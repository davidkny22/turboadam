"""CoState: first moment (m) compression.

Gradient-residual decomposition: m = α·g + δ
  - α = (m·g) / (g·g)  : scalar per parameter tensor
  - δ = m - α·g        : residual orthogonal to current gradient

Residual δ is partitioned into blocks and classified:
  - Null costate       (r < τ₀): reconstruct zero residual
  - Phase costate      (τ₀ ≤ r < τ₁): sign + block RMS magnitude
  - Amplitude costate  (r ≥ τ₁): sign + block RMS magnitude

The prior implementation persisted both an fp32 block norm and an fp16 copy of
that norm divided by sqrt(block_size). The fp16 amplitude value is derived on
decode instead, preserving the existing numerics without storing it twice. In
manager state, labels are encoded directly in the norm value: null stores +0,
phase stores +norm, and amplitude stores -norm. The magnitude is unchanged for
every block that uses it, and null-block magnitudes were never decoded.

Adaptive thresholds: τ₀ = P_10(r), τ₁ = P_90(r) per parameter tensor per step.
No warmup required: EMA error-washing handles cold-start.
"""

from __future__ import annotations

import math

import torch

from turboadam.utils import BLOCK_SIZE, pad_to_blocks

try:
    from turboadam.triton_kernels import (
        triton_costate_decode as _triton_costate_decode,
        triton_costate_encode as _triton_costate_encode,
        triton_decompose_ratios as _triton_decompose_ratios,
        triton_projection_alpha as _triton_projection_alpha,
    )

    _HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    _HAS_TRITON = False
    _triton_costate_decode = None
    _triton_costate_encode = None
    _triton_decompose_ratios = None
    _triton_projection_alpha = None


def _scaled_projection_alpha(
    m_flat: torch.Tensor, g_flat: torch.Tensor
) -> torch.Tensor:
    """Bounded-memory projection fallback for devices without safe fp64 dots."""
    m_scale = m_flat.abs().amax()
    g_scale = g_flat.abs().amax()
    if not bool(g_scale > 0):
        return g_scale.new_zeros(())

    numerator = g_scale.new_zeros(())
    denominator = g_scale.new_zeros(())
    chunk = 1 << 20
    for start in range(0, g_flat.numel(), chunk):
        m_part = m_flat[start : start + chunk] / m_scale.clamp_min(
            m_scale.new_tensor(2.0**-149)
        )
        g_part = g_flat[start : start + chunk] / g_scale
        numerator.add_(m_part.dot(g_part))
        denominator.add_(g_part.dot(g_part))

    safe_denominator = denominator.clamp_min(torch.finfo(denominator.dtype).tiny)
    alpha = (m_scale / g_scale) * (numerator / safe_denominator)
    return torch.where(denominator > 0, alpha, alpha.new_zeros(()))


def _projection_alpha(m: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Return the scalar least-squares projection coefficient of m onto g."""
    m_flat = m.reshape(-1)
    g_flat = g.reshape(-1)
    if _HAS_TRITON and m.is_cuda and g.is_cuda:
        return _triton_projection_alpha(m_flat, g_flat)

    g_dot_g = g_flat.dot(g_flat)
    m_dot_g = m_flat.dot(g_flat)

    # The ordinary fp32 dots are both faster and bit-for-bit compatible with
    # the original implementation. Fall back only when they overflow, or when
    # a nonzero gradient is so small that g·g underflows to zero.
    needs_fallback = not bool(torch.isfinite(g_dot_g)) or not bool(
        torch.isfinite(m_dot_g)
    )
    if not needs_fallback and g_dot_g == 0:
        needs_fallback = bool(torch.count_nonzero(g_flat))

    if needs_fallback:
        if m.device.type != "cpu":
            return _scaled_projection_alpha(m_flat, g_flat)

        # Chunked fp64 reductions bound temporary memory. CPU float64 is exact
        # enough for every finite fp32 product and avoids both overflow and
        # underflow in the projection scalar.
        numerator = torch.zeros((), dtype=torch.float64, device=m.device)
        denominator = torch.zeros((), dtype=torch.float64, device=m.device)
        chunk = 1 << 20
        for start in range(0, g_flat.numel(), chunk):
            m_part = m_flat[start : start + chunk].double()
            g_part = g_flat[start : start + chunk].double()
            numerator.add_(m_part.dot(g_part))
            denominator.add_(g_part.dot(g_part))
        if denominator == 0:
            return g_dot_g.new_zeros(())
        return (numerator / denominator).to(dtype=g_dot_g.dtype)

    # clamp_min avoids evaluating 0/0 inside torch.where. The final where keeps
    # the exact zero-gradient convention alpha=0 without a host synchronization.
    safe_denominator = g_dot_g.clamp_min(torch.finfo(g_dot_g.dtype).tiny)
    alpha = m_dot_g / safe_denominator
    return torch.where(g_dot_g > 0, alpha, g_dot_g.new_zeros(()))


def _scaled_block_norms(blocks: torch.Tensor) -> torch.Tensor:
    """Compute finite fp32 block norms without squaring overflow/underflow."""
    block_scale = blocks.abs().amax(dim=1)
    min_subnormal = block_scale.new_tensor(2.0**-149)
    safe_scale = block_scale.clamp_min(min_subnormal)
    unit = blocks / safe_scale.unsqueeze(1)
    factor = torch.sqrt(torch.sum(unit * unit, dim=1))

    # The true norm can exceed fp32 even when every element is finite. Saturate
    # only that unrepresentable case instead of emitting inf into the codec.
    max_value = torch.finfo(blocks.dtype).max
    bounded_scale = torch.minimum(
        block_scale, max_value / factor.clamp_min(1.0)
    )
    return torch.where(block_scale > 0, bounded_scale * factor, 0.0)


def _block_norms(blocks: torch.Tensor) -> torch.Tensor:
    """Use the original fast norm when safe, with a scaled rare-case fallback."""
    norms = torch.linalg.vector_norm(blocks, dim=1)
    nonfinite = ~torch.isfinite(norms)
    zero_norm = norms == 0

    if blocks.device.type == "cpu":
        # Avoid another full read of ordinary nonzero blocks. Only distinguish
        # true zero blocks from underflow after the cheap norm checks request it.
        if not bool(nonfinite.any()) and not bool(zero_norm.any()):
            return norms
        nonzero_blocks = blocks.ne(0).any(dim=1)
    else:
        nonzero_blocks = blocks.ne(0).any(dim=1)

    bad = nonfinite | (zero_norm & nonzero_blocks)
    stable = _scaled_block_norms(blocks)
    return torch.where(bad, stable, norms)


def _scaled_block_ratios(
    delta_blocks: torch.Tensor, m_blocks: torch.Tensor
) -> torch.Tensor:
    """Compute block-norm ratios without materializing overflowing norms."""
    delta_scale = delta_blocks.abs().amax(dim=1)
    m_scale = m_blocks.abs().amax(dim=1)
    min_subnormal = delta_scale.new_tensor(2.0**-149)

    delta_safe = delta_scale.clamp_min(min_subnormal)
    m_safe = m_scale.clamp_min(min_subnormal)
    delta_unit = delta_blocks / delta_safe.unsqueeze(1)
    m_unit = m_blocks / m_safe.unsqueeze(1)
    delta_factor = torch.sqrt(torch.sum(delta_unit * delta_unit, dim=1))
    m_factor = torch.sqrt(torch.sum(m_unit * m_unit, dim=1))

    log_ratio = (
        delta_safe.log()
        - m_safe.log()
        + delta_factor.clamp_min(min_subnormal).log()
        - m_factor.clamp_min(min_subnormal).log()
    )
    max_log = math.log(torch.finfo(torch.float32).max)
    ratio = log_ratio.clamp_max(max_log).exp()
    return torch.where(
        (delta_scale > 0) & (m_scale > 0), ratio, torch.zeros_like(ratio)
    )


def decompose(m: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose momentum into a gradient-aligned component and residual.

    m = α·g + δ  where α = (m·g) / (g·g)
    """
    alpha = _projection_alpha(m, g)
    delta = m - alpha * g
    return alpha, delta


# ---------------------------------------------------------------------------
# Block ratio computation
# ---------------------------------------------------------------------------


def compute_block_ratios(
    delta: torch.Tensor,
    m: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    *,
    return_delta_norms: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Compute r = norm(delta_block) / norm(m_block) per block.

    ``return_delta_norms`` lets CoState reuse the already-computed residual
    norms during encoding, avoiding a second full block reduction.
    """
    delta_flat = delta.reshape(-1).float()
    m_flat = m.reshape(-1).float()
    if delta_flat.numel() == 0:
        empty = delta_flat.new_empty(0)
        return (empty, empty) if return_delta_norms else empty

    delta_padded, _ = pad_to_blocks(delta_flat, block_size)
    m_padded, _ = pad_to_blocks(m_flat, block_size)
    delta_blocks = delta_padded.reshape(-1, block_size)
    m_blocks = m_padded.reshape(-1, block_size)

    delta_norms = _block_norms(delta_blocks)
    m_norms = _block_norms(m_blocks)
    safe_m_norms = m_norms.clamp_min(m_norms.new_tensor(2.0**-149))
    ratios = torch.where(m_norms > 0, delta_norms / safe_m_norms, 0.0)

    max_value = torch.finfo(ratios.dtype).max
    saturation_floor = max_value * 0.5
    suspect = (
        (delta_norms >= saturation_floor)
        | (m_norms >= saturation_floor)
        | ~torch.isfinite(ratios)
    )
    if delta_blocks.device.type != "cpu" or bool(suspect.any()):
        stable_ratios = _scaled_block_ratios(delta_blocks, m_blocks)
        ratios = torch.where(suspect, stable_ratios, ratios)

    return (ratios, delta_norms) if return_delta_norms else ratios


# ---------------------------------------------------------------------------
# Threshold computation
# ---------------------------------------------------------------------------


def compute_thresholds(
    ratios: torch.Tensor,
    null_pct: float = 0.10,
    amp_pct: float = 0.90,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute exact adaptive percentile thresholds without interpolation.

    Small tensors use one sort because it has lower launch overhead. Larger
    tensors use two order-statistic selections, avoiding a full O(n log n) sort
    and its full sorted output allocation.
    """
    n = ratios.numel()
    if n == 0:
        raise ValueError("cannot compute CoState thresholds for an empty tensor")

    idx_lo = max(0, int(null_pct * n) - 1)
    idx_hi = min(n - 1, int(amp_pct * n))
    if n < 512:
        sorted_r = ratios.sort().values
        return sorted_r[idx_lo], sorted_r[idx_hi]

    tau0 = ratios.kthvalue(idx_lo + 1).values
    tau1 = ratios.kthvalue(idx_hi + 1).values
    return tau0, tau1


# ---------------------------------------------------------------------------
# Block classification
# ---------------------------------------------------------------------------


def classify_blocks(
    ratios: torch.Tensor,
    tau0: float | torch.Tensor,
    tau1: float | torch.Tensor,
) -> torch.Tensor:
    """Assign uint8 labels 0=null, 1=phase, and 2=amplitude."""
    labels = (ratios >= tau0).to(torch.uint8)
    labels.add_((ratios >= tau1).to(torch.uint8))
    return labels


# ---------------------------------------------------------------------------
# Encoding and decoding
# ---------------------------------------------------------------------------


def _pack_signs(values: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """Pack sign bits (1 if negative, 0 otherwise) into uint8 bytes."""
    n = values.numel()
    pad = (-n) % 8
    if pad:
        values = torch.cat((values, values.new_zeros(pad)))
    sign_bits = (values < 0).to(torch.uint8).reshape(-1, 8)
    packed = (
        (sign_bits[:, 0] << 7)
        | (sign_bits[:, 1] << 6)
        | (sign_bits[:, 2] << 5)
        | (sign_bits[:, 3] << 4)
        | (sign_bits[:, 4] << 3)
        | (sign_bits[:, 5] << 2)
        | (sign_bits[:, 6] << 1)
        | sign_bits[:, 7]
    )
    if out is not None:
        if out.shape != packed.shape or out.dtype != torch.uint8:
            raise ValueError("sign output buffer has incompatible shape or dtype")
        out.copy_(packed)
        return out
    return packed


def _unpack_signs(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Unpack uint8 sign bytes to +1/-1 fp32 values."""
    orig_device = packed.device
    is_mps = orig_device.type == "mps"
    work_device = torch.device("cpu") if is_mps else orig_device

    packed_int = packed.to(dtype=torch.int32, device=work_device)
    shifts = torch.arange(7, -1, -1, dtype=torch.int32, device=work_device)
    bits = ((packed_int.unsqueeze(1) >> shifts) & 1).reshape(-1)[:n].float()
    return (1.0 - 2.0 * bits).to(orig_device)


def encode_blocks(
    delta: torch.Tensor,
    labels: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    *,
    out: dict | None = None,
    include_legacy_scale: bool = True,
    compact_labels: bool = False,
    block_norms: torch.Tensor | None = None,
) -> dict:
    """Encode a residual into labels, packed signs, and fp32 block norms.

    ``include_legacy_scale=True`` preserves the old standalone function's
    dictionary shape. CoStateManager disables it because ``scales`` is exactly
    derivable from ``block_norms`` and therefore wastes persistent memory. It
    also enables ``compact_labels`` to encode each costate in the sign/value of
    its existing fp32 norm, with no additional persistent label tensor.
    """
    delta_flat = delta.reshape(-1).float()
    num_blocks = (delta_flat.numel() + block_size - 1) // block_size
    if block_norms is None:
        delta_padded, _ = pad_to_blocks(delta_flat, block_size)
        delta_blocks = delta_padded.reshape(-1, block_size)
        block_norms_new = _block_norms(delta_blocks).float()
    else:
        if block_norms.numel() != num_blocks:
            raise ValueError("precomputed block_norms has an incompatible length")
        block_norms_new = block_norms.reshape(-1).to(
            device=delta.device, dtype=torch.float32
        )

    if compact_labels:
        # +0=null, +norm=phase, -norm=amplitude. A zero-magnitude phase
        # collapsing to null is exactly equivalent because both decode to zero.
        signed_norms_new = torch.where(
            labels == 0,
            torch.zeros_like(block_norms_new),
            torch.where(labels == 2, -block_norms_new, block_norms_new),
        )
        if out is None:
            encoded = {
                "sign_packed": _pack_signs(delta_flat),
                "block_norms": signed_norms_new,
            }
        else:
            _pack_signs(delta_flat, out=out["sign_packed"])
            out["block_norms"].copy_(signed_norms_new)
            out.pop("labels", None)
            out.pop("phase_packed", None)
            encoded = out
    elif out is None:
        encoded = {
            "labels": labels,
            "sign_packed": _pack_signs(delta_flat),
            "block_norms": block_norms_new,
        }
    else:
        out["labels"].copy_(labels)
        _pack_signs(delta_flat, out=out["sign_packed"])
        out["block_norms"].copy_(block_norms_new)
        out.pop("phase_packed", None)
        encoded = out

    if include_legacy_scale:
        scale_new = (block_norms_new * (1.0 / math.sqrt(block_size))).to(torch.float16)
        if out is not None and "scales" in out:
            out["scales"].copy_(scale_new)
        else:
            encoded["scales"] = scale_new
    else:
        encoded.pop("scales", None)

    return encoded


def decode_blocks(
    encoded: dict,
    alpha,
    g: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    original_numel: int | None = None,
) -> torch.Tensor:
    """Reconstruct m_hat = alpha*g + delta_hat from compressed CoState.

    Phase uses the fp32-derived RMS magnitude. Amplitude derives the same fp16
    rounding that the old ``scales`` tensor stored, preserving old checkpoints'
    numerical behavior without retaining the redundant tensor.
    """
    g_flat = g.reshape(-1).float()
    if original_numel is None:
        original_numel = g_flat.numel()
    if original_numel == 0:
        return torch.empty_like(g, dtype=torch.float32)

    device = g_flat.device
    sign_packed = encoded["sign_packed"].to(dtype=torch.uint8, device=device)
    stored_norms = encoded["block_norms"].to(dtype=torch.float32, device=device)
    num_blocks = (original_numel + block_size - 1) // block_size

    if "labels" not in encoded:
        # A negative zero amplitude may collapse to null: both decode to zero.
        amplitude_mask = stored_norms < 0
        block_norms = stored_norms.abs()
        phase_mask = (~amplitude_mask) & (block_norms > 0)
    else:
        labels = encoded["labels"].to(dtype=torch.uint8, device=device)
        phase_mask = labels == 1
        amplitude_mask = labels == 2
        block_norms = stored_norms

    signs_flat = _unpack_signs(sign_packed, original_numel)
    signs_padded, _ = pad_to_blocks(signs_flat, block_size)
    signs_blocks = signs_padded.reshape(num_blocks, block_size)

    phase_scales = block_norms / math.sqrt(block_size)
    amplitude_scales = phase_scales.to(torch.float16).float()
    block_scales = torch.where(
        phase_mask,
        phase_scales,
        torch.where(amplitude_mask, amplitude_scales, 0.0),
    )
    delta_hat = (block_scales.unsqueeze(1) * signs_blocks).reshape(-1)[:original_numel]
    return (alpha * g_flat + delta_hat).reshape(g.shape)


# ---------------------------------------------------------------------------
# CoStateManager: stateful per-step update loop
# ---------------------------------------------------------------------------


class CoStateManager:
    """Stateful manager for CoState first-moment compression."""

    def __init__(
        self,
        block_size: int = BLOCK_SIZE,
        error_feedback: bool = False,
        null_pct: float = 0.10,
        amp_pct: float = 0.90,
    ) -> None:
        self.block_size = block_size
        self._null_pct = null_pct
        self._amp_pct = amp_pct
        self._has_state: bool = False
        self._alpha = 0.0
        self._encoded: dict | None = None
        self._original_numel: int = 0
        self._error_feedback = error_feedback
        self._ef_residual: torch.Tensor | None = None

    def update(self, g: torch.Tensor, beta1: float) -> torch.Tensor:
        """Run one CoState reconstruction, EMA, decomposition, and encode step."""
        g = g.float()
        if g.numel() == 0:
            return torch.empty_like(g)

        use_triton = _HAS_TRITON and g.is_cuda
        decode = _triton_costate_decode if use_triton else decode_blocks

        # Reconstruct the old state, then reuse that transient buffer for m_new.
        if self._has_state:
            m_new = decode(
                self._encoded,
                self._alpha,
                g,
                self.block_size,
                self._original_numel,
            )
            if self._error_feedback and self._ef_residual is not None:
                m_new.lerp_(g + self._ef_residual, 1.0 - beta1)
            else:
                # Adam's EMA written as lerp is one fused elementwise pass.
                m_new.lerp_(g, 1.0 - beta1)
        else:
            if self._error_feedback and self._ef_residual is not None:
                m_new = (g + self._ef_residual).mul(1.0 - beta1)
            else:
                m_new = g.clone().mul_(1.0 - beta1)

        alpha_new = _projection_alpha(m_new, g)
        if use_triton:
            delta_new, ratios, block_norms = _triton_decompose_ratios(
                m_new,
                g,
                alpha_new,
                self.block_size,
                return_block_norms=True,
            )
        else:
            delta_new = m_new - alpha_new * g
            ratios, block_norms = compute_block_ratios(
                delta_new,
                m_new,
                self.block_size,
                return_delta_norms=True,
            )

        tau0, tau1 = compute_thresholds(ratios, self._null_pct, self._amp_pct)
        labels = classify_blocks(ratios, tau0, tau1)

        if use_triton:
            encoded_new = _triton_costate_encode(
                delta_new,
                labels,
                self.block_size,
                out=self._encoded,
                include_legacy_scale=False,
                compact_labels=True,
                block_norms=block_norms,
            )
        else:
            encoded_new = encode_blocks(
                delta_new,
                labels,
                self.block_size,
                out=self._encoded,
                include_legacy_scale=False,
                compact_labels=True,
                block_norms=block_norms,
            )

        # Optional error feedback is intentionally full precision and therefore
        # trades away the memory benefit; it remains opt-in for ablation parity.
        if self._error_feedback:
            zero_alpha = g.new_zeros(())
            delta_hat = decode(
                encoded_new, zero_alpha, g, self.block_size, m_new.numel()
            )
            ef_error = (delta_new - delta_hat).detach()
            if self._ef_residual is None:
                self._ef_residual = ef_error
            else:
                self._ef_residual.mul_(beta1).add_(ef_error, alpha=1.0 - beta1)

        if isinstance(self._alpha, torch.Tensor):
            self._alpha.copy_(alpha_new)
        else:
            self._alpha = alpha_new.detach()
        self._encoded = encoded_new
        self._original_numel = m_new.numel()
        self._has_state = True
        return m_new
