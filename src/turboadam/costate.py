"""CoState — first moment (m) compression.

Gradient-residual decomposition: m = α·g + δ
  - α = (m·g) / (g·g)  — scalar per parameter tensor
  - δ = m - α·g        — residual orthogonal to current gradient

Residual δ is partitioned into 128-element blocks and classified:
  - Null costate    (r < τ₀): store 1 bit in bitmap
  - Phase costate   (τ₀ ≤ r < τ₁): store 1-bit sign per element
  - Amplitude costate (r ≥ τ₁): store 1-bit sign + fp16 block scale

Adaptive thresholds: τ₀ = P_10(r), τ₁ = P_90(r) per parameter tensor per step.
No warmup required — EMA error-washing handles cold-start.
"""

import math
import torch

from turboadam.utils import pad_to_blocks, BLOCK_SIZE


def decompose(m: torch.Tensor, g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Decompose momentum into gradient-aligned component and residual.

    m = α·g + δ  where α = (m·g) / (g·g)

    Args:
        m: First moment tensor (any shape, will be treated as flat).
        g: Gradient tensor (same shape as m).

    Returns:
        (alpha, delta) where alpha is a scalar tensor (0-dim) and delta has
        the same shape as m.
    """
    m_flat = m.reshape(-1)
    g_flat = g.reshape(-1)
    g_dot_g = g_flat.dot(g_flat)
    # Keep alpha as a GPU scalar tensor — no .item() sync
    alpha = torch.where(
        g_dot_g > 0, m_flat.dot(g_flat) / g_dot_g, g_dot_g.new_zeros(())
    )
    delta = m - alpha * g
    return alpha, delta


# ---------------------------------------------------------------------------
# Block ratio computation
# ---------------------------------------------------------------------------


def compute_block_ratios(
    delta: torch.Tensor,
    m: torch.Tensor,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Compute per-block ratio r = norm(delta_block) / norm(m_block).

    Args:
        delta:      Residual tensor (any shape).
        m:          First moment tensor (same shape as delta).
        block_size: Elements per block (default: BLOCK_SIZE = 128).

    Returns:
        1-D float32 tensor of length num_blocks, each entry in [0, ∞).
        Blocks where norm(m_block) == 0 get ratio 0.
    """
    delta_flat = delta.reshape(-1).float()
    m_flat = m.reshape(-1).float()

    delta_padded, orig_len = pad_to_blocks(delta_flat, block_size)
    m_padded, _ = pad_to_blocks(m_flat, block_size)

    num_blocks = delta_padded.shape[0] // block_size
    delta_blocks = delta_padded.reshape(num_blocks, block_size)
    m_blocks = m_padded.reshape(num_blocks, block_size)

    delta_norms = delta_blocks.norm(dim=1)  # (num_blocks,)
    m_norms = m_blocks.norm(dim=1)  # (num_blocks,)

    # Guard: where m_norm is zero, ratio is 0
    safe_m_norms = m_norms.clone()
    safe_m_norms[safe_m_norms == 0.0] = 1.0  # avoid division by zero
    ratios = delta_norms / safe_m_norms
    ratios[m_norms == 0.0] = 0.0

    return ratios


# ---------------------------------------------------------------------------
# Threshold computation
# ---------------------------------------------------------------------------


def compute_thresholds(
    ratios: torch.Tensor,
    null_pct: float = 0.10,
    amp_pct: float = 0.90,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute adaptive thresholds as percentiles of ratios.

    Default P10/P90 gives 10% null, 80% phase, 10% amplitude.

    Uses sort+index instead of torch.quantile for ~4x speedup on GPU.
    Returns scalar tensors (no .item() sync).

    Args:
        ratios: 1-D float tensor of per-block ratios.
        null_pct: Percentile for null/phase boundary. Default: 0.10 (P10).
        amp_pct: Percentile for phase/amplitude boundary. Default: 0.90 (P90).

    Returns:
        (tau0, tau1) as scalar tensors on the same device as ratios.
    """
    sorted_r = ratios.sort().values
    n = sorted_r.shape[0]
    idx_lo = max(0, int(null_pct * n) - 1)
    idx_hi = min(n - 1, int(amp_pct * n))
    return sorted_r[idx_lo], sorted_r[idx_hi]


# ---------------------------------------------------------------------------
# Block classification
# ---------------------------------------------------------------------------


def classify_blocks(
    ratios: torch.Tensor,
    tau0: float,
    tau1: float,
) -> torch.Tensor:
    """Assign costate label to each block based on its ratio.

    Labels:
        0 (null)      : r < tau0
        1 (phase)     : tau0 <= r < tau1
        2 (amplitude) : r >= tau1

    Args:
        ratios: 1-D float tensor of per-block ratios.
        tau0:   Lower threshold (10th percentile).
        tau1:   Upper threshold (90th percentile).

    Returns:
        uint8 tensor of labels, same length as ratios.
    """
    labels = torch.zeros(ratios.shape[0], dtype=torch.uint8, device=ratios.device)
    labels[ratios >= tau0] = 1  # phase (will be overwritten for amplitude)
    labels[ratios >= tau1] = 2  # amplitude
    return labels


# ---------------------------------------------------------------------------
# Encoding and decoding
# ---------------------------------------------------------------------------


def _pack_signs(values: torch.Tensor) -> torch.Tensor:
    """Pack sign bits (1 if negative, 0 if non-negative) into uint8, 8 bits/byte.

    Args:
        values: 1-D float tensor of arbitrary length.

    Returns:
        uint8 tensor of length ceil(len(values) / 8).
    """
    n = values.shape[0]
    pad = (8 - n % 8) % 8
    if pad > 0:
        values = torch.cat([values, values.new_zeros(pad)])
    sign_bits = (values < 0).to(torch.uint8).reshape(-1, 8)
    # Pack via bitwise shifts (avoids creating multiplier tensor each call)
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
    return packed


def _unpack_signs(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Unpack sign bits from uint8 bytes back to +1/-1 float values.

    Args:
        packed: uint8 tensor of packed sign bytes.
        n:      Number of elements to unpack (may be less than len(packed)*8).

    Returns:
        float32 tensor of length n with values +1 or -1.
    """
    # Ensure uint8 (PyTorch load_state_dict _cast may change dtype).
    # Use vectorized bit extraction — no Python loop, stays on the original device.
    # MPS doesn't support integer bitwise ops, so fall back to CPU for MPS only.
    orig_device = packed.device
    is_mps = orig_device.type == "mps"
    work_device = torch.device("cpu") if is_mps else orig_device

    packed_int = packed.to(dtype=torch.int32, device=work_device)
    shifts = torch.tensor(
        [7, 6, 5, 4, 3, 2, 1, 0], dtype=torch.int32, device=work_device
    )
    # (num_bytes, 8): extract all 8 bits per byte in one shot
    bits = ((packed_int.unsqueeze(1) >> shifts.unsqueeze(0)) & 1).float()
    bits = bits.reshape(-1)[:n]
    signs = 1.0 - 2.0 * bits
    return signs.to(orig_device)


def encode_blocks(
    delta: torch.Tensor,
    labels: torch.Tensor,
    block_size: int = BLOCK_SIZE,
) -> dict:
    """Encode delta into per-block compressed representation.

    Args:
        delta:      Residual tensor (any shape, flattened internally).
        labels:     uint8 costate labels, one per block.
        block_size: Elements per block.

    Returns:
        dict with:
            labels      (uint8)  : costate label per block
            sign_packed (uint8)  : packed sign bits, ceil(numel/8) bytes
            block_norms (float32): L2 norm of each delta block
            scales      (float16): per-block amplitude scale = block_norm / sqrt(block_size),
                                   stored as fp16.  This is the uniform per-element magnitude
                                   that sign-only reconstruction (amplitude costate) uses.
    """
    delta_flat = delta.reshape(-1).float()

    delta_padded, _ = pad_to_blocks(delta_flat, block_size)
    num_blocks = delta_padded.shape[0] // block_size
    delta_blocks = delta_padded.reshape(num_blocks, block_size)

    # Per-block L2 norms
    block_norms = delta_blocks.norm(dim=1).float()  # (num_blocks,)

    # Per-block fp16 amplitude scales: block_norm / sqrt(block_size).
    # Storing the per-element uniform magnitude (rather than the block L2 norm) means
    # amplitude decode is: scale * sign(delta_block), yielding the correct element magnitudes.
    # Phase decode computes the same value on the fly from block_norms; amplitude stores it
    # explicitly in fp16 so the scale is preserved with full fp16 precision.
    scales = (block_norms / math.sqrt(block_size)).to(torch.float16)  # (num_blocks,)

    # Pack sign bits for the original (un-padded) elements
    sign_packed = _pack_signs(delta_flat)

    return {
        "labels": labels,
        "sign_packed": sign_packed,
        "block_norms": block_norms,
        "scales": scales,
    }


def decode_blocks(
    encoded: dict,
    alpha,
    g: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    original_numel: int = None,
) -> torch.Tensor:
    """Reconstruct approximated m = alpha*g + delta_hat from encoded representation.

    Per-costate delta_hat reconstruction:
        Null (0)      : delta_hat_block = 0
        Phase (1)     : delta_hat_block = (norm(delta_block)/sqrt(block_size)) * sign(delta_block)
        Amplitude (2) : delta_hat_block = fp16_scale * sign(delta_block)

    Args:
        encoded:        dict returned by encode_blocks.
        alpha:          Scalar float from decompose().
        g:              Gradient tensor (same original shape as delta).
        block_size:     Elements per block.
        original_numel: Number of elements in the original delta (before padding).

    Returns:
        Reconstructed m tensor with the same shape as g.
    """
    g_flat = g.reshape(-1).float()
    if original_numel is None:
        original_numel = g_flat.shape[0]

    device = g_flat.device
    labels = encoded["labels"].to(dtype=torch.uint8, device=device)
    sign_packed = encoded["sign_packed"]
    block_norms = encoded["block_norms"].to(dtype=torch.float32, device=device)
    scales = encoded["scales"].to(dtype=torch.float32, device=device)
    num_blocks = labels.shape[0]

    # Unpack sign bits and reshape to block layout
    signs_flat = _unpack_signs(sign_packed, original_numel)
    signs_padded, _ = pad_to_blocks(signs_flat, block_size)
    signs_blocks = signs_padded.reshape(num_blocks, block_size)

    # Compute per-block scale for each costate (vectorized — no Python loop):
    #   Null (0):      scale = 0
    #   Phase (1):     scale = block_norm / sqrt(block_size)
    #   Amplitude (2): scale = fp16 stored scale
    # Build a (num_blocks,) scale tensor, then broadcast over block_size.
    phase_scales = block_norms / math.sqrt(block_size)  # (num_blocks,)

    # Build per-block scale: null→0, phase→phase_scale, amplitude→stored scale
    # Use label as index: [0_scale, phase_scale, amp_scale] per block
    block_scales = torch.zeros_like(phase_scales)
    mask_phase = labels == 1
    mask_amp = labels == 2
    block_scales[mask_phase] = phase_scales[mask_phase]
    block_scales[mask_amp] = scales[mask_amp]

    # Broadcast: (num_blocks, 1) * (num_blocks, block_size) → (num_blocks, block_size)
    delta_hat_blocks = (
        block_scales.unsqueeze(1) * signs_blocks
    )  # (num_blocks, block_size)

    # Trim to original numel and reconstruct m
    delta_hat = delta_hat_blocks.reshape(-1)[:original_numel]
    result = alpha * g_flat + delta_hat
    return result.reshape(g.shape)


# ---------------------------------------------------------------------------
# CoStateManager — stateful per-step update loop (spec section 4.5)
# ---------------------------------------------------------------------------


class CoStateManager:
    """Stateful manager for CoState first moment compression.

    Implements the per-step update procedure from spec section 4.5:
      1. Load compressed δ̂ and costate bitmap from memory (if prior state exists)
      2. Reconstruct m̃ = α · g + decompress(δ̂)   [skip on first call, use m̃ = 0]
      3. Compute EMA update: m_new = β₁ · m̃ + (1 - β₁) · g
      4. Compute new projection: α_new = (m_new · g) / (g · g)
      5. Compute new residual: δ_new = m_new - α_new · g
      6. Classify blocks into costates using adaptive thresholds
      7. Compress and store δ̂_new according to costate assignments
      8. Store updated costate bitmap and α_new

    Usage:
        mgr = CoStateManager(block_size=128)
        m = mgr.update(g, beta1=0.9)  # call each optimizer step
    """

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
        self._alpha = 0.0  # becomes a scalar tensor after first update
        self._encoded: dict | None = None
        self._original_numel: int = 0
        self._error_feedback = error_feedback
        self._ef_residual: torch.Tensor | None = None

    def update(self, g: torch.Tensor, beta1: float) -> torch.Tensor:
        """Run one step of the CoState update procedure.

        Args:
            g:     Current gradient tensor (any shape).
            beta1: EMA decay for first moment (e.g. 0.9).

        Returns:
            m_new: Updated first moment tensor, same shape as g.
        """
        # Cast gradient to fp32 — CoState accumulators are fp32, and fp16/bf16
        # gradients would cause dtype mismatches in dot products.
        g = g.float()

        # Use Triton kernels if available and on CUDA
        try:
            from turboadam.triton_kernels import (
                triton_costate_decode,
                triton_costate_encode,
                triton_decompose_ratios,
            )

            _use_triton = g.is_cuda
        except ImportError:
            _use_triton = False

        _decode = triton_costate_decode if _use_triton else decode_blocks
        _encode = triton_costate_encode if _use_triton else encode_blocks
        _decompose_ratios = triton_decompose_ratios if _use_triton else None

        # Step 1-2: Reconstruct m̃ from compressed prior state (or zeros on first call)
        if self._has_state:
            m_hat = _decode(
                self._encoded,
                self._alpha,
                g,
                self.block_size,
                self._original_numel,
            )
        else:
            m_hat = torch.zeros_like(g, dtype=torch.float32)

        # Error feedback: compensate for previous step's encoding loss
        if self._error_feedback and self._ef_residual is not None:
            g_corrected = g + self._ef_residual
        else:
            g_corrected = g

        # Step 3: EMA update
        m_new = beta1 * m_hat + (1.0 - beta1) * g_corrected

        # Steps 4-6: Decompose + block ratios + classify
        alpha_new, delta_new = decompose(m_new, g)
        ratios = compute_block_ratios(delta_new, m_new, self.block_size)
        tau0, tau1 = compute_thresholds(ratios, self._null_pct, self._amp_pct)
        labels = classify_blocks(ratios, tau0, tau1)

        # Steps 7-8: Compress and store
        encoded_new = _encode(delta_new, labels, self.block_size)

        # Error feedback: measure what the encoding lost, accumulate for next step
        if self._error_feedback:
            zero_alpha = g.new_zeros(1)
            delta_hat = _decode(
                encoded_new, zero_alpha, g, self.block_size, m_new.numel()
            )
            ef_error = (delta_new - delta_hat).detach()
            if self._ef_residual is None:
                self._ef_residual = ef_error
            else:
                self._ef_residual = beta1 * self._ef_residual + (1.0 - beta1) * ef_error

        # Graph-stable buffer management: on first call allocate by cloning,
        # on subsequent calls copy data in-place to keep tensor addresses stable.
        if self._encoded is None:
            # First step: allocate graph-stable buffers by cloning
            self._alpha = (
                alpha_new.clone() if isinstance(alpha_new, torch.Tensor) else alpha_new
            )
            self._encoded = {
                "labels": encoded_new["labels"].clone(),
                "sign_packed": encoded_new["sign_packed"].clone(),
                "block_norms": encoded_new["block_norms"].clone(),
                "scales": encoded_new["scales"].clone(),
            }
        else:
            if isinstance(self._alpha, torch.Tensor):
                self._alpha.copy_(alpha_new)
            else:
                self._alpha = alpha_new
            self._encoded["labels"].copy_(encoded_new["labels"])
            self._encoded["sign_packed"].copy_(encoded_new["sign_packed"])
            self._encoded["block_norms"].copy_(encoded_new["block_norms"])
            self._encoded["scales"].copy_(encoded_new["scales"])
        self._original_numel = m_new.numel()
        self._has_state = True

        return m_new
