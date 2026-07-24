"""Log-scale quantization for strictly positive optimizer second moments.

Per block:
  1. Compute log(v_min) and log(v_max)
  2. Define a center-coded n-bit grid on that fp32 interval
  3. Store the two endpoints as fp16
  4. Pack the n-bit indices densely when requested

Log-scale is used because v is positive and spans orders of magnitude. Adam's
update rule divides by sqrt(v), so relative resolution at small values matters.

Supports 2, 3, 4, 6, and 8-bit generalized modes plus the legacy 2-bit API.
"""

from __future__ import annotations

import math

import torch

from turboadam.utils import BLOCK_SIZE, packed_num_bytes

_VALID_BITS = (2, 3, 4, 6, 8)
_LOG_FLOOR = 1e-38
_SPAN_FLOOR = 1e-10


def _validate_n_bits(n_bits: int) -> None:
    if n_bits not in _VALID_BITS:
        raise ValueError(f"n_bits must be one of {_VALID_BITS}, got {n_bits}")


def _infer_packed(
    indices: torch.Tensor,
    n_bits: int,
    num_blocks: int,
    block_size: int,
    packed: bool | None,
) -> bool:
    if packed is not None:
        return packed
    if n_bits == 8:
        return False  # packed and unpacked are byte-identical at 8 bits
    unpacked_numel = num_blocks * block_size
    return indices.numel() != unpacked_numel


# ---------------------------------------------------------------------------
# Dense n-bit packing
# ---------------------------------------------------------------------------


def pack_nbit_indices(
    indices: torch.Tensor,
    n_bits: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Losslessly pack uint8 bucket indices into a dense little-endian stream.

    ``out`` may be the old packed optimizer-state buffer after it has been
    decoded, allowing each step to rewrite that storage in place.
    """
    _validate_n_bits(n_bits)
    x = indices.reshape(-1).to(torch.uint8)
    n = x.numel()
    expected = packed_num_bytes(n, n_bits)
    if expected * 8 != n * n_bits:
        raise ValueError("index count must produce a whole number of packed bytes")

    if out is not None:
        if (
            out.dtype != torch.uint8
            or out.device != x.device
            or out.numel() != expected
        ):
            raise ValueError(
                "packed output buffer has incompatible shape, dtype, or device"
            )
        packed = out.reshape(-1)
    elif n_bits == 8:
        # At 8 bits, the unpacked uint8 representation is already densely packed.
        return x.contiguous()
    else:
        packed = torch.empty(expected, dtype=torch.uint8, device=x.device)

    if n == 0:
        return packed
    if n_bits == 8:
        packed.copy_(x)
        return packed

    if n_bits == 4:
        g = x.reshape(-1, 2)
        packed.copy_(g[:, 0])
        packed.bitwise_or_(g[:, 1] << 4)
        return packed

    if n_bits == 2:
        g = x.reshape(-1, 4)
        packed.copy_(g[:, 0])
        packed.bitwise_or_(g[:, 1] << 2)
        packed.bitwise_or_(g[:, 2] << 4)
        packed.bitwise_or_(g[:, 3] << 6)
        return packed

    if n_bits == 3:
        g = x.reshape(-1, 8)
        dest = packed.reshape(-1, 3)
        dest[:, 0] = g[:, 0] | (g[:, 1] << 3) | ((g[:, 2] & 0x03) << 6)
        dest[:, 1] = (
            (g[:, 2] >> 2)
            | (g[:, 3] << 1)
            | (g[:, 4] << 4)
            | ((g[:, 5] & 0x01) << 7)
        )
        dest[:, 2] = (g[:, 5] >> 1) | (g[:, 6] << 2) | (g[:, 7] << 5)
        return packed

    # 6 bits: four values occupy exactly three bytes.
    g = x.reshape(-1, 4)
    dest = packed.reshape(-1, 3)
    dest[:, 0] = g[:, 0] | ((g[:, 1] & 0x03) << 6)
    dest[:, 1] = (g[:, 1] >> 2) | ((g[:, 2] & 0x0F) << 4)
    dest[:, 2] = (g[:, 2] >> 4) | (g[:, 3] << 2)
    return packed


def unpack_nbit_indices(
    packed_indices: torch.Tensor,
    n_bits: int,
    num_values: int,
) -> torch.Tensor:
    """Unpack a dense n-bit stream to one uint8 bucket index per value."""
    _validate_n_bits(n_bits)
    p = packed_indices.reshape(-1).to(torch.uint8)
    if num_values == 0:
        return p.new_empty(0)
    expected = packed_num_bytes(num_values, n_bits)
    if p.numel() != expected:
        raise ValueError(
            f"packed index length mismatch: expected {expected}, got {p.numel()}"
        )

    if n_bits == 8:
        return p[:num_values].contiguous()

    if n_bits == 4:
        out = torch.empty((p.numel(), 2), dtype=torch.uint8, device=p.device)
        out[:, 0] = p & 0x0F
        out[:, 1] = p >> 4
        return out.reshape(-1)[:num_values]

    if n_bits == 2:
        out = torch.empty((p.numel(), 4), dtype=torch.uint8, device=p.device)
        out[:, 0] = p & 0x03
        out[:, 1] = (p >> 2) & 0x03
        out[:, 2] = (p >> 4) & 0x03
        out[:, 3] = p >> 6
        return out.reshape(-1)[:num_values]

    if n_bits == 3:
        g = p.reshape(-1, 3)
        out = torch.empty((g.shape[0], 8), dtype=torch.uint8, device=p.device)
        b0, b1, b2 = g[:, 0], g[:, 1], g[:, 2]
        out[:, 0] = b0 & 0x07
        out[:, 1] = (b0 >> 3) & 0x07
        out[:, 2] = ((b0 >> 6) | (b1 << 2)) & 0x07
        out[:, 3] = (b1 >> 1) & 0x07
        out[:, 4] = (b1 >> 4) & 0x07
        out[:, 5] = ((b1 >> 7) | (b2 << 1)) & 0x07
        out[:, 6] = (b2 >> 2) & 0x07
        out[:, 7] = (b2 >> 5) & 0x07
        return out.reshape(-1)[:num_values]

    g = p.reshape(-1, 3)
    out = torch.empty((g.shape[0], 4), dtype=torch.uint8, device=p.device)
    b0, b1, b2 = g[:, 0], g[:, 1], g[:, 2]
    out[:, 0] = b0 & 0x3F
    out[:, 1] = ((b0 >> 6) | (b1 << 2)) & 0x3F
    out[:, 2] = ((b1 >> 4) | (b2 << 4)) & 0x3F
    out[:, 3] = (b2 >> 2) & 0x3F
    return out.reshape(-1)[:num_values]


# ---------------------------------------------------------------------------
# Shared log-grid helpers
# ---------------------------------------------------------------------------


def _stored_log_grid(log_blocks: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Build the existing center-coded grid and its persisted fp16 endpoints."""
    log_min = log_blocks.amin(dim=1)
    log_max = log_blocks.amax(dim=1)
    scales = torch.stack((log_min, log_max), dim=1).to(torch.float16)
    span = (log_max - log_min).unsqueeze(1)
    normalized = log_blocks.sub_(log_min.unsqueeze(1)).div_(
        span.clamp_min(_SPAN_FLOOR)
    ).clamp_(0.0, 1.0)
    return scales, log_min.unsqueeze(1), log_max.unsqueeze(1), span, normalized


def _quantize_centers(
    normalized: torch.Tensor,
    n_buckets: int,
    stochastic_round: bool,
) -> torch.Tensor:
    """Quantize normalized values to the center-coded grid used by the decoder."""
    if stochastic_round:
        # Decode uses (idx + 0.5) / n_buckets. Stochastic rounding in this
        # coordinate is unbiased in log-space inside the representable range;
        # the two edge half-bins necessarily clip to the nearest endpoint code.
        position = normalized.mul_(n_buckets).sub_(0.5).clamp_(
            0.0, n_buckets - 1.0
        )
        lower = position.floor()
        position.sub_(lower)  # reuse the normalized buffer as the up-probability
        round_up = torch.rand_like(position).lt_(position)
        return lower.add_(round_up).to(torch.uint8)

    # For center-coded bins, floor(normalized*K) is nearest-center assignment.
    return normalized.mul_(n_buckets).clamp_(0.0, n_buckets - 1.0).to(
        torch.uint8
    )


def _dequantize_indices(
    indices: torch.Tensor,
    scales: torch.Tensor,
    n_buckets: int,
    block_size: int,
) -> torch.Tensor:
    num_blocks = scales.shape[0]
    values = indices.reshape(num_blocks, block_size).float()
    log_min = scales[:, 0].float().unsqueeze(1)
    span = (scales[:, 1].float() - scales[:, 0].float()).unsqueeze(1)
    return values.add_(0.5).div_(n_buckets).mul_(span).add_(log_min).exp_()


# ---------------------------------------------------------------------------
# Legacy 2-bit API
# ---------------------------------------------------------------------------


def quantize_logscale(
    v_flat: torch.Tensor, block_size: int = BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a flat positive tensor to packed 2-bit log-scale indices."""
    indices, scales, _ = quantize_logscale_nbits(
        v_flat,
        n_bits=2,
        block_size=block_size,
        stochastic_round=False,
        packed=True,
    )
    return indices, scales


def dequantize_logscale(
    packed: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    original_numel: int = 0,
) -> torch.Tensor:
    """Reconstruct fp32 values from packed 2-bit log-scale indices."""
    return dequantize_logscale_nbits(
        packed,
        scales,
        n_bits=2,
        block_size=block_size,
        original_numel=original_numel,
        packed=True,
    )


# ---------------------------------------------------------------------------
# Generalized n-bit log-scale quantization
# ---------------------------------------------------------------------------


def quantize_logscale_nbits(
    v_flat: torch.Tensor,
    n_bits: int = 3,
    block_size: int = BLOCK_SIZE,
    stochastic_round: bool = False,
    packed: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Quantize a flat positive tensor to n-bit log-scale indices per block.

    Args:
        v_flat: 1-D tensor of positive values, padded to full blocks.
        n_bits: Bits per element (2, 3, 4, 6, or 8).
        block_size: Elements per independent block.
        stochastic_round: Use stochastic rounding on the center-coded log grid.
        packed: Densely bit-pack the returned indices. The default remains
            False for compatibility; TurboAdam's persistent state uses True.

    Returns:
        (indices, scales, n_bits), where scales stores fp16 (log_min, log_max).
    """
    _validate_n_bits(n_bits)
    if v_flat.ndim != 1:
        raise ValueError(f"v_flat must be 1-D, got shape {tuple(v_flat.shape)}")
    if block_size <= 0 or v_flat.shape[0] % block_size != 0:
        raise ValueError("v_flat length must be a multiple of positive block_size")
    if v_flat.numel() == 0:
        return (
            torch.empty(0, dtype=torch.uint8, device=v_flat.device),
            torch.empty((0, 2), dtype=torch.float16, device=v_flat.device),
            n_bits,
        )

    num_blocks = v_flat.shape[0] // block_size
    n_buckets = 1 << n_bits
    log_blocks = v_flat.reshape(num_blocks, block_size).to(
        dtype=torch.float32, copy=True
    )
    log_blocks.clamp_min_(_LOG_FLOOR).log_()
    scales, _, _, _, normalized = _stored_log_grid(log_blocks)
    indices = _quantize_centers(normalized, n_buckets, stochastic_round).reshape(-1)
    if packed:
        indices = pack_nbit_indices(indices, n_bits)
    return indices, scales, n_bits


def dequantize_logscale_nbits(
    indices: torch.Tensor,
    scales: torch.Tensor,
    n_bits: int = 3,
    block_size: int = BLOCK_SIZE,
    original_numel: int = 0,
    packed: bool | None = None,
) -> torch.Tensor:
    """Reconstruct fp32 values from packed or unpacked n-bit indices."""
    _validate_n_bits(n_bits)
    num_blocks = scales.shape[0]
    padded_numel = num_blocks * block_size
    if original_numel == 0:
        original_numel = padded_numel
    if original_numel < 0 or original_numel > padded_numel:
        raise ValueError("original_numel must lie inside the encoded padded length")
    if padded_numel == 0:
        return torch.empty(0, dtype=torch.float32, device=scales.device)

    is_packed = _infer_packed(indices, n_bits, num_blocks, block_size, packed)
    if is_packed:
        decoded_indices = unpack_nbit_indices(indices, n_bits, padded_numel)
    else:
        if indices.numel() != padded_numel:
            raise ValueError(
                "unpacked index length mismatch: "
                f"expected {padded_numel}, got {indices.numel()}"
            )
        decoded_indices = indices.reshape(-1).to(torch.uint8)

    values = _dequantize_indices(decoded_indices, scales, 1 << n_bits, block_size)
    return values.reshape(-1)[:original_numel]


# ---------------------------------------------------------------------------
# Fused decompress → EMA → recompress
# ---------------------------------------------------------------------------


def _updated_v_state(
    indices: torch.Tensor,
    scales: torch.Tensor,
    grad: torch.Tensor,
    beta2: float,
    n_bits: int,
    block_size: int,
    original_numel: int,
    packed: bool | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
    """Internal PyTorch reference core used by both public update functions."""
    _validate_n_bits(n_bits)
    num_blocks = scales.shape[0]
    padded_numel = num_blocks * block_size
    if original_numel != grad.numel():
        raise ValueError(
            f"gradient has {grad.numel()} elements, expected {original_numel}"
        )
    if padded_numel == 0:
        is_packed = bool(packed)
        return (
            grad.new_empty((0, block_size), dtype=torch.float32),
            indices,
            scales,
            is_packed,
        )

    is_packed = _infer_packed(indices, n_bits, num_blocks, block_size, packed)
    decoded_indices = (
        unpack_nbit_indices(indices, n_bits, padded_numel)
        if is_packed
        else indices.reshape(-1).to(torch.uint8)
    )
    v_blocks = _dequantize_indices(decoded_indices, scales, 1 << n_bits, block_size)

    # Update valid values without materializing a padded g² tensor.
    v_blocks.mul_(beta2)
    v_flat = v_blocks.reshape(-1)
    g_flat = grad.reshape(-1).float()
    v_flat[:original_numel].addcmul_(g_flat, g_flat, value=1.0 - beta2)

    log_blocks = v_blocks.clone().clamp_min_(_LOG_FLOOR).log_()
    new_log_min = log_blocks.amin(dim=1)
    new_log_max = log_blocks.amax(dim=1)

    # Only the final block can contain padding. Exclude it from its statistics.
    remainder = original_numel % block_size
    if remainder:
        valid_last = log_blocks[-1, :remainder]
        new_log_min[-1] = valid_last.amin()
        new_log_max[-1] = valid_last.amax()

    # The old endpoints are dead after dequantization, so rewrite their tiny
    # persistent buffer in place instead of allocating a second scale tensor.
    scales[:, 0].copy_(new_log_min)
    scales[:, 1].copy_(new_log_max)
    new_scales = scales
    new_span = (new_log_max - new_log_min).unsqueeze(1).clamp_min(_SPAN_FLOOR)
    normalized = log_blocks.sub_(new_log_min.unsqueeze(1)).div_(
        new_span
    ).clamp_(0.0, 1.0)
    if remainder:
        normalized[-1, remainder:] = 0.0

    new_unpacked = _quantize_centers(normalized, 1 << n_bits, True).reshape(-1)
    new_indices = (
        pack_nbit_indices(new_unpacked, n_bits, out=indices)
        if is_packed
        else new_unpacked
    )
    return v_blocks, new_indices, new_scales, is_packed


def fused_v_update(
    indices: torch.Tensor,
    scales: torch.Tensor,
    grad: torch.Tensor,
    beta2: float,
    n_bits: int,
    block_size: int,
    original_numel: int,
    packed: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompress v, apply its EMA update, and recompress in one PyTorch path.

    The returned fp32 v is the exact post-EMA value used for the current Adam
    denominator; quantization affects the state carried into the next step.
    """
    v_blocks, new_indices, new_scales, _ = _updated_v_state(
        indices,
        scales,
        grad,
        beta2,
        n_bits,
        block_size,
        original_numel,
        packed,
    )
    return new_indices, new_scales, v_blocks.reshape(-1)[:original_numel]


@torch.no_grad()
def fused_adam_update(
    indices: torch.Tensor,
    scales: torch.Tensor,
    grad: torch.Tensor,
    param: torch.Tensor,
    first_moment: torch.Tensor,
    beta2: float,
    n_bits: int,
    block_size: int,
    original_numel: int,
    lr: float,
    bias_correction1: float,
    bias_correction2: float,
    eps: float,
    weight_decay: float,
    packed: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused reference path: v update, AdamW parameter update, and recompression.

    The v buffer is reused in-place as the denominator after its new quantized
    representation has been produced, avoiding another full-size fp32 tensor.
    """
    v_blocks, new_indices, new_scales, _ = _updated_v_state(
        indices,
        scales,
        grad,
        beta2,
        n_bits,
        block_size,
        original_numel,
        packed,
    )
    if original_numel == 0:
        return new_indices, new_scales

    denom = v_blocks.reshape(-1)[:original_numel]
    denom.sqrt_().div_(math.sqrt(bias_correction2)).add_(eps)
    if weight_decay != 0.0:
        param.mul_(1.0 - lr * weight_decay)
    param.addcdiv_(
        first_moment.reshape_as(param),
        denom.reshape_as(param),
        value=-(lr / bias_correction1),
    )
    return new_indices, new_scales
