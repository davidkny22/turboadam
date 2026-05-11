"""Log-scale quantization for strictly positive optimizer second moments.

Per 128-element block:
  1. Compute log(v_min) and log(v_max)
  2. Define 2^n_bits evenly-spaced buckets on log scale
  3. Quantize each element to nearest bucket index
  4. Store: n_bits/element + 2 fp16 scalars (min, max) per block

Log-scale chosen because v is strictly positive and spans orders of magnitude.
Adam's update rule (dividing by √v) is most sensitive to small v values;
log-scale spacing allocates more resolution there.

Supports 2-bit (legacy) and n-bit (generalized) modes.
"""

import torch

from turboadam.utils import BLOCK_SIZE


def quantize_logscale(
    v_flat: torch.Tensor, block_size: int = BLOCK_SIZE
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a flat positive tensor to 2-bit log-scale indices per block.

    Args:
        v_flat: 1-D tensor of strictly positive values, length must be a
                multiple of block_size.
        block_size: Elements per quantization block (default 128).

    Returns:
        (packed_indices, scales) where:
        - packed_indices: uint8 tensor, 4 values packed per byte.
          Shape: (num_blocks * block_size // 4,)
        - scales: fp16 tensor of (log_min, log_max) per block.
          Shape: (num_blocks, 2)
    """
    assert v_flat.ndim == 1
    assert v_flat.shape[0] % block_size == 0
    num_blocks = v_flat.shape[0] // block_size

    # Reshape to (num_blocks, block_size)
    blocks = v_flat.reshape(num_blocks, block_size)
    log_blocks = blocks.clamp(min=1e-38).log()

    # Per-block min/max on log scale
    log_min = log_blocks.min(dim=1).values  # (num_blocks,)
    log_max = log_blocks.max(dim=1).values  # (num_blocks,)
    scales = torch.stack([log_min, log_max], dim=1).to(torch.float16)

    # Compute bucket boundaries: 4 evenly-spaced centers on log scale
    # Bucket centers at: log_min + (0.5, 1.5, 2.5, 3.5) * step
    span = (log_max - log_min).unsqueeze(1)  # (num_blocks, 1)
    # Avoid division by zero for constant blocks
    span = span.clamp(min=1e-10)
    # Normalize to [0, 1] range within each block
    normalized = (log_blocks - log_min.unsqueeze(1)) / span  # (num_blocks, block_size)
    # Map to bucket index 0-3
    indices = (
        (normalized * 4.0).clamp(0, 3.999).to(torch.long)
    )  # (num_blocks, block_size)

    # Pack 4 x 2-bit indices into each uint8 byte
    indices_flat = indices.reshape(-1)  # (total_elements,)
    num_bytes = indices_flat.shape[0] // 4
    indices_groups = indices_flat.reshape(num_bytes, 4)
    packed = (
        (indices_groups[:, 0] & 0x03)
        | ((indices_groups[:, 1] & 0x03) << 2)
        | ((indices_groups[:, 2] & 0x03) << 4)
        | ((indices_groups[:, 3] & 0x03) << 6)
    ).to(torch.uint8)

    return packed, scales


def dequantize_logscale(
    packed: torch.Tensor,
    scales: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    original_numel: int = 0,
) -> torch.Tensor:
    """Reconstruct values from 2-bit log-scale quantized representation.

    Args:
        packed: uint8 tensor of packed 2-bit indices.
        scales: fp16 tensor of (log_min, log_max) per block, shape (num_blocks, 2).
        block_size: Elements per block.
        original_numel: Original number of elements (for output shape).

    Returns:
        Reconstructed fp32 tensor of shape (original_numel,).
    """
    num_blocks = scales.shape[0]
    if original_numel == 0:
        original_numel = num_blocks * block_size

    # Ensure packed is uint8 (PyTorch load_state_dict _cast may change dtype)
    packed = packed.to(torch.uint8)

    # Unpack 4 x 2-bit indices from each byte
    idx0 = (packed & 0x03).to(torch.long)
    idx1 = ((packed >> 2) & 0x03).to(torch.long)
    idx2 = ((packed >> 4) & 0x03).to(torch.long)
    idx3 = ((packed >> 6) & 0x03).to(torch.long)
    indices = torch.stack([idx0, idx1, idx2, idx3], dim=1).reshape(-1)
    indices = indices[: num_blocks * block_size].reshape(num_blocks, block_size)

    # Reconstruct bucket centers
    log_min = scales[:, 0].to(torch.float32).unsqueeze(1)  # (num_blocks, 1)
    log_max = scales[:, 1].to(torch.float32).unsqueeze(1)
    span = log_max - log_min

    # Bucket centers at (index + 0.5) / 4 * span + log_min
    log_values = log_min + (indices.float() + 0.5) / 4.0 * span
    values = log_values.exp()

    return values.reshape(-1)[:original_numel]


# ---------------------------------------------------------------------------
# Generalized n-bit log-scale quantization
# ---------------------------------------------------------------------------


def quantize_logscale_nbits(
    v_flat: torch.Tensor,
    n_bits: int = 3,
    block_size: int = BLOCK_SIZE,
    stochastic_round: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Quantize a flat positive tensor to n-bit log-scale indices per block.

    Stores indices as uint8 (one index per byte for simplicity).
    Theoretical cost: n_bits + 32/block_size bits/param.

    Args:
        v_flat: 1-D tensor of strictly positive values, length must be a
                multiple of block_size.
        n_bits: Bits per element (number of buckets = 2^n_bits).
        block_size: Elements per quantization block (default 128).
        stochastic_round: If True, use stochastic rounding (unbiased) instead
            of deterministic floor.  Essential for compress-every-step to
            prevent systematic bias accumulation in the EMA.

    Returns:
        (indices, scales, n_bits) where:
        - indices: uint8 tensor of bucket indices, shape (num_elements,).
        - scales: fp16 tensor of (log_min, log_max) per block, shape (num_blocks, 2).
        - n_bits: stored for dequantization.
    """
    assert v_flat.ndim == 1
    assert v_flat.shape[0] % block_size == 0
    num_blocks = v_flat.shape[0] // block_size
    n_buckets = 2**n_bits

    blocks = v_flat.reshape(num_blocks, block_size)
    log_blocks = blocks.clamp(min=1e-38).log()

    log_min = log_blocks.min(dim=1).values
    log_max = log_blocks.max(dim=1).values
    scales = torch.stack([log_min, log_max], dim=1).to(torch.float16)

    span = (log_max - log_min).unsqueeze(1).clamp(min=1e-10)
    normalized = (log_blocks - log_min.unsqueeze(1)) / span
    continuous = normalized * n_buckets  # (num_blocks, block_size)

    if stochastic_round:
        floor_idx = continuous.floor()
        frac = continuous - floor_idx
        # Round up with probability equal to fractional part (unbiased)
        indices = (
            (floor_idx + (torch.rand_like(frac) < frac).float())
            .clamp(0, n_buckets - 1)
            .to(torch.uint8)
        )
    else:
        indices = continuous.clamp(0, n_buckets - 0.001).to(torch.uint8)

    return indices.reshape(-1), scales, n_bits


def dequantize_logscale_nbits(
    indices: torch.Tensor,
    scales: torch.Tensor,
    n_bits: int = 3,
    block_size: int = BLOCK_SIZE,
    original_numel: int = 0,
) -> torch.Tensor:
    """Reconstruct values from n-bit log-scale quantized representation.

    Args:
        indices: uint8 tensor of bucket indices.
        scales: fp16 tensor of (log_min, log_max) per block, shape (num_blocks, 2).
        n_bits: Bits per element.
        block_size: Elements per block.
        original_numel: Original number of elements.

    Returns:
        Reconstructed fp32 tensor of shape (original_numel,).
    """
    num_blocks = scales.shape[0]
    if original_numel == 0:
        original_numel = num_blocks * block_size
    n_buckets = 2**n_bits

    idx = indices.to(torch.long).reshape(num_blocks, block_size)

    log_min = scales[:, 0].to(torch.float32).unsqueeze(1)
    log_max = scales[:, 1].to(torch.float32).unsqueeze(1)
    span = log_max - log_min

    log_values = log_min + (idx.float() + 0.5) / n_buckets * span
    return log_values.exp().reshape(-1)[:original_numel]


# ---------------------------------------------------------------------------
# Fused decompress → EMA → recompress (avoids redundant pad/reshape cycles)
# ---------------------------------------------------------------------------


def fused_v_update(
    indices: torch.Tensor,
    scales: torch.Tensor,
    grad: torch.Tensor,
    beta2: float,
    n_bits: int,
    block_size: int,
    original_numel: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decompress v, apply EMA update, recompress — all in one pass.

    Avoids the pad→unpad→reshape→pad cycle of separate decompress + compress.

    Args:
        indices:  Compressed v indices (uint8).
        scales:   Compressed v scales (fp16, num_blocks × 2).
        grad:     Current gradient (same shape as original param).
        beta2:    EMA decay for second moment.
        n_bits:   Quantization bits.
        block_size: Elements per block.
        original_numel: Original number of elements before padding.

    Returns:
        (new_indices, new_scales, v_flat): recompressed v and the fp32 v
        (trimmed to original_numel) for denominator computation.
    """
    num_blocks = scales.shape[0]
    n_buckets = 2**n_bits

    # --- Decompress in block layout (stay in blocks, no reshape to param shape) ---
    idx = indices.to(torch.long).reshape(num_blocks, block_size)
    log_min = scales[:, 0].to(torch.float32).unsqueeze(1)
    log_max = scales[:, 1].to(torch.float32).unsqueeze(1)
    span = log_max - log_min
    v_blocks = (log_min + (idx.float() + 0.5) / n_buckets * span).exp()

    # --- EMA update in block layout ---
    g_flat = grad.reshape(-1).float()
    # Pad g² to match block layout
    padded_len = num_blocks * block_size
    if g_flat.shape[0] < padded_len:
        g_sq = torch.zeros(padded_len, dtype=torch.float32, device=g_flat.device)
        g_sq[: g_flat.shape[0]] = g_flat * g_flat
    else:
        g_sq = g_flat * g_flat
    g_sq_blocks = g_sq.reshape(num_blocks, block_size)

    # v_new = β₂ · v_old + (1 - β₂) · g²
    v_blocks.mul_(beta2).add_(g_sq_blocks, alpha=1.0 - beta2)

    # --- Recompress (quantize in-place, already in block layout) ---
    # Mask padded elements out of block min/max so partial blocks keep
    # correct statistics.
    log_blocks = v_blocks.clamp(min=1e-38).log()
    block_indices = torch.arange(num_blocks, device=v_blocks.device).unsqueeze(1)
    elem_indices = torch.arange(block_size, device=v_blocks.device).unsqueeze(0)
    valid_mask = (block_indices * block_size + elem_indices) < original_numel
    new_log_min = log_blocks.masked_fill(~valid_mask, float("inf")).min(dim=1).values
    new_log_max = log_blocks.masked_fill(~valid_mask, float("-inf")).max(dim=1).values
    new_scales = torch.stack([new_log_min, new_log_max], dim=1).to(torch.float16)

    new_span = (new_log_max - new_log_min).unsqueeze(1).clamp(min=1e-10)
    normalized = (log_blocks - new_log_min.unsqueeze(1)) / new_span
    continuous = normalized * n_buckets

    # Stochastic rounding
    floor_idx = continuous.floor()
    frac = continuous - floor_idx
    new_indices = (
        (floor_idx + (torch.rand_like(frac) < frac).float())
        .clamp(0, n_buckets - 1)
        .to(torch.uint8)
    )

    # Return flat v (trimmed) for denominator computation
    v_flat = v_blocks.reshape(-1)[:original_numel]

    return new_indices.reshape(-1), new_scales, v_flat
