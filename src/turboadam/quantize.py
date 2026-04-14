"""2-bit log-scale quantization for non-matrix parameter second moments.

Per 128-element block:
  1. Compute log(v_min) and log(v_max)
  2. Define 4 evenly-spaced buckets on log scale
  3. Quantize each element to nearest bucket index (2 bits)
  4. Store: 2 bits/element + 2 fp16 scalars (min, max) per block = 2.25 bits/param

Log-scale chosen because v is strictly positive and spans orders of magnitude.
Adam's update rule (dividing by √v) is most sensitive to small v values;
log-scale spacing allocates more resolution there.
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
    indices = (normalized * 4.0).clamp(0, 3.999).to(torch.long)  # (num_blocks, block_size)

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
