"""Shared utilities: block operations, parameter routing, metadata helpers.

Constants:
  BLOCK_SIZE = 128          # GPU warp-aligned quantization block
  MATRIX_NUMEL_THRESHOLD = 10_000

Parameter routing:
  is_matrix_param(p) -> ndim >= 2 and numel > MATRIX_NUMEL_THRESHOLD

Block helpers:
  pad_to_blocks(tensor, block_size) -> (padded, original_length)
  unpad_from_blocks(padded, original_length) -> tensor
  packed_num_bytes(numel, n_bits) -> exact packed byte count
"""

from __future__ import annotations

import torch

BLOCK_SIZE = 128
MATRIX_NUMEL_THRESHOLD = 10_000


def is_matrix_param(param: torch.Tensor) -> bool:
    """Return True if param should use the matrix compression path."""
    return param.ndim >= 2 and param.numel() > MATRIX_NUMEL_THRESHOLD


def packed_num_bytes(numel: int, n_bits: int) -> int:
    """Return the exact number of bytes required for ``numel`` n-bit values."""
    if numel < 0:
        raise ValueError(f"numel must be non-negative, got {numel}")
    if n_bits <= 0 or n_bits > 8:
        raise ValueError(f"n_bits must be in [1, 8], got {n_bits}")
    return (numel * n_bits + 7) // 8


def pad_to_blocks(
    tensor: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    pad_value: float | torch.Tensor = 0.0,
) -> tuple[torch.Tensor, int]:
    """Pad a flat 1-D tensor to the next multiple of block_size.

    Args:
        tensor:     1-D tensor to pad.
        block_size: Block alignment target (default: BLOCK_SIZE = 128).
        pad_value:  Value to fill the padding with (default: 0.0). A scalar
                    tensor is accepted so CUDA callers can pad from a reduction
                    result without a synchronizing ``.item()`` call.

    Returns:
        (padded_tensor, original_length) where padded_tensor.shape[0] is a
        multiple of block_size and original_length is len(tensor).
    """
    if tensor.ndim != 1:
        raise ValueError(f"tensor must be flat (1-D), got shape {tuple(tensor.shape)}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    original_length = tensor.shape[0]
    remainder = original_length % block_size
    if remainder == 0:
        return tensor, original_length

    pad_size = block_size - remainder
    if isinstance(pad_value, torch.Tensor):
        if pad_value.numel() != 1:
            raise ValueError("tensor pad_value must contain exactly one element")
        fill = pad_value.to(device=tensor.device, dtype=tensor.dtype).reshape(1).expand(
            pad_size
        )
    else:
        fill = tensor.new_full((pad_size,), pad_value)

    return torch.cat((tensor, fill)), original_length


def unpad_from_blocks(padded: torch.Tensor, original_length: int) -> torch.Tensor:
    """Strip the padding added by pad_to_blocks.

    Args:
        padded:          Padded 1-D tensor (output of pad_to_blocks).
        original_length: Number of elements before padding.

    Returns:
        Tensor with exactly original_length elements.
    """
    return padded[:original_length]
