"""Shared utilities: block operations, parameter routing, metadata helpers.

Constants:
  BLOCK_SIZE = 128          # GPU warp-aligned quantization block
  MATRIX_NUMEL_THRESHOLD = 10_000
  SVD_RANK_DEFAULT = 8

Parameter routing:
  is_matrix_param(p) -> ndim >= 2 and numel > MATRIX_NUMEL_THRESHOLD

Block helpers:
  pad_to_blocks(tensor, block_size) -> (padded, original_length)
  unpad_from_blocks(padded, original_length) -> tensor
"""

import math
import torch

BLOCK_SIZE = 128
MATRIX_NUMEL_THRESHOLD = 10_000


def is_matrix_param(param: torch.Tensor) -> bool:
    """Return True if param should use the matrix (SVD) compression path."""
    return param.ndim >= 2 and param.numel() > MATRIX_NUMEL_THRESHOLD


def pad_to_blocks(
    tensor: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    pad_value: float = 0.0,
) -> tuple[torch.Tensor, int]:
    """Pad a flat 1-D tensor to the next multiple of block_size.

    Args:
        tensor:     1-D tensor to pad.
        block_size: Block alignment target (default: BLOCK_SIZE = 128).
        pad_value:  Value to fill the padding with (default: 0.0).
                    For log-scale quantization of positive tensors, callers
                    should pass the tensor minimum to avoid polluting block
                    statistics with log(-inf) from zero padding.

    Returns:
        (padded_tensor, original_length) where padded_tensor.shape[0] is a
        multiple of block_size and original_length is len(tensor).
    """
    original_length = tensor.shape[0]
    remainder = original_length % block_size
    if remainder == 0:
        return tensor, original_length
    pad_size = block_size - remainder
    fill = tensor.new_full((pad_size,), pad_value)
    padded = torch.cat([tensor, fill])
    return padded, original_length


def unpad_from_blocks(padded: torch.Tensor, original_length: int) -> torch.Tensor:
    """Strip the zero-padding added by pad_to_blocks.

    Args:
        padded:          Padded 1-D tensor (output of pad_to_blocks).
        original_length: Number of elements before padding.

    Returns:
        Tensor with exactly original_length elements.
    """
    return padded[:original_length]
