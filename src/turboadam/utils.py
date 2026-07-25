"""Shared block operations, parameter routing, and state accounting."""

from __future__ import annotations

import math

import torch

BLOCK_SIZE = 128
MATRIX_NUMEL_THRESHOLD = 10_000


def is_matrix_param(param: torch.Tensor) -> bool:
    """Return whether a parameter is a large matrix-like tensor."""
    return param.ndim >= 2 and param.numel() > MATRIX_NUMEL_THRESHOLD


def ceil_div(numerator: int, denominator: int) -> int:
    """Return the ceiling of a non-negative integer division."""
    if numerator < 0:
        raise ValueError(f"numerator must be non-negative, got {numerator}")
    if denominator <= 0:
        raise ValueError(f"denominator must be positive, got {denominator}")
    return (numerator + denominator - 1) // denominator


def validate_block_size(block_size: int) -> None:
    """Validate a block size accepted by the reference implementation."""
    if not isinstance(block_size, int) or isinstance(block_size, bool):
        raise TypeError(
            f"block_size must be an integer, got {type(block_size).__name__}"
        )
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")


def pad_to_blocks(
    tensor: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    pad_value: float | torch.Tensor = 0.0,
) -> tuple[torch.Tensor, int]:
    """Pad a flat tensor to a whole number of blocks."""
    validate_block_size(block_size)
    if tensor.ndim != 1:
        raise ValueError(f"tensor must be 1-D, got shape {tuple(tensor.shape)}")

    original_length = tensor.shape[0]
    remainder = original_length % block_size
    if remainder == 0:
        return tensor, original_length

    pad_size = block_size - remainder
    padded = tensor.new_empty(original_length + pad_size)
    padded[:original_length].copy_(tensor)
    if isinstance(pad_value, torch.Tensor):
        if pad_value.numel() != 1:
            raise ValueError("tensor pad_value must contain exactly one element")
        scalar = pad_value.to(device=tensor.device, dtype=tensor.dtype).reshape(())
        padded[original_length:].copy_(scalar.expand(pad_size))
    else:
        padded[original_length:].fill_(pad_value)
    return padded, original_length


def unpad_from_blocks(padded: torch.Tensor, original_length: int) -> torch.Tensor:
    """Strip block padding from a tensor."""
    if original_length < 0 or original_length > padded.shape[0]:
        raise ValueError(
            f"original_length must lie in [0, {padded.shape[0]}], got {original_length}"
        )
    return padded[:original_length]


def state_tensor_bytes(obj: object) -> int:
    """Count unique tensor-storage bytes reachable from nested optimizer state."""
    seen: set[tuple[str, int | None, int]] = set()

    def visit(value: object) -> int:
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            key = (value.device.type, value.device.index, storage.data_ptr())
            if key in seen:
                return 0
            seen.add(key)
            return storage.nbytes()
        if isinstance(value, dict):
            return sum(visit(item) for item in value.values())
        if isinstance(value, (tuple, list, set)):
            return sum(visit(item) for item in value)
        return 0

    return visit(obj)


def finite_scalar(value: float, name: str, *, non_negative: bool = False) -> float:
    """Validate and return a finite Python scalar."""
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value}")
    if non_negative and value < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value
