"""1Q packed block-log compression for Adam's second moment."""

from __future__ import annotations

import torch

from turboadam.quantize import (
    MIN_POSITIVE,
    decompress_v_state,
    initialize_v_state,
    quantize_logscale,
    validate_v_state,
)
from turboadam.utils import BLOCK_SIZE, pad_to_blocks, validate_block_size


def compress_v_logscale(
    second_moment: torch.Tensor,
    n_bits: int = 4,
    block_size: int = BLOCK_SIZE,
    stochastic_round: bool = False,
    *,
    seed: int = 0,
) -> dict:
    """Compress a non-empty second-moment tensor with packed log levels."""
    validate_block_size(block_size)
    if second_moment.numel() == 0:
        raise ValueError("cannot compress an empty second-moment tensor")
    original_shape = tuple(second_moment.shape)
    flat = second_moment.reshape(-1).float()
    remainder = flat.numel() % block_size
    pad_value = flat[-remainder:].amin() if remainder else MIN_POSITIVE
    padded, original_length = pad_to_blocks(flat, block_size, pad_value=pad_value)
    indices, scales = quantize_logscale(
        padded,
        n_bits,
        block_size,
        stochastic_round=stochastic_round,
        seed=seed,
    )
    state = {
        "indices": indices,
        "scales": scales,
        "n_bits": n_bits,
        "original_shape": original_shape,
        "original_length": original_length,
        "block_size": block_size,
    }
    validate_v_state(state)
    return state


def initialize_v_logscale(
    shape: torch.Size | tuple[int, ...],
    *,
    device: torch.device,
    n_bits: int = 4,
    block_size: int = BLOCK_SIZE,
) -> dict:
    """Initialize a compact near-zero second-moment state."""
    return initialize_v_state(
        shape,
        device=device,
        n_bits=n_bits,
        block_size=block_size,
    )


def decompress_v(state: dict) -> torch.Tensor:
    """Reconstruct a packed second moment as fp32."""
    return decompress_v_state(state)


__all__ = ["compress_v_logscale", "decompress_v", "initialize_v_logscale"]
