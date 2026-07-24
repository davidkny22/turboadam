"""1Q: second moment (v) compression.

N-bit log-scale quantization for all parameters.
Compress-every-step architecture.
"""

from __future__ import annotations

import math

import torch

from turboadam.quantize import dequantize_logscale_nbits, quantize_logscale_nbits
from turboadam.utils import BLOCK_SIZE, packed_num_bytes, pad_to_blocks


def init_compressed_v(
    shape: torch.Size | tuple[int, ...],
    *,
    device: torch.device,
    n_bits: int = 4,
    block_size: int = BLOCK_SIZE,
    packed: bool = True,
) -> dict:
    """Create the near-zero initial v state without a full-size fp32 tensor.

    All indices decode to the same value when log_min == log_max, so an
    all-zero byte stream plus one fp16 endpoint pair per block is sufficient.
    """
    if n_bits not in (2, 3, 4, 6, 8):
        raise ValueError(f"n_bits must be one of {{2, 3, 4, 6, 8}}, got {n_bits}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")

    original_shape = torch.Size(shape)
    original_length = math.prod(original_shape)
    num_blocks = (original_length + block_size - 1) // block_size
    padded_numel = num_blocks * block_size
    index_count = (
        packed_num_bytes(padded_numel, n_bits) if packed else padded_numel
    )
    indices = torch.zeros(index_count, dtype=torch.uint8, device=device)
    log_floor = float(math.log(1e-30))
    scales = torch.full(
        (num_blocks, 2), log_floor, dtype=torch.float16, device=device
    )
    return {
        "indices": indices,
        "scales": scales,
        "n_bits": n_bits,
        "packed": packed,
        "codec_version": 2,
        "original_shape": original_shape,
        "original_length": original_length,
        "block_size": block_size,
    }


def compress_v_logscale(
    v: torch.Tensor,
    n_bits: int = 3,
    block_size: int = BLOCK_SIZE,
    stochastic_round: bool = False,
    packed: bool = False,
) -> dict:
    """Compress a second-moment tensor with n-bit log-scale quantization.

    Parameters
    ----------
    v:
        Second-moment tensor of any shape. Values are converted to fp32 and
        padded to complete quantization blocks before encoding.
    n_bits:
        Number of bits per element, which determines the number of log buckets.
    block_size:
        Number of elements per independent quantization block.
    stochastic_round:
        Whether to use stochastic rounding when assigning bucket indices.
    packed:
        Whether to bit-pack indices to their physical n-bit representation.
        The optimizer enables this; False remains the public compatibility mode.

    Returns
    -------
    dict
        Compressed representation containing indices, fp16 per-block endpoints,
        format metadata, original shape, original length, and block size.
    """
    original_shape = v.shape
    v_flat = v.reshape(-1).float()

    if v_flat.numel() == 0:
        v_padded = v_flat
        original_length = 0
    else:
        remainder = v_flat.numel() % block_size
        if remainder:
            # Use the minimum of the *final partial block*, not a global minimum.
            # This preserves that block's dynamic range and stays on-device.
            pad_value = v_flat[-remainder:].amin().clamp_min(1e-38)
        else:
            pad_value = v_flat.new_tensor(1e-38)
        v_padded, original_length = pad_to_blocks(
            v_flat, block_size, pad_value=pad_value
        )

    indices, scales, encoded_bits = quantize_logscale_nbits(
        v_padded,
        n_bits=n_bits,
        block_size=block_size,
        stochastic_round=stochastic_round,
        packed=packed,
    )
    return {
        "indices": indices,
        "scales": scales,
        "n_bits": encoded_bits,
        "packed": packed,
        "codec_version": 2,
        "original_shape": original_shape,
        "original_length": original_length,
        "block_size": block_size,
    }


def decompress_v(compressed: dict) -> torch.Tensor:
    """Reconstruct fp32 v from a compressed representation."""
    v_flat = dequantize_logscale_nbits(
        compressed["indices"],
        compressed["scales"],
        n_bits=compressed["n_bits"],
        block_size=compressed["block_size"],
        original_numel=compressed["original_length"],
        packed=compressed.get("packed"),
    )
    return v_flat.reshape(compressed["original_shape"])
