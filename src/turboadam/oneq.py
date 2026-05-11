"""1Q — second moment (v) compression.

N-bit log-scale quantization for all parameters.
Compress-every-step architecture.
"""

import torch

from turboadam.utils import pad_to_blocks, BLOCK_SIZE
from turboadam.quantize import quantize_logscale_nbits, dequantize_logscale_nbits


def compress_v_logscale(
    v: torch.Tensor,
    n_bits: int = 3,
    block_size: int = BLOCK_SIZE,
    stochastic_round: bool = False,
) -> dict:
    """Compress v via n-bit log-scale quantization.

    Args:
        v:               Second-moment tensor (any shape, fp32, positive).
        n_bits:          Bits per element (default 3 = 8 buckets).
        block_size:      Quantization block size.
        stochastic_round: Use stochastic rounding (essential for compress-every-step).

    Returns:
        Compressed dict with 'indices', 'scales', metadata.
    """
    original_shape = v.shape
    v_flat = v.reshape(-1).float()
    v_min = v_flat.min().item()
    pad_value = max(v_min, 1e-38)
    v_padded, original_length = pad_to_blocks(v_flat, block_size, pad_value=pad_value)
    indices, scales, nb = quantize_logscale_nbits(
        v_padded, n_bits=n_bits, block_size=block_size, stochastic_round=stochastic_round,
    )
    return {
        "indices": indices,
        "scales": scales,
        "n_bits": n_bits,
        "original_shape": original_shape,
        "original_length": original_length,
        "block_size": block_size,
    }


def decompress_v(compressed: dict) -> torch.Tensor:
    """Reconstruct fp32 v from a compressed representation.

    Args:
        compressed: Dict produced by compress_v_logscale.

    Returns:
        fp32 tensor with the same shape as the original v.
    """
    v_flat = dequantize_logscale_nbits(
        compressed["indices"],
        compressed["scales"],
        n_bits=compressed["n_bits"],
        block_size=compressed["block_size"],
        original_numel=compressed["original_length"],
    )
    return v_flat.reshape(compressed["original_shape"])
