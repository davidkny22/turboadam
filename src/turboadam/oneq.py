"""1Q — second moment (v) compression.

Two paths based on parameter shape:
  - Matrix path (ndim >= 2 and numel > 10,000): low-rank SVD via svd.py
  - Non-matrix path: 2-bit log-scale quantization via quantize.py

Warmup detection monitors per-step relative change in v.
Freeze-refresh cycle runs every refresh_interval steps.

Public API
----------
check_warmup(v_current, v_previous, threshold) -> bool
compress_v(param, v, rank, block_size) -> dict
decompress_v(compressed) -> Tensor
refresh_v(compressed, new_v, param, rank, block_size) -> dict
"""

import torch

from turboadam.utils import is_matrix_param, pad_to_blocks, unpad_from_blocks, BLOCK_SIZE
from turboadam.svd import svd_compress, svd_reconstruct
from turboadam.quantize import quantize_logscale, dequantize_logscale


# ---------------------------------------------------------------------------
# Warmup detection
# ---------------------------------------------------------------------------

def check_warmup(
    v_current: torch.Tensor,
    v_previous: torch.Tensor,
    threshold: float,
) -> bool:
    """Return True when v has stabilised and warmup should end.

    Computes the relative change in v:

        ||v_current - v_previous|| / ||v_current||

    and returns True when the ratio is *strictly below* ``threshold``.

    Args:
        v_current:  Second-moment tensor at step t.
        v_previous: Second-moment tensor at step t-1.
        threshold:  Relative-change threshold ε_warmup (default 0.01).

    Returns:
        True if warmup is complete (relative change < threshold), else False.
    """
    v_norm = v_current.norm().item()
    if v_norm == 0.0:
        # Both tensors are zero → no change; treat as stable.
        return True
    delta_norm = (v_current - v_previous).norm().item()
    relative_change = delta_norm / v_norm
    return relative_change < threshold


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _compress_svd(
    param: torch.Tensor,
    v: torch.Tensor,
    rank: int,
) -> dict:
    """Compress v via truncated SVD.

    Reshapes v to param.shape (to match matrix dimensions), runs
    svd_compress on the 2-D view, and stores factors plus the original
    shape so decompress can restore it exactly.

    For tensors with ndim > 2 the first two dimensions are merged so the
    SVD sees a 2-D matrix; the original shape is recorded for restoration.
    """
    original_shape = v.shape
    # Flatten to 2-D: (first_dim, rest) for ndim > 2
    if v.ndim == 2:
        v_2d = v.reshape(param.shape[0], -1).float()
    else:
        v_2d = v.reshape(original_shape[0], -1).float()

    U_r, S_r, Vh_r = svd_compress(v_2d, rank=rank)
    return {
        "type": "svd",
        "U": U_r,
        "S": S_r,
        "Vh": Vh_r,
        "original_shape": original_shape,
    }


def _decompress_svd(compressed: dict) -> torch.Tensor:
    """Reconstruct v from SVD factors."""
    v_2d = svd_reconstruct(compressed["U"], compressed["S"], compressed["Vh"])
    return v_2d.reshape(compressed["original_shape"])


def _compress_logscale(
    v: torch.Tensor,
    block_size: int,
) -> dict:
    """Compress v via 2-bit log-scale block quantization."""
    original_shape = v.shape
    v_flat = v.reshape(-1).float()
    # Pad with the minimum value so the partial-block padding does not
    # inject log(-inf) = -87.5 into the block statistics, which would
    # collapse all bucket boundaries toward the padding value.
    v_min = v_flat.min().item()
    pad_value = max(v_min, 1e-38)  # guard against exact-zero v_flat
    v_padded, original_length = pad_to_blocks(v_flat, block_size, pad_value=pad_value)
    packed, scales = quantize_logscale(v_padded, block_size=block_size)
    return {
        "type": "logscale",
        "packed": packed,
        "scales": scales,
        "original_shape": original_shape,
        "original_length": original_length,
        "block_size": block_size,
    }


def _decompress_logscale(compressed: dict) -> torch.Tensor:
    """Reconstruct v from 2-bit log-scale quantized representation."""
    v_flat = dequantize_logscale(
        compressed["packed"],
        compressed["scales"],
        block_size=compressed["block_size"],
        original_numel=compressed["original_length"],
    )
    return v_flat.reshape(compressed["original_shape"])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compress_v(
    param: torch.Tensor,
    v: torch.Tensor,
    rank: int = 8,
    block_size: int = BLOCK_SIZE,
) -> dict:
    """Compress a second-moment tensor v, routing based on param shape.

    Routing rule (shared with ReQuant / QKVO-T):
    - ndim >= 2 AND numel > 10,000 → low-rank SVD path
    - everything else             → 2-bit log-scale quantization path

    Args:
        param:      The weight parameter whose shape governs routing.
        v:          Second-moment tensor (same shape as param, fp32, positive).
        rank:       SVD approximation rank for the matrix path.
        block_size: Quantization block size for the non-matrix path.

    Returns:
        Compressed representation as a dict with at minimum a ``"type"`` key
        set to ``"svd"`` or ``"logscale"``.
    """
    if is_matrix_param(param):
        return _compress_svd(param, v, rank)
    else:
        return _compress_logscale(v, block_size)


def decompress_v(compressed: dict) -> torch.Tensor:
    """Reconstruct fp32 v from a compressed representation.

    Dispatches on ``compressed["type"]``.

    Args:
        compressed: Dict produced by ``compress_v`` or ``refresh_v``.

    Returns:
        fp32 tensor with the same shape as the original v.
    """
    # Dispatch on structural keys rather than the 'type' string, because
    # PyTorch's load_state_dict _cast() corrupts strings inside nested dicts.
    if "U" in compressed:
        return _decompress_svd(compressed)
    elif "packed" in compressed:
        return _decompress_logscale(compressed)
    else:
        raise ValueError(f"Cannot determine compressed type from keys: {list(compressed.keys())}")


def refresh_v(
    compressed: dict,
    new_v: torch.Tensor,
    param: torch.Tensor,
    rank: int = 8,
    block_size: int = BLOCK_SIZE,
) -> dict:
    """Re-compress v with fresh data (called every K steps).

    Discards the old compressed representation and produces a new one
    from ``new_v``. This is the freeze-refresh cycle trigger.

    Args:
        compressed: Existing compressed dict (used only to check type, not data).
        new_v:      Freshly computed second-moment tensor (same shape as param).
        param:      The weight parameter (for routing).
        rank:       SVD approximation rank.
        block_size: Quantization block size.

    Returns:
        New compressed dict from the fresh data.
    """
    return compress_v(param, new_v, rank=rank, block_size=block_size)
