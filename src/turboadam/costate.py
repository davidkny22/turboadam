"""CoState — first moment (m) compression.

Gradient-residual decomposition: m = α·g + δ
  - α = (m·g) / (g·g)  — scalar per parameter tensor
  - δ = m - α·g        — residual orthogonal to current gradient

Residual δ is partitioned into 128-element blocks and classified:
  - Null costate    (r < τ₀): store 1 bit in bitmap
  - Phase costate   (τ₀ ≤ r < τ₁): store 1-bit sign per element
  - Amplitude costate (r ≥ τ₁): store 1-bit sign + fp16 block scale

Adaptive thresholds: τ₀ = P_10(r), τ₁ = P_90(r) per parameter tensor per step.
No warmup required — EMA error-washing handles cold-start.
"""

import math
import torch

from turboadam.utils import pad_to_blocks, unpad_from_blocks, BLOCK_SIZE


def decompose(m: torch.Tensor, g: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Decompose momentum into gradient-aligned component and residual.

    m = α·g + δ  where α = (m·g) / (g·g)

    Args:
        m: First moment tensor (any shape, will be treated as flat).
        g: Gradient tensor (same shape as m).

    Returns:
        (alpha, delta) where alpha is a float scalar and delta has
        the same shape as m.
    """
    m_flat = m.reshape(-1)
    g_flat = g.reshape(-1)
    g_dot_g = g_flat.dot(g_flat).item()
    if g_dot_g == 0.0:
        return 0.0, m.clone()
    alpha = m_flat.dot(g_flat).item() / g_dot_g
    delta = m - alpha * g
    return alpha, delta


# ---------------------------------------------------------------------------
# Block ratio computation
# ---------------------------------------------------------------------------

def compute_block_ratios(
    delta: torch.Tensor,
    m: torch.Tensor,
    block_size: int = BLOCK_SIZE,
) -> torch.Tensor:
    """Compute per-block ratio r = norm(delta_block) / norm(m_block).

    Args:
        delta:      Residual tensor (any shape).
        m:          First moment tensor (same shape as delta).
        block_size: Elements per block (default: BLOCK_SIZE = 128).

    Returns:
        1-D float32 tensor of length num_blocks, each entry in [0, ∞).
        Blocks where norm(m_block) == 0 get ratio 0.
    """
    delta_flat = delta.reshape(-1).float()
    m_flat = m.reshape(-1).float()

    delta_padded, orig_len = pad_to_blocks(delta_flat, block_size)
    m_padded, _ = pad_to_blocks(m_flat, block_size)

    num_blocks = delta_padded.shape[0] // block_size
    delta_blocks = delta_padded.reshape(num_blocks, block_size)
    m_blocks = m_padded.reshape(num_blocks, block_size)

    delta_norms = delta_blocks.norm(dim=1)   # (num_blocks,)
    m_norms = m_blocks.norm(dim=1)           # (num_blocks,)

    # Guard: where m_norm is zero, ratio is 0
    safe_m_norms = m_norms.clone()
    safe_m_norms[safe_m_norms == 0.0] = 1.0  # avoid division by zero
    ratios = delta_norms / safe_m_norms
    ratios[m_norms == 0.0] = 0.0

    return ratios


# ---------------------------------------------------------------------------
# Threshold computation
# ---------------------------------------------------------------------------

def compute_thresholds(ratios: torch.Tensor) -> tuple[float, float]:
    """Compute adaptive thresholds as 10th and 90th percentiles of ratios.

    Args:
        ratios: 1-D float tensor of per-block ratios.

    Returns:
        (tau0, tau1) where tau0 = P_10(ratios), tau1 = P_90(ratios).
    """
    tau0 = torch.quantile(ratios.float(), 0.10).item()
    tau1 = torch.quantile(ratios.float(), 0.90).item()
    return tau0, tau1


# ---------------------------------------------------------------------------
# Block classification
# ---------------------------------------------------------------------------

def classify_blocks(
    ratios: torch.Tensor,
    tau0: float,
    tau1: float,
) -> torch.Tensor:
    """Assign costate label to each block based on its ratio.

    Labels:
        0 (null)      : r < tau0
        1 (phase)     : tau0 <= r < tau1
        2 (amplitude) : r >= tau1

    Args:
        ratios: 1-D float tensor of per-block ratios.
        tau0:   Lower threshold (10th percentile).
        tau1:   Upper threshold (90th percentile).

    Returns:
        uint8 tensor of labels, same length as ratios.
    """
    labels = torch.zeros(ratios.shape[0], dtype=torch.uint8)
    labels[ratios >= tau0] = 1  # phase (will be overwritten for amplitude)
    labels[ratios >= tau1] = 2  # amplitude
    return labels


# ---------------------------------------------------------------------------
# Encoding and decoding
# ---------------------------------------------------------------------------

def _pack_signs(values: torch.Tensor) -> torch.Tensor:
    """Pack sign bits (1 if negative, 0 if non-negative) into uint8, 8 bits/byte.

    Args:
        values: 1-D float tensor of arbitrary length.

    Returns:
        uint8 tensor of length ceil(len(values) / 8).
    """
    n = values.shape[0]
    # Pad to multiple of 8
    pad = (8 - n % 8) % 8
    if pad > 0:
        values = torch.cat([values, values.new_zeros(pad)])
    sign_bits = (values < 0).to(torch.uint8)  # 1 if negative
    sign_bits = sign_bits.reshape(-1, 8)  # (num_bytes, 8)
    # Pack: bit 7 is index 0, bit 0 is index 7
    multipliers = torch.tensor([128, 64, 32, 16, 8, 4, 2, 1], dtype=torch.uint8)
    packed = (sign_bits * multipliers).sum(dim=1).to(torch.uint8)
    return packed


def _unpack_signs(packed: torch.Tensor, n: int) -> torch.Tensor:
    """Unpack sign bits from uint8 bytes back to +1/-1 float values.

    Args:
        packed: uint8 tensor of packed sign bytes.
        n:      Number of elements to unpack (may be less than len(packed)*8).

    Returns:
        float32 tensor of length n with values +1 or -1.
    """
    packed_int = packed.to(torch.int32)
    # Extract 8 bits per byte (bit 7 first)
    bits = torch.zeros(packed.shape[0] * 8, dtype=torch.float32)
    for i, shift in enumerate([7, 6, 5, 4, 3, 2, 1, 0]):
        byte_idx = torch.arange(packed.shape[0])
        bits[byte_idx * 8 + i] = ((packed_int >> shift) & 1).float()
    bits = bits[:n]  # trim to original length
    # Convert: 1 -> -1, 0 -> +1
    signs = 1.0 - 2.0 * bits
    return signs


def encode_blocks(
    delta: torch.Tensor,
    labels: torch.Tensor,
    block_size: int = BLOCK_SIZE,
) -> dict:
    """Encode delta into per-block compressed representation.

    Args:
        delta:      Residual tensor (any shape, flattened internally).
        labels:     uint8 costate labels, one per block.
        block_size: Elements per block.

    Returns:
        dict with:
            labels      (uint8)  : costate label per block
            sign_packed (uint8)  : packed sign bits, ceil(numel/8) bytes
            block_norms (float32): L2 norm of each delta block
            scales      (float16): per-block scale (block norm as fp16)
    """
    delta_flat = delta.reshape(-1).float()
    original_numel = delta_flat.shape[0]

    delta_padded, _ = pad_to_blocks(delta_flat, block_size)
    num_blocks = delta_padded.shape[0] // block_size
    delta_blocks = delta_padded.reshape(num_blocks, block_size)

    # Per-block L2 norms
    block_norms = delta_blocks.norm(dim=1).float()  # (num_blocks,)

    # Per-block fp16 scales (same as block norms but in fp16)
    scales = block_norms.to(torch.float16)  # (num_blocks,)

    # Pack sign bits for the original (un-padded) elements
    sign_packed = _pack_signs(delta_flat)

    return {
        "labels": labels,
        "sign_packed": sign_packed,
        "block_norms": block_norms,
        "scales": scales,
    }


def decode_blocks(
    encoded: dict,
    alpha: float,
    g: torch.Tensor,
    block_size: int = BLOCK_SIZE,
    original_numel: int = None,
) -> torch.Tensor:
    """Reconstruct approximated m = alpha*g + delta_hat from encoded representation.

    Per-costate delta_hat reconstruction:
        Null (0)      : delta_hat_block = 0
        Phase (1)     : delta_hat_block = (norm(delta_block)/sqrt(block_size)) * sign(delta_block)
        Amplitude (2) : delta_hat_block = fp16_scale * sign(delta_block)

    Args:
        encoded:        dict returned by encode_blocks.
        alpha:          Scalar float from decompose().
        g:              Gradient tensor (same original shape as delta).
        block_size:     Elements per block.
        original_numel: Number of elements in the original delta (before padding).

    Returns:
        Reconstructed m tensor with the same shape as g.
    """
    g_flat = g.reshape(-1).float()
    if original_numel is None:
        original_numel = g_flat.shape[0]

    labels = encoded["labels"]         # (num_blocks,)
    sign_packed = encoded["sign_packed"]
    block_norms = encoded["block_norms"].float()  # (num_blocks,)
    scales = encoded["scales"].to(torch.float32)  # (num_blocks,)
    num_blocks = labels.shape[0]

    # Unpack all sign bits for original elements
    signs_flat = _unpack_signs(sign_packed, original_numel)  # (original_numel,)

    # Pad signs to block boundary for block-wise processing
    signs_padded, _ = pad_to_blocks(signs_flat, block_size)
    signs_blocks = signs_padded.reshape(num_blocks, block_size)

    # Build delta_hat block by block
    delta_hat_padded = torch.zeros(num_blocks * block_size, dtype=torch.float32)
    delta_hat_blocks = delta_hat_padded.reshape(num_blocks, block_size)

    for i in range(num_blocks):
        label = labels[i].item()
        if label == 0:
            # Null: no delta contribution
            pass
        elif label == 1:
            # Phase: uniform-magnitude sign vector
            scale = block_norms[i] / math.sqrt(block_size)
            delta_hat_blocks[i] = scale * signs_blocks[i]
        else:
            # Amplitude: fp16 per-block scale
            scale = scales[i]
            delta_hat_blocks[i] = scale * signs_blocks[i]

    # Trim to original numel
    delta_hat = delta_hat_padded[:original_numel]

    # Reconstruct m
    result = alpha * g_flat + delta_hat
    return result.reshape(g.shape)
