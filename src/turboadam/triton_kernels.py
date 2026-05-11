"""Triton kernels for TurboAdam — fused operations to minimize kernel launch overhead."""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _rand_uniform_u32(seed, values):
    x = values.to(tl.uint32) ^ seed
    x = (x ^ (x >> 16)) * 0x7FEB352D
    x = (x ^ (x >> 15)) * 0x846CA68B
    x = x ^ (x >> 16)
    return x.to(tl.float32) * 2.3283064365386963e-10


# ---------------------------------------------------------------------------
# Fused v update: decompress → EMA → recompress in a single kernel
# ---------------------------------------------------------------------------
# Each program instance handles one block of BLOCK_SIZE elements.
# Input:  old indices (uint8), old scales (fp16 log_min/log_max), gradient, beta2
# Output: new indices (uint8), new scales (fp16 log_min/log_max), v values (fp32)
#
# Flow per block:
#   1. Load old indices, compute old log_min/log_max → reconstruct v_old via exp
#   2. Load gradient slice, compute g²
#   3. v_new = β₂ · v_old + (1-β₂) · g²
#   4. Compute log(v_new), find block min/max
#   5. Normalize to [0, n_buckets), stochastic round, store new indices
#   6. Store new scales (log_min, log_max) and v_new values


@triton.jit
def _fused_v_update_kernel(
    # Inputs
    old_indices_ptr,
    old_scales_ptr,
    grad_ptr,
    rand_buf_ptr,
    # Outputs
    new_indices_ptr,
    new_scales_ptr,
    v_out_ptr,
    # Params
    beta2: tl.constexpr,
    one_minus_beta2: tl.constexpr,
    n_buckets: tl.constexpr,
    original_numel,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    # Each program processes multiple blocks in a strided loop
    for block_id in range(pid, num_blocks, n_programs):
        block_start = block_id * BLOCK_SIZE
        offs = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offs < original_numel

        # Decompress
        old_idx = tl.load(old_indices_ptr + offs).to(tl.int32)
        log_min = tl.load(old_scales_ptr + block_id * 2).to(tl.float32)
        log_max = tl.load(old_scales_ptr + block_id * 2 + 1).to(tl.float32)
        span = log_max - log_min
        log_v = log_min + (old_idx.to(tl.float32) + 0.5) / n_buckets * span
        v_old = tl.exp(log_v)

        # g²
        g = tl.load(grad_ptr + offs, mask=mask, other=0.0)
        g_sq = g * g

        # EMA
        v_new = beta2 * v_old + one_minus_beta2 * g_sq

        # Recompress — mask padded elements out of block min/max so partial
        # blocks don't have their statistics corrupted by padding values.
        log_v_new = tl.log(tl.maximum(v_new, 1e-38))
        log_v_for_min = tl.where(mask, log_v_new, 1e38)
        log_v_for_max = tl.where(mask, log_v_new, -1e38)
        new_log_min = tl.min(log_v_for_min, axis=0)
        new_log_max = tl.max(log_v_for_max, axis=0)
        new_span = tl.maximum(new_log_max - new_log_min, 1e-10)
        normalized = (log_v_new - new_log_min) / new_span
        continuous = normalized * n_buckets

        # Stochastic rounding — load from pre-generated random buffer
        floor_idx = tl.math.floor(continuous)
        frac = continuous - floor_idx
        rand_val = tl.load(rand_buf_ptr + offs, mask=mask, other=0.5)
        rounded = floor_idx + (rand_val < frac).to(tl.float32)
        new_idx = tl.minimum(tl.maximum(rounded, 0.0), (n_buckets - 1) * 1.0).to(
            tl.uint8
        )

        # Store
        tl.store(new_indices_ptr + offs, new_idx)
        tl.store(new_scales_ptr + block_id * 2, new_log_min.to(tl.float16))
        tl.store(new_scales_ptr + block_id * 2 + 1, new_log_max.to(tl.float16))
        tl.store(v_out_ptr + offs, v_new, mask=mask)


def triton_fused_v_update(
    indices: torch.Tensor,
    scales: torch.Tensor,
    grad: torch.Tensor,
    beta2: float,
    n_bits: int,
    block_size: int,
    original_numel: int,
    rand_buf: torch.Tensor | None = None,
    out_indices: torch.Tensor | None = None,
    out_scales: torch.Tensor | None = None,
    v_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton-accelerated fused v update: decompress → EMA → recompress.

    Drop-in replacement for quantize.fused_v_update.

    Args:
        rand_buf: Pre-generated random buffer (same size as indices).
            Filled with torch.rand() before each call. CUDA graph safe
            because the buffer address is stable, only contents change.

    Returns:
        (new_indices, new_scales, v_flat) where v_flat has shape (original_numel,).
    """
    n_buckets = 2**n_bits
    num_blocks = scales.shape[0]
    padded_numel = num_blocks * block_size

    # Allocate outputs unless the caller supplies graph-stable buffers.
    new_indices = torch.empty_like(indices) if out_indices is None else out_indices
    new_scales = torch.empty_like(scales) if out_scales is None else out_scales
    if v_out is None:
        v_out = torch.empty(padded_numel, dtype=torch.float32, device=grad.device)
    if rand_buf is None:
        rand_buf = torch.rand(padded_numel, dtype=torch.float32, device=grad.device)

    # Flatten gradient for kernel access (cast to fp32 for numerical stability)
    grad_flat = grad.reshape(-1).float().contiguous()

    # Limit grid size: cap at ~2048 programs, each processes multiple blocks
    max_programs = 2048
    n_programs = min(num_blocks, max_programs)

    _fused_v_update_kernel[(n_programs,)](
        indices,
        scales.reshape(-1),
        grad_flat,
        rand_buf,
        new_indices,
        new_scales.reshape(-1),
        v_out,
        beta2=beta2,
        one_minus_beta2=1.0 - beta2,
        n_buckets=n_buckets,
        original_numel=original_numel,
        num_blocks=num_blocks,
        BLOCK_SIZE=block_size,
    )

    return new_indices, new_scales.reshape(-1, 2), v_out[:original_numel]


# ---------------------------------------------------------------------------
# Fused CoState decode: unpack signs + compute block scales + reconstruct m
# ---------------------------------------------------------------------------
# Each program handles one block of BLOCK_SIZE elements.
# Input:  packed sign bytes, labels, block_norms, scales, alpha, gradient
# Output: reconstructed m = alpha * g + delta_hat


@triton.jit
def _costate_decode_kernel(
    sign_packed_ptr,
    labels_ptr,
    block_norms_ptr,
    amp_scales_ptr,
    g_ptr,
    alpha_ptr,
    m_out_ptr,
    original_numel,
    num_blocks,
    inv_sqrt_bs: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    alpha = tl.load(alpha_ptr).to(tl.float32)

    for block_id in range(pid, num_blocks, n_programs):
        offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < original_numel

        label = tl.load(labels_ptr + block_id).to(tl.int32)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        byte_offsets = offs // 8
        bit_positions = 7 - (offs % 8)
        packed_bytes = tl.load(sign_packed_ptr + byte_offsets, mask=mask, other=0).to(
            tl.int32
        )
        sign_bits = (packed_bytes >> bit_positions) & 1
        signs = 1.0 - 2.0 * sign_bits.to(tl.float32)

        block_norm = tl.load(block_norms_ptr + block_id).to(tl.float32)
        phase_scale = block_norm * inv_sqrt_bs
        amp_scale = tl.load(amp_scales_ptr + block_id).to(tl.float32)
        scale = tl.where(label == 1, phase_scale, tl.where(label == 2, amp_scale, 0.0))

        m = alpha * g + scale * signs
        tl.store(m_out_ptr + offs, m, mask=mask)


def triton_costate_decode(
    encoded: dict,
    alpha,
    g: torch.Tensor,
    block_size: int,
    original_numel: int,
) -> torch.Tensor:
    """Triton-accelerated CoState decode: sign unpack + scale + reconstruct m.

    Drop-in replacement for costate.decode_blocks.
    """
    device = g.device
    labels = encoded["labels"].to(device)
    sign_packed = encoded["sign_packed"].to(device)
    block_norms = encoded["block_norms"].to(dtype=torch.float32, device=device)
    scales = encoded["scales"].to(dtype=torch.float32, device=device)

    # Store alpha in a 1-element tensor (avoids .item() which blocks CUDA graph capture)
    if isinstance(alpha, torch.Tensor):
        alpha_tensor = alpha.reshape(1).float().to(device)
    else:
        alpha_tensor = torch.tensor([alpha], dtype=torch.float32, device=device)

    g_flat = g.reshape(-1).float().contiguous()
    num_blocks = labels.shape[0]

    m_out = torch.empty(num_blocks * block_size, dtype=torch.float32, device=device)

    n_programs = min(num_blocks, 1024)

    _costate_decode_kernel[(n_programs,)](
        sign_packed,
        labels,
        block_norms,
        scales,
        g_flat,
        alpha_tensor,
        m_out,
        original_numel=original_numel,
        num_blocks=num_blocks,
        inv_sqrt_bs=1.0 / math.sqrt(block_size),
        BLOCK_SIZE=block_size,
    )

    return m_out[:original_numel].reshape(g.shape)


# ---------------------------------------------------------------------------
# Fused CoState encode: compute block norms + pack signs + compute scales
# ---------------------------------------------------------------------------


@triton.jit
def _costate_encode_kernel(
    delta_ptr,
    labels_ptr,
    sign_packed_ptr,
    block_norms_ptr,
    amp_scales_ptr,
    original_numel,
    num_blocks,
    inv_sqrt_bs: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    # Load from padded tensor — all elements valid (including padding)
    for block_id in range(pid, num_blocks, n_programs):
        block_start = block_id * BLOCK_SIZE
        offs = block_start + tl.arange(0, BLOCK_SIZE)

        delta = tl.load(delta_ptr + offs)

        block_norm_sq = tl.sum(delta * delta, axis=0)
        block_norm = tl.sqrt(block_norm_sq)
        tl.store(block_norms_ptr + block_id, block_norm)

        amp_scale = block_norm * inv_sqrt_bs
        tl.store(amp_scales_ptr + block_id, amp_scale.to(tl.float16))

        byte_ids = tl.arange(0, 16)
        byte_base = block_id * 16 + byte_ids
        elem_base = block_start + byte_ids * 8
        b0 = (tl.load(delta_ptr + elem_base + 0) < 0).to(tl.int32) << 7
        b1 = (tl.load(delta_ptr + elem_base + 1) < 0).to(tl.int32) << 6
        b2 = (tl.load(delta_ptr + elem_base + 2) < 0).to(tl.int32) << 5
        b3 = (tl.load(delta_ptr + elem_base + 3) < 0).to(tl.int32) << 4
        b4 = (tl.load(delta_ptr + elem_base + 4) < 0).to(tl.int32) << 3
        b5 = (tl.load(delta_ptr + elem_base + 5) < 0).to(tl.int32) << 2
        b6 = (tl.load(delta_ptr + elem_base + 6) < 0).to(tl.int32) << 1
        b7 = (tl.load(delta_ptr + elem_base + 7) < 0).to(tl.int32)
        packed = b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7
        tl.store(sign_packed_ptr + byte_base, packed.to(tl.uint8))


def triton_costate_encode(
    delta: torch.Tensor,
    labels: torch.Tensor,
    block_size: int,
) -> dict:
    """Triton-accelerated CoState encode: norms + sign pack + scales.

    Drop-in replacement for costate.encode_blocks.
    """
    from turboadam.utils import pad_to_blocks

    delta_flat = delta.reshape(-1).float()
    original_numel = delta_flat.shape[0]

    delta_padded, _ = pad_to_blocks(delta_flat, block_size)
    num_blocks = delta_padded.shape[0] // block_size

    # Each int32 element holds one packed byte (8 sign bits in low byte)
    padded_numel = num_blocks * block_size
    sign_bytes = (padded_numel + 7) // 8
    sign_packed = torch.empty(sign_bytes, dtype=torch.uint8, device=delta.device)
    block_norms = torch.empty(num_blocks, dtype=torch.float32, device=delta.device)
    amp_scales = torch.empty(num_blocks, dtype=torch.float16, device=delta.device)

    n_programs = min(num_blocks, 1024)

    _costate_encode_kernel[(n_programs,)](
        delta_padded,
        labels,
        sign_packed,
        block_norms,
        amp_scales,
        original_numel=original_numel,
        num_blocks=num_blocks,
        inv_sqrt_bs=1.0 / math.sqrt(block_size),
        BLOCK_SIZE=block_size,
    )

    # Cast int32 → uint8 (each int32 has packed byte in low 8 bits)
    orig_sign_bytes = (original_numel + 7) // 8
    return {
        "labels": labels,
        "sign_packed": sign_packed[:orig_sign_bytes],
        "block_norms": block_norms,
        "scales": amp_scales,
    }


# ---------------------------------------------------------------------------
# Fused decompose + block_ratios + classify in a single kernel
# ---------------------------------------------------------------------------
# Fuses three operations:
#   1. decompose: alpha = (m·g)/(g·g), delta = m - alpha*g
#   2. block_ratios: r_block = norm(delta_block) / norm(m_block)
#   3. classify: label = 0 if r < tau0, 1 if tau0 <= r < tau1, 2 if r >= tau1


@triton.jit
def _decompose_ratios_kernel(
    m_ptr,
    g_ptr,
    alpha_ptr,
    delta_ptr,
    ratios_ptr,
    original_numel,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    alpha = tl.load(alpha_ptr).to(tl.float32)

    for block_id in range(pid, num_blocks, n_programs):
        offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < original_numel

        m = tl.load(m_ptr + offs, mask=mask, other=0.0)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        delta = m - alpha * g
        tl.store(delta_ptr + offs, delta, mask=mask)

        delta_norm = tl.sqrt(tl.sum(delta * delta, axis=0))
        m_norm = tl.sqrt(tl.sum(m * m, axis=0))
        ratio = tl.where(m_norm > 0, delta_norm / m_norm, 0.0)
        tl.store(ratios_ptr + block_id, ratio)


def triton_decompose_ratios(
    m_new: torch.Tensor,
    g: torch.Tensor,
    alpha: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute residual delta and per-block ratios in one Triton kernel."""
    m_flat = m_new.reshape(-1).float()
    g_flat = g.reshape(-1).float()
    original_numel = m_flat.shape[0]
    num_blocks = triton.cdiv(original_numel, block_size)

    alpha_tensor = alpha.reshape(1).float()
    delta_padded = torch.empty(
        num_blocks * block_size, dtype=torch.float32, device=m_new.device
    )
    ratios = torch.empty(num_blocks, dtype=torch.float32, device=m_new.device)
    n_programs = min(num_blocks, 1024)

    _decompose_ratios_kernel[(n_programs,)](
        m_flat,
        g_flat,
        alpha_tensor,
        delta_padded,
        ratios,
        original_numel=original_numel,
        num_blocks=num_blocks,
        BLOCK_SIZE=block_size,
    )

    return delta_padded[:original_numel].reshape(m_new.shape), ratios


@triton.jit
def _decompose_ratios_classify_kernel(
    m_ptr,
    g_ptr,
    alpha,
    tau0,
    tau1,
    delta_ptr,
    ratios_ptr,
    labels_ptr,
    original_numel,
    num_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < original_numel

    m = tl.load(m_ptr + offs, mask=mask, other=0.0)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0)

    delta = m - alpha * g
    tl.store(delta_ptr + offs, delta, mask=mask)

    delta_norm = tl.sqrt(tl.sum(delta * delta, axis=0))
    m_norm = tl.sqrt(tl.sum(m * m, axis=0))

    ratio = tl.where(m_norm > 0, delta_norm / m_norm, 0.0)
    tl.store(ratios_ptr + block_id, ratio)

    label = tl.where(ratio >= tau1, 2, tl.where(ratio >= tau0, 1, 0)).to(tl.uint8)
    tl.store(labels_ptr + block_id, label)


def triton_decompose_ratios_classify(
    m_new: torch.Tensor,
    g: torch.Tensor,
    alpha: torch.Tensor,
    tau0: torch.Tensor,
    tau1: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fused decompose + block_ratios + classify.

    Returns:
        (delta, ratios, labels)
    """
    from turboadam.utils import pad_to_blocks

    m_flat = m_new.reshape(-1).float()
    g_flat = g.reshape(-1).float()
    original_numel = m_flat.shape[0]

    m_padded, _ = pad_to_blocks(m_flat, block_size)
    g_padded, _ = pad_to_blocks(g_flat, block_size)
    num_blocks = m_padded.shape[0] // block_size

    delta_padded = torch.empty_like(m_padded)
    ratios = torch.empty(num_blocks, dtype=torch.float32, device=m_new.device)
    labels = torch.empty(num_blocks, dtype=torch.uint8, device=m_new.device)

    # Thresholds are cached Python floats; alpha may be GPU tensor
    alpha_f = (
        alpha.detach().cpu().item() if isinstance(alpha, torch.Tensor) else float(alpha)
    )
    tau0_f = float(tau0)
    tau1_f = float(tau1)

    _decompose_ratios_classify_kernel[(num_blocks,)](
        m_padded,
        g_padded,
        alpha_f,
        tau0_f,
        tau1_f,
        delta_padded,
        ratios,
        labels,
        original_numel=original_numel,
        num_blocks=num_blocks,
        BLOCK_SIZE=block_size,
    )

    delta = delta_padded[:original_numel].reshape(m_new.shape)
    return delta, ratios, labels
