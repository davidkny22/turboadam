"""Triton kernels for TurboAdam: fused operations to minimize memory traffic."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from turboadam.utils import packed_num_bytes


@triton.jit
def _rand_uniform_u32(seed, values):
    """Hash an element key and runtime seed to a reproducible uniform [0, 1)."""
    x = values.to(tl.uint32) ^ seed.to(tl.uint32)
    x = (x ^ (x >> 16)) * 0x7FEB352D
    x = (x ^ (x >> 15)) * 0x846CA68B
    x = x ^ (x >> 16)
    return x.to(tl.float32) * 2.3283064365386963e-10


@triton.jit
def _load_nbit_index(
    packed_ptr,
    block_id,
    lane,
    N_BITS: tl.constexpr,
    BYTES_PER_BLOCK: tl.constexpr,
):
    """Load one n-bit value per lane from a block-local little-endian stream."""
    bit_offset = lane * N_BITS
    byte_offset = block_id * BYTES_PER_BLOCK + bit_offset // 8
    shift = bit_offset % 8
    low = tl.load(packed_ptr + byte_offset).to(tl.int32)
    needs_high = shift + N_BITS > 8
    high = tl.load(packed_ptr + byte_offset + 1, mask=needs_high, other=0).to(
        tl.int32
    )
    word = low | (high << 8)
    return (word >> shift) & ((1 << N_BITS) - 1)


# ---------------------------------------------------------------------------
# Dense n-bit packing for the v state
# ---------------------------------------------------------------------------


@triton.jit
def _pack_nbit_indices_kernel(
    unpacked_ptr,
    packed_ptr,
    num_blocks,
    N_BITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BYTES_PER_BLOCK: tl.constexpr,
    PACK_WIDTH: tl.constexpr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    for block_id in range(pid, num_blocks, n_programs):
        byte_lane = tl.arange(0, PACK_WIDTH)
        byte_mask = byte_lane < BYTES_PER_BLOCK
        bit_start = byte_lane * 8
        first_index = bit_start // N_BITS
        inner_shift = bit_start % N_BITS
        block_start = block_id * BLOCK_SIZE

        # Four source values cover every possible output byte for the supported
        # 2/3/4/6-bit formats. Loads past the block are masked to zero.
        i0 = first_index
        i1 = first_index + 1
        i2 = first_index + 2
        i3 = first_index + 3
        v0 = tl.load(
            unpacked_ptr + block_start + i0,
            mask=byte_mask & (i0 < BLOCK_SIZE),
            other=0,
        ).to(tl.int32)
        v1 = tl.load(
            unpacked_ptr + block_start + i1,
            mask=byte_mask & (i1 < BLOCK_SIZE),
            other=0,
        ).to(tl.int32)
        v2 = tl.load(
            unpacked_ptr + block_start + i2,
            mask=byte_mask & (i2 < BLOCK_SIZE),
            other=0,
        ).to(tl.int32)
        v3 = tl.load(
            unpacked_ptr + block_start + i3,
            mask=byte_mask & (i3 < BLOCK_SIZE),
            other=0,
        ).to(tl.int32)

        window = v0 | (v1 << N_BITS) | (v2 << (2 * N_BITS)) | (v3 << (3 * N_BITS))
        packed_byte = (window >> inner_shift) & 0xFF
        tl.store(
            packed_ptr + block_id * BYTES_PER_BLOCK + byte_lane,
            packed_byte.to(tl.uint8),
            mask=byte_mask,
        )


def _validate_kernel_layout(block_size: int, n_bits: int | None = None) -> None:
    if block_size < 8 or block_size % 8 != 0 or block_size & (block_size - 1):
        raise ValueError("Triton block_size must be a power of two and a multiple of 8")
    if n_bits is not None and n_bits not in (2, 3, 4, 6, 8):
        raise ValueError(f"unsupported n_bits={n_bits}")
    if n_bits is not None and (block_size * n_bits) % 8:
        raise ValueError("block_size*n_bits must form a whole number of bytes")


def _launch_pack(
    unpacked: torch.Tensor,
    packed: torch.Tensor,
    num_blocks: int,
    block_size: int,
    n_bits: int,
) -> None:
    bytes_per_block = block_size * n_bits // 8
    pack_width = triton.next_power_of_2(bytes_per_block)
    n_programs = min(num_blocks, 2048)
    _pack_nbit_indices_kernel[(n_programs,)](
        unpacked,
        packed,
        num_blocks,
        N_BITS=n_bits,
        BLOCK_SIZE=block_size,
        BYTES_PER_BLOCK=bytes_per_block,
        PACK_WIDTH=pack_width,
    )


# ---------------------------------------------------------------------------
# Fused v update and optional AdamW parameter update
# ---------------------------------------------------------------------------


@triton.jit
def _fused_v_update_kernel(
    old_indices_ptr,
    old_scales_ptr,
    grad_ptr,
    new_unpacked_indices_ptr,
    new_scales_ptr,
    v_out_ptr,
    param_ptr,
    first_moment_ptr,
    seed,
    step_size,
    inv_sqrt_bias_correction2,
    eps,
    decay,
    beta2: tl.constexpr,
    one_minus_beta2: tl.constexpr,
    n_buckets: tl.constexpr,
    original_numel,
    num_blocks,
    PACKED: tl.constexpr,
    UPDATE_PARAM: tl.constexpr,
    N_BITS: tl.constexpr,
    BYTES_PER_BLOCK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    lane = tl.arange(0, BLOCK_SIZE)

    for block_id in range(pid, num_blocks, n_programs):
        block_start = block_id * BLOCK_SIZE
        offs = block_start + lane
        mask = offs < original_numel

        if PACKED:
            old_idx = _load_nbit_index(
                old_indices_ptr,
                block_id,
                lane,
                N_BITS=N_BITS,
                BYTES_PER_BLOCK=BYTES_PER_BLOCK,
            )
        else:
            old_idx = tl.load(old_indices_ptr + offs).to(tl.int32)

        old_log_min = tl.load(old_scales_ptr + block_id * 2).to(tl.float32)
        old_log_max = tl.load(old_scales_ptr + block_id * 2 + 1).to(tl.float32)
        old_span = old_log_max - old_log_min
        old_log_v = old_log_min + (old_idx.to(tl.float32) + 0.5) / n_buckets * old_span
        v_old = tl.exp(old_log_v)

        g = tl.load(grad_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        v_new = beta2 * v_old + one_minus_beta2 * g * g

        log_v_new = tl.log(tl.maximum(v_new, 1e-38))
        min_input = tl.where(mask, log_v_new, 1e38)
        max_input = tl.where(mask, log_v_new, -1e38)
        new_log_min = tl.min(min_input, axis=0)
        new_log_max = tl.max(max_input, axis=0)

        # Preserve the established center-coded grid exactly; the fp16 casts
        # below affect persistence, not the current index assignment.
        stored_min_fp16 = new_log_min.to(tl.float16)
        stored_max_fp16 = new_log_max.to(tl.float16)
        new_span = tl.maximum(new_log_max - new_log_min, 1e-10)
        normalized = tl.maximum(
            0.0, tl.minimum(1.0, (log_v_new - new_log_min) / new_span)
        )
        normalized = tl.where(mask, normalized, 0.0)

        position = tl.maximum(
            0.0, tl.minimum((n_buckets - 1) * 1.0, normalized * n_buckets - 0.5)
        )
        lower = tl.floor(position)
        probability_up = position - lower
        rand_val = _rand_uniform_u32(seed, offs)
        new_idx = (
            lower + (rand_val < probability_up).to(tl.float32)
        ).to(tl.uint8)

        tl.store(new_unpacked_indices_ptr + offs, new_idx)
        tl.store(new_scales_ptr + block_id * 2, stored_min_fp16)
        tl.store(new_scales_ptr + block_id * 2 + 1, stored_max_fp16)

        if UPDATE_PARAM:
            p = tl.load(param_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            m = tl.load(first_moment_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            denom = tl.sqrt(v_new) * inv_sqrt_bias_correction2 + eps
            p_new = p * decay - step_size * m / denom
            tl.store(param_ptr + offs, p_new, mask=mask)
        else:
            tl.store(v_out_ptr + offs, v_new, mask=mask)


def _infer_packed(
    indices: torch.Tensor,
    n_bits: int,
    padded_numel: int,
    packed: bool | None,
) -> bool:
    if packed is not None:
        return packed
    if n_bits == 8:
        return False
    return indices.numel() != padded_numel


def triton_fused_v_update(
    indices: torch.Tensor,
    scales: torch.Tensor,
    grad: torch.Tensor,
    beta2: float,
    n_bits: int,
    block_size: int,
    original_numel: int,
    seed: int = 0,
    out_indices: torch.Tensor | None = None,
    out_scales: torch.Tensor | None = None,
    v_out: torch.Tensor | None = None,
    packed: bool | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Triton reference API: v EMA update, stochastic requantization, v return."""
    _validate_kernel_layout(block_size, n_bits)
    num_blocks = scales.shape[0]
    padded_numel = num_blocks * block_size
    is_packed = _infer_packed(indices, n_bits, padded_numel, packed)
    expected_indices = (
        packed_num_bytes(padded_numel, n_bits) if is_packed else padded_numel
    )
    if indices.numel() != expected_indices:
        raise ValueError("indices do not match their declared packed format")
    if num_blocks == 0:
        empty_v = torch.empty(0, dtype=torch.float32, device=grad.device)
        return indices.clone(), scales.clone(), empty_v

    new_index_count = expected_indices
    new_indices = (
        torch.empty(new_index_count, dtype=torch.uint8, device=indices.device)
        if out_indices is None
        else out_indices
    )
    new_scales = torch.empty_like(scales) if out_scales is None else out_scales
    if new_indices.numel() != new_index_count:
        raise ValueError("out_indices has an incompatible length")
    if v_out is None:
        v_out = torch.empty(padded_numel, dtype=torch.float32, device=grad.device)

    new_unpacked = (
        torch.empty(padded_numel, dtype=torch.uint8, device=indices.device)
        if is_packed and n_bits < 8
        else new_indices
    )
    grad_flat = grad.reshape(-1).contiguous()
    n_programs = min(num_blocks, 2048)
    bytes_per_block = block_size * n_bits // 8

    _fused_v_update_kernel[(n_programs,)](
        indices,
        scales.reshape(-1),
        grad_flat,
        new_unpacked,
        new_scales.reshape(-1),
        v_out,
        grad_flat,
        grad_flat,
        seed,
        0.0,
        1.0,
        0.0,
        1.0,
        beta2=beta2,
        one_minus_beta2=1.0 - beta2,
        n_buckets=1 << n_bits,
        original_numel=original_numel,
        num_blocks=num_blocks,
        PACKED=is_packed,
        UPDATE_PARAM=False,
        N_BITS=n_bits,
        BYTES_PER_BLOCK=bytes_per_block,
        BLOCK_SIZE=block_size,
    )
    if is_packed and n_bits < 8:
        _launch_pack(new_unpacked, new_indices, num_blocks, block_size, n_bits)

    return new_indices, new_scales.reshape(-1, 2), v_out[:original_numel]


def triton_fused_adam_update(
    indices: torch.Tensor,
    scales: torch.Tensor,
    grad: torch.Tensor,
    param: torch.Tensor,
    first_moment: torch.Tensor,
    beta2: float,
    n_bits: int,
    block_size: int,
    original_numel: int,
    lr: float,
    bias_correction1: float,
    bias_correction2: float,
    eps: float,
    weight_decay: float,
    seed: int,
    packed: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Optimizer fast path: update v, update the parameter, and rewrite state.

    Packed indices are rewritten only after all old values have been consumed;
    fp16 endpoints use a separate tiny output to avoid alias assumptions. No
    fp32 v or denominator buffer survives the kernel, and no second packed state
    buffer is allocated.
    """
    _validate_kernel_layout(block_size, n_bits)
    num_blocks = scales.shape[0]
    padded_numel = num_blocks * block_size
    if num_blocks == 0:
        return indices, scales

    is_packed = _infer_packed(indices, n_bits, padded_numel, packed)
    if not is_packed and n_bits < 8:
        raise ValueError("optimizer fast path expects packed indices")

    new_unpacked = torch.empty(
        padded_numel, dtype=torch.uint8, device=indices.device
    )
    new_scales = torch.empty_like(scales)
    grad_flat = grad.reshape(-1).contiguous()
    m_flat = first_moment.reshape(-1).contiguous()
    param_flat = param.reshape(-1)
    n_programs = min(num_blocks, 2048)
    bytes_per_block = block_size * n_bits // 8

    _fused_v_update_kernel[(n_programs,)](
        indices,
        scales.reshape(-1),
        grad_flat,
        new_unpacked,
        new_scales.reshape(-1),
        grad_flat,  # unused when UPDATE_PARAM=True
        param_flat,
        m_flat,
        seed,
        lr / bias_correction1,
        1.0 / math.sqrt(bias_correction2),
        eps,
        1.0 - lr * weight_decay,
        beta2=beta2,
        one_minus_beta2=1.0 - beta2,
        n_buckets=1 << n_bits,
        original_numel=original_numel,
        num_blocks=num_blocks,
        PACKED=is_packed,
        UPDATE_PARAM=True,
        N_BITS=n_bits,
        BYTES_PER_BLOCK=bytes_per_block,
        BLOCK_SIZE=block_size,
    )

    if n_bits < 8:
        _launch_pack(new_unpacked, indices, num_blocks, block_size, n_bits)
    else:
        indices.copy_(new_unpacked)
    return indices, new_scales


# ---------------------------------------------------------------------------
# Fused CoState decode
# ---------------------------------------------------------------------------


@triton.jit
def _costate_decode_kernel(
    sign_packed_ptr,
    labels_ptr,
    block_norms_ptr,
    g_ptr,
    alpha_ptr,
    m_out_ptr,
    original_numel,
    num_blocks,
    inv_sqrt_bs: tl.constexpr,
    COMPACT_LABELS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    alpha = tl.load(alpha_ptr).to(tl.float32)

    for block_id in range(pid, num_blocks, n_programs):
        lane = tl.arange(0, BLOCK_SIZE)
        offs = block_id * BLOCK_SIZE + lane
        mask = offs < original_numel

        stored_norm = tl.load(block_norms_ptr + block_id).to(tl.float32)
        if COMPACT_LABELS:
            amplitude = stored_norm < 0
            block_norm = tl.abs(stored_norm)
            phase = (stored_norm > 0) & (block_norm > 0)
        else:
            label = tl.load(labels_ptr + block_id).to(tl.int32)
            phase = label == 1
            amplitude = label == 2
            block_norm = stored_norm

        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        byte_offsets = offs // 8
        bit_positions = 7 - (offs % 8)
        packed_bytes = tl.load(
            sign_packed_ptr + byte_offsets, mask=mask, other=0
        ).to(tl.int32)
        sign_bits = (packed_bytes >> bit_positions) & 1
        signs = 1.0 - 2.0 * sign_bits.to(tl.float32)

        phase_scale = block_norm * inv_sqrt_bs
        amplitude_scale = phase_scale.to(tl.float16).to(tl.float32)
        scale = tl.where(
            phase, phase_scale, tl.where(amplitude, amplitude_scale, 0.0)
        )
        tl.store(m_out_ptr + offs, alpha * g + scale * signs, mask=mask)


def triton_costate_decode(
    encoded: dict,
    alpha,
    g: torch.Tensor,
    block_size: int,
    original_numel: int,
) -> torch.Tensor:
    """Triton-accelerated CoState sign unpack and reconstruction."""
    _validate_kernel_layout(block_size)
    if original_numel == 0:
        return torch.empty_like(g, dtype=torch.float32)

    device = g.device
    compact_labels = "labels" not in encoded
    sign_packed = encoded["sign_packed"].to(dtype=torch.uint8, device=device)
    block_norms = encoded["block_norms"].to(dtype=torch.float32, device=device)
    labels = (
        sign_packed
        if compact_labels
        else encoded["labels"].to(dtype=torch.uint8, device=device)
    )
    alpha_tensor = (
        alpha.reshape(1).to(dtype=torch.float32, device=device)
        if isinstance(alpha, torch.Tensor)
        else torch.tensor([alpha], dtype=torch.float32, device=device)
    )
    g_flat = g.reshape(-1).float().contiguous()
    num_blocks = block_norms.numel()
    m_out = torch.empty(num_blocks * block_size, dtype=torch.float32, device=device)
    n_programs = min(num_blocks, 1024)

    _costate_decode_kernel[(n_programs,)](
        sign_packed,
        labels,
        block_norms,
        g_flat,
        alpha_tensor,
        m_out,
        original_numel=original_numel,
        num_blocks=num_blocks,
        inv_sqrt_bs=1.0 / math.sqrt(block_size),
        COMPACT_LABELS=compact_labels,
        BLOCK_SIZE=block_size,
    )
    return m_out[:original_numel].reshape(g.shape)


# ---------------------------------------------------------------------------
# Fused CoState encode
# ---------------------------------------------------------------------------


@triton.jit
def _costate_encode_kernel(
    delta_ptr,
    labels_ptr,
    sign_packed_ptr,
    block_norms_ptr,
    precomputed_norms_ptr,
    legacy_scales_ptr,
    original_numel,
    num_blocks,
    sign_bytes,
    inv_sqrt_bs: tl.constexpr,
    STORE_LEGACY_SCALE: tl.constexpr,
    COMPACT_LABELS: tl.constexpr,
    PRECOMPUTED_NORMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BYTES_PER_BLOCK: tl.constexpr,
    BYTE_WIDTH: tl.constexpr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    for block_id in range(pid, num_blocks, n_programs):
        lane = tl.arange(0, BLOCK_SIZE)
        block_start = block_id * BLOCK_SIZE
        offs = block_start + lane
        elem_mask = offs < original_numel
        delta = tl.load(delta_ptr + offs, mask=elem_mask, other=0.0).to(tl.float32)

        if PRECOMPUTED_NORMS:
            block_norm = tl.load(precomputed_norms_ptr + block_id).to(tl.float32)
        else:
            # Scale before squaring so finite fp32 residuals cannot overflow the
            # reduction. Saturate only when the true norm itself exceeds fp32.
            block_scale = tl.max(tl.abs(delta), axis=0)
            unit = tl.where(block_scale > 0, delta / block_scale, 0.0)
            norm_factor = tl.sqrt(tl.sum(unit * unit, axis=0))
            bounded_scale = tl.minimum(
                block_scale, 3.402823466e38 / tl.maximum(norm_factor, 1.0)
            )
            block_norm = tl.where(
                block_scale > 0, bounded_scale * norm_factor, 0.0
            )
        label = tl.load(labels_ptr + block_id).to(tl.int32)
        if COMPACT_LABELS:
            signed_norm = tl.where(
                label == 0, 0.0, tl.where(label == 2, -block_norm, block_norm)
            )
            tl.store(block_norms_ptr + block_id, signed_norm)
        else:
            tl.store(block_norms_ptr + block_id, block_norm)
        if STORE_LEGACY_SCALE:
            tl.store(
                legacy_scales_ptr + block_id,
                (block_norm * inv_sqrt_bs).to(tl.float16),
            )

        byte_lane = tl.arange(0, BYTE_WIDTH)
        byte_base = block_id * BYTES_PER_BLOCK + byte_lane
        byte_mask = (byte_lane < BYTES_PER_BLOCK) & (byte_base < sign_bytes)
        elem_base = block_start + byte_lane * 8
        b0 = (
            tl.load(
                delta_ptr + elem_base + 0,
                mask=(elem_base + 0) < original_numel,
                other=0.0,
            )
            < 0
        ).to(tl.int32) << 7
        b1 = (
            tl.load(
                delta_ptr + elem_base + 1,
                mask=(elem_base + 1) < original_numel,
                other=0.0,
            )
            < 0
        ).to(tl.int32) << 6
        b2 = (
            tl.load(
                delta_ptr + elem_base + 2,
                mask=(elem_base + 2) < original_numel,
                other=0.0,
            )
            < 0
        ).to(tl.int32) << 5
        b3 = (
            tl.load(
                delta_ptr + elem_base + 3,
                mask=(elem_base + 3) < original_numel,
                other=0.0,
            )
            < 0
        ).to(tl.int32) << 4
        b4 = (
            tl.load(
                delta_ptr + elem_base + 4,
                mask=(elem_base + 4) < original_numel,
                other=0.0,
            )
            < 0
        ).to(tl.int32) << 3
        b5 = (
            tl.load(
                delta_ptr + elem_base + 5,
                mask=(elem_base + 5) < original_numel,
                other=0.0,
            )
            < 0
        ).to(tl.int32) << 2
        b6 = (
            tl.load(
                delta_ptr + elem_base + 6,
                mask=(elem_base + 6) < original_numel,
                other=0.0,
            )
            < 0
        ).to(tl.int32) << 1
        b7 = (
            tl.load(
                delta_ptr + elem_base + 7,
                mask=(elem_base + 7) < original_numel,
                other=0.0,
            )
            < 0
        ).to(tl.int32)
        tl.store(
            sign_packed_ptr + byte_base,
            (b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7).to(tl.uint8),
            mask=byte_mask,
        )


def triton_costate_encode(
    delta: torch.Tensor,
    labels: torch.Tensor,
    block_size: int,
    *,
    out: dict | None = None,
    include_legacy_scale: bool = True,
    compact_labels: bool = False,
    block_norms: torch.Tensor | None = None,
) -> dict:
    """Triton-accelerated CoState norm reduction and sign packing."""
    _validate_kernel_layout(block_size)
    delta_flat = delta.reshape(-1).float().contiguous()
    original_numel = delta_flat.numel()
    num_blocks = triton.cdiv(original_numel, block_size)
    sign_bytes = (original_numel + 7) // 8

    if out is None:
        encoded = {
            "sign_packed": torch.empty(
                sign_bytes, dtype=torch.uint8, device=delta.device
            ),
            "block_norms": torch.empty(
                num_blocks, dtype=torch.float32, device=delta.device
            ),
        }
        if not compact_labels:
            encoded["labels"] = labels
    else:
        encoded = out
        if compact_labels:
            encoded.pop("labels", None)
            encoded.pop("phase_packed", None)
        else:
            encoded["labels"].copy_(labels)

    if block_norms is None:
        precomputed_norms = encoded["block_norms"]
        has_precomputed_norms = False
    else:
        if block_norms.numel() != num_blocks:
            raise ValueError("precomputed block_norms has an incompatible length")
        precomputed_norms = block_norms.reshape(-1).to(
            device=delta.device, dtype=torch.float32
        )
        has_precomputed_norms = True

    if include_legacy_scale:
        if out is not None and "scales" in out:
            legacy_scales = out["scales"]
        else:
            legacy_scales = torch.empty(
                num_blocks, dtype=torch.float16, device=delta.device
            )
            encoded["scales"] = legacy_scales
    else:
        encoded.pop("scales", None)
        # The pointer is compile-time dead when STORE_LEGACY_SCALE=False.
        legacy_scales = encoded["block_norms"]

    if num_blocks == 0:
        return encoded

    bytes_per_block = block_size // 8
    byte_width = triton.next_power_of_2(bytes_per_block)
    n_programs = min(num_blocks, 1024)
    _costate_encode_kernel[(n_programs,)](
        delta_flat,
        labels,
        encoded["sign_packed"],
        encoded["block_norms"],
        precomputed_norms,
        legacy_scales,
        original_numel,
        num_blocks,
        sign_bytes,
        inv_sqrt_bs=1.0 / math.sqrt(block_size),
        STORE_LEGACY_SCALE=include_legacy_scale,
        COMPACT_LABELS=compact_labels,
        PRECOMPUTED_NORMS=has_precomputed_norms,
        BLOCK_SIZE=block_size,
        BYTES_PER_BLOCK=bytes_per_block,
        BYTE_WIDTH=byte_width,
    )
    return encoded


# ---------------------------------------------------------------------------
# Fused decompose + block ratios
# ---------------------------------------------------------------------------


@triton.jit
def _decompose_ratios_kernel(
    m_ptr,
    g_ptr,
    alpha_ptr,
    delta_ptr,
    ratios_ptr,
    block_norms_ptr,
    original_numel,
    num_blocks,
    STORE_BLOCK_NORMS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    alpha = tl.load(alpha_ptr).to(tl.float32)

    for block_id in range(pid, num_blocks, n_programs):
        offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < original_numel
        m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        delta = m - alpha * g
        tl.store(delta_ptr + offs, delta, mask=mask)

        delta_scale = tl.max(tl.abs(delta), axis=0)
        m_scale = tl.max(tl.abs(m), axis=0)
        delta_unit = tl.where(delta_scale > 0, delta / delta_scale, 0.0)
        m_unit = tl.where(m_scale > 0, m / m_scale, 0.0)
        delta_factor = tl.sqrt(tl.sum(delta_unit * delta_unit, axis=0))
        m_factor = tl.sqrt(tl.sum(m_unit * m_unit, axis=0))
        if STORE_BLOCK_NORMS:
            bounded_scale = tl.minimum(
                delta_scale, 3.402823466e38 / tl.maximum(delta_factor, 1.0)
            )
            delta_norm = tl.where(
                delta_scale > 0, bounded_scale * delta_factor, 0.0
            )
            tl.store(block_norms_ptr + block_id, delta_norm)
        valid_ratio = (delta_scale > 0) & (m_scale > 0)
        log_ratio = (
            tl.log(tl.maximum(delta_scale, 1.401298464e-45))
            - tl.log(tl.maximum(m_scale, 1.401298464e-45))
            + tl.log(tl.maximum(delta_factor, 1.401298464e-45))
            - tl.log(tl.maximum(m_factor, 1.401298464e-45))
        )
        ratio = tl.where(
            valid_ratio, tl.exp(tl.minimum(log_ratio, 88.7228390521)), 0.0
        )
        tl.store(ratios_ptr + block_id, ratio)


def triton_decompose_ratios(
    m_new: torch.Tensor,
    g: torch.Tensor,
    alpha: torch.Tensor,
    block_size: int,
    *,
    return_block_norms: bool = False,
):
    """Compute CoState residual and block ratios in one Triton pass."""
    _validate_kernel_layout(block_size)
    m_flat = m_new.reshape(-1).float().contiguous()
    g_flat = g.reshape(-1).float().contiguous()
    original_numel = m_flat.numel()
    num_blocks = triton.cdiv(original_numel, block_size)
    if num_blocks == 0:
        empty = m_flat.new_empty(0)
        if return_block_norms:
            return torch.empty_like(m_new), empty, empty
        return torch.empty_like(m_new), empty

    alpha_tensor = alpha.reshape(1).float().contiguous()
    delta_padded = torch.empty(
        num_blocks * block_size, dtype=torch.float32, device=m_new.device
    )
    ratios = torch.empty(num_blocks, dtype=torch.float32, device=m_new.device)
    block_norms = (
        torch.empty(num_blocks, dtype=torch.float32, device=m_new.device)
        if return_block_norms
        else ratios
    )
    n_programs = min(num_blocks, 1024)
    _decompose_ratios_kernel[(n_programs,)](
        m_flat,
        g_flat,
        alpha_tensor,
        delta_padded,
        ratios,
        block_norms,
        original_numel=original_numel,
        num_blocks=num_blocks,
        STORE_BLOCK_NORMS=return_block_norms,
        BLOCK_SIZE=block_size,
    )
    delta = delta_padded[:original_numel].reshape(m_new.shape)
    if return_block_norms:
        return delta, ratios, block_norms
    return delta, ratios


@triton.jit
def _decompose_ratios_classify_kernel(
    m_ptr,
    g_ptr,
    alpha_ptr,
    tau0_ptr,
    tau1_ptr,
    delta_ptr,
    ratios_ptr,
    labels_ptr,
    original_numel,
    BLOCK_SIZE: tl.constexpr,
):
    block_id = tl.program_id(0)
    offs = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < original_numel
    alpha = tl.load(alpha_ptr).to(tl.float32)
    tau0 = tl.load(tau0_ptr).to(tl.float32)
    tau1 = tl.load(tau1_ptr).to(tl.float32)

    m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    delta = m - alpha * g
    tl.store(delta_ptr + offs, delta, mask=mask)

    delta_scale = tl.max(tl.abs(delta), axis=0)
    m_scale = tl.max(tl.abs(m), axis=0)
    delta_unit = tl.where(delta_scale > 0, delta / delta_scale, 0.0)
    m_unit = tl.where(m_scale > 0, m / m_scale, 0.0)
    delta_factor = tl.sqrt(tl.sum(delta_unit * delta_unit, axis=0))
    m_factor = tl.sqrt(tl.sum(m_unit * m_unit, axis=0))
    valid_ratio = (delta_scale > 0) & (m_scale > 0)
    log_ratio = (
        tl.log(tl.maximum(delta_scale, 1.401298464e-45))
        - tl.log(tl.maximum(m_scale, 1.401298464e-45))
        + tl.log(tl.maximum(delta_factor, 1.401298464e-45))
        - tl.log(tl.maximum(m_factor, 1.401298464e-45))
    )
    ratio = tl.where(
        valid_ratio, tl.exp(tl.minimum(log_ratio, 88.7228390521)), 0.0
    )
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
    """Fused decompose, ratio, and classification without host scalar syncs."""
    _validate_kernel_layout(block_size)
    m_flat = m_new.reshape(-1).float().contiguous()
    g_flat = g.reshape(-1).float().contiguous()
    original_numel = m_flat.numel()
    num_blocks = triton.cdiv(original_numel, block_size)
    if num_blocks == 0:
        return (
            torch.empty_like(m_new),
            m_flat.new_empty(0),
            m_flat.new_empty(0, dtype=torch.uint8),
        )

    delta_padded = torch.empty(
        num_blocks * block_size, dtype=torch.float32, device=m_new.device
    )
    ratios = torch.empty(num_blocks, dtype=torch.float32, device=m_new.device)
    labels = torch.empty(num_blocks, dtype=torch.uint8, device=m_new.device)
    _decompose_ratios_classify_kernel[(num_blocks,)](
        m_flat,
        g_flat,
        alpha.reshape(1).float(),
        tau0.reshape(1).float(),
        tau1.reshape(1).float(),
        delta_padded,
        ratios,
        labels,
        original_numel,
        BLOCK_SIZE=block_size,
    )
    return delta_padded[:original_numel].reshape(m_new.shape), ratios, labels


# ---------------------------------------------------------------------------
# Overflow-safe global projection coefficient
# ---------------------------------------------------------------------------


@triton.jit
def _projection_max_kernel(
    m_ptr,
    g_ptr,
    max_ptr,
    original_numel,
    REDUCE_BLOCK: tl.constexpr,
):
    block_id = tl.program_id(0)
    offs = block_id * REDUCE_BLOCK + tl.arange(0, REDUCE_BLOCK)
    mask = offs < original_numel
    m = tl.abs(tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32))
    g = tl.abs(tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32))
    tl.store(max_ptr + block_id * 2, tl.max(m, axis=0))
    tl.store(max_ptr + block_id * 2 + 1, tl.max(g, axis=0))


@triton.jit
def _projection_dot_kernel(
    m_ptr,
    g_ptr,
    scales_ptr,
    partials_ptr,
    original_numel,
    REDUCE_BLOCK: tl.constexpr,
):
    block_id = tl.program_id(0)
    offs = block_id * REDUCE_BLOCK + tl.arange(0, REDUCE_BLOCK)
    mask = offs < original_numel
    m_scale = tl.load(scales_ptr).to(tl.float32)
    g_scale = tl.load(scales_ptr + 1).to(tl.float32)
    inv_m_scale = tl.where(m_scale > 0, 1.0 / m_scale, 0.0)
    inv_g_scale = tl.where(g_scale > 0, 1.0 / g_scale, 0.0)
    m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32) * inv_m_scale
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32) * inv_g_scale
    tl.store(partials_ptr + block_id * 2, tl.sum(m * g, axis=0))
    tl.store(partials_ptr + block_id * 2 + 1, tl.sum(g * g, axis=0))


def triton_projection_alpha(m: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Compute (m·g)/(g·g) without overflowing fp32 dot products.

    Two small partial arrays replace full-size normalized copies. Scaling is
    factored analytically back into the final scalar coefficient.
    """
    m_flat = m.reshape(-1).float().contiguous()
    g_flat = g.reshape(-1).float().contiguous()
    original_numel = m_flat.numel()
    if original_numel == 0:
        return m_flat.new_zeros(())

    reduce_block = 1024
    num_blocks = triton.cdiv(original_numel, reduce_block)
    partial_max = torch.empty(
        (num_blocks, 2), dtype=torch.float32, device=m.device
    )
    _projection_max_kernel[(num_blocks,)](
        m_flat,
        g_flat,
        partial_max,
        original_numel,
        REDUCE_BLOCK=reduce_block,
    )
    scales = partial_max.amax(dim=0)

    partial_dot = torch.empty_like(partial_max)
    _projection_dot_kernel[(num_blocks,)](
        m_flat,
        g_flat,
        scales,
        partial_dot,
        original_numel,
        REDUCE_BLOCK=reduce_block,
    )
    sums = partial_dot.sum(dim=0)
    numerator = sums[0]
    denominator = sums[1]
    m_scale = scales[0]
    g_scale = scales[1]
    valid = (g_scale > 0) & (denominator > 0) & (numerator != 0)
    log_abs_alpha = (
        m_scale.clamp_min(2.0**-149).log()
        - g_scale.clamp_min(2.0**-149).log()
        + numerator.abs().clamp_min(2.0**-149).log()
        - denominator.clamp_min(2.0**-149).log()
    )
    magnitude = log_abs_alpha.clamp_max(
        math.log(torch.finfo(torch.float32).max)
    ).exp()
    alpha = magnitude.copysign(numerator)
    return torch.where(valid, alpha, alpha.new_zeros(()))
