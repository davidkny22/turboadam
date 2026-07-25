"""Triton kernels for the packed UState and 1Q AdamW path."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl

from turboadam.quantize import packed_index_numel


def triton_supports_block_size(block_size: int) -> bool:
    """Return whether the fused block kernel supports a block size."""
    return bool(
        isinstance(block_size, int)
        and 32 <= block_size <= 1024
        and block_size % 8 == 0
        and (block_size & (block_size - 1)) == 0
    )


@triton.jit
def _rand_uniform_u24(seed, keys):
    x = keys.to(tl.uint32) ^ seed.to(tl.uint32)
    x = (x ^ (x >> 16)) * 0x7FEB352D
    x = (x ^ (x >> 15)) * 0x846CA68B
    x = x ^ (x >> 16)
    return (x >> 8).to(tl.float32) * 5.960464477539063e-8


@triton.jit
def _antithetic_uniform(seed, offsets):
    base = _rand_uniform_u24(seed, offsets // 2)
    reflected = tl.minimum(1.0 - base, 0.9999999403953552)
    return tl.where((offsets & 1) == 0, base, reflected)


@triton.jit
def _next_down_fp16(value):
    bits = value.to(tl.uint16, bitcast=True)
    stepped = tl.where(value > 0.0, bits - 1, bits + 1)
    stepped = tl.where(value == 0.0, 0x8001, stepped)
    return stepped.to(tl.float16, bitcast=True)


@triton.jit
def _next_up_fp16(value):
    bits = value.to(tl.uint16, bitcast=True)
    stepped = tl.where(value < 0.0, bits - 1, bits + 1)
    stepped = tl.where(value == 0.0, 0x0001, stepped)
    return stepped.to(tl.float16, bitcast=True)


@triton.jit
def _load_packed_indices(
    packed_ptr,
    block_id,
    local_offsets,
    N_BITS: tl.constexpr,
    PACKED_BYTES_PER_BLOCK: tl.constexpr,
):
    packed_base = block_id * PACKED_BYTES_PER_BLOCK
    if N_BITS == 8:
        return tl.load(packed_ptr + packed_base + local_offsets).to(tl.int32)
    if N_BITS == 4:
        byte_offsets = local_offsets // 2
        shifts = (local_offsets & 1) * 4
        byte = tl.load(packed_ptr + packed_base + byte_offsets).to(tl.int32)
        return (byte >> shifts) & 0x0F
    if N_BITS == 2:
        byte_offsets = local_offsets // 4
        shifts = (local_offsets & 3) * 2
        byte = tl.load(packed_ptr + packed_base + byte_offsets).to(tl.int32)
        return (byte >> shifts) & 0x03

    bit_offsets = local_offsets * N_BITS
    byte_offsets = bit_offsets // 8
    shifts = bit_offsets % 8
    low = tl.load(packed_ptr + packed_base + byte_offsets).to(tl.int32)
    high = tl.load(
        packed_ptr + packed_base + byte_offsets + 1,
        mask=(byte_offsets + 1) < PACKED_BYTES_PER_BLOCK,
        other=0,
    ).to(tl.int32)
    word = low | (high << 8)
    return (word >> shifts) & ((1 << N_BITS) - 1)


@triton.jit
def _store_packed_indices(
    packed_ptr,
    block_id,
    values,
    N_BITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    PACKED_BYTES_PER_BLOCK: tl.constexpr,
):
    values = values.to(tl.int32)
    base = block_id * PACKED_BYTES_PER_BLOCK
    if N_BITS == 8:
        byte_ids = tl.arange(0, BLOCK_SIZE)
        tl.store(packed_ptr + base + byte_ids, values.to(tl.uint8))
    elif N_BITS == 4:
        groups = tl.reshape(values, (BLOCK_SIZE // 2, 2))
        byte_ids = tl.arange(0, BLOCK_SIZE // 2)
        shifts = tl.arange(0, 2) * 4
        byte = tl.sum(groups << shifts, axis=1)
        tl.store(packed_ptr + base + byte_ids, byte.to(tl.uint8))
    elif N_BITS == 2:
        groups = tl.reshape(values, (BLOCK_SIZE // 4, 4))
        byte_ids = tl.arange(0, BLOCK_SIZE // 4)
        shifts = tl.arange(0, 4) * 2
        byte = tl.sum(groups << shifts, axis=1)
        tl.store(packed_ptr + base + byte_ids, byte.to(tl.uint8))
    elif N_BITS == 3:
        groups = tl.reshape(values, (BLOCK_SIZE // 8, 8))
        ids = tl.arange(0, BLOCK_SIZE // 8)
        out = base + ids * 3
        shifts = tl.arange(0, 8) * 3
        word = tl.sum(groups << shifts, axis=1)
        tl.store(packed_ptr + out, word.to(tl.uint8))
        tl.store(packed_ptr + out + 1, (word >> 8).to(tl.uint8))
        tl.store(packed_ptr + out + 2, (word >> 16).to(tl.uint8))
    else:
        groups = tl.reshape(values, (BLOCK_SIZE // 4, 4))
        ids = tl.arange(0, BLOCK_SIZE // 4)
        out = base + ids * 3
        shifts = tl.arange(0, 4) * 6
        word = tl.sum(groups << shifts, axis=1)
        tl.store(packed_ptr + out, word.to(tl.uint8))
        tl.store(packed_ptr + out + 1, (word >> 8).to(tl.uint8))
        tl.store(packed_ptr + out + 2, (word >> 16).to(tl.uint8))


@triton.jit
def _fused_ustate_adamw_kernel(
    param_ptr,
    grad_ptr,
    q_codes_ptr,
    q_means_ptr,
    q_decode_step_ptr,
    q_encode_step_ptr,
    q_rms_accumulator_ptr,
    v_codes_ptr,
    v_scales_ptr,
    previous_bc1,
    previous_bc2_sqrt,
    bc1,
    bc2_sqrt,
    beta1,
    one_minus_beta1,
    beta2,
    one_minus_beta2,
    step_size,
    eps,
    decay,
    m_seed,
    v_seed,
    original_numel,
    num_blocks,
    N_BITS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    M_BLOCK_SIZE: tl.constexpr,
    M_BLOCKS_PER_V: tl.constexpr,
    V_BYTES_PER_BLOCK: tl.constexpr,
    Q_BYTES_PER_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)
    local = tl.arange(0, BLOCK_SIZE)
    intervals = (1 << N_BITS) - 1
    previous_bc1 = previous_bc1.to(tl.float32)
    previous_bc2_sqrt = previous_bc2_sqrt.to(tl.float32)
    bc1 = bc1.to(tl.float32)
    bc2_sqrt = bc2_sqrt.to(tl.float32)
    beta1 = beta1.to(tl.float32)
    one_minus_beta1 = one_minus_beta1.to(tl.float32)
    beta2 = beta2.to(tl.float32)
    one_minus_beta2 = one_minus_beta2.to(tl.float32)
    step_size = step_size.to(tl.float32)
    eps = eps.to(tl.float32)
    decay = decay.to(tl.float32)
    decode_step = tl.load(q_decode_step_ptr).to(tl.float32)
    encode_step = tl.load(q_encode_step_ptr).to(tl.float32)
    safe_encode_step = tl.maximum(encode_step, 1.1754943508222875e-38)

    for block_id in range(pid, num_blocks, n_programs):
        offsets = block_id * BLOCK_SIZE + local
        mask = offsets < original_numel

        old_v_idx = _load_packed_indices(
            v_codes_ptr,
            block_id,
            local,
            N_BITS=N_BITS,
            PACKED_BYTES_PER_BLOCK=V_BYTES_PER_BLOCK,
        ).to(tl.float32)
        old_log_min = tl.load(v_scales_ptr + block_id * 2).to(tl.float32)
        old_log_max = tl.load(v_scales_ptr + block_id * 2 + 1).to(tl.float32)
        old_log_v = old_log_min + (old_v_idx / intervals) * (old_log_max - old_log_min)
        v_old = tl.exp(old_log_v)

        q_idx = _load_packed_indices(
            q_codes_ptr,
            block_id,
            local,
            N_BITS=2,
            PACKED_BYTES_PER_BLOCK=Q_BYTES_PER_BLOCK,
        ).to(tl.float32)
        q_raw = (q_idx - 1.5) * decode_step
        q_old = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        for sub in tl.static_range(0, M_BLOCKS_PER_V):
            sub_begin = sub * M_BLOCK_SIZE
            sub_end = sub_begin + M_BLOCK_SIZE
            sub_mask = mask & (local >= sub_begin) & (local < sub_end)
            sub_global_begin = block_id * BLOCK_SIZE + sub_begin
            remaining = original_numel - sub_global_begin
            valid_count = tl.maximum(0, tl.minimum(M_BLOCK_SIZE, remaining))
            valid_count_f = tl.maximum(valid_count.to(tl.float32), 1.0)
            mean_index = block_id * M_BLOCKS_PER_V + sub
            q_mean = tl.load(
                q_means_ptr + mean_index,
                mask=valid_count > 0,
                other=0.0,
            ).to(tl.float32)
            raw_sum = tl.sum(tl.where(sub_mask, q_raw, 0.0), axis=0)
            centered = q_mean + q_raw - raw_sum / valid_count_f
            q_old = tl.where(sub_mask, centered, q_old)

        previous_denom = tl.sqrt_rn(tl.maximum(v_old, 0.0)) / previous_bc2_sqrt + eps
        m_old = q_old * previous_bc1 * previous_denom
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        m_new = beta1 * m_old + one_minus_beta1 * grad
        v_new = beta2 * v_old + one_minus_beta2 * grad * grad
        v_new = tl.minimum(v_new, 3.2846709259524643e38)

        denominator = tl.sqrt_rn(tl.maximum(v_new, 0.0)) / bc2_sqrt + eps
        param = tl.load(param_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        param = param * decay - step_size * m_new / denominator
        tl.store(param_ptr + offsets, param, mask=mask)

        log_v_new = tl.log(tl.maximum(v_new, 1.0e-38))
        min_input = tl.where(mask, log_v_new, 1.0e38)
        max_input = tl.where(mask, log_v_new, -1.0e38)
        exact_min = tl.minimum(tl.min(min_input, axis=0), 88.6875)
        exact_max = tl.minimum(tl.max(max_input, axis=0), 88.6875)
        nearest_min = exact_min.to(tl.float16)
        nearest_max = exact_max.to(tl.float16)
        dynamic = exact_max > exact_min
        stored_min_h = tl.where(
            dynamic & (nearest_min.to(tl.float32) > exact_min),
            _next_down_fp16(nearest_min),
            nearest_min,
        )
        max_out = tl.minimum(_next_up_fp16(nearest_max).to(tl.float32), 88.6875).to(
            tl.float16
        )
        stored_max_h = tl.where(
            dynamic & (nearest_max.to(tl.float32) < exact_max),
            max_out,
            nearest_max,
        )
        stored_min = stored_min_h.to(tl.float32)
        stored_max = stored_max_h.to(tl.float32)
        tl.store(v_scales_ptr + block_id * 2, stored_min_h)
        tl.store(v_scales_ptr + block_id * 2 + 1, stored_max_h)

        log_step = (stored_max - stored_min) / intervals
        position = (log_v_new - stored_min) / tl.maximum(log_step, 1.0e-10)
        position = tl.minimum(tl.maximum(position, 0.0), intervals * 1.0)
        lower = tl.floor(position)
        upper = tl.minimum(lower + 1.0, intervals * 1.0)
        lower_value = tl.exp(stored_min + lower * log_step)
        upper_value = tl.exp(stored_min + upper * log_step)
        probability_up = (v_new - lower_value) / tl.maximum(
            upper_value - lower_value, 1.0e-30
        )
        probability_up = tl.minimum(tl.maximum(probability_up, 0.0), 1.0)
        probability_up = tl.where(upper > lower, probability_up, 0.0)
        v_random = _antithetic_uniform(v_seed, offsets)
        new_v_idx = (lower + (v_random < probability_up).to(tl.float32)).to(tl.float32)
        _store_packed_indices(
            v_codes_ptr,
            block_id,
            new_v_idx,
            N_BITS=N_BITS,
            BLOCK_SIZE=BLOCK_SIZE,
            PACKED_BYTES_PER_BLOCK=V_BYTES_PER_BLOCK,
        )

        v_persisted = tl.exp(
            stored_min + (new_v_idx / intervals) * (stored_max - stored_min)
        ).to(tl.float32)
        persisted_denom = tl.sqrt_rn(tl.maximum(v_persisted, 0.0)) / bc2_sqrt + eps
        q_new = tl.where(mask, (m_new / bc1) / persisted_denom, 0.0)

        new_q_idx = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        block_residual_sumsq = 0.0
        for sub in tl.static_range(0, M_BLOCKS_PER_V):
            sub_begin = sub * M_BLOCK_SIZE
            sub_end = sub_begin + M_BLOCK_SIZE
            sub_mask = mask & (local >= sub_begin) & (local < sub_end)
            sub_global_begin = block_id * BLOCK_SIZE + sub_begin
            remaining = original_numel - sub_global_begin
            valid_count = tl.maximum(0, tl.minimum(M_BLOCK_SIZE, remaining))
            valid_count_f = tl.maximum(valid_count.to(tl.float32), 1.0)
            mean_index = block_id * M_BLOCKS_PER_V + sub

            exact_mean = tl.sum(tl.where(sub_mask, q_new, 0.0), axis=0) / valid_count_f
            mean_h = exact_mean.to(tl.bfloat16)
            mean_stored = mean_h.to(tl.float32)
            tl.store(q_means_ptr + mean_index, mean_h, mask=valid_count > 0)

            residual = tl.where(sub_mask, q_new - mean_stored, 0.0)
            block_residual_sumsq += tl.sum(residual * residual, axis=0)
            q_position = residual / safe_encode_step + 1.5
            q_position = tl.minimum(tl.maximum(q_position, 0.0), 3.0)
            q_lower = tl.floor(q_position)
            q_fraction = q_position - q_lower
            q_random = _antithetic_uniform(m_seed, offsets)
            codes = q_lower + (q_random < q_fraction).to(tl.float32)
            codes = tl.where(encode_step > 0.0, codes, 0.0)
            new_q_idx = tl.where(sub_mask, codes, new_q_idx)

        tl.atomic_add(q_rms_accumulator_ptr, block_residual_sumsq)
        _store_packed_indices(
            q_codes_ptr,
            block_id,
            new_q_idx,
            N_BITS=2,
            BLOCK_SIZE=BLOCK_SIZE,
            PACKED_BYTES_PER_BLOCK=Q_BYTES_PER_BLOCK,
        )


@triton.jit
def _finalize_ustate_scale_kernel(
    q_decode_step_ptr,
    q_encode_step_ptr,
    q_rms_accumulator_ptr,
    step_factor,
    original_numel,
):
    step_factor = step_factor.to(tl.float32)
    used_step = tl.load(q_encode_step_ptr).to(tl.float32)
    sumsq = tl.load(q_rms_accumulator_ptr).to(tl.float32)
    next_step = tl.sqrt_rn(tl.maximum(sumsq / original_numel, 0.0)) * step_factor
    tl.store(q_decode_step_ptr, used_step)
    tl.store(q_encode_step_ptr, next_step)
    tl.store(q_rms_accumulator_ptr, 0.0)


def triton_fused_ustate_adamw_step(
    param: torch.Tensor,
    grad: torch.Tensor,
    q_codes: torch.Tensor,
    q_means: torch.Tensor,
    q_decode_step: torch.Tensor,
    q_encode_step: torch.Tensor,
    q_rms_accumulator: torch.Tensor,
    v_codes: torch.Tensor,
    v_scales: torch.Tensor,
    *,
    previous_step: int,
    step: int,
    beta1: float,
    beta2: float,
    lr: float,
    eps: float,
    weight_decay: float,
    n_bits: int,
    block_size: int,
    m_block_size: int,
    original_numel: int,
    m_step_factor: float,
    m_seed: int,
    v_seed: int,
) -> None:
    """Launch the fused packed-state AdamW step and scale finalizer."""
    if n_bits not in (2, 3, 4, 6, 8):
        raise ValueError(f"unsupported v bit width: {n_bits}")
    if not triton_supports_block_size(block_size):
        raise ValueError(f"unsupported Triton block_size: {block_size}")
    if not triton_supports_block_size(m_block_size):
        raise ValueError(f"unsupported Triton m_block_size: {m_block_size}")
    if block_size % m_block_size:
        raise ValueError("block_size must be divisible by m_block_size")
    if original_numel != param.numel() or grad.numel() != original_numel:
        raise ValueError("parameter, gradient, and state lengths disagree")
    tensors = (
        param,
        grad,
        q_codes,
        q_means,
        q_decode_step,
        q_encode_step,
        q_rms_accumulator,
        v_codes,
        v_scales,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise ValueError("fused Triton step requires CUDA tensors")
    if not all(tensor.is_contiguous() for tensor in tensors):
        raise ValueError("fused Triton step requires contiguous tensors")
    if q_codes.dtype != torch.uint8 or v_codes.dtype != torch.uint8:
        raise ValueError("packed code streams must be uint8")
    if q_means.dtype != torch.bfloat16 or v_scales.dtype != torch.float16:
        raise ValueError("expected bf16 UState means and fp16 v endpoints")
    for scalar in (q_decode_step, q_encode_step, q_rms_accumulator):
        if scalar.dtype != torch.float32 or scalar.numel() != 1:
            raise ValueError("UState scale tensors must be one-element fp32 tensors")

    num_blocks = triton.cdiv(original_numel, block_size)
    padded_numel = num_blocks * block_size
    num_mean_blocks = triton.cdiv(original_numel, m_block_size)
    if q_codes.numel() != packed_index_numel(padded_numel, 2):
        raise ValueError("invalid UState code storage length")
    if q_means.numel() != num_mean_blocks:
        raise ValueError("invalid UState mean storage length")
    if v_codes.numel() != packed_index_numel(padded_numel, n_bits):
        raise ValueError("invalid packed-v storage length")
    if v_scales.shape != (num_blocks, 2):
        raise ValueError("invalid packed-v scale shape")

    bc1 = 1.0 - beta1**step
    bc2_sqrt = math.sqrt(1.0 - beta2**step)
    if previous_step > 0:
        previous_bc1 = 1.0 - beta1**previous_step
        previous_bc2_sqrt = math.sqrt(1.0 - beta2**previous_step)
    else:
        previous_bc1 = 0.0
        previous_bc2_sqrt = 1.0
    programs = min(num_blocks, 4096)
    _fused_ustate_adamw_kernel[(programs,)](
        param,
        grad,
        q_codes,
        q_means,
        q_decode_step,
        q_encode_step,
        q_rms_accumulator,
        v_codes,
        v_scales,
        previous_bc1,
        previous_bc2_sqrt,
        bc1,
        bc2_sqrt,
        beta1,
        1.0 - beta1,
        beta2,
        1.0 - beta2,
        lr / bc1,
        eps,
        1.0 - lr * weight_decay,
        m_seed,
        v_seed,
        original_numel,
        num_blocks,
        N_BITS=n_bits,
        BLOCK_SIZE=block_size,
        M_BLOCK_SIZE=m_block_size,
        M_BLOCKS_PER_V=block_size // m_block_size,
        V_BYTES_PER_BLOCK=(block_size * n_bits) // 8,
        Q_BYTES_PER_BLOCK=block_size // 4,
    )
    _finalize_ustate_scale_kernel[(1,)](
        q_decode_step,
        q_encode_step,
        q_rms_accumulator,
        m_step_factor,
        original_numel,
    )


__all__ = ["triton_fused_ustate_adamw_step", "triton_supports_block_size"]
