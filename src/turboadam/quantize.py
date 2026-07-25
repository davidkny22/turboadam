"""Packed log-scale quantization for Adam's positive second moment."""

from __future__ import annotations

import math

import torch

from turboadam.utils import BLOCK_SIZE, ceil_div, pad_to_blocks, validate_block_size

SUPPORTED_N_BITS = (2, 3, 4, 6, 8)
MIN_POSITIVE = 1.0e-38
MAX_SAFE_LOG_FP16 = 88.6875

_V_STATE_KEYS = {
    "indices",
    "scales",
    "n_bits",
    "original_shape",
    "original_length",
    "block_size",
}


def counter_uniform(
    num_values: int,
    seed: int,
    *,
    device: torch.device,
    antithetic_pairs: bool = True,
) -> torch.Tensor:
    """Generate reproducible counter-based fp32 values in [0, 1)."""
    if num_values < 0:
        raise ValueError(f"num_values must be non-negative, got {num_values}")
    if num_values == 0:
        return torch.empty(0, dtype=torch.float32, device=device)

    key_count = ceil_div(num_values, 2) if antithetic_pairs else num_values
    keys = torch.arange(key_count, dtype=torch.int64, device=device)
    values = (keys ^ (int(seed) & 0xFFFFFFFF)) & 0xFFFFFFFF
    values = ((values ^ (values >> 16)) * 0x7FEB352D) & 0xFFFFFFFF
    values = ((values ^ (values >> 15)) * 0x846CA68B) & 0xFFFFFFFF
    values = (values ^ (values >> 16)) & 0xFFFFFFFF
    base = ((values >> 8) & 0xFFFFFF).to(torch.float32) * 2.0**-24
    if not antithetic_pairs:
        return base

    uniform = torch.empty(num_values, dtype=torch.float32, device=device)
    uniform[0::2] = base
    uniform[1::2] = (1.0 - base[: num_values // 2]).clamp_max(1.0 - 2.0**-24)
    return uniform


def packed_index_numel(num_values: int, n_bits: int) -> int:
    """Return the bytes required for a packed n-bit index stream."""
    if num_values < 0:
        raise ValueError(f"num_values must be non-negative, got {num_values}")
    if n_bits not in SUPPORTED_N_BITS:
        raise ValueError(f"n_bits must be one of {SUPPORTED_N_BITS}, got {n_bits}")
    return ceil_div(num_values * n_bits, 8)


def pack_nbit_indices(indices: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Pack uint8 indices into a little-endian n-bit stream."""
    if n_bits not in SUPPORTED_N_BITS:
        raise ValueError(f"n_bits must be one of {SUPPORTED_N_BITS}, got {n_bits}")
    flat = indices.reshape(-1).to(torch.uint8)
    original_numel = flat.numel()
    if n_bits == 8:
        return flat.contiguous()

    group_size = {2: 4, 3: 8, 4: 2, 6: 4}[n_bits]
    remainder = flat.numel() % group_size
    if remainder:
        padded = flat.new_zeros(flat.numel() + group_size - remainder)
        padded[: flat.numel()].copy_(flat)
        flat = padded
    values = flat.to(torch.int32)

    if n_bits == 2:
        groups = values.reshape(-1, 4)
        packed = (
            groups[:, 0]
            | (groups[:, 1] << 2)
            | (groups[:, 2] << 4)
            | (groups[:, 3] << 6)
        ).to(torch.uint8)
    elif n_bits == 4:
        groups = values.reshape(-1, 2)
        packed = (groups[:, 0] | (groups[:, 1] << 4)).to(torch.uint8)
    elif n_bits == 3:
        groups = values.reshape(-1, 8)
        packed = torch.empty(
            (groups.shape[0], 3), dtype=torch.uint8, device=indices.device
        )
        packed[:, 0] = (
            groups[:, 0] | (groups[:, 1] << 3) | ((groups[:, 2] & 0x03) << 6)
        ).to(torch.uint8)
        packed[:, 1] = (
            (groups[:, 2] >> 2)
            | (groups[:, 3] << 1)
            | (groups[:, 4] << 4)
            | ((groups[:, 5] & 0x01) << 7)
        ).to(torch.uint8)
        packed[:, 2] = (
            (groups[:, 5] >> 1) | (groups[:, 6] << 2) | (groups[:, 7] << 5)
        ).to(torch.uint8)
    else:
        groups = values.reshape(-1, 4)
        packed = torch.empty(
            (groups.shape[0], 3), dtype=torch.uint8, device=indices.device
        )
        packed[:, 0] = (groups[:, 0] | ((groups[:, 1] & 0x03) << 6)).to(torch.uint8)
        packed[:, 1] = ((groups[:, 1] >> 2) | ((groups[:, 2] & 0x0F) << 4)).to(
            torch.uint8
        )
        packed[:, 2] = ((groups[:, 2] >> 4) | (groups[:, 3] << 2)).to(torch.uint8)
    return packed.reshape(-1)[: packed_index_numel(original_numel, n_bits)]


def unpack_nbit_indices(
    packed: torch.Tensor,
    n_bits: int,
    num_values: int,
) -> torch.Tensor:
    """Unpack a little-endian n-bit stream to uint8 indices."""
    if n_bits not in SUPPORTED_N_BITS:
        raise ValueError(f"n_bits must be one of {SUPPORTED_N_BITS}, got {n_bits}")
    if num_values < 0:
        raise ValueError(f"num_values must be non-negative, got {num_values}")
    flat = packed.reshape(-1).to(torch.uint8)
    expected = packed_index_numel(num_values, n_bits)
    if flat.numel() != expected:
        raise ValueError(
            f"packed stream must contain {expected} bytes, got {flat.numel()}"
        )
    if n_bits == 8:
        return flat[:num_values].contiguous()

    values = flat.to(torch.int32)
    if n_bits == 2:
        out = torch.empty((values.numel(), 4), dtype=torch.uint8, device=flat.device)
        out[:, 0] = (values & 0x03).to(torch.uint8)
        out[:, 1] = ((values >> 2) & 0x03).to(torch.uint8)
        out[:, 2] = ((values >> 4) & 0x03).to(torch.uint8)
        out[:, 3] = ((values >> 6) & 0x03).to(torch.uint8)
        return out.reshape(-1)[:num_values]
    if n_bits == 4:
        out = torch.empty((values.numel(), 2), dtype=torch.uint8, device=flat.device)
        out[:, 0] = (values & 0x0F).to(torch.uint8)
        out[:, 1] = ((values >> 4) & 0x0F).to(torch.uint8)
        return out.reshape(-1)[:num_values]

    remainder = values.numel() % 3
    if remainder:
        padded = values.new_zeros(values.numel() + 3 - remainder)
        padded[: values.numel()].copy_(values)
        values = padded
    groups = values.reshape(-1, 3)
    b0, b1, b2 = groups[:, 0], groups[:, 1], groups[:, 2]
    if n_bits == 3:
        out = torch.empty((groups.shape[0], 8), dtype=torch.uint8, device=flat.device)
        out[:, 0] = (b0 & 0x07).to(torch.uint8)
        out[:, 1] = ((b0 >> 3) & 0x07).to(torch.uint8)
        out[:, 2] = (((b0 >> 6) & 0x03) | ((b1 & 0x01) << 2)).to(torch.uint8)
        out[:, 3] = ((b1 >> 1) & 0x07).to(torch.uint8)
        out[:, 4] = ((b1 >> 4) & 0x07).to(torch.uint8)
        out[:, 5] = (((b1 >> 7) & 0x01) | ((b2 & 0x03) << 1)).to(torch.uint8)
        out[:, 6] = ((b2 >> 2) & 0x07).to(torch.uint8)
        out[:, 7] = ((b2 >> 5) & 0x07).to(torch.uint8)
    else:
        out = torch.empty((groups.shape[0], 4), dtype=torch.uint8, device=flat.device)
        out[:, 0] = (b0 & 0x3F).to(torch.uint8)
        out[:, 1] = (((b0 >> 6) & 0x03) | ((b1 & 0x0F) << 2)).to(torch.uint8)
        out[:, 2] = (((b1 >> 4) & 0x0F) | ((b2 & 0x03) << 4)).to(torch.uint8)
        out[:, 3] = ((b2 >> 2) & 0x3F).to(torch.uint8)
    return out.reshape(-1)[:num_values]


def _validate_quantizer_input(
    values: torch.Tensor,
    n_bits: int,
    block_size: int,
) -> None:
    validate_block_size(block_size)
    if values.ndim != 1:
        raise ValueError(f"values must be 1-D, got shape {tuple(values.shape)}")
    if values.numel() == 0:
        raise ValueError("cannot quantize an empty tensor")
    if values.numel() % block_size:
        raise ValueError(
            f"length {values.numel()} must be divisible by block_size {block_size}"
        )
    if n_bits not in SUPPORTED_N_BITS:
        raise ValueError(f"n_bits must be one of {SUPPORTED_N_BITS}, got {n_bits}")


def _outward_fp16_log_scales(
    log_min: torch.Tensor,
    log_max: torch.Tensor,
) -> torch.Tensor:
    log_min = log_min.clamp(max=MAX_SAFE_LOG_FP16)
    log_max = log_max.clamp(max=MAX_SAFE_LOG_FP16)
    min_half = log_min.to(torch.float16)
    max_half = log_max.to(torch.float16)
    dynamic = log_max > log_min

    min_out = torch.nextafter(min_half, torch.full_like(min_half, -float("inf")))
    max_out = torch.nextafter(max_half, torch.full_like(max_half, float("inf")))
    min_half = torch.where(dynamic & (min_half.float() > log_min), min_out, min_half)
    max_half = torch.where(
        dynamic & (max_half.float() < log_max),
        max_out.clamp(max=MAX_SAFE_LOG_FP16),
        max_half,
    )
    return torch.stack((min_half, max_half), dim=1)


def quantize_logscale(
    values: torch.Tensor,
    n_bits: int = 4,
    block_size: int = BLOCK_SIZE,
    stochastic_round: bool = False,
    *,
    seed: int = 0,
    out_indices: torch.Tensor | None = None,
    out_scales: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a padded positive tensor with packed endpoint log levels."""
    _validate_quantizer_input(values, n_bits, block_size)
    num_blocks = values.numel() // block_size
    blocks = values.reshape(num_blocks, block_size).float().clamp_min(MIN_POSITIVE)
    blocks = blocks.clamp_max(torch.finfo(torch.float32).max)
    log_blocks = blocks.log()
    scales = _outward_fp16_log_scales(log_blocks.amin(1), log_blocks.amax(1))

    intervals = (1 << n_bits) - 1
    log_min = scales[:, 0].float().unsqueeze(1)
    log_max = scales[:, 1].float().unsqueeze(1)
    step = (log_max - log_min) / intervals
    position = ((log_blocks - log_min) / step.clamp_min(1.0e-10)).clamp(
        0.0, float(intervals)
    )
    if stochastic_round:
        lower = position.floor().clamp(0, intervals)
        upper = (lower + 1.0).clamp(max=intervals)
        lower_value = torch.exp(log_min + lower * step)
        upper_value = torch.exp(log_min + upper * step)
        probability_up = (
            (blocks - lower_value)
            / (upper_value - lower_value).clamp_min(torch.finfo(torch.float32).tiny)
        ).clamp(0.0, 1.0)
        probability_up = torch.where(
            upper > lower, probability_up, torch.zeros_like(probability_up)
        )
        random = counter_uniform(
            probability_up.numel(),
            seed,
            device=probability_up.device,
            antithetic_pairs=True,
        ).reshape_as(probability_up)
        unpacked = lower + (random < probability_up).to(lower.dtype)
    else:
        unpacked = torch.floor(position + 0.5).clamp(0, intervals)
    packed = pack_nbit_indices(unpacked.to(torch.uint8).reshape(-1), n_bits)

    if out_indices is not None:
        if (
            out_indices.dtype != torch.uint8
            or out_indices.shape != packed.shape
            or out_indices.device != packed.device
        ):
            raise ValueError("out_indices must match packed uint8 storage")
        out_indices.copy_(packed)
        packed = out_indices
    if out_scales is not None:
        if (
            out_scales.dtype != torch.float16
            or out_scales.shape != scales.shape
            or out_scales.device != scales.device
        ):
            raise ValueError("out_scales must match fp16 endpoint storage")
        out_scales.copy_(scales)
        scales = out_scales
    return packed, scales


def dequantize_logscale(
    indices: torch.Tensor,
    scales: torch.Tensor,
    n_bits: int = 4,
    block_size: int = BLOCK_SIZE,
    original_numel: int = 0,
) -> torch.Tensor:
    """Reconstruct fp32 values from packed endpoint log levels."""
    validate_block_size(block_size)
    if n_bits not in SUPPORTED_N_BITS:
        raise ValueError(f"n_bits must be one of {SUPPORTED_N_BITS}, got {n_bits}")
    if scales.ndim != 2 or scales.shape[1] != 2:
        raise ValueError(f"scales must have shape (num_blocks, 2), got {scales.shape}")
    num_blocks = scales.shape[0]
    padded_numel = num_blocks * block_size
    if original_numel == 0:
        original_numel = padded_numel
    if not 0 <= original_numel <= padded_numel:
        raise ValueError(
            f"original_numel must lie in [0, {padded_numel}], got {original_numel}"
        )
    unpacked = unpack_nbit_indices(indices, n_bits, padded_numel)
    coordinate = unpacked.reshape(num_blocks, block_size).float()
    coordinate.div_(float((1 << n_bits) - 1))
    log_min = scales[:, 0].float().unsqueeze(1)
    log_max = scales[:, 1].float().unsqueeze(1)
    values = torch.exp(log_min + coordinate * (log_max - log_min))
    return values.reshape(-1)[:original_numel]


def initialize_v_state(
    shape: torch.Size | tuple[int, ...],
    *,
    device: torch.device,
    n_bits: int = 4,
    block_size: int = BLOCK_SIZE,
    initial_value: float = 1.0e-30,
) -> dict:
    """Create a constant packed second-moment state."""
    validate_block_size(block_size)
    if n_bits not in SUPPORTED_N_BITS:
        raise ValueError(f"n_bits must be one of {SUPPORTED_N_BITS}, got {n_bits}")
    original_length = math.prod(shape)
    if original_length <= 0:
        raise ValueError("compressed optimizer state requires a non-empty tensor")
    num_blocks = ceil_div(original_length, block_size)
    padded_numel = num_blocks * block_size
    log_value = math.log(max(float(initial_value), MIN_POSITIVE))
    state = {
        "indices": torch.zeros(
            packed_index_numel(padded_numel, n_bits),
            dtype=torch.uint8,
            device=device,
        ),
        "scales": torch.full(
            (num_blocks, 2), log_value, dtype=torch.float16, device=device
        ),
        "n_bits": n_bits,
        "original_shape": tuple(shape),
        "original_length": original_length,
        "block_size": block_size,
    }
    validate_v_state(state)
    return state


def validate_v_state(state: dict) -> None:
    """Validate the current packed second-moment schema."""
    if set(state) != _V_STATE_KEYS:
        missing = sorted(_V_STATE_KEYS - set(state))
        unexpected = sorted(set(state) - _V_STATE_KEYS)
        raise ValueError(
            f"invalid v_state fields; missing={missing}, unexpected={unexpected}"
        )
    n_bits = int(state["n_bits"])
    block_size = int(state["block_size"])
    original_length = int(state["original_length"])
    shape = tuple(state["original_shape"])
    validate_block_size(block_size)
    if n_bits not in SUPPORTED_N_BITS:
        raise ValueError(f"n_bits must be one of {SUPPORTED_N_BITS}, got {n_bits}")
    if original_length <= 0 or math.prod(shape) != original_length:
        raise ValueError("v_state shape and original_length disagree")
    num_blocks = ceil_div(original_length, block_size)
    padded_numel = num_blocks * block_size
    indices = state["indices"]
    scales = state["scales"]
    if (
        not isinstance(indices, torch.Tensor)
        or indices.dtype != torch.uint8
        or indices.numel() != packed_index_numel(padded_numel, n_bits)
        or not indices.is_contiguous()
    ):
        raise ValueError("invalid packed v_state indices")
    if (
        not isinstance(scales, torch.Tensor)
        or scales.dtype != torch.float16
        or scales.shape != (num_blocks, 2)
        or not scales.is_contiguous()
    ):
        raise ValueError("invalid v_state endpoint scales")


def restore_v_state(state: dict, *, device: torch.device) -> dict:
    """Copy a current packed second-moment checkpoint to a parameter device."""
    validate_v_state(state)
    restored = {
        key: value.detach().to(device=device, copy=True).contiguous()
        if isinstance(value, torch.Tensor)
        else value
        for key, value in state.items()
    }
    validate_v_state(restored)
    return restored


def decompress_v_state(state: dict) -> torch.Tensor:
    """Decode a packed second-moment state to its original shape."""
    validate_v_state(state)
    flat = dequantize_logscale(
        state["indices"],
        state["scales"],
        int(state["n_bits"]),
        int(state["block_size"]),
        int(state["original_length"]),
    )
    return flat.reshape(tuple(state["original_shape"]))


def recompress_v_state(
    state: dict,
    current_v: torch.Tensor,
    *,
    seed: int,
) -> torch.Tensor:
    """Persist a current second moment and return its decoded state."""
    validate_v_state(state)
    if current_v.numel() != int(state["original_length"]):
        raise ValueError("current_v size does not match v_state")
    flat = current_v.reshape(-1).float()
    block_size = int(state["block_size"])
    remainder = flat.numel() % block_size
    pad_value = flat[-remainder:].amin() if remainder else MIN_POSITIVE
    padded, _ = pad_to_blocks(flat, block_size, pad_value=pad_value)
    quantize_logscale(
        padded,
        int(state["n_bits"]),
        block_size,
        stochastic_round=True,
        seed=seed,
        out_indices=state["indices"],
        out_scales=state["scales"],
    )
    return dequantize_logscale(
        state["indices"],
        state["scales"],
        int(state["n_bits"]),
        block_size,
        int(state["original_length"]),
    ).reshape(current_v.shape)


def update_v_state(
    state: dict,
    gradient: torch.Tensor,
    beta2: float,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode, update, persist, and return current and persisted v."""
    if not 0.0 <= beta2 < 1.0:
        raise ValueError(f"beta2 must lie in [0, 1), got {beta2}")
    current = decompress_v_state(state).reshape(-1)
    grad = gradient.reshape(-1).float()
    if grad.numel() != current.numel():
        raise ValueError("gradient size does not match v_state")
    current.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
    persisted = recompress_v_state(state, current, seed=seed)
    return current.reshape(gradient.shape), persisted.reshape(gradient.shape)


__all__ = [
    "MAX_SAFE_LOG_FP16",
    "MIN_POSITIVE",
    "SUPPORTED_N_BITS",
    "counter_uniform",
    "decompress_v_state",
    "dequantize_logscale",
    "initialize_v_state",
    "pack_nbit_indices",
    "packed_index_numel",
    "quantize_logscale",
    "recompress_v_state",
    "restore_v_state",
    "unpack_nbit_indices",
    "update_v_state",
    "validate_v_state",
]
