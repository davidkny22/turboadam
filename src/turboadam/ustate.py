"""Frame-stable compression for Adam's first moment."""

from __future__ import annotations

import math

import torch

from turboadam.quantize import (
    counter_uniform,
    pack_nbit_indices,
    packed_index_numel,
    unpack_nbit_indices,
)
from turboadam.utils import BLOCK_SIZE, ceil_div, validate_block_size

USTATE_BITS = 2
USTATE_LEVEL_CENTER = 1.5
USTATE_METADATA_DTYPE = torch.bfloat16
USTATE_MEAN_BLOCK_SIZE = 64
DEFAULT_STEP_FACTOR = 1.1

_STATE_KEYS = {
    "representation",
    "codes",
    "means",
    "decode_step",
    "encode_step",
    "rms_accumulator",
    "original_numel",
    "mean_block_size",
    "storage_block_size",
    "step_factor",
}


def ustate_bound(beta1: float, beta2: float, step: int) -> float:
    """Return the coordinatewise update-state bound when it is finite."""
    if step <= 0:
        return 0.0
    if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
        raise ValueError("betas must lie in [0, 1)")
    ratio = beta1 * beta1 / beta2 if beta2 > 0.0 else float("inf")
    if ratio >= 1.0:
        return float("inf")
    geometric = (1.0 - ratio**step) / (1.0 - ratio)
    numerator = (1.0 - beta1) * math.sqrt(geometric)
    numerator *= math.sqrt(1.0 - beta2**step)
    denominator = math.sqrt(1.0 - beta2) * (1.0 - beta1**step)
    return numerator / denominator


def _block_counts(
    original_numel: int,
    block_size: int,
    *,
    device: torch.device,
) -> torch.Tensor:
    num_blocks = ceil_div(original_numel, block_size)
    counts = torch.full((num_blocks,), block_size, dtype=torch.float32, device=device)
    final_count = original_numel - (num_blocks - 1) * block_size
    if final_count != block_size:
        counts[-1] = final_count
    return counts


def _mask_padding_(blocks: torch.Tensor, original_numel: int, block_size: int) -> None:
    remainder = original_numel % block_size
    if remainder:
        blocks[-1, remainder:] = 0


def _reuse_tensor(out: dict | None, key: str, value: torch.Tensor) -> torch.Tensor:
    if out is None:
        return value
    target = out.get(key)
    if (
        isinstance(target, torch.Tensor)
        and target.shape == value.shape
        and target.dtype == value.dtype
        and target.device == value.device
    ):
        target.copy_(value)
        return target
    return value


def initialize_ustate(
    original_numel: int,
    *,
    device: torch.device,
    mean_block_size: int = USTATE_MEAN_BLOCK_SIZE,
    storage_block_size: int = BLOCK_SIZE,
    step_factor: float = DEFAULT_STEP_FACTOR,
) -> dict:
    """Create a compact zero first-moment state."""
    validate_block_size(mean_block_size)
    validate_block_size(storage_block_size)
    if storage_block_size % mean_block_size != 0:
        raise ValueError("storage_block_size must be divisible by mean_block_size")
    if original_numel <= 0:
        raise ValueError(f"original_numel must be positive, got {original_numel}")
    if not math.isfinite(step_factor) or step_factor <= 0.0:
        raise ValueError("step_factor must be finite and positive")

    storage_numel = ceil_div(original_numel, storage_block_size) * storage_block_size
    num_mean_blocks = ceil_div(original_numel, mean_block_size)
    return {
        "representation": "ustate",
        "codes": torch.zeros(
            packed_index_numel(storage_numel, USTATE_BITS),
            dtype=torch.uint8,
            device=device,
        ),
        "means": torch.zeros(
            num_mean_blocks, dtype=USTATE_METADATA_DTYPE, device=device
        ),
        "decode_step": torch.zeros(1, dtype=torch.float32, device=device),
        "encode_step": torch.full(
            (1,), float(step_factor), dtype=torch.float32, device=device
        ),
        "rms_accumulator": torch.zeros(1, dtype=torch.float32, device=device),
        "original_numel": original_numel,
        "mean_block_size": mean_block_size,
        "storage_block_size": storage_block_size,
        "step_factor": float(step_factor),
    }


def encode_ustate(
    update_state: torch.Tensor,
    mean_block_size: int = USTATE_MEAN_BLOCK_SIZE,
    *,
    seed: int,
    step_factor: float = DEFAULT_STEP_FACTOR,
    storage_block_size: int | None = None,
    out: dict | None = None,
) -> dict:
    """Encode normalized Adam update units into a packed UState payload."""
    validate_block_size(mean_block_size)
    if storage_block_size is None:
        storage_block_size = mean_block_size
    validate_block_size(storage_block_size)
    if storage_block_size % mean_block_size != 0:
        raise ValueError("storage_block_size must be divisible by mean_block_size")
    if update_state.numel() == 0:
        raise ValueError("cannot encode an empty first-moment state")
    if not math.isfinite(step_factor) or step_factor <= 0.0:
        raise ValueError("step_factor must be finite and positive")

    flat = update_state.reshape(-1).float()
    original_numel = flat.numel()
    mean_padded_numel = ceil_div(original_numel, mean_block_size) * mean_block_size
    storage_numel = ceil_div(original_numel, storage_block_size) * storage_block_size
    padded = flat.new_zeros(storage_numel)
    padded[:original_numel].copy_(flat)

    num_mean_blocks = mean_padded_numel // mean_block_size
    blocks = padded[:mean_padded_numel].reshape(num_mean_blocks, mean_block_size)
    counts = _block_counts(original_numel, mean_block_size, device=flat.device)
    means = (blocks.sum(dim=1) / counts).to(USTATE_METADATA_DTYPE)
    residual = blocks - means.float().unsqueeze(1)
    _mask_padding_(residual, original_numel, mean_block_size)

    next_step = (
        (torch.sqrt(residual.square().sum() / float(original_numel)) * step_factor)
        .reshape(1)
        .float()
    )
    if out is not None:
        validate_ustate(out)
        used_step = out["encode_step"].float().reshape(1).clone()
    else:
        used_step = next_step.clone()

    safe_step = used_step.clamp_min(torch.finfo(torch.float32).tiny)
    position = (residual / safe_step + USTATE_LEVEL_CENTER).clamp(0.0, 3.0)
    lower = position.floor()
    probability_up = position - lower
    random = counter_uniform(
        mean_padded_numel,
        seed,
        device=flat.device,
        antithetic_pairs=True,
    ).reshape(num_mean_blocks, mean_block_size)
    codes = lower + (random < probability_up).to(lower.dtype)
    codes = torch.where(used_step > 0.0, codes, torch.zeros_like(codes)).to(torch.uint8)

    code_storage = torch.zeros(storage_numel, dtype=torch.uint8, device=flat.device)
    code_storage[:mean_padded_numel].copy_(codes.reshape(-1))
    packed = pack_nbit_indices(code_storage, USTATE_BITS)
    accumulator = flat.new_zeros(1, dtype=torch.float32)

    return {
        "representation": "ustate",
        "codes": _reuse_tensor(out, "codes", packed),
        "means": _reuse_tensor(out, "means", means),
        "decode_step": _reuse_tensor(out, "decode_step", used_step),
        "encode_step": _reuse_tensor(out, "encode_step", next_step),
        "rms_accumulator": _reuse_tensor(out, "rms_accumulator", accumulator),
        "original_numel": original_numel,
        "mean_block_size": mean_block_size,
        "storage_block_size": storage_block_size,
        "step_factor": float(step_factor),
    }


def validate_ustate(state: dict) -> None:
    """Validate the current UState schema and physical tensor layout."""
    if set(state) != _STATE_KEYS:
        missing = sorted(_STATE_KEYS - set(state))
        unexpected = sorted(set(state) - _STATE_KEYS)
        raise ValueError(
            f"invalid UState fields; missing={missing}, unexpected={unexpected}"
        )
    if state["representation"] != "ustate":
        raise ValueError("invalid UState representation")

    original_numel = int(state["original_numel"])
    mean_block_size = int(state["mean_block_size"])
    storage_block_size = int(state["storage_block_size"])
    validate_block_size(mean_block_size)
    validate_block_size(storage_block_size)
    if original_numel <= 0:
        raise ValueError("UState original_numel must be positive")
    if storage_block_size % mean_block_size != 0:
        raise ValueError("storage_block_size must be divisible by mean_block_size")
    if (
        not math.isfinite(float(state["step_factor"]))
        or float(state["step_factor"]) <= 0.0
    ):
        raise ValueError("UState step_factor must be finite and positive")

    storage_numel = ceil_div(original_numel, storage_block_size) * storage_block_size
    num_mean_blocks = ceil_div(original_numel, mean_block_size)
    tensors = {
        "codes": (
            torch.uint8,
            packed_index_numel(storage_numel, USTATE_BITS),
        ),
        "means": (USTATE_METADATA_DTYPE, num_mean_blocks),
        "decode_step": (torch.float32, 1),
        "encode_step": (torch.float32, 1),
        "rms_accumulator": (torch.float32, 1),
    }
    for key, (dtype, numel) in tensors.items():
        value = state[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"UState {key} must be a tensor")
        if value.dtype != dtype or value.numel() != numel or not value.is_contiguous():
            raise ValueError(
                f"invalid UState {key}: expected contiguous {dtype} tensor with "
                f"{numel} values"
            )


def restore_ustate(state: dict, *, device: torch.device) -> dict:
    """Copy a current UState checkpoint to a parameter device."""
    validate_ustate(state)
    restored = {
        key: value.detach().to(device=device, copy=True).contiguous()
        if isinstance(value, torch.Tensor)
        else value
        for key, value in state.items()
    }
    validate_ustate(restored)
    return restored


def decode_ustate(state: dict) -> torch.Tensor:
    """Decode a packed UState payload to normalized fp32 update units."""
    validate_ustate(state)
    original_numel = int(state["original_numel"])
    mean_block_size = int(state["mean_block_size"])
    storage_block_size = int(state["storage_block_size"])
    num_mean_blocks = ceil_div(original_numel, mean_block_size)
    mean_padded_numel = num_mean_blocks * mean_block_size
    storage_numel = ceil_div(original_numel, storage_block_size) * storage_block_size
    codes = unpack_nbit_indices(state["codes"], USTATE_BITS, storage_numel)
    levels = (
        codes[:mean_padded_numel].reshape(num_mean_blocks, mean_block_size).float()
        - USTATE_LEVEL_CENTER
    )
    raw = levels * state["decode_step"].float().reshape(())
    _mask_padding_(raw, original_numel, mean_block_size)
    counts = _block_counts(original_numel, mean_block_size, device=raw.device)
    raw -= (raw.sum(dim=1) / counts).unsqueeze(1)
    decoded = state["means"].float().reshape(-1, 1) + raw
    _mask_padding_(decoded, original_numel, mean_block_size)
    return decoded.reshape(-1)[:original_numel]


def reconstruct_first_moment(
    state: dict,
    persisted_v: torch.Tensor,
    step: int,
    beta1: float,
    beta2: float,
    eps: float,
) -> torch.Tensor:
    """Reconstruct the first moment from UState and its persisted v frame."""
    if step <= 0:
        return torch.zeros_like(persisted_v, dtype=torch.float32)
    update_state = decode_ustate(state).reshape(persisted_v.shape)
    bias_correction1 = 1.0 - beta1**step
    bias_correction2_sqrt = math.sqrt(1.0 - beta2**step)
    denominator = persisted_v.float().clamp_min(0).sqrt()
    denominator.div_(bias_correction2_sqrt).add_(eps)
    update_state.mul_(bias_correction1).mul_(denominator)
    return update_state


def encode_first_moment(
    first_moment: torch.Tensor,
    persisted_v: torch.Tensor,
    step: int,
    beta1: float,
    beta2: float,
    eps: float,
    *,
    seed: int,
    mean_block_size: int = USTATE_MEAN_BLOCK_SIZE,
    storage_block_size: int = BLOCK_SIZE,
    step_factor: float = DEFAULT_STEP_FACTOR,
    out: dict | None = None,
    reuse_m_buffer: bool = False,
) -> dict:
    """Normalize and encode the current first moment."""
    if step <= 0:
        raise ValueError(f"step must be positive, got {step}")
    update_state = (
        first_moment
        if reuse_m_buffer and first_moment.dtype == torch.float32
        else first_moment.float().clone()
    )
    bias_correction1 = 1.0 - beta1**step
    bias_correction2_sqrt = math.sqrt(1.0 - beta2**step)
    denominator = persisted_v.float().clamp_min(0).sqrt()
    denominator.div_(bias_correction2_sqrt).add_(eps)
    update_state.div_(bias_correction1).div_(denominator)
    return encode_ustate(
        update_state,
        mean_block_size,
        seed=seed,
        step_factor=step_factor,
        storage_block_size=storage_block_size,
        out=out,
    )


__all__ = [
    "DEFAULT_STEP_FACTOR",
    "USTATE_BITS",
    "USTATE_MEAN_BLOCK_SIZE",
    "decode_ustate",
    "encode_first_moment",
    "encode_ustate",
    "initialize_ustate",
    "reconstruct_first_moment",
    "restore_ustate",
    "ustate_bound",
    "validate_ustate",
]
