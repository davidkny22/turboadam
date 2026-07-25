"""AdamW with packed UState and 1Q optimizer state."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch
from torch.optim import Optimizer

from turboadam.oneq import initialize_v_logscale
from turboadam.quantize import (
    decompress_v_state,
    recompress_v_state,
    restore_v_state,
)
from turboadam.ustate import (
    DEFAULT_STEP_FACTOR,
    USTATE_MEAN_BLOCK_SIZE,
    encode_first_moment,
    initialize_ustate,
    reconstruct_first_moment,
    restore_ustate,
)
from turboadam.utils import BLOCK_SIZE, finite_scalar, validate_block_size

try:
    from turboadam.triton_kernels import (
        triton_fused_ustate_adamw_step as _triton_fused_step,
    )
    from turboadam.triton_kernels import (
        triton_supports_block_size as _triton_supports_block_size,
    )

    _HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    _HAS_TRITON = False
    _triton_fused_step = None
    _triton_supports_block_size = None


class TurboAdam(Optimizer):
    """AdamW with compact first- and second-moment persistence."""

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict],
        lr: float = 1.0e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1.0e-8,
        weight_decay: float = 0.0,
        block_size: int = BLOCK_SIZE,
        v_bits: int = 4,
        compress_m: bool = True,
        compress_v: bool = True,
        capturable: bool = False,
        min_m_compress_elements: int = 4096,
        min_v_compress_elements: int = 4096,
        m_block_size: int = USTATE_MEAN_BLOCK_SIZE,
        m_step_factor: float = DEFAULT_STEP_FACTOR,
        rounding_seed: int = 0x12345678,
    ) -> None:
        lr = finite_scalar(lr, "lr", non_negative=True)
        eps = finite_scalar(eps, "eps", non_negative=True)
        weight_decay = finite_scalar(weight_decay, "weight_decay", non_negative=True)
        beta1, beta2 = map(float, betas)
        if not math.isfinite(beta1) or not 0.0 <= beta1 < 1.0:
            raise ValueError(f"invalid beta1: {beta1}")
        if not math.isfinite(beta2) or not 0.0 <= beta2 < 1.0:
            raise ValueError(f"invalid beta2: {beta2}")
        if compress_m and beta1 * beta1 >= beta2:
            raise ValueError(
                "UState requires beta1**2 < beta2; disable compress_m for this "
                "beta pair"
            )
        validate_block_size(block_size)
        if v_bits not in (2, 3, 4, 6, 8):
            raise ValueError(f"v_bits must be one of {{2, 3, 4, 6, 8}}, got {v_bits}")
        validate_block_size(m_block_size)
        if block_size % m_block_size != 0:
            raise ValueError("block_size must be divisible by m_block_size")
        if not isinstance(min_m_compress_elements, int) or min_m_compress_elements < 0:
            raise ValueError("min_m_compress_elements must be a non-negative integer")
        if not isinstance(min_v_compress_elements, int) or min_v_compress_elements < 0:
            raise ValueError("min_v_compress_elements must be a non-negative integer")
        if capturable:
            raise NotImplementedError("TurboAdam does not support CUDA graph capture")
        m_step_factor = finite_scalar(m_step_factor, "m_step_factor", non_negative=True)
        if m_step_factor == 0.0:
            raise ValueError("m_step_factor must be positive")

        defaults = {
            "lr": lr,
            "betas": (beta1, beta2),
            "eps": eps,
            "weight_decay": weight_decay,
            "block_size": block_size,
            "v_bits": v_bits,
            "compress_m": bool(compress_m),
            "compress_v": bool(compress_v),
            "capturable": False,
            "min_m_compress_elements": min_m_compress_elements,
            "min_v_compress_elements": min_v_compress_elements,
            "m_block_size": m_block_size,
            "m_step_factor": m_step_factor,
            "rounding_seed": int(rounding_seed) & 0xFFFFFFFF,
        }
        super().__init__(params, defaults)
        for group in self.param_groups:
            self._validate_group(group)

    @staticmethod
    def _validate_group(group: dict) -> None:
        beta1, beta2 = map(float, group["betas"])
        if not math.isfinite(beta1) or not 0.0 <= beta1 < 1.0:
            raise ValueError(f"invalid beta1: {beta1}")
        if not math.isfinite(beta2) or not 0.0 <= beta2 < 1.0:
            raise ValueError(f"invalid beta2: {beta2}")
        if bool(group["compress_m"]) and beta1 * beta1 >= beta2:
            raise ValueError(
                "UState requires beta1**2 < beta2; disable compress_m for this "
                "parameter group"
            )
        validate_block_size(int(group["block_size"]))
        if int(group["v_bits"]) not in (2, 3, 4, 6, 8):
            raise ValueError("v_bits must be one of {2, 3, 4, 6, 8}")
        validate_block_size(int(group["m_block_size"]))
        if int(group["block_size"]) % int(group["m_block_size"]):
            raise ValueError("block_size must be divisible by m_block_size")
        if int(group["min_m_compress_elements"]) < 0:
            raise ValueError("min_m_compress_elements must be non-negative")
        if int(group["min_v_compress_elements"]) < 0:
            raise ValueError("min_v_compress_elements must be non-negative")
        if bool(group["capturable"]):
            raise NotImplementedError("TurboAdam does not support CUDA graph capture")
        finite_scalar(group["lr"], "lr", non_negative=True)
        finite_scalar(group["eps"], "eps", non_negative=True)
        finite_scalar(group["weight_decay"], "weight_decay", non_negative=True)
        step_factor = finite_scalar(
            group["m_step_factor"], "m_step_factor", non_negative=True
        )
        if step_factor == 0.0:
            raise ValueError("m_step_factor must be positive")
        group["rounding_seed"] = int(group["rounding_seed"]) & 0xFFFFFFFF

    def add_param_group(self, param_group: dict) -> None:
        """Add and validate a parameter group."""
        super().add_param_group(param_group)
        try:
            self._validate_group(self.param_groups[-1])
        except Exception:
            self.param_groups.pop()
            raise

    @staticmethod
    def _mix32(value: int) -> int:
        value &= 0xFFFFFFFF
        value ^= value >> 16
        value = (value * 0x7FEB352D) & 0xFFFFFFFF
        value ^= value >> 15
        value = (value * 0x846CA68B) & 0xFFFFFFFF
        value ^= value >> 16
        return value & 0xFFFFFFFF

    @classmethod
    def _seed_for(
        cls,
        seed_base: int,
        step: int,
        group_index: int,
        parameter_index: int,
        stream: int,
    ) -> int:
        value = int(seed_base) & 0xFFFFFFFF
        value ^= (step * 0x9E3779B9) & 0xFFFFFFFF
        value ^= ((group_index + 1) * 0x85EBCA6B) & 0xFFFFFFFF
        value ^= ((parameter_index + 1) * 0xC2B2AE35) & 0xFFFFFFFF
        value ^= (stream * 0x27D4EB2D) & 0xFFFFFFFF
        return cls._mix32(value)

    def _initialize_state(self, parameter: torch.Tensor, group: dict) -> dict:
        state = self.state[parameter]
        if state:
            return state
        state["step"] = 0
        use_ustate = bool(
            group["compress_m"]
            and parameter.numel() >= int(group["min_m_compress_elements"])
            and parameter.numel() > 0
        )
        state["_use_ustate"] = use_ustate
        if use_ustate:
            state["ustate"] = initialize_ustate(
                parameter.numel(),
                device=parameter.device,
                mean_block_size=int(group["m_block_size"]),
                storage_block_size=int(group["block_size"]),
                step_factor=float(group["m_step_factor"]),
            )
        else:
            state["exp_avg"] = torch.zeros_like(parameter, dtype=torch.float32)

        use_v_state = bool(
            group["compress_v"]
            and parameter.numel() >= int(group["min_v_compress_elements"])
            and parameter.numel() > 0
        )
        state["_use_v_state"] = use_v_state
        if use_v_state:
            state["v_state"] = initialize_v_logscale(
                parameter.shape,
                device=parameter.device,
                n_bits=int(group["v_bits"]),
                block_size=int(group["block_size"]),
            )
        else:
            state["exp_avg_sq"] = torch.zeros_like(parameter, dtype=torch.float32)
        return state

    def _can_use_triton(
        self,
        parameter: torch.Tensor,
        gradient: torch.Tensor,
        state: dict,
        group: dict,
    ) -> bool:
        if not _HAS_TRITON or not parameter.is_cuda:
            return False
        if not state["_use_ustate"] or not state["_use_v_state"]:
            return False
        if not parameter.is_contiguous() or not gradient.is_contiguous():
            return False
        if not _triton_supports_block_size(int(group["block_size"])):
            return False
        ustate = state["ustate"]
        v_state = state["v_state"]
        return bool(
            ustate["representation"] == "ustate"
            and int(ustate["mean_block_size"]) == int(group["m_block_size"])
            and int(ustate["storage_block_size"]) == int(group["block_size"])
            and v_state["indices"].is_contiguous()
            and v_state["scales"].is_contiguous()
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None):
        """Perform one AdamW step and persist its compact states."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_index, group in enumerate(self.param_groups):
            lr = float(group["lr"])
            beta1, beta2 = map(float, group["betas"])
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])
            block_size = int(group["block_size"])
            seed_base = int(group["rounding_seed"])

            for parameter_index, parameter in enumerate(group["params"]):
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("TurboAdam does not support sparse gradients")
                if parameter.is_complex() or gradient.is_complex():
                    raise RuntimeError("TurboAdam does not support complex parameters")
                if parameter.numel() == 0:
                    continue

                state = self._initialize_state(parameter, group)
                previous_step = int(state["step"])
                step = previous_step + 1
                m_seed = self._seed_for(
                    seed_base, step, group_index, parameter_index, 1
                )
                v_seed = self._seed_for(
                    seed_base, step, group_index, parameter_index, 2
                )

                if self._can_use_triton(parameter, gradient, state, group):
                    ustate = state["ustate"]
                    v_state = state["v_state"]
                    _triton_fused_step(
                        parameter,
                        gradient,
                        ustate["codes"],
                        ustate["means"],
                        ustate["decode_step"],
                        ustate["encode_step"],
                        ustate["rms_accumulator"],
                        v_state["indices"],
                        v_state["scales"],
                        previous_step=previous_step,
                        step=step,
                        beta1=beta1,
                        beta2=beta2,
                        lr=lr,
                        eps=eps,
                        weight_decay=weight_decay,
                        n_bits=int(v_state["n_bits"]),
                        block_size=block_size,
                        m_block_size=int(group["m_block_size"]),
                        original_numel=parameter.numel(),
                        m_step_factor=float(group["m_step_factor"]),
                        m_seed=m_seed,
                        v_seed=v_seed,
                    )
                    state["step"] = step
                    continue

                gradient_fp32 = gradient.float()
                if state["_use_v_state"]:
                    current_v = decompress_v_state(state["v_state"]).float()
                else:
                    current_v = state["exp_avg_sq"]

                if state["_use_ustate"]:
                    current_m = reconstruct_first_moment(
                        state["ustate"],
                        current_v,
                        previous_step,
                        beta1,
                        beta2,
                        eps,
                    )
                else:
                    current_m = state["exp_avg"]
                current_m.lerp_(gradient_fp32, 1.0 - beta1)
                current_v.mul_(beta2).addcmul_(
                    gradient_fp32, gradient_fp32, value=1.0 - beta2
                )

                bias_correction1 = 1.0 - beta1**step
                bias_correction2_sqrt = math.sqrt(1.0 - beta2**step)
                if state["_use_v_state"]:
                    persisted_v = recompress_v_state(
                        state["v_state"], current_v, seed=v_seed
                    )
                    denominator = current_v.clamp_min_(0).sqrt_()
                else:
                    persisted_v = current_v
                    denominator = current_v.clamp_min(0).sqrt()
                denominator.div_(bias_correction2_sqrt).add_(eps)

                if weight_decay != 0.0:
                    parameter.mul_(1.0 - lr * weight_decay)
                parameter.addcdiv_(
                    current_m,
                    denominator,
                    value=-(lr / bias_correction1),
                )

                if state["_use_ustate"]:
                    state["ustate"] = encode_first_moment(
                        current_m,
                        persisted_v,
                        step,
                        beta1,
                        beta2,
                        eps,
                        seed=m_seed,
                        mean_block_size=int(group["m_block_size"]),
                        storage_block_size=block_size,
                        step_factor=float(group["m_step_factor"]),
                        out=state["ustate"],
                        reuse_m_buffer=True,
                    )
                state["step"] = step

        return loss

    def load_state_dict(self, state_dict):
        """Load a current TurboAdam checkpoint without casting codec storage."""
        saved_for_parameter: dict[torch.Tensor, dict] = {}
        saved_groups = state_dict["param_groups"]
        if len(saved_groups) != len(self.param_groups):
            raise ValueError(
                "loaded state dict has a different number of parameter groups"
            )
        for saved_group, current_group in zip(
            saved_groups, self.param_groups, strict=True
        ):
            if len(saved_group["params"]) != len(current_group["params"]):
                raise ValueError("loaded parameter group size does not match")
            for saved_id, parameter in zip(
                saved_group["params"], current_group["params"], strict=True
            ):
                if saved_id in state_dict["state"]:
                    saved_for_parameter[parameter] = state_dict["state"][saved_id]

        super().load_state_dict(state_dict)
        for group in self.param_groups:
            self._validate_group(group)
            for parameter in group["params"]:
                source = saved_for_parameter.get(parameter)
                if source is None:
                    continue
                expected = {"step", "_use_ustate", "_use_v_state"}
                expected.add("ustate" if source["_use_ustate"] else "exp_avg")
                expected.add("v_state" if source["_use_v_state"] else "exp_avg_sq")
                if set(source) != expected:
                    missing = sorted(expected - set(source))
                    unexpected = sorted(set(source) - expected)
                    raise ValueError(
                        f"invalid optimizer state; missing={missing}, "
                        f"unexpected={unexpected}"
                    )

                state = self.state[parameter]
                saved_step = source["step"]
                if isinstance(saved_step, torch.Tensor):
                    if saved_step.numel() != 1:
                        raise ValueError("optimizer step must be scalar")
                    saved_step = saved_step.detach().cpu().item()
                state.clear()
                state["step"] = int(saved_step)
                state["_use_ustate"] = bool(source["_use_ustate"])
                state["_use_v_state"] = bool(source["_use_v_state"])
                if state["_use_ustate"]:
                    state["ustate"] = restore_ustate(
                        source["ustate"], device=parameter.device
                    )
                    if int(state["ustate"]["original_numel"]) != parameter.numel():
                        raise ValueError("UState size does not match parameter")
                else:
                    state["exp_avg"] = (
                        source["exp_avg"]
                        .detach()
                        .to(
                            device=parameter.device,
                            dtype=torch.float32,
                            copy=True,
                        )
                        .contiguous()
                    )
                    if state["exp_avg"].shape != parameter.shape:
                        raise ValueError("exp_avg shape does not match parameter")
                if state["_use_v_state"]:
                    state["v_state"] = restore_v_state(
                        source["v_state"], device=parameter.device
                    )
                    if int(state["v_state"]["original_length"]) != parameter.numel():
                        raise ValueError("v_state size does not match parameter")
                else:
                    state["exp_avg_sq"] = (
                        source["exp_avg_sq"]
                        .detach()
                        .to(
                            device=parameter.device,
                            dtype=torch.float32,
                            copy=True,
                        )
                        .contiguous()
                    )
                    if state["exp_avg_sq"].shape != parameter.shape:
                        raise ValueError("exp_avg_sq shape does not match parameter")
        return None
