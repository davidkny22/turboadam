"""TurboAdam optimizer: AdamW with compressed first and second moment states."""

from __future__ import annotations

import copy
import math

import torch
from torch.optim import Optimizer

from turboadam.costate import CoStateManager
from turboadam.oneq import init_compressed_v
from turboadam.quantize import fused_adam_update, pack_nbit_indices

try:
    from turboadam.triton_kernels import (
        triton_fused_adam_update as _triton_adam_update,
    )

    _HAS_TRITON = True
except (ImportError, ModuleNotFoundError):
    _HAS_TRITON = False
    _triton_adam_update = None


class TurboAdam(Optimizer):
    """AdamW with compressed first and second moment storage.

    The core design is unchanged:
      - CoState reconstructs and recompresses Adam's first moment every step.
      - 1Q stores Adam's second moment on a blockwise log-scale grid.
      - The current step uses the exact post-EMA v before v is recompressed.

    This implementation packs v indices to their real bit width, derives the
    redundant CoState scale instead of storing it, encodes costate labels inside
    the existing fp32 norm value, computes bias correction per parameter, and
    fuses the compressed-v parameter update so no full-size v state or
    denominator survives the step.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        block_size: int = 128,
        v_bits: int = 4,
        compress_m: bool = True,
        compress_v: bool = True,
        null_pct: float = 0.10,
        amp_pct: float = 0.90,
        error_feedback: bool = False,
        capturable: bool = False,
        min_m_compress_elements: int = 4096,
    ):
        if capturable:
            raise NotImplementedError(
                "CUDA graph capture is not yet supported. Set capturable=False."
            )

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            block_size=block_size,
            v_bits=v_bits,
            compress_m=compress_m,
            compress_v=compress_v,
            null_pct=null_pct,
            amp_pct=amp_pct,
            error_feedback=error_feedback,
            min_m_compress_elements=min_m_compress_elements,
        )
        self._validate_group_values(defaults)
        super().__init__(params, defaults)

    # ------------------------------------------------------------------
    # Validation and parameter groups
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_group_values(group: dict) -> None:
        lr = group["lr"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        block_size = group["block_size"]
        v_bits = group["v_bits"]
        null_pct = group["null_pct"]
        amp_pct = group["amp_pct"]
        min_m = group["min_m_compress_elements"]

        if not isinstance(lr, (int, float)) or not math.isfinite(lr) or lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not math.isfinite(beta1) or not 0.0 <= beta1 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {beta1}")
        if not math.isfinite(beta2) or not 0.0 <= beta2 < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {beta2}")
        if not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if (
            not isinstance(weight_decay, (int, float))
            or not math.isfinite(weight_decay)
            or weight_decay < 0.0
        ):
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if v_bits not in (2, 3, 4, 6, 8):
            raise ValueError(
                f"v_bits must be one of {{2, 3, 4, 6, 8}}, got {v_bits}"
            )
        if (
            not isinstance(block_size, int)
            or isinstance(block_size, bool)
            or block_size < 8
            or block_size % 8 != 0
            or block_size & (block_size - 1)
        ):
            raise ValueError(
                "block_size must be a power of two and a multiple of 8"
            )
        if not 0.0 <= null_pct < amp_pct <= 1.0:
            raise ValueError(
                "CoState thresholds must satisfy 0 <= null_pct < amp_pct <= 1"
            )
        if (
            not isinstance(min_m, int)
            or isinstance(min_m, bool)
            or min_m < 0
        ):
            raise ValueError(
                "min_m_compress_elements must be a non-negative integer"
            )

    def add_param_group(self, param_group: dict) -> None:
        # Validate the fully defaulted view before mutating optimizer state.
        merged = dict(self.defaults)
        merged.update({k: v for k, v in param_group.items() if k != "params"})
        self._validate_group_values(merged)
        super().add_param_group(param_group)

    # ------------------------------------------------------------------
    # State initialization and compatibility migration
    # ------------------------------------------------------------------

    @staticmethod
    def _seed_for(group_index: int, param_index: int, step: int) -> int:
        """Stable per-parameter, per-step seed independent of Python object ids."""
        x = 0x12345678
        x ^= ((group_index + 1) * 0x9E3779B1) & 0xFFFFFFFF
        x ^= ((param_index + 1) * 0x85EBCA77) & 0xFFFFFFFF
        x ^= (step * 0xC2B2AE3D) & 0xFFFFFFFF
        x ^= x >> 16
        x = (x * 0x7FEB352D) & 0xFFFFFFFF
        x ^= x >> 15
        return x & 0xFFFFFFFF

    def _initialize_state(self, p: torch.Tensor, group: dict) -> dict:
        state = self.state[p]
        state["step"] = 0
        compress_this_m = (
            group["compress_m"]
            and p.numel() >= group["min_m_compress_elements"]
        )
        state["_compress_m"] = compress_this_m
        if compress_this_m:
            state["m_mgr"] = CoStateManager(
                block_size=group["block_size"],
                null_pct=group["null_pct"],
                amp_pct=group["amp_pct"],
                error_feedback=group["error_feedback"],
            )
        else:
            state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)

        if group["compress_v"]:
            state["compressed_v"] = init_compressed_v(
                p.shape,
                device=p.device,
                n_bits=group["v_bits"],
                block_size=group["block_size"],
                packed=True,
            )
        else:
            state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        return state

    def load_state_dict(self, state_dict):
        """Load state, preserving codec dtypes and migrating legacy layouts."""
        # PyTorch's generic loader follows parameter dtype for floating states.
        # TurboAdam intentionally keeps uncompressed moments in fp32 and codec
        # endpoints in fp16, so retain direct references to the serialized
        # tensors and restore them without an fp16/bf16 round trip.
        saved_for_param = {}
        saved_groups = state_dict.get("param_groups", ())
        for saved_group, current_group in zip(
            saved_groups, self.param_groups, strict=False
        ):
            for saved_id, p in zip(
                saved_group.get("params", ()),
                current_group["params"],
                strict=False,
            ):
                saved_for_param[p] = state_dict.get("state", {}).get(saved_id, {})

        super().load_state_dict(state_dict)
        for group in self.param_groups:
            self._validate_group_values(group)
            for p in group["params"]:
                if p not in self.state:
                    continue
                state = self.state[p]
                saved_state = saved_for_param.get(p, {})
                state.pop("_bc1", None)
                state.pop("_bc2", None)

                step = state.get("step", 0)
                if isinstance(step, torch.Tensor):
                    step = int(step.detach().cpu().item())
                state["step"] = int(step)

                if "exp_avg" in state:
                    source = saved_state.get("exp_avg", state["exp_avg"])
                    state["exp_avg"] = source.detach().to(
                        device=p.device, dtype=torch.float32, copy=True
                    )
                if "exp_avg_sq" in state:
                    source = saved_state.get("exp_avg_sq", state["exp_avg_sq"])
                    state["exp_avg_sq"] = source.detach().to(
                        device=p.device, dtype=torch.float32, copy=True
                    )

                cv = state.get("compressed_v")
                if cv is not None:
                    saved_cv = saved_state.get("compressed_v", cv)
                    cv["indices"] = saved_cv["indices"].detach().to(
                        device=p.device, dtype=torch.uint8, copy=True
                    )
                    cv["scales"] = saved_cv["scales"].detach().to(
                        device=p.device, dtype=torch.float16, copy=True
                    )
                    cv.setdefault("original_shape", p.shape)
                    cv.setdefault("original_length", p.numel())
                    cv.setdefault("block_size", group["block_size"])
                    cv.setdefault("n_bits", group["v_bits"])
                    padded_numel = cv["scales"].shape[0] * cv["block_size"]
                    was_packed = cv.get("packed")
                    if was_packed is None:
                        was_packed = (
                            cv["n_bits"] < 8
                            and cv["indices"].numel() != padded_numel
                        )
                    if not was_packed and cv["n_bits"] < 8:
                        cv["indices"] = pack_nbit_indices(
                            cv["indices"], cv["n_bits"]
                        )
                    cv["packed"] = True
                    cv["codec_version"] = 2

                mgr = state.get("m_mgr")
                saved_mgr = saved_state.get("m_mgr")
                if saved_mgr is not None:
                    mgr = copy.deepcopy(saved_mgr)
                    state["m_mgr"] = mgr
                if mgr is not None:
                    device = p.device
                    if isinstance(mgr._alpha, torch.Tensor):
                        mgr._alpha = mgr._alpha.to(
                            device=device, dtype=torch.float32
                        )
                    if mgr._encoded is not None:
                        enc = mgr._encoded
                        enc["sign_packed"] = enc["sign_packed"].to(
                            device=device, dtype=torch.uint8
                        )
                        block_norms = enc["block_norms"].to(
                            device=device, dtype=torch.float32
                        )
                        if "labels" in enc:
                            labels = enc["labels"].to(
                                device=device, dtype=torch.uint8
                            )
                            positive_norms = block_norms.abs()
                            enc["block_norms"] = torch.where(
                                labels == 0,
                                torch.zeros_like(positive_norms),
                                torch.where(
                                    labels == 2, -positive_norms, positive_norms
                                ),
                            )
                            enc.pop("labels", None)
                        else:
                            enc["block_norms"] = block_norms
                        enc.pop("phase_packed", None)
                        enc.pop("scales", None)
                    if mgr._ef_residual is not None:
                        mgr._ef_residual = mgr._ef_residual.to(
                            device=device, dtype=torch.float32
                        )

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one AdamW step using compressed states where enabled."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_index, group in enumerate(self.param_groups):
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            use_compress_v = group["compress_v"]

            for param_index, p in enumerate(group["params"]):
                grad = p.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("TurboAdam does not support sparse gradients")
                if torch.is_complex(p) or torch.is_complex(grad):
                    raise RuntimeError("TurboAdam does not support complex parameters")

                state = self.state[p]
                if len(state) == 0:
                    state = self._initialize_state(p, group)

                state["step"] += 1
                step = state["step"]
                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step

                # --- m update ---
                if state["_compress_m"]:
                    m_new = state["m_mgr"].update(grad, beta1)
                else:
                    grad_fp32 = grad.float()
                    # Match torch.optim.AdamW's lerp formulation exactly.
                    state["exp_avg"].lerp_(grad_fp32, 1.0 - beta1)
                    m_new = state["exp_avg"]

                # --- v update + parameter update ---
                if use_compress_v:
                    cv = state["compressed_v"]
                    if (
                        _HAS_TRITON
                        and grad.is_cuda
                        and p.is_contiguous()
                        and grad.is_contiguous()
                        and m_new.is_contiguous()
                    ):
                        new_indices, new_scales = _triton_adam_update(
                            cv["indices"],
                            cv["scales"],
                            grad,
                            p,
                            m_new,
                            beta2,
                            cv["n_bits"],
                            cv["block_size"],
                            cv["original_length"],
                            lr,
                            bias_correction1,
                            bias_correction2,
                            eps,
                            weight_decay,
                            seed=self._seed_for(group_index, param_index, step),
                            packed=cv.get("packed", True),
                        )
                        cv["indices"] = new_indices
                        cv["scales"] = new_scales
                    else:
                        new_indices, new_scales = fused_adam_update(
                            cv["indices"],
                            cv["scales"],
                            grad,
                            p,
                            m_new,
                            beta2,
                            cv["n_bits"],
                            cv["block_size"],
                            cv["original_length"],
                            lr,
                            bias_correction1,
                            bias_correction2,
                            eps,
                            weight_decay,
                            packed=cv.get("packed", True),
                        )
                        cv["indices"] = new_indices
                        cv["scales"] = new_scales
                else:
                    grad_fp32 = grad.float()
                    v = state["exp_avg_sq"]
                    v.mul_(beta2).addcmul_(
                        grad_fp32, grad_fp32, value=1.0 - beta2
                    )
                    denom = v.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                    if weight_decay != 0.0:
                        p.mul_(1.0 - lr * weight_decay)
                    p.addcdiv_(
                        m_new,
                        denom,
                        value=-(lr / bias_correction1),
                    )

        return loss
