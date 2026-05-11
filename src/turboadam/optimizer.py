"""TurboAdam optimizer — drop-in replacement for Adam with compressed optimizer states."""

import torch
from torch.optim import Optimizer

from turboadam.costate import CoStateManager
from turboadam.oneq import compress_v_logscale, decompress_v
from turboadam.quantize import fused_v_update

# Try to use Triton-accelerated kernels; fall back to PyTorch if unavailable
try:
    from turboadam.triton_kernels import triton_fused_v_update as _triton_v_update
    _HAS_TRITON = True
except ImportError:
    _HAS_TRITON = False


class TurboAdam(Optimizer):
    """Adam optimizer with compressed first and second moment storage.

    Combines two compression techniques:
      - **CoState** (m compression): gradient-residual decomposition with
        three-tier costate encoding (null/phase/amplitude).  ~2 bits/param.
      - **Compress-every-step v**: v is stored as n-bit log-scale quantized
        representation.  Each step: decompress → EMA update → bias-correct
        for denominator → re-compress with stochastic rounding.

    Stochastic rounding is essential: it makes per-step quantization noise
    unbiased, preventing systematic drift in the EMA.

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate. Default: 1e-3.
        betas: EMA decay coefficients (β₁, β₂). Default: (0.9, 0.999).
        eps: Numerical stability term. Default: 1e-8.
        weight_decay: L2 penalty. Default: 0.
        block_size: Quantization block size (elements). Default: 128.
        v_bits: Bits per element for v compression (2, 3, 4, 6, or 8). Default: 4.
        compress_m: Enable CoState m compression. Default: True.
        compress_v: Enable v compression. Default: True.
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
        if v_bits not in (2, 3, 4, 6, 8):
            raise ValueError(f"v_bits must be one of {{2, 3, 4, 6, 8}}, got {v_bits}")
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
        super().__init__(params, defaults)
        self._group_step_tensors = {}

    # ------------------------------------------------------------------
    # Core step logic
    # ------------------------------------------------------------------

    def _full_step_kernel(self):
        """Complete optimizer step — m update + v update + weight update."""
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            use_compress_v = group["compress_v"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                grad = p.grad

                state["step"] += 1

                # Decoupled weight decay (AdamW style)
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # --- m update ---
                if state["_compress_m"]:
                    m_new = state["m_mgr"].update(grad, beta1)
                else:
                    state["exp_avg"].mul_(beta1).add_(grad, alpha=1.0 - beta1)
                    m_new = state["exp_avg"]

                # Bias corrections are tensors so CUDA graph replay sees new values.
                bias_correction1 = state["_bc1"]
                bias_correction2 = state["_bc2"]

                # --- v update ---
                if use_compress_v:
                    cv = state["compressed_v"]
                    _v_update_fn = _triton_v_update if (_HAS_TRITON and grad.is_cuda) else fused_v_update
                    if _HAS_TRITON and grad.is_cuda:
                        # Refill random buffer for stochastic rounding (graph-safe:
                        # same tensor address, only contents change)
                        cv["rand_buf"].uniform_()
                        # Allocate separate output buffers to avoid aliasing
                        # (Triton assumes no aliasing between pointer args)
                        _new_indices = torch.empty_like(cv["indices"])
                        _new_scales = torch.empty_like(cv["scales"])
                        new_indices, new_scales, v_flat = _v_update_fn(
                            cv["indices"], cv["scales"], grad, beta2,
                            cv["n_bits"], cv["block_size"], cv["original_length"],
                            rand_buf=cv["rand_buf"],
                            out_indices=_new_indices,
                            out_scales=_new_scales,
                        )
                        cv["indices"] = new_indices
                        cv["scales"] = new_scales
                    else:
                        new_indices, new_scales, v_flat = _v_update_fn(
                            cv["indices"], cv["scales"], grad, beta2,
                            cv["n_bits"], cv["block_size"], cv["original_length"],
                        )
                        cv["indices"] = new_indices
                        cv["scales"] = new_scales
                    v = v_flat.reshape(p.shape)
                    denom = (v / bias_correction2).sqrt().add_(eps)
                else:
                    v = state["exp_avg_sq"]
                    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                    denom = (v / bias_correction2).sqrt().add_(eps)

                # Weight update
                p.addcdiv_(m_new, denom * bias_correction1, value=-lr)

    def _prepare_step_scalars(self):
        """Update per-parameter scalar tensors for the next optimizer step.

        Each parameter computes its own bias correction based on its own step
        counter, so add_param_group() and checkpoint resume work correctly.
        """
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p not in self.state:
                    continue
                state = self.state[p]
                next_step = state["step"] + 1  # step increments inside _full_step_kernel
                state["_bc1"].fill_(1.0 - beta1 ** next_step)
                state["_bc2"].fill_(1.0 - beta2 ** next_step)

    # ------------------------------------------------------------------
    # State dict handling
    # ------------------------------------------------------------------

    def load_state_dict(self, state_dict):
        """Load optimizer state, migrating CoStateManager tensors to param device.

        PyTorch's default load_state_dict casts tensors in dicts but does not
        recurse into custom objects. CoStateManager stores tensors internally
        that must move to the correct device after loading.
        """
        super().load_state_dict(state_dict)
        # Migrate CoStateManager internal tensors to their parameter's device
        for group in self.param_groups:
            for p in group["params"]:
                if p not in self.state:
                    continue
                state = self.state[p]
                if "m_mgr" in state:
                    mgr = state["m_mgr"]
                    device = p.device
                    if isinstance(mgr._alpha, torch.Tensor):
                        mgr._alpha = mgr._alpha.to(device=device)
                    if mgr._encoded is not None:
                        for key in ("labels", "sign_packed", "block_norms", "scales"):
                            if key in mgr._encoded:
                                mgr._encoded[key] = mgr._encoded[key].to(device=device)
                    if mgr._ef_residual is not None:
                        mgr._ef_residual = mgr._ef_residual.to(device=device)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group_idx, group in enumerate(self.param_groups):
            block_size = group["block_size"]
            v_bits = group["v_bits"]
            use_compress_m = group["compress_m"]
            use_compress_v = group["compress_v"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # Lazy state initialisation
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    compress_this_m = use_compress_m and p.numel() >= group["min_m_compress_elements"]
                    state["_compress_m"] = compress_this_m
                    if compress_this_m:
                        state["m_mgr"] = CoStateManager(
                            block_size=block_size,
                            null_pct=group["null_pct"],
                            amp_pct=group["amp_pct"],
                            error_feedback=group["error_feedback"],
                        )
                    else:
                        state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
                    if not use_compress_v:
                        state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    device_key = (group_idx, p.device.type, p.device.index)
                    if device_key not in self._group_step_tensors:
                        self._group_step_tensors[device_key] = {
                            "bc1": torch.empty(1, dtype=torch.float32, device=p.device),
                            "bc2": torch.empty(1, dtype=torch.float32, device=p.device),
                        }
                    group_tensors = self._group_step_tensors[device_key]
                    state["_bc1"] = group_tensors["bc1"]
                    state["_bc2"] = group_tensors["bc2"]

                # First step: init compressed_v with near-zero so _step_kernel
                # can do the real first EMA update (avoids double-counting g²)
                if use_compress_v and "compressed_v" not in state:
                    v = torch.full_like(p, 1e-30, dtype=torch.float32)
                    state["compressed_v"] = compress_v_logscale(
                        v, n_bits=v_bits, block_size=block_size, stochastic_round=False,
                    )
                    if _HAS_TRITON and p.is_cuda:
                        num_blocks = state["compressed_v"]["scales"].shape[0]
                        padded_numel = num_blocks * block_size
                        state["compressed_v"]["rand_buf"] = torch.empty(
                            padded_numel, dtype=torch.float32, device=p.device
                        )

        # Prepare bias corrections for this step
        self._prepare_step_scalars()
        self._full_step_kernel()

        return loss
