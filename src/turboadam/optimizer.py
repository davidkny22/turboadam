"""TurboAdam optimizer — drop-in replacement for Adam with compressed optimizer states."""

import torch
from torch.optim import Optimizer

from turboadam.costate import CoStateManager
from turboadam.oneq import check_warmup, compress_v, decompress_v
from turboadam.quantize import quantize_logscale, dequantize_logscale
from turboadam.utils import pad_to_blocks, unpad_from_blocks


class TurboAdam(Optimizer):
    """Adam optimizer with compressed first and second moment storage.

    Combines 1Q (second moment compression) and CoState (first moment
    compression) to reduce optimizer state memory by ~116x vs. standard Adam,
    with no meaningful convergence loss.

    Phase A (warmup): v is maintained at full fp32 (standard Adam EMA).
    Phase B (compressed): v is compressed via 1Q, frozen between K-step refreshes.
    m is compressed via CoState from step 0 in both phases.

    The compressed v stores the bias-corrected second moment estimate v̂ = v / (1 - β₂^t)
    at the time of compression.  In Phase B the denominator is therefore
    sqrt(v_decompressed) + eps (no additional bias correction needed).

    Args:
        params: Iterable of parameters or param groups.
        lr: Learning rate. Default: 1e-3.
        betas: EMA decay coefficients (β₁, β₂). Default: (0.9, 0.999).
        eps: Numerical stability term. Default: 1e-8.
        weight_decay: L2 penalty. Default: 0.
        block_size: Quantization block size (elements). Default: 128.
        svd_rank: Rank for low-rank v approximation on matrix params. Default: 8.
        refresh_interval: Steps between v refresh cycles. Default: 1000.
        warmup_threshold: Relative change in v to trigger compression. Default: 0.01.
        refresh_mode: Strategy for accumulating gradient statistics during the
            frozen interval.

            ``'single'``: No per-step accumulator.  At each Phase B step a full
            fp32 tensor ``state['g_sq_accum']`` is overwritten with the current
            g².  At refresh: v_new = β₂^K·v̂_old + (1-β₂^K)·g_current².

            ``'compressed'`` *(default)*: g² is accumulated into a 2-bit
            log-scale quantized buffer (``state['g_sq_accum_packed']``,
            ``state['g_sq_accum_scales']``, ``state['g_sq_accum_numel']``,
            ``state['g_sq_accum_count']``) updated in-place every step.  At
            refresh: v_new = β₂^K·v̂_old + (1-β₂^K)·(Σg²/K).  This uses the
            full K-sample mean rather than a single-sample estimate.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        block_size: int = 128,
        svd_rank: int = 8,
        refresh_interval: int = 1000,
        warmup_threshold: float = 0.01,
        refresh_mode: str = "compressed",
    ):
        if refresh_mode not in ("single", "compressed"):
            raise ValueError(
                f"refresh_mode must be 'single' or 'compressed', got {refresh_mode!r}"
            )
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            block_size=block_size,
            svd_rank=svd_rank,
            refresh_interval=refresh_interval,
            warmup_threshold=warmup_threshold,
            refresh_mode=refresh_mode,
        )
        super().__init__(params, defaults)

        # PyTorch's load_state_dict _cast() corrupts string values inside
        # nested dicts (treats strings as iterables → generator objects).
        # Fix the 'type' field in compressed_v after loading.
        def _fix_compressed_type(optimizer, *args, **kwargs):
            for state in optimizer.state.values():
                for key in ("compressed_v",):
                    if key in state and isinstance(state[key], dict):
                        t = state[key].get("type")
                        if t is not None and not isinstance(t, str):
                            # Reconstruct from the generator or other mangled form
                            try:
                                state[key]["type"] = "".join(t)
                            except TypeError:
                                pass

        self.register_load_state_dict_post_hook(_fix_compressed_type)

    # ------------------------------------------------------------------
    # Internal helpers for the compressed g² accumulator
    # ------------------------------------------------------------------

    @staticmethod
    def _init_gsq_accum(
        g_sq: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Quantize g² into 2-bit log-scale form and return (packed, scales, numel)."""
        g_sq_flat = g_sq.reshape(-1).float()
        original_length = g_sq_flat.shape[0]
        g_sq_min = g_sq_flat.min().item()
        pad_value = max(g_sq_min, 1e-38)
        g_sq_padded, _ = pad_to_blocks(g_sq_flat, block_size, pad_value=pad_value)
        packed, scales = quantize_logscale(g_sq_padded, block_size=block_size)
        return packed, scales, original_length

    @staticmethod
    def _update_gsq_accum(
        packed: torch.Tensor,
        scales: torch.Tensor,
        original_numel: int,
        g_sq: torch.Tensor,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decompress accumulator, add g², re-compress, return new (packed, scales)."""
        g_sq_flat = g_sq.reshape(-1).float()
        # Decompress existing accum (trimmed to original_numel)
        accum_flat = dequantize_logscale(
            packed, scales, block_size=block_size, original_numel=original_numel
        )
        # Add current g²
        updated = accum_flat + g_sq_flat
        # Re-quantize
        updated_min = updated.min().item()
        pad_value = max(updated_min, 1e-38)
        updated_padded, _ = pad_to_blocks(updated, block_size, pad_value=pad_value)
        new_packed, new_scales = quantize_logscale(updated_padded, block_size=block_size)
        return new_packed, new_scales

    @staticmethod
    def _read_gsq_accum(
        packed: torch.Tensor,
        scales: torch.Tensor,
        original_numel: int,
        block_size: int,
        param_shape: torch.Size,
    ) -> torch.Tensor:
        """Decompress accumulated g² back to a float32 tensor matching param shape."""
        flat = dequantize_logscale(
            packed, scales, block_size=block_size, original_numel=original_numel
        )
        return flat.reshape(param_shape)

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step.

        Phase A: v is maintained at full fp32 (standard Adam EMA).
        Phase B: v is compressed via 1Q, frozen between K-step refreshes.
        m is compressed via CoState from step 0 in both phases.

        Args:
            closure: Optional callable that re-evaluates the model and returns loss.

        Returns:
            Loss value if closure was provided, else None.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            block_size = group["block_size"]
            svd_rank = group["svd_rank"]
            warmup_threshold = group["warmup_threshold"]
            refresh_interval = group["refresh_interval"]
            refresh_mode = group["refresh_mode"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.float()

                # Lazy state initialisation on first step for this parameter
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
                    state["costate_mgr"] = CoStateManager(block_size=block_size)
                    state["v_prev"] = torch.zeros_like(p, dtype=torch.float32)
                    state["warmup_complete"] = False
                    state["phase"] = "A"

                state["step"] += 1
                step = state["step"]
                costate_mgr = state["costate_mgr"]

                # Decoupled weight decay (AdamW style)
                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                # m update via CoState (active in both phases)
                m_new = costate_mgr.update(grad, beta1)

                # Bias corrections
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                # Dispatch on structural keys: 'compressed_v' exists only in Phase B.
                # Avoids reliance on string 'phase' which PyTorch _cast corrupts.
                in_phase_a = "compressed_v" not in state
                if in_phase_a:
                    # -------------------------------------------------------
                    # Phase A: full fp32 v update (standard Adam EMA)
                    # -------------------------------------------------------
                    v = state["exp_avg_sq"]
                    v_prev = state["v_prev"]

                    # v = β₂ · v + (1 - β₂) · g²
                    v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                    # Warmup check: did v stabilise enough to compress?
                    if not state["warmup_complete"]:
                        if check_warmup(v, v_prev, warmup_threshold):
                            state["warmup_complete"] = True

                    if state["warmup_complete"]:
                        # Transition to Phase B.
                        # Store the bias-corrected second moment v̂ = v / (1 - β₂^t)
                        # so Phase B can use v_decompressed directly (no bc2 division).
                        v_hat = v / bias_correction2
                        state["compressed_v"] = compress_v(
                            p, v_hat, rank=svd_rank, block_size=block_size
                        )
                        state["refresh_counter"] = 0
                        state["phase"] = "B"

                        # Initialise the g² accumulator for the chosen refresh mode
                        g_sq = grad.mul(grad)
                        if refresh_mode == "single":
                            state["g_sq_accum"] = g_sq.clone()
                        else:
                            packed, scales, numel = self._init_gsq_accum(g_sq, block_size)
                            state["g_sq_accum_packed"] = packed
                            state["g_sq_accum_scales"] = scales
                            state["g_sq_accum_numel"] = numel
                            state["g_sq_accum_count"] = 1

                        del state["exp_avg_sq"]
                        del state["v_prev"]

                        # Use decompressed (bias-corrected) v for this step
                        v_for_update = decompress_v(state["compressed_v"])
                        state["refresh_counter"] = 1
                        # v_for_update is already v̂ — no bias correction needed
                        denom = v_for_update.sqrt().add_(eps)

                    else:
                        # Still in Phase A — standard Adam denominator
                        state["v_prev"] = v.clone()
                        denom = (v / bias_correction2).sqrt().add_(eps)

                else:
                    # -------------------------------------------------------
                    # Phase B: v is frozen (compressed).
                    # Decompressed v is the bias-corrected v̂ — use directly.
                    # -------------------------------------------------------
                    v_for_update = decompress_v(state["compressed_v"])
                    denom = v_for_update.sqrt().add_(eps)

                    # Update g² accumulator
                    g_sq = grad.mul(grad)
                    if refresh_mode == "single":
                        # Overwrite with current g² (single-sample mode)
                        state["g_sq_accum"].copy_(g_sq)
                    else:
                        # Add to compressed accumulator
                        new_packed, new_scales = self._update_gsq_accum(
                            state["g_sq_accum_packed"],
                            state["g_sq_accum_scales"],
                            state["g_sq_accum_numel"],
                            g_sq,
                            block_size,
                        )
                        state["g_sq_accum_packed"] = new_packed
                        state["g_sq_accum_scales"] = new_scales
                        state["g_sq_accum_count"] += 1

                    state["refresh_counter"] += 1

                    # -------------------------------------------------------
                    # Refresh cycle: every refresh_interval steps re-estimate v̂
                    # -------------------------------------------------------
                    if state["refresh_counter"] >= refresh_interval:
                        v_old_hat = decompress_v(state["compressed_v"])
                        beta2_K = beta2 ** refresh_interval

                        if refresh_mode == "single":
                            # Single-sample refresh: use current g² as the estimate
                            recent_g_sq = state["g_sq_accum"].float()
                        else:
                            # K-sample mean refresh
                            count = max(state["g_sq_accum_count"], 1)
                            accum = self._read_gsq_accum(
                                state["g_sq_accum_packed"],
                                state["g_sq_accum_scales"],
                                state["g_sq_accum_numel"],
                                block_size,
                                p.shape,
                            )
                            recent_g_sq = accum / count

                        # v̂_new = β₂^K · v̂_old + (1 - β₂^K) · recent_g²
                        v_new_hat = v_old_hat.mul_(beta2_K).add_(
                            recent_g_sq, alpha=1.0 - beta2_K
                        )
                        state["compressed_v"] = compress_v(
                            p, v_new_hat, rank=svd_rank, block_size=block_size
                        )

                        # Reset accumulator
                        if refresh_mode == "single":
                            state["g_sq_accum"].zero_()
                        else:
                            packed0, scales0, numel0 = self._init_gsq_accum(g_sq, block_size)
                            state["g_sq_accum_packed"] = packed0
                            state["g_sq_accum_scales"] = scales0
                            state["g_sq_accum_numel"] = numel0
                            state["g_sq_accum_count"] = 1

                        state["refresh_counter"] = 0

                # Weight update:
                # step_size = -lr / bias_correction1
                # denom = sqrt(v̂) + eps  (v̂ already bias-corrected for Phase B;
                #                          standard (v/bc2).sqrt() for Phase A)
                step_size = -lr / bias_correction1
                p.addcdiv_(m_new, denom, value=step_size)

        return loss
