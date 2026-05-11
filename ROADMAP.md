# TurboAdam Roadmap

## Released

### v0.1.0 — Production Release (Current)
- Compress-every-step architecture (no freeze-refresh)
- 4-bit log-scale v compression with stochastic rounding
- CoState m compression (gradient-residual decomposition)
- Triton kernels for CUDA acceleration
- 6.5× optimizer-state memory reduction
- +0.25 loss gap (1.2% relative) on GPT-2 124M
- Drop-in AdamW replacement

## Upcoming

### v0.2.0 — Distributed Training Support
- **FSDP compatibility:** Flatten CoStateManager state into plain tensors for `torch.distributed.fsdp.FullyShardedDataParallel`
- **DeepSpeed ZeRO compatibility:** Ensure optimizer state shards correctly across ranks
- DDP validation and performance testing

### v0.3.0 — Scale Validation
- **7B model benchmark:** Validate convergence gap at 7B scale (Llama-2-7B or equivalent)
- **70B model benchmark:** Memory and convergence validation at 70B scale
- **AMP control experiment:** Matched-condition run with `torch.cuda.amp` enabled
- Gap-shrinking hypothesis validation (does gap decrease at scale?)

### v0.4.0 — Advanced Features
- **3-bit v compression:** Test viability of 3-bit (8 buckets) for additional memory savings
- **Multi-tensor Triton kernels:** Redesign without persistent buffer overhead
- **CUDA graph support:** Resolve Triton internal allocation incompatibility
- **Gradient compression composition:** Test GaLore + TurboAdam stacking

### v1.0.0 — Stable
- Convergence gap ≤1% on all tested scales (125M–70B)
- Full distributed training support (FSDP, DeepSpeed, DDP)
- Production-grade documentation and tutorials
- Benchmarked on multiple tasks (vision, RL, audio)

## Research Directions

1. **CoState cold-start elimination:** Initialize from first gradient instead of zero to reduce early-step gap
2. **Adaptive bit width:** Automatically select v_bits based on parameter shape and training dynamics
3. **Momentum low-rank revisited:** Test on fine-tuning workloads (different gradient statistics from pre-training)
4. **Per-parameter compression rates:** Different bit widths for different parameter groups

## How to Contribute

See [GitHub Issues](https://github.com/davidkny22/turboadam/issues) for tasks labeled `good first issue`.
