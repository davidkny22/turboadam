# TurboAdam roadmap

TurboAdam is evaluated on three coupled objectives:

1. Persistent optimizer-state memory.
2. End-to-end training speed.
3. Convergence relative to matched AdamW controls.

A candidate is adopted only when its tradeoff is measured on all three axes.

## Current product

- UState first-moment persistence at approximately 2.25 bits per value.
- 1Q second-moment persistence at 4.25 bits per value by default.
- A combined default state at approximately 6.50 bits per value.
- Exact fp32 state for parameters below configurable size thresholds.
- A fused Triton path for contiguous CUDA tensors.
- A PyTorch reference path for CPU, MPS, noncontiguous tensors, and kernel
  validation.
- Tensor-only checkpoints with deterministic rounding streams.
- Matched GPT-2 experiment paths for TinyStories and WikiText-103.

## Active optimization targets

### End-to-end speed

- Reduce transcendental work in packed second-moment decode and encode.
- Amortize launches across parameter tensors without adding persistent
  parameter-sized workspaces.
- Measure optimizer-only and full-model time separately.
- Preserve a direct reference-versus-kernel correctness gate for every fast
  path.

### Convergence

- Evaluate the UState scale factor on both TinyStories and WikiText-103.
- Measure final loss, trailing loss, and time-to-loss across matched seeds.
- Test memory-neutral changes to scale estimation, mean geometry, and
  stochastic rounding.
- Reject improvements that depend on changing the AdamW recurrence.

### Distributed training

- Define sharding rules for packed byte streams and per-block metadata.
- Validate state save, load, and resharding under FSDP.
- Validate optimizer-state partitioning under ZeRO.

### Execution environments

- Add a stable multi-tensor CUDA interface.
- Add graph-capture support when all state and launch addresses can remain
  stable.
- Measure `torch.compile` interaction without weakening eager correctness.

## Required gates

Every optimization must retain:

- Bit-exact agreement with PyTorch AdamW when compression is disabled.
- Finite behavior on partial blocks and extreme finite second moments.
- Exact checkpoint restoration and continuation within the documented device
  guarantees.
- No parameter-sized fp32 tensor in the default persistent state.
- Real CUDA execution for every claimed fused layout.
- Matched TinyStories and WikiText-103 controls with cache fingerprints.
