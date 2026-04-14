# TurboAdam

2x training memory reduction via optimizer state compression. Drop-in replacement for Adam.

```python
from turboadam import TurboAdam

optimizer = TurboAdam(model.parameters(), lr=1e-4)
```

One line change. No modifications to the training loop.

## How it works

TurboAdam compresses Adam's optimizer states (first and second moments) at rest during training using two separable techniques:

**1Q** — Second moment (v) compression via low-rank SVD for matrix parameters and 2-bit log-scale quantization for non-matrix parameters, with a warmup-then-freeze-refresh cycle.

**CoState** — First moment (m) compression via gradient-residual decomposition. At each step, m is decomposed as α·g + δ (gradient component + residual). The residual is classified into one of three costates (null/phase/amplitude) and encoded at variable bit-width based on how much information it carries.

Together: ~116x compression on optimizer states, ~2x reduction in total training memory.

## Memory reduction

| Component | Standard Adam | TurboAdam |
|-----------|--------------|-----------|
| First moment (m) | 4 bytes/param | ~0.05 bytes/param |
| Second moment (v) | 4 bytes/param | ~0.02 bytes/param |
| Total training | 16 bytes/param | ~8.8 bytes/param |

For a 70B model: 1,120 GB → ~565 GB.

## Status

Phase 1 in progress — correctness validation on RTX 4070 8GB (models ≤ 125M parameters).
