# TurboAdam

[![Tests](https://img.shields.io/badge/tests-101%20passing-brightgreen)](https://github.com/davidkny22/turboadam/tree/main/tests) [![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-orange)](https://pytorch.org/) [![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**A drop-in AdamW replacement with 9.845× smaller persistent optimizer state.**

One line change. No model modifications. No training-loop changes.

```python
from turboadam import TurboAdam

optimizer = TurboAdam(model.parameters(), lr=1e-3)
```

TurboAdam keeps the AdamW recurrences and parameter update. It computes each
step from fp32 transient moments, then persists the first moment with UState and
the second moment with 1Q.

---

## Why TurboAdam?

AdamW keeps two fp32 moment tensors for every trained parameter. That is 64
bits, or 8 bytes, of persistent optimizer state per parameter before allocator
overhead.

TurboAdam reduces the default state to approximately **6.50 bits per parameter
value** on large aligned tensors. The two moments remain active at every step;
only the representation persisted between steps changes.

| Parameters | AdamW moments | TurboAdam state | Difference |
| ---: | ---: | ---: | ---: |
| 125M | 1.00 GB | **0.102 GB** | **0.898 GB** |
| 7B | 56.0 GB | **5.69 GB** | **50.3 GB** |
| 70B | 560 GB | **56.9 GB** | **503 GB** |

These are decimal estimates for large aligned tensors. They exclude
parameters, gradients, allocator effects, block padding, and per-tensor
scalars.

Memory is only useful if the optimizer still trains well. In matched 500-step
GPT-2 runs, the default ends 1.224% behind AdamW on TinyStories and 0.094% ahead
on WikiText-103. The complete memory, convergence, and speed results are below.

---

## Quick start

### Install

Install the PyTorch build for your target accelerator, then install TurboAdam
with its platform Triton package:

```bash
pip install "turboadam[triton]"
```

An existing compatible CUDA PyTorch installation satisfies TurboAdam's
dependency and is left in place. When PyTorch is absent, pip resolves the build
available from the configured package index.

For CPU or MPS without Triton:

```bash
pip install turboadam
```

For an editable source installation:

```bash
pip install -e ".[triton]"
```

### Use

```python
from turboadam import TurboAdam

# Drop-in replacement for torch.optim.AdamW
optimizer = TurboAdam(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0.01,
)
```

---

## How it works

TurboAdam combines two separable state representations. UState compresses the
first moment to approximately 2.25 bits per value. 1Q compresses the second
moment to 4.25 bits per value by default. Either representation can be disabled
independently.

### UState: first-moment compression

UState stores the first moment in normalized Adam update units:

```text
q_t = (m_t / (1 - beta1^t))
      / (sqrt(v_persisted_t / (1 - beta2^t)) + eps)
```

Given the same persisted second moment, an unquantized `q_t` reconstructs
`m_t` exactly. The default UState layout is:

| State | Storage |
| --- | ---: |
| Four-level packed code | 2.00 bits/value |
| One bf16 mean per 64 values | 0.25 bits/value |
| Decode scale, encode scale, RMS accumulator | 12 bytes/tensor |

Each decoded mean block is recentered so its stored bf16 mean is preserved.
Codes use antithetic stochastic rounding and a one-step-lagged tensor scale.
The default scale factor is 1.1.

**Key insight:** AdamW applies the first moment through a normalized update.
UState persists that normalized quantity directly, so its compact codes spend
their resolution in the coordinate system that reaches the parameter update.

The finite coordinate bound requires `beta1**2 < beta2` when UState is active.
TurboAdam validates the condition at construction.

### 1Q: second-moment compression

The second moment is nonnegative and spans orders of magnitude. 1Q stores each
block on a logarithmic grid using packed indices and two fp16 log endpoints.

With the default 4-bit indices and 128-value blocks:

| State | Storage |
| --- | ---: |
| Packed log index | 4.00 bits/value |
| Two fp16 endpoints per 128 values | 0.25 bits/value |

Stochastic rounding is performed between decoded positive values. The PyTorch
and Triton paths use the same counter hash and require no persistent random
buffer.

**Key insight:** A logarithmic grid spends its levels on relative resolution.
That matches a positive moment whose coordinates may differ by many orders of
magnitude, while block-local endpoints adapt the grid to each region of the
tensor.

### The AdamW update remains AdamW

TurboAdam applies the decoupled AdamW update:

```text
m_t = beta1 * m_(t-1) + (1 - beta1) * g_t
v_t = beta2 * v_(t-1) + (1 - beta2) * g_t^2
theta_t = (1 - lr * weight_decay) * theta_(t-1)
          - lr * m_t / (1 - beta1^t)
          / (sqrt(v_t / (1 - beta2^t)) + eps)
```

Compression affects the state presented to the next optimizer step. It does
not replace the recurrence or the current parameter update with a different
optimizer rule.

### Fused CUDA path

For supported contiguous CUDA tensors, one Triton kernel owns each state block
and performs the complete update:

1. Decode UState and 1Q.
2. Reconstruct the prior first moment in the persisted second-moment frame.
3. Form the current fp32 Adam moments.
4. Apply the AdamW parameter update.
5. Recompress the second moment with 1Q.
6. Encode the next UState payload.

A one-program finalizer rotates the UState scale and clears its scalar RMS
accumulator. The wrapper allocates no parameter-sized optimizer workspace.

The fused path supports contiguous CUDA parameters, power-of-two block sizes
from 32 through 1024, UState mean blocks that divide the storage block, and 2,
3, 4, 6, or 8 second-moment bits. Other layouts use the PyTorch reference path.

---

## Results

### Persistent memory

For 131,072 aligned values, the default state occupies 106,508 bytes:

| Component | Bytes |
| --- | ---: |
| UState | 36,876 |
| 1Q | 69,632 |
| **Total** | **106,508** |

This is 6.500732 bits per value and 9.845× smaller than two fp32 moments.
The state contains no parameter-sized fp32 tensor when both representations are
active.

The included GPT-2-layer-shaped memory profile measures:

| Configuration | Persistent bytes | vs AdamW |
| --- | ---: | ---: |
| AdamW | 56,641,572 | 1.000× |
| TurboAdam, UState only | 30,320,712 | 0.535× |
| TurboAdam, 1Q only | 32,090,112 | 0.567× |
| **TurboAdam, UState + 1Q** | **5,769,288** | **0.102×** |

### Convergence

Matched 500-step GPT-2 124M runs use seed 42, sequence length 512, effective
batch size 16, no AMP, 100 linear warmup steps, and cosine decay to zero. The
TinyStories cache contains 12,000 chunks, so the run consumes 8,000 chunks
without repeating data.

| Dataset | AdamW final | TurboAdam final | Final gap | Trailing-50 gap |
| --- | ---: | ---: | ---: | ---: |
| TinyStories | 1.654943 | 1.675204 | +1.224% | +1.138% |
| WikiText-103 | 3.291123 | 3.288014 | -0.094% | -0.013% |

The TinyStories isolation identifies UState as the source of its late gap.
UState with exact fp32 second moments retains a +1.229% final gap. Exact fp32
first moments with 1Q finish within -0.060% of AdamW. Each result is a matched
single-seed trajectory, not a multi-seed estimate.

### Speed

On an RTX 4070 Laptop GPU, the included GPT-2-layer optimizer benchmark measures
6.26 ms per fused TurboAdam step and 2.23 ms per AdamW step.

The matched end-to-end language-model runs measure a 1.094× training-time ratio
on both TinyStories and WikiText-103 because model computation dominates the
optimizer step.

---

## API

```python
TurboAdam(
    params,                         # parameters or parameter groups
    lr=1e-3,                        # learning rate
    betas=(0.9, 0.999),             # AdamW EMA coefficients
    eps=1e-8,                       # numerical stability
    weight_decay=0.0,               # decoupled weight decay
    block_size=128,                 # UState and 1Q storage block size
    v_bits=4,                       # 1Q bits: 2, 3, 4, 6, or 8
    compress_m=True,                # enable UState
    compress_v=True,                # enable 1Q
    capturable=False,               # CUDA graph capture is unsupported
    min_m_compress_elements=4096,   # UState size threshold
    min_v_compress_elements=4096,   # 1Q size threshold
    m_block_size=64,                # UState mean block size
    m_step_factor=1.1,              # UState scale factor
    rounding_seed=0x12345678,       # counter-based rounding seed
)
```

The standard AdamW arguments retain their usual meaning. Parameters smaller
than a representation's threshold use exact fp32 state for that moment. Set a
threshold to zero to force compression for every nonempty parameter.

### Checkpoints

`state_dict()` contains tensors and ordinary Python values. Loading preserves
packed code dtypes even when parameters use fp16 or bf16. The rounding seed and
per-parameter step counters are part of the optimizer configuration and state.

Packed tensors restore exactly. CPU continuation is bit exact under the same
gradients. CUDA continuation is numerically equivalent within fp32 roundoff
from its parallel UState scale reduction.

---

## Reproduce the results

Build fixed GPT-2 token caches for both required datasets:

```bash
python experiments/prepare_language_data.py \
  --dataset tinystories \
  --output data/tinystories_gpt2_seq512.pt

python experiments/prepare_language_data.py \
  --dataset wikitext103 \
  --output data/wikitext103_gpt2_seq512.pt
```

Run AdamW and TurboAdam through the same trainer:

```bash
python experiments/train_language_model.py \
  --optimizer adamw \
  --dataset tinystories \
  --cache-path data/tinystories_gpt2_seq512.pt \
  --output runs/tinystories_adamw.jsonl

python experiments/train_language_model.py \
  --optimizer turboadam \
  --dataset tinystories \
  --cache-path data/tinystories_gpt2_seq512.pt \
  --output runs/tinystories_turboadam.jsonl
```

Repeat with `--dataset wikitext103` and its cache. The runner records the cache
SHA-256, model, seed, schedule, batch configuration, objective, state controls,
loss, trailing loss, gradient norm, and elapsed time. It passes
`labels=input_ids` to GPT-2 for the standard causal objective and logs the
unscaled mean cross-entropy.

Compare a matched pair with:

```bash
python scripts/compare_language_runs.py \
  --adamw runs/tinystories_adamw.jsonl \
  --turboadam runs/tinystories_turboadam.jsonl
```

---

## Verification

```bash
ruff format --check src tests experiments scripts benchmarks
ruff check src tests experiments scripts benchmarks
pytest -q
```

The suite covers the exact uncompressed AdamW identity, UState, 1Q, optimizer
state, checkpoint continuation, memory accounting, language-runner semantics,
and training smoke behavior.

The CUDA tests force both the PyTorch reference and Triton paths, compile every
supported 1Q width, verify parameter and decoded-state agreement, exercise
fp16, bf16, and fp32 parameters, check checkpoint restoration and continuation
within the documented device guarantees, and measure the persistent state.

---

## Limits

TurboAdam does not support sparse gradients, complex parameters, AMSGrad, or
CUDA graph capture. Noncontiguous parameters use the PyTorch reference path.

---

## Citation

```bibtex
@misc{kogan2026turboadam,
  title={TurboAdam: Memory-Efficient AdamW with Compressed Persistent State},
  author={Kogan, David},
  year={2026},
  howpublished={\url{https://github.com/davidkny22/turboadam}}
}
```

---

## License

MIT
