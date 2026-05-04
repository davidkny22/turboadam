"""Speed benchmark: AdamW vs TurboAdam configurations.

Mimics one GPT-2 layer param set (from optimization history):
    4x(768,768) + (768,3072) + (3072,768) + 3x(768,)

Reports wall-clock ms/step after warmup.
Usage:
    python scripts/benchmark_speed.py
"""

import gc
import time

import torch
import torch.nn as nn

from turboadam import TurboAdam


# ---------------------------------------------------------------------------
# Benchmark param set
# ---------------------------------------------------------------------------

def _make_params(device: str = "cuda"):
    """Return a list of Parameters mimicking one GPT-2 layer."""
    shapes = [
        (768, 768), (768, 768), (768, 768), (768, 768),
        (768, 3072), (3072, 768),
        (768,), (768,), (768,),
    ]
    params = []
    for s in shapes:
        p = nn.Parameter(torch.randn(s, device=device))
        p.grad = torch.randn_like(p)
        params.append(p)
    return params


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def _benchmark(opt_class, opt_kwargs, params, n_warmup: int = 50, n_measure: int = 200):
    """Return median ms/step over n_measure steps after warmup."""
    opt = opt_class(params, lr=1e-3, **opt_kwargs)

    # Warmup
    for _ in range(n_warmup):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_measure):
        for p in params:
            p.grad = torch.randn_like(p)
        opt.step()
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) / n_measure * 1000
    return elapsed_ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        print("CUDA not available — benchmark requires GPU.")
        return

    configs = [
        ("AdamW (baseline)", torch.optim.AdamW, {}),
        ("TurboAdam (m+v)", TurboAdam, {}),
        ("TurboAdam (v only)", TurboAdam, {"compress_m": False}),
        ("TurboAdam (m only)", TurboAdam, {"compress_v": False}),
        ("TurboAdam (no compression)", TurboAdam, {"compress_m": False, "compress_v": False}),
    ]

    print("=" * 60)
    print("TurboAdam Speed Benchmark")
    print("Param set: one GPT-2 layer (9 tensors)")
    print("Warmup: 50 steps | Measure: 200 steps")
    print("=" * 60)

    baseline_ms = None
    for name, cls, kwargs in configs:
        gc.collect()
        torch.cuda.empty_cache()

        # Fresh params for each config to avoid state contamination
        params = _make_params()
        ms = _benchmark(cls, kwargs, params)

        if baseline_ms is None:
            baseline_ms = ms
            overhead = 1.0
        else:
            overhead = ms / baseline_ms

        print(f"{name:30s} {ms:7.2f} ms/step  ({overhead:.2f}x vs baseline)")

    print("=" * 60)


if __name__ == "__main__":
    main()
