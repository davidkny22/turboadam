"""Memory profiling script for TurboAdam.

Measures actual GPU memory allocated for optimizer states under
various configurations using a synthetic GPT-2-layer param set.

Usage:
    python scripts/profile_memory.py
"""

import argparse
import gc
import json
from pathlib import Path

import torch
import torch.nn as nn

from turboadam import TurboAdam
from turboadam.utils import state_tensor_bytes


def _make_params(device: str = "cuda"):
    """Return a list of Parameters mimicking one GPT-2 layer."""
    shapes = [
        (768, 768),
        (768, 768),
        (768, 768),
        (768, 768),
        (768, 3072),
        (3072, 768),
        (768,),
        (768,),
        (768,),
    ]
    params = []
    for s in shapes:
        p = nn.Parameter(torch.randn(s, device=device))
        p.grad = torch.randn_like(p)
        params.append(p)
    return params


def _measure_memory(opt_class, opt_kwargs, params):
    """Return bytes allocated for optimizer states after 1 step."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()

    opt = opt_class(params, lr=1e-3, **opt_kwargs)
    opt.step()

    peak = torch.cuda.max_memory_allocated()
    return peak - base, state_tensor_bytes(opt.state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        print("CUDA not available: memory profiling requires GPU.")
        return

    configs = [
        ("AdamW (baseline)", torch.optim.AdamW, {}),
        ("TurboAdam (m+v)", TurboAdam, {}),
        ("TurboAdam (v only)", TurboAdam, {"compress_m": False}),
        ("TurboAdam (m only)", TurboAdam, {"compress_v": False}),
        (
            "TurboAdam (no compression)",
            TurboAdam,
            {"compress_m": False, "compress_v": False},
        ),
    ]

    print("=" * 60)
    print("TurboAdam Memory Profile")
    print("Param set: one GPT-2 layer (9 tensors)")
    print("=" * 60)

    results = []
    baseline_persistent = None
    for name, cls, kwargs in configs:
        params = _make_params()
        peak_delta, persistent = _measure_memory(cls, kwargs, params)

        if baseline_persistent is None:
            baseline_persistent = persistent
            ratio = 1.0
        else:
            ratio = persistent / baseline_persistent

        print(f"{name:30s} persistent={persistent:8,} B  ({ratio:.2f}x vs baseline)")
        results.append(
            {
                "config": name,
                "peak_delta_bytes": peak_delta,
                "persistent_bytes": persistent,
                "ratio_vs_baseline": ratio,
            }
        )

    print("=" * 60)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
