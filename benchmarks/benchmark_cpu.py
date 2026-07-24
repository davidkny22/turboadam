"""Small reproducible CPU benchmark for the optimized TurboAdam package.

This is diagnostic, not a pass/fail test. Run from the package root with:
    PYTHONPATH=. python benchmarks/benchmark_cpu.py
"""

from __future__ import annotations

import statistics
import time

import torch

from turboadam import TurboAdam


def measure(numel: int, compress_m: bool, compress_v: bool, repeats: int = 100):
    p = torch.nn.Parameter(torch.randn(numel))
    opt = TurboAdam(
        [p], compress_m=compress_m, compress_v=compress_v,
        min_m_compress_elements=0
    )
    times = []
    for step in range(repeats + 10):
        p.grad = torch.randn_like(p)
        start = time.perf_counter()
        opt.step()
        elapsed = (time.perf_counter() - start) * 1e3
        if step >= 10:
            times.append(elapsed)
    return statistics.median(times)


def main():
    torch.set_num_threads(1)
    print("numel, m, v, median_ms")
    for numel in (4096, 65536, 262144):
        for compress_m, compress_v in ((False, False), (False, True), (True, True)):
            result = measure(numel, compress_m, compress_v)
            print(numel, int(compress_m), int(compress_v), f"{result:.4f}", sep=", ")


if __name__ == "__main__":
    main()
