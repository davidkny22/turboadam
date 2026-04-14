"""Memory profiling instrumentation for TurboAdam.

Tracks per-component optimizer state memory at each step.
Produces JSONL logs and memory profile figures.

Usage:
    python scripts/profile_memory.py --steps 500 --warmup_threshold 0.01
    python scripts/profile_memory.py --steps 500 --warmup_threshold 100.0  # fast Phase B
"""

import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn


def parse_args():
    p = argparse.ArgumentParser(description="Profile TurboAdam memory usage")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--warmup_threshold", type=float, default=0.01)
    p.add_argument("--refresh_interval", type=int, default=1000)
    p.add_argument("--refresh_mode", type=str, default="compressed")
    p.add_argument("--output_dir", type=str, default="experiments/results")
    p.add_argument("--model_dim", type=int, default=256,
                   help="Hidden dimension for the test MLP (controls param count)")
    return p.parse_args()


def tensor_bytes(t):
    """Return the memory footprint of a tensor in bytes."""
    return t.nelement() * t.element_size()


def measure_state_bytes(optimizer):
    """Measure total optimizer state memory in bytes, broken down by component."""
    total = 0
    v_bytes = 0
    m_bytes = 0
    metadata_bytes = 0

    for state in optimizer.state.values():
        for key, val in state.items():
            if isinstance(val, torch.Tensor):
                b = tensor_bytes(val)
                total += b
                if key in ("exp_avg_sq", "v_prev"):
                    v_bytes += b
                elif key in ("g_sq_accum",):
                    v_bytes += b  # accumulator is v-related
                else:
                    metadata_bytes += b
            elif isinstance(val, dict):
                # Compressed dicts (compressed_v, encoded CoState state)
                for subkey, subval in val.items():
                    if isinstance(subval, torch.Tensor):
                        b = tensor_bytes(subval)
                        total += b
                        if key == "compressed_v":
                            v_bytes += b
                        elif key in ("g_sq_accum_packed", "g_sq_accum_scales"):
                            v_bytes += b
                        else:
                            m_bytes += b
            # Also count top-level packed accum tensors
            if key in ("g_sq_accum_packed", "g_sq_accum_scales"):
                if isinstance(val, torch.Tensor):
                    b = tensor_bytes(val)
                    v_bytes += b

    return {"total": total, "v": v_bytes, "m": m_bytes, "metadata": metadata_bytes}


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    from turboadam import TurboAdam

    # Build a small MLP to profile
    d = args.model_dim
    model = nn.Sequential(
        nn.Linear(d, d * 4),
        nn.ReLU(),
        nn.Linear(d * 4, d * 4),
        nn.ReLU(),
        nn.Linear(d * 4, d),
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters ({n_params * 4 / 1024:.1f} KB at fp32)")

    optimizer = TurboAdam(
        model.parameters(), lr=1e-3,
        warmup_threshold=args.warmup_threshold,
        refresh_interval=args.refresh_interval,
        refresh_mode=args.refresh_mode,
    )

    # Baseline: standard Adam state size
    baseline_bytes = n_params * 8  # m (fp32) + v (fp32) = 8 bytes/param

    log_path = os.path.join(args.output_dir, "memory_profile.jsonl")
    entries = []

    torch.manual_seed(42)
    x = torch.randn(16, d)
    target = torch.randn(16, d)

    for step in range(1, args.steps + 1):
        optimizer.zero_grad()
        out = model(x)
        loss = nn.functional.mse_loss(out, target)
        loss.backward()
        optimizer.step()

        mem = measure_state_bytes(optimizer)
        bits_per_param = (mem["total"] * 8) / n_params if n_params > 0 else 0
        entry = {
            "step": step,
            "total_bytes": mem["total"],
            "v_bytes": mem["v"],
            "m_bytes": mem["m"],
            "metadata_bytes": mem["metadata"],
            "baseline_bytes": baseline_bytes,
            "bits_per_param": round(bits_per_param, 2),
            "compression_ratio": round(baseline_bytes / max(mem["total"], 1), 1),
            "loss": loss.item(),
        }
        entries.append(entry)

        if step % 50 == 0 or step == 1:
            print(f"step={step} state={mem['total']:,}B "
                  f"({bits_per_param:.1f} bits/param, "
                  f"{entry['compression_ratio']}x compression) "
                  f"loss={loss.item():.4f}")

    with open(log_path, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    print(f"\nProfile saved to {log_path}")

    # Generate plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        steps = [e["step"] for e in entries]
        total = [e["total_bytes"] for e in entries]
        baseline = [e["baseline_bytes"] for e in entries]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        ax1.plot(steps, baseline, "r--", label="Baseline Adam", linewidth=2)
        ax1.plot(steps, total, "b-", label="TurboAdam", linewidth=2)
        ax1.set_ylabel("Optimizer State (bytes)")
        ax1.set_title("Optimizer State Memory Profile")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        bpp = [e["bits_per_param"] for e in entries]
        ax2.plot(steps, bpp, "g-", linewidth=2)
        ax2.axhline(y=64, color="r", linestyle="--", label="Baseline (64 bits/param)")
        ax2.set_xlabel("Training Step")
        ax2.set_ylabel("Bits per Parameter")
        ax2.set_title("Compression Ratio Over Training")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = os.path.join(args.output_dir, "memory_profile.png")
        plt.savefig(fig_path, dpi=150)
        print(f"Figure saved to {fig_path}")
    except ImportError:
        print("matplotlib not available — skipping plot generation")


if __name__ == "__main__":
    main()
