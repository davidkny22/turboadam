"""Visualization scripts for experiment results.

Plots:
- Loss curve overlays (TurboAdam variants vs. baseline)
- Memory profiles over training steps
- Costate distribution over time (null/phase/amplitude fractions)
- Ablation sweep heatmaps (block size, SVD rank, refresh interval, warmup length)
- Reconstruction fidelity by costate tier
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Plot TurboAdam vs baseline loss curves")
    p.add_argument(
        "--baseline_log",
        type=str,
        default="experiments/results/baseline_log.jsonl",
        help="Path to baseline JSONL log",
    )
    p.add_argument(
        "--turboadam_log",
        type=str,
        default="experiments/results/turboadam_log.jsonl",
        help="Path to TurboAdam JSONL log",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="experiments/results",
        help="Directory to save plots",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Log loading
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> list[dict]:
    """Load a JSONL file and return a list of dicts."""
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_loss_overlay(
    baseline_entries: list[dict],
    turboadam_entries: list[dict],
    output_path: str,
) -> None:
    """Plot overlaid loss curves (step vs loss) for both runs on the same axes.

    Args:
        baseline_entries:  Log entries from baseline run.
        turboadam_entries: Log entries from TurboAdam run.
        output_path:       Full path to save the PNG figure.
    """
    baseline_steps = [e["step"] for e in baseline_entries]
    baseline_loss = [e["loss"] for e in baseline_entries]

    turboadam_steps = [e["step"] for e in turboadam_entries]
    turboadam_loss = [e["loss"] for e in turboadam_entries]

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(baseline_steps, baseline_loss, label="AdamW (baseline)", color="steelblue", linewidth=1.5)
    ax.plot(turboadam_steps, turboadam_loss, label="TurboAdam", color="darkorange", linewidth=1.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (cross-entropy)")
    ax.set_title("TurboAdam vs AdamW — Loss Curves (GPT-2 124M / WikiText-103)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved loss overlay to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, "loss_overlay.png")

    # Load logs — both must exist
    if not os.path.isfile(args.baseline_log):
        raise FileNotFoundError(f"Baseline log not found: {args.baseline_log}")
    if not os.path.isfile(args.turboadam_log):
        raise FileNotFoundError(f"TurboAdam log not found: {args.turboadam_log}")

    baseline_entries = load_jsonl(args.baseline_log)
    turboadam_entries = load_jsonl(args.turboadam_log)

    print(f"Baseline entries:  {len(baseline_entries)}")
    print(f"TurboAdam entries: {len(turboadam_entries)}")

    plot_loss_overlay(baseline_entries, turboadam_entries, output_path)


if __name__ == "__main__":
    main()
