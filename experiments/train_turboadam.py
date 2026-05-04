"""TurboAdam training script — mirrors baseline.py with optimizer swapped.

Overlay loss curves against baseline to verify convergence equivalence.
Output: JSONL log at experiments/results/turboadam_log.jsonl
"""

import argparse
import json
import math
import os
import time

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer

from turboadam import TurboAdam


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="TurboAdam training on GPT-2 124M")
    p.add_argument("--steps",              type=int,   default=2000,          help="Total optimizer steps")
    p.add_argument("--batch_size",         type=int,   default=4,             help="Micro-batch size (sequences per GPU step)")
    p.add_argument("--accum_steps",        type=int,   default=4,             help="Gradient accumulation steps (effective batch = batch_size * accum_steps)")
    p.add_argument("--lr",                 type=float, default=6e-4,          help="Peak learning rate")
    p.add_argument("--warmup_steps",       type=int,   default=100,           help="Linear warmup steps")
    p.add_argument("--seed",               type=int,   default=42,            help="Random seed")
    p.add_argument("--seq_len",            type=int,   default=512,           help="Sequence length (tokens)")
    p.add_argument("--device",             type=str,   default=None,          help="Device: cuda / mps / cpu (auto-detected if omitted)")
    p.add_argument("--output_dir",         type=str,   default="experiments/results", help="Directory for log output")
    p.add_argument("--log_every",          type=int,   default=50,            help="Log interval (steps)")
    p.add_argument("--dry_run",            action="store_true",               help="Run 5 steps only (smoke test)")
    p.add_argument("--cache_path",         type=str,   default=None,          help="Path to pre-tokenized .pt chunk list (skips HF download)")
    # TurboAdam-specific args
    p.add_argument("--v_bits",             type=int,   default=4,             help="Bits per element for v compression: 4, 6, 8, or 16 (default 4)")
    p.add_argument("--no_compress_m",      action="store_true",               help="Ablation: disable CoState m compression (use fp32 m)")
    p.add_argument("--no_compress_v",      action="store_true",               help="Ablation: disable v compression (use fp32 v)")
    p.add_argument("--null_pct",           type=float, default=0.10,          help="CoState null threshold percentile (default 0.10)")
    p.add_argument("--amp_pct",            type=float, default=0.90,          help="CoState amplitude threshold percentile (default 0.90)")
    p.add_argument("--error_feedback",     action="store_true",               help="Enable CoState error feedback")
    p.add_argument("--no_amp",             action="store_true",               help="Disable AMP mixed precision")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------

def select_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Learning rate schedule: linear warmup + cosine decay
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, total_steps: int, peak_lr: float) -> float:
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    # Cosine decay from peak_lr to 0
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def load_chunks(cache_path: str) -> list:
    """Load pre-tokenized chunks from a .pt file (list of 512-token tensors)."""
    print(f"Loading tokenized cache from {cache_path}…")
    chunks = torch.load(cache_path, weights_only=False)
    print(f"  Chunks: {len(chunks):,}  seq_len: {chunks[0].shape[0]}")
    return chunks


class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx].long()
        return chunk[:-1], chunk[1:]  # input (511), target (511)


# ---------------------------------------------------------------------------
# Gradient norm
# ---------------------------------------------------------------------------

def compute_grad_norm(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total ** 0.5


# ---------------------------------------------------------------------------
# TurboAdam state inspection helpers
# ---------------------------------------------------------------------------

def count_compressed_params(optimizer: TurboAdam) -> int:
    """Count how many parameters have compressed v state."""
    count = 0
    for state in optimizer.state.values():
        if "compressed_v" in state:
            count += 1
    return count


def get_costate_fractions(optimizer: TurboAdam) -> dict | None:
    """Compute mean null/phase/amplitude fractions across all params in Phase B.

    Returns a dict with keys 'null_frac', 'phase_frac', 'amp_frac', or None
    if no parameters are in Phase B yet.
    """
    null_counts = 0
    phase_counts = 0
    amp_counts = 0
    total_blocks = 0

    for state in optimizer.state.values():
        costate_mgr = state.get("m_mgr")
        if costate_mgr is None or not costate_mgr._has_state:
            continue
        encoded = costate_mgr._encoded
        if encoded is None:
            continue
        labels = encoded["labels"]
        n = labels.numel()
        if n == 0:
            continue
        null_counts += (labels == 0).sum().item()
        phase_counts += (labels == 1).sum().item()
        amp_counts += (labels == 2).sum().item()
        total_blocks += n

    if total_blocks == 0:
        return None

    return {
        "null_frac": round(null_counts / total_blocks, 4),
        "phase_frac": round(phase_counts / total_blocks, 4),
        "amp_frac": round(amp_counts / total_blocks, 4),
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.dry_run:
        args.steps = 5
        print("[dry-run] Overriding --steps to 5")

    # Seed
    torch.manual_seed(args.seed)

    # Device
    device = select_device(args.device)
    print(f"Device: {device}")

    use_amp = (device.type == "cuda") and not getattr(args, 'no_amp', False)
    # MPS and CPU run in native precision
    print(f"Mixed precision (AMP): {use_amp}")

    # Output
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "turboadam_log.jsonl")
    print(f"Log: {log_path}")

    # -----------------------------------------------------------------------
    # Model + tokenizer
    # -----------------------------------------------------------------------
    print("Loading GPT-2 124M…")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token
    # Disable length warnings — we're concatenating tokens ourselves, not running
    # full articles through the model.
    tokenizer.model_max_length = int(1e30)

    model = AutoModelForCausalLM.from_pretrained("gpt2")
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}  ({n_params / 1e6:.1f}M)")

    # -----------------------------------------------------------------------
    # Dataset / DataLoader
    # -----------------------------------------------------------------------
    if args.cache_path:
        chunks = load_chunks(args.cache_path)
    else:
        from datasets import load_dataset
        print("No --cache_path provided, streaming WikiText-103 from HuggingFace…")
        def _build_token_chunks(tokenizer, seq_len, split="train"):
            dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
            token_ids = []
            for sample in dataset:
                text = sample["text"].strip()
                if not text:
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
                token_ids.extend(ids)
                token_ids.append(tokenizer.eos_token_id)
            chunks = []
            for i in range(0, len(token_ids) - seq_len, seq_len):
                chunk = token_ids[i : i + seq_len + 1]
                if len(chunk) == seq_len + 1:
                    chunks.append(torch.tensor(chunk, dtype=torch.long))
            return chunks
        chunks = _build_token_chunks(tokenizer, args.seq_len)
    dataset = ChunkDataset(chunks)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,       # keep simple; avoids multiprocessing issues on MPS
        pin_memory=(device.type == "cuda"),
        generator=torch.Generator().manual_seed(args.seed),
        drop_last=True,
    )

    # -----------------------------------------------------------------------
    # Optimizer — TurboAdam with same hyperparameters as baseline
    # -----------------------------------------------------------------------
    compress_m = not args.no_compress_m
    compress_v = not args.no_compress_v
    optimizer = TurboAdam(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        block_size=128,
        v_bits=args.v_bits,
        compress_m=compress_m,
        compress_v=compress_v,
        null_pct=args.null_pct,
        amp_pct=args.amp_pct,
        error_feedback=args.error_feedback,
    )

    print(
        f"TurboAdam config: v_bits={args.v_bits}, "
        f"compress_m={compress_m}, compress_v={compress_v}, "
        f"null_pct={args.null_pct}, amp_pct={args.amp_pct}"
    )

    # Use new-style torch.amp API (torch >= 2.0)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    model.train()
    step = 0
    running_loss = 0.0
    data_iter = iter(loader)

    log_entries = []

    print(f"\nTraining for {args.steps} steps  "
          f"(micro-batch={args.batch_size}, accum={args.accum_steps}, "
          f"eff-batch={args.batch_size * args.accum_steps})")

    t_start = time.time()

    while step < args.steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for _ in range(args.accum_steps):
            # Refill iterator if exhausted (epoch wrap)
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(input_ids=x, labels=y)
                loss = outputs.loss / args.accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # Unscale before clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # LR update
        lr = get_lr(step, args.warmup_steps, args.steps, args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        scaler.step(optimizer)
        scaler.update()

        step += 1
        running_loss += accum_loss

        # -------------------------------------------------------------------
        # Logging
        # -------------------------------------------------------------------
        if step % args.log_every == 0 or step == args.steps or step == 1:
            grad_norm = compute_grad_norm(model)
            avg_loss = running_loss / args.log_every if step > 1 else running_loss
            elapsed = time.time() - t_start

            compressed_count = count_compressed_params(optimizer)
            costate_info = get_costate_fractions(optimizer)

            entry = {
                "step": step,
                "loss": round(accum_loss * args.accum_steps, 6),  # full-scale loss
                "avg_loss": round(avg_loss * args.accum_steps, 6),
                "lr": lr,
                "grad_norm": round(grad_norm, 6),
                "elapsed_s": round(elapsed, 2),
                "compressed_count": compressed_count,
            }
            if costate_info is not None:
                entry.update(costate_info)

            log_entries.append(entry)

            costate_str = ""
            if costate_info is not None:
                costate_str = (
                    f"  null={costate_info['null_frac']:.2f} "
                    f"phase={costate_info['phase_frac']:.2f} "
                    f"amp={costate_info['amp_frac']:.2f}"
                )

            print(
                f"step {step:>5d}  loss {entry['loss']:.4f}  "
                f"lr {lr:.2e}  grad_norm {grad_norm:.3f}  "
                f"compressed {compressed_count}"
                f"{costate_str}  elapsed {elapsed:.1f}s"
            )
            if step % args.log_every == 0:
                running_loss = 0.0

    # -----------------------------------------------------------------------
    # Write log
    # -----------------------------------------------------------------------
    with open(log_path, "w") as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + "\n")

    total_time = time.time() - t_start
    print(f"\nDone. {step} steps in {total_time:.1f}s  ({total_time / step:.2f}s/step)")
    print(f"Log written to {log_path}")


if __name__ == "__main__":
    main()
