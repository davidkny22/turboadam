"""Ablation sweep script for TurboAdam hyperparameter sensitivity.

Runs shortened training loops with configurable mode and hyperparameters.
Supports: baseline, combined, m-only, v-only, and no-compression modes.
Logs to parameterized JSONL filenames for batch analysis.

Usage:
    python experiments/ablation.py --mode combined --v_bits 4 --steps 500
    python experiments/ablation.py --mode baseline --steps 500
    python experiments/ablation.py --mode costate_only --null_pct 0.05 --steps 500
"""

import argparse
import json
import math
import os
import time

import torch

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser(description="TurboAdam ablation sweep")
    # Mode
    p.add_argument(
        "--mode",
        type=str,
        default="combined",
        choices=["baseline", "combined", "costate_only", "1q_only", "no_compression"],
        help="Optimizer mode",
    )
    # TurboAdam hyperparameters
    p.add_argument("--block_size", type=int, default=128)
    p.add_argument(
        "--v_bits", type=int, default=4, help="Bits for v compression: 4, 6, 8, or 16"
    )
    p.add_argument(
        "--null_pct", type=float, default=0.10, help="CoState null threshold percentile"
    )
    p.add_argument(
        "--amp_pct",
        type=float,
        default=0.90,
        help="CoState amplitude threshold percentile",
    )
    p.add_argument(
        "--error_feedback", action="store_true", help="Enable CoState error feedback"
    )
    # Training
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accum_steps", type=int, default=4)
    p.add_argument("--lr", type=float, default=6e-4)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--seq_len", type=int, default=512)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--output_dir", type=str, default="experiments/results")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def select_device(requested):
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_lr(step, warmup_steps, total_steps, peak_lr):
    if step < warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def build_token_chunks(tokenizer, seq_len, split="train"):
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
    token_ids = []
    for sample in dataset:
        text = sample["text"].strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_ids.extend(ids)
        token_ids.append(tokenizer.eos_token_id)
    chunks = []
    for i in range(0, len(token_ids) - seq_len, seq_len):
        chunk = token_ids[i : i + seq_len + 1]
        if len(chunk) == seq_len + 1:
            chunks.append(chunk)
    return chunks


class ChunkDataset(torch.utils.data.Dataset):
    def __init__(self, chunks):
        self.chunks = chunks

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = torch.tensor(self.chunks[idx], dtype=torch.long)
        return chunk[:-1], chunk[1:]


def compute_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return total**0.5


def make_optimizer(mode, model, args):
    """Create optimizer based on mode."""
    if mode == "baseline":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8,
        )

    from turboadam import TurboAdam

    if mode == "costate_only":
        return TurboAdam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8,
            block_size=args.block_size,
            compress_m=True,
            compress_v=False,
            null_pct=args.null_pct,
            amp_pct=args.amp_pct,
            error_feedback=args.error_feedback,
        )
    elif mode == "1q_only":
        return TurboAdam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8,
            block_size=args.block_size,
            compress_m=False,
            compress_v=True,
            v_bits=args.v_bits,
        )
    elif mode == "no_compression":
        return TurboAdam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8,
            block_size=args.block_size,
            compress_m=False,
            compress_v=False,
        )
    else:
        # combined (default)
        return TurboAdam(
            model.parameters(),
            lr=args.lr,
            betas=(0.9, 0.999),
            weight_decay=0.01,
            eps=1e-8,
            block_size=args.block_size,
            v_bits=args.v_bits,
            compress_m=True,
            compress_v=True,
            null_pct=args.null_pct,
            amp_pct=args.amp_pct,
            error_feedback=args.error_feedback,
        )


def main():
    args = parse_args()
    if args.dry_run:
        args.steps = 5
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    use_amp = device.type == "cuda"
    os.makedirs(args.output_dir, exist_ok=True)

    # Parameterized log filename
    tag = (
        f"ablation_{args.mode}_bs{args.block_size}_vb{args.v_bits}"
        f"_np{args.null_pct}_ap{args.amp_pct}"
    )
    log_path = os.path.join(args.output_dir, f"{tag}.jsonl")
    print(f"Mode: {args.mode} | Log: {log_path}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

    chunks = build_token_chunks(tokenizer, args.seq_len)
    dataset = ChunkDataset(chunks)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        generator=torch.Generator().manual_seed(args.seed),
        drop_last=True,
    )

    optimizer = make_optimizer(args.mode, model, args)

    if use_amp:
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None

    model.train()
    step = 0
    micro_step = 0
    running_loss = 0.0
    data_iter = iter(loader)
    log_entries = []
    t0 = time.time()

    while step < args.steps:
        try:
            input_ids, labels = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            input_ids, labels = next(data_iter)

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        if use_amp:
            with torch.amp.autocast("cuda"):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs.loss / args.accum_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss / args.accum_steps
            loss.backward()

        running_loss += loss.item()
        micro_step += 1

        if micro_step % args.accum_steps == 0:
            if use_amp:
                scaler.unscale_(optimizer)

            grad_norm = compute_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            lr_now = get_lr(step, args.warmup_steps, args.steps, args.lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_now

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()

            step += 1
            if step % args.log_every == 0 or step == 1:
                avg_loss = running_loss / args.log_every if step > 1 else running_loss
                entry = {
                    "step": step,
                    "loss": loss.item() * args.accum_steps,
                    "avg_loss": avg_loss,
                    "lr": lr_now,
                    "grad_norm": grad_norm,
                    "elapsed_s": time.time() - t0,
                    "mode": args.mode,
                }
                log_entries.append(entry)
                print(
                    f"[{args.mode}] step={step} loss={entry['loss']:.4f} "
                    f"avg={avg_loss:.4f} lr={lr_now:.6f} grad={grad_norm:.2f}"
                )
                running_loss = 0.0

    with open(log_path, "w") as f:
        for entry in log_entries:
            f.write(json.dumps(entry) + "\n")
    print(f"Done. {step} steps. Log: {log_path}")


if __name__ == "__main__":
    main()
