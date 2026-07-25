"""Run matched GPT-2 language-model training with AdamW or TurboAdam."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import deque
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM

from turboadam import TurboAdam
from turboadam.utils import state_tensor_bytes

MODEL_ID = "openai-community/gpt2"


class TokenChunkDataset(torch.utils.data.Dataset):
    """Expose fixed token chunks with the causal labels expected by GPT-2."""

    def __init__(self, chunks: torch.Tensor | list[torch.Tensor], seq_len: int) -> None:
        if len(chunks) == 0:
            raise ValueError("token cache is empty")
        self.chunks = chunks
        self.seq_len = seq_len
        if isinstance(chunks, torch.Tensor):
            if chunks.ndim != 2 or chunks.shape[1] != seq_len + 1:
                raise ValueError(f"token tensor must have shape (n, {seq_len + 1})")
        elif any(chunk.numel() != seq_len + 1 for chunk in chunks):
            raise ValueError(f"every token chunk must contain {seq_len + 1} values")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.chunks[index][: self.seq_len].long()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimizer", choices=("adamw", "turboadam"), required=True)
    parser.add_argument(
        "--dataset", choices=("tinystories", "wikitext103"), required=True
    )
    parser.add_argument("--cache-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--accum-steps", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--lr", type=float, default=6.0e-4)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--device", default=None)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--v-bits", type=int, choices=(2, 3, 4, 6, 8), default=4)
    parser.add_argument("--m-step-factor", type=float, default=1.1)
    parser.add_argument("--no-compress-m", action="store_true")
    parser.add_argument("--no-compress-v", action="store_true")
    return parser.parse_args()


def select_device(requested: str | None) -> torch.device:
    if requested is not None:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def scheduled_lr(
    step: int,
    warmup_steps: int,
    total_steps: int,
    peak_lr: float,
) -> float:
    if step <= warmup_steps:
        return peak_lr * step / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_chunks(path: Path) -> torch.Tensor | list[torch.Tensor]:
    chunks = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(chunks, (torch.Tensor, list)):
        raise TypeError("token cache must contain a tensor or list of tensors")
    return chunks


def build_optimizer(
    args: argparse.Namespace,
    model: nn.Module,
) -> torch.optim.Optimizer:
    common = {
        "lr": args.lr,
        "betas": (0.9, 0.999),
        "eps": 1.0e-8,
        "weight_decay": args.weight_decay,
    }
    if args.optimizer == "adamw":
        return torch.optim.AdamW(model.parameters(), **common)
    return TurboAdam(
        model.parameters(),
        **common,
        v_bits=args.v_bits,
        compress_m=not args.no_compress_m,
        compress_v=not args.no_compress_v,
        m_step_factor=args.m_step_factor,
        rounding_seed=args.seed,
    )


def write_record(handle, record: dict) -> None:
    handle.write(json.dumps(record, sort_keys=True) + "\n")
    handle.flush()


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0 or args.accum_steps <= 0:
        raise ValueError("steps, batch size, and accumulation steps must be positive")
    if not 0 <= args.warmup_steps <= args.steps:
        raise ValueError("warmup steps must lie between zero and total steps")
    if args.log_every <= 0:
        raise ValueError("log interval must be positive")

    device = select_device(args.device)
    use_amp = bool(args.amp and device.type == "cuda")
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision("high")

    cache_path = args.cache_path.resolve()
    chunks = load_chunks(cache_path)
    dataset = TokenChunkDataset(chunks, args.seq_len)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(args.seed),
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        local_files_only=args.local_files_only,
    ).to(device)
    optimizer = build_optimizer(args, model)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = {
        "type": "config",
        "optimizer": args.optimizer,
        "dataset": args.dataset,
        "cache_path": str(cache_path),
        "cache_sha256": sha256_file(cache_path),
        "cache_chunks": len(dataset),
        "model": MODEL_ID,
        "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "accum_steps": args.accum_steps,
        "effective_batch_size": args.batch_size * args.accum_steps,
        "seq_len": args.seq_len,
        "lr": args.lr,
        "warmup_steps": args.warmup_steps,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "amp": use_amp,
        "device": str(device),
        "v_bits": args.v_bits if args.optimizer == "turboadam" else None,
        "m_step_factor": (
            args.m_step_factor if args.optimizer == "turboadam" else None
        ),
        "compress_m": (
            not args.no_compress_m if args.optimizer == "turboadam" else None
        ),
        "compress_v": (
            not args.no_compress_v if args.optimizer == "turboadam" else None
        ),
    }

    model.train()
    data_iterator = iter(loader)
    trailing_losses: deque[float] = deque(maxlen=args.log_every)
    if device.type == "cuda":
        torch.cuda.synchronize()
    started = time.perf_counter()

    with args.output.open("w", encoding="utf-8", newline="\n") as log:
        write_record(log, config)
        for step in range(1, args.steps + 1):
            optimizer.zero_grad(set_to_none=True)
            loss_sum = 0.0
            for _ in range(args.accum_steps):
                try:
                    input_ids = next(data_iterator)
                except StopIteration:
                    data_iterator = iter(loader)
                    input_ids = next(data_iterator)
                input_ids = input_ids.to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    mean_loss = model(input_ids=input_ids, labels=input_ids).loss
                    backward_loss = mean_loss / args.accum_steps
                scaler.scale(backward_loss).backward()
                loss_sum += float(mean_loss.detach())

            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            learning_rate = scheduled_lr(step, args.warmup_steps, args.steps, args.lr)
            for group in optimizer.param_groups:
                group["lr"] = learning_rate
            scaler.step(optimizer)
            scaler.update()

            step_loss = loss_sum / args.accum_steps
            trailing_losses.append(step_loss)
            if step == 1 or step % args.log_every == 0 or step == args.steps:
                if device.type == "cuda":
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - started
                record = {
                    "type": "step",
                    "step": step,
                    "loss": step_loss,
                    "trailing_loss": sum(trailing_losses) / len(trailing_losses),
                    "trailing_steps": len(trailing_losses),
                    "lr": learning_rate,
                    "grad_norm": float(grad_norm),
                    "elapsed_s": elapsed,
                }
                if isinstance(optimizer, TurboAdam):
                    record["persistent_state_bytes"] = state_tensor_bytes(
                        optimizer.state
                    )
                    record["ustate_parameters"] = sum(
                        bool(state.get("_use_ustate"))
                        for state in optimizer.state.values()
                    )
                    record["v_state_parameters"] = sum(
                        bool(state.get("_use_v_state"))
                        for state in optimizer.state.values()
                    )
                write_record(log, record)
                print(
                    f"{args.dataset} {args.optimizer} step={step} "
                    f"loss={step_loss:.6f} trailing={record['trailing_loss']:.6f} "
                    f"lr={learning_rate:.3e} elapsed={elapsed:.1f}s",
                    flush=True,
                )

        if device.type == "cuda":
            torch.cuda.synchronize()
        total_seconds = time.perf_counter() - started
        write_record(
            log,
            {
                "type": "summary",
                "steps": args.steps,
                "total_seconds": total_seconds,
                "seconds_per_step": total_seconds / args.steps,
            },
        )


if __name__ == "__main__":
    main()
