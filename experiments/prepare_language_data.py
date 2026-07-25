"""Build fixed GPT-2 token caches for language-model experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
from array import array
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_ID = "openai-community/gpt2"
DATASETS = {
    "tinystories": ("roneneldan/TinyStories", None),
    "wikitext103": ("wikitext", "wikitext-103-raw-v1"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if args.seq_len <= 0:
        raise ValueError("sequence length must be positive")
    if args.max_chunks is not None and args.max_chunks <= 0:
        raise ValueError("maximum chunks must be positive")
    dataset_id, dataset_config = DATASETS[args.dataset]
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        local_files_only=args.local_files_only,
    )
    dataset = load_dataset(
        dataset_id,
        dataset_config,
        split="train",
        streaming=args.streaming,
    )

    token_buffer = array("H")
    source_rows = 0
    nonempty_rows = 0
    for row in dataset:
        source_rows += 1
        text = row["text"].strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if any(token_id > 0xFFFF for token_id in ids):
            raise ValueError("token id does not fit uint16 storage")
        token_buffer.extend(ids)
        token_buffer.append(tokenizer.eos_token_id)
        nonempty_rows += 1
        if (
            args.max_chunks is not None
            and len(token_buffer) >= args.max_chunks * args.seq_len + 1
        ):
            break

    tokens = torch.frombuffer(token_buffer, dtype=torch.uint16)
    chunk_count = (tokens.numel() - 1) // args.seq_len
    if args.max_chunks is not None:
        chunk_count = min(chunk_count, args.max_chunks)
    if chunk_count <= 0:
        raise ValueError("dataset does not contain one complete token chunk")
    chunks = (
        tokens[: chunk_count * args.seq_len + 1]
        .unfold(0, args.seq_len + 1, args.seq_len)
        .contiguous()
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(chunks, args.output)
    manifest = {
        "dataset": args.dataset,
        "dataset_id": dataset_id,
        "dataset_config": dataset_config,
        "split": "train",
        "tokenizer": MODEL_ID,
        "source_rows": source_rows,
        "nonempty_rows": nonempty_rows,
        "source_tokens": len(token_buffer),
        "seq_len": args.seq_len,
        "chunks": chunks.shape[0],
        "streaming": args.streaming,
        "dtype": str(chunks.dtype),
        "cache_sha256": sha256_file(args.output),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(manifest_path)


if __name__ == "__main__":
    main()
