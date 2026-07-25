"""Compare matched AdamW and TurboAdam language-model runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_run(path: Path) -> tuple[dict, list[dict], dict]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    configs = [record for record in records if record.get("type") == "config"]
    summaries = [record for record in records if record.get("type") == "summary"]
    steps = [record for record in records if record.get("type") == "step"]
    if len(configs) != 1 or len(summaries) != 1 or not steps:
        raise ValueError(f"{path} is not a complete language-model run")
    return configs[0], steps, summaries[0]


def validate_match(adamw: dict, turboadam: dict) -> None:
    if adamw["optimizer"] != "adamw" or turboadam["optimizer"] != "turboadam":
        raise ValueError("expected one AdamW run and one TurboAdam run")
    fields = (
        "dataset",
        "cache_sha256",
        "cache_chunks",
        "model",
        "model_parameters",
        "steps",
        "batch_size",
        "accum_steps",
        "seq_len",
        "lr",
        "warmup_steps",
        "weight_decay",
        "seed",
        "amp",
        "device",
    )
    mismatched = [field for field in fields if adamw[field] != turboadam[field]]
    if mismatched:
        raise ValueError(f"run configurations differ in: {', '.join(mismatched)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adamw", type=Path, required=True)
    parser.add_argument("--turboadam", type=Path, required=True)
    args = parser.parse_args()

    adam_config, adam_steps, adam_summary = load_run(args.adamw)
    turbo_config, turbo_steps, turbo_summary = load_run(args.turboadam)
    validate_match(adam_config, turbo_config)
    adam_by_step = {record["step"]: record for record in adam_steps}
    turbo_by_step = {record["step"]: record for record in turbo_steps}
    if adam_by_step.keys() != turbo_by_step.keys():
        raise ValueError("logged step sets differ")

    print("step,adamw_loss,turboadam_loss,delta,relative_percent")
    for step in sorted(adam_by_step):
        adam_loss = adam_by_step[step]["loss"]
        turbo_loss = turbo_by_step[step]["loss"]
        delta = turbo_loss - adam_loss
        relative = 100.0 * delta / adam_loss
        print(f"{step},{adam_loss:.8f},{turbo_loss:.8f},{delta:.8f},{relative:.6f}")

    final_step = max(adam_by_step)
    adam_final = adam_by_step[final_step]
    turbo_final = turbo_by_step[final_step]
    trailing_delta = turbo_final["trailing_loss"] - adam_final["trailing_loss"]
    speed_ratio = turbo_summary["seconds_per_step"] / adam_summary["seconds_per_step"]
    print()
    print(f"dataset={adam_config['dataset']}")
    print(f"final_delta={turbo_final['loss'] - adam_final['loss']:.8f}")
    print(f"trailing_delta={trailing_delta:.8f}")
    print(f"speed_ratio={speed_ratio:.6f}")


if __name__ == "__main__":
    main()
