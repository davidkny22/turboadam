from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
SPEC = importlib.util.spec_from_file_location(
    "train_language_model", EXPERIMENTS / "train_language_model.py"
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
TokenChunkDataset = MODULE.TokenChunkDataset
scheduled_lr = MODULE.scheduled_lr


def test_chunk_dataset_uses_standard_causal_labels() -> None:
    chunk = torch.arange(513).to(torch.uint16).reshape(1, 513)
    dataset = TokenChunkDataset(chunk, 512)
    input_ids = dataset[0]
    assert input_ids.shape == (512,)
    assert torch.equal(input_ids, torch.arange(512))


def test_learning_rate_warms_up_and_reaches_zero() -> None:
    assert scheduled_lr(1, 100, 500, 6.0e-4) == pytest.approx(6.0e-6)
    assert scheduled_lr(100, 100, 500, 6.0e-4) == pytest.approx(6.0e-4)
    assert scheduled_lr(500, 100, 500, 6.0e-4) == pytest.approx(0.0)
