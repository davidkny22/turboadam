from __future__ import annotations

import statistics

import torch
from torch import nn

from turboadam import TurboAdam


class TinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(32, 128),
            nn.GELU(),
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, 8),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


def _train(seed: int, use_turboadam: bool) -> float:
    torch.manual_seed(seed)
    model = TinyRegressor()
    teacher = torch.randn(32, 8)
    if use_turboadam:
        optimizer = TurboAdam(
            model.parameters(),
            lr=2.0e-3,
            weight_decay=1.0e-3,
            min_m_compress_elements=4096,
            min_v_compress_elements=4096,
            rounding_seed=100 + seed,
        )
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=2.0e-3, weight_decay=1.0e-3
        )
    generator = torch.Generator().manual_seed(10_000 + seed)
    trailing = []
    for step in range(180):
        inputs = torch.randn(64, 32, generator=generator)
        targets = torch.sin(inputs @ teacher)
        targets += 0.05 * torch.randn(64, 8, generator=generator)
        prediction = model(inputs)
        loss = nn.functional.mse_loss(prediction, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step >= 160:
            trailing.append(float(loss.detach()))
    return sum(trailing) / len(trailing)


def test_compressed_training_tracks_adamw_on_matched_seeds() -> None:
    adamw = [_train(seed, False) for seed in range(3)]
    turboadam = [_train(seed, True) for seed in range(3)]
    ratio = statistics.median(turboadam) / statistics.median(adamw)
    assert ratio < 1.035
