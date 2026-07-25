from __future__ import annotations

import copy
import math

import pytest
import torch

from turboadam.ustate import (
    DEFAULT_STEP_FACTOR,
    USTATE_MEAN_BLOCK_SIZE,
    decode_ustate,
    encode_first_moment,
    encode_ustate,
    initialize_ustate,
    reconstruct_first_moment,
    restore_ustate,
    ustate_bound,
    validate_ustate,
)
from turboadam.utils import state_tensor_bytes


@pytest.mark.parametrize("numel", [1, 63, 64, 65, 127, 128, 129, 8193])
def test_layout_and_partial_blocks(numel: int) -> None:
    state = initialize_ustate(numel, device=torch.device("cpu"))
    assert state["representation"] == "ustate"
    assert state["codes"].dtype == torch.uint8
    assert state["means"].dtype == torch.bfloat16
    assert state["means"].numel() == math.ceil(numel / USTATE_MEAN_BLOCK_SIZE)
    assert state["codes"].numel() == math.ceil(numel / 128) * 32
    decoded = decode_ustate(state)
    assert decoded.shape == (numel,)
    assert torch.count_nonzero(decoded) == 0


def test_first_moment_state_uses_2_25_bits_per_value_plus_scalars() -> None:
    numel = 131_072
    state = initialize_ustate(numel, device=torch.device("cpu"))
    bytes_used = state_tensor_bytes(state)
    assert bytes_used == numel // 4 + (numel // 64) * 2 + 12
    assert bytes_used * 8 / numel == pytest.approx(2.250732421875)


def test_stored_means_are_reconstructed_exactly() -> None:
    generator = torch.Generator().manual_seed(12)
    update_state = torch.randn(8192, generator=generator) * 2.0
    update_state += torch.repeat_interleave(torch.linspace(-0.8, 0.8, 128), 64)
    encoded = encode_ustate(update_state, seed=4, storage_block_size=128)
    decoded = decode_ustate(encoded)
    decoded_means = decoded.reshape(-1, 64).mean(1)
    assert torch.allclose(decoded_means, encoded["means"].float(), atol=2.0e-7, rtol=0)


def test_encoding_is_reproducible_and_seed_sensitive() -> None:
    update_state = torch.randn(4096, generator=torch.Generator().manual_seed(91))
    first = encode_ustate(update_state, seed=5, storage_block_size=128)
    repeat = encode_ustate(update_state, seed=5, storage_block_size=128)
    different = encode_ustate(update_state, seed=6, storage_block_size=128)
    assert torch.equal(first["codes"], repeat["codes"])
    assert torch.equal(first["means"], repeat["means"])
    assert not torch.equal(first["codes"], different["codes"])


def test_lagged_step_rotation_matches_written_codes() -> None:
    state = initialize_ustate(
        4096,
        device=torch.device("cpu"),
        step_factor=DEFAULT_STEP_FACTOR,
    )
    first_value = torch.randn(4096, generator=torch.Generator().manual_seed(1))
    first = encode_ustate(first_value, seed=11, storage_block_size=128, out=state)
    assert first["decode_step"].item() == pytest.approx(DEFAULT_STEP_FACTOR)
    expected_next = (
        first_value.reshape(-1, 64) - first["means"].float().unsqueeze(1)
    ).square().mean().sqrt() * DEFAULT_STEP_FACTOR
    assert first["encode_step"].item() == pytest.approx(
        expected_next.item(), rel=1.0e-6
    )
    prior_next = first["encode_step"].item()
    second_value = torch.randn(4096, generator=torch.Generator().manual_seed(2))
    second = encode_ustate(second_value, seed=12, storage_block_size=128, out=first)
    assert second["decode_step"].item() == pytest.approx(prior_next)


def test_current_schema_restores_without_aliasing() -> None:
    encoded = encode_ustate(torch.randn(257), seed=9, storage_block_size=128)
    restored = restore_ustate(encoded, device=torch.device("cpu"))
    validate_ustate(restored)
    assert torch.equal(restored["codes"], encoded["codes"])
    assert restored["codes"].data_ptr() != encoded["codes"].data_ptr()

    invalid = copy.deepcopy(encoded)
    invalid["extra"] = True
    with pytest.raises(ValueError, match="unexpected"):
        validate_ustate(invalid)


def test_unquantized_parameterization_is_an_identity() -> None:
    generator = torch.Generator().manual_seed(88)
    first_moment = torch.randn(1024, generator=generator)
    second_moment = torch.rand(1024, generator=generator).mul_(0.2).add_(1.0e-5)
    beta1, beta2, step, eps = 0.9, 0.999, 37, 1.0e-8
    bc1 = 1.0 - beta1**step
    bc2 = 1.0 - beta2**step
    denominator = torch.sqrt(second_moment / bc2) + eps
    update_state = (first_moment / bc1) / denominator
    reconstructed = update_state * bc1 * denominator
    assert torch.allclose(reconstructed, first_moment, atol=2.0e-7, rtol=2.0e-7)


def test_first_moment_helpers_are_shape_safe_and_finite() -> None:
    generator = torch.Generator().manual_seed(67)
    first_moment = torch.zeros(129)
    second_moment = torch.zeros(129)
    for _ in range(9):
        gradient = torch.randn(129, generator=generator)
        first_moment.lerp_(gradient, 0.1)
        second_moment.mul_(0.999).addcmul_(gradient, gradient, value=0.001)
    encoded = encode_first_moment(
        first_moment.clone(),
        second_moment,
        9,
        0.9,
        0.999,
        1.0e-8,
        seed=4,
        reuse_m_buffer=True,
    )
    restored = reconstruct_first_moment(encoded, second_moment, 9, 0.9, 0.999, 1.0e-8)
    assert restored.shape == first_moment.shape
    assert bool(torch.isfinite(restored).all())
    assert torch.nn.functional.cosine_similarity(restored, first_moment, dim=0) > 0.72


def test_bound_dominates_random_adam_histories() -> None:
    beta1, beta2 = 0.9, 0.999
    generator = torch.Generator().manual_seed(4)
    first_moment = torch.zeros(2048)
    second_moment = torch.zeros(2048)
    for step in range(1, 201):
        gradient = torch.randn(2048, generator=generator)
        gradient *= torch.exp(torch.randn(2048, generator=generator) * 0.7)
        first_moment.lerp_(gradient, 1.0 - beta1)
        second_moment.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
        update_state = (first_moment / (1.0 - beta1**step)) / torch.sqrt(
            second_moment / (1.0 - beta2**step)
        )
        assert (
            update_state.abs().max().item() <= ustate_bound(beta1, beta2, step) + 2.0e-5
        )


def test_unbounded_beta_pair_is_reported() -> None:
    assert math.isinf(ustate_bound(0.9, 0.8, 10))
    with pytest.raises(ValueError):
        ustate_bound(-0.1, 0.999, 2)
