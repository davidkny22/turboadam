from __future__ import annotations

import copy

import pytest
import torch

from turboadam.oneq import compress_v_logscale, decompress_v, initialize_v_logscale
from turboadam.quantize import (
    MIN_POSITIVE,
    SUPPORTED_N_BITS,
    counter_uniform,
    dequantize_logscale,
    initialize_v_state,
    pack_nbit_indices,
    packed_index_numel,
    quantize_logscale,
    recompress_v_state,
    restore_v_state,
    unpack_nbit_indices,
    validate_v_state,
)


def test_oneq_helpers_roundtrip_current_state() -> None:
    generator = torch.Generator().manual_seed(73)
    second_moment = torch.rand(259, generator=generator).square().add_(1.0e-8)
    state = compress_v_logscale(
        second_moment,
        n_bits=4,
        block_size=128,
        stochastic_round=True,
        seed=91,
    )
    restored = decompress_v(state)
    assert restored.shape == second_moment.shape
    assert restored.dtype == torch.float32
    assert bool(torch.isfinite(restored).all())
    assert bool((restored > 0).all())

    initialized = initialize_v_logscale(
        second_moment.shape,
        device=torch.device("cpu"),
        n_bits=4,
        block_size=128,
    )
    assert decompress_v(initialized).shape == second_moment.shape

    with pytest.raises(ValueError, match="empty"):
        compress_v_logscale(torch.empty(0))


@pytest.mark.parametrize("n_bits", SUPPORTED_N_BITS)
@pytest.mark.parametrize("num_values", [1, 7, 8, 31, 128, 259])
def test_pack_roundtrip(n_bits: int, num_values: int) -> None:
    generator = torch.Generator().manual_seed(101 + n_bits + num_values)
    values = torch.randint(
        0,
        1 << n_bits,
        (num_values,),
        generator=generator,
        dtype=torch.uint8,
    )
    packed = pack_nbit_indices(values, n_bits)
    assert packed.numel() == packed_index_numel(num_values, n_bits)
    assert torch.equal(unpack_nbit_indices(packed, n_bits, num_values), values)


def test_counter_uniform_is_reproducible_antithetic_and_bounded() -> None:
    first = counter_uniform(103, 77, device=torch.device("cpu"))
    second = counter_uniform(103, 77, device=torch.device("cpu"))
    different = counter_uniform(103, 78, device=torch.device("cpu"))
    assert torch.equal(first, second)
    assert not torch.equal(first, different)
    assert bool(((first >= 0.0) & (first < 1.0)).all())
    assert torch.allclose(first[:102:2] + first[1:102:2], torch.ones(51))


@pytest.mark.parametrize("n_bits", SUPPORTED_N_BITS)
def test_log_quantizer_represents_block_extrema(n_bits: int) -> None:
    values = torch.logspace(-30, 20, 256)
    packed, scales = quantize_logscale(values, n_bits, 128)
    decoded = dequantize_logscale(packed, scales, n_bits, 128, values.numel())
    assert packed.dtype == torch.uint8
    assert scales.dtype == torch.float16
    assert bool(torch.isfinite(decoded).all())
    assert bool((decoded > 0).all())
    for block in range(2):
        source = values[block * 128 : (block + 1) * 128]
        restored = decoded[block * 128 : (block + 1) * 128]
        assert restored.min() <= source.min()
        assert restored.max() >= source.max()


def test_value_space_stochastic_rounding_is_empirically_unbiased() -> None:
    values = torch.exp(torch.linspace(-8.0, 2.0, 128))
    samples = []
    for seed in range(600):
        packed, scales = quantize_logscale(
            values,
            4,
            128,
            stochastic_round=True,
            seed=seed,
        )
        samples.append(dequantize_logscale(packed, scales, 4, 128, 128))
    mean = torch.stack(samples).mean(0)
    relative_l1 = ((mean - values).abs().sum() / values.abs().sum()).item()
    assert relative_l1 < 0.012


def test_v_state_schema_roundtrip_and_restore() -> None:
    state = initialize_v_state(
        (257,), device=torch.device("cpu"), n_bits=4, block_size=128
    )
    validate_v_state(state)
    exact = torch.full((257,), 0.001)
    persisted = recompress_v_state(state, exact, seed=42)
    assert persisted.shape == exact.shape
    assert bool(torch.isfinite(persisted).all())
    restored = restore_v_state(state, device=torch.device("cpu"))
    assert restored is not state
    assert torch.equal(restored["indices"], state["indices"])
    assert restored["indices"].data_ptr() != state["indices"].data_ptr()

    invalid = copy.deepcopy(state)
    invalid["extra"] = True
    with pytest.raises(ValueError, match="unexpected"):
        validate_v_state(invalid)


def test_extreme_finite_values_remain_finite() -> None:
    values = torch.tensor(
        [
            MIN_POSITIVE,
            1.0e-30,
            1.0e-10,
            1.0,
            1.0e10,
            1.0e30,
            torch.finfo(torch.float32).max,
        ],
        dtype=torch.float32,
    ).repeat(19)[:128]
    packed, scales = quantize_logscale(values, 4, 128, stochastic_round=True, seed=1)
    decoded = dequantize_logscale(packed, scales, 4, 128, 128)
    assert bool(torch.isfinite(decoded).all())
    assert decoded.max() <= torch.finfo(torch.float32).max
    assert decoded.min() > 0
