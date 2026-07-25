from __future__ import annotations

import copy
import io
import warnings

import pytest
import torch

from turboadam import TurboAdam
from turboadam.utils import state_tensor_bytes


def _run_uncompressed_pair(
    gradients: list[torch.Tensor | None],
    *,
    betas: tuple[float, float] = (0.9, 0.999),
    weight_decay: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor, TurboAdam, torch.optim.AdamW]:
    initial = torch.randn(257, generator=torch.Generator().manual_seed(17))
    reference = torch.nn.Parameter(initial.clone())
    candidate = torch.nn.Parameter(initial.clone())
    adamw = torch.optim.AdamW(
        [reference],
        lr=3.0e-4,
        betas=betas,
        eps=1.0e-8,
        weight_decay=weight_decay,
    )
    turbo = TurboAdam(
        [candidate],
        lr=3.0e-4,
        betas=betas,
        eps=1.0e-8,
        weight_decay=weight_decay,
        compress_m=False,
        compress_v=False,
    )
    for gradient in gradients:
        reference.grad = None if gradient is None else gradient.clone()
        candidate.grad = None if gradient is None else gradient.clone()
        adamw.step()
        turbo.step()
    return reference, candidate, turbo, adamw


def test_uncompressed_fp32_path_is_bit_exact_to_adamw() -> None:
    generator = torch.Generator().manual_seed(30)
    gradients = [torch.randn(257, generator=generator) for _ in range(30)]
    reference, candidate, turbo, adamw = _run_uncompressed_pair(gradients)
    assert torch.equal(reference, candidate)
    assert torch.equal(
        adamw.state[reference]["exp_avg"], turbo.state[candidate]["exp_avg"]
    )
    assert torch.equal(
        adamw.state[reference]["exp_avg_sq"], turbo.state[candidate]["exp_avg_sq"]
    )


def test_per_parameter_steps_match_intermittent_gradients() -> None:
    torch.manual_seed(2)
    references = [
        torch.nn.Parameter(torch.randn(17)),
        torch.nn.Parameter(torch.randn(17)),
    ]
    candidates = [torch.nn.Parameter(value.detach().clone()) for value in references]
    adamw = torch.optim.AdamW(references, lr=1.0e-3, weight_decay=0.01)
    turbo = TurboAdam(
        candidates,
        lr=1.0e-3,
        weight_decay=0.01,
        compress_m=False,
        compress_v=False,
    )
    schedule = [(True, False), (True, True), (False, True), (True, True)]
    for step, active in enumerate(schedule):
        for index, is_active in enumerate(active):
            gradient = torch.full((17,), float(step + index + 1)) if is_active else None
            references[index].grad = None if gradient is None else gradient.clone()
            candidates[index].grad = None if gradient is None else gradient.clone()
        adamw.step()
        turbo.step()
    for reference, candidate in zip(references, candidates, strict=True):
        assert torch.equal(reference, candidate)
    assert turbo.state[candidates[0]]["step"] == 3
    assert turbo.state[candidates[1]]["step"] == 3


def test_default_large_tensor_has_6_500732_bits_per_value() -> None:
    numel = 131_072
    parameter = torch.nn.Parameter(torch.zeros(numel))
    optimizer = TurboAdam([parameter], rounding_seed=5)
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    assert set(state) == {"step", "_use_ustate", "ustate", "_use_v_state", "v_state"}
    bytes_used = state_tensor_bytes(state)
    assert bytes_used == 106_508
    bits_per_value = bytes_used * 8 / numel
    assert bits_per_value == pytest.approx(6.500732421875)
    assert 64.0 / bits_per_value == pytest.approx(9.845044503699253, rel=1.0e-12)


def test_compact_state_contains_no_full_size_fp32_tensor() -> None:
    parameter = torch.nn.Parameter(torch.randn(32_768))
    optimizer = TurboAdam([parameter])
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    tensors = [
        value
        for family in (state["ustate"], state["v_state"])
        for value in family.values()
        if isinstance(value, torch.Tensor)
    ]
    assert not any(
        tensor.dtype == torch.float32 and tensor.numel() == parameter.numel()
        for tensor in tensors
    )
    assert not any("rand" in key or "error" in key for key in state)


def test_small_parameters_use_exact_states_by_default() -> None:
    parameter = torch.nn.Parameter(torch.randn(1024))
    optimizer = TurboAdam([parameter])
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    assert state["_use_ustate"] is False
    assert state["_use_v_state"] is False
    assert state["exp_avg"].dtype == torch.float32
    assert state["exp_avg_sq"].dtype == torch.float32


def test_thresholds_can_force_compression_for_small_tensors() -> None:
    parameter = torch.nn.Parameter(torch.randn(129))
    optimizer = TurboAdam(
        [parameter], min_m_compress_elements=0, min_v_compress_elements=0
    )
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    state = optimizer.state[parameter]
    assert state["_use_ustate"] is True
    assert state["_use_v_state"] is True
    assert state["ustate"]["codes"].numel() == 64


@pytest.mark.parametrize(
    ("compress_m", "compress_v"),
    [(True, False), (False, True), (False, False), (True, True)],
)
def test_state_ablation_combinations_are_finite(
    compress_m: bool, compress_v: bool
) -> None:
    parameter = torch.nn.Parameter(torch.randn(8192))
    optimizer = TurboAdam(
        [parameter],
        compress_m=compress_m,
        compress_v=compress_v,
        rounding_seed=11,
    )
    for _ in range(4):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
    assert bool(torch.isfinite(parameter).all())


def test_checkpoint_roundtrip_continues_exactly() -> None:
    generator = torch.Generator().manual_seed(44)
    gradients = [torch.randn(8193, generator=generator) for _ in range(9)]
    original_parameter = torch.nn.Parameter(torch.randn(8193, generator=generator))
    original_optimizer = TurboAdam([original_parameter], lr=1.0e-3, rounding_seed=123)
    for gradient in gradients[:5]:
        original_parameter.grad = gradient
        original_optimizer.step()
    saved_parameter = original_parameter.detach().clone()
    saved_state = copy.deepcopy(original_optimizer.state_dict())

    buffer = io.BytesIO()
    torch.save(saved_state, buffer)
    buffer.seek(0)
    loaded_state = torch.load(buffer, weights_only=True)

    for gradient in gradients[5:]:
        original_parameter.grad = gradient
        original_optimizer.step()

    resumed_parameter = torch.nn.Parameter(saved_parameter)
    resumed_optimizer = TurboAdam([resumed_parameter], lr=9.0e-9, rounding_seed=999)
    resumed_optimizer.load_state_dict(loaded_state)
    for gradient in gradients[5:]:
        resumed_parameter.grad = gradient
        resumed_optimizer.step()

    assert torch.equal(original_parameter, resumed_parameter)
    original_state = original_optimizer.state[original_parameter]
    resumed_state = resumed_optimizer.state[resumed_parameter]
    for family in ("ustate", "v_state"):
        for key, value in original_state[family].items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, resumed_state[family][key])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_checkpoint_preserves_codec_bits(dtype: torch.dtype) -> None:
    parameter = torch.nn.Parameter(torch.randn(8193, dtype=dtype))
    optimizer = TurboAdam([parameter], rounding_seed=77)
    for _ in range(3):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    loaded_parameter = torch.nn.Parameter(parameter.detach().clone())
    loaded_optimizer = TurboAdam([loaded_parameter], rounding_seed=1)
    loaded_optimizer.load_state_dict(saved)
    original = optimizer.state[parameter]
    loaded = loaded_optimizer.state[loaded_parameter]
    assert torch.equal(original["ustate"]["codes"], loaded["ustate"]["codes"])
    assert torch.equal(original["ustate"]["means"], loaded["ustate"]["means"])
    assert torch.equal(original["v_state"]["indices"], loaded["v_state"]["indices"])
    assert torch.equal(original["v_state"]["scales"], loaded["v_state"]["scales"])


def test_checkpoint_rejects_noncurrent_state_schema() -> None:
    parameter = torch.nn.Parameter(torch.randn(8193))
    optimizer = TurboAdam([parameter])
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    saved = copy.deepcopy(optimizer.state_dict())
    parameter_id = saved["param_groups"][0]["params"][0]
    saved["state"][parameter_id]["extra"] = True
    loaded = TurboAdam([torch.nn.Parameter(parameter.detach().clone())])
    with pytest.raises(ValueError, match="unexpected"):
        loaded.load_state_dict(saved)


def test_add_param_group_validates_and_steps_new_parameter() -> None:
    first = torch.nn.Parameter(torch.randn(10))
    second = torch.nn.Parameter(torch.randn(10))
    optimizer = TurboAdam([first], compress_m=False, compress_v=False)
    optimizer.add_param_group({"params": [second]})
    first.grad = torch.ones_like(first)
    second.grad = torch.ones_like(second)
    optimizer.step()
    assert optimizer.state[first]["step"] == 1
    assert optimizer.state[second]["step"] == 1
    with pytest.raises(ValueError, match="divisible"):
        optimizer.add_param_group(
            {
                "params": [torch.nn.Parameter(torch.ones(2))],
                "block_size": 96,
                "m_block_size": 64,
            }
        )


def test_noncontiguous_parameter_uses_reference_path() -> None:
    parameter = torch.nn.Parameter(torch.randn(128, 64).t())
    assert not parameter.is_contiguous()
    optimizer = TurboAdam(
        [parameter], min_m_compress_elements=0, min_v_compress_elements=0
    )
    for _ in range(4):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
    assert bool(torch.isfinite(parameter).all())


def test_closure_empty_and_error_paths() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = TurboAdam([parameter], compress_m=False, compress_v=False)
    called = []

    def closure() -> torch.Tensor:
        called.append(True)
        return parameter.square().sum()

    parameter.grad = torch.tensor([1.0])
    loss = optimizer.step(closure)
    assert called and loss.item() > 0

    empty = torch.nn.Parameter(torch.empty(0))
    empty_optimizer = TurboAdam([empty])
    empty.grad = torch.empty(0)
    empty_optimizer.step()
    assert empty not in empty_optimizer.state or not empty_optimizer.state[empty]

    sparse = torch.nn.Parameter(torch.ones(4))
    sparse_optimizer = TurboAdam([sparse])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Sparse invariant checks.*")
        sparse.grad = torch.sparse_coo_tensor(
            torch.tensor([[0]]),
            torch.tensor([1.0]),
            (4,),
            is_coalesced=True,
            check_invariants=True,
        )
    with pytest.raises(RuntimeError, match="sparse"):
        sparse_optimizer.step()


def test_configuration_validation() -> None:
    parameter = torch.nn.Parameter(torch.ones(4))
    with pytest.raises(ValueError, match=r"beta1\*\*2"):
        TurboAdam([parameter], betas=(0.9, 0.8))
    with pytest.raises(ValueError, match="divisible"):
        TurboAdam([parameter], block_size=96, m_block_size=64)
    with pytest.raises(NotImplementedError, match="capture"):
        TurboAdam([parameter], capturable=True)
    with pytest.raises(TypeError):
        TurboAdam([parameter], error_feedback=True)
