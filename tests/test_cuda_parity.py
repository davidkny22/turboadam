from __future__ import annotations

import copy

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required", allow_module_level=True)

import turboadam.optimizer as optimizer_module
from turboadam import TurboAdam
from turboadam.quantize import counter_uniform, decompress_v_state
from turboadam.ustate import decode_ustate
from turboadam.utils import state_tensor_bytes


def test_counter_hash_matches_cpu() -> None:
    cpu = counter_uniform(4099, 77, device=torch.device("cpu"))
    cuda = counter_uniform(4099, 77, device=torch.device("cuda")).cpu()
    assert torch.equal(cpu, cuda)


@pytest.mark.parametrize("n_bits", [2, 3, 4, 6, 8])
def test_fused_optimizer_tracks_torch_reference(
    monkeypatch: pytest.MonkeyPatch, n_bits: int
) -> None:
    if not optimizer_module._HAS_TRITON:
        pytest.skip("TurboAdam could not import Triton")
    generator = torch.Generator(device="cuda").manual_seed(300 + n_bits)
    initial = torch.randn(8193, generator=generator, device="cuda")
    reference_parameter = torch.nn.Parameter(initial.clone())
    fused_parameter = torch.nn.Parameter(initial.clone())
    reference = TurboAdam(
        [reference_parameter],
        lr=3.0e-4,
        weight_decay=0.01,
        v_bits=n_bits,
        min_m_compress_elements=0,
        min_v_compress_elements=0,
        rounding_seed=91,
    )
    fused = TurboAdam(
        [fused_parameter],
        lr=3.0e-4,
        weight_decay=0.01,
        v_bits=n_bits,
        min_m_compress_elements=0,
        min_v_compress_elements=0,
        rounding_seed=91,
    )
    gradients = [
        torch.randn(8193, generator=generator, device="cuda") for _ in range(6)
    ]
    for gradient in gradients:
        reference_parameter.grad = gradient.clone()
        monkeypatch.setattr(optimizer_module, "_HAS_TRITON", False)
        reference.step()
        fused_parameter.grad = gradient.clone()
        monkeypatch.setattr(optimizer_module, "_HAS_TRITON", True)
        fused.step()

    assert torch.allclose(
        fused_parameter, reference_parameter, atol=2.0e-5, rtol=2.0e-5
    )
    reference_state = reference.state[reference_parameter]
    fused_state = fused.state[fused_parameter]
    reference_v = decompress_v_state(reference_state["v_state"])
    fused_v = decompress_v_state(fused_state["v_state"])
    v_error = (fused_v - reference_v).abs()
    assert v_error.max() < 1.0e-4
    assert v_error.mean() < 1.0e-7
    reference_q = decode_ustate(reference_state["ustate"])
    fused_q = decode_ustate(fused_state["ustate"])
    assert torch.allclose(fused_q, reference_q, atol=3.0e-4, rtol=3.0e-4)


def test_cuda_optimizer_uses_fused_kernel(monkeypatch: pytest.MonkeyPatch) -> None:
    if not optimizer_module._HAS_TRITON:
        pytest.skip("TurboAdam could not import Triton")
    calls = []
    fused_step = optimizer_module._triton_fused_step

    def recording_step(*args, **kwargs):
        calls.append(True)
        return fused_step(*args, **kwargs)

    monkeypatch.setattr(optimizer_module, "_triton_fused_step", recording_step)
    parameter = torch.nn.Parameter(torch.randn(8192, device="cuda"))
    optimizer = TurboAdam(
        [parameter], min_m_compress_elements=0, min_v_compress_elements=0
    )
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()
    assert calls == [True]


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_fused_path_supports_parameter_dtypes(dtype: torch.dtype) -> None:
    parameter = torch.nn.Parameter(torch.randn(8192, device="cuda", dtype=dtype))
    optimizer = TurboAdam(
        [parameter], min_m_compress_elements=0, min_v_compress_elements=0
    )
    for _ in range(3):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()
    assert bool(torch.isfinite(parameter).all())


def test_cuda_checkpoint_continuation_is_numerically_equivalent() -> None:
    generator = torch.Generator(device="cuda").manual_seed(404)
    parameter = torch.nn.Parameter(
        torch.randn(8193, generator=generator, device="cuda")
    )
    optimizer = TurboAdam([parameter], rounding_seed=55)
    gradients = [
        torch.randn(8193, generator=generator, device="cuda") for _ in range(7)
    ]
    for gradient in gradients[:4]:
        parameter.grad = gradient
        optimizer.step()
    saved_parameter = parameter.detach().clone()
    saved_state = copy.deepcopy(optimizer.state_dict())
    for gradient in gradients[4:]:
        parameter.grad = gradient
        optimizer.step()

    resumed_parameter = torch.nn.Parameter(saved_parameter)
    resumed = TurboAdam([resumed_parameter], rounding_seed=1)
    resumed.load_state_dict(saved_state)
    loaded = resumed.state[resumed_parameter]
    saved_parameter_id = saved_state["param_groups"][0]["params"][0]
    saved = saved_state["state"][saved_parameter_id]
    for family in ("ustate", "v_state"):
        for key, value in saved[family].items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, loaded[family][key])
    for gradient in gradients[4:]:
        resumed_parameter.grad = gradient
        resumed.step()
    assert torch.allclose(parameter, resumed_parameter, atol=5.0e-7, rtol=0.0)


def test_cuda_persistent_state_matches_memory_contract() -> None:
    numel = 131_072
    parameter = torch.nn.Parameter(torch.zeros(numel, device="cuda"))
    optimizer = TurboAdam([parameter])
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    torch.cuda.synchronize()
    state = optimizer.state[parameter]
    assert state_tensor_bytes(state) == 106_508
    assert bool(torch.isfinite(parameter).all())
