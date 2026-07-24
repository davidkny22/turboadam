"""CUDA/Triton parity and stress tests for the optimized kernels.

These tests skip cleanly on CPU-only hosts. Run them on the target GPU with:
    PYTHONPATH=. pytest -q tests/test_cuda_parity.py
"""

from __future__ import annotations

import copy
import math

import pytest
import torch

pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required for Triton parity tests", allow_module_level=True)

from turboadam import TurboAdam
from turboadam.costate import (
    classify_blocks,
    compute_block_ratios,
    compute_thresholds,
    decode_blocks,
    encode_blocks,
)
from turboadam.oneq import compress_v_logscale
from turboadam.quantize import fused_adam_update, fused_v_update
from turboadam.triton_kernels import (
    triton_costate_decode,
    triton_costate_encode,
    triton_decompose_ratios,
    triton_fused_adam_update,
    triton_fused_v_update,
    triton_projection_alpha,
)


@pytest.mark.parametrize("n_bits", (2, 3, 4, 6, 8))
@pytest.mark.parametrize("block_size", (64, 128, 256))
def test_triton_v_matches_pytorch_prequantized_state(
    n_bits: int, block_size: int
):
    torch.manual_seed(101 + n_bits + block_size)
    numel = block_size * 7 + 13
    v = torch.rand(numel, device="cuda").mul_(0.1).add_(1e-12)
    grad = torch.randn(numel, device="cuda").mul_(0.03)
    compressed = compress_v_logscale(
        v, n_bits=n_bits, block_size=block_size, packed=True
    )

    pt_state = copy.deepcopy(compressed)
    tr_state = copy.deepcopy(compressed)
    pt_idx, pt_scales, pt_v = fused_v_update(
        pt_state["indices"],
        pt_state["scales"],
        grad,
        0.999,
        n_bits,
        block_size,
        numel,
        packed=True,
    )
    tr_idx, tr_scales, tr_v = triton_fused_v_update(
        tr_state["indices"],
        tr_state["scales"],
        grad,
        0.999,
        n_bits,
        block_size,
        numel,
        seed=12345,
        packed=True,
    )

    assert tr_idx.shape == pt_idx.shape
    assert torch.allclose(tr_v, pt_v, atol=2e-6, rtol=2e-5)
    assert torch.allclose(tr_scales, pt_scales, atol=2e-3, rtol=0.0)
    assert int(tr_idx.min()) >= 0


@pytest.mark.parametrize("n_bits", (2, 4, 8))
def test_triton_fused_parameter_update_matches_reference(n_bits: int):
    torch.manual_seed(201 + n_bits)
    numel = 128 * 17 + 7
    param_pt = torch.randn(numel, device="cuda")
    param_tr = param_pt.clone()
    grad = torch.randn_like(param_pt).mul_(0.02)
    first_moment = torch.randn_like(param_pt).mul_(0.01)
    v = torch.rand_like(param_pt).mul_(0.1).add_(1e-10)
    compressed = compress_v_logscale(v, n_bits=n_bits, packed=True)
    pt_state = copy.deepcopy(compressed)
    tr_state = copy.deepcopy(compressed)

    pt_idx, pt_scales = fused_adam_update(
        pt_state["indices"],
        pt_state["scales"],
        grad,
        param_pt,
        first_moment,
        0.999,
        n_bits,
        128,
        numel,
        3e-4,
        1.0 - 0.9**17,
        1.0 - 0.999**17,
        1e-8,
        0.01,
        packed=True,
    )
    tr_idx, tr_scales = triton_fused_adam_update(
        tr_state["indices"],
        tr_state["scales"],
        grad,
        param_tr,
        first_moment,
        0.999,
        n_bits,
        128,
        numel,
        3e-4,
        1.0 - 0.9**17,
        1.0 - 0.999**17,
        1e-8,
        0.01,
        seed=54321,
        packed=True,
    )

    assert tr_idx.shape == pt_idx.shape
    assert torch.allclose(param_tr, param_pt, atol=2e-6, rtol=2e-5)
    assert torch.allclose(tr_scales, pt_scales, atol=2e-3, rtol=0.0)


def test_triton_costate_codec_matches_pytorch_codec():
    torch.manual_seed(301)
    numel = 128 * 23 + 11
    delta = torch.randn(numel, device="cuda").mul_(0.03)
    m = torch.randn_like(delta).mul_(0.02)
    g = torch.randn_like(delta).mul_(0.02)
    ratios, block_norms = compute_block_ratios(
        delta, m, return_delta_norms=True
    )
    tau0, tau1 = compute_thresholds(ratios)
    labels = classify_blocks(ratios, tau0, tau1)

    pt = encode_blocks(
        delta,
        labels,
        compact_labels=True,
        include_legacy_scale=False,
        block_norms=block_norms,
    )
    tr = triton_costate_encode(
        delta,
        labels,
        128,
        compact_labels=True,
        include_legacy_scale=False,
        block_norms=block_norms,
    )
    assert torch.equal(tr["sign_packed"], pt["sign_packed"])
    assert torch.allclose(tr["block_norms"], pt["block_norms"], atol=2e-6)

    alpha = torch.tensor(0.13, device="cuda")
    pt_m = decode_blocks(pt, alpha, g, 128, numel)
    tr_m = triton_costate_decode(tr, alpha, g, 128, numel)
    assert torch.allclose(tr_m, pt_m, atol=2e-6, rtol=2e-6)


def test_triton_extreme_projection_and_ratio_stay_finite():
    g = torch.full((4096,), 1e-20, device="cuda")
    m = torch.full((4096,), 1e18, device="cuda")
    alpha = triton_projection_alpha(m, g)
    expected = (m.cpu().double().dot(g.cpu().double()) /
                g.cpu().double().dot(g.cpu().double()))
    assert torch.isfinite(alpha)
    assert float(alpha.cpu()) == pytest.approx(float(expected), rel=5e-5)

    orthogonal_g = torch.tensor([1e-20, 1e-20], device="cuda").repeat(2048)
    orthogonal_m = torch.tensor([1e18, -1e18], device="cuda").repeat(2048)
    orthogonal_alpha = triton_projection_alpha(orthogonal_m, orthogonal_g)
    assert torch.isfinite(orthogonal_alpha)
    assert abs(float(orthogonal_alpha.cpu())) < 1e-5

    m_new = torch.full((4096,), 5e37, device="cuda")
    ratio_g = torch.full((4096,), -5e37, device="cuda")
    _, ratios = triton_decompose_ratios(
        m_new, ratio_g, torch.tensor(1.0, device="cuda"), 128
    )
    assert torch.isfinite(ratios).all()
    assert torch.allclose(ratios, torch.full_like(ratios, 2.0), rtol=5e-5)


def test_default_optimizer_cuda_state_has_no_full_random_buffer():
    torch.manual_seed(401)
    parameter = torch.nn.Parameter(torch.randn(128 * 1024, device="cuda"))
    optimizer = TurboAdam([parameter], min_m_compress_elements=0)
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()

    state = optimizer.state[parameter]
    assert "rand_buf" not in state["compressed_v"]
    assert set(state["m_mgr"]._encoded) == {"sign_packed", "block_norms"}
    assert torch.isfinite(parameter).all()
