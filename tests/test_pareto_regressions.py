from __future__ import annotations

import copy
import math

import pytest
import torch

from turboadam import TurboAdam
from turboadam.costate import (
    CoStateManager,
    compute_block_ratios,
    decode_blocks,
    decompose,
    encode_blocks,
)
from turboadam.oneq import compress_v_logscale, decompress_v, init_compressed_v
from turboadam.quantize import (
    fused_adam_update,
    fused_v_update,
    pack_nbit_indices,
    quantize_logscale_nbits,
    unpack_nbit_indices,
)
from turboadam.utils import packed_num_bytes

BITS = (2, 3, 4, 6, 8)


@pytest.mark.parametrize("n_bits", BITS)
@pytest.mark.parametrize("num_blocks", (1, 3))
def test_dense_index_pack_is_lossless_and_exact_size(n_bits: int, num_blocks: int):
    num_values = num_blocks * 128
    indices = torch.randint(0, 1 << n_bits, (num_values,), dtype=torch.uint8)
    packed = pack_nbit_indices(indices, n_bits)
    restored = unpack_nbit_indices(packed, n_bits, num_values)

    assert torch.equal(restored, indices)
    assert packed.dtype == torch.uint8
    assert packed.numel() == packed_num_bytes(num_values, n_bits)


@pytest.mark.parametrize("n_bits", BITS)
def test_packed_and_unpacked_quantization_decode_identically(n_bits: int):
    torch.manual_seed(1)
    values = torch.randn(384).mul_(3).exp_()

    unpacked, scales_u, _ = quantize_logscale_nbits(
        values, n_bits=n_bits, block_size=128, packed=False
    )
    packed, scales_p, _ = quantize_logscale_nbits(
        values, n_bits=n_bits, block_size=128, packed=True
    )

    assert torch.equal(scales_u, scales_p)
    assert torch.equal(unpack_nbit_indices(packed, n_bits, values.numel()), unpacked)


@pytest.mark.parametrize("n_bits", BITS)
def test_packed_v_recurrence_is_numerically_identical_to_unpacked(n_bits: int):
    torch.manual_seed(2)
    v = torch.rand(384).mul_(0.1).add_(1e-7)
    grad = torch.randn_like(v).mul_(0.03)
    unpacked = compress_v_logscale(v, n_bits=n_bits, packed=False)
    packed = compress_v_logscale(v, n_bits=n_bits, packed=True)

    rng = torch.get_rng_state()
    u_idx, u_scales, u_v = fused_v_update(
        unpacked["indices"],
        unpacked["scales"],
        grad,
        0.999,
        n_bits,
        128,
        v.numel(),
        packed=False,
    )
    torch.set_rng_state(rng)
    p_idx, p_scales, p_v = fused_v_update(
        packed["indices"],
        packed["scales"],
        grad,
        0.999,
        n_bits,
        128,
        v.numel(),
        packed=True,
    )

    assert torch.equal(u_v, p_v)
    assert torch.equal(u_scales, p_scales)
    assert torch.equal(
        u_idx, unpack_nbit_indices(p_idx, n_bits, u_idx.numel())
    )


@pytest.mark.parametrize("n_bits", BITS)
def test_direct_initial_state_has_true_physical_bitwidth(n_bits: int):
    state = init_compressed_v(
        (1000,), device=torch.device("cpu"), n_bits=n_bits, block_size=128
    )
    padded = math.ceil(1000 / 128) * 128

    assert state["packed"] is True
    assert state["indices"].numel() == packed_num_bytes(padded, n_bits)
    assert state["indices"].dtype == torch.uint8
    assert state["scales"].dtype == torch.float16
    reconstructed = decompress_v(state)
    assert reconstructed.shape == (1000,)
    assert torch.isfinite(reconstructed).all()
    assert (reconstructed > 0).all()


def _compressed_state_bytes(opt: TurboAdam, p: torch.Tensor) -> int:
    state = opt.state[p]
    cv = state["compressed_v"]
    total = cv["indices"].numel() * cv["indices"].element_size()
    total += cv["scales"].numel() * cv["scales"].element_size()
    mgr = state["m_mgr"]
    total += mgr._alpha.numel() * mgr._alpha.element_size()
    for value in mgr._encoded.values():
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
    return total


def test_default_large_tensor_state_is_about_5_56_bits_per_parameter():
    torch.manual_seed(3)
    p = torch.nn.Parameter(torch.randn(128 * 1024))
    p.grad = torch.randn_like(p)
    opt = TurboAdam([p])
    opt.step()

    state = opt.state[p]
    assert state["compressed_v"]["packed"] is True
    assert "scales" not in state["m_mgr"]._encoded

    bits_per_param = 8.0 * _compressed_state_bytes(opt, p) / p.numel()
    # 4.25 bpp v + 1.25 bpp CoState + one fp32 alpha scalar.
    expected = 5.5 + 32.0 / p.numel()
    assert bits_per_param == pytest.approx(expected, abs=1e-12)
    assert bits_per_param < 5.5003


@pytest.mark.parametrize("weight_decay", (0.0, 0.01, 0.1))
def test_no_compression_matches_adamw_exactly(weight_decay: float):
    torch.manual_seed(4)
    p_ref = torch.nn.Parameter(torch.randn(257))
    p_new = torch.nn.Parameter(p_ref.detach().clone())
    ref = torch.optim.AdamW(
        [p_ref], lr=3e-3, betas=(0.87, 0.997), eps=1e-7,
        weight_decay=weight_decay, foreach=False
    )
    new = TurboAdam(
        [p_new], lr=3e-3, betas=(0.87, 0.997), eps=1e-7,
        weight_decay=weight_decay, compress_m=False, compress_v=False
    )

    for _ in range(50):
        grad = torch.randn_like(p_ref)
        p_ref.grad = grad.clone()
        p_new.grad = grad.clone()
        ref.step()
        new.step()

    assert torch.equal(p_new, p_ref)
    assert torch.equal(new.state[p_new]["exp_avg"], ref.state[p_ref]["exp_avg"])
    assert torch.equal(
        new.state[p_new]["exp_avg_sq"], ref.state[p_ref]["exp_avg_sq"]
    )


def test_bias_correction_is_per_parameter_with_intermittent_gradients():
    p1_ref = torch.nn.Parameter(torch.tensor([1.0]))
    p2_ref = torch.nn.Parameter(torch.tensor([1.0]))
    p1_new = torch.nn.Parameter(torch.tensor([1.0]))
    p2_new = torch.nn.Parameter(torch.tensor([1.0]))
    ref = torch.optim.AdamW(
        [p1_ref, p2_ref], lr=1e-2, weight_decay=0.0, foreach=False
    )
    new = TurboAdam(
        [p1_new, p2_new], lr=1e-2, weight_decay=0.0,
        compress_m=False, compress_v=False
    )

    for step in range(10):
        p1_ref.grad = torch.ones_like(p1_ref)
        p1_new.grad = torch.ones_like(p1_new)
        if step in (0, 4, 9):
            p2_ref.grad = torch.full_like(p2_ref, 2.0)
            p2_new.grad = torch.full_like(p2_new, 2.0)
        else:
            p2_ref.grad = None
            p2_new.grad = None
        ref.step()
        new.step()

    assert torch.equal(p1_new, p1_ref)
    assert torch.equal(p2_new, p2_ref)
    assert new.state[p1_new]["step"] == 10
    assert new.state[p2_new]["step"] == 3


def test_add_param_group_preserves_each_parameters_own_step():
    torch.manual_seed(5)
    p1_ref = torch.nn.Parameter(torch.randn(31))
    p1_new = torch.nn.Parameter(p1_ref.detach().clone())
    ref = torch.optim.AdamW(
        [p1_ref], lr=1e-3, weight_decay=0.0, foreach=False
    )
    new = TurboAdam([p1_new], lr=1e-3, compress_m=False, compress_v=False)

    for _ in range(7):
        grad = torch.randn_like(p1_ref)
        p1_ref.grad = grad.clone()
        p1_new.grad = grad.clone()
        ref.step()
        new.step()

    p2_ref = torch.nn.Parameter(torch.randn(31))
    p2_new = torch.nn.Parameter(p2_ref.detach().clone())
    ref.add_param_group({"params": [p2_ref], "lr": 2e-3})
    new.add_param_group({"params": [p2_new], "lr": 2e-3})

    for _ in range(5):
        g1 = torch.randn_like(p1_ref)
        g2 = torch.randn_like(p2_ref)
        p1_ref.grad, p1_new.grad = g1.clone(), g1.clone()
        p2_ref.grad, p2_new.grad = g2.clone(), g2.clone()
        ref.step()
        new.step()

    assert torch.equal(p1_new, p1_ref)
    assert torch.equal(p2_new, p2_ref)
    assert new.state[p1_new]["step"] == 12
    assert new.state[p2_new]["step"] == 5


@pytest.mark.parametrize("n_bits", BITS)
def test_v_only_optimizer_matches_unpacked_reference(n_bits: int):
    torch.manual_seed(6)
    p_opt = torch.nn.Parameter(torch.randn(513))
    p_ref = torch.nn.Parameter(p_opt.detach().clone())
    opt = TurboAdam(
        [p_opt], lr=2e-3, betas=(0.9, 0.999), eps=1e-8,
        compress_m=False, compress_v=True, v_bits=n_bits
    )
    exp_avg = torch.zeros_like(p_ref, dtype=torch.float32)
    cv = init_compressed_v(
        p_ref.shape, device=p_ref.device, n_bits=n_bits, packed=False
    )

    for step in range(1, 41):
        grad = torch.randn_like(p_opt)
        p_opt.grad = grad.clone()
        exp_avg.lerp_(grad, 0.1)
        bc1 = 1.0 - 0.9**step
        bc2 = 1.0 - 0.999**step

        rng = torch.get_rng_state()
        opt.step()
        torch.set_rng_state(rng)
        cv["indices"], cv["scales"] = fused_adam_update(
            cv["indices"], cv["scales"], grad, p_ref, exp_avg,
            0.999, n_bits, 128, p_ref.numel(), 2e-3, bc1, bc2,
            1e-8, 0.0, packed=False
        )

    packed_state = opt.state[p_opt]["compressed_v"]
    unpacked_opt = unpack_nbit_indices(
        packed_state["indices"],
        n_bits,
        packed_state["scales"].shape[0] * 128,
    )
    assert torch.equal(p_opt, p_ref)
    assert torch.equal(packed_state["scales"], cv["scales"])
    assert torch.equal(unpacked_opt, cv["indices"])


def test_state_dict_resume_is_exact_when_rng_state_is_replayed():
    torch.manual_seed(7)
    p1 = torch.nn.Parameter(torch.randn(5000))
    opt1 = TurboAdam([p1], lr=1e-3)
    for _ in range(6):
        p1.grad = torch.randn_like(p1)
        opt1.step()

    saved = copy.deepcopy(opt1.state_dict())
    p2 = torch.nn.Parameter(p1.detach().clone())
    opt2 = TurboAdam([p2], lr=1e-3)
    opt2.load_state_dict(saved)

    for _ in range(8):
        grad = torch.randn_like(p1)
        p1.grad = grad.clone()
        p2.grad = grad.clone()
        rng = torch.get_rng_state()
        opt1.step()
        torch.set_rng_state(rng)
        opt2.step()

    assert torch.equal(p1, p2)
    s1, s2 = opt1.state[p1], opt2.state[p2]
    assert s1["step"] == s2["step"]
    assert torch.equal(s1["compressed_v"]["indices"], s2["compressed_v"]["indices"])
    assert torch.equal(s1["compressed_v"]["scales"], s2["compressed_v"]["scales"])
    assert torch.equal(s1["m_mgr"]._alpha, s2["m_mgr"]._alpha)
    for key in s1["m_mgr"]._encoded:
        assert torch.equal(s1["m_mgr"]._encoded[key], s2["m_mgr"]._encoded[key])


def test_legacy_unpacked_state_is_migrated_without_numerical_loss():
    torch.manual_seed(8)
    p = torch.nn.Parameter(torch.randn(5000))
    opt = TurboAdam([p], v_bits=4)
    for _ in range(3):
        p.grad = torch.randn_like(p)
        opt.step()

    legacy = copy.deepcopy(opt.state_dict())
    legacy_state = next(iter(legacy["state"].values()))
    cv = legacy_state["compressed_v"]
    padded = cv["scales"].shape[0] * cv["block_size"]
    unpacked = unpack_nbit_indices(cv["indices"], cv["n_bits"], padded)
    cv["indices"] = unpacked
    cv.pop("packed", None)
    cv.pop("codec_version", None)
    legacy_state["_bc1"] = torch.ones(1)
    legacy_state["_bc2"] = torch.ones(1)
    mgr = legacy_state["m_mgr"]
    compact = mgr._encoded
    stored_norms = compact["block_norms"]
    amplitude = stored_norms < 0
    positive_norms = stored_norms.abs()
    phase = (~amplitude) & (positive_norms > 0)
    labels = torch.zeros(stored_norms.numel(), dtype=torch.uint8)
    labels[phase] = 1
    labels[amplitude] = 2
    compact["labels"] = labels
    compact["block_norms"] = positive_norms
    compact["scales"] = (positive_norms / math.sqrt(mgr.block_size)).half()

    p2 = torch.nn.Parameter(p.detach().clone())
    opt2 = TurboAdam([p2], v_bits=4)
    opt2.load_state_dict(legacy)
    migrated = opt2.state[p2]

    assert "_bc1" not in migrated and "_bc2" not in migrated
    assert migrated["compressed_v"]["packed"] is True
    assert migrated["compressed_v"]["indices"].numel() == packed_num_bytes(
        padded, 4
    )
    assert "scales" not in migrated["m_mgr"]._encoded
    assert torch.equal(
        unpack_nbit_indices(migrated["compressed_v"]["indices"], 4, padded),
        unpacked,
    )

    p2.grad = torch.randn_like(p2)
    opt2.step()
    assert torch.isfinite(p2).all()


@pytest.mark.parametrize("magnitude", (1e-40, 1e-30, 1e18))
def test_projection_is_safe_against_dot_underflow_and_overflow(magnitude: float):
    g = torch.full((4096,), magnitude, dtype=torch.float32)
    m = g * 0.125
    alpha, delta = decompose(m, g)

    expected = (m.double().dot(g.double()) / g.double().dot(g.double())).item()
    assert torch.isfinite(alpha)
    assert float(alpha) == pytest.approx(expected, rel=2e-6, abs=1e-7)
    assert torch.isfinite(delta).all()


def test_block_norms_and_ratios_avoid_intermediate_overflow():
    delta = torch.full((256,), 1e20, dtype=torch.float32)
    labels = torch.ones(2, dtype=torch.uint8)
    encoded = encode_blocks(delta, labels, 128)
    expected = math.sqrt(128.0) * 1e20

    assert torch.isfinite(encoded["block_norms"]).all()
    assert encoded["block_norms"][0].item() == pytest.approx(expected, rel=2e-6)
    assert torch.allclose(compute_block_ratios(delta, delta, 128), torch.ones(2))
    assert compute_block_ratios(
        torch.full((128,), 1e38), torch.full((128,), 5e37), 128
    ).item() == pytest.approx(2.0, rel=2e-6)
    assert compute_block_ratios(
        torch.full((128,), 1e-40), torch.full((128,), 5e-41), 128
    ).item() == pytest.approx(2.0, rel=2e-5)

    unrepresentable = torch.full((128,), 1e38, dtype=torch.float32)
    saturated = encode_blocks(unrepresentable, torch.ones(1, dtype=torch.uint8), 128)
    assert torch.isfinite(saturated["block_norms"]).all()
    assert saturated["block_norms"][0].item() == pytest.approx(
        torch.finfo(torch.float32).max, rel=2e-7
    )


def test_manager_state_omits_redundant_scale_but_public_codec_keeps_compatibility():
    torch.manual_seed(9)
    g = torch.randn(4096)
    manager = CoStateManager()
    manager.update(g, 0.9)
    assert set(manager._encoded) == {"sign_packed", "block_norms"}

    labels = torch.randint(0, 3, (32,), dtype=torch.uint8)
    delta = torch.randn(4096)
    public = encode_blocks(delta, labels)
    assert set(public) == {"labels", "sign_packed", "block_norms", "scales"}
    reconstructed = decode_blocks(public, torch.tensor(0.2), g)
    compact = dict(public)
    compact.pop("scales")
    reconstructed_compact = decode_blocks(compact, torch.tensor(0.2), g)
    assert torch.equal(reconstructed, reconstructed_compact)

    bit_compact = encode_blocks(
        delta, labels, compact_labels=True, include_legacy_scale=False
    )
    reconstructed_bit_compact = decode_blocks(
        bit_compact, torch.tensor(0.2), g
    )
    assert torch.equal(reconstructed, reconstructed_bit_compact)


@pytest.mark.parametrize(
    "compress_m,compress_v",
    ((False, False), (False, True), (True, False), (True, True)),
)
def test_quadratic_convergence_all_ablation_paths(
    compress_m: bool, compress_v: bool
):
    torch.manual_seed(10)
    x = torch.nn.Parameter(torch.randn(5000))
    opt = TurboAdam(
        [x], lr=1e-2, compress_m=compress_m, compress_v=compress_v
    )
    initial = x.square().sum().item()
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        x.square().sum().backward()
        opt.step()
    final = x.square().sum().item()

    assert torch.isfinite(x).all()
    assert final < initial * 0.1


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
@pytest.mark.parametrize("compress_m,compress_v", ((False, False), (True, True)))
def test_mixed_precision_cpu_stays_finite(
    dtype: torch.dtype, compress_m: bool, compress_v: bool
):
    torch.manual_seed(11)
    x = torch.nn.Parameter(torch.randn(5000, dtype=dtype))
    opt = TurboAdam(
        [x], lr=1e-3, compress_m=compress_m, compress_v=compress_v
    )
    for _ in range(8):
        opt.zero_grad(set_to_none=True)
        x.float().square().mean().backward()
        opt.step()
    assert torch.isfinite(x).all()


@pytest.mark.parametrize("block_size", (8, 16, 32, 64, 128, 256, 512))
@pytest.mark.parametrize("n_bits", BITS)
def test_partial_blocks_all_supported_layouts(block_size: int, n_bits: int):
    torch.manual_seed(block_size + n_bits)
    x = torch.nn.Parameter(torch.randn(block_size + 3))
    opt = TurboAdam(
        [x], lr=1e-3, v_bits=n_bits, block_size=block_size,
        compress_m=False, compress_v=True
    )
    x.grad = torch.randn_like(x)
    opt.step()

    cv = opt.state[x]["compressed_v"]
    assert decompress_v(cv).shape == x.shape
    assert torch.isfinite(x).all()
    assert cv["indices"].numel() == packed_num_bytes(
        cv["scales"].shape[0] * block_size, n_bits
    )


def test_scalar_and_empty_parameters_do_not_crash():
    scalar = torch.nn.Parameter(torch.tensor(1.0))
    empty = torch.nn.Parameter(torch.empty(0))
    opt = TurboAdam([scalar, empty])
    scalar.grad = torch.tensor(2.0)
    empty.grad = torch.empty(0)
    opt.step()
    assert torch.isfinite(scalar)
    assert empty.numel() == 0


def test_constructor_rejects_unsafe_layouts_and_bits():
    p = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match="v_bits"):
        TurboAdam([p], v_bits=5)
    with pytest.raises(ValueError, match="block_size"):
        TurboAdam([p], block_size=24)
    with pytest.raises(ValueError, match="thresholds"):
        TurboAdam([p], null_pct=0.9, amp_pct=0.1)

@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16))
@pytest.mark.parametrize(
    "compress_m,compress_v",
    ((False, False), (False, True), (True, True)),
)
def test_low_precision_checkpoint_preserves_full_state_bits(
    dtype: torch.dtype, compress_m: bool, compress_v: bool
):
    torch.manual_seed(12)
    p1 = torch.nn.Parameter(torch.randn(5000, dtype=dtype))
    opt1 = TurboAdam(
        [p1], compress_m=compress_m, compress_v=compress_v
    )
    for _ in range(4):
        p1.grad = torch.randn_like(p1)
        opt1.step()

    saved = opt1.state_dict()
    p2 = torch.nn.Parameter(p1.detach().clone())
    opt2 = TurboAdam(
        [p2], compress_m=compress_m, compress_v=compress_v
    )
    opt2.load_state_dict(saved)
    s1, s2 = opt1.state[p1], opt2.state[p2]

    for key in ("exp_avg", "exp_avg_sq"):
        if key in s1:
            assert s2[key].dtype == torch.float32
            assert torch.equal(s1[key], s2[key])
    if "compressed_v" in s1:
        assert torch.equal(
            s1["compressed_v"]["indices"], s2["compressed_v"]["indices"]
        )
        assert torch.equal(
            s1["compressed_v"]["scales"], s2["compressed_v"]["scales"]
        )
    if "m_mgr" in s1:
        assert s1["m_mgr"] is not s2["m_mgr"]
        assert torch.equal(s1["m_mgr"]._alpha, s2["m_mgr"]._alpha)
        for key in s1["m_mgr"]._encoded:
            assert torch.equal(
                s1["m_mgr"]._encoded[key], s2["m_mgr"]._encoded[key]
            )


def test_precomputed_costate_norms_preserve_encoded_result_exactly():
    torch.manual_seed(13)
    delta = torch.randn(4099)
    labels = torch.randint(0, 3, (math.ceil(delta.numel() / 128),), dtype=torch.uint8)
    _, block_norms = compute_block_ratios(
        delta, delta, 128, return_delta_norms=True
    )

    reference = encode_blocks(
        delta,
        labels,
        128,
        compact_labels=True,
        include_legacy_scale=False,
    )
    reused = encode_blocks(
        delta,
        labels,
        128,
        compact_labels=True,
        include_legacy_scale=False,
        block_norms=block_norms,
    )

    assert set(reference) == set(reused)
    for key in reference:
        assert torch.equal(reference[key], reused[key])


def test_noncontiguous_cpu_parameter_uses_safe_reference_path():
    torch.manual_seed(14)
    parameter = torch.nn.Parameter(torch.randn(65, 67).t())
    assert not parameter.is_contiguous()
    optimizer = TurboAdam(
        [parameter],
        lr=1e-3,
        compress_m=True,
        compress_v=True,
        min_m_compress_elements=0,
    )

    for _ in range(5):
        parameter.grad = torch.randn_like(parameter)
        optimizer.step()

    assert torch.isfinite(parameter).all()
    assert optimizer.state[parameter]["step"] == 5


@pytest.mark.parametrize(
    "keyword,value,match",
    (
        ("lr", float("nan"), "learning rate"),
        ("lr", float("inf"), "learning rate"),
        ("eps", float("nan"), "epsilon"),
        ("weight_decay", float("inf"), "weight_decay"),
        ("block_size", 128.0, "block_size"),
        ("min_m_compress_elements", 1.5, "non-negative integer"),
    ),
)
def test_constructor_rejects_nonfinite_or_ambiguous_hyperparameters(
    keyword: str, value, match: str
):
    parameter = torch.nn.Parameter(torch.ones(1))
    with pytest.raises(ValueError, match=match):
        TurboAdam([parameter], **{keyword: value})
