"""Tests for CoState first moment compression."""

import math
import torch
import pytest
from turboadam.costate import (
    decompose,
    compute_block_ratios,
    compute_thresholds,
    classify_blocks,
    encode_blocks,
    decode_blocks,
)


class TestDecompose:
    def test_aligned_gradient(self):
        """When m = 3*g, alpha should be 3 and delta should be zero."""
        g = torch.randn(256)
        m = 3.0 * g
        alpha, delta = decompose(m, g)
        assert abs(alpha - 3.0) < 1e-5
        assert delta.norm() < 1e-4

    def test_orthogonal(self):
        """When m is orthogonal to g, alpha should be 0 and delta should equal m."""
        g = torch.zeros(256)
        g[0] = 1.0
        m = torch.zeros(256)
        m[1] = 5.0  # orthogonal to g
        alpha, delta = decompose(m, g)
        assert abs(alpha) < 1e-5
        assert torch.allclose(delta, m, atol=1e-5)

    def test_reconstruction_identity(self):
        """m should equal alpha*g + delta within float precision."""
        torch.manual_seed(42)
        m = torch.randn(512)
        g = torch.randn(512)
        alpha, delta = decompose(m, g)
        m_reconstructed = alpha * g + delta
        assert torch.allclose(m, m_reconstructed, atol=1e-5)

    def test_zero_gradient(self):
        """When g is all zeros, alpha=0 and delta=m."""
        m = torch.randn(128)
        g = torch.zeros(128)
        alpha, delta = decompose(m, g)
        assert alpha == 0.0
        assert torch.allclose(delta, m)

    def test_alpha_is_scalar(self):
        """Alpha should be a single float, not a tensor."""
        m = torch.randn(256)
        g = torch.randn(256)
        alpha, delta = decompose(m, g)
        assert isinstance(alpha, float)

    def test_delta_shape_matches_m(self):
        """Delta should have the same shape as m."""
        m = torch.randn(300)
        g = torch.randn(300)
        alpha, delta = decompose(m, g)
        assert delta.shape == m.shape


class TestComputeBlockRatios:
    BLOCK_SIZE = 128

    def test_near_zero_delta(self):
        """When delta is near zero, all ratios should be near zero."""
        torch.manual_seed(0)
        m = torch.randn(256)
        delta = torch.zeros(256)
        ratios = compute_block_ratios(delta, m, self.BLOCK_SIZE)
        assert ratios.shape == (2,)
        assert (ratios < 1e-6).all(), f"Expected near-zero ratios, got {ratios}"

    def test_delta_equals_m(self):
        """When delta == m, each ratio should be 1.0."""
        torch.manual_seed(1)
        m = torch.randn(256)
        delta = m.clone()
        ratios = compute_block_ratios(delta, m, self.BLOCK_SIZE)
        assert ratios.shape == (2,)
        assert torch.allclose(ratios, torch.ones(2), atol=1e-5), f"Expected all-ones ratios, got {ratios}"

    def test_zero_m_block(self):
        """When an m block is all zeros, the ratio for that block should be 0."""
        m = torch.zeros(256)
        delta = torch.randn(256)
        ratios = compute_block_ratios(delta, m, self.BLOCK_SIZE)
        assert ratios.shape == (2,)
        assert (ratios == 0.0).all(), f"Expected zero ratios for zero m, got {ratios}"

    def test_non_block_aligned_size(self):
        """Should handle tensors not aligned to block boundaries."""
        torch.manual_seed(2)
        m = torch.randn(300)  # 300 elements, not a multiple of 128
        delta = torch.randn(300)
        ratios = compute_block_ratios(delta, m, self.BLOCK_SIZE)
        # ceil(300 / 128) = 3 blocks
        assert ratios.shape == (3,)
        assert (ratios >= 0).all()

    def test_ratio_output_dtype(self):
        """Ratios should be a float32 tensor."""
        m = torch.randn(128)
        delta = torch.randn(128)
        ratios = compute_block_ratios(delta, m, self.BLOCK_SIZE)
        assert ratios.dtype == torch.float32

    def test_ratio_values_scale_with_delta(self):
        """Doubling delta should double the ratio (m non-zero, same m)."""
        torch.manual_seed(3)
        m = torch.randn(128) + 1.0  # avoid near-zero m
        delta = torch.randn(128)
        r1 = compute_block_ratios(delta, m, self.BLOCK_SIZE)
        r2 = compute_block_ratios(2 * delta, m, self.BLOCK_SIZE)
        assert torch.allclose(r2, 2 * r1, atol=1e-5), f"Expected 2x ratio scaling, got {r1} vs {r2}"


class TestComputeThresholds:
    def test_known_uniform_distribution(self):
        """With uniform [0,1] ratios, tau0 ≈ 0.1, tau1 ≈ 0.9."""
        torch.manual_seed(42)
        ratios = torch.linspace(0.0, 1.0, steps=1000)
        tau0, tau1 = compute_thresholds(ratios)
        assert abs(tau0 - 0.1) < 0.01, f"Expected tau0 ≈ 0.1, got {tau0}"
        assert abs(tau1 - 0.9) < 0.01, f"Expected tau1 ≈ 0.9, got {tau1}"

    def test_tau0_less_than_tau1(self):
        """tau0 should always be less than or equal to tau1."""
        torch.manual_seed(5)
        ratios = torch.rand(512)
        tau0, tau1 = compute_thresholds(ratios)
        assert tau0 <= tau1, f"tau0={tau0} should be <= tau1={tau1}"

    def test_constant_ratios(self):
        """When all ratios are equal, tau0 == tau1 == that value."""
        ratios = torch.full((128,), 0.5)
        tau0, tau1 = compute_thresholds(ratios)
        assert abs(tau0 - 0.5) < 1e-5
        assert abs(tau1 - 0.5) < 1e-5

    def test_return_types_are_floats(self):
        """tau0 and tau1 should be Python floats (or scalar tensors)."""
        ratios = torch.rand(256)
        tau0, tau1 = compute_thresholds(ratios)
        # Accept both float and 0-dim tensor
        assert isinstance(tau0, (float, torch.Tensor))
        assert isinstance(tau1, (float, torch.Tensor))

    def test_single_element(self):
        """Single-element ratio tensor should produce tau0 == tau1 == that value."""
        ratios = torch.tensor([0.42])
        tau0, tau1 = compute_thresholds(ratios)
        assert abs(float(tau0) - 0.42) < 1e-5
        assert abs(float(tau1) - 0.42) < 1e-5


class TestClassifyBlocks:
    def test_all_null(self):
        """All ratios below tau0 → all labels are 0 (null)."""
        ratios = torch.tensor([0.01, 0.02, 0.03])
        tau0, tau1 = 0.1, 0.9
        labels = classify_blocks(ratios, tau0, tau1)
        assert labels.dtype == torch.uint8
        assert (labels == 0).all(), f"Expected all null, got {labels}"

    def test_all_phase(self):
        """All ratios in [tau0, tau1) → all labels are 1 (phase)."""
        ratios = torch.tensor([0.3, 0.5, 0.7])
        tau0, tau1 = 0.1, 0.9
        labels = classify_blocks(ratios, tau0, tau1)
        assert labels.dtype == torch.uint8
        assert (labels == 1).all(), f"Expected all phase, got {labels}"

    def test_all_amplitude(self):
        """All ratios >= tau1 → all labels are 2 (amplitude)."""
        ratios = torch.tensor([0.9, 1.1, 1.5])
        tau0, tau1 = 0.1, 0.9
        labels = classify_blocks(ratios, tau0, tau1)
        assert labels.dtype == torch.uint8
        assert (labels == 2).all(), f"Expected all amplitude, got {labels}"

    def test_mixed_labels(self):
        """Verify correct label assignment across all three classes."""
        ratios = torch.tensor([0.05, 0.5, 0.95])
        tau0, tau1 = 0.1, 0.9
        labels = classify_blocks(ratios, tau0, tau1)
        assert labels[0] == 0  # null
        assert labels[1] == 1  # phase
        assert labels[2] == 2  # amplitude

    def test_boundary_at_tau0(self):
        """Ratio exactly equal to tau0 should be phase (label=1)."""
        ratios = torch.tensor([0.1])
        tau0, tau1 = 0.1, 0.9
        labels = classify_blocks(ratios, tau0, tau1)
        assert labels[0] == 1, f"Boundary at tau0 should be phase, got {labels[0]}"

    def test_boundary_at_tau1(self):
        """Ratio exactly equal to tau1 should be amplitude (label=2)."""
        ratios = torch.tensor([0.9])
        tau0, tau1 = 0.1, 0.9
        labels = classify_blocks(ratios, tau0, tau1)
        assert labels[0] == 2, f"Boundary at tau1 should be amplitude, got {labels[0]}"

    def test_output_shape_matches_input(self):
        """Output labels tensor shape should match ratios tensor shape."""
        ratios = torch.rand(64)
        tau0, tau1 = 0.1, 0.9
        labels = classify_blocks(ratios, tau0, tau1)
        assert labels.shape == ratios.shape


class TestEncodeDecodeBlocks:
    BLOCK_SIZE = 128

    def _make_single_block_delta(self, label: int, seed: int = 7):
        """Create a 1-block delta with known label and return (delta, labels, block_size)."""
        torch.manual_seed(seed)
        delta = torch.randn(self.BLOCK_SIZE)
        labels = torch.tensor([label], dtype=torch.uint8)
        return delta, labels

    # --- encode_blocks structure tests ---

    def test_encode_returns_required_keys(self):
        """encode_blocks must return a dict with labels, sign_packed, block_norms, scales."""
        torch.manual_seed(10)
        delta = torch.randn(256)
        labels = torch.tensor([0, 1], dtype=torch.uint8)
        enc = encode_blocks(delta, labels, self.BLOCK_SIZE)
        for key in ("labels", "sign_packed", "block_norms", "scales"):
            assert key in enc, f"Missing key: {key}"

    def test_encode_labels_preserved(self):
        """Labels stored in encoded dict should match input labels."""
        delta = torch.randn(256)
        labels = torch.tensor([0, 2], dtype=torch.uint8)
        enc = encode_blocks(delta, labels, self.BLOCK_SIZE)
        assert torch.equal(enc["labels"], labels)

    def test_encode_sign_packed_dtype(self):
        """sign_packed should be uint8."""
        delta = torch.randn(256)
        labels = torch.tensor([1, 1], dtype=torch.uint8)
        enc = encode_blocks(delta, labels, self.BLOCK_SIZE)
        assert enc["sign_packed"].dtype == torch.uint8

    def test_encode_sign_packed_size(self):
        """sign_packed should have ceil(numel / 8) bytes per block (total ceil(n_elements / 8))."""
        n = 256  # 2 blocks of 128
        delta = torch.randn(n)
        labels = torch.tensor([1, 1], dtype=torch.uint8)
        enc = encode_blocks(delta, labels, self.BLOCK_SIZE)
        expected_bytes = math.ceil(n / 8)
        assert enc["sign_packed"].numel() == expected_bytes, (
            f"Expected {expected_bytes} sign bytes, got {enc['sign_packed'].numel()}"
        )

    def test_encode_block_norms_dtype_and_shape(self):
        """block_norms should be fp32 with one entry per block."""
        delta = torch.randn(256)
        labels = torch.tensor([0, 2], dtype=torch.uint8)
        enc = encode_blocks(delta, labels, self.BLOCK_SIZE)
        assert enc["block_norms"].dtype == torch.float32
        assert enc["block_norms"].shape == (2,)

    def test_encode_scales_dtype_and_shape(self):
        """scales should be fp16 with one entry per block."""
        delta = torch.randn(256)
        labels = torch.tensor([2, 2], dtype=torch.uint8)
        enc = encode_blocks(delta, labels, self.BLOCK_SIZE)
        assert enc["scales"].dtype == torch.float16
        assert enc["scales"].shape == (2,)

    # --- decode_blocks reconstruction tests ---

    def test_null_costate_reconstruction(self):
        """Null costate: decoded output = alpha*g (delta contribution is zero)."""
        torch.manual_seed(20)
        g = torch.randn(self.BLOCK_SIZE)
        delta = torch.randn(self.BLOCK_SIZE)
        alpha = 2.5
        labels = torch.tensor([0], dtype=torch.uint8)  # null
        enc = encode_blocks(delta, labels, self.BLOCK_SIZE)
        result = decode_blocks(enc, alpha, g, self.BLOCK_SIZE, self.BLOCK_SIZE)
        expected = alpha * g
        assert torch.allclose(result, expected, atol=1e-5), (
            f"Null costate: expected alpha*g, max diff={((result - expected).abs().max())}"
        )

    def test_phase_costate_reconstruction(self):
        """Phase costate: m_hat = alpha*g + (norm(delta_block)/sqrt(block_size)) * sign(delta_block)."""
        torch.manual_seed(21)
        g = torch.randn(self.BLOCK_SIZE)
        delta = torch.randn(self.BLOCK_SIZE)
        alpha = 1.0
        labels = torch.tensor([1], dtype=torch.uint8)  # phase
        enc = encode_blocks(delta, labels, self.BLOCK_SIZE)
        result = decode_blocks(enc, alpha, g, self.BLOCK_SIZE, self.BLOCK_SIZE)

        norm_d = delta.norm()
        scale = norm_d / math.sqrt(self.BLOCK_SIZE)
        delta_hat = scale * torch.sign(delta)
        expected = alpha * g + delta_hat
        assert torch.allclose(result, expected, atol=1e-4), (
            f"Phase costate max diff={((result - expected).abs().max())}"
        )

    def test_amplitude_costate_reconstruction(self):
        """Amplitude costate: m_hat = alpha*g + scale * sign(delta_block), scale is fp16 per-block."""
        torch.manual_seed(22)
        g = torch.randn(self.BLOCK_SIZE)
        delta = torch.randn(self.BLOCK_SIZE)
        alpha = 0.5
        labels = torch.tensor([2], dtype=torch.uint8)  # amplitude
        enc = encode_blocks(delta, labels, self.BLOCK_SIZE)
        result = decode_blocks(enc, alpha, g, self.BLOCK_SIZE, self.BLOCK_SIZE)

        # scale stored as fp16 = block norm (fp16 quantized)
        norm_d = delta.norm()
        scale_fp16 = norm_d.to(torch.float16).to(torch.float32)
        delta_hat = scale_fp16 * torch.sign(delta)
        expected = alpha * g + delta_hat
        assert torch.allclose(result, expected, atol=1e-2), (
            f"Amplitude costate max diff={((result - expected).abs().max())}"
        )

    def test_decode_output_shape_matches_g(self):
        """Decoded output should have the same shape as g (original_numel)."""
        torch.manual_seed(23)
        g = torch.randn(300)
        delta = torch.randn(300)
        alpha = 1.0
        # ceil(300/128) = 3 blocks
        labels = torch.tensor([0, 1, 2], dtype=torch.uint8)
        enc = encode_blocks(delta, labels, self.BLOCK_SIZE)
        result = decode_blocks(enc, alpha, g, self.BLOCK_SIZE, 300)
        assert result.shape == g.shape, f"Expected shape {g.shape}, got {result.shape}"

    def test_sign_packing_roundtrip(self):
        """Signs packed into uint8 bytes should be faithfully recovered during decode."""
        torch.manual_seed(24)
        # Use phase costate so sign bits are the only thing stored for delta
        g = torch.zeros(self.BLOCK_SIZE)
        delta = torch.randn(self.BLOCK_SIZE)
        alpha = 0.0
        labels = torch.tensor([1], dtype=torch.uint8)
        enc = encode_blocks(delta, labels, self.BLOCK_SIZE)
        result = decode_blocks(enc, alpha, g, self.BLOCK_SIZE, self.BLOCK_SIZE)
        # result should be (norm/sqrt(n)) * sign(delta)
        signs_expected = torch.sign(delta)
        signs_recovered = torch.sign(result)
        # Handle exact zeros (unlikely with randn)
        nonzero = delta != 0
        assert torch.all(signs_expected[nonzero] == signs_recovered[nonzero]), (
            "Sign roundtrip failed for phase costate"
        )

    def test_multi_block_mixed_costates(self):
        """Multi-block tensor with mixed costate labels should decode each block correctly."""
        torch.manual_seed(25)
        n = 384  # 3 blocks
        g = torch.randn(n)
        delta = torch.randn(n)
        alpha = 1.5
        labels = torch.tensor([0, 1, 2], dtype=torch.uint8)
        enc = encode_blocks(delta, labels, self.BLOCK_SIZE)
        result = decode_blocks(enc, alpha, g, self.BLOCK_SIZE, n)
        assert result.shape == (n,)

        # Manually compute expected per-block
        expected = alpha * g.clone()
        for i, label in enumerate([0, 1, 2]):
            start = i * self.BLOCK_SIZE
            end = start + self.BLOCK_SIZE
            db = delta[start:end]
            if label == 0:
                pass  # null: no delta contribution
            elif label == 1:
                scale = db.norm() / math.sqrt(self.BLOCK_SIZE)
                expected[start:end] += scale * torch.sign(db)
            else:
                scale_fp16 = db.norm().to(torch.float16).to(torch.float32)
                expected[start:end] += scale_fp16 * torch.sign(db)

        assert torch.allclose(result, expected, atol=1e-2), (
            f"Multi-block decode failed, max diff={((result - expected).abs().max())}"
        )
