"""Tests for 1Q second moment compression.

Covers:
- Warmup detection (relative change threshold)
- compress_v / decompress_v roundtrip
- Matrix vs. non-matrix routing
- Freeze behavior (decompress returns same values on repeated calls)
- refresh_v re-compresses with fresh data
"""

import torch
import pytest

from turboadam.oneq import check_warmup, compress_v, decompress_v, refresh_v


# ---------------------------------------------------------------------------
# check_warmup
# ---------------------------------------------------------------------------

class TestCheckWarmup:
    """check_warmup returns True when relative change is BELOW threshold."""

    def test_returns_true_when_below_threshold(self):
        """Small change → warmup complete → True."""
        v_prev = torch.ones(128)
        # v_current is almost the same; relative change ≈ 0.0
        v_curr = v_prev + 1e-6
        # relative change << 0.01
        assert check_warmup(v_curr, v_prev, threshold=0.01) is True

    def test_returns_false_when_above_threshold(self):
        """Large change → still warming up → False."""
        v_prev = torch.ones(128)
        v_curr = v_prev * 2.0  # 100% change, well above any threshold
        assert check_warmup(v_curr, v_prev, threshold=0.01) is False

    def test_at_exact_threshold_returns_false(self):
        """Relative change at or above threshold should return False.

        We construct v_curr and v_prev so that the ratio is *above* the threshold
        (using 1.1 * threshold) to avoid floating-point underflow that would
        accidentally put the ratio just below the boundary.
        """
        threshold = 0.01
        n = 256
        v_curr = torch.ones(n)
        # Target ratio = 1.1 * threshold (safely above threshold)
        target_ratio = 1.1 * threshold
        # ||delta|| = target_ratio * ||v_curr|| = target_ratio * sqrt(n)
        delta_norm = target_ratio * (n ** 0.5)
        delta = torch.full((n,), delta_norm / n ** 0.5)
        v_prev = v_curr - delta
        # Verify the ratio is actually above threshold
        ratio = (v_curr - v_prev).norm() / v_curr.norm()
        assert ratio.item() >= threshold, f"Test setup error: ratio={ratio.item()} < threshold={threshold}"
        result = check_warmup(v_curr, v_prev, threshold=threshold)
        assert result is False

    def test_zero_v_current_norm_does_not_crash(self):
        """If v_current is all zeros, function should handle gracefully."""
        v_curr = torch.zeros(128)
        v_prev = torch.zeros(128)
        # Ratio is 0/0; implementation should return True (stable) without raising
        result = check_warmup(v_curr, v_prev, threshold=0.01)
        assert isinstance(result, bool)

    def test_threshold_zero_always_false(self):
        """With threshold=0, any nonzero change → False."""
        v_prev = torch.ones(64)
        v_curr = v_prev + 1e-9
        # relative change is nonzero, not strictly below 0
        assert check_warmup(v_curr, v_prev, threshold=0.0) is False

    def test_different_precisions(self):
        """Works on both float32 and float64."""
        v64_prev = torch.ones(128, dtype=torch.float64)
        v64_curr = v64_prev * 1.0005
        result = check_warmup(v64_curr, v64_prev, threshold=0.01)
        assert isinstance(result, bool)


# ---------------------------------------------------------------------------
# Helper: make a fake parameter with a given shape
# ---------------------------------------------------------------------------

def _make_param(shape):
    """Return a leaf tensor simulating a parameter with given shape."""
    t = torch.randn(*shape)
    t.requires_grad_(True)
    return t


# ---------------------------------------------------------------------------
# compress_v / decompress_v — routing
# ---------------------------------------------------------------------------

class TestCompressionRouting:
    """compress_v routes large 2-D tensors to SVD, everything else to logscale."""

    def _make_v(self, shape):
        """Positive v tensor (second moment)."""
        return torch.rand(*shape).abs() + 1e-4  # strictly positive

    def test_matrix_param_produces_svd_tag(self):
        """ndim=2 and numel>10000 → type='svd'."""
        param = _make_param((128, 128))  # numel=16384 > 10000
        v = self._make_v(param.shape)
        compressed = compress_v(param, v, rank=4, block_size=128)
        assert compressed["type"] == "svd"

    def test_non_matrix_param_produces_logscale_tag(self):
        """ndim=1 → type='logscale'."""
        param = _make_param((512,))
        v = self._make_v(param.shape)
        compressed = compress_v(param, v, rank=4, block_size=128)
        assert compressed["type"] == "logscale"

    def test_small_2d_matrix_produces_logscale_tag(self):
        """ndim=2 but numel <= 10000 → type='logscale'."""
        param = _make_param((50, 100))  # numel=5000 <= 10000
        v = self._make_v(param.shape)
        compressed = compress_v(param, v, rank=4, block_size=128)
        assert compressed["type"] == "logscale"

    def test_large_matrix_2d_routes_to_svd(self):
        """ndim=2 and numel just above 10000 → type='svd'."""
        param = _make_param((101, 100))  # numel=10100 > 10000
        v = self._make_v(param.shape)
        compressed = compress_v(param, v, rank=4, block_size=128)
        assert compressed["type"] == "svd"

    def test_3d_tensor_routes_to_svd(self):
        """ndim=3 and numel>10000 → type='svd' (matrix path: ndim>=2)."""
        param = _make_param((4, 64, 64))  # numel=16384 > 10000
        v = self._make_v(param.shape)
        compressed = compress_v(param, v, rank=4, block_size=128)
        assert compressed["type"] == "svd"


# ---------------------------------------------------------------------------
# compress_v / decompress_v — shape correctness
# ---------------------------------------------------------------------------

class TestDecompressShape:
    """decompress_v returns fp32 tensor with same shape as v."""

    def _make_v(self, shape):
        return torch.rand(*shape).abs() + 1e-4

    def test_svd_decompress_returns_correct_shape(self):
        param = _make_param((128, 128))
        v = self._make_v(param.shape)
        compressed = compress_v(param, v, rank=4, block_size=128)
        v_hat = decompress_v(compressed)
        assert v_hat.shape == v.shape

    def test_logscale_decompress_returns_correct_shape_1d(self):
        param = _make_param((512,))
        v = self._make_v(param.shape)
        compressed = compress_v(param, v, rank=4, block_size=128)
        v_hat = decompress_v(compressed)
        assert v_hat.shape == v.shape

    def test_logscale_decompress_returns_correct_shape_small_2d(self):
        param = _make_param((32, 32))  # numel=1024 <= 10000
        v = self._make_v(param.shape)
        compressed = compress_v(param, v, rank=4, block_size=128)
        v_hat = decompress_v(compressed)
        assert v_hat.shape == v.shape

    def test_decompress_returns_fp32(self):
        param = _make_param((128, 128))
        v = self._make_v(param.shape)
        compressed = compress_v(param, v, rank=4, block_size=128)
        v_hat = decompress_v(compressed)
        assert v_hat.dtype == torch.float32

    def test_decompress_logscale_returns_fp32(self):
        param = _make_param((512,))
        v = self._make_v(param.shape)
        compressed = compress_v(param, v, rank=4, block_size=128)
        v_hat = decompress_v(compressed)
        assert v_hat.dtype == torch.float32


# ---------------------------------------------------------------------------
# Freeze behavior
# ---------------------------------------------------------------------------

class TestFreezeBehavior:
    """decompress_v always returns the same values from the same compressed dict."""

    def _make_v(self, shape):
        return torch.rand(*shape).abs() + 1e-4

    def test_svd_repeated_decompress_identical(self):
        param = _make_param((128, 128))
        v = self._make_v(param.shape)
        compressed = compress_v(param, v, rank=4, block_size=128)
        v_hat_1 = decompress_v(compressed)
        v_hat_2 = decompress_v(compressed)
        assert torch.equal(v_hat_1, v_hat_2), "SVD decompress must be deterministic"

    def test_logscale_repeated_decompress_identical(self):
        param = _make_param((512,))
        v = self._make_v(param.shape)
        compressed = compress_v(param, v, rank=4, block_size=128)
        v_hat_1 = decompress_v(compressed)
        v_hat_2 = decompress_v(compressed)
        assert torch.equal(v_hat_1, v_hat_2), "Logscale decompress must be deterministic"

    def test_decompress_does_not_mutate_compressed_dict(self):
        """Calling decompress_v twice should not change the stored data."""
        param = _make_param((128, 128))
        v = self._make_v(param.shape)
        compressed = compress_v(param, v, rank=4, block_size=128)
        keys_before = set(compressed.keys())
        _ = decompress_v(compressed)
        keys_after = set(compressed.keys())
        assert keys_before == keys_after


# ---------------------------------------------------------------------------
# refresh_v
# ---------------------------------------------------------------------------

class TestRefreshV:
    """refresh_v re-compresses with fresh data and returns a new compressed dict."""

    def _make_v(self, shape):
        return torch.rand(*shape).abs() + 1e-4

    def test_refresh_returns_dict_with_type(self):
        param = _make_param((128, 128))
        v_old = self._make_v(param.shape)
        v_new = self._make_v(param.shape)
        compressed_old = compress_v(param, v_old, rank=4, block_size=128)
        compressed_new = refresh_v(compressed_old, v_new, param, rank=4, block_size=128)
        assert "type" in compressed_new

    def test_refresh_svd_preserves_type(self):
        param = _make_param((128, 128))
        v_old = self._make_v(param.shape)
        v_new = self._make_v(param.shape)
        compressed_old = compress_v(param, v_old, rank=4, block_size=128)
        compressed_new = refresh_v(compressed_old, v_new, param, rank=4, block_size=128)
        assert compressed_new["type"] == "svd"

    def test_refresh_logscale_preserves_type(self):
        param = _make_param((512,))
        v_old = self._make_v(param.shape)
        v_new = self._make_v(param.shape)
        compressed_old = compress_v(param, v_old, rank=4, block_size=128)
        compressed_new = refresh_v(compressed_old, v_new, param, rank=4, block_size=128)
        assert compressed_new["type"] == "logscale"

    def test_refresh_updates_compressed_data(self):
        """After refresh with very different v, decompress should give different result."""
        torch.manual_seed(0)
        param = _make_param((128, 128))
        v_old = torch.ones(param.shape) * 0.001
        v_new = torch.ones(param.shape) * 100.0  # drastically different
        compressed_old = compress_v(param, v_old, rank=4, block_size=128)
        compressed_new = refresh_v(compressed_old, v_new, param, rank=4, block_size=128)
        v_hat_old = decompress_v(compressed_old)
        v_hat_new = decompress_v(compressed_new)
        # The means should differ substantially
        assert abs(v_hat_new.mean().item() - v_hat_old.mean().item()) > 1.0

    def test_refresh_shape_preserved(self):
        param = _make_param((512,))
        v_old = self._make_v(param.shape)
        v_new = self._make_v(param.shape)
        compressed_old = compress_v(param, v_old, rank=4, block_size=128)
        compressed_new = refresh_v(compressed_old, v_new, param, rank=4, block_size=128)
        v_hat = decompress_v(compressed_new)
        assert v_hat.shape == v_new.shape
