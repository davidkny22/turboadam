"""Tests for CoState first moment compression."""

import torch
import pytest
from turboadam.costate import decompose


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
