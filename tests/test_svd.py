"""Tests for truncated SVD compression."""

import torch
import pytest
from turboadam.svd import svd_compress, svd_reconstruct


class TestSvdCompress:
    def test_output_shapes(self):
        """U, S, Vh have correct shapes for rank-8 compression."""
        v = torch.randn(256, 256)
        U, S, Vh = svd_compress(v, rank=8)
        assert U.shape == (256, 8)
        assert S.shape == (8,)
        assert Vh.shape == (8, 256)

    def test_factors_are_fp32(self):
        """Factors stored in fp32 for numerical stability with bias-corrected v̂."""
        v = torch.randn(128, 128)
        U, S, Vh = svd_compress(v, rank=4)
        assert U.dtype == torch.float32
        assert S.dtype == torch.float32
        assert Vh.dtype == torch.float32

    def test_low_rank_matrix_near_exact(self):
        """A rank-4 matrix compressed at rank 8 should reconstruct near-exactly."""
        torch.manual_seed(42)
        A = torch.randn(128, 4)
        B = torch.randn(4, 128)
        v = A @ B  # true rank 4
        U, S, Vh = svd_compress(v, rank=8)
        v_hat = svd_reconstruct(U, S, Vh)
        rel_error = (v - v_hat).norm() / v.norm()
        # fp16 factors lose some precision, but should still be very close
        assert rel_error < 0.01

    def test_full_rank_bounded_error(self):
        """Full-rank random matrix — error nonzero but reconstruction works."""
        torch.manual_seed(0)
        v = torch.randn(256, 256)
        U, S, Vh = svd_compress(v, rank=8)
        v_hat = svd_reconstruct(U, S, Vh)
        # rank-8 approx of 256x256 random matrix captures top singular values
        assert v_hat.shape == v.shape
        # Error exists but reconstruction is a valid approximation
        rel_error = (v - v_hat).norm() / v.norm()
        assert rel_error < 1.0  # not exact but not garbage

    def test_storage_savings(self):
        """Factor storage is much smaller than original matrix."""
        rows, cols, rank = 768, 768, 8
        v = torch.randn(rows, cols)
        U, S, Vh = svd_compress(v, rank=rank)
        original_elements = rows * cols
        factor_elements = U.numel() + S.numel() + Vh.numel()
        assert factor_elements < original_elements * 0.03  # < 3%

    def test_reconstruct_dtype(self):
        """Reconstruction returns fp32."""
        v = torch.randn(64, 64)
        U, S, Vh = svd_compress(v, rank=4)
        v_hat = svd_reconstruct(U, S, Vh)
        assert v_hat.dtype == torch.float32

    def test_rectangular_matrix(self):
        """Works on non-square matrices."""
        v = torch.randn(512, 128)
        U, S, Vh = svd_compress(v, rank=8)
        assert U.shape == (512, 8)
        assert Vh.shape == (8, 128)
        v_hat = svd_reconstruct(U, S, Vh)
        assert v_hat.shape == (512, 128)
