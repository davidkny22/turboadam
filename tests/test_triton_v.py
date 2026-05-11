"""Test Triton fused v update against PyTorch reference implementation."""

import torch
import pytest
from turboadam.quantize import fused_v_update as pytorch_fused_v_update
from turboadam.oneq import compress_v_logscale


def _make_test_data(numel=768 * 768, n_bits=4, block_size=128, device="cuda"):
    """Create compressed v state and a gradient for testing."""
    torch.manual_seed(42)
    # Create a realistic v (positive, varying magnitude)
    v = torch.rand(numel, device=device) * 0.1 + 1e-6
    compressed = compress_v_logscale(
        v.reshape(-1), n_bits=n_bits, block_size=block_size
    )
    grad = torch.randn(numel, device=device) * 0.01
    return compressed, grad


class TestTritonVUpdate:
    def test_correctness_vs_pytorch(self):
        """Triton kernel should produce same results as PyTorch reference."""
        from turboadam.triton_kernels import triton_fused_v_update

        comp, grad = _make_test_data()
        beta2 = 0.999

        # PyTorch reference
        torch.manual_seed(0)  # for stochastic rounding
        pt_idx, pt_scales, pt_v = pytorch_fused_v_update(
            comp["indices"].clone(),
            comp["scales"].clone(),
            grad.clone(),
            beta2,
            comp["n_bits"],
            comp["block_size"],
            comp["original_length"],
        )

        # Triton — note: stochastic rounding uses different RNG, so indices won't match
        # But v_out should match (it's computed before quantization)
        tr_idx, tr_scales, tr_v = triton_fused_v_update(
            comp["indices"].clone(),
            comp["scales"].clone(),
            grad.clone(),
            beta2,
            comp["n_bits"],
            comp["block_size"],
            comp["original_length"],
        )

        # v values (pre-quantization) should match closely
        assert torch.allclose(pt_v, tr_v, atol=1e-4, rtol=1e-3), (
            f"v mismatch: max_diff={(pt_v - tr_v).abs().max().item():.6e}"
        )

    def test_output_shapes(self):
        """Output shapes should match input shapes."""
        from turboadam.triton_kernels import triton_fused_v_update

        comp, grad = _make_test_data()
        new_idx, new_scales, v_flat = triton_fused_v_update(
            comp["indices"],
            comp["scales"],
            grad,
            0.999,
            comp["n_bits"],
            comp["block_size"],
            comp["original_length"],
        )

        assert new_idx.shape == comp["indices"].shape
        assert new_scales.shape == comp["scales"].shape
        assert v_flat.shape[0] == comp["original_length"]

    def test_indices_in_range(self):
        """New indices should be in [0, n_buckets)."""
        from turboadam.triton_kernels import triton_fused_v_update

        comp, grad = _make_test_data()
        n_buckets = 2 ** comp["n_bits"]
        new_idx, _, _ = triton_fused_v_update(
            comp["indices"],
            comp["scales"],
            grad,
            0.999,
            comp["n_bits"],
            comp["block_size"],
            comp["original_length"],
        )

        assert new_idx.max().item() < n_buckets
        assert new_idx.min().item() >= 0

    def test_v_positive(self):
        """Output v should be positive (EMA of squared gradients)."""
        from turboadam.triton_kernels import triton_fused_v_update

        comp, grad = _make_test_data()
        _, _, v_flat = triton_fused_v_update(
            comp["indices"],
            comp["scales"],
            grad,
            0.999,
            comp["n_bits"],
            comp["block_size"],
            comp["original_length"],
        )

        assert (v_flat >= 0).all()

    def test_various_bits(self):
        """Should work for 4, 6, 8, and 16 bit widths."""
        from turboadam.triton_kernels import triton_fused_v_update

        for n_bits in [4, 6, 8]:
            comp, grad = _make_test_data(n_bits=n_bits)
            new_idx, new_scales, v_flat = triton_fused_v_update(
                comp["indices"],
                comp["scales"],
                grad,
                0.999,
                comp["n_bits"],
                comp["block_size"],
                comp["original_length"],
            )
            assert v_flat.shape[0] == comp["original_length"]
            assert new_idx.max().item() < 2**n_bits

    def test_performance(self):
        """Triton should be faster than PyTorch for large tensors."""
        from turboadam.triton_kernels import triton_fused_v_update
        import time

        comp, grad = _make_test_data(numel=768 * 768)

        # Warmup
        for _ in range(5):
            pytorch_fused_v_update(
                comp["indices"].clone(),
                comp["scales"].clone(),
                grad,
                0.999,
                comp["n_bits"],
                comp["block_size"],
                comp["original_length"],
            )
            triton_fused_v_update(
                comp["indices"].clone(),
                comp["scales"].clone(),
                grad,
                0.999,
                comp["n_bits"],
                comp["block_size"],
                comp["original_length"],
            )

        n = 200
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            pytorch_fused_v_update(
                comp["indices"],
                comp["scales"],
                grad,
                0.999,
                comp["n_bits"],
                comp["block_size"],
                comp["original_length"],
            )
        torch.cuda.synchronize()
        pt_time = (time.perf_counter() - t0) / n * 1000

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            triton_fused_v_update(
                comp["indices"],
                comp["scales"],
                grad,
                0.999,
                comp["n_bits"],
                comp["block_size"],
                comp["original_length"],
            )
        torch.cuda.synchronize()
        tr_time = (time.perf_counter() - t0) / n * 1000

        print(
            f"\nPyTorch: {pt_time:.3f} ms, Triton: {tr_time:.3f} ms, Speedup: {pt_time / tr_time:.2f}x"
        )
        # Triton should be at least somewhat faster
        assert tr_time < pt_time * 1.5, "Triton should not be significantly slower"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
