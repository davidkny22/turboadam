"""Tests for 2-bit log-scale quantization."""

import torch
from turboadam.quantize import quantize_logscale, dequantize_logscale


class TestQuantizeLogscale:
    def test_roundtrip_lognormal(self):
        """Quantize/dequantize lognormal data — error should be bounded."""
        torch.manual_seed(42)
        v = torch.randn(256).exp()  # lognormal, strictly positive
        packed, scales = quantize_logscale(v, block_size=128)
        v_hat = dequantize_logscale(packed, scales, block_size=128, original_numel=256)
        assert v_hat.shape == v.shape
        # 2-bit log-scale is coarse — relative error can be large per element,
        # but the overall L2 should be bounded
        rel_error = (v - v_hat).norm() / v.norm()
        assert rel_error < 0.5

    def test_output_shapes(self):
        """Packed indices and scales have correct shapes."""
        v = torch.ones(256) * 2.0
        packed, scales = quantize_logscale(v, block_size=128)
        num_blocks = 256 // 128
        # 128 elements / 4 per byte = 32 bytes per block
        assert packed.shape == (num_blocks * 32,)
        assert packed.dtype == torch.uint8
        assert scales.shape == (num_blocks, 2)
        assert scales.dtype == torch.float16

    def test_constant_block(self):
        """All-same-value block should roundtrip to the same constant."""
        v = torch.ones(128) * 3.14
        packed, scales = quantize_logscale(v, block_size=128)
        v_hat = dequantize_logscale(packed, scales, block_size=128, original_numel=128)
        # All values identical → all map to same bucket → exact roundtrip
        assert torch.allclose(v_hat, v, rtol=1e-2)

    def test_single_block(self):
        """Works on exactly one block."""
        v = torch.rand(128).exp()
        packed, scales = quantize_logscale(v, block_size=128)
        v_hat = dequantize_logscale(packed, scales, block_size=128, original_numel=128)
        assert v_hat.shape == (128,)

    def test_values_strictly_positive(self):
        """Output should be strictly positive (v values are always positive)."""
        torch.manual_seed(0)
        v = torch.rand(256).exp()
        packed, scales = quantize_logscale(v, block_size=128)
        v_hat = dequantize_logscale(packed, scales, block_size=128, original_numel=256)
        assert (v_hat > 0).all()

    def test_non_block_aligned_input(self):
        """Input not aligned to block_size — caller pads, but test the math."""
        # Pad to 256 manually, quantize, then check first 200 elements
        v = torch.rand(200).exp()
        padded = torch.cat([v, torch.ones(56)])  # pad to 256
        packed, scales = quantize_logscale(padded, block_size=128)
        v_hat = dequantize_logscale(packed, scales, block_size=128, original_numel=256)
        # First 200 should approximate original
        rel_error = (v - v_hat[:200]).norm() / v.norm()
        assert rel_error < 0.5
