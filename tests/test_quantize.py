"""Tests for 2-bit log-scale quantization."""

import pytest
import torch
from turboadam.quantize import (
    quantize_logscale,
    dequantize_logscale,
    quantize_logscale_nbits,
    dequantize_logscale_nbits,
    fused_v_update,
)


class TestQuantizeLogscale:
    def test_roundtrip_lognormal(self):
        """Quantize/dequantize lognormal data: error should be bounded."""
        torch.manual_seed(42)
        v = torch.randn(256).exp()  # lognormal, strictly positive
        packed, scales = quantize_logscale(v, block_size=128)
        v_hat = dequantize_logscale(packed, scales, block_size=128, original_numel=256)
        assert v_hat.shape == v.shape
        # 2-bit log-scale is coarse: relative error can be large per element,
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
        """Input not aligned to block_size: caller pads, but test the math."""
        # Pad to 256 manually, quantize, then check first 200 elements
        v = torch.rand(200).exp()
        padded = torch.cat([v, torch.ones(56)])  # pad to 256
        packed, scales = quantize_logscale(padded, block_size=128)
        v_hat = dequantize_logscale(packed, scales, block_size=128, original_numel=256)
        # First 200 should approximate original
        rel_error = (v - v_hat[:200]).norm() / v.norm()
        assert rel_error < 0.5


class TestStochasticRoundingUnbiased:
    """Stochastic rounding must be unbiased in decoded (value) space.

    Decode reconstructs each element at the bucket center (idx+0.5)/n_buckets
    of the normalized range.  Rounding (normalized*n_buckets) directly biases
    the decoded value upward by 1/(2*n_buckets); rounding
    (normalized*n_buckets - 0.5) is unbiased.  These tests pin that property.
    """

    @pytest.mark.parametrize("n_bits", [2, 3, 4, 6, 8])
    def test_quantize_nbits_unbiased_in_normalized_space(self, n_bits):
        # Cover the full normalized range [0,1]; average decoded normalized
        # value across many stochastic-rounding draws must match the input
        # within a small tolerance (CLT, ~1/sqrt(N)).
        n_buckets = 2 ** n_bits
        normalized = torch.linspace(0.001, 0.999, 256)  # across one block
        # Map to a real v by giving every block the same log span.
        log_min = torch.full((1,), -1.0)
        log_max = torch.full((1,), 1.0)
        span = (log_max - log_min).item()
        v_flat = (log_min + normalized * span).exp().repeat(1)
        # quantize_logscale_nbits requires block-size multiple; v_flat is 256
        n_draws = 2000
        decoded_means = []
        for _ in range(n_draws):
            idx, scales, _ = quantize_logscale_nbits(
                v_flat, n_bits=n_bits, block_size=64, stochastic_round=True
            )
            hat = dequantize_logscale_nbits(
                idx, scales, n_bits=n_bits, block_size=64, original_numel=256
            )
            decoded_means.append(hat)
        mean_hat = torch.stack(decoded_means).mean(dim=0)
        rel = (mean_hat - v_flat).abs() / (v_flat.abs() + 1e-12)
        # Unbiased => mean relative bias shrinks with sqrt(N); 5% is loose
        # enough for any v_bits yet catches the half-bin systematic shift.
        assert rel.mean().item() < 0.05, (
            f"n_bits={n_bits}: mean decoded|v bias {rel.mean().item():.5f} >= 5%"
        )

    @pytest.mark.parametrize("n_bits", [4, 8])
    def test_fused_v_update_unbiased(self, n_bits):
        """fused_v_update stochastic branch must be unbiased in v space.

        With g=0, v_new == v_old and only rounding noise contributes.  Across
        many fused steps, the mean decoded v must match the pre-quantization v.
        """
        torch.manual_seed(0)
        numel = 4096
        v = torch.rand(numel).exp() * 0.01  # positive, block multiple
        from turboadam.oneq import compress_v_logscale

        cv = compress_v_logscale(
            v, n_bits=n_bits, block_size=128, stochastic_round=False
        )
        grad = torch.zeros(numel)
        n_draws = 400
        # Ground truth: v_out is exact (g=0); the bias test is the mean of
        # decode(recompressed) vs v_out across repeated stochastic draws.
        ni0, ns0, v_out = fused_v_update(
            cv["indices"].clone(), cv["scales"].clone(), grad,
            beta2=0.999, n_bits=n_bits, block_size=128, original_numel=numel,
        )
        redecoded = []
        for _ in range(n_draws):
            ni, ns, _ = fused_v_update(
                cv["indices"].clone(), cv["scales"].clone(), grad,
                beta2=0.999, n_bits=n_bits, block_size=128, original_numel=numel,
            )
            d = dequantize_logscale_nbits(
                ni, ns, n_bits=n_bits, block_size=128, original_numel=numel,
            )
            redecoded.append(d)
        mean_decoded = torch.stack(redecoded).mean(dim=0)
        rel = (mean_decoded - v_out).abs() / (v_out.abs() + 1e-12)
        assert rel.mean().item() < 0.05, (
            f"n_bits={n_bits}: fused_v_update rounding bias {rel.mean().item():.5f} >= 5%"
        )
