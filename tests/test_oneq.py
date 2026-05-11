"""Tests for 1Q second moment compression (n-bit log-scale path)."""

import torch

from turboadam.oneq import compress_v_logscale, decompress_v


def _make_v(shape):
    """Positive v tensor (second moment)."""
    return torch.rand(*shape).abs() + 1e-4


class TestCompressVLogscale:
    """compress_v_logscale / decompress_v roundtrip."""

    def test_roundtrip_1d(self):
        v = _make_v((512,))
        compressed = compress_v_logscale(v, n_bits=4, block_size=128)
        v_hat = decompress_v(compressed)
        assert v_hat.shape == v.shape
        assert v_hat.dtype == torch.float32

    def test_roundtrip_2d(self):
        v = _make_v((64, 64))
        compressed = compress_v_logscale(v, n_bits=4, block_size=128)
        v_hat = decompress_v(compressed)
        assert v_hat.shape == v.shape

    def test_roundtrip_various_bits(self):
        for bits in [2, 3, 4, 6, 8]:
            v = _make_v((256,))
            compressed = compress_v_logscale(v, n_bits=bits, block_size=128)
            v_hat = decompress_v(compressed)
            assert v_hat.shape == v.shape

    def test_repeated_decompress_identical(self):
        v = _make_v((256,))
        compressed = compress_v_logscale(v, n_bits=4, block_size=128)
        v_hat_1 = decompress_v(compressed)
        v_hat_2 = decompress_v(compressed)
        assert torch.equal(v_hat_1, v_hat_2)

    def test_decompress_does_not_mutate_dict(self):
        v = _make_v((256,))
        compressed = compress_v_logscale(v, n_bits=4, block_size=128)
        keys_before = set(compressed.keys())
        _ = decompress_v(compressed)
        keys_after = set(compressed.keys())
        assert keys_before == keys_after

    def test_stochastic_rounding_different_each_call(self):
        v = _make_v((256,))
        compressed_1 = compress_v_logscale(
            v, n_bits=4, block_size=128, stochastic_round=True
        )
        compressed_2 = compress_v_logscale(
            v, n_bits=4, block_size=128, stochastic_round=True
        )
        # Stochastic rounding should produce different indices
        assert not torch.equal(compressed_1["indices"], compressed_2["indices"])

    def test_detinistic_rounding_same_each_call(self):
        v = _make_v((256,))
        compressed_1 = compress_v_logscale(
            v, n_bits=4, block_size=128, stochastic_round=False
        )
        compressed_2 = compress_v_logscale(
            v, n_bits=4, block_size=128, stochastic_round=False
        )
        assert torch.equal(compressed_1["indices"], compressed_2["indices"])
