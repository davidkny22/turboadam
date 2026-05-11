"""Test Triton CoState encode/decode against PyTorch reference."""

import torch
import pytest
import time
from turboadam.costate import (
    encode_blocks,
    decode_blocks,
    decompose,
    compute_block_ratios,
    compute_thresholds,
    classify_blocks,
)
from turboadam.triton_kernels import triton_costate_decode, triton_costate_encode


def _make_test_state(numel=768 * 768, block_size=128, device="cuda"):
    """Create a realistic CoState encoded state for testing."""
    torch.manual_seed(42)
    delta = torch.randn(numel, device=device) * 0.01
    m = torch.randn(numel, device=device) * 0.01
    g = torch.randn(numel, device=device) * 0.01
    alpha, _ = decompose(m, g)
    ratios = compute_block_ratios(delta, m, block_size)
    tau0, tau1 = compute_thresholds(ratios)
    labels = classify_blocks(ratios, tau0, tau1)
    encoded = encode_blocks(delta, labels, block_size)
    return encoded, alpha, g, delta, labels


class TestTritonCostateDecode:
    def test_correctness(self):
        encoded, alpha, g, _, _ = _make_test_state()
        numel = 768 * 768

        ref = decode_blocks(encoded, alpha, g, 128, numel)
        tri = triton_costate_decode(encoded, alpha, g, 128, numel)

        assert torch.allclose(ref, tri, atol=1e-5), (
            f"max diff: {(ref - tri).abs().max().item():.6e}"
        )

    def test_output_shape(self):
        encoded, alpha, g, _, _ = _make_test_state()
        result = triton_costate_decode(encoded, alpha, g, 128, 768 * 768)
        assert result.shape == g.shape

    def test_performance(self):
        encoded, alpha, g, _, _ = _make_test_state()
        numel = 768 * 768
        n = 500

        # Warmup
        for _ in range(5):
            decode_blocks(encoded, alpha, g, 128, numel)
            triton_costate_decode(encoded, alpha, g, 128, numel)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            decode_blocks(encoded, alpha, g, 128, numel)
        torch.cuda.synchronize()
        pt_t = (time.perf_counter() - t0) / n * 1000

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            triton_costate_decode(encoded, alpha, g, 128, numel)
        torch.cuda.synchronize()
        tr_t = (time.perf_counter() - t0) / n * 1000

        print(
            f"\nDecode - PyTorch: {pt_t:.3f} ms, Triton: {tr_t:.3f} ms, Speedup: {pt_t / tr_t:.2f}x"
        )


class TestTritonCostateEncode:
    def test_correctness_block_norms(self):
        """Block norms should match between PyTorch and Triton encode."""
        _, _, _, delta, labels = _make_test_state()
        ref = encode_blocks(delta, labels, 128)
        tri = triton_costate_encode(delta, labels, 128)

        assert torch.allclose(ref["block_norms"], tri["block_norms"], atol=1e-2), (
            f"block_norms max diff: {(ref['block_norms'] - tri['block_norms']).abs().max().item():.6e}"
        )

    def test_self_roundtrip(self):
        """Triton encode → Triton decode should reconstruct m with high fidelity.

        The Triton sign packing uses a different byte layout (int32 atomic_or)
        than PyTorch (_pack_signs), so cross-format comparison isn't meaningful.
        Instead, test that Triton's own encode→decode roundtrip preserves m.
        """
        from turboadam.triton_kernels import triton_costate_decode

        _, alpha, g, delta, labels = _make_test_state()
        numel = 768 * 768

        # Triton encode → Triton decode
        tri_enc = triton_costate_encode(delta, labels, 128)
        m_hat = triton_costate_decode(tri_enc, alpha, g, 128, numel)

        # Reconstruct expected m = alpha * g + delta (no compression)
        alpha_val = alpha.item() if isinstance(alpha, torch.Tensor) else float(alpha)
        m_exact = alpha_val * g.reshape(-1).float() + delta.reshape(-1).float()

        # Cosine similarity should be high (sign-only encoding loses magnitude info
        # for phase blocks, so exact match isn't expected)
        cos_sim = torch.nn.functional.cosine_similarity(
            m_hat.reshape(1, -1),
            m_exact.reshape(1, -1),
        ).item()
        assert cos_sim > 0.60, f"Self-roundtrip cosine sim {cos_sim:.4f} too low"

    def test_output_shapes(self):
        """Triton encode output shapes should match PyTorch encode."""
        _, _, _, delta, labels = _make_test_state()
        ref = encode_blocks(delta, labels, 128)
        tri = triton_costate_encode(delta, labels, 128)

        assert tri["block_norms"].shape == ref["block_norms"].shape
        assert tri["scales"].shape == ref["scales"].shape
        assert tri["labels"].shape == ref["labels"].shape

    def test_performance(self):
        _, _, _, delta, labels = _make_test_state()
        n = 500

        for _ in range(5):
            encode_blocks(delta, labels, 128)
            triton_costate_encode(delta, labels, 128)

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            encode_blocks(delta, labels, 128)
        torch.cuda.synchronize()
        pt_t = (time.perf_counter() - t0) / n * 1000

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            triton_costate_encode(delta, labels, 128)
        torch.cuda.synchronize()
        tr_t = (time.perf_counter() - t0) / n * 1000

        print(
            f"\nEncode - PyTorch: {pt_t:.3f} ms, Triton: {tr_t:.3f} ms, Speedup: {pt_t / tr_t:.2f}x"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
