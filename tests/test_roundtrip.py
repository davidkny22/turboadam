"""Roundtrip fidelity tests — compression then reconstruction.

Covers:
- 2-bit log-scale quantize → dequantize error bounds
- SVD compress → reconstruct relative L2 error
- Costate encode → decode cosine similarity by tier
- End-to-end: 100 optimizer steps, loss convergence vs. standard Adam
"""
