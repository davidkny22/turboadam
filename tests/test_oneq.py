"""Tests for 1Q second moment compression.

Covers:
- 2-bit log-scale quantization roundtrip fidelity per block
- Warmup detection (relative change threshold)
- Low-rank SVD compression and reconstruction error
- Freeze-refresh cycle correctness
- Matrix vs. non-matrix routing
"""
