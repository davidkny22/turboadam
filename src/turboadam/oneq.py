"""1Q — second moment (v) compression.

Two paths based on parameter shape:
  - Matrix path (ndim >= 2 and numel > 10,000): low-rank SVD via svd.py
  - Non-matrix path: 2-bit log-scale quantization via quantize.py

Warmup detection monitors per-step relative change in v.
Freeze-refresh cycle runs every refresh_interval steps.
"""
