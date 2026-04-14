"""CoState — first moment (m) compression.

Gradient-residual decomposition: m = α·g + δ
  - α = (m·g) / (g·g)  — scalar per layer
  - δ = m - α·g        — residual orthogonal to current gradient

Residual δ is partitioned into 128-element blocks and classified:
  - Null costate    (r < τ₀): store 1 bit in bitmap
  - Phase costate   (τ₀ ≤ r < τ₁): store 1-bit sign per element
  - Amplitude costate (r ≥ τ₁): store 1-bit sign + fp16 block scale

Adaptive thresholds: τ₀ = P_10(r), τ₁ = P_90(r) per layer per step.
No warmup required — EMA error-washing handles cold-start.
"""
