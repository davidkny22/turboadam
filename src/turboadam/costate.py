"""CoState — first moment (m) compression.

Gradient-residual decomposition: m = α·g + δ
  - α = (m·g) / (g·g)  — scalar per parameter tensor
  - δ = m - α·g        — residual orthogonal to current gradient

Residual δ is partitioned into 128-element blocks and classified:
  - Null costate    (r < τ₀): store 1 bit in bitmap
  - Phase costate   (τ₀ ≤ r < τ₁): store 1-bit sign per element
  - Amplitude costate (r ≥ τ₁): store 1-bit sign + fp16 block scale

Adaptive thresholds: τ₀ = P_10(r), τ₁ = P_90(r) per parameter tensor per step.
No warmup required — EMA error-washing handles cold-start.
"""

import torch


def decompose(m: torch.Tensor, g: torch.Tensor) -> tuple[float, torch.Tensor]:
    """Decompose momentum into gradient-aligned component and residual.

    m = α·g + δ  where α = (m·g) / (g·g)

    Args:
        m: First moment tensor (any shape, will be treated as flat).
        g: Gradient tensor (same shape as m).

    Returns:
        (alpha, delta) where alpha is a float scalar and delta has
        the same shape as m.
    """
    m_flat = m.reshape(-1)
    g_flat = g.reshape(-1)
    g_dot_g = g_flat.dot(g_flat).item()
    if g_dot_g == 0.0:
        return 0.0, m.clone()
    alpha = m_flat.dot(g_flat).item() / g_dot_g
    delta = m - alpha * g
    return alpha, delta
