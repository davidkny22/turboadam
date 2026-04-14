"""Tests for CoState first moment compression.

Covers:
- α computation and gradient-residual decomposition
- Residual energy ratio r per block
- Costate classification (null/phase/amplitude)
- Adaptive threshold computation (P_10, P_90)
- Variable-bit storage and reconstruction fidelity
- Per-step update loop correctness
- Error dissipation over EMA steps
"""
