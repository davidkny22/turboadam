"""Low-rank SVD factorization for matrix parameter second moments.

Computes truncated SVD: V ≈ U_r · S_r · W_rᵀ
Used by 1Q for matrix parameters (ndim >= 2 and numel > 10,000).

Storage: (rows*r + r + cols*r) * 2 bytes per matrix.
Reconstruction: small matrix multiply at each step.
"""
