"""Low-rank SVD factorization for matrix parameter second moments.

Computes truncated SVD: V ≈ U_r · S_r · W_rᵀ
Used by 1Q for matrix parameters (ndim >= 2 and numel > 10,000).

Storage: (rows*r + r + cols*r) * 2 bytes per matrix.
Reconstruction: small matrix multiply at each step.
"""

import torch


def svd_compress(
    v_matrix: torch.Tensor, rank: int = 8
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compress a 2D tensor via truncated SVD, storing factors in fp16.

    Args:
        v_matrix: 2D fp32 tensor to compress.
        rank: Number of singular values to keep.

    Returns:
        (U_r, S_r, Vh_r) in fp16 where U_r is (rows, r), S_r is (r,),
        Vh_r is (r, cols).
    """
    assert v_matrix.ndim == 2
    r = min(rank, min(v_matrix.shape))
    U, S, V = torch.svd_lowrank(v_matrix, q=r, niter=2)
    # U: (rows, r), S: (r,), V: (cols, r) — note V not Vh
    U_r = U[:, :r].to(torch.float16)
    S_r = S[:r].to(torch.float16)
    Vh_r = V[:, :r].t().to(torch.float16)  # transpose to (r, cols)
    return U_r, S_r, Vh_r


def svd_reconstruct(
    U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor
) -> torch.Tensor:
    """Reconstruct a matrix from truncated SVD factors.

    Args:
        U: (rows, r) factor.
        S: (r,) singular values.
        Vh: (r, cols) factor.

    Returns:
        Reconstructed fp32 matrix of shape (rows, cols).
    """
    # Promote to fp32 for reconstruction precision
    return (U.float() * S.float().unsqueeze(0)) @ Vh.float()
