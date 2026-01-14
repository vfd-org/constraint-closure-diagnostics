"""
Fourier-Based Spectrum Computation for Block-Circulant Kernels.

For translation-invariant operators on Z/CZ ⊗ C^internal_dim, the kernel
is block-circulant. FFT diagonalizes the cell structure, reducing the
problem to C independent internal_dim × internal_dim diagonalizations.

For K_can = I_internal ⊗ L_cell, each block is scalar (λ_j * I), so this
reduces to the analytic case. But this backend handles more general
block-circulant kernels where K_m are non-trivial internal blocks.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List, Optional, Dict, Any
from scipy.linalg import eigh


def extract_circulant_blocks(
    K_sparse,
    cell_count: int,
    internal_dim: int
) -> List[NDArray]:
    """
    Extract circulant blocks K_0, K_1, ..., K_{C-1} from kernel matrix.

    For a block-circulant matrix, K[n, m] depends only on (n-m) mod C.
    K_d = K[0, d] block (internal_dim × internal_dim).

    Args:
        K_sparse: Sparse kernel matrix
        cell_count: Number of cells
        internal_dim: Internal space dimension

    Returns:
        List of C blocks, each internal_dim × internal_dim
    """
    blocks = []

    for d in range(cell_count):
        # Extract K[0, d] block
        block = np.zeros((internal_dim, internal_dim), dtype=complex)

        row_start = 0
        row_end = internal_dim
        col_start = d * internal_dim
        col_end = (d + 1) * internal_dim

        # Get the block from sparse matrix
        block_dense = K_sparse[row_start:row_end, col_start:col_end].toarray()
        block[:, :] = block_dense

        blocks.append(block)

    return blocks


def fft_diagonalize_circulant_blocks(
    blocks: List[NDArray]
) -> List[NDArray]:
    """
    Apply block FFT to diagonalize circulant structure.

    For block-circulant matrix with blocks K_0, ..., K_{C-1}:
    The Fourier-transformed blocks are:
        K̂_j = Σ_{d=0}^{C-1} K_d * ω^{jd}

    where ω = exp(2πi/C).

    Args:
        blocks: List of C blocks (each internal_dim × internal_dim)

    Returns:
        List of C Fourier-space blocks K̂_j
    """
    C = len(blocks)
    internal_dim = blocks[0].shape[0]
    omega = np.exp(2j * np.pi / C)

    fourier_blocks = []

    for j in range(C):
        # K̂_j = Σ_d K_d * ω^{jd}
        K_hat_j = np.zeros((internal_dim, internal_dim), dtype=complex)

        for d in range(C):
            phase = omega ** (j * d)
            K_hat_j += blocks[d] * phase

        fourier_blocks.append(K_hat_j)

    return fourier_blocks


def fourier_cells_eigenvalues(
    K_sparse,
    cell_count: int,
    internal_dim: int = 600
) -> Tuple[NDArray, Dict[str, Any]]:
    """
    Compute spectrum via block-circulant FFT diagonalization.

    Steps:
    1. Extract circulant blocks K_d from kernel
    2. FFT to get Fourier blocks K̂_j
    3. Diagonalize each internal_dim × internal_dim block
    4. Collect all eigenvalues

    Args:
        K_sparse: Sparse kernel matrix
        cell_count: Number of cells
        internal_dim: Internal space dimension

    Returns:
        Tuple of (eigenvalues, info_dict)
    """
    # Extract circulant blocks
    blocks = extract_circulant_blocks(K_sparse, cell_count, internal_dim)

    # FFT diagonalize
    fourier_blocks = fft_diagonalize_circulant_blocks(blocks)

    # Diagonalize each Fourier block
    all_eigenvalues = []

    for j, K_hat_j in enumerate(fourier_blocks):
        # Each K̂_j is Hermitian, use eigh
        # Make Hermitian (numerical symmetrization)
        K_hat_j_hermitian = (K_hat_j + K_hat_j.conj().T) / 2

        eigenvalues_j = np.linalg.eigvalsh(K_hat_j_hermitian)
        all_eigenvalues.extend(eigenvalues_j)

    all_eigenvalues = np.array(all_eigenvalues)
    all_eigenvalues = np.sort(all_eigenvalues)

    info = {
        "backend": "fourier_cells",
        "cell_count": cell_count,
        "internal_dim": internal_dim,
        "total_eigenvalues": len(all_eigenvalues),
        "num_blocks_diagonalized": cell_count,
        "min_eigenvalue": float(all_eigenvalues[0]),
        "max_eigenvalue": float(all_eigenvalues[-1]),
    }

    return all_eigenvalues, info


def fourier_cells_kcan_fast(
    cell_count: int,
    propagation_range: int = 1,
    internal_dim: int = 600
) -> Tuple[NDArray, Dict[str, Any]]:
    """
    Fast Fourier computation for K_can = I_internal ⊗ L_cell.

    For this special case, K̂_j = λ_cell(θ_j) * I_internal.
    Each block is scalar, so no internal diagonalization needed.

    This is equivalent to analytic_kcan but structured for comparison.

    Args:
        cell_count: Number of cells
        propagation_range: Coupling range R
        internal_dim: Internal dimension

    Returns:
        Tuple of (eigenvalues, info_dict)
    """
    from .analytic import circulant_laplacian_eigenvalue

    # Compute cell eigenvalues at Fourier modes
    cell_eigenvalues = np.zeros(cell_count)

    for j in range(cell_count):
        theta_j = 2.0 * np.pi * j / cell_count
        cell_eigenvalues[j] = circulant_laplacian_eigenvalue(theta_j, propagation_range, cell_count)

    # Each has multiplicity internal_dim
    all_eigenvalues = np.repeat(cell_eigenvalues, internal_dim)
    all_eigenvalues = np.sort(all_eigenvalues)

    info = {
        "backend": "fourier_cells_kcan_fast",
        "cell_count": cell_count,
        "propagation_range": propagation_range,
        "internal_dim": internal_dim,
        "total_eigenvalues": len(all_eigenvalues),
        "unique_cell_eigenvalues": cell_count,
        "min_eigenvalue": float(all_eigenvalues[0]),
        "max_eigenvalue": float(all_eigenvalues[-1]),
    }

    return all_eigenvalues, info


def fourier_cells_with_eigenvectors(
    K_sparse,
    cell_count: int,
    internal_dim: int = 600,
    k: Optional[int] = None
) -> Tuple[NDArray, NDArray, Dict[str, Any]]:
    """
    Compute spectrum with eigenvectors via block-circulant FFT.

    Args:
        K_sparse: Sparse kernel matrix
        cell_count: Number of cells
        internal_dim: Internal space dimension
        k: Number of eigenvalues to return (None = all)

    Returns:
        Tuple of (eigenvalues, eigenvectors, info_dict)
    """
    # Extract and FFT blocks
    blocks = extract_circulant_blocks(K_sparse, cell_count, internal_dim)
    fourier_blocks = fft_diagonalize_circulant_blocks(blocks)

    total_dim = cell_count * internal_dim
    all_eigenvalues = []
    all_indices = []  # (fourier_mode_j, internal_idx)

    # Diagonalize each Fourier block
    block_eigenvectors = []

    for j, K_hat_j in enumerate(fourier_blocks):
        K_hat_j_hermitian = (K_hat_j + K_hat_j.conj().T) / 2
        eigenvalues_j, eigenvectors_j = eigh(K_hat_j_hermitian)

        block_eigenvectors.append(eigenvectors_j)

        for idx, ev in enumerate(eigenvalues_j):
            all_eigenvalues.append(ev)
            all_indices.append((j, idx))

    # Sort by eigenvalue
    all_eigenvalues = np.array(all_eigenvalues)
    sort_idx = np.argsort(all_eigenvalues)
    all_eigenvalues = all_eigenvalues[sort_idx]
    all_indices = [all_indices[i] for i in sort_idx]

    # Select k eigenvalues if specified
    if k is not None and k < len(all_eigenvalues):
        all_eigenvalues = all_eigenvalues[:k]
        all_indices = all_indices[:k]

    # Construct eigenvectors in original basis
    # Full eigenvector: Fourier_cell ⊗ block_eigenvector
    omega = np.exp(2j * np.pi / cell_count)
    n_return = len(all_eigenvalues)
    eigenvectors = np.zeros((total_dim, n_return), dtype=complex)

    for i, (j, internal_idx) in enumerate(all_indices):
        # Fourier vector for mode j: e_j[n] = ω^{jn} / sqrt(C)
        fourier_vec = np.array([omega ** (j * n) for n in range(cell_count)]) / np.sqrt(cell_count)

        # Internal eigenvector from block j
        internal_vec = block_eigenvectors[j][:, internal_idx]

        # Kronecker product
        eigenvector = np.kron(fourier_vec, internal_vec)
        eigenvectors[:, i] = eigenvector

    info = {
        "backend": "fourier_cells_with_eigenvectors",
        "cell_count": cell_count,
        "internal_dim": internal_dim,
        "eigenvalues_returned": n_return,
        "total_eigenvalues": cell_count * internal_dim,
    }

    return all_eigenvalues, eigenvectors, info
