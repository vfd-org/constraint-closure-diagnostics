"""
Sparse Fallback Backend for Spectrum Computation.

Uses scipy.sparse.linalg.eigsh with LinearOperator for general kernels
that don't have special structure. This is the most general but slowest backend.

Should only be used when:
- Kernel is not K_can (non-trivial internal coupling)
- Kernel is not block-circulant
- Need eigenvectors and analytic formula not available
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, Any, Optional
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigsh, LinearOperator
import warnings


def sparse_eigsh_eigenvalues(
    K_matrix,
    k: int = 100,
    which: str = "SM",
    tol: float = 1e-10,
    maxiter: int = 10000,
    sigma: Optional[float] = None
) -> Tuple[NDArray, Dict[str, Any]]:
    """
    Compute k eigenvalues using sparse eigsh.

    Args:
        K_matrix: Sparse or dense kernel matrix
        k: Number of eigenvalues to compute
        which: "SM" (smallest magnitude), "LM" (largest), "SA" (smallest algebraic)
        tol: Convergence tolerance
        maxiter: Maximum iterations
        sigma: Shift for shift-invert mode (None = no shift)

    Returns:
        Tuple of (eigenvalues, info_dict)
    """
    if not issparse(K_matrix):
        K_matrix = csr_matrix(K_matrix)

    dim = K_matrix.shape[0]
    k = min(k, dim - 2)  # eigsh requires k < dim - 1

    try:
        if sigma is not None:
            eigenvalues, _ = eigsh(
                K_matrix, k=k, which="LM", sigma=sigma,
                tol=tol, maxiter=maxiter
            )
        else:
            eigenvalues, _ = eigsh(
                K_matrix, k=k, which=which,
                tol=tol, maxiter=maxiter
            )

        eigenvalues = np.sort(eigenvalues)
        converged = True

    except Exception as e:
        warnings.warn(f"eigsh failed: {e}. Falling back to dense solver.")
        # Fallback to dense
        eigenvalues = np.sort(np.linalg.eigvalsh(K_matrix.toarray()))[:k]
        converged = False

    info = {
        "backend": "sparse_eigsh",
        "k_requested": k,
        "k_returned": len(eigenvalues),
        "matrix_dim": dim,
        "which": which,
        "converged": converged,
        "min_eigenvalue": float(eigenvalues[0]) if len(eigenvalues) > 0 else None,
        "max_eigenvalue": float(eigenvalues[-1]) if len(eigenvalues) > 0 else None,
    }

    return eigenvalues, info


def sparse_eigsh_with_eigenvectors(
    K_matrix,
    k: int = 100,
    which: str = "SM",
    tol: float = 1e-10,
    maxiter: int = 10000
) -> Tuple[NDArray, NDArray, Dict[str, Any]]:
    """
    Compute k eigenvalues and eigenvectors using sparse eigsh.

    Args:
        K_matrix: Sparse or dense kernel matrix
        k: Number of eigenvalues to compute
        which: "SM", "LM", "SA"
        tol: Convergence tolerance
        maxiter: Maximum iterations

    Returns:
        Tuple of (eigenvalues, eigenvectors, info_dict)
    """
    if not issparse(K_matrix):
        K_matrix = csr_matrix(K_matrix)

    dim = K_matrix.shape[0]
    k = min(k, dim - 2)

    try:
        eigenvalues, eigenvectors = eigsh(
            K_matrix, k=k, which=which,
            tol=tol, maxiter=maxiter
        )

        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        converged = True

    except Exception as e:
        warnings.warn(f"eigsh failed: {e}. Falling back to dense solver.")
        dense = K_matrix.toarray()
        eigenvalues, eigenvectors = np.linalg.eigh(dense)
        eigenvalues = eigenvalues[:k]
        eigenvectors = eigenvectors[:, :k]
        converged = False

    info = {
        "backend": "sparse_eigsh",
        "k_requested": k,
        "k_returned": len(eigenvalues),
        "matrix_dim": dim,
        "converged": converged,
    }

    return eigenvalues, eigenvectors, info


class KernelLinearOperator(LinearOperator):
    """
    LinearOperator wrapper for kernel application.

    Enables eigsh to use matrix-vector products without storing full matrix.
    """

    def __init__(self, kernel, total_dim: int):
        """
        Initialize LinearOperator for kernel.

        Args:
            kernel: Kernel object with .apply(state) method
            total_dim: Total dimension of the space
        """
        self.kernel = kernel
        super().__init__(dtype=complex, shape=(total_dim, total_dim))

    def _matvec(self, x):
        """Apply kernel to vector."""
        return self.kernel.apply(x)

    def _rmatvec(self, x):
        """Apply adjoint (same for self-adjoint kernel)."""
        return self.kernel.apply(x)


def sparse_eigsh_from_operator(
    kernel,
    total_dim: int,
    k: int = 100,
    which: str = "SM",
    tol: float = 1e-10,
    maxiter: int = 10000
) -> Tuple[NDArray, Dict[str, Any]]:
    """
    Compute eigenvalues using LinearOperator interface.

    This avoids constructing the full matrix, useful for very large systems.

    Args:
        kernel: Kernel object with .apply(state) method
        total_dim: Total dimension
        k: Number of eigenvalues
        which: "SM", "LM", "SA"
        tol: Tolerance
        maxiter: Max iterations

    Returns:
        Tuple of (eigenvalues, info_dict)
    """
    op = KernelLinearOperator(kernel, total_dim)
    k = min(k, total_dim - 2)

    try:
        eigenvalues, _ = eigsh(op, k=k, which=which, tol=tol, maxiter=maxiter)
        eigenvalues = np.sort(eigenvalues)
        converged = True
    except Exception as e:
        warnings.warn(f"eigsh with LinearOperator failed: {e}")
        eigenvalues = np.array([])
        converged = False

    info = {
        "backend": "sparse_eigsh_operator",
        "k_requested": k,
        "k_returned": len(eigenvalues),
        "total_dim": total_dim,
        "converged": converged,
    }

    return eigenvalues, info


def estimate_memory_gb(cell_count: int, internal_dim: int, sparse: bool = True) -> float:
    """
    Estimate memory usage for eigenvalue computation.

    Args:
        cell_count: Number of cells
        internal_dim: Internal dimension
        sparse: Whether using sparse storage

    Returns:
        Estimated memory in GB
    """
    total_dim = cell_count * internal_dim

    if sparse:
        # Sparse: ~10 * nnz * 8 bytes (data + indices)
        # For Laplacian: nnz ~ 2 * propagation_range * total_dim
        nnz_estimate = 6 * total_dim  # Assume R=3
        mem_bytes = 10 * nnz_estimate * 8
    else:
        # Dense: total_dim^2 * 16 bytes (complex)
        mem_bytes = total_dim ** 2 * 16

    return mem_bytes / (1024 ** 3)


def check_feasibility(
    cell_count: int,
    internal_dim: int,
    max_memory_gb: float = 8.0,
    max_dim_dense: int = 5000
) -> Dict[str, Any]:
    """
    Check if computation is feasible with given resources.

    Args:
        cell_count: Number of cells
        internal_dim: Internal dimension
        max_memory_gb: Maximum allowed memory
        max_dim_dense: Maximum dimension for dense methods

    Returns:
        Dictionary with feasibility information
    """
    total_dim = cell_count * internal_dim

    sparse_mem = estimate_memory_gb(cell_count, internal_dim, sparse=True)
    dense_mem = estimate_memory_gb(cell_count, internal_dim, sparse=False)

    return {
        "total_dim": total_dim,
        "sparse_memory_gb": sparse_mem,
        "dense_memory_gb": dense_mem,
        "sparse_feasible": sparse_mem < max_memory_gb,
        "dense_feasible": dense_mem < max_memory_gb and total_dim < max_dim_dense,
        "analytic_feasible": True,  # Always feasible for K_can
        "recommended_backend": (
            "analytic_kcan" if True else
            "sparse_fallback" if sparse_mem < max_memory_gb else
            "INFEASIBLE"
        ),
    }
