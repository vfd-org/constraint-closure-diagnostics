"""
Analytic Spectrum Computation for K_can.

Exploits the Kronecker structure: K_can = I_internal ⊗ L_cell

The circulant Laplacian L_cell on Z/CZ with coupling range R has
closed-form eigenvalues:

    λ_cell(θ_j) = 2R - 2 * Σ_{d=1}^R cos(d * θ_j)

where θ_j = 2π * j / C for j = 0, ..., C-1.

Each cell eigenvalue has multiplicity internal_dim, giving the full spectrum
without matrix construction or diagonalization.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
from functools import lru_cache


def circulant_laplacian_eigenvalue(theta: float, propagation_range: int, cell_count: int = None) -> float:
    """
    Compute single eigenvalue of circulant Laplacian.

    L_cell has eigenvalue λ(θ) = degree - Σ_{d} cos(d*θ) * (count of d)

    For standard case (R < C/2):
        λ(θ) = 2R - 2 * Σ_{d=1}^R cos(d*θ)

    For edge case (R >= C/2), neighbors wrap around and the formula adjusts
    to account for the actual graph structure.

    Args:
        theta: Fourier angle θ = 2π*j/C
        propagation_range: R, the coupling range
        cell_count: Number of cells (needed for edge case R >= C/2)

    Returns:
        Eigenvalue λ(θ)
    """
    R = propagation_range

    # Standard case: no wraparound overlap
    if cell_count is None or 2 * R < cell_count:
        # Standard formula: each d in 1..R contributes ±d neighbors
        result = 2.0 * R
        for d in range(1, R + 1):
            result -= 2.0 * np.cos(d * theta)
        return result

    # Edge case: R >= C/2, need to handle wraparound
    # Count distinct non-zero distances
    C = cell_count
    degree = 0
    cosine_sum = 0.0

    for d in range(1, R + 1):
        # d and C-d are equivalent distances
        # Only count d if d <= C-d (i.e., d <= C/2)
        if d < C - d:
            # Both +d and -d are distinct neighbors
            degree += 2
            cosine_sum += 2.0 * np.cos(d * theta)
        elif d == C - d:
            # d and C-d are the same (only happens when C is even and d = C/2)
            degree += 1
            cosine_sum += np.cos(d * theta)
        # If d > C - d, we've already counted this distance as C-d

    return degree - cosine_sum


def analytic_kcan_eigenvalues(
    cell_count: int,
    propagation_range: int = 1,
    internal_dim: int = 600
) -> NDArray:
    """
    Compute full spectrum of K_can analytically.

    K_can = I_internal ⊗ L_cell means each L_cell eigenvalue appears
    with multiplicity internal_dim.

    Args:
        cell_count: Number of cells C
        propagation_range: Coupling range R (default 1)
        internal_dim: Internal space dimension (default 600)

    Returns:
        Array of eigenvalues (length C * internal_dim), sorted ascending
    """
    # Compute cell eigenvalues at Fourier modes
    cell_eigenvalues = np.zeros(cell_count)

    for j in range(cell_count):
        theta_j = 2.0 * np.pi * j / cell_count
        cell_eigenvalues[j] = circulant_laplacian_eigenvalue(theta_j, propagation_range, cell_count)

    # Each cell eigenvalue has multiplicity internal_dim
    full_spectrum = np.repeat(cell_eigenvalues, internal_dim)

    # Sort for consistent ordering
    return np.sort(full_spectrum)


def analytic_kcan_cell_eigenvalues(
    cell_count: int,
    propagation_range: int = 1
) -> NDArray:
    """
    Compute just the cell eigenvalues (without multiplicity).

    Useful for bridge projection where we want unique eigenvalues.

    Args:
        cell_count: Number of cells C
        propagation_range: Coupling range R

    Returns:
        Array of C unique cell eigenvalues, sorted ascending
    """
    cell_eigenvalues = np.zeros(cell_count)

    for j in range(cell_count):
        theta_j = 2.0 * np.pi * j / cell_count
        cell_eigenvalues[j] = circulant_laplacian_eigenvalue(theta_j, propagation_range, cell_count)

    return np.sort(cell_eigenvalues)


def analytic_kcan_full(
    cell_count: int,
    propagation_range: int = 1,
    internal_dim: int = 600
) -> Tuple[NDArray, NDArray, dict]:
    """
    Full analytic spectrum computation with metadata.

    Returns eigenvalues, multiplicities, and diagnostic info.

    Args:
        cell_count: Number of cells C
        propagation_range: Coupling range R
        internal_dim: Internal dimension

    Returns:
        Tuple of:
        - unique_eigenvalues: Array of C unique eigenvalues
        - multiplicities: Array of multiplicities (all = internal_dim)
        - info: Dictionary with diagnostic information
    """
    unique_eigenvalues = analytic_kcan_cell_eigenvalues(cell_count, propagation_range)
    multiplicities = np.full(cell_count, internal_dim)

    info = {
        "backend": "analytic_kcan",
        "cell_count": cell_count,
        "propagation_range": propagation_range,
        "internal_dim": internal_dim,
        "total_eigenvalues": cell_count * internal_dim,
        "unique_eigenvalues": cell_count,
        "min_eigenvalue": float(unique_eigenvalues[0]),
        "max_eigenvalue": float(unique_eigenvalues[-1]),
        "spectral_gap": float(unique_eigenvalues[1] - unique_eigenvalues[0]) if cell_count > 1 else 0.0,
        # Verify zero mode: λ(0) = 0 for Laplacian
        "zero_mode_value": float(circulant_laplacian_eigenvalue(0.0, propagation_range, cell_count)),
    }

    return unique_eigenvalues, multiplicities, info


def analytic_eigenvector_cell(j: int, cell_count: int) -> NDArray:
    """
    Compute j-th Fourier eigenvector for the cell space.

    The eigenvector is e_j[n] = exp(i * θ_j * n) / sqrt(C)

    Args:
        j: Fourier mode index (0 to C-1)
        cell_count: Number of cells

    Returns:
        Complex eigenvector of length cell_count
    """
    theta_j = 2.0 * np.pi * j / cell_count
    n = np.arange(cell_count)
    eigenvector = np.exp(1j * theta_j * n) / np.sqrt(cell_count)
    return eigenvector


@lru_cache(maxsize=128)
def cached_cell_eigenvalues(cell_count: int, propagation_range: int) -> Tuple[float, ...]:
    """
    Cached computation of cell eigenvalues.

    Returns tuple (hashable) for caching purposes.
    """
    eigenvalues = analytic_kcan_cell_eigenvalues(cell_count, propagation_range)
    return tuple(eigenvalues)


def analytic_kcan_trace(cell_count: int, propagation_range: int, internal_dim: int = 600) -> float:
    """
    Compute trace(K_can) analytically.

    For K_can = I_internal ⊗ L_cell:
        trace(K_can) = internal_dim * trace(L_cell)

    For L_cell (circulant Laplacian with range R):
        trace(L_cell) = C * (diagonal entry) = C * 2R

    So: trace(K_can) = internal_dim * C * 2R

    Args:
        cell_count: Number of cells C
        propagation_range: Coupling range R
        internal_dim: Internal dimension

    Returns:
        trace(K_can)
    """
    # Each diagonal entry of L_cell is 2*R (degree in circulant graph)
    return float(internal_dim * cell_count * 2 * propagation_range)


def verify_analytic_formula(cell_count: int, propagation_range: int, tol: float = 1e-10) -> Tuple[bool, float]:
    """
    Verify the analytic formula against direct matrix computation.

    Constructs a small L_cell matrix and compares eigenvalues.
    Only practical for small cell_count.

    Args:
        cell_count: Number of cells (should be small for verification)
        propagation_range: Coupling range
        tol: Tolerance for comparison

    Returns:
        Tuple of (passes, max_error)
    """
    if cell_count > 1000:
        raise ValueError("Verification only practical for cell_count <= 1000")

    # Build L_cell directly using the proper Laplacian construction
    # that handles wraparound correctly
    L_cell = np.zeros((cell_count, cell_count))

    # Off-diagonal: -1 for neighbors within range, handling wraparound
    for n in range(cell_count):
        for d in range(1, propagation_range + 1):
            n_fwd = (n + d) % cell_count
            n_bwd = (n - d) % cell_count
            # Using = -1.0 means if n_fwd == n_bwd (wraparound), it's only set once
            L_cell[n, n_fwd] = -1.0
            L_cell[n, n_bwd] = -1.0

    # Diagonal: negative row sum (Laplacian property)
    for n in range(cell_count):
        L_cell[n, n] = -np.sum(L_cell[n, :])

    # Compute eigenvalues directly
    direct_eigenvalues = np.sort(np.linalg.eigvalsh(L_cell))

    # Compute analytically
    analytic_eigenvalues = analytic_kcan_cell_eigenvalues(cell_count, propagation_range)

    # Compare
    max_error = np.max(np.abs(direct_eigenvalues - analytic_eigenvalues))

    return max_error < tol, max_error
