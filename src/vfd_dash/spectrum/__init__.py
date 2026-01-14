"""
VFD Spectrum Computation Module.

Provides multiple backends for computing the spectrum of K_can:
- analytic_kcan: Closed-form eigenvalues using Kronecker structure
- fourier_cells: Block-circulant FFT diagonalization
- sparse_fallback: LinearOperator + eigsh for general kernels

The canonical kernel has Kronecker structure:
    K_can = I_internal ⊗ L_cell

where L_cell is the circulant Laplacian on Z/CZ with coupling range R.
This enables O(C) eigenvalue computation instead of O((C*internal_dim)^3).
"""

from .analytic import analytic_kcan_eigenvalues, analytic_kcan_full, analytic_kcan_cell_eigenvalues
from .fourier import fourier_cells_eigenvalues, fourier_cells_kcan_fast
from .sparse import sparse_eigsh_eigenvalues, sparse_eigsh_with_eigenvectors
from .backend import SpectralBackend, compute_spectrum, auto_select_backend, SpectralResult, clear_spectrum_cache
from .torsion_sectors import (
    get_torsion_projection_indices,
    torsion_sector_spectrum_kcan,
    torsion_fingerprint,
)
from .profiling import (
    estimate_spectrum_memory,
    check_feasibility,
    profile_computation,
    benchmark_all_backends,
    ComputationLimits,
    ProfileResult,
)


__all__ = [
    # Backend selection
    "SpectralBackend",
    "SpectralResult",
    "compute_spectrum",
    "auto_select_backend",
    "clear_spectrum_cache",
    # Analytic backend
    "analytic_kcan_eigenvalues",
    "analytic_kcan_full",
    "analytic_kcan_cell_eigenvalues",
    # Fourier backend
    "fourier_cells_eigenvalues",
    "fourier_cells_kcan_fast",
    # Sparse backend
    "sparse_eigsh_eigenvalues",
    "sparse_eigsh_with_eigenvectors",
    # Torsion sectors
    "get_torsion_projection_indices",
    "torsion_sector_spectrum_kcan",
    "torsion_fingerprint",
    # Profiling
    "estimate_spectrum_memory",
    "check_feasibility",
    "profile_computation",
    "benchmark_all_backends",
    "ComputationLimits",
    "ProfileResult",
]
