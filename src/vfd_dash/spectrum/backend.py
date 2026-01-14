"""
Spectral Backend Selector and Unified Interface.

Provides a unified interface for spectrum computation with automatic
backend selection, caching, and profiling.

Backend Selection Logic:
1. analytic_kcan: For K_can = I_internal ⊗ L_cell (fastest, O(C))
2. fourier_cells: For block-circulant kernels (O(C * d³) where d = internal_dim)
3. sparse_fallback: For general kernels (slowest, use eigsh)

The selector auto-detects kernel type and chooses optimal backend.
"""

import time
import hashlib
import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, Any, Optional, Union
from enum import Enum
from dataclasses import dataclass, field
from functools import lru_cache


class SpectralBackend(Enum):
    """Available spectral computation backends."""
    ANALYTIC_KCAN = "analytic_kcan"
    FOURIER_CELLS = "fourier_cells"
    SPARSE_FALLBACK = "sparse_fallback"
    AUTO = "auto"


@dataclass
class SpectralResult:
    """Result of spectrum computation."""
    eigenvalues: NDArray
    eigenvectors: Optional[NDArray] = None
    multiplicities: Optional[NDArray] = None
    backend_used: str = ""
    computation_time_ms: float = 0.0
    cache_hit: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


# Simple in-memory cache for eigenvalues
_eigenvalue_cache: Dict[str, SpectralResult] = {}


def _cache_key(
    cell_count: int,
    internal_dim: int,
    propagation_range: int,
    backend: str
) -> str:
    """Generate cache key for eigenvalue computation."""
    key_str = f"{cell_count}_{internal_dim}_{propagation_range}_{backend}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]


def clear_spectrum_cache():
    """Clear the eigenvalue cache."""
    global _eigenvalue_cache
    _eigenvalue_cache.clear()


def auto_select_backend(
    kernel_type: str,
    cell_count: int,
    internal_dim: int,
    needs_eigenvectors: bool = False
) -> SpectralBackend:
    """
    Automatically select the best backend for given parameters.

    Args:
        kernel_type: "K_can" or "custom"
        cell_count: Number of cells
        internal_dim: Internal dimension
        needs_eigenvectors: Whether eigenvectors are needed

    Returns:
        Best backend for the given situation
    """
    total_dim = cell_count * internal_dim

    if kernel_type == "K_can":
        # K_can has Kronecker structure, analytic is always best
        return SpectralBackend.ANALYTIC_KCAN

    # For custom kernels, check if block-circulant
    # (In practice, we'd need to verify this property)
    if total_dim > 10000:
        # Large systems: try fourier_cells if block-circulant
        return SpectralBackend.FOURIER_CELLS
    else:
        # Small enough for sparse eigsh
        return SpectralBackend.SPARSE_FALLBACK


def compute_spectrum(
    cell_count: int,
    internal_dim: int = 600,
    propagation_range: int = 1,
    backend: SpectralBackend = SpectralBackend.AUTO,
    k: Optional[int] = None,
    kernel_matrix=None,
    use_cache: bool = True,
    include_eigenvectors: bool = False
) -> SpectralResult:
    """
    Compute spectrum using the specified or auto-selected backend.

    For K_can (default), uses closed-form analytic eigenvalues.
    For custom kernels, provide kernel_matrix.

    Args:
        cell_count: Number of cells C
        internal_dim: Internal dimension (default 600)
        propagation_range: Coupling range R (default 1)
        backend: Which backend to use (default AUTO)
        k: Number of eigenvalues (None = all for analytic, 100 for sparse)
        kernel_matrix: Custom kernel matrix (None = use K_can)
        use_cache: Whether to use cached results
        include_eigenvectors: Whether to compute eigenvectors

    Returns:
        SpectralResult with eigenvalues and metadata
    """
    start_time = time.perf_counter()

    # Determine actual backend
    if backend == SpectralBackend.AUTO:
        kernel_type = "K_can" if kernel_matrix is None else "custom"
        backend = auto_select_backend(kernel_type, cell_count, internal_dim, include_eigenvectors)

    backend_name = backend.value

    # Check cache (only for K_can without eigenvectors)
    if use_cache and kernel_matrix is None and not include_eigenvectors:
        cache_key = _cache_key(cell_count, internal_dim, propagation_range, backend_name)
        if cache_key in _eigenvalue_cache:
            result = _eigenvalue_cache[cache_key]
            result.cache_hit = True
            return result

    # Dispatch to backend
    if backend == SpectralBackend.ANALYTIC_KCAN:
        result = _compute_analytic(cell_count, internal_dim, propagation_range, include_eigenvectors)

    elif backend == SpectralBackend.FOURIER_CELLS:
        if kernel_matrix is not None:
            result = _compute_fourier(kernel_matrix, cell_count, internal_dim, include_eigenvectors)
        else:
            # For K_can, fourier reduces to analytic
            from .fourier import fourier_cells_kcan_fast
            eigenvalues, info = fourier_cells_kcan_fast(cell_count, propagation_range, internal_dim)
            result = SpectralResult(
                eigenvalues=eigenvalues,
                backend_used="fourier_cells_kcan_fast",
                info=info
            )

    elif backend == SpectralBackend.SPARSE_FALLBACK:
        if kernel_matrix is not None:
            result = _compute_sparse(kernel_matrix, k or 100, include_eigenvectors)
        else:
            # Build K_can matrix and use sparse
            from ..vfd.canonical import VFDSpace, TorsionOperator, ShiftOperator
            from ..vfd.kernels import CanonicalKernel

            space = VFDSpace(cell_count=cell_count, internal_dim=internal_dim)
            T = TorsionOperator(space)
            S = ShiftOperator(space)
            kernel = CanonicalKernel(space, T, S, propagation_range=propagation_range)

            result = _compute_sparse(kernel.as_sparse(), k or 100, include_eigenvectors)

    else:
        raise ValueError(f"Unknown backend: {backend}")

    # Record timing
    elapsed_ms = (time.perf_counter() - start_time) * 1000
    result.computation_time_ms = elapsed_ms
    result.backend_used = backend_name

    # Cache result
    if use_cache and kernel_matrix is None and not include_eigenvectors:
        cache_key = _cache_key(cell_count, internal_dim, propagation_range, backend_name)
        _eigenvalue_cache[cache_key] = result

    return result


def _compute_analytic(
    cell_count: int,
    internal_dim: int,
    propagation_range: int,
    include_eigenvectors: bool
) -> SpectralResult:
    """Compute using analytic backend."""
    from .analytic import analytic_kcan_full

    unique_eigenvalues, multiplicities, info = analytic_kcan_full(
        cell_count, propagation_range, internal_dim
    )

    # Expand to full spectrum with multiplicities
    full_eigenvalues = np.repeat(unique_eigenvalues, multiplicities)

    eigenvectors = None
    if include_eigenvectors:
        # For K_can, eigenvectors are Fourier modes ⊗ standard basis
        # This would require significant memory for large systems
        from .analytic import analytic_eigenvector_cell
        # For now, skip eigenvector computation in analytic mode
        # Could implement on-demand eigenvector generation
        pass

    return SpectralResult(
        eigenvalues=full_eigenvalues,
        eigenvectors=eigenvectors,
        multiplicities=multiplicities,
        backend_used="analytic_kcan",
        info=info
    )


def _compute_fourier(
    kernel_matrix,
    cell_count: int,
    internal_dim: int,
    include_eigenvectors: bool
) -> SpectralResult:
    """Compute using Fourier backend."""
    if include_eigenvectors:
        from .fourier import fourier_cells_with_eigenvectors
        eigenvalues, eigenvectors, info = fourier_cells_with_eigenvectors(
            kernel_matrix, cell_count, internal_dim
        )
        return SpectralResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            backend_used="fourier_cells",
            info=info
        )
    else:
        from .fourier import fourier_cells_eigenvalues
        eigenvalues, info = fourier_cells_eigenvalues(kernel_matrix, cell_count, internal_dim)
        return SpectralResult(
            eigenvalues=eigenvalues,
            backend_used="fourier_cells",
            info=info
        )


def _compute_sparse(
    kernel_matrix,
    k: int,
    include_eigenvectors: bool
) -> SpectralResult:
    """Compute using sparse backend."""
    if include_eigenvectors:
        from .sparse import sparse_eigsh_with_eigenvectors
        eigenvalues, eigenvectors, info = sparse_eigsh_with_eigenvectors(kernel_matrix, k=k)
        return SpectralResult(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            backend_used="sparse_fallback",
            info=info
        )
    else:
        from .sparse import sparse_eigsh_eigenvalues
        eigenvalues, info = sparse_eigsh_eigenvalues(kernel_matrix, k=k)
        return SpectralResult(
            eigenvalues=eigenvalues,
            backend_used="sparse_fallback",
            info=info
        )


def benchmark_backends(
    cell_count: int,
    internal_dim: int = 600,
    propagation_range: int = 1,
    iterations: int = 3
) -> Dict[str, Dict[str, float]]:
    """
    Benchmark all backends for given parameters.

    Args:
        cell_count: Number of cells
        internal_dim: Internal dimension
        propagation_range: Coupling range
        iterations: Number of iterations per backend

    Returns:
        Dictionary mapping backend name to timing statistics
    """
    results = {}

    for backend in [SpectralBackend.ANALYTIC_KCAN, SpectralBackend.FOURIER_CELLS]:
        times = []

        for _ in range(iterations):
            # Clear cache for fair comparison
            clear_spectrum_cache()

            result = compute_spectrum(
                cell_count=cell_count,
                internal_dim=internal_dim,
                propagation_range=propagation_range,
                backend=backend,
                use_cache=False
            )

            times.append(result.computation_time_ms)

        results[backend.value] = {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "n_eigenvalues": len(result.eigenvalues),
        }

    return results


def verify_backend_agreement(
    cell_count: int,
    internal_dim: int = 600,
    propagation_range: int = 1,
    tol: float = 1e-8
) -> Tuple[bool, Dict[str, Any]]:
    """
    Verify that all backends produce consistent results.

    Args:
        cell_count: Number of cells
        internal_dim: Internal dimension
        propagation_range: Coupling range
        tol: Tolerance for eigenvalue comparison

    Returns:
        Tuple of (all_agree, comparison_details)
    """
    clear_spectrum_cache()

    # Compute with each backend
    results = {}
    for backend in [SpectralBackend.ANALYTIC_KCAN, SpectralBackend.FOURIER_CELLS]:
        result = compute_spectrum(
            cell_count=cell_count,
            internal_dim=internal_dim,
            propagation_range=propagation_range,
            backend=backend,
            use_cache=False
        )
        results[backend.value] = result.eigenvalues

    # Compare analytic vs fourier
    analytic_eigs = results["analytic_kcan"]
    fourier_eigs = results["fourier_cells"]

    # Sort for comparison
    analytic_sorted = np.sort(analytic_eigs)
    fourier_sorted = np.sort(fourier_eigs)

    if len(analytic_sorted) != len(fourier_sorted):
        return False, {
            "error": "Different number of eigenvalues",
            "analytic_count": len(analytic_sorted),
            "fourier_count": len(fourier_sorted),
        }

    max_diff = np.max(np.abs(analytic_sorted - fourier_sorted))
    mean_diff = np.mean(np.abs(analytic_sorted - fourier_sorted))

    agrees = max_diff < tol

    return agrees, {
        "max_difference": float(max_diff),
        "mean_difference": float(mean_diff),
        "tolerance": tol,
        "agrees": agrees,
        "n_eigenvalues": len(analytic_sorted),
    }
