"""
Profiling and Safety Guardrails for Spectrum Computation.

Provides:
- Memory estimation before computation
- Timeout wrappers for expensive operations
- Progress tracking for iterative methods
- Performance benchmarking utilities
"""

import time
import warnings
import functools
import signal
from typing import Dict, Any, Optional, Callable, TypeVar
from dataclasses import dataclass, field
import numpy as np


T = TypeVar('T')


@dataclass
class ComputationLimits:
    """Safety limits for spectrum computation."""
    max_memory_gb: float = 8.0
    max_time_seconds: float = 300.0
    max_matrix_dim: int = 100000
    max_dense_dim: int = 5000
    warn_memory_gb: float = 4.0
    warn_time_seconds: float = 60.0


@dataclass
class ProfileResult:
    """Result of profiled computation."""
    success: bool
    result: Any = None
    error: Optional[str] = None
    elapsed_seconds: float = 0.0
    memory_estimate_gb: float = 0.0
    warnings: list = field(default_factory=list)


# Global limits (can be modified)
LIMITS = ComputationLimits()


def estimate_spectrum_memory(
    cell_count: int,
    internal_dim: int,
    backend: str,
    k: Optional[int] = None
) -> Dict[str, float]:
    """
    Estimate memory usage for spectrum computation.

    Args:
        cell_count: Number of cells
        internal_dim: Internal dimension
        backend: Backend name
        k: Number of eigenvalues (for sparse methods)

    Returns:
        Dictionary with memory estimates in GB
    """
    total_dim = cell_count * internal_dim
    bytes_per_complex = 16  # complex128
    bytes_per_float = 8

    estimates = {
        "total_dim": total_dim,
    }

    if backend == "analytic_kcan":
        # Only need to store eigenvalues array
        eig_bytes = cell_count * bytes_per_float
        estimates["eigenvalues_gb"] = eig_bytes / (1024**3)
        estimates["total_gb"] = estimates["eigenvalues_gb"]

    elif backend == "fourier_cells":
        # C blocks of internal_dim x internal_dim
        block_bytes = cell_count * internal_dim**2 * bytes_per_complex
        eig_bytes = total_dim * bytes_per_float
        estimates["blocks_gb"] = block_bytes / (1024**3)
        estimates["eigenvalues_gb"] = eig_bytes / (1024**3)
        estimates["total_gb"] = estimates["blocks_gb"] + estimates["eigenvalues_gb"]

    elif backend == "sparse_fallback":
        # Sparse matrix: ~10 * nnz * 8 bytes
        # For Laplacian: nnz ~ 6 * total_dim (assuming R=3)
        nnz_estimate = 6 * total_dim
        sparse_bytes = 10 * nnz_estimate * bytes_per_float

        # Lanczos vectors: k * total_dim
        k_actual = k or 100
        lanczos_bytes = k_actual * total_dim * bytes_per_float

        estimates["sparse_matrix_gb"] = sparse_bytes / (1024**3)
        estimates["lanczos_gb"] = lanczos_bytes / (1024**3)
        estimates["total_gb"] = estimates["sparse_matrix_gb"] + estimates["lanczos_gb"]

    else:
        # Dense fallback
        dense_bytes = total_dim**2 * bytes_per_complex
        estimates["dense_matrix_gb"] = dense_bytes / (1024**3)
        estimates["total_gb"] = estimates["dense_matrix_gb"]

    return estimates


def check_feasibility(
    cell_count: int,
    internal_dim: int,
    backend: str = "auto",
    limits: Optional[ComputationLimits] = None
) -> Dict[str, Any]:
    """
    Check if computation is feasible within limits.

    Args:
        cell_count: Number of cells
        internal_dim: Internal dimension
        backend: Backend to check
        limits: Custom limits (uses global LIMITS if None)

    Returns:
        Dictionary with feasibility information
    """
    limits = limits or LIMITS
    total_dim = cell_count * internal_dim

    result = {
        "cell_count": cell_count,
        "internal_dim": internal_dim,
        "total_dim": total_dim,
        "backend": backend,
        "feasible": True,
        "warnings": [],
        "errors": [],
    }

    # Check dimension limits
    if total_dim > limits.max_matrix_dim:
        result["feasible"] = False
        result["errors"].append(
            f"Total dimension {total_dim} exceeds max {limits.max_matrix_dim}"
        )

    # Check memory
    if backend == "auto":
        backend = "analytic_kcan"  # Default for K_can

    mem_estimate = estimate_spectrum_memory(cell_count, internal_dim, backend)
    result["memory_estimate_gb"] = mem_estimate["total_gb"]

    if mem_estimate["total_gb"] > limits.max_memory_gb:
        result["feasible"] = False
        result["errors"].append(
            f"Memory estimate {mem_estimate['total_gb']:.2f} GB exceeds max {limits.max_memory_gb} GB"
        )
    elif mem_estimate["total_gb"] > limits.warn_memory_gb:
        result["warnings"].append(
            f"Memory estimate {mem_estimate['total_gb']:.2f} GB is high"
        )

    # Backend-specific checks
    if backend == "sparse_fallback" and total_dim > 50000:
        result["warnings"].append(
            "sparse_fallback may be slow for dim > 50000, consider analytic_kcan"
        )

    if backend not in ["analytic_kcan", "fourier_cells"] and total_dim > limits.max_dense_dim:
        result["warnings"].append(
            f"Dense methods not recommended for dim > {limits.max_dense_dim}"
        )

    # Recommend best backend
    if backend == "auto":
        result["recommended_backend"] = "analytic_kcan"
    else:
        result["recommended_backend"] = backend

    return result


class TimeoutError(Exception):
    """Raised when computation exceeds time limit."""
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Computation timed out")


def with_timeout(timeout_seconds: float):
    """
    Decorator to add timeout to a function.

    Note: Only works on Unix-like systems (uses signal.SIGALRM).
    On Windows, timeout is not enforced.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            # Try to use signal-based timeout (Unix only)
            try:
                old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                signal.alarm(int(timeout_seconds))
                try:
                    result = func(*args, **kwargs)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
                return result
            except (AttributeError, ValueError):
                # signal.SIGALRM not available (Windows)
                # Just run without timeout
                return func(*args, **kwargs)
        return wrapper
    return decorator


def profile_computation(
    func: Callable[[], T],
    cell_count: int,
    internal_dim: int,
    backend: str,
    limits: Optional[ComputationLimits] = None
) -> ProfileResult:
    """
    Profile a spectrum computation with safety checks.

    Args:
        func: Function to call (takes no arguments)
        cell_count: Number of cells
        internal_dim: Internal dimension
        backend: Backend being used
        limits: Safety limits

    Returns:
        ProfileResult with success status and metrics
    """
    limits = limits or LIMITS

    # Pre-check feasibility
    feasibility = check_feasibility(cell_count, internal_dim, backend, limits)

    if not feasibility["feasible"]:
        return ProfileResult(
            success=False,
            error="; ".join(feasibility["errors"]),
            memory_estimate_gb=feasibility.get("memory_estimate_gb", 0.0),
            warnings=feasibility["warnings"]
        )

    # Issue warnings
    for warning in feasibility["warnings"]:
        warnings.warn(warning)

    # Run with profiling
    start_time = time.perf_counter()

    try:
        # Apply timeout wrapper
        @with_timeout(limits.max_time_seconds)
        def timed_func():
            return func()

        result = timed_func()
        elapsed = time.perf_counter() - start_time

        return ProfileResult(
            success=True,
            result=result,
            elapsed_seconds=elapsed,
            memory_estimate_gb=feasibility.get("memory_estimate_gb", 0.0),
            warnings=feasibility["warnings"]
        )

    except TimeoutError:
        elapsed = time.perf_counter() - start_time
        return ProfileResult(
            success=False,
            error=f"Computation timed out after {elapsed:.1f} seconds",
            elapsed_seconds=elapsed,
            memory_estimate_gb=feasibility.get("memory_estimate_gb", 0.0),
            warnings=feasibility["warnings"]
        )

    except Exception as e:
        elapsed = time.perf_counter() - start_time
        return ProfileResult(
            success=False,
            error=str(e),
            elapsed_seconds=elapsed,
            memory_estimate_gb=feasibility.get("memory_estimate_gb", 0.0),
            warnings=feasibility["warnings"]
        )


def benchmark_all_backends(
    cell_count: int,
    internal_dim: int = 600,
    propagation_range: int = 1,
    iterations: int = 3
) -> Dict[str, Dict[str, Any]]:
    """
    Benchmark all available backends.

    Args:
        cell_count: Number of cells
        internal_dim: Internal dimension
        propagation_range: Coupling range
        iterations: Number of iterations per backend

    Returns:
        Dictionary mapping backend -> benchmark results
    """
    from .backend import compute_spectrum, SpectralBackend, clear_spectrum_cache

    results = {}

    for backend in [SpectralBackend.ANALYTIC_KCAN, SpectralBackend.FOURIER_CELLS]:
        times = []
        n_eigenvalues = 0

        for _ in range(iterations):
            clear_spectrum_cache()

            start = time.perf_counter()
            result = compute_spectrum(
                cell_count=cell_count,
                internal_dim=internal_dim,
                propagation_range=propagation_range,
                backend=backend,
                use_cache=False
            )
            elapsed = time.perf_counter() - start

            times.append(elapsed * 1000)  # Convert to ms
            n_eigenvalues = len(result.eigenvalues)

        results[backend.value] = {
            "mean_ms": np.mean(times),
            "std_ms": np.std(times),
            "min_ms": np.min(times),
            "max_ms": np.max(times),
            "n_eigenvalues": n_eigenvalues,
            "iterations": iterations,
        }

    return results


def format_benchmark_report(results: Dict[str, Dict[str, Any]]) -> str:
    """Format benchmark results as a readable report."""
    lines = ["Spectrum Backend Benchmark Results", "=" * 40]

    for backend, stats in results.items():
        lines.append(f"\n{backend}:")
        lines.append(f"  Mean time: {stats['mean_ms']:.2f} ms ± {stats['std_ms']:.2f} ms")
        lines.append(f"  Range: [{stats['min_ms']:.2f}, {stats['max_ms']:.2f}] ms")
        lines.append(f"  Eigenvalues: {stats['n_eigenvalues']}")

    return "\n".join(lines)
