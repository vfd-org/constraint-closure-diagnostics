"""
Tests for optimized spectrum computation backends.

Verifies:
1. Analytic formula correctness against direct matrix computation
2. Backend agreement (analytic vs fourier vs sparse)
3. Caching behavior
4. Torsion sector decomposition
5. Performance characteristics
"""

import pytest
import numpy as np
from numpy.testing import assert_allclose

from vfd_dash.vfd.canonical import VFDSpace, TorsionOperator, ShiftOperator
from vfd_dash.vfd.kernels import CanonicalKernel


class TestAnalyticBackend:
    """Tests for analytic_kcan backend."""

    def test_analytic_formula_small(self):
        """Verify analytic formula against direct computation for small C."""
        from vfd_dash.spectrum.analytic import verify_analytic_formula

        for cell_count in [4, 8, 16, 32]:
            for propagation_range in [1, 2, 3]:
                passes, max_error = verify_analytic_formula(cell_count, propagation_range)
                assert passes, f"Analytic formula failed for C={cell_count}, R={propagation_range}, error={max_error}"

    def test_zero_mode(self):
        """Verify that λ(θ=0) = 0 (Laplacian has zero mode)."""
        from vfd_dash.spectrum.analytic import circulant_laplacian_eigenvalue

        for R in [1, 2, 3, 5]:
            # Use large enough C to avoid wraparound edge case
            C = max(20, 4 * R)
            lambda_0 = circulant_laplacian_eigenvalue(0.0, R, C)
            assert abs(lambda_0) < 1e-14, f"Zero mode λ(0) = {lambda_0} != 0 for R={R}"

    def test_eigenvalue_nonnegativity(self):
        """Verify all eigenvalues are nonnegative (Laplacian is PSD)."""
        from vfd_dash.spectrum.analytic import analytic_kcan_cell_eigenvalues

        for cell_count in [8, 16, 64]:
            for R in [1, 2, 3]:
                eigenvalues = analytic_kcan_cell_eigenvalues(cell_count, R)
                assert np.all(eigenvalues >= -1e-14), f"Negative eigenvalue found for C={cell_count}, R={R}"

    def test_eigenvalue_count(self):
        """Verify correct number of eigenvalues."""
        from vfd_dash.spectrum.analytic import analytic_kcan_eigenvalues

        cell_count = 16
        internal_dim = 12  # Small for testing
        eigenvalues = analytic_kcan_eigenvalues(cell_count, propagation_range=1, internal_dim=internal_dim)

        expected_count = cell_count * internal_dim
        assert len(eigenvalues) == expected_count, f"Expected {expected_count} eigenvalues, got {len(eigenvalues)}"

    def test_multiplicity_structure(self):
        """Verify each cell eigenvalue has correct multiplicity."""
        from vfd_dash.spectrum.analytic import analytic_kcan_full

        cell_count = 8
        internal_dim = 24

        unique_eigs, multiplicities, info = analytic_kcan_full(cell_count, 1, internal_dim)

        assert len(unique_eigs) == cell_count
        assert np.all(multiplicities == internal_dim)
        assert info["total_eigenvalues"] == cell_count * internal_dim


class TestBackendAgreement:
    """Tests verifying all backends produce consistent results."""

    @pytest.fixture
    def small_config(self):
        """Small configuration for testing."""
        return {
            "cell_count": 8,
            "internal_dim": 12,
            "propagation_range": 1,
        }

    def test_analytic_vs_fourier(self, small_config):
        """Verify analytic and fourier backends agree."""
        from vfd_dash.spectrum.backend import verify_backend_agreement

        passes, details = verify_backend_agreement(**small_config)
        assert passes, f"Backend disagreement: {details}"
        assert details["max_difference"] < 1e-10

    def test_analytic_vs_direct_matrix(self, small_config):
        """Verify analytic backend agrees with direct matrix eigenvalues."""
        from vfd_dash.spectrum.analytic import analytic_kcan_eigenvalues

        # Build kernel matrix directly
        space = VFDSpace(
            cell_count=small_config["cell_count"],
            internal_dim=small_config["internal_dim"],
            orbit_count=small_config["internal_dim"] // 12,
            orbit_size=12
        )
        T = TorsionOperator(space)
        S = ShiftOperator(space)
        kernel = CanonicalKernel(space, T, S, propagation_range=small_config["propagation_range"])

        # Direct eigenvalues
        K_dense = kernel.as_matrix()
        direct_eigenvalues = np.sort(np.linalg.eigvalsh(K_dense))

        # Analytic eigenvalues
        analytic_eigenvalues = analytic_kcan_eigenvalues(
            small_config["cell_count"],
            small_config["propagation_range"],
            small_config["internal_dim"]
        )

        assert_allclose(
            direct_eigenvalues,
            analytic_eigenvalues,
            atol=1e-10,
            err_msg="Analytic eigenvalues don't match direct computation"
        )

    def test_all_backends_same_count(self, small_config):
        """All backends should return same number of eigenvalues."""
        from vfd_dash.spectrum.backend import compute_spectrum, SpectralBackend, clear_spectrum_cache

        counts = {}

        for backend in [SpectralBackend.ANALYTIC_KCAN, SpectralBackend.FOURIER_CELLS]:
            clear_spectrum_cache()
            result = compute_spectrum(
                cell_count=small_config["cell_count"],
                internal_dim=small_config["internal_dim"],
                propagation_range=small_config["propagation_range"],
                backend=backend,
                use_cache=False
            )
            counts[backend.value] = len(result.eigenvalues)

        # All counts should be equal
        values = list(counts.values())
        assert all(v == values[0] for v in values), f"Eigenvalue counts differ: {counts}"


class TestCaching:
    """Tests for eigenvalue caching."""

    def test_cache_hit(self):
        """Verify caching returns same results faster."""
        from vfd_dash.spectrum.backend import compute_spectrum, SpectralBackend, clear_spectrum_cache

        clear_spectrum_cache()

        # First call: cache miss
        result1 = compute_spectrum(cell_count=16, internal_dim=24, use_cache=True)
        assert not result1.cache_hit

        # Second call: cache hit
        result2 = compute_spectrum(cell_count=16, internal_dim=24, use_cache=True)
        assert result2.cache_hit

        # Results should be identical
        assert_allclose(result1.eigenvalues, result2.eigenvalues)

    def test_cache_invalidation(self):
        """Verify different parameters don't share cache."""
        from vfd_dash.spectrum.backend import compute_spectrum, clear_spectrum_cache

        clear_spectrum_cache()

        result1 = compute_spectrum(cell_count=8, internal_dim=12, use_cache=True)
        result2 = compute_spectrum(cell_count=16, internal_dim=12, use_cache=True)

        # Both should be cache misses (different cell_count)
        assert not result1.cache_hit
        assert not result2.cache_hit

        # Different number of eigenvalues
        assert len(result1.eigenvalues) != len(result2.eigenvalues)


class TestTorsionSectors:
    """Tests for torsion sector decomposition."""

    def test_sector_indices(self):
        """Verify sector indices are correct and complete."""
        from vfd_dash.spectrum.torsion_sectors import get_torsion_projection_indices

        internal_dim = 600
        orbit_count = 50
        orbit_size = 12

        indices = get_torsion_projection_indices(internal_dim, orbit_count, orbit_size)

        # Should have 12 sectors
        assert len(indices) == 12

        # Each sector should have orbit_count indices
        for q in range(12):
            assert len(indices[q]) == orbit_count

        # All indices together should cover 0..599
        all_indices = []
        for q in range(12):
            all_indices.extend(indices[q])
        all_indices = sorted(all_indices)
        assert all_indices == list(range(internal_dim))

    def test_sector_spectrum_kcan(self):
        """Verify sector spectrum matches full spectrum."""
        from vfd_dash.spectrum.torsion_sectors import torsion_sector_spectrum_kcan
        from vfd_dash.spectrum.analytic import analytic_kcan_eigenvalues

        cell_count = 8
        orbit_count = 4
        orbit_size = 12
        internal_dim = orbit_count * orbit_size

        # Full spectrum
        full_eigs = analytic_kcan_eigenvalues(cell_count, 1, internal_dim)

        # Sector spectrum
        all_eigs, sector_spectra, info = torsion_sector_spectrum_kcan(
            cell_count, 1, orbit_count, orbit_size
        )

        assert_allclose(np.sort(full_eigs), np.sort(all_eigs), atol=1e-12)
        assert info["sectors_identical"]  # For K_can, all sectors have same spectrum

    def test_torsion_fingerprint_shape(self):
        """Verify torsion fingerprint has correct shape."""
        from vfd_dash.spectrum.torsion_sectors import torsion_sector_spectrum_kcan, torsion_fingerprint

        _, sector_spectra, _ = torsion_sector_spectrum_kcan(16, 1, 4, 12)
        fingerprint = torsion_fingerprint(sector_spectra, n_bins=50)

        assert fingerprint.shape == (12, 50)


class TestKernelIntegration:
    """Tests for kernel integration with spectrum module."""

    def test_kernel_compute_spectrum_fast(self):
        """Verify kernel.compute_spectrum_fast() works."""
        space = VFDSpace(cell_count=8, internal_dim=24, orbit_count=2, orbit_size=12)
        T = TorsionOperator(space)
        S = ShiftOperator(space)
        kernel = CanonicalKernel(space, T, S, propagation_range=1)

        result = kernel.compute_spectrum_fast()

        assert result.eigenvalues is not None
        assert len(result.eigenvalues) == space.total_dim
        assert result.backend_used == "analytic_kcan"

    def test_kernel_get_cell_eigenvalues(self):
        """Verify kernel.get_cell_eigenvalues() returns unique values."""
        space = VFDSpace(cell_count=16, internal_dim=24, orbit_count=2, orbit_size=12)
        T = TorsionOperator(space)
        S = ShiftOperator(space)
        kernel = CanonicalKernel(space, T, S, propagation_range=1)

        cell_eigs = kernel.get_cell_eigenvalues()

        assert len(cell_eigs) == space.cell_count
        assert cell_eigs[0] >= -1e-14  # Zero mode
        assert np.all(np.diff(cell_eigs) >= -1e-14)  # Sorted

    def test_kernel_type_property(self):
        """Verify kernel type properties."""
        space = VFDSpace(cell_count=8, internal_dim=24, orbit_count=2, orbit_size=12)
        T = TorsionOperator(space)
        S = ShiftOperator(space)
        kernel = CanonicalKernel(space, T, S)

        assert kernel.kernel_type == "K_can"
        assert kernel.has_kronecker_structure


class TestProfiling:
    """Tests for profiling utilities."""

    def test_memory_estimation(self):
        """Verify memory estimation is reasonable."""
        from vfd_dash.spectrum.profiling import estimate_spectrum_memory

        # Analytic should be tiny
        mem_analytic = estimate_spectrum_memory(64, 600, "analytic_kcan")
        assert mem_analytic["total_gb"] < 0.01  # Less than 10 MB

        # Sparse should be moderate
        mem_sparse = estimate_spectrum_memory(64, 600, "sparse_fallback")
        assert mem_sparse["total_gb"] < 1.0  # Less than 1 GB

    def test_feasibility_check(self):
        """Verify feasibility check works."""
        from vfd_dash.spectrum.profiling import check_feasibility

        # Small should be feasible
        result = check_feasibility(8, 24, "analytic_kcan")
        assert result["feasible"]

        # Very large should fail
        result = check_feasibility(10000, 10000, "sparse_fallback")
        assert not result["feasible"]


class TestPerformance:
    """Performance regression tests."""

    @pytest.mark.slow
    def test_analytic_is_fastest(self):
        """Verify analytic backend is faster than others."""
        from vfd_dash.spectrum.backend import compute_spectrum, SpectralBackend, clear_spectrum_cache

        cell_count = 64
        internal_dim = 600

        clear_spectrum_cache()
        result_analytic = compute_spectrum(
            cell_count, internal_dim, backend=SpectralBackend.ANALYTIC_KCAN, use_cache=False
        )

        clear_spectrum_cache()
        result_fourier = compute_spectrum(
            cell_count, internal_dim, backend=SpectralBackend.FOURIER_CELLS, use_cache=False
        )

        # Analytic should be significantly faster
        assert result_analytic.computation_time_ms < result_fourier.computation_time_ms * 0.5

    def test_analytic_scales_linearly(self):
        """Verify analytic backend scales O(C)."""
        from vfd_dash.spectrum.backend import compute_spectrum, SpectralBackend, clear_spectrum_cache
        import time

        times = []
        cell_counts = [16, 32, 64, 128]

        for C in cell_counts:
            clear_spectrum_cache()
            start = time.perf_counter()
            compute_spectrum(C, 600, backend=SpectralBackend.ANALYTIC_KCAN, use_cache=False)
            elapsed = time.perf_counter() - start
            times.append(elapsed)

        # Doubling C should approximately double time (linear scaling)
        # Allow some slack for overhead
        for i in range(len(times) - 1):
            ratio = times[i + 1] / times[i]
            assert ratio < 5, f"Scaling appears superlinear: {times}"
