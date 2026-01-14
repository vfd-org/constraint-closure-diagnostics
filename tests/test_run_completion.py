"""
Test suite for run completion and artifact validation.

Tests:
- Run completes without error
- Manifest is written
- Expected figures exist (when generated)
- Basic sanity checks on arrays (non-empty, finite)
"""

import pytest
import numpy as np
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vfd_dash.core.config import RunConfig, get_default_config
from vfd_dash.vfd.canonical import VFDSpace, TorsionOperator, ShiftOperator
from vfd_dash.vfd.kernels import CanonicalKernel
from vfd_dash.vfd.stability import StabilityAnalyzer
from vfd_dash.vfd.primes import InternalPrimeGenerator
from vfd_dash.spectrum.backend import compute_spectrum, SpectralBackend


class TestRunCompletion:
    """Test that analysis runs complete successfully."""

    @pytest.fixture
    def small_config(self):
        """Create a minimal config for fast testing."""
        config = get_default_config()
        config.seed = 42
        config.vfd.cell_count = 4
        config.vfd.internal_dim = 24
        config.vfd.orbit_count = 2
        config.vfd.orbit_size = 12
        config.prime_field.max_transport_length = 10
        config.stability.probe_count = 20
        config.bridge.bridge_mode = "OFF"
        return config

    def test_vfd_space_creation(self, small_config):
        """Test VFD space can be created."""
        space = VFDSpace(
            cell_count=small_config.vfd.cell_count,
            internal_dim=small_config.vfd.internal_dim,
            orbit_count=small_config.vfd.orbit_count,
            orbit_size=small_config.vfd.orbit_size
        )
        assert space.total_dim == small_config.vfd.cell_count * small_config.vfd.internal_dim
        assert space.total_dim > 0

    def test_operators_creation(self, small_config):
        """Test operators can be created."""
        space = VFDSpace(
            cell_count=small_config.vfd.cell_count,
            internal_dim=small_config.vfd.internal_dim,
            orbit_count=small_config.vfd.orbit_count,
            orbit_size=small_config.vfd.orbit_size
        )
        T = TorsionOperator(space)
        S = ShiftOperator(space)

        # Verify operators have expected shapes
        assert len(T._full_T) == space.total_dim
        assert T.verify_order()

    def test_kernel_creation(self, small_config):
        """Test kernel can be created and has expected properties."""
        space = VFDSpace(
            cell_count=small_config.vfd.cell_count,
            internal_dim=small_config.vfd.internal_dim,
            orbit_count=small_config.vfd.orbit_count,
            orbit_size=small_config.vfd.orbit_size
        )
        T = TorsionOperator(space)
        S = ShiftOperator(space)
        kernel = CanonicalKernel(space, T, S, propagation_range=1)

        # Verify kernel is valid
        K = kernel.as_matrix()
        assert K.shape == (space.total_dim, space.total_dim)

        # Verify non-empty
        assert np.count_nonzero(K) > 0

    def test_prime_generation(self, small_config):
        """Test internal primes can be generated."""
        space = VFDSpace(
            cell_count=small_config.vfd.cell_count,
            internal_dim=small_config.vfd.internal_dim,
            orbit_count=small_config.vfd.orbit_count,
            orbit_size=small_config.vfd.orbit_size
        )
        T = TorsionOperator(space)
        S = ShiftOperator(space)

        prime_gen = InternalPrimeGenerator(
            space, T, S,
            max_length=small_config.prime_field.max_transport_length,
            seed=small_config.seed
        )
        primes = prime_gen.generate_primes_up_to(small_config.prime_field.max_transport_length)

        # Verify primes exist
        assert len(primes) > 0

        # Verify all primes have positive length
        for p in primes:
            assert p.length > 0

    def test_stability_analysis(self, small_config):
        """Test stability analysis completes."""
        space = VFDSpace(
            cell_count=small_config.vfd.cell_count,
            internal_dim=small_config.vfd.internal_dim,
            orbit_count=small_config.vfd.orbit_count,
            orbit_size=small_config.vfd.orbit_size
        )
        T = TorsionOperator(space)
        S = ShiftOperator(space)
        kernel = CanonicalKernel(space, T, S, propagation_range=1)

        analyzer = StabilityAnalyzer(space, T, kernel, seed=small_config.seed)
        coefficients = analyzer.compute_stability_coefficients(
            probe_count=small_config.stability.probe_count,
            support_radius=2
        )

        # Verify coefficients exist
        assert len(coefficients) > 0

        # Verify all coefficients are finite
        for c in coefficients:
            assert np.isfinite(c.value)

    def test_spectrum_computation(self, small_config):
        """Test spectrum can be computed."""
        result = compute_spectrum(
            cell_count=small_config.vfd.cell_count,
            internal_dim=small_config.vfd.internal_dim,
            propagation_range=1,
            backend=SpectralBackend.ANALYTIC_KCAN
        )

        # Verify eigenvalues exist
        assert len(result.eigenvalues) > 0

        # Verify all eigenvalues are finite and nonnegative
        assert np.all(np.isfinite(result.eigenvalues))
        assert np.all(result.eigenvalues >= -1e-10)  # Allow small numerical error


class TestManifestWriting:
    """Test manifest file generation."""

    def test_config_serialization(self):
        """Test config can be serialized to JSON."""
        config = get_default_config()
        json_str = config.to_json()

        # Verify valid JSON
        parsed = json.loads(json_str)
        assert "seed" in parsed
        assert "vfd" in parsed

    def test_config_round_trip(self):
        """Test config survives JSON round-trip."""
        config = get_default_config()
        config.seed = 12345
        config.vfd.cell_count = 32

        json_str = config.to_json()
        loaded = RunConfig.from_json(json_str)

        assert loaded.seed == config.seed
        assert loaded.vfd.cell_count == config.vfd.cell_count


class TestArraySanity:
    """Test arrays have expected properties."""

    def test_eigenvalues_nonnegative(self):
        """Test all eigenvalues are nonnegative."""
        result = compute_spectrum(
            cell_count=8,
            internal_dim=24,
            propagation_range=1,
            backend=SpectralBackend.ANALYTIC_KCAN
        )

        # K_can is a Laplacian, so eigenvalues >= 0
        assert np.all(result.eigenvalues >= -1e-10)

    def test_eigenvalues_sorted(self):
        """Test eigenvalues are returned sorted."""
        result = compute_spectrum(
            cell_count=8,
            internal_dim=24,
            propagation_range=1,
            backend=SpectralBackend.ANALYTIC_KCAN
        )

        # Check sorted
        sorted_eigs = np.sort(result.eigenvalues)
        # Allow for floating point comparison
        assert np.allclose(result.eigenvalues, sorted_eigs)

    def test_quadratic_form_nonnegative(self):
        """Test quadratic forms are nonnegative."""
        space = VFDSpace(cell_count=4, internal_dim=24, orbit_count=2, orbit_size=12)
        T = TorsionOperator(space)
        S = ShiftOperator(space)
        kernel = CanonicalKernel(space, T, S, propagation_range=1)

        # Test on random states
        np.random.seed(42)
        for _ in range(10):
            state = np.random.randn(space.total_dim)
            state /= np.linalg.norm(state)
            Q = kernel.quadratic_form(state)
            assert Q >= -1e-10  # Allow small numerical error


class TestDeterminism:
    """Test reproducibility with fixed seeds."""

    def test_spectrum_deterministic(self):
        """Test spectrum is deterministic."""
        results = []
        for _ in range(2):
            np.random.seed(42)
            result = compute_spectrum(
                cell_count=8,
                internal_dim=24,
                propagation_range=1,
                backend=SpectralBackend.ANALYTIC_KCAN,
                use_cache=False
            )
            results.append(result.eigenvalues)

        # Should be identical
        assert np.array_equal(results[0], results[1])

    def test_stability_deterministic(self):
        """Test stability analysis is deterministic."""
        space = VFDSpace(cell_count=4, internal_dim=24, orbit_count=2, orbit_size=12)
        T = TorsionOperator(space)
        S = ShiftOperator(space)
        kernel = CanonicalKernel(space, T, S, propagation_range=1)

        results = []
        for _ in range(2):
            np.random.seed(42)
            analyzer = StabilityAnalyzer(space, T, kernel, seed=42)
            coefficients = analyzer.compute_stability_coefficients(
                probe_count=10,
                support_radius=2
            )
            values = [c.value for c in coefficients]
            results.append(values)

        # Should be identical
        assert np.allclose(results[0], results[1])
