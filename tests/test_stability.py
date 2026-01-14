"""
Test stability analysis.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vfd_dash.vfd.canonical import VFDSpace, TorsionOperator, ShiftOperator
from vfd_dash.vfd.kernels import CanonicalKernel
from vfd_dash.vfd.probes import ProbeGenerator, Probe
from vfd_dash.vfd.stability import StabilityAnalyzer, SelfDualManifold


@pytest.fixture
def small_space():
    """Create small VFD space."""
    return VFDSpace(cell_count=8, internal_dim=24, orbit_count=2, orbit_size=12)


@pytest.fixture
def operators(small_space):
    """Create operators."""
    T = TorsionOperator(small_space)
    S = ShiftOperator(small_space)
    return T, S


@pytest.fixture
def kernel(small_space, operators):
    """Create canonical kernel."""
    T, S = operators
    return CanonicalKernel(small_space, T, S, propagation_range=1)


@pytest.fixture
def analyzer(small_space, operators, kernel):
    """Create stability analyzer."""
    T, _ = operators
    return StabilityAnalyzer(small_space, T, kernel, seed=42)


class TestProbes:
    """Test probe generation."""

    def test_pure_torsion_probe(self, small_space, operators):
        """Test generating pure torsion probes."""
        T, _ = operators
        gen = ProbeGenerator(small_space, T, seed=42)

        probe = gen.generate_pure_torsion_probe(torsion_degree=3, center_cell=4)

        assert probe.torsion_degree == 3
        assert 4 in probe.support_cells
        assert np.abs(np.linalg.norm(probe.state) - 1.0) < 1e-10

    def test_torsion_degree_verification(self, small_space, operators):
        """Verify probe has correct torsion degree."""
        T, _ = operators
        gen = ProbeGenerator(small_space, T, seed=42)

        for k in range(12):
            probe = gen.generate_pure_torsion_probe(torsion_degree=k, center_cell=2)
            assert gen.verify_torsion_degree(probe)

    def test_probe_family(self, small_space, operators):
        """Test generating probe family."""
        T, _ = operators
        gen = ProbeGenerator(small_space, T, seed=42)

        probes = gen.generate_probe_family(count=50, support_radius=1)

        assert len(probes) == 50
        assert all(isinstance(p, Probe) for p in probes)


class TestStabilityAnalysis:
    """Test stability coefficient computation."""

    def test_quadratic_form(self, analyzer, small_space):
        """Test quadratic form computation."""
        state = np.random.randn(small_space.total_dim)
        state /= np.linalg.norm(state)

        Q = analyzer.compute_quadratic_form(state)

        assert isinstance(Q, float)

    def test_stability_coefficients(self, analyzer):
        """Test computing stability coefficients."""
        coeffs = analyzer.compute_stability_coefficients(probe_count=50)

        assert len(coeffs) == 50
        assert all(c.value is not None for c in coeffs)

    def test_kernel_absoluteness(self, analyzer):
        """Test kernel absoluteness verification."""
        analyzer.compute_stability_coefficients(probe_count=100)
        passed, details = analyzer.verify_kernel_absoluteness()

        assert "min_Q_value" in details
        assert "negative_count" in details

        # For valid kernel, should pass
        assert passed, f"Kernel absoluteness failed: {details}"

    def test_all_nonnegative(self, analyzer):
        """All stability coefficients should be nonnegative."""
        coeffs = analyzer.compute_stability_coefficients(probe_count=100)

        negative = [c for c in coeffs if c.value < -1e-10]

        assert len(negative) == 0, f"Found {len(negative)} negative coefficients"

    def test_spectrum_computation(self, analyzer):
        """Test spectrum computation."""
        spectral = analyzer.compute_spectrum(k=20)

        assert len(spectral.eigenvalues) == 20
        assert spectral.eigenvalues[0] <= spectral.eigenvalues[-1]  # Sorted


class TestSelfDualManifold:
    """Test self-dual manifold."""

    def test_manifold_creation(self, small_space, operators):
        """Test creating self-dual manifold."""
        T, _ = operators
        manifold = SelfDualManifold(small_space, T)

        assert manifold is not None

    def test_manifold_projection(self, small_space, operators):
        """Test projecting to self-dual manifold."""
        T, _ = operators
        manifold = SelfDualManifold(small_space, T)

        state = np.random.randn(small_space.total_dim) + \
                1j * np.random.randn(small_space.total_dim)
        state /= np.linalg.norm(state)

        projected = manifold.project_to_manifold(state)

        # Projected state should be on manifold
        assert manifold.is_on_manifold(projected, tol=0.1)
