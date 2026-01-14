"""
Test VFD invariants.

These tests verify the fundamental VFD structure:
- Torsion order 12
- Weyl commutation relation
- Projector identities
- Kernel properties D1-D5
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vfd_dash.vfd.canonical import (
    VFDSpace, TorsionOperator, ShiftOperator,
    verify_weyl_relation, verify_torsion_order,
    TORSION_ORDER, OMEGA
)
from vfd_dash.vfd.operators import (
    create_torsion_projectors,
    verify_projector_resolution,
    verify_projector_orthogonality
)
from vfd_dash.vfd.kernels import CanonicalKernel


@pytest.fixture
def small_space():
    """Create small VFD space for testing."""
    return VFDSpace(cell_count=8, internal_dim=24, orbit_count=2, orbit_size=12)


@pytest.fixture
def operators(small_space):
    """Create operators for testing."""
    T = TorsionOperator(small_space)
    S = ShiftOperator(small_space)
    return T, S


class TestTorsionOrder:
    """Test torsion operator order."""

    def test_torsion_order_12(self, operators):
        """Verify T^12 = I."""
        T, _ = operators
        passed, error = verify_torsion_order(T)
        assert passed, f"T^12 != I, max error: {error}"

    def test_torsion_eigenvalues(self, operators):
        """Verify eigenvalues are 12th roots of unity."""
        T, _ = operators
        eigenvalues = np.unique(T._full_T)

        for ev in eigenvalues:
            # ev should be omega^k for some k
            found = False
            for k in range(TORSION_ORDER):
                if np.abs(ev - OMEGA**k) < 1e-10:
                    found = True
                    break
            assert found, f"Eigenvalue {ev} is not a 12th root of unity"


class TestWeylRelation:
    """Test Weyl commutation relation."""

    def test_weyl_commutation(self, operators):
        """Verify T S T^{-1} = omega S."""
        T, S = operators
        passed, error = verify_weyl_relation(T, S)
        assert passed, f"Weyl relation failed, max error: {error}"

    def test_weyl_on_random_states(self, operators, small_space):
        """Test Weyl relation on multiple random states."""
        T, S = operators
        rng = np.random.default_rng(42)

        for _ in range(10):
            state = rng.standard_normal(small_space.total_dim) + \
                    1j * rng.standard_normal(small_space.total_dim)
            state /= np.linalg.norm(state)

            lhs = T.apply(S.apply(T.apply_inverse(state)))
            rhs = OMEGA * S.apply(state)

            error = np.linalg.norm(lhs - rhs)
            assert error < 1e-10, f"Weyl relation error: {error}"


class TestProjectors:
    """Test torsion projector properties."""

    def test_projector_resolution(self, operators):
        """Verify sum_q P_q = I."""
        T, _ = operators
        projectors = create_torsion_projectors(T)
        passed, error = verify_projector_resolution(projectors)
        assert passed, f"Projector resolution failed, error: {error}"

    def test_projector_orthogonality(self, operators):
        """Verify P_q P_r = delta_{qr} P_q."""
        T, _ = operators
        projectors = create_torsion_projectors(T)
        passed, error = verify_projector_orthogonality(projectors)
        assert passed, f"Projector orthogonality failed, error: {error}"

    def test_projector_count(self, operators):
        """Verify 12 projectors."""
        T, _ = operators
        projectors = create_torsion_projectors(T)
        assert len(projectors) == TORSION_ORDER


class TestKernelProperties:
    """Test canonical kernel D1-D5 properties."""

    @pytest.fixture
    def kernel(self, small_space, operators):
        """Create canonical kernel."""
        T, S = operators
        return CanonicalKernel(small_space, T, S, propagation_range=1)

    def test_D1_selfadjoint(self, kernel):
        """Verify K = K* (self-adjoint)."""
        passed, error = kernel.verify_D1_selfadjoint()
        assert passed, f"Kernel not self-adjoint, error: {error}"

    def test_D2_torsion_commute(self, kernel):
        """Verify [K, T] = 0."""
        passed, error = kernel.verify_D2_torsion_commute()
        assert passed, f"Kernel doesn't commute with T, error: {error}"

    def test_D3_nonnegative(self, kernel):
        """Verify <v, Kv> >= 0."""
        passed, min_val = kernel.verify_D3_nonnegative()
        assert passed, f"Kernel not nonnegative, min eigenvalue: {min_val}"

    def test_D4_finite_propagation(self, kernel):
        """Verify finite propagation."""
        passed, prop_range = kernel.verify_D4_finite_propagation()
        assert passed
        assert prop_range >= 1

    def test_quadratic_form_positive(self, kernel, small_space):
        """Test quadratic form on random states."""
        rng = np.random.default_rng(42)

        for _ in range(10):
            state = rng.standard_normal(small_space.total_dim)
            state /= np.linalg.norm(state)

            Q = kernel.quadratic_form(state)
            assert Q >= -1e-10, f"Negative quadratic form: {Q}"
