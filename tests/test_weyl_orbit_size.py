"""
Test: Weyl relation requires orbit_size = 12.

This test demonstrates that the Weyl relation T S T^{-1} = ω S
only holds when orbit_size equals TORSION_ORDER (12).

BUG: When orbit_size != 12, the shift operator wraps at the wrong
period, causing a phase mismatch when position (orbit_size-1) wraps to 0.

FIX: orbit_size must be 12 to match the torsion order.
"""

import pytest
import numpy as np
import warnings


class TestWeylOrbitSize:
    """Tests for Weyl relation and orbit_size compatibility."""

    def test_weyl_fails_with_orbit_size_4(self):
        """
        BUG REPRODUCTION: Weyl relation FAILS when orbit_size=4.

        The shift S wraps position 3 → 0, giving:
          LHS: ω^{-3} × ω^0 = ω^{-3}
          RHS: ω

        Since ω = exp(2πi/12), ω^{-3} ≠ ω, so Weyl fails.
        """
        from vfd_dash.vfd.canonical import (
            VFDSpace, TorsionOperator, ShiftOperator,
            verify_weyl_relation, OMEGA, TORSION_ORDER
        )

        # This should emit a warning about orbit_size mismatch
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            space = VFDSpace(cell_count=4, internal_dim=8, orbit_count=2, orbit_size=4)
            assert len(w) == 1
            assert "orbit_size=4" in str(w[0].message)
            assert "Weyl relation" in str(w[0].message)

        T = TorsionOperator(space)
        S = ShiftOperator(space)

        # Weyl relation should FAIL
        passed, error = verify_weyl_relation(T, S, test_states=20)
        assert not passed, "Weyl should FAIL with orbit_size=4"
        assert error > 0.5, f"Expected large error, got {error}"

        # Verify the specific phase mismatch at wrap position
        e_last = np.zeros(space.total_dim, dtype=complex)
        e_last[3] = 1.0  # Last position in orbit (j=3)

        lhs = T.apply(S.apply(T.apply_inverse(e_last)))
        rhs = OMEGA * S.apply(e_last)

        # The phases should differ by ω^{-4}
        lhs_phase = lhs[np.argmax(np.abs(lhs))]
        rhs_phase = rhs[np.argmax(np.abs(rhs))]
        phase_ratio = lhs_phase / rhs_phase

        # ω^{-4} = exp(-8πi/12) = exp(-2πi/3)
        expected_ratio = OMEGA ** (-4)
        assert np.isclose(phase_ratio, expected_ratio, atol=1e-10), \
            f"Phase ratio {phase_ratio} should be ω^{{-4}} = {expected_ratio}"

    def test_weyl_passes_with_orbit_size_12(self):
        """
        FIX VERIFICATION: Weyl relation PASSES when orbit_size=12.

        With orbit_size=12 matching TORSION_ORDER=12:
          ω^12 = 1
        So when S wraps position 11 → 0:
          LHS: ω^{-11} × ω^0 = ω^{-11} = ω^1 = ω
          RHS: ω ✓
        """
        from vfd_dash.vfd.canonical import (
            VFDSpace, TorsionOperator, ShiftOperator,
            verify_weyl_relation, TORSION_ORDER
        )

        # No warning should be emitted with orbit_size=12
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            space = VFDSpace(cell_count=4, internal_dim=48, orbit_count=4, orbit_size=12)
            # Filter for our specific warning
            weyl_warnings = [x for x in w if "Weyl relation" in str(x.message)]
            assert len(weyl_warnings) == 0, "No warning expected with orbit_size=12"

        T = TorsionOperator(space)
        S = ShiftOperator(space)

        # Weyl relation should PASS
        passed, error = verify_weyl_relation(T, S, test_states=20)
        assert passed, f"Weyl should PASS with orbit_size=12, got error={error}"
        assert error < 1e-10, f"Expected negligible error, got {error}"

    def test_weyl_passes_canonical_config(self):
        """
        Weyl relation passes with canonical VFD config (600 dim).
        """
        from vfd_dash.vfd.canonical import (
            VFDSpace, TorsionOperator, ShiftOperator,
            verify_weyl_relation
        )

        # Canonical: 50 orbits × 12 = 600 internal dim
        space = VFDSpace(cell_count=8, internal_dim=600, orbit_count=50, orbit_size=12)
        T = TorsionOperator(space)
        S = ShiftOperator(space)

        passed, error = verify_weyl_relation(T, S, test_states=10)
        assert passed, f"Weyl should PASS with canonical config, error={error}"
        assert error < 1e-10

    def test_debug_weyl_identifies_failure(self):
        """
        Debug function correctly identifies wrap position as source of failure.
        """
        from vfd_dash.vfd.canonical import (
            VFDSpace, TorsionOperator, ShiftOperator,
            debug_weyl_on_basis, TORSION_ORDER
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            space = VFDSpace(cell_count=2, internal_dim=8, orbit_count=2, orbit_size=4)

        T = TorsionOperator(space)
        S = ShiftOperator(space)

        # Run debug (suppress output)
        result = debug_weyl_on_basis(T, S, verbose=False)

        # Should identify orbit_size mismatch
        assert not result["weyl_compatible"]
        assert result["orbit_size"] == 4
        assert result["torsion_order"] == TORSION_ORDER

        # Worst error should be at wrap position (j=3)
        assert result["worst_basis_idx"] == 3
        assert result["max_error"] > 1.0

        # Errors should be ~0 for non-wrap positions
        for j in range(3):
            assert result["errors_by_orbit_position"][j] < 1e-10

    def test_omega_twelfth_power_is_one(self):
        """
        Verify ω^12 = 1 (fundamental property for Weyl to work).
        """
        from vfd_dash.vfd.canonical import OMEGA, TORSION_ORDER

        assert TORSION_ORDER == 12
        omega_12 = OMEGA ** 12
        assert np.isclose(omega_12, 1.0, atol=1e-14), \
            f"ω^12 should be 1, got {omega_12}"

        # ω^k should cycle through 12th roots of unity
        for k in range(12):
            omega_k = OMEGA ** k
            assert np.isclose(abs(omega_k), 1.0), f"|ω^{k}| should be 1"

        # Only ω^12 = 1, not ω^4 or ω^6
        assert not np.isclose(OMEGA ** 4, 1.0), "ω^4 should NOT be 1"
        assert not np.isclose(OMEGA ** 6, 1.0), "ω^6 should NOT be 1"
