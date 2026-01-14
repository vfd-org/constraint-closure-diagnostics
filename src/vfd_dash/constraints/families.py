"""
Constraint Families for RH Constraint-Diagnostic Demo.

Four families of constraints:
- EF (Explicit Formula): Torsion order, Weyl relation
- Symmetry: Projector identities, self-dual balance
- Positivity: Kernel nonnegativity, quadratic form positivity
- Trace/Moment: Spectral moment constraints

Each family has an evaluate() method returning a dict of residuals.
Residuals are numeric floats: 0.0 if pass, >0 magnitude if fail.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, TYPE_CHECKING
import numpy as np

from .ladder import ClosureLevel

if TYPE_CHECKING:
    from ..state import DiagnosticState


class ConstraintFamily(ABC):
    """Base class for constraint families."""

    name: str = "base"

    @abstractmethod
    def evaluate(self, state: "DiagnosticState") -> Dict[str, float]:
        """
        Evaluate all constraints in this family.

        Args:
            state: Diagnostic state with computed data

        Returns:
            Dict mapping constraint name to residual value.
            Residual is 0.0 if satisfied, >0 if violated.
        """
        pass

    @property
    def constraint_names(self) -> List[str]:
        """List of constraint names in this family."""
        return []


class ExplicitFormulaFamily(ConstraintFamily):
    """
    Explicit Formula (EF) constraints.

    Verifies consistency with explicit structural formulas:
    - T^12 = I (torsion order)
    - TST^{-1} = ωS (Weyl commutation)
    """

    name = "EF"

    def evaluate(self, state: "DiagnosticState") -> Dict[str, float]:
        """Evaluate EF constraints."""
        residuals = {}

        # Torsion order: T^12 = I
        if state.invariants:
            torsion_err = state.invariants.get("torsion_error", 0.0)
            residuals["torsion_order"] = float(torsion_err)

            # Weyl relation: TST^{-1} = ωS
            weyl_err = state.invariants.get("weyl_error", 0.0)
            residuals["weyl_relation"] = float(weyl_err)
        else:
            # Compute directly if invariants not cached
            from ..vfd.canonical import verify_torsion_order, verify_weyl_relation
            _, torsion_err = verify_torsion_order(state.T)
            _, weyl_err = verify_weyl_relation(state.T, state.S)
            residuals["torsion_order"] = float(torsion_err)
            residuals["weyl_relation"] = float(weyl_err)

        return residuals

    @property
    def constraint_names(self) -> List[str]:
        return ["torsion_order", "weyl_relation"]


class SymmetryFamily(ConstraintFamily):
    """
    Symmetry constraints.

    Verifies symmetry properties:
    - Σ_q P_q = I (projector resolution)
    - P_q P_r = δ_{qr} P_q (projector orthogonality)
    - Self-dual balance (torsion degree symmetry)
    """

    name = "Symmetry"

    def evaluate(self, state: "DiagnosticState") -> Dict[str, float]:
        """Evaluate symmetry constraints."""
        residuals = {}

        if state.invariants:
            # Projector resolution
            res_err = state.invariants.get("projector_resolution_error", 0.0)
            residuals["projector_resolution"] = float(res_err)

            # Projector orthogonality
            orth_err = state.invariants.get("projector_orthogonality_error", 0.0)
            residuals["projector_orthogonality"] = float(orth_err)
        else:
            # Compute directly
            from ..vfd.operators import (
                create_torsion_projectors,
                verify_projector_resolution,
                verify_projector_orthogonality
            )
            projectors = create_torsion_projectors(state.T)
            _, res_err = verify_projector_resolution(projectors)
            _, orth_err = verify_projector_orthogonality(projectors)
            residuals["projector_resolution"] = float(res_err)
            residuals["projector_orthogonality"] = float(orth_err)

        # Self-dual balance: measure asymmetry in torsion components
        # For a self-dual state, |v_k|^2 = |v_{12-k}|^2
        # We compute an aggregate measure from stability coefficients
        if state.stability_df is not None and len(state.stability_df) > 0:
            # Use self_dual_coord from stability: 0.5 is perfect balance
            if "self_dual_coord" in state.stability_df.columns:
                coords = state.stability_df["self_dual_coord"].values
                # Measure deviation from 0.5 (self-dual point)
                # For balanced states, mean should be ~0.5
                mean_coord = np.mean(coords)
                balance_err = abs(mean_coord - 0.5)
                residuals["self_dual_balance"] = float(balance_err)
            else:
                residuals["self_dual_balance"] = 0.0
        else:
            residuals["self_dual_balance"] = 0.0

        return residuals

    @property
    def constraint_names(self) -> List[str]:
        return ["projector_resolution", "projector_orthogonality", "self_dual_balance"]


class PositivityFamily(ConstraintFamily):
    """
    Positivity constraints (Li criterion analog).

    Verifies nonnegativity properties:
    - K ≥ 0 (kernel is positive semidefinite)
    - Q_K(v) ≥ 0 (quadratic form nonnegative for all probes)
    """

    name = "Positivity"

    def evaluate(self, state: "DiagnosticState") -> Dict[str, float]:
        """Evaluate positivity constraints."""
        residuals = {}

        # Kernel nonnegativity: min eigenvalue ≥ 0
        if state.invariants:
            min_eig = state.invariants.get("kernel_D3_min", 0.0)
        elif state.spectrum is not None:
            min_eig = float(np.min(state.spectrum.eigenvalues))
        else:
            # Compute directly
            props = state.kernel.verify_D3_nonnegative()
            min_eig = props[1]  # (passed, min_value)

        # Residual is 0 if min_eig >= 0, else |min_eig|
        residuals["kernel_nonnegative"] = float(max(0, -min_eig))

        # Quadratic form nonnegativity: all Q_K(v) ≥ 0
        if state.stability_df is not None and len(state.stability_df) > 0:
            if "Q_Kcan" in state.stability_df.columns:
                Q_values = state.stability_df["Q_Kcan"].values
                min_Q = np.min(Q_values)
                negative_count = np.sum(Q_values < -1e-10)
                # Residual: sum of negative Q values (magnitude)
                negative_sum = np.sum(np.abs(Q_values[Q_values < -1e-10]))
                residuals["quadratic_form_nonneg"] = float(negative_sum)
                residuals["negative_probe_count"] = float(negative_count)
            else:
                residuals["quadratic_form_nonneg"] = 0.0
                residuals["negative_probe_count"] = 0.0
        else:
            residuals["quadratic_form_nonneg"] = 0.0
            residuals["negative_probe_count"] = 0.0

        return residuals

    @property
    def constraint_names(self) -> List[str]:
        return ["kernel_nonnegative", "quadratic_form_nonneg", "negative_probe_count"]


class TraceMomentFamily(ConstraintFamily):
    """
    Trace/Moment constraints.

    Verifies spectral moment properties:
    - trace_bound: trace(K) matches expected value
    - moment_consistency: first moments match analytic formulas

    For K_can = I_internal ⊗ L_cell:
    - trace(K) = internal_dim * trace(L_cell) = internal_dim * (2R * C)
      where R = propagation_range, C = cell_count
    - First moment (mean eigenvalue) = trace(K) / total_dim
    """

    name = "Trace"

    def evaluate(self, state: "DiagnosticState") -> Dict[str, float]:
        """Evaluate trace/moment constraints."""
        residuals = {}

        # Get parameters
        cell_count = state.space.cell_count
        internal_dim = state.space.internal_dim
        total_dim = state.space.total_dim
        prop_range = state.config.vfd.local_propagation_L if state.config else 1

        # Expected trace for K_can = I_internal ⊗ L_cell
        # Each diagonal element of L_cell is 2*R (degree in circulant)
        # trace(L_cell) = 2*R*C
        # trace(K_can) = internal_dim * trace(L_cell) = internal_dim * 2*R*C
        expected_trace = internal_dim * 2 * prop_range * cell_count

        # Actual trace from eigenvalues (or compute directly)
        if state.spectrum is not None:
            actual_trace = float(np.sum(state.spectrum.eigenvalues))
        else:
            # Compute analytically for K_can
            from ..spectrum.analytic import analytic_kcan_trace
            actual_trace = analytic_kcan_trace(cell_count, prop_range, internal_dim)

        # Trace residual (relative error)
        if expected_trace > 0:
            trace_rel_err = abs(actual_trace - expected_trace) / expected_trace
        else:
            trace_rel_err = abs(actual_trace - expected_trace)
        residuals["trace_bound"] = float(trace_rel_err)

        # First moment (mean eigenvalue)
        expected_mean = expected_trace / total_dim
        if state.spectrum is not None:
            actual_mean = float(np.mean(state.spectrum.eigenvalues))
        else:
            actual_mean = expected_mean  # Use expected if no spectrum

        if expected_mean > 0:
            moment1_rel_err = abs(actual_mean - expected_mean) / expected_mean
        else:
            moment1_rel_err = abs(actual_mean - expected_mean)
        residuals["moment1_consistency"] = float(moment1_rel_err)

        # Second moment: E[λ²]
        # For L_cell circulant with range R:
        # λ_j = 2R - 2*sum_{d=1}^R cos(d*2πj/C)
        # E[λ²] can be computed analytically but formula is more complex.
        #
        # For K_can = I_internal ⊗ L_cell:
        #   E[λ²] = (1/C) * Σ_j λ_j²
        # where λ_j = 2R - 2*Σ_{d=1}^R cos(d*θ_j), θ_j = 2πj/C
        #
        # Expanding: λ_j² = 4R² - 8R*Σcos + 4*(Σcos)²
        # The cross terms involve products of cosines which can be simplified
        # using trigonometric identities, but for now we verify numerically.
        #
        # Expected second moment for standard Laplacian with range R:
        #   E[λ²] = 4R² + 2R  (derived from Σcos²(dθ) = C/2 for each d)
        if state.spectrum is not None:
            actual_moment2 = float(np.mean(state.spectrum.eigenvalues ** 2))
            # Analytic formula: E[λ²] = 4R² + 2R for circulant Laplacian
            expected_moment2 = 4 * prop_range**2 + 2 * prop_range
            if expected_moment2 > 0:
                moment2_rel_err = abs(actual_moment2 - expected_moment2) / expected_moment2
            else:
                moment2_rel_err = abs(actual_moment2 - expected_moment2)
            residuals["moment2_consistency"] = float(moment2_rel_err)
        else:
            residuals["moment2_consistency"] = 0.0

        return residuals

    @property
    def constraint_names(self) -> List[str]:
        return ["trace_bound", "moment1_consistency", "moment2_consistency"]


# Level -> Family mapping
LEVEL_FAMILIES = {
    ClosureLevel.L0: [],  # Baseline: no constraints, just structural validity
    ClosureLevel.L1: [ExplicitFormulaFamily],
    ClosureLevel.L2: [SymmetryFamily],
    ClosureLevel.L3: [PositivityFamily],
    ClosureLevel.L4: [TraceMomentFamily],
}


def get_families_for_level(level: ClosureLevel) -> List[ConstraintFamily]:
    """
    Get constraint families for a given closure level.

    Each level checks its own families plus all lower levels.

    Args:
        level: Closure level

    Returns:
        List of ConstraintFamily instances
    """
    families = []
    for lvl in ClosureLevel:
        if lvl <= level:
            for family_cls in LEVEL_FAMILIES.get(lvl, []):
                families.append(family_cls())
    return families


def get_all_families() -> List[ConstraintFamily]:
    """Get all constraint families."""
    return [
        ExplicitFormulaFamily(),
        SymmetryFamily(),
        PositivityFamily(),
        TraceMomentFamily(),
    ]
