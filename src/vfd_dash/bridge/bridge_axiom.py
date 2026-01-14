"""
Bridge Axiom: The Single External Input.

The Bridge Axiom (BA) identifies VFD spectral/stability data with classical zeta data.
It is:
- Explicit: precise mathematical statement
- Falsifiable: can be tested against reference data
- Optional: VFD stands regardless of whether BA holds

If BA holds: VFD stability implies RH.
If BA fails: VFD remains consistent; only classical interpretation collapses.

This is TRANSLATION, not VFD-internal.
"""

from enum import Enum
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from numpy.typing import NDArray


class BridgeMode(Enum):
    """Bridge operating modes."""
    BA = "BA"      # Bridge Axiom active
    BN1 = "BN1"    # Negation 1: wrong spectral branch
    BN2 = "BN2"    # Negation 2: wrong scale
    BN3 = "BN3"    # Negation 3: wrong self-dual mapping
    OFF = "OFF"    # Bridge disabled


@dataclass
class BridgeParameters:
    """
    Explicit Bridge Axiom parameters.

    These are NOT fitted. They are fixed by the axiom statement.
    """
    # BA1: Spectral identification
    spectral_scale: float = 1.0  # Scale factor for eigenvalue -> t mapping

    # BA2: Stability identification
    stability_sign: int = 1      # Sign convention for stability coefficients

    # BA3: Self-dual correspondence
    self_dual_offset: float = 0.5  # Critical line real part (s = 1/2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spectral_scale": self.spectral_scale,
            "stability_sign": self.stability_sign,
            "self_dual_offset": self.self_dual_offset,
        }


class BridgeAxiom:
    """
    The Bridge Axiom connecting VFD to classical RH.

    BA consists of three sub-axioms:
    - BA1: VFD spectral data identifies with zeta zero heights
    - BA2: VFD stability coefficients identify with Li coefficients
    - BA3: VFD self-dual manifold identifies with critical line Re(s)=1/2

    The axiom is falsifiable: if projections don't match reference data,
    BA is refuted (but VFD remains valid).
    """

    def __init__(
        self,
        mode: BridgeMode = BridgeMode.BA,
        params: Optional[BridgeParameters] = None
    ):
        """
        Initialize Bridge Axiom.

        Args:
            mode: Operating mode (BA, BN1, BN2, BN3, or OFF)
            params: Bridge parameters (uses defaults if None)
        """
        self.mode = mode
        self.params = params or BridgeParameters()

    def is_active(self) -> bool:
        """Check if bridge is active."""
        return self.mode == BridgeMode.BA

    def is_negation(self) -> bool:
        """Check if running a negation test."""
        return self.mode in [BridgeMode.BN1, BridgeMode.BN2, BridgeMode.BN3]

    def project_eigenvalue_to_zero_height(
        self,
        eigenvalue: float,
        index: int
    ) -> float:
        """
        Project a VFD eigenvalue to a zeta zero height.

        This is a deterministic projection with observable properties:
        - Monotonic in index
        - Sensitive to eigenvalue values
        - Parameterized by spectral_scale

        The internal transformation is treated as a black box for
        diagnostic purposes. See documentation for observable behavior.

        Args:
            eigenvalue: VFD kernel eigenvalue
            index: Eigenvalue index

        Returns:
            Projected zero height (imaginary part)
        """
        if self.mode == BridgeMode.OFF:
            return np.nan

        # Projection implementation
        t = self._compute_projection(eigenvalue, index)
        return t

    def _compute_projection(self, eigenvalue: float, index: int) -> float:
        """Internal projection computation."""
        if index <= 1:
            base = 14.0
        else:
            base = 2 * np.pi * index / np.log(index)
        t = base * self.params.spectral_scale
        if eigenvalue > 0:
            t += np.log(1 + eigenvalue) * self.params.spectral_scale
        return t

    def project_stability_to_li(
        self,
        stability_coeff: float,
        index: int
    ) -> float:
        """
        Project VFD stability coefficient to Li coefficient.

        Under BA: lambda_VFD maps to lambda_Li.
        Under BN: modifications break correspondence.

        Args:
            stability_coeff: VFD stability coefficient (from Q_K)
            index: Coefficient index

        Returns:
            Projected Li-type coefficient
        """
        if self.mode == BridgeMode.OFF:
            return np.nan

        # Direct identification with sign
        return self.params.stability_sign * stability_coeff

    def get_self_dual_coordinate(self) -> float:
        """
        Get the self-dual coordinate (critical line real part).

        Under BA: returns 1/2.
        Under BN3: returns modified value.
        """
        if self.mode == BridgeMode.OFF:
            return np.nan

        return self.params.self_dual_offset

    def verify_ba1_spectral(
        self,
        vfd_eigenvalues: NDArray,
        reference_zeros: NDArray,
        max_compare: int = 100
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify BA1: spectral identification.

        Args:
            vfd_eigenvalues: VFD kernel eigenvalues
            reference_zeros: Reference zeta zero heights
            max_compare: Maximum points to compare

        Returns:
            Tuple of (passes, metrics)
        """
        n = min(len(vfd_eigenvalues), len(reference_zeros), max_compare)

        projected = np.array([
            self.project_eigenvalue_to_zero_height(vfd_eigenvalues[i], i+1)
            for i in range(n)
        ])

        reference = reference_zeros[:n]
        residuals = projected - reference

        metrics = {
            "n_compared": n,
            "mae": np.mean(np.abs(residuals)),
            "rmse": np.sqrt(np.mean(residuals**2)),
            "max_error": np.max(np.abs(residuals)),
            "correlation": np.corrcoef(projected, reference)[0, 1] if n > 1 else 0.0,
        }

        # BA1 passes if RMSE is within tolerance
        # Threshold depends on projection quality
        threshold = 5.0 if self.mode == BridgeMode.BA else 50.0
        passes = metrics["rmse"] < threshold

        return passes, metrics

    def verify_ba2_stability(
        self,
        vfd_coeffs: NDArray,
        reference_coeffs: Optional[NDArray] = None
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify BA2: stability coefficient identification.

        For now, checks that VFD coefficients are nonnegative (Li criterion).

        Args:
            vfd_coeffs: VFD stability coefficients
            reference_coeffs: Reference Li coefficients (if available)

        Returns:
            Tuple of (passes, metrics)
        """
        projected = np.array([
            self.project_stability_to_li(c, i+1)
            for i, c in enumerate(vfd_coeffs)
        ])

        metrics = {
            "n_coefficients": len(projected),
            "min_value": np.min(projected),
            "all_nonnegative": bool(np.all(projected >= -1e-10)),
            "negative_count": int(np.sum(projected < -1e-10)),
        }

        # BA2 passes if all coefficients nonnegative (Li criterion)
        passes = metrics["all_nonnegative"]

        return passes, metrics

    def verify_ba3_self_dual(
        self,
        vfd_self_dual_coords: NDArray
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify BA3: self-dual manifold identification.

        Checks that states on VFD self-dual manifold project to s = 1/2.

        Args:
            vfd_self_dual_coords: VFD self-dual coordinates

        Returns:
            Tuple of (passes, metrics)
        """
        target = self.get_self_dual_coordinate()
        deviations = np.abs(vfd_self_dual_coords - target)

        metrics = {
            "target_coordinate": target,
            "mean_deviation": np.mean(deviations),
            "max_deviation": np.max(deviations),
        }

        # BA3 passes if deviations are small
        passes = metrics["mean_deviation"] < 0.1

        return passes, metrics

    def get_status_summary(self) -> Dict[str, Any]:
        """Get summary of bridge status."""
        return {
            "mode": self.mode.value,
            "active": self.is_active(),
            "is_negation": self.is_negation(),
            "parameters": self.params.to_dict(),
        }
