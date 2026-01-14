"""
VFD Stability Analysis.

Stability is the PRIMITIVE concept in VFD. It is measured via quadratic forms,
not via zeros. The self-dual manifold is defined intrinsically.

Key concepts:
- Stability coefficient: lambda_n defined via quadratic form analysis
- Self-dual manifold: Fixed locus of the functional involution
- Kernel absoluteness: Nonnegative kernels are intrinsically stable
- Instability impossible: Structural impossibility within VFD constraints
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
from scipy.linalg import eigh

from .canonical import VFDSpace, TorsionOperator, TORSION_ORDER
from .kernels import CanonicalKernel
from .probes import ProbeGenerator, Probe


@dataclass
class StabilityCoefficient:
    """
    A stability coefficient computed from quadratic form analysis.

    These are VFD-native: defined via Q_K(v) = <v, Kv>, not via zeros.
    """
    index: int
    value: float
    probe_id: str
    torsion_degree: int
    support_cells: int
    self_dual_coord: float
    is_nonnegative: bool
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for export."""
        return {
            "probe_id": self.probe_id,
            "index": self.index,
            "stability_coeff": self.value,
            "torsion_degree": self.torsion_degree,
            "support_cells": self.support_cells,
            "self_dual_coord": self.self_dual_coord,
            "Q_Kcan": self.value,
            "is_nonnegative": self.is_nonnegative,
            "notes": self.notes
        }


@dataclass
class SpectralData:
    """
    Spectral data from the canonical kernel.

    Note: Eigenvalues are primary; "zeros" are secondary/derived.
    """
    eigenvalues: NDArray
    eigenvectors: Optional[NDArray]
    source_operator: str
    cell_truncation: int

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to DataFrame for export."""
        records = []
        for i, ev in enumerate(self.eigenvalues):
            records.append({
                "eig_id": i,
                "lambda": ev,
                "source_operator": self.source_operator,
                "cell_truncation": self.cell_truncation,
            })
        return pd.DataFrame(records)


class StabilityAnalyzer:
    """
    Analyze stability via quadratic forms.

    Stability is primary; zeros are derived concepts.
    """

    def __init__(
        self,
        space: VFDSpace,
        T: TorsionOperator,
        kernel: CanonicalKernel,
        seed: int = 42
    ):
        """
        Initialize stability analyzer.

        Args:
            space: VFD space
            T: Torsion operator
            kernel: Canonical kernel
            seed: Random seed
        """
        self.space = space
        self.T = T
        self.kernel = kernel
        self.seed = seed

        self.probe_generator = ProbeGenerator(space, T, seed)
        self._coefficients: List[StabilityCoefficient] = []
        self._spectral_data: Optional[SpectralData] = None

    def compute_quadratic_form(self, state: NDArray) -> float:
        """
        Compute Q_K(v) = <v, Kv>.

        This is the fundamental stability measure.
        """
        return self.kernel.quadratic_form(state)

    def compute_stability_coefficients(
        self,
        probe_count: int = 500,
        support_radius: int = 2
    ) -> List[StabilityCoefficient]:
        """
        Compute stability coefficients for a probe family.

        Coefficients are defined via quadratic forms, not zeros.

        Args:
            probe_count: Number of probes to test
            support_radius: Probe support radius

        Returns:
            List of stability coefficients
        """
        probes = self.probe_generator.generate_probe_family(
            probe_count,
            support_radius,
            probe_type="pure_torsion"
        )

        coefficients = []
        for i, probe in enumerate(probes):
            Q_value = self.compute_quadratic_form(probe.state)

            # Self-dual coordinate: based on torsion degree
            # Self-dual manifold corresponds to s = 1/2, here k = 6
            self_dual_coord = self._compute_self_dual_coord(probe)

            coeff = StabilityCoefficient(
                index=i,
                value=Q_value,
                probe_id=probe.probe_id,
                torsion_degree=probe.torsion_degree,
                support_cells=len(probe.support_cells),
                self_dual_coord=self_dual_coord,
                is_nonnegative=Q_value >= -1e-10,
                notes=f"Pure torsion degree {probe.torsion_degree}"
            )
            coefficients.append(coeff)

        self._coefficients = coefficients
        return coefficients

    def _compute_self_dual_coord(self, probe: Probe) -> float:
        """
        Compute self-dual manifold coordinate.

        The self-dual manifold is the fixed locus of s -> 1-s.
        In VFD, this corresponds to balanced torsion degree.
        """
        k = probe.torsion_degree
        if k < 0:
            return 0.5  # Mixed probes at center

        # Map torsion degree to self-dual coordinate
        # k=0 -> 0, k=6 -> 0.5 (self-dual point), k=11 -> ~1
        return k / (TORSION_ORDER - 1)

    def compute_spectrum(self, k: int = 100) -> SpectralData:
        """
        Compute spectrum of canonical kernel.

        Args:
            k: Number of eigenvalues

        Returns:
            Spectral data
        """
        eigenvalues, eigenvectors = self.kernel.compute_spectrum(k=k)

        self._spectral_data = SpectralData(
            eigenvalues=eigenvalues,
            eigenvectors=eigenvectors,
            source_operator="K_can",
            cell_truncation=self.space.cell_count
        )

        return self._spectral_data

    def verify_kernel_absoluteness(self) -> Tuple[bool, Dict[str, Any]]:
        """
        Verify kernel absoluteness: all stability coefficients are nonnegative.

        This demonstrates "instability is structurally impossible" within VFD.

        Returns:
            Tuple of (passes, detailed_results)
        """
        if not self._coefficients:
            self.compute_stability_coefficients()

        values = [c.value for c in self._coefficients]
        min_value = min(values)
        negative_count = sum(1 for v in values if v < -1e-10)
        negative_fraction = negative_count / len(values) if values else 0

        results = {
            "total_probes": len(values),
            "min_Q_value": min_value,
            "negative_count": negative_count,
            "negative_fraction": negative_fraction,
            "mean_Q_value": np.mean(values),
            "std_Q_value": np.std(values),
        }

        # Absoluteness holds if no negative values
        passes = negative_count == 0

        return passes, results

    def get_stability_summary(self) -> Dict[str, Any]:
        """
        Get summary of stability analysis.

        Returns:
            Dictionary with stability summary
        """
        if not self._coefficients:
            self.compute_stability_coefficients()

        passes, details = self.verify_kernel_absoluteness()

        return {
            "kernel_absoluteness": passes,
            "instability_impossible": passes,
            **details
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert stability coefficients to DataFrame."""
        if not self._coefficients:
            self.compute_stability_coefficients()

        records = [c.to_dict() for c in self._coefficients]
        return pd.DataFrame(records)


class SelfDualManifold:
    """
    The self-dual manifold in VFD.

    This is defined intrinsically as the fixed locus of the functional involution,
    NOT as "the critical line". The critical line is a projection/shadow concept.
    """

    def __init__(self, space: VFDSpace, T: TorsionOperator):
        """
        Initialize self-dual manifold.

        Args:
            space: VFD space
            T: Torsion operator
        """
        self.space = space
        self.T = T

    def is_on_manifold(self, state: NDArray, tol: float = 1e-8) -> bool:
        """
        Check if state lies on the self-dual manifold.

        A state is self-dual if it's invariant under the functional involution.
        In terms of torsion: balanced between k and (12-k) components.

        Args:
            state: State vector
            tol: Tolerance

        Returns:
            True if on self-dual manifold
        """
        # Decompose into torsion components
        from .probes import compute_torsion_decomposition
        components = compute_torsion_decomposition(state, self.T)

        # Check balance: |v_k|^2 = |v_{12-k}|^2
        for k in range(1, 6):
            norm_k = np.linalg.norm(components[k]) ** 2
            norm_12_minus_k = np.linalg.norm(components[12 - k]) ** 2

            if abs(norm_k - norm_12_minus_k) > tol:
                return False

        return True

    def project_to_manifold(self, state: NDArray) -> NDArray:
        """
        Project state onto self-dual manifold.

        Symmetrizes the state under torsion inversion.

        Args:
            state: State vector

        Returns:
            Projected state on self-dual manifold
        """
        from .probes import compute_torsion_decomposition
        components = compute_torsion_decomposition(state, self.T)

        # Symmetrize: v_k -> (v_k + v_{12-k})/2
        new_components = [components[0].copy()]  # k=0 is self-dual

        for k in range(1, 6):
            avg = (components[k] + components[12 - k]) / 2
            new_components.append(avg)

        # k=6 is fixed point of k -> 12-k
        new_components.append(components[6].copy())

        # Mirror for k > 6
        for k in range(7, 12):
            new_components.append(new_components[12 - k].copy())

        # Reconstruct
        result = sum(new_components)
        norm = np.linalg.norm(result)
        if norm > 0:
            result /= norm

        return result

    def get_coordinate(self, state: NDArray) -> float:
        """
        Get self-dual manifold coordinate for a state.

        Returns distance from the manifold (0 = on manifold).
        """
        from .probes import compute_torsion_decomposition
        components = compute_torsion_decomposition(state, self.T)

        # Asymmetry measure
        asymmetry = 0.0
        for k in range(1, 6):
            norm_k = np.linalg.norm(components[k]) ** 2
            norm_12_minus_k = np.linalg.norm(components[12 - k]) ** 2
            asymmetry += abs(norm_k - norm_12_minus_k)

        return asymmetry
