"""
VFD Probe States.

Probes are test states used to verify stability properties.
Each probe has definite torsion degree and finite support.

Stability is measured via Q_K(v) = <v, K v> on probe families.
"""

import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Iterator
from dataclasses import dataclass

from .canonical import VFDSpace, TorsionOperator, TORSION_ORDER, OMEGA


@dataclass
class Probe:
    """
    A probe state with definite properties.

    Attributes:
        state: The probe state vector
        torsion_degree: Torsion degree k (T v = omega^k v on support)
        support_cells: List of cell indices where probe is supported
        probe_id: Unique identifier
        metadata: Additional probe information
    """
    state: NDArray
    torsion_degree: int
    support_cells: List[int]
    probe_id: str
    metadata: dict

    def norm(self) -> float:
        """Compute L2 norm."""
        return np.linalg.norm(self.state)

    def normalize(self) -> "Probe":
        """Return normalized copy."""
        n = self.norm()
        if n > 0:
            return Probe(
                state=self.state / n,
                torsion_degree=self.torsion_degree,
                support_cells=self.support_cells,
                probe_id=self.probe_id,
                metadata=self.metadata
            )
        return self


class ProbeGenerator:
    """
    Generate probe families for stability testing.

    Probes are finite-support states with definite torsion degree.
    """

    def __init__(
        self,
        space: VFDSpace,
        T: TorsionOperator,
        seed: int = 42
    ):
        """
        Initialize probe generator.

        Args:
            space: VFD space
            T: Torsion operator
            seed: Random seed for reproducibility
        """
        self.space = space
        self.T = T
        self.rng = np.random.default_rng(seed)
        self._probe_counter = 0

    def generate_pure_torsion_probe(
        self,
        torsion_degree: int,
        center_cell: int,
        support_radius: int = 1
    ) -> Probe:
        """
        Generate a probe with pure torsion degree k.

        The probe is supported on cells [center - radius, center + radius].
        Within the internal space, it lives in the omega^k eigenspace.

        Args:
            torsion_degree: Target torsion degree k
            center_cell: Center cell of support
            support_radius: Radius of support in cells

        Returns:
            Probe with specified properties
        """
        k = torsion_degree % TORSION_ORDER
        state = np.zeros(self.space.total_dim, dtype=complex)

        # Support cells
        support_cells = []
        for delta in range(-support_radius, support_radius + 1):
            cell = (center_cell + delta) % self.space.cell_count
            support_cells.append(cell)

        # For each cell, populate only the omega^k eigenspace components
        # In our orbit structure, position j within orbit has eigenvalue omega^j
        for cell in support_cells:
            cell_start = cell * self.space.internal_dim

            for orbit in range(self.space.orbit_count):
                # Position k within orbit is in omega^k eigenspace
                internal_idx = orbit * self.space.orbit_size + k
                full_idx = cell_start + internal_idx

                # Random amplitude
                amplitude = self.rng.standard_normal() + 1j * self.rng.standard_normal()
                state[full_idx] = amplitude

        # Normalize
        norm = np.linalg.norm(state)
        if norm > 0:
            state /= norm

        self._probe_counter += 1
        probe_id = f"probe_{self._probe_counter:06d}_k{k}_c{center_cell}_r{support_radius}"

        return Probe(
            state=state,
            torsion_degree=k,
            support_cells=support_cells,
            probe_id=probe_id,
            metadata={
                "center_cell": center_cell,
                "support_radius": support_radius,
                "type": "pure_torsion"
            }
        )

    def generate_random_probe(
        self,
        support_radius: int = 1,
        center_cell: Optional[int] = None
    ) -> Probe:
        """
        Generate a random probe with finite support.

        This probe may have mixed torsion components.

        Args:
            support_radius: Radius of support
            center_cell: Center cell (random if None)

        Returns:
            Random probe
        """
        if center_cell is None:
            center_cell = self.rng.integers(0, self.space.cell_count)

        state = np.zeros(self.space.total_dim, dtype=complex)
        support_cells = []

        for delta in range(-support_radius, support_radius + 1):
            cell = (center_cell + delta) % self.space.cell_count
            support_cells.append(cell)

            cell_start = cell * self.space.internal_dim
            cell_end = cell_start + self.space.internal_dim

            # Random amplitudes in this cell
            state[cell_start:cell_end] = (
                self.rng.standard_normal(self.space.internal_dim) +
                1j * self.rng.standard_normal(self.space.internal_dim)
            )

        # Normalize
        norm = np.linalg.norm(state)
        if norm > 0:
            state /= norm

        self._probe_counter += 1
        probe_id = f"probe_{self._probe_counter:06d}_random_c{center_cell}_r{support_radius}"

        return Probe(
            state=state,
            torsion_degree=-1,  # Mixed
            support_cells=support_cells,
            probe_id=probe_id,
            metadata={
                "center_cell": center_cell,
                "support_radius": support_radius,
                "type": "random"
            }
        )

    def generate_probe_family(
        self,
        count: int,
        support_radius: int = 1,
        probe_type: str = "pure_torsion"
    ) -> List[Probe]:
        """
        Generate a family of probes for systematic testing.

        Args:
            count: Number of probes to generate
            support_radius: Support radius for each probe
            probe_type: "pure_torsion" or "random"

        Returns:
            List of probes
        """
        probes = []

        if probe_type == "pure_torsion":
            # Distribute probes across torsion degrees and cells
            probes_per_degree = count // TORSION_ORDER + 1

            for k in range(TORSION_ORDER):
                for i in range(probes_per_degree):
                    if len(probes) >= count:
                        break
                    center = self.rng.integers(0, self.space.cell_count)
                    probe = self.generate_pure_torsion_probe(k, center, support_radius)
                    probes.append(probe)

        elif probe_type == "random":
            for _ in range(count):
                probe = self.generate_random_probe(support_radius)
                probes.append(probe)

        return probes[:count]

    def verify_torsion_degree(self, probe: Probe, tol: float = 1e-10) -> bool:
        """
        Verify that probe has claimed torsion degree.

        For pure torsion probe with degree k:
        T v = omega^k v

        Args:
            probe: Probe to verify
            tol: Tolerance

        Returns:
            True if verification passes
        """
        if probe.torsion_degree < 0:
            return True  # Mixed probes don't have definite degree

        k = probe.torsion_degree
        Tv = self.T.apply(probe.state)
        expected = (OMEGA ** k) * probe.state

        error = np.linalg.norm(Tv - expected)
        return error < tol


def compute_torsion_decomposition(
    state: NDArray,
    T: TorsionOperator
) -> List[NDArray]:
    """
    Decompose state into torsion eigenspaces.

    Returns v = sum_k v_k where T v_k = omega^k v_k.

    Args:
        state: State to decompose
        T: Torsion operator

    Returns:
        List of components v_0, ..., v_11
    """
    components = []

    for k in range(TORSION_ORDER):
        # Project onto omega^k eigenspace
        # P_k = (1/12) sum_j omega^{-kj} T^j
        v_k = np.zeros_like(state)

        for j in range(TORSION_ORDER):
            phase = OMEGA ** (-k * j)
            T_j_state = T.apply_power(state, j)
            v_k += phase * T_j_state

        v_k /= TORSION_ORDER
        components.append(v_k)

    return components
