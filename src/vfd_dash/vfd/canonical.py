"""
VFD Canonical Framework: Core Space and Operators.

Implements the fundamental VFD structure:
- Extended Hilbert space H_ext = ell^2(Z) tensor C^600
- Torsion operator T of order 12
- Shift operator S with cocycle twist
- Weyl commutation relation: T S T^{-1} = omega S

All constructions are VFD-internal. No classical number theory references.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
from dataclasses import dataclass


# VFD constants
TORSION_ORDER = 12
OMEGA = np.exp(2j * np.pi / TORSION_ORDER)  # 12th root of unity


@dataclass
class VFDSpace:
    """
    VFD Extended Hilbert Space (truncated for computation).

    The full space is H_ext = ell^2(Z) tensor C^{internal_dim}.
    We truncate to cell_count cells for computation.

    Attributes:
        cell_count: Number of cells in truncated space
        internal_dim: Dimension of internal space (default 600 = 50 orbits * 12)
        orbit_count: Number of orbits (default 50)
        orbit_size: Size of each orbit (default 12)
        periodic: Use periodic boundary conditions
    """

    cell_count: int
    internal_dim: int = 600
    orbit_count: int = 50
    orbit_size: int = 12
    periodic: bool = True

    def __post_init__(self):
        """Validate configuration."""
        assert self.internal_dim == self.orbit_count * self.orbit_size, (
            f"internal_dim ({self.internal_dim}) must equal "
            f"orbit_count * orbit_size ({self.orbit_count * self.orbit_size})"
        )
        self.total_dim = self.cell_count * self.internal_dim

        # Validate orbit_size for Weyl relation compatibility
        # The Weyl relation T S T^{-1} = ω S requires ω^{orbit_size} = 1
        # Since ω = exp(2πi/12), this requires orbit_size = 12
        if self.orbit_size != TORSION_ORDER:
            import warnings
            warnings.warn(
                f"orbit_size={self.orbit_size} != TORSION_ORDER={TORSION_ORDER}. "
                f"The Weyl relation T S T^{{-1}} = ω S will NOT hold. "
                f"Set orbit_size=12 for correct VFD structure.",
                UserWarning
            )

    def cell_slice(self, n: int) -> slice:
        """Get slice for cell n in the full space."""
        n_mod = n % self.cell_count if self.periodic else n
        start = n_mod * self.internal_dim
        return slice(start, start + self.internal_dim)

    def random_state(self, seed: Optional[int] = None) -> NDArray:
        """Generate random state in the space."""
        rng = np.random.default_rng(seed)
        state = rng.standard_normal(self.total_dim) + 1j * rng.standard_normal(self.total_dim)
        return state / np.linalg.norm(state)


class TorsionOperator:
    """
    Torsion operator T of order 12.

    Acts on the internal space C^600 as block-diagonal with
    omega^j on the j-th component of each orbit.

    Properties (VFD Axiom 1.1):
    - T^12 = I (torsion order 12)
    - T is unitary
    - Eigenvalues are {omega^q : q = 0, ..., 11}
    """

    def __init__(self, space: VFDSpace):
        """
        Initialize torsion operator.

        Args:
            space: VFD space configuration
        """
        self.space = space
        self._build_operator()

    def _build_operator(self):
        """Build the internal torsion action."""
        # Internal torsion: diagonal matrix on C^{internal_dim}
        # Each orbit position j gets eigenvalue omega^j
        internal_phases = np.array([
            OMEGA ** (j % self.space.orbit_size)
            for orbit in range(self.space.orbit_count)
            for j in range(self.space.orbit_size)
        ])
        self._internal_T = internal_phases

        # Full torsion acts cell-by-cell
        self._full_T = np.tile(internal_phases, self.space.cell_count)

    def apply(self, state: NDArray) -> NDArray:
        """Apply T to a state."""
        return self._full_T * state

    def apply_power(self, state: NDArray, power: int) -> NDArray:
        """Apply T^power to a state."""
        phases = self._full_T ** power
        return phases * state

    def apply_inverse(self, state: NDArray) -> NDArray:
        """Apply T^{-1} to a state."""
        return np.conj(self._full_T) * state

    def as_matrix(self) -> NDArray:
        """Return T as explicit diagonal matrix (for small spaces)."""
        return np.diag(self._full_T)

    def verify_order(self, tol: float = 1e-12) -> bool:
        """Verify T^12 = I."""
        phases_12 = self._full_T ** 12
        return np.allclose(phases_12, 1.0, atol=tol)

    def get_eigenvalue(self, q: int) -> complex:
        """Get eigenvalue omega^q."""
        return OMEGA ** (q % TORSION_ORDER)


class ShiftOperator:
    """
    Shift operator S satisfying the Weyl relation.

    The Weyl relation T S T^{-1} = omega S requires S to map
    the omega^j eigenspace of T to the omega^{j+1} eigenspace.

    Implementation:
    - S shifts cells: n -> n+1
    - S shifts within orbits: position j -> j+1 (mod 12)

    This ensures T S T^{-1} = omega S exactly.
    """

    def __init__(self, space: VFDSpace, direction: int = 1):
        """
        Initialize shift operator.

        Args:
            space: VFD space configuration
            direction: +1 for forward shift, -1 for backward
        """
        self.space = space
        self.direction = direction
        self._build_permutation()

    def _build_permutation(self):
        """
        Build the internal permutation for Weyl relation.

        Within each orbit, shift position j -> j+1 (mod orbit_size).
        This maps omega^j eigenspace to omega^{j+1} eigenspace.
        """
        orbit_size = self.space.orbit_size
        orbit_count = self.space.orbit_count

        # Build permutation of internal indices
        # For each orbit, j -> (j + direction) mod orbit_size
        self._internal_perm = np.zeros(self.space.internal_dim, dtype=int)

        for orbit in range(orbit_count):
            for j in range(orbit_size):
                src_idx = orbit * orbit_size + j
                dst_j = (j + self.direction) % orbit_size
                dst_idx = orbit * orbit_size + dst_j
                self._internal_perm[src_idx] = dst_idx

        # Inverse permutation
        self._internal_perm_inv = np.argsort(self._internal_perm)

    def apply(self, state: NDArray) -> NDArray:
        """
        Apply S to a state.

        S shifts cells (n -> n+1) and shifts within orbits (j -> j+1).

        Args:
            state: Input state vector

        Returns:
            Shifted state
        """
        result = np.zeros_like(state)

        for n in range(self.space.cell_count):
            src_cell = n
            dst_cell = (n + self.direction) % self.space.cell_count

            src_slice = self.space.cell_slice(src_cell)
            dst_slice = self.space.cell_slice(dst_cell)

            # Get source cell data
            src_data = state[src_slice]

            # Permute internal indices: j -> j+1
            permuted_data = src_data[self._internal_perm_inv]

            result[dst_slice] = permuted_data

        return result

    def apply_inverse(self, state: NDArray) -> NDArray:
        """Apply S^{-1}."""
        result = np.zeros_like(state)

        for n in range(self.space.cell_count):
            src_cell = n
            dst_cell = (n - self.direction) % self.space.cell_count

            src_slice = self.space.cell_slice(src_cell)
            dst_slice = self.space.cell_slice(dst_cell)

            src_data = state[src_slice]

            # Inverse permutation: j -> j-1
            permuted_data = src_data[self._internal_perm]

            result[dst_slice] = permuted_data

        return result

    def apply_power(self, state: NDArray, power: int) -> NDArray:
        """Apply S^power to a state."""
        if power == 0:
            return state.copy()

        result = state.copy()
        op = self.apply if power > 0 else self.apply_inverse
        for _ in range(abs(power)):
            result = op(result)
        return result

    def as_matrix(self) -> NDArray:
        """Return S as explicit matrix (for small spaces)."""
        dim = self.space.total_dim
        S_matrix = np.zeros((dim, dim), dtype=complex)

        for n in range(self.space.cell_count):
            dst_cell = (n + self.direction) % self.space.cell_count
            src_slice = self.space.cell_slice(n)
            dst_slice = self.space.cell_slice(dst_cell)

            for j in range(self.space.internal_dim):
                dst_j = self._internal_perm_inv[j]
                S_matrix[dst_slice.start + dst_j, src_slice.start + j] = 1.0

        return S_matrix


def verify_weyl_relation(
    T: TorsionOperator,
    S: ShiftOperator,
    test_states: int = 10,
    seed: int = 42,
    tol: float = 1e-10
) -> Tuple[bool, float]:
    """
    Verify the Weyl commutation relation: T S T^{-1} = omega S.

    Args:
        T: Torsion operator
        S: Shift operator
        test_states: Number of random states to test
        seed: Random seed
        tol: Tolerance for equality

    Returns:
        Tuple of (passes, max_error)
    """
    space = T.space
    rng = np.random.default_rng(seed)
    max_error = 0.0

    for _ in range(test_states):
        # Random state
        state = rng.standard_normal(space.total_dim) + 1j * rng.standard_normal(space.total_dim)
        state /= np.linalg.norm(state)

        # Left side: T S T^{-1} |state>
        lhs = T.apply(S.apply(T.apply_inverse(state)))

        # Right side: omega S |state>
        rhs = OMEGA * S.apply(state)

        error = np.linalg.norm(lhs - rhs)
        max_error = max(max_error, error)

    return max_error < tol, max_error


def verify_torsion_order(T: TorsionOperator, tol: float = 1e-12) -> Tuple[bool, float]:
    """
    Verify T^12 = I.

    Args:
        T: Torsion operator
        tol: Tolerance

    Returns:
        Tuple of (passes, max_error)
    """
    phases_12 = T._full_T ** 12
    max_error = np.max(np.abs(phases_12 - 1.0))
    return max_error < tol, max_error


def debug_weyl_on_basis(
    T: TorsionOperator,
    S: ShiftOperator,
    verbose: bool = True
) -> dict:
    """
    Debug Weyl relation by testing on basis vectors.

    Tests T S T^{-1} e_i = ω S e_i for each basis vector.
    Returns detailed diagnostics showing where failures occur.

    Args:
        T: Torsion operator
        S: Shift operator
        verbose: Print detailed output

    Returns:
        Dictionary with debug info
    """
    space = T.space
    orbit_size = space.orbit_size

    results = {
        "orbit_size": orbit_size,
        "torsion_order": TORSION_ORDER,
        "omega": OMEGA,
        "omega_orbit_size": OMEGA ** orbit_size,
        "weyl_compatible": np.isclose(OMEGA ** orbit_size, 1.0),
        "max_error": 0.0,
        "worst_basis_idx": -1,
        "errors_by_orbit_position": {},
    }

    if verbose:
        print(f"=== Weyl Debug: orbit_size={orbit_size}, torsion_order={TORSION_ORDER} ===")
        print(f"ω = {OMEGA:.4f}")
        print(f"ω^{orbit_size} = {OMEGA**orbit_size:.4f} (should be 1 for Weyl to hold)")
        print(f"Weyl compatible: {results['weyl_compatible']}")
        print()

    # Test first orbit in first cell
    for j in range(orbit_size):
        e_j = np.zeros(space.total_dim, dtype=complex)
        e_j[j] = 1.0

        # LHS: T S T^{-1} e_j
        lhs = T.apply(S.apply(T.apply_inverse(e_j)))

        # RHS: ω S e_j
        rhs = OMEGA * S.apply(e_j)

        error = np.linalg.norm(lhs - rhs)
        results["errors_by_orbit_position"][j] = error

        if error > results["max_error"]:
            results["max_error"] = error
            results["worst_basis_idx"] = j

        if verbose:
            lhs_idx = np.argmax(np.abs(lhs))
            rhs_idx = np.argmax(np.abs(rhs))
            lhs_phase = lhs[lhs_idx] if np.abs(lhs[lhs_idx]) > 1e-10 else 0
            rhs_phase = rhs[rhs_idx] if np.abs(rhs[rhs_idx]) > 1e-10 else 0

            status = "OK" if error < 1e-10 else "FAIL"
            print(f"  e_{j}: LHS@{lhs_idx}={lhs_phase:.4f}, RHS@{rhs_idx}={rhs_phase:.4f}, "
                  f"error={error:.6f} [{status}]")

    if verbose:
        print()
        print(f"Worst error: {results['max_error']:.6f} at basis e_{results['worst_basis_idx']}")
        if not results["weyl_compatible"]:
            print()
            print("ROOT CAUSE: orbit_size does not match torsion_order.")
            print(f"  When S wraps position {orbit_size-1} → 0, we get:")
            print(f"    LHS phase = ω^{{-{orbit_size-1}}} × ω^0 = ω^{{1-{orbit_size}}} = {OMEGA**(1-orbit_size):.4f}")
            print(f"    RHS phase = ω = {OMEGA:.4f}")
            print(f"  These differ because ω^{orbit_size} = {OMEGA**orbit_size:.4f} ≠ 1")
            print()
            print(f"FIX: Set orbit_size = {TORSION_ORDER}")

    return results
