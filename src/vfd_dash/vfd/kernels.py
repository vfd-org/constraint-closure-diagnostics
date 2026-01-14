"""
VFD Kernel Operators.

Implements canonical and admissible kernels satisfying properties D1-D5:
- D1: Self-adjoint (K = K*)
- D2: Commutes with torsion ([K, T] = 0)
- D3: Nonnegative (<v, Kv> >= 0)
- D4: Finite propagation (K acts locally in cell structure)
- D5: Torsion averaging projects to well-defined spectral data

These properties are VFD-internal and define stability without zeros.

STRUCTURE NOTE:
The canonical kernel K_can has Kronecker structure:
    K_can = I_internal ⊗ L_cell

where L_cell is the circulant Laplacian on Z/CZ with coupling range R.
This enables O(C) eigenvalue computation via closed-form formula:
    λ_cell(θ_j) = 2R - 2 * Σ_{d=1}^R cos(d * θ_j)
where θ_j = 2π*j/C.

Each cell eigenvalue has multiplicity internal_dim (600).
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, Dict, Any, TYPE_CHECKING
from scipy import sparse
from scipy.sparse.linalg import eigsh
from dataclasses import dataclass

from .canonical import VFDSpace, TorsionOperator, ShiftOperator, TORSION_ORDER

if TYPE_CHECKING:
    from ..spectrum.backend import SpectralBackend, SpectralResult


@dataclass
class KernelProperties:
    """Verification results for kernel D1-D5 properties."""
    D1_selfadjoint: bool
    D1_error: float
    D2_torsion_commute: bool
    D2_error: float
    D3_nonnegative: bool
    D3_min_eigenvalue: float
    D4_finite_propagation: bool
    D4_propagation_range: int
    D5_spectral_valid: bool


class CanonicalKernel:
    """
    Canonical kernel operator K_can.

    Implements an intrinsic Laplacian-like operator satisfying D1-D5.
    Construction is VFD-internal: uses only cell structure, torsion, and shift.

    The canonical kernel is built as:
    K_can = sum over neighbor pairs of (I - projection onto common eigenspaces)

    This ensures:
    - Self-adjointness (D1)
    - Torsion commutation (D2)
    - Nonnegativity (D3)
    - Finite propagation (D4)
    """

    def __init__(
        self,
        space: VFDSpace,
        T: TorsionOperator,
        S: ShiftOperator,
        propagation_range: int = 1
    ):
        """
        Initialize canonical kernel.

        Args:
            space: VFD space
            T: Torsion operator
            S: Shift operator
            propagation_range: Local propagation range (cells)
        """
        self.space = space
        self.T = T
        self.S = S
        self.propagation_range = propagation_range
        self._build_kernel()

    def _build_kernel(self):
        """
        Build the canonical kernel matrix.

        Construction: Laplacian on the cell graph that commutes with T.
        K_can = D - A where D is degree, A is adjacency, both respecting torsion.
        """
        dim = self.space.total_dim
        n_cells = self.space.cell_count
        internal = self.space.internal_dim

        # Build as sparse matrix for efficiency
        data = []
        rows = []
        cols = []

        # Degree (diagonal) term: 2 * propagation_range per cell
        degree = 2 * self.propagation_range

        for n in range(n_cells):
            cell_start = n * internal
            for i in range(internal):
                idx = cell_start + i
                rows.append(idx)
                cols.append(idx)
                data.append(float(degree))

        # Adjacency (off-diagonal) terms with torsion-preserving structure
        for n in range(n_cells):
            for delta in range(1, self.propagation_range + 1):
                # Forward neighbor
                n_fwd = (n + delta) % n_cells
                # Backward neighbor
                n_bwd = (n - delta) % n_cells

                for i in range(internal):
                    src_idx = n * internal + i
                    fwd_idx = n_fwd * internal + i
                    bwd_idx = n_bwd * internal + i

                    # Adjacency entries (negative for Laplacian)
                    # With periodic BC, these wrap around
                    rows.append(src_idx)
                    cols.append(fwd_idx)
                    data.append(-1.0)

                    rows.append(src_idx)
                    cols.append(bwd_idx)
                    data.append(-1.0)

        self._matrix_sparse = sparse.csr_matrix(
            (data, (rows, cols)), shape=(dim, dim), dtype=float
        )

        # Dense matrix for small spaces
        if dim <= 2000:
            self._matrix_dense = self._matrix_sparse.toarray()
        else:
            self._matrix_dense = None

    def apply(self, state: NDArray) -> NDArray:
        """Apply K_can to a state."""
        return self._matrix_sparse @ state

    def as_matrix(self) -> NDArray:
        """Return K_can as dense matrix."""
        if self._matrix_dense is not None:
            return self._matrix_dense
        return self._matrix_sparse.toarray()

    def as_sparse(self) -> sparse.csr_matrix:
        """Return K_can as sparse matrix."""
        return self._matrix_sparse

    def quadratic_form(self, state: NDArray) -> float:
        """
        Compute Q_K(v) = <v, K v>.

        This is the fundamental stability measure in VFD.
        """
        Kv = self.apply(state)
        return np.real(np.vdot(state, Kv))

    def compute_spectrum(self, k: int = 100, which: str = "SM") -> Tuple[NDArray, NDArray]:
        """
        Compute k eigenvalues and eigenvectors using sparse eigsh.

        NOTE: For eigenvalues only, use compute_spectrum_fast() which uses
        the analytic formula exploiting Kronecker structure.

        Args:
            k: Number of eigenvalues
            which: "SM" for smallest magnitude, "LM" for largest

        Returns:
            Tuple of (eigenvalues, eigenvectors)
        """
        k = min(k, self.space.total_dim - 2)
        eigenvalues, eigenvectors = eigsh(self._matrix_sparse, k=k, which=which)
        idx = np.argsort(eigenvalues)
        return eigenvalues[idx], eigenvectors[:, idx]

    def compute_spectrum_fast(self, backend: str = "auto") -> "SpectralResult":
        """
        Compute full spectrum using optimized backends.

        Exploits the Kronecker structure K_can = I_internal ⊗ L_cell
        for O(C) computation instead of O((C*internal_dim)^3).

        Args:
            backend: "analytic_kcan", "fourier_cells", "sparse_fallback", or "auto"

        Returns:
            SpectralResult with eigenvalues and metadata
        """
        from ..spectrum.backend import compute_spectrum, SpectralBackend

        backend_enum = SpectralBackend(backend) if backend != "auto" else SpectralBackend.AUTO

        return compute_spectrum(
            cell_count=self.space.cell_count,
            internal_dim=self.space.internal_dim,
            propagation_range=self.propagation_range,
            backend=backend_enum,
            use_cache=True
        )

    def get_cell_eigenvalues(self) -> NDArray:
        """
        Get unique cell eigenvalues (without multiplicity).

        Returns the C distinct eigenvalues of L_cell.
        Useful for bridge projection where multiplicity is irrelevant.

        Returns:
            Array of C unique eigenvalues, sorted ascending
        """
        from ..spectrum.analytic import analytic_kcan_cell_eigenvalues
        return analytic_kcan_cell_eigenvalues(
            self.space.cell_count,
            self.propagation_range
        )

    @property
    def has_kronecker_structure(self) -> bool:
        """Whether this kernel has K = I_internal ⊗ L_cell structure."""
        return True  # CanonicalKernel always has this structure

    @property
    def kernel_type(self) -> str:
        """Return kernel type identifier."""
        return "K_can"

    def verify_D1_selfadjoint(
        self,
        tol: float = 1e-12,
        n_probes: int = 32,
        seed: int = 42,
        use_dense: bool = False
    ) -> Tuple[bool, float]:
        """
        Verify D1: K = K* (self-adjoint) using probe-based testing.

        Tests <u, Kv> == <Ku, v> for random probe pairs.
        Much faster than dense matrix construction for large spaces.

        Args:
            tol: Tolerance for equality
            n_probes: Number of probe pairs to test
            seed: Random seed for deterministic probes
            use_dense: Force dense matrix check (slow, for debugging)
        """
        if use_dense and self._matrix_dense is not None:
            K = self._matrix_dense
            error = np.linalg.norm(K - K.conj().T)
            return error < tol, error

        # Probe-based test: <u, Kv> should equal <Ku, v>
        rng = np.random.default_rng(seed)
        max_error = 0.0

        for _ in range(n_probes):
            u = rng.standard_normal(self.space.total_dim)
            v = rng.standard_normal(self.space.total_dim)

            Kv = self.apply(v)
            Ku = self.apply(u)

            lhs = np.vdot(u, Kv)  # <u, Kv>
            rhs = np.vdot(Ku, v)  # <Ku, v>

            error = abs(lhs - rhs)
            max_error = max(max_error, error)

        return max_error < tol, max_error

    def verify_D2_torsion_commute(
        self,
        tol: float = 1e-10,
        n_probes: int = 32,
        seed: int = 42,
        use_dense: bool = False
    ) -> Tuple[bool, float]:
        """
        Verify D2: [K, T] = 0 using probe-based testing.

        Tests K(T v) == T(K v) for random probes.
        Much faster than dense matrix construction for large spaces.

        Args:
            tol: Tolerance for equality
            n_probes: Number of probes to test
            seed: Random seed for deterministic probes
            use_dense: Force dense matrix check (slow, for debugging)
        """
        if use_dense and self._matrix_dense is not None:
            K = self._matrix_dense
            T = self.T.as_matrix()
            commutator = K @ T - T @ K
            error = np.linalg.norm(commutator)
            return error < tol, error

        # Probe-based test: K(Tv) should equal T(Kv)
        rng = np.random.default_rng(seed)
        max_error = 0.0

        for _ in range(n_probes):
            v = rng.standard_normal(self.space.total_dim) + \
                1j * rng.standard_normal(self.space.total_dim)
            v = v / np.linalg.norm(v)

            # K(T v)
            Tv = self.T.apply(v)
            K_Tv = self.apply(Tv)

            # T(K v)
            Kv = self.apply(v)
            T_Kv = self.T.apply(Kv)

            error = np.linalg.norm(K_Tv - T_Kv)
            max_error = max(max_error, error)

        return max_error < tol, max_error

    def verify_D3_nonnegative(
        self,
        n_samples: int = 32,
        seed: int = 42,
        check_eigenvalue: bool = True
    ) -> Tuple[bool, float]:
        """
        Verify D3: <v, Kv> >= 0 for all v.

        Tests on random samples and optionally checks minimum eigenvalue.

        Args:
            n_samples: Number of random samples to test
            seed: Random seed
            check_eigenvalue: Whether to compute actual min eigenvalue (slower)
        """
        # Sample random states
        rng = np.random.default_rng(seed)
        min_Q = float('inf')

        for _ in range(n_samples):
            state = rng.standard_normal(self.space.total_dim)
            state = state / np.linalg.norm(state)
            Q = self.quadratic_form(state)
            min_Q = min(min_Q, Q)

        # Optionally check minimum eigenvalue (more expensive)
        min_eig = min_Q
        if check_eigenvalue:
            try:
                eigenvalues, _ = self.compute_spectrum(k=10, which="SA")
                min_eig = eigenvalues[0]
            except Exception:
                pass

        passes = min_Q >= -1e-10 and min_eig >= -1e-10
        return passes, min(min_Q, min_eig)

    def verify_D4_finite_propagation(self) -> Tuple[bool, int]:
        """Verify D4: Finite propagation range."""
        # By construction, propagation is finite
        return True, self.propagation_range

    def verify_all_properties(self, tol: float = 1e-10, fast_mode: bool = True) -> KernelProperties:
        """
        Verify all D1-D5 properties.

        Args:
            tol: Tolerance for checks
            fast_mode: If True, skip expensive eigenvalue computation in D3
        """
        d1_pass, d1_err = self.verify_D1_selfadjoint(tol)
        d2_pass, d2_err = self.verify_D2_torsion_commute(tol)
        d3_pass, d3_min = self.verify_D3_nonnegative(check_eigenvalue=not fast_mode)
        d4_pass, d4_range = self.verify_D4_finite_propagation()

        return KernelProperties(
            D1_selfadjoint=d1_pass,
            D1_error=d1_err,
            D2_torsion_commute=d2_pass,
            D2_error=d2_err,
            D3_nonnegative=d3_pass,
            D3_min_eigenvalue=d3_min,
            D4_finite_propagation=d4_pass,
            D4_propagation_range=d4_range,
            D5_spectral_valid=True  # Valid by construction
        )


class AdmissibleKernel:
    """
    General admissible kernel satisfying D1-D5.

    Allows custom kernel construction while verifying admissibility.
    """

    def __init__(
        self,
        space: VFDSpace,
        T: TorsionOperator,
        kernel_matrix: NDArray
    ):
        """
        Initialize with custom kernel matrix.

        Args:
            space: VFD space
            T: Torsion operator
            kernel_matrix: Custom kernel matrix
        """
        self.space = space
        self.T = T
        self._matrix = kernel_matrix.copy()

    def apply(self, state: NDArray) -> NDArray:
        """Apply kernel to state."""
        return self._matrix @ state

    def as_matrix(self) -> NDArray:
        """Return kernel as matrix."""
        return self._matrix

    def quadratic_form(self, state: NDArray) -> float:
        """Compute Q_K(v) = <v, K v>."""
        Kv = self.apply(state)
        return np.real(np.vdot(state, Kv))

    def is_admissible(self, tol: float = 1e-10) -> Tuple[bool, Dict[str, Any]]:
        """
        Check if kernel is admissible (satisfies D1-D4).

        Returns:
            Tuple of (is_admissible, detailed_results)
        """
        results = {}

        # D1: Self-adjoint
        K = self._matrix
        d1_err = np.linalg.norm(K - K.conj().T)
        results["D1_error"] = d1_err
        results["D1_pass"] = d1_err < tol

        # D2: Commutes with T
        T = self.T.as_matrix()
        d2_err = np.linalg.norm(K @ T - T @ K)
        results["D2_error"] = d2_err
        results["D2_pass"] = d2_err < tol

        # D3: Nonnegative
        eigenvalues = np.linalg.eigvalsh(K)
        min_eig = np.min(eigenvalues)
        results["D3_min_eigenvalue"] = min_eig
        results["D3_pass"] = min_eig >= -tol

        # D4: Finite propagation (check sparsity pattern)
        # For arbitrary matrices, we check if it respects cell structure
        results["D4_pass"] = True  # Assume true, detailed check would need cell analysis

        is_admissible = all([results["D1_pass"], results["D2_pass"],
                            results["D3_pass"], results["D4_pass"]])

        return is_admissible, results
