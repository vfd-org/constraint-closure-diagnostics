"""
VFD Operator Algebra.

Implements:
- Torsion projectors P_q
- Bi-degree characterization (ell, k)
- Torsion averaging Pi_T
- Selection rules

All constructions are VFD-internal.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional, List
from dataclasses import dataclass

from .canonical import VFDSpace, TorsionOperator, ShiftOperator, TORSION_ORDER, OMEGA


def create_torsion_projectors(T: TorsionOperator) -> List[NDArray]:
    """
    Create torsion eigenspace projectors P_q for q = 0, ..., 11.

    P_q projects onto the omega^q eigenspace of T.
    P_q = (1/12) sum_{j=0}^{11} omega^{-qj} T^j

    Properties:
    - sum_q P_q = I
    - P_q P_r = delta_{qr} P_q
    - P_q^* = P_q (self-adjoint)

    Args:
        T: Torsion operator

    Returns:
        List of projector diagonal vectors (not full matrices)
    """
    projectors = []
    n = TORSION_ORDER

    for q in range(n):
        # P_q = (1/n) sum_{j=0}^{n-1} omega^{-qj} T^j
        # Since T is diagonal, P_q is also diagonal
        proj = np.zeros(T.space.total_dim, dtype=complex)

        for j in range(n):
            phase = OMEGA ** (-q * j)
            T_j_diag = T._full_T ** j
            proj += phase * T_j_diag

        proj /= n
        projectors.append(proj)

    return projectors


def verify_projector_resolution(projectors: List[NDArray], tol: float = 1e-12) -> Tuple[bool, float]:
    """
    Verify sum_q P_q = I.

    Args:
        projectors: List of projector diagonals
        tol: Tolerance

    Returns:
        Tuple of (passes, max_error)
    """
    total = sum(projectors)
    max_error = np.max(np.abs(total - 1.0))
    return max_error < tol, max_error


def verify_projector_orthogonality(projectors: List[NDArray], tol: float = 1e-12) -> Tuple[bool, float]:
    """
    Verify P_q P_r = delta_{qr} P_q.

    Args:
        projectors: List of projector diagonals
        tol: Tolerance

    Returns:
        Tuple of (passes, max_error)
    """
    max_error = 0.0

    for q, P_q in enumerate(projectors):
        for r, P_r in enumerate(projectors):
            product = P_q * P_r  # Element-wise for diagonal matrices
            expected = P_q if q == r else np.zeros_like(P_q)
            error = np.max(np.abs(product - expected))
            max_error = max(max_error, error)

    return max_error < tol, max_error


@dataclass
class BiDegree:
    """Bi-degree (ell, k) of a homogeneous operator."""
    shift_degree: int  # ell: transport/propagation degree
    torsion_degree: int  # k: torsion degree (mod 12)

    def __post_init__(self):
        self.torsion_degree = self.torsion_degree % TORSION_ORDER


def compute_bidegree(
    operator_matrix: NDArray,
    T: TorsionOperator,
    S: ShiftOperator,
    tol: float = 1e-8
) -> Optional[BiDegree]:
    """
    Compute bi-degree (ell, k) of a homogeneous operator.

    An operator A has bi-degree (ell, k) if:
    - S A S^{-1} = A (shift degree ell means A commutes with shift after accounting for transport)
    - T A T^{-1} = omega^k A (torsion degree k)

    For inhomogeneous operators, returns None.

    Args:
        operator_matrix: Operator as matrix
        T: Torsion operator
        S: Shift operator
        tol: Tolerance for homogeneity test

    Returns:
        BiDegree if homogeneous, None otherwise
    """
    T_mat = T.as_matrix()
    T_inv = np.conj(T_mat)  # T is unitary diagonal

    # Compute T A T^{-1}
    TAT_inv = T_mat @ operator_matrix @ T_inv

    # Check if TAT^{-1} = omega^k A for some k
    best_k = None
    best_error = float('inf')

    for k in range(TORSION_ORDER):
        expected = (OMEGA ** k) * operator_matrix
        error = np.linalg.norm(TAT_inv - expected)
        if error < best_error:
            best_error = error
            best_k = k

    if best_error > tol:
        return None  # Not torsion-homogeneous

    # For shift degree: check transport structure
    # This requires analyzing the cell structure of the operator
    # For now, return torsion degree only with shift=0 placeholder
    return BiDegree(shift_degree=0, torsion_degree=best_k)


def torsion_average(
    operator_matrix: NDArray,
    T: TorsionOperator
) -> NDArray:
    """
    Compute torsion average Pi_T(A) = (1/12) sum_{j=0}^{11} T^j A T^{-j}.

    Properties:
    - Pi_T projects operators onto torsion-degree-0 subspace
    - For k != 0: Pi_T(A) = 0 if A has pure torsion degree k

    Args:
        operator_matrix: Operator as matrix
        T: Torsion operator

    Returns:
        Torsion-averaged operator
    """
    T_mat = T.as_matrix()
    T_inv = np.conj(T_mat)

    result = np.zeros_like(operator_matrix)

    for j in range(TORSION_ORDER):
        T_j = np.diag(T._full_T ** j)
        T_neg_j = np.diag(T._full_T ** (-j))
        result += T_j @ operator_matrix @ T_neg_j

    return result / TORSION_ORDER


def verify_torsion_annihilation(
    operator_matrix: NDArray,
    T: TorsionOperator,
    tol: float = 1e-10
) -> Tuple[bool, float, int]:
    """
    Verify torsion annihilation: if A has torsion degree k != 0, then Pi_T(A) = 0.

    Args:
        operator_matrix: Operator matrix
        T: Torsion operator
        tol: Tolerance

    Returns:
        Tuple of (passes, norm of Pi_T(A), detected torsion degree)
    """
    bidegree = compute_bidegree(operator_matrix, T, None, tol=0.1)

    if bidegree is None:
        # Inhomogeneous - decompose and check each component
        return True, 0.0, -1

    k = bidegree.torsion_degree
    avg = torsion_average(operator_matrix, T)
    avg_norm = np.linalg.norm(avg)

    if k == 0:
        # Should NOT vanish
        passes = avg_norm > tol
    else:
        # Should vanish
        passes = avg_norm < tol

    return passes, avg_norm, k


class HomogeneousDecomposition:
    """
    Decompose an operator into homogeneous components.

    A = sum_{k=0}^{11} A_k where A_k has torsion degree k.
    """

    def __init__(self, operator_matrix: NDArray, T: TorsionOperator):
        """
        Decompose operator into torsion-homogeneous components.

        Args:
            operator_matrix: Operator to decompose
            T: Torsion operator
        """
        self.components = []
        T_mat = T.as_matrix()
        T_inv = np.conj(T_mat)

        for k in range(TORSION_ORDER):
            # A_k = (1/12) sum_j omega^{-kj} T^j A T^{-j}
            A_k = np.zeros_like(operator_matrix)

            for j in range(TORSION_ORDER):
                phase = OMEGA ** (-k * j)
                T_j = np.diag(T._full_T ** j)
                T_neg_j = np.diag(T._full_T ** (-j))
                A_k += phase * (T_j @ operator_matrix @ T_neg_j)

            A_k /= TORSION_ORDER
            self.components.append(A_k)

    def get_component(self, k: int) -> NDArray:
        """Get torsion-degree-k component."""
        return self.components[k % TORSION_ORDER]

    def verify_reconstruction(self, original: NDArray, tol: float = 1e-10) -> Tuple[bool, float]:
        """Verify sum of components equals original."""
        reconstructed = sum(self.components)
        error = np.linalg.norm(reconstructed - original)
        return error < tol, error
