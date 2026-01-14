"""
Torsion Sector Reduction for Spectrum Computation.

Since [K, T] = 0 (kernel commutes with torsion), the Hilbert space decomposes
into torsion eigenspaces. This allows block-diagonalization:

    H_ext = ⊕_{q=0}^{11} H_q

where H_q is the ω^q eigenspace of T.

For K_can = I_internal ⊗ L_cell with trivial internal coupling:
- Each H_q has dimension orbit_count * cell_count = 50 * C
- K_can restricted to H_q is just I_{50} ⊗ L_cell
- Spectrum on each sector is the same cell spectrum with multiplicity 50

This reduces internal_dim from 600 to 50 per sector, making computation
12x faster for general kernels.
"""

import numpy as np
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Any, Optional
from scipy.sparse import csr_matrix, block_diag
from scipy.sparse.linalg import eigsh


TORSION_ORDER = 12


def get_torsion_projection_indices(
    internal_dim: int,
    orbit_count: int,
    orbit_size: int = 12
) -> Dict[int, NDArray]:
    """
    Get indices for projecting onto each torsion sector.

    The internal space C^600 is organized as:
    - 50 orbits × 12 positions
    - Position j in each orbit belongs to ω^j eigenspace

    Args:
        internal_dim: Total internal dimension (600)
        orbit_count: Number of orbits (50)
        orbit_size: Orbit size (12)

    Returns:
        Dictionary mapping torsion degree q -> array of internal indices
    """
    assert internal_dim == orbit_count * orbit_size

    sector_indices = {q: [] for q in range(orbit_size)}

    for orbit in range(orbit_count):
        for j in range(orbit_size):
            idx = orbit * orbit_size + j
            q = j % orbit_size  # Torsion eigenvalue ω^q
            sector_indices[q].append(idx)

    return {q: np.array(indices) for q, indices in sector_indices.items()}


def project_to_torsion_sector(
    state: NDArray,
    torsion_degree: int,
    cell_count: int,
    internal_dim: int = 600,
    orbit_count: int = 50,
    orbit_size: int = 12
) -> NDArray:
    """
    Project a state onto a torsion sector.

    Args:
        state: Full state vector of length cell_count * internal_dim
        torsion_degree: Torsion sector q (0 to 11)
        cell_count: Number of cells
        internal_dim: Internal dimension
        orbit_count: Number of orbits
        orbit_size: Orbit size

    Returns:
        Projected state on sector q (length cell_count * orbit_count)
    """
    sector_indices = get_torsion_projection_indices(internal_dim, orbit_count, orbit_size)
    q_indices = sector_indices[torsion_degree]

    sector_dim = cell_count * orbit_count
    projected = np.zeros(sector_dim, dtype=state.dtype)

    for n in range(cell_count):
        # Extract sector components from cell n
        cell_start_full = n * internal_dim
        cell_start_sector = n * orbit_count

        for i, internal_idx in enumerate(q_indices):
            projected[cell_start_sector + i] = state[cell_start_full + internal_idx]

    return projected


def expand_from_torsion_sector(
    sector_state: NDArray,
    torsion_degree: int,
    cell_count: int,
    internal_dim: int = 600,
    orbit_count: int = 50,
    orbit_size: int = 12
) -> NDArray:
    """
    Expand a sector state to full space.

    Args:
        sector_state: State on sector q (length cell_count * orbit_count)
        torsion_degree: Torsion sector q
        cell_count: Number of cells
        internal_dim: Internal dimension
        orbit_count: Number of orbits
        orbit_size: Orbit size

    Returns:
        Full state with zeros outside sector q
    """
    sector_indices = get_torsion_projection_indices(internal_dim, orbit_count, orbit_size)
    q_indices = sector_indices[torsion_degree]

    full_dim = cell_count * internal_dim
    expanded = np.zeros(full_dim, dtype=sector_state.dtype)

    for n in range(cell_count):
        cell_start_full = n * internal_dim
        cell_start_sector = n * orbit_count

        for i, internal_idx in enumerate(q_indices):
            expanded[cell_start_full + internal_idx] = sector_state[cell_start_sector + i]

    return expanded


def extract_sector_kernel(
    K_sparse: csr_matrix,
    torsion_degree: int,
    cell_count: int,
    internal_dim: int = 600,
    orbit_count: int = 50,
    orbit_size: int = 12
) -> csr_matrix:
    """
    Extract the kernel restricted to a torsion sector.

    For K with [K, T] = 0, the restriction is well-defined.

    Args:
        K_sparse: Full kernel matrix
        torsion_degree: Sector q
        cell_count: Number of cells
        internal_dim: Full internal dimension
        orbit_count: Orbits per sector
        orbit_size: Orbit size (torsion order)

    Returns:
        Sparse matrix K|_q of size (cell_count * orbit_count)²
    """
    sector_indices = get_torsion_projection_indices(internal_dim, orbit_count, orbit_size)
    q_indices = sector_indices[torsion_degree]

    # Build full index list for sector q
    full_indices = []
    for n in range(cell_count):
        cell_start = n * internal_dim
        for internal_idx in q_indices:
            full_indices.append(cell_start + internal_idx)

    full_indices = np.array(full_indices)

    # Extract submatrix
    K_dense = K_sparse.toarray()
    K_sector = K_dense[np.ix_(full_indices, full_indices)]

    return csr_matrix(K_sector)


def compute_spectrum_by_sectors(
    K_sparse: csr_matrix,
    cell_count: int,
    internal_dim: int = 600,
    orbit_count: int = 50,
    orbit_size: int = 12,
    k_per_sector: Optional[int] = None
) -> Tuple[NDArray, Dict[str, Any]]:
    """
    Compute spectrum by diagonalizing each torsion sector.

    For K with [K, T] = 0, spectrum decomposes as union over sectors.

    Args:
        K_sparse: Full kernel matrix
        cell_count: Number of cells
        internal_dim: Internal dimension
        orbit_count: Orbits per sector
        orbit_size: Torsion order (12)
        k_per_sector: Eigenvalues per sector (None = all)

    Returns:
        Tuple of (all_eigenvalues, info_dict)
    """
    all_eigenvalues = []
    sector_info = {}

    sector_dim = cell_count * orbit_count

    for q in range(orbit_size):
        # Extract sector kernel
        K_q = extract_sector_kernel(
            K_sparse, q, cell_count, internal_dim, orbit_count, orbit_size
        )

        # Compute eigenvalues for this sector
        if k_per_sector is not None and k_per_sector < sector_dim - 2:
            eigenvalues_q, _ = eigsh(K_q, k=k_per_sector, which="SM")
        else:
            # Dense solver for small sectors
            eigenvalues_q = np.linalg.eigvalsh(K_q.toarray())

        all_eigenvalues.extend(eigenvalues_q)

        sector_info[f"sector_{q}"] = {
            "n_eigenvalues": len(eigenvalues_q),
            "min_eigenvalue": float(np.min(eigenvalues_q)),
            "max_eigenvalue": float(np.max(eigenvalues_q)),
        }

    all_eigenvalues = np.sort(np.array(all_eigenvalues))

    info = {
        "backend": "torsion_sectors",
        "total_eigenvalues": len(all_eigenvalues),
        "sectors_computed": orbit_size,
        "sector_dim": sector_dim,
        **sector_info,
    }

    return all_eigenvalues, info


def torsion_sector_spectrum_kcan(
    cell_count: int,
    propagation_range: int = 1,
    orbit_count: int = 50,
    orbit_size: int = 12
) -> Tuple[NDArray, Dict[int, NDArray], Dict[str, Any]]:
    """
    Compute K_can spectrum decomposed by torsion sector.

    For K_can = I_internal ⊗ L_cell, each sector has identical spectrum
    (cell eigenvalues with multiplicity orbit_count).

    Args:
        cell_count: Number of cells
        propagation_range: Coupling range
        orbit_count: Orbits per sector
        orbit_size: Torsion order

    Returns:
        Tuple of:
        - all_eigenvalues: Full sorted spectrum
        - sector_spectra: Dict mapping q -> sector eigenvalues
        - info: Diagnostic information
    """
    from .analytic import analytic_kcan_cell_eigenvalues

    # Cell eigenvalues (independent of sector)
    cell_eigs = analytic_kcan_cell_eigenvalues(cell_count, propagation_range)

    # Each sector has cell_eigs with multiplicity orbit_count
    sector_eigs = np.repeat(cell_eigs, orbit_count)
    sector_eigs_sorted = np.sort(sector_eigs)

    # All sectors have same spectrum
    sector_spectra = {q: sector_eigs_sorted.copy() for q in range(orbit_size)}

    # Full spectrum: 12 copies
    all_eigenvalues = np.repeat(sector_eigs_sorted, orbit_size)
    all_eigenvalues = np.sort(all_eigenvalues)

    info = {
        "backend": "torsion_sector_kcan_analytic",
        "cell_count": cell_count,
        "propagation_range": propagation_range,
        "orbit_count": orbit_count,
        "orbit_size": orbit_size,
        "sector_dim": cell_count * orbit_count,
        "eigenvalues_per_sector": len(sector_eigs_sorted),
        "total_eigenvalues": len(all_eigenvalues),
        "sectors_identical": True,  # For K_can, all sectors have same spectrum
    }

    return all_eigenvalues, sector_spectra, info


def torsion_fingerprint(
    eigenvalues_by_sector: Dict[int, NDArray],
    n_bins: int = 50
) -> NDArray:
    """
    Compute torsion fingerprint from sector spectra.

    The fingerprint is a (12, n_bins) array where each row is the
    eigenvalue histogram for that sector.

    Args:
        eigenvalues_by_sector: Dictionary q -> eigenvalues
        n_bins: Number of histogram bins

    Returns:
        (12, n_bins) fingerprint array
    """
    # Find global range
    all_eigs = np.concatenate(list(eigenvalues_by_sector.values()))
    eig_min, eig_max = np.min(all_eigs), np.max(all_eigs)

    # Add small margin
    margin = 0.01 * (eig_max - eig_min)
    bins = np.linspace(eig_min - margin, eig_max + margin, n_bins + 1)

    fingerprint = np.zeros((12, n_bins))

    for q in range(12):
        if q in eigenvalues_by_sector:
            hist, _ = np.histogram(eigenvalues_by_sector[q], bins=bins, density=True)
            fingerprint[q, :] = hist

    return fingerprint
