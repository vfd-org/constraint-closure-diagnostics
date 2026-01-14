"""
VFD Static Plot Generation.

All plots use matplotlib only. Each plot is self-contained with:
- Explicit figure size
- Axis labels and title
- Parameter annotation text box
- Consistent styling
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# Plot styling constants
FIGSIZE_STANDARD = (8, 4)
FIGSIZE_SQUARE = (6, 6)
FIGSIZE_WIDE = (10, 4)
FIGSIZE_HERO = (8, 12)
DPI_DEFAULT = 200
DPI_HIGH = 300

# Colors
COLOR_BA = '#2E86AB'  # Blue for Bridge Axiom
COLOR_BN = '#E94F37'  # Red for Bridge Negation
COLOR_SPECTRUM = '#1B998B'  # Teal for spectrum
COLOR_STABILITY = '#4CAF50'  # Green for stability
COLORS_TORSION = plt.cm.tab20(np.linspace(0, 1, 12))  # 12 distinct colors


def _add_param_box(ax, text: str, loc: str = 'upper right'):
    """Add parameter annotation box to plot."""
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    if loc == 'upper right':
        ax.text(0.97, 0.97, text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right', bbox=props)
    elif loc == 'upper left':
        ax.text(0.03, 0.97, text, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='left', bbox=props)
    elif loc == 'lower right':
        ax.text(0.97, 0.03, text, transform=ax.transAxes, fontsize=8,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)


def plot_cell_spectrum(
    cell_count: int,
    propagation_range: int,
    output_path: Path,
    format: str = 'png',
    dpi: int = DPI_DEFAULT
) -> Dict[str, Any]:
    """
    Plot 1: Cell spectrum (analytic fingerprint).

    Shows the closed-form eigenvalues of L_cell as a function of mode index.

    Args:
        cell_count: Number of cells C
        propagation_range: Coupling range R
        output_path: Output file path (without extension)
        format: 'png' or 'svg'
        dpi: Resolution

    Returns:
        Dictionary with plot data for saving
    """
    from ..spectrum.analytic import analytic_kcan_cell_eigenvalues, circulant_laplacian_eigenvalue

    # Compute eigenvalues
    eigenvalues = analytic_kcan_cell_eigenvalues(cell_count, propagation_range)

    # Also compute at finer theta resolution for the continuous curve
    theta_fine = np.linspace(0, 2 * np.pi, 500)
    lambda_fine = np.array([
        circulant_laplacian_eigenvalue(t, propagation_range, cell_count)
        for t in theta_fine
    ])

    # Mode indices and their theta values
    j_indices = np.arange(cell_count)
    theta_j = 2 * np.pi * j_indices / cell_count

    # Create figure
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    # Plot continuous curve
    ax.plot(theta_fine, lambda_fine, '-', color=COLOR_SPECTRUM, alpha=0.5,
            linewidth=1.5, label='Continuous $\\lambda(\\theta)$')

    # Plot discrete eigenvalues
    ax.scatter(theta_j, eigenvalues, s=50, c=COLOR_SPECTRUM,
               edgecolors='black', linewidths=0.5, zorder=5,
               label=f'Eigenvalues (C={cell_count})')

    # Labels and title
    ax.set_xlabel('$\\theta = 2\\pi j / C$', fontsize=11)
    ax.set_ylabel('$\\lambda_{\\mathrm{cell}}(\\theta)$', fontsize=11)
    ax.set_title(f'Cell Laplacian Spectrum (Analytic)', fontsize=12)

    # X-axis ticks at special angles
    ax.set_xticks([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi])
    ax.set_xticklabels(['0', '$\\pi/2$', '$\\pi$', '$3\\pi/2$', '$2\\pi$'])

    ax.set_xlim(-0.1, 2*np.pi + 0.1)
    ax.set_ylim(-0.2, max(eigenvalues) * 1.1)

    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)

    # Parameter box
    formula = '$\\lambda(\\theta) = 2R - 2\\sum_{d=1}^{R} \\cos(d\\theta)$'
    param_text = f'{formula}\nC = {cell_count}, R = {propagation_range}'
    _add_param_box(ax, param_text, 'lower right')

    plt.tight_layout()

    # Save
    filepath = f"{output_path}.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return {
        'j_indices': j_indices,
        'theta_j': theta_j,
        'eigenvalues': eigenvalues,
        'theta_fine': theta_fine,
        'lambda_fine': lambda_fine,
        'cell_count': cell_count,
        'propagation_range': propagation_range,
    }


def plot_spectral_density(
    cell_count: int,
    internal_dim: int,
    propagation_range: int,
    output_path: Path,
    ba_eigenvalues: Optional[NDArray] = None,
    bn_eigenvalues: Optional[NDArray] = None,
    n_bins: int = 50,
    format: str = 'png',
    dpi: int = DPI_DEFAULT
) -> Dict[str, Any]:
    """
    Plot 2: Spectral density histogram.

    Shows eigenvalue distribution, optionally comparing BA vs BN.

    Args:
        cell_count: Number of cells
        internal_dim: Internal dimension
        propagation_range: Coupling range
        output_path: Output file path
        ba_eigenvalues: Eigenvalues for BA mode (computed if None)
        bn_eigenvalues: Eigenvalues for BN mode (optional)
        n_bins: Number of histogram bins
        format: Output format
        dpi: Resolution

    Returns:
        Dictionary with plot data
    """
    from ..spectrum.analytic import analytic_kcan_eigenvalues

    # Compute BA eigenvalues if not provided
    if ba_eigenvalues is None:
        ba_eigenvalues = analytic_kcan_eigenvalues(cell_count, propagation_range, internal_dim)

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    # Determine bin edges from BA data
    eig_min = np.min(ba_eigenvalues)
    eig_max = np.max(ba_eigenvalues)
    margin = 0.05 * (eig_max - eig_min)
    bins = np.linspace(eig_min - margin, eig_max + margin, n_bins + 1)

    # Plot BA histogram
    ax.hist(ba_eigenvalues, bins=bins, alpha=0.7, color=COLOR_BA,
            edgecolor='black', linewidth=0.5, label='BA (Bridge Axiom)', density=True)

    # Plot BN histogram if provided
    if bn_eigenvalues is not None:
        ax.hist(bn_eigenvalues, bins=bins, alpha=0.5, color=COLOR_BN,
                edgecolor='black', linewidth=0.5, label='BN (Bridge Negation)', density=True)

    ax.set_xlabel('Eigenvalue $\\lambda$', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title(f'Spectral Density (C={cell_count}, internal_dim={internal_dim})', fontsize=12)

    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Parameter box
    param_text = f'C = {cell_count}\nR = {propagation_range}\nTotal: {len(ba_eigenvalues)} eigenvalues'
    _add_param_box(ax, param_text, 'upper left')

    plt.tight_layout()

    filepath = f"{output_path}.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return {
        'ba_eigenvalues': ba_eigenvalues,
        'bn_eigenvalues': bn_eigenvalues,
        'bins': bins,
        'cell_count': cell_count,
        'internal_dim': internal_dim,
    }


def plot_bridge_metrics(
    cell_counts: List[int],
    ba_metrics: Dict[str, List[float]],
    bn_metrics: Optional[Dict[str, List[float]]],
    output_path: Path,
    metric_name: str = 'rmse',
    format: str = 'png',
    dpi: int = DPI_DEFAULT
) -> Dict[str, Any]:
    """
    Plot 3: BA vs BN metric comparison across scales.

    Args:
        cell_counts: List of cell counts tested
        ba_metrics: Dictionary with metric lists for BA
        bn_metrics: Dictionary with metric lists for BN (optional)
        output_path: Output file path
        metric_name: Which metric to plot ('rmse', 'mae', 'ks_stat', 'correlation')
        format: Output format
        dpi: Resolution

    Returns:
        Dictionary with plot data
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    x = np.array(cell_counts)

    # Plot BA metrics
    ba_values = np.array(ba_metrics.get(metric_name, ba_metrics.get('values', [])))
    ax.plot(x, ba_values, 'o-', color=COLOR_BA, linewidth=2, markersize=8,
            label='BA (Bridge Axiom)')

    # Plot BN metrics if provided
    if bn_metrics is not None:
        bn_values = np.array(bn_metrics.get(metric_name, bn_metrics.get('values', [])))
        ax.plot(x, bn_values, 's--', color=COLOR_BN, linewidth=2, markersize=8,
                label='BN (Bridge Negation)')

    ax.set_xlabel('Cell Count (C)', fontsize=11)
    ax.set_ylabel(f'{metric_name.upper()}', fontsize=11)
    ax.set_title(f'Bridge Metric Comparison: {metric_name.upper()}', fontsize=12)

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Log scale for x-axis if range is large
    if max(cell_counts) / min(cell_counts) > 4:
        ax.set_xscale('log', base=2)
        ax.set_xticks(cell_counts)
        ax.set_xticklabels([str(c) for c in cell_counts])

    # Parameter box
    if bn_metrics is not None:
        ratio = np.mean(bn_values) / np.mean(ba_values) if np.mean(ba_values) > 0 else 0
        param_text = f'BN/BA ratio: {ratio:.2f}'
        _add_param_box(ax, param_text, 'upper right')

    plt.tight_layout()

    filepath = f"{output_path}.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return {
        'cell_counts': cell_counts,
        'ba_metrics': ba_metrics,
        'bn_metrics': bn_metrics,
        'metric_name': metric_name,
    }


def plot_convergence(
    cell_counts: List[int],
    metric_values: List[float],
    output_path: Path,
    metric_name: str = 'bridge_metric',
    confidence_bands: Optional[Tuple[List[float], List[float]]] = None,
    format: str = 'png',
    dpi: int = DPI_DEFAULT
) -> Dict[str, Any]:
    """
    Plot 4: Convergence plot showing metric stabilization.

    Args:
        cell_counts: List of cell counts
        metric_values: Metric values at each scale
        output_path: Output file path
        metric_name: Name of metric being plotted
        confidence_bands: Optional (lower, upper) confidence bands
        format: Output format
        dpi: Resolution

    Returns:
        Dictionary with plot data
    """
    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    x = np.array(cell_counts)
    y = np.array(metric_values)

    # Plot main line
    ax.plot(x, y, 'o-', color=COLOR_BA, linewidth=2, markersize=8, label=metric_name)

    # Plot confidence bands if provided
    if confidence_bands is not None:
        lower, upper = confidence_bands
        ax.fill_between(x, lower, upper, color=COLOR_BA, alpha=0.2, label='Confidence band')

    ax.set_xlabel('Cell Count (C)', fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    ax.set_title(f'Convergence: {metric_name} vs Scale', fontsize=12)

    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Log scale for x-axis
    if max(cell_counts) / min(cell_counts) > 4:
        ax.set_xscale('log', base=2)
        ax.set_xticks(cell_counts)
        ax.set_xticklabels([str(c) for c in cell_counts])

    # Annotate convergence behavior
    if len(metric_values) >= 2:
        rel_change = abs(metric_values[-1] - metric_values[-2]) / abs(metric_values[-2]) if metric_values[-2] != 0 else 0
        param_text = f'Final: {metric_values[-1]:.4f}\nRel. change: {rel_change:.2%}'
        _add_param_box(ax, param_text, 'upper right')

    plt.tight_layout()

    filepath = f"{output_path}.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return {
        'cell_counts': cell_counts,
        'metric_values': metric_values,
        'metric_name': metric_name,
    }


def plot_torsion_fingerprint(
    cell_count: int,
    propagation_range: int,
    output_path: Path,
    n_bins: int = 30,
    format: str = 'png',
    dpi: int = DPI_DEFAULT,
    single_figure: bool = True
) -> Dict[str, Any]:
    """
    Plot 5: Torsion fingerprint (12-sector spectral densities).

    Args:
        cell_count: Number of cells
        propagation_range: Coupling range
        output_path: Output file path
        n_bins: Histogram bins
        format: Output format
        dpi: Resolution
        single_figure: If True, create one figure; if False, create 12 separate

    Returns:
        Dictionary with plot data
    """
    from ..spectrum.torsion_sectors import torsion_sector_spectrum_kcan, torsion_fingerprint

    # Compute sector spectra
    all_eigs, sector_spectra, info = torsion_sector_spectrum_kcan(
        cell_count, propagation_range, orbit_count=50, orbit_size=12
    )

    # Compute fingerprint matrix
    fingerprint = torsion_fingerprint(sector_spectra, n_bins=n_bins)

    # Determine bin edges
    eig_min = np.min(all_eigs)
    eig_max = np.max(all_eigs)
    margin = 0.05 * (eig_max - eig_min)
    bins = np.linspace(eig_min - margin, eig_max + margin, n_bins + 1)

    if single_figure:
        # Create 12 subplots (4x3 grid)
        fig, axes = plt.subplots(4, 3, figsize=(10, 10))
        axes = axes.flatten()

        for q in range(12):
            ax = axes[q]
            sector_eigs = sector_spectra[q]

            ax.hist(sector_eigs, bins=bins, color=COLORS_TORSION[q],
                    edgecolor='black', linewidth=0.3, alpha=0.8, density=True)

            ax.set_title(f'Sector q={q} ($\\omega^{{{q}}}$)', fontsize=9)
            ax.set_xlim(bins[0], bins[-1])

            if q >= 9:  # Bottom row
                ax.set_xlabel('$\\lambda$', fontsize=8)
            if q % 3 == 0:  # Left column
                ax.set_ylabel('Density', fontsize=8)

            ax.tick_params(labelsize=7)

        fig.suptitle(f'Torsion Fingerprint (C={cell_count}, R={propagation_range})', fontsize=12)
        plt.tight_layout()

        filepath = f"{output_path}.{format}"
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        plt.close(fig)

    else:
        # Create 12 separate figures
        for q in range(12):
            fig, ax = plt.subplots(figsize=(4, 3))
            sector_eigs = sector_spectra[q]

            ax.hist(sector_eigs, bins=bins, color=COLORS_TORSION[q],
                    edgecolor='black', linewidth=0.5, alpha=0.8, density=True)

            ax.set_xlabel('$\\lambda$', fontsize=10)
            ax.set_ylabel('Density', fontsize=10)
            ax.set_title(f'Torsion Sector q={q} ($\\omega^{{{q}}}$)', fontsize=11)
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()

            filepath = f"{output_path}_q{q:02d}.{format}"
            fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
            plt.close(fig)

    return {
        'all_eigenvalues': all_eigs,
        'sector_spectra': sector_spectra,
        'fingerprint': fingerprint,
        'bins': bins,
        'cell_count': cell_count,
        'propagation_range': propagation_range,
    }


def plot_stability_certificate(
    cell_count: int,
    internal_dim: int,
    propagation_range: int,
    output_path: Path,
    n_probes: int = 500,
    seed: int = 42,
    format: str = 'png',
    dpi: int = DPI_DEFAULT
) -> Dict[str, Any]:
    """
    Plot 6: Stability certificate showing Q_K(v) distribution.

    Args:
        cell_count: Number of cells
        internal_dim: Internal dimension
        propagation_range: Coupling range
        output_path: Output file path
        n_probes: Number of probes to test
        seed: Random seed
        format: Output format
        dpi: Resolution

    Returns:
        Dictionary with plot data
    """
    from ..vfd.canonical import VFDSpace, TorsionOperator, ShiftOperator
    from ..vfd.kernels import CanonicalKernel
    from ..vfd.stability import StabilityAnalyzer

    # Build space and kernel
    orbit_count = internal_dim // 12
    space = VFDSpace(
        cell_count=cell_count,
        internal_dim=internal_dim,
        orbit_count=orbit_count,
        orbit_size=12
    )
    T = TorsionOperator(space)
    S = ShiftOperator(space)
    kernel = CanonicalKernel(space, T, S, propagation_range=propagation_range)

    # Compute stability coefficients
    analyzer = StabilityAnalyzer(space, T, kernel, seed=seed)
    coefficients = analyzer.compute_stability_coefficients(
        probe_count=n_probes, support_radius=2
    )

    # Extract Q_K values
    q_values = np.array([c.value for c in coefficients])

    fig, ax = plt.subplots(figsize=FIGSIZE_STANDARD)

    # Histogram of Q_K values
    n_bins = 40
    ax.hist(q_values, bins=n_bins, color=COLOR_STABILITY,
            edgecolor='black', linewidth=0.5, alpha=0.8)

    # Mark minimum and mean
    min_q = np.min(q_values)
    mean_q = np.mean(q_values)

    ax.axvline(min_q, color='red', linestyle='--', linewidth=2, label=f'Min: {min_q:.4f}')
    ax.axvline(mean_q, color='orange', linestyle='--', linewidth=2, label=f'Mean: {mean_q:.4f}')
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    ax.set_xlabel('$Q_K(v) = \\langle v, K v \\rangle$', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title(f'Stability Certificate: Kernel Nonnegativity', fontsize=12)

    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Parameter box
    nonneg_count = np.sum(q_values >= -1e-10)
    param_text = f'Probes: {n_probes}\nAll nonnegative: {nonneg_count == n_probes}\nMin Q_K: {min_q:.6f}'
    _add_param_box(ax, param_text, 'upper left')

    plt.tight_layout()

    filepath = f"{output_path}.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return {
        'q_values': q_values,
        'min_q': min_q,
        'mean_q': mean_q,
        'std_q': np.std(q_values),
        'n_probes': n_probes,
        'all_nonnegative': nonneg_count == n_probes,
    }


def create_hero_composite(
    cell_counts: List[int],
    ba_metrics: Dict[str, List[float]],
    bn_metrics: Optional[Dict[str, List[float]]],
    torsion_fingerprint_data: NDArray,
    stability_q_values: NDArray,
    output_path: Path,
    format: str = 'png',
    dpi: int = DPI_HIGH
) -> Dict[str, Any]:
    """
    Create hero composite image combining 3 key plots.

    Vertically stacked:
    1. BA vs BN metric comparison
    2. Torsion fingerprint heatmap summary
    3. Stability histogram

    Args:
        cell_counts: Cell counts for metric plot
        ba_metrics: BA metric values
        bn_metrics: BN metric values (optional)
        torsion_fingerprint_data: (12, n_bins) fingerprint array
        stability_q_values: Array of Q_K values
        output_path: Output file path
        format: Output format
        dpi: Resolution

    Returns:
        Dictionary with plot data
    """
    fig, axes = plt.subplots(3, 1, figsize=FIGSIZE_HERO)

    # --- Panel 1: BA vs BN metrics ---
    ax1 = axes[0]
    x = np.array(cell_counts)

    ba_values = np.array(ba_metrics.get('rmse', ba_metrics.get('values', [])))
    ax1.plot(x, ba_values, 'o-', color=COLOR_BA, linewidth=2, markersize=8,
             label='BA (Bridge Axiom)')

    if bn_metrics is not None:
        bn_values = np.array(bn_metrics.get('rmse', bn_metrics.get('values', [])))
        ax1.plot(x, bn_values, 's--', color=COLOR_BN, linewidth=2, markersize=8,
                 label='BN (Bridge Negation)')

    ax1.set_xlabel('Cell Count (C)', fontsize=10)
    ax1.set_ylabel('RMSE', fontsize=10)
    ax1.set_title('(A) Bridge Falsifiability: BA vs BN', fontsize=11, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)

    if max(cell_counts) / min(cell_counts) > 4:
        ax1.set_xscale('log', base=2)
        ax1.set_xticks(cell_counts)
        ax1.set_xticklabels([str(c) for c in cell_counts])

    # --- Panel 2: Torsion fingerprint heatmap ---
    ax2 = axes[1]

    im = ax2.imshow(torsion_fingerprint_data, aspect='auto', cmap='viridis',
                    interpolation='nearest')
    ax2.set_xlabel('Eigenvalue bin', fontsize=10)
    ax2.set_ylabel('Torsion sector q', fontsize=10)
    ax2.set_title('(B) Torsion Fingerprint (12-fold Structure)', fontsize=11, fontweight='bold')
    ax2.set_yticks(range(12))
    ax2.set_yticklabels([f'$\\omega^{{{q}}}$' for q in range(12)], fontsize=8)

    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Density', fontsize=9)

    # --- Panel 3: Stability histogram ---
    ax3 = axes[2]

    ax3.hist(stability_q_values, bins=40, color=COLOR_STABILITY,
             edgecolor='black', linewidth=0.5, alpha=0.8)

    min_q = np.min(stability_q_values)
    mean_q = np.mean(stability_q_values)

    ax3.axvline(min_q, color='red', linestyle='--', linewidth=2,
                label=f'Min: {min_q:.4f}')
    ax3.axvline(mean_q, color='orange', linestyle='--', linewidth=2,
                label=f'Mean: {mean_q:.4f}')
    ax3.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    ax3.set_xlabel('$Q_K(v) = \\langle v, K v \\rangle$', fontsize=10)
    ax3.set_ylabel('Count', fontsize=10)
    ax3.set_title('(C) Kernel Absoluteness: All Q_K(v) ≥ 0', fontsize=11, fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')

    # Overall title
    fig.suptitle('VFD Internal Spectral Proof Artifacts', fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    filepath = f"{output_path}.{format}"
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)

    return {
        'cell_counts': cell_counts,
        'ba_metrics': ba_metrics,
        'bn_metrics': bn_metrics,
        'fingerprint_shape': torsion_fingerprint_data.shape,
        'stability_min': min_q,
        'stability_mean': mean_q,
    }
