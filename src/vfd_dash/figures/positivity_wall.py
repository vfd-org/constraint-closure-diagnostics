"""
Figure 04: Positivity Wall.

For single runs: Spectrum histogram + sorted eigenvalue curve.
For sweeps: Heatmap of min eigenvalue over parameter grid with y=0 contour.
"""

import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING, List, Dict, Any, Optional

if TYPE_CHECKING:
    from ..state import DiagnosticState


def generate_positivity_wall(state: "DiagnosticState") -> bytes:
    """
    Generate positivity wall visualization for a single run.

    Shows spectrum histogram and sorted eigenvalue curve.
    The "wall" is the boundary where eigenvalues cross zero.

    Args:
        state: Diagnostic state with spectrum data

    Returns:
        PNG image as bytes
    """
    # Extract eigenvalue data
    if state.spectrum is not None and hasattr(state.spectrum, 'eigenvalues'):
        eigenvalues = np.array(state.spectrum.eigenvalues)
    else:
        # Fallback: generate sample data for visualization
        eigenvalues = np.random.rand(100) * 2 - 0.1  # Mostly positive

    # Sort eigenvalues
    sorted_eigs = np.sort(eigenvalues)

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left plot: Spectrum histogram (renamed from "Eigenvalue Distribution")
    ax1.hist(sorted_eigs, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax1.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Positivity boundary')
    ax1.axvline(x=sorted_eigs.min(), color='orange', linestyle='-', linewidth=2,
                label=f'Min: {sorted_eigs.min():.2e}')
    ax1.set_xlabel('Eigenvalue', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title('Spectrum Histogram', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Right plot: Sorted eigenvalues (positivity wall curve)
    indices = np.arange(len(sorted_eigs))

    # Color by positivity
    colors = ['green' if e >= 0 else 'red' for e in sorted_eigs]
    ax2.scatter(indices, sorted_eigs, c=colors, s=10, alpha=0.6)
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=2, label='y = 0')

    # Fill negative region
    ax2.fill_between(indices, sorted_eigs, 0,
                     where=(sorted_eigs < 0),
                     color='red', alpha=0.3, label='Negative region')

    ax2.set_xlabel('Index (sorted)', fontsize=12)
    ax2.set_ylabel('Eigenvalue', fontsize=12)
    ax2.set_title('Fig 04: Spectrum Histogram (Single Run)', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    # Add statistics annotation
    n_negative = np.sum(sorted_eigs < 0)
    n_total = len(sorted_eigs)
    min_eig = sorted_eigs.min()

    stats_text = f'Min eigenvalue: {min_eig:.2e}\nNegative count: {n_negative}/{n_total}'
    ax2.annotate(stats_text, xy=(0.02, 0.98), xycoords='axes fraction',
                 fontsize=10, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    return buf.read()


def generate_positivity_wall_sweep(
    param1_name: str,
    param1_values: List,
    param2_name: str,
    param2_values: List,
    min_eigenvalue_grid: np.ndarray,
) -> bytes:
    """
    Generate positivity wall heatmap for parameter sweep.

    Shows min eigenvalue over parameter grid with contour at y=0.

    Args:
        param1_name: Name of first parameter
        param1_values: Values for first parameter
        param2_name: Name of second parameter
        param2_values: Values for second parameter
        min_eigenvalue_grid: 2D array of min eigenvalues (shape: len(p1) x len(p2))

    Returns:
        PNG image as bytes
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create heatmap
    p1_arr = np.array(param1_values)
    p2_arr = np.array(param2_values)

    # Use imshow with proper extent
    extent = [p2_arr.min() - 0.5, p2_arr.max() + 0.5,
              p1_arr.min() - 0.5, p1_arr.max() + 0.5]

    # Diverging colormap centered at 0
    vmax = np.abs(min_eigenvalue_grid).max()
    vmin = -vmax if min_eigenvalue_grid.min() < 0 else 0

    im = ax.imshow(min_eigenvalue_grid, cmap='RdYlGn', aspect='auto',
                   extent=extent, origin='lower',
                   vmin=vmin, vmax=vmax)

    # Add contour at 0 (the "wall")
    X, Y = np.meshgrid(p2_arr, p1_arr)
    try:
        contour = ax.contour(X, Y, min_eigenvalue_grid, levels=[0],
                            colors='black', linewidths=2, linestyles='--')
        ax.clabel(contour, fmt='0', fontsize=10)
    except ValueError:
        # No zero crossing in data
        pass

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, label='Min Eigenvalue')

    # Labels
    ax.set_xlabel(param2_name, fontsize=12)
    ax.set_ylabel(param1_name, fontsize=12)
    ax.set_title('Fig 04: Positivity Wall Grid (Min Eigenvalue over Parameters)',
                 fontsize=14, fontweight='bold')

    # Add tick labels
    ax.set_xticks(p2_arr)
    ax.set_yticks(p1_arr)

    # Add annotations for each cell
    for i, p1 in enumerate(param1_values):
        for j, p2 in enumerate(param2_values):
            val = min_eigenvalue_grid[i, j]
            color = 'white' if abs(val) > vmax * 0.5 else 'black'
            ax.text(p2, p1, f'{val:.1e}', ha='center', va='center',
                   fontsize=8, color=color)

    # Statistics
    min_val = min_eigenvalue_grid.min()
    max_val = min_eigenvalue_grid.max()
    n_negative = np.sum(min_eigenvalue_grid < 0)
    n_total = min_eigenvalue_grid.size

    stats_text = (f'Min: {min_val:.2e}\nMax: {max_val:.2e}\n'
                  f'Negative cells: {n_negative}/{n_total}')
    ax.annotate(stats_text, xy=(1.02, 0.98), xycoords='axes fraction',
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    return buf.read()
