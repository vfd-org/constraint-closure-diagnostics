"""
Figure 05: Collapse Geometry.

Plots "effective dimension" proxy vs closure depth.
Shows how constraint satisfaction correlates with dimensionality reduction.
"""

import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..constraints.ladder import LadderResult


def generate_collapse_geometry(ladder_result: "LadderResult") -> bytes:
    """
    Generate collapse geometry visualization.

    Shows effective dimension (based on residuals) at each closure level.

    Args:
        ladder_result: Results from ClosureLadder.run()

    Returns:
        PNG image as bytes
    """
    # Extract data
    levels = ["L0", "L1", "L2", "L3", "L4"]
    level_results = ladder_result.to_dict().get("residuals_per_level", {})

    # Compute "effective dimension" proxy from residuals
    # Lower residual = more collapsed = lower effective dimension
    residuals = []
    for level in levels:
        if level in level_results:
            res = level_results[level].get("total_residual", 1e-16)
            residuals.append(max(res, 1e-20))
        else:
            residuals.append(1e-20)

    residuals = np.array(residuals)

    # Map residuals to "effective dimension" (inverse log scale)
    # Higher residual = more degrees of freedom = higher effective dim
    max_dim = 100  # Notional starting dimension
    effective_dims = max_dim * (1 - np.log10(1e-16) / np.log10(residuals + 1e-20))
    effective_dims = np.clip(effective_dims, 1, max_dim)

    # Reverse: lower residual = lower effective dim
    effective_dims = max_dim - effective_dims + 1

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Effective dimension vs level
    x = np.arange(len(levels))
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(levels)))

    bars = ax1.bar(x, effective_dims, color=colors, edgecolor='black', linewidth=1.2)

    ax1.set_xlabel('Closure Level', fontsize=12)
    ax1.set_ylabel('Effective Dimension (proxy)', fontsize=12)
    ax1.set_title('Dimension Collapse vs Closure Depth', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(levels)
    ax1.set_ylim(0, max_dim * 1.1)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, dim in zip(bars, effective_dims):
        height = bar.get_height()
        ax1.annotate(f'{dim:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    # Right: Residual decay (log scale)
    ax2.semilogy(x, residuals, 'o-', color='blue', linewidth=2, markersize=10, label='Total residual')
    ax2.axhline(y=1e-8, color='green', linestyle='--', linewidth=1.5, label='Tolerance')

    # Fill region above tolerance
    ax2.fill_between(x, residuals, 1e-8,
                     where=(residuals > 1e-8),
                     color='red', alpha=0.2, label='Above tolerance')
    ax2.fill_between(x, residuals, 1e-16,
                     where=(residuals <= 1e-8),
                     color='green', alpha=0.2, label='Below tolerance')

    ax2.set_xlabel('Closure Level', fontsize=12)
    ax2.set_ylabel('Residual (log scale)', fontsize=12)
    ax2.set_title('Residual Decay', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(levels)
    ax2.set_ylim(1e-18, 1e0)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.suptitle('Fig 05: Collapse Geometry', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    return buf.read()
