"""
Figure 03: Constraint Waterfall.

Stacked bar chart showing residuals per constraint family at each level.
"""

import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..constraints.ladder import LadderResult


def generate_constraint_waterfall(ladder_result: "LadderResult") -> bytes:
    """
    Generate constraint waterfall (stacked bars per family).

    Args:
        ladder_result: Results from ClosureLadder.run()

    Returns:
        PNG image as bytes
    """
    # Extract data
    levels = ["L0", "L1", "L2", "L3", "L4"]
    families = ["EF", "Symmetry", "Positivity", "Trace"]
    family_colors = {
        "EF": "#2ecc71",  # Green
        "Symmetry": "#3498db",  # Blue
        "Positivity": "#e74c3c",  # Red
        "Trace": "#9b59b6",  # Purple
    }

    level_results = ladder_result.to_dict().get("residuals_per_level", {})

    # Build data matrix
    data = {family: [] for family in families}

    for level in levels:
        if level in level_results:
            family_residuals = level_results[level].get("family_residuals", {})
            for family in families:
                res = family_residuals.get(family, 0.0)
                data[family].append(max(res, 1e-20))  # Avoid log(0)
        else:
            for family in families:
                data[family].append(1e-20)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(levels))
    width = 0.6

    # Stack bars
    bottom = np.zeros(len(levels))
    for family in families:
        values = np.array(data[family])
        ax.bar(x, values, width, bottom=bottom, label=family,
               color=family_colors[family], edgecolor='white', linewidth=0.5)
        bottom += values

    # Log scale
    ax.set_yscale('log')
    ax.set_ylim(1e-16, max(bottom) * 10)

    # Add tolerance line
    ax.axhline(y=1e-8, color='black', linestyle='--', linewidth=1, label='Tolerance')

    # Labels
    ax.set_xlabel('Closure Level', fontsize=12)
    ax.set_ylabel('Residual (log scale)', fontsize=12)
    ax.set_title('Fig 03: Constraint Family Waterfall', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(levels)

    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    return buf.read()
