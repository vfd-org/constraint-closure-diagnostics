"""
Figure 01: Residual Ladder.

Bar chart showing total residual at each closure level L0-L4.
Color indicates pass (green) or fail (red).
"""

import io
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..constraints.ladder import LadderResult


def generate_residual_ladder(ladder_result: "LadderResult") -> bytes:
    """
    Generate residual ladder bar chart.

    Args:
        ladder_result: Results from ClosureLadder.run()

    Returns:
        PNG image as bytes
    """
    # Extract data
    levels = ["L0", "L1", "L2", "L3", "L4"]
    residuals = []
    passed = []

    level_results = ladder_result.to_dict().get("residuals_per_level", {})

    for level in levels:
        if level in level_results:
            data = level_results[level]
            residuals.append(max(data.get("total_residual", 1e-20), 1e-20))  # Avoid log(0)
            passed.append(data.get("satisfied", False))
        else:
            residuals.append(1e-20)
            passed.append(True)  # Not checked = passed by default

    # Create figure
    fig, ax = plt.subplots(figsize=(8, 5))

    colors = ['green' if p else 'red' for p in passed]
    bars = ax.bar(levels, residuals, color=colors, edgecolor='black', linewidth=1.2)

    # Log scale for y-axis
    ax.set_yscale('log')
    ax.set_ylim(1e-16, max(residuals) * 10)

    # Add pass/fail threshold line
    ax.axhline(y=1e-8, color='blue', linestyle='--', linewidth=1, label='Tolerance (1e-8)')

    # Labels
    ax.set_xlabel('Closure Level', fontsize=12)
    ax.set_ylabel('Total Residual (log scale)', fontsize=12)
    ax.set_title('Fig 01: Closure Ladder Residuals', fontsize=14, fontweight='bold')

    # Add value labels on bars
    for bar, res, p in zip(bars, residuals, passed):
        height = bar.get_height()
        label = f'{res:.1e}' if res > 1e-16 else '0'
        status = '✓' if p else '✗'
        ax.annotate(f'{label}\n{status}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    ax.legend(loc='upper right')

    # Add grid
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    return buf.read()
