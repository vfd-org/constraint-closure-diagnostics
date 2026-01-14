"""
VFD Report Generation Package.

Provides static plot generation and markdown report export for
VFD proof artifacts. All plots use matplotlib only (no seaborn).

Outputs:
- figures/*.png, *.svg - Publication-quality plots
- REPORT.md - Markdown summary with embedded figures
- data/*.npz - Raw arrays for reproducibility
"""

from .plots import (
    plot_cell_spectrum,
    plot_spectral_density,
    plot_bridge_metrics,
    plot_convergence,
    plot_torsion_fingerprint,
    plot_stability_certificate,
    create_hero_composite,
)
from .generator import ReportGenerator
from .markdown import generate_report_markdown
from .release_report import generate_release_report, write_release_report
from .sweep_report import generate_sweep_report, write_sweep_report


__all__ = [
    "plot_cell_spectrum",
    "plot_spectral_density",
    "plot_bridge_metrics",
    "plot_convergence",
    "plot_torsion_fingerprint",
    "plot_stability_certificate",
    "create_hero_composite",
    "ReportGenerator",
    "generate_report_markdown",
    "generate_release_report",
    "write_release_report",
    "generate_sweep_report",
    "write_sweep_report",
]
