"""
Metrics Module: Computation and Reporting.

Computes all metrics for VFD invariants and bridge validation.
"""

from .report import MetricsReport, generate_metrics_report
from .stats import compute_distribution_stats, compute_comparison_stats

__all__ = [
    "MetricsReport",
    "generate_metrics_report",
    "compute_distribution_stats",
    "compute_comparison_stats",
]
