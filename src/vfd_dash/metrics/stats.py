"""
Statistical utilities for metrics computation.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Dict, Any, Optional
from scipy import stats


def compute_distribution_stats(data: NDArray) -> Dict[str, float]:
    """
    Compute basic distribution statistics.

    Args:
        data: Array of values

    Returns:
        Dictionary of statistics
    """
    if len(data) == 0:
        return {"error": "Empty data"}

    return {
        "count": len(data),
        "mean": float(np.mean(data)),
        "std": float(np.std(data)),
        "min": float(np.min(data)),
        "max": float(np.max(data)),
        "median": float(np.median(data)),
        "q25": float(np.percentile(data, 25)),
        "q75": float(np.percentile(data, 75)),
        "skewness": float(stats.skew(data)),
        "kurtosis": float(stats.kurtosis(data)),
    }


def compute_comparison_stats(
    data1: NDArray,
    data2: NDArray,
    labels: tuple = ("data1", "data2")
) -> Dict[str, Any]:
    """
    Compute comparison statistics between two datasets.

    Args:
        data1: First dataset
        data2: Second dataset
        labels: Labels for the datasets

    Returns:
        Dictionary of comparison statistics
    """
    if len(data1) == 0 or len(data2) == 0:
        return {"error": "Empty data"}

    # Basic stats for each
    stats1 = compute_distribution_stats(data1)
    stats2 = compute_distribution_stats(data2)

    # KS test
    ks_stat, ks_pvalue = stats.ks_2samp(data1, data2)

    # Wasserstein distance
    try:
        from scipy.stats import wasserstein_distance
        w_dist = wasserstein_distance(data1, data2)
    except Exception:
        w_dist = None

    # Correlation if same length
    if len(data1) == len(data2):
        corr = float(np.corrcoef(data1, data2)[0, 1])
    else:
        corr = None

    return {
        labels[0]: stats1,
        labels[1]: stats2,
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "wasserstein_distance": w_dist,
        "correlation": corr,
        "mean_difference": stats2["mean"] - stats1["mean"],
        "std_ratio": stats2["std"] / stats1["std"] if stats1["std"] > 0 else None,
    }


def compute_residual_stats(
    predicted: NDArray,
    actual: NDArray
) -> Dict[str, float]:
    """
    Compute residual statistics.

    Args:
        predicted: Predicted values
        actual: Actual values

    Returns:
        Dictionary of residual statistics
    """
    n = min(len(predicted), len(actual))
    if n == 0:
        return {"error": "Empty data"}

    residuals = predicted[:n] - actual[:n]

    return {
        "n": n,
        "mae": float(np.mean(np.abs(residuals))),
        "rmse": float(np.sqrt(np.mean(residuals**2))),
        "max_error": float(np.max(np.abs(residuals))),
        "mean_residual": float(np.mean(residuals)),
        "std_residual": float(np.std(residuals)),
        "median_residual": float(np.median(residuals)),
    }


def compute_invariant_check_stats(
    checks: Dict[str, bool]
) -> Dict[str, Any]:
    """
    Summarize invariant check results.

    Args:
        checks: Dictionary of check name -> pass/fail

    Returns:
        Summary statistics
    """
    total = len(checks)
    passed = sum(1 for v in checks.values() if v)
    failed = total - passed

    return {
        "total_checks": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": passed / total if total > 0 else 0.0,
        "all_passed": failed == 0,
        "failed_checks": [k for k, v in checks.items() if not v],
    }
