#!/usr/bin/env python3
"""
Fast verification of release metrics against paper-reported values.

This script reads the metrics.json from a release bundle and verifies
that the values match what is reported in the paper. No heavy computation
is performed - it only reads and compares stored values.

Usage:
    python tools/verify_release_metrics.py --bundle-dir runs/release_20260113_225750
    python tools/verify_release_metrics.py --metrics runs/.../metrics.json

Expected runtime: < 5 seconds
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple


# Paper-reported values (with tolerances for rounding)
EXPECTED = {
    "spearman_ba": {
        "value": 0.99973,
        "tolerance": 0.0001,
        "paper_rounded": "0.9997",
        "description": "Spearman rank correlation (BA mode)",
    },
    "rmse_ba": {
        "value": 1098.0,
        "tolerance": 1.0,
        "paper_rounded": "1098",
        "description": "RMSE (BA mode)",
    },
    "spearman_bn1": {
        "value": 0.00836,
        "tolerance": 0.001,
        "paper_rounded": "0.008",
        "description": "Spearman rank correlation (BN1 ordering perturbation)",
    },
    "rmse_bn2": {
        "value": 2395.0,
        "tolerance": 10.0,
        "paper_rounded": "2395",
        "description": "RMSE (BN2 scale perturbation)",
    },
    "beta_deviation_bn3": {
        "value": 0.2,
        "tolerance": 0.01,
        "paper_rounded": "0.2",
        "description": "Beta deviation (BN3 coordinate perturbation)",
    },
}


def find_metrics_file(bundle_dir: Path) -> Path:
    """Find the metrics.json file with bridge/projection data in a bundle directory."""
    # Look for hash-named subdirectories
    candidates = []
    for subdir in bundle_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("sweep_"):
            metrics_path = subdir / "metrics.json"
            if metrics_path.exists():
                candidates.append(metrics_path)

    # Find the one with projection/falsification (production run with bridge mode)
    for metrics_path in candidates:
        try:
            with open(metrics_path) as f:
                data = json.load(f)
            # Check for projection with falsification (the bridge run)
            proj = data.get("projection", {})
            if isinstance(proj, dict) and "falsification" in proj:
                return metrics_path
            # Also check bridge_overlay (alternative structure)
            if "bridge_overlay" in data and "falsification" in data.get("bridge_overlay", {}):
                return metrics_path
        except (json.JSONDecodeError, IOError):
            continue

    # Fall back to first candidate if no bridge data found
    if candidates:
        return candidates[0]

    raise FileNotFoundError(f"No metrics.json found in {bundle_dir}")


def extract_metrics(metrics_data: Dict) -> Dict:
    """Extract relevant metrics from the metrics.json structure."""
    extracted = {}

    # Try projection structure first (current format)
    if "projection" in metrics_data:
        proj = metrics_data["projection"]

        # Main overlay metrics
        if "overlay_metrics" in proj:
            overlay = proj["overlay_metrics"]
            if "rank_correlation" in overlay:
                extracted["spearman_ba"] = overlay["rank_correlation"]
            if "rmse" in overlay:
                extracted["rmse_ba"] = overlay["rmse"]

        # Falsification data
        if "falsification" in proj:
            fals = proj["falsification"]

            # BN1 Spearman
            if "BN1" in fals and "rank_correlation" in fals["BN1"]:
                extracted["spearman_bn1"] = fals["BN1"]["rank_correlation"]

            # BN2 RMSE - calculate from full overlay RMSE × ratio
            # (The direct BN2 rmse is from a small subset, paper reports full overlay scaled)
            if "falsification_ratios" in fals and "rmse_ba" in extracted:
                ratios = fals["falsification_ratios"]
                if "BN2_ratio" in ratios:
                    extracted["rmse_bn2"] = extracted["rmse_ba"] * ratios["BN2_ratio"]

            # BN3 beta deviation (from falsification_details)
            if "falsification_details" in fals:
                details = fals["falsification_details"]
                if "BN3_bn_value" in details:
                    extracted["beta_deviation_bn3"] = details["BN3_bn_value"]

    # Fallback to bridge_overlay structure (alternative format)
    elif "bridge_overlay" in metrics_data:
        overlay = metrics_data["bridge_overlay"]
        if "rank_correlation" in overlay:
            extracted["spearman_ba"] = overlay["rank_correlation"]
        if "rmse" in overlay:
            extracted["rmse_ba"] = overlay["rmse"]

        if "falsification" in overlay:
            fals = overlay["falsification"]
            if "falsification_details" in fals:
                details = fals["falsification_details"]
                if "BN1_bn_value" in details:
                    extracted["spearman_bn1"] = details["BN1_bn_value"]
                if "BN3_bn_value" in details:
                    extracted["beta_deviation_bn3"] = details["BN3_bn_value"]
            if "falsification_ratios" in fals:
                ratios = fals["falsification_ratios"]
                if "BN2_ratio" in ratios and "rmse_ba" in extracted:
                    extracted["rmse_bn2"] = extracted["rmse_ba"] * ratios["BN2_ratio"]

    return extracted


def verify_metrics(extracted: Dict) -> Tuple[bool, List]:
    """Verify extracted metrics against expected values."""
    results = []
    all_pass = True

    for key, expected in EXPECTED.items():
        if key not in extracted:
            results.append({
                "metric": expected["description"],
                "status": "MISSING",
                "expected": expected["paper_rounded"],
                "actual": "N/A",
                "pass": False,
            })
            all_pass = False
            continue

        actual = extracted[key]
        diff = abs(actual - expected["value"])
        passed = diff <= expected["tolerance"]

        results.append({
            "metric": expected["description"],
            "status": "PASS" if passed else "FAIL",
            "expected": f"{expected['paper_rounded']} (±{expected['tolerance']})",
            "actual": f"{actual:.6f}",
            "pass": passed,
        })

        if not passed:
            all_pass = False

    return all_pass, results


def print_results(results: List, all_pass: bool):
    """Print verification results in a readable format."""
    print("\n" + "=" * 70)
    print("RELEASE METRICS VERIFICATION")
    print("=" * 70)
    print()

    # Find max width for formatting
    max_metric = max(len(r["metric"]) for r in results)
    max_expected = max(len(str(r["expected"])) for r in results)
    max_actual = max(len(str(r["actual"])) for r in results)

    # Header
    print(f"{'Metric':<{max_metric}}  {'Expected':<{max_expected}}  {'Actual':<{max_actual}}  Status")
    print("-" * 70)

    # Results
    for r in results:
        status_symbol = "[PASS]" if r["pass"] else "[FAIL]" if r["status"] != "MISSING" else "[MISS]"
        print(f"{r['metric']:<{max_metric}}  {r['expected']:<{max_expected}}  {r['actual']:<{max_actual}}  {status_symbol}")

    print("-" * 70)
    print()

    if all_pass:
        print("OVERALL: PASS - All metrics match paper-reported values")
        print()
        print("Note: Paper values are rounded for readability.")
        print("      Full precision values are stored in metrics.json.")
    else:
        print("OVERALL: FAIL - Some metrics do not match expected values")

    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Fast verification of release metrics against paper values"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--bundle-dir",
        type=Path,
        help="Path to bundle directory (e.g., runs/release_20260113_225750)"
    )
    group.add_argument(
        "--metrics",
        type=Path,
        help="Direct path to metrics.json file"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON instead of formatted text"
    )

    args = parser.parse_args()

    # Find metrics file
    if args.metrics:
        metrics_path = args.metrics
    else:
        try:
            metrics_path = find_metrics_file(args.bundle_dir)
        except FileNotFoundError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

    if not metrics_path.exists():
        print(f"Error: Metrics file not found: {metrics_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading metrics from: {metrics_path}")

    # Load and verify
    try:
        with open(metrics_path) as f:
            metrics_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON in metrics file: {e}", file=sys.stderr)
        sys.exit(1)

    extracted = extract_metrics(metrics_data)
    all_pass, results = verify_metrics(extracted)

    if args.json:
        output = {
            "metrics_file": str(metrics_path),
            "overall_pass": all_pass,
            "results": results,
        }
        print(json.dumps(output, indent=2))
    else:
        print_results(results, all_pass)

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
