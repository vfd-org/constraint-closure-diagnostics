"""
Sweep Report Generator.

Generates markdown reports for parameter sweeps, summarizing
parameter ranges, results grid, and analysis.
"""

import json
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

from .release_report import SAFETY_DISCLAIMER, get_git_info, get_python_info


def generate_sweep_report(
    sweep_dir: Path,
    results: Dict[str, Any],
) -> str:
    """
    Generate a markdown report for a parameter sweep.

    Args:
        sweep_dir: Path to the sweep directory
        results: Sweep results dictionary

    Returns:
        Markdown report string
    """
    report = []

    # Header
    sweep_name = sweep_dir.name
    timestamp = results.get("timestamp", datetime.now().isoformat())

    report.append(f"# Sweep Report: {sweep_name}")
    report.append("")
    report.append(f"**Generated:** {timestamp}")
    report.append("")

    # Safety disclaimer
    report.append(SAFETY_DISCLAIMER)
    report.append("")

    # Sweep Configuration
    report.append("## Sweep Configuration")
    report.append("")

    param1_name = results.get("param1_name", "param1")
    param2_name = results.get("param2_name", "param2")
    param1_values = results.get("param1_values", [])
    param2_values = results.get("param2_values", [])

    report.append("| Parameter | Values |")
    report.append("|-----------|--------|")
    report.append(f"| {param1_name} | {param1_values} |")
    report.append(f"| {param2_name} | {param2_values} |")
    report.append(f"| Grid Size | {len(param1_values)} x {len(param2_values)} = {len(param1_values) * len(param2_values)} |")
    report.append(f"| Internal Dim | {results.get('internal_dim', 'N/A')} |")
    report.append(f"| Max Level | {results.get('max_level', 'L4')} |")
    report.append(f"| Seed | {results.get('seed', 42)} |")

    report.append("")

    # Results Summary
    report.append("## Results Summary")
    report.append("")

    grids = results.get("grids", {})

    # Level passed grid
    level_grid = grids.get("max_level_passed")
    if level_grid is not None:
        report.append("### Max Level Passed")
        report.append("")
        report.append(_format_grid_table(
            param1_name, param1_values,
            param2_name, param2_values,
            level_grid, format_func=str
        ))
        report.append("")

    # Min eigenvalue grid
    min_eig_grid = grids.get("min_eigenvalue")
    if min_eig_grid is not None:
        report.append("### Min Eigenvalue")
        report.append("")
        report.append(_format_grid_table(
            param1_name, param1_values,
            param2_name, param2_values,
            min_eig_grid, format_func=lambda x: f"{x:.2e}"
        ))
        report.append("")

        # Analysis
        min_val = np.min(min_eig_grid)
        max_val = np.max(min_eig_grid)
        n_negative = np.sum(np.array(min_eig_grid) < 0)
        n_total = len(param1_values) * len(param2_values)

        report.append("**Eigenvalue Analysis:**")
        report.append(f"- Minimum value: {min_val:.2e}")
        report.append(f"- Maximum value: {max_val:.2e}")
        report.append(f"- Cells with negative eigenvalues: {n_negative}/{n_total}")

        if n_negative == 0:
            report.append("- **All cells satisfy positivity constraint**")
        else:
            report.append(f"- **Warning:** {n_negative} cells violate positivity")

        report.append("")

    # Total residual grid
    residual_grid = grids.get("total_residual")
    if residual_grid is not None:
        report.append("### Total Residual")
        report.append("")
        report.append(_format_grid_table(
            param1_name, param1_values,
            param2_name, param2_values,
            residual_grid, format_func=lambda x: f"{x:.2e}"
        ))
        report.append("")

    # Figures
    report.append("## Generated Figures")
    report.append("")

    figures = []
    for fig_name in ["fig02_phase_map.png", "fig04_positivity_wall_grid.png"]:
        fig_path = sweep_dir / fig_name
        if fig_path.exists():
            figures.append(fig_name)

    if figures:
        report.append("| Figure | Description |")
        report.append("|--------|-------------|")
        for fig in figures:
            if "phase_map" in fig:
                desc = "Phase map showing pass/fail regions over parameter grid"
            elif "positivity_wall_grid" in fig:
                desc = "Heatmap of min eigenvalue over parameter grid"
            else:
                desc = "Sweep figure"
            report.append(f"| `{fig}` | {desc} |")
    else:
        report.append("No figures found.")

    report.append("")

    # Per-run details (if available)
    run_results = results.get("run_results", [])
    if run_results:
        report.append("## Individual Run Results")
        report.append("")
        report.append(f"| {param1_name} | {param2_name} | Max Level | Min Eigenvalue | Total Residual |")
        report.append("|---|---|---|---|---|")

        for run in run_results:
            p1 = run.get(param1_name, "N/A")
            p2 = run.get(param2_name, "N/A")
            max_level = run.get("max_level_passed", "N/A")
            min_eig = run.get("min_eigenvalue", 0)
            residual = run.get("total_residual", 0)
            report.append(f"| {p1} | {p2} | {max_level} | {min_eig:.2e} | {residual:.2e} |")

        report.append("")

    # Reproducibility
    report.append("## Reproducibility")
    report.append("")

    git_info = get_git_info()
    python_info = get_python_info()

    report.append("### Environment")
    report.append("")
    report.append("| Property | Value |")
    report.append("|----------|-------|")
    report.append(f"| Git Commit | `{git_info['git_hash']}` |")
    report.append(f"| Git Branch | {git_info['git_branch']} |")
    report.append(f"| Python Version | {python_info['python_version']} |")

    report.append("")

    report.append("### Reproduction Command")
    report.append("")
    report.append("```bash")
    p1_str = ",".join(str(v) for v in param1_values)
    p2_str = ",".join(str(v) for v in param2_values)
    cmd = f"rhdiag sweep --param1 {param1_name} --values1 {p1_str}"
    cmd += f" --param2 {param2_name} --values2 {p2_str}"
    cmd += f" --internal-dim {results.get('internal_dim', 96)}"
    cmd += f" --seed {results.get('seed', 42)}"
    report.append(cmd)
    report.append("```")

    report.append("")

    # Footer
    report.append("---")
    report.append("")
    report.append(f"*Report generated by VFD Proof Dashboard v1.0*")
    report.append(f"*Timestamp: {datetime.now().isoformat()}*")

    return "\n".join(report)


def _format_grid_table(
    param1_name: str,
    param1_values: List,
    param2_name: str,
    param2_values: List,
    grid: List[List],
    format_func=str,
) -> str:
    """Format a 2D grid as a markdown table."""
    lines = []

    # Header row
    header = f"| {param1_name} \\ {param2_name} |"
    for p2 in param2_values:
        header += f" {p2} |"
    lines.append(header)

    # Separator
    sep = "|---|"
    for _ in param2_values:
        sep += "---|"
    lines.append(sep)

    # Data rows
    for i, p1 in enumerate(param1_values):
        row = f"| {p1} |"
        for j in range(len(param2_values)):
            try:
                val = grid[i][j]
                row += f" {format_func(val)} |"
            except (IndexError, TypeError):
                row += " N/A |"
        lines.append(row)

    return "\n".join(lines)


def write_sweep_report(sweep_dir: Path) -> Path:
    """
    Write a sweep report for a sweep directory.

    Args:
        sweep_dir: Path to sweep directory containing sweep_results.json

    Returns:
        Path to the written report file
    """
    # Load results
    results_file = sweep_dir / "sweep_results.json"

    if not results_file.exists():
        raise FileNotFoundError(f"Sweep results not found: {results_file}")

    results = json.loads(results_file.read_text())

    # Generate report
    report_content = generate_sweep_report(
        sweep_dir=sweep_dir,
        results=results,
    )

    # Write report
    report_file = sweep_dir / "SWEEP_REPORT.md"
    report_file.write_text(report_content)

    return report_file
