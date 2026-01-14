"""
Parameter Sweep Infrastructure.

Runs diagnostic over a grid of parameter values and generates phase map.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .core.config import RunConfig
    from .constraints import ClosureLevel


def run_sweep(
    config: "RunConfig",
    param1_name: str,
    param1_values: List[int],
    param2_name: str,
    param2_values: List[int],
    max_level: "ClosureLevel" = None,
    seed: Optional[int] = None,
    compute_stability: bool = False,
    output_dir: str = "runs/",
) -> Dict[str, Any]:
    """
    Run parameter sweep over a 2D grid.

    Args:
        config: Base configuration
        param1_name: First parameter name (e.g., 'cell_count')
        param1_values: Values for first parameter
        param2_name: Second parameter name (e.g., 'propagation_range')
        param2_values: Values for second parameter
        max_level: Maximum closure level to check
        seed: Random seed
        compute_stability: Whether to compute stability (default False for speed)
        output_dir: Output directory for incremental writes

    Returns:
        Dictionary with sweep results
    """
    from .core.logging import get_logger
    from .runner import run_diagnostic
    from .constraints import ClosureLevel

    logger = get_logger()

    if max_level is None:
        max_level = ClosureLevel.L4

    results_grid = {}
    total_runs = len(param1_values) * len(param2_values)
    run_count = 0

    # Setup incremental output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(output_dir) / f"sweep_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    incremental_file = sweep_dir / "sweep_results_partial.json"

    for p1 in param1_values:
        for p2 in param2_values:
            run_count += 1
            logger.info(f"Sweep run {run_count}/{total_runs}: {param1_name}={p1}, {param2_name}={p2}")

            # Clone config and modify parameters
            run_config = _clone_config(config)
            _set_parameter(run_config, param1_name, p1)
            _set_parameter(run_config, param2_name, p2)

            if seed is not None:
                run_config.seed = seed + run_count  # Vary seed per run

            run_config.run_name = f"sweep_{param1_name}_{p1}_{param2_name}_{p2}"

            try:
                # Run diagnostic without figures for speed
                result = run_diagnostic(
                    config=run_config,
                    max_level=max_level,
                    generate_figures=False,
                    seed=run_config.seed,
                    compute_stability=compute_stability,
                )

                ladder = result.get("ladder_result", {})
                invariants = result.get("invariants", {})

                # Extract min eigenvalue from spectrum
                spectrum = result.get("spectrum", {})
                min_eigenvalue = spectrum.get("min", 0.0) if isinstance(spectrum, dict) else 0.0

                # Extract per-family residuals
                family_residuals = _extract_family_residuals(ladder)

                results_grid[f"{p1}_{p2}"] = {
                    "param1": p1,
                    "param2": p2,
                    "all_passed": ladder.get("all_passed", False),
                    "max_level_passed": _level_to_int(ladder.get("max_level_passed")),
                    "total_residual": _get_total_residual(ladder),
                    "min_eigenvalue": min_eigenvalue,
                    "family_residuals": family_residuals,
                    "gating_stop_reason": ladder.get("gating_stop_reason"),
                    "weyl_error": invariants.get("weyl_error", 0.0),
                    "torsion_error": invariants.get("torsion_error", 0.0),
                }
            except Exception as e:
                logger.error(f"Sweep run failed: {e}")
                results_grid[f"{p1}_{p2}"] = {
                    "param1": p1,
                    "param2": p2,
                    "all_passed": False,
                    "max_level_passed": -1,
                    "total_residual": 1.0,
                    "min_eigenvalue": 0.0,
                    "family_residuals": {"EF": 0.0, "Symmetry": 0.0, "Positivity": 0.0, "Trace": 0.0},
                    "gating_stop_reason": str(e),
                    "error": str(e),
                }

            # Incremental save after each grid point (for resumability)
            partial_results = {
                "param1_name": param1_name,
                "param1_values": param1_values,
                "param2_name": param2_name,
                "param2_values": param2_values,
                "max_level": f"L{max_level.value}",
                "results_grid": results_grid,
                "completed_runs": run_count,
                "total_runs": total_runs,
                "timestamp": datetime.now().isoformat(),
            }
            with open(incremental_file, "w") as f:
                json.dump(partial_results, f, indent=2, default=str)

    # Build structured grids for analysis
    grids = _build_grids(results_grid, param1_values, param2_values)

    return {
        "param1_name": param1_name,
        "param1_values": param1_values,
        "param2_name": param2_name,
        "param2_values": param2_values,
        "max_level": f"L{max_level.value}",
        "results_grid": results_grid,
        "grids": grids,
        "timestamp": datetime.now().isoformat(),
    }


def save_sweep_outputs(
    results: Dict[str, Any],
    output_dir: str = "runs/",
) -> Path:
    """
    Save sweep results and generate figures.

    Args:
        results: Results from run_sweep()
        output_dir: Output directory

    Returns:
        Path to sweep output directory
    """
    from .figures.phase_map import generate_phase_map
    from .figures.positivity_wall import generate_positivity_wall_sweep

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sweep_dir = Path(output_dir) / f"sweep_{timestamp}"
    sweep_dir.mkdir(parents=True, exist_ok=True)

    # Save results JSON
    results_file = sweep_dir / "sweep_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Generate phase map (fig02) - now encoding max_level_passed 0..4
    try:
        phase_map = generate_phase_map(
            sweep_results=results,
            param1_name=results.get("param1_name", "param1"),
            param2_name=results.get("param2_name", "param2"),
        )
        phase_map_file = sweep_dir / "fig02_phase_map.png"
        with open(phase_map_file, "wb") as f:
            f.write(phase_map)
    except Exception as e:
        print(f"Warning: Could not generate phase map: {e}")

    # Generate positivity wall grid (fig04) - min eigenvalue heatmap over parameter grid
    try:
        grids = results.get("grids", {})
        min_eig_grid = grids.get("min_eigenvalue")
        if min_eig_grid is not None:
            positivity_wall = generate_positivity_wall_sweep(
                param1_name=results.get("param1_name", "param1"),
                param1_values=results.get("param1_values", []),
                param2_name=results.get("param2_name", "param2"),
                param2_values=results.get("param2_values", []),
                min_eigenvalue_grid=np.array(min_eig_grid),
            )
            positivity_wall_file = sweep_dir / "fig04_positivity_wall_grid.png"
            with open(positivity_wall_file, "wb") as f:
                f.write(positivity_wall)
    except Exception as e:
        print(f"Warning: Could not generate positivity wall grid: {e}")

    # Generate summary
    summary = _generate_sweep_summary(results)
    summary_file = sweep_dir / "summary.txt"
    with open(summary_file, "w") as f:
        f.write(summary)

    return sweep_dir


def _clone_config(config: "RunConfig") -> "RunConfig":
    """Create a deep copy of config."""
    import copy
    return copy.deepcopy(config)


def _set_parameter(config: "RunConfig", param_name: str, value: int):
    """Set a parameter on the config object."""
    if param_name == "cell_count":
        config.vfd.cell_count = value
    elif param_name == "internal_dim":
        config.vfd.internal_dim = value
        # Adjust orbit count to match
        config.vfd.orbit_count = value // config.vfd.orbit_size
    elif param_name == "propagation_range":
        config.vfd.local_propagation_L = value
    elif param_name == "probe_count":
        config.stability.probe_count = value
    elif param_name == "seed":
        config.seed = value
    else:
        # Try setting on vfd subconfig
        if hasattr(config.vfd, param_name):
            setattr(config.vfd, param_name, value)


def _level_to_int(level_str: Optional[str]) -> int:
    """Convert level string to int."""
    if level_str is None:
        return -1
    if isinstance(level_str, int):
        return level_str
    # Parse "L0", "L1", etc.
    if level_str.startswith("L"):
        try:
            return int(level_str[1:])
        except ValueError:
            return -1
    return -1


def _get_total_residual(ladder_result: Dict[str, Any]) -> float:
    """Get total residual from ladder result."""
    residuals = ladder_result.get("residuals_per_level", {})
    total = 0.0
    for level_data in residuals.values():
        total += level_data.get("total_residual", 0.0)
    return total if total > 0 else 1.0


def _extract_family_residuals(ladder_result: Dict[str, Any]) -> Dict[str, float]:
    """Extract per-family residuals from ladder result."""
    family_totals = {"EF": 0.0, "Symmetry": 0.0, "Positivity": 0.0, "Trace": 0.0}

    residuals = ladder_result.get("residuals_per_level", {})
    for level_data in residuals.values():
        family_residuals = level_data.get("family_residuals", {})
        for family, value in family_residuals.items():
            if family in family_totals:
                family_totals[family] += value

    return family_totals


def _build_grids(
    results_grid: Dict[str, Any],
    param1_values: List[int],
    param2_values: List[int],
) -> Dict[str, Any]:
    """
    Build structured 2D grids from results.

    Returns:
        Dictionary with grids:
        - max_level_passed: int grid
        - total_residual: float grid
        - min_eigenvalue: float grid
        - family_residuals: dict of float grids per family
        - stop_reason: string grid
    """
    n1, n2 = len(param1_values), len(param2_values)

    # Initialize grids
    max_level_grid = [[0 for _ in range(n2)] for _ in range(n1)]
    residual_grid = [[0.0 for _ in range(n2)] for _ in range(n1)]
    min_eig_grid = [[0.0 for _ in range(n2)] for _ in range(n1)]
    stop_reason_grid = [["" for _ in range(n2)] for _ in range(n1)]

    family_grids = {
        "EF": [[0.0 for _ in range(n2)] for _ in range(n1)],
        "Symmetry": [[0.0 for _ in range(n2)] for _ in range(n1)],
        "Positivity": [[0.0 for _ in range(n2)] for _ in range(n1)],
        "Trace": [[0.0 for _ in range(n2)] for _ in range(n1)],
    }

    # Fill grids
    for i, p1 in enumerate(param1_values):
        for j, p2 in enumerate(param2_values):
            key = f"{p1}_{p2}"
            if key in results_grid:
                result = results_grid[key]
                max_level_grid[i][j] = result.get("max_level_passed", -1)
                residual_grid[i][j] = result.get("total_residual", 0.0)
                min_eig_grid[i][j] = result.get("min_eigenvalue", 0.0)
                stop_reason_grid[i][j] = result.get("gating_stop_reason") or ""

                family_res = result.get("family_residuals", {})
                for family in family_grids:
                    family_grids[family][i][j] = family_res.get(family, 0.0)

    return {
        "max_level_passed": max_level_grid,
        "total_residual": residual_grid,
        "min_eigenvalue": min_eig_grid,
        "family_residuals": family_grids,
        "stop_reason": stop_reason_grid,
    }


def _generate_sweep_summary(results: Dict[str, Any]) -> str:
    """Generate text summary of sweep results."""
    lines = [
        "=" * 60,
        "PARAMETER SWEEP SUMMARY",
        "=" * 60,
        "",
        f"Parameter 1: {results.get('param1_name')} = {results.get('param1_values')}",
        f"Parameter 2: {results.get('param2_name')} = {results.get('param2_values')}",
        f"Max Level: {results.get('max_level')}",
        f"Timestamp: {results.get('timestamp')}",
        "",
        "-" * 60,
        "STATISTICS",
        "-" * 60,
        "",
    ]

    grid = results.get("results_grid", {})
    grids = results.get("grids", {})
    p1_values = results.get("param1_values", [])
    p2_values = results.get("param2_values", [])

    # Count statistics
    total = len(grid)
    passed = sum(1 for r in grid.values() if r.get("all_passed", False))

    # Level distribution
    level_counts = {i: 0 for i in range(-1, 5)}
    for r in grid.values():
        lvl = r.get("max_level_passed", -1)
        if lvl in level_counts:
            level_counts[lvl] += 1

    lines.append(f"Total runs: {total}")
    lines.append(f"All levels passed: {passed}")
    lines.append(f"Pass rate: {passed/total*100:.1f}%" if total > 0 else "N/A")
    lines.append("")
    lines.append("Level distribution:")
    for lvl in range(0, 5):
        pct = level_counts[lvl] / total * 100 if total > 0 else 0
        lines.append(f"  L{lvl}: {level_counts[lvl]:3d} runs ({pct:5.1f}%)")
    if level_counts[-1] > 0:
        lines.append(f"  Failed: {level_counts[-1]} runs")
    lines.append("")

    # Min eigenvalue statistics
    min_eigs = [r.get("min_eigenvalue", 0.0) for r in grid.values()]
    if min_eigs:
        lines.append("Min eigenvalue statistics:")
        lines.append(f"  Best (max): {max(min_eigs):.2e}")
        lines.append(f"  Worst (min): {min(min_eigs):.2e}")
        negative_count = sum(1 for e in min_eigs if e < 0)
        lines.append(f"  Negative: {negative_count}/{total} runs")
        lines.append("")

    # Most common stop reasons
    stop_reasons = [r.get("gating_stop_reason") for r in grid.values() if r.get("gating_stop_reason")]
    if stop_reasons:
        from collections import Counter
        reason_counts = Counter(stop_reasons)
        lines.append("Most common stop reasons:")
        for reason, count in reason_counts.most_common(3):
            lines.append(f"  {count:3d}x: {reason[:50]}")
        lines.append("")

    lines.append("-" * 60)
    lines.append("PASS/FAIL GRID")
    lines.append("-" * 60)
    lines.append("")

    # Grid view - header row
    header = f"{'':>8}"
    for p1 in p1_values:
        header += f"{p1:>8}"
    lines.append(header)

    for p2 in p2_values:
        row = f"{p2:>8}"
        for p1 in p1_values:
            key = f"{p1}_{p2}"
            if key in grid:
                status = "PASS" if grid[key].get("all_passed") else "FAIL"
                row += f"{status:>8}"
            else:
                row += f"{'N/A':>8}"
        lines.append(row)

    lines.append("")
    lines.append("-" * 60)
    lines.append("MAX LEVEL PASSED GRID")
    lines.append("-" * 60)
    lines.append("")

    # Level grid view - header row
    header = f"{'':>8}"
    for p1 in p1_values:
        header += f"{p1:>8}"
    lines.append(header)

    for p2 in p2_values:
        row = f"{p2:>8}"
        for p1 in p1_values:
            key = f"{p1}_{p2}"
            if key in grid:
                lvl = grid[key].get("max_level_passed", -1)
                lvl_str = f"L{lvl}" if lvl >= 0 else "FAIL"
                row += f"{lvl_str:>8}"
            else:
                row += f"{'N/A':>8}"
        lines.append(row)

    lines.append("")
    lines.append("=" * 60)

    return "\n".join(lines)
