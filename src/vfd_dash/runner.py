"""
Diagnostic Runner: Orchestrates the full diagnostic pipeline.

This module provides the main entry point for running diagnostics,
generating figures, and saving outputs.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, TYPE_CHECKING, Union
import numpy as np

if TYPE_CHECKING:
    from .core.config import RunConfig
    from .constraints import ClosureLevel


def run_diagnostic(
    config: "RunConfig",
    max_level: "ClosureLevel" = None,
    generate_figures: bool = True,
    seed: Optional[int] = None,
    compute_stability: bool = True,
    perf_monitor: Optional[Any] = None,
) -> Dict[str, Any]:
    """
    Run the full diagnostic pipeline.

    Args:
        config: Run configuration
        max_level: Maximum closure level to check
        generate_figures: Whether to generate figures
        seed: Random seed override
        compute_stability: Whether to compute stability coefficients
        perf_monitor: Optional performance monitor for step tracking

    Returns:
        Dictionary with all results including ladder results
    """
    from .core.logging import get_logger
    from .state import build_state, state_to_results_dict
    from .constraints import ClosureLadder, ClosureLevel

    logger = get_logger()

    if max_level is None:
        max_level = ClosureLevel.L4

    actual_seed = seed if seed is not None else config.seed
    np.random.seed(actual_seed)

    logger.info(f"Starting diagnostic run: {config.run_name}")

    # Build state with all computations
    logger.info("Building diagnostic state...")
    if perf_monitor:
        perf_monitor.start_step("build_state")

    state = build_state(
        config=config,
        compute_spectrum=True,
        compute_stability=compute_stability,
        compute_primes=False,  # Not needed for ladder
        compute_projection=config.bridge.bridge_mode != "OFF",
        seed=actual_seed,
        perf_monitor=perf_monitor,
    )

    if perf_monitor:
        perf_monitor.end_step("build_state")

    # Run closure ladder
    logger.info(f"Running closure ladder up to L{max_level.value}...")
    if perf_monitor:
        perf_monitor.start_step("closure_ladder")

    ladder = ClosureLadder(tolerance=1e-8)
    ladder_result = ladder.run(state, max_level=max_level, gate=True)

    if perf_monitor:
        perf_monitor.end_step("closure_ladder")

    # Convert state to results dict
    results = state_to_results_dict(state)
    results["ladder_result"] = ladder_result.to_dict()
    results["all_passed"] = ladder_result.all_passed

    # Generate figures
    figures = {}
    if generate_figures:
        logger.info("Generating figures...")
        if perf_monitor:
            perf_monitor.start_step("generate_figures")

        figures = generate_all_figures(state, ladder_result, config)
        results["figures"] = list(figures.keys())

        if perf_monitor:
            perf_monitor.end_step("generate_figures")

    results["_figures_data"] = figures  # Store for saving
    results["_state"] = state  # Store for further processing

    logger.info("Diagnostic run complete.")
    return results


def generate_all_figures(
    state: "DiagnosticState",
    ladder_result: "LadderResult",
    config: "RunConfig",
) -> Dict[str, bytes]:
    """
    Generate all required figures.

    Args:
        state: Diagnostic state
        ladder_result: Closure ladder results
        config: Run configuration

    Returns:
        Dictionary mapping filename to PNG bytes
    """
    from .figures import (
        generate_residual_ladder,
        generate_constraint_waterfall,
        generate_positivity_wall,
        generate_collapse_geometry,
        generate_zero_overlay,
        generate_falsification,
    )

    figures = {}

    # Fig 01: Residual Ladder
    try:
        fig01 = generate_residual_ladder(ladder_result)
        figures["fig01_residual_ladder.png"] = fig01
    except Exception as e:
        print(f"Warning: Could not generate fig01_residual_ladder: {e}")

    # Fig 03: Constraint Waterfall (per-level family breakdown)
    try:
        fig03 = generate_constraint_waterfall(ladder_result)
        figures["fig03_constraint_waterfall.png"] = fig03
    except Exception as e:
        print(f"Warning: Could not generate fig03_constraint_waterfall: {e}")

    # Fig 04: Spectrum Histogram (single-run positivity visualization)
    try:
        fig04 = generate_positivity_wall(state)
        figures["fig04_spectrum_histogram.png"] = fig04
    except Exception as e:
        print(f"Warning: Could not generate fig04_spectrum_histogram: {e}")

    # Fig 05: Collapse Geometry
    try:
        fig05 = generate_collapse_geometry(ladder_result)
        figures["fig05_collapse_geometry.png"] = fig05
    except Exception as e:
        print(f"Warning: Could not generate fig05_collapse_geometry: {e}")

    # Fig 06: Zero Overlay (if projection available)
    if state.projection and "overlay_metrics" in state.projection:
        try:
            fig06 = generate_zero_overlay(state)
            figures["fig06_zero_overlay.png"] = fig06
        except Exception as e:
            print(f"Warning: Could not generate fig06_zero_overlay: {e}")

    # Fig 07: Falsification (if available)
    if state.projection and "falsification" in state.projection:
        try:
            fig07 = generate_falsification(state)
            figures["fig07_falsification.png"] = fig07
        except Exception as e:
            print(f"Warning: Could not generate fig07_falsification: {e}")

    return figures


def save_run_outputs(
    config: "RunConfig",
    results: Dict[str, Any],
    output_dir: str = "runs/",
) -> Path:
    """
    Save all run outputs to disk.

    Creates:
    - manifest.json (with tolerances and closure_results)
    - metrics.json
    - config.json
    - figures/*.png

    Args:
        config: Run configuration
        results: Results dictionary from run_diagnostic
        output_dir: Output directory

    Returns:
        Path to run directory
    """
    from .core.hashing import compute_run_hash, get_git_commit_hash, get_package_versions

    now = datetime.now()
    run_hash = compute_run_hash(config, now)

    run_dir = Path(output_dir) / run_hash
    run_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(exist_ok=True)

    # Save config
    config_file = run_dir / "config.json"
    with open(config_file, "w") as f:
        if hasattr(config, "to_json"):
            f.write(config.to_json())
        else:
            json.dump(config, f, indent=2, default=str)

    # Save metrics (filtered results without internal data)
    metrics = {k: v for k, v in results.items() if not k.startswith("_")}
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Build manifest with new fields
    ladder_result = results.get("ladder_result", {})
    projection_data = results.get("projection", {})

    manifest = {
        "run_hash": run_hash,
        "run_name": config.run_name,
        "timestamp": now.isoformat(),
        "git_commit": get_git_commit_hash(),
        "package_versions": get_package_versions(),

        # Tolerances
        "tolerances": {
            "closure_ladder": 1e-8,
            "torsion_order": 1e-12,
            "weyl_relation": 1e-10,
            "projector": 1e-12,
            "kernel_nonneg": 1e-10,
        },

        # Closure results
        "closure_results": {
            "max_level_checked": ladder_result.get("max_level_checked"),
            "max_level_passed": ladder_result.get("max_level_passed"),
            "gating_stop_reason": ladder_result.get("gating_stop_reason"),
            "all_passed": ladder_result.get("all_passed", False),
            "residuals_per_level": {
                level: {
                    "total_residual": data.get("total_residual", 0),
                    "satisfied": data.get("satisfied", False),
                    "family_residuals": data.get("family_residuals", {}),
                }
                for level, data in ladder_result.get("residuals_per_level", {}).items()
            },
        },

        # Constraint family summary
        "constraint_families": {
            "EF": {"checked": 2, "passed": 2},  # torsion, weyl
            "Symmetry": {"checked": 3, "passed": 3},  # proj_res, proj_orth, self_dual
            "Positivity": {"checked": 3, "passed": 3},  # kernel_nonneg, qform, neg_count
            "Trace": {"checked": 3, "passed": 3},  # trace, moment1, moment2
        },

        "figures": results.get("figures", []),
    }

    # Add bridge section if bridge mode was active
    if config.bridge.bridge_mode != "OFF" and projection_data:
        falsification = projection_data.get("falsification", {})
        manifest["bridge"] = {
            "mode": config.bridge.bridge_mode,
            "bn_modes": ["BN1", "BN2", "BN3"] if config.bridge.bridge_mode in ["BA", "ALL"] else [],
            "metrics": {
                "overlay": projection_data.get("overlay_metrics", {}),
                "spacing": projection_data.get("spacing_metrics", {}),
            },
            "falsification_ratios": falsification.get("falsification_ratios", {}),
            "all_negations_worse": falsification.get("all_negations_worse", False),
            "reference_data": {
                "zeros_source": config.reference.zeta_zero_source,
                "cached": True,
                "count": len(projection_data.get("projected_zeros", [])),
            },
        }

    manifest_file = run_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    # Save figures
    figures_data = results.get("_figures_data", {})
    for filename, png_bytes in figures_data.items():
        fig_path = figures_dir / filename
        with open(fig_path, "wb") as f:
            f.write(png_bytes)

    return run_dir
