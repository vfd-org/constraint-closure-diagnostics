#!/usr/bin/env python3
"""
RH Constraint-Diagnostic CLI.

Usage:
    rhdiag run [options]
    rhdiag sweep [options]
    rhdiag bundle [options]
    rhdiag replay --run-hash <hash>
    rhdiag list-runs

This is a DIAGNOSTIC AND VISUALIZATION FRAMEWORK.
It does NOT claim a proof of the Riemann Hypothesis.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Optional, List
from datetime import datetime

# Safety disclaimer - MUST be printed at start
DISCLAIMER = """
============================================================
 RH CONSTRAINT-DIAGNOSTIC TOOL
============================================================
 IMPORTANT: This is a diagnostic and visualization framework.
 It does NOT prove or claim to prove the Riemann Hypothesis.

 - VFD invariants are internal consistency checks.
 - The Bridge Axiom is a testable hypothesis, not proven.
 - Numerical agreement is not mathematical proof.
============================================================
"""


def print_disclaimer():
    """Print safety disclaimer."""
    print(DISCLAIMER, file=sys.stderr)


def cmd_run(args):
    """Run diagnostic analysis."""
    import numpy as np
    from .core.config import RunConfig, get_default_config
    from .core.logging import setup_logging, get_logger
    from .state import build_state, state_to_results_dict
    from .constraints import ClosureLadder, ClosureLevel
    from .runner import run_diagnostic, save_run_outputs

    # Setup performance monitoring if enabled
    perf_monitor = None
    if getattr(args, 'perf', False):
        from .diagnostics.perf import PerfMonitor, set_monitor
        perf_monitor = PerfMonitor(enabled=True, trace_exceptions=getattr(args, 'trace', False))
        set_monitor(perf_monitor)
        perf_monitor.start_step("cli_init")

    setup_logging()
    logger = get_logger()

    # Load or create config
    if args.config:
        logger.info(f"Loading config from {args.config}")
        config = RunConfig.from_file(args.config)
    else:
        config = get_default_config()

    # Override with CLI args
    if args.seed is not None:
        config.seed = args.seed
    if args.cell_count is not None:
        config.vfd.cell_count = args.cell_count

    # Handle orbit_size and orbit_count
    orbit_size = args.orbit_size if args.orbit_size is not None else 12
    config.vfd.orbit_size = orbit_size

    # Validate orbit_size (canonical Weyl condition)
    if orbit_size != 12:
        if not args.allow_noncanonical_orbit:
            print(f"\033[91mERROR: orbit_size={orbit_size} is noncanonical.\033[0m", file=sys.stderr)
            print("The Weyl relation T S T^{-1} = ω S requires orbit_size=12.", file=sys.stderr)
            print("Use --allow-noncanonical-orbit to override (NOT RECOMMENDED).", file=sys.stderr)
            return 1
        else:
            print("\033[93mWARNING: Noncanonical orbit config; Weyl relation may fail structurally.\033[0m", file=sys.stderr)

    # Resolve orbit structure: internal_dim = orbit_count * orbit_size
    if args.internal_dim is not None:
        # internal_dim explicitly set -> compute orbit_count
        if args.internal_dim % orbit_size != 0:
            print(f"\033[91mERROR: internal_dim={args.internal_dim} not divisible by orbit_size={orbit_size}.\033[0m", file=sys.stderr)
            return 1
        config.vfd.internal_dim = args.internal_dim
        config.vfd.orbit_count = args.internal_dim // orbit_size
    elif args.orbit_count is not None:
        # orbit_count explicitly set -> compute internal_dim
        config.vfd.orbit_count = args.orbit_count
        config.vfd.internal_dim = args.orbit_count * orbit_size
    else:
        # Neither specified -> use default orbit_count and compute internal_dim
        # This ensures internal_dim is consistent with orbit_size
        config.vfd.internal_dim = config.vfd.orbit_count * orbit_size

    if args.propagation_range is not None:
        config.vfd.local_propagation_L = args.propagation_range
    if args.probe_count is not None:
        config.stability.probe_count = args.probe_count
    if args.run_name:
        config.run_name = args.run_name
    else:
        config.run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Handle bridge mode
    config.bridge.bridge_mode = args.bridge_mode

    config.output.out_dir = args.outdir

    # Parse max level
    max_level = ClosureLevel.L4
    if args.max_level:
        levels = ClosureLevel.parse_range(args.max_level)
        max_level = max(levels)

    # Print config summary
    print(f"\n--- Configuration ---")
    print(f"Run Name: {config.run_name}")
    print(f"Seed: {config.seed}")
    print(f"Cells: {config.vfd.cell_count}")
    print(f"Internal Dim: {config.vfd.internal_dim}")
    print(f"Orbit Structure: {config.vfd.orbit_count} orbits × {config.vfd.orbit_size}")
    print(f"Propagation Range: {config.vfd.local_propagation_L}")
    print(f"Max Level: L{max_level.value}")
    print(f"Bridge Mode: {config.bridge.bridge_mode}")
    print(f"Stability: {args.stability}")
    print(f"Output Dir: {config.output.out_dir}")
    print()

    # Determine stability setting
    compute_stability = (args.stability == "on")

    # Apply adaptive probe count if stability is on
    if compute_stability and args.probe_count is None:
        total_dim = config.vfd.cell_count * config.vfd.internal_dim
        adaptive_cap = max(64, total_dim // 20)
        config.stability.probe_count = min(config.stability.probe_count, adaptive_cap)

    if perf_monitor:
        perf_monitor.end_step("cli_init")
        perf_monitor.snapshot("before_diagnostic")

    # Run diagnostic with exception handling
    try:
        if perf_monitor:
            perf_monitor.start_step("run_diagnostic")

        results = run_diagnostic(
            config=config,
            max_level=max_level,
            generate_figures=not args.no_figures,
            compute_stability=compute_stability,
            perf_monitor=perf_monitor,
        )

        if perf_monitor:
            perf_monitor.end_step("run_diagnostic")

    except Exception as e:
        if getattr(args, 'trace', False):
            print("\n[TRACE] Full exception traceback:", file=sys.stderr)
            traceback.print_exc()
        if perf_monitor:
            perf_monitor.end_step("run_diagnostic", error=str(e))
            perf_monitor.print_summary()
        raise

    # Print results summary
    print_results_summary(results)

    # Save outputs
    run_dir = None
    if not args.no_export:
        if perf_monitor:
            perf_monitor.start_step("save_outputs")

        run_dir = save_run_outputs(
            config=config,
            results=results,
            output_dir=args.outdir,
        )

        if perf_monitor:
            perf_monitor.end_step("save_outputs")

        print(f"\n--- Output ---")
        print(f"Run directory: {run_dir}")
        print(f"Manifest: {run_dir}/manifest.json")
        print(f"Figures: {run_dir}/figures/")

    # Save perf report
    if perf_monitor:
        perf_monitor.print_summary()
        if run_dir:
            perf_monitor.save(str(run_dir / "perf.json"))
        else:
            perf_monitor.save(f"runs/perf_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

    return 0 if results.get("all_passed", False) else 1


def cmd_sweep(args):
    """Run parameter sweep."""
    from .core.config import get_default_config
    from .core.logging import setup_logging
    from .sweep import run_sweep, save_sweep_outputs
    from .constraints import ClosureLevel

    setup_logging()

    config = get_default_config()
    config.seed = args.seed or 42

    # Set internal dimensions for sweep (canonical orbit_size=12)
    orbit_size = 12
    internal_dim = args.internal_dim
    if internal_dim % orbit_size != 0:
        print(f"\033[91mERROR: internal_dim={internal_dim} not divisible by orbit_size={orbit_size}.\033[0m", file=sys.stderr)
        return 1
    config.vfd.internal_dim = internal_dim
    config.vfd.orbit_size = orbit_size
    config.vfd.orbit_count = internal_dim // orbit_size

    # Parse parameter values
    param1_values = [int(x) for x in args.values1.split(",")]
    param2_values = [int(x) for x in args.values2.split(",")]

    # Check grid size limit
    total_runs = len(param1_values) * len(param2_values)
    if total_runs > args.max_grid_runs and not args.force_large:
        print(f"\033[91mERROR: Grid size {total_runs} exceeds limit {args.max_grid_runs}.\033[0m", file=sys.stderr)
        print(f"Use --force-large to override, or reduce parameter values.", file=sys.stderr)
        return 1

    max_level = ClosureLevel.from_string(args.max_level) if args.max_level else ClosureLevel.L4

    # Stability setting for sweep (default off)
    compute_stability = (args.stability == "on")

    print(f"\n--- Parameter Sweep ---")
    print(f"Param 1: {args.param1} = {param1_values}")
    print(f"Param 2: {args.param2} = {param2_values}")
    print(f"Internal Dim: {internal_dim} ({config.vfd.orbit_count} orbits × {orbit_size})")
    print(f"Max Level: L{max_level.value}")
    print(f"Stability: {args.stability}")
    print(f"Total runs: {total_runs}")
    print()

    # Run sweep
    results = run_sweep(
        config=config,
        param1_name=args.param1,
        param1_values=param1_values,
        param2_name=args.param2,
        param2_values=param2_values,
        max_level=max_level,
        compute_stability=compute_stability,
        output_dir=args.outdir,
    )

    # Save outputs
    sweep_dir = save_sweep_outputs(
        results=results,
        output_dir=args.outdir,
    )

    print(f"\n--- Sweep Complete ---")
    print(f"Output directory: {sweep_dir}")
    print(f"Phase map: {sweep_dir}/fig02_phase_map.png")

    return 0


def cmd_replay(args):
    """Replay a previous run."""
    from .core.config import RunConfig
    from .runner import run_diagnostic, save_run_outputs

    run_dir = Path(args.outdir) / args.run_hash
    config_file = run_dir / "config.json"

    if not config_file.exists():
        print(f"Error: Config not found at {config_file}", file=sys.stderr)
        return 1

    print(f"Replaying run: {args.run_hash}")
    config = RunConfig.from_file(str(config_file))

    # Re-run with same config
    results = run_diagnostic(config=config, generate_figures=True)
    print_results_summary(results)

    return 0 if results.get("all_passed", False) else 1


def cmd_list_runs(args):
    """List available runs."""
    from .io.export_bundle import list_available_bundles

    bundles = list_available_bundles(args.outdir)

    if not bundles:
        print("No runs found.")
        return 0

    print(f"\n{'Hash':<18} {'Name':<25} {'Timestamp':<25}")
    print("-" * 70)
    for b in bundles[:20]:
        print(f"{b.get('run_hash', 'N/A'):<18} {b.get('run_name', 'N/A'):<25} {b.get('timestamp', 'N/A'):<25}")

    if len(bundles) > 20:
        print(f"... and {len(bundles) - 20} more")

    return 0


def cmd_bundle(args):
    """Generate complete release bundle."""
    from datetime import datetime
    from pathlib import Path
    from .core.config import RunConfig, get_default_config
    from .core.logging import setup_logging
    from .runner import run_diagnostic, save_run_outputs
    from .sweep import run_sweep, save_sweep_outputs
    from .constraints import ClosureLevel
    from .reports.release_report import write_release_report
    from .reports.sweep_report import write_sweep_report

    setup_logging()

    # Create release directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    release_dir = Path(args.outdir) / f"release_{timestamp}"
    release_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("GENERATING RELEASE BUNDLE")
    print(f"{'='*60}")
    print(f"Output directory: {release_dir}")
    print()

    generated_runs = []
    generated_sweeps = []

    # Step 1: Standard run (bridge OFF)
    print("--- Step 1: Standard Run (Bridge OFF) ---")
    config = get_default_config()
    config.seed = args.seed
    config.vfd.cell_count = args.cell_count
    config.vfd.internal_dim = args.internal_dim
    config.vfd.orbit_size = 12
    config.vfd.orbit_count = args.internal_dim // 12
    config.vfd.local_propagation_L = args.propagation_range
    config.bridge.bridge_mode = "OFF"
    config.run_name = f"standard_run_{timestamp}"

    try:
        results = run_diagnostic(
            config=config,
            max_level=ClosureLevel.L4,
            generate_figures=True,
            compute_stability=False,
        )
        run_dir = save_run_outputs(config, results, str(release_dir))
        write_release_report(run_dir)
        generated_runs.append(("standard", run_dir))
        print(f"  Created: {run_dir}")
        print(f"  Status: {'PASS' if results.get('all_passed') else 'FAIL'}")
    except Exception as e:
        print(f"  ERROR: {e}")

    print()

    # Step 2: Bridge run (BA mode)
    if not args.skip_bridge:
        print("--- Step 2: Bridge Run (BA Mode) ---")
        config.bridge.bridge_mode = "BA"
        config.run_name = f"bridge_run_{timestamp}"

        try:
            results = run_diagnostic(
                config=config,
                max_level=ClosureLevel.L4,
                generate_figures=True,
                compute_stability=False,
            )
            run_dir = save_run_outputs(config, results, str(release_dir))
            write_release_report(run_dir)
            generated_runs.append(("bridge", run_dir))
            print(f"  Created: {run_dir}")
            print(f"  Status: {'PASS' if results.get('all_passed') else 'FAIL'}")
        except Exception as e:
            print(f"  ERROR: {e}")

        print()

    # Step 3: Small sweep (3x3)
    if not args.skip_sweep:
        print("--- Step 3: Parameter Sweep (3x3) ---")
        config.bridge.bridge_mode = "OFF"

        try:
            sweep_results = run_sweep(
                config=config,
                param1_name="cell_count",
                param1_values=[16, 32, 64],
                param2_name="propagation_range",
                param2_values=[1, 2, 3],
                max_level=ClosureLevel.L4,
                compute_stability=False,
                output_dir=str(release_dir),
            )
            sweep_dir = save_sweep_outputs(sweep_results, str(release_dir))
            write_sweep_report(sweep_dir)
            generated_sweeps.append(sweep_dir)
            print(f"  Created: {sweep_dir}")
        except Exception as e:
            print(f"  ERROR: {e}")

        print()

    # Step 4: Generate bundle report
    print("--- Step 4: Bundle Report ---")
    bundle_report = _generate_bundle_report(
        release_dir=release_dir,
        generated_runs=generated_runs,
        generated_sweeps=generated_sweeps,
        args=args,
    )
    bundle_report_file = release_dir / "BUNDLE_REPORT.md"
    bundle_report_file.write_text(bundle_report)
    print(f"  Created: {bundle_report_file}")

    print()
    print(f"{'='*60}")
    print("BUNDLE GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Release directory: {release_dir}")
    print(f"Runs generated: {len(generated_runs)}")
    print(f"Sweeps generated: {len(generated_sweeps)}")
    print()

    return 0


def _generate_bundle_report(release_dir, generated_runs, generated_sweeps, args):
    """Generate the top-level bundle report."""
    from datetime import datetime

    lines = []
    lines.append("# Release Bundle Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().isoformat()}")
    lines.append(f"**Directory:** `{release_dir}`")
    lines.append("")

    # Safety notice
    lines.append("## Important Notice")
    lines.append("")
    lines.append("This bundle is generated by a **diagnostic and visualization framework**.")
    lines.append("It does not claim, demonstrate, or provide a proof of RH.")
    lines.append("")

    # Configuration
    lines.append("## Bundle Configuration")
    lines.append("")
    lines.append("| Parameter | Value |")
    lines.append("|-----------|-------|")
    lines.append(f"| Cell Count | {args.cell_count} |")
    lines.append(f"| Internal Dim | {args.internal_dim} |")
    lines.append(f"| Propagation Range | {args.propagation_range} |")
    lines.append(f"| Seed | {args.seed} |")
    lines.append(f"| Skip Sweep | {args.skip_sweep} |")
    lines.append(f"| Skip Bridge | {args.skip_bridge} |")
    lines.append("")

    # Generated runs
    lines.append("## Generated Runs")
    lines.append("")
    if generated_runs:
        lines.append("| Type | Directory | Report |")
        lines.append("|------|-----------|--------|")
        for run_type, run_dir in generated_runs:
            report_file = run_dir / "RELEASE_REPORT.md"
            report_exists = "Yes" if report_file.exists() else "No"
            lines.append(f"| {run_type} | `{run_dir.name}` | {report_exists} |")
    else:
        lines.append("No runs generated.")
    lines.append("")

    # Generated sweeps
    lines.append("## Generated Sweeps")
    lines.append("")
    if generated_sweeps:
        lines.append("| Directory | Report |")
        lines.append("|-----------|--------|")
        for sweep_dir in generated_sweeps:
            report_file = sweep_dir / "SWEEP_REPORT.md"
            report_exists = "Yes" if report_file.exists() else "No"
            lines.append(f"| `{sweep_dir.name}` | {report_exists} |")
    else:
        lines.append("No sweeps generated.")
    lines.append("")

    # Contents summary
    lines.append("## Bundle Contents")
    lines.append("")
    lines.append("```")
    lines.append(f"{release_dir.name}/")
    lines.append("├── BUNDLE_REPORT.md")
    for run_type, run_dir in generated_runs:
        lines.append(f"├── {run_dir.name}/")
        lines.append(f"│   ├── config.json")
        lines.append(f"│   ├── manifest.json")
        lines.append(f"│   ├── metrics.json")
        lines.append(f"│   ├── RELEASE_REPORT.md")
        lines.append(f"│   └── figures/")
    for sweep_dir in generated_sweeps:
        lines.append(f"└── {sweep_dir.name}/")
        lines.append(f"    ├── sweep_results.json")
        lines.append(f"    ├── SWEEP_REPORT.md")
        lines.append(f"    ├── fig02_phase_map.png")
        lines.append(f"    └── fig04_positivity_wall_grid.png")
    lines.append("```")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("")
    lines.append("*Generated by VFD Proof Dashboard v1.0*")

    return "\n".join(lines)


def print_results_summary(results: dict):
    """Print summary of diagnostic results."""
    print("\n--- Closure Ladder Results ---")

    ladder_result = results.get("ladder_result", {})
    level_results = ladder_result.get("residuals_per_level", {})

    for level_name in ["L0", "L1", "L2", "L3", "L4"]:
        level_data = level_results.get(level_name, {})
        if level_data:
            passed = level_data.get("satisfied", False)
            residual = level_data.get("total_residual", 0.0)
            status = "\033[92mPASS\033[0m" if passed else "\033[91mFAIL\033[0m"
            print(f"  {level_name}: {status} (residual: {residual:.2e})")

            # Show family breakdown
            family_residuals = level_data.get("family_residuals", {})
            for family, res in family_residuals.items():
                print(f"       {family}: {res:.2e}")

    # Overall status
    all_passed = ladder_result.get("all_passed", False)
    max_passed = ladder_result.get("max_level_passed", "None")
    stop_reason = ladder_result.get("gating_stop_reason")

    print()
    if all_passed:
        print("\033[92m[ALL LEVELS PASSED]\033[0m")
    else:
        print(f"\033[93m[MAX LEVEL PASSED: {max_passed}]\033[0m")
        if stop_reason:
            print(f"Stop reason: {stop_reason}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RH Constraint-Diagnostic Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  rhdiag run --seed 42 --cell-count 16
  rhdiag run --max-level L2 --no-figures
  rhdiag sweep --param1 cell_count --values1 8,16,32 --param2 propagation_range --values2 1,2,3
  rhdiag list-runs
  rhdiag replay --run-hash abc123
"""
    )

    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress disclaimer")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run diagnostic analysis")
    run_parser.add_argument("--config", type=str, help="Path to config JSON file")
    run_parser.add_argument("--run-name", type=str, help="Run name")
    run_parser.add_argument("--seed", type=int, help="Random seed")
    run_parser.add_argument("--cell-count", type=int, help="Number of cells")
    run_parser.add_argument("--internal-dim", type=int, help="Internal dimension")
    run_parser.add_argument("--orbit-size", type=int, default=12, help="Orbit size (default: 12, required for Weyl)")
    run_parser.add_argument("--orbit-count", type=int, help="Number of orbits (default: internal_dim // orbit_size)")
    run_parser.add_argument("--propagation-range", type=int, help="Kernel coupling range")
    run_parser.add_argument("--probe-count", type=int, help="Number of stability probes")
    run_parser.add_argument("--max-level", type=str, default="L4", help="Max closure level (L0..L4)")
    run_parser.add_argument("--bridge-mode", type=str, default="OFF",
                           choices=["OFF", "BA", "BN1", "BN2", "BN3", "ALL"],
                           help="Bridge mode (default: OFF)")
    run_parser.add_argument("--allow-noncanonical-orbit", action="store_true",
                           help="Allow orbit_size != 12 (Weyl may fail)")
    run_parser.add_argument("--outdir", type=str, default="runs/", help="Output directory")
    run_parser.add_argument("--no-figures", action="store_true", help="Skip figure generation")
    run_parser.add_argument("--no-export", action="store_true", help="Skip export bundle")
    run_parser.add_argument("--stability", type=str, default="off", choices=["on", "off"],
                           help="Stability computation (default: off for speed)")
    run_parser.add_argument("--perf", action="store_true", help="Enable performance logging")
    run_parser.add_argument("--trace", action="store_true", help="Enable full stack traces on exceptions")

    # Sweep command
    sweep_parser = subparsers.add_parser("sweep", help="Run parameter sweep")
    sweep_parser.add_argument("--param1", type=str, default="cell_count", help="First parameter name")
    sweep_parser.add_argument("--values1", type=str, default="8,16,32", help="First parameter values (comma-sep)")
    sweep_parser.add_argument("--param2", type=str, default="propagation_range", help="Second parameter name")
    sweep_parser.add_argument("--values2", type=str, default="1,2,3", help="Second parameter values (comma-sep)")
    sweep_parser.add_argument("--max-level", type=str, default="L4", help="Max closure level")
    sweep_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    sweep_parser.add_argument("--internal-dim", type=int, default=96, help="Internal dimension (default: 96 for fast sweeps)")
    sweep_parser.add_argument("--outdir", type=str, default="runs/", help="Output directory")
    sweep_parser.add_argument("--stability", type=str, default="off", choices=["on", "off"],
                             help="Stability computation (default: off for sweeps)")
    sweep_parser.add_argument("--max-grid-runs", type=int, default=9,
                             help="Maximum grid points (default: 9, use --force-large for more)")
    sweep_parser.add_argument("--force-large", action="store_true",
                             help="Allow grid sizes larger than --max-grid-runs")
    sweep_parser.add_argument("--perf", action="store_true", help="Enable performance logging")
    sweep_parser.add_argument("--trace", action="store_true", help="Enable full stack traces on exceptions")

    # Replay command
    replay_parser = subparsers.add_parser("replay", help="Replay a previous run")
    replay_parser.add_argument("--run-hash", type=str, required=True, help="Run hash to replay")
    replay_parser.add_argument("--outdir", type=str, default="runs/", help="Output directory")

    # List runs command
    list_parser = subparsers.add_parser("list-runs", help="List available runs")
    list_parser.add_argument("--outdir", type=str, default="runs/", help="Output directory")

    # Bundle command - generate complete release bundle
    bundle_parser = subparsers.add_parser("bundle", help="Generate complete release bundle")
    bundle_parser.add_argument("--internal-dim", type=int, default=12,
                               help="Internal dimension (default: 12 for shareable profile)")
    bundle_parser.add_argument("--cell-count", type=int, default=32,
                               help="Cell count (default: 32)")
    bundle_parser.add_argument("--propagation-range", type=int, default=3,
                               help="Propagation range (default: 3)")
    bundle_parser.add_argument("--seed", type=int, default=42,
                               help="Random seed (default: 42)")
    bundle_parser.add_argument("--skip-sweep", action="store_true",
                               help="Skip sweep generation")
    bundle_parser.add_argument("--skip-bridge", action="store_true",
                               help="Skip bridge run generation")
    bundle_parser.add_argument("--perf", action="store_true",
                               help="Enable performance logging")
    bundle_parser.add_argument("--outdir", type=str, default="runs/",
                               help="Output directory")

    args = parser.parse_args()

    # Print disclaimer unless quiet mode
    if not args.quiet:
        print_disclaimer()

    # Dispatch command
    if args.command == "run":
        return cmd_run(args)
    elif args.command == "sweep":
        return cmd_sweep(args)
    elif args.command == "replay":
        return cmd_replay(args)
    elif args.command == "list-runs":
        return cmd_list_runs(args)
    elif args.command == "bundle":
        return cmd_bundle(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())
