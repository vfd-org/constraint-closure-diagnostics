#!/usr/bin/env python3
"""
Sync paper figures from bundle output to paper/figures/.

Usage:
    python tools/sync_paper_figures.py --bundle-dir runs/release_20260113_225750

This script:
1. Locates the production run hash folder (with bridge figures)
2. Locates the sweep output folder
3. Copies required PNGs to paper/figures/
4. Prints which files were copied
"""

import argparse
import shutil
from pathlib import Path
import sys


# Required figures for the paper
PRODUCTION_RUN_FIGURES = [
    "fig01_residual_ladder.png",
    "fig03_constraint_waterfall.png",
    "fig04_spectrum_histogram.png",
    "fig05_collapse_geometry.png",
    "fig06_zero_overlay.png",
    "fig07_falsification.png",
]

SWEEP_FIGURES = [
    "fig02_phase_map.png",
    "fig04_positivity_wall_grid.png",
]


def find_production_run(bundle_dir: Path) -> Path:
    """Find the production run folder (one with bridge figures like fig06, fig07)."""
    for subdir in bundle_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("sweep_"):
            figures_dir = subdir / "figures"
            if figures_dir.exists():
                # Check if it has bridge figures
                if (figures_dir / "fig06_zero_overlay.png").exists():
                    return subdir
    raise FileNotFoundError("Could not find production run with bridge figures")


def find_sweep_folder(bundle_dir: Path) -> Path:
    """Find the sweep output folder."""
    for subdir in bundle_dir.iterdir():
        if subdir.is_dir() and subdir.name.startswith("sweep_"):
            if (subdir / "fig02_phase_map.png").exists():
                return subdir
    raise FileNotFoundError("Could not find sweep folder with phase map")


def sync_figures(bundle_dir: Path, paper_figures_dir: Path, dry_run: bool = False):
    """Copy figures from bundle to paper/figures/."""
    copied = []
    errors = []

    # Find source directories
    try:
        prod_run = find_production_run(bundle_dir)
        print(f"Found production run: {prod_run.name}")
    except FileNotFoundError as e:
        errors.append(str(e))
        prod_run = None

    try:
        sweep_dir = find_sweep_folder(bundle_dir)
        print(f"Found sweep folder: {sweep_dir.name}")
    except FileNotFoundError as e:
        errors.append(str(e))
        sweep_dir = None

    # Create output directory
    if not dry_run:
        paper_figures_dir.mkdir(parents=True, exist_ok=True)

    # Copy production run figures
    if prod_run:
        figures_dir = prod_run / "figures"
        for fig in PRODUCTION_RUN_FIGURES:
            src = figures_dir / fig
            dst = paper_figures_dir / fig
            if src.exists():
                if not dry_run:
                    shutil.copy2(src, dst)
                copied.append(f"{src} -> {dst}")
            else:
                errors.append(f"Missing: {src}")

    # Copy sweep figures
    if sweep_dir:
        for fig in SWEEP_FIGURES:
            src = sweep_dir / fig
            dst = paper_figures_dir / fig
            if src.exists():
                if not dry_run:
                    shutil.copy2(src, dst)
                copied.append(f"{src} -> {dst}")
            else:
                errors.append(f"Missing: {src}")

    return copied, errors


def main():
    parser = argparse.ArgumentParser(
        description="Sync paper figures from bundle output to paper/figures/"
    )
    parser.add_argument(
        "--bundle-dir",
        type=Path,
        required=True,
        help="Path to bundle output directory (e.g., runs/release_20260113_225750)"
    )
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Output directory for paper figures (default: paper/figures)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be copied without copying"
    )

    args = parser.parse_args()

    if not args.bundle_dir.exists():
        print(f"Error: Bundle directory not found: {args.bundle_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Bundle directory: {args.bundle_dir}")
    print(f"Output directory: {args.paper_dir}")
    if args.dry_run:
        print("(Dry run - no files will be copied)")
    print()

    copied, errors = sync_figures(args.bundle_dir, args.paper_dir, args.dry_run)

    if copied:
        print(f"\nCopied {len(copied)} files:")
        for c in copied:
            print(f"  {c}")

    if errors:
        print(f"\nErrors ({len(errors)}):", file=sys.stderr)
        for e in errors:
            print(f"  {e}", file=sys.stderr)
        sys.exit(1)

    print("\nDone!")


if __name__ == "__main__":
    main()
