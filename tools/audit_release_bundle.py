#!/usr/bin/env python3
"""
Audit Release Bundle Tool.

Verifies that required documentation, run outputs, and sweep outputs
are present and properly named before release.

Usage:
    python tools/audit_release_bundle.py [--run-hash <hash>] [--sweep-dir <dir>]

Exit codes:
    0 = PASS (all required items present)
    1 = FAIL (missing items)
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# Required documentation files
REQUIRED_DOCS = [
    "SAFETY_BOUNDARY.md",
    "REPRODUCIBILITY_REPORT.md",
    "COMPUTATION_MAP.md",
    "PERFORMANCE_DIAGNOSTICS.md",
    "SHAREABLE_PROFILE.md",
    "PROOF_STATUS.md",
    "EVIDENCE_INVENTORY.md",
    "STATISTICAL_SIGNIFICANCE.md",
    "METHODOLOGY_AND_JOURNEY.md",
]

# Required files per single run
REQUIRED_RUN_FILES = [
    "config.json",
    "manifest.json",
    "metrics.json",
]

# Optional run files (checked but not required)
OPTIONAL_RUN_FILES = [
    "perf.json",
]

# Required figures for single run (non-bridge mode)
REQUIRED_RUN_FIGURES_BASE = [
    "fig01_residual_ladder.png",
    "fig03_constraint_waterfall.png",
    "fig05_collapse_geometry.png",
]

# Single-run specific figure (NOT sweep)
SINGLE_RUN_FIGURE = "fig04_spectrum_histogram.png"
SINGLE_RUN_FIGURE_ALT = "fig04_positivity_wall.png"  # Legacy name (pre-fix)

# Bridge mode additional figures
BRIDGE_FIGURES = [
    "fig06_zero_overlay.png",
    "fig07_falsification.png",
]

# Required sweep files
REQUIRED_SWEEP_FILES = [
    "sweep_results.json",
]

# Optional sweep files
OPTIONAL_SWEEP_FILES = [
    "sweep_results_partial.json",
    "summary.txt",
]

# Required sweep figures
REQUIRED_SWEEP_FIGURES = [
    "fig02_phase_map.png",
    "fig04_positivity_wall_grid.png",  # Target name (sweep-specific)
]

# Current sweep figure name (to be fixed)
SWEEP_FIGURE_CURRENT = "fig04_positivity_wall.png"


@dataclass
class AuditResult:
    """Result of an audit check."""
    category: str
    item: str
    status: str  # "PASS", "FAIL", "WARN", "SKIP"
    message: str = ""
    fix_command: str = ""


@dataclass
class AuditReport:
    """Complete audit report."""
    results: List[AuditResult] = field(default_factory=list)

    def add(self, result: AuditResult):
        self.results.append(result)

    @property
    def passed(self) -> bool:
        return all(r.status in ("PASS", "WARN", "SKIP") for r in self.results)

    @property
    def pass_count(self) -> int:
        return sum(1 for r in self.results if r.status == "PASS")

    @property
    def fail_count(self) -> int:
        return sum(1 for r in self.results if r.status == "FAIL")

    @property
    def warn_count(self) -> int:
        return sum(1 for r in self.results if r.status == "WARN")

    def print_report(self):
        """Print formatted audit report."""
        print("\n" + "=" * 70)
        print("RELEASE BUNDLE AUDIT REPORT")
        print("=" * 70)

        # Group by category
        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = []
            categories[r.category].append(r)

        for category, items in categories.items():
            print(f"\n--- {category} ---")
            for r in items:
                status_color = {
                    "PASS": "\033[92m",  # Green
                    "FAIL": "\033[91m",  # Red
                    "WARN": "\033[93m",  # Yellow
                    "SKIP": "\033[90m",  # Gray
                }
                reset = "\033[0m"
                color = status_color.get(r.status, "")
                print(f"  [{color}{r.status}{reset}] {r.item}")
                if r.message:
                    print(f"         {r.message}")
                if r.fix_command and r.status == "FAIL":
                    print(f"         Fix: {r.fix_command}")

        # Summary
        print("\n" + "=" * 70)
        total = len(self.results)
        print(f"SUMMARY: {self.pass_count}/{total} PASS, "
              f"{self.fail_count} FAIL, {self.warn_count} WARN")

        if self.passed:
            print("\033[92m[AUDIT PASSED]\033[0m")
        else:
            print("\033[91m[AUDIT FAILED]\033[0m")
            print("\nFailed items require attention before release.")
        print("=" * 70)


def find_repo_root() -> Path:
    """Find the repository root (contains src/, docs/, runs/)."""
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "src").exists() and (current / "docs").exists():
            return current
        current = current.parent
    # Fallback to current directory
    return Path.cwd()


def audit_docs(repo_root: Path, report: AuditReport):
    """Audit required documentation files."""
    docs_dir = repo_root / "docs"

    for doc in REQUIRED_DOCS:
        doc_path = docs_dir / doc
        if doc_path.exists():
            report.add(AuditResult(
                category="Documentation",
                item=doc,
                status="PASS",
            ))
        else:
            report.add(AuditResult(
                category="Documentation",
                item=doc,
                status="FAIL",
                message=f"File not found: {doc_path}",
                fix_command=f"Create docs/{doc} with required content",
            ))


def audit_run(run_dir: Path, report: AuditReport, is_bridge: bool = False):
    """Audit a single run directory."""
    if not run_dir.exists():
        report.add(AuditResult(
            category="Run Output",
            item=str(run_dir),
            status="FAIL",
            message="Run directory does not exist",
            fix_command="rhdiag run --seed 42",
        ))
        return

    # Check required files
    for file in REQUIRED_RUN_FILES:
        file_path = run_dir / file
        if file_path.exists():
            report.add(AuditResult(
                category="Run Output",
                item=file,
                status="PASS",
            ))
        else:
            report.add(AuditResult(
                category="Run Output",
                item=file,
                status="FAIL",
                message=f"Missing: {file_path}",
            ))

    # Check optional files
    for file in OPTIONAL_RUN_FILES:
        file_path = run_dir / file
        if file_path.exists():
            report.add(AuditResult(
                category="Run Output",
                item=f"{file} (optional)",
                status="PASS",
            ))
        else:
            report.add(AuditResult(
                category="Run Output",
                item=f"{file} (optional)",
                status="SKIP",
                message="Optional file not present",
            ))

    # Check figures
    figures_dir = run_dir / "figures"
    if not figures_dir.exists():
        report.add(AuditResult(
            category="Run Figures",
            item="figures/",
            status="FAIL",
            message="Figures directory missing",
        ))
        return

    # Base figures
    for fig in REQUIRED_RUN_FIGURES_BASE:
        fig_path = figures_dir / fig
        if fig_path.exists():
            report.add(AuditResult(
                category="Run Figures",
                item=fig,
                status="PASS",
            ))
        else:
            report.add(AuditResult(
                category="Run Figures",
                item=fig,
                status="FAIL",
                message=f"Missing: {fig_path}",
            ))

    # Single-run fig04 (check for naming collision issue)
    fig04_new = figures_dir / SINGLE_RUN_FIGURE
    fig04_old = figures_dir / SINGLE_RUN_FIGURE_ALT

    if fig04_new.exists():
        report.add(AuditResult(
            category="Run Figures",
            item=SINGLE_RUN_FIGURE,
            status="PASS",
        ))
    elif fig04_old.exists():
        report.add(AuditResult(
            category="Run Figures",
            item=SINGLE_RUN_FIGURE_ALT,
            status="WARN",
            message=f"Found {SINGLE_RUN_FIGURE_ALT}, should be {SINGLE_RUN_FIGURE}",
            fix_command="Update figure naming in plots.py",
        ))
    else:
        report.add(AuditResult(
            category="Run Figures",
            item="fig04_* (single run)",
            status="FAIL",
            message="No fig04 found for single run",
        ))

    # Bridge figures
    if is_bridge:
        for fig in BRIDGE_FIGURES:
            fig_path = figures_dir / fig
            if fig_path.exists():
                report.add(AuditResult(
                    category="Bridge Figures",
                    item=fig,
                    status="PASS",
                ))
            else:
                report.add(AuditResult(
                    category="Bridge Figures",
                    item=fig,
                    status="FAIL",
                    message=f"Missing bridge figure: {fig_path}",
                    fix_command="rhdiag run --bridge-mode BA",
                ))


def audit_sweep(sweep_dir: Path, report: AuditReport):
    """Audit a sweep directory."""
    if not sweep_dir.exists():
        report.add(AuditResult(
            category="Sweep Output",
            item=str(sweep_dir),
            status="FAIL",
            message="Sweep directory does not exist",
            fix_command="rhdiag sweep --values1 8,16,32 --values2 1,2,3",
        ))
        return

    # Check required files
    for file in REQUIRED_SWEEP_FILES:
        file_path = sweep_dir / file
        if file_path.exists():
            report.add(AuditResult(
                category="Sweep Output",
                item=file,
                status="PASS",
            ))
        else:
            report.add(AuditResult(
                category="Sweep Output",
                item=file,
                status="FAIL",
                message=f"Missing: {file_path}",
            ))

    # Check optional files
    for file in OPTIONAL_SWEEP_FILES:
        file_path = sweep_dir / file
        if file_path.exists():
            report.add(AuditResult(
                category="Sweep Output",
                item=f"{file} (optional)",
                status="PASS",
            ))
        else:
            report.add(AuditResult(
                category="Sweep Output",
                item=f"{file} (optional)",
                status="SKIP",
            ))

    # Check figures
    figures_dir = sweep_dir / "figures"
    if not figures_dir.exists():
        # Sweep figures might be in root
        figures_dir = sweep_dir

    # Phase map
    fig02 = figures_dir / "fig02_phase_map.png"
    if fig02.exists():
        report.add(AuditResult(
            category="Sweep Figures",
            item="fig02_phase_map.png",
            status="PASS",
        ))
    else:
        # Check root
        fig02_root = sweep_dir / "fig02_phase_map.png"
        if fig02_root.exists():
            report.add(AuditResult(
                category="Sweep Figures",
                item="fig02_phase_map.png",
                status="PASS",
            ))
        else:
            report.add(AuditResult(
                category="Sweep Figures",
                item="fig02_phase_map.png",
                status="FAIL",
                message="Phase map not found",
            ))

    # Positivity wall grid (check naming collision)
    fig04_new = figures_dir / "fig04_positivity_wall_grid.png"
    fig04_old = figures_dir / SWEEP_FIGURE_CURRENT
    fig04_root_new = sweep_dir / "fig04_positivity_wall_grid.png"
    fig04_root_old = sweep_dir / SWEEP_FIGURE_CURRENT

    if fig04_new.exists() or fig04_root_new.exists():
        report.add(AuditResult(
            category="Sweep Figures",
            item="fig04_positivity_wall_grid.png",
            status="PASS",
        ))
    elif fig04_old.exists() or fig04_root_old.exists():
        report.add(AuditResult(
            category="Sweep Figures",
            item=SWEEP_FIGURE_CURRENT,
            status="WARN",
            message=f"Found {SWEEP_FIGURE_CURRENT}, should be fig04_positivity_wall_grid.png",
            fix_command="Update figure naming in sweep.py",
        ))
    else:
        report.add(AuditResult(
            category="Sweep Figures",
            item="fig04_positivity_wall_grid.png",
            status="FAIL",
            message="Sweep positivity grid not found",
        ))


def check_fig04_collision(repo_root: Path, report: AuditReport):
    """Check for fig04 naming collision between single run and sweep."""
    # This is a structural check, not specific to a run

    # Check plots.py for single-run fig04 naming
    plots_file = repo_root / "src" / "vfd_dash" / "reports" / "plots.py"
    sweep_file = repo_root / "src" / "vfd_dash" / "sweep.py"

    collision_risk = False

    if plots_file.exists():
        content = plots_file.read_text()
        if "fig04_positivity_wall.png" in content:
            collision_risk = True
            report.add(AuditResult(
                category="Naming Collision",
                item="plots.py: fig04_positivity_wall.png",
                status="WARN",
                message="Single-run uses same fig04 name as sweep",
                fix_command="Rename to fig04_spectrum_histogram.png",
            ))

    if sweep_file.exists():
        content = sweep_file.read_text()
        if "fig04_positivity_wall.png" in content and collision_risk:
            report.add(AuditResult(
                category="Naming Collision",
                item="sweep.py: fig04_positivity_wall.png",
                status="WARN",
                message="Sweep uses same fig04 name as single-run",
                fix_command="Rename to fig04_positivity_wall_grid.png",
            ))
        elif "fig04_positivity_wall_grid.png" in content:
            report.add(AuditResult(
                category="Naming Collision",
                item="sweep.py: fig04 naming",
                status="PASS",
                message="Sweep uses distinct fig04 name",
            ))

    if not collision_risk:
        report.add(AuditResult(
            category="Naming Collision",
            item="fig04 naming",
            status="PASS" if not collision_risk else "WARN",
        ))


def audit_release_bundle(bundle_dir: Path) -> Dict:
    """
    Audit a release bundle directory programmatically.

    Args:
        bundle_dir: Path to the bundle root directory

    Returns:
        Dictionary with audit results:
        {
            "status": "PASS" or "FAIL",
            "docs": {"missing": [...], "present": [...]},
            "runs": [{"dir": ..., "missing": [...], "present": [...]}],
            "sweeps": [{"dir": ..., "missing": [...], "present": [...]}],
        }
    """
    result = {
        "status": "PASS",
        "docs": {"missing": [], "present": []},
        "runs": [],
        "sweeps": [],
    }

    # Check docs
    docs_dir = bundle_dir / "docs"
    required_docs = [
        "METHODOLOGY_AND_JOURNEY.md",
        "STATISTICAL_SIGNIFICANCE.md",
        "SAFETY_BOUNDARY.md",
    ]

    for doc in required_docs:
        doc_path = docs_dir / doc
        if doc_path.exists():
            result["docs"]["present"].append(doc)
        else:
            result["docs"]["missing"].append(doc)
            result["status"] = "FAIL"

    # Check runs
    runs_dir = bundle_dir / "runs"
    if runs_dir.exists():
        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue
            if run_dir.name.startswith("sweep_") or run_dir.name.startswith("release_"):
                continue

            run_result = {
                "dir": run_dir.name,
                "missing": [],
                "present": [],
            }

            for file in REQUIRED_RUN_FILES:
                if (run_dir / file).exists():
                    run_result["present"].append(file)
                else:
                    run_result["missing"].append(file)
                    result["status"] = "FAIL"

            result["runs"].append(run_result)

    # Check sweeps
    if runs_dir.exists():
        for sweep_dir in runs_dir.iterdir():
            if not sweep_dir.is_dir():
                continue
            if not sweep_dir.name.startswith("sweep_"):
                continue

            sweep_result = {
                "dir": sweep_dir.name,
                "missing": [],
                "present": [],
            }

            for file in REQUIRED_SWEEP_FILES:
                if (sweep_dir / file).exists():
                    sweep_result["present"].append(file)
                else:
                    sweep_result["missing"].append(file)
                    result["status"] = "FAIL"

            result["sweeps"].append(sweep_result)

    return result


def find_latest_run(runs_dir: Path) -> Optional[Path]:
    """Find the most recent run directory."""
    if not runs_dir.exists():
        return None

    runs = [d for d in runs_dir.iterdir()
            if d.is_dir() and not d.name.startswith("sweep_")
            and not d.name.startswith("release_")]

    if not runs:
        return None

    # Sort by modification time
    runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return runs[0]


def find_latest_sweep(runs_dir: Path) -> Optional[Path]:
    """Find the most recent sweep directory."""
    if not runs_dir.exists():
        return None

    sweeps = [d for d in runs_dir.iterdir()
              if d.is_dir() and d.name.startswith("sweep_")]

    if not sweeps:
        return None

    sweeps.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return sweeps[0]


def main():
    parser = argparse.ArgumentParser(
        description="Audit release bundle for completeness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-hash", type=str, help="Specific run hash to audit")
    parser.add_argument("--sweep-dir", type=str, help="Specific sweep directory to audit")
    parser.add_argument("--skip-docs", action="store_true", help="Skip documentation audit")
    parser.add_argument("--skip-run", action="store_true", help="Skip run output audit")
    parser.add_argument("--skip-sweep", action="store_true", help="Skip sweep output audit")

    args = parser.parse_args()

    repo_root = find_repo_root()
    runs_dir = repo_root / "runs"

    report = AuditReport()

    print(f"Repository root: {repo_root}")
    print(f"Runs directory: {runs_dir}")

    # Audit documentation
    if not args.skip_docs:
        audit_docs(repo_root, report)

    # Audit run output
    if not args.skip_run:
        if args.run_hash:
            run_dir = runs_dir / args.run_hash
        else:
            run_dir = find_latest_run(runs_dir)

        if run_dir:
            print(f"Auditing run: {run_dir.name}")

            # Check if bridge mode
            config_file = run_dir / "config.json"
            is_bridge = False
            if config_file.exists():
                try:
                    config = json.loads(config_file.read_text())
                    bridge_mode = config.get("bridge", {}).get("bridge_mode", "OFF")
                    is_bridge = bridge_mode not in ("OFF", None)
                except:
                    pass

            audit_run(run_dir, report, is_bridge=is_bridge)
        else:
            report.add(AuditResult(
                category="Run Output",
                item="Any run",
                status="FAIL",
                message="No run directories found",
                fix_command="rhdiag run --seed 42",
            ))

    # Audit sweep output
    if not args.skip_sweep:
        if args.sweep_dir:
            sweep_dir = runs_dir / args.sweep_dir
        else:
            sweep_dir = find_latest_sweep(runs_dir)

        if sweep_dir:
            print(f"Auditing sweep: {sweep_dir.name}")
            audit_sweep(sweep_dir, report)
        else:
            report.add(AuditResult(
                category="Sweep Output",
                item="Any sweep",
                status="WARN",
                message="No sweep directories found",
                fix_command="rhdiag sweep --values1 8,16,32 --values2 1,2,3",
            ))

    # Check naming collision
    check_fig04_collision(repo_root, report)

    # Print report
    report.print_report()

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
