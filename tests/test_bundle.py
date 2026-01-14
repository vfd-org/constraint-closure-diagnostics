"""
Tests for release bundle system.

Verifies:
1. Audit script detects missing items
2. Fig04 naming has no collisions between run and sweep
3. Release reports are written correctly
4. Bundle command creates correct folder structure
"""

import pytest
import tempfile
import json
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "tools"))


class TestAuditReleaseBundleDetection:
    """Tests that audit script detects missing items."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_audit_detects_missing_docs(self, temp_dir):
        """Audit should detect missing documentation files."""
        from audit_release_bundle import audit_release_bundle

        # Create minimal structure without docs
        runs_dir = temp_dir / "runs"
        runs_dir.mkdir()

        # Create a valid run
        run_dir = runs_dir / "test_run_abc123"
        run_dir.mkdir()
        (run_dir / "config.json").write_text('{"seed": 42}')
        (run_dir / "manifest.json").write_text('{"timestamp": "2024-01-01"}')
        (run_dir / "metrics.json").write_text('{}')
        figures_dir = run_dir / "figures"
        figures_dir.mkdir()
        (figures_dir / "fig01_residual_ladder.png").write_text("fake png")

        docs_dir = temp_dir / "docs"
        docs_dir.mkdir()
        # Don't create required docs

        result = audit_release_bundle(temp_dir)

        # Should fail because docs are missing
        assert result["status"] == "FAIL"
        assert len(result["docs"]["missing"]) > 0

    def test_audit_detects_missing_run_files(self, temp_dir):
        """Audit should detect missing run files."""
        from audit_release_bundle import audit_release_bundle

        # Create docs
        docs_dir = temp_dir / "docs"
        docs_dir.mkdir()
        for doc in ["METHODOLOGY_AND_JOURNEY.md", "STATISTICAL_SIGNIFICANCE.md", "SAFETY_BOUNDARY.md"]:
            (docs_dir / doc).write_text("# Doc")

        # Create run without manifest
        runs_dir = temp_dir / "runs"
        runs_dir.mkdir()
        run_dir = runs_dir / "test_run_abc123"
        run_dir.mkdir()
        (run_dir / "config.json").write_text('{"seed": 42}')
        # Missing manifest.json and metrics.json

        result = audit_release_bundle(temp_dir)

        # Should have run issues
        assert result["status"] == "FAIL"
        assert len(result["runs"]) > 0
        run_result = result["runs"][0]
        assert len(run_result["missing"]) > 0

    def test_audit_passes_with_complete_bundle(self, temp_dir):
        """Audit should pass with complete bundle."""
        from audit_release_bundle import audit_release_bundle

        # Create all required docs
        docs_dir = temp_dir / "docs"
        docs_dir.mkdir()
        for doc in ["METHODOLOGY_AND_JOURNEY.md", "STATISTICAL_SIGNIFICANCE.md", "SAFETY_BOUNDARY.md"]:
            (docs_dir / doc).write_text("# Doc content")

        # Create complete run
        runs_dir = temp_dir / "runs"
        runs_dir.mkdir()
        run_dir = runs_dir / "test_run_abc123"
        run_dir.mkdir()
        (run_dir / "config.json").write_text('{"seed": 42, "vfd": {}}')
        (run_dir / "manifest.json").write_text('{"timestamp": "2024-01-01"}')
        (run_dir / "metrics.json").write_text('{"all_passed": true}')

        figures_dir = run_dir / "figures"
        figures_dir.mkdir()
        (figures_dir / "fig01_residual_ladder.png").write_text("fake")

        result = audit_release_bundle(temp_dir)

        assert result["status"] == "PASS"


class TestFig04NamingNoCollision:
    """Tests that fig04 naming doesn't collide between runs and sweeps."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_single_run_fig04_name(self, temp_dir):
        """Single run should use fig04_spectrum_histogram.png."""
        from vfd_dash.core.config import get_default_config
        from vfd_dash.runner import run_diagnostic, save_run_outputs
        from vfd_dash.constraints import ClosureLevel

        config = get_default_config()
        config.seed = 42
        config.vfd.cell_count = 8
        config.vfd.internal_dim = 12
        config.vfd.orbit_size = 12
        config.vfd.orbit_count = 1
        config.vfd.local_propagation_L = 1
        config.run_name = "test_fig04_naming"

        results = run_diagnostic(
            config=config,
            max_level=ClosureLevel.L1,  # Fast test
            generate_figures=True,
            compute_stability=False,
        )

        run_dir = save_run_outputs(config, results, str(temp_dir))

        # Check for the correct filename
        fig04_spectrum = run_dir / "figures" / "fig04_spectrum_histogram.png"
        fig04_wall = run_dir / "figures" / "fig04_positivity_wall.png"

        assert fig04_spectrum.exists(), "fig04_spectrum_histogram.png should exist"
        assert not fig04_wall.exists(), "fig04_positivity_wall.png should NOT exist (old name)"

    def test_sweep_fig04_name(self, temp_dir):
        """Sweep should use fig04_positivity_wall_grid.png."""
        from vfd_dash.core.config import get_default_config
        from vfd_dash.sweep import run_sweep, save_sweep_outputs
        from vfd_dash.constraints import ClosureLevel

        config = get_default_config()
        config.seed = 42
        config.vfd.internal_dim = 12
        config.vfd.orbit_size = 12
        config.vfd.orbit_count = 1

        results = run_sweep(
            config=config,
            param1_name="cell_count",
            param1_values=[8, 16],
            param2_name="propagation_range",
            param2_values=[1, 2],
            max_level=ClosureLevel.L1,  # Fast test
            compute_stability=False,
            output_dir=str(temp_dir),
        )

        sweep_dir = save_sweep_outputs(results, str(temp_dir))

        # Check for the correct filename
        fig04_grid = sweep_dir / "fig04_positivity_wall_grid.png"
        fig04_wall = sweep_dir / "fig04_positivity_wall.png"

        assert fig04_grid.exists(), "fig04_positivity_wall_grid.png should exist"
        assert not fig04_wall.exists(), "fig04_positivity_wall.png should NOT exist (collision-prone)"

    def test_no_collision_in_same_directory(self, temp_dir):
        """Run and sweep in same dir should not overwrite each other's fig04."""
        from vfd_dash.core.config import get_default_config
        from vfd_dash.runner import run_diagnostic, save_run_outputs
        from vfd_dash.sweep import run_sweep, save_sweep_outputs
        from vfd_dash.constraints import ClosureLevel

        config = get_default_config()
        config.seed = 42
        config.vfd.cell_count = 8
        config.vfd.internal_dim = 12
        config.vfd.orbit_size = 12
        config.vfd.orbit_count = 1
        config.vfd.local_propagation_L = 1
        config.run_name = "collision_test"

        # Create run
        results = run_diagnostic(
            config=config,
            max_level=ClosureLevel.L1,
            generate_figures=True,
            compute_stability=False,
        )
        run_dir = save_run_outputs(config, results, str(temp_dir))
        run_fig04 = run_dir / "figures" / "fig04_spectrum_histogram.png"
        assert run_fig04.exists()

        # Create sweep
        sweep_results = run_sweep(
            config=config,
            param1_name="cell_count",
            param1_values=[8, 16],
            param2_name="propagation_range",
            param2_values=[1, 2],
            max_level=ClosureLevel.L1,
            compute_stability=False,
            output_dir=str(temp_dir),
        )
        sweep_dir = save_sweep_outputs(sweep_results, str(temp_dir))
        sweep_fig04 = sweep_dir / "fig04_positivity_wall_grid.png"
        assert sweep_fig04.exists()

        # Both should still exist (no collision)
        assert run_fig04.exists(), "Run fig04 should still exist after sweep"
        assert sweep_fig04.exists(), "Sweep fig04 should exist"


class TestReleaseReportWritten:
    """Tests that release reports are written correctly."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_release_report_generation(self, temp_dir):
        """Test that release report is generated correctly."""
        from vfd_dash.reports.release_report import generate_release_report, write_release_report

        # Create a mock run directory
        run_dir = temp_dir / "test_run"
        run_dir.mkdir()

        config = {
            "seed": 42,
            "run_name": "test_report",
            "vfd": {
                "cell_count": 32,
                "internal_dim": 96,
                "orbit_size": 12,
                "orbit_count": 8,
                "local_propagation_L": 3,
            },
            "bridge": {
                "bridge_mode": "OFF"
            }
        }

        manifest = {
            "timestamp": "2024-01-01T12:00:00",
            "closure_results": {
                "max_level_passed": "L4",
                "all_passed": True,
                "residuals_per_level": {
                    "L0": {"satisfied": True, "total_residual": 1e-10, "family_residuals": {}},
                    "L1": {"satisfied": True, "total_residual": 1e-10, "family_residuals": {}},
                    "L2": {"satisfied": True, "total_residual": 1e-10, "family_residuals": {}},
                    "L3": {"satisfied": True, "total_residual": 1e-10, "family_residuals": {}},
                    "L4": {"satisfied": True, "total_residual": 1e-10, "family_residuals": {}},
                }
            }
        }

        # Write files
        (run_dir / "config.json").write_text(json.dumps(config))
        (run_dir / "manifest.json").write_text(json.dumps(manifest))

        # Generate report
        report_path = write_release_report(run_dir)

        assert report_path.exists()
        assert report_path.name == "RELEASE_REPORT.md"

        content = report_path.read_text()
        assert "# Release Report" in content
        assert "Important Notice" in content
        assert "Configuration" in content
        assert "Closure Ladder Results" in content
        assert "Reproducibility" in content

    def test_release_report_contains_safety_disclaimer(self, temp_dir):
        """Test that release report contains safety disclaimer."""
        from vfd_dash.reports.release_report import generate_release_report

        run_dir = temp_dir / "test_run"
        run_dir.mkdir()

        config = {
            "seed": 42,
            "vfd": {"cell_count": 8, "internal_dim": 12, "orbit_size": 12, "orbit_count": 1, "local_propagation_L": 1},
            "bridge": {"bridge_mode": "OFF"}
        }
        manifest = {"timestamp": "2024-01-01", "closure_results": {}}

        report = generate_release_report(
            run_dir=run_dir,
            config=config,
            manifest=manifest,
        )

        # Check for safety disclaimer elements (accounting for possible newlines)
        assert "diagnostic and visualization framework" in report
        assert "does not" in report.lower()
        # The disclaimer says it "does not claim, demonstrate, or provide a proof of RH"
        # Check for key parts
        assert "proof of RH" in report or "proof of rh" in report.lower()

    def test_sweep_report_generation(self, temp_dir):
        """Test that sweep report is generated correctly."""
        from vfd_dash.reports.sweep_report import generate_sweep_report, write_sweep_report

        sweep_dir = temp_dir / "test_sweep"
        sweep_dir.mkdir()

        results = {
            "timestamp": "2024-01-01T12:00:00",
            "param1_name": "cell_count",
            "param2_name": "propagation_range",
            "param1_values": [8, 16, 32],
            "param2_values": [1, 2, 3],
            "internal_dim": 96,
            "max_level": "L4",
            "seed": 42,
            "grids": {
                "max_level_passed": [["L4", "L4", "L4"], ["L4", "L4", "L4"], ["L4", "L4", "L4"]],
                "min_eigenvalue": [[0.1, 0.2, 0.3], [0.15, 0.25, 0.35], [0.2, 0.3, 0.4]],
            }
        }

        (sweep_dir / "sweep_results.json").write_text(json.dumps(results))

        report_path = write_sweep_report(sweep_dir)

        assert report_path.exists()
        assert report_path.name == "SWEEP_REPORT.md"

        content = report_path.read_text()
        assert "# Sweep Report" in content
        assert "Sweep Configuration" in content
        assert "cell_count" in content
        assert "propagation_range" in content


class TestBundleCommandStructure:
    """Tests that bundle command creates correct folder structure."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def _find_run_dirs(self, release_dir: Path) -> list:
        """Find run directories (those containing config.json)."""
        run_dirs = []
        for item in release_dir.iterdir():
            if item.is_dir() and (item / "config.json").exists():
                run_dirs.append(item)
        return run_dirs

    def _find_sweep_dirs(self, release_dir: Path) -> list:
        """Find sweep directories (those starting with sweep_)."""
        return [d for d in release_dir.iterdir() if d.is_dir() and d.name.startswith("sweep_")]

    def test_bundle_creates_release_directory(self, temp_dir):
        """Bundle command should create timestamped release directory."""
        from vfd_dash.cli import cmd_bundle
        from argparse import Namespace

        args = Namespace(
            outdir=str(temp_dir),
            seed=42,
            cell_count=8,
            internal_dim=12,
            propagation_range=1,
            skip_sweep=True,  # Fast test
            skip_bridge=True,  # Fast test
            perf=False,
        )

        result = cmd_bundle(args)
        assert result == 0

        # Check that release directory was created
        release_dirs = list(temp_dir.glob("release_*"))
        assert len(release_dirs) == 1

        release_dir = release_dirs[0]
        assert release_dir.name.startswith("release_")

    def test_bundle_creates_bundle_report(self, temp_dir):
        """Bundle command should create BUNDLE_REPORT.md."""
        from vfd_dash.cli import cmd_bundle
        from argparse import Namespace

        args = Namespace(
            outdir=str(temp_dir),
            seed=42,
            cell_count=8,
            internal_dim=12,
            propagation_range=1,
            skip_sweep=True,
            skip_bridge=True,
            perf=False,
        )

        cmd_bundle(args)

        release_dirs = list(temp_dir.glob("release_*"))
        release_dir = release_dirs[0]

        bundle_report = release_dir / "BUNDLE_REPORT.md"
        assert bundle_report.exists()

        content = bundle_report.read_text()
        assert "# Release Bundle Report" in content
        assert "Important Notice" in content

    def test_bundle_creates_run_with_release_report(self, temp_dir):
        """Bundle command should create run with RELEASE_REPORT.md."""
        from vfd_dash.cli import cmd_bundle
        from argparse import Namespace

        args = Namespace(
            outdir=str(temp_dir),
            seed=42,
            cell_count=8,
            internal_dim=12,
            propagation_range=1,
            skip_sweep=True,
            skip_bridge=True,
            perf=False,
        )

        cmd_bundle(args)

        release_dirs = list(temp_dir.glob("release_*"))
        release_dir = release_dirs[0]

        # Find run directories (they have config.json)
        run_dirs = self._find_run_dirs(release_dir)
        assert len(run_dirs) >= 1, f"Expected at least 1 run, found: {[d.name for d in release_dir.iterdir()]}"

        # Check that at least one run has a release report
        has_report = any((run_dir / "RELEASE_REPORT.md").exists() for run_dir in run_dirs)
        assert has_report, "At least one run should have RELEASE_REPORT.md"

    def test_bundle_with_sweep_creates_sweep_report(self, temp_dir):
        """Bundle with sweep should create SWEEP_REPORT.md."""
        from vfd_dash.cli import cmd_bundle
        from argparse import Namespace

        args = Namespace(
            outdir=str(temp_dir),
            seed=42,
            cell_count=8,
            internal_dim=12,
            propagation_range=1,
            skip_sweep=False,  # Include sweep
            skip_bridge=True,
            perf=False,
        )

        cmd_bundle(args)

        release_dirs = list(temp_dir.glob("release_*"))
        release_dir = release_dirs[0]

        # Find sweep directory
        sweep_dirs = self._find_sweep_dirs(release_dir)
        assert len(sweep_dirs) == 1, f"Expected 1 sweep, found: {[d.name for d in sweep_dirs]}"

        sweep_dir = sweep_dirs[0]
        sweep_report = sweep_dir / "SWEEP_REPORT.md"
        assert sweep_report.exists()

    def test_bundle_structure_complete(self, temp_dir):
        """Full bundle should have complete structure."""
        from vfd_dash.cli import cmd_bundle
        from argparse import Namespace

        args = Namespace(
            outdir=str(temp_dir),
            seed=42,
            cell_count=8,
            internal_dim=12,
            propagation_range=1,
            skip_sweep=False,
            skip_bridge=False,
            perf=False,
        )

        cmd_bundle(args)

        release_dirs = list(temp_dir.glob("release_*"))
        release_dir = release_dirs[0]

        # Check bundle report
        assert (release_dir / "BUNDLE_REPORT.md").exists()

        # Check for at least 2 runs (standard and bridge)
        run_dirs = self._find_run_dirs(release_dir)
        assert len(run_dirs) >= 2, f"Expected at least 2 runs, found {len(run_dirs)}"

        # Each run should have essential files
        for run_dir in run_dirs:
            assert (run_dir / "config.json").exists(), f"Missing config.json in {run_dir.name}"
            assert (run_dir / "manifest.json").exists(), f"Missing manifest.json in {run_dir.name}"
            assert (run_dir / "RELEASE_REPORT.md").exists(), f"Missing RELEASE_REPORT.md in {run_dir.name}"

        # Check for sweep
        sweep_dirs = self._find_sweep_dirs(release_dir)
        assert len(sweep_dirs) == 1, f"Expected 1 sweep, found {len(sweep_dirs)}"
        assert (sweep_dirs[0] / "SWEEP_REPORT.md").exists()
