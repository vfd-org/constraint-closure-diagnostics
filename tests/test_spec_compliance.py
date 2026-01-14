"""
Spec Compliance Tests for RH Constraint-Diagnostic Demo.

These tests verify that the implementation follows the specification.
"""

import pytest
import numpy as np
from pathlib import Path


class TestClosureLevels:
    """Test closure level implementation."""

    def test_closure_levels_exist(self):
        """Verify all closure levels L0-L4 are defined."""
        from vfd_dash.constraints import ClosureLevel

        levels = [ClosureLevel.L0, ClosureLevel.L1, ClosureLevel.L2,
                  ClosureLevel.L3, ClosureLevel.L4]

        assert len(levels) == 5
        assert [l.value for l in levels] == [0, 1, 2, 3, 4]

    def test_level_parsing(self):
        """Test level string parsing."""
        from vfd_dash.constraints import ClosureLevel

        assert ClosureLevel.from_string("L0") == ClosureLevel.L0
        assert ClosureLevel.from_string("L4") == ClosureLevel.L4
        assert ClosureLevel.from_string("l2") == ClosureLevel.L2

    def test_level_range_parsing(self):
        """Test parsing level ranges."""
        from vfd_dash.constraints import ClosureLevel

        levels = ClosureLevel.parse_range("L0..L2")
        assert levels == [ClosureLevel.L0, ClosureLevel.L1, ClosureLevel.L2]


class TestConstraintFamilies:
    """Test constraint family implementation."""

    def test_all_families_defined(self):
        """Verify all constraint families are defined."""
        from vfd_dash.constraints.families import (
            ExplicitFormulaFamily,
            SymmetryFamily,
            PositivityFamily,
            TraceMomentFamily,
        )

        families = [
            ExplicitFormulaFamily(),
            SymmetryFamily(),
            PositivityFamily(),
            TraceMomentFamily(),
        ]

        assert len(families) == 4

    def test_family_evaluation(self):
        """Test that families return proper residual dictionaries."""
        from vfd_dash.constraints.families import ExplicitFormulaFamily
        from vfd_dash.state import DiagnosticState, build_state
        from vfd_dash.core.config import get_default_config

        # Use minimal config for fast test
        config = get_default_config()
        config.vfd.cell_count = 8
        config.vfd.internal_dim = 24
        config.vfd.orbit_count = 6
        config.vfd.orbit_size = 4

        np.random.seed(42)
        state = build_state(
            config=config,
            compute_spectrum=False,
            compute_stability=False,
            compute_primes=False,
            compute_projection=False,
            seed=42,
        )

        family = ExplicitFormulaFamily()
        residuals = family.evaluate(state)

        assert isinstance(residuals, dict)
        assert "torsion_order" in residuals
        assert "weyl_relation" in residuals
        assert all(isinstance(v, (int, float)) for v in residuals.values())

    def test_level_family_mapping(self):
        """Test that levels map to correct families."""
        from vfd_dash.constraints import ClosureLevel
        from vfd_dash.constraints.families import get_families_for_level

        # L0 has no families (baseline structural validity)
        l0_families = get_families_for_level(ClosureLevel.L0)
        assert len(l0_families) == 0

        # L1 should have EF
        l1_families = get_families_for_level(ClosureLevel.L1)
        assert any(f.__class__.__name__ == "ExplicitFormulaFamily" for f in l1_families)

        # L4 should have all 4 families
        l4_families = get_families_for_level(ClosureLevel.L4)
        assert len(l4_families) == 4


class TestLadderGating:
    """Test closure ladder gating behavior."""

    def test_ladder_basic_run(self):
        """Test basic ladder execution."""
        from vfd_dash.constraints import ClosureLadder, ClosureLevel, LadderResult
        from vfd_dash.state import build_state
        from vfd_dash.core.config import get_default_config

        config = get_default_config()
        config.vfd.cell_count = 8
        config.vfd.internal_dim = 24
        config.vfd.orbit_count = 6
        config.vfd.orbit_size = 4

        np.random.seed(42)
        state = build_state(
            config=config,
            compute_spectrum=True,
            compute_stability=False,
            compute_primes=False,
            compute_projection=False,
            seed=42,
        )

        ladder = ClosureLadder(tolerance=1e-8)
        result = ladder.run(state, max_level=ClosureLevel.L2, gate=True)

        assert isinstance(result, LadderResult)
        assert result.max_level_checked is not None
        assert hasattr(result, "all_passed")

    def test_ladder_result_serialization(self):
        """Test that ladder results can be serialized."""
        from vfd_dash.constraints import LadderResult, LevelResult, ClosureLevel

        level_result = LevelResult(
            level=ClosureLevel.L0,
            satisfied=True,
            residuals={"test": 1e-10},
            constraints_checked=["test"],
            gating_passed=True,
            total_residual=1e-10,
            family_residuals={"EF": 1e-10},
        )

        ladder_result = LadderResult(
            max_level_checked=ClosureLevel.L0,
            max_level_passed=ClosureLevel.L0,
            level_results={ClosureLevel.L0: level_result},
            gating_stop_reason=None,
        )

        result_dict = ladder_result.to_dict()

        assert "all_passed" in result_dict
        assert "max_level_checked" in result_dict
        assert "residuals_per_level" in result_dict


class TestManifestSchema:
    """Test manifest schema compliance."""

    def test_manifest_has_tolerances(self):
        """Test that manifest includes tolerances."""
        from vfd_dash.runner import save_run_outputs, run_diagnostic
        from vfd_dash.core.config import get_default_config
        from vfd_dash.constraints import ClosureLevel
        import tempfile
        import json

        config = get_default_config()
        config.vfd.cell_count = 8
        config.vfd.internal_dim = 24
        config.vfd.orbit_count = 6
        config.vfd.orbit_size = 4
        config.run_name = "test_manifest"
        config.bridge.bridge_mode = "OFF"  # Skip projection for speed

        np.random.seed(42)
        results = run_diagnostic(
            config=config,
            max_level=ClosureLevel.L0,
            generate_figures=False,
            seed=42,
            compute_stability=False,  # Skip for small test config
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = save_run_outputs(config, results, tmpdir)
            manifest_file = run_dir / "manifest.json"

            assert manifest_file.exists()

            with open(manifest_file) as f:
                manifest = json.load(f)

            assert "tolerances" in manifest
            assert "closure_ladder" in manifest["tolerances"]

    def test_manifest_has_closure_results(self):
        """Test that manifest includes closure results."""
        from vfd_dash.runner import save_run_outputs, run_diagnostic
        from vfd_dash.core.config import get_default_config
        from vfd_dash.constraints import ClosureLevel
        import tempfile
        import json

        config = get_default_config()
        config.vfd.cell_count = 8
        config.vfd.internal_dim = 24
        config.vfd.orbit_count = 6
        config.vfd.orbit_size = 4
        config.run_name = "test_closure_results"
        config.bridge.bridge_mode = "OFF"  # Skip projection for speed

        np.random.seed(42)
        results = run_diagnostic(
            config=config,
            max_level=ClosureLevel.L0,
            generate_figures=False,
            seed=42,
            compute_stability=False,  # Skip for small test config
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = save_run_outputs(config, results, tmpdir)
            manifest_file = run_dir / "manifest.json"

            with open(manifest_file) as f:
                manifest = json.load(f)

            assert "closure_results" in manifest
            assert "max_level_checked" in manifest["closure_results"]
            assert "all_passed" in manifest["closure_results"]


class TestFigureGeneration:
    """Test figure generation."""

    def test_residual_ladder_figure(self):
        """Test residual ladder figure generation."""
        from vfd_dash.figures import generate_residual_ladder
        from vfd_dash.constraints import LadderResult, LevelResult, ClosureLevel

        # Create mock ladder result with correct signature
        level_result = LevelResult(
            level=ClosureLevel.L0,
            satisfied=True,
            residuals={"test": 1e-10},
            constraints_checked=["test"],
            gating_passed=True,
            total_residual=1e-10,
            family_residuals={"EF": 1e-10},
        )

        ladder_result = LadderResult(
            max_level_checked=ClosureLevel.L0,
            max_level_passed=ClosureLevel.L0,
            level_results={ClosureLevel.L0: level_result},
            gating_stop_reason=None,
        )

        png_bytes = generate_residual_ladder(ladder_result)

        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        # Check PNG header
        assert png_bytes[:8] == b'\x89PNG\r\n\x1a\n'

    def test_constraint_waterfall_figure(self):
        """Test constraint waterfall figure generation."""
        from vfd_dash.figures import generate_constraint_waterfall
        from vfd_dash.constraints import LadderResult, LevelResult, ClosureLevel

        level_result = LevelResult(
            level=ClosureLevel.L0,
            satisfied=True,
            residuals={"test": 1e-10},
            constraints_checked=["test"],
            gating_passed=True,
            total_residual=1e-10,
            family_residuals={"EF": 1e-10, "Symmetry": 1e-11},
        )

        ladder_result = LadderResult(
            max_level_checked=ClosureLevel.L0,
            max_level_passed=ClosureLevel.L0,
            level_results={ClosureLevel.L0: level_result},
            gating_stop_reason=None,
        )

        png_bytes = generate_constraint_waterfall(ladder_result)

        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        assert png_bytes[:8] == b'\x89PNG\r\n\x1a\n'

    def test_phase_map_figure(self):
        """Test phase map figure generation."""
        from vfd_dash.figures import generate_phase_map

        sweep_results = {
            "param1_name": "cell_count",
            "param1_values": [4, 8],
            "param2_name": "propagation_range",
            "param2_values": [1, 2],
            "results_grid": {
                "4_1": {"all_passed": True, "max_level_passed": 4, "total_residual": 1e-10},
                "4_2": {"all_passed": True, "max_level_passed": 4, "total_residual": 1e-10},
                "8_1": {"all_passed": False, "max_level_passed": 2, "total_residual": 0.1},
                "8_2": {"all_passed": True, "max_level_passed": 4, "total_residual": 1e-10},
            },
        }

        png_bytes = generate_phase_map(sweep_results)

        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0
        assert png_bytes[:8] == b'\x89PNG\r\n\x1a\n'


class TestStateObject:
    """Test diagnostic state object."""

    def test_state_creation(self):
        """Test state object creation."""
        from vfd_dash.state import DiagnosticState, build_state
        from vfd_dash.core.config import get_default_config

        config = get_default_config()
        config.vfd.cell_count = 8
        config.vfd.internal_dim = 24
        config.vfd.orbit_count = 6
        config.vfd.orbit_size = 4

        np.random.seed(42)
        state = build_state(
            config=config,
            compute_spectrum=True,
            compute_stability=False,
            compute_primes=False,
            compute_projection=False,
            seed=42,
        )

        assert isinstance(state, DiagnosticState)
        assert state.config is not None
        assert state.space is not None
        assert state.T is not None
        assert state.S is not None
        assert state.kernel is not None

    def test_state_to_dict(self):
        """Test state serialization."""
        from vfd_dash.state import build_state, state_to_results_dict
        from vfd_dash.core.config import get_default_config

        config = get_default_config()
        config.vfd.cell_count = 8
        config.vfd.internal_dim = 24
        config.vfd.orbit_count = 6
        config.vfd.orbit_size = 4

        np.random.seed(42)
        state = build_state(
            config=config,
            compute_spectrum=False,
            compute_stability=False,
            compute_primes=False,
            compute_projection=False,
            seed=42,
        )

        results = state_to_results_dict(state)

        assert isinstance(results, dict)
        assert "config" in results or "vfd_config" in results or "seed" in results


class TestCLI:
    """Test CLI components."""

    def test_disclaimer_exists(self):
        """Test that safety disclaimer is defined."""
        from vfd_dash.cli import DISCLAIMER

        assert isinstance(DISCLAIMER, str)
        assert "NOT" in DISCLAIMER
        assert "proof" in DISCLAIMER.lower() or "prove" in DISCLAIMER.lower()

    def test_argparse_setup(self):
        """Test that CLI argument parser is properly configured."""
        from vfd_dash.cli import main
        import sys

        # Test with --help (should not raise)
        old_argv = sys.argv
        try:
            sys.argv = ["rhdiag", "--help"]
            with pytest.raises(SystemExit) as exc_info:
                main()
            assert exc_info.value.code == 0
        except SystemExit:
            pass
        finally:
            sys.argv = old_argv


class TestSweep:
    """Test parameter sweep infrastructure."""

    def test_sweep_module_exists(self):
        """Test that sweep module can be imported."""
        from vfd_dash.sweep import run_sweep, save_sweep_outputs

        assert callable(run_sweep)
        assert callable(save_sweep_outputs)


class TestIntegration:
    """Integration tests for full pipeline."""

    @pytest.mark.slow
    def test_full_diagnostic_run(self):
        """Test complete diagnostic run."""
        from vfd_dash.runner import run_diagnostic
        from vfd_dash.core.config import get_default_config
        from vfd_dash.constraints import ClosureLevel

        config = get_default_config()
        config.vfd.cell_count = 8
        config.vfd.internal_dim = 24
        config.vfd.orbit_count = 6
        config.vfd.orbit_size = 4
        config.run_name = "integration_test"
        config.bridge.bridge_mode = "OFF"  # Skip projection for speed

        np.random.seed(42)
        results = run_diagnostic(
            config=config,
            max_level=ClosureLevel.L2,
            generate_figures=True,
            seed=42,
            compute_stability=False,  # Skip for small test config
        )

        assert "ladder_result" in results
        assert "all_passed" in results
        assert "_figures_data" in results
        assert len(results["_figures_data"]) > 0
