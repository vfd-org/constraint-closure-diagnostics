"""
Tests for VFD report generation.

Verifies:
1. Report generation produces expected files
2. Plots generate without exceptions
3. REPORT.md references existing figure files
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import os


class TestPlotGeneration:
    """Tests that individual plots generate without errors."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_plot_cell_spectrum(self, temp_dir):
        """Test cell spectrum plot generation."""
        from vfd_dash.reports.plots import plot_cell_spectrum

        output_path = temp_dir / "test_cell_spectrum"
        data = plot_cell_spectrum(
            cell_count=8,
            propagation_range=1,
            output_path=output_path,
            format='png',
            dpi=100
        )

        # Check file was created
        assert (temp_dir / "test_cell_spectrum.png").exists()

        # Check data returned
        assert 'eigenvalues' in data
        assert 'theta_j' in data
        assert len(data['eigenvalues']) == 8

    def test_plot_spectral_density(self, temp_dir):
        """Test spectral density plot generation."""
        from vfd_dash.reports.plots import plot_spectral_density

        output_path = temp_dir / "test_density"
        data = plot_spectral_density(
            cell_count=8,
            internal_dim=24,
            propagation_range=1,
            output_path=output_path,
            format='png',
            dpi=100
        )

        assert (temp_dir / "test_density.png").exists()
        assert 'ba_eigenvalues' in data
        assert len(data['ba_eigenvalues']) == 8 * 24

    def test_plot_bridge_metrics(self, temp_dir):
        """Test bridge metrics plot generation."""
        from vfd_dash.reports.plots import plot_bridge_metrics

        output_path = temp_dir / "test_metrics"
        ba_metrics = {'rmse': [0.1, 0.08, 0.06, 0.05]}
        bn_metrics = {'rmse': [0.2, 0.18, 0.15, 0.12]}

        data = plot_bridge_metrics(
            cell_counts=[8, 16, 32, 64],
            ba_metrics=ba_metrics,
            bn_metrics=bn_metrics,
            output_path=output_path,
            format='png',
            dpi=100
        )

        assert (temp_dir / "test_metrics.png").exists()

    def test_plot_convergence(self, temp_dir):
        """Test convergence plot generation."""
        from vfd_dash.reports.plots import plot_convergence

        output_path = temp_dir / "test_convergence"
        data = plot_convergence(
            cell_counts=[8, 16, 32, 64],
            metric_values=[0.1, 0.08, 0.06, 0.055],
            output_path=output_path,
            format='png',
            dpi=100
        )

        assert (temp_dir / "test_convergence.png").exists()

    def test_plot_torsion_fingerprint_single(self, temp_dir):
        """Test torsion fingerprint plot (single figure)."""
        from vfd_dash.reports.plots import plot_torsion_fingerprint

        output_path = temp_dir / "test_torsion"
        data = plot_torsion_fingerprint(
            cell_count=8,
            propagation_range=1,
            output_path=output_path,
            format='png',
            dpi=100,
            single_figure=True
        )

        assert (temp_dir / "test_torsion.png").exists()
        assert 'fingerprint' in data
        assert data['fingerprint'].shape[0] == 12  # 12 torsion sectors

    def test_plot_torsion_fingerprint_multiple(self, temp_dir):
        """Test torsion fingerprint plot (multiple figures)."""
        from vfd_dash.reports.plots import plot_torsion_fingerprint

        output_path = temp_dir / "test_torsion_multi"
        data = plot_torsion_fingerprint(
            cell_count=8,
            propagation_range=1,
            output_path=output_path,
            format='png',
            dpi=100,
            single_figure=False
        )

        # Should create 12 separate files
        for q in range(12):
            assert (temp_dir / f"test_torsion_multi_q{q:02d}.png").exists()

    def test_plot_stability_certificate(self, temp_dir):
        """Test stability certificate plot generation."""
        from vfd_dash.reports.plots import plot_stability_certificate

        output_path = temp_dir / "test_stability"
        data = plot_stability_certificate(
            cell_count=8,
            internal_dim=24,  # Small for fast test
            propagation_range=1,
            output_path=output_path,
            n_probes=50,  # Small for fast test
            seed=42,
            format='png',
            dpi=100
        )

        assert (temp_dir / "test_stability.png").exists()
        assert 'q_values' in data
        assert 'min_q' in data
        assert data['all_nonnegative']  # Kernel is nonnegative

    def test_create_hero_composite(self, temp_dir):
        """Test hero composite image generation."""
        from vfd_dash.reports.plots import create_hero_composite

        output_path = temp_dir / "test_hero"

        ba_metrics = {'rmse': [0.1, 0.08, 0.06, 0.05]}
        bn_metrics = {'rmse': [0.2, 0.18, 0.15, 0.12]}
        fingerprint = np.random.rand(12, 50)
        q_values = np.random.exponential(2.0, 100) + 0.1

        data = create_hero_composite(
            cell_counts=[8, 16, 32, 64],
            ba_metrics=ba_metrics,
            bn_metrics=bn_metrics,
            torsion_fingerprint_data=fingerprint,
            stability_q_values=q_values,
            output_path=output_path,
            format='png',
            dpi=100
        )

        assert (temp_dir / "test_hero.png").exists()


class TestReportGenerator:
    """Tests for the full report generator."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_generator_produces_expected_files(self, temp_dir):
        """Smoke test: report generation produces expected files."""
        from vfd_dash.reports.generator import ReportGenerator, ReportConfig

        config = ReportConfig(
            output_dir=temp_dir,
            cell_counts=[8, 16],  # Small for fast test
            propagation_range=1,
            internal_dim=24,  # Small for fast test
            bridge_mode="COMPARE",
            seed=42,
            format="png",
            n_probes=50,
            torsion_plots=True,
        )

        generator = ReportGenerator(config)
        results = generator.generate_all()

        # Check output structure
        assert (temp_dir / "figures").exists()
        assert (temp_dir / "data").exists()
        assert (temp_dir / "REPORT.md").exists()

        # Check some expected files
        assert any("cell_spectrum" in f for f in results['files'])
        assert any("hero_proof" in f for f in results['files'])
        assert "REPORT.md" in results['files']

    def test_generator_no_exceptions(self, temp_dir):
        """Test that generator doesn't raise exceptions."""
        from vfd_dash.reports.generator import ReportGenerator, ReportConfig

        config = ReportConfig(
            output_dir=temp_dir,
            cell_counts=[8],
            propagation_range=1,
            internal_dim=24,
            bridge_mode="BA",  # Simpler mode
            seed=0,
            format="png",
            n_probes=20,
            torsion_plots=False,  # Skip for faster test
        )

        generator = ReportGenerator(config)

        # Should not raise
        results = generator.generate_all()
        assert results['report_generated']


class TestReportMarkdown:
    """Tests for markdown report generation."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_report_references_existing_figures(self, temp_dir):
        """Test that REPORT.md only references existing files."""
        from vfd_dash.reports.generator import ReportGenerator, ReportConfig

        config = ReportConfig(
            output_dir=temp_dir,
            cell_counts=[8, 16],
            propagation_range=1,
            internal_dim=24,
            seed=0,
            format="png",
            n_probes=30,
            torsion_plots=True,
        )

        generator = ReportGenerator(config)
        generator.generate_all()

        # Read the report
        report_path = temp_dir / "REPORT.md"
        report_content = report_path.read_text()

        # Find all image references
        import re
        image_refs = re.findall(r'!\[.*?\]\((.*?)\)', report_content)

        # Check each referenced image exists
        for ref in image_refs:
            ref_path = temp_dir / ref
            assert ref_path.exists(), f"Referenced image does not exist: {ref}"

    def test_report_contains_key_sections(self, temp_dir):
        """Test that report contains expected sections."""
        from vfd_dash.reports.generator import ReportGenerator, ReportConfig

        config = ReportConfig(
            output_dir=temp_dir,
            cell_counts=[8],
            propagation_range=1,
            internal_dim=24,
            seed=0,
            format="png",
            n_probes=20,
        )

        generator = ReportGenerator(config)
        generator.generate_all()

        report_content = (temp_dir / "REPORT.md").read_text()

        # Check for expected sections
        assert "# VFD Internal Spectral Proof Artifacts" in report_content
        assert "## Introduction" in report_content
        assert "## Parameters" in report_content
        assert "## Plots" in report_content
        assert "## Reproducibility" in report_content

    def test_report_contains_reproducibility_command(self, temp_dir):
        """Test that report includes reproducibility command."""
        from vfd_dash.reports.generator import ReportGenerator, ReportConfig

        config = ReportConfig(
            output_dir=temp_dir,
            cell_counts=[8, 16],
            propagation_range=1,
            seed=42,
            format="png",
        )

        generator = ReportGenerator(config)
        generator.generate_all()

        report_content = (temp_dir / "REPORT.md").read_text()

        # Check reproducibility section
        assert "python scripts/generate_report.py" in report_content
        assert "--seed 42" in report_content


class TestDataExport:
    """Tests for data export functionality."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_npz_files_created(self, temp_dir):
        """Test that NPZ data files are created."""
        from vfd_dash.reports.generator import ReportGenerator, ReportConfig

        config = ReportConfig(
            output_dir=temp_dir,
            cell_counts=[8],
            propagation_range=1,
            internal_dim=24,
            seed=0,
            n_probes=20,
        )

        generator = ReportGenerator(config)
        generator.generate_all()

        # Check data directory has NPZ files
        data_dir = temp_dir / "data"
        npz_files = list(data_dir.glob("*.npz"))
        assert len(npz_files) > 0

    def test_npz_files_loadable(self, temp_dir):
        """Test that NPZ files can be loaded."""
        from vfd_dash.reports.generator import ReportGenerator, ReportConfig

        config = ReportConfig(
            output_dir=temp_dir,
            cell_counts=[8],
            propagation_range=1,
            internal_dim=24,
            seed=0,
            n_probes=20,
        )

        generator = ReportGenerator(config)
        generator.generate_all()

        # Try loading each NPZ file
        data_dir = temp_dir / "data"
        for npz_file in data_dir.glob("*.npz"):
            data = np.load(npz_file)
            assert len(data.files) > 0  # Has at least one array

    def test_metrics_json_created(self, temp_dir):
        """Test that metrics JSON is created and valid."""
        from vfd_dash.reports.generator import ReportGenerator, ReportConfig
        import json

        config = ReportConfig(
            output_dir=temp_dir,
            cell_counts=[8],
            propagation_range=1,
            internal_dim=24,
            seed=0,
            n_probes=20,
        )

        generator = ReportGenerator(config)
        generator.generate_all()

        # Check metrics.json
        metrics_path = temp_dir / "data" / "metrics.json"
        assert metrics_path.exists()

        # Should be valid JSON
        with open(metrics_path) as f:
            metrics = json.load(f)

        assert isinstance(metrics, dict)
