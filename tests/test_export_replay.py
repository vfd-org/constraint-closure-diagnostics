"""
Test export bundle and replay functionality.
"""

import pytest
import tempfile
import json
from pathlib import Path
import pandas as pd
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vfd_dash.core.config import RunConfig, get_default_config
from vfd_dash.core.hashing import compute_config_hash, compute_run_hash
from vfd_dash.io.export_bundle import create_export_bundle, load_export_bundle, list_available_bundles
from vfd_dash.io.datasets import save_datasets, load_datasets
from vfd_dash.metrics.report import MetricsReport


class TestConfigHashing:
    """Test configuration hashing for reproducibility."""

    def test_config_hash_deterministic(self):
        """Same config should produce same hash."""
        config1 = get_default_config()
        config2 = get_default_config()

        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)

        assert hash1 == hash2

    def test_config_hash_changes(self):
        """Different config should produce different hash."""
        config1 = get_default_config()
        config2 = get_default_config()
        config2.seed = 999

        hash1 = compute_config_hash(config1)
        hash2 = compute_config_hash(config2)

        assert hash1 != hash2

    def test_config_json_roundtrip(self):
        """Config should survive JSON roundtrip."""
        config = get_default_config()
        config.run_name = "test_run"
        config.seed = 123

        json_str = config.to_json()
        restored = RunConfig.from_json(json_str)

        assert restored.run_name == "test_run"
        assert restored.seed == 123


class TestExportBundle:
    """Test export bundle creation."""

    def test_create_bundle(self, tmp_path):
        """Test creating export bundle."""
        config = get_default_config()
        config.run_name = "test_bundle"

        metrics = {"test": "metrics"}

        datasets = {
            "test_data": pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        }

        bundle = create_export_bundle(
            config=config,
            metrics=metrics,
            datasets=datasets,
            figures={},
            output_dir=str(tmp_path),
            run_name="test_bundle"
        )

        # Check files exist
        assert bundle.config_file.exists()
        assert bundle.metrics_file.exists()
        assert bundle.manifest_file.exists()
        assert (bundle.datasets_dir / "test_data.parquet").exists()

    def test_bundle_zip(self, tmp_path):
        """Test creating zip archive."""
        config = get_default_config()
        datasets = {"data": pd.DataFrame({"x": [1, 2]})}

        bundle = create_export_bundle(
            config=config,
            metrics={},
            datasets=datasets,
            figures={},
            output_dir=str(tmp_path)
        )

        zip_path = bundle.get_zip_path()
        assert zip_path.exists()

    def test_load_bundle(self, tmp_path):
        """Test loading export bundle."""
        config = get_default_config()
        config.seed = 999

        datasets = {"primes": pd.DataFrame({"m": [1, 2, 3]})}

        bundle = create_export_bundle(
            config=config,
            metrics={"run_hash": "test123"},
            datasets=datasets,
            figures={},
            output_dir=str(tmp_path)
        )

        # Load it back
        loaded = load_export_bundle(str(bundle.run_dir))

        assert "config" in loaded
        assert loaded["config"]["seed"] == 999
        assert "primes" in loaded["datasets"]
        assert len(loaded["datasets"]["primes"]) == 3


class TestDatasets:
    """Test dataset save/load."""

    def test_save_load_parquet(self, tmp_path):
        """Test saving and loading parquet datasets."""
        datasets = {
            "primes": pd.DataFrame({"m": [1, 2, 3], "dir": ["+", "-", "+"]}),
            "stability": pd.DataFrame({"Q": [0.1, 0.2, 0.3]})
        }

        save_datasets(datasets, str(tmp_path), format="parquet")

        loaded = load_datasets(str(tmp_path))

        assert "primes" in loaded
        assert "stability" in loaded
        assert len(loaded["primes"]) == 3

    def test_save_load_csv(self, tmp_path):
        """Test saving and loading CSV datasets."""
        datasets = {
            "test": pd.DataFrame({"x": [1, 2], "y": [3, 4]})
        }

        save_datasets(datasets, str(tmp_path), format="csv")

        loaded = load_datasets(str(tmp_path))

        assert "test" in loaded


class TestReplay:
    """Test replay functionality."""

    def test_list_bundles(self, tmp_path):
        """Test listing available bundles."""
        # Create a few bundles
        for i in range(3):
            config = get_default_config()
            config.seed = i

            create_export_bundle(
                config=config,
                metrics={"run_hash": f"hash{i}"},
                datasets={},
                figures={},
                output_dir=str(tmp_path),
                run_name=f"run_{i}"
            )

        bundles = list_available_bundles(str(tmp_path))

        assert len(bundles) == 3
        assert all("run_hash" in b for b in bundles)

    def test_replay_produces_same_hash(self, tmp_path):
        """Replaying a run should produce identical hash."""
        config = get_default_config()
        config.seed = 42

        hash1 = compute_config_hash(config)

        # Save and reload config
        config.save(str(tmp_path / "config.json"))
        loaded = RunConfig.from_file(str(tmp_path / "config.json"))

        hash2 = compute_config_hash(loaded)

        assert hash1 == hash2
