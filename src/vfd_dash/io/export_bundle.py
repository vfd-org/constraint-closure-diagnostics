"""
Export Bundle Creation.

Creates reproducible export bundles containing all run artifacts.
"""

import json
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import pandas as pd

from ..core.hashing import compute_run_hash, get_git_commit_hash, get_package_versions


@dataclass
class ExportBundle:
    """
    Export bundle for a VFD run.

    Contains all artifacts needed to reproduce and verify the run.
    """
    run_hash: str
    run_dir: Path
    config_file: Path
    datasets_dir: Path
    figures_dir: Path
    metrics_file: Path
    manifest_file: Path

    def get_zip_path(self) -> Path:
        """Get path to zip file."""
        return self.run_dir / "bundle.zip"

    def create_zip(self) -> Path:
        """Create zip archive of the bundle."""
        zip_path = self.get_zip_path()

        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Add config
            if self.config_file.exists():
                zf.write(self.config_file, "config.json")

            # Add metrics
            if self.metrics_file.exists():
                zf.write(self.metrics_file, "metrics.json")

            # Add manifest
            if self.manifest_file.exists():
                zf.write(self.manifest_file, "manifest.json")

            # Add datasets
            if self.datasets_dir.exists():
                for f in self.datasets_dir.iterdir():
                    zf.write(f, f"datasets/{f.name}")

            # Add figures
            if self.figures_dir.exists():
                for f in self.figures_dir.iterdir():
                    zf.write(f, f"figures/{f.name}")

        return zip_path


def create_export_bundle(
    config: Any,
    metrics: Any,
    datasets: Dict[str, pd.DataFrame],
    figures: Dict[str, bytes],
    output_dir: str = "runs",
    run_name: Optional[str] = None
) -> ExportBundle:
    """
    Create a complete export bundle.

    Args:
        config: Run configuration
        metrics: Metrics report
        datasets: Dictionary of dataset name -> DataFrame
        figures: Dictionary of figure name -> bytes (PNG/SVG)
        output_dir: Output directory
        run_name: Optional run name

    Returns:
        ExportBundle object
    """
    now = datetime.now()
    run_hash = compute_run_hash(config, now)

    # Create directory structure
    run_dir = Path(output_dir) / run_hash
    datasets_dir = run_dir / "datasets"
    figures_dir = run_dir / "figures"

    run_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(exist_ok=True)

    # Save config
    config_file = run_dir / "config.json"
    with open(config_file, "w") as f:
        if hasattr(config, "to_json"):
            f.write(config.to_json())
        else:
            json.dump(config, f, indent=2, default=str)

    # Save metrics
    metrics_file = run_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        if hasattr(metrics, "to_json"):
            f.write(metrics.to_json())
        else:
            json.dump(metrics, f, indent=2, default=str)

    # Save datasets
    for name, df in datasets.items():
        parquet_path = datasets_dir / f"{name}.parquet"
        csv_path = datasets_dir / f"{name}.csv"
        df.to_parquet(parquet_path, index=False)
        df.to_csv(csv_path, index=False)

    # Save figures
    for name, data in figures.items():
        fig_path = figures_dir / name
        with open(fig_path, "wb") as f:
            f.write(data)

    # Create manifest
    manifest = {
        "run_hash": run_hash,
        "run_name": run_name or config.run_name if hasattr(config, "run_name") else "default",
        "timestamp": now.isoformat(),
        "git_commit": get_git_commit_hash(),
        "package_versions": get_package_versions(),
        "datasets": list(datasets.keys()),
        "figures": list(figures.keys()),
    }

    manifest_file = run_dir / "manifest.json"
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    bundle = ExportBundle(
        run_hash=run_hash,
        run_dir=run_dir,
        config_file=config_file,
        datasets_dir=datasets_dir,
        figures_dir=figures_dir,
        metrics_file=metrics_file,
        manifest_file=manifest_file,
    )

    # Create zip
    bundle.create_zip()

    return bundle


def load_export_bundle(bundle_path: str) -> Dict[str, Any]:
    """
    Load an export bundle from disk.

    Args:
        bundle_path: Path to bundle directory or zip file

    Returns:
        Dictionary with config, metrics, datasets, manifest
    """
    path = Path(bundle_path)

    # Handle zip file
    if path.suffix == ".zip":
        extract_dir = path.parent / path.stem
        with zipfile.ZipFile(path, 'r') as zf:
            zf.extractall(extract_dir)
        path = extract_dir

    result = {}

    # Load config
    config_file = path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            result["config"] = json.load(f)

    # Load metrics
    metrics_file = path / "metrics.json"
    if metrics_file.exists():
        with open(metrics_file) as f:
            result["metrics"] = json.load(f)

    # Load manifest
    manifest_file = path / "manifest.json"
    if manifest_file.exists():
        with open(manifest_file) as f:
            result["manifest"] = json.load(f)

    # Load datasets
    datasets_dir = path / "datasets"
    result["datasets"] = {}
    if datasets_dir.exists():
        for f in datasets_dir.glob("*.parquet"):
            name = f.stem
            result["datasets"][name] = pd.read_parquet(f)

    return result


def list_available_bundles(output_dir: str = "runs") -> List[Dict[str, Any]]:
    """
    List available export bundles.

    Args:
        output_dir: Directory containing bundles

    Returns:
        List of bundle summaries
    """
    bundles = []
    runs_path = Path(output_dir)

    if not runs_path.exists():
        return bundles

    for run_dir in runs_path.iterdir():
        if run_dir.is_dir():
            manifest_file = run_dir / "manifest.json"
            if manifest_file.exists():
                with open(manifest_file) as f:
                    manifest = json.load(f)
                    bundles.append({
                        "run_hash": manifest.get("run_hash"),
                        "run_name": manifest.get("run_name"),
                        "timestamp": manifest.get("timestamp"),
                        "path": str(run_dir),
                    })

    # Sort by timestamp
    bundles.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

    return bundles
