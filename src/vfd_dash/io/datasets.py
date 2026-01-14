"""
Dataset saving and loading utilities.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, Optional


def save_datasets(
    datasets: Dict[str, pd.DataFrame],
    output_dir: str,
    format: str = "parquet"
) -> Dict[str, str]:
    """
    Save datasets to disk.

    Args:
        datasets: Dictionary of name -> DataFrame
        output_dir: Output directory
        format: Output format ("parquet" or "csv")

    Returns:
        Dictionary of name -> file path
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    paths = {}

    for name, df in datasets.items():
        if format == "parquet":
            file_path = out_path / f"{name}.parquet"
            df.to_parquet(file_path, index=False)
        else:
            file_path = out_path / f"{name}.csv"
            df.to_csv(file_path, index=False)

        paths[name] = str(file_path)

    return paths


def load_datasets(
    input_dir: str,
    names: Optional[list] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load datasets from disk.

    Args:
        input_dir: Input directory
        names: Specific dataset names to load (None = all)

    Returns:
        Dictionary of name -> DataFrame
    """
    in_path = Path(input_dir)
    datasets = {}

    # Find parquet files first, then CSV
    for ext in [".parquet", ".csv"]:
        for file_path in in_path.glob(f"*{ext}"):
            name = file_path.stem

            if names is not None and name not in names:
                continue

            if name in datasets:
                continue  # Already loaded

            if ext == ".parquet":
                datasets[name] = pd.read_parquet(file_path)
            else:
                datasets[name] = pd.read_csv(file_path)

    return datasets
