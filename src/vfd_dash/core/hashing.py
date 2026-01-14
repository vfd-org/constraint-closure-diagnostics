"""
Hashing utilities for reproducibility.

Provides deterministic hashing of configurations and run outputs.
"""

import hashlib
import json
from datetime import datetime
from typing import Any, Optional
import subprocess


def compute_config_hash(config: Any) -> str:
    """
    Compute deterministic hash of configuration.

    Args:
        config: Configuration object with to_dict() method or dict

    Returns:
        Hex string of SHA-256 hash (first 16 chars)
    """
    if hasattr(config, "to_dict"):
        config_dict = config.to_dict()
    else:
        config_dict = dict(config)

    # Sort keys for deterministic serialization
    config_str = json.dumps(config_dict, sort_keys=True, default=str)
    hash_obj = hashlib.sha256(config_str.encode("utf-8"))
    return hash_obj.hexdigest()[:16]


def compute_run_hash(config: Any, timestamp: Optional[datetime] = None) -> str:
    """
    Compute unique run hash from config and timestamp.

    Args:
        config: Run configuration
        timestamp: Run timestamp (defaults to now)

    Returns:
        Hex string of SHA-256 hash (first 16 chars)
    """
    if timestamp is None:
        timestamp = datetime.now()

    config_hash = compute_config_hash(config)
    time_str = timestamp.isoformat()

    combined = f"{config_hash}:{time_str}"
    hash_obj = hashlib.sha256(combined.encode("utf-8"))
    return hash_obj.hexdigest()[:16]


def compute_data_hash(data: bytes) -> str:
    """
    Compute hash of binary data.

    Args:
        data: Binary data

    Returns:
        Hex string of SHA-256 hash (first 16 chars)
    """
    return hashlib.sha256(data).hexdigest()[:16]


def get_git_commit_hash() -> Optional[str]:
    """
    Get current git commit hash if in a git repository.

    Returns:
        Git commit hash or None if not in a repo
    """
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()[:16]
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def get_package_versions() -> dict:
    """
    Get versions of key packages for reproducibility.

    Returns:
        Dictionary of package names to versions
    """
    versions = {}
    packages = ["numpy", "scipy", "pandas", "plotly", "dash", "pydantic", "networkx"]

    for pkg in packages:
        try:
            import importlib.metadata
            versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            versions[pkg] = "unknown"

    return versions
