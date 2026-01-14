#!/usr/bin/env python3
"""
Hash verification tool for VFD dashboard outputs.

Usage:
    python tools/hash_outputs.py runs/<hash>/
    python tools/hash_outputs.py runs/<hash1>/ runs/<hash2>/

Computes SHA-256 hashes of all artifacts in a run directory
and optionally compares two runs for determinism verification.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Optional


def hash_file(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def hash_directory(dirpath: Path) -> Dict[str, str]:
    """
    Hash all files in a directory recursively.

    Returns:
        Dictionary mapping relative path -> hash
    """
    hashes = {}
    dirpath = Path(dirpath)

    if not dirpath.exists():
        print(f"Error: Directory {dirpath} does not exist", file=sys.stderr)
        return hashes

    for filepath in sorted(dirpath.rglob("*")):
        if filepath.is_file():
            rel_path = filepath.relative_to(dirpath)
            # Skip certain files that may differ (e.g., timestamps in zip)
            if rel_path.suffix in [".zip"]:
                continue
            hashes[str(rel_path)] = hash_file(filepath)

    return hashes


def print_hashes(hashes: Dict[str, str], title: str = "File Hashes"):
    """Print hashes in a formatted table."""
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

    for path, hash_val in sorted(hashes.items()):
        print(f"  {hash_val[:16]}  {path}")

    print(f"{'='*60}")
    print(f" Total files: {len(hashes)}")

    # Compute aggregate hash
    all_hashes = "".join(sorted(hashes.values()))
    aggregate = hashlib.sha256(all_hashes.encode()).hexdigest()[:16]
    print(f" Aggregate hash: {aggregate}")
    print(f"{'='*60}\n")

    return aggregate


def compare_runs(dir1: Path, dir2: Path) -> bool:
    """
    Compare two run directories for determinism.

    Returns:
        True if runs are identical, False otherwise
    """
    hashes1 = hash_directory(dir1)
    hashes2 = hash_directory(dir2)

    agg1 = print_hashes(hashes1, f"Run 1: {dir1.name}")
    agg2 = print_hashes(hashes2, f"Run 2: {dir2.name}")

    print("\n" + "="*60)
    print(" COMPARISON RESULTS")
    print("="*60)

    # Check for missing files
    only_in_1 = set(hashes1.keys()) - set(hashes2.keys())
    only_in_2 = set(hashes2.keys()) - set(hashes1.keys())

    if only_in_1:
        print(f"\n Files only in run 1:")
        for f in sorted(only_in_1):
            print(f"   - {f}")

    if only_in_2:
        print(f"\n Files only in run 2:")
        for f in sorted(only_in_2):
            print(f"   + {f}")

    # Check for differing files
    common = set(hashes1.keys()) & set(hashes2.keys())
    differing = []
    for path in common:
        if hashes1[path] != hashes2[path]:
            differing.append(path)

    if differing:
        print(f"\n Files with different hashes:")
        for f in sorted(differing):
            print(f"   ! {f}")
            print(f"     Run 1: {hashes1[f][:16]}")
            print(f"     Run 2: {hashes2[f][:16]}")

    # Summary
    print("\n" + "-"*60)
    if agg1 == agg2 and not only_in_1 and not only_in_2:
        print(" RESULT: IDENTICAL (determinism verified)")
        return True
    else:
        print(" RESULT: DIFFERENT")
        print(f"   Aggregate 1: {agg1}")
        print(f"   Aggregate 2: {agg2}")
        print(f"   Files only in 1: {len(only_in_1)}")
        print(f"   Files only in 2: {len(only_in_2)}")
        print(f"   Files differing: {len(differing)}")
        return False


def hash_config(config_path: Path) -> str:
    """Hash a config file deterministically."""
    with open(config_path) as f:
        config = json.load(f)

    # Sort keys for deterministic serialization
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(
        description="Hash and compare VFD run outputs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python tools/hash_outputs.py runs/abc123/
  python tools/hash_outputs.py runs/abc123/ runs/def456/
  python tools/hash_outputs.py --config runs/abc123/config.json
"""
    )

    parser.add_argument(
        "directories",
        nargs="+",
        type=Path,
        help="Run directory(ies) to hash"
    )
    parser.add_argument(
        "--config",
        action="store_true",
        help="Hash config file only"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    args = parser.parse_args()

    if args.config:
        # Hash config files
        for dirpath in args.directories:
            config_path = dirpath / "config.json" if dirpath.is_dir() else dirpath
            if config_path.exists():
                config_hash = hash_config(config_path)
                print(f"{config_hash}  {config_path}")
            else:
                print(f"Error: {config_path} not found", file=sys.stderr)
        return 0

    if len(args.directories) == 1:
        # Single directory: just print hashes
        hashes = hash_directory(args.directories[0])
        if args.json:
            print(json.dumps(hashes, indent=2))
        else:
            print_hashes(hashes, f"Run: {args.directories[0].name}")
        return 0

    elif len(args.directories) == 2:
        # Two directories: compare
        identical = compare_runs(args.directories[0], args.directories[1])
        return 0 if identical else 1

    else:
        print("Error: Provide 1 or 2 directories", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
