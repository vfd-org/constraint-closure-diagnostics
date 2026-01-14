# Reproducibility Report

**Last Verified:** 2026-01-14 01:05 UTC
**Reference Run:** `85568e827299b531`
**Verification Run:** `879474f18d1eeba5`
**Changes:** Documentation updates only (BN1 metric presentation, EF→TW naming, AI-scope section)

---

## Verification Levels

### Fast Verification (no heavy compute)

Verify that the included release artifacts match paper-reported values. Runtime: **< 5 seconds**.

```bash
# After installation, verify release metrics
python tools/verify_release_metrics.py --bundle-dir runs/release_20260113_225750

# Expected output: OVERALL: PASS - All metrics match paper-reported values
```

This script reads the pre-computed `metrics.json` and checks:
- Spearman rank correlation (BA): 0.9997 ± 0.0001
- RMSE (BA): 1098 ± 1.0
- BN1 Spearman (ordering perturbation): 0.008 ± 0.001
- BN2 RMSE (scale perturbation): 2395 ± 10
- BN3 β deviation (coordinate perturbation): 0.2 ± 0.01

### Full Reproduction (regenerates figures and metrics)

Regenerate all metrics and figures from scratch. Runtime varies by hardware (may take 10-30+ minutes for large configurations).

```bash
# Complete bundle with bridge overlay and parameter sweep
rhdiag bundle --seed 42 --cell-count 64 --internal-dim 96

# Output: runs/release_<timestamp>/
# Expected: All L0-L4 levels PASS with residuals < 10⁻¹⁴
```

**Note:** Full reproduction uses deterministic seeding. Identical seeds produce bit-identical results regardless of when or where you run.

---

## D1: How to Run the Dashboard from a Clean Environment

### System Requirements
- Python 3.8 or higher
- OS: Linux, macOS, or Windows (WSL tested)
- Memory: Minimum 4GB RAM (8GB recommended for larger cell counts)

### Installation

```bash
# Clone or navigate to repository
cd constraint-closure-diagnostics

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install package in editable mode
pip install -e .
```

### Dependencies

From `pyproject.toml`:

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.20.0 | Array operations, linear algebra |
| scipy | >=1.6.0 | Sparse matrices, eigensolvers |
| pandas | >=1.3.0 | DataFrames for export |
| plotly | >=5.0.0 | Visualization |
| dash | >=2.0.0 | Web dashboard |
| dash-bootstrap-components | >=1.0.0 | UI components |
| pydantic | >=1.9.0 | Configuration validation |
| pytest | >=6.0.0 | Testing |
| networkx | >=2.6.0 | Factorization graphs |
| pyarrow | >=6.0.0 | Parquet export |
| mpmath | >=1.2.0 | High-precision zeta zeros |
| sympy | >=1.9 | Prime generation |

### Lockfile Generation

```bash
# Generate requirements.txt for exact reproducibility
pip freeze > requirements.txt

# Or use pip-tools for lockfile
pip install pip-tools
pip-compile pyproject.toml -o requirements.lock
```

### Running the Dashboard

#### Web Dashboard (UI Mode)
```bash
# Start Dash server
python3 -m vfd_dash.app
# or: vfd-dashboard

# Open browser to http://127.0.0.1:8050
```

#### Headless Mode (CLI)
```bash
# Default run
python3 scripts/run_headless.py

# With parameters
python3 scripts/run_headless.py \
    --seed 42 \
    --cell-count 16 \
    --max-prime-length 50 \
    --probe-count 200 \
    --bridge-mode BA \
    --run-name "my_run"

# From config file
python3 scripts/run_headless.py --config runs/<hash>/config.json
```

### Running Tests
```bash
# All tests
python3 -m pytest tests/ -v

# Specific test file
python3 -m pytest tests/test_invariants.py -v

# With coverage
python3 -m pytest tests/ -v --cov=vfd_dash
```

---

## Determinism

### Random Seeds

The system uses deterministic seeding via `numpy.random.seed(config.seed)`. The default seed is `42`.

Seeds are used in:
- `app.py:run_analysis()`: Sets global numpy seed at start
- `stability.py:StabilityAnalyzer`: Uses seed for probe generation
- `primes.py:InternalPrimeGenerator`: Uses seed for any randomized search
- `bridge_axiom.py:compare_ba_vs_bn()`: Uses seed for BN negation tests

### Numerical Tolerances

| Check | Tolerance | File |
|-------|-----------|------|
| T^12 = I | 1e-12 | canonical.py:314 |
| Weyl relation | 1e-10 | canonical.py:266 |
| Projector resolution | 1e-12 | operators.py:58-71 |
| Projector orthogonality | 1e-12 | operators.py:74-94 |
| Kernel D1 (self-adjoint) | 1e-12 | kernels.py:247 |
| Kernel D2 (T-commute) | 1e-10 | kernels.py:257 |
| Kernel D3 (nonnegative) | 1e-10 | kernels.py:259-283 |
| Bridge BA1 RMSE threshold | 5.0 | bridge_axiom.py:203 |

### Spectrum Computation

The analytic backend (`spectrum/backend.py`) computes eigenvalues using closed-form formulas:

```
lambda_cell(theta_j) = 2R - 2 * sum_{d=1}^R cos(d * theta_j)
where theta_j = 2*pi*j / C
```

This is fully deterministic and produces bit-identical results across runs.

---

## Verification of Determinism

### Hash-Based Verification

Two consecutive runs with identical parameters produce identical outputs:

```
Run 1: eigenvalues=192, hash=42e2052f3f811dce, Q=2.005854
Run 2: eigenvalues=192, hash=42e2052f3f811dce, Q=2.005854
```

### Determinism Test Script

```bash
python3 -c "
import numpy as np
import hashlib
from vfd_dash.vfd.canonical import VFDSpace, TorsionOperator, ShiftOperator
from vfd_dash.spectrum.backend import compute_spectrum, SpectralBackend

for run in [1, 2]:
    np.random.seed(42)
    space = VFDSpace(cell_count=8, internal_dim=24, orbit_count=2, orbit_size=12)
    result = compute_spectrum(
        cell_count=space.cell_count,
        internal_dim=space.internal_dim,
        propagation_range=1,
        backend=SpectralBackend.ANALYTIC_KCAN,
        use_cache=False
    )
    eig_hash = hashlib.sha256(result.eigenvalues.tobytes()).hexdigest()[:16]
    print(f'Run {run}: hash={eig_hash}')
"
```

### Using `tools/hash_outputs.py`

```bash
# Hash artifacts from a run
python3 tools/hash_outputs.py runs/<hash>/

# Compare two runs
python3 tools/hash_outputs.py runs/<hash1>/ runs/<hash2>/
```

---

## Test Suite Results

As of validation run:

| Test File | Tests | Status |
|-----------|-------|--------|
| test_invariants.py | 12 | PASS |
| test_bridge_controls.py | 12 | PASS |
| test_export_replay.py | 8 | PASS |
| test_primes.py | 10 | PASS |
| test_reports.py | 8 | PASS |
| test_spectrum_backends.py | 22 | PASS |
| test_stability.py | 10 | PASS |
| **Total** | **90** | **ALL PASS** |

Run time: ~67 seconds

---

## Output Artifacts

Each run produces:

```
runs/<hash>/
├── config.json           # Full configuration (replayable)
├── metrics.json          # All computed metrics
├── manifest.json         # Run metadata (git hash, timestamps, versions)
├── datasets/
│   ├── internal_primes.parquet
│   ├── internal_primes.csv
│   ├── stability.parquet
│   └── stability.csv
├── figures/
│   └── *.png
└── bundle.zip           # Complete archive
```

### Manifest Schema

```json
{
  "run_hash": "abc123...",
  "run_name": "my_run",
  "timestamp": "2026-01-12T15:00:00",
  "git_commit": "def456...",
  "package_versions": {
    "numpy": "1.24.0",
    "scipy": "1.11.0",
    ...
  },
  "datasets": ["internal_primes", "stability"],
  "figures": []
}
```

---

## Platform Notes

### Tested Configurations
- Python 3.8.10 on Linux (WSL2)
- Ubuntu 20.04 / Windows 10+11 WSL

### Known Issues
1. **Large cell counts**: With `cell_count=64` and `internal_dim=600` (total_dim=38,400), sparse eigsh can be slow. Use `analytic_kcan` backend for K_can.
2. **mpmath import**: First run with mpmath computes zeta zeros; subsequent runs use cache.
3. **Pydantic deprecation warnings**: Using Pydantic v1 config syntax; harmless.

### Performance Recommendations
- For production runs: `cell_count=64`, `probe_count=2000`
- For quick validation: `cell_count=8-16`, `probe_count=200`
- Always use `backend=ANALYTIC_KCAN` for K_can (O(C) vs O(n^3))

---

## Paper Figure Synchronization

The paper (`paper/rh_constraint_diagnostic.tex`) uses figures from `paper/figures/`.
After running `rhdiag bundle`, figures are generated in `runs/.../figures/`.

**Sync command:**

```bash
# After rhdiag bundle, use the sync script:
python tools/sync_paper_figures.py --bundle-dir runs/release_<timestamp>

# The script automatically locates production run and sweep outputs,
# then copies all required figures to paper/figures/
```

The release package includes pre-generated figures matching reference run `85568e827299b531`.
