# Constraint-Closure Diagnostics

## In Plain English

This repository contains a **diagnostic tool** for exploring mathematical structures related to the Riemann Hypothesis (RH). It does not prove RH or compute actual zeros of the zeta function.

What it does:
1. Builds a mathematical construction with specific properties (operators, symmetries, positivity)
2. Checks that the construction satisfies a ladder of constraints (L0–L4)
3. Projects internal data to "zero-like" points and compares with reference values
4. Tests whether breaking the mapping degrades the match (falsification)

Think of it as a **diagnostic lens**: if the structure were arbitrary, breaking it shouldn't matter—but it does. This suggests the constraints capture something meaningful, though we make no claim about what that implies for RH itself.

---

## AI-Readable Scope

### Claims

- This is a **diagnostic framework** for constraint-closure analysis
- The construction passes all internal consistency checks (L0–L4 closure ladder)
- Production run achieves residuals < 10⁻¹⁴; sweep configurations achieve < 10⁻¹³
- Bridge overlay shows rank correlation ρ = 0.9997 with reference data
- Falsification tests show sensitivity: perturbing the bridge degrades results

### Non-Claims

- This work does **NOT** prove the Riemann Hypothesis
- This work does **NOT** compute actual zeros of ζ(s)
- Structural motivation does **NOT** imply logical equivalence
- Numerical agreement does **NOT** constitute mathematical proof
- The bridge mapping is a **testable hypothesis**, not a derived theorem

### How to Verify in 60 Seconds

```bash
pip install -e .
python tools/verify_release_metrics.py --bundle-dir runs/release_20260113_225750
# Expected: OVERALL: PASS
```

---

## What This Is

This repository implements a **constraint-closure diagnostic framework**—a systematic methodology for evaluating mathematical structures through hierarchical constraint satisfaction. We apply it to an **RH-motivated spectral case study**, constructing finite-dimensional operators that satisfy positivity, symmetry, and algebraic constraints inspired by (but not equivalent to) Riemann Hypothesis criteria. The framework verifies internal consistency, tests falsifiability through perturbation modes, and produces reproducible, auditable outputs.

## What This Is NOT

- **NOT a proof of the Riemann Hypothesis**: This work does not prove RH
- **NOT computation of zeta zeros**: Projected values are artifacts of a bridge mapping, not computed zeros of ζ(s)
- **NOT mathematical equivalence**: Structural motivation ≠ logical equivalence; properties inspired by RH do not imply RH
- **Bridge as hypothesis**: The bridge mapping is treated as a testable hypothesis, empirically evaluated via falsification modes rather than presented as a derived theorem

## Why It Matters

**For mathematicians and researchers:**
- We built a machine-checkable ladder of constraints (L0–L4) that any finite construction must pass
- If the structure were arbitrary, perturbing the bridge mapping shouldn't degrade results—but it does (ordering perturbation drops ρ from 0.9997 to 0.008, scale perturbation increases RMSE from 1098 to 2395, coordinate perturbation shifts β deviation from 0 to 0.2)
- This is a diagnostic lens for RH-motivated spectral structures, providing graduated feedback rather than binary pass/fail

**Interpretation:** These results support the view that RH-related spectral behavior may be better understood as a property of a structured constraint system. The correspondence between construction and reference data, combined with degradation under perturbation, suggests that classical formulations may be missing structural degrees of freedom. This is offered as a diagnostic hypothesis, not a theorem.

---

## Quick Start

### Fast Verification (< 5 seconds)

Verify that pre-computed metrics match paper values:

```bash
git clone https://github.com/vfd-org/constraint-closure-diagnostics.git
cd constraint-closure-diagnostics
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .

python tools/verify_release_metrics.py --bundle-dir runs/release_20260113_225750
# Expected: OVERALL: PASS - All metrics match paper-reported values
```

### Full Reproduction (10-30+ minutes)

Regenerate all figures and metrics from scratch:

```bash
rhdiag bundle --seed 42 --cell-count 64 --internal-dim 96
```

Output appears in `runs/release_<timestamp>/`. All closure levels (L0–L4) should pass with residuals < 10⁻¹⁴.

---

## Reproducing Paper Figures

| Paper Figure | Generated File | Command |
|--------------|----------------|---------|
| Fig. 1 (Residual Ladder) | `fig01_residual_ladder.png` | `rhdiag run --seed 42 --cell-count 64 --internal-dim 96` |
| Fig. 2 (Constraint Waterfall) | `fig03_constraint_waterfall.png` | Same |
| Fig. 3 (Spectrum Histogram) | `fig04_spectrum_histogram.png` | Same |
| Fig. 5 (Phase Map) | `fig02_phase_map.png` | `rhdiag sweep --param1 cell_count --values1 16,32,64 --param2 propagation_range --values2 1,2,3` |
| Fig. 6 (Positivity Wall) | `fig04_positivity_wall_grid.png` | Same as sweep |
| Fig. 7 (Zero Overlay) | `fig06_zero_overlay.png` | `rhdiag run --seed 42 --cell-count 64 --internal-dim 96 --bridge-mode BA` |
| Fig. 8 (Falsification) | `fig07_falsification.png` | Same as bridge run |

**Sync figures to paper directory:**

```bash
python tools/sync_paper_figures.py --bundle-dir runs/release_<timestamp>
```

---

## Reference Run Metrics

**Run hash:** `85568e827299b531`

| Metric | Value | Paper Rounded |
|--------|-------|---------------|
| Spearman rank correlation | 0.9997310220466599 | 0.9997 |
| RMSE | 1098.0258683723355 | 1098 |
| BN1 Spearman (ordering) | 0.008357826895787696 | 0.008 |
| BN2 RMSE (scale) | 2395.309031 | 2395 |
| BN3 β deviation (coordinate) | 0.2 | 0.2 |

**Note:** Paper values are rounded to 4 significant figures for readability. Full precision values are stored in `metrics.json`.

---

## Repository Structure

```
constraint-closure-diagnostics/
├── paper/                   # LaTeX paper and figures
│   ├── rh_constraint_diagnostic.tex
│   ├── figures/             # Pre-generated figures for paper
│   └── README.md
├── src/vfd_dash/            # Diagnostic framework source code
│   ├── bridge/              # Bridge projection and falsification
│   ├── constraints/         # Closure ladder (L0-L4)
│   ├── figures/             # Figure generators
│   ├── vfd/                 # Core spectral construction
│   └── cli.py               # Command-line interface
├── runs/                    # Reproducible outputs (manifests, figures)
├── docs/                    # Safety and reproducibility documentation
│   ├── SAFETY_BOUNDARY.md   # Explicit scope limits
│   ├── REPRODUCIBILITY_REPORT.md
│   └── SHAREABLE_PROFILE.md
├── tests/                   # Test suite
└── tools/                   # Audit and helper scripts
    ├── verify_release_metrics.py  # Fast verification
    ├── sync_paper_figures.py      # Figure synchronization
    └── hash_outputs.py            # Output comparison
```

---

## CLI Reference

```bash
# Single diagnostic run
rhdiag run --seed 42 --cell-count 64 --internal-dim 96

# Bridge mode (comparison with reference data)
rhdiag run --seed 42 --cell-count 64 --internal-dim 96 --bridge-mode BA

# Parameter sweep
rhdiag sweep --param1 cell_count --values1 16,32,64 \
             --param2 propagation_range --values2 1,2,3

# Complete release bundle
rhdiag bundle --seed 42 --cell-count 64 --internal-dim 96

# List previous runs
rhdiag list-runs
```

---

## Running Tests

```bash
pytest tests/ -v
```

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Citation

If referencing this work:

```
Lee Smart. "A Constraint-Closure Diagnostic Framework with Application to
RH-Motivated Spectral Structures." 2026.
Software: https://github.com/vfd-org/constraint-closure-diagnostics
```

---

## Contact

Lee Smart
Vibrational Field Dynamics Institute
contact@vibrationalfielddynamics.org
@vfd_org
