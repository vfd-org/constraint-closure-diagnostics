# Release Audit

**Repository:** constraint-closure-diagnostics
**Date:** 2026-01-14
**Total Files:** 123

---

## Contents Tree

```
release_public/
├── .gitignore
├── FINAL_CONFIRMATION.md
├── LICENSE (MIT)
├── README.md
├── RELEASE_AUDIT.md (this file)
├── REPO_AUDIT_SUMMARY.md
├── pyproject.toml
├── docs/
│   ├── REPRODUCIBILITY_REPORT.md
│   ├── SAFETY_BOUNDARY.md
│   └── SHAREABLE_PROFILE.md
├── paper/
│   ├── ASSUMPTIONS_AND_SCOPE.md
│   ├── README.md
│   ├── rh_constraint_diagnostic.tex
│   └── figures/ (8 PNG files)
│       ├── fig01_residual_ladder.png
│       ├── fig02_phase_map.png
│       ├── fig03_constraint_waterfall.png
│       ├── fig04_positivity_wall_grid.png
│       ├── fig04_spectrum_histogram.png
│       ├── fig05_collapse_geometry.png
│       ├── fig06_zero_overlay.png
│       └── fig07_falsification.png
├── runs/
│   └── release_20260113_225750/
│       ├── BUNDLE_REPORT.md
│       ├── 83c751c958966979/ (initial run)
│       ├── 85568e827299b531/ (reference run with full figures)
│       └── sweep_20260113_225759/ (parameter sweep)
├── src/
│   └── vfd_dash/ (diagnostic framework code)
│       ├── bridge/ (bridge axiom implementation)
│       ├── constraints/ (closure families)
│       ├── core/ (config, hashing, logging)
│       ├── diagnostics/ (performance)
│       ├── figures/ (plot generators)
│       ├── io/ (cache, export)
│       ├── metrics/ (statistics)
│       ├── reports/ (markdown generation)
│       ├── spectrum/ (eigenvalue computation)
│       ├── ui/ (Dash dashboard)
│       └── vfd/ (VFD operators)
├── tests/ (18 test files)
└── tools/ (audit and hashing utilities)
```

---

## Safety Confirmations

| Check | Status | Notes |
|-------|--------|-------|
| No archive/ directory | PASS | Private materials excluded |
| No hidden API calls | PASS | All computation local |
| No sensitive constants | PASS | All parameters documented |
| No undisclosed dependencies | PASS | Standard PyPI packages only |
| Bridge treated as black box | PASS | No explicit formulas in paper |
| LaTeX self-contained | PASS | Figures in paper/figures/ |
| .gitignore excludes archive | PASS | `/archive/` in exclusion list |

---

## Archive Separation

The following private materials are in `/archive/` (NOT in release_public):

| File | Reason |
|------|--------|
| `VFD_Mathematical_Proofs.md` | Beyond diagnostic scope |
| `METHODOLOGY_AND_JOURNEY.md` | Internal documentation |
| `STATISTICAL_SIGNIFICANCE.md` | Contains strong claims |
| `EVIDENCE_INVENTORY.md` | Internal validation tracking |
| `PROOF_STATUS.md` | Status tracking |
| `UPGRADE_PLAN.md` | Development roadmap |
| `SPEC_COMPARISON.md` | Specification comparison |
| `COMPUTATION_MAP.md` | Implementation details |
| `WEYL_FAILURE_ANALYSIS.md` | Failure analysis |
| `PERFORMANCE_DIAGNOSTICS.md` | Profiling data |
| `VFD_Proofs.tex` | LaTeX proofs document |
| `public_draft/` | Draft disclosure docs |

---

## File Categories

### Documentation (8 files)
- README.md (main)
- LICENSE
- FINAL_CONFIRMATION.md
- REPO_AUDIT_SUMMARY.md
- RELEASE_AUDIT.md
- docs/REPRODUCIBILITY_REPORT.md
- docs/SAFETY_BOUNDARY.md
- docs/SHAREABLE_PROFILE.md

### Paper Assets (11 files)
- paper/rh_constraint_diagnostic.tex
- paper/README.md
- paper/ASSUMPTIONS_AND_SCOPE.md
- paper/figures/*.png (8 figures)

### Source Code (47 files)
- src/vfd_dash/**/*.py

### Tests (18 files)
- tests/test_*.py

### Tools (2 files)
- tools/audit_release_bundle.py
- tools/hash_outputs.py

### Run Data (15 files)
- runs/release_20260113_225750/**

### Configuration (2 files)
- pyproject.toml
- .gitignore

---

## Reference Run Metrics

| Metric | Value |
|--------|-------|
| Run hash | 85568e827299b531 |
| Rank correlation | 0.9997310220466599 |
| Pearson correlation | 0.9999445517627977 |
| RMSE | 1098.0258683723355 |
| MAE | 989.5176557969631 |
| BN1 ratio | 1.5011891832990705 |
| BN2 ratio | 2.1814686705966104 |
| BN3 ratio | 200001.00000000017 |

---

## Signed

Automated Release Audit
Date: 2026-01-14
