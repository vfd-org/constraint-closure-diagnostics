# Release Checklist

Use this checklist before publishing the repository.

## Pre-Release Verification

### Core Functionality

- [ ] Package installs cleanly: `pip install -e .`
- [ ] Fast verification passes: `python tools/verify_release_metrics.py --bundle-dir runs/release_20260113_225750`
- [ ] Tests pass: `pytest tests/ -v`

### Paper and Figures

- [ ] Paper builds without errors: `cd paper && pdflatex rh_constraint_diagnostic.tex`
- [ ] Figures sync script works: `python tools/sync_paper_figures.py --bundle-dir runs/release_20260113_225750 --dry-run`
- [ ] All 8 paper figures present in `paper/figures/`

### Documentation

- [ ] README.md contains:
  - [ ] Plain-English summary
  - [ ] AI-Readable Scope (Claims / Non-Claims)
  - [ ] Fast verification command
  - [ ] Full reproduction command
- [ ] LICENSE present (MIT)
- [ ] SAFETY_BOUNDARY.md present with non-claims
- [ ] REPRODUCIBILITY_REPORT.md has fast/full verification sections

### Metrics Consistency

Reference run `85568e827299b531` values (paper rounds for readability):

| Metric | Full Precision | Paper Rounded | Tolerance |
|--------|----------------|---------------|-----------|
| Spearman ρ (BA) | 0.9997310220466599 | 0.9997 | ±0.0001 |
| RMSE (BA) | 1098.0258683723355 | 1098 | ±1.0 |
| BN1 Spearman | 0.008357826895787696 | 0.008 | ±0.001 |
| BN2 RMSE | 2395.309031 | 2395 | ±10 |
| BN3 β deviation | 0.2 | 0.2 | ±0.01 |

- [ ] Paper abstract mentions correct residual thresholds: <10⁻¹⁴ production, <10⁻¹³ sweep
- [ ] Falsification metrics match in paper, README, and docs

### Scope and Claims

- [ ] NO claim of RH proof anywhere
- [ ] NO claim of computing actual zeta zeros
- [ ] Bridge described as "testable hypothesis" not "theorem"
- [ ] "Diagnostic framework" language used consistently

## Final Steps

- [ ] Remove any debug/test artifacts
- [ ] Update version tag if applicable
- [ ] Confirm .gitignore excludes archive/
- [ ] Final commit message is descriptive

## Post-Release

- [ ] Verify clone + install works from fresh environment
- [ ] Confirm fast verification passes on clean install
- [ ] Update citation if DOI assigned
