# Release Verification Report

**Date:** 2026-01-14
**Reference Run Hash:** `85568e827299b531`

---

## Verification Methods

### Fast Verification (< 5 seconds)

Verify that the included release artifacts match paper-reported values:

```bash
cd constraint-closure-diagnostics
pip install -e .
python tools/verify_release_metrics.py --bundle-dir runs/release_20260113_225750
```

**Expected output:** `OVERALL: PASS - All metrics match paper-reported values`

### Full Reproduction (10-30+ minutes)

Regenerate all figures and metrics from scratch:

```bash
rhdiag bundle --seed 42 --cell-count 64 --internal-dim 96
```

**Expected:** All L0-L4 levels PASS with residuals < 10⁻¹⁴

---

## Reference Metrics

The reference run `85568e827299b531` contains the following metrics (paper rounds these for readability):

| Metric | Full Precision | Paper Rounded |
|--------|----------------|---------------|
| Spearman rank correlation | 0.9997310220466599 | 0.9997 |
| Pearson correlation | 0.9999445517627977 | 0.9999 |
| RMSE | 1098.0258683723355 | 1098 |
| MAE | 989.5176557969631 | 990 |

---

## Falsification Metrics

| Mode | BA Value | BN Value | Effect |
|------|----------|----------|--------|
| BN1 (ordering) | Spearman ρ = 0.9997 | ρ = 0.008 | Ordering destroyed |
| BN2 (scale) | RMSE = 1098 | RMSE = 2395 | 2.2× worse |
| BN3 (coordinate) | β deviation = 0 (exact) | β deviation = 0.2 | Coordinate shifted |

---

## Closure Ladder

| Level | Status | Total Residual | Family Breakdown |
|-------|--------|----------------|------------------|
| L0 | PASS | 0 | (structural) |
| L1 | PASS | 1.35×10⁻¹⁴ | TW: 1.35×10⁻¹⁴ |
| L2 | PASS | 1.96×10⁻¹⁴ | +Symmetry: 6.08×10⁻¹⁵ |
| L3 | PASS | 1.96×10⁻¹⁴ | +Positivity: 0 |
| L4 | PASS | 1.96×10⁻¹⁴ | +Trace: 0 |

---

## Changes Since Reference Run

**Documentation only** (no core math/parameter changes):
- BN1 metric presentation: Changed from ratio to Spearman correlation values
- EF → TW naming: Renamed "Explicit Formula" to "Algebraic Relations (TW)"
- AI-scope section: Added "AI-Readable Scope (Claims and Non-Claims)" subsection
- Bridge clarification: Added note that implementation is included for reproducibility
- Verification tooling: Added `tools/verify_release_metrics.py` for fast verification

**Core math unchanged:**
- Eigenvalue computation: Unchanged
- Closure constraints: Unchanged
- Bridge projection: Unchanged
- Reference data: Unchanged (mpmath zeta zeros)
