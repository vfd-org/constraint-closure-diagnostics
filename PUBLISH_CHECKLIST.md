# Publish Checklist

**Repository:** constraint-closure-diagnostics
**Date:** 2026-01-14

Use this checklist to push the release to GitHub.

---

## Pre-Push Verification

- [x] All closure levels (L0-L4) pass
- [x] Closure residuals match reference run 85568e827299b531
- [x] LaTeX compiles with self-contained figures in paper/figures/
- [x] Archive directory NOT present in release/
- [x] .gitignore excludes /archive/
- [x] LICENSE (MIT) present
- [x] README.md has Quick Start and AI-Readable Scope sections
- [x] pyproject.toml has correct package name and author
- [x] All 8 paper figures present in paper/figures/
- [x] RELEASE_AUDIT.md generated
- [x] Fast verification passes

---

## Fast Verification Command

```bash
pip install -e .
python tools/verify_release_metrics.py --bundle-dir runs/release_20260113_225750
# Expected: OVERALL: PASS
```

---

## Reference Metrics (for validation)

**Reference run:** `85568e827299b531`

| Metric | Paper Rounded | Full Precision |
|--------|---------------|----------------|
| Spearman ρ (BA) | 0.9997 | 0.9997310220466599 |
| RMSE (BA) | 1098 | 1098.0258683723355 |
| Pearson correlation | 0.9999 | 0.9999445517627977 |
| MAE | 990 | 989.5176557969631 |

**Falsification metrics:**

| Mode | BA Value | BN Value | Effect |
|------|----------|----------|--------|
| BN1 (ordering) | Spearman ρ = 0.9997 | ρ = 0.008 | Ordering destroyed |
| BN2 (scale) | RMSE = 1098 | RMSE = 2395 | 2.2× worse |
| BN3 (coordinate) | β deviation = 0 | β deviation = 0.2 | Coordinate shifted |

**Note:** Paper values rounded to 4 significant figures for readability.

---

## GitHub Push Steps

### 1. Create New GitHub Repository

Go to https://github.com/new
- Repository name: constraint-closure-diagnostics
- Description: Constraint-closure diagnostic framework with RH-motivated spectral case study
- Visibility: Public
- Do NOT initialize with README, .gitignore, or license (we have these)

### 2. Initialize Git in release folder

```bash
cd release/constraint-closure-diagnostics
git init
git add .
git commit -m "Initial release: constraint-closure diagnostics v1.0.0

Reproducible diagnostic framework for constraint-closure analysis.
Reference run: 85568e827299b531

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>"
```

### 3. Push to GitHub

```bash
git remote add origin https://github.com/YOUR_USERNAME/constraint-closure-diagnostics.git
git branch -M main
git push -u origin main
```

### 4. Add Release Tag

```bash
git tag -a v1.0.0 -m "Release v1.0.0 - Initial public release"
git push origin v1.0.0
```

---

## Post-Push Verification

After pushing, verify:

1. **Clone test**: Clone fresh and run:
   ```bash
   git clone https://github.com/YOUR_USERNAME/constraint-closure-diagnostics.git
   cd constraint-closure-diagnostics
   pip install -e .
   python tools/verify_release_metrics.py --bundle-dir runs/release_20260113_225750
   ```
   Confirm: `OVERALL: PASS`

2. **Paper compilation**: Upload paper/ folder to Overleaf and verify it compiles.

3. **No archive leak**: Confirm /archive/ is NOT visible in the repository.

---

## Optional: GitHub Repository Settings

- Add topics: mathematics, riemann-hypothesis, spectral-theory, diagnostics
- Add description
- Enable Issues for feedback
- Consider adding a CITATION.cff for academic citation

---

## Safety Reminder

This repository:
- Does NOT claim a proof of the Riemann Hypothesis
- Presents a diagnostic framework with empirical results
- Treats the Bridge Axiom as a testable hypothesis
- Includes explicit scope limitations in docs/SAFETY_BOUNDARY.md

---

**Signed:** Automated Publish Checklist
**Date:** 2026-01-14
