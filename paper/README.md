# RH Constraint-Diagnostic Paper

This directory contains the LaTeX source for the paper:

**"A Constraint-Closure Diagnostic Framework with Application to RH-Motivated Spectral Structures"**

## Important Notice

This paper does NOT claim a proof of the Riemann Hypothesis. It presents a diagnostic framework, empirical results, and falsifiability analysis.

## Compilation

### Requirements

- LaTeX distribution (TeX Live, MiKTeX, or similar)
- Required packages: amsmath, amssymb, amsthm, graphicx, booktabs, hyperref, natbib, float, listings

### Compile Locally

```bash
cd paper/
pdflatex rh_constraint_diagnostic.tex
pdflatex rh_constraint_diagnostic.tex  # Run twice for references
```

### Compile with Overleaf

1. Create a new project on [Overleaf](https://www.overleaf.com)
2. Upload `rh_constraint_diagnostic.tex`
3. Upload the `figures/` folder (already included in this release)
4. Compile

## Figure Dependencies

All required figures are pre-generated and included in the `figures/` subdirectory.
The LaTeX file references them via `{figures/fig*.png}` paths.

To regenerate figures from source:

```bash
cd ..
pip install -e .
rhdiag bundle --seed 42 --cell-count 64 --internal-dim 96
```

Then copy the new figures from `runs/release_*/` to `paper/figures/`.

## Figure Mapping

| Paper Figure | Source File |
|--------------|-------------|
| Fig. 1 | `fig01_residual_ladder.png` |
| Fig. 2 | `fig03_constraint_waterfall.png` |
| Fig. 3 | `fig04_spectrum_histogram.png` |
| Fig. 4 | `fig05_collapse_geometry.png` (collapse geometry) |
| Fig. 5 | `fig02_phase_map.png` (from sweep) |
| Fig. 6 | `fig04_positivity_wall_grid.png` (from sweep) |
| Fig. 7 | `fig06_zero_overlay.png` |
| Fig. 8 | `fig07_falsification.png` |

## Scope and Limitations

This paper presents:
- A constraint-closure diagnostic framework
- Application to an RH-motivated spectral case study
- Falsifiability tests and statistical analysis

This paper does NOT:
- Claim proof of the Riemann Hypothesis
- Claim to compute actual zeta zeros
- Claim mathematical equivalence between VFD and zeta

See `../docs/SAFETY_BOUNDARY.md` for detailed scope limitations.

## Reproducibility

All results can be reproduced:

```bash
# Install software
cd ..
pip install -e .

# Verify existing run metrics
cat runs/release_20260113_225750/verification_20260113_225750/manifest.json | python3 -m json.tool

# Or regenerate from scratch
rhdiag bundle --seed 42 --cell-count 64 --internal-dim 96
```

## License

This paper and associated software are provided for research and educational purposes.
