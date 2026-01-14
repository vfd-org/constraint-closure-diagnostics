# Safety and Narrative Boundary

## D5: What This Tool Does and Does Not Assert

---

## Official Disclaimer (Include in README)

```
## Important Notice

This software is a **diagnostic and visualization framework** for exploring
constraint-based approaches to the Riemann Hypothesis. It does not claim,
demonstrate, or provide a proof of RH.

Specifically:
- The "VFD invariants" are properties of a specific mathematical construction,
  not properties of the Riemann zeta function.
- The "Bridge Axiom" connecting VFD data to classical zeta zeros is an
  unverified hypothesis, not a proven theorem.
- All "zero projections" are numerical artifacts of the projection map,
  not computed zeros of zeta(s).
- Passing all internal checks does NOT imply RH is true.

This tool is intended for:
- Educational exploration of constraint-satisfaction approaches
- Visualization of mathematical structures
- Reproducible numerical experiments
- Testing falsifiability of proposed correspondences

This tool is NOT intended for:
- Claiming progress toward proving RH
- Publication as evidence for or against RH
- Marketing or promotional purposes
```

---

## For Papers/Publications

```
\subsection*{Scope and Limitations}

The computational framework presented here is a diagnostic tool for
exploring constraint-satisfaction approaches related to the Riemann
Hypothesis. We emphasize the following limitations:

\begin{enumerate}
\item \textbf{No proof is claimed.} The software verifies internal
consistency of a mathematical construction (termed ``VFD''), not
properties of the Riemann zeta function itself.

\item \textbf{The Bridge Axiom is unverified.} The proposed correspondence
between VFD spectral data and zeta zero heights is a conjecture. If
this correspondence fails (as tested by the BN negation modes), only
the interpretation collapses; the internal VFD structure remains valid.

\item \textbf{Numerical evidence is not proof.} Agreement between
projected values and reference zeros up to some tolerance does not
constitute mathematical proof.

\item \textbf{Falsifiability is the goal.} The primary purpose of the
framework is to enable systematic testing and potential falsification
of proposed correspondences, not to accumulate confirming evidence.
\end{enumerate}
```

---

## What the Tool DOES

1. **Constructs a mathematical object** (the VFD space with operators T, S, K)
2. **Verifies internal consistency** (T^12 = I, Weyl relation, projector identities, kernel properties D1-D5)
3. **Generates derived quantities** (internal primes, stability coefficients, eigenvalues)
4. **Projects VFD data to "zero-like" points** (via the Bridge Axiom)
5. **Compares projections to reference data** (zeta zeros from mpmath)
6. **Tests falsifiability** (BA vs BN modes)
7. **Exports reproducible artifacts** (configs, manifests, datasets)

---

## What the Tool Does NOT

1. **Does NOT prove RH**
2. **Does NOT compute actual zeros of zeta(s)**
3. **Does NOT verify the Bridge Axiom**
4. **Does NOT establish any theorem about zeta**
5. **Does NOT provide evidence for RH** (numerical agreement is not evidence)
6. **Does NOT claim the VFD construction is unique or canonical**

---

## Terminology Guidance

| Avoid | Use Instead |
|-------|-------------|
| "proves RH" | "verifies VFD internal consistency" |
| "computes zeros" | "projects eigenvalues to zero-like points" |
| "demonstrates RH" | "tests correspondence hypothesis" |
| "confirms" | "is consistent with" |
| "breakthrough" | "diagnostic framework" |
| "evidence for RH" | "internal consistency check" |

---

## Falsifiability Statement

The Bridge Axiom is designed to be falsifiable:

> If the BN (Bridge Negation) modes produce metrics equal to or better
> than the BA (Bridge Axiom) mode, then the proposed correspondence is
> likely spurious (curve-fitting rather than genuine structure).

The dashboard includes explicit falsification tests:
- BN1: Wrong spectral ordering (shuffles eigenvalues)
- BN2: Wrong scale factor (incorrect scaling)
- BN3: Wrong coordinate (incorrect β offset)

A genuine correspondence should show:
- BA metrics significantly better than all BN metrics
- BN1: Ordering perturbation should drop rank correlation (observed: ρ drops from 0.9997 to 0.008)
- BN2: Scale perturbation should increase RMSE (observed: 1098 → 2395)
- BN3: Coordinate perturbation should increase β deviation (observed: 0 → 0.2)

The bridge implementation is included for reproducibility; we treat it as a
testable hypothesis and evaluate it via falsification modes rather than
presenting it as a derived theorem.

---

## Short Version (for Headers/Footers)

> This is a diagnostic and visualization framework. It does not claim a proof of RH.

---

## Recommended README Section

```markdown
## Disclaimer

This software is a **diagnostic tool**, not a proof of the Riemann Hypothesis.

- VFD invariants are internal consistency checks, not properties of zeta.
- The Bridge Axiom is a testable hypothesis, not a proven theorem.
- Numerical agreement does not constitute mathematical proof.
- The primary purpose is falsifiability testing, not confirmation.

See `docs/SAFETY_BOUNDARY.md` for detailed scope and limitations.
```
