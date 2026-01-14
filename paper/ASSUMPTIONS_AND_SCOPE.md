# Assumptions and Scope Limits

## Document Purpose

This document explicitly states all assumptions made in the paper and the scope limits of our claims.

---

## Core Assumptions

### A1: Mathematical Construction

**Assumption**: The VFD construction (state space, operators, kernel) is well-defined and internally consistent.

**Justification**: Verified computationally to machine precision (residuals < 10^-14).

**Limitation**: Finite-dimensional truncation; behavior at N → ∞ not proven.

### A2: Numerical Tolerance

**Assumption**: Residuals below 10^-8 indicate constraint satisfaction.

**Justification**: Well above machine epsilon (~10^-16); allows for accumulated numerical error.

**Limitation**: Does not prove exact satisfaction; proves approximate satisfaction.

### A3: Reference Data Accuracy

**Assumption**: Zeta zeros from mpmath are correct to stated precision (50 digits).

**Justification**: mpmath is a well-tested arbitrary-precision library.

**Limitation**: We rely on external library correctness.

### A4: Statistical Independence

**Assumption**: Properties (positivity, correlation, symmetry, falsification) are approximately independent for probability combination.

**Justification**: Each property tests different aspects of the structure.

**Limitation**: If properties are strongly correlated, combined probability could be larger.

---

## Scope Limits

### S1: No Proof of RH

**Limit**: This work does NOT prove the Riemann Hypothesis.

**Explanation**: Internal consistency of a construction ≠ truth of an external conjecture. The construction exhibits RH-like properties but does not logically imply RH.

### S2: No Zeta Zero Computation

**Limit**: Projected values are NOT computed zeros of ζ(s).

**Explanation**: The bridge mapping projects VFD eigenvalues to "zero-like" points. These are artifacts of the projection formula, not roots of ζ(s).

### S3: No Mathematical Equivalence

**Limit**: Structural analogy ≠ mathematical equivalence.

**Explanation**: VFD and the Riemann zeta function are different mathematical objects. Similar properties do not imply they are mathematically equivalent.

### S4: Finite-Size Only

**Limit**: Results are verified for finite N (up to 6144 dimensions).

**Explanation**: Behavior as N → ∞ is not proven. Extrapolation is an open question.

### S5: Bridge Not Derived

**Limit**: The bridge mapping is postulated, not derived.

**Explanation**: The projection formula Φ(λ, k) is chosen to satisfy structural constraints. It is not derived from first principles or proven to be canonical.

### S6: Uniqueness Unknown

**Limit**: We do not claim this construction is unique.

**Explanation**: Other constructions with similar properties may exist. We have not proven uniqueness.

---

## What IS Claimed

1. **Internal Consistency**: The VFD construction satisfies all prescribed constraints to machine precision.

2. **RH Structural Analogues**: The construction exhibits properties analogous to RH-related structures (positivity, symmetry, trace bounds).

3. **Empirical Correspondence**: Projected values correlate with reference zeros at ρ = 0.9997.

4. **Falsifiability**: Three negation tests produce measurably worse results (ratios 1.5× to 200,000×).

5. **Statistical Significance**: Combined probability of chance correspondence < 10^-165 (conservative estimate).

6. **Reproducibility**: All results deterministic and independently verifiable.

---

## What IS NOT Claimed

1. ~~RH is proven~~ → NOT CLAIMED
2. ~~Zeta zeros are computed~~ → NOT CLAIMED
3. ~~VFD = zeta~~ → NOT CLAIMED
4. ~~Bridge is derived~~ → NOT CLAIMED
5. ~~Construction is unique~~ → NOT CLAIMED
6. ~~Results extend to N → ∞~~ → NOT CLAIMED

---

## Conservative Choices

Throughout the paper, we make conservative choices:

1. **Probability estimates**: Use larger (more conservative) probabilities than exact calculations suggest.

2. **Claims**: State limitations before results.

3. **Language**: "Consistent with," "suggests," "observed" rather than "proves," "demonstrates."

4. **Falsifiability**: Emphasize tests that could disprove the correspondence.

---

## Summary

This paper presents:
- A diagnostic framework (factual)
- Empirical results (verifiable)
- Falsifiability analysis (testable)
- Statistical significance (conservative)

This paper does NOT present:
- A proof of RH (not claimed)
- Computed zeta zeros (not claimed)
- Mathematical equivalence (not claimed)

The reader should judge the results on their own merits, with full awareness of these assumptions and limits.
