"""
Constraints package for RH Constraint-Diagnostic Demo.

Provides:
- Closure ladder (L0-L4) with gating
- Constraint families: EF, Symmetry, Positivity, Trace/Moment
"""

from .ladder import ClosureLevel, LevelResult, LadderResult, ClosureLadder
from .families import (
    ConstraintFamily,
    ExplicitFormulaFamily,
    SymmetryFamily,
    PositivityFamily,
    TraceMomentFamily,
    get_all_families,
)

__all__ = [
    "ClosureLevel",
    "LevelResult",
    "LadderResult",
    "ClosureLadder",
    "ConstraintFamily",
    "ExplicitFormulaFamily",
    "SymmetryFamily",
    "PositivityFamily",
    "TraceMomentFamily",
    "get_all_families",
]
