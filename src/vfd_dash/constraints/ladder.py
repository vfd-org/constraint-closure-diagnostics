"""
Closure Ladder: Hierarchical constraint verification with gating.

The closure ladder defines 5 levels (L0-L4) of increasingly stringent constraints.
Each level must pass before proceeding to the next (gating).

Levels:
- L0: Baseline (structural validity)
- L1: Explicit formula consistency (torsion, Weyl)
- L2: Symmetry constraints (projectors, self-dual)
- L3: Positivity constraints (kernel nonnegativity, quadratic forms)
- L4: Trace/moment constraints (spectral moments)
"""

from enum import IntEnum
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from dataclasses import dataclass, field

if TYPE_CHECKING:
    from ..state import DiagnosticState


class ClosureLevel(IntEnum):
    """Closure ladder levels."""
    L0 = 0  # Baseline: structural validity
    L1 = 1  # Explicit formula consistency
    L2 = 2  # Symmetry constraints
    L3 = 3  # Positivity constraints
    L4 = 4  # Trace/moment constraints

    @classmethod
    def from_string(cls, s: str) -> "ClosureLevel":
        """Parse level from string like 'L2' or '2'."""
        s = s.upper().strip()
        if s.startswith("L"):
            s = s[1:]
        return cls(int(s))

    @classmethod
    def parse_range(cls, range_str: str) -> List["ClosureLevel"]:
        """Parse range like 'L0..L4' or 'L0-L2'."""
        range_str = range_str.upper().replace("-", "..")
        if ".." in range_str:
            start, end = range_str.split("..")
            start_level = cls.from_string(start)
            end_level = cls.from_string(end)
            return [cls(i) for i in range(start_level, end_level + 1)]
        else:
            return [cls.from_string(range_str)]


@dataclass
class LevelResult:
    """Result of checking constraints at a single closure level."""
    level: ClosureLevel
    satisfied: bool
    residuals: Dict[str, float]  # constraint_name -> residual value
    constraints_checked: List[str]
    gating_passed: bool
    total_residual: float = 0.0
    family_residuals: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self):
        """Compute total residual from individual residuals."""
        if self.residuals and self.total_residual == 0.0:
            self.total_residual = sum(abs(r) for r in self.residuals.values())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "level": f"L{self.level.value}",
            "satisfied": self.satisfied,
            "residuals": self.residuals,
            "constraints_checked": self.constraints_checked,
            "gating_passed": self.gating_passed,
            "total_residual": self.total_residual,
            "family_residuals": self.family_residuals,
            "notes": self.notes,
        }


@dataclass
class LadderResult:
    """Result of running the full closure ladder."""
    max_level_checked: ClosureLevel
    max_level_passed: Optional[ClosureLevel]
    level_results: Dict[ClosureLevel, LevelResult]
    gating_stop_reason: Optional[str] = None
    all_passed: bool = False

    def __post_init__(self):
        """Compute overall pass status."""
        if self.level_results:
            self.all_passed = all(r.satisfied for r in self.level_results.values())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "max_level_checked": f"L{self.max_level_checked.value}",
            "max_level_passed": f"L{self.max_level_passed.value}" if self.max_level_passed is not None else None,
            "gating_stop_reason": self.gating_stop_reason,
            "all_passed": self.all_passed,
            "residuals_per_level": {
                f"L{level.value}": result.to_dict()
                for level, result in self.level_results.items()
            },
        }

    def get_residual_ladder(self) -> List[tuple]:
        """Get (level, total_residual, passed) for plotting."""
        return [
            (level, result.total_residual, result.satisfied)
            for level, result in sorted(self.level_results.items())
        ]


class ClosureLadder:
    """
    Manages hierarchical constraint verification with gating.

    The ladder runs levels sequentially. If a level fails, subsequent
    levels are not checked (gating).
    """

    # Tolerance for considering a residual as "passed"
    DEFAULT_TOLERANCE = 1e-8

    def __init__(self, tolerance: float = None):
        """
        Initialize closure ladder.

        Args:
            tolerance: Residual tolerance for pass/fail (default 1e-8)
        """
        self.tolerance = tolerance or self.DEFAULT_TOLERANCE
        self._results: Dict[ClosureLevel, LevelResult] = {}

    def check_level(
        self,
        level: ClosureLevel,
        state: "DiagnosticState"
    ) -> LevelResult:
        """
        Check constraints at a given level.

        Args:
            level: Closure level to check
            state: Diagnostic state with computed data

        Returns:
            LevelResult with pass/fail and residuals
        """
        from .families import get_families_for_level

        families = get_families_for_level(level)
        all_residuals: Dict[str, float] = {}
        family_residuals: Dict[str, float] = {}
        constraints_checked: List[str] = []

        for family in families:
            family_result = family.evaluate(state)
            family_total = sum(abs(r) for r in family_result.values())
            family_residuals[family.name] = family_total

            for name, residual in family_result.items():
                full_name = f"{family.name}.{name}"
                all_residuals[full_name] = residual
                constraints_checked.append(full_name)

        total_residual = sum(abs(r) for r in all_residuals.values())
        satisfied = total_residual < self.tolerance

        return LevelResult(
            level=level,
            satisfied=satisfied,
            residuals=all_residuals,
            constraints_checked=constraints_checked,
            gating_passed=satisfied,
            total_residual=total_residual,
            family_residuals=family_residuals,
            notes=f"Checked {len(constraints_checked)} constraints across {len(families)} families"
        )

    def run(
        self,
        state: "DiagnosticState",
        max_level: ClosureLevel = ClosureLevel.L4,
        gate: bool = True
    ) -> LadderResult:
        """
        Run the closure ladder up to max_level.

        Args:
            state: Diagnostic state with computed data
            max_level: Maximum level to check
            gate: If True, stop at first failed level

        Returns:
            LadderResult with all level results
        """
        self._results = {}
        max_passed: Optional[ClosureLevel] = None
        stop_reason: Optional[str] = None

        for level in ClosureLevel:
            if level > max_level:
                break

            result = self.check_level(level, state)
            self._results[level] = result

            if result.satisfied:
                max_passed = level
            else:
                if gate:
                    stop_reason = f"Level L{level.value} failed with residual {result.total_residual:.2e}"
                    break

        return LadderResult(
            max_level_checked=max(self._results.keys()) if self._results else ClosureLevel.L0,
            max_level_passed=max_passed,
            level_results=self._results.copy(),
            gating_stop_reason=stop_reason,
        )

    def get_results(self) -> Dict[ClosureLevel, LevelResult]:
        """Get cached results from last run."""
        return self._results.copy()

    @property
    def tolerance_dict(self) -> Dict[str, float]:
        """Get tolerance as dict for manifest."""
        return {
            "closure_ladder": self.tolerance,
        }
