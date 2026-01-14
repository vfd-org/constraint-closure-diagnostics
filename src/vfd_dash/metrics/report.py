"""
Metrics Report Generation.

Produces comprehensive metrics.json for each run.
"""

import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from pathlib import Path

from ..core.hashing import compute_config_hash, compute_run_hash, get_git_commit_hash, get_package_versions


@dataclass
class InvariantResults:
    """Results of VFD invariant checks."""
    weyl_relation: bool = False
    weyl_error: float = 0.0
    torsion_order: bool = False
    torsion_error: float = 0.0
    projector_resolution: bool = False
    projector_resolution_error: float = 0.0
    projector_orthogonality: bool = False
    projector_orthogonality_error: float = 0.0
    kernel_D1_selfadjoint: bool = False
    kernel_D1_error: float = 0.0
    kernel_D2_torsion: bool = False
    kernel_D2_error: float = 0.0
    kernel_D3_nonnegative: bool = False
    kernel_D3_min: float = 0.0

    def all_passed(self) -> bool:
        return all([
            self.weyl_relation,
            self.torsion_order,
            self.projector_resolution,
            self.projector_orthogonality,
            self.kernel_D1_selfadjoint,
            self.kernel_D2_torsion,
            self.kernel_D3_nonnegative,
        ])


@dataclass
class BridgeMetrics:
    """Metrics for bridge validation."""
    mode: str = "OFF"
    overlay_mae: float = 0.0
    overlay_rmse: float = 0.0
    overlay_max_error: float = 0.0
    overlay_correlation: float = 0.0
    spacing_ks: float = 0.0
    spacing_wasserstein: float = 0.0


@dataclass
class FalsificationMetrics:
    """Metrics comparing BA vs BN modes."""
    bn1_rmse_ratio: float = 0.0
    bn2_rmse_ratio: float = 0.0
    bn3_rmse_ratio: float = 0.0
    falsification_successful: bool = False


@dataclass
class MetricsReport:
    """Complete metrics report for a run."""
    # Run identification
    run_hash: str = ""
    config_hash: str = ""
    timestamp: str = ""
    git_commit: Optional[str] = None
    package_versions: Dict[str, str] = field(default_factory=dict)

    # Configuration summary
    config_summary: Dict[str, Any] = field(default_factory=dict)

    # VFD invariants
    invariants: InvariantResults = field(default_factory=InvariantResults)

    # Prime statistics
    prime_count: int = 0
    prime_max_length: int = 0
    non_ufd_examples: int = 0

    # Stability statistics
    stability_probe_count: int = 0
    stability_min_Q: float = 0.0
    stability_all_nonnegative: bool = False
    kernel_absoluteness: bool = False

    # Bridge metrics (BA mode)
    bridge_ba: BridgeMetrics = field(default_factory=BridgeMetrics)

    # Falsification metrics
    falsification: FalsificationMetrics = field(default_factory=FalsificationMetrics)

    # Spectrum statistics
    spectrum_count: int = 0
    spectrum_min: float = 0.0
    spectrum_max: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = {
            "run_hash": self.run_hash,
            "config_hash": self.config_hash,
            "timestamp": self.timestamp,
            "git_commit": self.git_commit,
            "package_versions": self.package_versions,
            "config_summary": self.config_summary,
            "invariants": asdict(self.invariants),
            "invariants_all_passed": self.invariants.all_passed(),
            "prime_count": self.prime_count,
            "prime_max_length": self.prime_max_length,
            "non_ufd_examples": self.non_ufd_examples,
            "stability_probe_count": self.stability_probe_count,
            "stability_min_Q": self.stability_min_Q,
            "stability_all_nonnegative": self.stability_all_nonnegative,
            "kernel_absoluteness": self.kernel_absoluteness,
            "bridge_ba": asdict(self.bridge_ba),
            "falsification": asdict(self.falsification),
            "spectrum_count": self.spectrum_count,
            "spectrum_min": self.spectrum_min,
            "spectrum_max": self.spectrum_max,
        }
        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: str) -> None:
        """Save to JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_json(cls, json_str: str) -> "MetricsReport":
        """Load from JSON string."""
        data = json.loads(json_str)

        report = cls(
            run_hash=data.get("run_hash", ""),
            config_hash=data.get("config_hash", ""),
            timestamp=data.get("timestamp", ""),
            git_commit=data.get("git_commit"),
            package_versions=data.get("package_versions", {}),
            config_summary=data.get("config_summary", {}),
        )

        # Load invariants
        inv_data = data.get("invariants", {})
        report.invariants = InvariantResults(**inv_data)

        # Load other fields
        report.prime_count = data.get("prime_count", 0)
        report.prime_max_length = data.get("prime_max_length", 0)
        report.non_ufd_examples = data.get("non_ufd_examples", 0)
        report.stability_probe_count = data.get("stability_probe_count", 0)
        report.stability_min_Q = data.get("stability_min_Q", 0.0)
        report.stability_all_nonnegative = data.get("stability_all_nonnegative", False)
        report.kernel_absoluteness = data.get("kernel_absoluteness", False)

        # Load bridge metrics
        ba_data = data.get("bridge_ba", {})
        report.bridge_ba = BridgeMetrics(**ba_data)

        # Load falsification metrics
        fals_data = data.get("falsification", {})
        report.falsification = FalsificationMetrics(**fals_data)

        report.spectrum_count = data.get("spectrum_count", 0)
        report.spectrum_min = data.get("spectrum_min", 0.0)
        report.spectrum_max = data.get("spectrum_max", 0.0)

        return report


def generate_metrics_report(
    config: Any,
    invariant_results: Dict[str, Any],
    prime_results: Dict[str, Any],
    stability_results: Dict[str, Any],
    bridge_results: Dict[str, Any],
    falsification_results: Dict[str, Any],
    spectrum_results: Dict[str, Any],
) -> MetricsReport:
    """
    Generate complete metrics report from component results.

    Args:
        config: Run configuration
        invariant_results: VFD invariant check results
        prime_results: Prime generation results
        stability_results: Stability analysis results
        bridge_results: Bridge projection results
        falsification_results: BA vs BN comparison results
        spectrum_results: Spectrum computation results

    Returns:
        Complete MetricsReport
    """
    now = datetime.now()

    report = MetricsReport(
        run_hash=compute_run_hash(config, now),
        config_hash=compute_config_hash(config),
        timestamp=now.isoformat(),
        git_commit=get_git_commit_hash(),
        package_versions=get_package_versions(),
        config_summary=config.to_dict() if hasattr(config, "to_dict") else {},
    )

    # Invariants
    report.invariants = InvariantResults(
        weyl_relation=invariant_results.get("weyl_pass", False),
        weyl_error=invariant_results.get("weyl_error", 0.0),
        torsion_order=invariant_results.get("torsion_pass", False),
        torsion_error=invariant_results.get("torsion_error", 0.0),
        projector_resolution=invariant_results.get("projector_resolution_pass", False),
        projector_resolution_error=invariant_results.get("projector_resolution_error", 0.0),
        projector_orthogonality=invariant_results.get("projector_orthogonality_pass", False),
        projector_orthogonality_error=invariant_results.get("projector_orthogonality_error", 0.0),
        kernel_D1_selfadjoint=invariant_results.get("kernel_D1_pass", False),
        kernel_D1_error=invariant_results.get("kernel_D1_error", 0.0),
        kernel_D2_torsion=invariant_results.get("kernel_D2_pass", False),
        kernel_D2_error=invariant_results.get("kernel_D2_error", 0.0),
        kernel_D3_nonnegative=invariant_results.get("kernel_D3_pass", False),
        kernel_D3_min=invariant_results.get("kernel_D3_min", 0.0),
    )

    # Primes
    report.prime_count = prime_results.get("count", 0)
    report.prime_max_length = prime_results.get("max_length", 0)
    report.non_ufd_examples = prime_results.get("non_ufd_examples", 0)

    # Stability
    report.stability_probe_count = stability_results.get("probe_count", 0)
    report.stability_min_Q = stability_results.get("min_Q", 0.0)
    report.stability_all_nonnegative = stability_results.get("all_nonnegative", False)
    report.kernel_absoluteness = stability_results.get("kernel_absoluteness", False)

    # Bridge
    ba = bridge_results.get("BA", {})
    report.bridge_ba = BridgeMetrics(
        mode="BA",
        overlay_mae=ba.get("mae", 0.0),
        overlay_rmse=ba.get("rmse", 0.0),
        overlay_max_error=ba.get("max_error", 0.0),
        overlay_correlation=ba.get("correlation", 0.0),
        spacing_ks=bridge_results.get("spacing_ks", 0.0),
        spacing_wasserstein=bridge_results.get("spacing_wasserstein", 0.0),
    )

    # Falsification
    ratios = falsification_results.get("falsification_ratios", {})
    report.falsification = FalsificationMetrics(
        bn1_rmse_ratio=ratios.get("BN1_ratio", 0.0),
        bn2_rmse_ratio=ratios.get("BN2_ratio", 0.0),
        bn3_rmse_ratio=ratios.get("BN3_ratio", 0.0),
        falsification_successful=falsification_results.get("all_negations_worse", False),
    )

    # Spectrum
    report.spectrum_count = spectrum_results.get("count", 0)
    report.spectrum_min = spectrum_results.get("min", 0.0)
    report.spectrum_max = spectrum_results.get("max", 0.0)

    return report
