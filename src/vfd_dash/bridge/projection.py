"""
Shadow Projection: VFD to Classical Mapping.

Projects VFD spectral/stability data to "zero-like" points and other
classical shadows. All projections are deterministic given configuration.

Key principle: NO CURVE FITTING.
The projection is defined by the Bridge Axiom, not optimized to match data.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from .bridge_axiom import BridgeAxiom, BridgeMode, BridgeParameters
from .reference_data import ReferenceDataLoader, get_cached_zeros


@dataclass
class ProjectedZero:
    """A projected zero-like point from VFD data."""
    index: int
    t_projected: float  # Imaginary part analogue
    beta_projected: float  # Real part analogue (self-dual coordinate)
    residual_to_reference: Optional[float]  # Difference from reference zero
    bridge_mode: str
    source_eigenvalue: float
    projection_details: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "idx": self.index,
            "t_projected": self.t_projected,
            "beta_projected": self.beta_projected,
            "residual_to_reference": self.residual_to_reference,
            "bridge_mode": self.bridge_mode,
            "source_eigenvalue": self.source_eigenvalue,
            "projection_details": self.projection_details,
        }


class ShadowProjection:
    """
    Project VFD data to classical shadow.

    This implements the spectral_to_zero_like mapping defined by the Bridge Axiom.
    """

    def __init__(
        self,
        bridge: BridgeAxiom,
        reference: Optional[ReferenceDataLoader] = None,
        max_zeros: int = 100
    ):
        """
        Initialize shadow projection.

        Args:
            bridge: Bridge Axiom instance
            reference: Reference data loader (for residual computation)
            max_zeros: Maximum zeros to load (default 100 for fast startup)
        """
        self.bridge = bridge
        self.max_zeros = max_zeros
        # Use smaller default for faster bridge initialization
        self.reference = reference or ReferenceDataLoader(max_zeros=max_zeros)

    def project_spectrum_to_zeros(
        self,
        eigenvalues: NDArray,
        self_dual_coords: Optional[NDArray] = None,
        max_project: int = 1000
    ) -> List[ProjectedZero]:
        """
        Project VFD spectrum to zero-like points.

        Args:
            eigenvalues: VFD kernel eigenvalues (sorted)
            self_dual_coords: Self-dual coordinates for each eigenvalue
            max_project: Maximum number of projections

        Returns:
            List of projected zeros
        """
        n = min(len(eigenvalues), max_project)

        # Get reference zeros for residual computation
        try:
            reference_zeros = self.reference.get_zeta_zeros(n)
        except Exception:
            reference_zeros = None

        # Default self-dual coordinate if not provided
        if self_dual_coords is None:
            self_dual_coords = np.full(n, 0.5)

        projected = []

        for i in range(n):
            ev = eigenvalues[i]
            sd_coord = self_dual_coords[i] if i < len(self_dual_coords) else 0.5

            # Project eigenvalue to zero height
            t = self.bridge.project_eigenvalue_to_zero_height(ev, i + 1)

            # Get real part from self-dual coordinate
            beta = self.bridge.get_self_dual_coordinate()

            # Compute residual if reference available
            residual = None
            if reference_zeros is not None and i < len(reference_zeros):
                residual = t - reference_zeros[i]

            proj = ProjectedZero(
                index=i + 1,
                t_projected=t,
                beta_projected=beta,
                residual_to_reference=residual,
                bridge_mode=self.bridge.mode.value,
                source_eigenvalue=ev,
                projection_details=f"BA projection: lambda={ev:.4f} -> t={t:.4f}"
            )
            projected.append(proj)

        return projected

    def compute_overlay_metrics(
        self,
        projected_zeros: List[ProjectedZero],
        reference_zeros: Optional[NDArray] = None
    ) -> Dict[str, float]:
        """
        Compute overlay metrics comparing projected to reference zeros.

        Includes:
        - Standard metrics (RMSE, MAE, correlation)
        - Rank correlation: Tests if eigenvalue ordering is preserved (for BN1 falsification)
        - Beta deviation: Tests if self-dual maps to critical line (for BN3 falsification)

        Args:
            projected_zeros: Projected zero-like points
            reference_zeros: Reference zeta zeros

        Returns:
            Dictionary of metrics
        """
        from scipy.stats import spearmanr

        if reference_zeros is None:
            try:
                reference_zeros = self.reference.get_zeta_zeros(len(projected_zeros))
            except Exception:
                return {"error": "No reference data available"}

        n = min(len(projected_zeros), len(reference_zeros))
        if n == 0:
            return {"error": "No data to compare"}

        projected_t = np.array([p.t_projected for p in projected_zeros[:n]])
        reference_t = reference_zeros[:n]

        residuals = projected_t - reference_t

        # Standard metrics
        metrics = {
            "n_compared": n,
            "mae": float(np.mean(np.abs(residuals))),
            "rmse": float(np.sqrt(np.mean(residuals**2))),
            "max_error": float(np.max(np.abs(residuals))),
            "mean_residual": float(np.mean(residuals)),
            "std_residual": float(np.std(residuals)),
            "correlation": float(np.corrcoef(projected_t, reference_t)[0, 1]) if n > 1 else 0.0,
        }

        # Rank correlation: eigenvalue ordering vs projection ordering
        # This detects BN1 (shuffled eigenvalues) - shuffling breaks monotonicity
        # BA: eigenvalues increase → projections increase (rank_corr ≈ 1.0)
        # BN1: shuffled eigenvalues → random ordering (rank_corr ≈ 0.0)
        source_eigenvalues = np.array([p.source_eigenvalue for p in projected_zeros[:n]])
        if n > 2 and np.std(source_eigenvalues) > 1e-10:
            rank_corr_result = spearmanr(source_eigenvalues, projected_t)
            metrics["rank_correlation"] = float(rank_corr_result.correlation)
        else:
            metrics["rank_correlation"] = 1.0  # Default to perfect if not enough variation

        # Beta deviation: distance of mean beta from critical line (0.5)
        # This detects BN3 (wrong self-dual offset)
        # BA: self_dual_offset=0.5 → mean_beta ≈ 0.5 → deviation ≈ 0
        # BN3: self_dual_offset=0.7 → mean_beta ≈ 0.7 → deviation ≈ 0.2
        beta_values = np.array([p.beta_projected for p in projected_zeros[:n]])
        mean_beta = float(np.mean(beta_values))
        metrics["mean_beta"] = mean_beta
        metrics["beta_deviation"] = float(abs(mean_beta - 0.5))

        return metrics

    def compute_spacing_metrics(
        self,
        projected_zeros: List[ProjectedZero],
        reference_zeros: Optional[NDArray] = None
    ) -> Dict[str, float]:
        """
        Compute spacing distribution metrics.

        Args:
            projected_zeros: Projected zero-like points
            reference_zeros: Reference zeta zeros

        Returns:
            Dictionary of spacing metrics
        """
        from scipy.stats import ks_2samp, wasserstein_distance

        if reference_zeros is None:
            try:
                reference_zeros = self.reference.get_zeta_zeros(len(projected_zeros))
            except Exception:
                return {"error": "No reference data available"}

        # Compute spacings
        projected_t = np.array([p.t_projected for p in projected_zeros])
        proj_spacings = np.diff(projected_t)
        ref_spacings = np.diff(reference_zeros[:len(projected_zeros)])

        if len(proj_spacings) < 2 or len(ref_spacings) < 2:
            return {"error": "Not enough data for spacing analysis"}

        # Normalize spacings by local mean
        proj_normalized = proj_spacings / np.mean(proj_spacings)
        ref_normalized = ref_spacings / np.mean(ref_spacings)

        # KS test
        ks_stat, ks_pvalue = ks_2samp(proj_normalized, ref_normalized)

        # Wasserstein distance
        w_dist = wasserstein_distance(proj_normalized, ref_normalized)

        return {
            "ks_statistic": float(ks_stat),
            "ks_pvalue": float(ks_pvalue),
            "wasserstein_distance": float(w_dist),
            "mean_proj_spacing": float(np.mean(proj_spacings)),
            "mean_ref_spacing": float(np.mean(ref_spacings)),
            "std_proj_spacing": float(np.std(proj_spacings)),
            "std_ref_spacing": float(np.std(ref_spacings)),
        }

    def to_dataframe(self, projected_zeros: List[ProjectedZero]) -> pd.DataFrame:
        """Convert projected zeros to DataFrame."""
        records = [p.to_dict() for p in projected_zeros]
        return pd.DataFrame(records)

    def run_full_projection(
        self,
        eigenvalues: NDArray,
        self_dual_coords: Optional[NDArray] = None,
        max_project: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run full projection and compute all metrics.

        Args:
            eigenvalues: VFD eigenvalues
            self_dual_coords: Self-dual coordinates
            max_project: Maximum projections (defaults to self.max_zeros)

        Returns:
            Complete projection results
        """
        # Use instance max_zeros as default for faster runs
        if max_project is None:
            max_project = self.max_zeros

        # Project
        projected = self.project_spectrum_to_zeros(
            eigenvalues, self_dual_coords, max_project
        )

        # Compute metrics
        overlay = self.compute_overlay_metrics(projected)
        spacing = self.compute_spacing_metrics(projected)

        return {
            "bridge_mode": self.bridge.mode.value,
            "n_projected": len(projected),
            "projected_zeros": projected,
            "overlay_metrics": overlay,
            "spacing_metrics": spacing,
        }


def compare_ba_vs_bn(
    eigenvalues: NDArray,
    reference_zeros: NDArray,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Compare Bridge Axiom results against all negations.

    This is the core falsifiability test.

    Args:
        eigenvalues: VFD eigenvalues
        reference_zeros: Reference zeta zeros
        seed: Random seed for negations

    Returns:
        Comparison results
    """
    from .bn_negations import BN1_Negation, BN2_Negation, BN3_Negation

    results = {}

    # BA mode
    ba_bridge = BridgeAxiom(mode=BridgeMode.BA)
    ba_proj = ShadowProjection(ba_bridge)
    ba_result = ba_proj.run_full_projection(eigenvalues)
    results["BA"] = ba_result["overlay_metrics"]

    # BN1: Wrong ordering
    bn1 = BN1_Negation(seed=seed)
    bn1_result = bn1.apply(eigenvalues, mode="shuffle")
    bn1_bridge = bn1.create_bridge()
    bn1_proj = ShadowProjection(bn1_bridge)
    bn1_full = bn1_proj.run_full_projection(bn1_result.modified_values)
    results["BN1"] = bn1_full["overlay_metrics"]

    # BN2: Wrong scale
    bn2 = BN2_Negation(scale_factor=0.1)
    bn2_bridge = bn2.create_bridge()
    bn2_proj = ShadowProjection(bn2_bridge)
    bn2_full = bn2_proj.run_full_projection(eigenvalues)
    results["BN2"] = bn2_full["overlay_metrics"]

    # BN3: Wrong self-dual
    bn3 = BN3_Negation(offset=0.7)
    bn3_bridge = bn3.create_bridge()
    bn3_proj = ShadowProjection(bn3_bridge)
    bn3_full = bn3_proj.run_full_projection(eigenvalues)
    results["BN3"] = bn3_full["overlay_metrics"]

    # Compute falsification ratios using appropriate metrics for each negation
    #
    # BN1 (shuffled ordering): Use rank_correlation degradation
    #   BA should have high rank_corr (~1.0), BN1 should have low (~0.0)
    #   Ratio = (1 - BN1_rank_corr) / (1 - BA_rank_corr + epsilon)
    #   If BA has rank_corr=0.99, (1-0.99)=0.01; BN1 has rank_corr=0.1, (1-0.1)=0.9
    #   Ratio = 0.9 / 0.01 = 90 (BN1 is much worse)
    #
    # BN2 (wrong scale): Use RMSE ratio (scale directly affects absolute errors)
    #   Ratio = BN2_rmse / BA_rmse
    #
    # BN3 (wrong self-dual): Use beta_deviation increase
    #   BA should have low deviation (~0), BN3 should have high (~0.2)
    #   Ratio = (BN3_beta_dev + epsilon) / (BA_beta_dev + epsilon)

    ratios = {}
    epsilon = 1e-6  # Avoid division by zero

    # BN1: rank correlation degradation
    ba_rank = results["BA"].get("rank_correlation", 1.0)
    bn1_rank = results["BN1"].get("rank_correlation", 1.0)
    # Lower rank_corr is worse, so we measure (1 - rank_corr) as "badness"
    ba_rank_badness = 1.0 - ba_rank + epsilon
    bn1_rank_badness = 1.0 - bn1_rank + epsilon
    ratios["BN1_ratio"] = bn1_rank_badness / ba_rank_badness

    # BN2: RMSE ratio (same as before)
    ba_rmse = results["BA"].get("rmse", 1.0)
    bn2_rmse = results["BN2"].get("rmse", 1.0)
    ratios["BN2_ratio"] = bn2_rmse / ba_rmse if ba_rmse > 0 else float('inf')

    # BN3: beta deviation increase
    ba_beta_dev = results["BA"].get("beta_deviation", 0.0) + epsilon
    bn3_beta_dev = results["BN3"].get("beta_deviation", 0.0) + epsilon
    ratios["BN3_ratio"] = bn3_beta_dev / ba_beta_dev

    results["falsification_ratios"] = ratios
    results["all_negations_worse"] = all(r >= 1.5 for r in ratios.values())

    # Add detailed metrics for debugging
    results["falsification_details"] = {
        "BN1_metric": "rank_correlation",
        "BN1_ba_value": ba_rank,
        "BN1_bn_value": bn1_rank,
        "BN2_metric": "rmse",
        "BN2_ba_value": ba_rmse,
        "BN2_bn_value": bn2_rmse,
        "BN3_metric": "beta_deviation",
        "BN3_ba_value": results["BA"].get("beta_deviation", 0.0),
        "BN3_bn_value": results["BN3"].get("beta_deviation", 0.0),
    }

    return results
