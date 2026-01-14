"""
Bridge Negations (BN1-BN3): Falsification Controls.

These are deliberate modifications to the Bridge Axiom that SHOULD break
the match between VFD projections and classical reference data.

If BN modes don't produce worse metrics than BA mode, the bridge is not
genuinely connecting VFD to classical zeta - it's just curve fitting.

BN1: Wrong spectral branch identification
BN2: Wrong scaling law
BN3: Wrong self-dual coordinate mapping
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

from .bridge_axiom import BridgeAxiom, BridgeMode, BridgeParameters


@dataclass
class NegationResult:
    """Result of applying a bridge negation."""
    mode: BridgeMode
    original_values: NDArray
    modified_values: NDArray
    modification_description: str


class BN1_Negation:
    """
    BN1: Wrong spectral branch identification.

    Deliberately scrambles the spectral ordering so that VFD eigenvalues
    are matched to the wrong zeta zeros.

    This tests whether the ordering matters.
    """

    def __init__(self, seed: int = 12345):
        """
        Initialize BN1 negation.

        Args:
            seed: Random seed for scrambling
        """
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def apply(
        self,
        vfd_eigenvalues: NDArray,
        mode: str = "shuffle"
    ) -> NegationResult:
        """
        Apply BN1 modification to eigenvalues.

        Args:
            vfd_eigenvalues: Original VFD eigenvalues
            mode: Modification mode ("shuffle", "reverse", "random_permute")

        Returns:
            NegationResult with modified values
        """
        original = vfd_eigenvalues.copy()

        if mode == "shuffle":
            # Complete random shuffle
            modified = original.copy()
            self.rng.shuffle(modified)
            description = "Random shuffle of eigenvalue ordering"

        elif mode == "reverse":
            # Reverse the ordering
            modified = original[::-1]
            description = "Reversed eigenvalue ordering"

        elif mode == "random_permute":
            # Random permutation with partial structure preservation
            n = len(original)
            indices = np.arange(n)
            # Swap pairs randomly
            for i in range(0, n-1, 2):
                if self.rng.random() > 0.5:
                    indices[i], indices[i+1] = indices[i+1], indices[i]
            modified = original[indices]
            description = "Partial random permutation of eigenvalues"

        else:
            modified = original.copy()
            description = "No modification (unknown mode)"

        return NegationResult(
            mode=BridgeMode.BN1,
            original_values=original,
            modified_values=modified,
            modification_description=description
        )

    def create_bridge(self) -> BridgeAxiom:
        """Create a Bridge with BN1 mode active."""
        return BridgeAxiom(mode=BridgeMode.BN1)


class BN2_Negation:
    """
    BN2: Wrong scaling law.

    Applies incorrect scale factors to the projection, breaking the
    quantitative match with reference zeros.

    This tests whether the scale is meaningful.
    """

    def __init__(self, scale_factor: float = 0.1):
        """
        Initialize BN2 negation.

        Args:
            scale_factor: Wrong scale to apply (default 0.1 = 10x too small)
        """
        self.scale_factor = scale_factor

    def apply(
        self,
        vfd_data: NDArray,
        mode: str = "wrong_scale"
    ) -> NegationResult:
        """
        Apply BN2 modification to data.

        Args:
            vfd_data: Original VFD data
            mode: Modification mode ("wrong_scale", "inverted", "random_scale")

        Returns:
            NegationResult with modified values
        """
        original = vfd_data.copy()

        if mode == "wrong_scale":
            # Apply wrong scale factor
            modified = original * self.scale_factor
            description = f"Applied wrong scale factor {self.scale_factor}"

        elif mode == "inverted":
            # Invert the scale (multiply by 1/correct)
            modified = 1.0 / (original + 1e-10)
            description = "Inverted scaling law"

        elif mode == "random_scale":
            # Apply random scales to each value
            rng = np.random.default_rng(99999)
            scales = rng.uniform(0.1, 10.0, size=len(original))
            modified = original * scales
            description = "Random scaling applied to each value"

        else:
            modified = original.copy()
            description = "No modification (unknown mode)"

        return NegationResult(
            mode=BridgeMode.BN2,
            original_values=original,
            modified_values=modified,
            modification_description=description
        )

    def create_bridge(self) -> BridgeAxiom:
        """Create a Bridge with BN2 mode and wrong scale."""
        params = BridgeParameters(spectral_scale=self.scale_factor)
        return BridgeAxiom(mode=BridgeMode.BN2, params=params)


class BN3_Negation:
    """
    BN3: Wrong self-dual coordinate mapping.

    Maps VFD self-dual manifold to a line other than Re(s) = 1/2.

    This tests whether the specific self-dual identification matters.
    """

    def __init__(self, offset: float = 0.7):
        """
        Initialize BN3 negation.

        Args:
            offset: Wrong self-dual offset (default 0.7 instead of 0.5)
        """
        self.offset = offset

    def apply(
        self,
        self_dual_coords: NDArray,
        mode: str = "shift"
    ) -> NegationResult:
        """
        Apply BN3 modification to self-dual coordinates.

        Args:
            self_dual_coords: Original VFD self-dual coordinates
            mode: Modification mode ("shift", "mirror", "random")

        Returns:
            NegationResult with modified values
        """
        original = self_dual_coords.copy()

        if mode == "shift":
            # Shift away from 0.5
            modified = original + (self.offset - 0.5)
            description = f"Shifted self-dual coordinate to {self.offset}"

        elif mode == "mirror":
            # Mirror around wrong axis
            modified = 1.0 - original
            description = "Mirrored self-dual coordinates around s=1/2"

        elif mode == "random":
            # Random offset for each coordinate
            rng = np.random.default_rng(77777)
            offsets = rng.uniform(-0.3, 0.3, size=len(original))
            modified = original + offsets
            description = "Random offsets applied to self-dual coordinates"

        else:
            modified = original.copy()
            description = "No modification (unknown mode)"

        return NegationResult(
            mode=BridgeMode.BN3,
            original_values=original,
            modified_values=modified,
            modification_description=description
        )

    def create_bridge(self) -> BridgeAxiom:
        """Create a Bridge with BN3 mode and wrong offset."""
        params = BridgeParameters(self_dual_offset=self.offset)
        return BridgeAxiom(mode=BridgeMode.BN3, params=params)


def apply_all_negations(
    vfd_eigenvalues: NDArray,
    vfd_stability: NDArray,
    vfd_self_dual: NDArray,
    seed: int = 42
) -> Dict[str, NegationResult]:
    """
    Apply all three negations for comprehensive falsification testing.

    Args:
        vfd_eigenvalues: VFD eigenvalues
        vfd_stability: VFD stability coefficients
        vfd_self_dual: VFD self-dual coordinates
        seed: Random seed

    Returns:
        Dictionary of negation results
    """
    bn1 = BN1_Negation(seed=seed)
    bn2 = BN2_Negation(scale_factor=0.1)
    bn3 = BN3_Negation(offset=0.7)

    return {
        "BN1": bn1.apply(vfd_eigenvalues, mode="shuffle"),
        "BN2": bn2.apply(vfd_stability, mode="wrong_scale"),
        "BN3": bn3.apply(vfd_self_dual, mode="shift"),
    }


def compute_negation_metrics(
    ba_metrics: Dict[str, float],
    bn_metrics: Dict[str, float]
) -> Dict[str, Any]:
    """
    Compare BA metrics with BN metrics to verify falsifiability.

    Good bridge: BN metrics should be significantly worse than BA metrics.

    Args:
        ba_metrics: Metrics under Bridge Axiom
        bn_metrics: Metrics under Bridge Negation

    Returns:
        Comparison results
    """
    results = {}

    for key in ["rmse", "mae", "max_error"]:
        if key in ba_metrics and key in bn_metrics:
            ba_val = ba_metrics[key]
            bn_val = bn_metrics[key]

            if ba_val > 0:
                ratio = bn_val / ba_val
            else:
                ratio = float('inf') if bn_val > 0 else 1.0

            results[f"{key}_ratio"] = ratio
            results[f"{key}_delta"] = bn_val - ba_val

    # Falsification is successful if BN is significantly worse
    min_ratio = min(
        results.get("rmse_ratio", 1.0),
        results.get("mae_ratio", 1.0)
    )

    results["falsification_successful"] = min_ratio >= 2.0
    results["falsification_ratio"] = min_ratio

    return results
