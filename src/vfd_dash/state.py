"""
Diagnostic State: Container for all computed data.

The DiagnosticState holds:
- Configuration
- VFD space and operators (T, S, K)
- Computed spectrum
- Stability metrics
- Bridge projection results (optional)
- Cached invariant checks

This provides a unified interface for constraint evaluation.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, TYPE_CHECKING, Union
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .core.config import RunConfig
    from .vfd.canonical import VFDSpace, TorsionOperator, ShiftOperator
    from .vfd.kernels import CanonicalKernel
    from .spectrum.backend import SpectralResult


@dataclass
class DiagnosticState:
    """
    Container for all diagnostic state data.

    Attributes:
        config: Run configuration
        space: VFD space
        T: Torsion operator
        S: Shift operator
        kernel: Canonical kernel
        spectrum: Computed spectrum (optional)
        stability_df: Stability coefficients DataFrame (optional)
        invariants: Cached invariant check results
        projection: Bridge projection results (optional)
    """
    config: "RunConfig"
    space: "VFDSpace"
    T: "TorsionOperator"
    S: "ShiftOperator"
    kernel: "CanonicalKernel"
    spectrum: Optional["SpectralResult"] = None
    stability_df: Optional[pd.DataFrame] = None
    invariants: Dict[str, Any] = field(default_factory=dict)
    projection: Dict[str, Any] = field(default_factory=dict)
    primes_df: Optional[pd.DataFrame] = None
    non_ufd_examples: list = field(default_factory=list)

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of state for logging."""
        return {
            "cell_count": self.space.cell_count,
            "internal_dim": self.space.internal_dim,
            "total_dim": self.space.total_dim,
            "has_spectrum": self.spectrum is not None,
            "has_stability": self.stability_df is not None,
            "has_projection": bool(self.projection),
            "invariants_count": len(self.invariants),
        }


def build_state(
    config: "RunConfig",
    compute_spectrum: bool = True,
    compute_stability: bool = True,
    compute_primes: bool = False,
    compute_projection: bool = False,
    seed: Optional[int] = None,
    perf_monitor: Optional[Any] = None,
) -> DiagnosticState:
    """
    Build a DiagnosticState from configuration.

    Runs the computation pipeline and returns a state object
    suitable for constraint evaluation.

    Args:
        config: Run configuration
        compute_spectrum: Whether to compute spectrum
        compute_stability: Whether to compute stability coefficients
        compute_primes: Whether to generate internal primes
        compute_projection: Whether to run bridge projection
        seed: Random seed override (uses config.seed if None)
        perf_monitor: Optional performance monitor for step tracking

    Returns:
        DiagnosticState with computed data
    """
    from .vfd.canonical import (
        VFDSpace, TorsionOperator, ShiftOperator,
        verify_weyl_relation, verify_torsion_order
    )
    from .vfd.operators import (
        create_torsion_projectors,
        verify_projector_resolution,
        verify_projector_orthogonality
    )
    from .vfd.kernels import CanonicalKernel
    from .spectrum.backend import compute_spectrum as compute_spec, SpectralBackend

    # Set seed
    actual_seed = seed if seed is not None else config.seed
    np.random.seed(actual_seed)

    # Create space
    if perf_monitor:
        perf_monitor.start_step("create_vfdspace")

    space = VFDSpace(
        cell_count=config.vfd.cell_count,
        internal_dim=config.vfd.internal_dim,
        orbit_count=config.vfd.orbit_count,
        orbit_size=config.vfd.orbit_size,
        periodic=config.vfd.periodic_boundary
    )

    if perf_monitor:
        perf_monitor.end_step("create_vfdspace")

    # Create operators
    if perf_monitor:
        perf_monitor.start_step("create_operators_T_S")

    T = TorsionOperator(space)
    S = ShiftOperator(space)

    if perf_monitor:
        perf_monitor.end_step("create_operators_T_S")

    # Create kernel
    if perf_monitor:
        perf_monitor.start_step("create_kernel")

    kernel = CanonicalKernel(
        space, T, S,
        propagation_range=config.vfd.local_propagation_L
    )

    if perf_monitor:
        perf_monitor.end_step("create_kernel")

    # Compute invariants
    if perf_monitor:
        perf_monitor.start_step("compute_invariants")

    invariants = {}

    # Torsion order
    torsion_pass, torsion_err = verify_torsion_order(T)
    invariants["torsion_pass"] = torsion_pass
    invariants["torsion_error"] = float(torsion_err)

    # Weyl relation
    weyl_pass, weyl_err = verify_weyl_relation(T, S, seed=actual_seed)
    invariants["weyl_pass"] = weyl_pass
    invariants["weyl_error"] = float(weyl_err)

    # Projectors
    projectors = create_torsion_projectors(T)
    res_pass, res_err = verify_projector_resolution(projectors)
    orth_pass, orth_err = verify_projector_orthogonality(projectors)
    invariants["projector_resolution_pass"] = res_pass
    invariants["projector_resolution_error"] = float(res_err)
    invariants["projector_orthogonality_pass"] = orth_pass
    invariants["projector_orthogonality_error"] = float(orth_err)

    # Kernel properties
    kernel_props = kernel.verify_all_properties()
    invariants["kernel_D1_pass"] = kernel_props.D1_selfadjoint
    invariants["kernel_D1_error"] = float(kernel_props.D1_error)
    invariants["kernel_D2_pass"] = kernel_props.D2_torsion_commute
    invariants["kernel_D2_error"] = float(kernel_props.D2_error)
    invariants["kernel_D3_pass"] = kernel_props.D3_nonnegative
    invariants["kernel_D3_min"] = float(kernel_props.D3_min_eigenvalue)

    if perf_monitor:
        perf_monitor.end_step("compute_invariants")

    # Build state
    state = DiagnosticState(
        config=config,
        space=space,
        T=T,
        S=S,
        kernel=kernel,
        invariants=invariants,
    )

    # Compute spectrum
    if compute_spectrum:
        if perf_monitor:
            perf_monitor.start_step("compute_spectrum")

        spectrum_result = compute_spec(
            cell_count=space.cell_count,
            internal_dim=space.internal_dim,
            propagation_range=config.vfd.local_propagation_L,
            backend=SpectralBackend.ANALYTIC_KCAN,
            use_cache=True
        )
        state.spectrum = spectrum_result

        if perf_monitor:
            perf_monitor.end_step("compute_spectrum")

    # Compute stability
    if compute_stability:
        if perf_monitor:
            perf_monitor.start_step("compute_stability")

        from .vfd.stability import StabilityAnalyzer

        analyzer = StabilityAnalyzer(space, T, kernel, seed=actual_seed)
        analyzer.compute_stability_coefficients(
            probe_count=config.stability.probe_count,
            support_radius=config.stability.probe_support_radius
        )
        state.stability_df = analyzer.to_dataframe()

        if perf_monitor:
            perf_monitor.end_step("compute_stability")

    # Compute primes
    if compute_primes:
        if perf_monitor:
            perf_monitor.start_step("compute_primes")

        from .vfd.primes import InternalPrimeGenerator, NonUFDAnalyzer
        from .vfd.transport import TransportAlgebra

        prime_gen = InternalPrimeGenerator(
            space, T, S,
            max_length=config.prime_field.max_transport_length,
            seed=actual_seed
        )
        prime_gen.generate_primes_up_to(config.prime_field.max_transport_length)
        state.primes_df = prime_gen.to_dataframe()

        if config.prime_field.emit_non_ufd_examples:
            algebra = TransportAlgebra(space, T, S)
            non_ufd = NonUFDAnalyzer(algebra)
            state.non_ufd_examples = non_ufd.find_non_ufd_examples()

        if perf_monitor:
            perf_monitor.end_step("compute_primes")

    # Compute projection
    if compute_projection and config.bridge.bridge_mode != "OFF":
        if perf_monitor:
            perf_monitor.start_step("compute_projection_bridge")

        from .bridge.bridge_axiom import BridgeAxiom, BridgeMode
        from .bridge.projection import ShadowProjection, compare_ba_vs_bn
        from .bridge.reference_data import ReferenceDataLoader

        bridge_mode = BridgeMode[config.bridge.bridge_mode]
        bridge = BridgeAxiom(mode=bridge_mode)
        reference = ReferenceDataLoader(max_zeros=config.reference.max_reference_zeros)

        if state.spectrum is not None:
            projection = ShadowProjection(bridge, reference)
            eigenvalues = state.spectrum.eigenvalues

            proj_result = projection.run_full_projection(
                eigenvalues,
                max_project=config.bridge.max_zeros_compare
            )
            state.projection = {
                "projected_zeros": proj_result.get("projected_zeros", []),
                "overlay_metrics": proj_result.get("overlay_metrics", {}),
                "spacing_metrics": proj_result.get("spacing_metrics", {}),
                "bridge_mode": config.bridge.bridge_mode,
            }

            # Falsification comparison
            reference_zeros = reference.get_zeta_zeros(config.bridge.max_zeros_compare)
            falsification = compare_ba_vs_bn(eigenvalues, reference_zeros, seed=actual_seed)
            state.projection["falsification"] = falsification

        if perf_monitor:
            perf_monitor.end_step("compute_projection_bridge")

    return state


def state_to_results_dict(state: DiagnosticState) -> Dict[str, Any]:
    """
    Convert DiagnosticState to results dictionary for export.

    Args:
        state: Diagnostic state

    Returns:
        Dictionary suitable for metrics/manifest export
    """
    results = {
        "config": state.config.to_dict() if hasattr(state.config, "to_dict") else {},
        "invariants": state.invariants,
    }

    if state.spectrum is not None:
        results["spectrum"] = {
            "eigenvalues": state.spectrum.eigenvalues.tolist(),
            "count": len(state.spectrum.eigenvalues),
            "min": float(state.spectrum.eigenvalues.min()),
            "max": float(state.spectrum.eigenvalues.max()),
            "backend": state.spectrum.backend_used,
            "computation_time_ms": state.spectrum.computation_time_ms,
        }

    if state.stability_df is not None:
        Q_values = state.stability_df["Q_Kcan"].values if "Q_Kcan" in state.stability_df.columns else []
        results["stability_summary"] = {
            "probe_count": len(state.stability_df),
            "min_Q": float(np.min(Q_values)) if len(Q_values) > 0 else 0.0,
            "all_nonnegative": bool(np.all(np.array(Q_values) >= -1e-10)) if len(Q_values) > 0 else True,
            "kernel_absoluteness": bool(np.all(np.array(Q_values) >= -1e-10)) if len(Q_values) > 0 else True,
        }

    if state.projection:
        results["projection"] = state.projection

    return results
