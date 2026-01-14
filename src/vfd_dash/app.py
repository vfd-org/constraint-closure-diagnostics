"""
VFD Proof Dashboard - Main Application.

A proof-grade instrument for VFD Prime Field & RH Shadow Projection.
"""

import json
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
from dash import Dash, html, dcc, callback, Output, Input, State
import dash_bootstrap_components as dbc

from .core.config import RunConfig, get_default_config
from .core.hashing import compute_run_hash, compute_config_hash
from .core.logging import setup_logging, get_logger

from .vfd.canonical import VFDSpace, TorsionOperator, ShiftOperator, verify_weyl_relation, verify_torsion_order
from .vfd.operators import create_torsion_projectors, verify_projector_resolution, verify_projector_orthogonality
from .vfd.kernels import CanonicalKernel
from .vfd.probes import ProbeGenerator
from .vfd.primes import InternalPrimeGenerator, NonUFDAnalyzer
from .vfd.stability import StabilityAnalyzer
from .vfd.transport import TransportAlgebra

from .bridge.bridge_axiom import BridgeAxiom, BridgeMode
from .bridge.projection import ShadowProjection, compare_ba_vs_bn
from .bridge.reference_data import ReferenceDataLoader

from .metrics.report import MetricsReport, generate_metrics_report
from .io.export_bundle import create_export_bundle, list_available_bundles

from .ui.layout import create_layout
from .ui.tabs import (
    create_structure_tab,
    create_primes_tab,
    create_stability_tab,
    create_shadow_tab,
    create_audit_tab
)


# Initialize app
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="VFD Proof Dashboard"
)

app.layout = create_layout()


def run_analysis(config: RunConfig) -> Dict[str, Any]:
    """
    Run complete VFD analysis pipeline.

    Args:
        config: Run configuration

    Returns:
        Dictionary with all results
    """
    logger = get_logger()
    logger.info(f"Starting analysis run: {config.run_name}")

    results = {
        "config": config.to_dict(),
    }

    # Set random seed
    np.random.seed(config.seed)

    # 1. Create VFD space and operators
    logger.info("Creating VFD space and operators...")
    space = VFDSpace(
        cell_count=config.vfd.cell_count,
        internal_dim=config.vfd.internal_dim,
        orbit_count=config.vfd.orbit_count,
        orbit_size=config.vfd.orbit_size,
        periodic=config.vfd.periodic_boundary
    )

    T = TorsionOperator(space)
    S = ShiftOperator(space)

    # 2. Verify invariants
    logger.info("Verifying VFD invariants...")
    invariants = {}

    # Torsion order
    torsion_pass, torsion_err = verify_torsion_order(T)
    invariants["torsion_pass"] = torsion_pass
    invariants["torsion_error"] = torsion_err

    # Weyl relation
    weyl_pass, weyl_err = verify_weyl_relation(T, S)
    invariants["weyl_pass"] = weyl_pass
    invariants["weyl_error"] = weyl_err

    # Projectors
    projectors = create_torsion_projectors(T)
    res_pass, res_err = verify_projector_resolution(projectors)
    orth_pass, orth_err = verify_projector_orthogonality(projectors)
    invariants["projector_resolution_pass"] = res_pass
    invariants["projector_resolution_error"] = res_err
    invariants["projector_orthogonality_pass"] = orth_pass
    invariants["projector_orthogonality_error"] = orth_err

    # 3. Create canonical kernel
    logger.info("Creating canonical kernel...")
    kernel = CanonicalKernel(space, T, S, propagation_range=config.vfd.local_propagation_L)

    # Verify kernel properties
    kernel_props = kernel.verify_all_properties()
    invariants["kernel_D1_pass"] = kernel_props.D1_selfadjoint
    invariants["kernel_D1_error"] = kernel_props.D1_error
    invariants["kernel_D2_pass"] = kernel_props.D2_torsion_commute
    invariants["kernel_D2_error"] = kernel_props.D2_error
    invariants["kernel_D3_pass"] = kernel_props.D3_nonnegative
    invariants["kernel_D3_min"] = kernel_props.D3_min_eigenvalue

    results["invariants"] = invariants

    # 4. Generate internal primes
    logger.info("Generating internal primes...")
    prime_gen = InternalPrimeGenerator(
        space, T, S,
        max_length=config.prime_field.max_transport_length,
        seed=config.seed
    )
    primes = prime_gen.generate_primes_up_to(config.prime_field.max_transport_length)
    primes_df = prime_gen.to_dataframe()

    results["primes"] = primes_df

    # Non-UFD examples
    if config.prime_field.emit_non_ufd_examples:
        algebra = TransportAlgebra(space, T, S)
        non_ufd = NonUFDAnalyzer(algebra)
        non_ufd_examples = non_ufd.find_non_ufd_examples()
        results["non_ufd_examples"] = non_ufd_examples

    # 5. Stability analysis
    logger.info("Analyzing stability...")
    stability_analyzer = StabilityAnalyzer(space, T, kernel, seed=config.seed)

    coefficients = stability_analyzer.compute_stability_coefficients(
        probe_count=config.stability.probe_count,
        support_radius=config.stability.probe_support_radius
    )
    stability_df = stability_analyzer.to_dataframe()

    results["stability"] = stability_df

    # Kernel absoluteness
    abs_pass, abs_details = stability_analyzer.verify_kernel_absoluteness()
    results["stability_summary"] = {
        "kernel_absoluteness": abs_pass,
        "probe_count": len(coefficients),
        "min_Q": abs_details["min_Q_value"],
        "all_nonnegative": abs_details["negative_count"] == 0,
    }

    # Spectrum
    logger.info("Computing spectrum...")
    spectral_data = stability_analyzer.compute_spectrum(k=min(config.stability.spectrum_k, space.total_dim - 2))
    results["spectrum"] = {
        "eigenvalues": spectral_data.eigenvalues.tolist(),
        "count": len(spectral_data.eigenvalues),
        "min": float(spectral_data.eigenvalues.min()),
        "max": float(spectral_data.eigenvalues.max()),
    }

    # 6. Bridge projection (if enabled)
    if config.bridge.bridge_mode != "OFF":
        logger.info(f"Running bridge projection (mode: {config.bridge.bridge_mode})...")

        # Set up bridge
        bridge_mode = BridgeMode[config.bridge.bridge_mode]
        bridge = BridgeAxiom(mode=bridge_mode)

        # Reference data
        reference = ReferenceDataLoader(max_zeros=config.reference.max_reference_zeros)

        # Projection
        projection = ShadowProjection(bridge, reference)
        eigenvalues = np.array(spectral_data.eigenvalues)

        proj_result = projection.run_full_projection(
            eigenvalues,
            max_project=config.bridge.max_zeros_compare
        )

        results["projection"] = {
            "projected_zeros": [z.to_dict() for z in proj_result["projected_zeros"][:100]],
            "overlay_metrics": proj_result["overlay_metrics"],
            "spacing_metrics": proj_result["spacing_metrics"],
            "bridge_mode": config.bridge.bridge_mode,
        }

        # Compare BA vs BN
        logger.info("Running falsification comparison...")
        reference_zeros = reference.get_zeta_zeros(config.bridge.max_zeros_compare)
        falsification = compare_ba_vs_bn(eigenvalues, reference_zeros, seed=config.seed)
        results["falsification"] = falsification

    # 7. Generate metrics report
    logger.info("Generating metrics report...")
    report = generate_metrics_report(
        config=config,
        invariant_results=invariants,
        prime_results={
            "count": len(primes),
            "max_length": config.prime_field.max_transport_length,
            "non_ufd_examples": len(results.get("non_ufd_examples", [])),
        },
        stability_results=results.get("stability_summary", {}),
        bridge_results=results.get("projection", {}).get("overlay_metrics", {}),
        falsification_results=results.get("falsification", {}),
        spectrum_results=results.get("spectrum", {}),
    )

    results["metrics"] = report.to_dict()

    logger.info("Analysis complete.")
    return results


# Callbacks
@callback(
    Output("tab-content", "children"),
    Input("main-tabs", "active_tab"),
    State("run-results-store", "data")
)
def render_tab(active_tab: str, results: Optional[Dict[str, Any]]):
    """Render the selected tab content."""
    if active_tab == "tab-structure":
        return create_structure_tab(results)
    elif active_tab == "tab-primes":
        return create_primes_tab(results)
    elif active_tab == "tab-stability":
        return create_stability_tab(results)
    elif active_tab == "tab-shadow":
        return create_shadow_tab(results)
    elif active_tab == "tab-audit":
        return create_audit_tab(results)
    else:
        return html.Div("Select a tab")


@callback(
    Output("run-results-store", "data"),
    Output("loading-output", "children"),
    Input("run-button", "n_clicks"),
    State("run-name-input", "value"),
    State("seed-input", "value"),
    State("cell-count-input", "value"),
    State("max-prime-length-input", "value"),
    State("probe-count-input", "value"),
    State("bridge-mode-dropdown", "value"),
    State("export-bundle-checkbox", "value"),
    prevent_initial_call=True
)
def run_analysis_callback(
    n_clicks: int,
    run_name: str,
    seed: int,
    cell_count: int,
    max_prime_length: int,
    probe_count: int,
    bridge_mode: str,
    export_bundle: bool
):
    """Run the analysis when button is clicked."""
    if n_clicks is None:
        return None, ""

    # Build config
    config = get_default_config()
    config.run_name = run_name or "default_run"
    config.seed = seed or 42
    config.vfd.cell_count = cell_count or 16
    config.prime_field.max_transport_length = max_prime_length or 50
    config.stability.probe_count = probe_count or 200
    config.bridge.bridge_mode = bridge_mode or "BA"
    config.output.export_bundle = export_bundle

    # Run analysis
    results = run_analysis(config)

    # Export bundle if requested
    if export_bundle:
        datasets = {
            "internal_primes": results.get("primes", pd.DataFrame()),
            "stability": results.get("stability", pd.DataFrame()),
        }

        bundle = create_export_bundle(
            config=config,
            metrics=results.get("metrics", {}),
            datasets=datasets,
            figures={},
            output_dir=config.output.out_dir,
            run_name=config.run_name
        )
        results["export_path"] = str(bundle.run_dir)

    return results, "Analysis complete"


@callback(
    Output("weyl-check", "children"),
    Output("torsion-check", "children"),
    Output("projector-check", "children"),
    Output("kernel-check", "children"),
    Input("run-results-store", "data")
)
def update_invariant_checks(results: Optional[Dict[str, Any]]):
    """Update invariant check displays."""
    if not results or "invariants" not in results:
        pending = html.Span("\u23F3 Pending", className="text-muted")
        return pending, pending, pending, pending

    inv = results["invariants"]

    def check_badge(passed: bool, label: str, error: float = None) -> html.Span:
        if passed:
            text = f"\u2713 {label}"
            if error is not None:
                text += f" (err: {error:.2e})"
            return html.Span(text, className="text-success")
        else:
            return html.Span(f"\u2717 {label}", className="text-danger")

    weyl = check_badge(inv.get("weyl_pass", False), "Weyl", inv.get("weyl_error"))
    torsion = check_badge(inv.get("torsion_pass", False), "T^12=I", inv.get("torsion_error"))
    proj = check_badge(inv.get("projector_resolution_pass", False), "Projectors")
    kernel = check_badge(inv.get("kernel_D3_pass", False), "K>=0", inv.get("kernel_D3_min"))

    return weyl, torsion, proj, kernel


@callback(
    Output("replay-dropdown", "options"),
    Input("run-results-store", "data")
)
def update_replay_options(results):
    """Update replay dropdown with available bundles."""
    bundles = list_available_bundles()
    return [
        {"label": f"{b['run_name']} ({b['run_hash'][:8]})", "value": b['path']}
        for b in bundles[:10]
    ]


def main():
    """Run the dashboard server."""
    setup_logging()
    logger = get_logger()
    logger.info("Starting VFD Proof Dashboard...")
    app.run(debug=True, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()
