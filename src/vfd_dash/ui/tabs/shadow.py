"""
Shadow Tab: RH Shadow Projection.

This is TRANSLATION content - connecting VFD to classical RH.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
from typing import Dict, Any, Optional


def create_shadow_tab(results: Optional[Dict[str, Any]] = None):
    """Create the Shadow (RH Projection) tab content."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H4("RH Shadow Projection"),
                html.P([
                    "This tab shows ",
                    html.Strong("translation"),
                    " of VFD data to classical shadows. ",
                    "RH is a projection, not the goal."
                ], className="text-warning"),
            ])
        ]),

        # Bridge mode controls
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Bridge Controls"),
                    dbc.CardBody([
                        dbc.RadioItems(
                            id="bridge-mode-radio",
                            options=[
                                {"label": "BA (Bridge Axiom)", "value": "BA"},
                                {"label": "BN1 (Wrong Order)", "value": "BN1"},
                                {"label": "BN2 (Wrong Scale)", "value": "BN2"},
                                {"label": "BN3 (Wrong Self-Dual)", "value": "BN3"},
                            ],
                            value="BA",
                            inline=True
                        ),
                        html.P(
                            "Toggle controls to see falsification in action.",
                            className="text-muted mt-2"
                        )
                    ])
                ])
            ])
        ], className="mb-3"),

        dbc.Row([
            # Zero overlay plot
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Projected Zeros vs Reference"),
                    dbc.CardBody([
                        dcc.Graph(id="zero-overlay", figure=create_zero_overlay(results))
                    ])
                ])
            ], width=6),

            # Residual plot
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Residuals vs Index"),
                    dbc.CardBody([
                        dcc.Graph(id="residual-plot", figure=create_residual_plot(results))
                    ])
                ])
            ], width=6),
        ], className="mb-3"),

        dbc.Row([
            # Gap distribution
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Gap Distribution Comparison"),
                    dbc.CardBody([
                        dcc.Graph(id="gap-dist", figure=create_gap_distribution(results))
                    ])
                ])
            ], width=6),

            # Metrics table
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Bridge Metrics: BA vs BN"),
                    dbc.CardBody([
                        html.Div(id="bridge-metrics-table", children=create_metrics_table(results))
                    ])
                ])
            ], width=6),
        ]),
    ])


def create_zero_overlay(results: Optional[Dict[str, Any]] = None) -> go.Figure:
    """Create projected zeros vs reference zeros overlay."""
    fig = go.Figure()

    if results and "projection" in results:
        proj = results["projection"]
        projected = proj.get("projected_zeros", [])
        reference = proj.get("reference_zeros", [])

        t_proj = [z["t_projected"] for z in projected[:100]]
        t_ref = reference[:100] if len(reference) >= 100 else reference
    else:
        # Sample data
        n = 50
        t_ref = np.cumsum(np.random.exponential(2, n)) + 14
        t_proj = t_ref + np.random.randn(n) * 0.5

    indices = list(range(1, len(t_proj) + 1))

    # Reference zeros
    fig.add_trace(go.Scatter(
        x=indices,
        y=list(t_ref) if not isinstance(t_ref, list) else t_ref,
        mode="markers",
        name="Reference \u03B6",
        marker=dict(size=8, color="blue", symbol="circle-open")
    ))

    # Projected zeros
    fig.add_trace(go.Scatter(
        x=indices,
        y=t_proj,
        mode="markers",
        name="Projected VFD",
        marker=dict(size=6, color="red", symbol="x")
    ))

    fig.update_layout(
        title="Zero Heights: Projected vs Reference",
        xaxis_title="Index n",
        yaxis_title="t_n (imaginary part)",
        height=350,
        legend=dict(x=0.02, y=0.98)
    )

    return fig


def create_residual_plot(results: Optional[Dict[str, Any]] = None) -> go.Figure:
    """Create residual plot."""
    fig = go.Figure()

    if results and "projection" in results:
        projected = results["projection"].get("projected_zeros", [])
        residuals = [z.get("residual_to_reference", 0) for z in projected[:100]]
    else:
        n = 50
        residuals = np.random.randn(n) * 0.5

    indices = list(range(1, len(residuals) + 1))

    fig.add_trace(go.Scatter(
        x=indices,
        y=residuals,
        mode="lines+markers",
        name="Residual",
        line=dict(width=1),
        marker=dict(size=4)
    ))

    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    # Add error bands
    if len(residuals) > 0:
        std = np.std(residuals)
        fig.add_hrect(y0=-std, y1=std, fillcolor="green", opacity=0.1,
                      annotation_text="\u00B11\u03C3")

    fig.update_layout(
        title="Residuals (Projected - Reference)",
        xaxis_title="Index n",
        yaxis_title="Residual",
        height=350
    )

    return fig


def create_gap_distribution(results: Optional[Dict[str, Any]] = None) -> go.Figure:
    """Create gap distribution comparison."""
    fig = go.Figure()

    if results and "projection" in results:
        # Use actual data
        pass
    else:
        # Sample data: GUE-like distribution
        n = 200
        gaps_ref = np.random.exponential(1, n)
        gaps_proj = gaps_ref + np.random.randn(n) * 0.1

    # Normalize
    gaps_ref = gaps_ref / np.mean(gaps_ref)
    gaps_proj = gaps_proj / np.mean(gaps_proj)

    # Histograms
    bins = np.linspace(0, 4, 30)

    fig.add_trace(go.Histogram(
        x=gaps_ref,
        nbinsx=30,
        name="Reference",
        opacity=0.7,
        marker_color="blue"
    ))

    fig.add_trace(go.Histogram(
        x=gaps_proj,
        nbinsx=30,
        name="Projected",
        opacity=0.7,
        marker_color="red"
    ))

    fig.update_layout(
        title="Normalized Gap Distribution",
        xaxis_title="Normalized Gap",
        yaxis_title="Count",
        barmode="overlay",
        height=350
    )

    return fig


def create_metrics_table(results: Optional[Dict[str, Any]] = None) -> list:
    """Create metrics comparison table."""
    content = []

    modes = ["BA", "BN1", "BN2", "BN3"]

    if results and "falsification" in results:
        metrics = results["falsification"]
    else:
        # Sample metrics
        metrics = {
            "BA": {"rmse": 0.5, "mae": 0.3, "ks": 0.05},
            "BN1": {"rmse": 5.2, "mae": 4.1, "ks": 0.35},
            "BN2": {"rmse": 8.1, "mae": 6.5, "ks": 0.42},
            "BN3": {"rmse": 3.8, "mae": 2.9, "ks": 0.28},
        }

    table_rows = []
    for mode in modes:
        m = metrics.get(mode, {})
        row = html.Tr([
            html.Td(mode, style={"fontWeight": "bold"}),
            html.Td(f"{m.get('rmse', 'N/A'):.3f}" if isinstance(m.get('rmse'), (int, float)) else "N/A"),
            html.Td(f"{m.get('mae', 'N/A'):.3f}" if isinstance(m.get('mae'), (int, float)) else "N/A"),
            html.Td(f"{m.get('ks', 'N/A'):.3f}" if isinstance(m.get('ks'), (int, float)) else "N/A"),
        ])
        table_rows.append(row)

    content.append(dbc.Table([
        html.Thead(html.Tr([
            html.Th("Mode"),
            html.Th("RMSE"),
            html.Th("MAE"),
            html.Th("KS Stat"),
        ])),
        html.Tbody(table_rows)
    ], bordered=True, striped=True, hover=True))

    # Falsification status
    if results and "falsification" in results:
        ratios = metrics.get("falsification_ratios", {})
        all_worse = all(r >= 1.5 for r in ratios.values()) if ratios else False

        status = "SUCCESSFUL" if all_worse else "INCONCLUSIVE"
        color = "success" if all_worse else "warning"

        content.append(dbc.Alert([
            html.Strong(f"Falsification: {status}"),
            html.Br(),
            "BN modes show significantly worse metrics." if all_worse else
            "BN modes should show worse metrics for valid bridge."
        ], color=color, className="mt-3"))
    else:
        content.append(dbc.Alert(
            "Run analysis to compare BA vs BN metrics",
            color="secondary", className="mt-3"
        ))

    return content
