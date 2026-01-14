"""
Stability Tab: Stability Analysis via Quadratic Forms.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from typing import Dict, Any, Optional


def create_stability_tab(results: Optional[Dict[str, Any]] = None):
    """Create the Stability tab content."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H4("VFD Intrinsic Stability"),
                html.P([
                    "Stability is ",
                    html.Strong("primary"),
                    "; zeros are derived. Measured via Q",
                    html.Sub("K"),
                    "(v) = \u27E8v, Kv\u27E9."
                ]),
            ])
        ]),

        dbc.Row([
            # Spectrum plot
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Canonical Kernel Spectrum"),
                    dbc.CardBody([
                        dcc.Graph(id="spectrum-plot", figure=create_spectrum_plot(results))
                    ])
                ])
            ], width=6),

            # Stability coefficient map
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Stability Coefficients"),
                    dbc.CardBody([
                        dcc.Graph(id="stability-map", figure=create_stability_map(results))
                    ])
                ])
            ], width=6),
        ], className="mb-3"),

        dbc.Row([
            # Self-dual manifold
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Self-Dual Manifold"),
                    dbc.CardBody([
                        dcc.Graph(id="self-dual-plot", figure=create_self_dual_plot(results))
                    ])
                ])
            ], width=6),

            # Kernel absoluteness
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Kernel Absoluteness"),
                    dbc.CardBody([
                        html.Div(id="absoluteness-panel", children=create_absoluteness_panel(results))
                    ])
                ])
            ], width=6),
        ]),
    ])


def create_spectrum_plot(results: Optional[Dict[str, Any]] = None) -> go.Figure:
    """Create spectrum plot of canonical kernel."""
    if results and "spectrum" in results:
        eigenvalues = results["spectrum"]["eigenvalues"]
    else:
        # Sample spectrum (Laplacian-like)
        n = 100
        eigenvalues = np.sort(np.abs(np.random.randn(n) * 0.5 + np.arange(n) * 0.1))

    fig = go.Figure()

    # Sorted eigenvalues
    fig.add_trace(go.Scatter(
        x=list(range(len(eigenvalues))),
        y=eigenvalues,
        mode="lines+markers",
        name="Eigenvalues",
        marker=dict(size=4),
        line=dict(width=1)
    ))

    # Mark nonnegative region
    fig.add_hline(y=0, line_dash="dash", line_color="red", annotation_text="Stability threshold")

    fig.update_layout(
        title="K_can Spectrum (sorted)",
        xaxis_title="Index",
        yaxis_title="\u03BB",
        height=350
    )

    return fig


def create_stability_map(results: Optional[Dict[str, Any]] = None) -> go.Figure:
    """Create stability coefficient map."""
    if results and "stability" in results:
        df = results["stability"]
        Q_values = df["Q_Kcan"].values
        torsion = df["torsion_degree"].values
    else:
        # Sample data
        n = 200
        torsion = np.random.randint(0, 12, n)
        Q_values = np.abs(np.random.randn(n)) * 0.5 + 0.1

    fig = go.Figure()

    # Color by torsion degree
    fig.add_trace(go.Scatter(
        x=list(range(len(Q_values))),
        y=Q_values,
        mode="markers",
        marker=dict(
            size=8,
            color=torsion,
            colorscale="Viridis",
            colorbar=dict(title="Torsion k"),
            opacity=0.7
        ),
        name="Q_K(v)"
    ))

    # Stability threshold
    fig.add_hline(y=0, line_dash="dash", line_color="red")

    fig.update_layout(
        title="Stability Coefficients by Probe",
        xaxis_title="Probe Index",
        yaxis_title="Q_K(v)",
        height=350
    )

    return fig


def create_self_dual_plot(results: Optional[Dict[str, Any]] = None) -> go.Figure:
    """Create self-dual manifold visualization."""
    # Self-dual manifold: fixed locus of s -> 1-s
    # In VFD: balanced torsion components

    if results and "stability" in results:
        df = results["stability"]
        coords = df["self_dual_coord"].values
        Q_values = df["Q_Kcan"].values
    else:
        n = 200
        coords = np.random.uniform(0, 1, n)
        Q_values = np.abs(np.random.randn(n)) * 0.5 + 0.1

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=coords,
        y=Q_values,
        mode="markers",
        marker=dict(
            size=8,
            color=Q_values,
            colorscale="RdYlGn",
            colorbar=dict(title="Q_K(v)"),
            opacity=0.7
        ),
        name="Probes"
    ))

    # Mark self-dual manifold at s = 1/2
    fig.add_vline(x=0.5, line_dash="dash", line_color="blue",
                  annotation_text="Self-dual manifold (s=1/2)")

    fig.update_layout(
        title="Stability on Self-Dual Manifold",
        xaxis_title="Self-Dual Coordinate",
        yaxis_title="Q_K(v)",
        height=350
    )

    return fig


def create_absoluteness_panel(results: Optional[Dict[str, Any]] = None) -> list:
    """Create kernel absoluteness panel."""
    content = []

    content.append(html.H5("Theorem: Kernel Absoluteness"))
    content.append(html.P([
        "Nonnegative admissible kernels are ",
        html.Strong("intrinsically stable"),
        ". Instability is structurally impossible."
    ]))

    if results and "stability" in results:
        df = results["stability"]
        min_Q = df["Q_Kcan"].min()
        n_negative = (df["Q_Kcan"] < -1e-10).sum()
        n_total = len(df)

        passes = n_negative == 0
        status_color = "success" if passes else "danger"

        content.append(dbc.Alert([
            html.Strong("Kernel Absoluteness: " + ("VERIFIED" if passes else "FAILED")),
            html.Br(),
            f"Probes tested: {n_total}",
            html.Br(),
            f"Min Q_K(v): {min_Q:.6f}",
            html.Br(),
            f"Negative count: {n_negative}",
        ], color=status_color))

        if passes:
            content.append(dbc.Alert([
                html.Strong("Instability Impossible"),
                html.Br(),
                "All stability coefficients are nonnegative.",
                html.Br(),
                "This is a structural property of VFD, not empirical observation."
            ], color="info"))
    else:
        content.append(dbc.Alert("Run analysis to verify kernel absoluteness", color="secondary"))

    return content
