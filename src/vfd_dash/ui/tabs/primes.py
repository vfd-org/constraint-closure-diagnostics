"""
Prime Field Tab: Internal Primes Visualization.
"""

from dash import html, dcc, dash_table
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional


def create_primes_tab(results: Optional[Dict[str, Any]] = None):
    """Create the Prime Field tab content."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H4("VFD Internal Prime Field"),
                html.P("Internal primes arise from transport structure without reference to \u2115."),
            ])
        ]),

        dbc.Row([
            # Prime field lattice plot
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Prime Field Lattice"),
                    dbc.CardBody([
                        dcc.Graph(id="prime-lattice", figure=create_prime_lattice(results))
                    ])
                ])
            ], width=6),

            # Prime counting plot
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Prime Counting Function \u03C0_VFD(x)"),
                    dbc.CardBody([
                        dcc.Graph(id="prime-counting", figure=create_prime_counting(results))
                    ])
                ])
            ], width=6),
        ], className="mb-3"),

        dbc.Row([
            # Prime table
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Internal Primes"),
                    dbc.CardBody([
                        dbc.Input(
                            id="prime-search",
                            placeholder="Search primes...",
                            className="mb-2"
                        ),
                        html.Div(id="prime-table-container", children=create_prime_table(results))
                    ])
                ])
            ], width=8),

            # Non-UFD examples
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Non-UFD Examples"),
                    dbc.CardBody([
                        html.Div(id="non-ufd-panel", children=create_non_ufd_panel(results))
                    ])
                ])
            ], width=4),
        ]),
    ])


def create_prime_lattice(results: Optional[Dict[str, Any]] = None) -> go.Figure:
    """Create prime field lattice visualization."""
    if results and "primes" in results:
        df = results["primes"]
        lengths = df["m"].values
        torsion = df["torsion_class"].values
        directions = df["direction"].values
    else:
        # Sample data
        lengths = list(range(1, 51))
        torsion = [m % 12 for m in lengths]
        directions = ["+" if i % 2 == 0 else "-" for i in range(50)]

    # Create scatter plot
    fig = go.Figure()

    # Separate by direction
    for d, color, name in [("+", "blue", "Forward"), ("-", "red", "Backward")]:
        mask = [dir == d for dir in directions]
        fig.add_trace(go.Scatter(
            x=[lengths[i] for i in range(len(lengths)) if mask[i]],
            y=[torsion[i] for i in range(len(torsion)) if mask[i]],
            mode="markers",
            name=name,
            marker=dict(size=8, color=color, opacity=0.7)
        ))

    fig.update_layout(
        title="Prime Lattice: Transport Length vs Torsion Class",
        xaxis_title="Transport Length m",
        yaxis_title="Torsion Class (m mod 12)",
        height=400,
        yaxis=dict(tickmode="array", tickvals=list(range(12)))
    )

    return fig


def create_prime_counting(results: Optional[Dict[str, Any]] = None) -> go.Figure:
    """Create prime counting function plot."""
    if results and "primes" in results:
        df = results["primes"]
        max_length = df["m"].max()
    else:
        max_length = 50

    x = np.arange(1, max_length + 1)

    # Count primes up to each x
    if results and "primes" in results:
        counts = [len(results["primes"][results["primes"]["m"] <= i]) for i in x]
    else:
        # Approximate: internal primes are sparser
        counts = [int(i / np.log(i + 2) * 1.5) for i in x]

    # Density
    density = [c / i if i > 0 else 0 for c, i in zip(counts, x)]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x, y=counts,
        mode="lines",
        name="\u03C0_VFD(x)",
        line=dict(color="blue", width=2)
    ))

    fig.add_trace(go.Scatter(
        x=x, y=density,
        mode="lines",
        name="Density",
        yaxis="y2",
        line=dict(color="green", width=2, dash="dash")
    ))

    fig.update_layout(
        title="Internal Prime Counting",
        xaxis_title="x",
        yaxis_title="\u03C0_VFD(x)",
        yaxis2=dict(
            title="Density",
            overlaying="y",
            side="right"
        ),
        height=400
    )

    return fig


def create_prime_table(results: Optional[Dict[str, Any]] = None):
    """Create searchable prime table."""
    if results and "primes" in results:
        df = results["primes"].head(100)
    else:
        df = pd.DataFrame({
            "prime_id": [f"P_{i:06d}" for i in range(1, 21)],
            "m": list(range(1, 21)),
            "direction": ["+" if i % 2 == 0 else "-" for i in range(20)],
            "torsion_class": [i % 12 for i in range(20)],
        })

    return dash_table.DataTable(
        id="prime-table",
        columns=[{"name": c, "id": c} for c in df.columns[:5]],
        data=df.to_dict("records"),
        page_size=10,
        filter_action="native",
        sort_action="native",
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "left", "padding": "5px"},
        style_header={"fontWeight": "bold"},
    )


def create_non_ufd_panel(results: Optional[Dict[str, Any]] = None) -> list:
    """Create non-UFD examples panel."""
    content = []

    content.append(html.H6("Theorem: A is Non-UFD"))
    content.append(html.P(
        "The interaction algebra does not satisfy unique factorization.",
        className="text-muted"
    ))

    if results and "non_ufd_examples" in results and results["non_ufd_examples"]:
        examples = results["non_ufd_examples"][:3]
        for i, ex in enumerate(examples):
            content.append(dbc.Card([
                dbc.CardHeader(f"Example {i+1}: {ex.get('mode', 'unknown')}"),
                dbc.CardBody([
                    html.P(f"Factorizations: {ex.get('distinct_signatures', 0)}"),
                    html.Ul([
                        html.Li(" \u00D7 ".join(f.get("factors", [])))
                        for f in ex.get("factorizations", [])[:2]
                    ])
                ])
            ], className="mb-2", color="warning", outline=True))
    else:
        content.append(dbc.Alert(
            "Run analysis to find non-UFD examples",
            color="secondary"
        ))

    return content
