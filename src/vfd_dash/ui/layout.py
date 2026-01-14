"""
Main Dashboard Layout.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc


def create_sidebar():
    """Create the left sidebar with run controls."""
    return dbc.Card([
        dbc.CardHeader("Run Controls"),
        dbc.CardBody([
            # Run name
            dbc.Label("Run Name"),
            dbc.Input(id="run-name-input", value="default_run", type="text"),
            html.Br(),

            # Seed
            dbc.Label("Random Seed"),
            dbc.Input(id="seed-input", value=42, type="number"),
            html.Br(),

            # Cell count
            dbc.Label("Cell Count"),
            dbc.Input(id="cell-count-input", value=16, type="number", min=4, max=128),
            html.Br(),

            # Max primes length
            dbc.Label("Max Prime Length"),
            dbc.Input(id="max-prime-length-input", value=50, type="number", min=10, max=500),
            html.Br(),

            # Probe count
            dbc.Label("Probe Count"),
            dbc.Input(id="probe-count-input", value=200, type="number", min=50, max=2000),
            html.Br(),

            # Bridge mode
            dbc.Label("Bridge Mode"),
            dcc.Dropdown(
                id="bridge-mode-dropdown",
                options=[
                    {"label": "BA (Bridge Axiom)", "value": "BA"},
                    {"label": "BN1 (Wrong Order)", "value": "BN1"},
                    {"label": "BN2 (Wrong Scale)", "value": "BN2"},
                    {"label": "BN3 (Wrong Self-Dual)", "value": "BN3"},
                    {"label": "OFF (Disabled)", "value": "OFF"},
                ],
                value="BA"
            ),
            html.Br(),

            # Run button
            dbc.Button(
                "Run Analysis",
                id="run-button",
                color="primary",
                className="w-100 mb-2"
            ),

            # Export toggle
            dbc.Checkbox(
                id="export-bundle-checkbox",
                label="Export Bundle",
                value=True
            ),
            html.Br(),

            # Replay dropdown
            dbc.Label("Replay Previous Run"),
            dcc.Dropdown(
                id="replay-dropdown",
                options=[],
                placeholder="Select run to replay..."
            ),
        ])
    ], className="h-100")


def create_assertions_panel():
    """Create the right panel showing assertions and metrics."""
    return dbc.Card([
        dbc.CardHeader("Assertions & Metrics"),
        dbc.CardBody([
            html.Div(id="invariant-checks", children=[
                html.H6("VFD Invariants"),
                html.Div(id="weyl-check", className="check-item"),
                html.Div(id="torsion-check", className="check-item"),
                html.Div(id="projector-check", className="check-item"),
                html.Div(id="kernel-check", className="check-item"),
            ]),
            html.Hr(),
            html.Div(id="bridge-metrics", children=[
                html.H6("Bridge Metrics (BA)"),
                html.Div(id="overlay-rmse", className="metric-item"),
                html.Div(id="spacing-ks", className="metric-item"),
            ]),
            html.Hr(),
            html.Div(id="falsification-status", children=[
                html.H6("Falsification Status"),
                html.Div(id="bn1-ratio", className="metric-item"),
                html.Div(id="bn2-ratio", className="metric-item"),
                html.Div(id="bn3-ratio", className="metric-item"),
            ]),
        ])
    ], className="h-100")


def create_tabs():
    """Create the main content tabs."""
    return dbc.Tabs([
        dbc.Tab(label="Structure", tab_id="tab-structure"),
        dbc.Tab(label="Prime Field", tab_id="tab-primes"),
        dbc.Tab(label="Stability", tab_id="tab-stability"),
        dbc.Tab(label="Shadow", tab_id="tab-shadow"),
        dbc.Tab(label="Audit", tab_id="tab-audit"),
    ], id="main-tabs", active_tab="tab-structure")


def create_layout():
    """Create the main dashboard layout."""
    return dbc.Container([
        # Header
        dbc.Row([
            dbc.Col([
                html.H2("VFD Prime Field & RH Shadow Projection Dashboard"),
                html.P("A proof-grade instrument for VFD analysis", className="text-muted"),
            ])
        ], className="mb-3"),

        # Main content
        dbc.Row([
            # Left sidebar
            dbc.Col(create_sidebar(), width=2),

            # Main tabs area
            dbc.Col([
                create_tabs(),
                html.Div(id="tab-content", className="mt-3"),
            ], width=8),

            # Right assertions panel
            dbc.Col(create_assertions_panel(), width=2),
        ], className="g-3"),

        # Hidden stores for data
        dcc.Store(id="run-results-store"),
        dcc.Store(id="config-store"),

        # Loading indicator
        dcc.Loading(
            id="loading-indicator",
            type="circle",
            children=html.Div(id="loading-output")
        ),

    ], fluid=True, className="p-4")
