"""
Audit Tab: Reproducibility and Export.
"""

from dash import html, dcc
import dash_bootstrap_components as dbc
import json
from typing import Dict, Any, Optional


def create_audit_tab(results: Optional[Dict[str, Any]] = None):
    """Create the Audit/Reproducibility tab content."""
    return dbc.Container([
        dbc.Row([
            dbc.Col([
                html.H4("Audit & Reproducibility"),
                html.P("All runs are deterministic, hashable, and replayable."),
            ])
        ]),

        dbc.Row([
            # Config viewer
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Run Configuration"),
                    dbc.CardBody([
                        html.Div(id="config-viewer", children=create_config_viewer(results))
                    ])
                ])
            ], width=6),

            # Run info
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Run Information"),
                    dbc.CardBody([
                        html.Div(id="run-info", children=create_run_info(results))
                    ])
                ])
            ], width=6),
        ], className="mb-3"),

        dbc.Row([
            # Export controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Export Bundle"),
                    dbc.CardBody([
                        html.Div(id="export-controls", children=create_export_controls(results))
                    ])
                ])
            ], width=6),

            # Replay controls
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader("Replay Previous Run"),
                    dbc.CardBody([
                        html.Div(id="replay-controls", children=create_replay_controls(results))
                    ])
                ])
            ], width=6),
        ]),
    ])


def create_config_viewer(results: Optional[Dict[str, Any]] = None) -> list:
    """Create configuration JSON viewer."""
    content = []

    if results and "config" in results:
        config = results["config"]
        config_json = json.dumps(config, indent=2, default=str)
    else:
        config_json = json.dumps({
            "run_name": "default_run",
            "seed": 42,
            "vfd": {
                "torsion_order": 12,
                "cell_count": 16,
            },
            "bridge": {
                "mode": "BA",
            }
        }, indent=2)

    content.append(html.Pre(
        config_json,
        style={
            "backgroundColor": "#f8f9fa",
            "padding": "10px",
            "borderRadius": "5px",
            "maxHeight": "300px",
            "overflow": "auto",
            "fontSize": "12px"
        }
    ))

    return content


def create_run_info(results: Optional[Dict[str, Any]] = None) -> list:
    """Create run information panel."""
    content = []

    if results and "metrics" in results:
        metrics = results["metrics"]
        run_hash = metrics.get("run_hash", "N/A")
        config_hash = metrics.get("config_hash", "N/A")
        timestamp = metrics.get("timestamp", "N/A")
        git_commit = metrics.get("git_commit", "N/A")
    else:
        run_hash = "Not yet run"
        config_hash = "N/A"
        timestamp = "N/A"
        git_commit = "N/A"

    info_items = [
        ("Run Hash", run_hash),
        ("Config Hash", config_hash),
        ("Timestamp", timestamp),
        ("Git Commit", git_commit),
    ]

    for label, value in info_items:
        content.append(dbc.Row([
            dbc.Col(html.Strong(f"{label}:"), width=4),
            dbc.Col(html.Code(value), width=8),
        ], className="mb-2"))

    # Package versions
    if results and "metrics" in results:
        versions = results["metrics"].get("package_versions", {})
        if versions:
            content.append(html.Hr())
            content.append(html.H6("Package Versions"))
            for pkg, ver in list(versions.items())[:5]:
                content.append(html.Small(f"{pkg}: {ver}", className="d-block"))

    return content


def create_export_controls(results: Optional[Dict[str, Any]] = None) -> list:
    """Create export controls."""
    content = []

    content.append(html.P("Export complete bundle including:"))
    content.append(html.Ul([
        html.Li("Configuration (config.json)"),
        html.Li("Datasets (parquet/csv)"),
        html.Li("Figures (png)"),
        html.Li("Metrics (metrics.json)"),
        html.Li("Manifest (manifest.json)"),
    ]))

    if results and "export_path" in results:
        path = results["export_path"]
        content.append(dbc.Alert([
            html.Strong("Bundle Created"),
            html.Br(),
            html.Code(path)
        ], color="success"))

        content.append(dbc.Button(
            "Download Bundle",
            id="download-bundle-btn",
            color="primary",
            className="mt-2"
        ))
    else:
        content.append(dbc.Alert(
            "Run analysis with 'Export Bundle' checked to create bundle",
            color="secondary"
        ))

    return content


def create_replay_controls(results: Optional[Dict[str, Any]] = None) -> list:
    """Create replay controls."""
    content = []

    content.append(html.P(
        "Select a previous run to replay exactly. "
        "All outputs will be regenerated with identical hashes."
    ))

    # Dropdown will be populated by callback
    content.append(dcc.Dropdown(
        id="replay-run-dropdown",
        options=[],
        placeholder="Select run hash...",
        className="mb-3"
    ))

    content.append(dbc.Button(
        "Replay Selected Run",
        id="replay-btn",
        color="secondary",
        disabled=True
    ))

    return content
