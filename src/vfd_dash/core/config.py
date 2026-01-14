"""
Configuration schema for VFD Proof Dashboard.

All run parameters are captured in a single RunConfig object,
which is exportable to JSON for reproducibility.
"""

from typing import Optional
import json

try:
    from pydantic import BaseModel, Field
    PYDANTIC_V2 = hasattr(BaseModel, "model_dump")
except ImportError:
    raise ImportError("pydantic is required")


class VFDConfig(BaseModel):
    """VFD canonical framework configuration."""

    torsion_order: int = 12
    orbit_count: int = 50
    orbit_size: int = 12
    internal_dim: int = 600
    cell_count: int = 64
    kernel_type: str = "K_can"
    kernel_params: dict = {}
    local_propagation_L: int = 3
    enable_geometry_model_claims: bool = False
    periodic_boundary: bool = True

    class Config:
        extra = "allow"


class PrimeFieldConfig(BaseModel):
    """Internal prime field configuration."""

    max_transport_length: int = 500
    prime_definition: str = "irreducible_transport_mode"
    factorization_search_depth: int = 6
    graph_max_nodes: int = 2000
    emit_non_ufd_examples: bool = True

    class Config:
        extra = "allow"


class StabilityConfig(BaseModel):
    """Stability analysis configuration."""

    spectrum_method: str = "eigsh"
    spectrum_k: int = 256
    probe_family: str = "finite_support_pure_degree"
    probe_count: int = 2000
    probe_support_radius: int = 3
    stability_metric: str = "quadratic_form_nonnegativity"

    class Config:
        extra = "allow"


class BridgeConfig(BaseModel):
    """Bridge Axiom and projection configuration."""

    bridge_mode: str = "BA"
    projection_map: str = "spectral_to_zero_like"
    projection_params: dict = {}
    fit_for_alignment: bool = False
    allow_scale_only: bool = True
    max_zeros_compare: int = 5000

    class Config:
        extra = "allow"


class ReferenceConfig(BaseModel):
    """Reference data configuration."""

    zeta_zero_source: str = "mpmath_compute"
    primes_source: str = "sympy_generate"
    max_reference_zeros: int = 5000
    max_reference_primes: int = 200000

    class Config:
        extra = "allow"


class OutputConfig(BaseModel):
    """Output configuration."""

    out_dir: str = "runs/"
    export_bundle: bool = True
    save_parquet: bool = True
    save_figures: bool = True
    figure_format: str = "png"
    cache_enabled: bool = True

    class Config:
        extra = "allow"


class RunConfig(BaseModel):
    """Complete run configuration."""

    run_name: str = "default_run"
    seed: int = 42
    vfd: VFDConfig = None
    prime_field: PrimeFieldConfig = None
    stability: StabilityConfig = None
    bridge: BridgeConfig = None
    reference: ReferenceConfig = None
    output: OutputConfig = None

    class Config:
        extra = "allow"

    def __init__(self, **data):
        # Set defaults for nested configs
        if "vfd" not in data or data["vfd"] is None:
            data["vfd"] = VFDConfig()
        if "prime_field" not in data or data["prime_field"] is None:
            data["prime_field"] = PrimeFieldConfig()
        if "stability" not in data or data["stability"] is None:
            data["stability"] = StabilityConfig()
        if "bridge" not in data or data["bridge"] is None:
            data["bridge"] = BridgeConfig()
        if "reference" not in data or data["reference"] is None:
            data["reference"] = ReferenceConfig()
        if "output" not in data or data["output"] is None:
            data["output"] = OutputConfig()
        super().__init__(**data)

    def to_json(self) -> str:
        """Export configuration to JSON string."""
        return json.dumps(self.to_dict(), indent=2)

    def to_dict(self) -> dict:
        """Export configuration to dictionary."""
        if PYDANTIC_V2:
            return self.model_dump()
        else:
            return self.dict()

    @classmethod
    def from_json(cls, json_str: str) -> "RunConfig":
        """Load configuration from JSON string."""
        data = json.loads(json_str)
        return cls(**data)

    @classmethod
    def from_file(cls, path: str) -> "RunConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            return cls.from_json(f.read())

    def save(self, path: str) -> None:
        """Save configuration to JSON file."""
        with open(path, "w") as f:
            f.write(self.to_json())


def get_default_config() -> RunConfig:
    """Get default run configuration."""
    return RunConfig()
