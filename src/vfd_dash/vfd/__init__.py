"""
VFD Canonical Framework Implementation.

This module implements the VFD internal universe:
- Torsion operators and projectors
- Shift operators with cocycle twist
- Canonical kernel operator
- Selection rules and invariants

All definitions are VFD-internal with no reference to classical number theory.
"""

from .canonical import VFDSpace, TorsionOperator, ShiftOperator
from .operators import create_torsion_projectors, compute_bidegree, torsion_average
from .kernels import CanonicalKernel, AdmissibleKernel
from .probes import ProbeGenerator, Probe
from .transport import TransportMode, TransportAlgebra

__all__ = [
    "VFDSpace",
    "TorsionOperator",
    "ShiftOperator",
    "create_torsion_projectors",
    "compute_bidegree",
    "torsion_average",
    "CanonicalKernel",
    "AdmissibleKernel",
    "ProbeGenerator",
    "Probe",
    "TransportMode",
    "TransportAlgebra",
]
