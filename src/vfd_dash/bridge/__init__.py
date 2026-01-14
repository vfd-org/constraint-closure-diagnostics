"""
Bridge Module: VFD to Classical Shadow Projection.

The Bridge Axiom is the single external input connecting VFD to classical RH.
All content here is labeled as TRANSLATION, not VFD-internal.

Key components:
- Bridge Axiom (BA): Identification of VFD spectral data with zeta zeros
- Bridge Negations (BN1-BN3): Falsification controls
- Projection: Mapping VFD data to classical shadow
- Reference Data: Classical zeta zeros and primes for comparison
"""

from .bridge_axiom import BridgeAxiom, BridgeMode
from .bn_negations import BN1_Negation, BN2_Negation, BN3_Negation
from .projection import ShadowProjection, ProjectedZero
from .reference_data import ReferenceDataLoader, get_cached_zeros, EMBEDDED_ZEROS_100

__all__ = [
    "BridgeAxiom",
    "BridgeMode",
    "BN1_Negation",
    "BN2_Negation",
    "BN3_Negation",
    "ShadowProjection",
    "ProjectedZero",
    "ReferenceDataLoader",
    "get_cached_zeros",
    "EMBEDDED_ZEROS_100",
]
