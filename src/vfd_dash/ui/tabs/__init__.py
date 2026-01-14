"""
Tab Components for Dashboard.
"""

from .structure import create_structure_tab
from .primes import create_primes_tab
from .stability import create_stability_tab
from .shadow import create_shadow_tab
from .audit import create_audit_tab

__all__ = [
    "create_structure_tab",
    "create_primes_tab",
    "create_stability_tab",
    "create_shadow_tab",
    "create_audit_tab",
]
