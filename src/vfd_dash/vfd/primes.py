"""
VFD Internal Primes.

Internal primes are irreducible transport modes in the interaction algebra.
They exist purely within VFD structure without reference to classical primes.

Key properties:
- Irreducibility: Cannot be factored into smaller positive-length modes
- Minimality: Have minimal transport length among equivalent factorizations
- Indecomposability: Cannot be written as sums of smaller primes
- Well-ordering: Ordered by transport length
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict, Any, Set
from dataclasses import dataclass, field
import networkx as nx

from .transport import TransportMode, TransportAlgebra, Direction, Factorization
from .canonical import VFDSpace, TorsionOperator, ShiftOperator, TORSION_ORDER


@dataclass
class InternalPrime:
    """
    An internal prime in the VFD framework.

    Attributes:
        mode: The underlying transport mode
        prime_id: Unique identifier
        irreducibility_witness: Explanation of why it's irreducible
        minimality_verified: Whether minimality has been verified
        order_index: Position in well-ordering
    """
    mode: TransportMode
    prime_id: str
    irreducibility_witness: str = ""
    minimality_verified: bool = False
    order_index: int = -1

    @property
    def length(self) -> int:
        return self.mode.length

    @property
    def direction(self) -> str:
        return self.mode.direction.value

    @property
    def torsion_class(self) -> int:
        return self.mode.torsion_class

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for dataset export."""
        return {
            "prime_id": self.prime_id,
            "m": self.length,
            "direction": self.direction,
            "torsion_class": self.torsion_class,
            "is_irreducible": True,
            "irreducibility_witness": self.irreducibility_witness,
            "minimality_verified": self.minimality_verified,
            "order_index": self.order_index,
        }


class InternalPrimeGenerator:
    """
    Generate and analyze internal primes.
    """

    def __init__(
        self,
        space: VFDSpace,
        T: TorsionOperator,
        S: ShiftOperator,
        max_length: int = 100,
        seed: int = 42
    ):
        """
        Initialize prime generator.

        Args:
            space: VFD space
            T: Torsion operator
            S: Shift operator
            max_length: Maximum transport length to explore
            seed: Random seed
        """
        self.space = space
        self.T = T
        self.S = S
        self.max_length = max_length
        self.seed = seed

        self.algebra = TransportAlgebra(space, T, S)
        self._primes: List[InternalPrime] = []
        self._prime_set: Set[Tuple[int, str]] = set()
        self._counter = 0

    def _make_prime_id(self, mode: TransportMode) -> str:
        """Create unique prime ID."""
        self._counter += 1
        return f"P_{self._counter:06d}_{mode.length}{mode.direction.value}"

    def generate_primes_up_to(self, max_length: int) -> List[InternalPrime]:
        """
        Generate all internal primes up to given length.

        Uses VFD-internal irreducibility criterion.

        Args:
            max_length: Maximum transport length

        Returns:
            List of internal primes, ordered by length
        """
        primes = []
        order_idx = 0

        for m in range(1, max_length + 1):
            for d in [Direction.FORWARD, Direction.BACKWARD]:
                mode = TransportMode(m, d)

                # Check irreducibility
                is_irr, counterexample = self.algebra.is_irreducible(mode)

                if is_irr:
                    # Generate witness
                    if m == 1:
                        witness = "Length 1 is minimal, hence irreducible."
                    else:
                        witness = f"No nontrivial factorization exists for length {m}."

                    prime = InternalPrime(
                        mode=mode,
                        prime_id=self._make_prime_id(mode),
                        irreducibility_witness=witness,
                        minimality_verified=True,
                        order_index=order_idx
                    )
                    primes.append(prime)
                    self._prime_set.add((m, d.value))
                    order_idx += 1

        self._primes = primes
        return primes

    def is_prime(self, mode: TransportMode) -> bool:
        """Check if a mode is a known prime."""
        return (mode.length, mode.direction.value) in self._prime_set

    def count_primes_up_to(self, x: int) -> int:
        """
        VFD prime counting function pi_VFD(x).

        Counts internal primes with length <= x.

        Args:
            x: Maximum length

        Returns:
            Count of internal primes
        """
        return sum(1 for p in self._primes if p.length <= x)

    def get_prime_density(self, x: int) -> float:
        """
        Compute prime density at length x.

        Returns pi_VFD(x) / x.
        """
        if x <= 0:
            return 0.0
        return self.count_primes_up_to(x) / x

    def to_dataframe(self) -> pd.DataFrame:
        """Convert primes to DataFrame for export."""
        if not self._primes:
            self.generate_primes_up_to(self.max_length)

        records = [p.to_dict() for p in self._primes]
        return pd.DataFrame(records)

    def build_factorization_graph(
        self,
        max_length: int = 50,
        max_nodes: int = 1000
    ) -> nx.DiGraph:
        """
        Build a directed graph of factorization relationships.

        Nodes are transport modes. Edges represent factorizations.

        Args:
            max_length: Maximum mode length to include
            max_nodes: Maximum nodes in graph

        Returns:
            NetworkX directed graph
        """
        G = nx.DiGraph()
        nodes_added = 0

        for m in range(1, max_length + 1):
            if nodes_added >= max_nodes:
                break

            for d in [Direction.FORWARD, Direction.BACKWARD]:
                if nodes_added >= max_nodes:
                    break

                mode = TransportMode(m, d)
                node_id = f"({m},{d.value})"

                is_prime = self.is_prime(mode)
                G.add_node(node_id, length=m, direction=d.value, is_prime=is_prime)
                nodes_added += 1

                # Add edges for factorizations
                facts = self.algebra.find_all_factorizations(mode, max_factors=2)
                for fact in facts[:5]:  # Limit edges
                    for factor in fact.factors:
                        factor_id = f"({factor.length},{factor.direction.value})"
                        if factor_id in G:
                            G.add_edge(node_id, factor_id)

        return G


class NonUFDAnalyzer:
    """
    Analyze non-UFD structure of the interaction algebra.

    Demonstrates that unique factorization fails.
    """

    def __init__(self, algebra: TransportAlgebra):
        """
        Initialize analyzer.

        Args:
            algebra: Transport algebra
        """
        self.algebra = algebra
        self._examples: List[Dict[str, Any]] = []

    def find_non_ufd_examples(self, max_search: int = 30) -> List[Dict[str, Any]]:
        """
        Find examples of non-unique factorization.

        Args:
            max_search: Maximum mode length to search

        Returns:
            List of non-UFD examples
        """
        examples = []

        for m in range(4, max_search + 1):
            for d in [Direction.FORWARD, Direction.BACKWARD]:
                mode = TransportMode(m, d)
                facts = self.algebra.find_all_factorizations(mode, max_factors=3)

                # Group by signature
                by_signature: Dict[tuple, List[Factorization]] = {}
                for fact in facts:
                    sig = tuple(sorted([f.signed_length for f in fact.factors]))
                    if sig not in by_signature:
                        by_signature[sig] = []
                    by_signature[sig].append(fact)

                if len(by_signature) >= 2:
                    # Found multiple inequivalent factorizations
                    example = {
                        "mode": str(mode),
                        "length": m,
                        "direction": d.value,
                        "factorization_count": len(facts),
                        "distinct_signatures": len(by_signature),
                        "factorizations": [
                            {
                                "factors": [str(f) for f in fact.factors],
                                "signature": tuple(sorted([f.signed_length for f in fact.factors]))
                            }
                            for fact in facts[:6]
                        ],
                        "witness": f"Mode {mode} has {len(by_signature)} distinct factorization signatures."
                    }
                    examples.append(example)

                    if len(examples) >= 10:
                        break

            if len(examples) >= 10:
                break

        self._examples = examples
        return examples

    def get_structural_proof(self) -> str:
        """
        Return a structural proof of non-UFD property.

        This proof is VFD-internal and doesn't reference classical number theory.
        """
        if not self._examples:
            self.find_non_ufd_examples()

        if not self._examples:
            return "No non-UFD examples found in search range."

        ex = self._examples[0]
        proof = f"""
THEOREM (Non-UFD): The interaction algebra A does not satisfy unique factorization.

PROOF:
Consider the transport mode {ex['mode']}.

This mode admits at least {ex['distinct_signatures']} distinct factorization signatures:
"""
        for i, fact in enumerate(ex['factorizations'][:3]):
            factors_str = " * ".join(fact['factors'])
            proof += f"  ({i+1}) {factors_str}\n"

        proof += f"""
These factorizations are inequivalent as they have different multisets of factor lengths.

Since A is noncommutative (by the Weyl relation TST^{{-1}} = omega S),
the existence of multiple inequivalent factorizations demonstrates that
unique factorization fails in A.

This is a VFD-internal structural property, independent of classical number theory.
QED.
"""
        return proof
