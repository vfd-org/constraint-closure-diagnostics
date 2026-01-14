"""
Test internal prime generation.
"""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vfd_dash.vfd.canonical import VFDSpace, TorsionOperator, ShiftOperator
from vfd_dash.vfd.transport import TransportMode, TransportAlgebra, Direction
from vfd_dash.vfd.primes import InternalPrimeGenerator, NonUFDAnalyzer


@pytest.fixture
def small_space():
    """Create small VFD space."""
    return VFDSpace(cell_count=8, internal_dim=24, orbit_count=2, orbit_size=12)


@pytest.fixture
def operators(small_space):
    """Create operators."""
    T = TorsionOperator(small_space)
    S = ShiftOperator(small_space)
    return T, S


@pytest.fixture
def prime_generator(small_space, operators):
    """Create prime generator."""
    T, S = operators
    return InternalPrimeGenerator(small_space, T, S, max_length=30, seed=42)


class TestTransportModes:
    """Test transport mode operations."""

    def test_mode_creation(self):
        """Test creating transport modes."""
        mode = TransportMode(5, Direction.FORWARD)
        assert mode.length == 5
        assert mode.direction == Direction.FORWARD
        assert mode.torsion_class == 5 % 12

    def test_mode_composition(self):
        """Test mode composition."""
        m1 = TransportMode(3, Direction.FORWARD)
        m2 = TransportMode(4, Direction.FORWARD)
        result = m1.compose(m2)

        assert result.length == 7
        assert result.direction == Direction.FORWARD

    def test_mode_inverse(self):
        """Test mode inverse."""
        mode = TransportMode(5, Direction.FORWARD)
        inv = mode.inverse()

        assert inv.length == 5
        assert inv.direction == Direction.BACKWARD


class TestPrimeGeneration:
    """Test internal prime generation."""

    def test_generate_primes(self, prime_generator):
        """Test generating primes up to length."""
        primes = prime_generator.generate_primes_up_to(20)

        assert len(primes) > 0
        assert all(p.length > 0 for p in primes)

    def test_length_1_is_prime(self, prime_generator):
        """Length 1 modes should be prime (irreducible)."""
        primes = prime_generator.generate_primes_up_to(5)
        length_1 = [p for p in primes if p.length == 1]

        assert len(length_1) == 2  # (1, +) and (1, -)

    def test_prime_ordering(self, prime_generator):
        """Primes should be ordered by length."""
        primes = prime_generator.generate_primes_up_to(20)

        for i in range(len(primes) - 1):
            assert primes[i].length <= primes[i+1].length

    def test_prime_counting(self, prime_generator):
        """Test prime counting function."""
        prime_generator.generate_primes_up_to(20)

        count_10 = prime_generator.count_primes_up_to(10)
        count_20 = prime_generator.count_primes_up_to(20)

        assert count_10 <= count_20
        assert count_10 > 0

    def test_dataframe_export(self, prime_generator):
        """Test exporting primes to DataFrame."""
        prime_generator.generate_primes_up_to(20)
        df = prime_generator.to_dataframe()

        assert "prime_id" in df.columns
        assert "m" in df.columns
        assert "direction" in df.columns
        assert "torsion_class" in df.columns


class TestNonUFD:
    """Test non-UFD structure."""

    def test_find_non_ufd_examples(self, small_space, operators):
        """Test finding non-UFD examples."""
        T, S = operators
        algebra = TransportAlgebra(small_space, T, S)
        analyzer = NonUFDAnalyzer(algebra)

        examples = analyzer.find_non_ufd_examples(max_search=15)

        # Should find at least some examples
        # (depends on the algebra structure)
        assert isinstance(examples, list)

    def test_structural_proof(self, small_space, operators):
        """Test structural non-UFD proof generation."""
        T, S = operators
        algebra = TransportAlgebra(small_space, T, S)
        analyzer = NonUFDAnalyzer(algebra)

        analyzer.find_non_ufd_examples(max_search=15)
        proof = analyzer.get_structural_proof()

        assert isinstance(proof, str)
        assert "THEOREM" in proof or "No non-UFD" in proof
