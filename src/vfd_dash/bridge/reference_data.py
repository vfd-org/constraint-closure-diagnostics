"""
Reference Data for Classical Comparison.

This module provides classical zeta zeros and prime numbers
for comparison with VFD shadow projections.

All data here is EXTERNAL to VFD - used only for translation validation.
"""

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from typing import List, Optional, Tuple, Dict
from pathlib import Path
import os

# Global cache for singleton pattern - avoids reloading zeros across instances
_GLOBAL_ZEROS_CACHE: Dict[int, NDArray] = {}
_GLOBAL_PRIMES_CACHE: Dict[int, NDArray] = {}

# First 100 zeta zeros (imaginary parts) embedded for instant startup
# These are high-precision values, no mpmath computation needed
EMBEDDED_ZEROS_100 = np.array([
    14.134725141734693, 21.022039638771554, 25.010857580145688, 30.424876125859513,
    32.935061587739189, 37.586178158825671, 40.918719012147495, 43.327073280914999,
    48.005150881167159, 49.773832477672302, 52.970321477714460, 56.446247697063394,
    59.347044002602353, 60.831778524609809, 65.112544048081606, 67.079810529494173,
    69.546401711173979, 72.067157674481907, 75.704690699083933, 77.144840068874805,
    79.337375020249367, 82.910380854086030, 84.735492980517050, 87.425274613125229,
    88.809111207634465, 92.491899270558484, 94.651344040519848, 95.870634228245309,
    98.831194218193692, 101.31785100573139, 103.72553804047833, 105.44662305232897,
    107.16861118427640, 111.02953554316967, 111.87465917699263, 114.32022091545271,
    116.22668032085755, 118.79078286597621, 121.37012500242066, 122.94682929355258,
    124.25681855434864, 127.51668387959649, 129.57870419995605, 131.08768853093265,
    133.49773720299758, 134.75650975337387, 138.11604205453344, 139.73620895212138,
    141.12370740402112, 143.11184580762063, 146.00098248149990, 147.42276534331089,
    150.05352042078298, 150.92525768847491, 153.02469388692735, 156.11290929488189,
    157.59759160078638, 158.84998812287518, 161.18896413873550, 163.03070968698884,
    165.53706943059837, 167.18443997377605, 169.09451541487243, 169.91197647149082,
    173.41153665461373, 174.75419194167414, 176.44143413064684, 178.37740777609739,
    179.91648402026806, 182.20707848436646, 184.87446783224795, 185.59878367571505,
    187.22892258423602, 189.41615865377470, 192.02665636825408, 193.07972660217856,
    195.26539684777033, 196.87648158723787, 198.01530985154437, 201.26475194370653,
    202.49359417134171, 204.18967180042656, 205.39469722753668, 207.90625898159494,
    209.57650932306960, 211.69086259298865, 213.34791936192336, 214.54704478294061,
    216.16953850025384, 219.06759630824481, 220.71491886890145, 221.43070547674078,
    224.00700025280777, 224.98332466958367, 227.42144426657101, 229.33741330917046,
    231.25018869548723, 231.98723519352982, 233.69340352294875, 236.52422967263282,
], dtype=np.float64)


def get_cached_zeros(count: int, data_dir: Optional[Path] = None) -> NDArray:
    """
    Get zeta zeros from global cache, loading from disk/computing if needed.

    This is the preferred way to get zeros - avoids redundant loading.

    Args:
        count: Number of zeros needed
        data_dir: Directory for cached files

    Returns:
        Array of zero imaginary parts
    """
    global _GLOBAL_ZEROS_CACHE

    # Check if we have enough in global cache
    for cached_count, cached_zeros in _GLOBAL_ZEROS_CACHE.items():
        if cached_count >= count:
            return cached_zeros[:count]

    # Load from disk or compute
    loader = ReferenceDataLoader(data_dir=str(data_dir) if data_dir else None, max_zeros=count)
    zeros = loader.get_zeta_zeros(count)

    # Store in global cache
    _GLOBAL_ZEROS_CACHE[count] = zeros

    return zeros


def clear_global_cache():
    """Clear the global zeros/primes cache (for testing)."""
    global _GLOBAL_ZEROS_CACHE, _GLOBAL_PRIMES_CACHE
    _GLOBAL_ZEROS_CACHE.clear()
    _GLOBAL_PRIMES_CACHE.clear()


class ReferenceDataLoader:
    """
    Load reference classical data for shadow comparison.

    This data is external to VFD and used only for Bridge validation.
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        max_zeros: int = 5000,
        max_primes: int = 200000
    ):
        """
        Initialize reference data loader.

        Args:
            data_dir: Directory for cached data files
            max_zeros: Maximum number of zeta zeros
            max_primes: Maximum number of primes
        """
        self.data_dir = Path(data_dir) if data_dir else Path("data/reference")
        self.max_zeros = max_zeros
        self.max_primes = max_primes

        self._zeros: Optional[NDArray] = None
        self._primes: Optional[NDArray] = None

    def get_zeta_zeros(self, count: Optional[int] = None) -> NDArray:
        """
        Get imaginary parts of nontrivial zeta zeros.

        Priority order:
        1. In-memory cache (fastest)
        2. Embedded first 100 zeros (instant, no I/O)
        3. Disk cache (.npy files)
        4. mpmath computation (slow, ~0.1s per zero)
        5. Asymptotic approximation fallback

        Args:
            count: Number of zeros (defaults to max_zeros)

        Returns:
            Array of zero imaginary parts t_n
        """
        if count is None:
            count = self.max_zeros

        # 1. Check in-memory cache
        if self._zeros is not None and len(self._zeros) >= count:
            return self._zeros[:count]

        # 2. Use embedded zeros if count <= 100 (instant, no I/O)
        if count <= 100:
            self._zeros = EMBEDDED_ZEROS_100[:count].copy()
            return self._zeros

        # 3. Try to load from disk cache
        cache_file = self.data_dir / f"zeta_zeros_{count}.npy"
        if cache_file.exists():
            self._zeros = np.load(cache_file)
            return self._zeros[:count]

        # Also check if we have a larger cached file we can use
        for cached_file in sorted(self.data_dir.glob("zeta_zeros_*.npy"), reverse=True):
            try:
                cached_count = int(cached_file.stem.split("_")[-1])
                if cached_count >= count:
                    self._zeros = np.load(cached_file)
                    return self._zeros[:count]
            except (ValueError, IndexError):
                continue

        # 4. Compute using mpmath
        try:
            from mpmath import zetazero, mp
            mp.dps = 25  # 25 decimal places

            # Start with embedded zeros to save computation
            zeros = list(EMBEDDED_ZEROS_100)

            # Compute remaining zeros
            for n in range(101, count + 1):
                z = zetazero(n)
                zeros.append(float(z.imag))

            self._zeros = np.array(zeros[:count])

            # Cache for future use
            self.data_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, self._zeros)

            return self._zeros

        except ImportError:
            # 5. Fallback: use approximate zeros
            return self._approximate_zeros(count)

    def _approximate_zeros(self, count: int) -> NDArray:
        """
        Generate approximate zeros using asymptotic formula.

        This is for testing when mpmath is not available.
        Real analysis requires actual zeros.
        """
        # Approximate: t_n ~ 2*pi*n / ln(n) for large n
        zeros = []
        for n in range(1, count + 1):
            if n == 1:
                t = 14.134725  # First zero
            elif n == 2:
                t = 21.022040
            elif n == 3:
                t = 25.010858
            else:
                # Asymptotic approximation
                t = 2 * np.pi * n / np.log(n)
            zeros.append(t)

        return np.array(zeros)

    def get_primes(self, count: Optional[int] = None) -> NDArray:
        """
        Get first count prime numbers.

        Args:
            count: Number of primes (defaults to max_primes)

        Returns:
            Array of primes
        """
        if count is None:
            count = self.max_primes

        if self._primes is not None and len(self._primes) >= count:
            return self._primes[:count]

        # Try to load from cache
        cache_file = self.data_dir / f"primes_{count}.npy"
        if cache_file.exists():
            self._primes = np.load(cache_file)
            return self._primes[:count]

        # Generate using sympy
        try:
            from sympy import primerange

            primes = list(primerange(2, self._estimate_nth_prime(count) + 1000))
            self._primes = np.array(primes[:count])

            # Cache
            self.data_dir.mkdir(parents=True, exist_ok=True)
            np.save(cache_file, self._primes)

            return self._primes

        except ImportError:
            # Fallback: simple sieve
            return self._sieve_primes(count)

    def _estimate_nth_prime(self, n: int) -> int:
        """Estimate the nth prime using prime number theorem."""
        if n < 6:
            return [2, 3, 5, 7, 11, 13][n]
        # p_n ~ n * (ln(n) + ln(ln(n)))
        import math
        return int(n * (math.log(n) + math.log(math.log(n)) + 2))

    def _sieve_primes(self, count: int) -> NDArray:
        """Simple Sieve of Eratosthenes fallback."""
        limit = self._estimate_nth_prime(count) + 1000
        sieve = [True] * limit
        sieve[0] = sieve[1] = False

        for i in range(2, int(limit**0.5) + 1):
            if sieve[i]:
                for j in range(i*i, limit, i):
                    sieve[j] = False

        primes = [i for i, is_prime in enumerate(sieve) if is_prime]
        return np.array(primes[:count])

    def get_zero_spacings(self, count: Optional[int] = None) -> NDArray:
        """
        Get spacings between consecutive zeros.

        Args:
            count: Number of zeros to use

        Returns:
            Array of spacings delta_n = t_{n+1} - t_n
        """
        zeros = self.get_zeta_zeros(count)
        return np.diff(zeros)

    def get_normalized_spacings(self, count: Optional[int] = None) -> NDArray:
        """
        Get normalized zero spacings (scaled by local mean).

        The normalized spacings should follow GUE statistics.

        Args:
            count: Number of zeros to use

        Returns:
            Array of normalized spacings
        """
        zeros = self.get_zeta_zeros(count)
        spacings = np.diff(zeros)

        # Local normalization using running average
        # Mean spacing ~ 2*pi / ln(t)
        mean_spacings = 2 * np.pi / np.log(zeros[:-1])

        return spacings / mean_spacings

    def prime_counting(self, x: float) -> int:
        """
        Prime counting function pi(x).

        Args:
            x: Upper bound

        Returns:
            Count of primes <= x
        """
        primes = self.get_primes()
        return np.sum(primes <= x)

    def to_zeros_dataframe(self, count: Optional[int] = None) -> pd.DataFrame:
        """Export zeros as DataFrame."""
        zeros = self.get_zeta_zeros(count)
        return pd.DataFrame({
            "index": range(1, len(zeros) + 1),
            "t": zeros,
            "source": "reference_zeta"
        })

    def to_primes_dataframe(self, count: Optional[int] = None) -> pd.DataFrame:
        """Export primes as DataFrame."""
        primes = self.get_primes(count)
        return pd.DataFrame({
            "index": range(1, len(primes) + 1),
            "p": primes,
            "source": "reference_classical"
        })
