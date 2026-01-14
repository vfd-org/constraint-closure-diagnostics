"""
Results caching for faster reruns.
"""

import pickle
import hashlib
from pathlib import Path
from typing import Any, Optional
from datetime import datetime, timedelta


class ResultsCache:
    """
    Cache computed results for faster reruns.
    """

    def __init__(
        self,
        cache_dir: str = ".cache",
        max_age_hours: int = 24,
        enabled: bool = True
    ):
        """
        Initialize cache.

        Args:
            cache_dir: Cache directory
            max_age_hours: Maximum cache age in hours
            enabled: Whether caching is enabled
        """
        self.cache_dir = Path(cache_dir)
        self.max_age = timedelta(hours=max_age_hours)
        self.enabled = enabled

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = str((args, sorted(kwargs.items())))
        return hashlib.sha256(key_data.encode()).hexdigest()[:16]

    def _get_path(self, key: str) -> Path:
        """Get cache file path for key."""
        return self.cache_dir / f"{key}.pkl"

    def get(self, key: str) -> Optional[Any]:
        """
        Get cached value.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/expired
        """
        if not self.enabled:
            return None

        path = self._get_path(key)
        if not path.exists():
            return None

        # Check age
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        if datetime.now() - mtime > self.max_age:
            path.unlink()
            return None

        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception:
            return None

    def set(self, key: str, value: Any) -> None:
        """
        Set cached value.

        Args:
            key: Cache key
            value: Value to cache
        """
        if not self.enabled:
            return

        path = self._get_path(key)
        try:
            with open(path, "wb") as f:
                pickle.dump(value, f)
        except Exception:
            pass

    def clear(self) -> int:
        """
        Clear all cached values.

        Returns:
            Number of items cleared
        """
        count = 0
        for path in self.cache_dir.glob("*.pkl"):
            path.unlink()
            count += 1
        return count

    def cached(self, func):
        """
        Decorator for caching function results.

        Usage:
            @cache.cached
            def expensive_computation(x, y):
                ...
        """
        def wrapper(*args, **kwargs):
            key = self._get_key(func.__name__, *args, **kwargs)
            result = self.get(key)
            if result is not None:
                return result

            result = func(*args, **kwargs)
            self.set(key, result)
            return result

        return wrapper
