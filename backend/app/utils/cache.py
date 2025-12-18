"""Caching utilities."""

from typing import Optional, Any
from functools import wraps
import hashlib
import json
import time
from pathlib import Path


class Cache:
    """Simple file-based cache."""

    def __init__(self, cache_dir: str = "./data/cache"):
        """Initialize cache.

        Args:
            cache_dir: Cache directory path
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None
        """
        cache_file = self.cache_dir / f"{self._hash_key(key)}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    # Check expiration
                    if data.get("expires", 0) > time.time():
                        return data.get("value")
                    else:
                        cache_file.unlink()  # Remove expired cache
            except Exception:
                pass
        return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds
        """
        cache_file = self.cache_dir / f"{self._hash_key(key)}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(
                    {
                        "value": value,
                        "expires": time.time() + ttl,
                    },
                    f,
                )
        except Exception:
            pass

    def _hash_key(self, key: str) -> str:
        """Hash cache key.

        Args:
            key: Cache key

        Returns:
            Hashed key
        """
        return hashlib.md5(key.encode()).hexdigest()


def cache_result(ttl: int = 3600):
    """Decorator to cache function results.

    Args:
        ttl: Time to live in seconds

    Returns:
        Decorated function
    """
    cache = Cache()

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}:{json.dumps(args, default=str)}:{json.dumps(kwargs, default=str)}"
            cached = cache.get(cache_key)
            if cached is not None:
                return cached
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result

        return wrapper

    return decorator

