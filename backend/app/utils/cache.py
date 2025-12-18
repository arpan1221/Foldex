"""File caching and temporary storage management."""

from typing import Optional, Any, Dict
from functools import wraps
import hashlib
import json
import time
import shutil
from pathlib import Path
import structlog

from app.config.settings import settings

logger = structlog.get_logger(__name__)


class Cache:
    """Simple file-based cache for key-value storage."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize cache.

        Args:
            cache_dir: Cache directory path (defaults to settings.CACHE_DIR)
        """
        self.cache_dir = Path(cache_dir or settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get(self, key: str) -> Optional[Any]:
        """Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found or expired
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
                        logger.debug("Cache expired", key=key)
            except Exception as e:
                logger.warning("Failed to read cache", key=key, error=str(e))
        return None

    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        """Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON serializable)
            ttl: Time to live in seconds
        """
        cache_file = self.cache_dir / f"{self._hash_key(key)}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(
                    {
                        "value": value,
                        "expires": time.time() + ttl,
                        "created_at": time.time(),
                    },
                    f,
                )
            logger.debug("Cache set", key=key, ttl=ttl)
        except Exception as e:
            logger.warning("Failed to write cache", key=key, error=str(e))

    def delete(self, key: str) -> bool:
        """Delete value from cache.

        Args:
            key: Cache key

        Returns:
            True if deleted, False if not found
        """
        cache_file = self.cache_dir / f"{self._hash_key(key)}.json"
        if cache_file.exists():
            cache_file.unlink()
            logger.debug("Cache deleted", key=key)
            return True
        return False

    def clear(self) -> int:
        """Clear all cache entries.

        Returns:
            Number of files deleted
        """
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                cache_file.unlink()
                count += 1
            except Exception:
                pass
        logger.info("Cache cleared", deleted_count=count)
        return count

    def _hash_key(self, key: str) -> str:
        """Hash cache key.

        Args:
            key: Cache key

        Returns:
            Hashed key
        """
        return hashlib.md5(key.encode()).hexdigest()


class FileCache:
    """File-based cache for downloaded files with metadata."""

    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize file cache.

        Args:
            cache_dir: Cache directory path (defaults to settings.CACHE_DIR)
        """
        self.cache_dir = Path(cache_dir or settings.CACHE_DIR)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.cache_dir / ".metadata.json"
        self.metadata: Dict[str, Dict[str, Any]] = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Load cache metadata.

        Returns:
            Metadata dictionary
        """
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Failed to load cache metadata", error=str(e))
        return {}

    def _save_metadata(self) -> None:
        """Save cache metadata."""
        try:
            with open(self.metadata_file, "w") as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            logger.warning("Failed to save cache metadata", error=str(e))

    def get_file_path(self, file_id: str, user_id: str) -> Optional[Path]:
        """Get cached file path if exists and not expired.

        Args:
            file_id: File identifier
            user_id: User identifier

        Returns:
            File path if cached, None otherwise
        """
        cache_key = f"{user_id}_{file_id}"
        metadata = self.metadata.get(cache_key)

        if not metadata:
            return None

        file_path = self.cache_dir / user_id / metadata.get("filename", file_id)

        # Check if file exists and not expired
        if file_path.exists():
            expires_at = metadata.get("expires_at", 0)
            if expires_at > time.time():
                logger.debug("Cache hit", file_id=file_id, user_id=user_id)
                return file_path
            else:
                # Expired, remove
                logger.debug("Cache expired", file_id=file_id, user_id=user_id)
                self.delete_file(file_id, user_id)
                return None

        return None

    def store_file(
        self,
        file_id: str,
        user_id: str,
        file_path: Path,
        ttl: int = 86400,  # 24 hours default
    ) -> Path:
        """Store file in cache with metadata.

        Args:
            file_id: File identifier
            user_id: User identifier
            file_path: Source file path
            ttl: Time to live in seconds

        Returns:
            Cached file path
        """
        cache_key = f"{user_id}_{file_id}"
        user_cache_dir = self.cache_dir / user_id
        user_cache_dir.mkdir(parents=True, exist_ok=True)

        # Determine cached filename
        cached_filename = file_path.name
        cached_path = user_cache_dir / cached_filename

        # Copy file to cache (only if not already there)
        if file_path.resolve() != cached_path.resolve():
            shutil.copy2(file_path, cached_path)

        # Store metadata
        self.metadata[cache_key] = {
            "file_id": file_id,
            "user_id": user_id,
            "filename": cached_filename,
            "original_path": str(file_path),
            "size": cached_path.stat().st_size,
            "created_at": time.time(),
            "expires_at": time.time() + ttl,
        }

        self._save_metadata()

        logger.info(
            "File cached",
            file_id=file_id,
            user_id=user_id,
            path=str(cached_path),
            size=cached_path.stat().st_size,
        )

        return cached_path

    def delete_file(self, file_id: str, user_id: str) -> bool:
        """Delete cached file.

        Args:
            file_id: File identifier
            user_id: User identifier

        Returns:
            True if deleted, False if not found
        """
        cache_key = f"{user_id}_{file_id}"
        metadata = self.metadata.get(cache_key)

        if not metadata:
            return False

        filename = metadata.get("filename")
        if filename:
            file_path = self.cache_dir / user_id / filename
            if file_path.exists():
                file_path.unlink()

        del self.metadata[cache_key]
        self._save_metadata()

        logger.debug("File cache deleted", file_id=file_id, user_id=user_id)
        return True

    def cleanup_expired(self) -> int:
        """Clean up expired cache entries.

        Returns:
            Number of files deleted
        """
        count = 0
        expired_keys = []

        for cache_key, metadata in self.metadata.items():
            if metadata.get("expires_at", 0) < time.time():
                file_id = metadata.get("file_id")
                user_id = metadata.get("user_id")
                if self.delete_file(file_id, user_id):
                    count += 1
                expired_keys.append(cache_key)

        logger.info("Cache cleanup completed", deleted_count=count)
        return count

    def get_cache_size(self, user_id: Optional[str] = None) -> int:
        """Get total cache size in bytes.

        Args:
            user_id: Optional user ID to filter by

        Returns:
            Total size in bytes
        """
        total_size = 0
        cache_dir = self.cache_dir / user_id if user_id else self.cache_dir

        if cache_dir.exists():
            for file_path in cache_dir.rglob("*"):
                if file_path.is_file() and file_path.name != ".metadata.json":
                    total_size += file_path.stat().st_size

        return total_size

    def clear_user_cache(self, user_id: str) -> int:
        """Clear all cache for a user.

        Args:
            user_id: User identifier

        Returns:
            Number of files deleted
        """
        count = 0
        user_cache_dir = self.cache_dir / user_id

        if user_cache_dir.exists():
            for file_path in user_cache_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()
                    count += 1
            user_cache_dir.rmdir()

        # Remove from metadata
        keys_to_remove = [
            key for key, metadata in self.metadata.items()
            if metadata.get("user_id") == user_id
        ]
        for key in keys_to_remove:
            del self.metadata[key]

        self._save_metadata()

        logger.info("User cache cleared", user_id=user_id, deleted_count=count)
        return count


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
                logger.debug("Cache hit", function=func.__name__)
                return cached
            result = await func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            logger.debug("Cache miss, stored", function=func.__name__)
            return result

        return wrapper

    return decorator
