"""Embedding cache for query optimization."""

import hashlib
import time
from typing import Optional, List, Dict, Any
from collections import OrderedDict
import structlog

logger = structlog.get_logger(__name__)


class EmbeddingCache:
    """
    LRU cache for query embeddings with TTL support.
    
    Caches embeddings for identical/similar queries to reduce latency.
    Uses query hash as key and stores embeddings with timestamps.
    """

    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        """
        Initialize embedding cache.
        
        Args:
            max_size: Maximum number of embeddings to cache
            ttl_seconds: Time-to-live for cached embeddings (default: 1 hour)
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._hits = 0
        self._misses = 0
        
        logger.info(
            "Initialized EmbeddingCache",
            max_size=max_size,
            ttl_seconds=ttl_seconds,
        )

    def _hash_query(self, query: str) -> str:
        """
        Generate a hash for the query.
        
        Args:
            query: Query text to hash
            
        Returns:
            SHA256 hash of the normalized query
        """
        # Normalize query: lowercase, strip whitespace
        normalized = query.lower().strip()
        return hashlib.sha256(normalized.encode()).hexdigest()

    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """
        Check if a cache entry has expired.
        
        Args:
            entry: Cache entry with timestamp
            
        Returns:
            True if entry is expired, False otherwise
        """
        current_time = time.time()
        return (current_time - entry["timestamp"]) > self.ttl_seconds

    def get(self, query: str) -> Optional[List[float]]:
        """
        Get cached embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            Cached embedding if found and not expired, None otherwise
        """
        query_hash = self._hash_query(query)
        
        if query_hash in self._cache:
            entry = self._cache[query_hash]
            
            # Check if expired
            if self._is_expired(entry):
                logger.debug(
                    "Cache entry expired",
                    query_hash=query_hash[:16],
                )
                del self._cache[query_hash]
                self._misses += 1
                return None
            
            # Move to end (most recently used)
            self._cache.move_to_end(query_hash)
            self._hits += 1
            
            logger.debug(
                "Cache hit",
                query_hash=query_hash[:16],
                hit_rate=self.get_hit_rate(),
            )
            
            return entry["embedding"]
        
        self._misses += 1
        logger.debug(
            "Cache miss",
            query_hash=query_hash[:16],
            hit_rate=self.get_hit_rate(),
        )
        return None

    def put(self, query: str, embedding: List[float]) -> None:
        """
        Cache an embedding for a query.
        
        Args:
            query: Query text
            embedding: Query embedding vector
        """
        query_hash = self._hash_query(query)
        
        # Remove oldest entry if cache is full
        if len(self._cache) >= self.max_size and query_hash not in self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            logger.debug(
                "Cache full, evicted oldest entry",
                cache_size=len(self._cache),
            )
        
        # Add or update entry
        self._cache[query_hash] = {
            "embedding": embedding,
            "timestamp": time.time(),
        }
        
        # Move to end (most recently used)
        self._cache.move_to_end(query_hash)
        
        logger.debug(
            "Cached embedding",
            query_hash=query_hash[:16],
            cache_size=len(self._cache),
        )

    def clear(self) -> None:
        """Clear all cached embeddings."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Cleared embedding cache")

    def get_hit_rate(self) -> float:
        """
        Calculate cache hit rate.
        
        Returns:
            Hit rate as a percentage (0-100)
        """
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return (self._hits / total) * 100

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.get_hit_rate(),
            "ttl_seconds": self.ttl_seconds,
        }

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        expired_keys = [
            key for key, entry in self._cache.items()
            if self._is_expired(entry)
        ]
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.info(
                "Cleaned up expired cache entries",
                removed_count=len(expired_keys),
                remaining_count=len(self._cache),
            )
        
        return len(expired_keys)


# Global cache instance
_embedding_cache: Optional[EmbeddingCache] = None


def get_embedding_cache() -> EmbeddingCache:
    """Get the global embedding cache instance."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache(max_size=1000, ttl_seconds=3600)
    return _embedding_cache

