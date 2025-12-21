"""
Caching layer for Foldex with Redis support and in-memory fallback.

Provides caching for embeddings, retrieval results, and other expensive operations.
"""

import hashlib
import json
import pickle
import time
from functools import lru_cache
from typing import Optional, List, Dict, Any
import structlog

logger = structlog.get_logger(__name__)

# Try to import Redis
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

# Fallback to in-memory cache
from collections import OrderedDict


class FoldexCache:
    """Redis-based caching with in-memory fallback for embeddings and retrieval results."""
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        use_redis: bool = True,
        max_memory_size: int = 1000,
    ):
        """Initialize cache.
        
        Args:
            redis_url: Redis connection URL (e.g., "redis://localhost:6379")
            use_redis: Whether to use Redis (if available)
            max_memory_size: Maximum size for in-memory cache
        """
        self.use_redis = use_redis and REDIS_AVAILABLE
        self.max_memory_size = max_memory_size
        self._hits = 0
        self._misses = 0
        
        if self.use_redis and redis_url:
            try:
                self.redis = redis.from_url(redis_url, decode_responses=False)
                # Test connection
                self.redis.ping()
                self.enabled = True
                logger.info("Redis cache enabled", redis_url=redis_url)
            except Exception as e:
                logger.warning("Redis connection failed, using in-memory cache", error=str(e))
                self.redis = None
                self.enabled = False
                self._init_memory_cache()
        else:
            self.redis = None
            self.enabled = False
            self._init_memory_cache()
    
    def _init_memory_cache(self):
        """Initialize in-memory cache."""
        self._embedding_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._retrieval_cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        logger.info("Using in-memory cache", max_size=self.max_memory_size)
    
    def _hash_key(self, text: str, prefix: str = "") -> str:
        """Generate cache key from text.
        
        Args:
            text: Text to hash
            prefix: Key prefix
            
        Returns:
            Cache key string
        """
        # Normalize: lowercase, strip whitespace
        normalized = text.lower().strip()
        text_hash = hashlib.sha256(normalized.encode()).hexdigest()
        return f"{prefix}:{text_hash}" if prefix else text_hash
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get cached embedding.
        
        Args:
            text: Query text
            
        Returns:
            Cached embedding or None
        """
        key = self._hash_key(text, "emb")
        
        if self.enabled and self.redis:
            try:
                cached = self.redis.get(key)
                if cached:
                    self._hits += 1
                    return pickle.loads(cached)
            except Exception as e:
                logger.warning("Redis get failed", error=str(e))
        
        # Fallback to memory cache
        if key in self._embedding_cache:
            entry = self._embedding_cache[key]
            if not self._is_expired(entry):
                self._embedding_cache.move_to_end(key)
                self._hits += 1
                return entry["value"]
            else:
                del self._embedding_cache[key]
        
        self._misses += 1
        return None
    
    def set_embedding(self, text: str, embedding: List[float], ttl: int = 3600):
        """Cache embedding with TTL.
        
        Args:
            text: Query text
            embedding: Embedding vector
            ttl: Time-to-live in seconds (default: 1 hour)
        """
        key = self._hash_key(text, "emb")
        
        if self.enabled and self.redis:
            try:
                self.redis.setex(key, ttl, pickle.dumps(embedding))
                return
            except Exception as e:
                logger.warning("Redis set failed", error=str(e))
        
        # Fallback to memory cache
        if len(self._embedding_cache) >= self.max_memory_size:
            oldest_key = next(iter(self._embedding_cache))
            del self._embedding_cache[oldest_key]
        
        self._embedding_cache[key] = {
            "value": embedding,
            "timestamp": time.time(),
            "ttl": ttl,
        }
        self._embedding_cache.move_to_end(key)
    
    def get_retrieval(self, query: str, folder_id: Optional[str] = None) -> Optional[List]:
        """Get cached retrieval results.
        
        Args:
            query: Search query
            folder_id: Optional folder ID for scoping
            
        Returns:
            Cached retrieval results or None
        """
        # Include folder_id in key for scoping
        cache_key = f"{folder_id}:{query}" if folder_id else query
        key = self._hash_key(cache_key, "ret")
        
        if self.enabled and self.redis:
            try:
                cached = self.redis.get(key)
                if cached:
                    self._hits += 1
                    return pickle.loads(cached)
            except Exception as e:
                logger.warning("Redis get failed", error=str(e))
        
        # Fallback to memory cache
        if key in self._retrieval_cache:
            entry = self._retrieval_cache[key]
            if not self._is_expired(entry):
                self._retrieval_cache.move_to_end(key)
                self._hits += 1
                return entry["value"]
            else:
                del self._retrieval_cache[key]
        
        self._misses += 1
        return None
    
    def set_retrieval(
        self,
        query: str,
        results: List,
        ttl: int = 600,
        folder_id: Optional[str] = None,
    ):
        """Cache retrieval results.
        
        Args:
            query: Search query
            results: Retrieval results (list of Documents)
            ttl: Time-to-live in seconds (default: 10 minutes)
            folder_id: Optional folder ID for scoping
        """
        # Include folder_id in key for scoping
        cache_key = f"{folder_id}:{query}" if folder_id else query
        key = self._hash_key(cache_key, "ret")
        
        # Serialize results (Documents need special handling)
        try:
            # Convert Documents to dicts for serialization
            serializable_results = []
            for doc in results:
                if hasattr(doc, "page_content") and hasattr(doc, "metadata"):
                    serializable_results.append({
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    })
                else:
                    serializable_results.append(doc)
        except Exception as e:
            logger.warning("Failed to serialize retrieval results", error=str(e))
            return
        
        if self.enabled and self.redis:
            try:
                self.redis.setex(key, ttl, pickle.dumps(serializable_results))
                return
            except Exception as e:
                logger.warning("Redis set failed", error=str(e))
        
        # Fallback to memory cache
        if len(self._retrieval_cache) >= self.max_memory_size:
            oldest_key = next(iter(self._retrieval_cache))
            del self._retrieval_cache[oldest_key]
        
        self._retrieval_cache[key] = {
            "value": serializable_results,
            "timestamp": time.time(),
            "ttl": ttl,
        }
        self._retrieval_cache.move_to_end(key)
    
    def _is_expired(self, entry: Dict[str, Any]) -> bool:
        """Check if cache entry is expired.
        
        Args:
            entry: Cache entry with timestamp and ttl
            
        Returns:
            True if expired
        """
        current_time = time.time()
        ttl = entry.get("ttl", 3600)
        return (current_time - entry["timestamp"]) > ttl
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate.
        
        Returns:
            Hit rate as percentage (0-100)
        """
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return (self._hits / total) * 100
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache stats
        """
        stats = {
            "enabled": self.enabled,
            "redis_enabled": self.enabled and self.redis is not None,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.get_hit_rate(),
        }
        
        if not self.enabled or not self.redis:
            stats["memory_cache_size"] = len(self._embedding_cache) + len(self._retrieval_cache)
            stats["max_memory_size"] = self.max_memory_size
        
        return stats
    
    def clear(self):
        """Clear all caches."""
        if self.enabled and self.redis:
            try:
                # Clear all keys with our prefixes
                for prefix in ["emb:", "ret:"]:
                    keys = self.redis.keys(f"{prefix}*")
                    if keys:
                        self.redis.delete(*keys)
            except Exception as e:
                logger.warning("Redis clear failed", error=str(e))
        
        self._embedding_cache.clear()
        self._retrieval_cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Cache cleared")


# Global cache instance (lazy initialization)
_cache_instance: Optional[FoldexCache] = None


def get_cache() -> FoldexCache:
    """Get global cache instance.
    
    Returns:
        FoldexCache instance
    """
    global _cache_instance
    if _cache_instance is None:
        from app.config.settings import settings
        
        redis_url = getattr(settings, "REDIS_URL", None)
        use_redis = getattr(settings, "USE_REDIS_CACHE", True)
        
        _cache_instance = FoldexCache(
            redis_url=redis_url,
            use_redis=use_redis,
            max_memory_size=getattr(settings, "CACHE_MAX_SIZE", 1000),
        )
    
    return _cache_instance

