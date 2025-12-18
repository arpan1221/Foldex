"""LangChain-compatible caching for expensive operations."""

from typing import Optional, Dict, Any, List
import structlog
import hashlib
import json
import time

try:
    from langchain.cache import InMemoryCache, SQLiteCache
    from langchain.globals import set_llm_cache, get_llm_cache
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    InMemoryCache = None
    SQLiteCache = None
    set_llm_cache = None
    get_llm_cache = None

from app.core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)


class EmbeddingCache:
    """Cache for embedding generation."""

    def __init__(self, ttl: int = 3600):
        """Initialize embedding cache.

        Args:
            ttl: Time to live in seconds
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl
        self.logger = structlog.get_logger(__name__)

    def _generate_key(self, text: str) -> str:
        """Generate cache key for text.

        Args:
            text: Text to cache

        Returns:
            Cache key
        """
        return hashlib.md5(text.encode()).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache.

        Args:
            text: Text to get embedding for

        Returns:
            Embedding vector or None
        """
        try:
            key = self._generate_key(text)
            if key in self.cache:
                entry = self.cache[key]
                # Check if expired
                if time.time() - entry["timestamp"] < self.ttl:
                    self.logger.debug("Cache hit for embedding", key=key[:8])
                    return entry["embedding"]
                else:
                    # Expired, remove
                    del self.cache[key]

            return None

        except Exception as e:
            logger.error("Failed to get from embedding cache", error=str(e))
            return None

    def set(self, text: str, embedding: List[float]) -> None:
        """Store embedding in cache.

        Args:
            text: Text
            embedding: Embedding vector
        """
        try:
            key = self._generate_key(text)
            self.cache[key] = {
                "embedding": embedding,
                "timestamp": time.time(),
            }

            logger.debug("Cached embedding", key=key[:8])

        except Exception as e:
            logger.error("Failed to cache embedding", error=str(e))

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        logger.debug("Cleared embedding cache")


class LLMResponseCache:
    """Cache for LLM responses."""

    def __init__(self, ttl: int = 3600):
        """Initialize LLM response cache.

        Args:
            ttl: Time to live in seconds
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl
        self.logger = structlog.get_logger(__name__)

    def _generate_key(self, prompt: str, **kwargs: Any) -> str:
        """Generate cache key for prompt.

        Args:
            prompt: Prompt text
            **kwargs: Additional parameters

        Returns:
            Cache key
        """
        cache_data = {"prompt": prompt, **kwargs}
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()

    def get(self, prompt: str, **kwargs: Any) -> Optional[str]:
        """Get response from cache.

        Args:
            prompt: Prompt text
            **kwargs: Additional parameters

        Returns:
            Cached response or None
        """
        try:
            key = self._generate_key(prompt, **kwargs)
            if key in self.cache:
                entry = self.cache[key]
                # Check if expired
                if time.time() - entry["timestamp"] < self.ttl:
                    self.logger.debug("Cache hit for LLM response", key=key[:8])
                    return entry["response"]
                else:
                    # Expired, remove
                    del self.cache[key]

            return None

        except Exception as e:
            logger.error("Failed to get from LLM cache", error=str(e))
            return None

    def set(self, prompt: str, response: str, **kwargs: Any) -> None:
        """Store response in cache.

        Args:
            prompt: Prompt text
            response: LLM response
            **kwargs: Additional parameters
        """
        try:
            key = self._generate_key(prompt, **kwargs)
            self.cache[key] = {
                "response": response,
                "timestamp": time.time(),
            }

            logger.debug("Cached LLM response", key=key[:8])

        except Exception as e:
            logger.error("Failed to cache LLM response", error=str(e))

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        logger.debug("Cleared LLM response cache")


class RetrievalCache:
    """Cache for retrieval results."""

    def __init__(self, ttl: int = 1800):
        """Initialize retrieval cache.

        Args:
            ttl: Time to live in seconds
        """
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl
        self.logger = structlog.get_logger(__name__)

    def _generate_key(self, query: str, k: int = 10, **kwargs: Any) -> str:
        """Generate cache key for query.

        Args:
            query: Search query
            k: Number of results
            **kwargs: Additional parameters

        Returns:
            Cache key
        """
        cache_data = {"query": query, "k": k, **kwargs}
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_str.encode()).hexdigest()

    def get(self, query: str, k: int = 10, **kwargs: Any) -> Optional[List[Any]]:
        """Get retrieval results from cache.

        Args:
            query: Search query
            k: Number of results
            **kwargs: Additional parameters

        Returns:
            Cached results or None
        """
        try:
            key = self._generate_key(query, k, **kwargs)
            if key in self.cache:
                entry = self.cache[key]
                # Check if expired
                if time.time() - entry["timestamp"] < self.ttl:
                    self.logger.debug("Cache hit for retrieval", key=key[:8])
                    return entry["results"]
                else:
                    # Expired, remove
                    del self.cache[key]

            return None

        except Exception as e:
            logger.error("Failed to get from retrieval cache", error=str(e))
            return None

    def set(self, query: str, results: List[Any], k: int = 10, **kwargs: Any) -> None:
        """Store retrieval results in cache.

        Args:
            query: Search query
            results: Retrieval results
            k: Number of results
            **kwargs: Additional parameters
        """
        try:
            key = self._generate_key(query, k, **kwargs)
            self.cache[key] = {
                "results": results,
                "timestamp": time.time(),
            }

            logger.debug("Cached retrieval results", key=key[:8])

        except Exception as e:
            logger.error("Failed to cache retrieval results", error=str(e))

    def clear(self) -> None:
        """Clear cache."""
        self.cache.clear()
        logger.debug("Cleared retrieval cache")


class LangChainCacheManager:
    """Manager for LangChain-compatible caching."""

    def __init__(self, use_sqlite: bool = False, cache_path: Optional[str] = None):
        """Initialize LangChain cache manager.

        Args:
            use_sqlite: Whether to use SQLite cache
            cache_path: Optional path for SQLite cache
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        self.use_sqlite = use_sqlite
        self.cache_path = cache_path

        # Initialize LangChain cache
        try:
            if use_sqlite and cache_path:
                self.langchain_cache = SQLiteCache(database_path=cache_path)
            else:
                self.langchain_cache = InMemoryCache()

            # Set as global cache
            set_llm_cache(self.langchain_cache)

            logger.info("Initialized LangChain cache", use_sqlite=use_sqlite)

        except Exception as e:
            logger.error("Failed to initialize LangChain cache", error=str(e))
            self.langchain_cache = None

        # Initialize custom caches
        self.embedding_cache = EmbeddingCache()
        self.llm_cache = LLMResponseCache()
        self.retrieval_cache = RetrievalCache()

    def enable_llm_caching(self) -> None:
        """Enable LLM response caching."""
        try:
            if self.langchain_cache:
                set_llm_cache(self.langchain_cache)
                logger.info("Enabled LLM caching")

        except Exception as e:
            logger.error("Failed to enable LLM caching", error=str(e))

    def disable_llm_caching(self) -> None:
        """Disable LLM response caching."""
        try:
            set_llm_cache(None)
            logger.info("Disabled LLM caching")

        except Exception as e:
            logger.error("Failed to disable LLM caching", error=str(e))

    def clear_all_caches(self) -> None:
        """Clear all caches."""
        try:
            self.embedding_cache.clear()
            self.llm_cache.clear()
            self.retrieval_cache.clear()

            if self.langchain_cache and hasattr(self.langchain_cache, "clear"):
                self.langchain_cache.clear()

            logger.info("Cleared all caches")

        except Exception as e:
            logger.error("Failed to clear caches", error=str(e))

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        return {
            "embedding_cache_size": len(self.embedding_cache.cache),
            "llm_cache_size": len(self.llm_cache.cache),
            "retrieval_cache_size": len(self.retrieval_cache.cache),
            "langchain_cache_enabled": self.langchain_cache is not None,
        }

