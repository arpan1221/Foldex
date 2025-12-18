"""Time-To-First-Token (TTFT) optimization strategies for RAG system.

This module provides comprehensive TTFT optimization including:
- Prompt caching to reduce repeated processing
- Context window optimization
- Model pre-warming on startup
- Streaming improvements
"""

from typing import Optional, Dict, Any, List, Callable
import hashlib
import time
import structlog
from dataclasses import dataclass
from collections import OrderedDict

try:
    from langchain_community.chat_models import ChatOllama
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.chat_models import ChatOllama
        from langchain.prompts import PromptTemplate
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        ChatOllama = None
        PromptTemplate = None

from app.config.settings import settings

logger = structlog.get_logger(__name__)


@dataclass
class CachedPromptContext:
    """Cached prompt context to reduce TTFT."""

    system_prompt: str
    query_type: str
    cache_key: str
    created_at: float
    use_count: int = 0
    last_used: float = 0.0


class PromptCache:
    """LRU cache for prompt contexts to reduce TTFT.

    This cache stores pre-formatted system prompts and static context
    to avoid reprocessing on every request.
    """

    def __init__(self, max_size: int = 100, ttl_seconds: int = 3600):
        """Initialize prompt cache.

        Args:
            max_size: Maximum number of cached prompts
            ttl_seconds: Time-to-live for cached entries in seconds
        """
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: OrderedDict[str, CachedPromptContext] = OrderedDict()
        self._hits = 0
        self._misses = 0

        logger.info(
            "Initialized PromptCache",
            max_size=max_size,
            ttl_seconds=ttl_seconds
        )

    def _generate_cache_key(self, query_type: str, folder_id: Optional[str] = None) -> str:
        """Generate cache key for prompt context.

        Args:
            query_type: Type of query (factual, synthesis, etc.)
            folder_id: Optional folder identifier for folder-specific caching

        Returns:
            Cache key string
        """
        key_parts = [query_type]
        if folder_id:
            key_parts.append(folder_id)

        key_string = ":".join(key_parts)
        return hashlib.md5(key_string.encode()).hexdigest()[:16]

    def get(self, query_type: str, folder_id: Optional[str] = None) -> Optional[CachedPromptContext]:
        """Get cached prompt context if available.

        Args:
            query_type: Type of query
            folder_id: Optional folder identifier

        Returns:
            Cached prompt context or None if not found/expired
        """
        cache_key = self._generate_cache_key(query_type, folder_id)

        if cache_key not in self.cache:
            self._misses += 1
            logger.debug("Prompt cache miss", cache_key=cache_key)
            return None

        cached = self.cache[cache_key]

        # Check if expired
        if time.time() - cached.created_at > self.ttl_seconds:
            logger.debug("Prompt cache entry expired", cache_key=cache_key)
            del self.cache[cache_key]
            self._misses += 1
            return None

        # Move to end (most recently used)
        self.cache.move_to_end(cache_key)

        # Update usage stats
        cached.use_count += 1
        cached.last_used = time.time()

        self._hits += 1
        logger.debug(
            "Prompt cache hit",
            cache_key=cache_key,
            use_count=cached.use_count
        )

        return cached

    def put(
        self,
        query_type: str,
        system_prompt: str,
        folder_id: Optional[str] = None
    ) -> CachedPromptContext:
        """Store prompt context in cache.

        Args:
            query_type: Type of query
            system_prompt: System prompt to cache
            folder_id: Optional folder identifier

        Returns:
            Cached prompt context
        """
        cache_key = self._generate_cache_key(query_type, folder_id)

        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            logger.debug("Evicted oldest cache entry", key=oldest_key)

        cached = CachedPromptContext(
            system_prompt=system_prompt,
            query_type=query_type,
            cache_key=cache_key,
            created_at=time.time(),
            use_count=0,
            last_used=time.time()
        )

        self.cache[cache_key] = cached

        logger.debug("Stored prompt in cache", cache_key=cache_key)

        return cached

    def clear(self):
        """Clear all cached prompts."""
        self.cache.clear()
        self._hits = 0
        self._misses = 0
        logger.info("Cleared prompt cache")

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        total_requests = self._hits + self._misses
        hit_rate = self._hits / total_requests if total_requests > 0 else 0.0

        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": hit_rate,
            "total_requests": total_requests
        }


class ContextOptimizer:
    """Optimizes context window to reduce TTFT while maintaining quality."""

    def __init__(
        self,
        max_context_chars: int = 6000,
        max_documents: int = 5,
        truncate_long_docs: bool = True,
        doc_char_limit: int = 1500
    ):
        """Initialize context optimizer.

        Args:
            max_context_chars: Maximum total characters in context
            max_documents: Maximum number of documents to include
            truncate_long_docs: Whether to truncate long documents
            doc_char_limit: Character limit per document if truncating
        """
        self.max_context_chars = max_context_chars
        self.max_documents = max_documents
        self.truncate_long_docs = truncate_long_docs
        self.doc_char_limit = doc_char_limit

        logger.info(
            "Initialized ContextOptimizer",
            max_context_chars=max_context_chars,
            max_documents=max_documents
        )

    def optimize_documents(
        self,
        documents: List[Any],
        query: str
    ) -> List[Any]:
        """Optimize document list to reduce context size.

        Args:
            documents: List of retrieved documents
            query: User query for relevance scoring

        Returns:
            Optimized list of documents
        """
        if not documents:
            return []

        # Limit number of documents
        docs = documents[:self.max_documents]

        # Truncate long documents if enabled
        if self.truncate_long_docs:
            optimized_docs = []
            total_chars = 0

            for doc in docs:
                # Get content
                if hasattr(doc, "page_content"):
                    content = doc.page_content
                elif hasattr(doc, "content"):
                    content = doc.content
                else:
                    content = str(doc)

                # Truncate if needed
                if len(content) > self.doc_char_limit:
                    # Try to truncate at sentence boundary
                    truncated = content[:self.doc_char_limit]
                    last_period = truncated.rfind(".")
                    if last_period > self.doc_char_limit * 0.7:  # At least 70% retained
                        truncated = truncated[:last_period + 1]
                    else:
                        truncated = truncated + "..."

                    # Update document
                    if hasattr(doc, "page_content"):
                        doc.page_content = truncated
                    elif hasattr(doc, "content"):
                        doc.content = truncated

                    content = truncated

                # Check if we've exceeded total character limit
                if total_chars + len(content) > self.max_context_chars:
                    # Include partial document if it fits at least 50%
                    remaining = self.max_context_chars - total_chars
                    if remaining > len(content) * 0.5:
                        truncated = content[:remaining] + "..."
                        if hasattr(doc, "page_content"):
                            doc.page_content = truncated
                        elif hasattr(doc, "content"):
                            doc.content = truncated
                        optimized_docs.append(doc)
                    break

                total_chars += len(content)
                optimized_docs.append(doc)

            logger.debug(
                "Optimized context",
                original_docs=len(documents),
                optimized_docs=len(optimized_docs),
                total_chars=total_chars
            )

            return optimized_docs

        return docs

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count from text.

        Uses rough approximation: 1 token ≈ 4 characters for English text.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        return len(text) // 4


class ModelPrewarmer:
    """Pre-warms LLM model on startup to reduce first-request latency."""

    def __init__(self, llm: Optional[ChatOllama] = None):
        """Initialize model pre-warmer.

        Args:
            llm: ChatOllama instance to pre-warm
        """
        self.llm = llm
        self.is_warmed = False

    async def warmup(
        self,
        warmup_prompt: str = "Hello, I am ready to assist you with your documents.",
        timeout: int = 30
    ):
        """Pre-warm the model with a simple prompt.

        Args:
            warmup_prompt: Simple prompt to warm up model
            timeout: Timeout in seconds
        """
        if not self.llm:
            logger.warning("No LLM provided for warmup, skipping")
            return

        if self.is_warmed:
            logger.debug("Model already warmed, skipping")
            return

        logger.info("Starting model warmup")
        start_time = time.time()

        try:
            # Simple invoke to load model into memory
            result = await self.llm.ainvoke(warmup_prompt)

            elapsed = time.time() - start_time
            self.is_warmed = True

            logger.info(
                "Model warmup completed",
                elapsed_seconds=round(elapsed, 2)
            )

        except Exception as e:
            logger.error(
                "Model warmup failed",
                error=str(e),
                elapsed_seconds=round(time.time() - start_time, 2)
            )
            # Don't fail startup, just log


class TTFTOptimizer:
    """Main TTFT optimization orchestrator.

    Combines prompt caching, context optimization, and pre-warming
    to minimize time-to-first-token.
    """

    def __init__(
        self,
        enable_prompt_cache: bool = True,
        enable_context_optimization: bool = True,
        prompt_cache_size: int = 100,
        max_context_chars: int = 6000,
        max_documents: int = 5
    ):
        """Initialize TTFT optimizer.

        Args:
            enable_prompt_cache: Enable prompt caching
            enable_context_optimization: Enable context optimization
            prompt_cache_size: Maximum size of prompt cache
            max_context_chars: Maximum characters in context
            max_documents: Maximum number of documents
        """
        self.enable_prompt_cache = enable_prompt_cache
        self.enable_context_optimization = enable_context_optimization

        # Initialize components
        self.prompt_cache = PromptCache(max_size=prompt_cache_size) if enable_prompt_cache else None
        self.context_optimizer = ContextOptimizer(
            max_context_chars=max_context_chars,
            max_documents=max_documents
        ) if enable_context_optimization else None

        logger.info(
            "Initialized TTFTOptimizer",
            prompt_cache=enable_prompt_cache,
            context_optimization=enable_context_optimization
        )

    def get_cached_prompt(
        self,
        query_type: str,
        folder_id: Optional[str] = None
    ) -> Optional[str]:
        """Get cached system prompt if available.

        Args:
            query_type: Type of query
            folder_id: Optional folder identifier

        Returns:
            Cached system prompt or None
        """
        if not self.prompt_cache:
            return None

        cached = self.prompt_cache.get(query_type, folder_id)
        return cached.system_prompt if cached else None

    def cache_prompt(
        self,
        query_type: str,
        system_prompt: str,
        folder_id: Optional[str] = None
    ):
        """Cache system prompt for future use.

        Args:
            query_type: Type of query
            system_prompt: System prompt to cache
            folder_id: Optional folder identifier
        """
        if not self.prompt_cache:
            return

        self.prompt_cache.put(query_type, system_prompt, folder_id)

    def optimize_context(
        self,
        documents: List[Any],
        query: str
    ) -> List[Any]:
        """Optimize context to reduce size.

        Args:
            documents: Retrieved documents
            query: User query

        Returns:
            Optimized documents
        """
        if not self.context_optimizer:
            return documents

        return self.context_optimizer.optimize_documents(documents, query)

    def get_stats(self) -> Dict[str, Any]:
        """Get optimization statistics.

        Returns:
            Dictionary with stats
        """
        stats = {
            "enabled": {
                "prompt_cache": self.enable_prompt_cache,
                "context_optimization": self.enable_context_optimization
            }
        }

        if self.prompt_cache:
            stats["prompt_cache"] = self.prompt_cache.get_stats()

        return stats


# Global TTFT optimizer instance
_ttft_optimizer: Optional[TTFTOptimizer] = None


def get_ttft_optimizer() -> TTFTOptimizer:
    """Get global TTFT optimizer instance.

    Returns:
        TTFTOptimizer instance
    """
    global _ttft_optimizer
    if _ttft_optimizer is None:
        _ttft_optimizer = TTFTOptimizer()
    return _ttft_optimizer


async def warmup_model(llm: ChatOllama):
    """Warmup model on startup.

    This should be called during application startup to reduce
    first-request latency.

    Args:
        llm: ChatOllama instance to warm up
    """
    warmer = ModelPrewarmer(llm)
    await warmer.warmup()
