"""LangChain chain optimization and resource management."""

from typing import Optional, Dict, Any, List, Callable
import structlog
import asyncio
from enum import Enum

try:
    from langchain.chains import LLMChain
    from langchain.chains.base import Chain
    from langchain_core.callbacks import CallbackManager
    from langchain_core.tokens import TokenUsage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.chains import LLMChain
        from langchain.chains.base import Chain
        from langchain.callbacks.manager import CallbackManager
        LANGCHAIN_AVAILABLE = True
        TokenUsage = None
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        LLMChain = None
        Chain = None
        CallbackManager = None
        TokenUsage = None

from app.core.exceptions import ProcessingError
from app.rag.llm_chains import OllamaLLM
from app.rag.chain_caching import LangChainCacheManager

logger = structlog.get_logger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels."""

    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


class ChainOptimizer:
    """Optimizer for LangChain chains with resource management."""

    def __init__(
        self,
        llm: Optional[OllamaLLM] = None,
        cache_manager: Optional[LangChainCacheManager] = None,
    ):
        """Initialize chain optimizer.

        Args:
            llm: Optional Ollama LLM instance
            cache_manager: Optional cache manager
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        self.llm = llm or OllamaLLM()
        self.cache_manager = cache_manager
        self.logger = structlog.get_logger(__name__)

    def estimate_query_complexity(self, query: str) -> QueryComplexity:
        """Estimate query complexity.

        Args:
            query: User query

        Returns:
            QueryComplexity level
        """
        query_length = len(query)
        word_count = len(query.split())

        # Simple heuristics for complexity
        if word_count < 5 and query_length < 50:
            return QueryComplexity.SIMPLE
        elif word_count < 15 and query_length < 150:
            return QueryComplexity.MODERATE
        elif word_count < 30 and query_length < 300:
            return QueryComplexity.COMPLEX
        else:
            return QueryComplexity.VERY_COMPLEX

    def optimize_chain_for_complexity(
        self,
        chain: Any,
        complexity: QueryComplexity,
    ) -> Any:
        """Optimize chain based on query complexity.

        Args:
            chain: Chain to optimize
            complexity: Query complexity level

        Returns:
            Optimized chain
        """
        try:
            if complexity == QueryComplexity.SIMPLE:
                # Simple queries: use caching, no streaming needed
                if self.cache_manager:
                    self.cache_manager.enable_llm_caching()
                logger.debug("Optimized chain for simple query")

            elif complexity == QueryComplexity.MODERATE:
                # Moderate queries: enable caching and async
                if self.cache_manager:
                    self.cache_manager.enable_llm_caching()
                logger.debug("Optimized chain for moderate query")

            elif complexity == QueryComplexity.COMPLEX:
                # Complex queries: enable streaming, async, caching
                if self.cache_manager:
                    self.cache_manager.enable_llm_caching()
                logger.debug("Optimized chain for complex query")

            else:  # VERY_COMPLEX
                # Very complex: full optimization
                if self.cache_manager:
                    self.cache_manager.enable_llm_caching()
                logger.debug("Optimized chain for very complex query")

            return chain

        except Exception as e:
            logger.error("Chain optimization failed", error=str(e))
            return chain

    def count_tokens(self, text: str) -> int:
        """Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token
        return len(text) // 4

    def check_context_window(
        self,
        query: str,
        context: str,
        max_tokens: int = 2000,
    ) -> bool:
        """Check if content fits in context window.

        Args:
            query: Query text
            context: Context text
            max_tokens: Maximum tokens

        Returns:
            True if fits, False otherwise
        """
        query_tokens = self.count_tokens(query)
        context_tokens = self.count_tokens(context)
        total_tokens = query_tokens + context_tokens

        return total_tokens <= max_tokens

    def truncate_context(
        self,
        context: str,
        max_tokens: int = 2000,
    ) -> str:
        """Truncate context to fit in window.

        Args:
            context: Context to truncate
            max_tokens: Maximum tokens

        Returns:
            Truncated context
        """
        context_tokens = self.count_tokens(context)
        if context_tokens <= max_tokens:
            return context

        # Truncate proportionally
        ratio = max_tokens / context_tokens
        target_length = int(len(context) * ratio)

        # Try to truncate at sentence boundary
        truncated = context[:target_length]
        last_period = truncated.rfind('.')
        if last_period > target_length * 0.8:
            truncated = truncated[:last_period + 1]

        return truncated + "..."

    def optimize_chain_composition(
        self,
        chains: List[Any],
    ) -> Any:
        """Optimize chain composition for reduced latency.

        Args:
            chains: List of chains to compose

        Returns:
            Optimized composed chain
        """
        try:
            # For now, return first chain
            # In production, this would implement actual composition optimization
            if chains:
                return chains[0]

            raise ProcessingError("No chains provided for composition")

        except Exception as e:
            logger.error("Chain composition optimization failed", error=str(e))
            raise ProcessingError(f"Chain composition failed: {str(e)}") from e


class AsyncChainExecutor:
    """Executor for concurrent LangChain chain execution."""

    def __init__(self, max_concurrent: int = 5):
        """Initialize async chain executor.

        Args:
            max_concurrent: Maximum concurrent executions
        """
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.logger = structlog.get_logger(__name__)

    async def execute_chain_async(
        self,
        chain: Any,
        inputs: Dict[str, Any],
        callbacks: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Execute chain asynchronously.

        Args:
            chain: Chain to execute
            inputs: Chain inputs
            callbacks: Optional callbacks

        Returns:
            Chain outputs
        """
        async with self.semaphore:
            try:
                if hasattr(chain, "ainvoke"):
                    result = await chain.ainvoke(inputs, callbacks=callbacks)
                else:
                    # Fallback to sync execution
                    result = chain.invoke(inputs, callbacks=callbacks)

                return result

            except Exception as e:
                logger.error("Async chain execution failed", error=str(e))
                raise ProcessingError(f"Chain execution failed: {str(e)}") from e

    async def execute_chains_concurrent(
        self,
        chains_and_inputs: List[tuple[Any, Dict[str, Any]]],
        callbacks: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Execute multiple chains concurrently.

        Args:
            chains_and_inputs: List of (chain, inputs) tuples
            callbacks: Optional callbacks

        Returns:
            List of results
        """
        try:
            tasks = [
                self.execute_chain_async(chain, inputs, callbacks)
                for chain, inputs in chains_and_inputs
            ]

            gathered_results = await asyncio.gather(*tasks, return_exceptions=True)

            # Filter out exceptions and ensure type
            valid_results: List[Dict[str, Any]] = []
            for r in gathered_results:
                if isinstance(r, Exception):
                    logger.warning("Chain execution exception", error=str(r))
                    continue
                if isinstance(r, dict):
                    valid_results.append(r)
                else:
                    # Convert non-dict results to dict
                    valid_results.append({"result": r})

            logger.info(
                "Concurrent chain execution completed",
                total=len(chains_and_inputs),
                successful=len(valid_results),
            )

            return valid_results

        except Exception as e:
            logger.error("Concurrent chain execution failed", error=str(e))
            raise ProcessingError(f"Concurrent execution failed: {str(e)}") from e


class StreamingChainWrapper:
    """Wrapper for LangChain chains with streaming support."""

    def __init__(self, chain: Any):
        """Initialize streaming chain wrapper.

        Args:
            chain: Chain to wrap
        """
        self.chain = chain
        self.logger = structlog.get_logger(__name__)

    async def stream_response(
        self,
        inputs: Dict[str, Any],
        callback: Optional[Callable[[str], None]] = None,
    ) -> str:
        """Stream chain response.

        Args:
            inputs: Chain inputs
            callback: Optional callback for each token

        Returns:
            Complete response
        """
        try:
            if hasattr(self.chain, "stream"):
                # Use chain's streaming capability
                response_parts = []
                async for chunk in self.chain.astream(inputs):
                    if isinstance(chunk, dict):
                        text = chunk.get("answer", "") or chunk.get("text", "")
                    else:
                        text = str(chunk)

                    if text:
                        response_parts.append(text)
                        if callback:
                            callback(text)

                return "".join(response_parts)

            else:
                # Fallback to non-streaming
                if hasattr(self.chain, "ainvoke"):
                    result = await self.chain.ainvoke(inputs)
                else:
                    result = self.chain.invoke(inputs)

                if isinstance(result, dict):
                    return result.get("answer", "") or result.get("text", "")
                return str(result)

        except Exception as e:
            logger.error("Streaming failed", error=str(e))
            raise ProcessingError(f"Streaming failed: {str(e)}") from e


class RetryChainWrapper:
    """Wrapper for LangChain chains with retry mechanisms."""

    def __init__(
        self,
        chain: Any,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ):
        """Initialize retry chain wrapper.

        Args:
            chain: Chain to wrap
            max_retries: Maximum retry attempts
            retry_delay: Delay between retries in seconds
        """
        self.chain = chain
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = structlog.get_logger(__name__)

    async def invoke_with_retry(
        self,
        inputs: Dict[str, Any],
        callbacks: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Invoke chain with retry logic.

        Args:
            inputs: Chain inputs
            callbacks: Optional callbacks

        Returns:
            Chain outputs
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                if hasattr(self.chain, "ainvoke"):
                    result = await self.chain.ainvoke(inputs, callbacks=callbacks)
                else:
                    result = self.chain.invoke(inputs, callbacks=callbacks)

                logger.debug(
                    "Chain invocation succeeded",
                    attempt=attempt + 1,
                )

                return result

            except Exception as e:
                last_error = e
                logger.warning(
                    "Chain invocation failed, retrying",
                    attempt=attempt + 1,
                    max_retries=self.max_retries,
                    error=str(e),
                )

                if attempt < self.max_retries - 1:
                    await asyncio.sleep(self.retry_delay * (attempt + 1))

        # All retries failed
        logger.error("Chain invocation failed after retries", error=str(last_error))
        raise ProcessingError(f"Chain invocation failed: {str(last_error)}") from last_error


class DynamicChainSelector:
    """Dynamic chain selection based on query complexity and resources."""

    def __init__(
        self,
        chains: Dict[str, Any],
        optimizer: Optional[ChainOptimizer] = None,
    ):
        """Initialize dynamic chain selector.

        Args:
            chains: Dictionary of chain names to chain instances
            optimizer: Optional chain optimizer
        """
        self.chains = chains
        self.optimizer = optimizer or ChainOptimizer()
        self.logger = structlog.get_logger(__name__)

    def select_chain(
        self,
        query: str,
        resource_availability: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, Any]:
        """Select optimal chain based on query and resources.

        Args:
            query: User query
            resource_availability: Optional resource availability info

        Returns:
            Tuple of (chain_name, chain_instance)
        """
        try:
            # Estimate complexity
            complexity = self.optimizer.estimate_query_complexity(query)

            # Check resource availability
            if resource_availability is None:
                resource_availability = {
                    "memory_available": True,
                    "cpu_available": True,
                    "cache_enabled": True,
                }

            # Select chain based on complexity and resources
            if complexity == QueryComplexity.SIMPLE:
                chain_name = "simple_chain"
            elif complexity == QueryComplexity.MODERATE:
                chain_name = "moderate_chain"
            elif complexity == QueryComplexity.COMPLEX:
                chain_name = "complex_chain"
            else:
                chain_name = "very_complex_chain"

            # Fallback to default if specific chain not available
            if chain_name not in self.chains:
                available_chains = list(self.chains.keys())
                if available_chains:
                    chain_name = available_chains[0]
                else:
                    raise ProcessingError("No chains available")

            chain = self.chains[chain_name]

            # Optimize chain
            optimized_chain = self.optimizer.optimize_chain_for_complexity(
                chain=chain,
                complexity=complexity,
            )

            logger.info(
                "Selected chain",
                chain_name=chain_name,
                complexity=complexity.value,
            )

            return chain_name, optimized_chain

        except Exception as e:
            logger.error("Chain selection failed", error=str(e))
            raise ProcessingError(f"Chain selection failed: {str(e)}") from e

