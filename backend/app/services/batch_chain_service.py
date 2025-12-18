"""Efficient batch processing with LangChain."""

from typing import Optional, Dict, Any, List, Callable
import structlog
import asyncio

try:
    from langchain.chains.base import Chain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    Chain = None

from app.core.exceptions import ProcessingError
from app.rag.chain_optimization import AsyncChainExecutor, RetryChainWrapper
from app.rag.chain_caching import LangChainCacheManager

logger = structlog.get_logger(__name__)


class BatchChainService:
    """Service for efficient batch processing of LangChain chains."""

    def __init__(
        self,
        max_concurrent: int = 5,
        cache_manager: Optional[LangChainCacheManager] = None,
        enable_retry: bool = True,
    ):
        """Initialize batch chain service.

        Args:
            max_concurrent: Maximum concurrent executions
            cache_manager: Optional cache manager
            enable_retry: Whether to enable retry logic
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        self.max_concurrent = max_concurrent
        self.cache_manager = cache_manager
        self.enable_retry = enable_retry
        self.executor = AsyncChainExecutor(max_concurrent=max_concurrent)
        self.logger = structlog.get_logger(__name__)

    async def process_batch(
        self,
        chain: Any,
        inputs_list: List[Dict[str, Any]],
        callbacks: Optional[List[Any]] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[Dict[str, Any]]:
        """Process batch of inputs.

        Args:
            chain: Chain to use
            inputs_list: List of input dictionaries
            callbacks: Optional callbacks
            progress_callback: Optional progress callback (current, total)

        Returns:
            List of results
        """
        try:
            logger.info(
                "Processing batch",
                input_count=len(inputs_list),
                max_concurrent=self.max_concurrent,
            )

            # Wrap chain with retry if enabled
            if self.enable_retry:
                wrapped_chain = RetryChainWrapper(chain)
            else:
                wrapped_chain = chain

            # Process in batches
            results = []
            total = len(inputs_list)

            for i in range(0, total, self.max_concurrent):
                batch = inputs_list[i:i + self.max_concurrent]

                # Create tasks for batch
                tasks = [
                    self.executor.execute_chain_async(
                        chain=wrapped_chain,
                        inputs=inputs,
                        callbacks=callbacks,
                    )
                    for inputs in batch
                ]

                # Execute batch
                batch_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Filter out exceptions
                valid_results = [
                    r for r in batch_results
                    if not isinstance(r, Exception)
                ]

                results.extend(valid_results)

                # Report progress
                if progress_callback:
                    progress_callback(len(results), total)

                logger.debug(
                    "Batch processed",
                    batch_start=i,
                    batch_end=min(i + self.max_concurrent, total),
                    successful=len(valid_results),
                )

            logger.info(
                "Batch processing completed",
                total=total,
                successful=len(results),
            )

            return results

        except Exception as e:
            logger.error("Batch processing failed", error=str(e), exc_info=True)
            raise ProcessingError(f"Batch processing failed: {str(e)}") from e

    async def process_multiple_chains(
        self,
        chains_and_inputs: List[tuple[Any, Dict[str, Any]]],
        callbacks: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Process multiple chains with different inputs.

        Args:
            chains_and_inputs: List of (chain, inputs) tuples
            callbacks: Optional callbacks

        Returns:
            List of results
        """
        try:
            logger.info(
                "Processing multiple chains",
                chain_count=len(chains_and_inputs),
            )

            # Wrap chains with retry if enabled
            if self.enable_retry:
                wrapped_chains_and_inputs = [
                    (RetryChainWrapper(chain), inputs)
                    for chain, inputs in chains_and_inputs
                ]
            else:
                wrapped_chains_and_inputs = chains_and_inputs

            # Execute concurrently
            gathered_results = await self.executor.execute_chains_concurrent(
                chains_and_inputs=wrapped_chains_and_inputs,
                callbacks=callbacks,
            )

            # Filter and ensure type - execute_chains_concurrent already filters exceptions
            # but we ensure dict type for type safety
            results: List[Dict[str, Any]] = [
                r for r in gathered_results
                if isinstance(r, dict)
            ]

            return results

        except Exception as e:
            logger.error("Multiple chain processing failed", error=str(e))
            raise ProcessingError(f"Multiple chain processing failed: {str(e)}") from e

    async def process_with_priority(
        self,
        chain: Any,
        prioritized_inputs: List[tuple[int, Dict[str, Any]]],
        callbacks: Optional[List[Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Process inputs with priority ordering.

        Args:
            chain: Chain to use
            prioritized_inputs: List of (priority, inputs) tuples (lower priority = higher priority)
            callbacks: Optional callbacks

        Returns:
            List of results in priority order
        """
        try:
            # Sort by priority
            sorted_inputs = sorted(prioritized_inputs, key=lambda x: x[0])

            # Extract inputs
            inputs_list = [inputs for _, inputs in sorted_inputs]

            # Process batch
            results = await self.process_batch(
                chain=chain,
                inputs_list=inputs_list,
                callbacks=callbacks,
            )

            return results

        except Exception as e:
            logger.error("Priority processing failed", error=str(e))
            raise ProcessingError(f"Priority processing failed: {str(e)}") from e

    async def process_with_deduplication(
        self,
        chain: Any,
        inputs_list: List[Dict[str, Any]],
        callbacks: Optional[List[Any]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Process inputs with deduplication.

        Args:
            chain: Chain to use
            inputs_list: List of input dictionaries
            callbacks: Optional callbacks

        Returns:
            Dictionary mapping input hash to result
        """
        try:
            import hashlib
            import json

            # Deduplicate inputs
            seen = {}
            unique_inputs = []
            input_map = {}

            for inputs in inputs_list:
                # Create hash of inputs
                inputs_str = json.dumps(inputs, sort_keys=True)
                inputs_hash = hashlib.md5(inputs_str.encode()).hexdigest()

                if inputs_hash not in seen:
                    seen[inputs_hash] = inputs
                    unique_inputs.append(inputs)
                    input_map[inputs_hash] = [inputs]
                else:
                    input_map[inputs_hash].append(inputs)

            logger.info(
                "Deduplicating inputs",
                original_count=len(inputs_list),
                unique_count=len(unique_inputs),
            )

            # Process unique inputs
            unique_results = await self.process_batch(
                chain=chain,
                inputs_list=unique_inputs,
                callbacks=callbacks,
            )

            # Map results back to all inputs
            results = {}
            for i, inputs_hash in enumerate(seen.keys()):
                result = unique_results[i]
                # Store result for all inputs with this hash
                for inputs in input_map[inputs_hash]:
                    inputs_str = json.dumps(inputs, sort_keys=True)
                    results[inputs_str] = result

            return results

        except Exception as e:
            logger.error("Deduplication processing failed", error=str(e))
            raise ProcessingError(f"Deduplication processing failed: {str(e)}") from e

    def get_batch_statistics(self) -> Dict[str, Any]:
        """Get batch processing statistics.

        Returns:
            Dictionary with statistics
        """
        return {
            "max_concurrent": self.max_concurrent,
            "cache_enabled": self.cache_manager is not None,
            "retry_enabled": self.enable_retry,
        }

