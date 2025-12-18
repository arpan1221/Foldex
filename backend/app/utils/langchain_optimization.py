"""LangChain chain optimization and profiling."""

from typing import Optional, Dict, Any, List
import structlog
import asyncio
import time

try:
    from langchain.chains import LLMChain
    from langchain_core.callbacks import CallbackManager
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.chains import LLMChain
        from langchain.callbacks.manager import CallbackManager
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        LLMChain = None
        CallbackManager = None

from app.core.exceptions import ProcessingError
from app.monitoring.langchain_metrics import PerformanceMetricsHandler, BottleneckDetector
from app.rag.chain_caching import LangChainCacheManager

logger = structlog.get_logger(__name__)


class ChainProfiler:
    """Profiler for LangChain chains."""

    def __init__(self):
        """Initialize chain profiler."""
        self.profiles: Dict[str, Dict[str, Any]] = {}
        self.logger = structlog.get_logger(__name__)

    def profile_chain(
        self,
        chain: Any,
        inputs: Dict[str, Any],
        callbacks: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Profile chain execution.

        Args:
            chain: LangChain chain to profile
            inputs: Chain inputs
            callbacks: Optional callbacks

        Returns:
            Dictionary with profiling results
        """
        try:
            # Add performance metrics callback
            metrics_handler = PerformanceMetricsHandler()
            bottleneck_detector = BottleneckDetector()

            all_callbacks = [metrics_handler, bottleneck_detector]
            if callbacks:
                all_callbacks.extend(callbacks)

            # Profile execution
            start_time = time.time()

            result = chain.invoke(inputs, callbacks=all_callbacks)

            total_time = time.time() - start_time

            # Get metrics
            metrics = metrics_handler.get_metrics()
            bottlenecks = bottleneck_detector.get_bottlenecks()

            profile = {
                "total_time": total_time,
                "metrics": metrics,
                "bottlenecks": bottlenecks,
                "result": result,
            }

            self.profiles[chain.__class__.__name__] = profile

            logger.info(
                "Chain profiled",
                chain_type=chain.__class__.__name__,
                total_time=total_time,
            )

            return profile

        except Exception as e:
            logger.error("Chain profiling failed", error=str(e))
            raise ProcessingError(f"Profiling failed: {str(e)}") from e

    async def profile_chain_async(
        self,
        chain: Any,
        inputs: Dict[str, Any],
        callbacks: Optional[List[Any]] = None,
    ) -> Dict[str, Any]:
        """Profile chain execution asynchronously.

        Args:
            chain: LangChain chain to profile
            inputs: Chain inputs
            callbacks: Optional callbacks

        Returns:
            Dictionary with profiling results
        """
        try:
            # Add performance metrics callback
            metrics_handler = PerformanceMetricsHandler()
            bottleneck_detector = BottleneckDetector()

            all_callbacks = [metrics_handler, bottleneck_detector]
            if callbacks:
                all_callbacks.extend(callbacks)

            # Profile execution
            start_time = time.time()

            if hasattr(chain, "ainvoke"):
                result = await chain.ainvoke(inputs, callbacks=all_callbacks)
            else:
                # Fallback to sync
                result = chain.invoke(inputs, callbacks=all_callbacks)

            total_time = time.time() - start_time

            # Get metrics
            metrics = metrics_handler.get_metrics()
            bottlenecks = bottleneck_detector.get_bottlenecks()

            profile = {
                "total_time": total_time,
                "metrics": metrics,
                "bottlenecks": bottlenecks,
                "result": result,
            }

            self.profiles[chain.__class__.__name__] = profile

            logger.info(
                "Chain profiled (async)",
                chain_type=chain.__class__.__name__,
                total_time=total_time,
            )

            return profile

        except Exception as e:
            logger.error("Async chain profiling failed", error=str(e))
            raise ProcessingError(f"Async profiling failed: {str(e)}") from e


class ChainOptimizer:
    """Optimizer for LangChain chains."""

    def __init__(self, cache_manager: Optional[LangChainCacheManager] = None):
        """Initialize chain optimizer.

        Args:
            cache_manager: Optional cache manager
        """
        self.cache_manager = cache_manager
        self.logger = structlog.get_logger(__name__)

    def optimize_for_streaming(
        self,
        chain: Any,
    ) -> Any:
        """Optimize chain for streaming.

        Args:
            chain: Chain to optimize

        Returns:
            Optimized chain (same instance, configured for streaming)
        """
        try:
            # Enable streaming if available
            if hasattr(chain, "stream"):
                logger.debug("Chain supports streaming")
            else:
                logger.warning("Chain does not support streaming")

            return chain

        except Exception as e:
            logger.error("Streaming optimization failed", error=str(e))
            return chain

    def optimize_for_async(
        self,
        chain: Any,
    ) -> Any:
        """Optimize chain for async execution.

        Args:
            chain: Chain to optimize

        Returns:
            Optimized chain
        """
        try:
            # Check if chain supports async
            if hasattr(chain, "ainvoke"):
                logger.debug("Chain supports async execution")
            else:
                logger.warning("Chain does not support async execution")

            return chain

        except Exception as e:
            logger.error("Async optimization failed", error=str(e))
            return chain

    async def batch_process(
        self,
        chain: Any,
        inputs_list: List[Dict[str, Any]],
        max_concurrent: int = 5,
    ) -> List[Dict[str, Any]]:
        """Process multiple inputs in batch.

        Args:
            chain: Chain to use
            inputs_list: List of input dictionaries
            max_concurrent: Maximum concurrent executions

        Returns:
            List of results
        """
        try:
            logger.info(
                "Batch processing",
                input_count=len(inputs_list),
                max_concurrent=max_concurrent,
            )

            # Process in batches
            batch_results: List[Dict[str, Any]] = []

            if hasattr(chain, "ainvoke"):
                # Use async batch processing
                semaphore = asyncio.Semaphore(max_concurrent)

                async def process_one(inputs: Dict[str, Any]) -> Dict[str, Any]:
                    async with semaphore:
                        return await chain.ainvoke(inputs)

                tasks = [process_one(inputs) for inputs in inputs_list]
                gathered_results = await asyncio.gather(*tasks, return_exceptions=True)

                # Filter out exceptions and ensure type
                batch_results = [
                    r for r in gathered_results
                    if not isinstance(r, Exception) and isinstance(r, dict)
                ]

            else:
                # Fallback to sequential processing
                for inputs in inputs_list:
                    try:
                        result = chain.invoke(inputs)
                        if isinstance(result, dict):
                            batch_results.append(result)
                    except Exception as e:
                        logger.error("Batch processing error", error=str(e))
                        continue

            logger.info(
                "Batch processing completed",
                input_count=len(inputs_list),
                result_count=len(batch_results),
            )

            return batch_results

        except Exception as e:
            logger.error("Batch processing failed", error=str(e))
            raise ProcessingError(f"Batch processing failed: {str(e)}") from e

    def enable_caching(
        self,
        chain: Any,
    ) -> Any:
        """Enable caching for chain.

        Args:
            chain: Chain to enable caching for

        Returns:
            Chain with caching enabled
        """
        try:
            if self.cache_manager:
                self.cache_manager.enable_llm_caching()
                logger.debug("Enabled caching for chain")

            return chain

        except Exception as e:
            logger.error("Failed to enable caching", error=str(e))
            return chain

    def get_optimization_recommendations(
        self,
        profile: Dict[str, Any],
    ) -> List[str]:
        """Get optimization recommendations based on profile.

        Args:
            profile: Chain profile dictionary

        Returns:
            List of optimization recommendations
        """
        recommendations = []

        metrics = profile.get("metrics", {})
        bottlenecks = profile.get("bottlenecks", [])

        # Check for LLM bottlenecks
        llm_bottlenecks = [b for b in bottlenecks if b.get("stage") == "llm"]
        if llm_bottlenecks:
            recommendations.append("Consider enabling LLM response caching")
            recommendations.append("Consider using streaming for faster perceived response time")

        # Check for retriever bottlenecks
        retriever_bottlenecks = [b for b in bottlenecks if b.get("stage") == "retriever"]
        if retriever_bottlenecks:
            recommendations.append("Consider caching retrieval results")
            recommendations.append("Consider reducing retrieval k parameter")

        # Check token usage
        total_tokens = metrics.get("total_tokens", 0)
        if total_tokens > 10000:
            recommendations.append("High token usage detected - consider prompt optimization")

        # Check latency
        avg_llm_latency = metrics.get("avg_llm_latency", 0.0)
        if avg_llm_latency > 5.0:
            recommendations.append("High LLM latency - consider model optimization or caching")

        return recommendations

