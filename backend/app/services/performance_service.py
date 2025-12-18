"""LangChain performance analysis and optimization service."""

from typing import Optional, Dict, Any, List
import structlog

from app.core.exceptions import ProcessingError
from app.monitoring.langchain_metrics import (
    PerformanceMetricsHandler,
    MemoryTrackingHandler,
    BottleneckDetector,
)
from app.rag.chain_caching import LangChainCacheManager
from app.utils.langchain_optimization import ChainProfiler, ChainOptimizer
from app.rag.llm_chains import OllamaLLM

logger = structlog.get_logger(__name__)


class PerformanceService:
    """Service for LangChain performance analysis and optimization."""

    def __init__(
        self,
        llm: Optional[OllamaLLM] = None,
        enable_caching: bool = True,
        cache_path: Optional[str] = None,
    ):
        """Initialize performance service.

        Args:
            llm: Optional Ollama LLM instance
            enable_caching: Whether to enable caching
            cache_path: Optional path for SQLite cache

        Raises:
            ProcessingError: If initialization fails
        """
        self.llm = llm

        # Initialize cache manager
        self.cache_manager: Optional[LangChainCacheManager] = None
        try:
            self.cache_manager = LangChainCacheManager(
                use_sqlite=(cache_path is not None),
                cache_path=cache_path,
            )

            if enable_caching:
                self.cache_manager.enable_llm_caching()

            logger.info("Initialized performance service with caching")

        except Exception as e:
            logger.error("Failed to initialize cache manager", error=str(e))
            self.cache_manager = None

        # Initialize profiler and optimizer
        self.profiler = ChainProfiler()
        self.optimizer = ChainOptimizer(cache_manager=self.cache_manager)

        # Initialize metrics handlers
        self.metrics_handler = PerformanceMetricsHandler()
        self.memory_handler = MemoryTrackingHandler()
        self.bottleneck_detector = BottleneckDetector()

        logger.info("Initialized performance service")

    def get_performance_callbacks(self) -> List[Any]:
        """Get list of performance tracking callbacks.

        Returns:
            List of callback handlers
        """
        return [
            self.metrics_handler,
            self.memory_handler,
            self.bottleneck_detector,
        ]

    async def analyze_chain_performance(
        self,
        chain: Any,
        inputs: Dict[str, Any],
        use_async: bool = True,
    ) -> Dict[str, Any]:
        """Analyze chain performance.

        Args:
            chain: LangChain chain to analyze
            inputs: Chain inputs
            use_async: Whether to use async profiling

        Returns:
            Dictionary with performance analysis
        """
        try:
            logger.info("Analyzing chain performance")

            # Profile chain
            if use_async and hasattr(chain, "ainvoke"):
                profile = await self.profiler.profile_chain_async(
                    chain=chain,
                    inputs=inputs,
                    callbacks=self.get_performance_callbacks(),
                )
            else:
                profile = self.profiler.profile_chain(
                    chain=chain,
                    inputs=inputs,
                    callbacks=self.get_performance_callbacks(),
                )

            # Get optimization recommendations
            recommendations = self.optimizer.get_optimization_recommendations(profile)

            # Get cache stats
            cache_stats = {}
            if self.cache_manager:
                cache_stats = self.cache_manager.get_cache_stats()

            analysis = {
                "profile": profile,
                "recommendations": recommendations,
                "cache_stats": cache_stats,
                "memory_stats": self.memory_handler.get_memory_stats(),
            }

            logger.info(
                "Chain performance analyzed",
                total_time=profile.get("total_time", 0.0),
                recommendation_count=len(recommendations),
            )

            return analysis

        except Exception as e:
            logger.error("Performance analysis failed", error=str(e), exc_info=True)
            raise ProcessingError(f"Performance analysis failed: {str(e)}") from e

    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get collected performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        try:
            metrics = self.metrics_handler.get_metrics()
            memory_stats = self.memory_handler.get_memory_stats()
            bottlenecks = self.bottleneck_detector.get_bottlenecks()
            cache_stats = {}

            if self.cache_manager:
                cache_stats = self.cache_manager.get_cache_stats()

            return {
                "metrics": metrics,
                "memory_stats": memory_stats,
                "bottlenecks": bottlenecks,
                "cache_stats": cache_stats,
            }

        except Exception as e:
            logger.error("Failed to get performance metrics", error=str(e))
            return {}

    def optimize_chain(
        self,
        chain: Any,
        optimization_options: Optional[Dict[str, bool]] = None,
    ) -> Any:
        """Optimize chain based on options.

        Args:
            chain: Chain to optimize
            optimization_options: Optional optimization options

        Returns:
            Optimized chain
        """
        try:
            if optimization_options is None:
                optimization_options = {
                    "streaming": True,
                    "async": True,
                    "caching": True,
                }

            # Apply optimizations
            if optimization_options.get("streaming", False):
                chain = self.optimizer.optimize_for_streaming(chain)

            if optimization_options.get("async", False):
                chain = self.optimizer.optimize_for_async(chain)

            if optimization_options.get("caching", False):
                chain = self.optimizer.enable_caching(chain)

            logger.info("Chain optimized", options=optimization_options)

            return chain

        except Exception as e:
            logger.error("Chain optimization failed", error=str(e))
            return chain

    async def batch_optimize_processing(
        self,
        chain: Any,
        inputs_list: List[Dict[str, Any]],
        max_concurrent: int = 5,
    ) -> List[Dict[str, Any]]:
        """Optimize batch processing.

        Args:
            chain: Chain to use
            inputs_list: List of inputs
            max_concurrent: Maximum concurrent executions

        Returns:
            List of results
        """
        try:
            logger.info(
                "Optimizing batch processing",
                input_count=len(inputs_list),
                max_concurrent=max_concurrent,
            )

            results = await self.optimizer.batch_process(
                chain=chain,
                inputs_list=inputs_list,
                max_concurrent=max_concurrent,
            )

            return results

        except Exception as e:
            logger.error("Batch optimization failed", error=str(e))
            raise ProcessingError(f"Batch optimization failed: {str(e)}") from e

    def clear_metrics(self) -> None:
        """Clear collected metrics."""
        try:
            self.metrics_handler.clear_metrics()
            self.bottleneck_detector.clear_bottlenecks()
            logger.info("Cleared performance metrics")

        except Exception as e:
            logger.error("Failed to clear metrics", error=str(e))

    def get_cache_hit_rate(self) -> float:
        """Get cache hit rate.

        Returns:
            Cache hit rate (0.0 to 1.0)
        """
        try:
            # This would track cache hits/misses
            # For now, return placeholder
            return 0.0

        except Exception as e:
            logger.error("Failed to get cache hit rate", error=str(e))
            return 0.0

    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report.

        Returns:
            Dictionary with performance report
        """
        try:
            metrics = self.get_performance_metrics()
            cache_stats = self.cache_manager.get_cache_stats() if self.cache_manager else {}

            report = {
                "summary": {
                    "total_llm_calls": metrics.get("metrics", {}).get("llm_calls", 0),
                    "total_retriever_calls": metrics.get("metrics", {}).get("retriever_calls", 0),
                    "total_chain_calls": metrics.get("metrics", {}).get("chain_calls", 0),
                    "total_tokens": metrics.get("metrics", {}).get("total_tokens", 0),
                    "total_latency": metrics.get("metrics", {}).get("total_latency", 0.0),
                    "bottleneck_count": len(metrics.get("bottlenecks", [])),
                },
                "detailed_metrics": metrics,
                "cache_statistics": cache_stats,
                "recommendations": self._generate_recommendations(metrics),
            }

            logger.info("Generated performance report")

            return report

        except Exception as e:
            logger.error("Failed to generate performance report", error=str(e))
            return {"error": str(e)}

    def _generate_recommendations(
        self,
        metrics: Dict[str, Any],
    ) -> List[str]:
        """Generate optimization recommendations.

        Args:
            metrics: Performance metrics

        Returns:
            List of recommendations
        """
        recommendations = []

        metrics_data = metrics.get("metrics", {})
        bottlenecks = metrics.get("bottlenecks", [])

        # Check for bottlenecks
        if bottlenecks:
            recommendations.append("Performance bottlenecks detected - review bottleneck details")

        # Check latency
        avg_llm_latency = metrics_data.get("avg_llm_latency", 0.0)
        if avg_llm_latency > 3.0:
            recommendations.append("High LLM latency - consider enabling caching or using faster model")

        avg_retriever_latency = metrics_data.get("avg_retriever_latency", 0.0)
        if avg_retriever_latency > 2.0:
            recommendations.append("High retriever latency - consider caching retrieval results")

        # Check token usage
        total_tokens = metrics_data.get("total_tokens", 0)
        if total_tokens > 50000:
            recommendations.append("High token usage - consider prompt optimization")

        # Check cache usage
        cache_stats = metrics.get("cache_stats", {})
        if cache_stats.get("langchain_cache_enabled", False):
            recommendations.append("Caching is enabled - monitor cache hit rates")

        return recommendations


# Global performance service instance
_performance_service: Optional[PerformanceService] = None


def get_performance_service(
    llm: Optional[OllamaLLM] = None,
    enable_caching: bool = True,
) -> PerformanceService:
    """Get global performance service instance.

    Args:
        llm: Optional LLM instance
        enable_caching: Whether to enable caching

    Returns:
        PerformanceService instance
    """
    global _performance_service
    if _performance_service is None:
        _performance_service = PerformanceService(
            llm=llm,
            enable_caching=enable_caching,
        )
    return _performance_service

