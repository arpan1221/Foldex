"""LangChain callback handlers for performance tracking."""

from typing import Dict, Any, List
import structlog
import time
from collections import defaultdict

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.schema import LLMResult
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler
        from langchain.schema import LLMResult
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        BaseCallbackHandler = None
        LLMResult = None

from app.core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)


class PerformanceMetricsHandler(BaseCallbackHandler):
    """LangChain callback handler for tracking performance metrics."""

    def __init__(self):
        """Initialize performance metrics handler."""
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        super().__init__()
        self.metrics: Dict[str, Any] = {
            "llm_calls": 0,
            "retriever_calls": 0,
            "chain_calls": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_latency": 0.0,
            "llm_latency": 0.0,
            "retriever_latency": 0.0,
            "chain_latency": 0.0,
            "memory_usage": 0,
            "errors": [],
        }
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.logger = structlog.get_logger(__name__)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Handle LLM start event.

        Args:
            serialized: Serialized LLM configuration
            prompts: Input prompts
            **kwargs: Additional arguments
        """
        try:
            self.metrics["llm_calls"] += 1
            self.timings["llm_start"] = [time.time()]  # Store as list for consistency

            # Track prompt tokens
            total_prompt_length = sum(len(p) for p in prompts)
            estimated_tokens = total_prompt_length // 4  # Rough estimate
            self.metrics["prompt_tokens"] += estimated_tokens

        except Exception as e:
            logger.error("Error in LLM start callback", error=str(e))

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Handle LLM end event.

        Args:
            response: LLM result
            **kwargs: Additional arguments
        """
        try:
            if "llm_start" in self.timings and self.timings["llm_start"]:
                elapsed = time.time() - self.timings["llm_start"][0]
                self.metrics["llm_latency"] += elapsed
                self.metrics["total_latency"] += elapsed
                self.timings["llm_latency"].append(elapsed)

            # Track token usage
            if hasattr(response, "llm_output") and response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})
                self.metrics["total_tokens"] += token_usage.get("total_tokens", 0)
                self.metrics["prompt_tokens"] += token_usage.get("prompt_tokens", 0)
                self.metrics["completion_tokens"] += token_usage.get(
                    "completion_tokens", 0
                )

            elapsed = time.time() - self.timings["llm_start"][0] if "llm_start" in self.timings and self.timings["llm_start"] else 0.0
            logger.debug("LLM call completed", latency=elapsed)

        except Exception as e:
            logger.error("Error in LLM end callback", error=str(e))

    def on_retriever_start(
        self,
        query: str,
        **kwargs: Any,
    ) -> None:
        """Handle retriever start event.

        Args:
            query: Search query
            **kwargs: Additional arguments
        """
        try:
            self.metrics["retriever_calls"] += 1
            self.timings["retriever_start"] = [time.time()]  # Store as list

        except Exception as e:
            logger.error("Error in retriever start callback", error=str(e))

    def on_retriever_end(
        self,
        documents: List[Any],
        **kwargs: Any,
    ) -> None:
        """Handle retriever end event.

        Args:
            documents: Retrieved documents
            **kwargs: Additional arguments
        """
        try:
            if "retriever_start" in self.timings and self.timings["retriever_start"]:
                elapsed = time.time() - self.timings["retriever_start"][0]
                self.metrics["retriever_latency"] += elapsed
                self.metrics["total_latency"] += elapsed
                self.timings["retriever_latency"].append(elapsed)

            elapsed = time.time() - self.timings["retriever_start"][0] if "retriever_start" in self.timings and self.timings["retriever_start"] else 0.0
            logger.debug(
                "Retriever call completed",
                document_count=len(documents),
                latency=elapsed,
            )

        except Exception as e:
            logger.error("Error in retriever end callback", error=str(e))

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Handle chain start event.

        Args:
            serialized: Serialized chain configuration
            inputs: Chain inputs
            **kwargs: Additional arguments
        """
        try:
            self.metrics["chain_calls"] += 1
            self.timings["chain_start"] = [time.time()]  # Store as list

        except Exception as e:
            logger.error("Error in chain start callback", error=str(e))

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Handle chain end event.

        Args:
            outputs: Chain outputs
            **kwargs: Additional arguments
        """
        try:
            if "chain_start" in self.timings and self.timings["chain_start"]:
                elapsed = time.time() - self.timings["chain_start"][0]
                self.metrics["chain_latency"] += elapsed
                self.metrics["total_latency"] += elapsed
                self.timings["chain_latency"].append(elapsed)

            elapsed = time.time() - self.timings["chain_start"][0] if "chain_start" in self.timings and self.timings["chain_start"] else 0.0
            logger.debug("Chain call completed", latency=elapsed)

        except Exception as e:
            logger.error("Error in chain end callback", error=str(e))

    def on_chain_error(
        self,
        error: Exception,
        **kwargs: Any,
    ) -> None:
        """Handle chain error event.

        Args:
            error: Exception that occurred
            **kwargs: Additional arguments
        """
        try:
            self.metrics["errors"].append({
                "error_type": type(error).__name__,
                "error_message": str(error),
                "timestamp": time.time(),
            })

            logger.error("Chain error", error=str(error))

        except Exception as e:
            logger.error("Error in chain error callback", error=str(e))

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        metrics = self.metrics.copy()

        # Calculate averages
        if self.metrics["llm_calls"] > 0:
            metrics["avg_llm_latency"] = (
                self.metrics["llm_latency"] / self.metrics["llm_calls"]
            )
        else:
            metrics["avg_llm_latency"] = 0.0

        if self.metrics["retriever_calls"] > 0:
            metrics["avg_retriever_latency"] = (
                self.metrics["retriever_latency"] / self.metrics["retriever_calls"]
            )
        else:
            metrics["avg_retriever_latency"] = 0.0

        if self.metrics["chain_calls"] > 0:
            metrics["avg_chain_latency"] = (
                self.metrics["chain_latency"] / self.metrics["chain_calls"]
            )
        else:
            metrics["avg_chain_latency"] = 0.0

        # Add timing statistics
        metrics["timing_stats"] = {
            "llm_latency": {
                "min": min(self.timings["llm_latency"]) if self.timings["llm_latency"] else 0.0,
                "max": max(self.timings["llm_latency"]) if self.timings["llm_latency"] else 0.0,
                "avg": sum(self.timings["llm_latency"]) / len(self.timings["llm_latency"]) if self.timings["llm_latency"] else 0.0,
            },
            "retriever_latency": {
                "min": min(self.timings["retriever_latency"]) if self.timings["retriever_latency"] else 0.0,
                "max": max(self.timings["retriever_latency"]) if self.timings["retriever_latency"] else 0.0,
                "avg": sum(self.timings["retriever_latency"]) / len(self.timings["retriever_latency"]) if self.timings["retriever_latency"] else 0.0,
            },
            "chain_latency": {
                "min": min(self.timings["chain_latency"]) if self.timings["chain_latency"] else 0.0,
                "max": max(self.timings["chain_latency"]) if self.timings["chain_latency"] else 0.0,
                "avg": sum(self.timings["chain_latency"]) / len(self.timings["chain_latency"]) if self.timings["chain_latency"] else 0.0,
            },
        }

        return metrics

    def clear_metrics(self) -> None:
        """Clear collected metrics."""
        self.metrics = {
            "llm_calls": 0,
            "retriever_calls": 0,
            "chain_calls": 0,
            "total_tokens": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_latency": 0.0,
            "llm_latency": 0.0,
            "retriever_latency": 0.0,
            "chain_latency": 0.0,
            "memory_usage": 0,
            "errors": [],
        }
        self.timings.clear()
        logger.debug("Cleared performance metrics")


class MemoryTrackingHandler(BaseCallbackHandler):
    """LangChain callback handler for memory usage tracking."""

    def __init__(self):
        """Initialize memory tracking handler."""
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        super().__init__()
        self.memory_snapshots: List[Dict[str, Any]] = []
        self.logger = structlog.get_logger(__name__)

    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Handle chain start event.

        Args:
            serialized: Serialized chain configuration
            inputs: Chain inputs
            **kwargs: Additional arguments
        """
        try:
            try:
                import psutil
                import os

                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()

                self.memory_snapshots.append({
                    "stage": "chain_start",
                    "rss": memory_info.rss,
                    "vms": memory_info.vms,
                    "timestamp": time.time(),
                })
            except ImportError:
                # psutil not available, skip memory tracking
                pass
        except Exception as e:
            logger.error("Error in memory tracking", error=str(e))

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Handle chain end event.

        Args:
            outputs: Chain outputs
            **kwargs: Additional arguments
        """
        try:
            try:
                import psutil
                import os

                process = psutil.Process(os.getpid())
                memory_info = process.memory_info()

                self.memory_snapshots.append({
                    "stage": "chain_end",
                    "rss": memory_info.rss,
                    "vms": memory_info.vms,
                    "timestamp": time.time(),
                })
            except ImportError:
                # psutil not available, skip memory tracking
                pass
        except Exception as e:
            logger.error("Error in memory tracking", error=str(e))

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary with memory statistics
        """
        if not self.memory_snapshots:
            return {"snapshots": []}

        return {
            "snapshots": self.memory_snapshots,
            "peak_rss": max(s["rss"] for s in self.memory_snapshots) if self.memory_snapshots else 0,
            "peak_vms": max(s["vms"] for s in self.memory_snapshots) if self.memory_snapshots else 0,
        }


class BottleneckDetector(BaseCallbackHandler):
    """LangChain callback handler for detecting performance bottlenecks."""

    def __init__(self, threshold_seconds: float = 5.0):
        """Initialize bottleneck detector.

        Args:
            threshold_seconds: Threshold in seconds for bottleneck detection
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        super().__init__()
        self.threshold = threshold_seconds
        self.bottlenecks: List[Dict[str, Any]] = []
        self.stage_timings: Dict[str, float] = {}
        self.logger = structlog.get_logger(__name__)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        **kwargs: Any,
    ) -> None:
        """Handle LLM start event."""
        self.stage_timings["llm"] = time.time()

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Handle LLM end event."""
        if "llm" in self.stage_timings:
            elapsed = time.time() - self.stage_timings["llm"]
            if elapsed > self.threshold:
                self.bottlenecks.append({
                    "stage": "llm",
                    "latency": elapsed,
                    "threshold": self.threshold,
                    "timestamp": time.time(),
                })
                logger.warning(
                    "LLM bottleneck detected",
                    latency=elapsed,
                    threshold=self.threshold,
                )

    def on_retriever_start(
        self,
        query: str,
        **kwargs: Any,
    ) -> None:
        """Handle retriever start event."""
        self.stage_timings["retriever"] = time.time()

    def on_retriever_end(
        self,
        documents: List[Any],
        **kwargs: Any,
    ) -> None:
        """Handle retriever end event."""
        if "retriever" in self.stage_timings:
            elapsed = time.time() - self.stage_timings["retriever"]
            if elapsed > self.threshold:
                self.bottlenecks.append({
                    "stage": "retriever",
                    "latency": elapsed,
                    "threshold": self.threshold,
                    "timestamp": time.time(),
                })
                logger.warning(
                    "Retriever bottleneck detected",
                    latency=elapsed,
                    threshold=self.threshold,
                )

    def get_bottlenecks(self) -> List[Dict[str, Any]]:
        """Get detected bottlenecks.

        Returns:
            List of bottleneck dictionaries
        """
        return self.bottlenecks.copy()

    def clear_bottlenecks(self) -> None:
        """Clear detected bottlenecks."""
        self.bottlenecks.clear()
        self.stage_timings.clear()

