"""LangChain callbacks for quality metrics collection."""

from typing import List, Dict, Any, Optional
import structlog

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


class QualityMetricsCallbackHandler(BaseCallbackHandler):
    """Callback handler for collecting quality metrics during chain execution."""

    def __init__(self):
        """Initialize quality metrics callback handler."""
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        super().__init__()
        self.metrics: Dict[str, Any] = {
            "response_length": 0,
            "token_count": 0,
            "citation_count": 0,
            "source_diversity": 0,
            "processing_time": 0.0,
            "errors": [],
        }
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
            import time
            self.metrics["llm_start_time"] = time.time()
            self.metrics["prompt_count"] = len(prompts)

        except Exception as e:
            logger.error("Error in LLM start callback", error=str(e))

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        """Handle LLM end event.

        Args:
            response: LLM result
            **kwargs: Additional arguments
        """
        try:
            import time
            if "llm_start_time" in self.metrics:
                elapsed = time.time() - self.metrics["llm_start_time"]
                self.metrics["processing_time"] = elapsed

            # Count tokens
            if hasattr(response, "llm_output") and response.llm_output:
                token_usage = response.llm_output.get("token_usage", {})
                self.metrics["token_count"] = token_usage.get("total_tokens", 0)

            logger.debug("LLM completed", metrics=self.metrics)

        except Exception as e:
            logger.error("Error in LLM end callback", error=str(e))

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
            # Track query length
            if "question" in inputs:
                self.metrics["query_length"] = len(inputs["question"])

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
            # Track response length
            if "answer" in outputs:
                self.metrics["response_length"] = len(outputs["answer"])

            # Track citations
            if "citations" in outputs:
                self.metrics["citation_count"] = len(outputs["citations"])

            # Calculate source diversity
            if "source_documents" in outputs:
                sources = outputs["source_documents"]
                unique_files = set()
                for doc in sources:
                    if hasattr(doc, "metadata"):
                        file_id = doc.metadata.get("file_id")
                        if file_id:
                            unique_files.add(file_id)
                self.metrics["source_diversity"] = len(unique_files)

            logger.debug("Chain completed", metrics=self.metrics)

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
            })

            logger.error("Chain error", error=str(error))

        except Exception as e:
            logger.error("Error in chain error callback", error=str(e))

    def get_metrics(self) -> Dict[str, Any]:
        """Get collected quality metrics.

        Returns:
            Dictionary with quality metrics
        """
        return self.metrics.copy()

    def clear_metrics(self) -> None:
        """Clear collected metrics."""
        self.metrics = {
            "response_length": 0,
            "token_count": 0,
            "citation_count": 0,
            "source_diversity": 0,
            "processing_time": 0.0,
            "errors": [],
        }
        logger.debug("Cleared quality metrics")


class ResponseQualityTracker:
    """Tracks response quality metrics across multiple dimensions."""

    def __init__(self):
        """Initialize quality tracker."""
        self.metrics_history: List[Dict[str, Any]] = []
        self.logger = structlog.get_logger(__name__)

    def track_response(
        self,
        response: str,
        citations: List[Dict[str, Any]],
        source_documents: List[Any],
        metrics: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Track response quality metrics.

        Args:
            response: Generated response
            citations: List of citations
            source_documents: List of source documents
            metrics: Optional additional metrics

        Returns:
            Dictionary with quality metrics
        """
        try:
            quality_metrics = {
                "response_length": len(response),
                "citation_count": len(citations),
                "source_count": len(source_documents),
                "source_diversity": self._calculate_source_diversity(source_documents),
                "citation_coverage": self._calculate_citation_coverage(response, citations),
            }

            if metrics:
                quality_metrics.update(metrics)

            self.metrics_history.append(quality_metrics)

            logger.debug("Tracked response quality", metrics=quality_metrics)

            return quality_metrics

        except Exception as e:
            logger.error("Failed to track response quality", error=str(e))
            return {}

    def _calculate_source_diversity(self, source_documents: List[Any]) -> float:
        """Calculate source diversity score.

        Args:
            source_documents: List of source documents

        Returns:
            Diversity score (0.0 to 1.0)
        """
        try:
            if not source_documents:
                return 0.0

            unique_files = set()
            for doc in source_documents:
                if hasattr(doc, "metadata"):
                    file_id = doc.metadata.get("file_id")
                    if file_id:
                        unique_files.add(file_id)

            diversity = len(unique_files) / len(source_documents)
            return min(diversity, 1.0)

        except Exception as e:
            logger.error("Failed to calculate source diversity", error=str(e))
            return 0.0

    def _calculate_citation_coverage(
        self,
        response: str,
        citations: List[Dict[str, Any]],
    ) -> float:
        """Calculate citation coverage score.

        Args:
            response: Generated response
            citations: List of citations

        Returns:
            Coverage score (0.0 to 1.0)
        """
        try:
            if not citations:
                return 0.0

            # Simple heuristic: more citations relative to response length
            response_length = len(response)
            citation_count = len(citations)

            # Ideal: 1 citation per 200 characters
            ideal_citations = max(response_length / 200, 1)
            coverage = min(citation_count / ideal_citations, 1.0)

            return coverage

        except Exception as e:
            logger.error("Failed to calculate citation coverage", error=str(e))
            return 0.0

    def get_average_quality(self) -> Dict[str, float]:
        """Get average quality metrics across history.

        Returns:
            Dictionary with average metrics
        """
        try:
            if not self.metrics_history:
                return {}

            averages = {}
            for key in self.metrics_history[0].keys():
                if isinstance(self.metrics_history[0][key], (int, float)):
                    values = [m.get(key, 0) for m in self.metrics_history]
                    averages[key] = sum(values) / len(values)

            return averages

        except Exception as e:
            logger.error("Failed to calculate average quality", error=str(e))
            return {}

