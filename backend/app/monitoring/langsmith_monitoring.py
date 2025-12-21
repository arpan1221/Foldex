"""LangSmith monitoring and observability for LangChain and LangGraph workflows."""

from typing import Optional, Dict, Any, List
import structlog
import os

logger = structlog.get_logger(__name__)

# Try to import LangSmith
try:
    from langsmith import Client, traceable
    from langsmith.run_helpers import tracing_context
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.tracers import LangChainTracer
    LANGSMITH_AVAILABLE = True
except ImportError:
    LANGSMITH_AVAILABLE = False
    Client = None
    traceable = None
    tracing_context = None
    BaseCallbackHandler = None
    LangChainTracer = None

from app.config.settings import settings


class LangSmithMonitor:
    """LangSmith monitoring utility for LangChain and LangGraph observability."""

    def __init__(self):
        """Initialize LangSmith monitor."""
        self.enabled = False
        self.client: Optional[Any] = None
        self.project_name = settings.LANGSMITH_PROJECT
        self.tracing_enabled = settings.LANGSMITH_TRACING

        if not LANGSMITH_AVAILABLE:
            logger.warning("LangSmith not installed. Install with: pip install langsmith")
            return

        # Check if LangSmith is configured
        api_key = settings.LANGSMITH_API_KEY or os.getenv("LANGCHAIN_API_KEY")
        
        if not api_key:
            logger.info(
                "LangSmith API key not configured. Set LANGSMITH_API_KEY or LANGCHAIN_API_KEY to enable monitoring"
            )
            return

        if not self.tracing_enabled:
            logger.info("LangSmith tracing is disabled")
            return

        try:
            # Initialize LangSmith client
            self.client = Client(
                api_key=api_key,
                api_url=settings.LANGSMITH_ENDPOINT,
            )
            self.enabled = True
            
            # Set environment variables for automatic tracing
            os.environ["LANGCHAIN_API_KEY"] = api_key
            os.environ["LANGCHAIN_PROJECT"] = self.project_name
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGSMITH_ENDPOINT

            logger.info(
                "LangSmith monitoring enabled",
                project=self.project_name,
                endpoint=settings.LANGSMITH_ENDPOINT,
            )
        except Exception as e:
            logger.error("Failed to initialize LangSmith", error=str(e))
            self.enabled = False

    def get_callbacks(self) -> List[Any]:
        """Get LangChain callback handlers for tracing.

        Returns:
            List of callback handlers (empty if LangSmith not enabled)
        """
        if not self.enabled or not LANGSMITH_AVAILABLE:
            return []

        try:
            tracer = LangChainTracer(project_name=self.project_name)
            return [tracer]
        except Exception as e:
            logger.error("Failed to create LangChain tracer", error=str(e))
            return []

    def get_langgraph_config(self, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Get LangGraph configuration with LangSmith tracing.

        Args:
            metadata: Optional metadata to attach to traces

        Returns:
            LangGraph configuration dictionary
        """
        config: Dict[str, Any] = {
            "configurable": {
                "thread_id": metadata.get("thread_id", "default") if metadata else "default",
            }
        }

        if self.enabled and metadata:
            # Add metadata to config for tracing
            config["metadata"] = metadata

        return config

    def trace_function(self, name: Optional[str] = None):
        """Decorator to trace a function with LangSmith.

        Args:
            name: Optional name for the trace

        Returns:
            Decorator function
        """
        if not self.enabled or not LANGSMITH_AVAILABLE:
            # Return no-op decorator
            def noop_decorator(func):
                return func
            return noop_decorator

        if traceable is None:
            def noop_decorator(func):
                return func
            return noop_decorator

        return traceable(name=name, project_name=self.project_name)

    def create_tracing_context(self, metadata: Optional[Dict[str, Any]] = None):
        """Create a LangSmith tracing context.

        Args:
            metadata: Optional metadata for the trace

        Returns:
            Context manager for tracing
        """
        if not self.enabled or not LANGSMITH_AVAILABLE or tracing_context is None:
            from contextlib import nullcontext
            return nullcontext()

        return tracing_context(
            project_name=self.project_name,
            metadata=metadata or {},
        )

    def log_feedback(
        self,
        run_id: str,
        score: Optional[float] = None,
        comment: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log feedback for a LangSmith run.

        Args:
            run_id: LangSmith run ID
            score: Optional score (0-1)
            comment: Optional comment
            metadata: Optional metadata
        """
        if not self.enabled or not self.client:
            return

        try:
            self.client.create_feedback(
                run_id=run_id,
                score=score,
                comment=comment,
                metadata=metadata or {},
            )
            logger.info("Logged feedback to LangSmith", run_id=run_id)
        except Exception as e:
            logger.error("Failed to log feedback to LangSmith", error=str(e))

    def get_run_url(self, run_id: str) -> Optional[str]:
        """Get LangSmith UI URL for a run.

        Args:
            run_id: LangSmith run ID

        Returns:
            URL to view the run in LangSmith UI, or None if not enabled
        """
        if not self.enabled:
            return None

        try:
            base_url = settings.LANGSMITH_ENDPOINT.replace("/api", "")
            return f"{base_url}/runs/{run_id}"
        except Exception:
            return None


# Global LangSmith monitor instance
_langsmith_monitor: Optional[LangSmithMonitor] = None


def get_langsmith_monitor() -> LangSmithMonitor:
    """Get global LangSmith monitor instance.

    Returns:
        LangSmithMonitor instance
    """
    global _langsmith_monitor
    if _langsmith_monitor is None:
        _langsmith_monitor = LangSmithMonitor()
    return _langsmith_monitor

