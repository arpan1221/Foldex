"""Performance metrics collection."""

from typing import Dict, Optional
from datetime import datetime
import time
import structlog

logger = structlog.get_logger(__name__)


class MetricsCollector:
    """Collects and logs performance metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: Dict[str, list] = {}

    def record_processing_time(
        self, operation: str, duration: float, metadata: Optional[Dict] = None
    ) -> None:
        """Record processing time for an operation.

        Args:
            operation: Operation name
            duration: Duration in seconds
            metadata: Optional metadata
        """
        if operation not in self.metrics:
            self.metrics[operation] = []

        self.metrics[operation].append({
            "duration": duration,
            "timestamp": datetime.utcnow().isoformat(),
            "metadata": metadata or {},
        })

        logger.info(
            "Processing time recorded",
            operation=operation,
            duration_seconds=duration,
            metadata=metadata,
        )

    def get_average_time(self, operation: str) -> Optional[float]:
        """Get average processing time for an operation.

        Args:
            operation: Operation name

        Returns:
            Average time in seconds or None
        """
        if operation not in self.metrics or not self.metrics[operation]:
            return None

        durations = [m["duration"] for m in self.metrics[operation]]
        return sum(durations) / len(durations)


class Timer:
    """Context manager for timing operations."""

    def __init__(self, operation: str, collector: MetricsCollector):
        """Initialize timer.

        Args:
            operation: Operation name
            collector: Metrics collector
        """
        self.operation = operation
        self.collector = collector
        self.start_time: Optional[float] = None

    def __enter__(self):
        """Start timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and record metric."""
        if self.start_time:
            duration = time.time() - self.start_time
            self.collector.record_processing_time(self.operation, duration)


# Global metrics collector
metrics = MetricsCollector()

