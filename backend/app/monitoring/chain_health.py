"""LangChain chain health monitoring and alerts."""

from typing import Optional, Dict, Any, List, Callable
import structlog
import time
from enum import Enum
from collections import defaultdict

try:
    from langchain.chains.base import Chain
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    Chain = None

from app.core.exceptions import ProcessingError
from app.monitoring.langchain_metrics import PerformanceMetricsHandler, BottleneckDetector

logger = structlog.get_logger(__name__)


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class ChainHealthMonitor:
    """Monitor for LangChain chain health."""

    def __init__(
        self,
        metrics_handler: Optional[PerformanceMetricsHandler] = None,
        bottleneck_detector: Optional[BottleneckDetector] = None,
    ):
        """Initialize chain health monitor.

        Args:
            metrics_handler: Optional metrics handler
            bottleneck_detector: Optional bottleneck detector
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        self.metrics_handler = metrics_handler or PerformanceMetricsHandler()
        self.bottleneck_detector = bottleneck_detector or BottleneckDetector()
        self.health_history: List[Dict[str, Any]] = []
        self.alert_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.logger = structlog.get_logger(__name__)

    def check_health(self) -> Dict[str, Any]:
        """Check chain health.

        Returns:
            Dictionary with health status and details
        """
        try:
            metrics = self.metrics_handler.get_metrics()
            bottlenecks = self.bottleneck_detector.get_bottlenecks()

            # Determine health status
            status = self._determine_health_status(metrics, bottlenecks)

            health_report = {
                "status": status.value,
                "timestamp": time.time(),
                "metrics": metrics,
                "bottlenecks": bottlenecks,
                "recommendations": self._generate_health_recommendations(
                    status,
                    metrics,
                    bottlenecks,
                ),
            }

            # Store in history
            self.health_history.append(health_report)

            # Keep only last 100 entries
            if len(self.health_history) > 100:
                self.health_history = self.health_history[-100:]

            # Trigger alerts if needed
            if status in [HealthStatus.DEGRADED, HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                self._trigger_alert(health_report)

            logger.info(
                "Health check completed",
                status=status.value,
            )

            return health_report

        except Exception as e:
            logger.error("Health check failed", error=str(e))
            return {
                "status": HealthStatus.CRITICAL.value,
                "error": str(e),
                "timestamp": time.time(),
            }

    def _determine_health_status(
        self,
        metrics: Dict[str, Any],
        bottlenecks: List[Dict[str, Any]],
    ) -> HealthStatus:
        """Determine health status from metrics and bottlenecks.

        Args:
            metrics: Performance metrics
            bottlenecks: List of bottlenecks

        Returns:
            HealthStatus
        """
        # Check for critical issues
        error_count = len(metrics.get("errors", []))
        if error_count > 10:
            return HealthStatus.CRITICAL

        # Check for bottlenecks
        if len(bottlenecks) > 5:
            return HealthStatus.UNHEALTHY

        # Check latency
        avg_llm_latency = metrics.get("avg_llm_latency", 0.0)
        if avg_llm_latency > 10.0:
            return HealthStatus.UNHEALTHY
        elif avg_llm_latency > 5.0:
            return HealthStatus.DEGRADED

        # Check retriever latency
        avg_retriever_latency = metrics.get("avg_retriever_latency", 0.0)
        if avg_retriever_latency > 5.0:
            return HealthStatus.DEGRADED

        # Check error rate
        total_calls = (
            metrics.get("llm_calls", 0) +
            metrics.get("retriever_calls", 0) +
            metrics.get("chain_calls", 0)
        )
        if total_calls > 0:
            error_rate = error_count / total_calls
            if error_rate > 0.1:  # 10% error rate
                return HealthStatus.UNHEALTHY
            elif error_rate > 0.05:  # 5% error rate
                return HealthStatus.DEGRADED

        return HealthStatus.HEALTHY

    def _generate_health_recommendations(
        self,
        status: HealthStatus,
        metrics: Dict[str, Any],
        bottlenecks: List[Dict[str, Any]],
    ) -> List[str]:
        """Generate health recommendations.

        Args:
            status: Health status
            metrics: Performance metrics
            bottlenecks: List of bottlenecks

        Returns:
            List of recommendations
        """
        recommendations = []

        if status == HealthStatus.CRITICAL:
            recommendations.append("CRITICAL: High error rate detected - review error logs immediately")
            recommendations.append("Consider restarting the service or reducing load")

        if status == HealthStatus.UNHEALTHY:
            recommendations.append("UNHEALTHY: Multiple bottlenecks detected - optimize chain execution")
            recommendations.append("Consider enabling caching or reducing query complexity")

        if status == HealthStatus.DEGRADED:
            recommendations.append("DEGRADED: Performance degradation detected - monitor closely")
            recommendations.append("Consider optimizing chain configuration")

        # Specific recommendations based on metrics
        avg_llm_latency = metrics.get("avg_llm_latency", 0.0)
        if avg_llm_latency > 5.0:
            recommendations.append("High LLM latency - consider model optimization or caching")

        avg_retriever_latency = metrics.get("avg_retriever_latency", 0.0)
        if avg_retriever_latency > 3.0:
            recommendations.append("High retriever latency - consider caching retrieval results")

        if bottlenecks:
            recommendations.append(f"{len(bottlenecks)} bottlenecks detected - review bottleneck details")

        return recommendations

    def _trigger_alert(self, health_report: Dict[str, Any]) -> None:
        """Trigger health alert.

        Args:
            health_report: Health report dictionary
        """
        try:
            for callback in self.alert_callbacks:
                try:
                    callback(health_report)
                except Exception as e:
                    logger.error("Alert callback failed", error=str(e))

            logger.warning(
                "Health alert triggered",
                status=health_report.get("status"),
            )

        except Exception as e:
            logger.error("Failed to trigger alert", error=str(e))

    def register_alert_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Register alert callback.

        Args:
            callback: Callback function for alerts
        """
        self.alert_callbacks.append(callback)
        logger.debug("Registered alert callback")

    def get_health_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get health history.

        Args:
            limit: Maximum number of entries

        Returns:
            List of health reports
        """
        return self.health_history[-limit:]

    def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary.

        Returns:
            Dictionary with health summary
        """
        if not self.health_history:
            return {"status": "unknown", "history_count": 0}

        recent_history = self.health_history[-10:] if len(self.health_history) > 10 else self.health_history

        status_counts = defaultdict(int)
        for report in recent_history:
            status = report.get("status", "unknown")
            status_counts[status] += 1

        return {
            "recent_status": recent_history[-1].get("status") if recent_history else "unknown",
            "status_distribution": dict(status_counts),
            "history_count": len(self.health_history),
            "recent_history": recent_history,
        }


class ChainHealthScheduler:
    """Scheduler for periodic health checks."""

    def __init__(
        self,
        monitor: ChainHealthMonitor,
        check_interval: int = 60,
    ):
        """Initialize health scheduler.

        Args:
            monitor: Chain health monitor
            check_interval: Check interval in seconds
        """
        self.monitor = monitor
        self.check_interval = check_interval
        self.running = False
        self.logger = structlog.get_logger(__name__)

    async def start(self) -> None:
        """Start health scheduler."""
        self.running = True
        logger.info("Started chain health scheduler")

        while self.running:
            try:
                await asyncio.sleep(self.check_interval)
                if self.running:
                    self.monitor.check_health()

            except Exception as e:
                logger.error("Health scheduler error", error=str(e))

    def stop(self) -> None:
        """Stop health scheduler."""
        self.running = False
        logger.info("Stopped chain health scheduler")


class ChainHealthAlert:
    """Alert system for chain health issues."""

    def __init__(self):
        """Initialize chain health alert."""
        self.alerts: List[Dict[str, Any]] = []
        self.logger = structlog.get_logger(__name__)

    def create_alert(
        self,
        severity: str,
        message: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create health alert.

        Args:
            severity: Alert severity
            message: Alert message
            details: Optional alert details
        """
        alert = {
            "severity": severity,
            "message": message,
            "details": details or {},
            "timestamp": time.time(),
        }

        self.alerts.append(alert)

        # Keep only last 100 alerts
        if len(self.alerts) > 100:
            self.alerts = self.alerts[-100:]

        logger.warning(
            "Health alert created",
            severity=severity,
            message=message,
        )

    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts.

        Args:
            limit: Maximum number of alerts

        Returns:
            List of alerts
        """
        return self.alerts[-limit:]

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        self.alerts.clear()
        logger.info("Cleared all alerts")

