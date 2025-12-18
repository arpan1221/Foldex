"""Structured logging configuration with structlog."""

import logging
import sys

import structlog
from structlog.types import Processor


def setup_logging(
    log_level: str = "INFO",
    enable_json: bool = False,
    enable_colors: bool = True,
) -> None:
    """Configure structured logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_json: Force JSON output even in development
        enable_colors: Enable colored output in console (development only)

    Example:
        >>> setup_logging("DEBUG")
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started", version="0.1.0")
    """
    # Validate log level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if log_level.upper() not in valid_levels:
        raise ValueError(
            f"Invalid log level: {log_level}. Must be one of {valid_levels}"
        )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper()),
        force=True,  # Override any existing configuration
    )

    # Build processor chain
    processors: list[Processor] = [
        # Add context variables
        structlog.contextvars.merge_contextvars,
        # Add log level
        structlog.processors.add_log_level,
        # Add stack info for exceptions
        structlog.processors.StackInfoRenderer(),
        # Format exception info
        structlog.dev.set_exc_info,
        # Add timestamp
        structlog.processors.TimeStamper(fmt="iso"),
    ]

    # Choose renderer based on environment
    is_debug = log_level.upper() == "DEBUG"
    if enable_json or (not is_debug and not enable_colors):
        # JSON output for production/logging systems
        processors.append(structlog.processors.JSONRenderer())
    else:
        # Pretty console output for development
        processors.append(
            structlog.dev.ConsoleRenderer(colors=enable_colors)
        )

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.BoundLogger:
    """Get a structured logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        Configured structlog logger

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Processing file", file_path="/path/to/file.pdf")
        >>> logger.error("Processing failed", error=str(e), file_id="abc123")
    """
    return structlog.get_logger(name)


def configure_request_logging() -> None:
    """Configure logging for HTTP requests.

    This sets up middleware-compatible logging for FastAPI requests.
    """
    # Additional configuration for request logging can be added here
    # For now, the base setup_logging handles it
    pass

