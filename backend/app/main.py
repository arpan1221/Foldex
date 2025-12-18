"""FastAPI application entry point with middleware and routing."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from app.api.v1 import auth, folders, chat, websocket
from app.config.settings import settings
from app.config.logging import setup_logging, get_logger
from app.core.exceptions import FoldexException
from app.database.base import initialize_database, close_database


# Initialize logging before creating app
setup_logging(
    log_level=settings.LOG_LEVEL,
    enable_json=settings.is_production,
    enable_colors=settings.is_development,
)

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager for startup and shutdown events.

    Yields:
        None: Application is ready
    """
    # Startup
    logger.info(
        "Starting Foldex API",
        environment=settings.APP_ENV,
        debug=settings.DEBUG,
        port=settings.BACKEND_PORT,
    )

    # Ensure directories exist
    settings.ensure_directories()
    logger.info("Application directories verified")

    # Initialize database
    try:
        await initialize_database()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error("Database initialization failed", error=str(e), exc_info=True)
        raise

    yield

    # Shutdown
    logger.info("Shutting down Foldex API")
    
    # Close database connections
    try:
        await close_database()
        logger.info("Database closed successfully")
    except Exception as e:
        logger.error("Error closing database", error=str(e), exc_info=True)


# Create FastAPI application
app = FastAPI(
    title="Foldex API",
    description="Local-first multimodal RAG system for Google Drive folders",
    version="0.1.0",
    docs_url="/api/docs" if settings.DEBUG else None,
    redoc_url="/api/redoc" if settings.DEBUG else None,
    openapi_url="/api/openapi.json" if settings.DEBUG else None,
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all HTTP requests with structured logging."""
    import time

    start_time = time.time()
    logger.info(
        "Request started",
        method=request.method,
        path=request.url.path,
        client_ip=request.client.host if request.client else None,
    )

    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        logger.info(
            "Request completed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            process_time_seconds=round(process_time, 3),
        )
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(
            "Request failed",
            method=request.method,
            path=request.url.path,
            error=str(e),
            error_type=type(e).__name__,
            process_time_seconds=round(process_time, 3),
        )
        raise


# Exception handlers
@app.exception_handler(FoldexException)
async def foldex_exception_handler(
    request: Request, exc: FoldexException
) -> JSONResponse:
    """Handle custom Foldex exceptions.

    Args:
        request: FastAPI request object
        exc: Foldex exception instance

    Returns:
        JSON error response
    """
    logger.error(
        "Foldex exception",
        error_type=exc.__class__.__name__,
        message=exc.message,
        path=request.url.path,
        status_code=exc.status_code,
    )

    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.message,
            "error_type": exc.__class__.__name__,
        },
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Handle Pydantic validation errors.

    Args:
        request: FastAPI request object
        exc: Validation error

    Returns:
        JSON error response
    """
    logger.warning(
        "Validation error",
        path=request.url.path,
        errors=exc.errors(),
    )

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": "Validation error",
            "error_type": "ValidationError",
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(
    request: Request, exc: Exception
) -> JSONResponse:
    """Handle unexpected exceptions.

    Args:
        request: FastAPI request object
        exc: Exception instance

    Returns:
        JSON error response
    """
    logger.error(
        "Unexpected error",
        error_type=type(exc).__name__,
        error=str(exc),
        path=request.url.path,
        exc_info=True,
    )

    # Don't expose internal errors in production
    error_message = (
        "Internal server error"
        if settings.is_production
        else f"Internal server error: {str(exc)}"
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": error_message,
            "error_type": "InternalServerError",
        },
    )


# Include API routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["Authentication"])
app.include_router(
    folders.router, prefix="/api/v1/folders", tags=["Folder Processing"]
)
app.include_router(chat.router, prefix="/api/v1/chat", tags=["Chat & Queries"])
app.include_router(
    websocket.router, prefix="/ws", tags=["WebSocket"]
)


# Health and info endpoints
@app.get("/health", tags=["System"])
async def health_check() -> dict:
    """Health check endpoint for monitoring.

    Returns:
        Health status dictionary
    """
    return {
        "status": "healthy",
        "version": "0.1.0",
        "environment": settings.APP_ENV,
    }


@app.get("/", tags=["System"])
async def root() -> dict:
    """Root endpoint with API information.

    Returns:
        API information dictionary
    """
    return {
        "name": settings.APP_NAME,
        "version": "0.1.0",
        "description": "Local-first multimodal RAG system API",
        "docs_url": "/api/docs" if settings.DEBUG else None,
    }

