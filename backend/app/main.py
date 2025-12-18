"""FastAPI application entry point."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import auth, folders, chat, websocket
from app.config.settings import settings
from app.core.exceptions import FoldexException

app = FastAPI(
    title="Foldex API",
    description="Local-first multimodal RAG system API",
    version="0.1.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(folders.router, prefix="/api/v1/folders", tags=["folders"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])
app.include_router(websocket.router, prefix="/ws", tags=["websocket"])


@app.exception_handler(FoldexException)
async def foldex_exception_handler(request, exc: FoldexException):
    """Handle custom Foldex exceptions."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.message, "error_type": exc.__class__.__name__},
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "Foldex API", "version": "0.1.0"}

