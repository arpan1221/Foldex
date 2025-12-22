"""Application configuration settings with environment variable support."""

from pydantic import Field, field_validator, ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Literal, Any, Optional
from pathlib import Path


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    All settings can be overridden via environment variables.
    For local development, create a .env file in the project root.
    """

    # Application
    APP_NAME: str = Field(default="Foldex", description="Application name")
    APP_ENV: Literal["development", "production", "testing"] = Field(
        default="development", description="Application environment"
    )
    DEBUG: bool = Field(
        default=True, description="Enable debug mode (auto-disabled in production)"
    )
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Logging level"
    )

    # Server Configuration
    BACKEND_HOST: str = Field(
        default="0.0.0.0", description="Backend server host"
    )
    BACKEND_PORT: int = Field(
        default=8000, ge=1, le=65535, description="Backend server port"
    )
    FRONTEND_PORT: int = Field(
        default=3000, ge=1, le=65535, description="Frontend server port"
    )

    # CORS Configuration
    CORS_ORIGINS: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:5173"],
        description="Allowed CORS origins",
    )

    # Database Configuration
    DATABASE_PATH: str = Field(
        default="./data/foldex.db", description="SQLite database path"
    )
    VECTOR_DB_PATH: str = Field(
        default="./data/vector_db", description="ChromaDB persistence path"
    )

    # Google Drive API Configuration
    GOOGLE_CLIENT_ID: str = Field(
        default="", description="Google OAuth2 client ID"
    )
    GOOGLE_CLIENT_SECRET: str = Field(
        default="", description="Google OAuth2 client secret"
    )
    GOOGLE_REDIRECT_URI: str = Field(
        default="http://localhost:3000/auth/callback",
        description="Google OAuth2 redirect URI",
    )

    # Local LLM Configuration (Ollama)
    OLLAMA_BASE_URL: str = Field(
        default="http://ollama:11434", description="Ollama API base URL (use 'ollama' for Docker, 'localhost' for local)"
    )
    OLLAMA_MODEL: str = Field(
        default="llama3.2:3b", description="Ollama model name (llama3.2:3b optimized for speed and efficiency)"
    )
    OLLAMA_TIMEOUT: int = Field(
        default=300, description="Ollama request timeout in seconds"
    )
    OLLAMA_KEEP_ALIVE: str = Field(
        default="-1", description="Keep model loaded in memory (-1 = indefinitely, 0 = unload immediately)"
    )
    OLLAMA_NUM_PREDICT: int = Field(
        default=4096, description="Maximum number of tokens to generate in response (default: 4096 for longer responses)"
    )

    # Embedding Model Configuration
    EMBEDDING_MODEL: str = Field(
        default="nomic-embed-text:latest",
        description="Ollama embedding model name (nomic-embed-text:latest recommended)",
    )
    EMBEDDING_TYPE: Literal["ollama", "huggingface"] = Field(
        default="ollama", description="Embedding provider (ollama or huggingface)"
    )
    EMBEDDING_DEVICE: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu", description="Device for embedding model (huggingface only)"
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default=32, ge=1, description="Batch size for embedding generation"
    )
    
    # Reranking Configuration
    USE_RERANKING: bool = Field(
        default=True, description="Enable cross-encoder reranking (requires HuggingFace models)"
    )
    RERANKER_MODEL: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model for reranking",
    )
    
    # Smart Reranking Configuration
    USE_SMART_RERANKING: bool = Field(
        default=True, description="Enable query-type-aware smart reranking"
    )
    DIVERSITY_RERANK_WEIGHT: float = Field(
        default=0.3, ge=0.0, le=1.0,
        description="Weight for diversity penalty in reranking (0.0-1.0)"
    )
    RECENCY_RERANK_WEIGHT: float = Field(
        default=0.2, ge=0.0, le=1.0,
        description="Weight for recency boost in reranking (0.0-1.0)"
    )
    ENTITY_BOOST_MULTIPLIER: float = Field(
        default=1.3, ge=1.0,
        description="Multiplier for entity matches in entity search queries (>= 1.0)"
    )
    
    # MMR (Maximal Marginal Relevance) Configuration
    USE_MMR: bool = Field(
        default=True, description="Enable MMR for diversity-aware retrieval (balances relevance and diversity)"
    )
    MMR_LAMBDA: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="MMR lambda parameter: 0.0 = max diversity, 1.0 = max relevance (default: 0.7)"
    )
    
    # File Rebalancing Configuration
    MAX_CHUNKS_PER_FILE: int = Field(
        default=5, ge=1,
        description="Maximum chunks per file in retrieval results (prevents overrepresented files from dominating)"
    )
    USE_FILE_REBALANCING: bool = Field(
        default=True, description="Enable file rebalancing to downweight files with many chunks"
    )
    FILE_REBALANCING_NORMALIZATION: float = Field(
        default=0.5, ge=0.0, le=2.0,
        description="Normalization exponent for file rebalancing (0.5 = sqrt, reduces bias from high chunk counts)"
    )

    # LangSmith Observability Configuration
    LANGSMITH_API_KEY: str = Field(
        default="", description="LangSmith API key for observability (optional)"
    )
    LANGSMITH_PROJECT: str = Field(
        default="foldex", description="LangSmith project name"
    )
    LANGSMITH_TRACING: bool = Field(
        default=True, description="Enable LangSmith tracing"
    )
    LANGSMITH_ENDPOINT: str = Field(
        default="https://api.smith.langchain.com",
        description="LangSmith API endpoint"
    )

    # Performance Optimization Configuration
    EMBEDDING_CACHE_ENABLED: bool = Field(
        default=True, description="Enable embedding cache for query optimization"
    )
    EMBEDDING_CACHE_MAX_SIZE: int = Field(
        default=1000, description="Maximum number of embeddings to cache"
    )
    EMBEDDING_CACHE_TTL: int = Field(
        default=3600, description="Cache TTL in seconds (default: 1 hour)"
    )
    
    # Redis Cache Configuration (optional)
    REDIS_URL: Optional[str] = Field(
        default=None, description="Redis connection URL (e.g., redis://localhost:6379)"
    )
    USE_REDIS_CACHE: bool = Field(
        default=False, description="Enable Redis caching (requires redis package)"
    )
    CACHE_MAX_SIZE: int = Field(
        default=1000, description="Maximum size for in-memory cache"
    )
    RETRIEVAL_CACHE_TTL: int = Field(
        default=600, description="Retrieval cache TTL in seconds (default: 10 minutes)"
    )

    # Whisper Configuration
    WHISPER_MODEL: Literal["tiny", "base", "small", "medium", "large"] = Field(
        default="small", description="Whisper model size (small recommended for demo)"
    )
    WHISPER_MODEL_SIZE: Literal["tiny", "base", "small", "medium", "large"] = Field(
        default="base", description="Whisper model size for audio transcription (base recommended for balance)"
    )
    WHISPER_DEVICE: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu", description="Device for Whisper model"
    )
    MAX_AUDIO_DURATION_MINUTES: int = Field(
        default=60, ge=1, le=480, description="Maximum audio duration in minutes for processing"
    )

    # Security Configuration
    SECRET_KEY: str = Field(
        default="change-me-in-production-please-use-a-secure-key-32-chars-min",
        min_length=32,
        description="Secret key for JWT signing (must be at least 32 characters)",
    )
    JWT_ALGORITHM: Literal["HS256", "HS512"] = Field(
        default="HS256", description="JWT signing algorithm"
    )
    JWT_EXPIRATION_HOURS: int = Field(
        default=24, ge=1, description="JWT token expiration in hours"
    )

    # Multimodal Processing Configuration
    UNSTRUCTURED_STRATEGY: Literal["fast", "hi_res", "auto"] = Field(
        default="fast", description="Unstructured.io processing strategy for PDFs (fast=quick, hi_res=accurate, auto=adaptive)"
    )
    ENABLE_OCR: bool = Field(
        default=True, description="Enable OCR processing for scanned documents and images"
    )
    ENABLE_RESPONSE_POST_PROCESSING: bool = Field(
        default=False, description="Enable post-processing of LLM responses (cleaning, formatting). Disabled by default for testing."
    )

    # Performance Configuration
    MAX_CONCURRENT_PROCESSING: int = Field(
        default=4, ge=1, le=16, description="Maximum concurrent file processing tasks"
    )
    CHUNK_SIZE: int = Field(
        default=800, ge=100, le=5000, description="Default text chunk size (optimized for Llama 3.2:3b - ~200 tokens)"
    )
    CHUNK_OVERLAP: int = Field(
        default=150, ge=0, le=500, description="Default chunk overlap size (optimized for Llama 3.2:3b)"
    )
    MAX_FILE_SIZE_MB: int = Field(
        default=100, ge=1, description="Maximum file size in MB"
    )

    # TTFT (Time-To-First-Token) Optimization
    ENABLE_TTFT_OPTIMIZATION: bool = Field(
        default=True, description="Enable TTFT optimization techniques"
    )
    ENABLE_PROMPT_CACHE: bool = Field(
        default=True, description="Enable prompt caching to reduce TTFT"
    )
    PROMPT_CACHE_SIZE: int = Field(
        default=100, ge=10, le=1000, description="Maximum size of prompt cache"
    )
    PROMPT_CACHE_TTL: int = Field(
        default=3600, ge=60, description="Prompt cache TTL in seconds"
    )
    MAX_CONTEXT_CHARS: int = Field(
        default=2400, ge=1000, le=20000, description="Maximum characters in context window (optimized for Llama 3.2:3b - ~600 tokens)"
    )
    MAX_RETRIEVAL_DOCS: int = Field(
        default=4, ge=1, le=50, description="Maximum documents to retrieve initially (optimized for Llama 3.2:3b)"
    )
    MAX_CONTEXT_DOCS: int = Field(
        default=3, ge=1, le=20, description="Maximum documents in final context (optimized for Llama 3.2:3b)"
    )
    RELEVANCE_THRESHOLD: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Minimum relevance score to include chunk (0.0-1.0, lower = more permissive)"
    )
    ENABLE_MODEL_WARMUP: bool = Field(
        default=True, description="Pre-warm model on startup to reduce first-request latency"
    )

    # Storage Paths
    CACHE_DIR: str = Field(
        default="./data/cache", description="Temporary file cache directory"
    )
    KNOWLEDGE_GRAPH_DIR: str = Field(
        default="./data/knowledge_graphs",
        description="Knowledge graph storage directory",
    )
    SESSIONS_DIR: str = Field(
        default="./data/sessions", description="User session storage directory"
    )

    # Model Storage Paths
    MODELS_DIR: str = Field(
        default="./models", description="Base directory for ML models"
    )

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
        env_ignore_empty=True,  # Ignore empty .env files
    )

    @field_validator("DEBUG", mode="before")
    @classmethod
    def validate_debug(cls, v: Any, info: ValidationInfo) -> bool:
        """Auto-disable debug in production."""
        if info.data and info.data.get("APP_ENV") == "production":
            return False
        return bool(v)

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> List[str]:
        """Parse CORS origins from string or list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        if isinstance(v, list):
            return v
        return []

    @field_validator("SECRET_KEY")
    @classmethod
    def validate_secret_key(cls, v: str, info: ValidationInfo) -> str:
        """Warn if using default secret key in production."""
        default_key = "change-me-in-production-please-use-a-secure-key-32-chars-min"
        if (
            v == default_key
            and info.data
            and info.data.get("APP_ENV") == "production"
        ):
            raise ValueError(
                "SECRET_KEY must be changed from default value in production"
            )
        return v

    def ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        directories = [
            self.CACHE_DIR,
            self.VECTOR_DB_PATH,
            self.KNOWLEDGE_GRAPH_DIR,
            self.SESSIONS_DIR,
            Path(self.DATABASE_PATH).parent,
        ]
        for directory in directories:
            Path(str(directory)).mkdir(parents=True, exist_ok=True)

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.APP_ENV == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.APP_ENV == "production"


# Global settings instance
# Handle .env file loading errors gracefully (permission errors, etc.)
try:
    settings = Settings()
except (PermissionError, OSError, FileNotFoundError) as e:
    # If we can't read .env file, create settings without it
    # This can happen if .env file has wrong permissions or is in restricted location
    import warnings
    warnings.warn(
        f"Could not load .env file: {e}. Using environment variables and defaults only.",
        UserWarning
    )
    # Create settings instance without .env file
    class _SettingsWithoutEnvFile(Settings):
        model_config = SettingsConfigDict(
            env_file=None,  # Don't try to load .env file
            env_file_encoding="utf-8",
            case_sensitive=True,
            extra="ignore",
        )
    settings = _SettingsWithoutEnvFile()

# Ensure directories exist on import
settings.ensure_directories()

