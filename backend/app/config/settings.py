"""Application configuration settings with environment variable support."""

from pydantic import Field, field_validator, ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List, Literal, Any
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
        default="http://localhost:11434", description="Ollama API base URL"
    )
    OLLAMA_MODEL: str = Field(
        default="llama3.2", description="Ollama model name"
    )
    OLLAMA_TIMEOUT: int = Field(
        default=300, description="Ollama request timeout in seconds"
    )

    # Embedding Model Configuration
    EMBEDDING_MODEL: str = Field(
        default="sentence-transformers/all-mpnet-base-v2",
        description="Sentence transformer model name",
    )
    EMBEDDING_DEVICE: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu", description="Device for embedding model"
    )
    EMBEDDING_BATCH_SIZE: int = Field(
        default=32, ge=1, description="Batch size for embedding generation"
    )

    # Whisper Configuration
    WHISPER_MODEL: Literal["tiny", "base", "small", "medium", "large"] = Field(
        default="base", description="Whisper model size"
    )
    WHISPER_DEVICE: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu", description="Device for Whisper model"
    )

    # Security Configuration
    SECRET_KEY: str = Field(
        default="change-me-in-production",
        min_length=32,
        description="Secret key for JWT signing (must be at least 32 characters)",
    )
    JWT_ALGORITHM: Literal["HS256", "HS512"] = Field(
        default="HS256", description="JWT signing algorithm"
    )
    JWT_EXPIRATION_HOURS: int = Field(
        default=24, ge=1, description="JWT token expiration in hours"
    )

    # Performance Configuration
    MAX_CONCURRENT_PROCESSING: int = Field(
        default=4, ge=1, le=16, description="Maximum concurrent file processing tasks"
    )
    CHUNK_SIZE: int = Field(
        default=1000, ge=100, le=5000, description="Default text chunk size"
    )
    CHUNK_OVERLAP: int = Field(
        default=200, ge=0, le=500, description="Default chunk overlap size"
    )
    MAX_FILE_SIZE_MB: int = Field(
        default=100, ge=1, description="Maximum file size in MB"
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
        if (
            v == "change-me-in-production"
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
            Path(directory).mkdir(parents=True, exist_ok=True)

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.APP_ENV == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.APP_ENV == "production"


# Global settings instance
settings = Settings()

# Ensure directories exist on import
settings.ensure_directories()

