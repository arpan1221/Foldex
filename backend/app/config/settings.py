"""Application configuration settings."""

from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Application
    APP_NAME: str = "Foldex"
    APP_ENV: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Server
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    FRONTEND_PORT: int = 3000

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:5173"]

    # Database
    DATABASE_PATH: str = "./data/foldex.db"
    VECTOR_DB_PATH: str = "./data/vector_db"

    # Google Drive API
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""
    GOOGLE_REDIRECT_URI: str = "http://localhost:3000/auth/callback"

    # Local LLM (Ollama)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3.2"

    # Embedding Model
    EMBEDDING_MODEL: str = "sentence-transformers/all-mpnet-base-v2"

    # Whisper
    WHISPER_MODEL: str = "base"
    WHISPER_DEVICE: str = "cpu"

    # Security
    SECRET_KEY: str = "change-me-in-production"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRATION_HOURS: int = 24

    # Performance
    MAX_CONCURRENT_PROCESSING: int = 4
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200

    # Storage Paths
    CACHE_DIR: str = "./data/cache"
    KNOWLEDGE_GRAPH_DIR: str = "./data/knowledge_graphs"
    SESSIONS_DIR: str = "./data/sessions"

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

