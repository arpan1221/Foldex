"""Dependency injection for FastAPI routes."""

from typing import Generator
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.core.auth import verify_token
from app.core.exceptions import AuthenticationError
from app.database.sqlite_manager import SQLiteManager
from app.database.vector_store import VectorStore

security = HTTPBearer()


def get_db() -> Generator[SQLiteManager, None, None]:
    """Dependency for database connection.

    Yields:
        SQLiteManager instance
    """
    db = SQLiteManager()
    try:
        yield db
    finally:
        db.close()


def get_vector_store() -> VectorStore:
    """Dependency for vector store.

    Returns:
        VectorStore instance
    """
    return VectorStore()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Dependency to get current authenticated user.

    Args:
        credentials: HTTP Bearer token credentials

    Returns:
        User information from token

    Raises:
        HTTPException: If authentication fails
    """
    try:
        token = credentials.credentials
        payload = verify_token(token)
        return payload
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.message,
            headers={"WWW-Authenticate": "Bearer"},
        )

