"""Dependency injection for FastAPI routes."""

from typing import AsyncGenerator
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import verify_token
from app.core.exceptions import AuthenticationError
from app.database.base import get_db_manager, DatabaseSessionManager
from app.database.sqlite_manager import SQLiteManager
from app.database.vector_store import VectorStore

security = HTTPBearer()


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency for database session.

    Yields:
        AsyncSession instance

    Example:
        >>> @router.get("/items")
        >>> async def get_items(session: AsyncSession = Depends(get_db_session)):
        ...     result = await session.execute(select(Item))
        ...     return result.scalars().all()
    """
    db_manager = get_db_manager()
    async with db_manager.get_session() as session:
        yield session


def get_sqlite_manager() -> SQLiteManager:
    """Dependency for SQLite manager.

    Returns:
        SQLiteManager instance
    """
    return SQLiteManager()


def get_vector_store() -> VectorStore:
    """Dependency for vector store.

    Returns:
        VectorStore instance
    """
    return VectorStore()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db_session),
) -> dict:
    """Dependency to get current authenticated user.

    Args:
        credentials: HTTP Bearer token credentials
        db: Database session

    Returns:
        User information dictionary

    Raises:
        HTTPException: If authentication fails
    """
    try:
        token = credentials.credentials
        payload = verify_token(token)
        user_id = payload.get("sub")

        if not user_id:
            raise AuthenticationError("Invalid token payload")

        # Get user from database
        from app.database.sqlite_manager import SQLiteManager
        db_manager = SQLiteManager()
        user = await db_manager.get_user_by_id(user_id)

        if not user:
            raise AuthenticationError("User not found")

        return user

    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=e.message,
            headers={"WWW-Authenticate": "Bearer"},
        )

