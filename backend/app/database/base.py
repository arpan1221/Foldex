"""SQLAlchemy base classes and session management."""

from typing import AsyncGenerator, Optional
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    AsyncEngine,
    create_async_engine,
    async_sessionmaker,
)
from sqlalchemy.orm import DeclarativeBase, declared_attr
from sqlalchemy.pool import StaticPool
import structlog

from app.config.settings import settings
from app.core.exceptions import DatabaseError

logger = structlog.get_logger(__name__)


class Base(DeclarativeBase):
    """Base class for all database models."""

    @declared_attr
    def __tablename__(cls) -> str:
        """Generate table name from class name."""
        return cls.__name__.lower() + "s"


class DatabaseSessionManager:
    """Manages database sessions with connection pooling."""

    def __init__(self, database_url: str):
        """Initialize session manager.

        Args:
            database_url: Database connection URL
        """
        self.database_url = database_url
        self.engine: Optional[AsyncEngine] = None
        self.session_factory: Optional[async_sessionmaker[AsyncSession]] = None

    async def initialize(self) -> None:
        """Initialize database engine and session factory.

        Raises:
            DatabaseError: If initialization fails
        """
        try:
            logger.info("Initializing database engine", database_url=self.database_url)

            # Create async engine with connection pooling
            self.engine = create_async_engine(
                self.database_url,
                poolclass=StaticPool,  # SQLite requires StaticPool
                connect_args={"check_same_thread": False},
                echo=settings.DEBUG,  # Log SQL queries in debug mode
                pool_pre_ping=True,  # Verify connections before using
            )

            # Create session factory
            self.session_factory = async_sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False,
                autocommit=False,
                autoflush=False,
            )

            logger.info("Database engine initialized successfully")
        except Exception as e:
            logger.error("Database initialization failed", error=str(e), exc_info=True)
            raise DatabaseError(f"Failed to initialize database: {str(e)}") from e

    async def close(self) -> None:
        """Close database engine and connections.

        Raises:
            DatabaseError: If closing fails
        """
        try:
            if self.engine:
                await self.engine.dispose()
                logger.info("Database engine closed")
        except Exception as e:
            logger.error("Error closing database engine", error=str(e))
            raise DatabaseError(f"Failed to close database: {str(e)}") from e

    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get database session with automatic cleanup.

        Yields:
            AsyncSession: Database session

        Raises:
            DatabaseError: If session creation fails

        Example:
            >>> async with db_manager.get_session() as session:
            ...     result = await session.execute(select(User))
            ...     users = result.scalars().all()
        """
        if not self.session_factory:
            raise DatabaseError("Database not initialized. Call initialize() first.")

        session = self.session_factory()
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error("Database session error", error=str(e), exc_info=True)
            raise DatabaseError(f"Database operation failed: {str(e)}") from e
        finally:
            await session.close()

    async def create_tables(self) -> None:
        """Create all database tables.

        Raises:
            DatabaseError: If table creation fails
        """
        try:
            if not self.engine:
                raise DatabaseError("Database engine not initialized")

            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error("Failed to create database tables", error=str(e), exc_info=True)
            raise DatabaseError(f"Failed to create tables: {str(e)}") from e

    async def drop_tables(self) -> None:
        """Drop all database tables.

        Warning: This will delete all data!

        Raises:
            DatabaseError: If table dropping fails
        """
        try:
            if not self.engine:
                raise DatabaseError("Database engine not initialized")

            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)

            logger.warning("Database tables dropped")
        except Exception as e:
            logger.error("Failed to drop database tables", error=str(e), exc_info=True)
            raise DatabaseError(f"Failed to drop tables: {str(e)}") from e


# Global database session manager
_db_manager: Optional[DatabaseSessionManager] = None


def get_database_url() -> str:
    """Get database URL from settings.

    Returns:
        Database connection URL
    """
    # Convert SQLite path to async URL
    db_path = settings.DATABASE_PATH
    if not db_path.startswith("sqlite"):
        # Add sqlite+aiosqlite:// prefix for async SQLite
        if db_path.startswith("./") or not db_path.startswith("/"):
            db_path = f"sqlite+aiosqlite:///{db_path}"
        else:
            db_path = f"sqlite+aiosqlite://{db_path}"
    return db_path


async def initialize_database() -> DatabaseSessionManager:
    """Initialize global database session manager.

    Returns:
        Initialized database session manager

    Raises:
        DatabaseError: If initialization fails
    """
    global _db_manager

    if _db_manager is None:
        database_url = get_database_url()
        _db_manager = DatabaseSessionManager(database_url)
        await _db_manager.initialize()
        await _db_manager.create_tables()

    return _db_manager


async def close_database() -> None:
    """Close global database session manager.

    Raises:
        DatabaseError: If closing fails
    """
    global _db_manager

    if _db_manager:
        await _db_manager.close()
        _db_manager = None


def get_db_manager() -> DatabaseSessionManager:
    """Get global database session manager.

    Returns:
        Database session manager

    Raises:
        DatabaseError: If database not initialized
    """
    if _db_manager is None:
        raise DatabaseError(
            "Database not initialized. Call initialize_database() first."
        )
    return _db_manager

