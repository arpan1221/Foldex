"""Database base classes and interfaces."""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict


class DatabaseBase(ABC):
    """Base class for database operations."""

    @abstractmethod
    async def connect(self) -> None:
        """Connect to database."""
        pass

    @abstractmethod
    async def close(self) -> None:
        """Close database connection."""
        pass

