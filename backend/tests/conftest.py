"""Pytest configuration and fixtures."""

import pytest
from typing import Generator
import tempfile
import os

from app.database.sqlite_manager import SQLiteManager
from app.database.vector_store import VectorStore


@pytest.fixture
def temp_db() -> Generator[SQLiteManager, None, None]:
    """Create temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    db = SQLiteManager(db_path=db_path)
    yield db
    db.close()
    os.unlink(db_path)


@pytest.fixture
def temp_vector_store() -> Generator[VectorStore, None, None]:
    """Create temporary vector store for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(persist_directory=tmpdir)
        yield store

