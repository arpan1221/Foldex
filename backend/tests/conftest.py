"""Pytest configuration and shared fixtures for all tests."""

import pytest
import asyncio
import tempfile
import shutil
from typing import AsyncGenerator, Generator
from pathlib import Path
from unittest.mock import Mock, AsyncMock
import structlog

from app.database.base import DatabaseSessionManager
from app.database.sqlite_manager import SQLiteManager
from app.rag.vector_store import LangChainVectorStore
from app.config.settings import Settings
from app.models.documents import DocumentChunk, FileMetadata


# Configure structlog for tests
structlog.configure(
    processors=[structlog.dev.ConsoleRenderer()],
    wrapper_class=structlog.make_filtering_bound_logger(50),  # Only show errors
)


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for test files."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def temp_db_path(temp_dir: Path) -> Path:
    """Create temporary database path."""
    return temp_dir / "test.db"


@pytest.fixture
async def db_session_manager(temp_db_path: Path) -> AsyncGenerator[DatabaseSessionManager, None]:
    """Create database session manager for testing."""
    db_url = f"sqlite+aiosqlite:///{temp_db_path}"
    manager = DatabaseSessionManager(db_url)
    await manager.initialize()
    await manager.create_tables()
    
    yield manager
    
    await manager.close()
    if temp_db_path.exists():
        temp_db_path.unlink()


@pytest.fixture
async def db_manager(db_session_manager: DatabaseSessionManager) -> AsyncGenerator[SQLiteManager, None]:
    """Create SQLite manager instance for testing."""
    manager = SQLiteManager()
    yield manager


@pytest.fixture
def temp_vector_store_dir(temp_dir: Path) -> Path:
    """Create temporary directory for vector store."""
    vector_dir = temp_dir / "vector_store"
    vector_dir.mkdir()
    return vector_dir


@pytest.fixture
def vector_store(temp_vector_store_dir: Path) -> LangChainVectorStore:
    """Create LangChain vector store instance for testing."""
    return LangChainVectorStore(persist_directory=str(temp_vector_store_dir))
    """Create vector store instance for testing."""
    return VectorStore(persist_directory=str(temp_vector_store_dir))


@pytest.fixture
def test_settings(temp_dir: Path) -> Settings:
    """Create test settings with temporary paths."""
    return Settings(
        DATABASE_PATH=str(temp_dir / "test.db"),
        VECTOR_DB_PATH=str(temp_dir / "vector_db"),
        CACHE_DIR=str(temp_dir / "cache"),
        KNOWLEDGE_GRAPH_DIR=str(temp_dir / "kg"),
        SESSIONS_DIR=str(temp_dir / "sessions"),
        APP_ENV="test",
        DEBUG=True,
    )


@pytest.fixture
def sample_file_metadata() -> FileMetadata:
    """Create sample file metadata for testing."""
    return FileMetadata(
        file_id="test_file_123",
        file_name="test_document.pdf",
        mime_type="application/pdf",
        size=1024,
        folder_id="test_folder_456",
    )


@pytest.fixture
def sample_chunks() -> list[DocumentChunk]:
    """Create sample document chunks for testing."""
    return [
        DocumentChunk(
            chunk_id="chunk_1",
            file_id="test_file_123",
            content="This is the first chunk of text.",
            metadata={"page": 1, "section": "introduction"},
            file_name="test_document.pdf",
        ),
        DocumentChunk(
            chunk_id="chunk_2",
            file_id="test_file_123",
            content="This is the second chunk of text.",
            metadata={"page": 2, "section": "main"},
            file_name="test_document.pdf",
        ),
    ]


@pytest.fixture
def mock_google_drive_service():
    """Create mock Google Drive service."""
    mock_service = AsyncMock()
    
    # Mock list_folder_files
    mock_service.list_folder_files = AsyncMock(return_value=[
        {
            "id": "file_1",
            "name": "test.pdf",
            "mimeType": "application/pdf",
            "size": "1024",
            "createdTime": "2024-01-01T00:00:00Z",
            "modifiedTime": "2024-01-01T00:00:00Z",
        },
        {
            "id": "file_2",
            "name": "test.txt",
            "mimeType": "text/plain",
            "size": "512",
            "createdTime": "2024-01-01T00:00:00Z",
            "modifiedTime": "2024-01-01T00:00:00Z",
        },
    ])
    
    # Mock download_file
    mock_service.download_file = AsyncMock(return_value="/tmp/test_file.pdf")
    
    # Mock is_supported_file_type
    mock_service.is_supported_file_type = Mock(return_value=True)
    
    return mock_service


@pytest.fixture
def mock_progress_callback():
    """Create mock progress callback."""
    return Mock()


@pytest.fixture
def sample_pdf_path(temp_dir: Path) -> Path:
    """Create a sample PDF file for testing."""
    # Create a minimal PDF file
    pdf_path = temp_dir / "sample.pdf"
    
    # Minimal PDF structure (valid PDF header)
    pdf_content = b"""%PDF-1.4
1 0 obj
<<
/Type /Catalog
/Pages 2 0 R
>>
endobj
2 0 obj
<<
/Type /Pages
/Kids [3 0 R]
/Count 1
>>
endobj
3 0 obj
<<
/Type /Page
/Parent 2 0 R
/MediaBox [0 0 612 792]
/Contents 4 0 R
/Resources <<
/Font <<
/F1 <<
/Type /Font
/Subtype /Type1
/BaseFont /Helvetica
>>
>>
>>
>>
endobj
4 0 obj
<<
/Length 44
>>
stream
BT
/F1 12 Tf
100 700 Td
(Test PDF Content) Tj
ET
endstream
endobj
xref
0 5
0000000000 65535 f
0000000009 00000 n
0000000058 00000 n
0000000115 00000 n
0000000306 00000 n
trailer
<<
/Size 5
/Root 1 0 R
>>
startxref
398
%%EOF"""
    
    pdf_path.write_bytes(pdf_content)
    return pdf_path


@pytest.fixture
def sample_text_path(temp_dir: Path) -> Path:
    """Create a sample text file for testing."""
    text_path = temp_dir / "sample.txt"
    text_path.write_text("This is a sample text file.\n\nIt has multiple paragraphs.\n\nFor testing purposes.")
    return text_path


@pytest.fixture
def sample_markdown_path(temp_dir: Path) -> Path:
    """Create a sample Markdown file for testing."""
    md_path = temp_dir / "sample.md"
    md_content = """# Sample Markdown Document

This is a **sample** markdown file for testing.

## Section 1

Some content here.

## Section 2

More content with `code` examples.
"""
    md_path.write_text(md_content)
    return md_path


@pytest.fixture
def mock_embedding_model():
    """Create mock embedding model."""
    mock_model = Mock()
    mock_model.encode = Mock(return_value=[0.1] * 384)  # Mock 384-dim embedding
    return mock_model


@pytest.fixture
def mock_llm_response():
    """Create mock LLM response."""
    return {
        "response": "This is a mock LLM response.",
        "citations": [
            {
                "file_id": "file_1",
                "file_name": "test.pdf",
                "chunk_id": "chunk_1",
                "confidence": 0.95,
            }
        ],
    }


@pytest.fixture
def mock_user():
    """Create mock user for authentication tests."""
    return {
        "user_id": "test_user_123",
        "email": "test@example.com",
        "name": "Test User",
        "google_id": "google_123",
    }


@pytest.fixture
def mock_jwt_token():
    """Create mock JWT token."""
    return "mock.jwt.token.here"


@pytest.fixture(autouse=True)
def reset_logging():
    """Reset logging configuration before each test."""
    yield
    # Reset structlog if needed
    structlog.reset_defaults()


@pytest.fixture
def mock_websocket():
    """Create mock WebSocket connection."""
    mock_ws = AsyncMock()
    mock_ws.send_json = AsyncMock()
    mock_ws.receive_text = AsyncMock(return_value='{"type": "ping"}')
    return mock_ws
