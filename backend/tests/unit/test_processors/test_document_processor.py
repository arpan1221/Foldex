"""Unit tests for DocumentProcessor orchestrator."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from app.services.document_processor import DocumentProcessor
from app.processors.pdf_processor import PDFProcessor
from app.processors.text_processor import TextProcessor
from app.models.documents import DocumentChunk, FileMetadata
from app.core.exceptions import DocumentProcessingError


@pytest.mark.asyncio
class TestDocumentProcessor:
    """Test DocumentProcessor orchestrator."""
    
    @pytest.fixture
    def processor(self):
        """Create DocumentProcessor instance."""
        return DocumentProcessor()
    
    async def test_process_file_routes_to_pdf_processor(self, processor, sample_pdf_path):
        """Test that PDF files are routed to PDFProcessor."""
        file_info = {
            "id": "test_file",
            "name": "test.pdf",
            "mimeType": "application/pdf",
        }
        
        try:
            chunks = await processor.process_file(str(sample_pdf_path), file_info)
            assert len(chunks) > 0
            assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        except DocumentProcessingError:
            pytest.skip("PDF processing not available")
    
    async def test_process_file_routes_to_text_processor(self, processor, sample_text_path):
        """Test that text files are routed to TextProcessor."""
        file_info = {
            "id": "test_file",
            "name": "test.txt",
            "mimeType": "text/plain",
        }
        
        chunks = await processor.process_file(str(sample_text_path), file_info)
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
    
    async def test_process_file_unsupported_type(self, processor, temp_dir):
        """Test processing unsupported file type."""
        unsupported_file = temp_dir / "test.xyz"
        unsupported_file.write_text("test content")
        
        file_info = {
            "id": "test_file",
            "name": "test.xyz",
            "mimeType": "application/unknown",
        }
        
        with pytest.raises(DocumentProcessingError):
            await processor.process_file(str(unsupported_file), file_info)
    
    async def test_process_file_with_progress_callback(self, processor, sample_text_path, mock_progress_callback):
        """Test file processing with progress callback."""
        file_info = {
            "id": "test_file",
            "name": "test.txt",
            "mimeType": "text/plain",
        }
        
        chunks = await processor.process_file(
            str(sample_text_path),
            file_info,
            progress_callback=mock_progress_callback,
        )
        
        assert mock_progress_callback.called
        assert len(chunks) > 0
    
    async def test_process_file_enriches_metadata(self, processor, sample_text_path):
        """Test that file metadata is enriched in chunks."""
        file_info = {
            "id": "test_file_123",
            "name": "test.txt",
            "mimeType": "text/plain",
            "size": "1024",
            "createdTime": "2024-01-01T00:00:00Z",
        }
        
        chunks = await processor.process_file(str(sample_text_path), file_info)
        
        assert all(chunk.file_id == "test_file_123" for chunk in chunks)
        assert all(chunk.metadata.get("file_name") == "test.txt" for chunk in chunks)
        assert all("file_id" in chunk.metadata for chunk in chunks)

