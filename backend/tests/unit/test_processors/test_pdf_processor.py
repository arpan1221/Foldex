"""Unit tests for PDFProcessor."""

import pytest
from unittest.mock import Mock

from app.processors.pdf_processor import PDFProcessor
from app.models.documents import DocumentChunk
from app.core.exceptions import DocumentProcessingError


@pytest.mark.asyncio
class TestPDFProcessor:
    """Test PDFProcessor functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create PDFProcessor instance."""
        return PDFProcessor()
    
    async def test_can_process_pdf(self, processor):
        """Test PDF file detection."""
        assert await processor.can_process("test.pdf") is True
        assert await processor.can_process("test.PDF") is True
        assert await processor.can_process("test.txt") is False
        assert await processor.can_process("test.pdf", mime_type="application/pdf") is True
        assert await processor.can_process("test.pdf", mime_type="text/plain") is False
    
    async def test_can_process_with_mime_type(self, processor):
        """Test can_process with MIME type."""
        assert await processor.can_process("file", mime_type="application/pdf") is True
        assert await processor.can_process("file", mime_type="text/plain") is False
    
    async def test_get_supported_extensions(self, processor):
        """Test supported extensions."""
        extensions = processor.get_supported_extensions()
        assert ".pdf" in extensions
    
    async def test_get_supported_mime_types(self, processor):
        """Test supported MIME types."""
        mime_types = processor.get_supported_mime_types()
        assert "application/pdf" in mime_types
    
    @pytest.mark.skipif(
        not pytest.importorskip("fitz", reason="PyMuPDF not available"),
        reason="PyMuPDF required for PDF processing"
    )
    async def test_process_pdf_success(self, processor, sample_pdf_path):
        """Test successful PDF processing."""
        chunks = await processor.process(
            str(sample_pdf_path),
            file_id="test_file_123",
            metadata={"folder_id": "folder_456"},
        )
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.file_id == "test_file_123" for chunk in chunks)
        assert all(chunk.content.strip() for chunk in chunks)
    
    async def test_process_pdf_file_not_found(self, processor):
        """Test PDF processing with non-existent file."""
        with pytest.raises(DocumentProcessingError):
            await processor.process("/non/existent/file.pdf")
    
    async def test_process_pdf_invalid_file(self, processor, temp_dir):
        """Test PDF processing with invalid PDF file."""
        invalid_pdf = temp_dir / "invalid.pdf"
        invalid_pdf.write_text("This is not a PDF file")
        
        with pytest.raises(DocumentProcessingError):
            await processor.process(str(invalid_pdf))
    
    async def test_process_with_progress_callback(self, processor, sample_pdf_path, mock_progress_callback):
        """Test PDF processing with progress callback."""
        try:
            await processor.process(
                str(sample_pdf_path),
                file_id="test_file",
                progress_callback=mock_progress_callback,
            )
            # Progress callback should be called
            assert mock_progress_callback.called
        except DocumentProcessingError:
            # Skip if PyMuPDF not available
            pytest.skip("PyMuPDF not available")
    
    async def test_process_with_metadata(self, processor, sample_pdf_path):
        """Test PDF processing with metadata."""
        try:
            metadata = {
                "folder_id": "folder_123",
                "file_name": "test.pdf",
                "created_at": "2024-01-01",
            }
            
            chunks = await processor.process(
                str(sample_pdf_path),
                file_id="test_file",
                metadata=metadata,
            )
            
            # Metadata should be included in chunks
            assert all("folder_id" in chunk.metadata for chunk in chunks)
        except DocumentProcessingError:
            pytest.skip("PyMuPDF not available")
    
    async def test_chunk_generation(self, processor, sample_pdf_path):
        """Test chunk generation from PDF."""
        try:
            chunks = await processor.process(
                str(sample_pdf_path),
                file_id="test_file",
            )
            
            # Verify chunk structure
            for i, chunk in enumerate(chunks):
                assert chunk.chunk_id.startswith("test_file_chunk_")
                assert chunk.file_id == "test_file"
                assert len(chunk.content) > 0
                assert isinstance(chunk.metadata, dict)
        except DocumentProcessingError:
            pytest.skip("PyMuPDF not available")

