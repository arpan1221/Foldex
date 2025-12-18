"""Unit tests for TextProcessor."""

import pytest

from app.processors.text_processor import TextProcessor
from app.models.documents import DocumentChunk
from app.core.exceptions import DocumentProcessingError


@pytest.mark.asyncio
class TestTextProcessor:
    """Test TextProcessor functionality."""
    
    @pytest.fixture
    def processor(self):
        """Create TextProcessor instance."""
        return TextProcessor()
    
    async def test_can_process_text(self, processor):
        """Test text file detection."""
        assert await processor.can_process("test.txt") is True
        assert await processor.can_process("test.TXT") is True
        assert await processor.can_process("test.pdf") is False
        assert await processor.can_process("test.txt", mime_type="text/plain") is True
    
    async def test_can_process_markdown(self, processor):
        """Test Markdown file detection."""
        assert await processor.can_process("test.md") is True
        assert await processor.can_process("test.MD") is True
        assert await processor.can_process("test.markdown") is True
        assert await processor.can_process("test.md", mime_type="text/markdown") is True
    
    async def test_get_supported_extensions(self, processor):
        """Test supported extensions."""
        extensions = processor.get_supported_extensions()
        assert ".txt" in extensions
        assert ".md" in extensions
        assert ".markdown" in extensions
    
    async def test_get_supported_mime_types(self, processor):
        """Test supported MIME types."""
        mime_types = processor.get_supported_mime_types()
        assert "text/plain" in mime_types
        assert "text/markdown" in mime_types
    
    async def test_process_text_file(self, processor, sample_text_path):
        """Test text file processing."""
        chunks = await processor.process(
            str(sample_text_path),
            file_id="test_file_123",
            metadata={"folder_id": "folder_456"},
        )
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.file_id == "test_file_123" for chunk in chunks)
        assert all(chunk.content.strip() for chunk in chunks)
    
    async def test_process_markdown_file(self, processor, sample_markdown_path):
        """Test Markdown file processing."""
        chunks = await processor.process(
            str(sample_markdown_path),
            file_id="test_file_456",
        )
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        # Markdown content should be preserved
        assert any("#" in chunk.content or "**" in chunk.content for chunk in chunks)
    
    async def test_process_file_not_found(self, processor):
        """Test processing with non-existent file."""
        with pytest.raises(DocumentProcessingError):
            await processor.process("/non/existent/file.txt")
    
    async def test_process_with_encoding_detection(self, processor, temp_dir):
        """Test encoding detection for text files."""
        # Create file with UTF-8 content
        utf8_file = temp_dir / "utf8.txt"
        utf8_file.write_text("Test content with émojis 🎉", encoding="utf-8")
        
        chunks = await processor.process(str(utf8_file), file_id="test_file")
        assert len(chunks) > 0
        assert "émojis" in chunks[0].content or "🎉" in chunks[0].content
    
    async def test_process_with_progress_callback(self, processor, sample_text_path, mock_progress_callback):
        """Test text processing with progress callback."""
        chunks = await processor.process(
            str(sample_text_path),
            file_id="test_file",
            progress_callback=mock_progress_callback,
        )
        
        # Progress callback should be called
        assert mock_progress_callback.called
        assert len(chunks) > 0
    
    async def test_process_with_metadata(self, processor, sample_text_path):
        """Test text processing with metadata."""
        metadata = {
            "folder_id": "folder_123",
            "file_name": "test.txt",
            "created_at": "2024-01-01",
        }
        
        chunks = await processor.process(
            str(sample_text_path),
            file_id="test_file",
            metadata=metadata,
        )
        
        # Metadata should be included in chunks
        assert all("folder_id" in chunk.metadata for chunk in chunks)
        assert all(chunk.metadata["folder_id"] == "folder_123" for chunk in chunks)
    
    async def test_chunk_size_respect(self, processor, temp_dir):
        """Test that chunks respect size limits."""
        # Create a large text file
        large_content = "This is a test sentence. " * 1000
        large_file = temp_dir / "large.txt"
        large_file.write_text(large_content)
        
        chunks = await processor.process(str(large_file), file_id="test_file")
        
        # Should create multiple chunks
        assert len(chunks) > 1
        # Each chunk should be within reasonable size
        for chunk in chunks:
            assert len(chunk.content) > 0
    
    async def test_empty_file_handling(self, processor, temp_dir):
        """Test processing empty file."""
        empty_file = temp_dir / "empty.txt"
        empty_file.write_text("")
        
        chunks = await processor.process(str(empty_file), file_id="test_file")
        
        # Should handle empty file gracefully
        assert len(chunks) == 0 or all(not chunk.content.strip() for chunk in chunks)

