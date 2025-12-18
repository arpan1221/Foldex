"""Unit tests for BaseProcessor interface."""

import pytest
from unittest.mock import Mock
from app.processors.base import BaseProcessor
from app.models.documents import DocumentChunk


class ConcreteProcessor(BaseProcessor):
    """Concrete implementation for testing."""
    
    async def can_process(self, file_path: str, mime_type=None):
        return file_path.endswith(".test")
    
    async def process(self, file_path: str, file_id=None, metadata=None, progress_callback=None):
        return [
            DocumentChunk(
                chunk_id="chunk_1",
                file_id=file_id or "test",
                content="Test content",
                metadata=metadata or {},
            )
        ]
    
    def get_supported_extensions(self):
        return [".test"]
    
    def get_supported_mime_types(self):
        return ["application/test"]


@pytest.mark.asyncio
class TestBaseProcessor:
    """Test BaseProcessor abstract interface."""
    
    async def test_can_process(self):
        """Test can_process method."""
        processor = ConcreteProcessor()
        assert await processor.can_process("test.test") is True
        assert await processor.can_process("test.txt") is False
    
    async def test_process(self):
        """Test process method."""
        processor = ConcreteProcessor()
        chunks = await processor.process("test.test", file_id="file_123")
        
        assert len(chunks) == 1
        assert chunks[0].chunk_id == "chunk_1"
        assert chunks[0].file_id == "file_123"
        assert chunks[0].content == "Test content"
    
    def test_generate_chunk_id(self):
        """Test chunk ID generation."""
        processor = ConcreteProcessor()
        chunk_id = processor._generate_chunk_id("file_123", 0)
        assert chunk_id == "file_123_chunk_0"
        
        chunk_id = processor._generate_chunk_id("file_456", 42)
        assert chunk_id == "file_456_chunk_42"
    
    def test_update_progress(self):
        """Test progress callback."""
        processor = ConcreteProcessor()
        callback = Mock()
        
        processor._update_progress(callback, 0.5)
        callback.assert_called_once_with(0.5)
        
        # Test with None callback
        processor._update_progress(None, 0.75)
        # Should not raise error
    
    def test_update_progress_clamping(self):
        """Test progress value clamping."""
        processor = ConcreteProcessor()
        callback = Mock()
        
        # Test values outside 0-1 range
        processor._update_progress(callback, -0.1)
        callback.assert_called_with(0.0)
        
        processor._update_progress(callback, 1.5)
        callback.assert_called_with(1.0)
    
    def test_update_progress_callback_error(self):
        """Test progress callback error handling."""
        processor = ConcreteProcessor()
        callback = Mock(side_effect=Exception("Callback error"))
        
        # Should not raise, just log warning
        processor._update_progress(callback, 0.5)
        callback.assert_called_once()

