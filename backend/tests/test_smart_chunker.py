"""Tests for smart chunking with structure preservation."""

import pytest
from app.ingestion.smart_chunker import SmartChunker
from app.ingestion.metadata_schema import FileType, ChunkType

try:
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.schema import Document
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        Document = None


@pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="LangChain not available")
class TestSmartChunker:
    """Test SmartChunker class."""

    def test_basic_chunking(self):
        """Test basic chunking with context."""
        chunker = SmartChunker(chunk_size=100, chunk_overlap=20, context_window=30)

        text = "This is a test document. " * 10  # ~280 chars
        file_metadata = {
            "file_id": "123",
            "file_name": "test.txt",
            "mime_type": "text/plain",
        }

        chunks = chunker.chunk_with_structure(text, file_metadata)

        assert len(chunks) > 0
        assert all(isinstance(chunk, Document) for chunk in chunks)
        assert all(chunk.metadata.get("file_id") == "123" for chunk in chunks)
        assert all("prev_context" in chunk.metadata for chunk in chunks)
        assert all("next_context" in chunk.metadata for chunk in chunks)

    def test_section_aware_chunking(self):
        """Test chunking with section boundaries."""
        chunker = SmartChunker(chunk_size=100, chunk_overlap=20, context_window=30)

        text = "Section 1 content. " * 20 + "\n\nSection 2 content. " * 20
        file_metadata = {
            "file_id": "456",
            "file_name": "test.md",
            "mime_type": "text/markdown",
        }

        sections = [
            {"title": "Introduction", "start": 0, "end": 200},
            {"title": "Methods", "start": 200, "end": 400},
        ]

        chunks = chunker.chunk_with_structure(text, file_metadata, sections=sections)

        assert len(chunks) > 0
        # Check that section info is preserved
        assert any("Introduction" in chunk.metadata.get("section", "") for chunk in chunks)
        assert any("Methods" in chunk.metadata.get("section", "") for chunk in chunks)
        assert all("section_index" in chunk.metadata for chunk in chunks)

    def test_header_preservation(self):
        """Test chunking with headers preserved."""
        chunker = SmartChunker(chunk_size=100, chunk_overlap=20, context_window=30)

        text = """# Introduction

This is the introduction section with some content.

# Methods

This describes the methodology.

# Results

These are the results."""
        
        file_metadata = {
            "file_id": "789",
            "file_name": "paper.md",
            "mime_type": "text/markdown",
        }

        chunks = chunker.chunk_with_headers(text, file_metadata)

        assert len(chunks) > 0
        # Check that headers are in chunk content
        assert any("Introduction" in chunk.page_content for chunk in chunks)
        assert any("Methods" in chunk.page_content for chunk in chunks)
        assert any("Results" in chunk.page_content for chunk in chunks)

    def test_markdown_chunking(self):
        """Test Markdown-specific chunking."""
        chunker = SmartChunker(chunk_size=100, chunk_overlap=20, context_window=30)

        text = """# Title

## Subsection 1

Content here.

## Subsection 2

More content."""
        
        file_metadata = {
            "file_id": "101",
            "file_name": "doc.md",
            "mime_type": "text/markdown",
        }

        chunks = chunker.chunk_markdown(text, file_metadata)

        assert len(chunks) > 0
        # Should preserve headers
        assert any("Title" in chunk.page_content or "Subsection" in chunk.page_content for chunk in chunks)

    def test_context_windows(self):
        """Test that context windows are properly added."""
        chunker = SmartChunker(chunk_size=50, chunk_overlap=10, context_window=20)

        text = "Chunk 1 content. " * 5 + "Chunk 2 content. " * 5 + "Chunk 3 content. " * 5
        file_metadata = {
            "file_id": "202",
            "file_name": "test.txt",
        }

        chunks = chunker.chunk_with_structure(text, file_metadata)

        assert len(chunks) >= 2

        # Check middle chunk has both prev and next context
        if len(chunks) >= 2:
            middle_chunk = chunks[1]
            assert "prev_context" in middle_chunk.metadata
            assert "next_context" in middle_chunk.metadata
            assert len(middle_chunk.metadata["prev_context"]) <= chunker.context_window
            assert len(middle_chunk.metadata["next_context"]) <= chunker.context_window

        # First chunk should have no prev context
        assert chunks[0].metadata.get("prev_context") == ""

        # Last chunk should have no next context
        assert chunks[-1].metadata.get("next_context") == ""

    def test_metadata_schema_compliance(self):
        """Test that chunks use standardized metadata schema."""
        chunker = SmartChunker(chunk_size=100, chunk_overlap=20)

        text = "Test content here."
        file_metadata = {
            "file_id": "303",
            "file_name": "test.pdf",
            "mime_type": "application/pdf",
            "drive_url": "https://drive.google.com/file/303",
        }

        chunks = chunker.chunk_with_structure(text, file_metadata)

        assert len(chunks) > 0

        # Check standardized metadata fields
        for chunk in chunks:
            metadata = chunk.metadata
            assert "file_id" in metadata
            assert "file_name" in metadata
            assert "file_type" in metadata
            assert "chunk_type" in metadata
            assert "chunk_id" in metadata
            assert "ingestion_date" in metadata
            assert "schema_version" in metadata
            assert metadata["file_type"] == FileType.PDF.value
            assert metadata["chunk_type"] == ChunkType.DOCUMENT_SECTION.value

    def test_semantic_boundaries(self):
        """Test chunking with semantic boundaries."""
        chunker = SmartChunker(chunk_size=100, chunk_overlap=20)

        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."
        file_metadata = {
            "file_id": "404",
            "file_name": "test.txt",
        }

        chunks = chunker.chunk_with_semantic_boundaries(
            text,
            file_metadata,
            boundary_markers=["\n\n"],  # Paragraph boundaries
        )

        assert len(chunks) > 0
        # Should respect paragraph boundaries
        assert all(isinstance(chunk, Document) for chunk in chunks)

    def test_empty_text(self):
        """Test handling of empty text."""
        chunker = SmartChunker()

        chunks = chunker.chunk_with_structure("", {"file_id": "505", "file_name": "empty.txt"})

        assert chunks == []

    def test_file_type_detection(self):
        """Test automatic file type detection."""
        chunker = SmartChunker()

        # Test PDF
        chunks = chunker.chunk_with_structure(
            "PDF content",
            {"file_id": "1", "file_name": "doc.pdf", "mime_type": "application/pdf"}
        )
        assert chunks[0].metadata["file_type"] == FileType.PDF.value

        # Test Markdown
        chunks = chunker.chunk_with_structure(
            "Markdown content",
            {"file_id": "2", "file_name": "readme.md", "mime_type": "text/markdown"}
        )
        assert chunks[0].metadata["file_type"] == FileType.MARKDOWN.value

        # Test Code
        chunks = chunker.chunk_with_structure(
            "Code content",
            {"file_id": "3", "file_name": "script.py"}
        )
        assert chunks[0].metadata["file_type"] == FileType.CODE.value

    def test_chunk_id_generation(self):
        """Test that chunk IDs are unique and deterministic."""
        chunker = SmartChunker()

        text = "Test content"
        file_metadata = {"file_id": "606", "file_name": "test.txt"}

        chunks1 = chunker.chunk_with_structure(text, file_metadata)
        chunks2 = chunker.chunk_with_structure(text, file_metadata)

        # Same content should generate same IDs
        assert chunks1[0].metadata["chunk_id"] == chunks2[0].metadata["chunk_id"]

        # Different content should generate different IDs
        chunks3 = chunker.chunk_with_structure("Different content", file_metadata)
        assert chunks1[0].metadata["chunk_id"] != chunks3[0].metadata["chunk_id"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

