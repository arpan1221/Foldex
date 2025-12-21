"""Tests for standardized metadata schema."""

import pytest
from datetime import datetime

from app.ingestion.metadata_schema import (
    MetadataBuilder,
    FileType,
    ChunkType,
)


class TestMetadataBuilder:
    """Test MetadataBuilder class."""

    def test_base_metadata(self):
        """Test base metadata creation."""
        base = MetadataBuilder.base_metadata(
            file_id="123",
            file_name="test.pdf",
            file_type=FileType.PDF,
            chunk_type=ChunkType.DOCUMENT_SECTION,
            chunk_id="chunk_123",
            drive_url="https://drive.google.com/file/123",
        )

        assert base["file_id"] == "123"
        assert base["file_name"] == "test.pdf"
        assert base["file_type"] == "pdf"
        assert base["chunk_type"] == "document_section"
        assert base["chunk_id"] == "chunk_123"
        assert base["drive_url"] == "https://drive.google.com/file/123"
        assert "ingestion_date" in base
        assert base["schema_version"] == "1.0"

    def test_pdf_metadata(self):
        """Test PDF-specific metadata."""
        base = MetadataBuilder.base_metadata(
            file_id="456",
            file_name="paper.pdf",
            file_type=FileType.PDF,
            chunk_type=ChunkType.DOCUMENT_SECTION,
            chunk_id="chunk_456",
            drive_url="https://drive.google.com/file/456",
        )

        pdf_meta = MetadataBuilder.pdf_metadata(
            base=base,
            page_number=5,
            total_pages=12,
            section="Methodology",
            authors=["John Doe", "Jane Smith"],
            title="Research Paper",
        )

        assert pdf_meta["file_type"] == "pdf"
        assert pdf_meta["page_number"] == 5
        assert pdf_meta["total_pages"] == 12
        assert pdf_meta["section"] == "Methodology"
        assert pdf_meta["authors"] == ["John Doe", "Jane Smith"]
        assert pdf_meta["document_title"] == "Research Paper"
        assert pdf_meta["searchable_location"] == "page 5"
        assert pdf_meta["citation_format"] == "paper.pdf, p.5"

    def test_audio_metadata(self):
        """Test audio-specific metadata."""
        base = MetadataBuilder.base_metadata(
            file_id="789",
            file_name="interview.m4a",
            file_type=FileType.AUDIO,
            chunk_type=ChunkType.AUDIO_TRANSCRIPTION,
            chunk_id="chunk_789",
            drive_url="https://drive.google.com/file/789",
        )

        audio_meta = MetadataBuilder.audio_metadata(
            base=base,
            start_time=65.5,
            end_time=72.3,
            segment_index=0,
            total_segments=10,
            speaker="Speaker 1",
            confidence=0.95,
            language="en",
        )

        assert audio_meta["file_type"] == "audio"
        assert audio_meta["chunk_type"] == "audio_transcription"
        assert audio_meta["start_time"] == 65.5
        assert audio_meta["end_time"] == 72.3
        assert audio_meta["duration"] == pytest.approx(6.8, abs=0.1)
        assert audio_meta["segment_index"] == 0
        assert audio_meta["total_segments"] == 10
        assert audio_meta["speaker"] == "Speaker 1"
        assert audio_meta["confidence"] == 0.95
        assert audio_meta["language"] == "en"
        assert audio_meta["timestamp_display"] == "01:05"
        assert audio_meta["time_range"] == "01:05 - 01:12"
        assert audio_meta["searchable_location"] == "at 01:05"
        assert audio_meta["citation_format"] == "interview.m4a [01:05]"

    def test_audio_metadata_long_duration(self):
        """Test audio metadata with long duration (hours)."""
        base = MetadataBuilder.base_metadata(
            file_id="789",
            file_name="long_audio.m4a",
            file_type=FileType.AUDIO,
            chunk_type=ChunkType.AUDIO_TRANSCRIPTION,
            chunk_id="chunk_789",
        )

        audio_meta = MetadataBuilder.audio_metadata(
            base=base,
            start_time=3665.0,  # 1 hour, 1 minute, 5 seconds
            end_time=3670.0,
            segment_index=5,
            total_segments=20,
        )

        assert audio_meta["timestamp_display"] == "01:01:05"
        assert audio_meta["time_range"] == "01:01:05 - 01:01:10"

    def test_text_metadata(self):
        """Test text-specific metadata."""
        base = MetadataBuilder.base_metadata(
            file_id="101",
            file_name="document.txt",
            file_type=FileType.TEXT,
            chunk_type=ChunkType.DOCUMENT_SECTION,
            chunk_id="chunk_101",
        )

        text_meta = MetadataBuilder.text_metadata(
            base=base,
            line_start=10,
            line_end=25,
            section="Introduction",
        )

        assert text_meta["file_type"] == "text"
        assert text_meta["line_start"] == 10
        assert text_meta["line_end"] == 25
        assert text_meta["section"] == "Introduction"
        assert text_meta["searchable_location"] == "lines 10-25"
        assert text_meta["citation_format"] == "document.txt, lines 10-25"

    def test_code_metadata(self):
        """Test code-specific metadata."""
        base = MetadataBuilder.base_metadata(
            file_id="202",
            file_name="script.py",
            file_type=FileType.CODE,
            chunk_type=ChunkType.CODE_FUNCTION,
            chunk_id="chunk_202",
        )

        code_meta = MetadataBuilder.code_metadata(
            base=base,
            language="python",
            function_name="process_data",
            class_name="DataProcessor",
            line_start=50,
            line_end=75,
        )

        assert code_meta["file_type"] == "code"
        assert code_meta["chunk_type"] == "code_function"
        assert code_meta["language"] == "python"
        assert code_meta["function_name"] == "process_data"
        assert code_meta["class_name"] == "DataProcessor"
        assert code_meta["line_start"] == 50
        assert code_meta["line_end"] == 75
        assert code_meta["searchable_location"] == "DataProcessor.process_data"
        assert code_meta["citation_format"] == "script.py:50"

    def test_code_metadata_function_only(self):
        """Test code metadata with function but no class."""
        base = MetadataBuilder.base_metadata(
            file_id="303",
            file_name="utils.py",
            file_type=FileType.CODE,
            chunk_type=ChunkType.CODE_FUNCTION,
            chunk_id="chunk_303",
        )

        code_meta = MetadataBuilder.code_metadata(
            base=base,
            language="python",
            function_name="helper_function",
            class_name="",
            line_start=100,
            line_end=120,
        )

        assert code_meta["searchable_location"] == "helper_function"
        assert code_meta["citation_format"] == "utils.py:100"

    def test_merge_metadata(self):
        """Test merging additional metadata."""
        base = MetadataBuilder.base_metadata(
            file_id="404",
            file_name="test.pdf",
            file_type=FileType.PDF,
            chunk_type=ChunkType.DOCUMENT_SECTION,
            chunk_id="chunk_404",
        )

        additional = {
            "custom_field": "custom_value",
            "another_field": 123,
            "list_field": ["item1", "item2"],
            "none_field": None,
        }

        merged = MetadataBuilder.merge_metadata(base, additional)

        assert merged["custom_field"] == "custom_value"
        assert merged["another_field"] == 123
        assert merged["list_field"] == "item1, item2"
        assert "none_field" not in merged or merged["none_field"] is None

    def test_merge_metadata_preserves_base(self):
        """Test that merge doesn't overwrite base fields."""
        base = MetadataBuilder.base_metadata(
            file_id="505",
            file_name="test.pdf",
            file_type=FileType.PDF,
            chunk_type=ChunkType.DOCUMENT_SECTION,
            chunk_id="chunk_505",
        )

        additional = {
            "file_id": "should_not_override",
            "file_type": "should_not_override",
        }

        merged = MetadataBuilder.merge_metadata(base, additional)

        # Base fields should be preserved
        assert merged["file_id"] == "505"
        assert merged["file_type"] == "pdf"
        # But additional fields are still added
        assert "file_id" in merged  # The key exists, but value is from base

    def test_format_time(self):
        """Test time formatting."""
        # Test seconds only
        assert MetadataBuilder._format_time(65.5) == "01:05"
        assert MetadataBuilder._format_time(125.0) == "02:05"

        # Test hours
        assert MetadataBuilder._format_time(3665.0) == "01:01:05"
        assert MetadataBuilder._format_time(7325.0) == "02:02:05"

        # Test edge cases
        assert MetadataBuilder._format_time(0.0) == "00:00"
        assert MetadataBuilder._format_time(59.0) == "00:59"
        assert MetadataBuilder._format_time(3600.0) == "01:00:00"

    def test_metadata_schema_version(self):
        """Test that schema version is consistent."""
        base1 = MetadataBuilder.base_metadata(
            file_id="1",
            file_name="test1.pdf",
            file_type=FileType.PDF,
            chunk_type=ChunkType.DOCUMENT_SECTION,
            chunk_id="chunk1",
        )

        base2 = MetadataBuilder.base_metadata(
            file_id="2",
            file_name="test2.pdf",
            file_type=FileType.PDF,
            chunk_type=ChunkType.DOCUMENT_SECTION,
            chunk_id="chunk2",
        )

        assert base1["schema_version"] == base2["schema_version"]
        assert base1["schema_version"] == MetadataBuilder.SCHEMA_VERSION

    def test_all_file_types(self):
        """Test that all FileType enum values work."""
        for file_type in FileType:
            base = MetadataBuilder.base_metadata(
                file_id="test",
                file_name="test.file",
                file_type=file_type,
                chunk_type=ChunkType.GENERIC_TEXT,
                chunk_id="chunk_test",
            )
            assert base["file_type"] == file_type.value

    def test_all_chunk_types(self):
        """Test that all ChunkType enum values work."""
        for chunk_type in ChunkType:
            base = MetadataBuilder.base_metadata(
                file_id="test",
                file_name="test.file",
                file_type=FileType.UNKNOWN,
                chunk_type=chunk_type,
                chunk_id="chunk_test",
            )
            assert base["chunk_type"] == chunk_type.value


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

