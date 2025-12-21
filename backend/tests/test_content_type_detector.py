"""Tests for content type detection."""

import pytest
from app.rag.content_type_detector import ContentTypeDetector, ContentType


class TestContentTypeDetector:
    """Test ContentTypeDetector class."""

    def test_audio_detection(self):
        """Test audio content type detection."""
        detector = ContentTypeDetector()
        
        result = detector.detect("what is the audio file about?")
        assert result["content_type"] == ContentType.AUDIO
        assert result["confidence"] > 0
        assert result["metadata_filter"] == {"file_type": {"$in": ["audio"]}}
        assert "audio" in result["explanation"].lower()

    def test_explicit_file_reference(self):
        """Test explicit file reference detection."""
        detector = ContentTypeDetector()
        
        files = [
            {"file_id": "123", "file_name": "interview.m4a"},
            {"file_id": "456", "file_name": "paper.pdf"},
        ]
        
        result = detector.detect("what is in interview.m4a?", files)
        assert result["content_type"] == ContentType.AUDIO
        assert result["confidence"] == 1.0
        assert result["metadata_filter"] == {"file_id": {"$eq": "123"}}
        assert "interview.m4a" in result["explanation"]

    def test_document_detection(self):
        """Test document content type detection."""
        detector = ContentTypeDetector()
        
        result = detector.detect("what does the paper say about methodology?")
        assert result["content_type"] == ContentType.DOCUMENT
        assert result["confidence"] > 0
        assert "pdf" in result["metadata_filter"]["file_type"]["$in"] or "text" in result["metadata_filter"]["file_type"]["$in"]

    def test_code_detection(self):
        """Test code content type detection."""
        detector = ContentTypeDetector()
        
        result = detector.detect("what does the function do?")
        assert result["content_type"] == ContentType.CODE
        assert result["confidence"] > 0
        assert result["metadata_filter"] == {"file_type": {"$in": ["code"]}}

    def test_no_specific_type(self):
        """Test queries with no specific content type."""
        detector = ContentTypeDetector()
        
        result = detector.detect("what are the common themes?")
        assert result["content_type"] == ContentType.ANY
        assert result["confidence"] == 0.0
        assert result["metadata_filter"] is None
        assert "all files" in result["explanation"].lower()

    def test_video_detection(self):
        """Test video content type detection."""
        detector = ContentTypeDetector()
        
        result = detector.detect("what does the video show?")
        assert result["content_type"] == ContentType.VIDEO
        assert result["confidence"] > 0
        assert result["metadata_filter"] == {"file_type": {"$in": ["video"]}}

    def test_multiple_keywords(self):
        """Test detection with multiple keywords increases confidence."""
        detector = ContentTypeDetector()
        
        # Single keyword
        result1 = detector.detect("audio")
        confidence1 = result1["confidence"]
        
        # Multiple keywords
        result2 = detector.detect("what is in the audio recording transcription?")
        confidence2 = result2["confidence"]
        
        assert result2["content_type"] == ContentType.AUDIO
        assert confidence2 >= confidence1  # More keywords should increase confidence

    def test_file_extension_detection(self):
        """Test detection from file extensions."""
        detector = ContentTypeDetector()
        
        files = [
            {"file_id": "1", "file_name": "script.py"},
            {"file_id": "2", "file_name": "document.pdf"},
        ]
        
        result = detector.detect("what is in script.py?", files)
        assert result["content_type"] == ContentType.CODE
        assert result["confidence"] == 1.0
        assert result["metadata_filter"] == {"file_id": {"$eq": "1"}}

    def test_case_insensitive(self):
        """Test that detection is case insensitive."""
        detector = ContentTypeDetector()
        
        result1 = detector.detect("AUDIO FILE")
        result2 = detector.detect("audio file")
        
        assert result1["content_type"] == result2["content_type"]
        assert result1["confidence"] == result2["confidence"]

    def test_empty_query(self):
        """Test empty query handling."""
        detector = ContentTypeDetector()
        
        result = detector.detect("")
        assert result["content_type"] == ContentType.ANY
        assert result["metadata_filter"] is None

    def test_no_files_provided(self):
        """Test detection without available files."""
        detector = ContentTypeDetector()
        
        result = detector.detect("what is in interview.m4a?", available_files=None)
        # Should still detect audio from keywords, not file reference
        assert result["content_type"] == ContentType.AUDIO
        assert result["metadata_filter"] == {"file_type": {"$in": ["audio"]}}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

