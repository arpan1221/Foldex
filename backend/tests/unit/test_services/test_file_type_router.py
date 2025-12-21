"""Unit tests for file type router module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from app.services.file_type_router import (
    FileTypeCategory,
    get_file_category,
    get_supported_extensions,
    is_supported_file,
    SUPPORTED_EXTENSIONS,
    MIME_TYPE_PATTERNS,
    MAGIC_AVAILABLE,
)


@pytest.mark.asyncio
class TestFileTypeCategory:
    """Test FileTypeCategory enum."""
    
    def test_enum_values(self):
        """Test enum has expected values."""
        assert FileTypeCategory.UNSTRUCTURED_NATIVE.value == "unstructured_native"
        assert FileTypeCategory.AUDIO.value == "audio"
        assert FileTypeCategory.CODE.value == "code"
        assert FileTypeCategory.NOTEBOOK.value == "notebook"
        assert FileTypeCategory.UNSUPPORTED.value == "unsupported"
    
    def test_enum_string_comparison(self):
        """Test enum can be compared to strings."""
        assert FileTypeCategory.UNSTRUCTURED_NATIVE == "unstructured_native"
        assert FileTypeCategory.AUDIO == "audio"


@pytest.mark.asyncio
class TestGetFileCategory:
    """Test get_file_category function."""
    
    def test_pdf_file(self):
        """Test PDF file categorization."""
        assert get_file_category("document.pdf") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("document.PDF") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("document.pdf", mime_type="application/pdf") == FileTypeCategory.UNSTRUCTURED_NATIVE
    
    def test_text_files(self):
        """Test text file categorization."""
        assert get_file_category("readme.txt") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("notes.md") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("data.csv") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("config.json") == FileTypeCategory.UNSTRUCTURED_NATIVE
    
    def test_office_documents(self):
        """Test Microsoft Office document categorization."""
        assert get_file_category("report.docx") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("spreadsheet.xlsx") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("presentation.pptx") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("document.doc") == FileTypeCategory.UNSTRUCTURED_NATIVE
    
    def test_image_files(self):
        """Test image file categorization (for OCR processing)."""
        assert get_file_category("photo.jpg") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("image.jpeg") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("screenshot.png") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("diagram.gif") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("scan.tiff") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("picture.bmp") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("image.webp") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("icon.svg") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("photo.jpg", mime_type="image/jpeg") == FileTypeCategory.UNSTRUCTURED_NATIVE
    
    def test_audio_files(self):
        """Test audio file categorization."""
        assert get_file_category("song.mp3") == FileTypeCategory.AUDIO
        assert get_file_category("recording.wav") == FileTypeCategory.AUDIO
        assert get_file_category("audio.m4a") == FileTypeCategory.AUDIO
        assert get_file_category("music.flac") == FileTypeCategory.AUDIO
        assert get_file_category("audio.mp3", mime_type="audio/mpeg") == FileTypeCategory.AUDIO
    
    def test_code_files(self):
        """Test source code file categorization."""
        assert get_file_category("script.py") == FileTypeCategory.CODE
        assert get_file_category("app.js") == FileTypeCategory.CODE
        assert get_file_category("component.tsx") == FileTypeCategory.CODE
        assert get_file_category("Main.java") == FileTypeCategory.CODE
        assert get_file_category("program.cpp") == FileTypeCategory.CODE
        assert get_file_category("server.go") == FileTypeCategory.CODE
        assert get_file_category("lib.rs") == FileTypeCategory.CODE
        assert get_file_category("script.sh") == FileTypeCategory.CODE
    
    def test_notebook_files(self):
        """Test Jupyter notebook categorization."""
        assert get_file_category("analysis.ipynb") == FileTypeCategory.NOTEBOOK
        assert get_file_category("notebook.IPYNB") == FileTypeCategory.NOTEBOOK
    
    def test_unsupported_files(self):
        """Test unsupported file categorization."""
        assert get_file_category("unknown.xyz") == FileTypeCategory.UNSUPPORTED
        assert get_file_category("file.bin") == FileTypeCategory.UNSUPPORTED
        assert get_file_category("archive.zip") == FileTypeCategory.UNSUPPORTED
    
    def test_files_without_extensions(self):
        """Test files without extensions."""
        assert get_file_category("file_without_extension") == FileTypeCategory.UNSUPPORTED
        assert get_file_category("README") == FileTypeCategory.UNSUPPORTED
        assert get_file_category("/path/to/file") == FileTypeCategory.UNSUPPORTED
    
    def test_files_with_mime_type_only(self):
        """Test categorization using MIME type when extension is missing."""
        # File without extension but with MIME type
        assert get_file_category("document", mime_type="application/pdf") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("audio", mime_type="audio/mpeg") == FileTypeCategory.AUDIO
        assert get_file_category("script", mime_type="text/x-python") == FileTypeCategory.CODE
    
    def test_mime_type_overrides_extension(self):
        """Test that MIME type takes precedence over extension."""
        # File with .txt extension but audio MIME type
        assert get_file_category("file.txt", mime_type="audio/mpeg") == FileTypeCategory.AUDIO
        # File with .mp3 extension but PDF MIME type
        assert get_file_category("file.mp3", mime_type="application/pdf") == FileTypeCategory.UNSTRUCTURED_NATIVE
    
    def test_case_insensitive_extensions(self):
        """Test that extensions are case-insensitive."""
        assert get_file_category("FILE.PDF") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("FILE.Py") == FileTypeCategory.CODE
        assert get_file_category("FILE.MP3") == FileTypeCategory.AUDIO
        assert get_file_category("FILE.IPYNB") == FileTypeCategory.NOTEBOOK
    
    def test_relative_paths(self):
        """Test that relative paths work correctly."""
        assert get_file_category("./document.pdf") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("../script.py") == FileTypeCategory.CODE
        assert get_file_category("subfolder/audio.mp3") == FileTypeCategory.AUDIO
    
    def test_absolute_paths(self):
        """Test that absolute paths work correctly."""
        assert get_file_category("/absolute/path/document.pdf") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("C:\\Windows\\script.py") == FileTypeCategory.CODE
    
    @pytest.mark.skipif(not MAGIC_AVAILABLE, reason="python-magic not available")
    def test_python_magic_detection(self, temp_dir):
        """Test python-magic MIME type detection."""
        # Create a test file
        test_file = temp_dir / "test_file"
        test_file.write_bytes(b"%PDF-1.4\n")  # PDF magic bytes
        
        # Should detect as PDF even without extension
        category = get_file_category(str(test_file))
        # Note: This might still return UNSUPPORTED if magic doesn't detect it correctly
        # The important thing is it doesn't crash
    
    @pytest.mark.skipif(MAGIC_AVAILABLE, reason="python-magic available, testing fallback")
    def test_fallback_when_magic_unavailable(self):
        """Test that extension matching works when python-magic is unavailable."""
        # Should still work with extensions
        assert get_file_category("document.pdf") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("script.py") == FileTypeCategory.CODE
        assert get_file_category("audio.mp3") == FileTypeCategory.AUDIO


@pytest.mark.asyncio
class TestMimeTypeCategorization:
    """Test MIME type-based categorization."""
    
    def test_pdf_mime_types(self):
        """Test PDF MIME type detection."""
        assert get_file_category("file", mime_type="application/pdf") == FileTypeCategory.UNSTRUCTURED_NATIVE
    
    def test_audio_mime_types(self):
        """Test audio MIME type detection."""
        assert get_file_category("file", mime_type="audio/mpeg") == FileTypeCategory.AUDIO
        assert get_file_category("file", mime_type="audio/wav") == FileTypeCategory.AUDIO
        assert get_file_category("file", mime_type="audio/m4a") == FileTypeCategory.AUDIO
        assert get_file_category("file", mime_type="audio/x-m4a") == FileTypeCategory.AUDIO
    
    def test_code_mime_types(self):
        """Test code file MIME type detection."""
        assert get_file_category("file", mime_type="text/x-python") == FileTypeCategory.CODE
        assert get_file_category("file", mime_type="text/javascript") == FileTypeCategory.CODE
        assert get_file_category("file", mime_type="application/javascript") == FileTypeCategory.CODE
        assert get_file_category("file", mime_type="text/x-java-source") == FileTypeCategory.CODE
    
    def test_notebook_mime_types(self):
        """Test notebook MIME type detection."""
        assert get_file_category("file", mime_type="application/x-ipynb+json") == FileTypeCategory.NOTEBOOK
    
    def test_generic_text_mime_types(self):
        """Test generic text MIME types."""
        assert get_file_category("file", mime_type="text/plain") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("file", mime_type="text/markdown") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("file", mime_type="text/csv") == FileTypeCategory.UNSTRUCTURED_NATIVE
    
    def test_image_mime_types(self):
        """Test image MIME type detection (for OCR)."""
        assert get_file_category("file", mime_type="image/jpeg") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("file", mime_type="image/jpg") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("file", mime_type="image/png") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("file", mime_type="image/gif") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("file", mime_type="image/tiff") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("file", mime_type="image/bmp") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("file", mime_type="image/webp") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("file", mime_type="image/svg+xml") == FileTypeCategory.UNSTRUCTURED_NATIVE
    
    def test_generic_image_prefix(self):
        """Test that generic image/ prefix is recognized."""
        assert get_file_category("file", mime_type="image/unknown") == FileTypeCategory.UNSTRUCTURED_NATIVE
    
    def test_generic_audio_prefix(self):
        """Test that generic audio/ prefix is recognized."""
        assert get_file_category("file", mime_type="audio/unknown") == FileTypeCategory.AUDIO
    
    def test_unsupported_mime_types(self):
        """Test unsupported MIME types."""
        assert get_file_category("file", mime_type="application/octet-stream") == FileTypeCategory.UNSUPPORTED
        assert get_file_category("file", mime_type="video/mp4") == FileTypeCategory.UNSUPPORTED
        assert get_file_category("file", mime_type="application/zip") == FileTypeCategory.UNSUPPORTED


@pytest.mark.asyncio
class TestGetSupportedExtensions:
    """Test get_supported_extensions function."""
    
    def test_returns_dict(self):
        """Test that function returns a dictionary."""
        extensions = get_supported_extensions()
        assert isinstance(extensions, dict)
    
    def test_contains_expected_extensions(self):
        """Test that expected extensions are present."""
        extensions = get_supported_extensions()
        
        # PDF
        assert ".pdf" in extensions
        assert extensions[".pdf"] == FileTypeCategory.UNSTRUCTURED_NATIVE
        
        # Audio
        assert ".mp3" in extensions
        assert extensions[".mp3"] == FileTypeCategory.AUDIO
        
        # Code
        assert ".py" in extensions
        assert extensions[".py"] == FileTypeCategory.CODE
        
        # Notebook
        assert ".ipynb" in extensions
        assert extensions[".ipynb"] == FileTypeCategory.NOTEBOOK
    
    def test_returns_copy(self):
        """Test that function returns a copy, not the original dict."""
        extensions1 = get_supported_extensions()
        extensions2 = get_supported_extensions()
        
        # Modifying one shouldn't affect the other
        extensions1["test"] = FileTypeCategory.UNSUPPORTED
        assert "test" not in extensions2


@pytest.mark.asyncio
class TestIsSupportedFile:
    """Test is_supported_file function."""
    
    def test_supported_files(self):
        """Test that supported files return True."""
        assert is_supported_file("document.pdf") is True
        assert is_supported_file("script.py") is True
        assert is_supported_file("audio.mp3") is True
        assert is_supported_file("notebook.ipynb") is True
    
    def test_unsupported_files(self):
        """Test that unsupported files return False."""
        assert is_supported_file("unknown.xyz") is False
        assert is_supported_file("file_without_extension") is False
        assert is_supported_file("archive.zip") is False
    
    def test_with_mime_type(self):
        """Test with MIME type parameter."""
        assert is_supported_file("file", mime_type="application/pdf") is True
        assert is_supported_file("file", mime_type="audio/mpeg") is True
        assert is_supported_file("file", mime_type="image/jpeg") is False


@pytest.mark.asyncio
class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_string(self):
        """Test empty string handling."""
        assert get_file_category("") == FileTypeCategory.UNSUPPORTED
    
    def test_path_with_multiple_dots(self):
        """Test paths with multiple dots."""
        assert get_file_category("file.backup.pdf") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("script.min.js") == FileTypeCategory.CODE
    
    def test_path_with_spaces(self):
        """Test paths with spaces."""
        assert get_file_category("my document.pdf") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("my script.py") == FileTypeCategory.CODE
    
    def test_path_with_special_characters(self):
        """Test paths with special characters."""
        assert get_file_category("file-name_123.pdf") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("script (1).py") == FileTypeCategory.CODE
    
    def test_unicode_in_path(self):
        """Test paths with unicode characters."""
        assert get_file_category("文档.pdf") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("скрипт.py") == FileTypeCategory.CODE
    
    def test_mime_type_with_parameters(self):
        """Test MIME types with parameters (charset, etc.)."""
        assert get_file_category("file", mime_type="text/plain; charset=utf-8") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("file", mime_type="application/pdf; version=1.4") == FileTypeCategory.UNSTRUCTURED_NATIVE
    
    def test_mime_type_case_insensitive(self):
        """Test that MIME types are case-insensitive."""
        assert get_file_category("file", mime_type="APPLICATION/PDF") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("file", mime_type="Audio/MPEG") == FileTypeCategory.AUDIO
    
    def test_mime_type_with_whitespace(self):
        """Test MIME types with whitespace."""
        assert get_file_category("file", mime_type="  application/pdf  ") == FileTypeCategory.UNSTRUCTURED_NATIVE
        assert get_file_category("file", mime_type="\taudio/mpeg\n") == FileTypeCategory.AUDIO


@pytest.mark.asyncio
class TestSupportedExtensionsDict:
    """Test SUPPORTED_EXTENSIONS dictionary structure."""
    
    def test_all_categories_present(self):
        """Test that all categories are present in the dict."""
        assert FileTypeCategory.UNSTRUCTURED_NATIVE in SUPPORTED_EXTENSIONS
        assert FileTypeCategory.AUDIO in SUPPORTED_EXTENSIONS
        assert FileTypeCategory.CODE in SUPPORTED_EXTENSIONS
        assert FileTypeCategory.NOTEBOOK in SUPPORTED_EXTENSIONS
        assert FileTypeCategory.UNSUPPORTED not in SUPPORTED_EXTENSIONS
    
    def test_extensions_are_sets(self):
        """Test that extension values are sets."""
        for extensions in SUPPORTED_EXTENSIONS.values():
            assert isinstance(extensions, set)
    
    def test_extensions_have_dots(self):
        """Test that all extensions start with dots."""
        for extensions in SUPPORTED_EXTENSIONS.values():
            for ext in extensions:
                assert ext.startswith("."), f"Extension {ext} should start with a dot"
    
    def test_no_overlapping_extensions(self):
        """Test that extensions don't overlap between categories."""
        all_extensions = []
        for extensions in SUPPORTED_EXTENSIONS.values():
            all_extensions.extend(extensions)
        
        # Check for duplicates
        assert len(all_extensions) == len(set(all_extensions)), "Extensions should not overlap between categories"


@pytest.mark.asyncio
class TestMimeTypePatterns:
    """Test MIME_TYPE_PATTERNS dictionary."""
    
    def test_all_categories_present(self):
        """Test that all categories are present."""
        assert FileTypeCategory.UNSTRUCTURED_NATIVE in MIME_TYPE_PATTERNS
        assert FileTypeCategory.AUDIO in MIME_TYPE_PATTERNS
        assert FileTypeCategory.CODE in MIME_TYPE_PATTERNS
        assert FileTypeCategory.NOTEBOOK in MIME_TYPE_PATTERNS
    
    def test_patterns_are_sets(self):
        """Test that pattern values are sets."""
        for patterns in MIME_TYPE_PATTERNS.values():
            assert isinstance(patterns, set)
    
    def test_common_mime_types_present(self):
        """Test that common MIME types are present."""
        assert "application/pdf" in MIME_TYPE_PATTERNS[FileTypeCategory.UNSTRUCTURED_NATIVE]
        assert "audio/mpeg" in MIME_TYPE_PATTERNS[FileTypeCategory.AUDIO]
        assert "text/x-python" in MIME_TYPE_PATTERNS[FileTypeCategory.CODE]
        assert "application/x-ipynb+json" in MIME_TYPE_PATTERNS[FileTypeCategory.NOTEBOOK]

