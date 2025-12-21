"""File type categorization and routing for multimodal processing.

This module provides file type detection and categorization for Foldex's
local-first multimodal RAG system. It uses python-magic for reliable
MIME type detection with fallback to extension matching.

The categorization supports routing files to appropriate processors:
- UNSTRUCTURED_NATIVE: Text, PDFs, Office docs, images (for OCR) processed by Unstructured.io
- AUDIO: Audio files for Whisper transcription
- CODE: Source code files with syntax preservation
- NOTEBOOK: Jupyter notebooks
- UNSUPPORTED: Files that cannot be processed

All processing happens locally with no cloud dependencies, maintaining
the local-first architecture principle.
"""

from enum import Enum
from pathlib import Path
from typing import Optional, Dict, Set
import structlog

try:
    import magic
    MAGIC_AVAILABLE = True
except ImportError:
    MAGIC_AVAILABLE = False
    magic = None

logger = structlog.get_logger(__name__)


class FileTypeCategory(str, Enum):
    """File type categories for routing to appropriate processors.
    
    Categories are designed to align with Foldex's processor architecture
    and support local-first processing without cloud dependencies.
    
    Attributes:
        UNSTRUCTURED_NATIVE: Files processed by Unstructured.io (PDFs, Office docs, text, images for OCR)
        AUDIO: Audio files for Whisper transcription
        CODE: Source code files requiring syntax preservation
        NOTEBOOK: Jupyter notebook files
        UNSUPPORTED: Files that cannot be processed
    """
    
    UNSTRUCTURED_NATIVE = "unstructured_native"
    AUDIO = "audio"
    CODE = "code"
    NOTEBOOK = "notebook"
    UNSUPPORTED = "unsupported"


# Extension to category mapping
# Organized by category for maintainability
SUPPORTED_EXTENSIONS: Dict[FileTypeCategory, Set[str]] = {
    FileTypeCategory.UNSTRUCTURED_NATIVE: {
        # Text files
        ".txt", ".md", ".markdown", ".rst", ".csv", ".json", ".xml", ".html", ".htm",
        # PDFs
        ".pdf",
        # Microsoft Office
        ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        # OpenDocument
        ".odt", ".ods", ".odp",
        # Google Docs (exported)
        ".gdoc", ".gsheet", ".gslides",
        # Other document formats
        ".rtf", ".epub", ".mobi",
        # Image files (for OCR processing)
        ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp", ".svg",
    },
    FileTypeCategory.AUDIO: {
        ".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma", ".opus",
    },
    FileTypeCategory.CODE: {
        # Python
        ".py", ".pyw", ".pyc",
        # JavaScript/TypeScript
        ".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs",
        # Java
        ".java", ".class", ".jar",
        # C/C++
        ".c", ".cpp", ".cc", ".cxx", ".h", ".hpp", ".hxx",
        # Go
        ".go",
        # Rust
        ".rs",
        # Ruby
        ".rb", ".rake",
        # PHP
        ".php", ".phtml",
        # Swift
        ".swift",
        # Kotlin
        ".kt", ".kts",
        # Scala
        ".scala",
        # Shell scripts
        ".sh", ".bash", ".zsh", ".fish",
        # PowerShell
        ".ps1", ".psm1",
        # Configuration files (often code-like)
        ".yaml", ".yml", ".toml", ".ini", ".cfg", ".conf",
        # Other
        ".r", ".R", ".m", ".matlab", ".lua", ".pl", ".pm", ".sql",
    },
    FileTypeCategory.NOTEBOOK: {
        ".ipynb",
    },
}

# Reverse mapping: extension -> category for fast lookup
_EXTENSION_TO_CATEGORY: Dict[str, FileTypeCategory] = {}
for category, extensions in SUPPORTED_EXTENSIONS.items():
    for ext in extensions:
        _EXTENSION_TO_CATEGORY[ext.lower()] = category

# MIME type patterns for category detection
# Used when python-magic is available for more reliable detection
MIME_TYPE_PATTERNS: Dict[FileTypeCategory, Set[str]] = {
    FileTypeCategory.UNSTRUCTURED_NATIVE: {
        # Text
        "text/plain", "text/markdown", "text/csv", "text/html", "text/xml",
        "application/json", "application/xml",
        # PDF
        "application/pdf",
        # Microsoft Office
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.ms-excel",
        "application/vnd.ms-powerpoint",
        # OpenDocument
        "application/vnd.oasis.opendocument.text",
        "application/vnd.oasis.opendocument.spreadsheet",
        "application/vnd.oasis.opendocument.presentation",
        # Google Docs
        "application/vnd.google-apps.document",
        "application/vnd.google-apps.spreadsheet",
        "application/vnd.google-apps.presentation",
        # Other
        "application/rtf", "application/epub+zip",
        # Image files (for OCR processing)
        "image/jpeg", "image/jpg", "image/png", "image/gif", "image/bmp",
        "image/tiff", "image/webp", "image/svg+xml", "image/x-icon",
    },
    FileTypeCategory.AUDIO: {
        "audio/mpeg", "audio/mp3", "audio/wav", "audio/x-wav", "audio/m4a",
        "audio/x-m4a", "audio/flac", "audio/ogg", "audio/aac", "audio/x-ms-wma",
        "audio/opus", "audio/webm",
    },
    FileTypeCategory.CODE: {
        "text/x-python", "application/x-python-code",
        "text/javascript", "application/javascript", "text/x-javascript",
        "application/typescript", "text/x-typescript",
        "text/x-java-source", "text/x-java",
        "text/x-c", "text/x-c++", "text/x-c++src",
        "text/x-go", "text/x-rust",
        "text/x-ruby", "application/x-php",
        "text/x-shellscript", "application/x-sh",
        "text/x-yaml", "application/x-yaml",
        "text/x-sql",
    },
    FileTypeCategory.NOTEBOOK: {
        "application/x-ipynb+json",
    },
}


def get_file_category(
    file_path: str,
    mime_type: Optional[str] = None,
) -> FileTypeCategory:
    """Determine file category for routing to appropriate processor.
    
    Uses python-magic for reliable MIME type detection when available,
    with fallback to extension matching. This ensures accurate file type
    detection even when file extensions are missing or incorrect.
    
    Args:
        file_path: Path to the file (can be relative or absolute)
        mime_type: Optional pre-detected MIME type (skips detection if provided)
        
    Returns:
        FileTypeCategory enum value indicating the file's category
        
    Examples:
        >>> get_file_category("document.pdf")
        <FileTypeCategory.UNSTRUCTURED_NATIVE: 'unstructured_native'>
        
        >>> get_file_category("audio.mp3", mime_type="audio/mpeg")
        <FileTypeCategory.AUDIO: 'audio'>
        
        >>> get_file_category("script.py")
        <FileTypeCategory.CODE: 'code'>
        
        >>> get_file_category("notebook.ipynb")
        <FileTypeCategory.NOTEBOOK: 'notebook'>
        
        >>> get_file_category("unknown.xyz")
        <FileTypeCategory.UNSUPPORTED: 'unsupported'>
        
        >>> get_file_category("file_without_extension")
        <FileTypeCategory.UNSUPPORTED: 'unsupported'>
    
    Note:
        This function prioritizes MIME type detection over extension matching
        for accuracy, especially important for files without extensions or
        files with incorrect extensions. All processing is local-first with
        no cloud dependencies.
    """
    path_obj = Path(file_path)
    
    # Step 1: Use provided MIME type if available
    if mime_type:
        detected_mime = mime_type.lower().strip()
        category = _categorize_by_mime_type(detected_mime)
        if category != FileTypeCategory.UNSUPPORTED:
            logger.debug(
                "File categorized by provided MIME type",
                file_path=file_path,
                mime_type=detected_mime,
                category=category.value,
            )
            return category
    
    # Step 2: Try python-magic MIME type detection if available
    if MAGIC_AVAILABLE and path_obj.exists():
        try:
            detected_mime = magic.from_file(str(path_obj), mime=True).lower()
            category = _categorize_by_mime_type(detected_mime)
            if category != FileTypeCategory.UNSUPPORTED:
                logger.debug(
                    "File categorized by python-magic",
                    file_path=file_path,
                    mime_type=detected_mime,
                    category=category.value,
                )
                return category
        except Exception as e:
            logger.warning(
                "python-magic detection failed, falling back to extension",
                file_path=file_path,
                error=str(e),
            )
    
    # Step 3: Fallback to extension matching
    if path_obj.suffix:
        extension = path_obj.suffix.lower()
        category = _EXTENSION_TO_CATEGORY.get(extension)
        if category is not None:
            logger.debug(
                "File categorized by extension",
                file_path=file_path,
                extension=extension,
                category=category.value,
            )
            return category
    
    # Step 4: No match found
    logger.debug(
        "File type unsupported",
        file_path=file_path,
        extension=path_obj.suffix if path_obj.suffix else "none",
    )
    return FileTypeCategory.UNSUPPORTED


def _categorize_by_mime_type(mime_type: str) -> FileTypeCategory:
    """Categorize file by MIME type.
    
    Args:
        mime_type: MIME type string (should be lowercase)
        
    Returns:
        FileTypeCategory or UNSUPPORTED if no match
    """
    mime_lower = mime_type.lower().strip()
    
    # Check exact matches first
    for category, patterns in MIME_TYPE_PATTERNS.items():
        if mime_lower in patterns:
            return category
    
    # Check prefix matches for generic types
    if mime_lower.startswith("audio/"):
        return FileTypeCategory.AUDIO
    
    if mime_lower.startswith("text/"):
        # Text files could be code or unstructured_native
        # Check if it matches code patterns
        if any(pattern in mime_lower for pattern in [
            "x-python", "javascript", "typescript", "x-java",
            "x-c", "x-c++", "x-go", "x-rust", "x-ruby",
            "x-php", "x-shellscript", "x-sh", "x-sql",
        ]):
            return FileTypeCategory.CODE
        return FileTypeCategory.UNSTRUCTURED_NATIVE
    
    # Image files for OCR processing
    if mime_lower.startswith("image/"):
        return FileTypeCategory.UNSTRUCTURED_NATIVE
    
    return FileTypeCategory.UNSUPPORTED


def get_supported_extensions() -> Dict[str, FileTypeCategory]:
    """Get all supported file extensions mapped to categories.
    
    Returns:
        Dictionary mapping extension (with dot) to FileTypeCategory
        
    Example:
        >>> extensions = get_supported_extensions()
        >>> extensions[".pdf"]
        <FileTypeCategory.UNSTRUCTURED_NATIVE: 'unstructured_native'>
        >>> extensions[".py"]
        <FileTypeCategory.CODE: 'code'>
    """
    return _EXTENSION_TO_CATEGORY.copy()


def is_supported_file(file_path: str, mime_type: Optional[str] = None) -> bool:
    """Check if a file is supported for processing.
    
    Args:
        file_path: Path to the file
        mime_type: Optional pre-detected MIME type
        
    Returns:
        True if file can be processed, False otherwise
        
    Example:
        >>> is_supported_file("document.pdf")
        True
        >>> is_supported_file("unknown.xyz")
        False
    """
    category = get_file_category(file_path, mime_type)
    return category != FileTypeCategory.UNSUPPORTED

