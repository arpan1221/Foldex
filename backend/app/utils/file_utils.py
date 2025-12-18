"""File handling utilities for validation and metadata extraction."""

import os
import hashlib
import mimetypes
from typing import Optional, Dict, Any
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)


def ensure_directory(path: str) -> None:
    """Ensure directory exists, create if not.

    Args:
        path: Directory path
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def get_file_hash(file_path: str) -> str:
    """Calculate SHA256 hash of file.

    Args:
        file_path: Path to file

    Returns:
        Hex digest of file hash

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()


def get_file_size(file_path: str) -> int:
    """Get file size in bytes.

    Args:
        file_path: Path to file

    Returns:
        File size in bytes

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return os.path.getsize(file_path)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    # Remove path components
    filename = os.path.basename(filename)
    # Remove dangerous characters
    dangerous_chars = ['/', '\\', '..', '<', '>', ':', '"', '|', '?', '*']
    for char in dangerous_chars:
        filename = filename.replace(char, '_')
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
    return filename


def get_mime_type(file_path: str) -> Optional[str]:
    """Get MIME type for a file.

    Args:
        file_path: Path to file

    Returns:
        MIME type or None if unknown
    """
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type


def extract_file_metadata(file_path: str, file_id: str, folder_id: str) -> Dict[str, Any]:
    """Extract metadata from a file.

    Args:
        file_path: Path to file
        file_id: File identifier
        folder_id: Parent folder ID

    Returns:
        File metadata dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        stat = os.stat(file_path)
        mime_type = get_mime_type(file_path)

        return {
            "file_id": file_id,
            "file_name": os.path.basename(file_path),
            "mime_type": mime_type or "application/octet-stream",
            "size": stat.st_size,
            "folder_id": folder_id,
            "created_at": stat.st_ctime,
            "modified_at": stat.st_mtime,
            "file_hash": get_file_hash(file_path),
        }
    except Exception as e:
        logger.error("Failed to extract file metadata", file_path=file_path, error=str(e))
        raise


def validate_file_size(file_path: str, max_size_mb: int = 100) -> bool:
    """Validate file size is within limits.

    Args:
        file_path: Path to file
        max_size_mb: Maximum file size in MB

    Returns:
        True if file size is valid

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file exceeds size limit
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    file_size = get_file_size(file_path)
    max_size_bytes = max_size_mb * 1024 * 1024

    if file_size > max_size_bytes:
        raise ValueError(f"File size {file_size} exceeds maximum {max_size_bytes} bytes")

    return True


def get_file_extension(file_path: str) -> str:
    """Get file extension from path.

    Args:
        file_path: Path to file

    Returns:
        File extension (with dot, e.g., ".pdf")
    """
    _, ext = os.path.splitext(file_path)
    return ext.lower()


def is_text_file(mime_type: Optional[str]) -> bool:
    """Check if file is a text file based on MIME type.

    Args:
        mime_type: File MIME type

    Returns:
        True if file is a text file
    """
    if not mime_type:
        return False

    text_types = [
        "text/",
        "application/json",
        "application/xml",
        "application/javascript",
        "application/x-sh",
        "application/x-python",
    ]
    return any(mime_type.startswith(prefix) for prefix in text_types)


def is_audio_file(mime_type: Optional[str]) -> bool:
    """Check if file is an audio file based on MIME type.

    Args:
        mime_type: File MIME type

    Returns:
        True if file is an audio file
    """
    return mime_type is not None and mime_type.startswith("audio/")


def is_video_file(mime_type: Optional[str]) -> bool:
    """Check if file is a video file based on MIME type.

    Args:
        mime_type: File MIME type

    Returns:
        True if file is a video file
    """
    return mime_type is not None and mime_type.startswith("video/")


def is_document_file(mime_type: Optional[str]) -> bool:
    """Check if file is a document file based on MIME type.

    Args:
        mime_type: File MIME type

    Returns:
        True if file is a document file
    """
    if not mime_type:
        return False

    document_types = [
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument",
        "application/vnd.google-apps",
    ]
    return any(mime_type.startswith(prefix) for prefix in document_types)
