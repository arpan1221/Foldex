"""Security utilities and helpers."""

import re
from typing import Optional


def sanitize_folder_id(folder_id: str) -> str:
    """Sanitize and validate Google Drive folder ID.

    Args:
        folder_id: Raw folder ID string

    Returns:
        Sanitized folder ID

    Raises:
        ValueError: If folder ID format is invalid
    """
    # Remove any URL components
    folder_id = folder_id.strip()
    if "/" in folder_id:
        folder_id = folder_id.split("/")[-1]
    if "?" in folder_id:
        folder_id = folder_id.split("?")[0]

    # Validate format (alphanumeric, hyphens, underscores)
    if not re.match(r"^[a-zA-Z0-9_-]+$", folder_id):
        raise ValueError("Invalid folder ID format")

    # Limit length
    if len(folder_id) > 64:
        raise ValueError("Folder ID too long")

    return folder_id


def sanitize_file_path(file_path: str) -> str:
    """Sanitize file path to prevent directory traversal.

    Args:
        file_path: Raw file path

    Returns:
        Sanitized file path

    Raises:
        ValueError: If path contains dangerous patterns
    """
    # Remove any path traversal attempts
    if ".." in file_path or file_path.startswith("/"):
        raise ValueError("Invalid file path: path traversal detected")

    # Normalize path
    normalized = file_path.replace("\\", "/")
    return normalized


def mask_sensitive_data(data: str, visible_chars: int = 4) -> str:
    """Mask sensitive data for logging.

    Args:
        data: Sensitive string to mask
        visible_chars: Number of characters to show at start

    Returns:
        Masked string (e.g., "abcd****")
    """
    if len(data) <= visible_chars:
        return "*" * len(data)
    return data[:visible_chars] + "*" * (len(data) - visible_chars)

