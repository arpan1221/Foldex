"""Security utilities, session management, and user authentication helpers."""

import re
import hashlib
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import structlog

from app.core.exceptions import AuthenticationError

logger = structlog.get_logger(__name__)


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


def generate_session_id() -> str:
    """Generate a secure random session ID.

    Returns:
        Hex-encoded session ID string
    """
    return secrets.token_hex(32)


def hash_user_id(user_id: str) -> str:
    """Hash user ID for storage (one-way hash).

    Args:
        user_id: User identifier

    Returns:
        SHA-256 hash of user ID
    """
    return hashlib.sha256(user_id.encode()).hexdigest()


def validate_email(email: str) -> bool:
    """Validate email format.

    Args:
        email: Email address to validate

    Returns:
        True if email format is valid
    """
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email))


def create_user_session(
    user_id: str, google_access_token: str, expires_in: int = 3600
) -> Dict[str, Any]:
    """Create a user session with metadata.

    Args:
        user_id: User identifier
        google_access_token: Google OAuth2 access token
        expires_in: Session expiration in seconds

    Returns:
        Session dictionary with metadata
    """
    session_id = generate_session_id()
    expires_at = datetime.utcnow() + timedelta(seconds=expires_in)

    return {
        "session_id": session_id,
        "user_id": user_id,
        "google_access_token": google_access_token,
        "created_at": datetime.utcnow().isoformat(),
        "expires_at": expires_at.isoformat(),
    }


def is_session_expired(session: Dict[str, Any]) -> bool:
    """Check if session has expired.

    Args:
        session: Session dictionary

    Returns:
        True if session is expired
    """
    if "expires_at" not in session:
        return True

    try:
        expires_at = datetime.fromisoformat(session["expires_at"])
        return datetime.utcnow() >= expires_at
    except (ValueError, TypeError):
        return True


def validate_google_drive_access(google_access_token: str) -> bool:
    """Validate that Google access token has Drive access.

    This is a lightweight check - full validation should be done
    in the auth module.

    Args:
        google_access_token: Google OAuth2 access token

    Returns:
        True if token appears valid (basic check)
    """
    if not google_access_token or len(google_access_token) < 20:
        return False
    return True
