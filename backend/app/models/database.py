"""Database schema models."""

from typing import Optional
from datetime import datetime


class DatabaseModel:
    """Base database model."""

    def __init__(self, **kwargs):
        """Initialize database model."""
        for key, value in kwargs.items():
            setattr(self, key, value)


class ChunkRecord(DatabaseModel):
    """Database record for document chunk."""

    chunk_id: str
    file_id: str
    content: str
    metadata: dict
    embedding: Optional[bytes] = None
    created_at: datetime


class FileRecord(DatabaseModel):
    """Database record for file."""

    file_id: str
    folder_id: str
    file_name: str
    mime_type: str
    size: int
    created_at: datetime
    modified_at: datetime


class ConversationRecord(DatabaseModel):
    """Database record for conversation."""

    conversation_id: str
    user_id: str
    folder_id: Optional[str]
    created_at: datetime
    updated_at: datetime

