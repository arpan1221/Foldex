"""Pydantic models for database entities and SQLAlchemy ORM models."""

from typing import Optional, Dict, Any
from datetime import datetime
from sqlalchemy import (
    Column,
    String,
    Integer,
    Text,
    DateTime,
    ForeignKey,
    BLOB,
    JSON,
    Boolean,
)
from sqlalchemy.orm import relationship
from pydantic import BaseModel, Field

from app.database.base import Base


# ============================================================================
# SQLAlchemy ORM Models
# ============================================================================

class FileRecord(Base):
    """SQLAlchemy model for file metadata."""

    __tablename__ = "files"

    file_id = Column(String, primary_key=True, index=True)
    folder_id = Column(String, nullable=False, index=True)
    file_name = Column(String, nullable=False)
    mime_type = Column(String, nullable=True)
    size = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)
    modified_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    chunks = relationship("ChunkRecord", back_populates="file", cascade="all, delete-orphan")


class ChunkRecord(Base):
    """SQLAlchemy model for document chunks."""

    __tablename__ = "chunks"

    chunk_id = Column(String, primary_key=True, index=True)
    file_id = Column(String, ForeignKey("files.file_id"), nullable=False, index=True)
    content = Column(Text, nullable=False)
    chunk_metadata = Column(JSON, default=dict, name="metadata")  # Use name="metadata" for DB column
    embedding = Column(BLOB, nullable=True)  # Embeddings stored in vector DB
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    file = relationship("FileRecord", back_populates="chunks")


class ConversationRecord(Base):
    """SQLAlchemy model for conversations."""

    __tablename__ = "conversations"

    conversation_id = Column(String, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    folder_id = Column(String, nullable=True, index=True)
    title = Column(String, nullable=True)  # Added title for UI
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    messages = relationship(
        "MessageRecord", back_populates="conversation", cascade="all, delete-orphan"
    )


class MessageRecord(Base):
    """SQLAlchemy model for chat messages."""

    __tablename__ = "messages"

    message_id = Column(String, primary_key=True, index=True)
    conversation_id = Column(
        String, ForeignKey("conversations.conversation_id"), nullable=False, index=True
    )
    role = Column(String, nullable=False)  # 'user' or 'assistant'
    content = Column(Text, nullable=False)
    citations = Column(JSON, nullable=True)  # Store citations as JSON
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    # Relationships
    conversation = relationship("ConversationRecord", back_populates="messages")


class UserRecord(Base):
    """SQLAlchemy model for users."""

    __tablename__ = "users"

    user_id = Column(String, primary_key=True, index=True)
    email = Column(String, nullable=False, unique=True, index=True)
    name = Column(String, nullable=True)
    google_id = Column(String, nullable=True, unique=True, index=True)
    picture = Column(String, nullable=True)
    verified_email = Column(Boolean, default=False)
    google_access_token = Column(String, nullable=True)  # Store Google OAuth token
    google_refresh_token = Column(String, nullable=True)  # Store Google refresh token
    google_token_expiry = Column(DateTime, nullable=True)  # Token expiration time
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class FolderRecord(Base):
    """SQLAlchemy model for folders."""

    __tablename__ = "folders"

    folder_id = Column(String, primary_key=True, index=True)
    user_id = Column(String, nullable=False, index=True)
    folder_name = Column(String, nullable=False)
    folder_url = Column(String, nullable=True)
    parent_folder_id = Column(String, nullable=True, index=True)  # Parent folder ID for hierarchy
    root_folder_id = Column(String, nullable=True, index=True)  # Root folder ID (top-level folder)
    file_count = Column(Integer, default=0)
    status = Column(String, nullable=False, default="processing")  # processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Folder knowledge base / summarization fields
    summary = Column(Text, nullable=True)  # Master folder summary
    learning_status = Column(String, nullable=True)  # learning_pending, learning_in_progress, learning_complete, learning_failed
    insights = Column(JSON, nullable=True)  # Structured insights (themes, entities, capabilities)
    file_type_distribution = Column(JSON, nullable=True)  # File type counts {"pdf": 5, "audio": 2}
    entity_summary = Column(JSON, nullable=True)  # Top entities across all files
    relationship_summary = Column(JSON, nullable=True)  # Key cross-file relationships
    capabilities = Column(JSON, nullable=True)  # What questions can be answered from this folder
    graph_statistics = Column(JSON, nullable=True)  # Knowledge graph metrics (node count, edge count, etc.)
    learning_completed_at = Column(DateTime, nullable=True)  # When summarization finished


class KnowledgeGraphRecord(Base):
    """SQLAlchemy model for knowledge graphs."""

    __tablename__ = "knowledge_graphs"

    folder_id = Column(String, primary_key=True, index=True)
    graph_data = Column(BLOB, nullable=False)  # Pickled NetworkX graph
    created_at = Column(DateTime, default=datetime.utcnow)


class ProcessingCacheRecord(Base):
    """SQLAlchemy model for processing cache."""

    __tablename__ = "processing_cache"

    file_id = Column(String, primary_key=True, index=True)
    file_hash = Column(String, nullable=False, index=True)  # MD5 hash for change detection
    folder_id = Column(String, nullable=True, index=True)
    file_name = Column(String, nullable=False)
    file_type = Column(String, nullable=False)  # FileTypeCategory value
    chunk_count = Column(Integer, default=0)
    processing_time_seconds = Column(Integer, default=0)  # Stored as integer (milliseconds)
    cached_at = Column(DateTime, default=datetime.utcnow, index=True)
    expires_at = Column(DateTime, nullable=True, index=True)  # Optional expiration


# ============================================================================
# Pydantic Models for API/Service Layer
# ============================================================================

class FileMetadataModel(BaseModel):
    """Pydantic model for file metadata."""

    file_id: str = Field(..., description="Unique file identifier")
    folder_id: str = Field(..., description="Parent folder identifier")
    file_name: str = Field(..., description="File name")
    mime_type: Optional[str] = Field(None, description="MIME type")
    size: int = Field(default=0, description="File size in bytes")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    modified_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True


class ChunkMetadataModel(BaseModel):
    """Pydantic model for chunk metadata."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    file_id: str = Field(..., description="Source file identifier")
    content: str = Field(..., description="Chunk content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Chunk metadata")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True


class ConversationModel(BaseModel):
    """Pydantic model for conversation."""

    conversation_id: str = Field(..., description="Unique conversation identifier")
    user_id: str = Field(..., description="User identifier")
    folder_id: Optional[str] = Field(None, description="Associated folder ID")
    title: Optional[str] = Field(None, description="Conversation title")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True


class MessageModel(BaseModel):
    """Pydantic model for chat message."""

    message_id: str = Field(..., description="Unique message identifier")
    conversation_id: str = Field(..., description="Conversation identifier")
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")
    citations: Optional[Dict[str, Any]] = Field(None, description="Source citations")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        from_attributes = True


class FolderSummaryModel(BaseModel):
    """Pydantic model for folder summary/knowledge base."""

    folder_id: str = Field(..., description="Folder identifier")
    folder_name: Optional[str] = Field(None, description="Folder name")
    summary: Optional[str] = Field(None, description="Master folder summary")
    learning_status: Optional[str] = Field(None, description="Learning status")
    insights: Optional[Dict[str, Any]] = Field(None, description="Structured insights")
    file_type_distribution: Optional[Dict[str, int]] = Field(None, description="File type counts")
    entity_summary: Optional[Dict[str, Any]] = Field(None, description="Top entities")
    relationship_summary: Optional[Dict[str, Any]] = Field(None, description="Key relationships")
    capabilities: Optional[list] = Field(None, description="Folder capabilities")
    graph_statistics: Optional[Dict[str, Any]] = Field(None, description="Graph metrics")
    learning_completed_at: Optional[datetime] = Field(None, description="Completion timestamp")

    class Config:
        from_attributes = True

