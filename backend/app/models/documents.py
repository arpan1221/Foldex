"""Document data models for API requests and responses."""

from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, List, Optional, Union
from datetime import datetime


class DocumentChunk(BaseModel):
    """Represents a chunk of a document."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., min_length=1, max_length=2000, description="Chunk content")
    file_id: str = Field(..., description="Source file identifier")
    metadata: Dict[str, Union[str, int, float]] = Field(
        default_factory=dict, description="Chunk metadata"
    )
    embedding: Optional[List[float]] = Field(
        None, description="Vector embedding for the chunk"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "chunk_id": "chunk_123",
                "content": "This is a sample chunk of text.",
                "file_id": "file_456",
                "metadata": {"page_number": 1, "section": "Introduction"},
            }
        }


class FileMetadata(BaseModel):
    """Metadata for a file."""

    file_id: str = Field(..., description="File identifier")
    file_name: str = Field(..., description="File name")
    mime_type: str = Field(..., description="MIME type")
    size: int = Field(..., description="File size in bytes")
    folder_id: str = Field(..., description="Parent folder ID")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    modified_at: Optional[datetime] = Field(None, description="Modification timestamp")
    web_view_link: Optional[HttpUrl] = Field(None, description="Google Drive web view link")
    web_content_link: Optional[HttpUrl] = Field(None, description="Google Drive download link")


class FolderMetadata(BaseModel):
    """Metadata for a folder."""

    folder_id: str = Field(..., description="Folder identifier")
    folder_name: str = Field(..., description="Folder name")
    file_count: int = Field(default=0, description="Number of files in folder")
    total_size: int = Field(default=0, description="Total size of files in bytes")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    modified_at: Optional[datetime] = Field(None, description="Modification timestamp")
    web_view_link: Optional[HttpUrl] = Field(None, description="Google Drive web view link")


class ProcessFolderRequest(BaseModel):
    """Request model for folder processing."""

    folder_id: str = Field(..., description="Google Drive folder ID")
    folder_url: Optional[str] = Field(None, description="Google Drive folder URL (alternative to folder_id)")


class ProcessFolderResponse(BaseModel):
    """Response model for folder processing."""

    folder_id: str = Field(..., description="Folder identifier")
    status: str = Field(..., description="Processing status")
    message: str = Field(..., description="Status message")
    files_detected: Optional[int] = Field(None, description="Number of files detected")


class FolderStatusResponse(BaseModel):
    """Response model for folder status."""

    folder_id: str = Field(..., description="Folder identifier")
    status: str = Field(..., description="Processing status")
    files_processed: int = Field(default=0, description="Number of files processed")
    total_files: int = Field(default=0, description="Total number of files")
    progress: float = Field(default=0.0, ge=0.0, le=1.0, description="Processing progress (0.0 to 1.0)")
    error: Optional[str] = Field(None, description="Error message if processing failed")

