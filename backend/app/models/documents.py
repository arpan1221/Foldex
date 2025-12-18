"""Document data models."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union


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
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    modified_at: Optional[str] = Field(None, description="Modification timestamp")

