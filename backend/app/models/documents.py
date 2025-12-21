"""Document data models for API requests and responses."""

from pydantic import BaseModel, Field, HttpUrl
from typing import Dict, List, Optional, Union
from datetime import datetime


class DocumentChunk(BaseModel):
    """Represents a chunk of a document."""

    chunk_id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., min_length=1, max_length=50000, description="Chunk content")
    file_id: str = Field(..., description="Source file identifier")
    metadata: Dict[str, Union[str, int, float, None]] = Field(
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
    """Metadata for a file or folder."""

    file_id: str = Field(..., description="File or folder identifier")
    file_name: str = Field(..., description="File or folder name")
    mime_type: str = Field(..., description="MIME type")
    size: int = Field(default=0, description="File size in bytes (0 for folders)")
    folder_id: Optional[str] = Field(None, description="Parent folder ID")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    modified_at: Optional[datetime] = Field(None, description="Modification timestamp")
    web_view_link: Optional[HttpUrl] = Field(None, description="Google Drive web view link")
    web_content_link: Optional[HttpUrl] = Field(None, description="Google Drive download link")
    is_folder: Optional[bool] = Field(False, description="Whether this is a folder")


class FolderMetadata(BaseModel):
    """Metadata for a folder."""

    folder_id: str = Field(..., description="Folder identifier")
    folder_name: str = Field(..., description="Folder name")
    file_count: int = Field(default=0, description="Number of files in folder")
    folder_count: int = Field(default=0, description="Number of subfolders in folder")
    status: str = Field(default="processing", description="Processing status")
    created_at: Optional[str] = Field(None, description="Creation timestamp")
    updated_at: Optional[str] = Field(None, description="Last update timestamp")


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


class TreeNode(BaseModel):
    """Represents a node in the folder tree structure."""

    id: str = Field(..., description="File or folder ID")
    name: str = Field(..., description="File or folder name")
    is_folder: bool = Field(..., description="Whether this is a folder")
    mime_type: str = Field(..., description="MIME type")
    size: int = Field(default=0, description="File size in bytes (0 for folders)")
    created_at: Optional[datetime] = Field(None, description="Creation timestamp")
    modified_at: Optional[datetime] = Field(None, description="Modification timestamp")
    web_view_link: Optional[HttpUrl] = Field(None, description="Google Drive web view link")
    web_content_link: Optional[HttpUrl] = Field(None, description="Google Drive download link")
    children: List["TreeNode"] = Field(default_factory=list, description="Child nodes (files and subfolders)")

    class Config:
        json_schema_extra = {
            "example": {
                "id": "folder_123",
                "name": "My Folder",
                "is_folder": True,
                "mime_type": "application/vnd.google-apps.folder",
                "size": 0,
                "children": [
                    {
                        "id": "file_456",
                        "name": "document.pdf",
                        "is_folder": False,
                        "mime_type": "application/pdf",
                        "size": 1024,
                        "children": []
                    }
                ]
            }
        }


# Forward reference resolution
TreeNode.model_rebuild()


# LangChain Document compatibility
try:
    from langchain_core.documents import Document as LangChainDocument
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback for older LangChain versions
    try:
        from langchain.schema import Document as LangChainDocument
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        LangChainDocument = None


def document_chunk_to_langchain(chunk: DocumentChunk) -> "LangChainDocument":
    """Convert DocumentChunk to LangChain Document.

    Args:
        chunk: DocumentChunk instance

    Returns:
        LangChain Document object

    Raises:
        ImportError: If LangChain is not installed
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is not installed. Install with: pip install langchain")

    return LangChainDocument(
        page_content=chunk.content,
        metadata={
            "chunk_id": chunk.chunk_id,
            "file_id": chunk.file_id,
            **chunk.metadata,
        },
    )


def langchain_to_document_chunk(
    doc: "LangChainDocument",
    file_id: str,
    chunk_id: Optional[str] = None,
) -> DocumentChunk:
    """Convert LangChain Document to DocumentChunk.

    Args:
        doc: LangChain Document instance
        file_id: File identifier
        chunk_id: Optional chunk identifier (extracted from metadata if not provided)

    Returns:
        DocumentChunk object

    Raises:
        ImportError: If LangChain is not installed
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain is not installed. Install with: pip install langchain")

    chunk_id = chunk_id or doc.metadata.get("chunk_id", f"{file_id}_chunk_0")

    # Extract chunk-specific metadata
    chunk_metadata = {
        k: v
        for k, v in doc.metadata.items()
        if k not in ["chunk_id", "file_id"]
    }

    return DocumentChunk(
        chunk_id=chunk_id,
        content=doc.page_content,
        file_id=file_id,
        metadata=chunk_metadata,
        embedding=None,  # Embedding will be generated separately
    )

