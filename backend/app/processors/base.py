"""Base processor interface for all document processors."""

from abc import ABC, abstractmethod
from typing import List, Optional, Callable
import structlog

from app.models.documents import DocumentChunk
from app.core.exceptions import DocumentProcessingError

logger = structlog.get_logger(__name__)


class BaseProcessor(ABC):
    """Base interface for all document processors.
    
    All processors must implement this interface to ensure consistent
    processing behavior across different file types.
    """

    def __init__(self):
        """Initialize base processor."""
        self.logger = structlog.get_logger(self.__class__.__name__)

    @abstractmethod
    async def can_process(self, file_path: str, mime_type: Optional[str] = None) -> bool:
        """Check if processor can handle the file type.

        Args:
            file_path: Path to the file
            mime_type: Optional MIME type for faster detection

        Returns:
            True if processor can handle the file
        """
        pass

    @abstractmethod
    async def process(
        self,
        file_path: str,
        file_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[DocumentChunk]:
        """Extract content and return structured chunks.

        Args:
            file_path: Path to the file to process
            file_id: Optional file identifier for chunk IDs
            metadata: Optional file metadata to include in chunks
            progress_callback: Optional callback for progress updates (0.0 to 1.0)

        Returns:
            List of document chunks with metadata

        Raises:
            DocumentProcessingError: If processing fails
        """
        pass

    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions.

        Returns:
            List of file extensions (e.g., ['.pdf', '.txt'])
        """
        pass

    @abstractmethod
    def get_supported_mime_types(self) -> List[str]:
        """Return list of supported MIME types.

        Returns:
            List of MIME types (e.g., ['application/pdf', 'text/plain'])
        """
        pass

    def _generate_chunk_id(self, file_id: str, chunk_index: int) -> str:
        """Generate a unique chunk ID.

        Args:
            file_id: File identifier
            chunk_index: Chunk index within the file

        Returns:
            Unique chunk ID
        """
        return f"{file_id}_chunk_{chunk_index}"

    def _update_progress(
        self, progress_callback: Optional[Callable[[float], None]], progress: float
    ) -> None:
        """Update processing progress.

        Args:
            progress_callback: Optional progress callback
            progress: Progress value (0.0 to 1.0)
        """
        if progress_callback:
            try:
                progress_callback(min(1.0, max(0.0, progress)))
            except Exception as e:
                self.logger.warning("Progress callback failed", error=str(e))
