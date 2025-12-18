"""Base processor interface."""

from abc import ABC, abstractmethod
from typing import List

from app.models.documents import DocumentChunk


class BaseProcessor(ABC):
    """Base interface for all document processors."""

    @abstractmethod
    async def can_process(self, file_path: str) -> bool:
        """Check if processor can handle the file type.

        Args:
            file_path: Path to the file

        Returns:
            True if processor can handle the file
        """
        pass

    @abstractmethod
    async def process(self, file_path: str) -> List[DocumentChunk]:
        """Extract content and return structured chunks.

        Args:
            file_path: Path to the file to process

        Returns:
            List of document chunks

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

