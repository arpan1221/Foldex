"""Text and Markdown file processor."""

from typing import List
import structlog

from app.processors.base import BaseProcessor
from app.models.documents import DocumentChunk
from app.core.exceptions import DocumentProcessingError
from app.utils.text_utils import chunk_text

logger = structlog.get_logger(__name__)


class TextProcessor(BaseProcessor):
    """Processor for text and Markdown files."""

    def __init__(self):
        """Initialize text processor."""
        pass

    async def can_process(self, file_path: str) -> bool:
        """Check if file is a text file.

        Args:
            file_path: Path to file

        Returns:
            True if file is text
        """
        text_extensions = [".txt", ".md", ".markdown", ".rst"]
        return any(file_path.lower().endswith(ext) for ext in text_extensions)

    async def process(self, file_path: str) -> List[DocumentChunk]:
        """Process text file and create chunks.

        Args:
            file_path: Path to text file

        Returns:
            List of document chunks

        Raises:
            DocumentProcessingError: If processing fails
        """
        try:
            logger.info("Processing text file", file_path=file_path)
            # Read file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Chunk text
            text_chunks = chunk_text(content)

            # Create DocumentChunk objects
            chunks: List[DocumentChunk] = []
            for i, chunk_text in enumerate(text_chunks):
                chunks.append(
                    DocumentChunk(
                        chunk_id=f"{file_path}_chunk_{i}",
                        content=chunk_text,
                        file_id="",  # Will be set by document processor
                        metadata={"chunk_index": i},
                    )
                )

            return chunks
        except Exception as e:
            logger.error("Text processing failed", file_path=file_path, error=str(e))
            raise DocumentProcessingError(file_path, str(e))

    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions.

        Returns:
            List of text extensions
        """
        return [".txt", ".md", ".markdown", ".rst"]

