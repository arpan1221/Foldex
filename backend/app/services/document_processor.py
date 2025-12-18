"""Document processing service."""

from typing import List, Dict
import structlog

from app.models.documents import DocumentChunk
from app.processors.base import BaseProcessor
from app.processors.pdf_processor import PDFProcessor
from app.processors.audio_processor import AudioProcessor
from app.processors.text_processor import TextProcessor
from app.processors.code_processor import CodeProcessor
from app.core.exceptions import DocumentProcessingError

logger = structlog.get_logger(__name__)


class DocumentProcessor:
    """Orchestrates document processing with appropriate processors."""

    def __init__(self):
        """Initialize document processor with all available processors."""
        self.processors: List[BaseProcessor] = [
            PDFProcessor(),
            AudioProcessor(),
            TextProcessor(),
            CodeProcessor(),
        ]

    async def process_file(
        self, file_path: str, file_metadata: Dict
    ) -> List[DocumentChunk]:
        """Process a file and return document chunks.

        Args:
            file_path: Local file path
            file_metadata: File metadata from Google Drive

        Returns:
            List of document chunks

        Raises:
            DocumentProcessingError: If processing fails
        """
        try:
            logger.info("Processing file", file_path=file_path)

            # Find appropriate processor
            processor = await self._get_processor(file_path)
            if not processor:
                raise DocumentProcessingError(
                    file_path, "No suitable processor found"
                )

            # Process file
            chunks = await processor.process(file_path)

            # Add file metadata to chunks
            for chunk in chunks:
                chunk.file_id = file_metadata.get("id", "")
                chunk.metadata.update(
                    {
                        "file_name": file_metadata.get("name", ""),
                        "mime_type": file_metadata.get("mimeType", ""),
                    }
                )

            logger.info(
                "File processing completed",
                file_path=file_path,
                chunk_count=len(chunks),
            )
            return chunks
        except Exception as e:
            logger.error("File processing failed", file_path=file_path, error=str(e))
            raise DocumentProcessingError(file_path, str(e))

    async def _get_processor(self, file_path: str) -> BaseProcessor | None:
        """Get appropriate processor for file.

        Args:
            file_path: File path

        Returns:
            Processor instance or None
        """
        for processor in self.processors:
            if await processor.can_process(file_path):
                return processor
        return None

