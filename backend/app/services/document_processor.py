"""Document processing service with file type routing and progress tracking."""

from typing import List, Dict, Optional, Callable
import os
import structlog

from app.models.documents import DocumentChunk
from app.processors.base import BaseProcessor
from app.processors.pdf_processor import PDFProcessor
from app.processors.text_processor import TextProcessor
from app.processors.audio_processor import AudioProcessor
from app.processors.code_processor import CodeProcessor
from app.core.exceptions import DocumentProcessingError
from app.utils.file_utils import get_mime_type, validate_file_size, extract_file_metadata

logger = structlog.get_logger(__name__)


class DocumentProcessor:
    """Orchestrates document processing with appropriate processors.
    
    Routes files to appropriate processors based on file type,
    tracks processing progress, and handles errors gracefully.
    """

    def __init__(self):
        """Initialize document processor with all available processors."""
        self.processors: List[BaseProcessor] = [
            PDFProcessor(),
            TextProcessor(),
            AudioProcessor(),
            CodeProcessor(),
        ]
        self.logger = structlog.get_logger(__name__)

    async def process_file(
        self,
        file_path: str,
        file_metadata: Dict,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[DocumentChunk]:
        """Process a file and return document chunks.

        Args:
            file_path: Local file path
            file_metadata: File metadata from Google Drive
            progress_callback: Optional callback for progress updates (0.0 to 1.0)

        Returns:
            List of document chunks with metadata

        Raises:
            DocumentProcessingError: If processing fails
        """
        try:
            file_id = file_metadata.get("id", "")
            file_name = file_metadata.get("name", os.path.basename(file_path))
            mime_type = file_metadata.get("mimeType") or get_mime_type(file_path)

            self.logger.info(
                "Processing file",
                file_path=file_path,
                file_id=file_id,
                file_name=file_name,
                mime_type=mime_type,
            )

            # Validate file exists
            if not os.path.exists(file_path):
                raise DocumentProcessingError(file_path, "File not found")

            # Validate file size
            try:
                validate_file_size(file_path, max_size_mb=100)  # Use settings.MAX_FILE_SIZE_MB
            except ValueError as e:
                raise DocumentProcessingError(file_path, str(e))

            # Find appropriate processor
            processor = await self._get_processor(file_path, mime_type)
            if not processor:
                raise DocumentProcessingError(
                    file_path,
                    f"No suitable processor found for file type: {mime_type or 'unknown'}",
                )

            self.logger.debug(
                "Using processor",
                processor=processor.__class__.__name__,
                file_path=file_path,
            )

            # Prepare metadata for chunks
            chunk_metadata = {
                "file_name": file_name,
                "mime_type": mime_type,
                "file_size": file_metadata.get("size", 0),
                "created_time": file_metadata.get("createdTime"),
                "modified_time": file_metadata.get("modifiedTime"),
            }

            # Process file with progress tracking
            chunks = await processor.process(
                file_path,
                file_id=file_id,
                metadata=chunk_metadata,
                progress_callback=progress_callback,
            )

            # Validate chunks
            if not chunks:
                raise DocumentProcessingError(
                    file_path, "No chunks extracted from file"
                )

            # Ensure all chunks have file_id
            for chunk in chunks:
                if not chunk.file_id:
                    chunk.file_id = file_id

            self.logger.info(
                "File processing completed",
                file_path=file_path,
                file_id=file_id,
                chunk_count=len(chunks),
            )

            return chunks

        except DocumentProcessingError:
            raise
        except Exception as e:
            self.logger.error(
                "File processing failed",
                file_path=file_path,
                error=str(e),
                exc_info=True,
            )
            raise DocumentProcessingError(
                file_path, f"File processing failed: {str(e)}"
            ) from e

    async def _get_processor(
        self, file_path: str, mime_type: Optional[str] = None
    ) -> Optional[BaseProcessor]:
        """Get appropriate processor for file.

        Args:
            file_path: File path
            mime_type: Optional MIME type for faster detection

        Returns:
            Processor instance or None if no suitable processor found
        """
        for processor in self.processors:
            try:
                if await processor.can_process(file_path, mime_type):
                    return processor
            except Exception as e:
                self.logger.warning(
                    "Processor check failed",
                    processor=processor.__class__.__name__,
                    file_path=file_path,
                    error=str(e),
                )
                continue

        return None

    def get_supported_file_types(self) -> Dict[str, List[str]]:
        """Get all supported file types by extension and MIME type.

        Returns:
            Dictionary with 'extensions' and 'mime_types' keys
        """
        extensions = set()
        mime_types = set()

        for processor in self.processors:
            extensions.update(processor.get_supported_extensions())
            mime_types.update(processor.get_supported_mime_types())

        return {
            "extensions": sorted(list(extensions)),
            "mime_types": sorted(list(mime_types)),
        }

    async def is_supported_file_type(
        self, file_path: str, mime_type: Optional[str] = None
    ) -> bool:
        """Check if file type is supported.

        Args:
            file_path: File path
            mime_type: Optional MIME type

        Returns:
            True if file type is supported
        """
        for processor in self.processors:
            try:
                if await processor.can_process(file_path, mime_type):
                    return True
            except Exception:
                continue
        return False
