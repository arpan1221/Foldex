"""Text and Markdown file processor with smart chunking."""

from typing import List, Optional, Callable
import os
import uuid
import structlog

from app.processors.base import BaseProcessor
from app.models.documents import DocumentChunk
from app.core.exceptions import DocumentProcessingError
from app.config.settings import settings
from app.utils.text_utils import chunk_text, clean_text, extract_sentences

logger = structlog.get_logger(__name__)


class TextProcessor(BaseProcessor):
    """Processor for plain text and Markdown files.
    
    Handles various text formats with proper encoding detection
    and smart chunking that preserves document structure.
    """

    def __init__(self):
        """Initialize text processor."""
        super().__init__()
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.supported_extensions = [".txt", ".md", ".markdown", ".rst", ".csv"]
        self.supported_mime_types = [
            "text/plain",
            "text/markdown",
            "text/csv",
            "text/x-markdown",
        ]

    async def can_process(self, file_path: str, mime_type: Optional[str] = None) -> bool:
        """Check if file is a text file.

        Args:
            file_path: Path to file
            mime_type: Optional MIME type

        Returns:
            True if file is text
        """
        if mime_type:
            return mime_type in self.supported_mime_types or mime_type.startswith("text/")

        file_lower = file_path.lower()
        return any(file_lower.endswith(ext) for ext in self.supported_extensions)

    async def process(
        self,
        file_path: str,
        file_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[DocumentChunk]:
        """Process text file and create chunks.

        Args:
            file_path: Path to text file
            file_id: Optional file identifier
            metadata: Optional file metadata
            progress_callback: Optional progress callback

        Returns:
            List of document chunks

        Raises:
            DocumentProcessingError: If processing fails
        """
        if not os.path.exists(file_path):
            raise DocumentProcessingError(file_path, "File not found")

        try:
            self.logger.info("Processing text file", file_path=file_path, file_id=file_id)

            # Generate file_id if not provided
            if not file_id:
                file_id = str(uuid.uuid4())

            self._update_progress(progress_callback, 0.1)

            # Detect encoding and read file
            content = self._read_file_with_encoding(file_path)
            
            if not content or not content.strip():
                raise DocumentProcessingError(file_path, "File is empty or contains no text")

            self._update_progress(progress_callback, 0.3)

            # Clean text
            content = clean_text(content)

            # Determine if it's Markdown
            is_markdown = file_path.lower().endswith((".md", ".markdown"))

            # Chunk text with smart boundaries
            if is_markdown:
                text_chunks = self._chunk_markdown(content)
            else:
                text_chunks = self._chunk_plain_text(content)

            self._update_progress(progress_callback, 0.7)

            # Create DocumentChunk objects
            chunks: List[DocumentChunk] = []
            for i, chunk_text in enumerate(text_chunks):
                if not chunk_text.strip():
                    continue

                chunk_id = self._generate_chunk_id(file_id, i)
                chunk_metadata = {
                    "chunk_index": i,
                    "total_chunks": len(text_chunks),
                    "file_name": metadata.get("file_name", "") if metadata else "",
                    "mime_type": metadata.get("mime_type", "text/plain") if metadata else "text/plain",
                    "is_markdown": is_markdown,
                }

                # Add any additional metadata
                if metadata:
                    chunk_metadata.update({
                        k: v for k, v in metadata.items()
                        if k not in ["file_name", "mime_type"]
                    })

                chunks.append(
                    DocumentChunk(
                        chunk_id=chunk_id,
                        content=chunk_text,
                        file_id=file_id,
                        metadata=chunk_metadata,
                    )
                )

            self._update_progress(progress_callback, 1.0)

            self.logger.info(
                "Text processing completed",
                file_path=file_path,
                chunks=len(chunks),
            )

            return chunks

        except DocumentProcessingError:
            raise
        except Exception as e:
            self.logger.error(
                "Text processing failed",
                file_path=file_path,
                error=str(e),
                exc_info=True,
            )
            raise DocumentProcessingError(file_path, f"Text processing failed: {str(e)}") from e

    def _read_file_with_encoding(self, file_path: str) -> str:
        """Read file with encoding detection.

        Args:
            file_path: Path to file

        Returns:
            File content as string

        Raises:
            DocumentProcessingError: If file cannot be read
        """
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                raise DocumentProcessingError(
                    file_path, f"Failed to read file: {str(e)}"
                )

        raise DocumentProcessingError(
            file_path, "Could not decode file with any supported encoding"
        )

    def _chunk_plain_text(self, text: str) -> List[str]:
        """Chunk plain text with sentence awareness.

        Args:
            text: Text content

        Returns:
            List of text chunks
        """
        # Split by paragraphs first
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        chunks = []
        current_chunk = ""

        for paragraph in paragraphs:
            # If adding this paragraph would exceed chunk size, save current chunk
            if current_chunk and len(current_chunk) + len(paragraph) + 2 > self.chunk_size:
                chunks.append(current_chunk.strip())
                # Start new chunk with overlap
                overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                current_chunk = overlap_text + "\n\n" + paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph

        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Fallback to simple chunking if no chunks created
        if not chunks:
            chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)

        return chunks

    def _chunk_markdown(self, text: str) -> List[str]:
        """Chunk Markdown text preserving structure.

        Args:
            text: Markdown content

        Returns:
            List of text chunks
        """
        # Split by markdown headers (##, ###, etc.)
        import re
        sections = re.split(r"\n(#{1,6}\s+.+)\n", text)

        chunks = []
        current_chunk = ""

        for i, section in enumerate(sections):
            section = section.strip()
            if not section:
                continue

            # If section is a header, add it to current chunk
            if section.startswith("#"):
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    # Start new chunk with header
                    current_chunk = section + "\n\n"
                else:
                    current_chunk = section + "\n\n"
            else:
                # Regular content
                if current_chunk and len(current_chunk) + len(section) + 2 > self.chunk_size:
                    chunks.append(current_chunk.strip())
                    # Start new chunk with overlap
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else ""
                    current_chunk = overlap_text + "\n\n" + section
                else:
                    current_chunk += section + "\n\n"

        # Add remaining chunk
        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        # Fallback to plain text chunking if no chunks created
        if not chunks:
            chunks = self._chunk_plain_text(text)

        return chunks

    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions.

        Returns:
            List of text extensions
        """
        return self.supported_extensions

    def get_supported_mime_types(self) -> List[str]:
        """Get supported MIME types.

        Returns:
            List of text MIME types
        """
        return self.supported_mime_types
