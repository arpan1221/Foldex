"""PDF document processor using PyMuPDF (fitz) for text extraction."""

from typing import List, Optional, Callable
import os
import uuid
import structlog

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

from app.processors.base import BaseProcessor
from app.models.documents import DocumentChunk
from app.core.exceptions import DocumentProcessingError
from app.config.settings import settings
from app.utils.text_utils import chunk_text, clean_text

logger = structlog.get_logger(__name__)


class PDFProcessor(BaseProcessor):
    """Processor for PDF documents using PyMuPDF.
    
    Extracts text with page numbers, preserves document structure,
    and creates chunks with page/section metadata for citations.
    """

    def __init__(self):
        """Initialize PDF processor."""
        super().__init__()
        if fitz is None:
            logger.warning("PyMuPDF not available, PDF processing will fail")
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP

    async def can_process(self, file_path: str, mime_type: Optional[str] = None) -> bool:
        """Check if file is a PDF.

        Args:
            file_path: Path to file
            mime_type: Optional MIME type

        Returns:
            True if file is PDF
        """
        if mime_type:
            return mime_type == "application/pdf"
        return file_path.lower().endswith(".pdf")

    async def process(
        self,
        file_path: str,
        file_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[DocumentChunk]:
        """Process PDF and extract chunks with page numbers.

        Args:
            file_path: Path to PDF file
            file_id: Optional file identifier
            metadata: Optional file metadata
            progress_callback: Optional progress callback

        Returns:
            List of document chunks with page metadata

        Raises:
            DocumentProcessingError: If processing fails
        """
        if fitz is None:
            raise DocumentProcessingError(
                file_path, "PyMuPDF library not installed"
            )

        if not os.path.exists(file_path):
            raise DocumentProcessingError(file_path, "File not found")

        try:
            self.logger.info("Processing PDF", file_path=file_path, file_id=file_id)
            
            # Generate file_id if not provided
            if not file_id:
                file_id = str(uuid.uuid4())

            # Open PDF document
            doc = fitz.open(file_path)
            total_pages = len(doc)
            
            if total_pages == 0:
                raise DocumentProcessingError(file_path, "PDF has no pages")

            self._update_progress(progress_callback, 0.1)

            chunks: List[DocumentChunk] = []
            chunk_index = 0

            # Extract text from each page
            for page_num in range(total_pages):
                try:
                    page = doc[page_num]
                    
                    # Extract text from page
                    text = page.get_text()
                    
                    if not text or not text.strip():
                        # Skip empty pages
                        continue

                    # Clean text
                    text = clean_text(text)

                    # Chunk text by sentences/paragraphs, respecting page boundaries
                    page_chunks = self._chunk_page_text(text, page_num + 1)

                    # Create chunks with page metadata
                    for chunk_text in page_chunks:
                        if not chunk_text.strip():
                            continue

                        chunk_id = self._generate_chunk_id(file_id, chunk_index)
                        chunk_metadata = {
                            "page_number": page_num + 1,
                            "total_pages": total_pages,
                            "chunk_index": chunk_index,
                            "file_name": metadata.get("file_name", "") if metadata else "",
                            "mime_type": "application/pdf",
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
                        chunk_index += 1

                    # Update progress
                    progress = 0.1 + (page_num + 1) / total_pages * 0.9
                    self._update_progress(progress_callback, progress)

                except Exception as e:
                    self.logger.warning(
                        "Failed to process PDF page",
                        page_num=page_num + 1,
                        error=str(e),
                    )
                    # Continue with next page
                    continue

            doc.close()

            if not chunks:
                raise DocumentProcessingError(
                    file_path, "No text content extracted from PDF"
                )

            self.logger.info(
                "PDF processing completed",
                file_path=file_path,
                pages=total_pages,
                chunks=len(chunks),
            )

            return chunks

        except DocumentProcessingError:
            raise
        except Exception as e:
            self.logger.error(
                "PDF processing failed",
                file_path=file_path,
                error=str(e),
                exc_info=True,
            )
            raise DocumentProcessingError(file_path, f"PDF processing failed: {str(e)}") from e

    def _chunk_page_text(self, text: str, page_number: int) -> List[str]:
        """Chunk text from a single page.

        Args:
            text: Text content from page
            page_number: Page number for metadata

        Returns:
            List of text chunks
        """
        # Use smart chunking that respects sentence boundaries
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

        # If no chunks created (very short text), create one chunk
        if not chunks and text.strip():
            chunks = chunk_text(text, self.chunk_size, self.chunk_overlap)

        return chunks

    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions.

        Returns:
            List of PDF extensions
        """
        return [".pdf"]

    def get_supported_mime_types(self) -> List[str]:
        """Get supported MIME types.

        Returns:
            List of PDF MIME types
        """
        return ["application/pdf"]
