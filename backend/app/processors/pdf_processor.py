"""PDF document processor using intelligent hierarchical chunking."""

from typing import List, Optional, Callable
import os
import uuid
import structlog

from app.processors.base import BaseProcessor
from app.models.documents import DocumentChunk
from app.core.exceptions import DocumentProcessingError
from app.config.settings import settings
from app.ingestion.chunking import get_foldex_chunker
from app.ingestion.metadata_schema import MetadataBuilder, FileType, ChunkType

logger = structlog.get_logger(__name__)


class PDFProcessor(BaseProcessor):
    """Processor for PDF documents using PyMuPDF.
    
    Extracts text with page numbers, preserves document structure,
    and creates chunks with page/section metadata for citations.
    """

    def __init__(self):
        """Initialize PDF processor."""
        super().__init__()
        # Use intelligent chunking system
        self.chunker = get_foldex_chunker(
            chunk_size=600,  # ~150 tokens for qwen3:4b
            chunk_overlap=100,
            context_window_size=200,
        )

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
        if not os.path.exists(file_path):
            raise DocumentProcessingError(file_path, "File not found")

        try:
            self.logger.info("Processing PDF", file_path=file_path, file_id=file_id)
            
            # Generate file_id if not provided
            if not file_id:
                file_id = str(uuid.uuid4())

            self._update_progress(progress_callback, 0.1)

            # Prepare file metadata for chunker
            file_metadata = {
                "file_id": file_id,
                "file_name": metadata.get("file_name", "") if metadata else "",
                "mime_type": "application/pdf",
            }

            # Add any additional metadata
            if metadata:
                file_metadata.update({
                    k: v for k, v in metadata.items()
                    if k not in ["file_name", "mime_type"]
                })

            self._update_progress(progress_callback, 0.3)

            # Use intelligent chunking system
            langchain_docs = self.chunker.chunk_pdf(file_path, file_metadata)

            self._update_progress(progress_callback, 0.7)

            # Convert LangChain Documents to DocumentChunk objects with standardized metadata
            chunks: List[DocumentChunk] = []
            total_pages = max(
                (doc.metadata.get("page_number", 0) for doc in langchain_docs),
                default=0
            )
            
            for doc in langchain_docs:
                doc_meta = doc.metadata
                
                # Extract values from existing metadata
                chunk_id = doc_meta.get("chunk_id", str(uuid.uuid4()))
                file_name = doc_meta.get("file_name", "")
                drive_url = doc_meta.get("drive_url", doc_meta.get("web_view_link", ""))
                page_number = doc_meta.get("page_number", 0)
                section = doc_meta.get("section", "")
                authors = doc_meta.get("authors", [])
                document_title = doc_meta.get("document_title", "")
                
                # Convert authors list if it's a string
                if isinstance(authors, str):
                    authors = [a.strip() for a in authors.split(",") if a.strip()]
                elif not isinstance(authors, list):
                    authors = []
                
                # Build base metadata
                base_meta = MetadataBuilder.base_metadata(
                    file_id=file_id,
                    file_name=file_name,
                    file_type=FileType.PDF,
                    chunk_type=ChunkType.DOCUMENT_SECTION,
                    chunk_id=chunk_id,
                    drive_url=drive_url,
                )
                
                # Add PDF-specific metadata
                if page_number > 0:
                    pdf_meta = MetadataBuilder.pdf_metadata(
                        base=base_meta,
                        page_number=page_number,
                        total_pages=total_pages,
                        section=section,
                        authors=authors,
                        title=document_title,
                    )
                else:
                    pdf_meta = base_meta
                
                # Merge with any additional metadata from chunker
                additional_meta = {
                    k: v for k, v in doc_meta.items()
                    if k not in [
                        "chunk_id", "file_id", "file_name", "drive_url", "web_view_link",
                        "page_number", "section", "authors", "document_title",
                        "file_type", "chunk_type", "ingestion_date", "schema_version"
                    ]
                }
                
                final_metadata = MetadataBuilder.merge_metadata(pdf_meta, additional_meta)
                
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=doc.page_content,
                    file_id=file_id,
                    metadata=final_metadata,
                )
                chunks.append(chunk)

            self._update_progress(progress_callback, 1.0)

            self.logger.info(
                "PDF processing completed",
                file_path=file_path,
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
