"""Text and Markdown file processor with intelligent hierarchical chunking."""

from typing import List, Optional, Callable
import os
import uuid
import structlog

from app.processors.base import BaseProcessor
from app.models.documents import DocumentChunk
from app.core.exceptions import DocumentProcessingError
from app.config.settings import settings
from app.ingestion.chunking import get_foldex_chunker
from app.ingestion.smart_chunker import SmartChunker
from app.ingestion.metadata_schema import MetadataBuilder, FileType, ChunkType

logger = structlog.get_logger(__name__)


class TextProcessor(BaseProcessor):
    """Processor for plain text and Markdown files.
    
    Handles various text formats with proper encoding detection
    and smart chunking that preserves document structure.
    """

    def __init__(self):
        """Initialize text processor."""
        super().__init__()
        # Use SmartChunker for better structure preservation and header detection
        self.smart_chunker = SmartChunker(
            chunk_size=600,  # ~150 tokens for qwen3:4b
            chunk_overlap=100,
            context_window=200,
        )
        # Keep FoldexChunker as fallback for edge cases
        self.foldex_chunker = get_foldex_chunker(
            chunk_size=600,
            chunk_overlap=100,
            context_window_size=200,
        )
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
        file_lower = file_path.lower()
        
        # Check by extension first (more specific)
        if any(file_lower.endswith(ext) for ext in self.supported_extensions):
            # Exclude HTML files - they should be handled by UnstructuredProcessor
            if file_lower.endswith((".html", ".htm")):
                return False
            return True
        
        # Check by MIME type (more restrictive)
        if mime_type:
            # Only accept specific MIME types, not all text/* 
            # This allows UnstructuredProcessor to handle HTML, XML, etc.
            return mime_type in self.supported_mime_types
        
        return False

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

            self._update_progress(progress_callback, 0.3)

            # Prepare file metadata for chunker
            file_metadata = {
                "file_id": file_id,
                "file_name": metadata.get("file_name", "") if metadata else os.path.basename(file_path),
                "mime_type": metadata.get("mime_type", "text/plain") if metadata else "text/plain",
                "source": file_path,  # For SmartChunker
            }

            # Add any additional metadata (drive_url, folder_id, etc.)
            if metadata:
                file_metadata.update({
                    k: v for k, v in metadata.items()
                    if k not in ["file_name", "mime_type"]
                })
                # Ensure drive_url is available
                if "drive_url" not in file_metadata and "web_view_link" in metadata:
                    file_metadata["drive_url"] = metadata["web_view_link"]

            self._update_progress(progress_callback, 0.5)

            # Read file content
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with different encodings
                encodings = ["utf-8-sig", "latin-1", "cp1252"]
                content = None
                for encoding in encodings:
                    try:
                        with open(file_path, "r", encoding=encoding) as f:
                            content = f.read()
                        break
                    except UnicodeDecodeError:
                        continue
                
                if content is None:
                    raise DocumentProcessingError(file_path, "Could not decode file with any supported encoding")

            if not content or not content.strip():
                self.logger.warning("File is empty or contains no text", file_path=file_path)
                return []

            # Determine if Markdown
            is_markdown = file_path.lower().endswith((".md", ".markdown"))
            file_metadata["is_markdown"] = is_markdown

            # Use SmartChunker for better structure preservation
            # For Markdown, use header-preserving chunking
            if is_markdown:
                langchain_docs = self.smart_chunker.chunk_markdown(content, file_metadata)
            else:
                # For plain text, use structure-aware chunking
                langchain_docs = self.smart_chunker.chunk_with_structure(content, file_metadata)

            if not langchain_docs:
                self.logger.warning("No chunks created from text", file_path=file_path)
                return []

            self._update_progress(progress_callback, 0.8)

            # Convert LangChain Documents to DocumentChunk objects with standardized metadata
            chunks: List[DocumentChunk] = []
            for doc in langchain_docs:
                doc_meta = doc.metadata
                
                # SmartChunker already creates standardized metadata using MetadataBuilder
                # We just need to enhance it with text-specific fields if needed
                chunk_id = doc_meta.get("chunk_id", str(uuid.uuid4()))
                section = doc_meta.get("section", "")
                line_start = doc_meta.get("line_start", 0)
                line_end = doc_meta.get("line_end", 0)
                
                # If SmartChunker already created standardized metadata, use it
                if "schema_version" in doc_meta:
                    # SmartChunker already created base metadata, just add text-specific fields
                    final_metadata = doc_meta.copy()
                    
                    # Add text-specific metadata if line numbers are available
                    if line_start > 0 or line_end > 0:
                        # Enhance with text metadata builder
                        text_meta = MetadataBuilder.text_metadata(
                            base={
                                "file_id": doc_meta.get("file_id", file_id),
                                "file_name": doc_meta.get("file_name", ""),
                                "file_type": doc_meta.get("file_type", FileType.TEXT.value),
                                "chunk_type": doc_meta.get("chunk_type", ChunkType.DOCUMENT_SECTION.value),
                                "chunk_id": chunk_id,
                                "drive_url": doc_meta.get("drive_url", ""),
                            },
                            line_start=line_start,
                            line_end=line_end,
                            section=section,
                        )
                        # Merge text-specific fields into final metadata
                        final_metadata.update({
                            k: v for k, v in text_meta.items()
                            if k in ["line_start", "line_end", "searchable_location", "citation_format"]
                        })
                        if section and "section" not in final_metadata:
                            final_metadata["section"] = section
                else:
                    # Fallback: build metadata from scratch (shouldn't happen with SmartChunker)
                    is_markdown = doc_meta.get("is_markdown", False) or file_path.lower().endswith((".md", ".markdown"))
                    file_type = FileType.MARKDOWN if is_markdown else FileType.TEXT
                    
                    base_meta = MetadataBuilder.base_metadata(
                        file_id=file_id,
                        file_name=doc_meta.get("file_name", ""),
                        file_type=file_type,
                        chunk_type=ChunkType.DOCUMENT_SECTION,
                        chunk_id=chunk_id,
                        drive_url=doc_meta.get("drive_url", doc_meta.get("web_view_link", "")),
                    )
                    
                    if line_start > 0 or line_end > 0:
                        final_metadata = MetadataBuilder.text_metadata(
                            base=base_meta,
                            line_start=line_start,
                            line_end=line_end,
                            section=section,
                        )
                    else:
                        final_metadata = base_meta.copy()
                        if section:
                            final_metadata["section"] = section
                    
                    # Preserve SmartChunker fields
                    for field in ["prev_context", "next_context", "section_index", "chunk_index_in_section",
                                  "total_chunks_in_section", "chunk_index", "total_chunks", "chunk_char_count"]:
                        if field in doc_meta:
                            final_metadata[field] = doc_meta[field]
                
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=doc.page_content,
                    file_id=file_id,
                    metadata=final_metadata,
                    embedding=None,
                )
                chunks.append(chunk)

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
