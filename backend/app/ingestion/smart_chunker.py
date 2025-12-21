"""Smart chunking with structure preservation and context windows.

Enhanced chunking that preserves document structure, maintains section boundaries,
and adds context windows for better retrieval and understanding.
"""

from typing import List, Dict, Optional, Any
import re
import hashlib
import structlog
from pathlib import Path

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        RecursiveCharacterTextSplitter = None
        Document = None

from app.core.exceptions import DocumentProcessingError
from app.ingestion.metadata_schema import MetadataBuilder, FileType, ChunkType

logger = structlog.get_logger(__name__)


class SmartChunker:
    """
    Intelligent chunking that preserves document structure and maintains context.
    
    Features:
    1. Preserves document structure (sections, headers)
    2. Adds context windows (prev/next chunk snippets)
    3. Respects semantic boundaries
    4. Maintains metadata richness
    5. Section-aware chunking (chunks by sections when available)
    6. Header preservation (includes headers in chunk content)
    """

    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 100,
        context_window: int = 200,
    ):
        """
        Initialize smart chunker.

        Args:
            chunk_size: Target characters per chunk
            chunk_overlap: Overlap between chunks in characters
            context_window: Characters of context to include from adjacent chunks
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for chunking. Install with: pip install langchain")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.context_window = context_window

        # Initialize text splitter with intelligent separators
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",  # Multiple blank lines (section breaks)
                "\n\n",   # Paragraph breaks
                "\n",     # Line breaks
                ". ",     # Sentence boundaries
                "! ",     # Exclamations
                "? ",     # Questions
                "; ",     # Semicolons
                ", ",     # Clauses
                " ",      # Word boundaries
                "",       # Character-level (last resort)
            ],
        )

        logger.info(
            "Initialized SmartChunker",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            context_window=context_window,
        )

    def chunk_with_structure(
        self,
        text: str,
        file_metadata: Dict[str, Any],
        sections: Optional[List[Dict[str, Any]]] = None,
    ) -> List[Document]:
        """
        Chunk text while preserving structure.

        Args:
            text: Full document text
            file_metadata: File-level metadata (must include file_id, file_name)
            sections: Optional list of section dicts with {title, start, end} for sections

        Returns:
            List of Documents with enhanced metadata and context windows
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for chunking")
            return []

        # If sections provided, chunk each section separately
        if sections:
            return self._chunk_by_sections(text, file_metadata, sections)

        # Otherwise, intelligent chunking with context
        return self._chunk_with_context(text, file_metadata)

    def _chunk_by_sections(
        self,
        text: str,
        file_metadata: Dict[str, Any],
        sections: List[Dict[str, Any]],
    ) -> List[Document]:
        """
        Chunk respecting section boundaries.

        Each section is chunked separately, preserving section context
        in each chunk's metadata and content.

        Args:
            text: Full document text
            file_metadata: File-level metadata
            sections: List of section dicts with {title, start, end}

        Returns:
            List of Documents with section-aware chunking
        """
        all_chunks = []
        global_chunk_index = 0

        for section_idx, section_info in enumerate(sections):
            section_title = section_info.get("title", f"Section {section_idx + 1}")
            section_start = section_info.get("start", 0)
            section_end = section_info.get("end", len(text))

            # Extract section text
            section_text = text[section_start:section_end].strip()

            if not section_text:
                continue

            # Chunk this section
            section_chunks = self.splitter.split_text(section_text)

            for chunk_idx, chunk_text in enumerate(section_chunks):
                if not chunk_text.strip():
                    continue

                # Enrich chunk with section context
                # Include section title in chunk content for better retrieval
                enriched_text = f"[Section: {section_title}]\n\n{chunk_text}"

                # Generate chunk ID
                chunk_id = self._generate_chunk_id(
                    file_metadata.get("file_id", "unknown"),
                    chunk_text,
                )

                # Build base metadata using standardized schema
                file_type = self._determine_file_type(file_metadata)
                base_meta = MetadataBuilder.base_metadata(
                    file_id=file_metadata.get("file_id", "unknown"),
                    file_name=file_metadata.get("file_name", "Unknown"),
                    file_type=file_type,
                    chunk_type=ChunkType.DOCUMENT_SECTION,
                    chunk_id=chunk_id,
                    drive_url=file_metadata.get("drive_url", file_metadata.get("web_view_link", "")),
                )

                # Add section-specific metadata
                chunk_metadata = {
                    **base_meta,
                    "section": section_title,
                    "section_index": section_idx,
                    "chunk_index": global_chunk_index,
                    "chunk_index_in_section": chunk_idx,
                    "total_chunks_in_section": len(section_chunks),
                    "source": file_metadata.get("source", ""),
                }

                # Add any additional file metadata
                for key in ["folder_id", "user_id", "created_at", "modified_at", "mime_type"]:
                    if key in file_metadata:
                        chunk_metadata[key] = file_metadata[key]

                doc = Document(
                    page_content=enriched_text,
                    metadata=chunk_metadata,
                )
                all_chunks.append(doc)
                global_chunk_index += 1

        # Update total_chunks count
        total_chunks = len(all_chunks)
        for chunk in all_chunks:
            chunk.metadata["total_chunks"] = total_chunks

        # Add context windows
        all_chunks = self._add_context_window(all_chunks)

        logger.info(
            "Section-aware chunking completed",
            section_count=len(sections),
            total_chunks=total_chunks,
        )

        return all_chunks

    def _chunk_with_context(
        self,
        text: str,
        file_metadata: Dict[str, Any],
    ) -> List[Document]:
        """
        Chunk with surrounding context windows.

        Adds prev/next snippets to each chunk for better retrieval
        and understanding of chunk position in document.

        Args:
            text: Full document text
            file_metadata: File-level metadata

        Returns:
            List of Documents with context windows
        """
        # Split into chunks
        chunks = self.splitter.split_text(text)

        if not chunks:
            logger.warning("No chunks created from text")
            return []

        documents = []
        file_type = self._determine_file_type(file_metadata)

        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue

            # Get context from surrounding chunks
            prev_context = ""
            next_context = ""

            if i > 0:
                # Last N chars of previous chunk
                prev_chunk = chunks[i - 1]
                prev_context = prev_chunk[-self.context_window:] if len(prev_chunk) > self.context_window else prev_chunk

            if i < len(chunks) - 1:
                # First N chars of next chunk
                next_chunk = chunks[i + 1]
                next_context = next_chunk[:self.context_window] if len(next_chunk) > self.context_window else next_chunk

            # Generate chunk ID
            chunk_id = self._generate_chunk_id(
                file_metadata.get("file_id", "unknown"),
                chunk_text,
            )

            # Build base metadata using standardized schema
            base_meta = MetadataBuilder.base_metadata(
                file_id=file_metadata.get("file_id", "unknown"),
                file_name=file_metadata.get("file_name", "Unknown"),
                file_type=file_type,
                chunk_type=ChunkType.DOCUMENT_SECTION,
                chunk_id=chunk_id,
                drive_url=file_metadata.get("drive_url", file_metadata.get("web_view_link", "")),
            )

            # Build enriched metadata
            metadata = {
                **base_meta,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_char_count": len(chunk_text),
                "source": file_metadata.get("source", ""),
            }

            # Add context windows (for better retrieval)
            metadata["prev_context"] = prev_context
            metadata["next_context"] = next_context

            # Add any additional file metadata
            for key in ["folder_id", "user_id", "created_at", "modified_at", "mime_type"]:
                if key in file_metadata:
                    metadata[key] = file_metadata[key]

            doc = Document(
                page_content=chunk_text,
                metadata=metadata,
            )
            documents.append(doc)

        logger.info(
            "Context-aware chunking completed",
            total_chunks=len(documents),
            context_window=self.context_window,
        )

        return documents

    def chunk_with_headers(
        self,
        text: str,
        file_metadata: Dict[str, Any],
        header_pattern: str = r"^#+\s+",  # Markdown headers by default
    ) -> List[Document]:
        """
        Chunk preserving headers in each chunk.

        Useful for markdown/structured documents where headers indicate
        section boundaries. Each chunk includes its section header.

        Args:
            text: Full document text
            file_metadata: File-level metadata
            header_pattern: Regex pattern to match headers (default: Markdown headers)

        Returns:
            List of Documents with headers preserved in content
        """
        # Find all headers
        headers = []
        for match in re.finditer(header_pattern, text, re.MULTILINE):
            line_start = match.start()
            # Get full header line
            line_end = text.find("\n", line_start)
            if line_end == -1:
                line_end = len(text)

            header_text = text[line_start:line_end].strip()
            # Clean header (remove markdown markers)
            clean_header = re.sub(r"^#+\s*", "", header_text).strip()

            headers.append({
                "text": clean_header,
                "original": header_text,
                "position": line_start,
            })

        if not headers:
            # No headers found, use regular chunking
            logger.debug("No headers found, using regular chunking")
            return self._chunk_with_context(text, file_metadata)

        # Build sections from headers
        sections = []
        for i, header in enumerate(headers):
            start = header["position"]
            end = headers[i + 1]["position"] if i < len(headers) - 1 else len(text)

            sections.append({
                "title": header["text"],
                "start": start,
                "end": end,
            })

        logger.info(
            "Detected headers for chunking",
            header_count=len(headers),
            section_count=len(sections),
        )

        return self._chunk_by_sections(text, file_metadata, sections)

    def _generate_chunk_id(self, file_id: str, content: str) -> str:
        """
        Generate unique chunk ID.

        Args:
            file_id: File identifier
            content: Chunk content

        Returns:
            Unique chunk ID
        """
        # Use first 100 chars of content + file_id for uniqueness
        content_hash = hashlib.md5(content[:100].encode()).hexdigest()[:8]
        return f"{file_id}_{content_hash}"

    def _determine_file_type(self, file_metadata: Dict[str, Any]) -> FileType:
        """
        Determine file type from metadata.

        Args:
            file_metadata: File metadata dictionary

        Returns:
            FileType enum value
        """
        mime_type = file_metadata.get("mime_type", "").lower()
        file_name = file_metadata.get("file_name", "").lower()

        # Check MIME type first
        if "pdf" in mime_type:
            return FileType.PDF
        elif "audio" in mime_type:
            return FileType.AUDIO
        elif "video" in mime_type:
            return FileType.VIDEO
        elif "text" in mime_type or "markdown" in mime_type:
            if file_name.endswith((".md", ".markdown")):
                return FileType.MARKDOWN
            return FileType.TEXT

        # Check file extension
        if file_name.endswith(".pdf"):
            return FileType.PDF
        elif file_name.endswith((".md", ".markdown")):
            return FileType.MARKDOWN
        elif file_name.endswith((".py", ".js", ".ts", ".java", ".cpp", ".go", ".rs", ".rb")):
            return FileType.CODE
        elif file_name.endswith((".m4a", ".mp3", ".wav", ".flac", ".ogg")):
            return FileType.AUDIO
        elif file_name.endswith((".mp4", ".mov", ".avi")):
            return FileType.VIDEO

        return FileType.UNKNOWN

    def _add_context_window(self, chunks: List[Document]) -> List[Document]:
        """
        Add previous/next context to each chunk for better retrieval.

        Args:
            chunks: List of Document objects

        Returns:
            List of Document objects with context added to metadata
        """
        for i, chunk in enumerate(chunks):
            # Previous context (last N chars of previous chunk)
            if i > 0:
                prev_content = chunks[i - 1].page_content
                chunk.metadata["prev_context"] = (
                    prev_content[-self.context_window:]
                    if len(prev_content) > self.context_window
                    else prev_content
                )
            else:
                chunk.metadata["prev_context"] = ""

            # Next context (first N chars of next chunk)
            if i < len(chunks) - 1:
                next_content = chunks[i + 1].page_content
                chunk.metadata["next_context"] = (
                    next_content[:self.context_window]
                    if len(next_content) > self.context_window
                    else next_content
                )
            else:
                chunk.metadata["next_context"] = ""

        logger.debug(
            "Added context windows",
            chunk_count=len(chunks),
            context_window_size=self.context_window,
        )

        return chunks

    def chunk_markdown(
        self,
        text: str,
        file_metadata: Dict[str, Any],
    ) -> List[Document]:
        """
        Chunk Markdown files with header preservation.

        Convenience method for Markdown files that automatically detects
        and preserves headers.

        Args:
            text: Markdown text
            file_metadata: File-level metadata

        Returns:
            List of Documents with headers preserved
        """
        return self.chunk_with_headers(
            text,
            file_metadata,
            header_pattern=r"^#+\s+",  # Markdown headers
        )

    def chunk_with_semantic_boundaries(
        self,
        text: str,
        file_metadata: Dict[str, Any],
        boundary_markers: Optional[List[str]] = None,
    ) -> List[Document]:
        """
        Chunk respecting semantic boundaries (paragraphs, sections, etc.).

        Args:
            text: Full document text
            file_metadata: File-level metadata
            boundary_markers: Optional list of boundary markers (e.g., ["\n\n\n", "\n\n"])

        Returns:
            List of Documents chunked at semantic boundaries
        """
        if boundary_markers:
            # Use custom separators
            custom_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                length_function=len,
                separators=boundary_markers + [" ", ""],  # Always include word/char fallbacks
            )
            chunks = custom_splitter.split_text(text)
        else:
            # Use default splitter
            chunks = self.splitter.split_text(text)

        # Convert to Documents with context
        documents = []
        file_type = self._determine_file_type(file_metadata)

        for i, chunk_text in enumerate(chunks):
            if not chunk_text.strip():
                continue

            chunk_id = self._generate_chunk_id(
                file_metadata.get("file_id", "unknown"),
                chunk_text,
            )

            base_meta = MetadataBuilder.base_metadata(
                file_id=file_metadata.get("file_id", "unknown"),
                file_name=file_metadata.get("file_name", "Unknown"),
                file_type=file_type,
                chunk_type=ChunkType.DOCUMENT_SECTION,
                chunk_id=chunk_id,
                drive_url=file_metadata.get("drive_url", file_metadata.get("web_view_link", "")),
            )

            metadata = {
                **base_meta,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "source": file_metadata.get("source", ""),
            }

            # Add file metadata
            for key in ["folder_id", "user_id", "created_at", "modified_at", "mime_type"]:
                if key in file_metadata:
                    metadata[key] = file_metadata[key]

            documents.append(Document(
                page_content=chunk_text,
                metadata=metadata,
            ))

        # Add context windows
        documents = self._add_context_window(documents)

        return documents

