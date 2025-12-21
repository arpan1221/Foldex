"""
Smart chunking system for Foldex with hierarchical structure preservation.

Preserves document structure, extracts metadata, and adds sliding window context.
Optimized for qwen3:4b with 500-800 token chunks.
"""

from typing import List, Dict, Any, Optional
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

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    fitz = None

from app.core.exceptions import DocumentProcessingError
from app.ingestion.metadata_extractor import MetadataExtractor

logger = structlog.get_logger(__name__)


class FoldexChunker:
    """
    Intelligent chunking with metadata preservation and hierarchical structure.
    
    Features:
    - Document-aware chunking (PDFs, text, code)
    - Metadata extraction (authors, titles, sections, page numbers)
    - Sliding window context (prev/next chunks)
    - Optimized for 500-800 tokens (qwen3:4b)
    - Preserves document structure and hierarchy
    """

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 150,
        context_window_size: int = 150,
    ):
        """Initialize Foldex chunker.

        Args:
            chunk_size: Target chunk size in characters (~200 tokens for 800 chars, optimized for Llama 3.2:3b)
            chunk_overlap: Overlap between chunks in characters
            context_window_size: Size of prev/next context window in characters
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain is required for chunking. Install with: pip install langchain")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.context_window_size = context_window_size

        # Initialize text splitter with intelligent separators
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=[
                "\n\n\n",  # Section breaks
                "\n\n",    # Paragraphs
                "\n",      # Lines
                ". ",      # Sentences
                "! ",      # Exclamations
                "? ",      # Questions
                "; ",      # Semicolons
                ", ",      # Clauses
                " ",       # Words
                "",        # Characters
            ],
        )

        # Initialize metadata extractor
        self.metadata_extractor = MetadataExtractor()

        logger.info(
            "Initialized FoldexChunker",
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            context_window=context_window_size,
        )

    def chunk_pdf(
        self,
        file_path: str,
        file_metadata: Dict[str, Any],
    ) -> List[Document]:
        """
        Extract and chunk PDF with rich metadata preservation.

        Args:
            file_path: Path to PDF file
            file_metadata: Dict with file_name, file_id, drive_url, etc.

        Returns:
            List of LangChain Document objects with comprehensive metadata

        Raises:
            DocumentProcessingError: If PDF processing fails
        """
        if not PYMUPDF_AVAILABLE:
            raise DocumentProcessingError(
                file_path,
                "PyMuPDF is required for PDF processing. Install with: pip install pymupdf"
            )

        try:
            logger.info("Chunking PDF", file_path=file_path, file_id=file_metadata.get("file_id"))

            # Open PDF
            doc = fitz.open(file_path)
            total_pages = len(doc)

            if total_pages == 0:
                raise DocumentProcessingError(file_path, "PDF has no pages")

            # Extract document-level metadata (title, authors, etc.)
            doc_metadata = self.metadata_extractor.extract_pdf_metadata(doc, file_path)
            logger.info(
                "Extracted PDF metadata",
                title=doc_metadata.get("document_title"),
                authors=doc_metadata.get("authors"),
                pages=total_pages,
            )

            # Extract text with page and section information
            page_contents = []
            for page_num in range(total_pages):
                page = doc[page_num]
                text = page.get_text()

                if text and text.strip():
                    # Clean and normalize text
                    text = self._clean_text(text)

                    # Detect section headers in text
                    section = self.metadata_extractor.detect_section(text, page_num)

                    page_contents.append({
                        "text": text,
                        "page_number": page_num + 1,
                        "section": section,
                    })

            doc.close()

            if not page_contents:
                raise DocumentProcessingError(file_path, "No text content extracted from PDF")

            # Create chunks with sliding window context
            all_chunks = []
            chunk_index = 0

            for page_data in page_contents:
                page_text = page_data["text"]
                page_number = page_data["page_number"]
                section = page_data["section"]

                # Split page text into chunks
                text_chunks = self.splitter.split_text(page_text)

                for chunk_text in text_chunks:
                    if not chunk_text.strip():
                        continue

                    # Generate unique chunk ID
                    chunk_id = self._generate_chunk_id(
                        chunk_text,
                        file_metadata.get("file_id", "unknown")
                    )

                    # Build comprehensive metadata
                    chunk_metadata = {
                        "chunk_id": chunk_id,
                        "file_id": file_metadata.get("file_id"),
                        "file_name": file_metadata.get("file_name"),
                        "drive_url": file_metadata.get("web_view_link") or file_metadata.get("drive_url"),
                        "page_number": page_number,
                        "section": section,
                        "document_title": doc_metadata.get("document_title"),
                        "authors": doc_metadata.get("authors"),
                        "chunk_index": chunk_index,
                        "total_chunks": None,  # Will be updated after all chunks are created
                        "mime_type": "application/pdf",
                        "source": file_path,
                    }

                    # Add any additional file metadata
                    for key in ["folder_id", "user_id", "created_at", "modified_at"]:
                        if key in file_metadata:
                            chunk_metadata[key] = file_metadata[key]

                    # Create LangChain Document
                    all_chunks.append(Document(
                        page_content=chunk_text,
                        metadata=chunk_metadata,
                    ))

                    chunk_index += 1

            # Update total_chunks count
            total_chunks = len(all_chunks)
            for chunk in all_chunks:
                chunk.metadata["total_chunks"] = total_chunks

            # Add sliding window context
            all_chunks = self._add_context_window(all_chunks)

            logger.info(
                "PDF chunking completed",
                file_path=file_path,
                total_pages=total_pages,
                total_chunks=total_chunks,
            )

            return all_chunks

        except DocumentProcessingError:
            raise
        except Exception as e:
            logger.error(
                "PDF chunking failed",
                file_path=file_path,
                error=str(e),
                exc_info=True,
            )
            raise DocumentProcessingError(
                file_path,
                f"PDF chunking failed: {str(e)}"
            ) from e

    def chunk_text(
        self,
        file_path: str,
        file_metadata: Dict[str, Any],
    ) -> List[Document]:
        """
        Chunk plain text files with metadata.

        Args:
            file_path: Path to text file
            file_metadata: Dict with file_name, file_id, drive_url, etc.

        Returns:
            List of LangChain Document objects with metadata

        Raises:
            DocumentProcessingError: If text processing fails
        """
        try:
            logger.info("Chunking text file", file_path=file_path)

            # Read file with encoding detection
            content = self._read_file_with_encoding(file_path)

            if not content or not content.strip():
                logger.warning("Text file is empty", file_path=file_path)
                return []

            # Clean text
            content = self._clean_text(content)

            # Determine if Markdown
            is_markdown = Path(file_path).suffix.lower() in [".md", ".markdown"]

            # Extract document-level metadata
            doc_metadata = self.metadata_extractor.extract_text_metadata(
                content,
                file_path,
                is_markdown=is_markdown
            )

            # Split into chunks
            text_chunks = self.splitter.split_text(content)

            # Create Document objects with metadata
            all_chunks = []
            for chunk_index, chunk_text in enumerate(text_chunks):
                if not chunk_text.strip():
                    continue

                chunk_id = self._generate_chunk_id(
                    chunk_text,
                    file_metadata.get("file_id", "unknown")
                )

                chunk_metadata = {
                    "chunk_id": chunk_id,
                    "file_id": file_metadata.get("file_id"),
                    "file_name": file_metadata.get("file_name"),
                    "drive_url": file_metadata.get("web_view_link") or file_metadata.get("drive_url"),
                    "document_title": doc_metadata.get("document_title"),
                    "chunk_index": chunk_index,
                    "total_chunks": len(text_chunks),
                    "mime_type": file_metadata.get("mime_type", "text/plain"),
                    "is_markdown": is_markdown,
                    "source": file_path,
                }

                # Add any additional file metadata
                for key in ["folder_id", "user_id", "created_at", "modified_at"]:
                    if key in file_metadata:
                        chunk_metadata[key] = file_metadata[key]

                all_chunks.append(Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata,
                ))

            # Add sliding window context
            all_chunks = self._add_context_window(all_chunks)

            logger.info(
                "Text chunking completed",
                file_path=file_path,
                total_chunks=len(all_chunks),
            )

            return all_chunks

        except Exception as e:
            logger.error(
                "Text chunking failed",
                file_path=file_path,
                error=str(e),
                exc_info=True,
            )
            raise DocumentProcessingError(
                file_path,
                f"Text chunking failed: {str(e)}"
            ) from e

    def chunk_code(
        self,
        file_path: str,
        file_metadata: Dict[str, Any],
    ) -> List[Document]:
        """
        Chunk code files preserving function/class boundaries.

        Args:
            file_path: Path to code file
            file_metadata: Dict with file_name, file_id, drive_url, etc.

        Returns:
            List of LangChain Document objects with metadata

        Raises:
            DocumentProcessingError: If code processing fails
        """
        try:
            logger.info("Chunking code file", file_path=file_path)

            # Read file
            content = self._read_file_with_encoding(file_path)

            if not content or not content.strip():
                logger.warning("Code file is empty", file_path=file_path)
                return []

            # Detect programming language
            file_ext = Path(file_path).suffix.lower()
            language = self._detect_language(file_ext)

            # Use AST-based chunking for supported languages
            supported_ast_languages = ["python", "javascript", "typescript", "js", "ts"]
            if language in supported_ast_languages:
                try:
                    from app.ingestion.ast_code_chunker import ASTCodeChunker
                    ast_chunker = ASTCodeChunker(
                        max_chunk_size=self.chunk_size,
                        min_chunk_size=self.chunk_overlap,
                    )
                    all_chunks = ast_chunker.chunk_code(content, file_metadata, language)
                    
                    # Add sliding window context
                    all_chunks = self._add_context_window(all_chunks)
                    
                    logger.info(
                        "AST-based code chunking completed",
                        file_path=file_path,
                        language=language,
                        chunks=len(all_chunks),
                    )
                    
                    return all_chunks
                except Exception as e:
                    logger.warning(
                        "AST-based chunking failed, falling back to text-based",
                        language=language,
                        error=str(e),
                        file_path=file_path,
                    )
                    # Fall through to text-based chunking

            # Use language-specific splitter for other languages or as fallback
            try:
                from langchain.text_splitter import (
                    PythonCodeTextSplitter,
                    Language,
                    RecursiveCharacterTextSplitter,
                )

                if language == "python":
                    code_splitter = PythonCodeTextSplitter(
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    )
                else:
                    # Use RecursiveCharacterTextSplitter with code-aware separators
                    code_splitter = RecursiveCharacterTextSplitter.from_language(
                        language=Language.PYTHON if language == "python" else Language.JS,
                        chunk_size=self.chunk_size,
                        chunk_overlap=self.chunk_overlap,
                    )

                text_chunks = code_splitter.split_text(content)
            except Exception as e:
                logger.warning(
                    "Language-specific splitting failed, using general splitter",
                    error=str(e)
                )
                text_chunks = self.splitter.split_text(content)

            # Create Document objects
            all_chunks = []
            for chunk_index, chunk_text in enumerate(text_chunks):
                if not chunk_text.strip():
                    continue

                chunk_id = self._generate_chunk_id(
                    chunk_text,
                    file_metadata.get("file_id", "unknown")
                )

                chunk_metadata = {
                    "chunk_id": chunk_id,
                    "file_id": file_metadata.get("file_id"),
                    "file_name": file_metadata.get("file_name"),
                    "drive_url": file_metadata.get("web_view_link") or file_metadata.get("drive_url"),
                    "language": language,
                    "chunk_index": chunk_index,
                    "total_chunks": len(text_chunks),
                    "mime_type": file_metadata.get("mime_type", "text/plain"),
                    "source": file_path,
                }

                # Add any additional file metadata
                for key in ["folder_id", "user_id", "created_at", "modified_at"]:
                    if key in file_metadata:
                        chunk_metadata[key] = file_metadata[key]

                all_chunks.append(Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata,
                ))

            # Add sliding window context
            all_chunks = self._add_context_window(all_chunks)

            logger.info(
                "Code chunking completed",
                file_path=file_path,
                language=language,
                total_chunks=len(all_chunks),
            )

            return all_chunks

        except Exception as e:
            logger.error(
                "Code chunking failed",
                file_path=file_path,
                error=str(e),
                exc_info=True,
            )
            raise DocumentProcessingError(
                file_path,
                f"Code chunking failed: {str(e)}"
            ) from e

    def _generate_chunk_id(self, content: str, file_id: str) -> str:
        """Generate unique chunk ID from content and file ID.

        Args:
            content: Chunk content
            file_id: File identifier

        Returns:
            Unique chunk ID (MD5 hash)
        """
        # Use first 100 chars of content + file_id for uniqueness
        unique_string = f"{file_id}:{content[:100]}"
        return hashlib.md5(unique_string.encode()).hexdigest()

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
                chunk.metadata["prev_context"] = prev_content[-self.context_window_size:]
            else:
                chunk.metadata["prev_context"] = ""

            # Next context (first N chars of next chunk)
            if i < len(chunks) - 1:
                next_content = chunks[i + 1].page_content
                chunk.metadata["next_context"] = next_content[:self.context_window_size]
            else:
                chunk.metadata["next_context"] = ""

        return chunks

    def _clean_text(self, text: str) -> str:
        """Clean and normalize text.

        Args:
            text: Raw text

        Returns:
            Cleaned text
        """
        # Remove excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)

        # Remove control characters except newlines and tabs
        text = ''.join(char for char in text if char.isprintable() or char in '\n\t')

        return text.strip()

    def _read_file_with_encoding(self, file_path: str) -> str:
        """Read file with encoding detection.

        Args:
            file_path: Path to file

        Returns:
            File content as string

        Raises:
            DocumentProcessingError: If file cannot be read
        """
        encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252", "iso-8859-1"]

        for encoding in encodings:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
            except Exception as e:
                raise DocumentProcessingError(
                    file_path,
                    f"Failed to read file: {str(e)}"
                )

        raise DocumentProcessingError(
            file_path,
            "Could not decode file with any supported encoding"
        )

    def _detect_language(self, file_ext: str) -> str:
        """Detect programming language from file extension.

        Args:
            file_ext: File extension (e.g., '.py', '.js')

        Returns:
            Language name
        """
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".java": "java",
            ".cpp": "cpp",
            ".c": "c",
            ".go": "go",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
        }

        return language_map.get(file_ext.lower(), "unknown")


# Global instance
_foldex_chunker: Optional[FoldexChunker] = None


def get_foldex_chunker(
    chunk_size: int = 800,
    chunk_overlap: int = 150,
    context_window_size: int = 150,
) -> FoldexChunker:
    """Get or create global FoldexChunker instance.

    Args:
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between chunks
        context_window_size: Size of context window

    Returns:
        FoldexChunker instance
    """
    global _foldex_chunker

    if _foldex_chunker is None:
        _foldex_chunker = FoldexChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            context_window_size=context_window_size,
        )

    return _foldex_chunker

