"""Source code file processor."""

from typing import List, Optional, Callable
import structlog

from app.processors.base import BaseProcessor
from app.models.documents import DocumentChunk
from app.core.exceptions import DocumentProcessingError

logger = structlog.get_logger(__name__)


class CodeProcessor(BaseProcessor):
    """Processor for source code files using tree-sitter."""

    def __init__(self):
        """Initialize code processor."""
        # TODO: Initialize tree-sitter parsers for different languages

    async def can_process(self, file_path: str, mime_type: Optional[str] = None) -> bool:
        """Check if file is source code.

        Args:
            file_path: Path to file
            mime_type: Optional MIME type for faster detection

        Returns:
            True if file is source code
        """
        # Check MIME type first if provided
        if mime_type:
            code_mime_types = self.get_supported_mime_types()
            if mime_type in code_mime_types:
                return True
        
        # Fall back to extension check
        code_extensions = self.get_supported_extensions()
        return any(file_path.lower().endswith(ext) for ext in code_extensions)

    async def process(
        self,
        file_path: str,
        file_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[DocumentChunk]:
        """Process code file, preserving function/class boundaries.

        Args:
            file_path: Path to code file
            file_id: Optional file identifier for chunk IDs
            metadata: Optional file metadata to include in chunks
            progress_callback: Optional callback for progress updates (0.0 to 1.0)

        Returns:
            List of document chunks with code structure metadata

        Raises:
            DocumentProcessingError: If processing fails
        """
        try:
            logger.info("Processing code file", file_path=file_path, file_id=file_id)
            
            # TODO: Implement code processing with tree-sitter
            # For now, read the file content and create a basic chunk
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with latin-1 encoding as fallback
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            
            if not content.strip():
                raise DocumentProcessingError(file_path, "Empty code file")
            
            file_name = metadata.get("file_name", "unknown") if metadata else "unknown"
            
            # Create a single chunk with the entire file content
            # TODO: Split by functions/classes using tree-sitter
            code_chunk = DocumentChunk(
                chunk_id=self._generate_chunk_id(file_id or "unknown", 0),
                content=content,
                file_id=file_id or "unknown",
                metadata={
                    **(metadata or {}),
                    "chunk_index": 0,
                    "content_type": "code",
                    "file_name": file_name,
                    "note": "Basic code processing - tree-sitter parsing not yet implemented"
                },
                embedding=None
            )
            
            if progress_callback:
                progress_callback(1.0)
            
            return [code_chunk]
            
        except DocumentProcessingError:
            raise
        except Exception as e:
            logger.error("Code processing failed", file_path=file_path, error=str(e))
            raise DocumentProcessingError(file_path, str(e))

    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions.

        Returns:
            List of code extensions
        """
        return [
            ".py",
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".java",
            ".cpp",
            ".c",
            ".go",
            ".rs",
            ".rb",
        ]

    def get_supported_mime_types(self) -> List[str]:
        """Get supported MIME types.

        Returns:
            List of code MIME types
        """
        return [
            "text/x-python",
            "application/x-python-code",
            "text/javascript",
            "application/javascript",
            "application/x-javascript",
            "text/typescript",
            "application/typescript",
            "text/x-java-source",
            "text/x-c",
            "text/x-c++",
            "text/x-go",
            "text/x-rust",
            "text/x-ruby",
            "application/x-ruby",
            "text/plain",  # Many code files are served as text/plain
        ]

