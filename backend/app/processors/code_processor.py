"""Source code file processor."""

from typing import List
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

    async def can_process(self, file_path: str) -> bool:
        """Check if file is source code.

        Args:
            file_path: Path to file

        Returns:
            True if file is source code
        """
        code_extensions = [
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
        return any(file_path.lower().endswith(ext) for ext in code_extensions)

    async def process(self, file_path: str) -> List[DocumentChunk]:
        """Process code file, preserving function/class boundaries.

        Args:
            file_path: Path to code file

        Returns:
            List of document chunks with code structure metadata

        Raises:
            DocumentProcessingError: If processing fails
        """
        try:
            logger.info("Processing code file", file_path=file_path)
            # TODO: Implement code processing with tree-sitter
            # 1. Parse code with tree-sitter
            # 2. Extract functions, classes, modules
            # 3. Create chunks preserving structure
            # 4. Include metadata (function name, class name, etc.)
            chunks: List[DocumentChunk] = []
            return chunks
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

