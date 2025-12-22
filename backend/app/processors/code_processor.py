"""Source code file processor with intelligent hierarchical chunking."""

from typing import List, Optional, Callable
import uuid
import structlog

from app.processors.base import BaseProcessor
from app.models.documents import DocumentChunk
from app.core.exceptions import DocumentProcessingError
from app.ingestion.chunking import get_foldex_chunker
from app.ingestion.metadata_schema import MetadataBuilder, FileType, ChunkType

logger = structlog.get_logger(__name__)


class CodeProcessor(BaseProcessor):
    """Processor for source code files with function/class boundary preservation."""

    def __init__(self):
        """Initialize code processor."""
        super().__init__()
        # Use intelligent chunking system
        self.chunker = get_foldex_chunker(
            chunk_size=600,  # ~150 tokens for qwen3:4b
            chunk_overlap=100,
            context_window_size=200,
        )

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
            
            # Generate file_id if not provided
            if not file_id:
                file_id = str(uuid.uuid4())

            # Prepare file metadata for chunker
            file_metadata = {
                "file_id": file_id,
                "file_name": metadata.get("file_name", "") if metadata else "",
                "mime_type": metadata.get("mime_type", "text/plain") if metadata else "text/plain",
            }

            # Add any additional metadata
            if metadata:
                file_metadata.update({
                    k: v for k, v in metadata.items()
                    if k not in ["file_name", "mime_type"]
                })

            # Use intelligent chunking system with document context
            langchain_docs = await self.chunker.chunk_code_with_context(
                file_path,
                file_metadata,
                include_summary_in_content=True,
            )

            if not langchain_docs:
                logger.warning("Code file is empty", file_path=file_path)
                return []

            # Convert LangChain Documents to DocumentChunk objects with standardized metadata
            chunks: List[DocumentChunk] = []
            for doc in langchain_docs:
                doc_meta = doc.metadata
                
                # Extract values from existing metadata
                chunk_id = doc_meta.get("chunk_id", str(uuid.uuid4()))
                file_name = doc_meta.get("file_name", "")
                drive_url = doc_meta.get("drive_url", doc_meta.get("web_view_link", ""))
                language = doc_meta.get("language", "unknown")
                
                # Try to extract function/class names from content (simple regex)
                function_name = doc_meta.get("function_name", "")
                class_name = doc_meta.get("class_name", "")
                line_start = doc_meta.get("line_start", 0)
                line_end = doc_meta.get("line_end", 0)
                
                # Determine chunk type based on content
                chunk_type = ChunkType.CODE_FUNCTION if function_name else (
                    ChunkType.CODE_CLASS if class_name else ChunkType.CODE_MODULE
                )
                
                # Build base metadata
                base_meta = MetadataBuilder.base_metadata(
                    file_id=file_id,
                    file_name=file_name,
                    file_type=FileType.CODE,
                    chunk_type=chunk_type,
                    chunk_id=chunk_id,
                    drive_url=drive_url,
                )
                
                # Add code-specific metadata
                code_meta = MetadataBuilder.code_metadata(
                    base=base_meta,
                    language=language,
                    function_name=function_name,
                    class_name=class_name,
                    line_start=line_start,
                    line_end=line_end,
                )
                
                # Merge with any additional metadata
                additional_meta = {
                    k: v for k, v in doc_meta.items()
                    if k not in [
                        "chunk_id", "file_id", "file_name", "drive_url", "web_view_link",
                        "language", "function_name", "class_name", "line_start", "line_end",
                        "file_type", "chunk_type", "ingestion_date", "schema_version"
                    ]
                }
                
                final_metadata = MetadataBuilder.merge_metadata(code_meta, additional_meta)
                
                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=doc.page_content,
                    file_id=file_id,
                    metadata=final_metadata,
                )
                chunks.append(chunk)

            if progress_callback:
                progress_callback(1.0)

            logger.info("Code processing completed", file_path=file_path, chunks=len(chunks))
            return chunks
                
        except DocumentProcessingError:
            raise
        except Exception as e:
            logger.error("Code processing failed", file_path=file_path, error=str(e), exc_info=True)
            raise DocumentProcessingError(file_path, f"Code processing failed: {str(e)}") from e

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
