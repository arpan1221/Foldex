"""LangChain source document tracking and metadata preservation."""

from typing import List, Dict, Any, Optional
import structlog

try:
    from langchain_core.documents import Document
    from langchain_core.callbacks import BaseCallbackHandler
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.schema import Document
        from langchain.callbacks.base import BaseCallbackHandler
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        Document = None
        BaseCallbackHandler = None

from app.core.exceptions import ProcessingError
from app.models.documents import DocumentChunk

logger = structlog.get_logger(__name__)


class SourceTracker:
    """Tracks source documents and preserves metadata for citations."""

    def __init__(self):
        """Initialize source tracker."""
        self.tracked_sources: List[Dict[str, Any]] = []
        self.logger = structlog.get_logger(__name__)

    def track_document(
        self,
        document: Document,
        retrieval_score: Optional[float] = None,
        retrieval_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Track a source document with metadata.

        Args:
            document: LangChain Document object
            retrieval_score: Optional retrieval relevance score
            retrieval_method: Optional retrieval method used

        Returns:
            Dictionary with tracked source information
        """
        try:
            metadata = document.metadata if hasattr(document, "metadata") else {}
            content = document.page_content if hasattr(document, "page_content") else str(document)

            source_info = {
                "document_id": metadata.get("chunk_id") or metadata.get("id"),
                "file_id": metadata.get("file_id"),
                "file_name": metadata.get("file_name", "Unknown"),
                "content": content[:500],  # Truncate for tracking
                "full_content": content,
                "page_number": metadata.get("page_number"),
                "chunk_index": metadata.get("chunk_index"),
                "start_time": metadata.get("start_time"),
                "end_time": metadata.get("end_time"),
                "mime_type": metadata.get("mime_type"),
                "retrieval_score": retrieval_score,
                "retrieval_method": retrieval_method,
                "metadata": metadata,
            }

            self.tracked_sources.append(source_info)

            logger.debug(
                "Tracked source document",
                document_id=source_info["document_id"],
                file_name=source_info["file_name"],
            )

            return source_info

        except Exception as e:
            logger.error("Failed to track document", error=str(e))
            return {}

    def track_chunk(
        self,
        chunk: DocumentChunk,
        retrieval_score: Optional[float] = None,
        retrieval_method: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Track a DocumentChunk as source.

        Args:
            chunk: DocumentChunk object
            retrieval_score: Optional retrieval relevance score
            retrieval_method: Optional retrieval method used

        Returns:
            Dictionary with tracked source information
        """
        try:
            source_info = {
                "document_id": chunk.chunk_id,
                "file_id": chunk.file_id,
                "file_name": chunk.metadata.get("file_name", "Unknown"),
                "content": chunk.content[:500],
                "full_content": chunk.content,
                "page_number": chunk.metadata.get("page_number"),
                "chunk_index": chunk.metadata.get("chunk_index"),
                "start_time": chunk.metadata.get("start_time"),
                "end_time": chunk.metadata.get("end_time"),
                "mime_type": chunk.metadata.get("mime_type"),
                "retrieval_score": retrieval_score,
                "retrieval_method": retrieval_method,
                "metadata": chunk.metadata,
            }

            self.tracked_sources.append(source_info)

            return source_info

        except Exception as e:
            logger.error("Failed to track chunk", error=str(e))
            return {}

    def get_tracked_sources(self) -> List[Dict[str, Any]]:
        """Get all tracked source documents.

        Returns:
            List of tracked source dictionaries
        """
        return self.tracked_sources.copy()

    def clear_tracking(self) -> None:
        """Clear all tracked sources."""
        self.tracked_sources.clear()
        logger.debug("Cleared source tracking")

    def calculate_source_reliability(
        self,
        source: Dict[str, Any],
    ) -> float:
        """Calculate reliability score for a source.

        Args:
            source: Source dictionary

        Returns:
            Reliability score (0.0 to 1.0)
        """
        try:
            reliability = 0.5  # Base reliability

            # Boost for high retrieval score
            retrieval_score = source.get("retrieval_score", 0.0)
            if retrieval_score > 0.8:
                reliability += 0.2
            elif retrieval_score > 0.6:
                reliability += 0.1

            # Boost for complete metadata
            if source.get("file_name") and source.get("file_name") != "Unknown":
                reliability += 0.1

            if source.get("page_number") is not None:
                reliability += 0.1

            # Boost for specific file types
            mime_type = source.get("mime_type", "")
            if mime_type == "application/pdf":
                reliability += 0.05  # PDFs are generally reliable

            # Normalize
            reliability = min(reliability, 1.0)

            return reliability

        except Exception as e:
            logger.error("Failed to calculate reliability", error=str(e))
            return 0.5

    def preserve_metadata(
        self,
        document: Document,
    ) -> Dict[str, Any]:
        """Extract and preserve metadata from document.

        Args:
            document: LangChain Document object

        Returns:
            Dictionary with preserved metadata
        """
        try:
            if not hasattr(document, "metadata"):
                return {}

            metadata = document.metadata.copy()

            # Ensure critical fields are preserved
            preserved = {
                "file_id": metadata.get("file_id"),
                "file_name": metadata.get("file_name"),
                "chunk_id": metadata.get("chunk_id") or metadata.get("id"),
                "page_number": metadata.get("page_number"),
                "chunk_index": metadata.get("chunk_index"),
                "start_time": metadata.get("start_time"),
                "end_time": metadata.get("end_time"),
                "mime_type": metadata.get("mime_type"),
                "source": metadata.get("source"),
                "file_path": metadata.get("file_path"),
                "created_at": metadata.get("created_at"),
                "modified_at": metadata.get("modified_at"),
            }

            # Remove None values
            preserved = {k: v for k, v in preserved.items() if v is not None}

            return preserved

        except Exception as e:
            logger.error("Failed to preserve metadata", error=str(e))
            return {}

    def enrich_document_metadata(
        self,
        document: Document,
        additional_metadata: Dict[str, Any],
    ) -> Document:
        """Enrich document with additional metadata.

        Args:
            document: LangChain Document object
            additional_metadata: Additional metadata to add

        Returns:
            Document with enriched metadata
        """
        try:
            if not LANGCHAIN_AVAILABLE:
                raise ProcessingError("LangChain is not installed")

            # Create new document with enriched metadata
            existing_metadata = document.metadata if hasattr(document, "metadata") else {}
            enriched_metadata = {**existing_metadata, **additional_metadata}

            content = document.page_content if hasattr(document, "page_content") else str(document)

            return Document(
                page_content=content,
                metadata=enriched_metadata,
            )

        except Exception as e:
            logger.error("Failed to enrich metadata", error=str(e))
            return document

