"""Custom LangChain callbacks for citation collection."""

from typing import List, Dict, Any, Optional
import structlog

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.schema import LLMResult
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler
        from langchain.schema import LLMResult
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        BaseCallbackHandler = None
        LLMResult = None

from app.core.exceptions import ProcessingError
from app.rag.source_tracking import SourceTracker

logger = structlog.get_logger(__name__)


class CitationCallbackHandler(BaseCallbackHandler):
    """Custom callback handler for collecting citations during chain execution."""

    def __init__(self, source_tracker: Optional[SourceTracker] = None):
        """Initialize citation callback handler.

        Args:
            source_tracker: Optional SourceTracker instance
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        super().__init__()
        self.source_tracker = source_tracker or SourceTracker()
        self.citations: List[Dict[str, Any]] = []
        self.source_documents: List[Any] = []
        self.logger = structlog.get_logger(__name__)

    def on_retriever_end(
        self,
        documents: List[Any],
        **kwargs: Any,
    ) -> None:
        """Handle retriever end event.

        Args:
            documents: Retrieved documents
            **kwargs: Additional arguments
        """
        try:
            self.source_documents.extend(documents)

            # Track each document
            for doc in documents:
                if hasattr(doc, "metadata"):
                    self.source_tracker.track_document(
                        document=doc,
                        retrieval_method=kwargs.get("retrieval_method", "unknown"),
                    )

            logger.debug(
                "Retriever completed",
                document_count=len(documents),
            )

        except Exception as e:
            logger.error("Error in retriever callback", error=str(e))

    def on_chain_end(
        self,
        outputs: Dict[str, Any],
        **kwargs: Any,
    ) -> None:
        """Handle chain end event.

        Args:
            outputs: Chain outputs
            **kwargs: Additional arguments
        """
        try:
            # Extract source documents from outputs
            if "source_documents" in outputs:
                source_docs = outputs["source_documents"]
                self.source_documents.extend(source_docs)

                # Generate citations from source documents
                for doc in source_docs:
                    citation = self._generate_citation_from_document(doc)
                    if citation:
                        self.citations.append(citation)

            logger.debug(
                "Chain completed",
                citation_count=len(self.citations),
            )

        except Exception as e:
            logger.error("Error in chain callback", error=str(e))

    def _generate_citation_from_document(
        self,
        document: Any,
    ) -> Optional[Dict[str, Any]]:
        """Generate citation from document.

        Args:
            document: Document object

        Returns:
            Citation dictionary or None
        """
        try:
            if hasattr(document, "metadata"):
                metadata = document.metadata
            else:
                return None

            citation = {
                "file_id": metadata.get("file_id"),
                "file_name": metadata.get("file_name", "Unknown"),
                "chunk_id": metadata.get("chunk_id") or metadata.get("id"),
                "page_number": metadata.get("page_number"),
                "chunk_index": metadata.get("chunk_index"),
                "start_time": metadata.get("start_time"),
                "end_time": metadata.get("end_time"),
                "content_preview": (
                    document.page_content[:200]
                    if hasattr(document, "page_content")
                    else str(document)[:200]
                ),
                "metadata": metadata,
            }

            # Calculate reliability
            source_info = {
                "retrieval_score": metadata.get("retrieval_score"),
                "file_name": citation["file_name"],
                "page_number": citation["page_number"],
                "mime_type": metadata.get("mime_type"),
            }
            citation["reliability"] = self.source_tracker.calculate_source_reliability(
                source_info
            )

            return citation

        except Exception as e:
            logger.error("Failed to generate citation", error=str(e))
            return None

    def get_citations(self) -> List[Dict[str, Any]]:
        """Get collected citations.

        Returns:
            List of citation dictionaries
        """
        return self.citations.copy()

    def get_source_documents(self) -> List[Any]:
        """Get collected source documents.

        Returns:
            List of source documents
        """
        return self.source_documents.copy()

    def clear_citations(self) -> None:
        """Clear collected citations and source documents."""
        self.citations.clear()
        self.source_documents.clear()
        self.source_tracker.clear_tracking()
        logger.debug("Cleared citations")


class SourceTrackingCallbackHandler(BaseCallbackHandler):
    """Callback handler for tracking source documents throughout chain execution."""

    def __init__(self):
        """Initialize source tracking callback handler."""
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        super().__init__()
        self.source_tracker = SourceTracker()
        self.tracked_documents: List[Dict[str, Any]] = []
        self.logger = structlog.get_logger(__name__)

    def on_retriever_start(
        self,
        query: str,
        **kwargs: Any,
    ) -> None:
        """Handle retriever start event.

        Args:
            query: Search query
            **kwargs: Additional arguments
        """
        try:
            logger.debug("Retriever started", query_length=len(query))
            self.source_tracker.clear_tracking()

        except Exception as e:
            logger.error("Error in retriever start callback", error=str(e))

    def on_retriever_end(
        self,
        documents: List[Any],
        **kwargs: Any,
    ) -> None:
        """Handle retriever end event.

        Args:
            documents: Retrieved documents
            **kwargs: Additional arguments
        """
        try:
            for doc in documents:
                if hasattr(doc, "metadata"):
                    source_info = self.source_tracker.track_document(
                        document=doc,
                        retrieval_method=kwargs.get("retrieval_method", "unknown"),
                    )
                    if source_info:
                        self.tracked_documents.append(source_info)

            logger.debug(
                "Retriever completed",
                document_count=len(documents),
                tracked_count=len(self.tracked_documents),
            )

        except Exception as e:
            logger.error("Error in retriever end callback", error=str(e))

    def get_tracked_sources(self) -> List[Dict[str, Any]]:
        """Get tracked source documents.

        Returns:
            List of tracked source dictionaries
        """
        return self.tracked_documents.copy()

    def clear_tracking(self) -> None:
        """Clear tracked documents."""
        self.tracked_documents.clear()
        self.source_tracker.clear_tracking()

