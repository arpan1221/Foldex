"""Debug service for diagnosing RAG pipeline issues."""

import time
from typing import Dict, Any, List, Optional
import structlog

logger = structlog.get_logger(__name__)


class DebugMetrics:
    """Collects and stores debug metrics for a single query."""

    def __init__(self, query: str):
        """Initialize debug metrics collector.

        Args:
            query: Original user query
        """
        self.query = query
        self.query_embedding: Optional[List[float]] = None
        self.retrieval_start_time: Optional[float] = None
        self.retrieval_end_time: Optional[float] = None
        self.retrieved_chunks: List[Dict[str, Any]] = []
        self.final_context: Optional[str] = None
        self.context_token_count: Optional[int] = None
        self.llm_start_time: Optional[float] = None
        self.llm_end_time: Optional[float] = None
        self.llm_raw_response: Optional[str] = None
        self.llm_cleaned_response: Optional[str] = None
        self.average_similarity_score: Optional[float] = None
        self.error: Optional[str] = None

    def start_retrieval(self) -> None:
        """Mark retrieval start time."""
        self.retrieval_start_time = time.time()

    def end_retrieval(self) -> None:
        """Mark retrieval end time."""
        self.retrieval_end_time = time.time()

    def set_query_embedding(self, embedding: List[float]) -> None:
        """Store query embedding.

        Args:
            embedding: Query embedding vector
        """
        self.query_embedding = embedding

    def add_retrieved_chunk(
        self,
        chunk: Any,
        score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add a retrieved chunk with metadata.

        Args:
            chunk: Retrieved document chunk
            score: Similarity score (if available)
            metadata: Additional metadata
        """
        # Extract content
        if hasattr(chunk, "page_content"):
            content = chunk.page_content
            chunk_metadata = chunk.metadata if hasattr(chunk, "metadata") else {}
        elif hasattr(chunk, "content"):
            content = chunk.content
            chunk_metadata = chunk.metadata if hasattr(chunk, "metadata") else {}
        else:
            content = str(chunk)
            chunk_metadata = {}

        # Merge metadata
        if metadata:
            chunk_metadata.update(metadata)

        chunk_info = {
            "content": content[:500],  # First 500 chars
            "content_length": len(content),
            "score": score,
            "file_name": chunk_metadata.get("file_name", "Unknown"),
            "file_id": chunk_metadata.get("file_id"),
            "page_number": chunk_metadata.get("page_number"),
            "chunk_id": chunk_metadata.get("chunk_id"),
            "chunk_index": chunk_metadata.get("chunk_index"),
            "metadata": chunk_metadata,
        }

        self.retrieved_chunks.append(chunk_info)

    def set_final_context(self, context: str, token_count: Optional[int] = None) -> None:
        """Store final context sent to LLM.

        Args:
            context: Formatted context string
            token_count: Estimated token count
        """
        self.final_context = context
        self.context_token_count = token_count

    def start_llm(self) -> None:
        """Mark LLM inference start time."""
        self.llm_start_time = time.time()

    def end_llm(self) -> None:
        """Mark LLM inference end time."""
        self.llm_end_time = time.time()

    def set_llm_response(self, raw: str, cleaned: Optional[str] = None) -> None:
        """Store LLM response.

        Args:
            raw: Raw LLM response
            cleaned: Cleaned response (if available)
        """
        self.llm_raw_response = raw
        self.llm_cleaned_response = cleaned

    def calculate_average_score(self) -> None:
        """Calculate average similarity score from retrieved chunks."""
        scores = [chunk["score"] for chunk in self.retrieved_chunks if chunk["score"] is not None]
        if scores:
            self.average_similarity_score = sum(scores) / len(scores)

    def set_error(self, error: str) -> None:
        """Record error.

        Args:
            error: Error message
        """
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for API response.

        Returns:
            Dictionary with all debug metrics
        """
        retrieval_time = None
        if self.retrieval_start_time and self.retrieval_end_time:
            retrieval_time = self.retrieval_end_time - self.retrieval_start_time

        llm_time = None
        if self.llm_start_time and self.llm_end_time:
            llm_time = self.llm_end_time - self.llm_start_time

        # Get first 5 dimensions of embedding
        embedding_preview = None
        if self.query_embedding:
            embedding_preview = self.query_embedding[:5]

        return {
            "query": self.query,
            "query_embedding_preview": embedding_preview,
            "retrieval_metrics": {
                "time_seconds": retrieval_time,
                "chunk_count": len(self.retrieved_chunks),
                "average_similarity_score": self.average_similarity_score,
            },
            "retrieved_chunks": self.retrieved_chunks,
            "context": {
                "text": self.final_context or "",
                "token_count": self.context_token_count,
                "length": len(self.final_context) if self.final_context else 0,
            },
            "llm_metrics": {
                "time_seconds": llm_time,
                "raw_response": self.llm_raw_response,
                "cleaned_response": self.llm_cleaned_response,
                "raw_length": len(self.llm_raw_response) if self.llm_raw_response else 0,
                "cleaned_length": len(self.llm_cleaned_response) if self.llm_cleaned_response else 0,
            },
            "error": self.error,
        }

    def log_summary(self) -> None:
        """Log summary of metrics for debugging."""
        retrieval_time = None
        if self.retrieval_start_time and self.retrieval_end_time:
            retrieval_time = self.retrieval_end_time - self.retrieval_start_time

        llm_time = None
        if self.llm_start_time and self.llm_end_time:
            llm_time = self.llm_end_time - self.llm_start_time

        logger.info(
            "Query debug metrics",
            query=self.query[:100],
            retrieval_time_seconds=retrieval_time,
            llm_time_seconds=llm_time,
            chunk_count=len(self.retrieved_chunks),
            average_score=self.average_similarity_score,
            context_length=len(self.final_context) if self.final_context else 0,
            context_tokens=self.context_token_count,
            error=self.error,
        )

