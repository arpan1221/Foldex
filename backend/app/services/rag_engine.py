"""RAG engine core logic."""

from typing import List, Dict, Optional
import structlog

from app.models.documents import DocumentChunk
from app.rag.embeddings import EmbeddingService
from app.rag.retriever import HybridRetriever
from app.rag.reranker import Reranker
from app.rag.llm_interface import LLMInterface

logger = structlog.get_logger(__name__)


class RAGEngine:
    """Main RAG engine orchestrating retrieval and generation."""

    def __init__(
        self,
        embedding_service: Optional[EmbeddingService] = None,
        retriever: Optional[HybridRetriever] = None,
        reranker: Optional[Reranker] = None,
        llm: Optional[LLMInterface] = None,
    ):
        """Initialize RAG engine.

        Args:
            embedding_service: Embedding service
            retriever: Hybrid retriever
            reranker: Reranker service
            llm: LLM interface
        """
        self.embedding_service = embedding_service or EmbeddingService()
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or Reranker()
        self.llm = llm or LLMInterface()

    async def query(
        self, query: str, folder_id: str, k: int = 10
    ) -> Dict:
        """Process a query using RAG.

        Args:
            query: User query
            folder_id: Folder ID to search
            k: Number of chunks to retrieve

        Returns:
            Dictionary with response and source chunks
        """
        try:
            logger.info("Processing RAG query", query_length=len(query), folder_id=folder_id)

            # Generate query embedding
            query_embedding = await self.embedding_service.embed_text(query)

            # Retrieve relevant chunks
            chunks = await self.retriever.retrieve(
                query_embedding, query, folder_id, k=k * 2
            )

            # Rerank chunks
            reranked_chunks = await self.reranker.rerank(query, chunks, top_k=k)

            # Generate response
            response = await self.llm.generate_response(query, reranked_chunks)

            return {
                "response": response["text"],
                "chunks": reranked_chunks,
                "citations": self._extract_citations(reranked_chunks),
            }
        except Exception as e:
            logger.error("RAG query failed", error=str(e))
            raise

    def _extract_citations(self, chunks: List[DocumentChunk]) -> List[Dict]:
        """Extract citation information from chunks.

        Args:
            chunks: Retrieved chunks

        Returns:
            List of citation dictionaries
        """
        citations = []
        for chunk in chunks:
            citations.append(
                {
                    "file_id": chunk.file_id,
                    "file_name": chunk.metadata.get("file_name", ""),
                    "chunk_id": chunk.chunk_id,
                    "page_number": chunk.metadata.get("page_number"),
                    "timestamp": chunk.metadata.get("timestamp"),
                    "confidence": chunk.metadata.get("confidence", 0.0),
                }
            )
        return citations

