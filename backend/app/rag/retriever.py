"""Hybrid retrieval logic combining semantic, keyword, and graph search."""

from typing import List
import structlog

from app.models.documents import DocumentChunk
from app.database.vector_store import VectorStore
from app.database.sqlite_manager import SQLiteManager
from app.knowledge_graph.graph_queries import GraphQueries

logger = structlog.get_logger(__name__)


class HybridRetriever:
    """Hybrid retriever combining multiple retrieval strategies."""

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        db: SQLiteManager | None = None,
        graph_queries: GraphQueries | None = None,
    ):
        """Initialize hybrid retriever.

        Args:
            vector_store: Vector store for semantic search
            db: Database for keyword search
            graph_queries: Graph query service
        """
        self.vector_store = vector_store or VectorStore()
        self.db = db or SQLiteManager()
        self.graph_queries = graph_queries or GraphQueries()

    async def retrieve(
        self,
        query_embedding: List[float],
        query_text: str,
        folder_id: str,
        k: int = 10,
    ) -> List[DocumentChunk]:
        """Retrieve relevant chunks using hybrid approach.

        Args:
            query_embedding: Query embedding vector
            query_text: Original query text
            folder_id: Folder ID to search
            k: Number of chunks to retrieve

        Returns:
            List of retrieved document chunks
        """
        try:
            logger.info("Retrieving chunks", query_length=len(query_text), k=k)

            # 1. Semantic similarity search (vector)
            semantic_chunks = await self.vector_store.similarity_search(
                query_embedding, folder_id, k=k
            )

            # 2. Keyword/BM25 search for exact matches
            keyword_chunks = await self.db.keyword_search(query_text, folder_id, k=k)

            # 3. Knowledge graph traversal for relationships
            graph_chunks = await self.graph_queries.traverse_related(
                query_text, folder_id, k=k
            )

            # Combine and deduplicate
            all_chunks = self._combine_results(
                semantic_chunks, keyword_chunks, graph_chunks
            )

            # Return top k
            return all_chunks[:k]
        except Exception as e:
            logger.error("Retrieval failed", error=str(e))
            raise

    def _combine_results(
        self,
        semantic: List[DocumentChunk],
        keyword: List[DocumentChunk],
        graph: List[DocumentChunk],
    ) -> List[DocumentChunk]:
        """Combine results from different retrieval methods.

        Args:
            semantic: Semantic search results
            keyword: Keyword search results
            graph: Graph traversal results

        Returns:
            Combined and deduplicated chunks
        """
        # Simple combination - can be improved with scoring
        seen = set()
        combined = []

        for chunk in semantic + keyword + graph:
            if chunk.chunk_id not in seen:
                seen.add(chunk.chunk_id)
                combined.append(chunk)

        return combined

