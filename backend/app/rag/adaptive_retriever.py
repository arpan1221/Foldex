
"""Adaptive retrieval system that routes queries to appropriate strategies.

Routes queries to different retrieval strategies based on query classification:
- FACTUAL_SPECIFIC: Precision-focused, strict filtering
- RELATIONSHIP: Diversity-focused, cross-document sampling
- COMPARISON: Balanced retrieval for two entities
- ENTITY_SEARCH: Exhaustive coverage, extreme diversity
- FACTUAL_GENERAL: Balanced relevance and diversity
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass
import structlog

from app.rag.query_classifier import QueryUnderstanding, QueryType
from app.rag.vector_store import LangChainVectorStore
from app.rag.smart_reranker import SmartReranker, get_smart_reranker
from app.config.settings import settings
from langchain_core.documents import Document

logger = structlog.get_logger(__name__)


@dataclass
class RetrievalResult:
    """Standardized retrieval result format."""
    
    chunks: List[Dict[str, Any]]
    metadata: Dict[str, Any]


class AdaptiveRetriever:
    """
    Adaptive retriever that routes queries to appropriate strategies
    based on query classification.
    
    Each strategy is optimized for its query type:
    - Factual: Precision over diversity
    - Relationship: Diversity over precision
    - Comparison: Balanced representation
    - Entity Search: Exhaustive coverage
    - General: Balanced approach
    """
    
    def __init__(
        self,
        vector_store: LangChainVectorStore,
        folder_id: Optional[str] = None,
        reranker: Optional[SmartReranker] = None,
        use_smart_reranking: Optional[bool] = None,
    ):
        """Initialize adaptive retriever.
        
        Args:
            vector_store: LangChain vector store instance
            folder_id: Optional folder ID for filtering
            reranker: Optional smart reranker instance
            use_smart_reranking: Whether to use smart reranking (default: from settings)
        """
        self.vector_store = vector_store
        self.folder_id = folder_id
        self.use_smart_reranking = use_smart_reranking if use_smart_reranking is not None else settings.USE_SMART_RERANKING
        self.reranker = reranker or (get_smart_reranker() if self.use_smart_reranking else None)
    
    async def retrieve(
        self,
        query: str,
        understanding: QueryUnderstanding,
        top_k: int = 10,
        folder_id: Optional[str] = None,
    ) -> RetrievalResult:
        """
        Main entry point for adaptive retrieval.
        
        Routes query to appropriate strategy based on query type.
        
        Args:
            query: User query string
            understanding: QueryUnderstanding from classifier
            top_k: Number of chunks to return
            folder_id: Optional folder ID (overrides instance folder_id)
            
        Returns:
            RetrievalResult with standardized format
        """
        folder_id = folder_id or self.folder_id
        
        query_type = understanding.query_type
        
        logger.info(
            "🧠 Adaptive retrieval routing",
            query_type=query_type.value,
            confidence=f"{understanding.confidence:.0%}",
            query=query[:100],
        )
        
        # Route to appropriate strategy
        if query_type == QueryType.FACTUAL_SPECIFIC:
            result = await self._factual_retrieval(query, understanding, top_k, folder_id)
        elif query_type == QueryType.RELATIONSHIP:
            result = await self._relationship_retrieval(query, understanding, top_k, folder_id)
        elif query_type == QueryType.COMPARISON:
            result = await self._comparison_retrieval(query, understanding, top_k, folder_id)
        elif query_type == QueryType.ENTITY_SEARCH:
            result = await self._entity_search_retrieval(query, understanding, top_k, folder_id)
        elif query_type == QueryType.FACTUAL_GENERAL:
            result = await self._general_retrieval(query, understanding, top_k, folder_id)
        else:
            # Default to general retrieval
            logger.warning(f"Unknown query type {query_type}, using general retrieval")
            result = await self._general_retrieval(query, understanding, top_k, folder_id)
        
        return result
    
    async def _factual_retrieval(
        self,
        query: str,
        understanding: QueryUnderstanding,
        top_k: int,
        folder_id: Optional[str],
    ) -> RetrievalResult:
        """
        Factual retrieval: Precision-focused, strict filtering.
        
        Strategy:
        - Apply strict content-type filter (file_type or file_id)
        - Use semantic search only (no diversity needed)
        - Return top K by relevance score
        - Precision over diversity
        """
        logger.info("🔍 Strategy: factual_retrieval (precision-focused)")
        
        # Build filter with strict content-type filtering
        filter_clause = self._build_filter(folder_id, understanding, apply_content_filter=True)
        
        if filter_clause:
            logger.info(f"   Filter applied: {filter_clause}")
        else:
            logger.info("   No filter applied")
        
        # Retrieve top K (precision-focused, no over-retrieval needed)
        results = await self.vector_store.similarity_search_with_score(
            query=query,
            k=top_k,
            filter=filter_clause,
        )
        
        # Format results
        chunks = self._format_chunks(results)
        
        # Get unique files
        unique_files = {chunk["file_id"] for chunk in chunks if chunk.get("file_id")}
        
        logger.info(
            "✅ Factual retrieval complete",
            chunks_returned=len(chunks),
            files_retrieved_from=len(unique_files),
        )
        
        return RetrievalResult(
            chunks=chunks,
            metadata={
                "strategy_used": "factual_filtered",
                "query_type": understanding.query_type.value,
                "files_retrieved_from": len(unique_files),
                "diversity_enforced": False,
                "content_filter_applied": filter_clause.get("file_type") or filter_clause.get("file_id") if filter_clause else None,
            },
        )
    
    async def _relationship_retrieval(
        self,
        query: str,
        understanding: QueryUnderstanding,
        top_k: int,
        folder_id: Optional[str],
    ) -> RetrievalResult:
        """
        Relationship retrieval: Diversity-focused, cross-document sampling.
        
        Strategy:
        - NO content-type filtering (need cross-document view)
        - Over-retrieve (top_k * 4 candidates)
        - Enforce strict diversity: max 2 chunks per file
        - Sample from as many different files as possible
        - Diversity over precision
        """
        logger.info("🔍 Strategy: relationship_retrieval (diversity-focused)")
        logger.info("   No content-type filter (need cross-document view)")
        
        # Build filter WITHOUT content-type filtering
        filter_clause = self._build_filter(folder_id, understanding, apply_content_filter=False)
        
        # Over-retrieve to get diverse candidates
        over_retrieve_k = top_k * 4
        logger.info(f"   Over-retrieving: {over_retrieve_k} candidates")
        
        results = await self.vector_store.similarity_search_with_score(
            query=query,
            k=over_retrieve_k,
            filter=filter_clause,
        )
        
        # Format candidates for reranking
        candidate_chunks = self._format_chunks(results)
        
        # Apply smart reranking (diversity-boosting) before diversity enforcement
        if self.reranker and self.use_smart_reranking:
            logger.info("   Applying diversity-boosting reranking")
            candidate_chunks = self.reranker.rerank(candidate_chunks, understanding.query_type, understanding)
            # Convert back to (Document, score) format for diversity enforcement
            diverse_results = [
                (Document(page_content=chunk["content"], metadata=chunk["metadata"]), chunk["score"])
                for chunk in candidate_chunks
            ]
        else:
            diverse_results = results
        
        # Enforce strict diversity: max 2 chunks per file
        logger.info("   Enforcing diversity: max 2 per file")
        diverse_results = self._enforce_diversity(diverse_results, max_per_file=2, target_count=top_k)
        
        # Format results
        chunks = self._format_chunks(diverse_results)
        
        # Analyze diversity
        file_counts = self._count_chunks_per_file(chunks)
        unique_files = len(file_counts)
        
        logger.info(
            "✅ Relationship retrieval complete",
            chunks_returned=len(chunks),
            files_retrieved_from=unique_files,
            file_distribution=file_counts,
        )
        
        return RetrievalResult(
            chunks=chunks,
            metadata={
                "strategy_used": "relationship_diverse",
                "query_type": understanding.query_type.value,
                "files_retrieved_from": unique_files,
                "diversity_enforced": True,
                "max_chunks_per_file": 2,
                "file_distribution": file_counts,
                "content_filter_applied": None,  # Explicitly None for relationship queries
            },
        )
    
    async def _comparison_retrieval(
        self,
        query: str,
        understanding: QueryUnderstanding,
        top_k: int,
        folder_id: Optional[str],
    ) -> RetrievalResult:
        """
        Comparison retrieval: Balanced retrieval for two entities.
        
        Strategy:
        - Extract the two entities being compared
        - Retrieve top_k/2 for entity A
        - Retrieve top_k/2 for entity B
        - Interleave results to show both perspectives
        - Balanced representation
        """
        logger.info("🔍 Strategy: comparison_retrieval (balanced)")
        
        # Extract entities from understanding
        entities = understanding.entities or []
        file_refs = understanding.file_references or []
        
        # Use file references if available, otherwise entities
        if len(file_refs) >= 2:
            entity_a, entity_b = file_refs[0], file_refs[1]
            logger.info(f"   Comparing files: {entity_a} vs {entity_b}")
            
            # Build base filter
            base_filter = self._build_filter(folder_id, understanding, apply_content_filter=False)
            if not base_filter:
                base_filter = {}
            
            # Retrieve for entity A
            filter_a = base_filter.copy()
            filter_a["file_name"] = {"$eq": entity_a}
            
            results_a = await self.vector_store.similarity_search_with_score(
                query=query,  # Use original query for better semantic matching
                k=top_k // 2 + 1,  # Get a bit more to ensure we have enough
                filter=filter_a,
            )
            
            # Retrieve for entity B
            filter_b = base_filter.copy()
            filter_b["file_name"] = {"$eq": entity_b}
            
            results_b = await self.vector_store.similarity_search_with_score(
                query=query,  # Use original query for better semantic matching
                k=top_k // 2 + 1,  # Get a bit more to ensure we have enough
                filter=filter_b,
            )
            
            # Interleave results
            interleaved = self._interleave_results(results_a, results_b)
            # Limit to top_k after interleaving
            chunks = self._format_chunks(interleaved[:top_k])
            
        elif len(entities) >= 2:
            # Use entities if file references not available
            entity_a, entity_b = entities[0], entities[1]
            logger.info(f"   Comparing entities: {entity_a} vs {entity_b}")
            
            # Retrieve for both entities using original query
            filter_clause = self._build_filter(folder_id, understanding, apply_content_filter=False)
            
            results_a = await self.vector_store.similarity_search_with_score(
                query=f"{query} {entity_a}",
                k=top_k // 2,
                filter=filter_clause,
            )
            
            results_b = await self.vector_store.similarity_search_with_score(
                query=f"{query} {entity_b}",
                k=top_k // 2,
                filter=filter_clause,
            )
            
            # Interleave results
            interleaved = self._interleave_results(results_a, results_b)
            chunks = self._format_chunks(interleaved)
        else:
            # Fallback: regular retrieval with diversity
            logger.warning("   Could not extract two entities, using regular retrieval")
            filter_clause = self._build_filter(folder_id, understanding, apply_content_filter=False)
            results = await self.vector_store.similarity_search_with_score(
                query=query,
                k=top_k * 2,
                filter=filter_clause,
            )
            
            # Apply smart reranking if enabled
            candidate_chunks = self._format_chunks(results)
            if self.reranker and self.use_smart_reranking:
                candidate_chunks = self.reranker.rerank(candidate_chunks, understanding.query_type, understanding)
                diverse_results = [
                    (Document(page_content=chunk["content"], metadata=chunk["metadata"]), chunk["score"])
                    for chunk in candidate_chunks
                ]
            else:
                diverse_results = results
            
            diverse_results = self._enforce_diversity(diverse_results, max_per_file=top_k // 2, target_count=top_k)
            chunks = self._format_chunks(diverse_results)
        
        # Analyze balance
        file_counts = self._count_chunks_per_file(chunks)
        unique_files = len(file_counts)
        
        logger.info(
            "✅ Comparison retrieval complete",
            chunks_returned=len(chunks),
            files_retrieved_from=unique_files,
            file_distribution=file_counts,
        )
        
        return RetrievalResult(
            chunks=chunks,
            metadata={
                "strategy_used": "comparison_balanced",
                "query_type": understanding.query_type.value,
                "files_retrieved_from": unique_files,
                "diversity_enforced": True,
                "file_distribution": file_counts,
            },
        )
    
    async def _entity_search_retrieval(
        self,
        query: str,
        understanding: QueryUnderstanding,
        top_k: int,
        folder_id: Optional[str],
    ) -> RetrievalResult:
        """
        Entity search retrieval: Exhaustive coverage, extreme diversity.
        
        Strategy:
        - Search for entity name as query (not original query)
        - Enforce extreme diversity: max 1 chunk per file
        - Exhaustive coverage over relevance
        - Goal: find all locations where entity appears
        """
        logger.info("🔍 Strategy: entity_search_retrieval (exhaustive)")
        
        # Extract entity from query or understanding
        entities = understanding.entities or []
        entity_query = entities[0] if entities else query
        
        logger.info(f"   Searching for entity: {entity_query}")
        logger.info("   Enforcing extreme diversity: max 1 per file")
        
        # Build filter without content-type filtering
        filter_clause = self._build_filter(folder_id, understanding, apply_content_filter=False)
        
        # Over-retrieve significantly to find all occurrences
        over_retrieve_k = top_k * 5
        logger.info(f"   Over-retrieving: {over_retrieve_k} candidates for exhaustive coverage")
        
        results = await self.vector_store.similarity_search_with_score(
            query=entity_query,  # Use entity name as query
            k=over_retrieve_k,
            filter=filter_clause,
        )
        
        # Format candidates for reranking
        candidate_chunks = self._format_chunks(results)
        
        # Apply entity boost reranking if enabled
        if self.reranker and self.use_smart_reranking:
            logger.info("   Applying entity boost reranking")
            candidate_chunks = self.reranker.rerank(candidate_chunks, understanding.query_type, understanding)
            # Convert back to (Document, score) format for diversity enforcement
            diverse_results = [
                (Document(page_content=chunk["content"], metadata=chunk["metadata"]), chunk["score"])
                for chunk in candidate_chunks
            ]
        else:
            diverse_results = results
        
        # Enforce extreme diversity: max 1 chunk per file
        diverse_results = self._enforce_diversity(diverse_results, max_per_file=1, target_count=top_k)
        
        # Format results
        chunks = self._format_chunks(diverse_results)
        
        # Analyze coverage
        file_counts = self._count_chunks_per_file(chunks)
        unique_files = len(file_counts)
        
        logger.info(
            "✅ Entity search retrieval complete",
            chunks_returned=len(chunks),
            files_retrieved_from=unique_files,
            file_distribution=file_counts,
        )
        
        return RetrievalResult(
            chunks=chunks,
            metadata={
                "strategy_used": "entity_search_exhaustive",
                "query_type": understanding.query_type.value,
                "files_retrieved_from": unique_files,
                "diversity_enforced": True,
                "max_chunks_per_file": 1,
                "file_distribution": file_counts,
            },
        )
    
    async def _general_retrieval(
        self,
        query: str,
        understanding: QueryUnderstanding,
        top_k: int,
        folder_id: Optional[str],
    ) -> RetrievalResult:
        """
        General retrieval: Balanced relevance and diversity.
        
        Strategy:
        - Over-retrieve (top_k * 2)
        - Enforce moderate diversity: max 2 chunks per file
        - Balance relevance and diversity
        - Representative sampling
        """
        logger.info("🔍 Strategy: general_retrieval (balanced)")
        
        # Build filter (may include content-type if specified)
        filter_clause = self._build_filter(folder_id, understanding, apply_content_filter=True)
        
        # Over-retrieve for diversity
        over_retrieve_k = top_k * 2
        logger.info(f"   Over-retrieving: {over_retrieve_k} candidates")
        logger.info("   Enforcing moderate diversity: max 2 per file")
        
        results = await self.vector_store.similarity_search_with_score(
            query=query,
            k=over_retrieve_k,
            filter=filter_clause,
        )
        
        # Format candidates for reranking
        candidate_chunks = self._format_chunks(results)
        
        # Apply smart reranking (balanced with diversity boost) before diversity enforcement
        if self.reranker and self.use_smart_reranking:
            logger.info("   Applying balanced reranking with diversity boost")
            candidate_chunks = self.reranker.rerank(candidate_chunks, understanding.query_type, understanding)
            # Convert back to (Document, score) format for diversity enforcement
            diverse_results = [
                (Document(page_content=chunk["content"], metadata=chunk["metadata"]), chunk["score"])
                for chunk in candidate_chunks
            ]
        else:
            diverse_results = results
        
        # Enforce moderate diversity: max 2 chunks per file
        diverse_results = self._enforce_diversity(diverse_results, max_per_file=2, target_count=top_k)
        
        # Format results
        chunks = self._format_chunks(diverse_results)
        
        # Analyze diversity
        file_counts = self._count_chunks_per_file(chunks)
        unique_files = len(file_counts)
        
        logger.info(
            "✅ General retrieval complete",
            chunks_returned=len(chunks),
            files_retrieved_from=unique_files,
            file_distribution=file_counts,
        )
        
        return RetrievalResult(
            chunks=chunks,
            metadata={
                "strategy_used": "general_balanced",
                "query_type": understanding.query_type.value,
                "files_retrieved_from": unique_files,
                "diversity_enforced": True,
                "max_chunks_per_file": 2,
                "file_distribution": file_counts,
                "content_filter_applied": filter_clause.get("file_type") if filter_clause else None,
            },
        )
    
    def _enforce_diversity(
        self,
        results: List[Tuple[Document, float]],
        max_per_file: int,
        target_count: int,
    ) -> List[Tuple[Document, float]]:
        """
        Enforce diversity by limiting chunks per file.
        
        Algorithm:
        1. Iterate through chunks in relevance order
        2. Track count per file_id
        3. Skip chunks from files that hit max_per_file limit
        4. Continue until target_count selected
        5. Return selected chunks
        
        Args:
            results: List of (Document, score) tuples in relevance order
            max_per_file: Maximum chunks allowed per file
            target_count: Target number of chunks to return
            
        Returns:
            List of diverse (Document, score) tuples
        """
        selected: List[Tuple[Document, float]] = []
        file_counts: Dict[str, int] = {}
        
        for doc, score in results:
            if len(selected) >= target_count:
                break
            
            file_id = doc.metadata.get("file_id") or doc.metadata.get("file_name", "unknown")
            current_count = file_counts.get(file_id, 0)
            
            if current_count < max_per_file:
                selected.append((doc, score))
                file_counts[file_id] = current_count + 1
        
        skipped_count = len(results) - len(selected)
        if skipped_count > 0:
            logger.debug(
                "Diversity enforcement",
                selected=len(selected),
                skipped=skipped_count,
                max_per_file=max_per_file,
                file_distribution=file_counts,
            )
        
        return selected
    
    def _build_filter(
        self,
        folder_id: Optional[str],
        understanding: QueryUnderstanding,
        apply_content_filter: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """
        Build ChromaDB filter clause.
        
        Logic:
        1. Start with base filter: {'folder_id': folder_id}
        2. If understanding.content_type exists and apply_content_filter is True, add file_type filter
        3. If understanding.file_references exists, add file_id filter
        4. If query_type is RELATIONSHIP, do NOT add content-type filter (handled by apply_content_filter)
        
        Args:
            folder_id: Folder ID for filtering
            understanding: Query understanding with content type and file references
            apply_content_filter: Whether to apply content-type filtering
            
        Returns:
            Combined where clause for ChromaDB or None
        """
        filter_clause: Dict[str, Any] = {}
        
        # Always filter by folder_id if provided
        if folder_id:
            filter_clause["folder_id"] = folder_id
        
        # Apply content-type filter only if requested
        if apply_content_filter and understanding.content_type:
            filter_clause["file_type"] = understanding.content_type
        
        # Apply file_id filter if file references exist
        if understanding.file_references:
            file_names = understanding.file_references
            if len(file_names) == 1:
                filter_clause["file_name"] = {"$eq": file_names[0]}
            else:
                filter_clause["file_name"] = {"$in": file_names}
        
        return filter_clause if filter_clause else None
    
    def _format_chunks(
        self,
        results: List[Tuple[Document, float]],
    ) -> List[Dict[str, Any]]:
        """
        Format retrieval results to standardized chunk format.
        
        Args:
            results: List of (Document, score) tuples
            
        Returns:
            List of formatted chunk dicts
        """
        chunks = []
        
        for doc, score in results:
            chunk = {
                "id": doc.metadata.get("chunk_id", ""),
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": score,
                "file_name": doc.metadata.get("file_name", "unknown"),
                "file_id": doc.metadata.get("file_id", doc.metadata.get("file_name", "unknown")),
            }
            chunks.append(chunk)
        
        return chunks
    
    def _count_chunks_per_file(self, chunks: List[Dict[str, Any]]) -> Dict[str, int]:
        """Count chunks per file for diversity analysis."""
        file_counts: Dict[str, int] = {}
        
        for chunk in chunks:
            file_id = chunk.get("file_id") or chunk.get("file_name", "unknown")
            file_counts[file_id] = file_counts.get(file_id, 0) + 1
        
        return file_counts
    
    def _interleave_results(
        self,
        results_a: List[Tuple[Document, float]],
        results_b: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """
        Interleave two result lists to show balanced representation.
        
        Args:
            results_a: First set of results
            results_b: Second set of results
            
        Returns:
            Interleaved list of (Document, score) tuples
        """
        interleaved = []
        max_len = max(len(results_a), len(results_b))
        
        for i in range(max_len):
            if i < len(results_a):
                interleaved.append(results_a[i])
            if i < len(results_b):
                interleaved.append(results_b[i])
        
        return interleaved

