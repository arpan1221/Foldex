"""Custom LangChain retrievers combining multiple retrieval strategies."""

from typing import List, Optional, Dict, Any
import structlog

try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from pydantic import Field
    
    try:
        # Try langchain-community first (recommended)
        from langchain_community.retrievers import BM25Retriever
        from langchain.retrievers import EnsembleRetriever
        from langchain.retrievers.multi_query import MultiQueryRetriever
        from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import LLMChainExtractor
    except ImportError:
        # Fallback to deprecated import
        try:
            from langchain.retrievers import EnsembleRetriever, BM25Retriever
            from langchain.retrievers.multi_query import MultiQueryRetriever
            from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
            from langchain.retrievers.document_compressors import LLMChainExtractor
        except ImportError:
            # These are optional
            EnsembleRetriever = None
            BM25Retriever = None
            MultiQueryRetriever = None
            ContextualCompressionRetriever = None
            LLMChainExtractor = None
    
    # Verify BaseRetriever is actually imported
    if BaseRetriever is None:
        raise ImportError("BaseRetriever is None")
    
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    # Fallback for older LangChain versions
    try:
        from langchain.schema import BaseRetriever, Document
        from langchain.callbacks.manager import CallbackManagerForRetrieverRun
        try:
            from langchain.retrievers import EnsembleRetriever, BM25Retriever
            from langchain.retrievers.multi_query import MultiQueryRetriever
            from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
            from langchain.retrievers.document_compressors import LLMChainExtractor
        except ImportError:
            EnsembleRetriever = None
            BM25Retriever = None
            MultiQueryRetriever = None
            ContextualCompressionRetriever = None
            LLMChainExtractor = None
        Field = None
        
        if BaseRetriever is None:
            raise ImportError("BaseRetriever is None")
        
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        BaseRetriever = None
        Document = None
        CallbackManagerForRetrieverRun = None
        EnsembleRetriever = None
        BM25Retriever = None
        MultiQueryRetriever = None
        ContextualCompressionRetriever = None
        LLMChainExtractor = None
        Field = None

from app.core.exceptions import ProcessingError
from app.rag.vector_store import LangChainVectorStore
from app.rag.content_type_detector import ContentTypeDetector
from app.rag.document_diversity import get_document_diversifier
from app.rag.mmr_retrieval import get_mmr_retriever
from app.rag.file_rebalancer import get_file_rebalancer
from app.rag.query_classifier import QueryClassifier, QueryType, get_query_classifier

logger = structlog.get_logger(__name__)

# Ensure BaseRetriever is available before defining classes
if not LANGCHAIN_AVAILABLE or BaseRetriever is None:
    raise ProcessingError(
        "LangChain is not installed or BaseRetriever is not available. "
        "Install with: pip install langchain langchain-core"
    )


class VectorStoreRetriever(BaseRetriever):
    """Custom LangChain retriever for vector similarity search."""
    
    # Declare fields as class variables for Pydantic
    vector_store: Any  # Use Any to avoid Pydantic validation issues
    k: int = 10
    search_kwargs: Dict[str, Any] = {}

    def __init__(
        self,
        vector_store: LangChainVectorStore,
        k: int = 10,
        search_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """Initialize vector store retriever.

        Args:
            vector_store: LangChain vector store instance
            k: Number of documents to retrieve
            search_kwargs: Additional search parameters
            **kwargs: Additional arguments passed to BaseRetriever
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        # Initialize Pydantic model properly
        init_values = {
            "vector_store": vector_store,
            "k": k,
            "search_kwargs": search_kwargs or {},
            **kwargs,
        }
        super().__init__(**init_values)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Get relevant documents for query.

        Args:
            query: Search query
            run_manager: Optional callback manager

        Returns:
            List of relevant Document objects
        """
        try:
            # Use vector store similarity search
            # Note: This is synchronous but vector_store methods are async
            # In practice, this should be called from async context
            import asyncio

            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, we need to use a different approach
                    # For now, use the vector_store's synchronous interface if available
                    results = self.vector_store.vector_store.similarity_search(
                        query=query, k=self.k, **self.search_kwargs
                    )
                else:
                    results = loop.run_until_complete(
                        self.vector_store.similarity_search(query=query, k=self.k)
                    )
            except RuntimeError:
                # No event loop, create new one
                results = asyncio.run(
                    self.vector_store.similarity_search(query=query, k=self.k)
                )

            logger.debug(
                "Vector store retrieval completed",
                query_length=len(query),
                result_count=len(results),
            )

            return results

        except Exception as e:
            logger.error(
                "Vector store retrieval failed",
                query=query[:50],
                error=str(e),
                exc_info=True,
            )
            return []

    async def aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Async version of get_relevant_documents.

        Args:
            query: Search query
            run_manager: Optional callback manager

        Returns:
            List of relevant Document objects
        """
        try:
            results = await self.vector_store.similarity_search(
                query=query, k=self.k, filter=self.search_kwargs.get("filter")
            )

            logger.debug(
                "Async vector store retrieval completed",
                query_length=len(query),
                result_count=len(results),
            )

            return results

        except Exception as e:
            logger.error(
                "Async vector store retrieval failed",
                query=query[:50],
                error=str(e),
                exc_info=True,
            )
            return []


class BM25KeywordRetriever(BaseRetriever):
    """Custom LangChain retriever for BM25 keyword search."""
    
    # Declare fields as class variables for Pydantic
    documents: List[Document] = []
    k: int = 10
    bm25_retriever: Any = None  # LangChain BM25Retriever instance

    def __init__(
        self,
        documents: List[Document],
        k: int = 10,
        **kwargs,
    ):
        """Initialize BM25 retriever.

        Args:
            documents: List of documents to search
            k: Number of documents to retrieve
            **kwargs: Additional arguments passed to BaseRetriever
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        # Initialize BM25 retriever first (before super().__init__)
        try:
            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = k
            logger.debug("Initialized BM25 retriever", document_count=len(documents))
        except Exception as e:
            logger.error("Failed to initialize BM25 retriever", error=str(e))
            raise ProcessingError(f"Failed to initialize BM25 retriever: {str(e)}") from e

        # Initialize Pydantic model properly with all fields
        init_values = {
            "documents": documents,
            "k": k,
            "bm25_retriever": bm25_retriever,
            **kwargs,
        }
        super().__init__(**init_values)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Get relevant documents using BM25.

        Args:
            query: Search query
            run_manager: Optional callback manager

        Returns:
            List of relevant Document objects
        """
        try:
            results = self.bm25_retriever.get_relevant_documents(query)

            logger.debug(
                "BM25 retrieval completed",
                query_length=len(query),
                result_count=len(results),
            )

            return results

        except Exception as e:
            logger.error(
                "BM25 retrieval failed",
                query=query[:50],
                error=str(e),
                exc_info=True,
            )
            return []

    async def aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Async version of get_relevant_documents.

        Args:
            query: Search query
            run_manager: Optional callback manager

        Returns:
            List of relevant Document objects
        """
        try:
            results = await self.bm25_retriever.aget_relevant_documents(query)

            logger.debug(
                "Async BM25 retrieval completed",
                query_length=len(query),
                result_count=len(results),
            )

            return results

        except Exception as e:
            logger.error(
                "Async BM25 retrieval failed",
                query=query[:50],
                error=str(e),
                exc_info=True,
            )
            return []


class HybridLangChainRetriever:
    """Factory for creating hybrid LangChain retrievers with multiple strategies."""

    @staticmethod
    def create_ensemble_retriever(
        vector_store: LangChainVectorStore,
        documents: List[Document],
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        k: int = 10,
    ) -> EnsembleRetriever:
        """Create ensemble retriever combining vector and BM25 search.

        Args:
            vector_store: LangChain vector store
            documents: List of documents for BM25
            vector_weight: Weight for vector search results
            bm25_weight: Weight for BM25 search results
            k: Number of documents to retrieve

        Returns:
            EnsembleRetriever instance

        Raises:
            ProcessingError: If LangChain is not available or creation fails
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        try:
            # Create vector retriever
            vector_retriever = VectorStoreRetriever(
                vector_store=vector_store,
                k=k,
            )

            # Create BM25 retriever
            bm25_retriever = BM25KeywordRetriever(
                documents=documents,
                k=k,
            )

            # Create ensemble retriever
            ensemble = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[vector_weight, bm25_weight],
            )

            logger.info(
                "Created ensemble retriever",
                vector_weight=vector_weight,
                bm25_weight=bm25_weight,
                k=k,
            )

            return ensemble

        except Exception as e:
            logger.error(
                "Failed to create ensemble retriever",
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(f"Failed to create ensemble retriever: {str(e)}") from e

    @staticmethod
    def create_multi_query_retriever(
        base_retriever: BaseRetriever,
        llm: Any,
        k: int = 10,
    ) -> MultiQueryRetriever:
        """Create multi-query retriever for query expansion.

        Args:
            base_retriever: Base retriever to use
            llm: LLM for query generation
            k: Number of documents to retrieve

        Returns:
            MultiQueryRetriever instance

        Raises:
            ProcessingError: If creation fails
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        try:
            multi_query = MultiQueryRetriever.from_llm(
                retriever=base_retriever,
                llm=llm,
            )

            logger.info("Created multi-query retriever", k=k)

            return multi_query

        except Exception as e:
            logger.error(
                "Failed to create multi-query retriever",
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(
                f"Failed to create multi-query retriever: {str(e)}"
            ) from e

    @staticmethod
    def create_compression_retriever(
        base_retriever: BaseRetriever,
        llm: Any,
        k: int = 10,
    ) -> ContextualCompressionRetriever:
        """Create contextual compression retriever for relevance filtering.

        Args:
            base_retriever: Base retriever to use
            llm: LLM for compression/extraction
            k: Number of documents to retrieve

        Returns:
            ContextualCompressionRetriever instance

        Raises:
            ProcessingError: If creation fails
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        try:
            # Create LLM chain extractor for compression
            compressor = LLMChainExtractor.from_llm(llm)

            # Create compression retriever
            compression_retriever = ContextualCompressionRetriever(
                base_compressor=compressor,
                base_retriever=base_retriever,
            )

            logger.info("Created compression retriever", k=k)

            return compression_retriever

        except Exception as e:
            logger.error(
                "Failed to create compression retriever",
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(
                f"Failed to create compression retriever: {str(e)}"
            ) from e

    @staticmethod
    def create_hybrid_retriever(
        vector_store: LangChainVectorStore,
        documents: List[Document],
        llm: Any,
        use_multi_query: bool = True,
        use_compression: bool = True,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        k: int = 10,
    ) -> BaseRetriever:
        """Create comprehensive hybrid retriever with all strategies.

        Args:
            vector_store: LangChain vector store
            documents: List of documents for BM25
            llm: LLM for query expansion and compression
            use_multi_query: Whether to use multi-query expansion
            use_compression: Whether to use contextual compression
            vector_weight: Weight for vector search
            bm25_weight: Weight for BM25 search
            k: Number of documents to retrieve

        Returns:
            BaseRetriever instance (with all enhancements applied)

        Raises:
            ProcessingError: If creation fails
        """
        try:
            # PERFORMANCE: Use vector-only retriever if BM25 is disabled
            if bm25_weight == 0.0:
                logger.info("Using vector-only retriever (BM25 disabled for performance)")
                base_retriever = VectorStoreRetriever(
                    vector_store=vector_store,
                    k=k,
                )
            else:
                # Start with ensemble retriever
                base_retriever = HybridLangChainRetriever.create_ensemble_retriever(
                    vector_store=vector_store,
                    documents=documents,
                    vector_weight=vector_weight,
                    bm25_weight=bm25_weight,
                    k=k,
                )

            # Apply multi-query expansion if enabled
            if use_multi_query:
                base_retriever = HybridLangChainRetriever.create_multi_query_retriever(
                    base_retriever=base_retriever,
                    llm=llm,
                    k=k,
                )

            # Apply compression if enabled
            if use_compression:
                base_retriever = HybridLangChainRetriever.create_compression_retriever(
                    base_retriever=base_retriever,
                    llm=llm,
                    k=k,
                )

            logger.info(
                "Created hybrid retriever",
                use_multi_query=use_multi_query,
                use_compression=use_compression,
                vector_only=bm25_weight == 0.0,
                k=k,
            )

            return base_retriever

        except Exception as e:
            logger.error(
                "Failed to create hybrid retriever",
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(f"Failed to create hybrid retriever: {str(e)}") from e


class AdaptiveRetriever(BaseRetriever):
    """Adaptive retriever that chooses strategy based on query type.
    
    Automatically selects the best retrieval strategy:
    - exact_match: BM25 for keyword matching
    - conceptual: Vector search for semantic understanding
    - complex: Hybrid search with reranking
    
    Supports folder-specific filtering to ensure queries only return
    results from the current conversation's folder.
    """
    
    # Declare fields as class variables for Pydantic
    vector_store: Any
    documents: List[Document] = []
    reranker: Optional[Any] = None
    k: int = 10
    vector_weight: float = 0.7
    bm25_weight: float = 0.3
    use_reranking: bool = True
    use_mmr: bool = True  # Use MMR for diversity-aware retrieval
    mmr_lambda: float = 0.7  # MMR lambda parameter
    folder_id: Optional[str] = None  # Add folder_id for filtering
    min_relevance_score: float = 0.3  # Minimum relevance threshold
    
    # Internal retrievers (declared as fields to avoid Pydantic validation errors)
    # Using Any type to bypass validation, similar to BM25KeywordRetriever pattern
    _vector_retriever: Any = None
    _bm25_retriever: Any = None
    _ensemble_retriever: Any = None
    _content_type_detector: Optional[ContentTypeDetector] = None
    _query_classifier: Optional[QueryClassifier] = None
    _available_files: Optional[List[Dict]] = None
    
    def __init__(
        self,
        vector_store: LangChainVectorStore,
        documents: List[Document],
        reranker: Optional[Any] = None,
        k: int = 10,
        vector_weight: float = 0.7,
        bm25_weight: float = 0.3,
        use_reranking: bool = True,
        use_mmr: Optional[bool] = None,
        mmr_lambda: Optional[float] = None,
        folder_id: Optional[str] = None,
        min_relevance_score: Optional[float] = None,
        **kwargs,
    ):
        """Initialize adaptive retriever.

        Args:
            vector_store: LangChain vector store instance
            documents: List of documents for BM25 search (already filtered by folder)
            reranker: Optional reranker instance
            k: Number of documents to retrieve
            vector_weight: Weight for vector search in hybrid mode
            bm25_weight: Weight for BM25 search in hybrid mode
            use_reranking: Whether to use reranking for complex queries
            folder_id: Optional folder ID for filtering vector search results
            min_relevance_score: Minimum relevance score threshold (0.0-1.0)
            **kwargs: Additional arguments passed to BaseRetriever
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )
        
        # Initialize Pydantic model first
        from app.config.settings import settings
        init_values = {
            "vector_store": vector_store,
            "documents": documents,
            "reranker": reranker,
            "k": k,
            "vector_weight": vector_weight,
            "bm25_weight": bm25_weight,
            "use_reranking": use_reranking,
            "use_mmr": use_mmr if use_mmr is not None else settings.USE_MMR,
            "mmr_lambda": mmr_lambda if mmr_lambda is not None else settings.MMR_LAMBDA,
            "folder_id": folder_id,
            "min_relevance_score": min_relevance_score if min_relevance_score is not None else settings.RELEVANCE_THRESHOLD,
            **kwargs,
        }
        super().__init__(**init_values)
        
        # Initialize content type detector
        object.__setattr__(self, "_content_type_detector", ContentTypeDetector())
        
        # Initialize query classifier (with LLM enabled for better classification)
        object.__setattr__(self, "_query_classifier", get_query_classifier(available_files=documents or [], use_llm=True))
        
        # Initialize classification cache
        object.__setattr__(self, "_classification_cache", {})
        
        # Initialize base retrievers after super().__init__()
        # Use object.__setattr__ to bypass Pydantic validation for private attributes
        # Create vector retriever with folder filtering support
        search_kwargs = {"k": k}
        if folder_id:
            search_kwargs["filter"] = {"folder_id": folder_id}
            logger.debug("Creating vector retriever with folder filter", folder_id=folder_id)
        
        vector_retriever = VectorStoreRetriever(
            vector_store=vector_store,  # Pass the LangChainVectorStore instance
            k=k,
            search_kwargs=search_kwargs,
        )
        object.__setattr__(self, "_vector_retriever", vector_retriever)
        
        bm25_retriever = None
        if documents and bm25_weight > 0.0:
            try:
                bm25_retriever = BM25KeywordRetriever(
                    documents=documents,
                    k=k,
                )
                object.__setattr__(self, "_bm25_retriever", bm25_retriever)
            except Exception as e:
                logger.warning("Failed to initialize BM25 retriever", error=str(e))
                object.__setattr__(self, "_bm25_retriever", None)
        
        ensemble_retriever = None
        if bm25_retriever and bm25_weight > 0.0:
            try:
                ensemble_retriever = EnsembleRetriever(
                    retrievers=[vector_retriever, bm25_retriever],
                    weights=[vector_weight, bm25_weight],
                )
                object.__setattr__(self, "_ensemble_retriever", ensemble_retriever)
            except Exception as e:
                logger.warning("Failed to create ensemble retriever", error=str(e))
                object.__setattr__(self, "_ensemble_retriever", None)
    
    def _classify_query(self, query: str) -> str:
        """Classify query type to determine retrieval strategy.
        
        Args:
            query: User query
            
        Returns:
            Query type: 'exact_match', 'conceptual', or 'complex'
        """
        query_lower = query.lower()
        
        # Exact match indicators
        exact_indicators = ["function", "class", "variable", "import", "def ", "def(", 
                           "calculateTax", "authService", "file:", "line "]
        if any(indicator in query_lower for indicator in exact_indicators):
            return "exact_match"
        
        # Complex query indicators
        complex_indicators = ["compare", "contrast", "difference", "inconsistent", 
                             "gap", "check if", "verify", "consistent with", 
                             "every place where", "find all"]
        if any(indicator in query_lower for indicator in complex_indicators):
            return "complex"
        
        # Default to conceptual
        return "conceptual"
    
    def _reciprocal_rank_fusion(
        self, 
        results_a: List[Document], 
        results_b: List[Document],
        k: int = 60
    ) -> List[Document]:
        """Merge two ranked lists using Reciprocal Rank Fusion (RRF).
        
        Args:
            results_a: First ranked list
            results_b: Second ranked list
            k: RRF constant (default 60)
            
        Returns:
            Merged and deduplicated list of documents
        """
        scores = {}
        
        # Score documents from first list
        for rank, doc in enumerate(results_a):
            doc_id = doc.metadata.get("chunk_id") or str(id(doc))
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rank + k)
        
        # Score documents from second list
        for rank, doc in enumerate(results_b):
            doc_id = doc.metadata.get("chunk_id") or str(id(doc))
            scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (rank + k)
        
        # Sort by score
        ranked_ids = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
        
        # Return documents in ranked order (deduplicated)
        seen = set()
        merged = []
        doc_map = {doc.metadata.get("chunk_id") or str(id(doc)): doc 
                  for doc in results_a + results_b}
        
        for doc_id in ranked_ids:
            if doc_id not in seen and doc_id in doc_map:
                merged.append(doc_map[doc_id])
                seen.add(doc_id)
        
        return merged
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    ) -> List[Document]:
        """Get relevant documents using adaptive strategy.
        
        Args:
            query: Search query
            run_manager: Optional callback manager
            
        Returns:
            List of relevant Document objects
        """
        try:
            logger.info(
                "Starting retrieval",
                query_length=len(query),
                folder_id=self.folder_id,
                retriever_k=self.k,
            )
            # CRITICAL: Classify query type and detect content type
            # Cache classification to avoid duplicate work when retriever is called multiple times
            query_understanding = None
            content_type_filter = None
            
            if self._query_classifier:
                # Check cache first
                cache_key = f"{query}_{id(self._available_files)}"
                if cache_key in self._classification_cache:
                    query_understanding = self._classification_cache[cache_key]
                    logger.debug("Using cached query classification", query=query[:50])
                else:
                    # Update classifier with available files if needed
                    if self._available_files:
                        self._query_classifier.update_available_files(self._available_files)
                    
                    # Classify query (use sync version for synchronous method)
                    query_understanding = self._query_classifier.classify_sync(query)
                    # Cache result (limit cache size to prevent memory issues)
                    if len(self._classification_cache) > 100:
                        self._classification_cache.clear()
                    self._classification_cache[cache_key] = query_understanding
                
                logger.info(
                    "Query classified",
                    query_type=query_understanding.query_type.value,
                    confidence=query_understanding.confidence,
                    explanation=query_understanding.explanation,
                    entities=query_understanding.entities,
                    file_references=query_understanding.file_references,
                )
            
            # Also use content type detector for backward compatibility
            if self._content_type_detector:
                detection = self._content_type_detector.detect(
                    query=query,
                    available_files=self._available_files,
                )
                if detection["metadata_filter"]:
                    content_type_filter = detection["metadata_filter"]
                    # Use content type from query understanding if available (more comprehensive)
                    if query_understanding and query_understanding.content_type:
                        content_type_filter = {"file_type": {"$in": [query_understanding.content_type]}}
            
            # Map query understanding to legacy query_type for compatibility
            if query_understanding:
                query_type_map = {
                    QueryType.FACTUAL_SPECIFIC: "exact_match",
                    QueryType.FACTUAL_GENERAL: "conceptual",
                    QueryType.RELATIONSHIP: "complex",
                    QueryType.COMPARISON: "complex",
                    QueryType.ENTITY_SEARCH: "complex",
                    QueryType.TEMPORAL: "complex",
                }
                query_type = query_type_map.get(query_understanding.query_type, "conceptual")
            else:
                query_type = self._classify_query(query)  # Fallback to legacy classification
                logger.debug("Query classified (legacy)", query_type=query_type, query_length=len(query))
            
            # Build base filter combining folder_id, file_references (highest priority), and content_type
            base_filter = {}
            if self.folder_id:
                base_filter["folder_id"] = self.folder_id
            
            # Priority 1: file_references (specific file names mentioned) - highest priority
            if query_understanding and query_understanding.file_references:
                file_refs = query_understanding.file_references
                if len(file_refs) == 1:
                    base_filter["file_name"] = {"$eq": file_refs[0]}
                elif len(file_refs) > 1:
                    base_filter["file_name"] = {"$in": file_refs}
                logger.debug("Applied file_references filter", file_references=file_refs)
            # Priority 2: content_type filter (only if no specific file references)
            elif content_type_filter and "file_type" in content_type_filter:
                # Merge file_type filter into base_filter
                file_type_filter = content_type_filter["file_type"]
                if isinstance(file_type_filter, dict) and "$in" in file_type_filter:
                    # ChromaDB supports $in operator
                    base_filter["file_type"] = {"$in": file_type_filter["$in"]}
                elif isinstance(file_type_filter, str):
                    base_filter["file_type"] = file_type_filter
            
            if query_type == "exact_match":
                # Use BM25 for exact keyword matching
                # Skip BM25 if content type filter is active (BM25 can't pre-filter)
                if self._bm25_retriever and not self._has_content_type_filter(base_filter):
                    results = self._bm25_retriever.get_relevant_documents(query)
                    logger.debug("BM25 retrieval", result_count=len(results))
                    final_results = results[:self.k]
                    if len(final_results) == 0:
                        logger.warning("BM25 retrieval returned empty results", query=query[:100], folder_id=self.folder_id)
                    return final_results
                else:
                    # Fallback to vector search when content type filter is active or BM25 unavailable
                    if self._has_content_type_filter(base_filter):
                        logger.debug(
                            "Skipping BM25 for exact_match due to content type filter, using vector search",
                            filter=base_filter
                        )
                    # Continue to vector search (conceptual path will handle it)
                    query_type = "conceptual"  # Fall through to vector search
            
            if query_type == "conceptual":
                # Use vector search for semantic understanding with content type filter
                if base_filter and hasattr(self.vector_store, 'similarity_search'):
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            # If loop is running, use vector retriever (won't have filter, but will filter after)
                            results = self._vector_retriever.get_relevant_documents(query)
                        else:
                            results = loop.run_until_complete(
                                self.vector_store.similarity_search(
                                    query=query,
                                    k=self.k,
                                    filter=base_filter if base_filter else None,
                                )
                            )
                    except RuntimeError:
                        results = asyncio.run(
                            self.vector_store.similarity_search(
                                query=query,
                                k=self.k,
                                filter=base_filter if base_filter else None,
                            )
                        )
                else:
                    results = self._vector_retriever.get_relevant_documents(query)
                    # Apply content type filter after retrieval
                    if base_filter:
                        results = self._apply_content_type_filter(results, base_filter)
                logger.debug("Vector retrieval", result_count=len(results), filter_applied=bool(base_filter))
                final_results = results[:self.k]
                if len(final_results) == 0:
                    logger.warning(
                        "Vector retrieval returned empty results",
                        query=query[:100],
                        folder_id=self.folder_id,
                        filter_applied=bool(base_filter),
                        filter=base_filter,
                    )
                return final_results
            
            elif query_type == "complex":
                # Use hybrid search with reranking
                if self._ensemble_retriever:
                    # Hybrid: vector + BM25 with content type filtering
                    # Get vector results with filter
                    vector_results = []
                    if base_filter and hasattr(self.vector_store, 'similarity_search'):
                        import asyncio
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                vector_results = self._vector_retriever.get_relevant_documents(query)
                            else:
                                vector_results = loop.run_until_complete(
                                    self.vector_store.similarity_search(
                                        query=query,
                                        k=self.k * 2,  # Get more for RRF
                                        filter=base_filter if base_filter else None,
                                    )
                                )
                        except RuntimeError:
                            vector_results = asyncio.run(
                                self.vector_store.similarity_search(
                                    query=query,
                                    k=self.k * 2,
                                    filter=base_filter if base_filter else None,
                                )
                            )
                    else:
                        vector_results = self._vector_retriever.get_relevant_documents(query)
                    
                    # Get BM25 results and filter
                    # NOTE: Skip BM25 if content type filter is active, since BM25's in-memory index
                    # doesn't support pre-filtering and will return irrelevant documents that get filtered out
                    bm25_results = []
                    if self._bm25_retriever and not self._has_content_type_filter(base_filter):
                        bm25_results = self._bm25_retriever.get_relevant_documents(query)
                    elif self._bm25_retriever and self._has_content_type_filter(base_filter):
                        logger.debug(
                            "Skipping BM25 retrieval due to content type filter",
                            filter=base_filter
                        )
                    
                    # Merge using RRF
                    merged = self._reciprocal_rank_fusion(vector_results, bm25_results)
                    
                    # Get scored documents for reranking/MMR
                    # Use vector store's similarity_search_with_score to get relevance scores
                    scored_candidates = []
                    try:
                        # Get scored results from vector store
                        if hasattr(self.vector_store, 'similarity_search_with_score'):
                            import asyncio
                            try:
                                loop = asyncio.get_event_loop()
                                if loop.is_running():
                                    # If loop is running, we need to use a different approach
                                    # For now, use merged results and compute scores manually
                                    scored_candidates = [(doc, 0.8) for doc in merged[:self.k * 3]]
                                else:
                                    scored_candidates = loop.run_until_complete(
                                        self.vector_store.similarity_search_with_score(
                                            query=query,
                                            k=self.k * 3,
                                            filter=base_filter if base_filter else None,
                                        )
                                    )
                            except RuntimeError:
                                scored_candidates = asyncio.run(
                                    self.vector_store.similarity_search_with_score(
                                        query=query,
                                        k=self.k * 3,
                                        filter=base_filter if base_filter else None,
                                    )
                                )
                        else:
                            # Fallback: assign default scores
                            scored_candidates = [(doc, 0.8) for doc in merged[:self.k * 3]]
                    except Exception as e:
                        logger.warning("Failed to get scored candidates, using merged results", error=str(e))
                        scored_candidates = [(doc, 0.8) for doc in merged[:self.k * 3]]
                    
                    # Rerank if enabled with relevance threshold
                    if self.use_reranking and self.reranker and scored_candidates:
                        try:
                            docs_for_reranking = [doc for doc, _ in scored_candidates]
                            scored_docs = self.reranker.rerank_documents(
                                query=query,
                                documents=docs_for_reranking,
                                top_k=self.k * 2,  # Retrieve more for MMR/diversity filtering
                                min_relevance_score=self.min_relevance_score,
                            )
                            scored_candidates = scored_docs  # Update with reranked scores
                        except Exception as e:
                            logger.warning("Reranking failed, using original scores", error=str(e))
                    
                    # Apply file rebalancing to prevent overrepresented files from dominating
                    from app.config.settings import settings
                    if settings.USE_FILE_REBALANCING and scored_candidates:
                        try:
                            rebalancer = get_file_rebalancer(
                                max_chunks_per_file=settings.MAX_CHUNKS_PER_FILE,
                                use_score_normalization=True,
                                normalization_factor=settings.FILE_REBALANCING_NORMALIZATION,
                            )
                            scored_candidates = rebalancer.rebalance_scored_documents(scored_candidates)
                            logger.debug(
                                "File rebalancing applied",
                                result_count=len(scored_candidates),
                                max_chunks_per_file=settings.MAX_CHUNKS_PER_FILE,
                            )
                        except Exception as e:
                            logger.warning("File rebalancing failed, continuing without it", error=str(e))

                    # Apply MMR or document diversity
                    if self.use_mmr and scored_candidates:
                        try:
                            # Get query embedding for MMR
                            query_embedding = None
                            try:
                                if hasattr(self.vector_store, 'embeddings'):
                                    query_embedding = self.vector_store.embeddings.embed_query(query)
                            except Exception as e:
                                logger.warning("Failed to get query embedding for MMR", error=str(e))
                            
                            if query_embedding:
                                # Use MMR for diversity-aware selection
                                mmr_retriever = get_mmr_retriever(lambda_param=self.mmr_lambda)
                                # Provide embedding function to compute document embeddings if needed
                                embedding_function = None
                                if hasattr(self.vector_store, 'embeddings'):
                                    embedding_function = self.vector_store.embeddings.embed_query
                                results = mmr_retriever.select_with_mmr_from_scored_docs(
                                    query_embedding=query_embedding,
                                    scored_documents=scored_candidates,
                                    k=self.k,
                                    embedding_function=embedding_function,
                                )
                                logger.info(
                                    "Hybrid retrieval with reranking and MMR",
                                    result_count=len(results),
                                    mmr_lambda=self.mmr_lambda,
                                )
                                return results
                            else:
                                # Fallback to document diversity if embeddings unavailable
                                logger.warning("Query embedding unavailable, falling back to document diversity")
                                diversifier = get_document_diversifier(
                                    min_chunks_per_file=2,
                                    max_total_chunks=self.k,
                                )
                                results = [doc for doc, _ in scored_candidates[:self.k * 2]]
                                return diversifier.diversify_by_document(results)
                        except Exception as e:
                            logger.warning("MMR failed, falling back to document diversity", error=str(e))
                            # Fallback to document diversity
                            diversifier = get_document_diversifier(
                                min_chunks_per_file=2,
                                max_total_chunks=self.k,
                            )
                            results = [doc for doc, _ in scored_candidates[:self.k * 2]]
                            return diversifier.diversify_by_document(results)
                    else:
                        # Apply document diversity (MMR disabled)
                        diversifier = get_document_diversifier(
                            min_chunks_per_file=2,
                            max_total_chunks=self.k,
                        )
                        results = [doc for doc, _ in scored_candidates[:self.k * 2]] if scored_candidates else merged[:self.k * 2]
                        diversified = diversifier.diversify_by_document(results)
                        logger.debug(
                            "Hybrid retrieval with document diversity",
                            result_count=len(diversified),
                            file_distribution=diversifier.get_file_distribution(diversified),
                        )
                        return diversified
                else:
                    # Fallback to vector only with filter
                    if base_filter and hasattr(self.vector_store, 'similarity_search'):
                        import asyncio
                        try:
                            loop = asyncio.get_event_loop()
                            if loop.is_running():
                                results = self._vector_retriever.get_relevant_documents(query)
                            else:
                                results = loop.run_until_complete(
                                    self.vector_store.similarity_search(
                                        query=query,
                                        k=self.k,
                                        filter=base_filter if base_filter else None,
                                    )
                                )
                        except RuntimeError:
                            results = asyncio.run(
                                self.vector_store.similarity_search(
                                    query=query,
                                    k=self.k,
                                    filter=base_filter if base_filter else None,
                                )
                            )
                    else:
                        results = self._vector_retriever.get_relevant_documents(query)
                        if base_filter:
                            results = self._apply_content_type_filter(results, base_filter)
                    return results[:self.k]
            
            else:
                # Default to vector search with filter
                if base_filter and hasattr(self.vector_store, 'similarity_search'):
                    import asyncio
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            results = self._vector_retriever.get_relevant_documents(query)
                        else:
                            results = loop.run_until_complete(
                                self.vector_store.similarity_search(
                                    query=query,
                                    k=self.k,
                                    filter=base_filter if base_filter else None,
                                )
                            )
                    except RuntimeError:
                        results = asyncio.run(
                            self.vector_store.similarity_search(
                                query=query,
                                k=self.k,
                                filter=base_filter if base_filter else None,
                            )
                        )
                else:
                    results = self._vector_retriever.get_relevant_documents(query)
                    if base_filter:
                        results = self._apply_content_type_filter(results, base_filter)
                return results[:self.k]
        
        except Exception as e:
            logger.error(
                "Adaptive retrieval failed",
                query=query[:50],
                error=str(e),
                exc_info=True,
            )
            # Fallback to vector search
            try:
                return self._vector_retriever.get_relevant_documents(query)[:self.k]
            except Exception:
                return []
    
    async def aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        folder_id: Optional[str] = None,
        available_files: Optional[List[Dict]] = None,
    ) -> List[Document]:
        """Async version of get_relevant_documents with caching.
        
        Args:
            query: Search query
            run_manager: Optional callback manager
            folder_id: Optional folder ID for cache scoping
            
        Returns:
            List of relevant Document objects
        """
        try:
            # Check cache first
            from app.utils.caching import get_cache
            cache = get_cache()
            
            # Store available_files for this retrieval
            if available_files is not None:
                object.__setattr__(self, "_available_files", available_files)
            
            # Use stored available_files if not provided
            files_to_use = available_files or self._available_files
            
            # CRITICAL: Classify query type and detect content type
            # Cache classification to avoid duplicate work when retriever is called multiple times
            query_understanding = None
            content_type_filter = None
            
            if self._query_classifier:
                # Check cache first
                cache_key = f"{query}_{id(files_to_use)}"
                if cache_key in self._classification_cache:
                    query_understanding = self._classification_cache[cache_key]
                    logger.debug("Using cached query classification (async)", query=query[:50])
                else:
                    # Update classifier with available files if needed
                    if files_to_use:
                        self._query_classifier.update_available_files(files_to_use)
                    
                    # Classify query (async version)
                    query_understanding = await self._query_classifier.classify(query)
                    # Cache result (limit cache size to prevent memory issues)
                    if len(self._classification_cache) > 100:
                        self._classification_cache.clear()
                    self._classification_cache[cache_key] = query_understanding
                
                logger.info(
                    "Query classified",
                    query_type=query_understanding.query_type.value,
                    confidence=query_understanding.confidence,
                    explanation=query_understanding.explanation,
                    entities=query_understanding.entities,
                    file_references=query_understanding.file_references,
                )
            
            # Also use content type detector for backward compatibility
            if self._content_type_detector:
                detection = self._content_type_detector.detect(
                    query=query,
                    available_files=files_to_use,
                )
                if detection["metadata_filter"]:
                    content_type_filter = detection["metadata_filter"]
                    content_type_val = detection.get("content_type")
                    logger.info(
                        "Content type detected for query",
                        content_type=content_type_val.value if hasattr(content_type_val, 'value') else content_type_val,
                        confidence=detection.get("confidence", 0.0),
                        explanation=detection.get("explanation", ""),
                    )
                    # Use content type from query understanding if available (more comprehensive)
                    if query_understanding and hasattr(query_understanding, 'content_type') and query_understanding.content_type:
                        content_type_filter = {"file_type": {"$in": [query_understanding.content_type]}}
            
            combined_filter = None
            if folder_id or self.folder_id or query_understanding and query_understanding.file_references or content_type_filter:
                combined_filter = {}
                if folder_id or self.folder_id:
                    combined_filter["folder_id"] = folder_id or self.folder_id
                
                # Priority 1: file_references (specific file names mentioned) - highest priority
                if query_understanding and query_understanding.file_references:
                    file_refs = query_understanding.file_references
                    if len(file_refs) == 1:
                        combined_filter["file_name"] = {"$eq": file_refs[0]}
                    elif len(file_refs) > 1:
                        combined_filter["file_name"] = {"$in": file_refs}
                    logger.debug("Applied file_references filter (async)", file_references=file_refs)
                # Priority 2: content_type filter (only if no specific file references)
                elif content_type_filter:
                    # Merge content type filter (e.g., file_type)
                    combined_filter.update(content_type_filter)
            
            # Update vector retriever search_kwargs with combined filter
            if combined_filter and hasattr(self._vector_retriever, "search_kwargs"):
                self._vector_retriever.search_kwargs["filter"] = combined_filter
                logger.debug("Updated vector retriever filter", filter=combined_filter)
            
            cached_results = cache.get_retrieval(query, folder_id=folder_id or self.folder_id)
            if cached_results:
                logger.debug(
                    "Cache hit for adaptive retrieval",
                    query_length=len(query),
                    result_count=len(cached_results),
                )
                # Convert back to Documents
                from langchain_core.documents import Document
                results = [
                    Document(page_content=r["page_content"], metadata=r["metadata"])
                    for r in cached_results
                ]
                # Apply content type filter to cached results if needed
                if content_type_filter:
                    results = self._apply_content_type_filter(results, content_type_filter)
                return results
            
            # Map query understanding to legacy query_type for compatibility
            if query_understanding:
                query_type_map = {
                    QueryType.FACTUAL_SPECIFIC: "exact_match",
                    QueryType.FACTUAL_GENERAL: "conceptual",
                    QueryType.RELATIONSHIP: "complex",
                    QueryType.COMPARISON: "complex",
                    QueryType.ENTITY_SEARCH: "complex",
                    QueryType.TEMPORAL: "complex",
                }
                query_type = query_type_map.get(query_understanding.query_type, "conceptual")
            else:
                query_type = self._classify_query(query)  # Fallback to legacy classification
                logger.debug("Query classified (legacy)", query_type=query_type, query_length=len(query))
            
            if query_type == "exact_match" and self._bm25_retriever:
                # Use BM25 for exact keyword matching
                if hasattr(self._bm25_retriever, "aget_relevant_documents"):
                    results = await self._bm25_retriever.aget_relevant_documents(query)
                else:
                    results = self._bm25_retriever.get_relevant_documents(query)
                logger.debug("BM25 retrieval", result_count=len(results))
                results = results[:self.k]
                # Apply content type filter if available
                if content_type_filter:
                    results = self._apply_content_type_filter(results, content_type_filter)
                return results
            
            elif query_type == "conceptual":
                # Use vector search for semantic understanding
                results = await self._vector_retriever.aget_relevant_documents(query)
                logger.debug("Vector retrieval", result_count=len(results))
                results = results[:self.k]
                # Apply content type filter if available
                if content_type_filter:
                    results = self._apply_content_type_filter(results, content_type_filter)
                return results
            
            elif query_type == "complex":
                # Use hybrid search with reranking
                # Get results from both retrievers
                vector_results = await self._vector_retriever.aget_relevant_documents(query)
                bm25_results = []
                # Skip BM25 if content type filter is active (BM25 can't pre-filter)
                if self._bm25_retriever and not content_type_filter:
                    if hasattr(self._bm25_retriever, "aget_relevant_documents"):
                        bm25_results = await self._bm25_retriever.aget_relevant_documents(query)
                    else:
                        bm25_results = self._bm25_retriever.get_relevant_documents(query)
                elif self._bm25_retriever and content_type_filter:
                    logger.debug(
                        "Skipping BM25 retrieval due to content type filter (async)",
                        filter=content_type_filter
                    )
                
                # Merge using RRF
                merged = self._reciprocal_rank_fusion(vector_results, bm25_results)
                
                # Rerank if enabled
                if self.use_reranking and self.reranker and merged:
                    try:
                        scored_docs = self.reranker.rerank_documents(
                            query=query,
                            documents=merged[:self.k * 3],
                            top_k=self.k * 2,  # Retrieve more for diversity filtering
                        )
                        results = [doc for doc, _ in scored_docs]

                        # Apply document diversity
                        diversifier = get_document_diversifier(
                            min_chunks_per_file=2,
                            max_total_chunks=self.k,
                        )
                        results = diversifier.diversify_by_document(results)

                        logger.debug(
                            "Hybrid + reranking + diversity",
                            result_count=len(results),
                            file_distribution=diversifier.get_file_distribution(results),
                        )
                        # Apply content type filter if available
                        if content_type_filter:
                            results = self._apply_content_type_filter(results, content_type_filter)
                        return results
                    except Exception as e:
                        logger.warning("Reranking failed, using merged results", error=str(e))
                        # Still apply diversity
                        diversifier = get_document_diversifier(
                            min_chunks_per_file=2,
                            max_total_chunks=self.k,
                        )
                        diversified = diversifier.diversify_by_document(merged[:self.k * 2])
                        # Apply content type filter if available
                        if content_type_filter:
                            diversified = self._apply_content_type_filter(diversified, content_type_filter)
                        return diversified

                # Apply diversity even without reranking
                diversifier = get_document_diversifier(
                    min_chunks_per_file=2,
                    max_total_chunks=self.k,
                )
                diversified = diversifier.diversify_by_document(merged[:self.k * 2])
                logger.debug(
                    "Hybrid retrieval with diversity",
                    result_count=len(diversified),
                    file_distribution=diversifier.get_file_distribution(diversified),
                )
                # Apply content type filter if available
                if content_type_filter:
                    diversified = self._apply_content_type_filter(diversified, content_type_filter)
                # Cache results
                from app.config.settings import settings
                cache.set_retrieval(
                    query,
                    diversified,
                    ttl=settings.RETRIEVAL_CACHE_TTL,
                    folder_id=folder_id or self.folder_id,
                )
                return diversified
            
            else:
                # Default to vector search
                results = await self._vector_retriever.aget_relevant_documents(query)
                results = results[:self.k]
                # Apply content type filter if available
                if content_type_filter:
                    results = self._apply_content_type_filter(results, content_type_filter)
            
            # Cache results
            from app.config.settings import settings
            cache.set_retrieval(
                query,
                results,
                ttl=settings.RETRIEVAL_CACHE_TTL,
                folder_id=folder_id or self.folder_id,
            )
            
            return results
        
        except Exception as e:
            logger.error(
                "Adaptive retrieval failed",
                query=query[:50],
                error=str(e),
                exc_info=True,
            )
            # Fallback to vector search
            try:
                # Re-detect content type if not already done
                if content_type_filter is None and self._content_type_detector:
                    files_to_use = available_files or self._available_files
                    detection = self._content_type_detector.detect(
                        query=query,
                        available_files=files_to_use,
                    )
                    if detection["metadata_filter"]:
                        content_type_filter = detection["metadata_filter"]
                
                results = await self._vector_retriever.aget_relevant_documents(query)[:self.k]
                # Apply content type filter if available
                if content_type_filter:
                    results = self._apply_content_type_filter(results, content_type_filter)
                return results
            except Exception:
                return []
    
    def _has_content_type_filter(self, filter_dict: Dict[str, Any]) -> bool:
        """Check if filter contains a content type (file_type) filter.
        
        Args:
            filter_dict: Filter dictionary
            
        Returns:
            True if filter contains file_type filter
        """
        if not filter_dict:
            return False
        
        # Check for file_type in filter or in $and conditions
        if "file_type" in filter_dict:
            return True
        
        if "$and" in filter_dict:
            for condition in filter_dict["$and"]:
                if isinstance(condition, dict) and "file_type" in condition:
                    return True
        
        return False
    
    def _apply_content_type_filter(
        self, documents: List[Document], filter_dict: Dict[str, Any]
    ) -> List[Document]:
        """Apply content type filter to documents.
        
        Args:
            documents: List of Document objects
            filter_dict: Filter dictionary (e.g., {'file_type': {'$in': ['audio']}})
            
        Returns:
            Filtered list of Document objects
        """
        if not filter_dict:
            return documents
        
        filtered = []
        file_types = None
        
        # Extract file types from filter (handle both direct and $and formats)
        if "file_type" in filter_dict:
            file_type_value = filter_dict["file_type"]
            if isinstance(file_type_value, dict) and "$in" in file_type_value:
                file_types = set(file_type_value["$in"])
            elif isinstance(file_type_value, str):
                file_types = {file_type_value}
        elif "$and" in filter_dict:
            # Extract file_type from $and conditions
            for condition in filter_dict["$and"]:
                if isinstance(condition, dict) and "file_type" in condition:
                    file_type_value = condition["file_type"]
                    if isinstance(file_type_value, dict) and "$in" in file_type_value:
                        file_types = set(file_type_value["$in"])
                    elif isinstance(file_type_value, str):
                        file_types = {file_type_value}
                    break
        
        for doc in documents:
            metadata = doc.metadata if hasattr(doc, "metadata") else {}
            
            # Filter by file_type (case-insensitive matching)
            if file_types:
                doc_file_type = metadata.get("file_type")
                # Convert to lowercase for case-insensitive comparison
                doc_file_type_lower = doc_file_type.lower() if doc_file_type else None
                file_types_lower = {ft.lower() for ft in file_types}
                
                if doc_file_type_lower not in file_types_lower:
                    # Log first few mismatches for debugging
                    if len(filtered) < 3:
                        logger.debug(
                            "Document filtered out by file_type",
                            doc_file_type=doc_file_type,
                            expected_types=list(file_types),
                            file_name=metadata.get("file_name", "Unknown"),
                        )
                    continue
            
            # Filter by file_id if specified
            if "file_id" in filter_dict:
                file_id_filter = filter_dict["file_id"]
                if isinstance(file_id_filter, dict) and "$eq" in file_id_filter:
                    if metadata.get("file_id") != file_id_filter["$eq"]:
                        continue
                elif metadata.get("file_id") != file_id_filter:
                    continue
            
            filtered.append(doc)
        
        logger.debug(
            "Applied content type filter",
            original_count=len(documents),
            filtered_count=len(filtered),
            filter_dict=filter_dict,
            expected_file_types=list(file_types) if file_types else None,
        )
        
        # If filtering removed all documents, log sample metadata values for debugging
        if len(filtered) == 0 and len(documents) > 0 and file_types:
            sample_meta = documents[0].metadata if hasattr(documents[0], "metadata") else {}
            actual_file_types = set()
            for doc in documents[:5]:  # Check first 5 documents
                if hasattr(doc, "metadata"):
                    doc_file_type = doc.metadata.get("file_type")
                    if doc_file_type:
                        actual_file_types.add(doc_file_type)
            
            logger.warning(
                "Content type filter removed all documents",
                expected_types=list(file_types),
                actual_file_types=list(actual_file_types),
                sample_file_type=sample_meta.get("file_type"),
                sample_file_name=sample_meta.get("file_name"),
                total_documents=len(documents),
            )
            
            # Don't return all documents as fallback - if filter is too strict, 
            # return empty list and let the caller handle it (e.g., relax filter or show error)
        
        return filtered

