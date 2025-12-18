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
        from langchain.retrievers import EnsembleRetriever, BM25Retriever
        from langchain.retrievers.multi_query import MultiQueryRetriever
        from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import LLMChainExtractor
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback for older LangChain versions
    try:
        from langchain.schema import BaseRetriever, Document
        from langchain.callbacks.manager import CallbackManagerForRetrieverRun
        from langchain.retrievers import EnsembleRetriever, BM25Retriever
        from langchain.retrievers.multi_query import MultiQueryRetriever
        from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
        from langchain.retrievers.document_compressors import LLMChainExtractor
        Field = None
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

logger = structlog.get_logger(__name__)


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

