"""Main LangChain RAG service orchestration with hybrid retrieval."""

from typing import Optional, Dict, Any, List, Callable
import structlog

from app.core.exceptions import ProcessingError
from app.rag.vector_store import LangChainVectorStore
from app.rag.retrievers import HybridLangChainRetriever
from app.rag.retrieval_chains import HybridRetrievalChain
from app.rag.reranking import LangChainReranker, RerankingRetriever
from app.rag.llm_chains import OllamaLLM
from app.rag.memory import get_memory_manager
from app.rag.prompt_management import PromptManager, get_prompt_manager

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.schema import Document, BaseRetriever
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        Document = None
        BaseRetriever = None

logger = structlog.get_logger(__name__)


class LangChainRAGService:
    """Main RAG service orchestrating LangChain-based retrieval and generation.

    Combines multiple retrieval strategies, reranking, and LLM generation
    with proper citation tracking and metadata propagation.
    """

    def __init__(
        self,
        vector_store: LangChainVectorStore,
        llm: Optional[OllamaLLM] = None,
        memory_manager: Optional[Any] = None,
        prompt_manager: Optional[PromptManager] = None,
        use_multi_query: bool = True,
        use_compression: bool = True,
        use_reranking: bool = True,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        retrieval_k: int = 20,
        final_k: int = 10,
    ):
        """Initialize LangChain RAG service.

        Args:
            vector_store: LangChain vector store instance
            llm: Ollama LLM instance
            memory_manager: Memory manager instance
            prompt_manager: Prompt manager instance
            use_multi_query: Whether to use multi-query expansion
            use_compression: Whether to use contextual compression
            use_reranking: Whether to use reranking
            vector_weight: Weight for vector search in ensemble
            bm25_weight: Weight for BM25 search in ensemble
            retrieval_k: Number of documents to retrieve initially
            final_k: Number of documents to return after reranking

        Raises:
            ProcessingError: If initialization fails
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        self.vector_store = vector_store
        self.llm = llm or OllamaLLM()
        self.memory_manager = memory_manager or get_memory_manager()
        self.prompt_manager = prompt_manager or get_prompt_manager()
        # PERFORMANCE: Disable expensive features for faster responses
        self.use_multi_query = False  # Disabled: 3x LLM calls
        self.use_compression = False  # Disabled: 1x LLM call
        self.use_reranking = False  # PERFORMANCE: Disabled cross-encoder reranking
        # PERFORMANCE: Use vector-only search (BM25 requires loading all docs)
        self.vector_weight = 1.0  # Vector only
        self.bm25_weight = 0.0  # Disable BM25
        self.retrieval_k = 5  # PERFORMANCE: Reduced from 20 to 5
        self.final_k = 5  # PERFORMANCE: Reduced from 10 to 5

        # Initialize reranker
        self.reranker = LangChainReranker(use_cross_encoder=use_reranking)

        # Cache for retrievers and chains per folder
        self.retrievers: Dict[str, BaseRetriever] = {}
        self.chains: Dict[str, HybridRetrievalChain] = {}

        logger.info(
            "Initialized LangChain RAG service",
            use_multi_query=use_multi_query,
            use_compression=use_compression,
            use_reranking=use_reranking,
        )

    async def initialize_for_folder(
        self,
        folder_id: str,
        documents: Optional[List[Document]] = None,
    ) -> None:
        """Initialize retriever and chain for a specific folder.

        Args:
            folder_id: Folder identifier
            documents: Optional list of documents for BM25 (will be loaded if not provided)

        Raises:
            ProcessingError: If initialization fails
        """
        try:
            # Load documents if not provided
            if documents is None:
                documents = await self._load_folder_documents(folder_id)

            # Create hybrid retriever
            retriever = HybridLangChainRetriever.create_hybrid_retriever(
                vector_store=self.vector_store,
                documents=documents,
                llm=self.llm.get_llm(),
                use_multi_query=self.use_multi_query,
                use_compression=self.use_compression,
                vector_weight=self.vector_weight,
                bm25_weight=self.bm25_weight,
                k=self.retrieval_k,
            )

            # Apply reranking if enabled
            if self.use_reranking:
                retriever = RerankingRetriever(
                    base_retriever=retriever,
                    reranker=self.reranker,
                    top_k=self.final_k,
                )

            # Create retrieval chain
            chain = HybridRetrievalChain(
                retriever=retriever,
                llm=self.llm,
                prompt_manager=self.prompt_manager,
            )

            # Cache retriever and chain
            self.retrievers[folder_id] = retriever
            self.chains[folder_id] = chain

            logger.info(
                "Initialized RAG service for folder",
                folder_id=folder_id,
                document_count=len(documents),
            )

        except Exception as e:
            logger.error(
                "Failed to initialize RAG service for folder",
                folder_id=folder_id,
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(
                f"Failed to initialize RAG service for folder: {str(e)}"
            ) from e

    async def _load_folder_documents(
        self, folder_id: str
    ) -> List[Document]:
        """Load all documents for a folder from vector store.

        Args:
            folder_id: Folder identifier

        Returns:
            List of Document objects filtered by folder_id
        """
        try:
            logger.debug("Loading folder documents", folder_id=folder_id)
            
            # Use similarity search with empty query and folder_id filter
            # to retrieve all documents for this folder
            documents = await self.vector_store.similarity_search(
                query="",  # Empty query to match all
                k=10000,  # Large number to get all documents
                filter={"folder_id": folder_id}
            )
            
            logger.info(
                "Loaded folder documents",
                folder_id=folder_id,
                document_count=len(documents),
            )
            
            return documents

        except Exception as e:
            logger.error(
                "Failed to load folder documents",
                folder_id=folder_id,
                error=str(e),
                exc_info=True,
            )
            # Return empty list on error to allow graceful degradation
            return []

    async def query(
        self,
        query: str,
        folder_id: str,
        conversation_id: Optional[str] = None,
        query_type: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Execute RAG query with hybrid retrieval.

        Args:
            query: User query
            folder_id: Folder identifier
            conversation_id: Optional conversation identifier for memory
            query_type: Optional query type (factual, synthesis, relationship, default)
            streaming_callback: Optional callback for streaming tokens

        Returns:
            Dictionary with answer, sources, citations, and metadata

        Raises:
            ProcessingError: If query execution fails
        """
        try:
            # Ensure folder is initialized
            if folder_id not in self.chains:
                await self.initialize_for_folder(folder_id)

            # Get chain for folder
            chain = self.chains[folder_id]

            # Get or create memory if conversation_id provided
            memory = None
            if conversation_id:
                memory = self.memory_manager.get_or_create_memory(conversation_id)

            # Invoke chain
            result = await chain.invoke(
                query=query,
                query_type=query_type,
                streaming_callback=streaming_callback,
            )

            # Extract citations from source documents
            source_documents = result.get("source_documents", [])
            citations = self._extract_citations(source_documents)

            # Update memory if available
            if memory:
                memory.add_user_message(query)
                if result.get("answer"):
                    memory.add_ai_message(result["answer"])

            logger.info(
                "RAG query completed",
                folder_id=folder_id,
                query_length=len(query),
                answer_length=len(result.get("answer", "")),
                citation_count=len(citations),
            )

            return {
                "answer": result.get("answer", ""),
                "sources": source_documents,
                "citations": citations,
                "query_type": result.get("query_type", "default"),
                "conversation_id": conversation_id,
                "folder_id": folder_id,
            }

        except Exception as e:
            logger.error(
                "RAG query failed",
                folder_id=folder_id,
                query=query[:100],
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(f"RAG query failed: {str(e)}") from e

    def _extract_citations(
        self, source_documents: List[Document]
    ) -> List[Dict[str, Any]]:
        """Extract citation information from source documents with metadata propagation.

        Args:
            source_documents: List of source Document objects

        Returns:
            List of citation dictionaries with full metadata
        """
        citations = []

        for doc in source_documents:
            metadata = doc.metadata if hasattr(doc, "metadata") else {}

            # Extract all relevant metadata for citations
            citation = {
                "file_id": metadata.get("file_id"),
                "file_name": metadata.get("file_name", "Unknown"),
                "chunk_id": metadata.get("chunk_id"),
                "page_number": metadata.get("page_number"),
                "chunk_index": metadata.get("chunk_index"),
                "start_time": metadata.get("start_time"),
                "end_time": metadata.get("end_time"),
                "relevance_score": metadata.get("relevance_score"),
                "content_preview": (
                    doc.page_content[:200]
                    if hasattr(doc, "page_content")
                    else str(doc)[:200]
                ),
                "source": metadata.get("source"),
                "file_path": metadata.get("file_path"),
                "mime_type": metadata.get("mime_type"),
                "metadata": metadata,  # Include full metadata for reference
            }

            citations.append(citation)

        return citations

    async def add_documents(
        self,
        documents: List[Document],
        folder_id: Optional[str] = None,
    ) -> List[str]:
        """Add documents to vector store and update retrievers.

        Args:
            documents: List of Document objects to add
            folder_id: Optional folder identifier

        Returns:
            List of document IDs

        Raises:
            ProcessingError: If adding documents fails
        """
        try:
            # Add documents to vector store
            doc_ids = await self.vector_store.add_documents(documents)

            # If folder_id provided and retriever exists, reinitialize
            if folder_id and folder_id in self.retrievers:
                await self.initialize_for_folder(folder_id, documents=documents)

            logger.info(
                "Added documents to RAG service",
                document_count=len(doc_ids),
                folder_id=folder_id,
            )

            return doc_ids

        except Exception as e:
            logger.error(
                "Failed to add documents to RAG service",
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(f"Failed to add documents: {str(e)}") from e

    def clear_folder_cache(self, folder_id: str) -> None:
        """Clear cached retriever and chain for folder.

        Args:
            folder_id: Folder identifier
        """
        if folder_id in self.retrievers:
            del self.retrievers[folder_id]
        if folder_id in self.chains:
            del self.chains[folder_id]

        logger.debug("Cleared folder cache", folder_id=folder_id)


# Global RAG service instance
_rag_service: Optional[LangChainRAGService] = None


def get_rag_service(
    vector_store: Optional[LangChainVectorStore] = None,
) -> LangChainRAGService:
    """Get global RAG service instance.

    Args:
        vector_store: Optional vector store to initialize service with

    Returns:
        LangChainRAGService instance

    Raises:
        ProcessingError: If vector store is required but not provided
    """
    global _rag_service
    if _rag_service is None:
        if vector_store is None:
            raise ProcessingError("Vector store is required to initialize RAG service")
        _rag_service = LangChainRAGService(vector_store=vector_store)
    elif vector_store and _rag_service.vector_store != vector_store:
        _rag_service.vector_store = vector_store
        _rag_service.retrievers.clear()
        _rag_service.chains.clear()
    return _rag_service

