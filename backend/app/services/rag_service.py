"""Main LangChain RAG service orchestration with hybrid retrieval."""

from typing import Optional, Dict, Any, List, Callable
import structlog

from app.core.exceptions import ProcessingError
from app.rag.vector_store import LangChainVectorStore
from app.rag.retrievers import AdaptiveRetriever
from app.rag.retrieval_chains import HybridRetrievalChain
from app.rag.reranking import LangChainReranker
from app.rag.llm_chains import OllamaLLM
from app.rag.memory import get_memory_manager
from app.rag.prompt_management import PromptManager, get_prompt_manager
from app.knowledge_graph.graph_queries import GraphQueries
from app.monitoring.langsmith_monitoring import get_langsmith_monitor
from app.services.debug_service import DebugMetrics
from app.services.citation_engine import CitationEngine
from app.database.sqlite_manager import SQLiteManager

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
        use_reranking: Optional[bool] = None,
        vector_weight: float = 0.5,
        bm25_weight: float = 0.5,
        retrieval_k: Optional[int] = None,
        final_k: Optional[int] = None,
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
        # Enable hybrid search with adaptive retrieval
        # Use settings default if not explicitly provided
        from app.config.settings import settings
        self.use_reranking = use_reranking if use_reranking is not None else settings.USE_RERANKING
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        # Use settings defaults for TTFT optimization if not provided
        self.retrieval_k = retrieval_k if retrieval_k is not None else settings.MAX_RETRIEVAL_DOCS
        self.final_k = final_k if final_k is not None else settings.MAX_CONTEXT_DOCS

        logger.info(
            "RAG service retrieval configuration",
            retrieval_k=self.retrieval_k,
            final_k=self.final_k,
            use_reranking=self.use_reranking,
        )

        # Initialize reranker only if enabled
        self.reranker = None
        if self.use_reranking:
            logger.info("Initializing reranker (will download HuggingFace models if not cached)")
            self.reranker = LangChainReranker(
                use_cross_encoder=True,
                model_name=settings.RERANKER_MODEL
            )
        else:
            logger.info("Reranking disabled - using simple relevance scoring")

        # Initialize graph queries for graph-augmented retrieval
        self.graph_queries = GraphQueries()

        # Initialize citation engine for sophisticated citation mapping
        self.citation_engine = CitationEngine(db=SQLiteManager())

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

            # Create adaptive retriever with folder filtering
            retriever = AdaptiveRetriever(
                vector_store=self.vector_store,
                documents=documents,
                reranker=self.reranker,
                k=self.retrieval_k,
                vector_weight=self.vector_weight,
                bm25_weight=self.bm25_weight,
                use_reranking=self.use_reranking,
                folder_id=folder_id,  # Pass folder_id for vector search filtering
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
            
            # Use get() method with filter instead of similarity_search for getting all documents
            # This avoids HNSW index issues with large k values
            try:
                # Try using ChromaDB's get() method directly if available
                chroma_collection = self.vector_store.vector_store._collection
                if hasattr(chroma_collection, 'get'):
                    # Get all document IDs for this folder using where filter
                    where_filter = {"folder_id": folder_id}
                    results = chroma_collection.get(where=where_filter)
                    
                    # Convert results to Document format
                    documents = []
                    if results and "documents" in results and results["documents"]:
                        for i, doc_text in enumerate(results["documents"]):
                            metadata = {}
                            if "metadatas" in results and results["metadatas"] and i < len(results["metadatas"]):
                                metadata = results["metadatas"][i]
                            
                            from langchain_core.documents import Document
                            documents.append(Document(page_content=doc_text, metadata=metadata))
                    
                    logger.info(
                        "Loaded folder documents using get()",
                        folder_id=folder_id,
                        document_count=len(documents),
                    )
                    return documents
            except Exception as get_error:
                logger.debug(
                    "get() method failed, falling back to similarity_search",
                    error=str(get_error),
                )
            
            # Fallback: Use similarity search with smaller k and multiple queries if needed
            # Start with reasonable k value to avoid HNSW errors
            documents = await self.vector_store.similarity_search(
                query="document",  # Use a generic term instead of empty string
                k=500,  # Reasonable default - can be increased if needed
                filter={"folder_id": folder_id}
            )
            
            logger.info(
                "Loaded folder documents using similarity_search",
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
        status_callback: Optional[Callable[[str], None]] = None,
        citations_callback: Optional[Callable[[list], None]] = None,
    ) -> Dict[str, Any]:
        """Execute RAG query with hybrid retrieval.

        Args:
            query: User query
            folder_id: Folder identifier
            conversation_id: Optional conversation identifier for memory
            query_type: Optional query type (factual, synthesis, relationship, default)
            streaming_callback: Optional callback for streaming tokens
            status_callback: Optional callback for status updates
            citations_callback: Optional callback for progressive citations

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
            
            # Get available files for content type detection
            available_files = await self._get_folder_files_metadata(folder_id)
            
            # Update retriever with available files for content type detection
            if folder_id in self.retrievers:
                retriever = self.retrievers[folder_id]
                if hasattr(retriever, "_available_files"):
                    # Store available_files in retriever for content type detection
                    object.__setattr__(retriever, "_available_files", available_files)
                    logger.debug(
                        "Updated retriever with available files",
                        folder_id=folder_id,
                        file_count=len(available_files),
                    )

            # Get or create memory if conversation_id provided
            memory = None
            if conversation_id:
                memory = self.memory_manager.get_or_create_memory(conversation_id)

            # Get LangSmith callbacks for observability
            langsmith_monitor = get_langsmith_monitor()
            callbacks = langsmith_monitor.get_callbacks()

            # Invoke chain with LangSmith callbacks and progress callbacks
            invoke_kwargs: Dict[str, Any] = {
                "query": query,
                "query_type": query_type,
                "streaming_callback": streaming_callback,
                "status_callback": status_callback,
                "citations_callback": citations_callback,
            }
            if callbacks:
                invoke_kwargs["callbacks"] = callbacks

            result = await chain.invoke(**invoke_kwargs)

            # Get citations from the result (now extracted inline by the chain)
            citations = result.get("citations", [])
            
            # Get source documents from result
            source_documents = result.get("source_documents", [])
            
            # Get LLM response text
            answer = result.get("answer", "")
            
            # If no inline citations were extracted, use CitationEngine to map response to sources
            if not citations and answer and source_documents:
                try:
                    # Use CitationEngine to generate sophisticated citations by mapping response to chunks
                    citation_objects = await self.citation_engine.generate_citations(
                        response=answer,
                        retrieved_chunks=source_documents,
                        min_confidence=0.3,
                    )
                    
                    # Convert Citation objects to dict format for API compatibility
                    citations = self._convert_citations_to_dict(citation_objects)
                    
                    logger.info(
                        "Generated citations using CitationEngine",
                        citation_count=len(citations),
                        high_confidence=len([c for c in citations if c.get("confidence", 0) >= 0.7]),
                    )
                except Exception as e:
                    logger.warning(
                        "CitationEngine failed, falling back to simple extraction",
                        error=str(e),
                    )
                    # Fallback to simple metadata extraction
                    citations = self._extract_citations(source_documents)
            elif not citations:
                # No answer or source documents, use simple extraction
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

            response = {
                "answer": result.get("answer", ""),
                "sources": source_documents,
                "citations": citations,
                "query_type": result.get("query_type", "default"),
                "conversation_id": conversation_id,
                "folder_id": folder_id,
            }

            return response

        except Exception as e:
            
            logger.error(
                "RAG query failed",
                folder_id=folder_id,
                query=query[:100],
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(f"RAG query failed: {str(e)}") from e

    async def _get_folder_files_metadata(self, folder_id: str) -> List[Dict[str, Any]]:
        """Get file metadata for a folder to enable content type detection.
        
        Args:
            folder_id: Folder identifier
            
        Returns:
            List of file metadata dictionaries with file_id and file_name
        """
        try:
            # Get unique files from documents in the folder
            if folder_id not in self.retrievers:
                await self.initialize_for_folder(folder_id)
            
            # Load folder documents to extract file metadata
            documents = await self._load_folder_documents(folder_id)
            
            # Extract unique files
            files_map = {}
            for doc in documents:
                metadata = doc.metadata if hasattr(doc, "metadata") else {}
                file_id = metadata.get("file_id")
                file_name = metadata.get("file_name")
                
                if file_id and file_id not in files_map:
                    files_map[file_id] = {
                        "file_id": file_id,
                        "file_name": file_name or "Unknown",
                    }
            
            files_list = list(files_map.values())
            logger.debug(
                "Extracted folder files metadata",
                folder_id=folder_id,
                file_count=len(files_list),
            )
            
            return files_list
            
        except Exception as e:
            logger.warning(
                "Failed to get folder files metadata",
                folder_id=folder_id,
                error=str(e),
            )
            return []

    async def retrieve_with_graph_expansion(
        self,
        query: str,
        folder_id: str,
        k: int = 10,
    ) -> List[Document]:
        """Retrieve documents with graph-augmented expansion.
        
        Flow:
        1. Initial vector retrieval (k=10)
        2. Expand via knowledge graph (find related chunks via relationships)
        3. Fetch expanded chunks from vector store
        4. Rerank expanded set
        5. Return top k*2 results
        
        Args:
            query: User query
            folder_id: Folder identifier
            k: Number of documents to return
            
        Returns:
            List of Document objects with expanded context
        """
        try:
            logger.info(
                "Graph-augmented retrieval",
                query_length=len(query),
                folder_id=folder_id,
                k=k,
            )
            
            # Step 1: Initial vector retrieval
            if folder_id not in self.retrievers:
                await self.initialize_for_folder(folder_id)
            
            retriever = self.retrievers[folder_id]
            
            # Get initial documents
            if hasattr(retriever, "aget_relevant_documents"):
                initial_docs = await retriever.aget_relevant_documents(query)
            else:
                initial_docs = retriever.get_relevant_documents(query)
            
            logger.debug("Initial retrieval", count=len(initial_docs))
            
            # Step 2: Expand via knowledge graph
            expanded_chunk_ids = set()
            
            # Add initial chunk IDs
            for doc in initial_docs:
                chunk_id = doc.metadata.get("chunk_id")
                if chunk_id:
                    expanded_chunk_ids.add(chunk_id)
            
            # Find related chunks via graph
            for doc in initial_docs[:5]:  # Limit to top 5 for performance
                chunk_id = doc.metadata.get("chunk_id")
                if chunk_id:
                    related = await self.graph_queries.find_related_chunks(
                        chunk_id=chunk_id,
                        folder_id=folder_id,
                        relationship_types=["cross_reference", "entity_overlap"],
                        max_depth=1,
                    )
                    
                    # Add related chunks (top 3 per doc)
                    for related_id, _ in related[:3]:
                        expanded_chunk_ids.add(related_id)
            
            logger.debug("Graph expansion", expanded_count=len(expanded_chunk_ids))
            
            # Step 3: Fetch expanded chunks from vector store
            # For now, we'll use the retriever to get documents by chunk_id
            # This is a simplified approach - in production, you'd query the vector store directly
            expanded_docs = list(initial_docs)  # Start with initial docs
            
            # Step 4: Rerank if enabled
            if self.use_reranking and self.reranker and expanded_docs:
                try:
                    scored_docs = self.reranker.rerank_documents(
                        query=query,
                        documents=expanded_docs[:k * 2],
                        top_k=k * 2,
                    )
                    expanded_docs = [doc for doc, _ in scored_docs]
                    logger.debug("Reranking completed", count=len(expanded_docs))
                except Exception as e:
                    logger.warning("Reranking failed, using original order", error=str(e))
            
            logger.info(
                "Graph-augmented retrieval completed",
                initial_count=len(initial_docs),
                expanded_count=len(expanded_docs),
                final_count=min(len(expanded_docs), k * 2),
            )
            
            return expanded_docs[:k * 2]  # Return top 2k
            
        except Exception as e:
            logger.error(
                "Graph-augmented retrieval failed",
                folder_id=folder_id,
                error=str(e),
                exc_info=True,
            )
            # Fallback to regular retrieval
            if folder_id in self.retrievers:
                retriever = self.retrievers[folder_id]
                if hasattr(retriever, "aget_relevant_documents"):
                    return await retriever.aget_relevant_documents(query)
                else:
                    return retriever.get_relevant_documents(query)
            return []

    def _convert_citations_to_dict(
        self, citation_objects: List[Any]
    ) -> List[Dict[str, Any]]:
        """Convert Citation objects to dictionary format for API compatibility.
        
        Args:
            citation_objects: List of Citation Pydantic objects from CitationEngine
            
        Returns:
            List of citation dictionaries compatible with existing API format
        """
        citations = []
        
        for citation in citation_objects:
            # Extract timestamp range if available
            timestamp_range = citation.timestamp_range
            start_time = timestamp_range[0] if timestamp_range else None
            end_time = timestamp_range[1] if timestamp_range else None
            
            # Extract line range if available
            line_range = citation.line_range
            
            citation_dict = {
                "file_id": citation.source_file_id,
                "file_name": citation.source_file_name,
                "chunk_id": citation.chunk_id,
                "page_number": citation.page_number,
                "start_time": start_time,
                "end_time": end_time,
                "line_range": line_range,
                "excerpt": citation.excerpt,
                "confidence": citation.confidence,
                "formatted": citation.formatted,
                "file_type": citation.file_type.value if hasattr(citation.file_type, "value") else str(citation.file_type),
                "content_preview": citation.excerpt[:200],  # Use excerpt as preview
                "relevance_score": citation.confidence,  # Use confidence as relevance
            }
            
            citations.append(citation_dict)
        
        return citations

    def _extract_citations(
        self, source_documents: List[Document]
    ) -> List[Dict[str, Any]]:
        """Extract citation information from source documents with metadata propagation.
        
        Fallback method when CitationEngine is not available or fails.
        Uses simple metadata extraction without sophisticated mapping.

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
                "start_time": metadata.get("start_time") or metadata.get("segment_start"),
                "end_time": metadata.get("end_time") or metadata.get("segment_end"),
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

