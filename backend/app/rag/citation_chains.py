"""LangChain chains with automatic citation generation."""

from typing import Optional, Dict, Any, List, Callable
import structlog

try:
    from langchain.chains import RetrievalQAWithSourcesChain
    from langchain.chains.question_answering import load_qa_chain
    from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.chains import RetrievalQAWithSourcesChain
        from langchain.chains.question_answering import load_qa_chain
        from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
        from langchain.schema import BaseRetriever, Document
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        RetrievalQAWithSourcesChain = None
        load_qa_chain = None
        ConversationalRetrievalChain = None
        BaseRetriever = None
        Document = None

from app.core.exceptions import ProcessingError
from app.rag.llm_chains import OllamaLLM
from app.rag.prompt_management import PromptManager, get_prompt_manager
from app.rag.source_tracking import SourceTracker
from app.rag.citation_callbacks import CitationCallbackHandler, SourceTrackingCallbackHandler

logger = structlog.get_logger(__name__)


class CitationChain:
    """LangChain chain with automatic citation generation."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: Optional[OllamaLLM] = None,
        prompt_manager: Optional[PromptManager] = None,
        return_source_documents: bool = True,
    ):
        """Initialize citation chain.

        Args:
            retriever: LangChain retriever instance
            llm: Ollama LLM instance
            prompt_manager: Prompt manager instance
            return_source_documents: Whether to return source documents

        Raises:
            ProcessingError: If LangChain is not available or initialization fails
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        self.retriever = retriever
        self.llm = llm or OllamaLLM()
        self.prompt_manager = prompt_manager or get_prompt_manager()
        self.return_source_documents = return_source_documents

        # Initialize source tracking
        self.source_tracker = SourceTracker()
        self.citation_callback = CitationCallbackHandler(source_tracker=self.source_tracker)

        # Initialize chain
        self.chain = None
        self._initialize_chain()

    def _initialize_chain(self) -> None:
        """Initialize RetrievalQAWithSourcesChain."""
        try:
            # Get default prompt
            prompt = self.prompt_manager.get_prompt("default")

            # Create RetrievalQAWithSourcesChain
            self.chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm=self.llm.get_llm(),
                chain_type="stuff",
                retriever=self.retriever,
                return_source_documents=self.return_source_documents,
                chain_type_kwargs={"prompt": prompt},
            )

            logger.info("Initialized citation chain with source tracking")

        except Exception as e:
            logger.error(
                "Failed to initialize citation chain",
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(f"Failed to initialize chain: {str(e)}") from e

    async def invoke(
        self,
        question: str,
        query_type: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Invoke chain with question and generate citations.

        Args:
            question: User question
            query_type: Optional query type for prompt selection
            streaming_callback: Optional callback for streaming tokens

        Returns:
            Dictionary with answer, sources, and citations

        Raises:
            ProcessingError: If chain invocation fails
        """
        try:
            # Update prompt if query type is different
            if query_type and query_type != "default":
                prompt = self.prompt_manager.get_prompt(query_type)
                # Reinitialize chain with new prompt
                self.chain = RetrievalQAWithSourcesChain.from_chain_type(
                    llm=self.llm.get_llm(),
                    chain_type="stuff",
                    retriever=self.retriever,
                    return_source_documents=self.return_source_documents,
                    chain_type_kwargs={"prompt": prompt},
                )

            # Clear previous citations
            self.citation_callback.clear_citations()

            # Set up callbacks
            callbacks = [self.citation_callback]
            if streaming_callback:
                from app.rag.llm_chains import StreamingCallbackHandler
                callbacks.append(StreamingCallbackHandler(streaming_callback))

            # Prepare input
            chain_input = {"question": question}

            # Invoke chain
            if self.chain is None:
                raise ProcessingError("Chain is not initialized")

            result = self.chain.invoke(chain_input, callbacks=callbacks)

            # Extract citations
            citations = self.citation_callback.get_citations()

            # If no citations from callback, extract from result
            if not citations and "source_documents" in result:
                citations = self._extract_citations_from_documents(
                    result["source_documents"]
                )

            logger.info(
                "Citation chain invocation completed",
                answer_length=len(result.get("answer", "")),
                citation_count=len(citations),
            )

            return {
                "answer": result.get("answer", ""),
                "sources": result.get("sources", ""),
                "source_documents": result.get("source_documents", []),
                "citations": citations,
            }

        except Exception as e:
            logger.error(
                "Citation chain invocation failed",
                question=question[:100],
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(f"Chain invocation failed: {str(e)}") from e

    def _extract_citations_from_documents(
        self,
        documents: List[Document],
    ) -> List[Dict[str, Any]]:
        """Extract citations from source documents.

        Args:
            documents: List of source Document objects

        Returns:
            List of citation dictionaries
        """
        citations = []

        for doc in documents:
            if hasattr(doc, "metadata"):
                citation = {
                    "file_id": doc.metadata.get("file_id"),
                    "file_name": doc.metadata.get("file_name", "Unknown"),
                    "chunk_id": doc.metadata.get("chunk_id"),
                    "page_number": doc.metadata.get("page_number"),
                    "chunk_index": doc.metadata.get("chunk_index"),
                    "start_time": doc.metadata.get("start_time"),
                    "end_time": doc.metadata.get("end_time"),
                    "content_preview": doc.page_content[:200] if hasattr(doc, "page_content") else str(doc)[:200],
                    "metadata": doc.metadata,
                }

                # Calculate reliability
                source_info = {
                    "retrieval_score": doc.metadata.get("retrieval_score"),
                    "file_name": citation["file_name"],
                    "page_number": citation["page_number"],
                    "mime_type": doc.metadata.get("mime_type"),
                }
                citation["reliability"] = self.source_tracker.calculate_source_reliability(
                    source_info
                )

                citations.append(citation)

        return citations


class ConversationalCitationChain:
    """ConversationalRetrievalChain enhanced with source tracking."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: Optional[OllamaLLM] = None,
        memory: Optional[Any] = None,
        prompt_manager: Optional[PromptManager] = None,
    ):
        """Initialize conversational citation chain.

        Args:
            retriever: LangChain retriever instance
            llm: Ollama LLM instance
            memory: Conversation memory instance
            prompt_manager: Prompt manager instance
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        self.retriever = retriever
        self.llm = llm or OllamaLLM()
        self.memory = memory
        self.prompt_manager = prompt_manager or get_prompt_manager()

        # Initialize source tracking
        self.source_tracker = SourceTracker()
        self.source_callback = SourceTrackingCallbackHandler()

        # Initialize chain
        self.chain = None
        self._initialize_chain()

    def _initialize_chain(self) -> None:
        """Initialize ConversationalRetrievalChain with source tracking."""
        try:
            from langchain.chains import ConversationalRetrievalChain

            # Get default prompt
            prompt = self.prompt_manager.get_prompt("default")

            # Create conversational chain
            chain_kwargs = {
                "llm": self.llm.get_llm(),
                "retriever": self.retriever,
                "return_source_documents": True,
                "combine_docs_chain_kwargs": {"prompt": prompt},
            }

            if self.memory:
                chain_kwargs["memory"] = self.memory.get_memory() if hasattr(self.memory, "get_memory") else self.memory

            self.chain = ConversationalRetrievalChain.from_llm(**chain_kwargs)

            logger.info("Initialized conversational citation chain")

        except Exception as e:
            logger.error(
                "Failed to initialize conversational citation chain",
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(
                f"Failed to initialize chain: {str(e)}"
            ) from e

    async def invoke(
        self,
        question: str,
        query_type: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Invoke chain with question and track sources.

        Args:
            question: User question
            query_type: Optional query type
            streaming_callback: Optional streaming callback

        Returns:
            Dictionary with answer and citations
        """
        try:
            # Clear previous tracking
            self.source_callback.clear_tracking()

            # Set up callbacks
            callbacks = [self.source_callback]
            if streaming_callback:
                from app.rag.llm_chains import StreamingCallbackHandler
                callbacks.append(StreamingCallbackHandler(streaming_callback))

            # Invoke chain
            if self.chain is None:
                raise ProcessingError("Chain is not initialized")

            # Prepare input
            chain_input = {"question": question}

            # Invoke chain (synchronous but wrapped for async compatibility)
            result = self.chain.invoke(chain_input, callbacks=callbacks)

            # Get tracked sources
            tracked_sources = self.source_callback.get_tracked_sources()

            # Generate citations
            citations = []
            for source in tracked_sources:
                citation = {
                    "file_id": source.get("file_id"),
                    "file_name": source.get("file_name", "Unknown"),
                    "chunk_id": source.get("document_id"),
                    "page_number": source.get("page_number"),
                    "reliability": source.get("reliability", 0.5),
                    "retrieval_score": source.get("retrieval_score"),
                    "metadata": source.get("metadata", {}),
                }
                citations.append(citation)

            return {
                "answer": result.get("answer", ""),
                "source_documents": result.get("source_documents", []),
                "citations": citations,
                "tracked_sources": tracked_sources,
            }

        except Exception as e:
            logger.error(
                "Conversational citation chain invocation failed",
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(f"Chain invocation failed: {str(e)}") from e

