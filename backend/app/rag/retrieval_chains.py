"""LangChain RetrievalQA chains with custom retrievers."""

from typing import Optional, Dict, Any, List, Callable
import structlog

try:
    from langchain.chains import RetrievalQA
    from langchain.chains.question_answering import load_qa_chain
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback for older LangChain versions
    try:
        from langchain.chains import RetrievalQA
        from langchain.chains.question_answering import load_qa_chain
        from langchain.schema import BaseRetriever, Document
        from langchain_core.runnables import RunnablePassthrough, RunnableParallel
        from langchain_core.output_parsers import StrOutputParser
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        RetrievalQA = None
        load_qa_chain = None
        BaseRetriever = None
        Document = None
        RunnablePassthrough = None
        RunnableParallel = None
        StrOutputParser = None

from app.core.exceptions import ProcessingError
from app.rag.prompt_management import PromptManager, get_prompt_manager
from app.rag.llm_chains import OllamaLLM
from app.rag.ttft_optimization import get_ttft_optimizer

logger = structlog.get_logger(__name__)


class RetrievalQAChain:
    """LangChain RetrievalQA chain with custom retriever and prompt management."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: Optional[OllamaLLM] = None,
        prompt_manager: Optional[PromptManager] = None,
        chain_type: str = "stuff",
        return_source_documents: bool = True,
        enable_ttft_optimization: bool = True,
    ):
        """Initialize RetrievalQA chain.

        Args:
            retriever: LangChain retriever instance
            llm: Ollama LLM instance
            prompt_manager: Prompt manager instance
            chain_type: Chain type ("stuff", "map_reduce", "refine", "map_rerank")
            return_source_documents: Whether to return source documents
            enable_ttft_optimization: Enable TTFT optimizations

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
        self.chain_type = chain_type
        self.return_source_documents = return_source_documents
        self.enable_ttft_optimization = enable_ttft_optimization

        # TTFT optimization
        self.ttft_optimizer = get_ttft_optimizer() if enable_ttft_optimization else None

        # Initialize chain
        self.chain = None
        self._initialize_chain()

    def _initialize_chain(self) -> None:
        """Initialize retrieval chain using LCEL for better streaming."""
        try:
            # Get default prompt
            prompt = self.prompt_manager.get_prompt("default")

            # Create format_docs function with TTFT optimization
            def format_docs_factory(query_context=None):
                """Factory to create format_docs with query context."""
                def format_docs(docs):
                    # Apply TTFT context optimization if enabled
                    optimized_docs = docs
                    if self.ttft_optimizer and query_context:
                        optimized_docs = self.ttft_optimizer.optimize_context(docs, query_context)
                    return "\n\n".join(doc.page_content for doc in optimized_docs)
                return format_docs

            # Define the LCEL chain
            # We use RunnableParallel to get both the answer and the source documents
            self.chain = RunnableParallel({
                "context": self.retriever | format_docs_factory(),
                "question": RunnablePassthrough(),
                "source_documents": self.retriever
            }) | {
                "answer": (
                    RunnablePassthrough.assign(
                        context=lambda x: x["context"],
                        question=lambda x: x["question"]
                    ) | prompt | self.llm.get_llm() | StrOutputParser()
                ),
                "source_documents": lambda x: x["source_documents"]
            }

            logger.info(
                "Initialized LCEL retrieval chain",
                chain_type="lcel",
                ttft_optimization=self.enable_ttft_optimization,
            )

        except Exception as e:
            logger.error(
                "Failed to initialize LCEL retrieval chain",
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(
                f"Failed to initialize retrieval chain: {str(e)}"
            ) from e

    async def invoke(
        self,
        query: str,
        query_type: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Invoke chain with query.

        Args:
            query: User query
            query_type: Optional query type for prompt selection
            streaming_callback: Optional callback for streaming tokens

        Returns:
            Dictionary with answer and source documents

        Raises:
            ProcessingError: If chain invocation fails
        """
        try:
            # TTFT Optimization: Create optimized format_docs function
            def format_docs(docs):
                # Apply context optimization if enabled
                if self.ttft_optimizer:
                    docs = self.ttft_optimizer.optimize_context(docs, query)
                    logger.debug(
                        "Applied TTFT context optimization",
                        original_count=len(docs),
                        optimized_count=len(docs)
                    )
                return "\n\n".join(doc.page_content for doc in docs)

            # Update prompt if query type is different
            if query_type and query_type != "default":
                prompt = self.prompt_manager.get_prompt(query_type)

                # Reinitialize chain with new prompt
                self.chain = RunnableParallel({
                    "context": self.retriever | format_docs,
                    "question": RunnablePassthrough(),
                    "source_documents": self.retriever
                }) | {
                    "answer": (
                        RunnablePassthrough.assign(
                            context=lambda x: x["context"],
                            question=lambda x: x["question"]
                        ) | prompt | self.llm.get_llm() | StrOutputParser()
                    ),
                    "source_documents": lambda x: x["source_documents"]
                }

            # Set up streaming if callback provided
            callbacks = []
            if streaming_callback:
                from app.rag.llm_chains import StreamingCallbackHandler
                callbacks.append(StreamingCallbackHandler(streaming_callback))

            # Invoke chain using astream for better streaming performance
            if self.chain is None:
                raise ProcessingError("Chain is not initialized")
            
            final_result = {
                "answer": "",
                "source_documents": []
            }

            # LCEL astream allows us to get tokens as they are generated
            async for chunk in self.chain.astream(query, config={"callbacks": callbacks}):
                if "answer" in chunk:
                    # Token found in chunk
                    token = chunk["answer"]
                    final_result["answer"] += token
                    # Note: We rely on the callbacks in config to handle real-time streaming 
                    # to the frontend. No manual callback trigger needed here.
                            
                if "source_documents" in chunk:
                    # Source documents found in chunk (usually at the start or end)
                    final_result["source_documents"].extend(chunk["source_documents"])

            logger.info(
                "Chain streaming completed",
                query_length=len(query),
                answer_length=len(final_result["answer"]),
                source_count=len(final_result["source_documents"]),
            )

            return {
                "answer": final_result["answer"],
                "source_documents": final_result["source_documents"],
                "query_type": query_type,
            }

        except Exception as e:
            logger.error(
                "Chain invocation failed",
                query=query[:100],
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(f"Chain invocation failed: {str(e)}") from e

    def get_chain(self) -> RetrievalQA:
        """Get the underlying LangChain RetrievalQA chain.

        Returns:
            RetrievalQA chain instance
        """
        return self.chain


class HybridRetrievalChain:
    """Hybrid retrieval chain combining multiple retrieval strategies."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: Optional[OllamaLLM] = None,
        prompt_manager: Optional[PromptManager] = None,
        chain_type: str = "stuff",
    ):
        """Initialize hybrid retrieval chain.

        Args:
            retriever: Hybrid retriever instance
            llm: Ollama LLM instance
            prompt_manager: Prompt manager instance
            chain_type: Chain type
        """
        self.retrieval_qa_chain = RetrievalQAChain(
            retriever=retriever,
            llm=llm,
            prompt_manager=prompt_manager,
            chain_type=chain_type,
        )

    async def invoke(
        self,
        query: str,
        query_type: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Invoke hybrid retrieval chain.

        Args:
            query: User query
            query_type: Optional query type
            streaming_callback: Optional streaming callback

        Returns:
            Dictionary with answer and sources
        """
        return await self.retrieval_qa_chain.invoke(
            query=query,
            query_type=query_type,
            streaming_callback=streaming_callback,
        )

    def _extract_citations(self, source_documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract citation information from source documents.

        Args:
            source_documents: List of source Document objects

        Returns:
            List of citation dictionaries with metadata
        """
        citations = []

        for doc in source_documents:
            metadata = doc.metadata if hasattr(doc, "metadata") else {}

            citation = {
                "file_id": metadata.get("file_id"),
                "file_name": metadata.get("file_name", "Unknown"),
                "chunk_id": metadata.get("chunk_id"),
                "page_number": metadata.get("page_number"),
                "chunk_index": metadata.get("chunk_index"),
                "start_time": metadata.get("start_time"),
                "end_time": metadata.get("end_time"),
                "content_preview": doc.page_content[:200] if hasattr(doc, "page_content") else str(doc)[:200],
                "metadata": metadata,
            }

            citations.append(citation)

        return citations

