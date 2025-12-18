"""LangChain-compatible reranking and relevance scoring."""

from typing import List, Optional, Any, Tuple
import structlog

try:
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback for older LangChain versions
    try:
        from langchain.schema import Document, BaseRetriever
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        Document = None
        BaseRetriever = None

from app.core.exceptions import ProcessingError
from app.models.documents import DocumentChunk

logger = structlog.get_logger(__name__)


class LangChainReranker:
    """LangChain-compatible reranker for relevance scoring and filtering."""

    def __init__(
        self,
        use_cross_encoder: bool = True,
        model_name: Optional[str] = None,
    ):
        """Initialize reranker.

        Args:
            use_cross_encoder: Whether to use cross-encoder model for reranking
            model_name: Optional cross-encoder model name

        Raises:
            ProcessingError: If initialization fails
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        self.use_cross_encoder = use_cross_encoder
        self.model_name = model_name or "cross-encoder/ms-marco-MiniLM-L-6-v2"

        # Initialize cross-encoder if enabled
        self.cross_encoder = None
        if use_cross_encoder:
            try:
                from sentence_transformers import CrossEncoder
                self.cross_encoder = CrossEncoder(self.model_name)
                logger.info(
                    "Initialized cross-encoder reranker",
                    model=self.model_name,
                )
            except ImportError:
                logger.warning(
                    "CrossEncoder not available, using simple reranking",
                )
                self.use_cross_encoder = False
            except Exception as e:
                logger.warning(
                    "Failed to initialize cross-encoder, using simple reranking",
                    error=str(e),
                )
                self.use_cross_encoder = False

    def rerank_documents(
        self,
        query: str,
        documents: List[Document],
        top_k: int = 10,
    ) -> List[Tuple[Document, float]]:
        """Rerank documents by relevance to query.

        Args:
            query: Search query
            documents: List of Document objects to rerank
            top_k: Number of top documents to return

        Returns:
            List of tuples (Document, relevance_score) sorted by score
        """
        try:
            if not documents:
                return []

            logger.debug(
                "Reranking documents",
                query_length=len(query),
                document_count=len(documents),
                top_k=top_k,
            )

            if self.use_cross_encoder and self.cross_encoder:
                # Use cross-encoder for reranking
                scored_documents = self._rerank_with_cross_encoder(
                    query=query,
                    documents=documents,
                )
            else:
                # Use simple scoring based on keyword matching
                scored_documents = self._rerank_with_keywords(
                    query=query,
                    documents=documents,
                )

            # Sort by score (descending) and return top_k
            scored_documents.sort(key=lambda x: x[1], reverse=True)
            top_documents = scored_documents[:top_k]

            logger.debug(
                "Reranking completed",
                top_k=len(top_documents),
                max_score=top_documents[0][1] if top_documents else 0.0,
            )

            return top_documents

        except Exception as e:
            logger.error(
                "Reranking failed",
                error=str(e),
                exc_info=True,
            )
            # Return documents with default scores on error
            return [(doc, 0.5) for doc in documents[:top_k]]

    def _rerank_with_cross_encoder(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Tuple[Document, float]]:
        """Rerank using cross-encoder model.

        Args:
            query: Search query
            documents: List of documents

        Returns:
            List of (Document, score) tuples
        """
        try:
            # Prepare pairs for cross-encoder
            pairs = []
            for doc in documents:
                content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                pairs.append([query, content])

            # Score pairs
            if self.cross_encoder is None:
                raise ProcessingError("Cross-encoder not initialized")
            scores = self.cross_encoder.predict(pairs)

            # Combine documents with scores
            scored_documents = [
                (doc, float(score)) for doc, score in zip(documents, scores)
            ]

            return scored_documents

        except Exception as e:
            logger.error(
                "Cross-encoder reranking failed",
                error=str(e),
                exc_info=True,
            )
            # Fallback to keyword reranking
            return self._rerank_with_keywords(query, documents)

    def _rerank_with_keywords(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Tuple[Document, float]]:
        """Rerank using simple keyword matching.

        Args:
            query: Search query
            documents: List of documents

        Returns:
            List of (Document, score) tuples
        """
        query_lower = query.lower()
        query_words = set(query_lower.split())

        scored_documents = []

        for doc in documents:
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            content_lower = content.lower()

            # Calculate keyword match score
            matches = sum(1 for word in query_words if word in content_lower)
            score = matches / len(query_words) if query_words else 0.0

            # Boost score if query appears as phrase
            if query_lower in content_lower:
                score += 0.3

            scored_documents.append((doc, score))

        return scored_documents

    async def rerank_document_chunks(
        self,
        query: str,
        chunks: List[DocumentChunk],
        top_k: int = 10,
    ) -> List[DocumentChunk]:
        """Rerank DocumentChunk objects.

        Args:
            query: Search query
            chunks: List of DocumentChunk objects
            top_k: Number of top chunks to return

        Returns:
            List of reranked DocumentChunk objects
        """
        try:
            # Convert DocumentChunks to LangChain Documents
            from app.models.documents import document_chunk_to_langchain

            documents = [document_chunk_to_langchain(chunk) for chunk in chunks]

            # Rerank
            scored_documents = self.rerank_documents(
                query=query,
                documents=documents,
                top_k=top_k,
            )

            # Convert back to DocumentChunks and update scores in metadata
            reranked_chunks = []
            for doc, score in scored_documents:
                # Find original chunk
                chunk_id = doc.metadata.get("chunk_id")
                original_chunk = next(
                    (c for c in chunks if c.chunk_id == chunk_id), None
                )

                if original_chunk:
                    # Update metadata with relevance score
                    original_chunk.metadata["relevance_score"] = score
                    reranked_chunks.append(original_chunk)

            logger.info(
                "Reranked document chunks",
                original_count=len(chunks),
                reranked_count=len(reranked_chunks),
            )

            return reranked_chunks

        except Exception as e:
            logger.error(
                "Failed to rerank document chunks",
                error=str(e),
                exc_info=True,
            )
            # Return original chunks on error
            return chunks[:top_k]


class RerankingRetriever(BaseRetriever):
    """LangChain retriever wrapper that applies reranking to results."""
    
    # Declare fields as class variables for Pydantic
    base_retriever: Any  # BaseRetriever instance
    reranker: Any  # LangChainReranker instance
    top_k: int = 10

    def __init__(
        self,
        base_retriever: BaseRetriever,
        reranker: LangChainReranker,
        top_k: int = 10,
        **kwargs,
    ):
        """Initialize reranking retriever.

        Args:
            base_retriever: Base retriever to wrap
            reranker: Reranker instance
            top_k: Number of top documents to return after reranking
            **kwargs: Additional arguments passed to BaseRetriever
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        # Initialize Pydantic model properly
        init_values = {
            "base_retriever": base_retriever,
            "reranker": reranker,
            "top_k": top_k,
            **kwargs,
        }
        super().__init__(**init_values)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[Any] = None,
    ) -> List[Document]:
        """Get relevant documents with reranking.

        Args:
            query: Search query
            run_manager: Optional callback manager

        Returns:
            List of reranked Document objects
        """
        try:
            # Get documents from base retriever
            documents = self.base_retriever.get_relevant_documents(query)

            # Rerank documents
            scored_documents = self.reranker.rerank_documents(
                query=query,
                documents=documents,
                top_k=self.top_k,
            )

            # Return just the documents (without scores)
            return [doc for doc, _ in scored_documents]

        except Exception as e:
            logger.error(
                "Reranking retriever failed",
                query=query[:50],
                error=str(e),
                exc_info=True,
            )
            # Fallback to base retriever results
            return self.base_retriever.get_relevant_documents(query)[:self.top_k]

    async def aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[Any] = None,
    ) -> List[Document]:
        """Async version of get_relevant_documents.

        Args:
            query: Search query
            run_manager: Optional callback manager

        Returns:
            List of reranked Document objects
        """
        try:
            # Get documents from base retriever
            if hasattr(self.base_retriever, "aget_relevant_documents"):
                documents = await self.base_retriever.aget_relevant_documents(query)
            else:
                documents = self.base_retriever.get_relevant_documents(query)

            # Rerank documents
            scored_documents = self.reranker.rerank_documents(
                query=query,
                documents=documents,
                top_k=self.top_k,
            )

            # Return just the documents (without scores)
            return [doc for doc, _ in scored_documents]

        except Exception as e:
            logger.error(
                "Async reranking retriever failed",
                query=query[:50],
                error=str(e),
                exc_info=True,
            )
            # Fallback to base retriever results
            if hasattr(self.base_retriever, "aget_relevant_documents"):
                return (await self.base_retriever.aget_relevant_documents(query))[:self.top_k]
            else:
                return self.base_retriever.get_relevant_documents(query)[:self.top_k]

