"""Maximal Marginal Relevance (MMR) for diversity-aware retrieval.

MMR selects documents that are both relevant to the query AND diverse from each other,
avoiding redundant results while maintaining high relevance.
"""

from typing import List, Optional, Tuple, Callable, Any
import numpy as np
import structlog

try:
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.schema import Document
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        Document = None

logger = structlog.get_logger(__name__)


class MMRRetriever:
    """Maximal Marginal Relevance retriever for diversity-aware document selection.
    
    MMR balances relevance to the query with diversity among selected documents.
    Formula: MMR(d) = λ * Sim(d, q) - (1-λ) * max(Sim(d, d_i)) for d_i in selected
    
    Where:
        - λ (lambda) controls trade-off: 0.0 = max diversity, 1.0 = max relevance
        - Sim(d, q) is similarity between document and query
        - max(Sim(d, d_i)) is maximum similarity to already selected documents
    """

    def __init__(
        self,
        lambda_param: float = 0.7,
        similarity_function: Optional[Callable[[Any, Any], float]] = None,
    ):
        """Initialize MMR retriever.

        Args:
            lambda_param: MMR lambda parameter (0.0-1.0)
                - 0.0: Maximum diversity (ignore relevance)
                - 1.0: Maximum relevance (ignore diversity)
                - 0.7: Balanced (default, slightly favors relevance)
            similarity_function: Optional function to compute document-document similarity.
                If None, uses cosine similarity of embeddings.
                Signature: (doc1: Document, doc2: Document) -> float
        """
        if not 0.0 <= lambda_param <= 1.0:
            raise ValueError(f"lambda_param must be between 0.0 and 1.0, got {lambda_param}")
        
        self.lambda_param = lambda_param
        self.similarity_function = similarity_function

    def select_with_mmr(
        self,
        query_embedding: List[float],
        candidate_documents: List[Tuple[Document, float]],
        k: int,
        document_embeddings: Optional[List[List[float]]] = None,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
    ) -> List[Document]:
        """Select k documents using Maximal Marginal Relevance.

        Args:
            query_embedding: Query embedding vector
            candidate_documents: List of (Document, relevance_score) tuples
            k: Number of documents to select
            document_embeddings: Optional list of embeddings for each candidate document.
                If None, will attempt to extract from document metadata.

        Returns:
            List of k selected documents ordered by MMR score
        """
        if not candidate_documents:
            return []
        
        if k >= len(candidate_documents):
            return [doc for doc, _ in candidate_documents]
        
        # Extract documents and their relevance scores
        docs = [doc for doc, _ in candidate_documents]
        relevance_scores = [score for _, score in candidate_documents]
        
        # Get embeddings for each document
        doc_embeddings = self._extract_embeddings(docs, document_embeddings, embedding_function)
        
        if not doc_embeddings:
            logger.warning("No embeddings available for MMR, falling back to relevance-only")
            return docs[:k]
        
        # Normalize embeddings for cosine similarity
        query_emb = np.array(query_embedding, dtype=np.float32)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        
        doc_embeddings_norm = []
        for emb in doc_embeddings:
            emb_arr = np.array(emb, dtype=np.float32)
            emb_arr = emb_arr / (np.linalg.norm(emb_arr) + 1e-8)
            doc_embeddings_norm.append(emb_arr)
        
        # Compute query-document similarities
        query_doc_similarities = [
            float(np.dot(query_emb, doc_emb))
            for doc_emb in doc_embeddings_norm
        ]
        
        # Greedy MMR selection
        selected_indices = []
        selected_docs = []
        
        # Start with most relevant document
        first_idx = max(range(len(docs)), key=lambda i: relevance_scores[i])
        selected_indices.append(first_idx)
        selected_docs.append(docs[first_idx])
        
        # Iteratively select documents with highest MMR score
        while len(selected_docs) < k and len(selected_docs) < len(docs):
            best_mmr = float('-inf')
            best_idx = None
            
            for i, doc in enumerate(docs):
                if i in selected_indices:
                    continue
                
                # Relevance component: similarity to query
                relevance = query_doc_similarities[i]
                
                # Diversity component: max similarity to already selected documents
                max_similarity_to_selected = 0.0
                if selected_indices:
                    similarities_to_selected = [
                        float(np.dot(doc_embeddings_norm[i], doc_embeddings_norm[j]))
                        for j in selected_indices
                    ]
                    max_similarity_to_selected = max(similarities_to_selected) if similarities_to_selected else 0.0
                
                # MMR score: λ * relevance - (1-λ) * max_similarity_to_selected
                mmr_score = (
                    self.lambda_param * relevance -
                    (1 - self.lambda_param) * max_similarity_to_selected
                )
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i
            
            if best_idx is not None:
                selected_indices.append(best_idx)
                selected_docs.append(docs[best_idx])
            else:
                break
        
        logger.debug(
            "MMR selection completed",
            selected_count=len(selected_docs),
            lambda_param=self.lambda_param,
            avg_relevance=sum(relevance_scores[i] for i in selected_indices) / len(selected_indices) if selected_indices else 0.0,
        )
        
        return selected_docs

    def _extract_embeddings(
        self,
        documents: List[Document],
        provided_embeddings: Optional[List[List[float]]] = None,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
    ) -> List[List[float]]:
        """Extract embeddings from documents.

        Args:
            documents: List of Document objects
            provided_embeddings: Optional pre-computed embeddings
            embedding_function: Optional function to compute embeddings from text.
                Signature: (text: str) -> List[float]

        Returns:
            List of embedding vectors
        """
        if provided_embeddings and len(provided_embeddings) == len(documents):
            return provided_embeddings
        
        # Try to extract from document metadata
        embeddings = []
        missing_indices = []
        
        for i, doc in enumerate(documents):
            if hasattr(doc, "metadata") and doc.metadata:
                # Check for embedding in metadata
                emb = doc.metadata.get("embedding")
                if emb and isinstance(emb, list):
                    embeddings.append(emb)
                else:
                    # No embedding available, will compute if function provided
                    embeddings.append(None)
                    missing_indices.append(i)
            else:
                embeddings.append(None)
                missing_indices.append(i)
        
        # Compute missing embeddings if function provided
        if missing_indices and embedding_function:
            try:
                for idx in missing_indices:
                    doc = documents[idx]
                    text = doc.page_content if hasattr(doc, "page_content") else str(doc)
                    emb = embedding_function(text)
                    embeddings[idx] = emb
                logger.debug(
                    "Computed missing embeddings for MMR",
                    computed_count=len(missing_indices),
                )
            except Exception as e:
                logger.warning(
                    "Failed to compute embeddings for MMR",
                    error=str(e),
                    missing_count=len(missing_indices),
                )
        
        # Filter out None embeddings
        valid_embeddings = [emb for emb in embeddings if emb is not None]
        
        if len(valid_embeddings) != len(documents):
            logger.warning(
                "Not all documents have embeddings for MMR",
                total_docs=len(documents),
                docs_with_embeddings=len(valid_embeddings),
            )
            return []
        
        return embeddings

    def select_with_mmr_from_scored_docs(
        self,
        query_embedding: List[float],
        scored_documents: List[Tuple[Document, float]],
        k: int,
        embedding_function: Optional[Callable[[str], List[float]]] = None,
    ) -> List[Document]:
        """Select documents using MMR from pre-scored documents.

        Convenience method that extracts embeddings from documents.

        Args:
            query_embedding: Query embedding vector
            scored_documents: List of (Document, relevance_score) tuples
            k: Number of documents to select
            embedding_function: Optional function to compute embeddings from text

        Returns:
            List of k selected documents
        """
        return self.select_with_mmr(
            query_embedding=query_embedding,
            candidate_documents=scored_documents,
            k=k,
            document_embeddings=None,
            embedding_function=embedding_function,
        )


# Global MMR retriever instance
_mmr_retriever: Optional[MMRRetriever] = None


def get_mmr_retriever(
    lambda_param: float = 0.7,
    similarity_function: Optional[Callable[[Any, Any], float]] = None,
) -> MMRRetriever:
    """Get global MMR retriever instance.

    Args:
        lambda_param: MMR lambda parameter (0.0-1.0)
        similarity_function: Optional custom similarity function

    Returns:
        MMRRetriever instance
    """
    global _mmr_retriever
    if _mmr_retriever is None or _mmr_retriever.lambda_param != lambda_param:
        _mmr_retriever = MMRRetriever(
            lambda_param=lambda_param,
            similarity_function=similarity_function,
        )
    return _mmr_retriever

