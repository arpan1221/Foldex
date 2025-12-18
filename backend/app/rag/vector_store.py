"""LangChain ChromaDB integration with HuggingFace embeddings."""

from typing import List, Optional, Dict, Any
from pathlib import Path
import structlog

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback for older LangChain versions
    try:
        from langchain.vectorstores import Chroma
        from langchain.embeddings import HuggingFaceEmbeddings
        from langchain.schema import Document
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        Chroma = None
        HuggingFaceEmbeddings = None
        Document = None

from app.config.settings import settings
from app.core.exceptions import VectorStoreError

logger = structlog.get_logger(__name__)


class LangChainVectorStore:
    """LangChain-based vector store using ChromaDB and HuggingFace embeddings.

    Provides async-compatible interface for LangChain's Chroma vector store
    with proper collection management and metadata tracking.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        collection_name: Optional[str] = None,
        embedding_model: Optional[str] = None,
    ):
        """Initialize LangChain vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data
            collection_name: Name of the ChromaDB collection
            embedding_model: HuggingFace model name for embeddings

        Raises:
            VectorStoreError: If LangChain/ChromaDB not available or initialization fails
        """
        if not LANGCHAIN_AVAILABLE:
            raise VectorStoreError(
                "LangChain is not installed. Install with: pip install langchain langchain-community"
            )

        self.persist_directory = Path(
            persist_directory or settings.VECTOR_DB_PATH
        )
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.collection_name = collection_name or "foldex_chunks"
        self.embedding_model_name = embedding_model or settings.EMBEDDING_MODEL

        # Initialize embeddings
        try:
            logger.info(
                "Initializing HuggingFace embeddings",
                model=self.embedding_model_name,
            )
            self.embeddings = HuggingFaceEmbeddings(
                model_name=self.embedding_model_name,
                model_kwargs={"device": "cpu"},  # Use CPU for local-first
                encode_kwargs={"normalize_embeddings": True},  # Normalize for better similarity
            )
            logger.info("HuggingFace embeddings initialized successfully")
        except Exception as e:
            logger.error(
                "Failed to initialize embeddings",
                model=self.embedding_model_name,
                error=str(e),
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to initialize embeddings: {str(e)}") from e

        # Initialize Chroma vector store
        try:
            logger.info(
                "Initializing LangChain Chroma vector store",
                persist_directory=str(self.persist_directory),
                collection_name=self.collection_name,
            )

            self.vector_store = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
            )

            logger.info("LangChain Chroma vector store initialized successfully")
        except Exception as e:
            logger.error(
                "Failed to initialize vector store",
                error=str(e),
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to initialize vector store: {str(e)}") from e

    async def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add documents to the vector store.

        Args:
            documents: List of LangChain Document objects
            ids: Optional list of document IDs (auto-generated if not provided)

        Returns:
            List of document IDs

        Raises:
            VectorStoreError: If adding documents fails
        """
        try:
            logger.debug(
                "Adding documents to vector store",
                document_count=len(documents),
            )

            # LangChain Chroma.add_documents is synchronous but we wrap it for async compatibility
            result_ids = self.vector_store.add_documents(
                documents=documents,
                ids=ids,
            )

            # Persist to disk
            self.vector_store.persist()

            logger.info(
                "Documents added to vector store",
                count=len(result_ids),
            )

            return result_ids

        except Exception as e:
            logger.error(
                "Failed to add documents to vector store",
                error=str(e),
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to add documents: {str(e)}") from e

    async def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add texts to the vector store.

        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of document IDs

        Returns:
            List of document IDs

        Raises:
            VectorStoreError: If adding texts fails
        """
        try:
            logger.debug("Adding texts to vector store", text_count=len(texts))

            result_ids = self.vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                ids=ids,
            )

            # Persist to disk
            self.vector_store.persist()

            logger.info("Texts added to vector store", count=len(result_ids))

            return result_ids

        except Exception as e:
            logger.error(
                "Failed to add texts to vector store",
                error=str(e),
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to add texts: {str(e)}") from e

    async def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Search for similar documents.

        Args:
            query: Search query text
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of similar Document objects

        Raises:
            VectorStoreError: If search fails
        """
        try:
            logger.debug("Performing similarity search", query=query[:50], k=k)

            if filter:
                results = self.vector_store.similarity_search(
                    query=query,
                    k=k,
                    filter=filter,
                )
            else:
                results = self.vector_store.similarity_search(query=query, k=k)

            logger.debug("Similarity search completed", result_count=len(results))

            return results

        except Exception as e:
            logger.error(
                "Similarity search failed",
                error=str(e),
                exc_info=True,
            )
            raise VectorStoreError(f"Similarity search failed: {str(e)}") from e

    async def similarity_search_with_score(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[tuple[Document, float]]:
        """Search for similar documents with similarity scores.

        Args:
            query: Search query text
            k: Number of results to return
            filter: Optional metadata filter

        Returns:
            List of tuples (Document, similarity_score)

        Raises:
            VectorStoreError: If search fails
        """
        try:
            logger.debug(
                "Performing similarity search with scores",
                query=query[:50],
                k=k,
            )

            if filter:
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=k,
                    filter=filter,
                )
            else:
                results = self.vector_store.similarity_search_with_score(
                    query=query, k=k
                )

            logger.debug(
                "Similarity search with scores completed",
                result_count=len(results),
            )

            return results

        except Exception as e:
            logger.error(
                "Similarity search with scores failed",
                error=str(e),
                exc_info=True,
            )
            raise VectorStoreError(
                f"Similarity search with scores failed: {str(e)}"
            ) from e

    async def delete(
        self,
        ids: Optional[List[str]] = None,
        filter: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Delete documents from the vector store.

        Args:
            ids: Optional list of document IDs to delete
            filter: Optional metadata filter for deletion

        Raises:
            VectorStoreError: If deletion fails
        """
        try:
            logger.debug("Deleting documents from vector store", ids=ids)

            if ids:
                self.vector_store.delete(ids=ids)
            elif filter:
                self.vector_store.delete(filter=filter)
            else:
                raise VectorStoreError("Must provide either ids or filter for deletion")

            # Persist to disk
            self.vector_store.persist()

            logger.info("Documents deleted from vector store")

        except Exception as e:
            logger.error(
                "Failed to delete documents from vector store",
                error=str(e),
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to delete documents: {str(e)}") from e

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the vector store collection.

        Returns:
            Dictionary with collection information

        Raises:
            VectorStoreError: If getting info fails
        """
        try:
            collection = self.vector_store._collection
            count = collection.count()

            return {
                "collection_name": self.collection_name,
                "document_count": count,
                "embedding_model": self.embedding_model_name,
                "persist_directory": str(self.persist_directory),
            }

        except Exception as e:
            logger.error(
                "Failed to get collection info",
                error=str(e),
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to get collection info: {str(e)}") from e

    def persist(self) -> None:
        """Persist vector store to disk."""
        try:
            self.vector_store.persist()
            logger.debug("Vector store persisted to disk")
        except Exception as e:
            logger.error("Failed to persist vector store", error=str(e))
            raise VectorStoreError(f"Failed to persist vector store: {str(e)}") from e

