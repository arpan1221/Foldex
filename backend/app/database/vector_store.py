"""ChromaDB vector store interface with collection management."""

from typing import List, Optional, Dict, Any
from pathlib import Path
import structlog

try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

from app.config.settings import settings
from app.models.documents import DocumentChunk
from app.core.exceptions import VectorStoreError

logger = structlog.get_logger(__name__)


class VectorStore:
    """Manages vector database operations using ChromaDB.

    This class provides async-like interface for ChromaDB operations
    with proper collection management and error handling.
    """

    def __init__(self, persist_directory: Optional[str] = None):
        """Initialize vector store.

        Args:
            persist_directory: Directory to persist ChromaDB data

        Raises:
            VectorStoreError: If ChromaDB is not available or initialization fails
        """
        if not CHROMADB_AVAILABLE:
            raise VectorStoreError(
                "ChromaDB is not installed. Install with: pip install chromadb"
            )

        self.persist_directory = Path(
            persist_directory or settings.VECTOR_DB_PATH
        )
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.client: Optional[chromadb.PersistentClient] = None
        self.collection: Optional[chromadb.Collection] = None
        self._collection_name = "foldex_chunks"
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize ChromaDB client and collection.

        Raises:
            VectorStoreError: If initialization fails
        """
        try:
            logger.info(
                "Initializing ChromaDB client",
                path=str(self.persist_directory),
            )

            # Create persistent client
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )

            # Get or create collection
            try:
                self.collection = self.client.get_collection(
                    name=self._collection_name
                )
                logger.info(
                    "Using existing ChromaDB collection",
                    collection_name=self._collection_name,
                )
            except Exception:
                # Collection doesn't exist, create it
                self.collection = self.client.create_collection(
                    name=self._collection_name,
                    metadata={"description": "Foldex document chunks"},
                )
                logger.info(
                    "Created new ChromaDB collection",
                    collection_name=self._collection_name,
                )

            logger.info("Vector store initialized successfully")
        except Exception as e:
            logger.error(
                "Vector store initialization failed",
                error=str(e),
                exc_info=True,
            )
            raise VectorStoreError(
                f"Failed to initialize vector store: {str(e)}"
            ) from e

    async def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add chunks to vector store.

        Args:
            chunks: List of document chunks with embeddings

        Raises:
            VectorStoreError: If addition fails

        Example:
            >>> chunks = [DocumentChunk(..., embedding=[...]), ...]
            >>> await vector_store.add_chunks(chunks)
        """
        if not chunks:
            logger.warning("Attempted to add empty chunk list")
            return

        if not self.collection:
            raise VectorStoreError("Vector store not initialized")

        try:
            # Filter chunks with embeddings
            chunks_with_embeddings = [
                c for c in chunks if c.embedding is not None
            ]

            if not chunks_with_embeddings:
                logger.warning("No chunks with embeddings to add")
                return

            # Prepare data for ChromaDB
            ids = [chunk.chunk_id for chunk in chunks_with_embeddings]
            embeddings = [
                chunk.embedding for chunk in chunks_with_embeddings
            ]
            documents = [chunk.content for chunk in chunks_with_embeddings]
            metadatas = [
                {
                    "file_id": chunk.file_id,
                    "folder_id": chunk.metadata.get("folder_id", ""),
                    "file_name": chunk.metadata.get("file_name", ""),
                    "page_number": chunk.metadata.get("page_number"),
                    "timestamp": chunk.metadata.get("timestamp"),
                }
                for chunk in chunks_with_embeddings
            ]

            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )

            logger.info(
                "Chunks added to vector store",
                count=len(chunks_with_embeddings),
                collection_name=self._collection_name,
            )
        except Exception as e:
            logger.error(
                "Failed to add chunks",
                chunk_count=len(chunks),
                error=str(e),
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to add chunks: {str(e)}") from e

    async def similarity_search(
        self,
        query_embedding: List[float],
        folder_id: Optional[str] = None,
        k: int = 10,
    ) -> List[DocumentChunk]:
        """Search for similar chunks using vector similarity.

        Args:
            query_embedding: Query embedding vector
            folder_id: Optional folder ID to filter results
            k: Number of results to return

        Returns:
            List of similar document chunks

        Raises:
            VectorStoreError: If search fails

        Example:
            >>> embedding = [0.1, 0.2, ...]
            >>> results = await vector_store.similarity_search(embedding, "folder123", k=10)
        """
        if not self.collection:
            raise VectorStoreError("Vector store not initialized")

        try:
            # Build where clause for filtering
            where: Dict[str, Any] = {}
            if folder_id:
                where["folder_id"] = folder_id

            # Perform similarity search
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where if where else None,
            )

            # Convert results to DocumentChunk objects
            chunks = []
            if results["ids"] and len(results["ids"][0]) > 0:
                for i, chunk_id in enumerate(results["ids"][0]):
                    metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                    document = results["documents"][0][i] if results["documents"] else ""

                    chunks.append(
                        DocumentChunk(
                            chunk_id=chunk_id,
                            content=document,
                            file_id=metadata.get("file_id", ""),
                            metadata={
                                "file_name": metadata.get("file_name", ""),
                                "page_number": metadata.get("page_number"),
                                "timestamp": metadata.get("timestamp"),
                                "distance": results["distances"][0][i]
                                if results["distances"]
                                else None,
                            },
                        )
                    )

            logger.info(
                "Similarity search completed",
                query_length=len(query_embedding),
                folder_id=folder_id,
                results_count=len(chunks),
                k=k,
            )
            return chunks
        except Exception as e:
            logger.error(
                "Similarity search failed",
                folder_id=folder_id,
                k=k,
                error=str(e),
                exc_info=True,
            )
            raise VectorStoreError(f"Similarity search failed: {str(e)}") from e

    async def delete_chunks_by_folder(self, folder_id: str) -> int:
        """Delete all chunks for a folder.

        Args:
            folder_id: Folder ID

        Returns:
            Number of chunks deleted

        Raises:
            VectorStoreError: If deletion fails
        """
        if not self.collection:
            raise VectorStoreError("Vector store not initialized")

        try:
            # Get all chunk IDs for this folder
            results = self.collection.get(where={"folder_id": folder_id})
            chunk_ids = results["ids"] if results["ids"] else []

            if not chunk_ids:
                logger.info("No chunks found for folder", folder_id=folder_id)
                return 0

            # Delete chunks
            self.collection.delete(ids=chunk_ids)

            logger.info(
                "Chunks deleted",
                folder_id=folder_id,
                count=len(chunk_ids),
            )
            return len(chunk_ids)
        except Exception as e:
            logger.error(
                "Failed to delete chunks",
                folder_id=folder_id,
                error=str(e),
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to delete chunks: {str(e)}") from e

    async def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the collection.

        Returns:
            Dictionary with collection information

        Raises:
            VectorStoreError: If info retrieval fails
        """
        if not self.collection:
            raise VectorStoreError("Vector store not initialized")

        try:
            count = self.collection.count()
            return {
                "collection_name": self._collection_name,
                "chunk_count": count,
                "persist_directory": str(self.persist_directory),
            }
        except Exception as e:
            logger.error("Failed to get collection info", error=str(e))
            raise VectorStoreError(f"Failed to get collection info: {str(e)}") from e

    def close(self) -> None:
        """Close vector store connections.

        Note: ChromaDB PersistentClient manages its own connections.
        This method is provided for interface consistency.
        """
        # ChromaDB client handles cleanup automatically
        pass

