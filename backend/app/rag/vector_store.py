"""LangChain ChromaDB integration with Ollama or HuggingFace embeddings.

Enhanced for multimodal chunk storage with:
- Element types from Unstructured
- Timestamp ranges for audio segments
- Page numbers for documents
- Source file IDs for citations
- Enhanced metadata filtering
- Recursive chunk splitting for long texts
- BM25 indexing preparation for hybrid search
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
from datetime import datetime
import structlog
import shutil

try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback for older LangChain versions
    try:
        from langchain.vectorstores import Chroma
        from langchain.embeddings import HuggingFaceEmbeddings, OllamaEmbeddings
        from langchain.schema import Document
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        Chroma = None
        HuggingFaceEmbeddings = None
        OllamaEmbeddings = None
        Document = None

from app.config.settings import settings
from app.core.exceptions import VectorStoreError
from app.utils.embedding_cache import get_embedding_cache
from app.models.documents import DocumentChunk
from app.rag.reranking import LangChainReranker

logger = structlog.get_logger(__name__)

# Token estimation: ~4 characters per token (conservative)
CHARS_PER_TOKEN = 4
MAX_CHUNK_TOKENS = 8000  # Maximum tokens before recursive splitting
CHUNK_OVERLAP_TOKENS = 200  # Overlap between chunks when splitting


def _distance_to_similarity(distance: float, distance_metric: Optional[str] = None) -> float:
    """Convert distance score to similarity score (0-1).
    
    ChromaDB returns distance scores. The conversion depends on the distance metric.
    If distance_metric is None, auto-detects based on distance value:
    - If distance > 10: assumes L2 (Euclidean) distance
    - Otherwise: assumes cosine distance
    
    For cosine distance:
    - Cosine distance = 1 - cosine_similarity
    - Range: 0 (identical) to 2 (opposite), typically 0-1 for normalized embeddings
    - Conversion: similarity = 1 - distance
    
    For L2 (Euclidean) distance:
    - Lower distance = more similar
    - Range: 0 (identical) to infinity (very different)
    - Conversion: similarity = 1 / (1 + distance)
    
    Args:
        distance: Distance score from ChromaDB
        distance_metric: Distance metric used ("cosine" or "l2"). If None, auto-detects.
        
    Returns:
        Similarity score between 0.0 and 1.0 (1.0 = most similar, 0.0 = least similar)
    """
    # Auto-detect if not specified: L2 distances are typically >> 1, cosine distances are 0-2
    if distance_metric is None:
        distance_metric = "l2" if distance > 10.0 else "cosine"
    
    if distance_metric == "cosine":
        # Cosine distance: 0 = identical, 1 = orthogonal, 2 = opposite
        # For normalized embeddings, typically in [0, 1] range
        # Convert: similarity = 1 - distance
        return max(0.0, min(1.0, 1.0 - distance))
    else:  # L2 distance
        # L2 distance: 0 = identical, higher = less similar
        # Convert: similarity = 1 / (1 + distance)
        # This normalizes large distances to a 0-1 range
        return 1.0 / (1.0 + distance)


class LangChainVectorStore:
    """LangChain-based vector store using ChromaDB with Ollama or HuggingFace embeddings.

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
            embedding_model: Embedding model name (Ollama or HuggingFace)

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
        self.embedding_type = settings.EMBEDDING_TYPE

        # Initialize embedding cache
        self.cache_enabled = settings.EMBEDDING_CACHE_ENABLED
        if self.cache_enabled:
            self.embedding_cache = get_embedding_cache()  # type: ignore
            logger.info("Embedding cache enabled for query optimization")
        else:
            self.embedding_cache = None  # type: ignore
            logger.info("Embedding cache disabled")
        
        # BM25 index for hybrid search (lazy initialization)
        self._bm25_index: Optional[Any] = None
        self._bm25_documents: List[Document] = []


        # Initialize reranker if enabled
        self.use_reranking = settings.USE_RERANKING
        self.reranker = None
        if self.use_reranking:
            try:
                logger.info("Initializing reranker for vector store")
                self.reranker = LangChainReranker(
                    use_cross_encoder=True,
                    model_name=settings.RERANKER_MODEL,
                )
            except Exception as e:
                logger.warning(
                    "Failed to initialize reranker, continuing without reranking",
                    error=str(e),
                )
                self.use_reranking = False

        # Initialize embeddings based on type
        try:
            if self.embedding_type == "ollama":
                logger.info(
                    "Initializing Ollama embeddings",
                    model=self.embedding_model_name,
                    base_url=settings.OLLAMA_BASE_URL,
                )
                self.embeddings = OllamaEmbeddings(
                    model=self.embedding_model_name,
                    base_url=settings.OLLAMA_BASE_URL,
                )
                logger.info("Ollama embeddings initialized successfully")
            else:
                logger.info(
                    "Initializing HuggingFace embeddings",
                    model=self.embedding_model_name,
                )
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,
                    model_kwargs={"device": settings.EMBEDDING_DEVICE},
                    encode_kwargs={"normalize_embeddings": True},
                )
                logger.info("HuggingFace embeddings initialized successfully")
        except Exception as e:
            logger.error(
                "Failed to initialize embeddings",
                model=self.embedding_model_name,
                embedding_type=self.embedding_type,
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

            # Configure ChromaDB to use cosine similarity for better text embeddings
            # Cosine similarity is better for text as it measures direction (semantic meaning)
            # rather than magnitude, which is more suitable for normalized embeddings
            self.vector_store = Chroma(
                persist_directory=str(self.persist_directory),
                embedding_function=self.embeddings,
                collection_name=self.collection_name,
                collection_metadata={"hnsw:space": "cosine"},
            )

            logger.info(
                "LangChain Chroma vector store initialized successfully",
                distance_metric="cosine",
            )
        except Exception as e:
            error_str = str(e)
            # Check if it's a schema mismatch error
            if "no such column" in error_str.lower() or "schema" in error_str.lower():
                logger.warning(
                    "ChromaDB schema mismatch detected. Attempting to reset database.",
                    error=error_str,
                )
                # Try to reset by deleting the old database files
                try:
                    self._reset_chromadb_database()
                    # Retry initialization
                    self.vector_store = Chroma(
                        persist_directory=str(self.persist_directory),
                        embedding_function=self.embeddings,
                        collection_name=self.collection_name,
                        collection_metadata={"hnsw:space": "cosine"},
                    )
                    logger.info(
                        "ChromaDB database reset and reinitialized successfully"
                    )
                except Exception as reset_error:
                    logger.error(
                        "Failed to reset ChromaDB database",
                        error=str(reset_error),
                        exc_info=True,
                    )
                    raise VectorStoreError(
                        f"ChromaDB schema mismatch. Please delete {self.persist_directory} and restart. "
                        f"Original error: {error_str}"
                    ) from reset_error
            else:
                logger.error(
                    "Failed to initialize vector store",
                    error=error_str,
                    exc_info=True,
                )
                raise VectorStoreError(f"Failed to initialize vector store: {error_str}") from e

    def _reset_chromadb_database(self) -> None:
        """Reset ChromaDB database by removing old database files.
        
        This is used when there's a schema mismatch between ChromaDB versions.
        WARNING: This will delete all existing data in the vector store.
        """
        try:
            logger.warning(
                "Resetting ChromaDB database due to schema mismatch",
                persist_directory=str(self.persist_directory),
            )
            
            # Find and remove ChromaDB SQLite database files
            db_files = [
                self.persist_directory / "chroma.sqlite3",
                self.persist_directory / "chroma.sqlite3-wal",
                self.persist_directory / "chroma.sqlite3-shm",
            ]
            
            for db_file in db_files:
                if db_file.exists():
                    logger.info(f"Removing old database file: {db_file}")
                    db_file.unlink()
            
            # Also check for subdirectories that might contain old collections
            # (ChromaDB 0.4+ uses a different structure)
            for item in self.persist_directory.iterdir():
                if item.is_dir() and item.name != "__pycache__":
                    # Check if it's a ChromaDB collection directory
                    if (item / "chroma.sqlite3").exists() or any(
                        f.name.startswith("chroma") for f in item.iterdir()
                    ):
                        logger.info(f"Removing old collection directory: {item}")
                        shutil.rmtree(item)
            
            logger.info("ChromaDB database reset complete")
            
        except Exception as e:
            logger.error(
                "Failed to reset ChromaDB database",
                error=str(e),
                exc_info=True,
            )
            raise

    async def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add documents to the vector store with multimodal metadata support.

        Handles long chunks by recursive splitting and preserves all metadata
        through the chunking process. Also updates BM25 index for hybrid search.

        Args:
            documents: List of LangChain Document objects with multimodal metadata
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

            # Process documents: split long chunks and preserve metadata
            processed_docs: List[Document] = []
            processed_ids: List[str] = []
            
            for idx, doc in enumerate(documents):
                doc_id = ids[idx] if ids and idx < len(ids) else None
                
                # Check if chunk needs splitting
                estimated_tokens = len(doc.page_content) // CHARS_PER_TOKEN
                
                if estimated_tokens > MAX_CHUNK_TOKENS:
                    # Recursive splitting for long chunks
                    split_docs = await self._split_long_chunk(doc, doc_id)
                    processed_docs.extend(split_docs)
                    # Generate IDs for split chunks
                    base_id = doc_id or f"doc_{idx}"
                    processed_ids.extend([
                        f"{base_id}_part_{i}" for i in range(len(split_docs))
                    ])
                else:
                    processed_docs.append(doc)
                    # Always append an ID - use provided ID or generate one
                    if doc_id:
                        processed_ids.append(doc_id)
                    else:
                        # Generate ID from metadata if available, otherwise use index
                        chunk_id = doc.metadata.get("chunk_id") if doc.metadata else None
                        if chunk_id:
                            processed_ids.append(str(chunk_id))
                        else:
                            processed_ids.append(f"doc_{idx}")

            # Filter metadata to remove None values and incompatible types for ChromaDB
            for doc in processed_docs:
                if doc.metadata:
                    # Remove None values and convert complex types
                    filtered_metadata = {}
                    for key, value in doc.metadata.items():
                        if value is not None:
                            # ChromaDB supports: str, int, float, bool
                            if isinstance(value, (str, int, float, bool)):
                                filtered_metadata[key] = value
                            elif isinstance(value, (list, dict, set)):
                                # Convert complex types to string
                                filtered_metadata[key] = str(value)
                    doc.metadata = filtered_metadata

            # Only pass IDs if we have the same number as documents (ChromaDB requirement)
            ids_to_pass = processed_ids if len(processed_ids) == len(processed_docs) else None

            # Add to vector store
            result_ids = self.vector_store.add_documents(
                documents=processed_docs,
                ids=ids_to_pass,
            )

            # Update BM25 index for hybrid search
            self._bm25_documents.extend(processed_docs)
            self._invalidate_bm25_index()

            # Persist to disk
            self.vector_store.persist()

            logger.info(
                "Documents added to vector store",
                original_count=len(documents),
                processed_count=len(processed_docs),
                split_count=len(processed_docs) - len(documents),
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

    def _get_cached_embedding(self, query: str) -> Optional[List[float]]:
        """Get cached embedding for a query.
        
        Args:
            query: Query text
            
        Returns:
            Cached embedding if found, None otherwise
        """
        if not self.cache_enabled or not self.embedding_cache:
            return None
        
        return self.embedding_cache.get(query)
    
    def _cache_embedding(self, query: str, embedding: List[float]) -> None:
        """Cache an embedding for a query.
        
        Args:
            query: Query text
            embedding: Query embedding vector
        """
        if self.cache_enabled and self.embedding_cache:
            self.embedding_cache.put(query, embedding)

    async def similarity_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Search for similar documents with embedding caching and enhanced filtering.

        Args:
            query: Search query text
            k: Number of results to return
            filter: Optional metadata filter with support for:
                - file_type: Filter by FileTypeCategory
                - element_type: Filter by element type (e.g., "title", "narrative_text")
                - date_range: Filter by date range {"start": datetime, "end": datetime}
                - source_file_id: Filter by source file ID
                - page_number: Filter by page number (for documents)
                - timestamp_range: Filter by timestamp range (for audio)

        Returns:
            List of similar Document objects

        Raises:
            VectorStoreError: If search fails
        """
        try:
            logger.debug("Performing similarity search", query=query[:50], k=k)

            # Build enhanced filter from metadata
            chroma_filter = self._build_chroma_filter(filter) if filter else None

            # Check cache for query embedding
            cached_embedding = self._get_cached_embedding(query)
            
            if cached_embedding:
                logger.debug("Using cached embedding for query")
                # Use cached embedding for search
                if chroma_filter:
                    results = self.vector_store.similarity_search_by_vector(
                        embedding=cached_embedding,
                        k=k,
                        filter=chroma_filter,
                    )
                else:
                    results = self.vector_store.similarity_search_by_vector(
                        embedding=cached_embedding,
                        k=k,
                    )
            else:
                # Generate new embedding and cache it
                logger.debug("Generating new embedding for query")
                
                # Get embedding from the embeddings model
                query_embedding = self.embeddings.embed_query(query)
                
                # Cache the embedding
                self._cache_embedding(query, query_embedding)
                
                # Perform search with the new embedding
                if chroma_filter:
                    results = self.vector_store.similarity_search_by_vector(
                        embedding=query_embedding,
                        k=k,
                        filter=chroma_filter,
                    )
                else:
                    results = self.vector_store.similarity_search_by_vector(
                        embedding=query_embedding,
                        k=k,
                    )

            logger.debug("Similarity search completed", result_count=len(results))

            return results

        except Exception as e:
            error_str = str(e)
            # Handle HNSW index errors - these occur when the index parameters are too small
            if "contigious" in error_str.lower() or "contiguous" in error_str.lower() or "ef or M is too small" in error_str.lower():
                logger.warning(
                    "HNSW index error detected - index parameters may be too small for dataset size",
                    error=error_str,
                    k=k,
                )
                # Try with a smaller k value as a workaround
                if k > 4:
                    logger.info(f"Retrying similarity search with reduced k={min(4, k)}")
                    try:
                        results = await self.similarity_search(query, k=min(4, k), filter=filter)
                        return results
                    except Exception as retry_error:
                        logger.error(
                            "Retry with reduced k also failed",
                            error=str(retry_error),
                            exc_info=True,
                        )
                        # Fallback: return empty results instead of failing completely
                        logger.warning("Returning empty results as fallback for HNSW error")
                        return []
            
            logger.error(
                "Similarity search failed",
                error=error_str,
                exc_info=True,
            )
            raise VectorStoreError(f"Similarity search failed: {error_str}") from e

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
            List of tuples (Document, similarity_score) where similarity_score
            is normalized to 0.0-1.0 range (1.0 = most similar, 0.0 = least similar).
            ChromaDB is configured to use cosine distance, which is converted to
            cosine similarity using: similarity = 1 - distance. Cosine similarity
            is ideal for text embeddings as it measures semantic direction rather
            than magnitude.

        Raises:
            VectorStoreError: If search fails
        """
        try:
            logger.debug(
                "Performing similarity search with scores",
                query=query[:50],
                k=k,
            )

            # If reranking is enabled, retrieve more candidates (2-3x k) for better reranking results
            # Otherwise, just retrieve k results
            retrieval_k = (k * 3) if (self.use_reranking and self.reranker) else k
            
            if filter:
                chroma_filter = self._build_chroma_filter(filter)
                results = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=retrieval_k,
                    filter=chroma_filter,
                )
            else:
                results = self.vector_store.similarity_search_with_score(
                    query=query, k=retrieval_k
                )

            # Convert distance scores to similarity scores (0-1)
            # ChromaDB can return either cosine distance (0-2 range) or L2 distance (0-infinity)
            # We auto-detect which metric is being used based on distance values
            # Cosine: similarity = 1 - distance
            # L2: similarity = 1 / (1 + distance)
            converted_results = [
                (doc, _distance_to_similarity(distance, distance_metric=None))
                for doc, distance in results
            ]

            # Apply reranking if enabled
            if self.use_reranking and self.reranker and converted_results:
                try:
                    logger.debug(
                        "Applying reranking to search results",
                        initial_count=len(converted_results),
                    )
                    
                    # Extract documents for reranking
                    documents = [doc for doc, _ in converted_results]
                    
                    # Rerank (returns list of (doc, score) tuples)
                    reranked_results = self.reranker.rerank_documents(
                        query=query,
                        documents=documents,
                        top_k=k,
                    )
                    
                    # Convert reranker scores (which can be any range) to 0-1 similarity scores
                    # Cross-encoder scores are typically in a wider range, so we normalize them
                    if reranked_results:
                        # Normalize scores to 0-1 range using min-max normalization
                        scores = [score for _, score in reranked_results]
                        if scores:
                            min_score = min(scores)
                            max_score = max(scores)
                            score_range = max_score - min_score if max_score > min_score else 1.0
                            
                            # Normalize and update results
                            converted_results = [
                                (doc, (score - min_score) / score_range if score_range > 0 else 0.5)
                                for doc, score in reranked_results
                            ]
                        
                        logger.debug(
                            "Reranking completed",
                            reranked_count=len(converted_results),
                        )
                    else:
                        logger.warning("Reranker returned empty results, using original order")
                        
                except Exception as e:
                    logger.warning(
                        "Reranking failed, using original results",
                        error=str(e),
                    )
                    # Continue with original results if reranking fails

            logger.debug(
                "Similarity search with scores completed",
                result_count=len(converted_results),
                reranked=self.use_reranking and self.reranker is not None,
            )

            return converted_results

        except Exception as e:
            error_str = str(e)
            # Handle HNSW index errors - these occur when the index parameters are too small
            if "contigious" in error_str.lower() or "contiguous" in error_str.lower() or "ef or M is too small" in error_str.lower():
                logger.warning(
                    "HNSW index error detected in similarity_search_with_score - index parameters may be too small",
                    error=error_str,
                    k=k,
                )
                # Try with a smaller k value as a workaround
                if k > 4:
                    logger.info(f"Retrying similarity_search_with_score with reduced k={min(4, k)}")
                    try:
                        results = await self.similarity_search_with_score(query, k=min(4, k), filter=filter)
                        return results
                    except Exception as retry_error:
                        logger.error(
                            "Retry with reduced k also failed",
                            error=str(retry_error),
                            exc_info=True,
                        )
                        # Fallback: return empty results instead of failing completely
                        logger.warning("Returning empty results as fallback for HNSW error")
                        return []
            
            logger.error(
                "Similarity search with scores failed",
                error=error_str,
                exc_info=True,
            )
            raise VectorStoreError(
                f"Similarity search with scores failed: {error_str}"
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
    
    async def add_chunks(
        self,
        chunks: List[DocumentChunk],
        folder_id: Optional[str] = None,
    ) -> List[str]:
        """Add DocumentChunk objects to vector store with multimodal metadata.
        
        Converts DocumentChunk to LangChain Document with enhanced metadata:
        - element_type from Unstructured
        - timestamp ranges for audio segments
        - page_numbers for documents
        - source_file_id for citations
        
        Args:
            chunks: List of DocumentChunk objects
            folder_id: Optional folder ID to add to metadata
            
        Returns:
            List of document IDs
        """
        try:
            documents = []
            for chunk in chunks:
                # Build enhanced metadata from DocumentChunk
                metadata = {
                    "chunk_id": chunk.chunk_id,
                    "file_id": chunk.file_id,
                    "source_file_id": chunk.file_id,  # For citations
                    **chunk.metadata,  # Include all chunk metadata
                }
                
                # Add folder_id if provided
                if folder_id:
                    metadata["folder_id"] = folder_id
                
                # Extract multimodal-specific metadata
                if "element_type" in chunk.metadata:
                    metadata["element_type"] = chunk.metadata["element_type"]
                
                if "page_number" in chunk.metadata:
                    metadata["page_number"] = chunk.metadata["page_number"]
                
                if "segment_start" in chunk.metadata and "segment_end" in chunk.metadata:
                    metadata["timestamp_start"] = chunk.metadata["segment_start"]
                    metadata["timestamp_end"] = chunk.metadata["segment_end"]
                
                if "file_name" in chunk.metadata:
                    metadata["file_name"] = chunk.metadata["file_name"]
                
                if "mime_type" in chunk.metadata:
                    metadata["mime_type"] = chunk.metadata["mime_type"]
                    # Extract file_type from mime_type if possible
                    try:
                        from app.services.file_type_router import get_file_category
                        file_name_val = chunk.metadata.get("file_name")
                        mime_type_val = chunk.metadata.get("mime_type")
                        # Ensure types are correct
                        file_name_str = str(file_name_val) if file_name_val is not None else ""
                        mime_type_str = str(mime_type_val) if mime_type_val is not None else None
                        if file_name_str:
                            file_category = get_file_category(
                                file_name_str,
                                mime_type_str
                            )
                            metadata["file_type"] = file_category.value
                    except Exception:
                        pass
                
                # Create LangChain Document
                doc = Document(
                    page_content=chunk.content,
                    metadata=metadata,
                )
                documents.append(doc)
            
            # Add documents with automatic splitting
            return await self.add_documents(documents)
            
        except Exception as e:
            logger.error(
                "Failed to add chunks to vector store",
                error=str(e),
                exc_info=True,
            )
            raise VectorStoreError(f"Failed to add chunks: {str(e)}") from e
    
    async def _split_long_chunk(
        self,
        document: Document,
        base_id: Optional[str] = None,
    ) -> List[Document]:
        """Recursively split long chunks while preserving metadata.
        
        Args:
            document: LangChain Document to split
            base_id: Base ID for generated chunk IDs
            
        Returns:
            List of split Document objects
        """
        text = document.page_content
        estimated_tokens = len(text) // CHARS_PER_TOKEN
        
        if estimated_tokens <= MAX_CHUNK_TOKENS:
            return [document]
        
        # Calculate chunk size in characters
        chunk_size_chars = MAX_CHUNK_TOKENS * CHARS_PER_TOKEN
        overlap_chars = CHUNK_OVERLAP_TOKENS * CHARS_PER_TOKEN
        
        split_docs: List[Document] = []
        start = 0
        part_num = 0
        
        while start < len(text):
            end = min(start + chunk_size_chars, len(text))
            
            # Try to split at sentence boundary
            if end < len(text):
                # Look for sentence endings near the end
                for punct in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct > start + chunk_size_chars * 0.7:  # Only if not too early
                        end = last_punct + len(punct)
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                # Create new document with preserved metadata
                chunk_metadata = document.metadata.copy()
                chunk_metadata["chunk_part"] = part_num
                chunk_metadata["is_split"] = True
                chunk_metadata["original_length"] = len(text)
                
                # Generate chunk ID
                chunk_id = base_id or chunk_metadata.get("chunk_id", "chunk")
                chunk_metadata["chunk_id"] = f"{chunk_id}_part_{part_num}"
                
                chunk_doc = Document(
                    page_content=chunk_text,
                    metadata=chunk_metadata,
                )
                split_docs.append(chunk_doc)
                part_num += 1
            
            # Move start position with overlap
            start = end - overlap_chars if end < len(text) else end
        
        logger.debug(
            "Split long chunk",
            original_length=len(text),
            estimated_tokens=estimated_tokens,
            split_count=len(split_docs),
        )
        
        return split_docs
    
    def _build_chroma_filter(self, filter_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Build ChromaDB-compatible filter from enhanced metadata filter.
        
        Args:
            filter_dict: Enhanced filter dictionary with support for:
                - file_type: FileTypeCategory value
                - element_type: Element type string
                - date_range: {"start": datetime, "end": datetime}
                - source_file_id: File ID string
                - page_number: Page number (int or list)
                - timestamp_range: {"start": float, "end": float}
        
        Returns:
            ChromaDB-compatible filter dictionary
        
        Note: ChromaDB requires $and operator when multiple conditions are present.
        """
        # Direct metadata filters (ChromaDB supports these directly)
        direct_keys = [
            "file_id", "source_file_id", "folder_id", "file_name",
            "element_type", "mime_type", "file_type",
        ]
        
        # Collect simple equality filters
        simple_filters = {}
        for key in direct_keys:
            if key in filter_dict:
                value = filter_dict[key]
                # Extract value from {"$eq": value} format if present
                if isinstance(value, dict) and "$eq" in value:
                    simple_filters[key] = value["$eq"]
                else:
                    simple_filters[key] = value
        
        # If we have multiple simple filters, use $and
        if len(simple_filters) > 1:
            chroma_filter = {"$and": [{k: v} for k, v in simple_filters.items()]}
        elif len(simple_filters) == 1:
            chroma_filter = simple_filters
        else:
            chroma_filter = {}
        
        # Collect additional filters
        additional_filters = {}
        
        # Page number filter
        if "page_number" in filter_dict:
            page_num = filter_dict["page_number"]
            if isinstance(page_num, (int, str)):
                additional_filters["page_number"] = page_num
            elif isinstance(page_num, list):
                additional_filters["page_number"] = {"$in": page_num}
        
        # Date range filter
        if "date_range" in filter_dict:
            date_range = filter_dict["date_range"]
            if "start" in date_range or "end" in date_range:
                date_filter: Dict[str, Any] = {}
                if "start" in date_range:
                    date_filter["$gte"] = date_range["start"].isoformat()
                if "end" in date_range:
                    date_filter["$lte"] = date_range["end"].isoformat()
                if date_filter:
                    additional_filters["created_at"] = date_filter
        
        # Timestamp range filter (for audio)
        if "timestamp_range" in filter_dict:
            ts_range = filter_dict["timestamp_range"]
            if "start" in ts_range or "end" in ts_range:
                ts_filter: Dict[str, Any] = {}
                if "start" in ts_range:
                    ts_filter["$gte"] = ts_range["start"]
                if "end" in ts_range:
                    ts_filter["$lte"] = ts_range["end"]
                if ts_filter:
                    additional_filters["timestamp_start"] = ts_filter
        
        # Element type filter (e.g., only titles) - already handled in simple_filters if it's a direct key
        if "element_type" in filter_dict and "element_type" not in simple_filters:
            element_type = filter_dict["element_type"]
            if isinstance(element_type, list):
                additional_filters["element_type"] = {"$in": element_type}
            else:
                additional_filters["element_type"] = element_type
        
        # Combine all filters
        all_conditions = []
        if chroma_filter:
            if "$and" in chroma_filter:
                all_conditions.extend(chroma_filter["$and"])
            else:
                all_conditions.append(chroma_filter)
        
        # Add additional filters
        for key, value in additional_filters.items():
            all_conditions.append({key: value})
        
        # Build final filter
        if len(all_conditions) > 1:
            final_filter = {"$and": all_conditions}
        elif len(all_conditions) == 1:
            final_filter = all_conditions[0]
        else:
            final_filter = {}
        
        logger.debug("Built ChromaDB filter", filter=final_filter)
        
        return final_filter
    
    def _invalidate_bm25_index(self) -> None:
        """Invalidate BM25 index to force rebuild on next use."""
        self._bm25_index = None
    
    def _get_bm25_index(self) -> Any:
        """Get or build BM25 index for hybrid search.
        
        Returns:
            BM25Retriever instance
        """
        if self._bm25_index is None and self._bm25_documents:
            try:
                from langchain_community.retrievers import BM25Retriever
                self._bm25_index = BM25Retriever.from_documents(self._bm25_documents)
                logger.debug("BM25 index built", document_count=len(self._bm25_documents))
            except ImportError:
                logger.warning("BM25Retriever not available, skipping BM25 indexing")
                return None
            except Exception as e:
                logger.error("Failed to build BM25 index", error=str(e))
                return None
        
        return self._bm25_index
    
    async def bm25_search(
        self,
        query: str,
        k: int = 10,
        filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        """Perform BM25 keyword search with metadata filtering.
        
        Args:
            query: Search query text
            k: Number of results to return
            filter: Optional metadata filter (applied post-search)
        
        Returns:
            List of relevant Document objects
        """
        try:
            bm25_index = self._get_bm25_index()
            if not bm25_index:
                logger.warning("BM25 index not available, returning empty results")
                return []
            
            # Perform BM25 search
            results = bm25_index.get_relevant_documents(query, k=k * 2)  # Get more for filtering
            
            # Apply metadata filter if provided
            if filter:
                results = self._apply_metadata_filter(results, filter)
            
            # Limit to k results
            results = results[:k]
            
            logger.debug("BM25 search completed", result_count=len(results))
            
            return results
            
        except Exception as e:
            logger.error("BM25 search failed", error=str(e), exc_info=True)
            return []
    
    def _apply_metadata_filter(
        self,
        documents: List[Document],
        filter_dict: Dict[str, Any],
    ) -> List[Document]:
        """Apply metadata filter to documents.
        
        Args:
            documents: List of Document objects
            filter_dict: Filter dictionary
        
        Returns:
            Filtered list of Document objects
        """
        filtered = []
        
        for doc in documents:
            metadata = doc.metadata
            
            # Check file_type
            if "file_type" in filter_dict:
                if metadata.get("file_type") != filter_dict["file_type"]:
                    continue
            
            # Check element_type
            if "element_type" in filter_dict:
                element_type = filter_dict["element_type"]
                if isinstance(element_type, list):
                    if metadata.get("element_type") not in element_type:
                        continue
                elif metadata.get("element_type") != element_type:
                    continue
            
            # Check date range
            if "date_range" in filter_dict:
                date_range = filter_dict["date_range"]
                created_at_str = metadata.get("created_at")
                if created_at_str:
                    try:
                        created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                        if "start" in date_range and created_at < date_range["start"]:
                            continue
                        if "end" in date_range and created_at > date_range["end"]:
                            continue
                    except Exception:
                        pass
            
            # Check source_file_id
            if "source_file_id" in filter_dict:
                if metadata.get("source_file_id") != filter_dict["source_file_id"]:
                    continue
            
            filtered.append(doc)
        
        return filtered

