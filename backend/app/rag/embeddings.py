"""Local embedding service using sentence-transformers."""

from typing import List, Optional
import structlog
import asyncio
from functools import lru_cache

from app.config.settings import settings
from app.core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)

# Global model cache to avoid reloading
_model_cache: Optional[any] = None


def _load_model():
    """Load sentence-transformers model (cached)."""
    global _model_cache
    if _model_cache is not None:
        return _model_cache
    
    try:
        from sentence_transformers import SentenceTransformer
        model_name = settings.EMBEDDING_MODEL
        logger.info("Loading embedding model", model=model_name)
        _model_cache = SentenceTransformer(model_name)
        logger.info("Embedding model loaded successfully", model=model_name)
        return _model_cache
    except ImportError:
        logger.error("sentence-transformers not installed")
        raise ProcessingError("sentence-transformers not installed. Install with: pip install sentence-transformers")
    except Exception as e:
        logger.error("Failed to load embedding model", error=str(e), exc_info=True)
        raise ProcessingError(f"Failed to load embedding model: {str(e)}")


class EmbeddingService:
    """Service for generating text embeddings locally."""

    def __init__(self):
        """Initialize embedding service."""
        self.model_name = settings.EMBEDDING_MODEL
        self._model = None
        self._load_lock = asyncio.Lock()

    def _get_model(self):
        """Get or load the embedding model."""
        if self._model is None:
            self._model = _load_model()
        return self._model

    async def embed_text(self, text: str) -> List[float]:
        """Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector

        Raises:
            ProcessingError: If embedding generation fails
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return []
        
        try:
            async with self._load_lock:
                model = self._get_model()
            
            # Run encoding in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                lambda: model.encode(text, convert_to_numpy=True)
            )
            
            # Convert to list
            embedding_list = embedding.tolist()
            
            logger.debug(
                "Generated embedding",
                text_length=len(text),
                embedding_dim=len(embedding_list)
            )
            
            return embedding_list
            
        except Exception as e:
            logger.error("Embedding generation failed", error=str(e), exc_info=True)
            raise ProcessingError(f"Embedding generation failed: {str(e)}")

    async def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors

        Raises:
            ProcessingError: If batch embedding generation fails
        """
        if not texts:
            logger.warning("Empty text list provided for batch embedding")
            return []
        
        # Filter out empty texts
        valid_texts = [t for t in texts if t and t.strip()]
        if not valid_texts:
            logger.warning("All texts in batch are empty")
            return [[] for _ in texts]
        
        try:
            async with self._load_lock:
                model = self._get_model()
            
            # Run batch encoding in thread pool
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: model.encode(valid_texts, convert_to_numpy=True, show_progress_bar=False)
            )
            
            # Convert to list of lists
            embeddings_list = [emb.tolist() for emb in embeddings]
            
            logger.debug(
                "Generated batch embeddings",
                count=len(embeddings_list),
                embedding_dim=len(embeddings_list[0]) if embeddings_list else 0
            )
            
            return embeddings_list
            
        except Exception as e:
            logger.error("Batch embedding generation failed", error=str(e), exc_info=True)
            raise ProcessingError(f"Batch embedding generation failed: {str(e)}")

