"""Warmup endpoints for model pre-loading and cache management."""

from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status
import structlog

from app.startup.warmup import warmup_llm_model, initialize_ttft_optimizer
from app.utils.embedding_cache import get_embedding_cache

router = APIRouter(prefix="/warmup", tags=["warmup"])
logger = structlog.get_logger(__name__)


@router.post("/model", response_model=Dict[str, Any])
async def warmup_model():
    """
    Warmup the LLM model by loading it into memory.
    
    This endpoint can be called manually to pre-load the model
    and reduce first-request latency.
    
    Returns:
        Status message and warmup details
    """
    try:
        logger.info("Manual model warmup requested")
        await warmup_llm_model()
        
        return {
            "status": "success",
            "message": "Model warmup completed successfully",
        }
    except Exception as e:
        logger.error("Manual model warmup failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model warmup failed: {str(e)}"
        )


@router.post("/ttft", response_model=Dict[str, Any])
async def warmup_ttft_optimizer():
    """
    Initialize and warmup the TTFT optimizer.
    
    This endpoint can be called manually to initialize
    the TTFT optimizer and its caches.
    
    Returns:
        Status message and optimizer details
    """
    try:
        logger.info("Manual TTFT optimizer warmup requested")
        await initialize_ttft_optimizer()
        
        return {
            "status": "success",
            "message": "TTFT optimizer initialized successfully",
        }
    except Exception as e:
        logger.error("Manual TTFT optimizer warmup failed", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"TTFT optimizer warmup failed: {str(e)}"
        )


@router.get("/cache/stats", response_model=Dict[str, Any])
async def get_cache_stats():
    """
    Get embedding cache statistics.
    
    Returns cache hit rate, size, and other metrics.
    
    Returns:
        Cache statistics
    """
    try:
        cache = get_embedding_cache()
        stats = cache.get_stats()
        
        return {
            "status": "success",
            "cache_stats": stats,
        }
    except Exception as e:
        logger.error("Failed to get cache stats", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get cache stats: {str(e)}"
        )


@router.post("/cache/clear", response_model=Dict[str, Any])
async def clear_embedding_cache():
    """
    Clear the embedding cache.
    
    This removes all cached query embeddings.
    
    Returns:
        Status message
    """
    try:
        logger.info("Manual cache clear requested")
        cache = get_embedding_cache()
        cache.clear()
        
        return {
            "status": "success",
            "message": "Embedding cache cleared successfully",
        }
    except Exception as e:
        logger.error("Failed to clear cache", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear cache: {str(e)}"
        )


@router.post("/cache/cleanup", response_model=Dict[str, Any])
async def cleanup_expired_cache():
    """
    Clean up expired entries from the embedding cache.
    
    This removes only the expired entries, keeping valid ones.
    
    Returns:
        Status message and number of entries removed
    """
    try:
        logger.info("Manual cache cleanup requested")
        cache = get_embedding_cache()
        removed_count = cache.cleanup_expired()
        
        return {
            "status": "success",
            "message": f"Removed {removed_count} expired cache entries",
            "removed_count": removed_count,
        }
    except Exception as e:
        logger.error("Failed to cleanup cache", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cleanup cache: {str(e)}"
        )

