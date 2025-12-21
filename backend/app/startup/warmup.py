"""Application startup warmup for TTFT optimization.

This module handles model pre-warming and initialization tasks
to reduce first-request latency.
"""

import asyncio
import time
import structlog

from app.config.settings import settings
from app.rag.ttft_optimization import warmup_model, get_ttft_optimizer
from app.rag.llm_chains import OllamaLLM
from app.rag.query_classifier import get_query_classifier

logger = structlog.get_logger(__name__)


async def warmup_llm_model():
    """Pre-warm the LLM model on startup.

    This reduces first-request latency by loading the model
    into memory and running a simple inference.
    """
    if not settings.ENABLE_MODEL_WARMUP:
        logger.info("Model warmup disabled in settings")
        return

    logger.info("Starting LLM model warmup")
    start_time = time.time()

    try:
        # Initialize LLM
        llm = OllamaLLM()

        # Warmup the model
        await warmup_model(llm.get_llm())

        elapsed = time.time() - start_time
        logger.info(
            "LLM model warmup completed",
            elapsed_seconds=round(elapsed, 2)
        )

    except Exception as e:
        # Don't fail startup, just log
        logger.error(
            "LLM model warmup failed",
            error=str(e),
            elapsed_seconds=round(time.time() - start_time, 2)
        )


async def warmup_query_classifier_model():
    """Pre-warm the query classifier LLM model (llama3.2:1b) on startup.
    
    This reduces first-request latency for query classification by loading
    the model into memory and running a simple inference.
    """
    if not settings.ENABLE_MODEL_WARMUP:
        logger.info("Model warmup disabled in settings, skipping query classifier warmup")
        return
    
    logger.info("Starting query classifier model warmup (llama3.2:1b)")
    start_time = time.time()
    
    try:
        # Get query classifier instance (this will initialize the LLM)
        classifier = get_query_classifier(use_llm=True)
        
        # Only warmup if LLM is actually initialized
        if classifier.llm:
            # Warmup the model by running a simple classification
            llm_instance = classifier.llm.get_llm()
            await warmup_model(llm_instance)
            
            elapsed = time.time() - start_time
            logger.info(
                "Query classifier model warmup completed",
                model="llama3.2:1b",
                elapsed_seconds=round(elapsed, 2)
            )
        else:
            logger.info("Query classifier LLM not available, skipping warmup (pattern matching will be used)")
    
    except Exception as e:
        # Don't fail startup, just log
        logger.error(
            "Query classifier model warmup failed",
            error=str(e),
            elapsed_seconds=round(time.time() - start_time, 2)
        )


async def initialize_ttft_optimizer():
    """Initialize TTFT optimizer with settings.

    This configures the global TTFT optimizer instance
    with settings from configuration.
    """
    if not settings.ENABLE_TTFT_OPTIMIZATION:
        logger.info("TTFT optimization disabled in settings")
        return

    logger.info("Initializing TTFT optimizer")

    try:
        # Get optimizer instance (creates if not exists)
        optimizer = get_ttft_optimizer()

        # Configure with settings
        if optimizer.prompt_cache:
            optimizer.prompt_cache.max_size = settings.PROMPT_CACHE_SIZE
            optimizer.prompt_cache.ttl_seconds = settings.PROMPT_CACHE_TTL

        if optimizer.context_optimizer:
            optimizer.context_optimizer.max_context_chars = settings.MAX_CONTEXT_CHARS

        logger.info(
            "TTFT optimizer initialized",
            prompt_cache_enabled=optimizer.enable_prompt_cache,
            context_optimization_enabled=optimizer.enable_context_optimization,
            prompt_cache_size=settings.PROMPT_CACHE_SIZE,
            max_context_chars=settings.MAX_CONTEXT_CHARS,
        )

    except Exception as e:
        logger.error(
            "TTFT optimizer initialization failed",
            error=str(e)
        )


async def startup_warmup():
    """Run all startup warmup tasks.

    This function orchestrates all warmup tasks including:
    - TTFT optimizer initialization
    - Main LLM model pre-warming (llama3.2:3b)
    - Query classifier model pre-warming (llama3.2:1b)
    """
    logger.info("Starting application warmup")
    start_time = time.time()

    # Run warmup tasks
    tasks = [
        initialize_ttft_optimizer(),
        warmup_llm_model(),
        warmup_query_classifier_model(),
    ]

    # Run concurrently
    await asyncio.gather(*tasks, return_exceptions=True)

    elapsed = time.time() - start_time
    logger.info(
        "Application warmup completed",
        elapsed_seconds=round(elapsed, 2)
    )
