"""
Model warmup script to keep qwen3:4b loaded in memory.

Run on startup: python scripts/warmup_model.py

This ensures the model is loaded and ready for fast first-token generation.
"""

import os
import sys
import time
from pathlib import Path

# Add backend to path
backend_path = Path(__file__).parent.parent
sys.path.insert(0, str(backend_path))

from app.config.settings import settings
import structlog

logger = structlog.get_logger(__name__)


def warmup_ollama():
    """Send dummy request to load model into memory.
    
    Returns:
        True if successful
    """
    try:
        logger.info("Warming up Ollama model", model=settings.OLLAMA_MODEL)
        
        # Set keep-alive to permanent if not already set
        if not os.environ.get("OLLAMA_KEEP_ALIVE"):
            os.environ["OLLAMA_KEEP_ALIVE"] = settings.OLLAMA_KEEP_ALIVE
        
        # Import LLM class
        try:
            from app.rag.llm_chains import OllamaLLM
        except ImportError:
            logger.error("Failed to import OllamaLLM")
            return False
        
        # Initialize LLM
        llm = OllamaLLM(
            model=settings.OLLAMA_MODEL,
            base_url=settings.OLLAMA_BASE_URL,
        )
        
        # Send warmup request
        logger.info("Sending warmup request...")
        start = time.time()
        
        warmup_prompt = "Hello, this is a warmup request. Please respond with 'OK'."
        response = llm.get_llm().invoke(warmup_prompt)
        
        elapsed = time.time() - start
        
        logger.info(
            "Model warmed up successfully",
            model=settings.OLLAMA_MODEL,
            warmup_time_seconds=elapsed,
            keep_alive=os.environ.get("OLLAMA_KEEP_ALIVE", "-1"),
        )
        
        # Verify model is loaded
        try:
            import requests
            ollama_url = settings.OLLAMA_BASE_URL.replace("ollama", "localhost")
            ps_response = requests.get(f"{ollama_url}/api/ps", timeout=5)
            if ps_response.status_code == 200:
                models = ps_response.json().get("models", [])
                loaded_models = [m.get("name") for m in models]
                if settings.OLLAMA_MODEL in loaded_models:
                    logger.info(
                        "Model confirmed loaded in Ollama",
                        loaded_models=loaded_models,
                    )
        except Exception as e:
            logger.warning("Could not verify model status", error=str(e))
        
        return True
        
    except Exception as e:
        logger.error("Model warmup failed", error=str(e), exc_info=True)
        return False


if __name__ == "__main__":
    success = warmup_ollama()
    sys.exit(0 if success else 1)

