#!/usr/bin/env python3
"""Download required ML models for Foldex."""

import sys
from pathlib import Path
from typing import Optional

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

try:
    from app.config.settings import settings
except ImportError:
    # Fallback if settings not available
    class Settings:
        EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
        EMBEDDING_DEVICE = "cpu"
        WHISPER_MODEL = "base"
        WHISPER_DEVICE = "cpu"
        OLLAMA_BASE_URL = "http://localhost:11434"
        OLLAMA_MODEL = "llama3.2"
        MODELS_DIR = "./models"
    settings = Settings()


def download_embedding_model(model_name: Optional[str] = None) -> bool:
    """Download sentence-transformers embedding model.
    
    Args:
        model_name: Optional model name override
        
    Returns:
        True if successful, False otherwise
    """
    print("📥 Downloading embedding model...")
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = model_name or settings.EMBEDDING_MODEL
        model_path = Path(settings.MODELS_DIR) / "embeddings"
        model_path.mkdir(parents=True, exist_ok=True)
        
        print(f"   Model: {model_name}")
        print(f"   Device: {settings.EMBEDDING_DEVICE}")
        print("   Loading model (this may take a few minutes)...")
        
        model = SentenceTransformer(model_name, device=settings.EMBEDDING_DEVICE)
        
        # Test the model
        test_embedding = model.encode("test", show_progress_bar=False)
        print(f"   ✓ Model loaded successfully (embedding dim: {len(test_embedding)})")
        del model  # Free memory
        return True
        
    except ImportError:
        print("   ✗ sentence-transformers not installed")
        print("   Install with: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def download_whisper_model(model_name: Optional[str] = None) -> bool:
    """Download Whisper model for audio transcription.
    
    Args:
        model_name: Optional model name override
        
    Returns:
        True if successful, False otherwise
    """
    print("\n📥 Downloading Whisper model...")
    try:
        import whisper
        
        model_name = model_name or settings.WHISPER_MODEL
        model_path = Path(settings.MODELS_DIR) / "whisper"
        model_path.mkdir(parents=True, exist_ok=True)
        
        print(f"   Model: {model_name}")
        print(f"   Device: {settings.WHISPER_DEVICE}")
        print("   Loading model (this may take a few minutes)...")
        
        model = whisper.load_model(model_name, device=settings.WHISPER_DEVICE)
        print("   ✓ Model loaded successfully")
        del model  # Free memory
        return True
        
    except ImportError:
        print("   ✗ openai-whisper not installed")
        print("   Install with: pip install openai-whisper")
        return False
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False


def check_ollama_connection() -> bool:
    """Check if Ollama is running and accessible.
    
    Returns:
        True if Ollama is accessible, False otherwise
    """
    try:
        import httpx
        
        response = httpx.get(f"{settings.OLLAMA_BASE_URL}/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "") for m in models]
            print("   ✓ Ollama is running")
            available = ", ".join(model_names) if model_names else "None"
            print(f"   Available models: {available}")
            
            # Check if required model is available
            required_model = settings.OLLAMA_MODEL
            if any(required_model in name for name in model_names):
                print(f"   ✓ Required model '{required_model}' is available")
            else:
                print(f"   ⚠ Required model '{required_model}' not found")
                print(f"   Pull it with: ollama pull {required_model}")
            return True
        return False
    except Exception:
        print(f"   ⚠ Ollama not accessible at {settings.OLLAMA_BASE_URL}")
        print(f"   Start Ollama with: ollama serve")
        return False


def main():
    """Main function to download all models."""
    print("=" * 60)
    print("Foldex Model Downloader")
    print("=" * 60)
    print()
    
    success_count = 0
    total_count = 2
    
    # Download embedding model
    if download_embedding_model():
        success_count += 1
    
    # Download Whisper model
    if download_whisper_model():
        success_count += 1
    
    # Check Ollama (optional)
    print("\n🔍 Checking Ollama connection...")
    check_ollama_connection()
    
    print()
    print("=" * 60)
    if success_count == total_count:
        print("✓ All required models downloaded successfully!")
    else:
        print(f"⚠ Downloaded {success_count}/{total_count} models")
        print("  Some models failed to download. Check errors above.")
    print("=" * 60)
    print()
    
    if success_count < total_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
