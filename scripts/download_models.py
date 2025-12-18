#!/usr/bin/env python3
"""Download required ML models for Foldex."""

import os
import sys
from pathlib import Path

def download_embedding_model():
    """Download sentence-transformers embedding model."""
    print("Downloading embedding model...")
    try:
        from sentence_transformers import SentenceTransformer
        
        model_name = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
        model_path = Path("models/embeddings")
        model_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading model: {model_name}")
        model = SentenceTransformer(model_name)
        # Model will be cached automatically
        print("Embedding model ready!")
    except ImportError:
        print("sentence-transformers not installed. Install with: pip install sentence-transformers")
        sys.exit(1)
    except Exception as e:
        print(f"Error downloading embedding model: {e}")
        sys.exit(1)


def download_whisper_model():
    """Download Whisper model."""
    print("Downloading Whisper model...")
    try:
        import whisper
        
        model_name = os.getenv("WHISPER_MODEL", "base")
        model_path = Path("models/whisper")
        model_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)
        # Model will be cached automatically
        print("Whisper model ready!")
    except ImportError:
        print("openai-whisper not installed. Install with: pip install openai-whisper")
        sys.exit(1)
    except Exception as e:
        print(f"Error downloading Whisper model: {e}")
        sys.exit(1)


def main():
    """Main function."""
    print("Foldex Model Downloader")
    print("=" * 50)
    
    # Download embedding model
    download_embedding_model()
    print()
    
    # Download Whisper model
    download_whisper_model()
    print()
    
    print("=" * 50)
    print("All models downloaded successfully!")


if __name__ == "__main__":
    main()

