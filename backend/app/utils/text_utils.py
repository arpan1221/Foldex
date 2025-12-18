"""Text processing utilities."""

from typing import List, Optional
from app.config.settings import settings


def chunk_text(text: str, chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> List[str]:
    """Split text into chunks with overlap.

    Args:
        text: Text to chunk
        chunk_size: Size of each chunk (default from settings)
        overlap: Overlap between chunks (default from settings)

    Returns:
        List of text chunks
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    if not text or not text.strip():
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:  # Only add non-empty chunks
            chunks.append(chunk)
        start = end - overlap
        if start >= text_length:
            break

    return chunks


def clean_text(text: str) -> str:
    """Clean and normalize text.

    Args:
        text: Raw text

    Returns:
        Cleaned text
    """
    # Remove extra whitespace
    text = " ".join(text.split())
    return text.strip()


def extract_sentences(text: str) -> List[str]:
    """Extract sentences from text.

    Args:
        text: Text to extract sentences from

    Returns:
        List of sentences
    """
    import re
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]

