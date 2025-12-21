"""Intelligent document ingestion and chunking system for Foldex."""

from app.ingestion.chunking import FoldexChunker
from app.ingestion.metadata_extractor import MetadataExtractor

__all__ = ["FoldexChunker", "MetadataExtractor"]

