"""Citation engine for generating accurate, validated citations from LLM responses.

This module implements Foldex's citation-driven architecture by:
- Mapping LLM response segments to source chunks
- Generating structured citations with file type-specific formatting
- Deduplicating overlapping citations
- Validating citations against database and metadata

Critical for maintaining trust and traceability in RAG responses.
"""

from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, field_validator
from datetime import datetime
import structlog
import re
from difflib import SequenceMatcher

from app.services.file_type_router import FileTypeCategory, get_file_category
from app.database.sqlite_manager import SQLiteManager
from app.core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)

try:
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.schema import Document
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        Document = None


class Citation(BaseModel):
    """Structured citation model for accurate source attribution.
    
    Supports multimodal citations with file type-specific metadata.
    """
    
    source_file_id: str = Field(..., description="Source file identifier")
    source_file_name: str = Field(..., description="Source file name")
    file_type: FileTypeCategory = Field(..., description="File type category")
    page_number: Optional[int] = Field(None, description="Page number for documents")
    timestamp_range: Optional[Tuple[float, float]] = Field(
        None, description="Timestamp range (start, end) for audio/video"
    )
    excerpt: str = Field(..., description="Snippet of original text from source")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score for citation accuracy"
    )
    chunk_id: Optional[str] = Field(None, description="Source chunk identifier")
    line_range: Optional[Tuple[int, int]] = Field(
        None, description="Line range (start, end) for code files"
    )
    formatted: Optional[str] = Field(None, description="Formatted citation string")
    
    @field_validator("timestamp_range")
    @classmethod
    def validate_timestamp_range(cls, v: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """Validate timestamp range."""
        if v is not None:
            start, end = v
            if start < 0 or end < 0:
                raise ValueError("Timestamps must be non-negative")
            if start >= end:
                raise ValueError("Start timestamp must be less than end timestamp")
        return v
    
    @field_validator("line_range")
    @classmethod
    def validate_line_range(cls, v: Optional[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Validate line range."""
        if v is not None:
            start, end = v
            if start < 1 or end < 1:
                raise ValueError("Line numbers must be positive")
            if start > end:
                raise ValueError("Start line must be less than or equal to end line")
        return v


class CitationEngine:
    """Engine for generating accurate citations from LLM responses and retrieved chunks.
    
    Implements citation-driven architecture by mapping response segments to sources,
    validating citations, and formatting them appropriately per file type.
    """
    
    def __init__(self, db: Optional[SQLiteManager] = None):
        """Initialize citation engine.
        
        Args:
            db: Optional SQLiteManager for citation validation
        """
        self.db = db or SQLiteManager()
        self._file_cache: Dict[str, Dict[str, Any]] = {}  # Cache file metadata
        
        logger.info("CitationEngine initialized")
    
    async def generate_citations(
        self,
        response: str,
        retrieved_chunks: List[Any],
        min_confidence: float = 0.3,
    ) -> List[Citation]:
        """Generate citations by mapping LLM response to source chunks.
        
        Args:
            response: LLM-generated response text
            retrieved_chunks: List of retrieved Document objects or DocumentChunk objects
            min_confidence: Minimum confidence threshold for citations
        
        Returns:
            List of Citation objects with validated metadata
        
        Raises:
            ProcessingError: If citation generation fails
        """
        try:
            logger.debug(
                "Generating citations",
                response_length=len(response),
                chunk_count=len(retrieved_chunks),
            )
            
            # Convert chunks to standardized format
            standardized_chunks = self._standardize_chunks(retrieved_chunks)
            
            # Map response segments to chunks
            citation_candidates = self._map_response_to_chunks(
                response=response,
                chunks=standardized_chunks,
            )
            
            # Deduplicate overlapping citations
            deduplicated = self._deduplicate_citations(citation_candidates)
            
            # Build Citation objects
            citations = []
            for candidate in deduplicated:
                try:
                    citation = await self._build_citation(candidate)
                    if citation and citation.confidence >= min_confidence:
                        citations.append(citation)
                except Exception as e:
                    logger.warning(
                        "Failed to build citation",
                        candidate=candidate,
                        error=str(e),
                    )
                    continue
            
            # Validate citations
            validated_citations = await self._validate_citations(citations)
            
            # Format citations
            for citation in validated_citations:
                citation.formatted = self._format_citation(citation)
            
            logger.info(
                "Citations generated",
                total=len(validated_citations),
                validated=len([c for c in validated_citations if c.confidence >= 0.7]),
            )
            
            return validated_citations
            
        except Exception as e:
            logger.error(
                "Citation generation failed",
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(f"Citation generation failed: {str(e)}") from e
    
    def _standardize_chunks(self, chunks: List[Any]) -> List[Dict[str, Any]]:
        """Standardize chunks to common format.
        
        Args:
            chunks: List of Document or DocumentChunk objects
        
        Returns:
            List of standardized chunk dictionaries
        """
        standardized = []
        
        for chunk in chunks:
            if LANGCHAIN_AVAILABLE and isinstance(chunk, Document):
                # LangChain Document
                metadata = chunk.metadata if hasattr(chunk, "metadata") else {}
                content = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
                
                standardized.append({
                    "content": content,
                    "file_id": metadata.get("file_id") or metadata.get("source_file_id"),
                    "file_name": metadata.get("file_name", "Unknown"),
                    "chunk_id": metadata.get("chunk_id"),
                    "page_number": metadata.get("page_number"),
                    "timestamp_start": metadata.get("timestamp_start") or metadata.get("segment_start"),
                    "timestamp_end": metadata.get("timestamp_end") or metadata.get("segment_end"),
                    "mime_type": metadata.get("mime_type"),
                    "element_type": metadata.get("element_type"),
                    "metadata": metadata,
                })
            elif hasattr(chunk, "content") and hasattr(chunk, "file_id"):
                # DocumentChunk
                standardized.append({
                    "content": chunk.content,
                    "file_id": chunk.file_id,
                    "file_name": chunk.metadata.get("file_name", "Unknown") if hasattr(chunk, "metadata") else "Unknown",
                    "chunk_id": chunk.chunk_id if hasattr(chunk, "chunk_id") else None,
                    "page_number": chunk.metadata.get("page_number") if hasattr(chunk, "metadata") else None,
                    "timestamp_start": chunk.metadata.get("segment_start") if hasattr(chunk, "metadata") else None,
                    "timestamp_end": chunk.metadata.get("segment_end") if hasattr(chunk, "metadata") else None,
                    "mime_type": chunk.metadata.get("mime_type") if hasattr(chunk, "metadata") else None,
                    "element_type": chunk.metadata.get("element_type") if hasattr(chunk, "metadata") else None,
                    "metadata": chunk.metadata if hasattr(chunk, "metadata") else {},
                })
            else:
                logger.warning("Unknown chunk type, skipping", chunk_type=type(chunk))
                continue
        
        return standardized
    
    def _map_response_to_chunks(
        self,
        response: str,
        chunks: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Map response segments to source chunks using text similarity.
        
        Args:
            response: LLM response text
            chunks: List of standardized chunk dictionaries
        
        Returns:
            List of citation candidates with confidence scores
        """
        citation_candidates = []
        
        # Split response into sentences for granular mapping
        sentences = self._split_into_sentences(response)
        
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip very short sentences
                continue
            
            best_match = None
            best_score = 0.0
            
            for chunk in chunks:
                # Calculate similarity between sentence and chunk
                similarity = self._calculate_similarity(sentence, chunk["content"])
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = chunk
            
            if best_match and best_score > 0.3:  # Minimum similarity threshold
                # Extract relevant excerpt from chunk
                excerpt = self._extract_excerpt(sentence, best_match["content"])
                
                citation_candidates.append({
                    "sentence": sentence,
                    "chunk": best_match,
                    "confidence": best_score,
                    "excerpt": excerpt,
                })
        
        return citation_candidates
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts.
        
        Uses SequenceMatcher for semantic similarity approximation.
        
        Args:
            text1: First text
            text2: Second text
        
        Returns:
            Similarity score between 0.0 and 1.0
        """
        # Normalize texts
        text1_lower = text1.lower().strip()
        text2_lower = text2.lower().strip()
        
        # Check for exact substring match (higher confidence)
        if text1_lower in text2_lower or text2_lower in text1_lower:
            return 0.9
        
        # Use SequenceMatcher for similarity
        matcher = SequenceMatcher(None, text1_lower, text2_lower)
        similarity = matcher.ratio()
        
        # Boost similarity if key phrases match
        text1_words = set(text1_lower.split())
        text2_words = set(text2_lower.split())
        if len(text1_words) > 0:
            word_overlap = len(text1_words & text2_words) / len(text1_words)
            similarity = max(similarity, word_overlap * 0.8)
        
        return similarity
    
    def _extract_excerpt(self, query: str, content: str, context_chars: int = 100) -> str:
        """Extract relevant excerpt from content matching query.
        
        Args:
            query: Query text
            content: Full content text
            context_chars: Number of context characters around match
        
        Returns:
            Excerpt string
        """
        query_lower = query.lower()
        content_lower = content.lower()
        
        # Find best match position
        match_pos = content_lower.find(query_lower[:50])  # Match first 50 chars
        
        if match_pos == -1:
            # No direct match, return beginning
            return content[:200]
        
        # Extract excerpt with context
        start = max(0, match_pos - context_chars)
        end = min(len(content), match_pos + len(query) + context_chars)
        
        excerpt = content[start:end]
        
        # Clean up excerpt
        if start > 0:
            excerpt = "..." + excerpt
        if end < len(content):
            excerpt = excerpt + "..."
        
        return excerpt.strip()
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Text to split
        
        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be enhanced with NLTK/spaCy)
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _deduplicate_citations(
        self,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Deduplicate overlapping citations.
        
        Args:
            candidates: List of citation candidates
        
        Returns:
            Deduplicated list of citations
        """
        if not candidates:
            return []
        
        # Sort by confidence (highest first)
        sorted_candidates = sorted(candidates, key=lambda x: x["confidence"], reverse=True)
        
        deduplicated = []
        seen_chunks = set()
        
        for candidate in sorted_candidates:
            chunk_id = candidate["chunk"].get("chunk_id")
            file_id = candidate["chunk"].get("file_id")
            
            # Create unique key for chunk
            chunk_key = (file_id, chunk_id) if chunk_id else (file_id,)
            
            # Skip if we've already cited this chunk
            if chunk_key in seen_chunks:
                continue
            
            # Check for overlap with existing citations
            is_duplicate = False
            for existing in deduplicated:
                existing_chunk = existing["chunk"]
                
                # Same file and similar content = duplicate
                if (existing_chunk.get("file_id") == file_id and
                    self._calculate_similarity(
                        candidate["sentence"],
                        existing["sentence"]
                    ) > 0.7):
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(candidate)
                seen_chunks.add(chunk_key)
        
        return deduplicated
    
    async def _build_citation(self, candidate: Dict[str, Any]) -> Optional[Citation]:
        """Build Citation object from candidate.
        
        Args:
            candidate: Citation candidate dictionary
        
        Returns:
            Citation object or None if build fails
        """
        try:
            chunk = candidate["chunk"]
            confidence = candidate["confidence"]
            excerpt = candidate["excerpt"]
            
            file_id = chunk.get("file_id")
            file_name = chunk.get("file_name", "Unknown")
            mime_type = chunk.get("mime_type", "")
            
            if not file_id:
                logger.warning("Citation candidate missing file_id")
                return None
            
            # Determine file type
            file_type = get_file_category(file_name, mime_type)
            
            # Extract file type-specific metadata
            page_number = chunk.get("page_number")
            timestamp_start = chunk.get("timestamp_start")
            timestamp_end = chunk.get("timestamp_end")
            timestamp_range = None
            line_range = None
            
            if timestamp_start is not None and timestamp_end is not None:
                try:
                    timestamp_range = (float(timestamp_start), float(timestamp_end))
                except (ValueError, TypeError):
                    pass
            
            # Extract line range for code files
            if file_type == FileTypeCategory.CODE:
                line_range = self._extract_line_range(excerpt, chunk.get("content", ""))
            
            # Build citation
            citation = Citation(
                source_file_id=file_id,
                source_file_name=file_name,
                file_type=file_type,
                page_number=page_number,
                timestamp_range=timestamp_range,
                excerpt=excerpt,
                confidence=confidence,
                chunk_id=chunk.get("chunk_id"),
                line_range=line_range,
            )
            
            return citation
            
        except Exception as e:
            logger.error("Failed to build citation", error=str(e), exc_info=True)
            return None
    
    async def _validate_citations(self, citations: List[Citation]) -> List[Citation]:
        """Validate citations against database and metadata.
        
        Args:
            citations: List of Citation objects
        
        Returns:
            List of validated Citation objects (low-confidence ones flagged)
        """
        validated = []
        
        for citation in citations:
            try:
                # Verify file_id exists
                file_exists = await self._verify_file_exists(citation.source_file_id)
                
                if not file_exists:
                    logger.warning(
                        "Citation references non-existent file",
                        file_id=citation.source_file_id,
                    )
                    # Lower confidence but don't remove
                    citation.confidence *= 0.5
                
                # Verify page/timestamp references
                if citation.page_number is not None:
                    # Could validate against actual document page count
                    # For now, just check it's positive
                    if citation.page_number < 1:
                        citation.confidence *= 0.7
                
                if citation.timestamp_range is not None:
                    start, end = citation.timestamp_range
                    if start < 0 or end < 0:
                        citation.confidence *= 0.7
                
                validated.append(citation)
                
            except Exception as e:
                logger.warning(
                    "Citation validation failed",
                    citation=citation.source_file_id,
                    error=str(e),
                )
                # Include citation but with lower confidence
                citation.confidence *= 0.6
                validated.append(citation)
        
        return validated
    
    async def _verify_file_exists(self, file_id: str) -> bool:
        """Verify file exists in database.
        
        Args:
            file_id: File ID to verify
        
        Returns:
            True if file exists, False otherwise
        """
        try:
            # Check cache first
            if file_id in self._file_cache:
                return True
            
            # Query database
            async with self.db._get_db_manager().get_session() as session:
                from sqlalchemy import select
                from app.models.database import FileRecord
                
                result = await session.execute(
                    select(FileRecord).where(FileRecord.file_id == file_id)
                )
                file_record = result.scalar_one_or_none()
                
                if file_record:
                    # Cache file metadata
                    self._file_cache[file_id] = {
                        "file_id": file_record.file_id,
                        "file_name": file_record.file_name,
                        "mime_type": file_record.mime_type,
                    }
                    return True
                
                return False
                
        except Exception as e:
            logger.warning(
                "File existence check failed",
                file_id=file_id,
                error=str(e),
            )
            # Assume file exists if check fails (don't block citation)
            return True
    
    def _format_citation(self, citation: Citation) -> str:
        """Format citation string based on file type.
        
        Args:
            citation: Citation object
        
        Returns:
            Formatted citation string
        """
        file_name = citation.source_file_name
        
        if citation.file_type == FileTypeCategory.UNSTRUCTURED_NATIVE:
            # Documents: "Source: filename.pdf, Page 5"
            if citation.page_number:
                return f"Source: {file_name}, Page {citation.page_number}"
            else:
                return f"Source: {file_name}"
        
        elif citation.file_type == FileTypeCategory.AUDIO:
            # Audio: "Source: meeting.m4a, 12:34-13:45"
            if citation.timestamp_range:
                start, end = citation.timestamp_range
                start_str = self._format_timestamp(start)
                end_str = self._format_timestamp(end)
                return f"Source: {file_name}, {start_str}-{end_str}"
            else:
                return f"Source: {file_name}"
        
        elif citation.file_type == FileTypeCategory.CODE:
            # Code: "Source: auth.py, Lines 45-52"
            if citation.line_range:
                start_line, end_line = citation.line_range
                if start_line == end_line:
                    return f"Source: {file_name}, Line {start_line}"
                else:
                    return f"Source: {file_name}, Lines {start_line}-{end_line}"
            else:
                return f"Source: {file_name}"
        
        else:
            # Default format
            return f"Source: {file_name}"
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format timestamp in MM:SS or HH:MM:SS format.
        
        Args:
            seconds: Timestamp in seconds
        
        Returns:
            Formatted timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def _extract_line_range(self, excerpt: str, full_content: str) -> Optional[Tuple[int, int]]:
        """Extract line range from code excerpt.
        
        Args:
            excerpt: Excerpt text
            full_content: Full content text
        
        Returns:
            Tuple of (start_line, end_line) or None
        """
        try:
            # Find excerpt position in full content
            excerpt_clean = excerpt.replace("...", "").strip()
            if not excerpt_clean:
                return None
            
            # Find excerpt in content (case-insensitive)
            content_lower = full_content.lower()
            excerpt_lower = excerpt_clean.lower()
            
            # Try to find excerpt
            pos = content_lower.find(excerpt_lower[:100])  # Match first 100 chars
            if pos == -1:
                return None
            
            # Count lines up to position
            lines_before = full_content[:pos].count('\n')
            excerpt_lines = excerpt_clean.count('\n')
            
            start_line = lines_before + 1
            end_line = start_line + excerpt_lines
            
            return (start_line, end_line)
            
        except Exception as e:
            logger.debug("Failed to extract line range", error=str(e))
            return None

