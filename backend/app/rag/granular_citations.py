"""Granular citation system with precise text matching and quote extraction."""

import re
from typing import List, Dict, Any, Optional, Tuple
from difflib import SequenceMatcher
import structlog

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

logger = structlog.get_logger(__name__)


class GranularCitationExtractor:
    """
    Extracts granular citations with precise text matching, quotes, and semantic validation.

    Features:
    - Exact quote extraction from source documents
    - Character-level position tracking for UI highlighting
    - Sentence and paragraph-level granularity
    - Semantic similarity scoring between claims and sources
    - Contextual snippets with surrounding text
    """

    def __init__(self, min_quote_similarity: float = 0.6):
        """Initialize granular citation extractor.

        Args:
            min_quote_similarity: Minimum similarity score for quote matching (0.0-1.0)
        """
        self.min_quote_similarity = min_quote_similarity
        self.citation_pattern = re.compile(r'\[(\d+)\]')

    def extract_citations_with_quotes(
        self,
        response: str,
        source_documents: List[Document],
        include_context: bool = True,
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract citations with precise quotes and text positions.

        Args:
            response: LLM response with inline citations
            source_documents: Original source documents
            include_context: Whether to include surrounding context

        Returns:
            Tuple of (response, granular_citations)
        """
        # Find all citation numbers and their context
        citations_with_context = self._extract_citation_contexts(response)

        # Build granular citations
        granular_citations = []

        for citation_info in citations_with_context:
            cite_num = citation_info["citation_number"]
            claim_text = citation_info["claim_text"]

            # Get the source document
            doc_idx = cite_num - 1
            if doc_idx >= len(source_documents):
                logger.warning("Citation number out of range", citation_number=cite_num)
                continue

            source_doc = source_documents[doc_idx]

            # Extract quotes and positions
            quote_data = self._extract_best_quote(
                claim_text=claim_text,
                source_document=source_doc,
                include_context=include_context,
            )

            # Build granular citation
            metadata = source_doc.metadata if hasattr(source_doc, "metadata") else {}

            citation = {
                "citation_number": cite_num,
                "claim_text": claim_text,

                # File metadata
                "file_id": metadata.get("file_id"),
                "file_name": metadata.get("file_name", "Unknown"),
                "mime_type": metadata.get("mime_type"),

                # Location metadata
                "page_number": metadata.get("page_number"),
                "chunk_id": metadata.get("chunk_id"),
                "chunk_index": metadata.get("chunk_index"),

                # Audio/video metadata
                "start_time": metadata.get("start_time"),
                "end_time": metadata.get("end_time"),

                # Granular quote data
                "exact_quote": quote_data["exact_quote"],
                "quote_confidence": quote_data["confidence"],
                "char_start": quote_data["char_start"],
                "char_end": quote_data["char_end"],

                # Context
                "context_before": quote_data.get("context_before", ""),
                "context_after": quote_data.get("context_after", ""),

                # Sentence/paragraph level
                "sentence_index": quote_data.get("sentence_index"),
                "paragraph_index": quote_data.get("paragraph_index"),

                # URLs
                "google_drive_url": self._get_drive_url(metadata),

                # Full metadata
                "metadata": metadata,
            }

            granular_citations.append(citation)

        logger.info(
            "Extracted granular citations",
            total_citations=len(granular_citations),
            avg_confidence=sum(c["quote_confidence"] for c in granular_citations) / len(granular_citations) if granular_citations else 0,
        )

        return response, granular_citations

    def _extract_citation_contexts(self, response: str) -> List[Dict[str, str]]:
        """
        Extract citation numbers along with the surrounding claim text.

        Args:
            response: LLM response with citations

        Returns:
            List of dicts with citation_number and claim_text
        """
        citations = []

        # Split response into sentences
        sentences = self._split_into_sentences(response)

        for sentence in sentences:
            # Find citations in this sentence
            cite_matches = list(self.citation_pattern.finditer(sentence))

            if cite_matches:
                # Remove citation markers to get clean claim text
                claim_text = self.citation_pattern.sub('', sentence).strip()

                # Extract all citation numbers from this sentence
                for match in cite_matches:
                    try:
                        cite_num = int(match.group(1))
                        citations.append({
                            "citation_number": cite_num,
                            "claim_text": claim_text,
                            "full_sentence": sentence,
                        })
                    except ValueError:
                        continue

        return citations

    def _extract_best_quote(
        self,
        claim_text: str,
        source_document: Document,
        include_context: bool = True,
    ) -> Dict[str, Any]:
        """
        Find the best matching quote in the source document for a claim.

        Args:
            claim_text: The claim text from the response
            source_document: The source document to search
            include_context: Whether to include surrounding context

        Returns:
            Dictionary with quote data
        """
        source_content = (
            source_document.page_content
            if hasattr(source_document, "page_content")
            else str(source_document)
        )

        # Split source into sentences
        source_sentences = self._split_into_sentences(source_content)

        # Find best matching sentence(s)
        best_match = None
        best_score = 0.0

        for idx, sentence in enumerate(source_sentences):
            # Calculate similarity between claim and source sentence
            similarity = self._calculate_text_similarity(claim_text, sentence)

            if similarity > best_score and similarity >= self.min_quote_similarity:
                best_score = similarity
                best_match = {
                    "sentence": sentence,
                    "sentence_index": idx,
                    "similarity": similarity,
                }

        # If no good sentence match, try phrase matching
        if not best_match:
            phrase_match = self._find_best_phrase_match(claim_text, source_content)
            if phrase_match:
                best_match = phrase_match
                best_score = phrase_match["similarity"]

        # Build quote data
        if best_match:
            exact_quote = best_match["sentence"]
            char_start = source_content.find(exact_quote)
            char_end = char_start + len(exact_quote) if char_start != -1 else -1

            # Extract context
            context_before = ""
            context_after = ""

            if include_context and char_start != -1:
                # Get 100 chars before and after
                context_start = max(0, char_start - 100)
                context_end = min(len(source_content), char_end + 100)

                if context_start < char_start:
                    context_before = source_content[context_start:char_start].strip()
                if char_end < context_end:
                    context_after = source_content[char_end:context_end].strip()

            # Find paragraph index
            paragraph_index = self._find_paragraph_index(source_content, char_start)

            return {
                "exact_quote": exact_quote,
                "confidence": best_score,
                "char_start": char_start,
                "char_end": char_end,
                "context_before": context_before,
                "context_after": context_after,
                "sentence_index": best_match.get("sentence_index"),
                "paragraph_index": paragraph_index,
            }
        else:
            # No good match found, return fallback
            logger.warning(
                "No good quote match found",
                claim_text=claim_text[:100],
                min_similarity=self.min_quote_similarity,
            )

            # Return first 200 chars as fallback
            fallback_quote = source_content[:200].strip()

            return {
                "exact_quote": fallback_quote,
                "confidence": 0.0,
                "char_start": 0,
                "char_end": len(fallback_quote),
                "context_before": "",
                "context_after": source_content[200:300].strip() if len(source_content) > 200 else "",
                "sentence_index": 0,
                "paragraph_index": 0,
            }

    def _find_best_phrase_match(
        self,
        claim_text: str,
        source_content: str
    ) -> Optional[Dict[str, Any]]:
        """
        Find best matching phrase when sentence matching fails.

        Uses sliding window to find best substring match.

        Args:
            claim_text: Claim to match
            source_content: Source text

        Returns:
            Match data or None
        """
        # Extract key phrases from claim (3-10 word sequences)
        words = claim_text.split()
        best_match = None
        best_score = 0.0

        for window_size in range(min(10, len(words)), 2, -1):  # Try larger windows first
            for i in range(len(words) - window_size + 1):
                phrase = " ".join(words[i:i + window_size])

                # Find this phrase or similar in source
                if phrase.lower() in source_content.lower():
                    # Found exact match
                    start_idx = source_content.lower().find(phrase.lower())
                    actual_phrase = source_content[start_idx:start_idx + len(phrase)]

                    return {
                        "sentence": actual_phrase,
                        "sentence_index": None,
                        "similarity": 1.0,
                    }

                # Try fuzzy matching
                for match in re.finditer(r'[^.!?]+[.!?]', source_content):
                    sentence = match.group(0).strip()
                    similarity = self._calculate_text_similarity(phrase, sentence)

                    if similarity > best_score:
                        best_score = similarity
                        best_match = {
                            "sentence": sentence,
                            "sentence_index": None,
                            "similarity": similarity,
                        }

        return best_match if best_score >= self.min_quote_similarity else None

    def _split_into_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences.

        Args:
            text: Text to split

        Returns:
            List of sentences
        """
        # Simple sentence splitting (can be enhanced with NLP library)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate similarity between two text strings.

        Uses SequenceMatcher for now, can be enhanced with embeddings.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score 0.0-1.0
        """
        # Normalize texts
        t1 = text1.lower().strip()
        t2 = text2.lower().strip()

        # Use SequenceMatcher
        return SequenceMatcher(None, t1, t2).ratio()

    def _find_paragraph_index(self, text: str, char_position: int) -> int:
        """
        Find which paragraph a character position belongs to.

        Args:
            text: Full text
            char_position: Character position

        Returns:
            Paragraph index (0-based)
        """
        if char_position < 0:
            return 0

        # Split into paragraphs (double newline or single newline with indentation)
        paragraphs = re.split(r'\n\n+|\n(?=\s{2,})', text)

        current_pos = 0
        for idx, para in enumerate(paragraphs):
            current_pos += len(para) + 2  # +2 for newlines
            if current_pos > char_position:
                return idx

        return len(paragraphs) - 1

    def _get_drive_url(self, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Extract or construct Google Drive URL from metadata.

        Args:
            metadata: Document metadata

        Returns:
            Google Drive URL if available
        """
        if "google_drive_url" in metadata:
            return metadata["google_drive_url"]

        file_id = metadata.get("file_id")
        if file_id:
            return f"https://drive.google.com/file/d/{file_id}/view"

        return None

    def format_citation_for_ui(
        self,
        citation: Dict[str, Any],
        format_type: str = "detailed"
    ) -> Dict[str, Any]:
        """
        Format citation for UI display with highlighting support.

        Args:
            citation: Granular citation dictionary
            format_type: "detailed" or "compact"

        Returns:
            Formatted citation for UI
        """
        if format_type == "compact":
            return {
                "display_text": f"{citation['file_name']}, p.{citation['page_number']}" if citation.get('page_number') else citation['file_name'],
                "tooltip": citation['exact_quote'][:100] + "...",
                "url": citation.get('google_drive_url'),
            }

        # Detailed format
        return {
            "file_name": citation['file_name'],
            "page_number": citation.get('page_number'),
            "exact_quote": citation['exact_quote'],
            "confidence": citation['quote_confidence'],

            # For highlighting in UI
            "highlight": {
                "start": citation['char_start'],
                "end": citation['char_end'],
                "text": citation['exact_quote'],
            },

            # Context for preview
            "context": {
                "before": citation.get('context_before', ''),
                "quote": citation['exact_quote'],
                "after": citation.get('context_after', ''),
            },

            # Navigation
            "location": {
                "page": citation.get('page_number'),
                "paragraph": citation.get('paragraph_index'),
                "sentence": citation.get('sentence_index'),
                "timestamp": f"{citation.get('start_time')}-{citation.get('end_time')}" if citation.get('start_time') else None,
            },

            # Link
            "url": citation.get('google_drive_url'),
        }


# Global instance
_granular_extractor: Optional[GranularCitationExtractor] = None


def get_granular_citation_extractor(
    min_quote_similarity: float = 0.6
) -> GranularCitationExtractor:
    """Get global granular citation extractor instance.

    Args:
        min_quote_similarity: Minimum similarity for quote matching

    Returns:
        GranularCitationExtractor instance
    """
    global _granular_extractor
    if _granular_extractor is None:
        _granular_extractor = GranularCitationExtractor(
            min_quote_similarity=min_quote_similarity
        )
    return _granular_extractor
