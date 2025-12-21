"""Document-level relationship detection for cross-document analysis."""

from typing import List, Dict, Set, Optional, Tuple
from collections import defaultdict
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


class DocumentRelationshipDetector:
    """Detect relationships between entire documents (files)."""

    def __init__(self):
        """Initialize document relationship detector."""
        pass

    def detect_document_relationships(
        self,
        documents: List[Document],
    ) -> Dict[str, any]:
        """Detect relationships between documents.

        Args:
            documents: List of document chunks

        Returns:
            Dictionary with relationship analysis including:
            - shared_themes: List of themes appearing in multiple documents
            - document_topics: Mapping of file_name to primary topics
            - cross_references: Files that reference each other
            - unique_files: List of unique file names
        """
        if not documents:
            return {
                "shared_themes": [],
                "document_topics": {},
                "cross_references": [],
                "unique_files": [],
                "file_summaries": {},
            }

        # Group chunks by file
        file_chunks: Dict[str, List[Document]] = defaultdict(list)
        for doc in documents:
            metadata = doc.metadata if hasattr(doc, "metadata") else {}
            file_name = metadata.get("file_name", "Unknown")
            file_chunks[file_name].append(doc)

        unique_files = sorted(file_chunks.keys())

        # Extract topics/keywords from each file
        document_topics = {}
        file_keywords: Dict[str, Set[str]] = {}

        for file_name, chunks in file_chunks.items():
            # Combine all chunk content for this file
            combined_content = " ".join(
                doc.page_content if hasattr(doc, "page_content") else str(doc)
                for doc in chunks
            )

            # Extract keywords (simple approach - can be enhanced with NLP)
            keywords = self._extract_keywords(combined_content)
            file_keywords[file_name] = keywords

            # Get top keywords as topics
            top_topics = list(keywords)[:5]
            document_topics[file_name] = top_topics

        # Find shared themes across documents
        shared_themes = self._find_shared_themes(file_keywords)

        # Detect cross-references
        cross_references = self._detect_cross_references(file_chunks)

        # Generate file summaries
        file_summaries = {}
        for file_name, chunks in file_chunks.items():
            file_summaries[file_name] = {
                "chunk_count": len(chunks),
                "topics": document_topics.get(file_name, []),
                "total_content_length": sum(
                    len(doc.page_content if hasattr(doc, "page_content") else str(doc))
                    for doc in chunks
                ),
            }

        logger.info(
            "Document relationship detection completed",
            unique_files_count=len(unique_files),
            shared_themes_count=len(shared_themes),
            cross_references_count=len(cross_references),
        )

        return {
            "shared_themes": shared_themes,
            "document_topics": document_topics,
            "cross_references": cross_references,
            "unique_files": unique_files,
            "file_summaries": file_summaries,
        }

    def _extract_keywords(self, text: str, min_length: int = 4) -> Set[str]:
        """Extract keywords from text using simple heuristics.

        Args:
            text: Text to analyze
            min_length: Minimum keyword length

        Returns:
            Set of keywords
        """
        # Convert to lowercase and split
        words = text.lower().split()

        # Filter stopwords and short words
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "can", "this",
            "that", "these", "those", "it", "its", "they", "their", "them", "we",
            "our", "you", "your", "he", "she", "him", "her", "his", "i", "me",
            "my", "not", "no", "yes", "into", "out", "up", "down", "over", "under",
            "again", "further", "then", "once", "here", "there", "when", "where",
            "why", "how", "all", "both", "each", "few", "more", "most", "other",
            "some", "such", "than", "too", "very", "said", "just", "also",
        }

        keywords = set()
        for word in words:
            # Remove punctuation
            cleaned = "".join(c for c in word if c.isalnum())
            if cleaned and len(cleaned) >= min_length and cleaned not in stopwords:
                keywords.add(cleaned)

        return keywords

    def _find_shared_themes(
        self, file_keywords: Dict[str, Set[str]]
    ) -> List[Dict[str, any]]:
        """Find themes (keywords) shared across multiple files.

        Args:
            file_keywords: Mapping of file_name to keywords

        Returns:
            List of shared theme dictionaries
        """
        if len(file_keywords) < 2:
            return []

        # Count keyword occurrences across files
        keyword_files: Dict[str, Set[str]] = defaultdict(set)
        for file_name, keywords in file_keywords.items():
            for keyword in keywords:
                keyword_files[keyword].add(file_name)

        # Find keywords appearing in multiple files
        shared_themes = []
        for keyword, files in keyword_files.items():
            if len(files) >= 2:  # Appears in at least 2 files
                shared_themes.append({
                    "theme": keyword,
                    "files": sorted(list(files)),
                    "file_count": len(files),
                })

        # Sort by number of files (most shared first)
        shared_themes.sort(key=lambda x: x["file_count"], reverse=True)

        return shared_themes[:10]  # Top 10 shared themes

    def _detect_cross_references(
        self, file_chunks: Dict[str, List[Document]]
    ) -> List[Dict[str, str]]:
        """Detect if files reference each other by name.

        Args:
            file_chunks: Mapping of file_name to chunks

        Returns:
            List of cross-reference dictionaries
        """
        cross_references = []

        for file_a, chunks_a in file_chunks.items():
            # Combine content from file A
            content_a = " ".join(
                doc.page_content if hasattr(doc, "page_content") else str(doc)
                for doc in chunks_a
            )

            for file_b in file_chunks.keys():
                if file_a == file_b:
                    continue

                # Check if file A mentions file B
                file_b_base = file_b.rsplit(".", 1)[0] if "." in file_b else file_b

                if file_b in content_a or file_b_base in content_a:
                    cross_references.append({
                        "source_file": file_a,
                        "target_file": file_b,
                        "reference_type": "file_mention",
                    })

        return cross_references

    def generate_cross_document_summary(
        self, relationship_data: Dict[str, any]
    ) -> str:
        """Generate a human-readable summary of cross-document relationships.

        Args:
            relationship_data: Output from detect_document_relationships

        Returns:
            Summary string
        """
        unique_files = relationship_data.get("unique_files", [])
        shared_themes = relationship_data.get("shared_themes", [])
        document_topics = relationship_data.get("document_topics", {})

        if not unique_files:
            return "No documents found."

        if len(unique_files) == 1:
            return f"Single document: {unique_files[0]}"

        summary_parts = []
        summary_parts.append(f"{len(unique_files)} documents: {', '.join(unique_files)}")

        if shared_themes:
            top_themes = [t["theme"] for t in shared_themes[:3]]
            summary_parts.append(f"Shared themes: {', '.join(top_themes)}")

        if document_topics:
            for file_name, topics in document_topics.items():
                if topics:
                    summary_parts.append(f"{file_name}: {', '.join(topics[:3])}")

        return " | ".join(summary_parts)


# Global instance
_detector: Optional[DocumentRelationshipDetector] = None


def get_document_relationship_detector() -> DocumentRelationshipDetector:
    """Get global document relationship detector instance.

    Returns:
        DocumentRelationshipDetector instance
    """
    global _detector
    if _detector is None:
        _detector = DocumentRelationshipDetector()
    return _detector
