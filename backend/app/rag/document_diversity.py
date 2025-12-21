"""Document diversity utilities for cross-document retrieval."""

from typing import List, Dict, Set, Optional
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


class DocumentDiversifier:
    """Ensures diverse document representation in retrieved results."""

    def __init__(
        self,
        min_chunks_per_file: int = 2,
        max_total_chunks: int = 10,
    ):
        """Initialize document diversifier.

        Args:
            min_chunks_per_file: Minimum chunks to retrieve from each unique file
            max_total_chunks: Maximum total chunks to return
        """
        self.min_chunks_per_file = min_chunks_per_file
        self.max_total_chunks = max_total_chunks

    def diversify_by_document(
        self,
        documents: List[Document],
        preserve_scores: bool = True,
    ) -> List[Document]:
        """Ensure diverse representation across multiple files.

        Strategy:
        1. Group documents by file_name
        2. Ensure at least min_chunks_per_file from each unique file
        3. Fill remaining slots with highest-scored chunks
        4. Maintain score-based ordering within constraints

        Args:
            documents: List of retrieved documents with metadata
            preserve_scores: If True, preserve relevance_score in metadata

        Returns:
            Diversified list of documents with balanced file representation
        """
        if not documents:
            return []

        # Group by file
        file_groups: Dict[str, List[Document]] = defaultdict(list)
        for doc in documents:
            metadata = doc.metadata if hasattr(doc, "metadata") else {}
            file_name = metadata.get("file_name", "Unknown")
            file_groups[file_name].append(doc)

        unique_files = list(file_groups.keys())
        num_files = len(unique_files)

        logger.debug(
            "Document diversity analysis",
            total_chunks=len(documents),
            unique_files=num_files,
            files=unique_files,
        )

        # If only one file, return as-is
        if num_files == 1:
            return documents[:self.max_total_chunks]

        # Calculate slots per file
        guaranteed_slots = self.min_chunks_per_file * num_files

        if guaranteed_slots > self.max_total_chunks:
            # Too many files for guaranteed minimum, distribute evenly
            slots_per_file = max(1, self.max_total_chunks // num_files)
            logger.debug(
                "Too many files for guaranteed minimum, distributing evenly",
                slots_per_file=slots_per_file,
            )
        else:
            slots_per_file = self.min_chunks_per_file

        # Select diverse chunks
        selected_docs = []
        file_chunk_counts: Dict[str, int] = defaultdict(int)

        # Phase 1: Ensure minimum representation from each file
        for file_name in sorted(unique_files):
            file_docs = file_groups[file_name]
            # Take top chunks from this file
            for doc in file_docs[:slots_per_file]:
                selected_docs.append(doc)
                file_chunk_counts[file_name] += 1

                if len(selected_docs) >= self.max_total_chunks:
                    break

            if len(selected_docs) >= self.max_total_chunks:
                break

        # Phase 2: Fill remaining slots with highest-scored chunks
        if len(selected_docs) < self.max_total_chunks:
            remaining_slots = self.max_total_chunks - len(selected_docs)

            # Collect remaining chunks not yet selected
            selected_ids = set()
            for doc in selected_docs:
                chunk_id = doc.metadata.get("chunk_id") if hasattr(doc, "metadata") else None
                if chunk_id:
                    selected_ids.add(chunk_id)

            remaining_docs = []
            for doc in documents:
                chunk_id = doc.metadata.get("chunk_id") if hasattr(doc, "metadata") else None
                if chunk_id and chunk_id not in selected_ids:
                    remaining_docs.append(doc)

            # Add top remaining chunks
            selected_docs.extend(remaining_docs[:remaining_slots])

        # Sort by original order (preserves relevance ranking as much as possible)
        # Create a mapping from chunk_id to original index
        original_order = {}
        for idx, doc in enumerate(documents):
            chunk_id = doc.metadata.get("chunk_id") if hasattr(doc, "metadata") else None
            if chunk_id:
                original_order[chunk_id] = idx

        selected_docs.sort(
            key=lambda doc: original_order.get(
                doc.metadata.get("chunk_id") if hasattr(doc, "metadata") else None,
                float('inf')
            )
        )

        logger.info(
            "Document diversification completed",
            original_count=len(documents),
            diversified_count=len(selected_docs),
            files_represented=len(set(
                doc.metadata.get("file_name", "Unknown")
                for doc in selected_docs
                if hasattr(doc, "metadata")
            )),
        )

        return selected_docs[:self.max_total_chunks]

    def get_file_distribution(self, documents: List[Document]) -> Dict[str, int]:
        """Get distribution of chunks across files.

        Args:
            documents: List of documents

        Returns:
            Dictionary mapping file_name to chunk count
        """
        distribution: Dict[str, int] = defaultdict(int)
        for doc in documents:
            metadata = doc.metadata if hasattr(doc, "metadata") else {}
            file_name = metadata.get("file_name", "Unknown")
            distribution[file_name] += 1

        return dict(distribution)


# Global diversifier instance
_diversifier: Optional[DocumentDiversifier] = None


def get_document_diversifier(
    min_chunks_per_file: int = 2,
    max_total_chunks: int = 10,
) -> DocumentDiversifier:
    """Get global document diversifier instance.

    Args:
        min_chunks_per_file: Minimum chunks per file
        max_total_chunks: Maximum total chunks

    Returns:
        DocumentDiversifier instance
    """
    global _diversifier
    if _diversifier is None:
        _diversifier = DocumentDiversifier(
            min_chunks_per_file=min_chunks_per_file,
            max_total_chunks=max_total_chunks,
        )
    return _diversifier
