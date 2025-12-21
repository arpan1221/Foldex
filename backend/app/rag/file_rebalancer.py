"""Rebalance retrieval results to prevent overrepresented files from dominating.

Applies downweighting to files with many chunks to ensure diverse results.
"""

from typing import List, Dict, Tuple
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


class FileRebalancer:
    """Rebalance retrieval results by downweighting overrepresented files.
    
    Strategies:
    1. Score normalization: Divide score by sqrt(chunks_per_file) to reduce bias
    2. Chunk capping: Limit max chunks per file in final results
    3. Inverse frequency weighting: Downweight files with many chunks
    """
    
    def __init__(
        self,
        max_chunks_per_file: int = 5,
        use_score_normalization: bool = True,
        normalization_factor: float = 0.5,  # Exponent for normalization (0.5 = sqrt, 1.0 = linear)
    ):
        """Initialize file rebalancer.
        
        Args:
            max_chunks_per_file: Maximum chunks to include per file in final results
            use_score_normalization: Whether to normalize scores by file chunk count
            normalization_factor: Exponent for normalization (0.5 = sqrt reduces bias)
        """
        self.max_chunks_per_file = max_chunks_per_file
        self.use_score_normalization = use_score_normalization
        self.normalization_factor = normalization_factor
    
    def rebalance_scored_documents(
        self,
        scored_documents: List[Tuple[Document, float]],
    ) -> List[Tuple[Document, float]]:
        """Rebalance scored documents by downweighting overrepresented files.
        
        Args:
            scored_documents: List of (Document, score) tuples
            
        Returns:
            Rebalanced list of (Document, score) tuples
        """
        if not scored_documents:
            return scored_documents
        
        # Count chunks per file
        file_chunk_counts: Dict[str, int] = defaultdict(int)
        file_chunks: Dict[str, List[Tuple[Document, float]]] = defaultdict(list)
        
        for doc, score in scored_documents:
            file_id = doc.metadata.get("file_id") if hasattr(doc, "metadata") and doc.metadata else None
            file_name = doc.metadata.get("file_name") if hasattr(doc, "metadata") and doc.metadata else "unknown"
            file_key = file_id or file_name
            
            file_chunk_counts[file_key] += 1
            file_chunks[file_key].append((doc, score))
        
        # Calculate normalization factors
        max_count = max(file_chunk_counts.values()) if file_chunk_counts else 1
        normalization_factors: Dict[str, float] = {}
        
        for file_key, count in file_chunk_counts.items():
            if self.use_score_normalization and count > 1:
                # Normalize: score / (count ** normalization_factor)
                # For normalization_factor=0.5: score / sqrt(count)
                # This reduces the advantage of files with many chunks
                normalization_factors[file_key] = count ** self.normalization_factor
            else:
                normalization_factors[file_key] = 1.0
        
        # Apply normalization and rebalance
        rebalanced: List[Tuple[Document, float]] = []
        
        for file_key, chunks in file_chunks.items():
            norm_factor = normalization_factors[file_key]
            
            # Normalize scores
            normalized_chunks = [
                (doc, score / norm_factor)
                for doc, score in chunks
            ]
            
            # Sort by normalized score (descending)
            normalized_chunks.sort(key=lambda x: x[1], reverse=True)
            
            # Cap chunks per file
            capped_chunks = normalized_chunks[:self.max_chunks_per_file]
            
            rebalanced.extend(capped_chunks)
            
            if len(chunks) > self.max_chunks_per_file:
                logger.debug(
                    "Capped chunks for file",
                    file_key=file_key,
                    original_count=len(chunks),
                    capped_count=len(capped_chunks),
                    normalization_factor=norm_factor,
                )
        
        # Sort all rebalanced results by normalized score
        rebalanced.sort(key=lambda x: x[1], reverse=True)
        
        logger.info(
            "File rebalancing completed",
            original_count=len(scored_documents),
            rebalanced_count=len(rebalanced),
            files_represented=len(file_chunk_counts),
            max_chunks_per_file=self.max_chunks_per_file,
            normalization_applied=self.use_score_normalization,
        )
        
        return rebalanced
    
    def rebalance_documents(
        self,
        documents: List[Document],
    ) -> List[Document]:
        """Rebalance documents by limiting chunks per file.
        
        Simple version without scores - just caps chunks per file.
        
        Args:
            documents: List of Document objects
            
        Returns:
            Rebalanced list of Document objects
        """
        if not documents:
            return documents
        
        # Group by file
        file_chunks: Dict[str, List[Document]] = defaultdict(list)
        
        for doc in documents:
            file_id = doc.metadata.get("file_id") if hasattr(doc, "metadata") and doc.metadata else None
            file_name = doc.metadata.get("file_name") if hasattr(doc, "metadata") and doc.metadata else "unknown"
            file_key = file_id or file_name
            
            file_chunks[file_key].append(doc)
        
        # Cap chunks per file and maintain order
        rebalanced: List[Document] = []
        seen_doc_ids = set()
        
        for file_key, chunks in file_chunks.items():
            # Take first N chunks (preserving original order)
            capped = chunks[:self.max_chunks_per_file]
            
            for doc in capped:
                doc_id = doc.metadata.get("chunk_id") if hasattr(doc, "metadata") and doc.metadata else None
                if doc_id and doc_id not in seen_doc_ids:
                    rebalanced.append(doc)
                    seen_doc_ids.add(doc_id)
            
            if len(chunks) > self.max_chunks_per_file:
                logger.debug(
                    "Capped chunks for file",
                    file_key=file_key,
                    original_count=len(chunks),
                    capped_count=len(capped),
                )
        
        logger.info(
            "File rebalancing completed (no scores)",
            original_count=len(documents),
            rebalanced_count=len(rebalanced),
            files_represented=len(file_chunks),
            max_chunks_per_file=self.max_chunks_per_file,
        )
        
        return rebalanced


# Global rebalancer instance
_rebalancer: FileRebalancer = None


def get_file_rebalancer(
    max_chunks_per_file: int = 5,
    use_score_normalization: bool = True,
    normalization_factor: float = 0.5,
) -> FileRebalancer:
    """Get global file rebalancer instance.
    
    Args:
        max_chunks_per_file: Maximum chunks per file
        use_score_normalization: Whether to normalize scores
        normalization_factor: Normalization exponent
        
    Returns:
        FileRebalancer instance
    """
    global _rebalancer
    if _rebalancer is None or _rebalancer.max_chunks_per_file != max_chunks_per_file:
        _rebalancer = FileRebalancer(
            max_chunks_per_file=max_chunks_per_file,
            use_score_normalization=use_score_normalization,
            normalization_factor=normalization_factor,
        )
    return _rebalancer

