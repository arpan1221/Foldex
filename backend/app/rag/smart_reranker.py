"""Query-type-aware reranking system.

Applies different reranking strategies based on query classification:
- FACTUAL_SPECIFIC: Skip reranking (filtering ensures precision)
- RELATIONSHIP: Diversity-boosting reranking
- TEMPORAL: Recency-boosting reranking
- ENTITY_SEARCH: Entity match boosting
- FACTUAL_GENERAL: Balanced reranking with diversity boost
"""

from typing import List, Dict, Optional, Any, Tuple
from datetime import datetime, timezone
from dataclasses import dataclass
import structlog

from app.rag.query_classifier import QueryUnderstanding, QueryType
from app.config.settings import settings

logger = structlog.get_logger(__name__)


@dataclass
class RerankedChunk:
    """Chunk with reranking metadata."""
    
    chunk: Dict[str, Any]
    original_score: float
    reranked_score: float
    adjustments: Dict[str, float]  # Breakdown of score adjustments


class SmartReranker:
    """
    Query-type-aware reranker that applies appropriate strategies
    based on query classification.
    
    Reranking is conditional - only applied when it improves results.
    """
    
    def __init__(
        self,
        diversity_weight: Optional[float] = None,
        recency_weight: Optional[float] = None,
        entity_boost: Optional[float] = None,
    ):
        """Initialize smart reranker.
        
        Args:
            diversity_weight: How much to penalize overrepresented files (0.0-1.0)
            recency_weight: How much to boost recent files (0.0-1.0)
            entity_boost: Multiplier for entity matches (>= 1.0)
        """
        self.diversity_weight = diversity_weight or 0.3
        self.recency_weight = recency_weight or 0.2
        self.entity_boost = entity_boost or 1.3
        
        # Validate weights
        assert 0.0 <= self.diversity_weight <= 1.0, "diversity_weight must be 0.0-1.0"
        assert 0.0 <= self.recency_weight <= 1.0, "recency_weight must be 0.0-1.0"
        assert self.entity_boost >= 1.0, "entity_boost must be >= 1.0"
    
    def should_rerank(self, query_type: QueryType) -> bool:
        """
        Determine if reranking should be applied for this query type.
        
        Args:
            query_type: Query type from classification
            
        Returns:
            True if reranking would help, False otherwise
        """
        # Skip reranking for FACTUAL_SPECIFIC - filtering already ensures precision
        if query_type == QueryType.FACTUAL_SPECIFIC:
            return False
        
        # ENTITY_SEARCH can benefit from entity boost, so we allow reranking
        # The rerank() method will only apply entity boost, not diversity reranking
        
        # Apply reranking for other types
        return True
    
    def rerank(
        self,
        chunks: List[Dict[str, Any]],
        query_type: QueryType,
        understanding: QueryUnderstanding,
    ) -> List[Dict[str, Any]]:
        """
        Apply appropriate reranking strategy based on query type.
        
        Args:
            chunks: List of chunk dicts with 'score' key
            query_type: Query type from classification
            understanding: Query understanding with entities and metadata
            
        Returns:
            Reranked list of chunks
        """
        if not self.should_rerank(query_type):
            logger.debug("Skipping reranking", query_type=query_type.value, reason="not beneficial for this query type")
            return chunks
        
        if not chunks:
            return chunks
        
        logger.info(
            "🎯 Applying smart reranking",
            query_type=query_type.value,
            chunks_count=len(chunks),
        )
        
        # Apply strategy based on query type
        if query_type == QueryType.RELATIONSHIP:
            reranked = self._diversity_rerank(chunks, self.diversity_weight)
        elif query_type == QueryType.TEMPORAL:
            reranked = self._recency_rerank(chunks, self.recency_weight)
        elif query_type == QueryType.ENTITY_SEARCH:
            # Extract entity and apply entity boost
            entities = understanding.entities or []
            entity = entities[0] if entities else None
            if entity:
                reranked = self._entity_boost_rerank(chunks, entity, self.entity_boost)
            else:
                reranked = chunks
        elif query_type == QueryType.COMPARISON:
            # Apply diversity reranking for comparison to ensure balanced representation
            reranked = self._diversity_rerank(chunks, self.diversity_weight * 0.5)  # Lighter penalty
        elif query_type == QueryType.FACTUAL_GENERAL:
            # Balanced: apply diversity reranking with moderate weight
            reranked = self._diversity_rerank(chunks, self.diversity_weight * 0.7)  # Moderate penalty
        else:
            # Unknown type - skip reranking
            logger.warning("Unknown query type for reranking, skipping", query_type=query_type.value)
            return chunks
        
        logger.info(
            "✅ Smart reranking complete",
            query_type=query_type.value,
            reranked_count=len(reranked),
        )
        
        return reranked
    
    def _diversity_rerank(
        self,
        chunks: List[Dict[str, Any]],
        weight: float,
    ) -> List[Dict[str, Any]]:
        """
        Apply diversity-boosting reranking to penalize overrepresented files.
        
        Algorithm:
        1. Track how many times each file appears in candidates
        2. For each chunk, calculate diversity penalty:
           penalty = (file_count / max_file_count) * diversity_weight
        3. Adjust score: new_score = original_score * (1 - penalty)
        4. Re-sort by new scores
        
        Args:
            chunks: List of chunk dicts with 'score' and 'file_id'/'file_name' keys
            weight: Diversity penalty weight (0.0-1.0)
            
        Returns:
            Reranked chunks with adjusted scores
        """
        if not chunks:
            return chunks
        
        # Count chunks per file
        file_counts: Dict[str, int] = {}
        for chunk in chunks:
            file_id = chunk.get("file_id") or chunk.get("file_name", "unknown")
            file_counts[file_id] = file_counts.get(file_id, 0) + 1
        
        if not file_counts:
            return chunks
        
        max_file_count = max(file_counts.values())
        if max_file_count == 0:
            return chunks
        
        logger.debug(
            "Diversity reranking",
            unique_files=len(file_counts),
            max_chunks_per_file=max_file_count,
            weight=weight,
        )
        
        # Calculate adjusted scores
        reranked_chunks = []
        file_positions: Dict[str, int] = {}  # Track position of each file
        
        for chunk in chunks:
            file_id = chunk.get("file_id") or chunk.get("file_name", "unknown")
            original_score = chunk.get("score", 0.0)
            
            # Track how many chunks from this file we've already seen
            file_positions[file_id] = file_positions.get(file_id, 0) + 1
            file_position = file_positions[file_id]
            file_count = file_counts[file_id]
            
            # Calculate penalty: more chunks from this file = higher penalty
            # Penalty increases with position (first chunk gets less penalty)
            position_factor = file_position / file_count  # 0.0 (first) to 1.0 (last)
            file_representation = file_count / max_file_count  # 0.0 to 1.0
            
            # Penalty formula: penalize files with many chunks, and later chunks more
            penalty = file_representation * position_factor * weight
            
            # Adjust score: new_score = original_score * (1 - penalty)
            adjusted_score = original_score * (1.0 - penalty)
            
            reranked_chunk = chunk.copy()
            reranked_chunk["score"] = adjusted_score
            reranked_chunk["_rerank_penalty"] = penalty
            reranked_chunk["_original_score"] = original_score
            
            reranked_chunks.append(reranked_chunk)
        
        # Re-sort by adjusted scores (descending)
        reranked_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
        logger.debug(
            "Diversity reranking complete",
            file_distribution_before={k: v for k, v in list(file_counts.items())[:5]},
        )
        
        return reranked_chunks
    
    def _recency_rerank(
        self,
        chunks: List[Dict[str, Any]],
        weight: float,
    ) -> List[Dict[str, Any]]:
        """
        Apply recency-boosting reranking for temporal queries.
        
        Algorithm:
        1. Extract file modification time from metadata
        2. Calculate recency_score = 1 / (days_since_modified + 1)
        3. Boost score: new_score = original_score * (1 + recency_score * recency_weight)
        
        Args:
            chunks: List of chunk dicts with 'score' and metadata
            weight: Recency boost weight (0.0-1.0)
            
        Returns:
            Reranked chunks with recency-boosted scores
        """
        if not chunks:
            return chunks
        
        current_time = datetime.now(timezone.utc)
        
        logger.debug(
            "Recency reranking",
            weight=weight,
            current_time=current_time.isoformat(),
        )
        
        reranked_chunks = []
        
        for chunk in chunks:
            original_score = chunk.get("score", 0.0)
            metadata = chunk.get("metadata", {})
            
            # Try to extract modification time from metadata
            # Common fields: modified_at, created_at, date, timestamp
            mod_time = None
            for field in ["modified_at", "created_at", "date", "timestamp", "file_modified_at"]:
                if field in metadata:
                    mod_time = metadata[field]
                    break
            
            if mod_time:
                # Parse datetime if string
                if isinstance(mod_time, str):
                    try:
                        mod_time = datetime.fromisoformat(mod_time.replace('Z', '+00:00'))
                    except (ValueError, AttributeError):
                        mod_time = None
                
                # Calculate days since modification
                if isinstance(mod_time, datetime):
                    if mod_time.tzinfo is None:
                        mod_time = mod_time.replace(tzinfo=timezone.utc)
                    
                    time_diff = current_time - mod_time
                    days_since_modified = time_diff.total_seconds() / (24 * 3600)
                    
                    # Recency score: 1.0 for today, decreasing over time
                    recency_score = 1.0 / (days_since_modified + 1.0)
                    
                    # Boost score: new_score = original_score * (1 + recency_score * weight)
                    boost = recency_score * weight
                    adjusted_score = original_score * (1.0 + boost)
                    
                    reranked_chunk = chunk.copy()
                    reranked_chunk["score"] = adjusted_score
                    reranked_chunk["_recency_boost"] = boost
                    reranked_chunk["_original_score"] = original_score
                    reranked_chunk["_days_since_modified"] = days_since_modified
                    
                    reranked_chunks.append(reranked_chunk)
                    continue
            
            # No modification time available - keep original score
            reranked_chunk = chunk.copy()
            reranked_chunk["_original_score"] = original_score
            reranked_chunks.append(reranked_chunk)
        
        # Re-sort by adjusted scores (descending)
        reranked_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
        logger.debug("Recency reranking complete", reranked_count=len(reranked_chunks))
        
        return reranked_chunks
    
    def _entity_boost_rerank(
        self,
        chunks: List[Dict[str, Any]],
        entity: str,
        boost: float,
    ) -> List[Dict[str, Any]]:
        """
        Apply entity match boosting for entity search queries.
        
        Algorithm:
        1. Extract entity from query understanding
        2. Check if chunk content contains entity (case-insensitive)
        3. If match: boost_score = original_score * boost
        4. If no match: keep original score
        
        Args:
            chunks: List of chunk dicts with 'score' and 'content' keys
            entity: Entity name to search for
            boost: Boost multiplier for matches (>= 1.0)
            
        Returns:
            Reranked chunks with entity-boosted scores
        """
        if not chunks:
            return chunks
        
        entity_lower = entity.lower()
        
        logger.debug(
            "Entity boost reranking",
            entity=entity,
            boost=boost,
        )
        
        reranked_chunks = []
        matched_count = 0
        
        for chunk in chunks:
            original_score = chunk.get("score", 0.0)
            content = chunk.get("content", "")
            
            # Check if content contains entity (case-insensitive)
            if entity_lower in content.lower():
                # Boost score
                adjusted_score = original_score * boost
                
                reranked_chunk = chunk.copy()
                reranked_chunk["score"] = adjusted_score
                reranked_chunk["_entity_boost"] = boost
                reranked_chunk["_original_score"] = original_score
                reranked_chunk["_entity_matched"] = True
                
                matched_count += 1
            else:
                # No match - keep original score
                reranked_chunk = chunk.copy()
                reranked_chunk["_original_score"] = original_score
                reranked_chunk["_entity_matched"] = False
            
            reranked_chunks.append(reranked_chunk)
        
        # Re-sort by adjusted scores (descending)
        reranked_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        
        logger.info(
            "Entity boost reranking complete",
            entity=entity,
            matched_chunks=matched_count,
            total_chunks=len(chunks),
            boost_applied=boost,
        )
        
        return reranked_chunks


# Global reranker instance
_reranker: Optional[SmartReranker] = None


def get_smart_reranker() -> SmartReranker:
    """Get global smart reranker instance."""
    global _reranker
    if _reranker is None:
        _reranker = SmartReranker()
    return _reranker

