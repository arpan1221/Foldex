"""Relationship detection between documents and entities."""

from typing import List, Dict, Optional
import re
import structlog
from datetime import datetime

from app.models.documents import DocumentChunk
from app.knowledge_graph.entity_extractor import EntityExtractor

logger = structlog.get_logger(__name__)

# Try to import numpy for cosine similarity
try:
    import numpy as np
    from numpy.linalg import norm
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None


class RelationshipDetector:
    """Detects relationships between documents and entities."""

    RELATIONSHIP_TYPES = {
        "entity_overlap": "Documents mention same entities",
        "temporal": "Documents created in sequence",
        "cross_reference": "Document A references Document B",
        "topical_similarity": "Documents share themes",
        "implementation_gap": "Code doesn't match specifications",
    }

    def __init__(self, entity_extractor: Optional[EntityExtractor] = None):
        """Initialize relationship detector.
        
        Args:
            entity_extractor: Optional entity extractor instance
        """
        self.entity_extractor = entity_extractor or EntityExtractor()

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity score between -1 and 1
        """
        if not NUMPY_AVAILABLE:
            # Fallback to manual calculation
            dot_product = sum(a * b for a, b in zip(vec1, vec2))
            magnitude1 = sum(a * a for a in vec1) ** 0.5
            magnitude2 = sum(b * b for b in vec2) ** 0.5
            if magnitude1 == 0 or magnitude2 == 0:
                return 0.0
            return dot_product / (magnitude1 * magnitude2)
        
        vec1_array = np.array(vec1)
        vec2_array = np.array(vec2)
        return float(np.dot(vec1_array, vec2_array) / (norm(vec1_array) * norm(vec2_array)))

    async def _detect_entity_overlap(
        self, chunks: List[DocumentChunk]
    ) -> List[Dict]:
        """Detect relationships based on shared entities.
        
        Two chunks share ≥2 entities → entity_overlap relationship
        
        Args:
            chunks: Document chunks to analyze
            
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        # Extract entities for each chunk
        chunk_entities = {}
        for chunk in chunks:
            entities = await self.entity_extractor.extract_entities(chunk.content)
            # Create set of entity texts for fast lookup
            entity_texts = set(e["text"].lower() for e in entities)
            chunk_entities[chunk.chunk_id] = entity_texts
        
        # Find overlaps
        for i, chunk_a in enumerate(chunks):
            for chunk_b in chunks[i + 1:]:
                overlap = chunk_entities[chunk_a.chunk_id] & chunk_entities[chunk_b.chunk_id]
                
                if len(overlap) >= 2:  # At least 2 shared entities
                    confidence = min(len(overlap) / 5.0, 1.0)  # More overlap = higher confidence
                    relationships.append({
                        "source_chunk_id": chunk_a.chunk_id,
                        "target_chunk_id": chunk_b.chunk_id,
                        "type": "entity_overlap",
                        "confidence": confidence,
                        "metadata": {
                            "shared_entities": list(overlap),
                            "overlap_count": len(overlap),
                        },
                    })
        
        logger.debug("Entity overlap detection", relationship_count=len(relationships))
        return relationships

    async def _detect_cross_references(
        self, chunks: List[DocumentChunk]
    ) -> List[Dict]:
        """Detect cross-references between chunks.
        
        Chunk A mentions filename/function from Chunk B → cross_reference relationship
        
        Args:
            chunks: Document chunks to analyze
            
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        # Build index of file names and function names
        file_index = {}
        function_index = {}
        
        for chunk in chunks:
            file_name = chunk.metadata.get("file_name", "")
            if file_name:
                file_index[chunk.chunk_id] = file_name
            
            # Extract functions from chunk content
            functions = re.findall(
                r'\b(?:def|function)\s+(\w+)\s*\(', chunk.content
            )
            function_index[chunk.chunk_id] = functions
        
        # Check if any chunk mentions another's file/function
        for chunk_a in chunks:
            for chunk_b in chunks:
                if chunk_a.chunk_id == chunk_b.chunk_id:
                    continue
                
                # Does A mention B's file?
                if chunk_b.chunk_id in file_index:
                    file_name = file_index[chunk_b.chunk_id]
                    # Check for file name mentions (with or without extension)
                    file_base = file_name.rsplit('.', 1)[0] if '.' in file_name else file_name
                    if file_name in chunk_a.content or file_base in chunk_a.content:
                        relationships.append({
                            "source_chunk_id": chunk_a.chunk_id,
                            "target_chunk_id": chunk_b.chunk_id,
                            "type": "cross_reference",
                            "confidence": 0.9,
                            "metadata": {
                                "reference_type": "file_mention",
                                "referenced_file": file_name,
                            },
                        })
                
                # Does A mention B's functions?
                if chunk_b.chunk_id in function_index:
                    for func in function_index[chunk_b.chunk_id]:
                        # Check for function calls or references
                        if re.search(rf'\b{re.escape(func)}\s*\(', chunk_a.content):
                            relationships.append({
                                "source_chunk_id": chunk_a.chunk_id,
                                "target_chunk_id": chunk_b.chunk_id,
                                "type": "cross_reference",
                                "confidence": 0.95,
                                "metadata": {
                                    "reference_type": "function_call",
                                    "function": func,
                                },
                            })
        
        logger.debug("Cross-reference detection", relationship_count=len(relationships))
        return relationships

    async def _detect_temporal(
        self, chunks: List[DocumentChunk]
    ) -> List[Dict]:
        """Detect temporal relationships based on creation/modification times.
        
        Args:
            chunks: Document chunks to analyze
            
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        # Group chunks by file and sort by creation time
        chunks_with_time = []
        for chunk in chunks:
            created = chunk.metadata.get("created_at")
            modified = chunk.metadata.get("modified_at")
            
            # Try to parse timestamp
            timestamp = None
            if created:
                try:
                    if isinstance(created, str):
                        timestamp = datetime.fromisoformat(created.replace('Z', '+00:00'))
                    elif isinstance(created, datetime):
                        timestamp = created
                except (ValueError, AttributeError):
                    pass
            
            if timestamp:
                chunks_with_time.append((chunk, timestamp))
        
        # Sort by timestamp
        chunks_with_time.sort(key=lambda x: x[1])
        
        # Create relationships for sequential chunks
        for i in range(len(chunks_with_time) - 1):
            chunk_a, time_a = chunks_with_time[i]
            chunk_b, time_b = chunks_with_time[i + 1]
            
            # Only relate chunks from same file or related files
            file_a = chunk_a.metadata.get("file_name", "")
            file_b = chunk_b.metadata.get("file_name", "")
            
            if file_a == file_b or file_a in file_b or file_b in file_a:
                time_diff = (time_b - time_a).total_seconds()
                # Only relate if within reasonable time window (e.g., same day)
                if time_diff < 86400:  # 24 hours
                    relationships.append({
                        "source_chunk_id": chunk_a.chunk_id,
                        "target_chunk_id": chunk_b.chunk_id,
                        "type": "temporal",
                        "confidence": 0.7,
                        "metadata": {
                            "time_difference_seconds": time_diff,
                        },
                    })
        
        logger.debug("Temporal detection", relationship_count=len(relationships))
        return relationships

    async def _detect_topical_similarity(
        self, chunks: List[DocumentChunk]
    ) -> List[Dict]:
        """Detect topical similarity using embedding cosine similarity.
        
        High embedding similarity (>0.75) → topical_similarity relationship
        
        Args:
            chunks: Document chunks to analyze
            
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        # Get embeddings for all chunks
        embeddings = {}
        for chunk in chunks:
            if chunk.embedding:
                embeddings[chunk.chunk_id] = chunk.embedding
            else:
                logger.warning("Chunk missing embedding", chunk_id=chunk.chunk_id)
        
        # Compute pairwise cosine similarity
        chunk_list = list(chunks)
        for i, chunk_a in enumerate(chunk_list):
            if chunk_a.chunk_id not in embeddings:
                continue
                
            for chunk_b in chunk_list[i + 1:]:
                if chunk_b.chunk_id not in embeddings:
                    continue
                
                similarity = self._cosine_similarity(
                    embeddings[chunk_a.chunk_id],
                    embeddings[chunk_b.chunk_id],
                )
                
                if similarity > 0.75:  # High similarity threshold
                    relationships.append({
                        "source_chunk_id": chunk_a.chunk_id,
                        "target_chunk_id": chunk_b.chunk_id,
                        "type": "topical_similarity",
                        "confidence": float(similarity),
                        "metadata": {
                            "similarity_score": float(similarity),
                        },
                    })
        
        logger.debug("Topical similarity detection", relationship_count=len(relationships))
        return relationships

    async def _detect_implementation_gaps(
        self, chunks: List[DocumentChunk]
    ) -> List[Dict]:
        """Detect implementation gaps between specs and code.
        
        Uses heuristics to identify potential gaps. For full LLM-based verification,
        this would require additional LLM calls to compare spec vs code.
        
        Args:
            chunks: Document chunks to analyze
            
        Returns:
            List of relationship dictionaries
        """
        relationships = []
        
        # Separate specs from code
        spec_chunks = []
        code_chunks = []
        
        for chunk in chunks:
            mime_type = chunk.metadata.get("mime_type", "")
            file_name = chunk.metadata.get("file_name", "").lower()
            
            # Identify spec chunks (markdown, text files with "spec", "guide", "doc" in name)
            if (mime_type.endswith("markdown") or 
                mime_type.endswith("plain") or
                any(keyword in file_name for keyword in ["spec", "guide", "doc", "readme"])):
                spec_chunks.append(chunk)
            
            # Identify code chunks
            elif mime_type in ["text/x-python", "application/javascript", "text/javascript"]:
                code_chunks.append(chunk)
        
        # For each spec-code pair, check for keyword matches
        # This is a simple heuristic - full implementation would use LLM
        for spec_chunk in spec_chunks:
            spec_content_lower = spec_chunk.content.lower()
            
            for code_chunk in code_chunks:
                # Check if spec mentions concepts that code should implement
                # Simple heuristic: check for function/class names in spec
                code_functions = re.findall(r'\b(?:def|function|class)\s+(\w+)', code_chunk.content)
                
                # Check if spec mentions any of these functions/classes
                mentions_code = any(
                    func.lower() in spec_content_lower for func in code_functions
                )
                
                if mentions_code:
                    # Potential gap: spec mentions code, but we'd need LLM to verify implementation
                    # For now, create a low-confidence relationship
                    relationships.append({
                        "source_chunk_id": spec_chunk.chunk_id,
                        "target_chunk_id": code_chunk.chunk_id,
                        "type": "implementation_gap",
                        "confidence": 0.5,  # Low confidence without LLM verification
                        "metadata": {
                            "gap_description": "Spec mentions code concepts - requires LLM verification",
                            "detection_method": "heuristic",
                        },
                    })
        
        logger.debug("Implementation gap detection", relationship_count=len(relationships))
        return relationships

    async def detect_relationships(
        self, chunks: List[DocumentChunk]
    ) -> List[Dict]:
        """Detect relationships between chunks.

        Args:
            chunks: Document chunks to analyze

        Returns:
            List of relationship dictionaries with source_chunk_id, target_chunk_id, type, confidence, metadata
        """
        try:
            logger.info("Detecting relationships", chunk_count=len(chunks))
            
            if len(chunks) < 2:
                return []

            relationships = []

            # 1. Entity overlap (fast)
            relationships.extend(await self._detect_entity_overlap(chunks))
            
            # 2. Cross-references (fast)
            relationships.extend(await self._detect_cross_references(chunks))
            
            # 3. Temporal (medium)
            relationships.extend(await self._detect_temporal(chunks))
            
            # 4. Topical similarity (medium - requires embeddings)
            relationships.extend(await self._detect_topical_similarity(chunks))
            
            # 5. Implementation gap (slow, optional - uses heuristics for now)
            relationships.extend(await self._detect_implementation_gaps(chunks))

            logger.info(
                "Relationship detection completed",
                relationship_count=len(relationships),
                chunk_count=len(chunks),
            )

            return relationships
            
        except Exception as e:
            logger.error("Relationship detection failed", error=str(e), exc_info=True)
            return []

