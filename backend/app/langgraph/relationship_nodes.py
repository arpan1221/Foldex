"""Individual LangGraph nodes for relationship detection."""

from typing import Dict, Any, List
import structlog
from datetime import datetime

try:
    from langgraph.graph import StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None

from app.langgraph.knowledge_state import KnowledgeGraphState, KnowledgeStateManager
from app.models.documents import DocumentChunk

logger = structlog.get_logger(__name__)


class RelationshipNodes:
    """Collection of LangGraph nodes for relationship detection."""

    def __init__(self, llm: Any = None):
        """Initialize relationship nodes.

        Args:
            llm: Optional LLM for entity extraction and relationship detection
        """
        self.llm = llm
        self.state_manager = KnowledgeStateManager()

    def classify_documents_node(self, state: KnowledgeGraphState) -> KnowledgeGraphState:
        """Node: Classify document types for conditional routing.

        Args:
            state: Current workflow state

        Returns:
            Updated state with document type classifications
        """
        try:
            logger.debug("Classifying document types", document_count=len(state["documents"]))

            document_types = {}
            for doc in state["documents"]:
                mime_type = doc.metadata.get("mime_type", "")
                mime_type_str = str(mime_type) if mime_type else ""
                if mime_type_str == "application/pdf":
                    doc_type = "pdf"
                elif mime_type_str.startswith("audio/"):
                    doc_type = "audio"
                elif doc.metadata.get("is_code") or mime_type_str in ["text/x-python", "text/javascript"]:
                    doc_type = "code"
                else:
                    doc_type = "text"
                
                document_types[doc.chunk_id] = doc_type

            updates = {
                "document_types": document_types,
                "workflow_step": "classified",
            }

            return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Document classification failed", error=str(e), exc_info=True)
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "classification_failed",
                    "error_message": f"Classification failed: {str(e)}",
                },
            )

    def extract_entities_node(self, state: KnowledgeGraphState) -> KnowledgeGraphState:
        """Node: Extract entities from documents.

        Args:
            state: Current workflow state

        Returns:
            Updated state with extracted entities
        """
        try:
            logger.debug("Extracting entities", document_count=len(state["documents"]))

            entities = {}
            processed = set()

            for doc in state["documents"]:
                try:
                    doc_entities = self._extract_entities_from_document(doc)
                    entities[doc.chunk_id] = doc_entities
                    processed.add(doc.chunk_id)

                    logger.debug(
                        "Extracted entities from document",
                        chunk_id=doc.chunk_id,
                        entity_count=len(doc_entities),
                    )

                except Exception as e:
                    logger.warning(
                        "Failed to extract entities from document",
                        chunk_id=doc.chunk_id,
                        error=str(e),
                    )
                    processed.add(doc.chunk_id)  # Mark as processed even if failed
                    entities[doc.chunk_id] = []  # Empty entities on failure

            updates = {
                "entities": entities,
                "processed_documents": processed,
                "workflow_step": "entities_extracted",
            }

            return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Entity extraction failed", error=str(e), exc_info=True)
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "entity_extraction_failed",
                    "error_message": f"Entity extraction failed: {str(e)}",
                },
            )

    def _extract_entities_from_document(self, doc: DocumentChunk) -> List[Dict[str, Any]]:
        """Extract entities from a document chunk.

        Args:
            doc: Document chunk

        Returns:
            List of entity dictionaries
        """
        # Simple entity extraction based on patterns
        # In production, this would use NER models or LLM
        entities = []
        content = doc.content.lower()

        # Extract potential entities (simplified)
        # Look for capitalized words, numbers, dates, etc.
        import re

        # Extract potential person names (capitalized words)
        person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
        persons = re.findall(person_pattern, doc.content)
        for person in set(persons):
            entities.append({
                "text": person,
                "type": "PERSON",
                "confidence": 0.7,
            })

        # Extract dates
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
        dates = re.findall(date_pattern, doc.content, re.IGNORECASE)
        for date in set(dates):
            entities.append({
                "text": date if isinstance(date, str) else " ".join(date),
                "type": "DATE",
                "confidence": 0.8,
            })

        # Extract organizations (words with "Inc", "Corp", "LLC", etc.)
        org_pattern = r'\b[A-Z][a-zA-Z]+ (Inc|Corp|LLC|Ltd|Company|Corporation)\b'
        orgs = re.findall(org_pattern, doc.content)
        for org in set(orgs):
            entities.append({
                "text": org if isinstance(org, str) else " ".join(org),
                "type": "ORGANIZATION",
                "confidence": 0.75,
            })

        return entities

    def detect_entity_overlap_node(self, state: KnowledgeGraphState) -> KnowledgeGraphState:
        """Node: Detect entity overlap between documents.

        Args:
            state: Current workflow state

        Returns:
            Updated state with entity overlap relationships
        """
        try:
            logger.debug("Detecting entity overlaps")

            overlaps = []
            entities = state["entities"]
            documents = state["documents"]

            # Compare entities across all document pairs
            doc_ids = [doc.chunk_id for doc in documents]
            for i, doc_id1 in enumerate(doc_ids):
                entities1 = entities.get(doc_id1, [])
                entity_texts1 = {e["text"].lower() for e in entities1}

                for doc_id2 in doc_ids[i + 1:]:
                    entities2 = entities.get(doc_id2, [])
                    entity_texts2 = {e["text"].lower() for e in entities2}

                    # Find overlapping entities
                    overlap = entity_texts1.intersection(entity_texts2)
                    if overlap:
                        overlap_count = len(overlap)
                        total_entities = len(entity_texts1) + len(entity_texts2)
                        confidence = overlap_count / max(total_entities, 1)

                        overlaps.append({
                            "source_chunk_id": doc_id1,
                            "target_chunk_id": doc_id2,
                            "type": "entity_overlap",
                            "confidence": confidence,
                            "overlapping_entities": list(overlap),
                            "overlap_count": overlap_count,
                        })

            updates = {
                "entity_overlaps": overlaps,
                "workflow_step": "entity_overlaps_detected",
            }

            return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Entity overlap detection failed", error=str(e), exc_info=True)
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "entity_overlap_failed",
                    "error_message": f"Entity overlap detection failed: {str(e)}",
                },
            )

    def detect_temporal_relationships_node(self, state: KnowledgeGraphState) -> KnowledgeGraphState:
        """Node: Detect temporal relationships between documents.

        Args:
            state: Current workflow state

        Returns:
            Updated state with temporal relationships
        """
        try:
            logger.debug("Detecting temporal relationships")

            temporal_rels = []
            documents = state["documents"]

            # Extract creation/modification times
            doc_times = {}
            for doc in documents:
                created_at = doc.metadata.get("created_at")
                
                if created_at:
                    if isinstance(created_at, str):
                        try:
                            doc_times[doc.chunk_id] = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        except Exception:
                            pass
                    elif isinstance(created_at, datetime):
                        doc_times[doc.chunk_id] = created_at

            # Compare document times
            doc_ids = list(doc_times.keys())
            for i, doc_id1 in enumerate(doc_ids):
                time1 = doc_times[doc_id1]
                for doc_id2 in doc_ids[i + 1:]:
                    time2 = doc_times[doc_id2]

                    if time1 < time2:
                        # doc1 created before doc2
                        temporal_rels.append({
                            "source_chunk_id": doc_id1,
                            "target_chunk_id": doc_id2,
                            "type": "temporal",
                            "relationship": "precedes",
                            "confidence": 0.9,
                            "time_difference": (time2 - time1).total_seconds(),
                        })
                    elif time2 < time1:
                        # doc2 created before doc1
                        temporal_rels.append({
                            "source_chunk_id": doc_id2,
                            "target_chunk_id": doc_id1,
                            "type": "temporal",
                            "relationship": "precedes",
                            "confidence": 0.9,
                            "time_difference": (time1 - time2).total_seconds(),
                        })

            updates = {
                "temporal_relationships": temporal_rels,
                "workflow_step": "temporal_relationships_detected",
            }

            return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Temporal relationship detection failed", error=str(e), exc_info=True)
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "temporal_detection_failed",
                    "error_message": f"Temporal detection failed: {str(e)}",
                },
            )

    def detect_cross_references_node(self, state: KnowledgeGraphState) -> KnowledgeGraphState:
        """Node: Detect cross-references between documents.

        Args:
            state: Current workflow state

        Returns:
            Updated state with cross-reference relationships
        """
        try:
            logger.debug("Detecting cross-references")

            cross_refs = []
            documents = state["documents"]

            # Look for references to other documents
            # Patterns: "see document X", "refer to file Y", "as mentioned in Z"
            import re

            reference_patterns = [
                r'see\s+(?:document|file|section)\s+([A-Za-z0-9_-]+)',
                r'refer(?:s|ring)?\s+to\s+(?:document|file)\s+([A-Za-z0-9_-]+)',
                r'mentioned\s+in\s+([A-Za-z0-9_-]+)',
                r'according\s+to\s+([A-Za-z0-9_-]+)',
            ]

            for doc in documents:
                doc_content = doc.content.lower()

                for pattern in reference_patterns:
                    matches = re.finditer(pattern, doc_content, re.IGNORECASE)
                    for match in matches:
                        referenced_name = match.group(1).lower()
                        
                        # Try to find matching document
                        for other_doc in documents:
                            if other_doc.chunk_id != doc.chunk_id:
                                other_file_name_raw = other_doc.metadata.get("file_name", "")
                                other_file_name = str(other_file_name_raw).lower() if other_file_name_raw else ""
                                
                                # Check if reference matches document name
                                if other_file_name and (referenced_name in other_file_name or other_file_name in referenced_name):
                                    cross_refs.append({
                                        "source_chunk_id": doc.chunk_id,
                                        "target_chunk_id": other_doc.chunk_id,
                                        "type": "cross_reference",
                                        "confidence": 0.8,
                                        "reference_text": match.group(0),
                                    })

            updates = {
                "cross_references": cross_refs,
                "workflow_step": "cross_references_detected",
            }

            return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Cross-reference detection failed", error=str(e), exc_info=True)
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "cross_reference_failed",
                    "error_message": f"Cross-reference detection failed: {str(e)}",
                },
            )

    def detect_semantic_similarity_node(self, state: KnowledgeGraphState) -> KnowledgeGraphState:
        """Node: Detect semantic similarity between documents.

        Args:
            state: Current workflow state

        Returns:
            Updated state with semantic similarity relationships
        """
        try:
            logger.debug("Detecting semantic similarities")

            similarities = []
            documents = state["documents"]

            # Use embeddings for semantic similarity
            # Note: Embeddings should be pre-computed and stored in document metadata

            # Generate embeddings for all documents
            # Note: In production, embeddings should be pre-computed and stored
            # For now, use simple text similarity as fallback
            doc_embeddings = {}
            
            # Try to get pre-computed embeddings from metadata if available
            for doc in documents:
                if "embedding" in doc.metadata and doc.metadata["embedding"]:
                    doc_embeddings[doc.chunk_id] = doc.metadata["embedding"]
                else:
                    # Use simple text-based similarity as fallback
                    # In production, this would use pre-computed embeddings
                    logger.debug("No pre-computed embedding found, using text similarity", chunk_id=doc.chunk_id)

            # Compare documents - use embeddings if available, otherwise text similarity
            doc_ids = list(doc_embeddings.keys())
            
            if doc_ids:
                # Use embeddings for similarity
                for i, doc_id1 in enumerate(doc_ids):
                    emb1 = doc_embeddings[doc_id1]
                    for doc_id2 in doc_ids[i + 1:]:
                        emb2 = doc_embeddings[doc_id2]

                        # Calculate cosine similarity
                        try:
                            import numpy as np
                            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
                            
                            if similarity > 0.7:  # Threshold for similarity
                                similarities.append({
                                    "source_chunk_id": doc_id1,
                                    "target_chunk_id": doc_id2,
                                    "type": "topical_similarity",
                                    "confidence": float(similarity),
                                    "similarity_score": float(similarity),
                                })
                        except Exception as e:
                            logger.warning("Similarity calculation failed", error=str(e))
                            continue
            else:
                # Fallback to simple text similarity
                for i, doc1 in enumerate(documents):
                    content1 = doc1.content.lower()
                    words1 = set(content1.split())
                    
                    for doc2 in documents[i + 1:]:
                        content2 = doc2.content.lower()
                        words2 = set(content2.split())
                        
                        # Jaccard similarity
                        intersection = len(words1.intersection(words2))
                        union = len(words1.union(words2))
                        similarity = intersection / union if union > 0 else 0.0
                        
                        if similarity > 0.3:  # Lower threshold for text similarity
                            similarities.append({
                                "source_chunk_id": doc1.chunk_id,
                                "target_chunk_id": doc2.chunk_id,
                                "type": "topical_similarity",
                                "confidence": float(similarity),
                                "similarity_score": float(similarity),
                            })

            updates = {
                "semantic_similarities": similarities,
                "workflow_step": "semantic_similarities_detected",
            }

            return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Semantic similarity detection failed", error=str(e), exc_info=True)
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "semantic_similarity_failed",
                    "error_message": f"Semantic similarity detection failed: {str(e)}",
                },
            )

    def detect_implementation_gaps_node(self, state: KnowledgeGraphState) -> KnowledgeGraphState:
        """Node: Detect implementation gaps (code vs specifications).

        Args:
            state: Current workflow state

        Returns:
            Updated state with implementation gap relationships
        """
        try:
            logger.debug("Detecting implementation gaps")

            gaps = []
            documents = state["documents"]
            document_types = state["document_types"]

            # Find code documents and specification documents
            code_docs = [
                doc for doc in documents
                if document_types.get(doc.chunk_id) == "code"
            ]
            spec_docs = [
                doc for doc in documents
                if document_types.get(doc.chunk_id) in ["pdf", "text"]
            ]

            # Compare code with specifications
            for code_doc in code_docs:
                code_content = code_doc.content.lower()

                for spec_doc in spec_docs:
                    spec_content = spec_doc.content.lower()

                    # Look for function/class names in code
                    import re
                    code_functions = re.findall(r'def\s+(\w+)', code_content)

                    # Check if functions/classes are mentioned in spec
                    mentioned_in_spec = []
                    for func in code_functions:
                        if func in spec_content:
                            mentioned_in_spec.append(func)

                    # If code has functions not mentioned in spec, it's a gap
                    if code_functions and len(mentioned_in_spec) < len(code_functions) * 0.5:
                        gaps.append({
                            "source_chunk_id": code_doc.chunk_id,
                            "target_chunk_id": spec_doc.chunk_id,
                            "type": "implementation_gap",
                            "confidence": 0.7,
                            "gap_type": "missing_specification",
                            "missing_functions": [f for f in code_functions if f not in mentioned_in_spec],
                        })

            updates = {
                "implementation_gaps": gaps,
                "workflow_step": "implementation_gaps_detected",
            }

            return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Implementation gap detection failed", error=str(e), exc_info=True)
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "implementation_gap_failed",
                    "error_message": f"Implementation gap detection failed: {str(e)}",
                },
            )

    def consolidate_relationships_node(self, state: KnowledgeGraphState) -> KnowledgeGraphState:
        """Node: Consolidate all detected relationships.

        Args:
            state: Current workflow state

        Returns:
            Updated state with consolidated relationships
        """
        try:
            logger.debug("Consolidating relationships")

            all_relationships = []

            # Combine all relationship types
            all_relationships.extend(state["entity_overlaps"])
            all_relationships.extend(state["temporal_relationships"])
            all_relationships.extend(state["cross_references"])
            all_relationships.extend(state["semantic_similarities"])
            all_relationships.extend(state["implementation_gaps"])

            # Deduplicate relationships
            seen = set()
            unique_relationships = []

            for rel in all_relationships:
                key = (
                    rel["source_chunk_id"],
                    rel["target_chunk_id"],
                    rel["type"],
                )
                if key not in seen:
                    seen.add(key)
                    unique_relationships.append(rel)

            # Update stats
            stats = state["stats"].copy()
            stats["relationship_count"] = len(unique_relationships)
            stats["processed_count"] = len(state["processed_documents"])

            updates = {
                "relationships": unique_relationships,
                "workflow_step": "relationships_consolidated",
                "stats": stats,
            }

            logger.info(
                "Relationships consolidated",
                total_relationships=len(unique_relationships),
            )

            return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Relationship consolidation failed", error=str(e), exc_info=True)
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "consolidation_failed",
                    "error_message": f"Consolidation failed: {str(e)}",
                },
            )

    def should_continue(self, state: KnowledgeGraphState) -> str:
        """Conditional routing: Determine next step based on document types.

        Args:
            state: Current workflow state

        Returns:
            Next node name
        """
        workflow_step = state["workflow_step"]

        # Error handling - route to error node if failed
        if "failed" in workflow_step:
            return "error_handler"

        # Continue based on workflow step
        if workflow_step == "classified":
            return "extract_entities"
        elif workflow_step == "entities_extracted":
            return "detect_entity_overlap"
        elif workflow_step == "entity_overlaps_detected":
            return "detect_temporal"
        elif workflow_step == "temporal_relationships_detected":
            return "detect_cross_references"
        elif workflow_step == "cross_references_detected":
            return "detect_semantic_similarity"
        elif workflow_step == "semantic_similarities_detected":
            return "detect_implementation_gaps"
        elif workflow_step == "implementation_gaps_detected":
            return "consolidate_relationships"
        elif workflow_step == "relationships_consolidated":
            return "end"

        return "extract_entities"  # Default

    def error_handler_node(self, state: KnowledgeGraphState) -> KnowledgeGraphState:
        """Node: Handle errors and attempt recovery.

        Args:
            state: Current workflow state

        Returns:
            Updated state with error handling
        """
        try:
            logger.warning(
                "Error handler invoked",
                workflow_step=state["workflow_step"],
                error_message=state.get("error_message"),
            )

            # Try to continue with remaining documents
            if len(state["processed_documents"]) < len(state["documents"]):
                # Continue workflow
                updates = {
                    "workflow_step": "error_recovered",
                    "error_message": None,  # Clear error to continue
                }
                return self.state_manager.update_state(state, updates)
            else:
                # All documents failed, end workflow
                updates = {
                    "workflow_step": "failed",
                }
                return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Error handler failed", error=str(e))
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "critical_failure",
                    "error_message": f"Error handler failed: {str(e)}",
                },
            )

