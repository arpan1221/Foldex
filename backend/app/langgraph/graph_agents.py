"""Specialized LangGraph agents for different graph operations."""

from typing import Any, Dict, List
import structlog

try:
    from langgraph.graph import StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None

from app.langgraph.graph_state import GraphAgentState, GraphStateManager

logger = structlog.get_logger(__name__)


class GraphAgents:
    """Collection of specialized agents for graph operations."""

    def __init__(self, llm: Any = None):
        """Initialize graph agents.

        Args:
            llm: Optional LLM for agent reasoning
        """
        self.llm = llm
        self.state_manager = GraphStateManager()

    def conflict_detection_agent(self, state: GraphAgentState) -> GraphAgentState:
        """Agent: Detect conflicts in knowledge graph.

        Args:
            state: Current workflow state

        Returns:
            Updated state with detected conflicts
        """
        try:
            logger.debug("Conflict detection agent running")

            conflicts = []
            entities = state["entities"]
            relationships = state["relationships"]

            # Detect duplicate entities
            entity_names: Dict[str, Any] = {}
            for entity in entities:
                entity_text = entity.get("text", "").lower()
                if entity_text in entity_names:
                    conflicts.append({
                        "type": "duplicate_entity",
                        "entity1": entity_names[entity_text],
                        "entity2": entity,
                        "confidence": 0.9,
                    })
                else:
                    entity_names[entity_text] = entity

            # Detect conflicting relationships
            relationship_map: Dict[tuple, Any] = {}
            for rel in relationships:
                key = (rel.get("source"), rel.get("target"), rel.get("type"))
                if key in relationship_map:
                    # Check for conflicting relationship types
                    existing = relationship_map[key]
                    if existing.get("type") != rel.get("type"):
                        conflicts.append({
                            "type": "conflicting_relationship",
                            "relationship1": existing,
                            "relationship2": rel,
                            "confidence": 0.8,
                        })
                else:
                    relationship_map[key] = rel

            # Detect circular relationships
            for rel in relationships:
                source = rel.get("source")
                target = rel.get("target")
                # Check if reverse relationship exists
                reverse_key = (target, source, rel.get("type"))
                if reverse_key in relationship_map:
                    conflicts.append({
                        "type": "circular_relationship",
                        "relationship": rel,
                        "confidence": 0.7,
                    })

            updates = {
                "detected_conflicts": conflicts,
                "active_agents": {"conflict_detection"},
                "workflow_step": "conflicts_detected",
                "stats": {"conflicts_detected": len(conflicts)},
            }

            logger.info(
                "Conflict detection completed",
                conflict_count=len(conflicts),
            )

            return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Conflict detection agent failed", error=str(e), exc_info=True)
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "conflict_detection_failed",
                    "error_message": f"Conflict detection failed: {str(e)}",
                },
            )

    def entity_validation_agent(self, state: GraphAgentState) -> GraphAgentState:
        """Agent: Validate entities in knowledge graph.

        Args:
            state: Current workflow state

        Returns:
            Updated state with validation results
        """
        try:
            logger.debug("Entity validation agent running")

            validations: List[Dict[str, Any]] = []
            entities = state["entities"]

            for entity in entities:
                issues_list: List[str] = []
                validation: Dict[str, Any] = {
                    "entity": entity,
                    "is_valid": True,
                    "issues": issues_list,
                    "confidence": 1.0,
                }

                # Validate entity text
                entity_text = entity.get("text", "")
                if not entity_text or len(entity_text) < 2:
                    validation["is_valid"] = False
                    issues_list.append("Entity text too short")
                    validation["confidence"] = 0.0

                # Validate entity type
                entity_type = entity.get("type", "")
                if not entity_type:
                    validation["is_valid"] = False
                    issues_list.append("Missing entity type")
                    validation["confidence"] = 0.5

                # Validate confidence score
                confidence = entity.get("confidence", 0.0)
                if confidence < 0.5:
                    issues_list.append("Low confidence score")
                    validation["confidence"] = confidence

                validations.append(validation)

            updates = {
                "entity_validations": validations,
                "active_agents": {"entity_validation"},
                "workflow_step": "entities_validated",
            }

            logger.info(
                "Entity validation completed",
                validation_count=len(validations),
            )

            return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Entity validation agent failed", error=str(e), exc_info=True)
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "entity_validation_failed",
                    "error_message": f"Entity validation failed: {str(e)}",
                },
            )

    def relationship_scoring_agent(self, state: GraphAgentState) -> GraphAgentState:
        """Agent: Score and validate relationships.

        Args:
            state: Current workflow state

        Returns:
            Updated state with relationship scores
        """
        try:
            logger.debug("Relationship scoring agent running")

            validations: List[Dict[str, Any]] = []
            relationships = state["relationships"]

            for rel in relationships:
                issues_list: List[str] = []
                score = 0.5  # Default score
                validation: Dict[str, Any] = {
                    "relationship": rel,
                    "score": score,
                    "is_valid": True,
                    "issues": issues_list,
                }

                # Score based on confidence
                confidence = float(rel.get("confidence", 0.0))
                score = confidence

                # Score based on relationship type
                rel_type = rel.get("type", "")
                if rel_type in ["entity_overlap", "cross_reference"]:
                    score = score + 0.2
                elif rel_type == "temporal":
                    score = score + 0.1

                # Validate source and target exist
                source = rel.get("source_chunk_id") or rel.get("source")
                target = rel.get("target_chunk_id") or rel.get("target")
                if not source or not target:
                    validation["is_valid"] = False
                    issues_list.append("Missing source or target")
                    score = 0.0

                # Normalize score
                score = min(score, 1.0)
                validation["score"] = score

                validations.append(validation)

            updates = {
                "relationship_validations": validations,
                "active_agents": {"relationship_scoring"},
                "workflow_step": "relationships_scored",
            }

            logger.info(
                "Relationship scoring completed",
                validation_count=len(validations),
            )

            return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Relationship scoring agent failed", error=str(e), exc_info=True)
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "relationship_scoring_failed",
                    "error_message": f"Relationship scoring failed: {str(e)}",
                },
            )

    def conflict_resolution_agent(self, state: GraphAgentState) -> GraphAgentState:
        """Agent: Resolve detected conflicts.

        Args:
            state: Current workflow state

        Returns:
            Updated state with conflict resolutions
        """
        try:
            logger.debug("Conflict resolution agent running")

            resolutions = []
            conflicts = state["detected_conflicts"]

            for conflict in conflicts:
                resolution = {
                    "conflict": conflict,
                    "resolution": "keep_both",  # Default resolution
                    "action": "no_action",
                    "confidence": 0.5,
                }

                conflict_type = conflict.get("type")

                if conflict_type == "duplicate_entity":
                    # Merge duplicate entities
                    resolution["resolution"] = "merge"
                    resolution["action"] = "merge_entities"
                    resolution["confidence"] = 0.8

                elif conflict_type == "conflicting_relationship":
                    # Keep relationship with higher confidence
                    rel1 = conflict.get("relationship1", {})
                    rel2 = conflict.get("relationship2", {})
                    if rel1.get("confidence", 0.0) > rel2.get("confidence", 0.0):
                        resolution["resolution"] = "keep_first"
                        resolution["action"] = "remove_second"
                    else:
                        resolution["resolution"] = "keep_second"
                        resolution["action"] = "remove_first"
                    resolution["confidence"] = 0.7

                elif conflict_type == "circular_relationship":
                    # Keep one direction
                    resolution["resolution"] = "keep_one_direction"
                    resolution["action"] = "remove_reverse"
                    resolution["confidence"] = 0.6

                resolutions.append(resolution)

            updates = {
                "conflict_resolutions": resolutions,
                "active_agents": {"conflict_resolution"},
                "workflow_step": "conflicts_resolved",
            }

            logger.info(
                "Conflict resolution completed",
                resolution_count=len(resolutions),
            )

            return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Conflict resolution agent failed", error=str(e), exc_info=True)
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "conflict_resolution_failed",
                    "error_message": f"Conflict resolution failed: {str(e)}",
                },
            )

    def graph_maintenance_agent(self, state: GraphAgentState) -> GraphAgentState:
        """Agent: Perform graph maintenance operations.

        Args:
            state: Current workflow state

        Returns:
            Updated state with maintenance updates
        """
        try:
            logger.debug("Graph maintenance agent running")

            updates = []

            # Remove invalid entities
            entity_validations = state["entity_validations"]
            for validation in entity_validations:
                if not validation.get("is_valid", True):
                    entity = validation.get("entity")
                    if entity:
                        updates.append({
                            "type": "remove_entity",
                            "entity": entity,
                            "reason": "Invalid entity",
                        })

            # Remove low-scoring relationships
            relationship_validations = state["relationship_validations"]
            for validation in relationship_validations:
                if validation.get("score", 0.0) < 0.3:
                    rel = validation.get("relationship")
                    if rel:
                        updates.append({
                            "type": "remove_relationship",
                            "relationship": rel,
                            "reason": "Low score",
                        })

            # Apply conflict resolutions
            conflict_resolutions = state["conflict_resolutions"]
            for resolution in conflict_resolutions:
                action = resolution.get("action")
                if action == "merge_entities":
                    updates.append({
                        "type": "merge_entities",
                        "conflict": resolution.get("conflict"),
                        "reason": "Resolve duplicate",
                    })
                elif action in ["remove_first", "remove_second"]:
                    updates.append({
                        "type": "remove_relationship",
                        "conflict": resolution.get("conflict"),
                        "reason": "Resolve conflict",
                    })

            # Apply updates
            applied_updates = []
            for update in updates:
                # In production, this would actually modify the graph
                applied_updates.append({
                    **update,
                    "status": "applied",
                })

            updates_dict = {
                "proposed_updates": updates,
                "applied_updates": applied_updates,
                "active_agents": {"graph_maintenance"},
                "workflow_step": "maintenance_completed",
                "stats": {"updates_applied": len(applied_updates)},
            }

            logger.info(
                "Graph maintenance completed",
                update_count=len(applied_updates),
            )

            return self.state_manager.update_state(state, updates_dict)

        except Exception as e:
            logger.error("Graph maintenance agent failed", error=str(e), exc_info=True)
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "maintenance_failed",
                    "error_message": f"Graph maintenance failed: {str(e)}",
                },
            )

    def quality_assessment_agent(self, state: GraphAgentState) -> GraphAgentState:
        """Agent: Assess graph quality and suggest improvements.

        Args:
            state: Current workflow state

        Returns:
            Updated state with quality scores and suggestions
        """
        try:
            logger.debug("Quality assessment agent running")

            entities = state["entities"]
            relationships = state["relationships"]

            # Calculate quality scores
            quality_scores = {}

            # Entity quality
            entity_count = len(entities)
            valid_entities = sum(
                1 for v in state["entity_validations"]
                if v.get("is_valid", True)
            )
            quality_scores["entity_quality"] = (
                valid_entities / entity_count if entity_count > 0 else 0.0
            )

            # Relationship quality
            relationship_count = len(relationships)
            avg_relationship_score = sum(
                v.get("score", 0.0) for v in state["relationship_validations"]
            ) / relationship_count if relationship_count > 0 else 0.0
            quality_scores["relationship_quality"] = avg_relationship_score

            # Conflict rate
            conflict_count = len(state["detected_conflicts"])
            quality_scores["conflict_rate"] = (
                conflict_count / max(relationship_count, 1)
            )

            # Overall quality
            quality_scores["overall_quality"] = (
                quality_scores["entity_quality"] * 0.4 +
                quality_scores["relationship_quality"] * 0.4 +
                (1.0 - min(quality_scores["conflict_rate"], 1.0)) * 0.2
            )

            # Generate improvement suggestions
            suggestions = []

            if quality_scores["entity_quality"] < 0.7:
                suggestions.append({
                    "type": "improve_entities",
                    "priority": "high",
                    "description": "Entity quality is below threshold",
                })

            if quality_scores["relationship_quality"] < 0.7:
                suggestions.append({
                    "type": "improve_relationships",
                    "priority": "high",
                    "description": "Relationship quality is below threshold",
                })

            if quality_scores["conflict_rate"] > 0.1:
                suggestions.append({
                    "type": "resolve_conflicts",
                    "priority": "medium",
                    "description": "High conflict rate detected",
                })

            updates = {
                "quality_scores": quality_scores,
                "improvement_suggestions": suggestions,
                "active_agents": {"quality_assessment"},
                "workflow_step": "quality_assessed",
            }

            logger.info(
                "Quality assessment completed",
                overall_quality=quality_scores.get("overall_quality", 0.0),
            )

            return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Quality assessment agent failed", error=str(e), exc_info=True)
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "quality_assessment_failed",
                    "error_message": f"Quality assessment failed: {str(e)}",
                },
            )

