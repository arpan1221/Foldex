"""Shared state management for LangGraph graph agents."""

from typing import List, Dict, Any, Optional, Set, TypedDict
import structlog

try:
    from langgraph.graph import StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None

logger = structlog.get_logger(__name__)


class GraphAgentState(TypedDict):
    """State schema for LangGraph multi-agent graph operations.

    Tracks graph operations, agent decisions, conflicts, and improvements
    across multiple coordinated agents.
    """

    # Graph data
    graph_data: Dict[str, Any]  # Current graph state
    entities: List[Dict[str, Any]]  # Entities in graph
    relationships: List[Dict[str, Any]]  # Relationships in graph
    
    # Agent operations
    agent_results: Dict[str, List[Dict[str, Any]]]  # Results from each agent
    agent_decisions: Dict[str, Dict[str, Any]]  # Decisions made by agents
    
    # Conflict detection
    detected_conflicts: List[Dict[str, Any]]
    conflict_resolutions: List[Dict[str, Any]]
    
    # Validation results
    entity_validations: List[Dict[str, Any]]
    relationship_validations: List[Dict[str, Any]]
    
    # Graph improvements
    proposed_updates: List[Dict[str, Any]]
    applied_updates: List[Dict[str, Any]]
    
    # Quality metrics
    quality_scores: Dict[str, float]
    improvement_suggestions: List[Dict[str, Any]]
    
    # Agent coordination
    active_agents: Set[str]
    completed_agents: Set[str]
    agent_communication: List[Dict[str, Any]]  # Messages between agents
    
    # Workflow metadata
    workflow_step: str
    folder_id: Optional[str]
    error_message: Optional[str]
    
    # Processing statistics
    stats: Dict[str, Any]


class GraphStateManager:
    """Manages shared state for LangGraph graph agents."""

    def __init__(self):
        """Initialize state manager."""
        self.logger = structlog.get_logger(__name__)

    @staticmethod
    def create_initial_state(
        graph_data: Dict[str, Any],
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        folder_id: Optional[str] = None,
    ) -> GraphAgentState:
        """Create initial state for graph agent workflow.

        Args:
            graph_data: Current graph data
            entities: List of entities
            relationships: List of relationships
            folder_id: Optional folder identifier

        Returns:
            Initial GraphAgentState
        """
        return GraphAgentState(
            graph_data=graph_data,
            entities=entities,
            relationships=relationships,
            agent_results={},
            agent_decisions={},
            detected_conflicts=[],
            conflict_resolutions=[],
            entity_validations=[],
            relationship_validations=[],
            proposed_updates=[],
            applied_updates=[],
            quality_scores={},
            improvement_suggestions=[],
            active_agents=set(),
            completed_agents=set(),
            agent_communication=[],
            workflow_step="initialized",
            folder_id=folder_id,
            error_message=None,
            stats={
                "total_entities": len(entities),
                "total_relationships": len(relationships),
                "conflicts_detected": 0,
                "updates_applied": 0,
            },
        )

    @staticmethod
    def update_state(
        state: GraphAgentState,
        updates: Dict[str, Any],
    ) -> GraphAgentState:
        """Update state with new values.

        Args:
            state: Current state
            updates: Dictionary of updates to apply

        Returns:
            Updated state
        """
        # Create new state dict with updates
        new_state: Dict[str, Any] = {
            "graph_data": state["graph_data"].copy(),
            "entities": state["entities"].copy(),
            "relationships": state["relationships"].copy(),
            "agent_results": state["agent_results"].copy(),
            "agent_decisions": state["agent_decisions"].copy(),
            "detected_conflicts": state["detected_conflicts"].copy(),
            "conflict_resolutions": state["conflict_resolutions"].copy(),
            "entity_validations": state["entity_validations"].copy(),
            "relationship_validations": state["relationship_validations"].copy(),
            "proposed_updates": state["proposed_updates"].copy(),
            "applied_updates": state["applied_updates"].copy(),
            "quality_scores": state["quality_scores"].copy(),
            "improvement_suggestions": state["improvement_suggestions"].copy(),
            "active_agents": state["active_agents"].copy(),
            "completed_agents": state["completed_agents"].copy(),
            "agent_communication": state["agent_communication"].copy(),
            "workflow_step": state["workflow_step"],
            "folder_id": state["folder_id"],
            "error_message": state["error_message"],
            "stats": state["stats"].copy(),
        }

        # Apply updates
        for key, value in updates.items():
            if key in ["entities", "relationships", "detected_conflicts", "conflict_resolutions",
                      "entity_validations", "relationship_validations", "proposed_updates",
                      "applied_updates", "improvement_suggestions", "agent_communication"]:
                if isinstance(value, list):
                    new_state[key] = state[key] + value
                else:
                    new_state[key] = value
            elif key in ["active_agents", "completed_agents"]:
                if isinstance(value, set):
                    new_state[key] = state[key].union(value)
                else:
                    new_state[key] = state[key].union(set(value))
            elif key in ["agent_results", "agent_decisions", "quality_scores", "graph_data", "stats"]:
                if isinstance(value, dict):
                    new_state[key] = {**state[key], **value}
                else:
                    new_state[key] = value
            else:
                new_state[key] = value

        # Build TypedDict properly
        result: GraphAgentState = {
            "graph_data": new_state["graph_data"],
            "entities": new_state["entities"],
            "relationships": new_state["relationships"],
            "agent_results": new_state["agent_results"],
            "agent_decisions": new_state["agent_decisions"],
            "detected_conflicts": new_state["detected_conflicts"],
            "conflict_resolutions": new_state["conflict_resolutions"],
            "entity_validations": new_state["entity_validations"],
            "relationship_validations": new_state["relationship_validations"],
            "proposed_updates": new_state["proposed_updates"],
            "applied_updates": new_state["applied_updates"],
            "quality_scores": new_state["quality_scores"],
            "improvement_suggestions": new_state["improvement_suggestions"],
            "active_agents": new_state["active_agents"],
            "completed_agents": new_state["completed_agents"],
            "agent_communication": new_state["agent_communication"],
            "workflow_step": new_state["workflow_step"],
            "folder_id": new_state["folder_id"],
            "error_message": new_state["error_message"],
            "stats": new_state["stats"],
        }
        return result

    @staticmethod
    def get_state_summary(state: GraphAgentState) -> Dict[str, Any]:
        """Get summary of current state.

        Args:
            state: Current state

        Returns:
            Dictionary with state summary
        """
        return {
            "workflow_step": state["workflow_step"],
            "active_agents": list(state["active_agents"]),
            "completed_agents": list(state["completed_agents"]),
            "conflicts_detected": len(state["detected_conflicts"]),
            "updates_applied": len(state["applied_updates"]),
            "quality_scores": state["quality_scores"],
            "folder_id": state["folder_id"],
            "error_message": state["error_message"],
        }

    @staticmethod
    def add_agent_message(
        state: GraphAgentState,
        from_agent: str,
        to_agent: Optional[str],
        message: str,
        data: Optional[Dict[str, Any]] = None,
    ) -> GraphAgentState:
        """Add communication message between agents.

        Args:
            state: Current state
            from_agent: Source agent name
            to_agent: Target agent name (None for broadcast)
            message: Message content
            data: Optional message data

        Returns:
            Updated state with message
        """
        communication = {
            "from_agent": from_agent,
            "to_agent": to_agent,
            "message": message,
            "data": data or {},
            "timestamp": structlog.get_logger().info("timestamp"),  # Placeholder
        }

        return GraphStateManager.update_state(
            state,
            {"agent_communication": [communication]},
        )

