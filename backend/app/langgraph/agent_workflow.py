"""LangGraph multi-agent coordination for graph operations."""

from typing import Optional, Dict, Any, List
import structlog

try:
    from langgraph.graph import StateGraph, END
    from langgraph.checkpoint.memory import MemorySaver
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None
    MemorySaver = None

from app.core.exceptions import ProcessingError
from app.langgraph.graph_state import GraphAgentState, GraphStateManager
from app.langgraph.graph_agents import GraphAgents

logger = structlog.get_logger(__name__)


class AgentWorkflow:
    """LangGraph workflow for multi-agent graph coordination."""

    def __init__(
        self,
        llm: Optional[Any] = None,
        enable_checkpointing: bool = True,
    ):
        """Initialize agent workflow.

        Args:
            llm: Optional LLM for agent reasoning
            enable_checkpointing: Whether to enable checkpointing

        Raises:
            ProcessingError: If LangGraph is not available or initialization fails
        """
        if not LANGGRAPH_AVAILABLE:
            raise ProcessingError(
                "LangGraph is not installed. Install with: pip install langgraph"
            )

        self.llm = llm
        self.enable_checkpointing = enable_checkpointing
        self.state_manager = GraphStateManager()
        self.agents = GraphAgents(llm=llm)

        # Initialize checkpointing
        self.checkpointer = None
        if enable_checkpointing:
            try:
                self.checkpointer = MemorySaver()
                logger.info("Initialized LangGraph checkpointing for agents")
            except Exception as e:
                logger.warning("Failed to initialize checkpointing", error=str(e))
                self.checkpointer = None

        # Build workflow graph
        self.graph = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build LangGraph StateGraph for multi-agent coordination.

        Returns:
            Configured StateGraph instance
        """
        try:
            # Create StateGraph
            workflow = StateGraph(GraphAgentState)

            # Add agent nodes
            workflow.add_node(
                "conflict_detection",
                self.agents.conflict_detection_agent,
            )
            workflow.add_node(
                "entity_validation",
                self.agents.entity_validation_agent,
            )
            workflow.add_node(
                "relationship_scoring",
                self.agents.relationship_scoring_agent,
            )
            workflow.add_node(
                "conflict_resolution",
                self.agents.conflict_resolution_agent,
            )
            workflow.add_node(
                "graph_maintenance",
                self.agents.graph_maintenance_agent,
            )
            workflow.add_node(
                "quality_assessment",
                self.agents.quality_assessment_agent,
            )

            # Set entry point
            workflow.set_entry_point("conflict_detection")

            # Define workflow: detect conflicts → validate → score → resolve → maintain → assess
            workflow.add_edge("conflict_detection", "entity_validation")
            workflow.add_edge("entity_validation", "relationship_scoring")
            workflow.add_edge("relationship_scoring", "conflict_resolution")
            workflow.add_edge("conflict_resolution", "graph_maintenance")
            workflow.add_edge("graph_maintenance", "quality_assessment")
            workflow.add_edge("quality_assessment", END)

            # Compile workflow
            if self.checkpointer:
                compiled = workflow.compile(checkpointer=self.checkpointer)
            else:
                compiled = workflow.compile()

            logger.info("Built LangGraph agent workflow")

            return compiled

        except Exception as e:
            logger.error(
                "Failed to build agent workflow",
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(f"Failed to build workflow: {str(e)}") from e

    async def run(
        self,
        graph_data: Dict[str, Any],
        entities: List[Dict[str, Any]],
        relationships: List[Dict[str, Any]],
        folder_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> GraphAgentState:
        """Run graph enhancement workflow.

        Args:
            graph_data: Current graph data
            entities: List of entities
            relationships: List of relationships
            folder_id: Optional folder identifier
            config: Optional LangGraph configuration

        Returns:
            Final workflow state with improvements

        Raises:
            ProcessingError: If workflow execution fails
        """
        try:
            logger.info(
                "Starting graph enhancement workflow",
                entity_count=len(entities),
                relationship_count=len(relationships),
                folder_id=folder_id,
            )

            # Create initial state
            initial_state = self.state_manager.create_initial_state(
                graph_data=graph_data,
                entities=entities,
                relationships=relationships,
                folder_id=folder_id,
            )

            # Prepare config for checkpointing
            if config is None:
                config = {}
            if self.checkpointer and "configurable" not in config:
                thread_id = folder_id or "default"
                config["configurable"] = {"thread_id": thread_id}

            # Run workflow
            final_state = None
            async for state in self.graph.astream(initial_state, config=config):
                # Log progress
                current_step = state.get("workflow_step", "unknown")
                logger.debug(
                    "Agent workflow step completed",
                    step=current_step,
                    active_agents=list(state.get("active_agents", set())),
                )
                final_state = state

            if final_state is None:
                raise ProcessingError("Workflow did not produce final state")

            logger.info(
                "Graph enhancement workflow completed",
                conflicts_detected=len(final_state.get("detected_conflicts", [])),
                updates_applied=len(final_state.get("applied_updates", [])),
                overall_quality=final_state.get("quality_scores", {}).get("overall_quality", 0.0),
            )

            return final_state

        except Exception as e:
            logger.error(
                "Agent workflow execution failed",
                folder_id=folder_id,
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(f"Workflow execution failed: {str(e)}") from e

    def validate_workflow(self) -> Dict[str, Any]:
        """Validate workflow configuration.

        Returns:
            Dictionary with validation results
        """
        try:
            validation_results: Dict[str, Any] = {
                "valid": True,
                "errors": [],
                "warnings": [],
            }

            # Check if graph is compiled
            if self.graph is None:
                validation_results["valid"] = False
                validation_results["errors"].append("Workflow graph is not compiled")

            # Check agents
            if self.agents is None:
                validation_results["valid"] = False
                validation_results["errors"].append("Graph agents not initialized")

            # Check checkpointing
            if self.enable_checkpointing and self.checkpointer is None:
                validation_results["warnings"].append(
                    "Checkpointing enabled but checkpointer not initialized"
                )

            logger.debug("Agent workflow validation completed", **validation_results)

            return validation_results

        except Exception as e:
            logger.error("Agent workflow validation failed", error=str(e))
            return {
                "valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": [],
            }

