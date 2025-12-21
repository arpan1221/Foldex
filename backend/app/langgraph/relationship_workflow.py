"""LangGraph workflow for relationship detection and knowledge graph construction."""

from typing import Optional, Dict, Any
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
from app.langgraph.knowledge_state import KnowledgeGraphState, KnowledgeStateManager
from app.langgraph.relationship_nodes import RelationshipNodes
from app.monitoring.langsmith_monitoring import get_langsmith_monitor

logger = structlog.get_logger(__name__)


class RelationshipWorkflow:
    """LangGraph workflow for multi-step relationship detection."""

    def __init__(
        self,
        llm: Optional[Any] = None,
        enable_checkpointing: bool = True,
    ):
        """Initialize relationship workflow.

        Args:
            llm: Optional LLM for entity extraction
            enable_checkpointing: Whether to enable checkpointing for resumable workflows

        Raises:
            ProcessingError: If LangGraph is not available or initialization fails
        """
        if not LANGGRAPH_AVAILABLE:
            raise ProcessingError(
                "LangGraph is not installed. Install with: pip install langgraph"
            )

        self.llm = llm
        self.enable_checkpointing = enable_checkpointing
        self.state_manager = KnowledgeStateManager()
        self.nodes = RelationshipNodes(llm=llm)

        # Initialize checkpointing
        self.checkpointer = None
        if enable_checkpointing:
            try:
                self.checkpointer = MemorySaver()
                logger.info("Initialized LangGraph checkpointing")
            except Exception as e:
                logger.warning("Failed to initialize checkpointing", error=str(e))
                self.checkpointer = None

        # Get LangSmith callbacks for observability (stored for use in run method)
        langsmith_monitor = get_langsmith_monitor()
        self.langsmith_callbacks = langsmith_monitor.get_callbacks()
        self.langsmith_enabled = langsmith_monitor.enabled

        # Build workflow graph
        self.graph = self._build_workflow()

    def _build_workflow(self) -> StateGraph:
        """Build LangGraph StateGraph workflow.

        Returns:
            Configured StateGraph instance
        """
        try:
            # Create StateGraph
            workflow = StateGraph(KnowledgeGraphState)

            # Add nodes
            workflow.add_node("classify_documents", self.nodes.classify_documents_node)
            workflow.add_node("extract_entities", self.nodes.extract_entities_node)
            workflow.add_node("detect_entity_overlap", self.nodes.detect_entity_overlap_node)
            workflow.add_node("detect_temporal", self.nodes.detect_temporal_relationships_node)
            workflow.add_node("detect_cross_references", self.nodes.detect_cross_references_node)
            workflow.add_node("detect_semantic_similarity", self.nodes.detect_semantic_similarity_node)
            workflow.add_node("detect_implementation_gaps", self.nodes.detect_implementation_gaps_node)
            workflow.add_node("consolidate_relationships", self.nodes.consolidate_relationships_node)
            workflow.add_node("error_handler", self.nodes.error_handler_node)

            # Set entry point
            workflow.set_entry_point("classify_documents")

            # Add conditional edges for routing
            workflow.add_conditional_edges(
                "classify_documents",
                self.nodes.should_continue,
                {
                    "extract_entities": "extract_entities",
                    "error_handler": "error_handler",
                },
            )

            workflow.add_conditional_edges(
                "extract_entities",
                self.nodes.should_continue,
                {
                    "detect_entity_overlap": "detect_entity_overlap",
                    "error_handler": "error_handler",
                },
            )

            workflow.add_conditional_edges(
                "detect_entity_overlap",
                self.nodes.should_continue,
                {
                    "detect_temporal": "detect_temporal",
                    "error_handler": "error_handler",
                },
            )

            workflow.add_conditional_edges(
                "detect_temporal",
                self.nodes.should_continue,
                {
                    "detect_cross_references": "detect_cross_references",
                    "error_handler": "error_handler",
                },
            )

            workflow.add_conditional_edges(
                "detect_cross_references",
                self.nodes.should_continue,
                {
                    "detect_semantic_similarity": "detect_semantic_similarity",
                    "error_handler": "error_handler",
                },
            )

            workflow.add_conditional_edges(
                "detect_semantic_similarity",
                self.nodes.should_continue,
                {
                    "detect_implementation_gaps": "detect_implementation_gaps",
                    "error_handler": "error_handler",
                },
            )

            workflow.add_conditional_edges(
                "detect_implementation_gaps",
                self.nodes.should_continue,
                {
                    "consolidate_relationships": "consolidate_relationships",
                    "error_handler": "error_handler",
                },
            )

            workflow.add_conditional_edges(
                "consolidate_relationships",
                self.nodes.should_continue,
                {
                    "end": END,
                    "error_handler": "error_handler",
                },
            )

            # Error handler can retry or end
            workflow.add_conditional_edges(
                "error_handler",
                self._error_handler_route,
                {
                    "extract_entities": "extract_entities",  # Retry from entity extraction
                    "end": END,  # End if critical failure
                },
            )

            # Compile workflow (callbacks are passed in config during invocation, not compilation)
            if self.checkpointer:
                compiled = workflow.compile(checkpointer=self.checkpointer)
            else:
                compiled = workflow.compile()

            logger.info(
                "Built LangGraph relationship workflow",
                langsmith_enabled=self.langsmith_enabled,
            )

            return compiled

        except Exception as e:
            logger.error(
                "Failed to build workflow",
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(f"Failed to build workflow: {str(e)}") from e

    def _error_handler_route(self, state: KnowledgeGraphState) -> str:
        """Route after error handling.

        Args:
            state: Current workflow state

        Returns:
            Next node name
        """
        workflow_step = state["workflow_step"]

        if workflow_step == "error_recovered":
            # Try to continue from entity extraction
            return "extract_entities"
        elif workflow_step in ["failed", "critical_failure"]:
            return "end"
        else:
            # Default: try to continue
            return "extract_entities"

    async def run(
        self,
        documents: list,
        folder_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> KnowledgeGraphState:
        """Run relationship detection workflow.

        Args:
            documents: List of DocumentChunk objects to process
            folder_id: Optional folder identifier
            config: Optional LangGraph configuration

        Returns:
            Final workflow state with detected relationships

        Raises:
            ProcessingError: If workflow execution fails
        """
        try:
            # Get LangSmith monitor and create config with tracing
            langsmith_monitor = get_langsmith_monitor()
            
            # Merge LangSmith config with provided config
            if config is None:
                config = {}
            
            langsmith_config = langsmith_monitor.get_langgraph_config(
                metadata={
                    "folder_id": folder_id,
                    "document_count": len(documents),
                }
            )
            
            # Merge configs (provided config takes precedence)
            final_config = {**langsmith_config, **config}
            
            # Add LangSmith callbacks to config if available
            if self.langsmith_callbacks:
                final_config["callbacks"] = self.langsmith_callbacks

            logger.info(
                "Starting relationship detection workflow",
                document_count=len(documents),
                folder_id=folder_id,
                langsmith_enabled=langsmith_monitor.enabled,
            )

            # Create initial state
            initial_state = self.state_manager.create_initial_state(
                documents=documents,
                folder_id=folder_id,
            )

            # Prepare config for checkpointing (merge with LangSmith config)
            if self.checkpointer and "configurable" not in final_config:
                final_config["configurable"] = {"thread_id": folder_id or "default"}

            # Run workflow with LangSmith tracing
            final_state = None
            async for state in self.graph.astream(initial_state, config=final_config):
                # Log progress
                current_step = state.get("workflow_step", "unknown")
                logger.debug(
                    "Workflow step completed",
                    step=current_step,
                    processed_count=len(state.get("processed_documents", set())),
                )
                final_state = state

            if final_state is None:
                raise ProcessingError("Workflow did not produce final state")

            logger.info(
                "Relationship detection workflow completed",
                relationship_count=len(final_state.get("relationships", [])),
                folder_id=folder_id,
            )

            return final_state

        except Exception as e:
            logger.error(
                "Workflow execution failed",
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

            # Check nodes
            if self.nodes is None:
                validation_results["valid"] = False
                validation_results["errors"].append("Relationship nodes not initialized")

            # Check checkpointing
            if self.enable_checkpointing and self.checkpointer is None:
                validation_results["warnings"].append(
                    "Checkpointing enabled but checkpointer not initialized"
                )

            logger.debug("Workflow validation completed", **validation_results)

            return validation_results

        except Exception as e:
            logger.error("Workflow validation failed", error=str(e))
            return {
                "valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": [],
            }

