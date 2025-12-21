"""LangGraph workflow for intelligent query processing and multi-step reasoning."""

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
from app.langgraph.query_state import QueryState, QueryStateManager
from app.langgraph.query_nodes import QueryNodes
from app.langgraph.query_routing import QueryRouter
from app.monitoring.langsmith_monitoring import get_langsmith_monitor

logger = structlog.get_logger(__name__)


class QueryWorkflow:
    """LangGraph workflow for multi-step query processing and reasoning."""

    def __init__(
        self,
        llm: Optional[Any] = None,
        enable_checkpointing: bool = True,
    ):
        """Initialize query workflow.

        Args:
            llm: Optional LLM for query analysis and answer generation
            enable_checkpointing: Whether to enable checkpointing for conversations

        Raises:
            ProcessingError: If LangGraph is not available or initialization fails
        """
        if not LANGGRAPH_AVAILABLE:
            raise ProcessingError(
                "LangGraph is not installed. Install with: pip install langgraph"
            )

        self.llm = llm
        self.enable_checkpointing = enable_checkpointing
        self.state_manager = QueryStateManager()
        self.nodes = QueryNodes(llm=llm)
        self.router = QueryRouter()

        # Initialize checkpointing
        self.checkpointer = None
        if enable_checkpointing:
            try:
                self.checkpointer = MemorySaver()
                logger.info("Initialized LangGraph checkpointing for queries")
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
            workflow = StateGraph(QueryState)

            # Add nodes
            workflow.add_node("parse_query", self.nodes.parse_query_node)
            workflow.add_node("detect_intent", self.nodes.detect_intent_node)
            workflow.add_node("extract_entities", self.nodes.extract_entities_node)
            workflow.add_node("retrieve_vector", self.nodes.retrieve_vector_node)
            workflow.add_node("retrieve_keyword", self.nodes.retrieve_keyword_node)
            workflow.add_node("traverse_graph", self.nodes.traverse_graph_node)
            workflow.add_node("assemble_context", self.nodes.assemble_context_node)
            workflow.add_node("synthesize_answer", self.nodes.synthesize_answer_node)
            workflow.add_node("error_handler", self.nodes.error_handler_node)

            # Set entry point
            workflow.set_entry_point("parse_query")

            # Add conditional edges for routing
            workflow.add_conditional_edges(
                "parse_query",
                self.router.route_after_parse,
                {
                    "detect_intent": "detect_intent",
                    "error_handler": "error_handler",
                },
            )

            workflow.add_conditional_edges(
                "detect_intent",
                self.router.route_after_intent,
                {
                    "extract_entities": "extract_entities",
                    "error_handler": "error_handler",
                },
            )

            workflow.add_conditional_edges(
                "extract_entities",
                self.router.route_after_entities,
                {
                    "retrieve_vector": "retrieve_vector",
                    "traverse_graph": "traverse_graph",
                    "error_handler": "error_handler",
                },
            )

            workflow.add_conditional_edges(
                "retrieve_vector",
                self.router.route_after_vector_retrieval,
                {
                    "retrieve_keyword": "retrieve_keyword",
                    "traverse_graph": "traverse_graph",
                    "assemble_context": "assemble_context",
                    "error_handler": "error_handler",
                },
            )

            workflow.add_conditional_edges(
                "retrieve_keyword",
                self.router.route_after_keyword_retrieval,
                {
                    "assemble_context": "assemble_context",
                    "error_handler": "error_handler",
                },
            )

            workflow.add_conditional_edges(
                "traverse_graph",
                self.router.route_after_graph_traversal,
                {
                    "retrieve_vector": "retrieve_vector",
                    "assemble_context": "assemble_context",
                    "error_handler": "error_handler",
                },
            )

            workflow.add_conditional_edges(
                "assemble_context",
                self.router.route_after_context_assembly,
                {
                    "synthesize_answer": "synthesize_answer",
                    "error_handler": "error_handler",
                },
            )

            workflow.add_conditional_edges(
                "synthesize_answer",
                self.router.route_after_synthesis,
                {
                    "end": END,
                    "error_handler": "error_handler",
                },
            )

            # Error handler can retry or end
            workflow.add_conditional_edges(
                "error_handler",
                self.router.route_after_error,
                {
                    "retrieve_vector": "retrieve_vector",
                    "assemble_context": "assemble_context",
                    "synthesize_answer": "synthesize_answer",
                    "end": END,
                },
            )

            # Compile workflow (callbacks are passed in config during invocation, not compilation)
            if self.checkpointer:
                compiled = workflow.compile(checkpointer=self.checkpointer)
            else:
                compiled = workflow.compile()

            logger.info(
                "Built LangGraph query workflow",
                langsmith_enabled=self.langsmith_enabled,
            )

            return compiled

        except Exception as e:
            logger.error(
                "Failed to build query workflow",
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(f"Failed to build workflow: {str(e)}") from e

    async def run(
        self,
        query: str,
        folder_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> QueryState:
        """Run query processing workflow.

        Args:
            query: User query
            folder_id: Optional folder identifier
            conversation_id: Optional conversation identifier for context continuity
            conversation_history: Optional conversation history
            config: Optional LangGraph configuration

        Returns:
            Final workflow state with answer and citations

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
                    "conversation_id": conversation_id,
                    "query_length": len(query),
                }
            )
            
            # Merge configs (provided config takes precedence)
            final_config = {**langsmith_config, **config}
            
            # Add LangSmith callbacks to config if available
            if self.langsmith_callbacks:
                final_config["callbacks"] = self.langsmith_callbacks

            logger.info(
                "Starting query processing workflow",
                query_length=len(query),
                folder_id=folder_id,
                conversation_id=conversation_id,
                langsmith_enabled=langsmith_monitor.enabled,
            )

            # Create initial state
            initial_state = self.state_manager.create_initial_state(
                query=query,
                folder_id=folder_id,
                conversation_id=conversation_id,
                conversation_history=conversation_history,
            )

            # Prepare config for checkpointing (merge with LangSmith config)
            if self.checkpointer and "configurable" not in final_config:
                thread_id = conversation_id or folder_id or "default"
                if "configurable" not in final_config:
                    final_config["configurable"] = {}
                final_config["configurable"]["thread_id"] = thread_id

            # Run workflow with LangSmith tracing
            final_state = None
            async for state in self.graph.astream(initial_state, config=final_config):
                # Log progress
                current_step = state.get("workflow_step", "unknown")
                logger.debug(
                    "Query workflow step completed",
                    step=current_step,
                    routing_path=state.get("routing_path", []),
                )
                final_state = state

            if final_state is None:
                raise ProcessingError("Workflow did not produce final state")

            logger.info(
                "Query processing workflow completed",
                answer_length=len(final_state.get("final_answer", "")),
                citation_count=len(final_state.get("citations", [])),
                confidence=final_state.get("confidence_score", 0.0),
            )

            return final_state

        except Exception as e:
            logger.error(
                "Query workflow execution failed",
                query=query[:100],
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
                validation_results["errors"].append("Query nodes not initialized")

            # Check router
            if self.router is None:
                validation_results["valid"] = False
                validation_results["errors"].append("Query router not initialized")

            # Check checkpointing
            if self.enable_checkpointing and self.checkpointer is None:
                validation_results["warnings"].append(
                    "Checkpointing enabled but checkpointer not initialized"
                )

            logger.debug("Query workflow validation completed", **validation_results)

            return validation_results

        except Exception as e:
            logger.error("Query workflow validation failed", error=str(e))
            return {
                "valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": [],
            }

    def get_conversation_state(
        self,
        conversation_id: str,
    ) -> Optional[QueryState]:
        """Get conversation state from checkpoint.

        Args:
            conversation_id: Conversation identifier

        Returns:
            QueryState if found, None otherwise
        """
        if not self.checkpointer:
            return None

        try:
            # Get state from checkpointer
            # Note: This is a simplified version
            # Full implementation would query checkpointer for state
            logger.debug("Getting conversation state", conversation_id=conversation_id)
            return None  # Placeholder

        except Exception as e:
            logger.error("Failed to get conversation state", error=str(e))
            return None

