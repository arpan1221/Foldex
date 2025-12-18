"""LangGraph state management for query analysis and routing."""

from typing import List, Dict, Any, Optional, Set, TypedDict
import structlog

from app.models.documents import DocumentChunk

try:
    from langgraph.graph import StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None

logger = structlog.get_logger(__name__)


class QueryState(TypedDict):
    """State schema for LangGraph query processing workflow.

    Tracks query, analysis results, retrieved context, and response
    across multiple workflow steps.
    """

    # Input query
    query: str
    original_query: str
    
    # Query analysis
    query_intent: Optional[str]  # factual, relational, temporal, synthesis, comparison
    query_type: Optional[str]  # More specific classification
    entities: List[Dict[str, Any]]  # Extracted entities from query
    keywords: List[str]  # Extracted keywords
    
    # Context and retrieval
    retrieved_chunks: List[DocumentChunk]
    graph_traversal_results: List[Dict[str, Any]]
    context_documents: List[Dict[str, Any]]
    
    # Multi-step reasoning
    reasoning_steps: List[Dict[str, Any]]
    intermediate_answers: List[str]
    
    # Response generation
    final_answer: Optional[str]
    citations: List[Dict[str, Any]]
    confidence_score: float
    
    # Conversation state
    conversation_id: Optional[str]
    conversation_history: List[Dict[str, str]]  # Previous Q&A pairs
    follow_up_context: Optional[str]
    
    # Workflow metadata
    folder_id: Optional[str]
    workflow_step: str  # Current step in workflow
    error_message: Optional[str]
    
    # Routing decisions
    selected_retrieval_strategies: List[str]  # vector, keyword, graph, hybrid
    routing_path: List[str]  # Path taken through workflow
    
    # Processing statistics
    stats: Dict[str, Any]


class QueryStateManager:
    """Manages state for LangGraph query processing workflow."""

    def __init__(self):
        """Initialize state manager."""
        self.logger = structlog.get_logger(__name__)

    @staticmethod
    def create_initial_state(
        query: str,
        folder_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> QueryState:
        """Create initial state for query workflow.

        Args:
            query: User query
            folder_id: Optional folder identifier
            conversation_id: Optional conversation identifier
            conversation_history: Optional conversation history

        Returns:
            Initial QueryState
        """
        return QueryState(
            query=query,
            original_query=query,
            query_intent=None,
            query_type=None,
            entities=[],
            keywords=[],
            retrieved_chunks=[],
            graph_traversal_results=[],
            context_documents=[],
            reasoning_steps=[],
            intermediate_answers=[],
            final_answer=None,
            citations=[],
            confidence_score=0.0,
            conversation_id=conversation_id,
            conversation_history=conversation_history or [],
            follow_up_context=None,
            folder_id=folder_id,
            workflow_step="initialized",
            error_message=None,
            selected_retrieval_strategies=[],
            routing_path=["initialized"],
            stats={
                "retrieval_count": 0,
                "reasoning_steps_count": 0,
                "processing_time": 0.0,
            },
        )

    @staticmethod
    def update_state(
        state: QueryState,
        updates: Dict[str, Any],
    ) -> QueryState:
        """Update state with new values.

        Args:
            state: Current state
            updates: Dictionary of updates to apply

        Returns:
            Updated state
        """
        # Create new state dict with updates
        new_state: Dict[str, Any] = {
            "query": state["query"],
            "original_query": state["original_query"],
            "query_intent": state["query_intent"],
            "query_type": state["query_type"],
            "entities": state["entities"].copy(),
            "keywords": state["keywords"].copy(),
            "retrieved_chunks": state["retrieved_chunks"].copy(),
            "graph_traversal_results": state["graph_traversal_results"].copy(),
            "context_documents": state["context_documents"].copy(),
            "reasoning_steps": state["reasoning_steps"].copy(),
            "intermediate_answers": state["intermediate_answers"].copy(),
            "final_answer": state["final_answer"],
            "citations": state["citations"].copy(),
            "confidence_score": state["confidence_score"],
            "conversation_id": state["conversation_id"],
            "conversation_history": state["conversation_history"].copy(),
            "follow_up_context": state["follow_up_context"],
            "folder_id": state["folder_id"],
            "workflow_step": state["workflow_step"],
            "error_message": state["error_message"],
            "selected_retrieval_strategies": state["selected_retrieval_strategies"].copy(),
            "routing_path": state["routing_path"].copy(),
            "stats": state["stats"].copy(),
        }

        # Apply updates
        for key, value in updates.items():
            if key in ["entities", "keywords", "retrieved_chunks", "graph_traversal_results",
                      "context_documents", "reasoning_steps", "intermediate_answers",
                      "citations", "conversation_history", "selected_retrieval_strategies",
                      "routing_path"]:
                if isinstance(value, list):
                    # Append to existing list
                    new_state[key] = state[key] + value
                else:
                    new_state[key] = value
            elif key == "routing_path" and isinstance(value, str):
                # Append step to routing path
                new_state[key] = state[key] + [value]
            elif key == "stats" and isinstance(value, dict):
                # Merge stats
                new_state[key] = {**state[key], **value}
            else:
                new_state[key] = value

        # Build TypedDict properly
        result: QueryState = {
            "query": new_state["query"],
            "original_query": new_state["original_query"],
            "query_intent": new_state["query_intent"],
            "query_type": new_state["query_type"],
            "entities": new_state["entities"],
            "keywords": new_state["keywords"],
            "retrieved_chunks": new_state["retrieved_chunks"],
            "graph_traversal_results": new_state["graph_traversal_results"],
            "context_documents": new_state["context_documents"],
            "reasoning_steps": new_state["reasoning_steps"],
            "intermediate_answers": new_state["intermediate_answers"],
            "final_answer": new_state["final_answer"],
            "citations": new_state["citations"],
            "confidence_score": new_state["confidence_score"],
            "conversation_id": new_state["conversation_id"],
            "conversation_history": new_state["conversation_history"],
            "follow_up_context": new_state["follow_up_context"],
            "folder_id": new_state["folder_id"],
            "workflow_step": new_state["workflow_step"],
            "error_message": new_state["error_message"],
            "selected_retrieval_strategies": new_state["selected_retrieval_strategies"],
            "routing_path": new_state["routing_path"],
            "stats": new_state["stats"],
        }
        return result

    @staticmethod
    def get_state_summary(state: QueryState) -> Dict[str, Any]:
        """Get summary of current state.

        Args:
            state: Current state

        Returns:
            Dictionary with state summary
        """
        return {
            "workflow_step": state["workflow_step"],
            "query_intent": state["query_intent"],
            "query_type": state["query_type"],
            "entity_count": len(state["entities"]),
            "retrieved_chunk_count": len(state["retrieved_chunks"]),
            "reasoning_steps_count": len(state["reasoning_steps"]),
            "confidence_score": state["confidence_score"],
            "routing_path": state["routing_path"],
            "folder_id": state["folder_id"],
            "error_message": state["error_message"],
        }

