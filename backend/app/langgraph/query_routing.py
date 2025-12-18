"""Conditional routing logic for different query types."""

from typing import List
import structlog

from app.langgraph.query_state import QueryState

logger = structlog.get_logger(__name__)


class QueryRouter:
    """Handles conditional routing for query workflow."""

    @staticmethod
    def route_after_parse(state: QueryState) -> str:
        """Route after query parsing.

        Args:
            state: Current workflow state

        Returns:
            Next node name
        """
        workflow_step = state["workflow_step"]

        if "failed" in workflow_step:
            return "error_handler"

        return "detect_intent"

    @staticmethod
    def route_after_intent(state: QueryState) -> str:
        """Route after intent detection.

        Args:
            state: Current workflow state

        Returns:
            Next node name
        """
        workflow_step = state["workflow_step"]

        if "failed" in workflow_step:
            return "error_handler"

        return "extract_entities"

    @staticmethod
    def route_after_entities(state: QueryState) -> str:
        """Route after entity extraction.

        Args:
            state: Current workflow state

        Returns:
            Next node name
        """
        workflow_step = state["workflow_step"]

        if "failed" in workflow_step:
            return "error_handler"

        # Route to appropriate retrieval strategy based on intent
        query_intent = state["query_intent"]

        if query_intent == "relational":
            # Relational queries benefit from graph traversal
            return "traverse_graph"
        elif query_intent in ["factual", "synthesis"]:
            # Factual and synthesis queries use vector search
            return "retrieve_vector"
        elif query_intent == "comparison":
            # Comparison queries need multiple retrieval strategies
            return "retrieve_vector"  # Start with vector, then keyword
        elif query_intent == "temporal":
            # Temporal queries may need both vector and graph
            return "retrieve_vector"
        else:
            # Default to vector retrieval
            return "retrieve_vector"

    @staticmethod
    def route_after_vector_retrieval(state: QueryState) -> str:
        """Route after vector retrieval.

        Args:
            state: Current workflow state

        Returns:
            Next node name
        """
        workflow_step = state["workflow_step"]

        if "failed" in workflow_step:
            return "error_handler"

        query_intent = state["query_intent"]

        # For comparison queries, also do keyword search
        if query_intent == "comparison":
            return "retrieve_keyword"

        # For relational queries, also do graph traversal
        if query_intent == "relational" and not state["graph_traversal_results"]:
            return "traverse_graph"

        # Otherwise, assemble context
        return "assemble_context"

    @staticmethod
    def route_after_keyword_retrieval(state: QueryState) -> str:
        """Route after keyword retrieval.

        Args:
            state: Current workflow state

        Returns:
            Next node name
        """
        workflow_step = state["workflow_step"]

        if "failed" in workflow_step:
            return "error_handler"

        # After keyword retrieval, assemble context
        return "assemble_context"

    @staticmethod
    def route_after_graph_traversal(state: QueryState) -> str:
        """Route after graph traversal.

        Args:
            state: Current workflow state

        Returns:
            Next node name
        """
        workflow_step = state["workflow_step"]

        if "failed" in workflow_step:
            return "error_handler"

        # After graph traversal, check if we need vector retrieval
        if not state["retrieved_chunks"] and state["query_intent"] != "relational":
            return "retrieve_vector"

        # Otherwise, assemble context
        return "assemble_context"

    @staticmethod
    def route_after_context_assembly(state: QueryState) -> str:
        """Route after context assembly.

        Args:
            state: Current workflow state

        Returns:
            Next node name
        """
        workflow_step = state["workflow_step"]

        if "failed" in workflow_step:
            return "error_handler"

        # After context assembly, synthesize answer
        return "synthesize_answer"

    @staticmethod
    def route_after_synthesis(state: QueryState) -> str:
        """Route after answer synthesis.

        Args:
            state: Current workflow state

        Returns:
            Next node name (should be END)
        """
        workflow_step = state["workflow_step"]

        if "failed" in workflow_step:
            return "error_handler"

        # Workflow complete
        return "end"

    @staticmethod
    def route_after_error(state: QueryState) -> str:
        """Route after error handling.

        Args:
            state: Current workflow state

        Returns:
            Next node name
        """
        workflow_step = state["workflow_step"]

        if workflow_step == "error_recovered":
            # Try to continue from context assembly
            if state["context_documents"]:
                return "synthesize_answer"
            elif state["retrieved_chunks"]:
                return "assemble_context"
            else:
                return "retrieve_vector"  # Retry retrieval
        elif workflow_step in ["failed", "critical_failure"]:
            return "end"
        else:
            # Default: try to continue
            return "synthesize_answer"

    @staticmethod
    def should_use_multi_strategy(state: QueryState) -> bool:
        """Determine if query requires multiple retrieval strategies.

        Args:
            state: Current workflow state

        Returns:
            True if multiple strategies needed
        """
        query_intent = state["query_intent"]

        # Comparison and relational queries benefit from multiple strategies
        return query_intent in ["comparison", "relational"]

    @staticmethod
    def get_required_strategies(state: QueryState) -> List[str]:
        """Get list of required retrieval strategies for query.

        Args:
            state: Current workflow state

        Returns:
            List of strategy names
        """
        query_intent = state["query_intent"]
        strategies = []

        if query_intent in ["factual", "synthesis", "comparison", "temporal"]:
            strategies.append("vector")

        if query_intent == "comparison":
            strategies.append("keyword")

        if query_intent == "relational":
            strategies.append("graph")

        return strategies if strategies else ["vector"]  # Default to vector

