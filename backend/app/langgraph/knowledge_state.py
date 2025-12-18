"""LangGraph state management for knowledge graph construction."""

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


class KnowledgeGraphState(TypedDict):
    """State schema for LangGraph knowledge graph construction workflow.

    Tracks documents, entities, relationships, and processing status
    across multiple workflow steps.
    """

    # Input documents
    documents: List[DocumentChunk]
    
    # Extracted entities per document
    entities: Dict[str, List[Dict[str, Any]]]  # document_id -> [entities]
    
    # Detected relationships
    relationships: List[Dict[str, Any]]
    
    # Processing status
    processed_documents: Set[str]
    failed_documents: Set[str]
    
    # Workflow metadata
    folder_id: Optional[str]
    workflow_step: str  # Current step in workflow
    error_message: Optional[str]
    
    # Intermediate results
    entity_overlaps: List[Dict[str, Any]]
    temporal_relationships: List[Dict[str, Any]]
    cross_references: List[Dict[str, Any]]
    semantic_similarities: List[Dict[str, Any]]
    implementation_gaps: List[Dict[str, Any]]
    
    # Document type classification
    document_types: Dict[str, str]  # document_id -> type (pdf, audio, code, text)
    
    # Processing statistics
    stats: Dict[str, Any]


class KnowledgeStateManager:
    """Manages state for LangGraph knowledge graph construction workflow."""

    def __init__(self):
        """Initialize state manager."""
        self.logger = structlog.get_logger(__name__)

    @staticmethod
    def create_initial_state(
        documents: List[DocumentChunk],
        folder_id: Optional[str] = None,
    ) -> KnowledgeGraphState:
        """Create initial state for workflow.

        Args:
            documents: List of document chunks to process
            folder_id: Optional folder identifier

        Returns:
            Initial KnowledgeGraphState
        """
        # Classify document types
        document_types = {}
        for doc in documents:
            mime_type = doc.metadata.get("mime_type", "")
            mime_type_str = str(mime_type) if mime_type else ""
            if mime_type_str == "application/pdf":
                document_types[doc.chunk_id] = "pdf"
            elif mime_type_str.startswith("audio/"):
                document_types[doc.chunk_id] = "audio"
            elif mime_type_str.startswith("text/") or doc.metadata.get("is_code"):
                document_types[doc.chunk_id] = "code" if doc.metadata.get("is_code") else "text"
            else:
                document_types[doc.chunk_id] = "text"

        return KnowledgeGraphState(
            documents=documents,
            entities={},
            relationships=[],
            processed_documents=set(),
            failed_documents=set(),
            folder_id=folder_id,
            workflow_step="initialized",
            error_message=None,
            entity_overlaps=[],
            temporal_relationships=[],
            cross_references=[],
            semantic_similarities=[],
            implementation_gaps=[],
            document_types=document_types,
            stats={
                "total_documents": len(documents),
                "processed_count": 0,
                "failed_count": 0,
                "relationship_count": 0,
            },
        )

    @staticmethod
    def update_state(
        state: KnowledgeGraphState,
        updates: Dict[str, Any],
    ) -> KnowledgeGraphState:
        """Update state with new values.

        Args:
            state: Current state
            updates: Dictionary of updates to apply

        Returns:
            Updated state
        """
        # Create new state dict with updates
        new_state: Dict[str, Any] = {
            "documents": state["documents"],
            "entities": state["entities"].copy(),
            "relationships": state["relationships"].copy(),
            "processed_documents": state["processed_documents"].copy(),
            "failed_documents": state["failed_documents"].copy(),
            "folder_id": state["folder_id"],
            "workflow_step": state["workflow_step"],
            "error_message": state["error_message"],
            "entity_overlaps": state["entity_overlaps"].copy(),
            "temporal_relationships": state["temporal_relationships"].copy(),
            "cross_references": state["cross_references"].copy(),
            "semantic_similarities": state["semantic_similarities"].copy(),
            "implementation_gaps": state["implementation_gaps"].copy(),
            "document_types": state["document_types"].copy(),
            "stats": state["stats"].copy(),
        }

        # Apply updates
        for key, value in updates.items():
            if key == "processed_documents":
                if isinstance(value, set):
                    new_state[key] = state[key].union(value)
                else:
                    new_state[key] = state[key].union(set(value))
            elif key == "failed_documents":
                if isinstance(value, set):
                    new_state[key] = state[key].union(value)
                else:
                    new_state[key] = state[key].union(set(value))
            elif key in ["relationships", "entity_overlaps", "temporal_relationships", 
                        "cross_references", "semantic_similarities", "implementation_gaps"]:
                if isinstance(value, list):
                    new_state[key] = state[key] + value
                else:
                    new_state[key] = state[key]
            elif key in ["entities", "document_types"]:
                if isinstance(value, dict):
                    new_state[key] = {**state[key], **value}
                else:
                    new_state[key] = state[key]
            else:
                new_state[key] = value

        # Build TypedDict properly
        result: KnowledgeGraphState = {
            "documents": new_state["documents"],
            "entities": new_state["entities"],
            "relationships": new_state["relationships"],
            "processed_documents": new_state["processed_documents"],
            "failed_documents": new_state["failed_documents"],
            "folder_id": new_state["folder_id"],
            "workflow_step": new_state["workflow_step"],
            "error_message": new_state["error_message"],
            "entity_overlaps": new_state["entity_overlaps"],
            "temporal_relationships": new_state["temporal_relationships"],
            "cross_references": new_state["cross_references"],
            "semantic_similarities": new_state["semantic_similarities"],
            "implementation_gaps": new_state["implementation_gaps"],
            "document_types": new_state["document_types"],
            "stats": new_state["stats"],
        }
        return result

    @staticmethod
    def get_state_summary(state: KnowledgeGraphState) -> Dict[str, Any]:
        """Get summary of current state.

        Args:
            state: Current state

        Returns:
            Dictionary with state summary
        """
        return {
            "workflow_step": state["workflow_step"],
            "total_documents": len(state["documents"]),
            "processed_count": len(state["processed_documents"]),
            "failed_count": len(state["failed_documents"]),
            "relationship_count": len(state["relationships"]),
            "entity_count": sum(len(entities) for entities in state["entities"].values()),
            "folder_id": state["folder_id"],
            "error_message": state["error_message"],
        }

