"""LangGraph workflow orchestration for knowledge graph construction."""

from typing import Optional, List, Dict, Any
import structlog

from app.core.exceptions import ProcessingError
from app.langgraph.relationship_workflow import RelationshipWorkflow
from app.langgraph.knowledge_state import KnowledgeGraphState
from app.knowledge_graph.graph_builder import GraphBuilder
from app.models.documents import DocumentChunk
from app.rag.llm_chains import OllamaLLM

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

logger = structlog.get_logger(__name__)


class LangGraphKnowledgeService:
    """Service for orchestrating LangGraph-based knowledge graph construction.

    Manages relationship detection workflows, graph construction, and
    checkpointing for resumable processing of large folders.
    """

    def __init__(
        self,
        llm: Optional[OllamaLLM] = None,
        enable_checkpointing: bool = True,
    ):
        """Initialize LangGraph knowledge service.

        Args:
            llm: Optional Ollama LLM instance
            enable_checkpointing: Whether to enable checkpointing

        Raises:
            ProcessingError: If initialization fails
        """
        self.llm = llm or OllamaLLM()
        self.enable_checkpointing = enable_checkpointing

        # Initialize workflow
        try:
            self.workflow = RelationshipWorkflow(
                llm=self.llm.get_llm() if hasattr(self.llm, "get_llm") else None,
                enable_checkpointing=enable_checkpointing,
            )
            logger.info("Initialized LangGraph knowledge service")
        except Exception as e:
            logger.error("Failed to initialize workflow", error=str(e))
            raise ProcessingError(f"Failed to initialize knowledge service: {str(e)}") from e

        # Initialize graph builder
        self.graph_builder = GraphBuilder()

        # Cache for graphs per folder
        self.graphs: Dict[str, nx.Graph] = {}

    async def build_knowledge_graph(
        self,
        documents: List[DocumentChunk],
        folder_id: Optional[str] = None,
        resume_from_checkpoint: bool = False,
    ) -> nx.Graph:
        """Build knowledge graph from documents using LangGraph workflow.

        Args:
            documents: List of document chunks to process
            folder_id: Optional folder identifier for checkpointing
            resume_from_checkpoint: Whether to resume from checkpoint if available

        Returns:
            NetworkX graph with nodes (documents) and edges (relationships)

        Raises:
            ProcessingError: If graph construction fails
        """
        try:
            logger.info(
                "Building knowledge graph",
                document_count=len(documents),
                folder_id=folder_id,
                resume_from_checkpoint=resume_from_checkpoint,
            )

            # Prepare config for checkpointing
            config = {}
            if self.enable_checkpointing and folder_id:
                config["configurable"] = {"thread_id": folder_id}

            # Run relationship detection workflow
            final_state = await self.workflow.run(
                documents=documents,
                folder_id=folder_id,
                config=config if config else None,
            )

            # Extract relationships from state
            relationships = final_state.get("relationships", [])

            logger.info(
                "Relationship detection completed",
                relationship_count=len(relationships),
                folder_id=folder_id,
            )

            # Build NetworkX graph
            graph = await self.graph_builder.build_graph(
                chunks=documents,
                relationships=relationships,
            )

            # Cache graph
            if folder_id:
                self.graphs[folder_id] = graph

            logger.info(
                "Knowledge graph built",
                node_count=graph.number_of_nodes(),
                edge_count=graph.number_of_edges(),
                folder_id=folder_id,
            )

            return graph

        except Exception as e:
            logger.error(
                "Knowledge graph construction failed",
                folder_id=folder_id,
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(
                f"Knowledge graph construction failed: {str(e)}"
            ) from e

    async def detect_relationships(
        self,
        documents: List[DocumentChunk],
        folder_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Detect relationships between documents using LangGraph workflow.

        Args:
            documents: List of document chunks
            folder_id: Optional folder identifier

        Returns:
            List of relationship dictionaries

        Raises:
            ProcessingError: If relationship detection fails
        """
        try:
            logger.info(
                "Detecting relationships",
                document_count=len(documents),
                folder_id=folder_id,
            )

            # Run workflow
            final_state = await self.workflow.run(
                documents=documents,
                folder_id=folder_id,
            )

            # Extract relationships
            relationships = final_state.get("relationships", [])

            logger.info(
                "Relationships detected",
                relationship_count=len(relationships),
                folder_id=folder_id,
            )

            return relationships

        except Exception as e:
            logger.error(
                "Relationship detection failed",
                folder_id=folder_id,
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(f"Relationship detection failed: {str(e)}") from e

    def get_workflow_state_summary(
        self,
        folder_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Get summary of workflow state for a folder.

        Args:
            folder_id: Folder identifier

        Returns:
            Dictionary with workflow state summary

        Raises:
            ProcessingError: If state retrieval fails
        """
        try:
            if not self.enable_checkpointing or not folder_id:
                return {"checkpointing_enabled": False}

            # Get state from checkpointer
            if self.workflow.checkpointer:
                # Note: This is a simplified version
                # Full implementation would query checkpointer for state
                return {
                    "checkpointing_enabled": True,
                    "folder_id": folder_id,
                    "has_checkpoint": True,  # Placeholder
                }

            return {"checkpointing_enabled": False}

        except Exception as e:
            logger.error("Failed to get workflow state", error=str(e))
            return {"error": str(e)}

    def get_graph(self, folder_id: str) -> Optional[nx.Graph]:
        """Get cached knowledge graph for folder.

        Args:
            folder_id: Folder identifier

        Returns:
            NetworkX graph or None if not found
        """
        return self.graphs.get(folder_id)

    def clear_graph_cache(self, folder_id: Optional[str] = None) -> None:
        """Clear cached graphs.

        Args:
            folder_id: Optional folder identifier (clears all if None)
        """
        if folder_id:
            if folder_id in self.graphs:
                del self.graphs[folder_id]
                logger.debug("Cleared graph cache", folder_id=folder_id)
        else:
            self.graphs.clear()
            logger.debug("Cleared all graph caches")

    def validate_service(self) -> Dict[str, Any]:
        """Validate service configuration.

        Returns:
            Dictionary with validation results
        """
        try:
            validation_results = {
                "valid": True,
                "errors": [],
                "warnings": [],
            }

            # Validate workflow
            workflow_validation = self.workflow.validate_workflow()
            if not workflow_validation.get("valid", False):
                validation_results["valid"] = False
                validation_results["errors"].extend(workflow_validation.get("errors", []))

            validation_results["warnings"].extend(workflow_validation.get("warnings", []))

            # Check NetworkX
            if not NETWORKX_AVAILABLE:
                validation_results["warnings"].append("NetworkX not available")

            logger.debug("Service validation completed", **validation_results)

            return validation_results

        except Exception as e:
            logger.error("Service validation failed", error=str(e))
            return {
                "valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": [],
            }


# Global service instance
_knowledge_service: Optional[LangGraphKnowledgeService] = None


def get_knowledge_service(
    llm: Optional[OllamaLLM] = None,
) -> LangGraphKnowledgeService:
    """Get global LangGraph knowledge service instance.

    Args:
        llm: Optional LLM instance

    Returns:
        LangGraphKnowledgeService instance
    """
    global _knowledge_service
    if _knowledge_service is None:
        _knowledge_service = LangGraphKnowledgeService(llm=llm)
    return _knowledge_service

