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
            logger.info(
                "Running relationship detection workflow",
                document_count=len(documents),
                folder_id=folder_id,
            )
            
            try:
                final_state = await self.workflow.run(
                    documents=documents,
                    folder_id=folder_id,
                    config=config if config else None,
                )
                
                logger.info(
                    "Workflow run completed",
                    folder_id=folder_id,
                    state_keys=list(final_state.keys()) if final_state else [],
                )
            except Exception as workflow_error:
                logger.error(
                    "Workflow run failed, falling back to GraphBuilder",
                    folder_id=folder_id,
                    error=str(workflow_error),
                    error_type=type(workflow_error).__name__,
                    exc_info=True,
                )
                # Fallback: use GraphBuilder which does entity/relationship extraction
                logger.info("Using GraphBuilder as fallback for graph construction")
                graph = await self.graph_builder.build_graph(
                    chunks=documents,
                    relationships=None,  # Will extract relationships
                )
                return graph

            # Extract relationships and entities from state
            relationships = final_state.get("relationships", [])
            entities = final_state.get("entities", {})  # Dict mapping file_id to entity list

            logger.info(
                "Relationship detection completed",
                relationship_count=len(relationships),
                entity_count=sum(len(ent_list) for ent_list in entities.values()),
                entities_keys=list(entities.keys()) if entities else [],
                folder_id=folder_id,
            )
            
            # If no relationships were found, log warning and try GraphBuilder as fallback
            if not relationships and not entities:
                logger.warning(
                    "No relationships or entities found from workflow, using GraphBuilder fallback",
                    folder_id=folder_id,
                )
                # Fallback: use GraphBuilder
                graph = await self.graph_builder.build_graph(
                    chunks=documents,
                    relationships=None,  # Will extract relationships
                )
                logger.info(
                    "GraphBuilder fallback completed",
                    node_count=graph.number_of_nodes(),
                    edge_count=graph.number_of_edges(),
                    folder_id=folder_id,
                )
                if folder_id:
                    self.graphs[folder_id] = graph
                return graph

            # Build NetworkX graph directly from workflow results
            # IMPORTANT: Don't use graph_builder.build_graph() as it would do
            # duplicate entity extraction via FoldexKnowledgeGraph
            import networkx as nx
            graph = nx.DiGraph()

            # Add document nodes
            # Build multiple mappings for robust ID lookups
            doc_ids_to_names = {}
            chunk_ids_to_names = {}
            for doc in documents:
                file_id = doc.file_id if hasattr(doc, 'file_id') else doc.metadata.get('file_id')
                chunk_id = doc.chunk_id if hasattr(doc, 'chunk_id') else None
                file_name = doc.metadata.get('file_name', 'unknown') if hasattr(doc, 'metadata') else 'unknown'

                # Map both file_id and chunk_id to file_name for robust lookups
                if file_id:
                    doc_ids_to_names[file_id] = file_name
                if chunk_id:
                    chunk_ids_to_names[chunk_id] = file_name
                    doc_ids_to_names[chunk_id] = file_name  # Also map chunk_id

                # Only add document node once (use file_name as unique key)
                if not graph.has_node(file_name):
                    graph.add_node(
                        file_name,
                        type="document",
                        label=file_name,
                        node_type="document",
                        file_id=file_id,
                        chunk_id=chunk_id,
                    )

            # Add entities from workflow (already extracted, no LLM call needed)
            # entities dict is keyed by document_id (could be chunk_id or file_id)
            for doc_id, entity_list in entities.items():
                for entity in entity_list:
                    entity_name = entity.get("name", "").strip()
                    entity_type = entity.get("type", "unknown")

                    if not entity_name:
                        continue

                    # Add or update node
                    if graph.has_node(entity_name):
                        # Update existing node
                        graph.nodes[entity_name]["type"] = entity_type
                    else:
                        # Add new node
                        graph.add_node(
                            entity_name,
                            type=entity_type,
                            label=entity_name,
                            node_type="entity",
                        )

                    # Add edge from entity to document (try both mappings)
                    file_name = doc_ids_to_names.get(doc_id) or chunk_ids_to_names.get(doc_id)
                    if file_name and graph.has_node(file_name):
                        graph.add_edge(
                            entity_name,
                            file_name,
                            relation="mentioned_in",
                        )

            # Add relationships from workflow
            for rel in relationships:
                source = rel.get("source", "").strip()
                target = rel.get("target", "").strip()
                relation = rel.get("relation", "related")

                if not source or not target:
                    continue

                # Ensure nodes exist (they might be file IDs, convert to names)
                source_name = doc_ids_to_names.get(source, source)
                target_name = doc_ids_to_names.get(target, target)

                if not graph.has_node(source_name):
                    graph.add_node(
                        source_name,
                        type="entity",
                        label=source_name,
                        node_type="entity",
                    )

                if not graph.has_node(target_name):
                    graph.add_node(
                        target_name,
                        type="entity",
                        label=target_name,
                        node_type="entity",
                    )

                # Add edge
                graph.add_edge(source_name, target_name, relation=relation)

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

