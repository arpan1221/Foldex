"""Master LangGraph workflow orchestration for graph intelligence and RAG integration."""

from typing import Optional, Dict, Any, List, Callable
import structlog

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

from app.core.exceptions import ProcessingError
from app.langgraph.agent_workflow import AgentWorkflow
from app.langgraph.query_workflow import QueryWorkflow
from app.langgraph.relationship_workflow import RelationshipWorkflow
from app.rag.llm_chains import OllamaLLM
from app.services.rag_service import LangChainRAGService
from app.services.langgraph_knowledge_service import LangGraphKnowledgeService
from app.knowledge_graph.graph_builder import GraphBuilder

logger = structlog.get_logger(__name__)


class LangGraphOrchestrator:
    """Master orchestrator for all LangGraph workflows and RAG integration.

    Coordinates query processing, knowledge graph enhancement, and
    RAG retrieval with graph intelligence.
    """

    def __init__(
        self,
        llm: Optional[OllamaLLM] = None,
        rag_service: Optional[LangChainRAGService] = None,
        knowledge_service: Optional[LangGraphKnowledgeService] = None,
        enable_checkpointing: bool = True,
    ):
        """Initialize LangGraph orchestrator.

        Args:
            llm: Optional Ollama LLM instance
            rag_service: Optional RAG service instance
            knowledge_service: Optional knowledge graph service
            enable_checkpointing: Whether to enable checkpointing

        Raises:
            ProcessingError: If initialization fails
        """
        self.llm = llm or OllamaLLM()
        
        # Initialize RAG service with vector store if not provided
        if rag_service is None:
            try:
                from app.rag.vector_store import LangChainVectorStore
                from app.services.rag_service import get_rag_service
                
                logger.info("Initializing LangChain vector store for RAG service")
                vector_store = LangChainVectorStore()
                rag_service = get_rag_service(vector_store=vector_store)
                logger.info("RAG service initialized successfully")
            except Exception as e:
                logger.warning(
                    "Failed to initialize RAG service, will be unavailable",
                    error=str(e),
                    exc_info=True,
                )
                rag_service = None
        
        self.rag_service = rag_service
        self.knowledge_service = knowledge_service
        self.enable_checkpointing = enable_checkpointing

        # Initialize workflows
        try:
            llm_instance = self.llm.get_llm() if hasattr(self.llm, "get_llm") else None

            self.query_workflow = QueryWorkflow(
                llm=llm_instance,
                enable_checkpointing=enable_checkpointing,
            )

            self.relationship_workflow = RelationshipWorkflow(
                llm=llm_instance,
                enable_checkpointing=enable_checkpointing,
            )

            self.agent_workflow = AgentWorkflow(
                llm=llm_instance,
                enable_checkpointing=enable_checkpointing,
            )

            # Initialize graph builder
            self.graph_builder = GraphBuilder()

            logger.info("Initialized LangGraph orchestrator")

        except Exception as e:
            logger.error("Failed to initialize orchestrator", error=str(e))
            raise ProcessingError(f"Failed to initialize orchestrator: {str(e)}") from e

    async def process_query_with_graph_intelligence(
        self,
        query: str,
        folder_id: str,
        conversation_id: Optional[str] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        use_graph_enhancement: bool = True,
        streaming_callback: Optional[Callable[[str], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        citations_callback: Optional[Callable[[list], None]] = None,
    ) -> Dict[str, Any]:
        """Process query with graph-enhanced RAG.

        Args:
            query: User query
            folder_id: Folder identifier
            conversation_id: Optional conversation identifier
            conversation_history: Optional conversation history
            use_graph_enhancement: Whether to use graph enhancement
            streaming_callback: Optional callback for streaming tokens
            status_callback: Optional callback for status updates
            citations_callback: Optional callback for progressive citations

        Returns:
            Dictionary with answer, citations, and graph insights

        Raises:
            ProcessingError: If processing fails
        """
        try:
            logger.info(
                "Processing query with graph intelligence",
                query_length=len(query),
                folder_id=folder_id,
                use_graph_enhancement=use_graph_enhancement,
            )

            # Step 1: Process query through query workflow (skip if streaming for now)
            query_state = None
            if not streaming_callback:
                try:
                    query_state = await self.query_workflow.run(
                        query=query,
                        folder_id=folder_id,
                        conversation_id=conversation_id,
                        conversation_history=conversation_history,
                    )
                except Exception as e:
                    logger.warning("Query workflow failed, continuing with RAG service", error=str(e))
                    query_state = None
            else:
                logger.info("Skipping query workflow for streaming requests")

            # Step 2: Enhance graph if enabled
            graph_insights = None
            if use_graph_enhancement and self.knowledge_service:
                graph_insights = await self._enhance_graph_for_query(
                    query_state=query_state,
                    folder_id=folder_id,
                )

            # Step 3: Use RAG service with graph context
            # For now, skip query workflow and use RAG service directly for streaming
            rag_result = None
            if self.rag_service and streaming_callback:
                # If streaming is requested, use RAG service directly (bypass query workflow for now)
                logger.info("Using RAG service directly for streaming")
                rag_result = await self.rag_service.query(
                    query=query,
                    folder_id=folder_id,
                    conversation_id=conversation_id,
                    query_type=query_state.get("query_intent") if query_state else None,
                    streaming_callback=streaming_callback,
                    status_callback=status_callback,
                    citations_callback=citations_callback,
                )
            elif self.rag_service:
                # Non-streaming: use RAG service
                rag_result = await self.rag_service.query(
                    query=query,
                    folder_id=folder_id,
                    conversation_id=conversation_id,
                    query_type=query_state.get("query_intent") if query_state else None,
                    streaming_callback=None,
                    status_callback=None,
                    citations_callback=None,
                )

            # Combine results (RAG service returns "answer", query_state might have "final_answer")
            result = {
                "answer": (
                    (query_state.get("final_answer") if query_state else None)
                    or rag_result.get("answer", "")
                    or rag_result.get("response", "")
                ) if rag_result else "",
                "citations": (
                    (query_state.get("citations") if query_state else None)
                    or rag_result.get("citations", [])
                ) if rag_result else [],
                "confidence": (
                    (query_state.get("confidence_score", 0.0) if query_state else 0.0)
                    or rag_result.get("confidence", 0.0)
                ) if rag_result else 0.0,
                "query_intent": query_state.get("query_intent") if query_state else None,
                "graph_insights": graph_insights,
                "routing_path": query_state.get("routing_path", []) if query_state else [],
            }
            
            if not rag_result:
                logger.error("No RAG result available", folder_id=folder_id)
                raise ProcessingError("RAG service did not return a result")

            logger.info(
                "Query processing completed",
                answer_length=len(result["answer"]),
                citation_count=len(result["citations"]),
            )

            return result

        except Exception as e:
            logger.error(
                "Query processing with graph intelligence failed",
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(f"Query processing failed: {str(e)}") from e

    async def _enhance_graph_for_query(
        self,
        query_state: Dict[str, Any],
        folder_id: str,
    ) -> Optional[Dict[str, Any]]:
        """Enhance knowledge graph based on query context.

        Args:
            query_state: Query workflow state
            folder_id: Folder identifier

        Returns:
            Dictionary with graph insights
        """
        try:
            # Extract entities from query
            entities = query_state.get("entities", [])

            if not entities:
                return None

            # Get current graph
            # This would integrate with knowledge service to get graph
            # For now, return placeholder
            return {
                "entities_found": len(entities),
                "graph_enhanced": True,
            }

        except Exception as e:
            logger.error("Graph enhancement failed", error=str(e))
            return None

    async def enhance_knowledge_graph(
        self,
        folder_id: str,
        entities: Optional[List[Dict[str, Any]]] = None,
        relationships: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Enhance knowledge graph using agent workflow.

        Args:
            folder_id: Folder identifier
            entities: Optional list of entities
            relationships: Optional list of relationships

        Returns:
            Dictionary with enhancement results

        Raises:
            ProcessingError: If enhancement fails
        """
        try:
            logger.info(
                "Enhancing knowledge graph",
                folder_id=folder_id,
                entity_count=len(entities) if entities else 0,
                relationship_count=len(relationships) if relationships else 0,
            )

            # Get graph data
            graph_data = {"folder_id": folder_id}

            # Run agent workflow
            agent_state = await self.agent_workflow.run(
                graph_data=graph_data,
                entities=entities or [],
                relationships=relationships or [],
                folder_id=folder_id,
            )

            # Extract results
            result = {
                "conflicts_detected": len(agent_state.get("detected_conflicts", [])),
                "conflicts_resolved": len(agent_state.get("conflict_resolutions", [])),
                "updates_applied": len(agent_state.get("applied_updates", [])),
                "quality_scores": agent_state.get("quality_scores", {}),
                "improvement_suggestions": agent_state.get("improvement_suggestions", []),
            }

            logger.info(
                "Knowledge graph enhancement completed",
                overall_quality=result["quality_scores"].get("overall_quality", 0.0),
            )

            return result

        except Exception as e:
            logger.error(
                "Knowledge graph enhancement failed",
                folder_id=folder_id,
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(f"Graph enhancement failed: {str(e)}") from e

    async def build_enhanced_knowledge_graph(
        self,
        documents: List[Any],
        folder_id: str,
    ) -> nx.Graph:
        """Build and enhance knowledge graph from documents.

        Args:
            documents: List of document chunks
            folder_id: Folder identifier

        Returns:
            Enhanced NetworkX graph

        Raises:
            ProcessingError: If graph building fails
        """
        try:
            logger.info(
                "Building enhanced knowledge graph",
                document_count=len(documents),
                folder_id=folder_id,
            )

            # Step 1: Detect relationships
            relationship_state = await self.relationship_workflow.run(
                documents=documents,
                folder_id=folder_id,
            )

            relationships = relationship_state.get("relationships", [])

            # Step 2: Build initial graph
            graph = await self.graph_builder.build_graph(
                chunks=documents,
                relationships=relationships,
            )

            # Step 3: Enhance graph with agents
            entities = relationship_state.get("entities", {})
            entity_list = []
            for entity_group in entities.values():
                entity_list.extend(entity_group)

            enhancement_result = await self.enhance_knowledge_graph(
                folder_id=folder_id,
                entities=entity_list,
                relationships=relationships,
            )

            logger.info(
                "Enhanced knowledge graph built",
                node_count=graph.number_of_nodes(),
                edge_count=graph.number_of_edges(),
                overall_quality=enhancement_result["quality_scores"].get("overall_quality", 0.0),
            )

            return graph

        except Exception as e:
            logger.error(
                "Enhanced knowledge graph building failed",
                folder_id=folder_id,
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(f"Graph building failed: {str(e)}") from e

    def validate_orchestrator(self) -> Dict[str, Any]:
        """Validate orchestrator configuration.

        Returns:
            Dictionary with validation results
        """
        try:
            validation_results: Dict[str, Any] = {
                "valid": True,
                "errors": [],
                "warnings": [],
            }

            # Validate workflows
            query_validation = self.query_workflow.validate_workflow()
            if not query_validation.get("valid", False):
                validation_results["valid"] = False
                validation_results["errors"].extend(query_validation.get("errors", []))

            agent_validation = self.agent_workflow.validate_workflow()
            if not agent_validation.get("valid", False):
                validation_results["valid"] = False
                validation_results["errors"].extend(agent_validation.get("errors", []))

            # Check NetworkX
            if not NETWORKX_AVAILABLE:
                validation_results["warnings"].append("NetworkX not available")

            logger.debug("Orchestrator validation completed", **validation_results)

            return validation_results

        except Exception as e:
            logger.error("Orchestrator validation failed", error=str(e))
            return {
                "valid": False,
                "errors": [f"Validation failed: {str(e)}"],
                "warnings": [],
            }


# Global orchestrator instance
_orchestrator: Optional[LangGraphOrchestrator] = None


def get_orchestrator(
    llm: Optional[OllamaLLM] = None,
    rag_service: Optional[LangChainRAGService] = None,
    knowledge_service: Optional[LangGraphKnowledgeService] = None,
) -> LangGraphOrchestrator:
    """Get global LangGraph orchestrator instance.

    Args:
        llm: Optional LLM instance
        rag_service: Optional RAG service
        knowledge_service: Optional knowledge service

    Returns:
        LangGraphOrchestrator instance
    """
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = LangGraphOrchestrator(
            llm=llm,
            rag_service=rag_service,
            knowledge_service=knowledge_service,
        )
    return _orchestrator

