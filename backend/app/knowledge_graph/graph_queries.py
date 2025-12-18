"""Graph traversal and query logic."""

from typing import List
import structlog
import networkx as nx

from app.models.documents import DocumentChunk
from app.database.sqlite_manager import SQLiteManager

logger = structlog.get_logger(__name__)


class GraphQueries:
    """Query interface for knowledge graphs."""

    def __init__(self, db: SQLiteManager | None = None):
        """Initialize graph queries.

        Args:
            db: Database manager
        """
        self.db = db or SQLiteManager()

    async def traverse_related(
        self, query: str, folder_id: str, k: int = 10
    ) -> List[DocumentChunk]:
        """Traverse graph to find related chunks.

        Args:
            query: Query text
            folder_id: Folder ID
            k: Number of chunks to return

        Returns:
            List of related document chunks
        """
        try:
            logger.info("Traversing knowledge graph", query_length=len(query), folder_id=folder_id)

            # Load graph from database
            graph_data = await self.db.get_knowledge_graph(folder_id)
            if not graph_data or not graph_data.get("nodes") or not graph_data.get("edges"):
                logger.warning("No knowledge graph found for folder", folder_id=folder_id)
                return []
            
            # Build NetworkX graph
            G = nx.DiGraph()
            
            # Add nodes (chunks)
            chunk_map = {}
            for node in graph_data["nodes"]:
                node_id = node.get("id")
                if node_id:
                    G.add_node(node_id, **node)
                    # Store chunk data for later retrieval
                    if node.get("type") == "chunk":
                        chunk_map[node_id] = node
            
            # Add edges (relationships)
            for edge in graph_data["edges"]:
                source = edge.get("source")
                target = edge.get("target")
                if source and target:
                    G.add_edge(source, target, **edge)
            
            if not chunk_map:
                logger.warning("No chunk nodes in graph", folder_id=folder_id)
                return []
            
            # Find starting nodes by keyword matching (simple approach)
            query_terms = set(query.lower().split())
            scored_nodes = []
            
            for node_id, node_data in chunk_map.items():
                content = node_data.get("content", "")
                if content:
                    content_terms = set(content.lower().split())
                    overlap = len(query_terms & content_terms)
                    if overlap > 0:
                        scored_nodes.append((node_id, overlap))
            
            if not scored_nodes:
                logger.info("No matching starting nodes found")
                return []
            
            # Sort by relevance
            scored_nodes.sort(key=lambda x: x[1], reverse=True)
            
            # Traverse from top starting nodes
            related_node_ids = set()
            max_start_nodes = min(3, len(scored_nodes))
            
            for node_id, _ in scored_nodes[:max_start_nodes]:
                # Add the node itself
                related_node_ids.add(node_id)
                
                # Add neighbors (1-hop traversal)
                if G.has_node(node_id):
                    # Outgoing edges
                    related_node_ids.update(G.successors(node_id))
                    # Incoming edges
                    related_node_ids.update(G.predecessors(node_id))
            
            # Retrieve chunks for related nodes - create DocumentChunk from node data
            from app.models.documents import DocumentChunk
            
            related_chunks = []
            for node_id in related_node_ids:
                if node_id in chunk_map:
                    node_data = chunk_map[node_id]
                    # Create DocumentChunk from node data
                    try:
                        chunk = DocumentChunk(
                            chunk_id=node_data.get("chunk_id", node_id),
                            file_id=node_data.get("file_id", ""),
                            content=node_data.get("content", ""),
                            metadata=node_data.get("metadata", {}),
                            embedding=None,  # Not needed for retrieval
                        )
                        if chunk.content:  # Only add if has content
                            related_chunks.append(chunk)
                    except Exception as e:
                        logger.warning("Failed to create chunk from node data", node_id=node_id, error=str(e))
                        continue
            
            logger.info(
                "Graph traversal completed",
                start_nodes=max_start_nodes,
                related_nodes=len(related_node_ids),
                chunks_found=len(related_chunks)
            )
            
            return related_chunks[:k]
            
        except Exception as e:
            logger.error("Graph traversal failed", error=str(e), exc_info=True)
            return []

