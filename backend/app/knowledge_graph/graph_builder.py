"""NetworkX graph construction."""

from typing import List, Dict
import networkx as nx
import structlog

from app.models.documents import DocumentChunk

logger = structlog.get_logger(__name__)


class GraphBuilder:
    """Builds knowledge graphs using NetworkX."""

    def __init__(self):
        """Initialize graph builder."""
        self.graph: nx.Graph = nx.Graph()

    async def build_graph(
        self, chunks: List[DocumentChunk], relationships: List[Dict]
    ) -> nx.Graph:
        """Build NetworkX graph from chunks and relationships.

        Args:
            chunks: Document chunks
            relationships: Detected relationships

        Returns:
            NetworkX graph
        """
        try:
            logger.info(
                "Building knowledge graph",
                chunk_count=len(chunks),
                relationship_count=len(relationships),
            )

            # Create graph
            graph = nx.Graph()

            # Add nodes (chunks)
            for chunk in chunks:
                graph.add_node(
                    chunk.chunk_id,
                    **{
                        "file_id": chunk.file_id,
                        "content": chunk.content[:100],  # Truncate for storage
                        "metadata": chunk.metadata,
                    }
                )

            # Add edges (relationships)
            for rel in relationships:
                graph.add_edge(
                    rel["source_chunk_id"],
                    rel["target_chunk_id"],
                    **{
                        "relationship_type": rel["type"],
                        "confidence": rel.get("confidence", 0.0),
                    }
                )

            self.graph = graph
            return graph
        except Exception as e:
            logger.error("Graph construction failed", error=str(e))
            raise

