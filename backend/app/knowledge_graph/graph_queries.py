"""Graph traversal and query logic."""

from typing import List, Optional, Tuple, Dict
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

    async def _load_graph(self, folder_id: str) -> Tuple[Optional[nx.DiGraph], Dict]:
        """Load and build NetworkX graph from database.
        
        Args:
            folder_id: Folder identifier
            
        Returns:
            Tuple of (graph, chunk_map)
        """
        try:
            graph_data = await self.db.get_knowledge_graph(folder_id)
            if not graph_data:
                return None, {}
            
            # Parse graph_data if it's bytes (JSON)
            import json
            if isinstance(graph_data, bytes):
                graph_data = json.loads(graph_data.decode('utf-8'))
            elif not isinstance(graph_data, dict):
                logger.warning("Unexpected graph_data format", type=type(graph_data))
                return None, {}
        except Exception as e:
            logger.warning("Failed to load graph from database", error=str(e))
            return None, {}
        if not graph_data or not graph_data.get("nodes") or not graph_data.get("edges"):
            return None, {}
        
        G = nx.DiGraph()
        chunk_map = {}
        
        # Add nodes
        for node in graph_data["nodes"]:
            node_id = node.get("id")
            if node_id:
                G.add_node(node_id, **node)
                if node.get("type") == "chunk":
                    chunk_map[node_id] = node
        
        # Add edges
        for edge in graph_data["edges"]:
            source = edge.get("source")
            target = edge.get("target")
            if source and target:
                G.add_edge(source, target, **edge)
        
        return G, chunk_map

    async def find_related_chunks(
        self,
        chunk_id: str,
        folder_id: str,
        relationship_types: Optional[List[str]] = None,
        max_depth: int = 2,
    ) -> List[Tuple[str, List[str]]]:
        """Find all chunks related to a given chunk via graph traversal.
        
        Args:
            chunk_id: Starting chunk ID
            folder_id: Folder identifier
            relationship_types: Optional list of relationship types to filter
            max_depth: Maximum traversal depth
            
        Returns:
            List of tuples (chunk_id, path) where path is list of relationship types
        """
        try:
            G, chunk_map = await self._load_graph(folder_id)
            if G is None or not G.has_node(chunk_id):
                return []
            
            if relationship_types is None:
                relationship_types = ["entity_overlap", "cross_reference", "topical_similarity"]
            
            # BFS traversal
            visited = set()
            queue = [(chunk_id, [])]  # (chunk_id, path)
            results = []
            
            while queue:
                current_id, path = queue.pop(0)
                
                if current_id in visited or len(path) > max_depth:
                    continue
                
                visited.add(current_id)
                
                if current_id != chunk_id:
                    results.append((current_id, path))
                
                # Find neighbors with matching relationship types
                for source, target, data in G.edges(current_id, data=True):
                    rel_type = data.get("relationship_type") or data.get("type")
                    if rel_type in relationship_types:
                        queue.append((target, path + [rel_type]))
            
            logger.debug(
                "Found related chunks",
                chunk_id=chunk_id,
                related_count=len(results),
                max_depth=max_depth,
            )
            
            return results
            
        except Exception as e:
            logger.error("find_related_chunks failed", error=str(e), exc_info=True)
            return []

    async def find_contradictions(self, folder_id: str) -> List[Tuple[str, str, str]]:
        """Find chunks with 'implementation_gap' relationships.
        
        Args:
            folder_id: Folder identifier
            
        Returns:
            List of tuples (source_chunk_id, target_chunk_id, gap_description)
        """
        try:
            G, _ = await self._load_graph(folder_id)
            if G is None:
                return []
            
            contradictions = []
            for source, target, data in G.edges(data=True):
                rel_type = data.get("relationship_type") or data.get("type")
                if rel_type == "implementation_gap":
                    gap_desc = data.get("metadata", {}).get("gap_description", "Implementation gap detected")
                    contradictions.append((source, target, gap_desc))
            
            logger.debug("Found contradictions", count=len(contradictions))
            return contradictions
            
        except Exception as e:
            logger.error("find_contradictions failed", error=str(e), exc_info=True)
            return []

    async def find_entity_occurrences(self, entity_text: str, folder_id: str) -> List[str]:
        """Find all chunks mentioning an entity.
        
        Args:
            entity_text: Entity text to search for
            folder_id: Folder identifier
            
        Returns:
            List of chunk IDs
        """
        try:
            _, chunk_map = await self._load_graph(folder_id)
            if not chunk_map:
                return []
            
            matching_chunks = []
            entity_lower = entity_text.lower()
            
            for node_id, node_data in chunk_map.items():
                content = node_data.get("content", "").lower()
                if entity_lower in content:
                    matching_chunks.append(node_id)
            
            logger.debug(
                "Found entity occurrences",
                entity=entity_text,
                count=len(matching_chunks),
            )
            
            return matching_chunks
            
        except Exception as e:
            logger.error("find_entity_occurrences failed", error=str(e), exc_info=True)
            return []

    async def find_cross_references(
        self, file_name: str, folder_id: str
    ) -> List[Tuple[str, str]]:
        """Find chunks that reference a specific file.
        
        Args:
            file_name: File name to search for
            folder_id: Folder identifier
            
        Returns:
            List of tuples (referencing_chunk_id, referenced_chunk_id)
        """
        try:
            G, chunk_map = await self._load_graph(folder_id)
            if G is None:
                return []
            
            references = []
            file_base = file_name.rsplit('.', 1)[0] if '.' in file_name else file_name
            
            for source, target, data in G.edges(data=True):
                rel_type = data.get("relationship_type") or data.get("type")
                if rel_type == "cross_reference":
                    metadata = data.get("metadata", {})
                    if metadata.get("referenced_file") == file_name or file_base in metadata.get("referenced_file", ""):
                        references.append((source, target))
            
            logger.debug(
                "Found cross-references",
                file_name=file_name,
                count=len(references),
            )
            
            return references
            
        except Exception as e:
            logger.error("find_cross_references failed", error=str(e), exc_info=True)
            return []

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

            G, chunk_map = await self._load_graph(folder_id)
            if G is None or not chunk_map:
                logger.warning("No knowledge graph found for folder", folder_id=folder_id)
                return []
            
            # Find starting nodes by keyword matching
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
                related_node_ids.add(node_id)
                if G.has_node(node_id):
                    related_node_ids.update(G.successors(node_id))
                    related_node_ids.update(G.predecessors(node_id))
            
            # Create DocumentChunk objects
            from app.models.documents import DocumentChunk
            
            related_chunks = []
            for node_id in related_node_ids:
                if node_id in chunk_map:
                    node_data = chunk_map[node_id]
                    try:
                        chunk = DocumentChunk(
                            chunk_id=node_data.get("chunk_id", node_id),
                            file_id=node_data.get("file_id", ""),
                            content=node_data.get("content", ""),
                            metadata=node_data.get("metadata", {}),
                            embedding=None,
                        )
                        if chunk.content:
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

