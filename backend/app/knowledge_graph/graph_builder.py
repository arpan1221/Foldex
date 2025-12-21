"""
Knowledge graph construction for Foldex.

Extracts entities and relationships from documents and builds a graph
for visualization and cross-document insights.
"""

from typing import List, Dict, Any, Optional
import json
import structlog

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

try:
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.schema import Document
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        Document = None

from app.core.exceptions import ProcessingError
from app.rag.llm_chains import OllamaLLM

logger = structlog.get_logger(__name__)


class FoldexKnowledgeGraph:
    """Build and query knowledge graph of folder contents."""
    
    def __init__(self, llm: Optional[Any] = None):
        """Initialize knowledge graph.
        
        Args:
            llm: Optional LLM instance (defaults to OllamaLLM)
        """
        if not NETWORKX_AVAILABLE:
            raise ProcessingError(
                "NetworkX is not installed. Install with: pip install networkx"
            )
        
        self.graph = nx.DiGraph()  # Directed graph
        self.llm = llm or OllamaLLM()
        self._entity_cache: Dict[str, Dict] = {}  # Cache entity extractions
    
    def build_from_documents(self, chunks: List[Document]) -> nx.DiGraph:
        """Extract entities and relationships from documents.
        
        Args:
            chunks: List of document chunks
            
        Returns:
            NetworkX graph with entities and relationships
        """
        if not chunks:
            logger.warning("No chunks provided for graph construction")
            return self.graph
        
        logger.info("Building knowledge graph", chunk_count=len(chunks))
        
        # Group chunks by document
        docs: Dict[str, List[str]] = {}
        for chunk in chunks:
            if hasattr(chunk, "metadata"):
                file_name = chunk.metadata.get("file_name", "unknown")
            elif isinstance(chunk, dict):
                file_name = chunk.get("file_name", "unknown")
            else:
                file_name = "unknown"
            
            if file_name not in docs:
                docs[file_name] = []
            
            content = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
            docs[file_name].append(content)
        
        logger.info("Grouped chunks by document", document_count=len(docs))
        
        # Extract entities and relationships per document
        for file_name, content_list in docs.items():
            try:
                # Use first 5 chunks to avoid token limits
                combined_content = "\n\n".join(content_list[:5])
                
                entities_and_rels = self._extract_graph_data(
                    file_name,
                    combined_content
                )
                
                self._add_to_graph(file_name, entities_and_rels)
                
                logger.debug(
                    "Processed document",
                    file_name=file_name,
                    entity_count=len(entities_and_rels.get("entities", [])),
                    relationship_count=len(entities_and_rels.get("relationships", [])),
                )
            except Exception as e:
                logger.warning(
                    "Failed to process document for graph",
                    file_name=file_name,
                    error=str(e),
                )
                continue
        
        logger.info(
            "Knowledge graph built",
            node_count=self.graph.number_of_nodes(),
            edge_count=self.graph.number_of_edges(),
        )
        
        return self.graph
    
    def _extract_graph_data(self, file_name: str, content: str) -> Dict[str, Any]:
        """Use LLM to extract entities and relationships.
        
        Args:
            file_name: Name of the document
            content: Document content
            
        Returns:
            Dictionary with entities and relationships
        """
        # Check cache first
        cache_key = f"{file_name}:{hash(content[:500])}"
        if cache_key in self._entity_cache:
            logger.debug("Using cached entity extraction", file_name=file_name)
            return self._entity_cache[cache_key]
        
        # Limit content length
        content_preview = content[:2000] if len(content) > 2000 else content
        
        prompt = f"""Analyze this document and extract entities and relationships.

Document: {file_name}
Content: {content_preview}

Extract:
1. People (authors, researchers, mentioned individuals)
2. Algorithms/Methods (DDPG, A2C, PPO, etc.)
3. Concepts (Deep Reinforcement Learning, MDP, Actor-Critic, etc.)
4. Datasets used
5. Relationships between entities

Output ONLY valid JSON (no markdown, no code blocks):
{{
    "entities": [
        {{"type": "person", "name": "Arpan Nookala"}},
        {{"type": "algorithm", "name": "DDPG"}},
        {{"type": "concept", "name": "Deep Reinforcement Learning"}},
        {{"type": "dataset", "name": "Atari"}}
    ],
    "relationships": [
        {{"source": "Arpan Nookala", "target": "{file_name}", "relation": "authored"}},
        {{"source": "{file_name}", "target": "DDPG", "relation": "uses"}},
        {{"source": "DDPG", "target": "Deep Reinforcement Learning", "relation": "type_of"}},
        {{"source": "{file_name}", "target": "Atari", "relation": "uses_dataset"}}
    ]
}}

JSON only:"""
        
        try:
            response = self.llm.get_llm().invoke(prompt)
            
            # Extract content from response
            if hasattr(response, "content"):
                content_str = response.content
            else:
                content_str = str(response)
            
            # Clean up response (remove markdown code blocks if present)
            content_str = content_str.strip()
            if content_str.startswith("```json"):
                content_str = content_str[7:]
            if content_str.startswith("```"):
                content_str = content_str[3:]
            if content_str.endswith("```"):
                content_str = content_str[:-3]
            content_str = content_str.strip()
            
            # Parse JSON
            graph_data = json.loads(content_str)
            
            # Validate structure
            if not isinstance(graph_data, dict):
                raise ValueError("Response is not a dictionary")
            
            if "entities" not in graph_data:
                graph_data["entities"] = []
            if "relationships" not in graph_data:
                graph_data["relationships"] = []
            
            # Cache result
            self._entity_cache[cache_key] = graph_data
            
            return graph_data
            
        except json.JSONDecodeError as e:
            logger.warning(
                "Failed to parse entity extraction JSON",
                file_name=file_name,
                error=str(e),
                response_preview=content_str[:200] if 'content_str' in locals() else "N/A",
            )
            return {"entities": [], "relationships": []}
        except Exception as e:
            logger.warning(
                "Entity extraction failed",
                file_name=file_name,
                error=str(e),
            )
            return {"entities": [], "relationships": []}
    
    def _add_to_graph(self, file_name: str, graph_data: Dict[str, Any]):
        """Add entities and relationships to NetworkX graph.
        
        Args:
            file_name: Name of the document
            graph_data: Dictionary with entities and relationships
        """
        # Add file node
        self.graph.add_node(
            file_name,
            type="document",
            label=file_name,
            node_type="document",
        )
        
        # Add entities
        for entity in graph_data.get("entities", []):
            entity_name = entity.get("name", "").strip()
            entity_type = entity.get("type", "unknown")
            
            if not entity_name:
                continue
            
            # Add or update node
            if self.graph.has_node(entity_name):
                # Update existing node
                self.graph.nodes[entity_name]["type"] = entity_type
            else:
                # Add new node
                self.graph.add_node(
                    entity_name,
                    type=entity_type,
                    label=entity_name,
                    node_type="entity",
                )
        
        # Add relationships
        for rel in graph_data.get("relationships", []):
            source = rel.get("source", "").strip()
            target = rel.get("target", "").strip()
            relation = rel.get("relation", "related")
            
            if not source or not target:
                continue
            
            # Ensure nodes exist
            if not self.graph.has_node(source):
                self.graph.add_node(
                    source,
                    type="unknown",
                    label=source,
                    node_type="entity",
                )
            
            if not self.graph.has_node(target):
                self.graph.add_node(
                    target,
                    type="unknown",
                    label=target,
                    node_type="entity",
                )
            
            # Add edge
            self.graph.add_edge(
                source,
                target,
                relation=relation,
            )
    
    def find_common_entities(self) -> List[Dict[str, Any]]:
        """Find entities connected to multiple documents.
        
        Returns:
            List of entities with their connected documents
        """
        results: List[Dict[str, Any]] = []
        
        # Find all document nodes
        doc_nodes = [
            n for n, d in self.graph.nodes(data=True)
            if d.get("node_type") == "document"
        ]
        
        if len(doc_nodes) < 2:
            logger.debug("Not enough documents for common entity analysis")
            return results
        
        # For each non-document node, check if connected to multiple docs
        for node in self.graph.nodes():
            if node in doc_nodes:
                continue
            
            # Find which documents reference this entity
            connected_docs = []
            for doc in doc_nodes:
                # Check both directions (direct edges)
                if self.graph.has_edge(doc, node) or self.graph.has_edge(node, doc):
                    connected_docs.append(doc)
                # Also check if there's a path through other nodes (indirect connection)
                else:
                    try:
                        if nx.has_path(self.graph, doc, node) or nx.has_path(self.graph, node, doc):
                            connected_docs.append(doc)
                    except Exception:
                        # If path check fails, skip
                        pass
            
            if len(connected_docs) > 1:
                node_data = self.graph.nodes[node]
                results.append({
                    "entity": node,
                    "type": node_data.get("type", "unknown"),
                    "label": node_data.get("label", node),
                    "documents": connected_docs,
                    "document_count": len(connected_docs),
                })
        
        # Sort by document count (most shared first)
        results.sort(key=lambda x: x["document_count"], reverse=True)
        
        logger.info(
            "Found common entities",
            count=len(results),
        )
        
        return results  # type: ignore
    
    def to_json(self) -> Dict[str, Any]:
        """Export graph as JSON for D3.js visualization.
        
        Returns:
            Dictionary with nodes and links
        """
        nodes = []
        for node, data in self.graph.nodes(data=True):
            nodes.append({
                "id": node,
                "label": data.get("label", node),
                "type": data.get("type", "unknown"),
                "node_type": data.get("node_type", "entity"),
            })
        
        links = []
        for source, target, data in self.graph.edges(data=True):
            links.append({
                "source": source,
                "target": target,
                "relation": data.get("relation", "related"),
            })
        
        return {
            "nodes": nodes,
            "links": links,
            "stats": {
                "node_count": len(nodes),
                "link_count": len(links),
                "document_count": len([
                    n for n in nodes if n.get("node_type") == "document"
                ]),
            },
        }
    
    def get_entity_subgraph(self, entity_name: str, depth: int = 2) -> Dict[str, Any]:
        """Get subgraph centered on a specific entity.
        
        Args:
            entity_name: Name of the entity
            depth: Maximum distance from entity
            
        Returns:
            Subgraph as JSON
        """
        if not self.graph.has_node(entity_name):
            return {"nodes": [], "links": []}
        
        # Get neighbors within depth
        subgraph_nodes = {entity_name}
        
        # BFS to find nodes within depth
        current_level = {entity_name}
        for _ in range(depth):
            next_level = set()
            for node in current_level:
                # Add neighbors
                next_level.update(self.graph.successors(node))
                next_level.update(self.graph.predecessors(node))
            subgraph_nodes.update(next_level)
            current_level = next_level
        
        # Create subgraph
        subgraph = self.graph.subgraph(subgraph_nodes)
        
        # Convert to JSON
        nodes = []
        for node in subgraph.nodes():
            data = subgraph.nodes[node]
            nodes.append({
                "id": node,
                "label": data.get("label", node),
                "type": data.get("type", "unknown"),
                "node_type": data.get("node_type", "entity"),
            })
        
        links = []
        for source, target, data in subgraph.edges(data=True):
            links.append({
                "source": source,
                "target": target,
                "relation": data.get("relation", "related"),
            })
        
        return {"nodes": nodes, "links": links}


class GraphBuilder:
    """Async wrapper for building knowledge graphs from DocumentChunks and relationships.
    
    This class provides the interface expected by services that need to build
    graphs from DocumentChunk objects and relationship data.
    """
    
    def __init__(self, llm: Optional[Any] = None):
        """Initialize graph builder.
        
        Args:
            llm: Optional LLM instance (defaults to OllamaLLM)
        """
        self.kg = FoldexKnowledgeGraph(llm=llm)
    
    async def build_graph(
        self,
        chunks: List[Any],
        relationships: Optional[List[Dict[str, Any]]] = None,
    ) -> Any:
        """Build knowledge graph from chunks and relationships.
        
        Args:
            chunks: List of DocumentChunk objects or LangChain Documents
            relationships: Optional list of relationship dictionaries
            
        Returns:
            NetworkX graph
        """
        # Simple Document-like class for fallback
        class SimpleDoc:
            def __init__(self, content, metadata):
                self.page_content = content
                self.metadata = metadata
        
        # Convert DocumentChunk to LangChain Document if needed
        langchain_docs = []
        for chunk in chunks:
            if hasattr(chunk, "page_content"):
                # Already a LangChain Document
                langchain_docs.append(chunk)
            elif hasattr(chunk, "content"):
                # DocumentChunk - convert to LangChain Document
                if LANGCHAIN_AVAILABLE:
                    from app.models.documents import document_chunk_to_langchain
                    langchain_docs.append(document_chunk_to_langchain(chunk))
                else:
                    # Fallback: create a simple Document-like object
                    langchain_docs.append(SimpleDoc(
                        content=chunk.content,
                        metadata={
                            "chunk_id": getattr(chunk, "chunk_id", ""),
                            "file_id": getattr(chunk, "file_id", ""),
                            "file_name": getattr(chunk, "metadata", {}).get("file_name", "unknown"),
                            **getattr(chunk, "metadata", {}),
                        }
                    ))
            else:
                # Unknown type, try to convert to string
                logger.warning("Unknown chunk type, converting to string", chunk_type=type(chunk))
                langchain_docs.append(SimpleDoc(
                    content=str(chunk),
                    metadata={"file_name": "unknown"}
                ))
        
        # Build graph using FoldexKnowledgeGraph
        graph = self.kg.build_from_documents(langchain_docs)
        
        # Add relationships if provided
        if relationships:
            for rel in relationships:
                source = rel.get("source", "").strip()
                target = rel.get("target", "").strip()
                relation = rel.get("relation", "related")
                
                if source and target:
                    # Ensure nodes exist
                    if not graph.has_node(source):
                        graph.add_node(
                            source,
                            type="entity",
                            label=source,
                            node_type="entity",
                        )
                    
                    if not graph.has_node(target):
                        graph.add_node(
                            target,
                            type="entity",
                            label=target,
                            node_type="entity",
                        )
                    
                    # Add edge
                    graph.add_edge(source, target, relation=relation)
        
        return graph
