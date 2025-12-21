"""
Knowledge graph construction for Foldex.

Extracts entities and relationships from documents and builds a graph
for visualization and cross-document insights.
"""

from typing import List, Dict, Any, Optional
import json
import re
import asyncio
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
    
    async def build_from_documents(self, chunks: List[Document]) -> nx.DiGraph:
        """Extract entities and relationships from documents (async to avoid blocking).
        
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
        
        # Extract entities and relationships per document (async)
        for file_name, content_list in docs.items():
            try:
                # Use first 5 chunks to avoid token limits
                combined_content = "\n\n".join(content_list[:5])
                
                entities_and_rels = await self._extract_graph_data(
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
    
    async def _extract_graph_data(self, file_name: str, content: str) -> Dict[str, Any]:
        """Use LLM to extract entities and relationships (async to avoid blocking).
        
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
        
        # Clean and validate content before sending to LLM
        # Remove excessive whitespace and non-printable characters
        cleaned_content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
        cleaned_content = ''.join(char for char in cleaned_content if char.isprintable() or char.isspace())
        
        # Skip leading non-text content (common in PDFs with metadata)
        # Look for first substantial text block (at least 50 chars of readable text)
        text_start = 0
        for i in range(len(cleaned_content) - 50):
            chunk = cleaned_content[i:i+50]
            # Check if chunk has reasonable text density (at least 70% alphabetic)
            alpha_count = sum(1 for c in chunk if c.isalpha())
            if alpha_count / len(chunk) >= 0.7:
                text_start = i
                break
        
        # Get content starting from first substantial text
        content_preview = cleaned_content[text_start:text_start + 2000] if len(cleaned_content) > text_start + 2000 else cleaned_content[text_start:]
        
        # Validate we have enough meaningful content
        if not content_preview or len(content_preview.strip()) < 50:
            logger.warning(
                "Content preview too short or invalid after cleaning",
                file_name=file_name,
                original_length=len(content),
                preview_length=len(content_preview)
            )
            return {"entities": [], "relationships": []}
        
        prompt = f"""Analyze this document and extract entities and relationships.

Document: {file_name}
Content: {content_preview}

Extract relevant entities and relationships based on the document's content. The entities and relationships should be appropriate for the document type (e.g., for research papers extract authors and methods; for code files extract functions and classes; for data files extract key datasets and fields).

Extract:
1. People (names of individuals mentioned: authors, researchers, team members, etc.)
2. Methods/Techniques (algorithms, approaches, methodologies, processes mentioned)
3. Concepts/Ideas (key topics, theories, principles, domains discussed)
4. Tools/Resources (datasets, software, platforms, technologies, systems used or mentioned)
5. Organizations/Institutions (companies, universities, agencies, etc.)
6. Other relevant entities specific to this document's domain

Also extract relationships between entities, such as:
- authored, created, wrote (person -> document)
- uses, implements, applies (document/entity -> method/tool)
- contains, includes, references (document -> entity)
- type_of, instance_of, related_to (entity -> entity)
- works_with, collaborates_with (person -> person)
- Any other relevant relationships based on the content

Output ONLY valid JSON (no markdown, no code blocks):
{{
    "entities": [
        {{"type": "person", "name": "Entity Name"}},
        {{"type": "method", "name": "Entity Name"}},
        {{"type": "concept", "name": "Entity Name"}},
        {{"type": "tool", "name": "Entity Name"}}
    ],
    "relationships": [
        {{"source": "Entity Name", "target": "{file_name}", "relation": "relationship_type"}},
        {{"source": "Entity A", "target": "Entity B", "relation": "relationship_type"}}
    ]
}}

JSON only:"""
        
        try:
            # Use async invoke to avoid blocking the event loop
            response = await asyncio.wait_for(
                self.llm.get_llm().ainvoke(prompt),
                timeout=90.0  # Extended timeout (> 1 minute) for entity extraction
            )
            
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
            
            # Try to extract JSON from response if it's not pure JSON
            # Look for JSON object boundaries { ... }
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content_str, re.DOTALL)
            if json_match:
                content_str = json_match.group(0)
            
            # Check if response indicates refusal or no content
            refusal_patterns = [
                "I can't help", "I cannot help", "I cannot", "I can not",
                "I'm unable", "I am unable", "There is no", "no information",
                "I don't have", "I do not have"
            ]
            is_refusal = any(pattern.lower() in content_str.lower() for pattern in refusal_patterns) if content_str else True
            
            if not content_str or is_refusal:
                logger.warning(
                    "LLM refused or indicated no entities/relationships",
                    file_name=file_name,
                    response_preview=content_str[:200] if content_str else "empty response"
                )
                # Try fallback: extract names from filename
                fallback_entities = []
                fallback_relationships = []
                name_pattern = r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)'
                potential_names = re.findall(name_pattern, file_name.replace('_', ' ').replace('-', ' '))
                for name in potential_names[:3]:
                    if len(name.split()) >= 2:
                        fallback_entities.append({"type": "person", "name": name})
                        fallback_relationships.append({
                            "source": name,
                            "target": file_name,
                            "relation": "authored"
                        })
                if fallback_entities:
                    logger.info("Created fallback entities from filename", file_name=file_name)
                    return {"entities": fallback_entities, "relationships": fallback_relationships}
                return {"entities": [], "relationships": []}
            
            # Parse JSON
            try:
                graph_data = json.loads(content_str)
            except json.JSONDecodeError:
                # If direct parsing fails, try to find and extract JSON object
                # Look for first { and last } to extract potential JSON
                first_brace = content_str.find('{')
                last_brace = content_str.rfind('}')
                if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                    try:
                        potential_json = content_str[first_brace:last_brace + 1]
                        graph_data = json.loads(potential_json)
                    except json.JSONDecodeError:
                        # Final fallback: return empty structure
                        logger.debug(
                            "Could not extract valid JSON from response, returning empty structure",
                            file_name=file_name,
                            response_preview=content_str[:200]
                        )
                        return {"entities": [], "relationships": []}
                else:
                    # No JSON structure found, return empty
                    logger.debug(
                        "No JSON structure found in response, returning empty structure",
                        file_name=file_name,
                        response_preview=content_str[:200]
                    )
                    return {"entities": [], "relationships": []}
            
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
        except asyncio.TimeoutError:
            logger.debug(
                "Entity extraction timed out (non-fatal, continuing)",
                file_name=file_name,
                timeout_seconds=10.0,
            )
            return {"entities": [], "relationships": []}
        except Exception as e:
            # Log with exception type if error message is empty
            error_msg = str(e) if str(e) else f"{type(e).__name__} (no message)"
            logger.debug(
                "Entity extraction failed (non-fatal, continuing)",
                file_name=file_name,
                error=error_msg,
                error_type=type(e).__name__,
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
        
        # Build graph using FoldexKnowledgeGraph (async to avoid blocking)
        graph = await self.kg.build_from_documents(langchain_docs)
        
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
