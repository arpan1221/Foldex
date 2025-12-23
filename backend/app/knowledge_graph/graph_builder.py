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
from app.knowledge_graph.entity_extractor import EntityExtractor
from app.services.document_summarizer import get_document_summarizer

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
        self.entity_extractor = EntityExtractor()  # For NER fallback
        self.document_summarizer = get_document_summarizer()  # For document context
    
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
                # Select best chunks for extraction (intelligent selection, not just first N)
                # Score chunks by informativeness and select top ones up to 4000 chars
                scored_chunks = []
                for i, chunk in enumerate(content_list):
                    score = 0
                    # Entity density (capitalized words, proper nouns)
                    capitalized = len(re.findall(r'\b[A-Z][a-z]{2,}\b', chunk))
                    all_caps = len(re.findall(r'\b[A-Z]{2,}\b', chunk))
                    score += int((capitalized + all_caps) * 0.3)
                    # Vocabulary diversity
                    words = chunk.lower().split()
                    unique_words = len(set(words))
                    total_words = len(words)
                    if total_words > 0:
                        diversity = unique_words / total_words
                        score += int(diversity * 10.0)
                    # Content density (alphabetic vs symbols)
                    alpha_chars = sum(1 for c in chunk if c.isalpha())
                    total_chars = len(chunk.strip())
                    if total_chars > 0:
                        density = alpha_chars / total_chars
                        if density > 0.6:
                            score += int((density - 0.6) * 20)
                    scored_chunks.append((score, i, chunk))
                
                # Sort by score and select top chunks up to 4000 chars
                scored_chunks.sort(reverse=True, key=lambda x: x[0])
                selected_chunks = []
                total_length = 0
                for score, idx, chunk in scored_chunks:
                    if total_length + len(chunk) <= 4000:
                        selected_chunks.append((idx, chunk))
                        total_length += len(chunk)
                    elif total_length < 2800:  # If under 70% of target, add one more
                        selected_chunks.append((idx, chunk))
                        total_length += len(chunk)
                    else:
                        break
                
                # Sort by original order and combine
                selected_chunks.sort(key=lambda x: x[0])
                combined_content = "\n\n".join(chunk for _, chunk in selected_chunks)
                
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
    
    async def _extract_graph_data(
        self, 
        file_name: str, 
        content: str,
        document_summary: Optional[str] = None
    ) -> Dict[str, Any]:
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
        
        # Improved prompt with better LLM refusal handling and domain adaptability
        summary_context = ""
        if document_summary:
            summary_context = f"\n\nDocument Summary (for context): {document_summary}\n"
        
        prompt = f"""You are an information extraction system. Extract entities and relationships from this document.

Document: {file_name}{summary_context}
Content: {content_preview}

TASK: Extract ALL entities and relationships mentioned in the content. This is a factual extraction task - extract what is actually stated, regardless of document type. Adapt entity types to match the document's domain.

Entity Types (adapt to document type):
- person: People mentioned (names of individuals, authors, creators, participants, etc.)
- method: Methods, techniques, processes, approaches (adapt to domain: algorithms, procedures, workflows, methodologies, etc.)
- concept: Ideas, theories, principles, topics, themes, concepts discussed
- tool: Tools, software, platforms, datasets, resources, technologies used or mentioned
- organization: Companies, institutions, universities, agencies, groups, organizations
- location: Places, locations, addresses (if relevant to document type)
- event: Events, meetings, occurrences (if relevant to document type)

Relationship Types (adapt to context):
- authored/created/wrote: person -> document (who created/wrote it)
- discusses/mentions/contains: document -> entity (what the document is about)
- uses/employs/implements: document/entity -> tool/method (what is used)
- related_to/associated_with: entity -> entity (how entities relate)
- works_with/collaborates_with: person -> person
- located_in/part_of: entity -> location/organization
- occurs_at/happens_at: event -> location/time
- Any other relevant relationships based on the content

IMPORTANT:
- Adapt entity and relationship extraction to the document's domain (research papers, legal docs, medical records, code, audio transcripts, etc.)
- Extract ALL entities you can identify from the content
- Extract relationships between entities based on what is stated in the content
- This is NOT a conversation - you are performing factual extraction from the provided content

Output ONLY valid JSON (no markdown, no code blocks, no explanations):
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

Output JSON now:"""
        
        try:
            # Use async invoke to avoid blocking the event loop
            response = await asyncio.wait_for(
                self.llm.get_llm().ainvoke(prompt),
                timeout=120.0  # Extended timeout (2 minutes) for entity extraction
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
                "i can't help", "i cannot help", "i cannot", "i can not",
                "i'm unable", "i am unable", "there is no", "no information",
                "i don't have", "i do not have", "i can help you", "i notice"
            ]
            content_lower = content_str.lower() if content_str else ""
            is_refusal = any(pattern in content_lower for pattern in refusal_patterns) if content_str else True
            
            if not content_str or is_refusal:
                logger.warning(
                    "LLM refused or indicated no entities/relationships",
                    file_name=file_name,
                    response_preview=content_str[:200] if content_str else "empty response"
                )
                # Retry with simpler prompt and different approach
                return await self._retry_extraction_with_fallback(file_name, content_preview)
            
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
    
    async def _retry_extraction_with_fallback(
        self, 
        file_name: str, 
        content: str,
        attempt: int = 0
    ) -> Dict[str, Any]:
        """Retry extraction with simpler prompt and fallback strategies.
        
        Args:
            file_name: File name
            content: Document content
            attempt: Retry attempt number
            
        Returns:
            Dictionary with entities and relationships
        """
        if attempt >= 2:  # Max 3 attempts (0, 1, 2)
            logger.debug("Max retry attempts reached, returning empty", file_name=file_name)
            return {"entities": [], "relationships": []}
        
        # Simpler prompt for retry (completely domain-agnostic)
        if attempt == 0:
            retry_prompt = f"""Extract entities from this document as JSON. This is factual extraction from the provided content.

Document: {file_name}
Content: {content[:3000]}

Extract entities (person, method, concept, tool, organization) and relationships (authored, discusses, uses, related_to). Adapt types to the document's domain.

Output JSON only: {{"entities": [{{"type": "...", "name": "..."}}], "relationships": [{{"source": "...", "target": "...", "relation": "..."}}]}}"""
        else:
            # Final attempt - minimal prompt
            retry_prompt = f"""List entities from: {file_name}

{content[:2000]}

JSON: {{"entities": [{{"type": "person|method|concept|tool|organization", "name": "..."}}], "relationships": []}}"""
        
        try:
            response = await asyncio.wait_for(
                self.llm.get_llm().ainvoke(retry_prompt),
                timeout=60.0
            )
            
            content_str = response.content.strip() if hasattr(response, "content") else str(response).strip()
            
            # Clean response
            if content_str.startswith("```json"):
                content_str = content_str[7:]
            if content_str.startswith("```"):
                content_str = content_str[3:]
            if content_str.endswith("```"):
                content_str = content_str[:-3]
            content_str = content_str.strip()
            
            # Extract JSON
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', content_str, re.DOTALL)
            if json_match:
                content_str = json_match.group(0)
            
            # Check for refusals
            refusal_patterns = [
                "i can't help", "i cannot help", "i'm unable", "i am unable",
                "i don't have", "i do not have", "no information", "there is no"
            ]
            is_refusal = any(pattern in content_str.lower() for pattern in refusal_patterns) if content_str else True
            
            if is_refusal or not content_str:
                # Try next attempt
                return await self._retry_extraction_with_fallback(file_name, content, attempt + 1)
            
            # Parse JSON
            try:
                graph_data = json.loads(content_str)
                if not isinstance(graph_data, dict):
                    raise ValueError("Response is not a dictionary")
                
                if "entities" not in graph_data:
                    graph_data["entities"] = []
                if "relationships" not in graph_data:
                    graph_data["relationships"] = []
                
                logger.info("Retry extraction succeeded", file_name=file_name, attempt=attempt + 1)
                return graph_data
            except json.JSONDecodeError:
                # Try next attempt
                return await self._retry_extraction_with_fallback(file_name, content, attempt + 1)
                
        except Exception as e:
            logger.debug(f"Retry extraction failed (attempt {attempt})", file_name=file_name, error=str(e))
            if attempt < 2:
                return await self._retry_extraction_with_fallback(file_name, content, attempt + 1)
            
            # Final fallback: Use NER extraction (domain-agnostic)
            try:
                logger.debug("Using NER fallback extraction", file_name=file_name)
                ner_entities = await self.entity_extractor.extract_entities(content[:2000])
                
                # Convert NER entities to graph format
                graph_entities = []
                seen = set()
                for ent in ner_entities:
                    # Map spaCy types to graph types (generic mapping)
                    type_mapping = {
                        'PERSON': 'person',
                        'ORG': 'organization',
                        'GPE': 'organization',  # Geographic -> organization
                        'MONEY': 'concept',
                        'DATE': 'concept',
                        'EVENT': 'event',
                        'LAW': 'concept',
                        'PRODUCT': 'tool',
                        'WORK_OF_ART': 'concept',
                    }
                    
                    entity_type = type_mapping.get(ent.get('type', ''), 'concept')
                    entity_name = ent.get('text', '').strip()
                    
                    if entity_name and len(entity_name) > 2 and entity_name.lower() not in seen:
                        seen.add(entity_name.lower())
                        graph_entities.append({
                            "type": entity_type,
                            "name": entity_name,
                            "confidence": ent.get('confidence', 0.5),
                            "source": "NER"
                        })
                
                return {
                    "entities": graph_entities,
                    "relationships": []  # NER doesn't extract relationships
                }
            except Exception as ner_error:
                logger.debug("NER fallback also failed", file_name=file_name, error=str(ner_error))
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
