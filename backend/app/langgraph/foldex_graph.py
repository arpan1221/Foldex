"""
LangGraph pipeline for stateful multi-document analysis.

This module implements a multi-step reasoning workflow that:
1. Retrieves relevant chunks from multiple documents
2. Extracts structured entities from each document
3. Synthesizes findings across all documents
4. Generates granular citations

State is maintained throughout the workflow for observability.
LangSmith tracing is enabled for full pipeline visibility.
"""

from typing import TypedDict, List, Dict, Any, Optional
import json
import structlog
import asyncio

try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None
    END = None

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

from app.monitoring.langsmith_monitoring import get_langsmith_monitor
from app.core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)


class FoldexState(TypedDict, total=False):
    """State schema for the Foldex multi-document analysis workflow."""

    # Input
    query: str
    folder_id: Optional[str]

    # Retrieval results
    retrieved_chunks: List[Document]
    document_map: Dict[str, List[Document]]  # file_name -> chunks

    # Entity extraction
    entities_per_file: Dict[str, Dict[str, Any]]

    # Synthesis
    synthesis: str
    synthesis_raw: Optional[str]  # Raw synthesis before citation parsing
    chunk_map: Optional[Dict[str, Dict[str, Any]]]  # chunk_id -> metadata
    used_citations: Optional[List[Dict[str, Any]]]  # Citations actually used in synthesis

    # Citations
    citations: List[Dict[str, Any]]

    # Error handling
    error: Optional[str]

    # Metadata
    unique_files: List[str]
    total_chunks: int


class FoldexGraph:
    """LangGraph workflow for multi-document analysis."""

    def __init__(self, retriever, llm):
        """Initialize the Foldex graph.

        Args:
            retriever: Retriever instance (AdaptiveRetriever)
            llm: LLM instance (OllamaLLM)
        """
        self.retriever = retriever
        self.llm = llm
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the LangGraph workflow.

        Returns:
            Compiled StateGraph
        """
        if not LANGGRAPH_AVAILABLE:
            raise ProcessingError(
                "LangGraph is not installed. Install with: pip install langgraph"
            )

        workflow = StateGraph(FoldexState)

        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("extract_entities", self._extract_entities_node)
        workflow.add_node("synthesize", self._synthesis_node)
        workflow.add_node("generate_citations", self._citation_node)

        # Add edges (linear flow for now)
        workflow.add_edge("retrieve", "extract_entities")
        workflow.add_edge("extract_entities", "synthesize")
        workflow.add_edge("synthesize", "generate_citations")
        workflow.add_edge("generate_citations", END)

        # Set entry point
        workflow.set_entry_point("retrieve")

        return workflow.compile()

    def _retrieve_node(self, state: FoldexState) -> Dict[str, Any]:
        """Node 1: Retrieve relevant chunks and group by document.

        Args:
            state: Current workflow state

        Returns:
            Updated state with retrieved chunks and document map
        """
        try:
            logger.info("Retrieve node started", query=state["query"][:100])

            # Retrieve chunks using the retriever (support both sync and async)
            if hasattr(self.retriever, "aget_relevant_documents"):
                # Async retriever - run in event loop
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # If loop is running, we need to use a different approach
                        # Create a task or use run_until_complete in a thread
                        import concurrent.futures
                        with concurrent.futures.ThreadPoolExecutor() as executor:
                            future = executor.submit(
                                asyncio.run,
                                self.retriever.aget_relevant_documents(state["query"])
                            )
                            chunks = future.result()
                    else:
                        chunks = loop.run_until_complete(
                            self.retriever.aget_relevant_documents(state["query"])
                        )
                except RuntimeError:
                    # No event loop, create new one
                    chunks = asyncio.run(
                        self.retriever.aget_relevant_documents(state["query"])
                    )
            else:
                # Sync retriever
                chunks = self.retriever.get_relevant_documents(state["query"])

            # Group chunks by document
            doc_map: Dict[str, List[Document]] = {}
            for chunk in chunks:
                metadata = chunk.metadata if hasattr(chunk, "metadata") else {}
                file_name = metadata.get("file_name", "Unknown")

                if file_name not in doc_map:
                    doc_map[file_name] = []
                doc_map[file_name].append(chunk)

            unique_files = list(doc_map.keys())

            logger.info(
                "Retrieve node completed",
                total_chunks=len(chunks),
                unique_files=len(unique_files),
                files=unique_files,
            )

            return {
                "retrieved_chunks": chunks,
                "document_map": doc_map,
                "unique_files": unique_files,
                "total_chunks": len(chunks),
            }

        except Exception as e:
            logger.error("Retrieve node failed", error=str(e), exc_info=True)
            return {"error": f"Retrieval failed: {str(e)}"}

    def _extract_entities_node(self, state: FoldexState) -> Dict[str, Any]:
        """Node 2: Extract structured entities from each document.

        Args:
            state: Current workflow state

        Returns:
            Updated state with entities per file
        """
        try:
            logger.info("Extract entities node started", num_files=len(state["document_map"]))

            entities_per_file = {}

            for file_name, chunks in state["document_map"].items():
                # Combine chunks for this file (limit to first 3 chunks for speed)
                combined_chunks = chunks[:3]
                combined_text = "\n\n".join([
                    chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
                    for chunk in combined_chunks
                ])

                # Limit context length
                combined_text = combined_text[:4000]

                # Entity extraction prompt
                entity_prompt = f"""You are analyzing a document titled: "{file_name}"

Extract the following structured entities from this text:
1. **Authors**: People who wrote this (if mentioned)
2. **Key Methods/Algorithms**: Specific algorithms or methods used (e.g., DDPG, A2C, PPO, MDP)
3. **Datasets**: Datasets mentioned or used
4. **Main Objective**: Brief description of the research goal or purpose
5. **Technical Frameworks**: Frameworks or theoretical models used (e.g., Markov Decision Process, Actor-Critic, Neural Networks)
6. **Application Domain**: What field this applies to (e.g., traffic control, stock trading, healthcare)

Text excerpt:
{combined_text}

Respond ONLY with valid JSON (no markdown, no code blocks):
{{
    "authors": ["name1", "name2"],
    "algorithms": ["algo1", "algo2"],
    "datasets": ["dataset1"],
    "objective": "brief description",
    "frameworks": ["framework1"],
    "domain": "application domain"
}}"""

                # Extract entities using LLM
                try:
                    response = self.llm.get_llm().invoke(entity_prompt)

                    # Extract content from response
                    if hasattr(response, "content"):
                        content = response.content
                    else:
                        content = str(response)

                    # Clean up response (remove markdown code blocks if present)
                    content = content.strip()
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.startswith("```"):
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()

                    # Parse JSON
                    entities = json.loads(content)
                    entities_per_file[file_name] = entities

                    logger.debug(
                        "Entities extracted",
                        file_name=file_name,
                        entities=entities,
                    )

                except json.JSONDecodeError as e:
                    logger.warning(
                        "Failed to parse entities JSON",
                        file_name=file_name,
                        error=str(e),
                        content=content[:200] if 'content' in locals() else "N/A",
                    )
                    entities_per_file[file_name] = {
                        "error": "Failed to parse entities",
                        "raw_response": content[:500] if 'content' in locals() else "N/A"
                    }
                except Exception as e:
                    logger.warning(
                        "Entity extraction failed",
                        file_name=file_name,
                        error=str(e),
                    )
                    entities_per_file[file_name] = {"error": str(e)}

            logger.info(
                "Extract entities node completed",
                num_files=len(entities_per_file),
            )

            return {"entities_per_file": entities_per_file}

        except Exception as e:
            logger.error("Extract entities node failed", error=str(e), exc_info=True)
            return {"error": f"Entity extraction failed: {str(e)}"}

    def _synthesis_node(self, state: FoldexState) -> Dict[str, Any]:
        """Node 3: Cross-document synthesis with inline citations.

        Args:
            state: Current workflow state

        Returns:
            Updated state with synthesis and chunk_map
        """
        try:
            logger.info("Synthesis node started", num_files=len(state["document_map"]))

            # Check if we have an error or no documents
            if state.get("error"):
                return {
                    "synthesis": f"I couldn't find any documents to analyze. {state['error']}",
                    "chunk_map": {},
                }
            
            if len(state["document_map"]) == 0:
                return {
                    "synthesis": (
                        "I couldn't find any documents in this folder to analyze. "
                        "Please ensure the folder has been processed and contains documents. "
                        "You may need to process the folder first using the folder processing endpoint."
                    ),
                    "chunk_map": {},
                }

            # Build chunk_id to metadata mapping
            chunk_map = {}
            chunk_references = []
            
            for chunk in state["retrieved_chunks"]:
                chunk_id = None
                if hasattr(chunk, "metadata"):
                    chunk_id = chunk.metadata.get("chunk_id")
                elif isinstance(chunk, dict):
                    chunk_id = chunk.get("chunk_id") or chunk.get("id")
                
                if not chunk_id:
                    # Generate a temporary ID if missing
                    import hashlib
                    content = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
                    chunk_id = hashlib.md5(content[:100].encode()).hexdigest()[:12]
                
                # Extract metadata
                if hasattr(chunk, "metadata"):
                    meta = chunk.metadata
                    content = chunk.page_content if hasattr(chunk, "page_content") else str(chunk)
                elif isinstance(chunk, dict):
                    meta = chunk.get("metadata", {})
                    content = chunk.get("page_content") or chunk.get("content", "")
                else:
                    meta = {}
                    content = str(chunk)
                
                chunk_map[chunk_id] = {
                    "file_id": meta.get("file_id", ""),  # REQUIRED for citations
                    "file_name": meta.get("file_name", "Unknown"),
                    "page_number": meta.get("page_number"),
                    "mime_type": meta.get("mime_type"),  # For file type detection
                    "url": meta.get("drive_url") or meta.get("google_drive_url") or meta.get("url"),
                    "drive_url": meta.get("drive_url"),  # Explicit drive URL
                    "google_drive_url": meta.get("google_drive_url"),  # Explicit Google Drive URL
                    "section": meta.get("section", ""),
                    "content_preview": content[:150],
                    "start_time": meta.get("start_time"),  # For audio/video citations
                    "end_time": meta.get("end_time"),  # For audio/video citations
                }
                
                # Build chunk reference for prompt
                file_name = meta.get("file_name", "Unknown")
                page = meta.get("page_number", "N/A")
                chunk_references.append(
                    f"- [cid:{chunk_id}]: {file_name} (p.{page}) - {content[:100]}..."
                )

            # Store chunk_map in state for later use
            state["chunk_map"] = chunk_map

            # Build file list
            file_list = "\n".join([f"- {f}" for f in state["document_map"].keys()])

            # Build entities summary
            entities_json = json.dumps(state["entities_per_file"], indent=2)

            # Build chunk references text
            chunk_refs_text = "\n".join(chunk_references[:20])  # Limit to first 20 chunks

            # Get folder context for enhanced understanding
            folder_context = self._get_folder_context(state.get("folder_id"))

            # Enhanced synthesis prompt with citation instructions
            synthesis_prompt = f"""You are analyzing a folder with {len(state["document_map"])} documents.

{folder_context}

CRITICAL: You MUST discuss ALL files below. Do not focus on only one file.

Files analyzed:
{file_list}

Extracted entities per file:
{entities_json}

Available chunks with IDs (use these for citations):
{chunk_refs_text}

User question: {state["query"]}

TASK: Identify what is COMMON across ALL files and what is UNIQUE to each file.

INSTRUCTIONS:
1. Look for shared authors, algorithms, or frameworks across documents
2. Identify common methodologies or research approaches
3. Note overlapping domains or objectives
4. Highlight unique contributions of each file
5. For EVERY factual claim, add the supporting chunk ID in brackets immediately after the claim

CRITICAL CITATION FORMAT:
- Use inline citations with format: [cid:chunk_id_here]
- Place the citation marker IMMEDIATELY after the claim it supports
- Multiple claims can reference the same chunk ID
- Every specific fact, method, or finding needs a citation

Example:
"Both papers use Deep Reinforcement Learning[cid:abc123] with Actor-Critic methods[cid:def456] including DDPG[cid:abc123] and A2C[cid:ghi789]."

Now provide your comprehensive synthesis with inline citations for every claim:"""

            # Generate synthesis
            response = self.llm.get_llm().invoke(synthesis_prompt)

            # Extract content
            if hasattr(response, "content"):
                synthesis = response.content
            else:
                synthesis = str(response)

            synthesis = synthesis.strip()

            # Parse inline citations and convert to HTML
            from app.utils.citation_parser import parse_inline_citations
            
            parsed_synthesis, used_citations = parse_inline_citations(
                synthesis,
                chunk_map
            )

            logger.info(
                "Synthesis node completed",
                synthesis_length=len(synthesis),
                parsed_length=len(parsed_synthesis),
                citation_count=len(used_citations),
            )

            return {
                "synthesis": parsed_synthesis,  # HTML with inline citations
                "synthesis_raw": synthesis,  # Keep raw for debugging
                "chunk_map": chunk_map,
                "used_citations": used_citations,
            }

        except Exception as e:
            logger.error("Synthesis node failed", error=str(e), exc_info=True)
            return {"error": f"Synthesis failed: {str(e)}"}

    def _citation_node(self, state: FoldexState) -> Dict[str, Any]:
        """Node 4: Generate structured citations.

        Args:
            state: Current workflow state

        Returns:
            Updated state with citations
        """
        try:
            logger.info("Citation node started", num_chunks=len(state["retrieved_chunks"]))

            # Use inline citations if available, otherwise generate from all chunks
            used_citations = state.get("used_citations")
            if used_citations:
                citations = used_citations
                logger.info(
                    "Using inline citations",
                    num_citations=len(used_citations),
                )
            else:
                # Fallback: generate citations from all retrieved chunks
                citations = []
                citation_num = 1

                # Group by file and create citations
                for file_name in state["document_map"].keys():
                    chunks = state["document_map"][file_name]

                    # Get first chunk metadata for this file
                    if chunks:
                        first_chunk = chunks[0]
                        metadata = first_chunk.metadata if hasattr(first_chunk, "metadata") else {}

                        citation = {
                            "citation_number": citation_num,
                            "file_name": file_name,
                            "file_id": metadata.get("file_id"),
                            "page_number": metadata.get("page_number"),
                            "page_display": f"p.{metadata.get('page_number')}" if metadata.get("page_number") else None,
                            "chunk_id": metadata.get("chunk_id"),
                            "google_drive_url": metadata.get("google_drive_url") or self._get_drive_url(metadata),
                            "content_preview": (
                                first_chunk.page_content[:200]
                                if hasattr(first_chunk, "page_content")
                                else str(first_chunk)[:200]
                            ),
                            "total_chunks_from_file": len(chunks),
                        }

                        citations.append(citation)
                        citation_num += 1

            logger.info(
                "Citation node completed",
                num_citations=len(citations) if citations else 0,
            )

            return {"citations": citations}

        except Exception as e:
            logger.error("Citation node failed", error=str(e), exc_info=True)
            return {"error": f"Citation generation failed: {str(e)}"}

    def _get_drive_url(self, metadata: Dict[str, Any]) -> Optional[str]:
        """Construct Google Drive URL from file_id.

        Args:
            metadata: Document metadata

        Returns:
            Google Drive URL or None
        """
        file_id = metadata.get("file_id")
        if file_id:
            return f"https://drive.google.com/file/d/{file_id}/view"
        return None

    def _get_folder_context(self, folder_id: Optional[str]) -> str:
        """Get folder context summary for injection into prompts.

        Args:
            folder_id: Folder ID

        Returns:
            Formatted folder context string
        """
        if not folder_id:
            return ""

        try:
            from app.database.sqlite_manager import SQLiteManager

            db = SQLiteManager()

            # Get folder summary (sync wrapper for async method)
            summary_data = asyncio.run(db.get_folder_summary(folder_id))

            if not summary_data or not summary_data.get("summary"):
                return ""

            # Format folder context for prompt injection
            context_parts = []

            # Master summary
            context_parts.append("FOLDER OVERVIEW:")
            context_parts.append(summary_data["summary"])
            context_parts.append("")

            # File types
            if summary_data.get("file_type_distribution"):
                types_str = ", ".join([
                    f"{count} {ftype}"
                    for ftype, count in summary_data["file_type_distribution"].items()
                ])
                context_parts.append(f"Available file types: {types_str}")

            # Capabilities
            if summary_data.get("capabilities"):
                caps = summary_data["capabilities"][:3]  # Top 3 capabilities
                context_parts.append("This folder can help you:")
                for cap in caps:
                    context_parts.append(f"  - {cap}")
                context_parts.append("")

            # Top themes
            if summary_data.get("insights", {}).get("top_themes"):
                themes = summary_data["insights"]["top_themes"][:3]
                context_parts.append(f"Main themes: {', '.join(themes)}")
                context_parts.append("")

            return "\n".join(context_parts)

        except Exception as e:
            logger.warning("Failed to get folder context", folder_id=folder_id, error=str(e))
            return ""

    def invoke(
        self, 
        query: str, 
        folder_id: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Execute the graph workflow.

        Args:
            query: User query
            folder_id: Optional folder ID
            config: Optional LangGraph configuration (for LangSmith tracing)

        Returns:
            Final state with synthesis and citations
        """
        try:
            logger.info("Graph invocation started", query=query[:100], folder_id=folder_id)

            # Initialize state
            initial_state: FoldexState = {
                "query": query,
                "folder_id": folder_id,
                "retrieved_chunks": [],
                "document_map": {},
                "entities_per_file": {},
                "synthesis": "",
                "citations": [],
                "error": None,
                "unique_files": [],
                "total_chunks": 0,
            }

            # Get LangSmith config if not provided
            if config is None:
                langsmith_monitor = get_langsmith_monitor()
                config = langsmith_monitor.get_langgraph_config(
                    metadata={
                        "query": query[:200],
                        "folder_id": folder_id,
                        "workflow": "foldex_multi_document_synthesis"
                    }
                )

            # Execute graph with LangSmith tracing
            final_state = self.graph.invoke(initial_state, config=config)

            logger.info(
                "Graph invocation completed",
                has_error=bool(final_state.get("error")),
                num_files=len(final_state.get("unique_files", [])),
                synthesis_length=len(final_state.get("synthesis", "")),
                citation_count=len(final_state.get("citations", [])),
            )

            return final_state

        except Exception as e:
            logger.error("Graph invocation failed", error=str(e), exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "synthesis": "",
                "citations": [],
                "entities_per_file": {},
            }


def create_foldex_graph(retriever, llm) -> FoldexGraph:
    """Factory function to create a Foldex graph instance.

    Args:
        retriever: Retriever instance
        llm: LLM instance

    Returns:
        FoldexGraph instance
    """
    return FoldexGraph(retriever, llm)
