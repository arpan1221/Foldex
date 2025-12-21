"""Chat service for processing user queries."""

from typing import Dict, Optional, Callable, List
import uuid
import structlog

from app.services.langgraph_orchestrator import LangGraphOrchestrator
from app.database.sqlite_manager import SQLiteManager
from app.langgraph.foldex_graph import create_foldex_graph
from app.services.rag_service import get_rag_service

logger = structlog.get_logger(__name__)


class ChatService:
    """Service for handling chat interactions with LangChain and LangGraph."""

    def __init__(
        self,
        orchestrator: Optional[LangGraphOrchestrator] = None,
        db: Optional[SQLiteManager] = None,
    ):
        """Initialize chat service.

        Args:
            orchestrator: LangGraph orchestrator instance
            db: Database manager
        """
        self.db = db or SQLiteManager()
        
        # Initialize LangGraph orchestrator (primary workflow)
        try:
            self.orchestrator = orchestrator or LangGraphOrchestrator()
            logger.info("Initialized LangGraph orchestrator for chat service")
        except Exception as e:
            logger.error("Failed to initialize LangGraph orchestrator", error=str(e), exc_info=True)
            raise RuntimeError(f"Failed to initialize chat service: {str(e)}") from e

    async def process_query(
        self,
        query: str,
        folder_id: Optional[str],
        user_id: str,
        conversation_id: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        citations_callback: Optional[Callable[[list], None]] = None,
        use_graph_intelligence: bool = True,
    ) -> Dict:
        """Process a chat query with LangGraph orchestration.

        Args:
            query: User query
            folder_id: Optional folder ID to search
            user_id: User ID
            conversation_id: Optional conversation ID
            streaming_callback: Optional callback for streaming tokens
            status_callback: Optional callback for status updates
            citations_callback: Optional callback for progressive citations
            use_graph_intelligence: Whether to use graph-enhanced processing

        Returns:
            Response dictionary with answer and citations
        """
        try:
            logger.info(
                "Processing chat query with LangGraph",
                query_length=len(query),
                folder_id=folder_id,
                user_id=user_id[:8] + "...",
            )

            # Get or create conversation ID
            if not conversation_id:
                conversation_id = str(uuid.uuid4())

            # Store user message
            await self.db.store_message(
                conversation_id, "user", query, user_id, folder_id=folder_id
            )

            # Get conversation history
            conversation_history = await self._get_conversation_history(conversation_id)

            # Check if this is a cross-document synthesis query
            is_synthesis_query = self._is_synthesis_query(query)
            
            # Process query with LangGraph orchestrator (or fallback to RAG service)
            if folder_id:
                # Use FoldexGraph for cross-document synthesis queries
                if is_synthesis_query:
                    try:
                        result = await self._process_with_foldex_graph(
                            query=query,
                            folder_id=folder_id,
                            conversation_id=conversation_id,
                            streaming_callback=streaming_callback,
                            status_callback=status_callback,
                            citations_callback=citations_callback,
                        )
                    except Exception as e:
                        logger.warning(
                            "FoldexGraph processing failed, falling back to orchestrator",
                            error=str(e),
                            exc_info=True
                        )
                        # Fall through to orchestrator
                        is_synthesis_query = False
                
                if not is_synthesis_query:
                    try:
                        # Use LangGraph orchestrator for advanced processing
                        result = await self.orchestrator.process_query_with_graph_intelligence(
                            query=query,
                            folder_id=folder_id,
                            conversation_id=conversation_id,
                            conversation_history=conversation_history,
                            use_graph_enhancement=use_graph_intelligence,
                            streaming_callback=streaming_callback,
                            status_callback=status_callback,
                            citations_callback=citations_callback,
                        )
                        
                        # Map orchestrator result to expected format
                        result = {
                            "response": result.get("answer", ""),
                            "citations": result.get("citations", []),
                            "query_intent": result.get("query_intent"),
                            "confidence": result.get("confidence", 0.0),
                        }
                    except Exception as e:
                        logger.error(
                            "LangGraph orchestrator failed, this should not happen",
                            error=str(e),
                            exc_info=True
                        )
                        # Re-raise since we don't have a fallback RAG service here
                        raise
            else:
                # Handle queries without folder context
                result = {
                    "response": "Please specify a folder to search in.",
                    "citations": [],
                    "query_intent": None,
                    "confidence": 0.0,
                }

            # Validate result structure
            if "response" not in result:
                logger.error("Result missing 'response' key", result_keys=list(result.keys()) if result else None)
                raise ValueError(f"Result missing 'response' key. Got keys: {list(result.keys()) if result else None}")
            if "citations" not in result:
                logger.warning("Result missing 'citations' key, defaulting to empty list")
                result["citations"] = []

            # Store assistant response
            await self.db.store_message(
                conversation_id, "assistant", result["response"], user_id, folder_id=folder_id, citations=result["citations"]
            )

            return {
                "response": result["response"],
                "citations": result["citations"],
                "conversation_id": conversation_id,
                "query_intent": result.get("query_intent"),
                "confidence": result.get("confidence", 0.0),
            }
        except Exception as e:
            logger.error("Chat query processing failed", error=str(e), exc_info=True)
            raise

    async def _get_conversation_history(self, conversation_id: str) -> List[Dict[str, str]]:
        """Get conversation history for context.
        
        Args:
            conversation_id: Conversation identifier
            
        Returns:
            List of message dictionaries with 'role' and 'content' keys
        """
        try:
            messages = await self.db.get_conversation_messages(conversation_id, limit=10)
            history = []
            for msg in messages:
                history.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })
            return history
        except Exception as e:
            logger.warning("Failed to get conversation history", error=str(e))
            return []

    def _is_synthesis_query(self, query: str) -> bool:
        """Detect if query is asking for cross-document synthesis.
        
        Args:
            query: User query
            
        Returns:
            True if query is asking for cross-document analysis
        """
        query_lower = query.lower()
        
        # Keywords that indicate cross-document synthesis
        synthesis_keywords = [
            "common",
            "shared",
            "between",
            "across",
            "compare",
            "similar",
            "different",
            "all files",
            "all documents",
            "both",
            "multiple",
            "relationship",
            "entities",
            "synthesis",
            "together",
            "collectively",
        ]
        
        # Check if query contains synthesis keywords
        has_keyword = any(keyword in query_lower for keyword in synthesis_keywords)
        
        # Also check for questions about multiple documents
        has_multiple_refs = any(phrase in query_lower for phrase in [
            "files in this folder",
            "documents in",
            "papers",
            "both papers",
            "all papers",
        ])
        
        return has_keyword or has_multiple_refs

    async def _process_with_foldex_graph(
        self,
        query: str,
        folder_id: str,
        conversation_id: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        citations_callback: Optional[Callable[[list], None]] = None,
    ) -> Dict:
        """Process query using FoldexGraph for cross-document synthesis.
        
        Args:
            query: User query
            folder_id: Folder identifier
            conversation_id: Optional conversation ID
            streaming_callback: Optional callback for streaming tokens
            status_callback: Optional status callback
            citations_callback: Optional citations callback
            
        Returns:
            Response dictionary with synthesis and citations
        """
        try:
            if status_callback:
                status_callback("Initializing multi-document analysis...")
            
            # Get RAG service to access retriever
            rag_service = get_rag_service()
            
            # Ensure folder is initialized
            if folder_id not in rag_service.retrievers:
                if status_callback:
                    status_callback("Loading folder documents...")
                await rag_service.initialize_for_folder(folder_id)
            
            # Get retriever and LLM
            retriever = rag_service.retrievers[folder_id]
            llm = rag_service.llm
            
            # Create FoldexGraph
            foldex_graph = create_foldex_graph(retriever, llm)
            
            if status_callback:
                status_callback("Retrieving relevant chunks from multiple documents...")
            
            # Execute graph
            graph_result = foldex_graph.invoke(
                query=query,
                folder_id=folder_id,
            )
            
            if graph_result.get("error"):
                raise RuntimeError(f"Graph execution failed: {graph_result['error']}")
            
            if status_callback:
                status_callback("Synthesizing findings across documents...")
            
            # Extract results
            synthesis = graph_result.get("synthesis", "")
            citations = graph_result.get("citations", [])
            entities_per_file = graph_result.get("entities_per_file", {})
            used_citations = graph_result.get("used_citations", [])
            
            # Use inline citations if available, otherwise fall back to regular citations
            if used_citations:
                # Send inline citations (these are the ones actually referenced in the text)
                if citations_callback:
                    citations_callback(used_citations)
            elif citations:
                # Fall back to regular citations
                if citations_callback:
                    citations_callback(citations)
            
            # Stream the synthesis text if streaming callback provided
            if streaming_callback and synthesis:
                import asyncio
                logger.info("Streaming FoldexGraph synthesis", synthesis_length=len(synthesis))
                # Stream word by word for natural appearance (better than character-by-character)
                words = synthesis.split()
                total_words = len(words)
                for i, word in enumerate(words):
                    # Add space before word (except first)
                    if i > 0:
                        streaming_callback(" ")
                    streaming_callback(word)
                    # Small delay every few words to prevent overwhelming the client
                    if i % 3 == 0:  # Every 3 words, yield control
                        await asyncio.sleep(0.01)  # 10ms delay for smooth streaming
                logger.info(
                    "Finished streaming FoldexGraph synthesis",
                    word_count=total_words,
                    total_chars=len(synthesis)
                )
            elif not streaming_callback:
                logger.warning("No streaming callback provided for FoldexGraph - synthesis will not stream")
            
            # Build response - use inline citations if available
            final_citations = used_citations if used_citations else citations
            
            response = {
                "response": synthesis,  # HTML with inline citations
                "citations": final_citations,
                "conversation_id": conversation_id,
                "query_intent": "cross_document_synthesis",
                "confidence": 0.9,  # High confidence for synthesis queries
            }
            
            logger.info(
                "FoldexGraph processing completed",
                folder_id=folder_id,
                synthesis_length=len(synthesis),
                citation_count=len(final_citations),
                num_files=len(entities_per_file),
            )
            
            return response
            
        except Exception as e:
            logger.error(
                "FoldexGraph processing failed",
                error=str(e),
                exc_info=True
            )
            raise

