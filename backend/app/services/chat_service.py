"""Chat service for processing user queries."""

from typing import Dict, Optional, Callable, List
import uuid
import structlog

from app.services.langgraph_orchestrator import LangGraphOrchestrator
from app.database.sqlite_manager import SQLiteManager

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
        use_graph_intelligence: bool = True,
    ) -> Dict:
        """Process a chat query with LangGraph orchestration.

        Args:
            query: User query
            folder_id: Optional folder ID to search
            user_id: User ID
            conversation_id: Optional conversation ID
            streaming_callback: Optional callback for streaming tokens
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

            # Process query with LangGraph orchestrator (or fallback to RAG service)
            if folder_id:
                try:
                    # Use LangGraph orchestrator for advanced processing
                    result = await self.orchestrator.process_query_with_graph_intelligence(
                        query=query,
                        folder_id=folder_id,
                        conversation_id=conversation_id,
                        conversation_history=conversation_history,
                        use_graph_enhancement=use_graph_intelligence,
                        streaming_callback=streaming_callback,
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

