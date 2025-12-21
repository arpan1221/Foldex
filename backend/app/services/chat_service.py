"""Chat service for processing user queries."""

from typing import Dict, Optional, Callable, List
import uuid
import structlog

from app.database.sqlite_manager import SQLiteManager
from app.services.rag_service import get_rag_service

logger = structlog.get_logger(__name__)


class ChatService:
    """Service for handling chat interactions with RAG."""

    def __init__(
        self,
        db: Optional[SQLiteManager] = None,
    ):
        """Initialize chat service.

        Args:
            db: Database manager
        """
        self.db = db or SQLiteManager()
        logger.info("Initialized chat service with standard RAG")

    async def process_query(
        self,
        query: str,
        folder_id: Optional[str],
        user_id: str,
        conversation_id: Optional[str] = None,
        file_id: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        citations_callback: Optional[Callable[[list], None]] = None,
        use_graph_intelligence: bool = False,  # Not used - kept for backward compatibility
    ) -> Dict:
        """Process a chat query with standard RAG.

        Args:
            query: User query
            folder_id: Optional folder ID to search
            user_id: User ID
            conversation_id: Optional conversation ID
            file_id: Optional file ID for file-specific chats
            streaming_callback: Optional callback for streaming tokens
            status_callback: Optional callback for status updates
            citations_callback: Optional callback for progressive citations
            use_graph_intelligence: Not used (kept for backward compatibility)

        Returns:
            Response dictionary with answer and citations
        """
        try:
            logger.info(
                "Processing chat query",
                query_length=len(query),
                folder_id=folder_id,
                user_id=user_id[:8] + "...",
            )

            # Get or create conversation ID
            if not conversation_id:
                conversation_id = str(uuid.uuid4())

            # Store user message
            await self.db.store_message(
                conversation_id, "user", query, user_id, folder_id=folder_id, file_id=file_id
            )

            # Get conversation history
            conversation_history = await self._get_conversation_history(conversation_id)

            # If file_id is provided, add file context to query
            enhanced_query = query
            if file_id:
                # Get file name from database
                try:
                    files = await self.db.get_files_by_folder(folder_id)
                    file_info = next((f for f in files if f.get("file_id") == file_id), None)
                    if file_info:
                        file_name = file_info.get("file_name", "the file")
                        enhanced_query = f"Regarding {file_name}: {query}"
                        logger.info(
                            "Added file context to query",
                            file_id=file_id,
                            file_name=file_name,
                        )
                except Exception as e:
                    logger.warning("Failed to get file name for context", file_id=file_id, error=str(e))
            
            # Use standard RAG service for all queries (simplified workflow)
            if folder_id:
                # Initialize vector store if needed
                from app.rag.vector_store import LangChainVectorStore
                vector_store = LangChainVectorStore()
                rag_service = get_rag_service(vector_store=vector_store)
                
                # Ensure folder is initialized
                if folder_id not in rag_service.retrievers:
                    if status_callback:
                        status_callback("Loading folder documents...")
                    await rag_service.initialize_for_folder(folder_id)
                
                # Use standard RAG service query method
                try:
                    rag_result = await rag_service.query(
                        query=enhanced_query,
                        folder_id=folder_id,
                        conversation_id=conversation_id,
                        streaming_callback=streaming_callback,
                        status_callback=status_callback,
                        citations_callback=citations_callback,
                        file_id=file_id,
                    )
                    
                    # Map RAG service result to expected format
                    result = {
                        "response": rag_result.get("answer", ""),
                        "citations": rag_result.get("citations", []),
                        "query_intent": rag_result.get("query_intent", "factual"),
                        "confidence": rag_result.get("confidence", 0.8),
                    }
                except Exception as e:
                    logger.error(
                        "RAG service query failed",
                        error=str(e),
                        exc_info=True
                    )
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
                conversation_id, "assistant", result["response"], user_id, folder_id=folder_id, file_id=file_id, citations=result["citations"]
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


