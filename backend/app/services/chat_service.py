"""Chat service for processing user queries."""

from typing import Dict, Optional
import uuid
import structlog

from app.services.rag_engine import RAGEngine
from app.database.sqlite_manager import SQLiteManager

logger = structlog.get_logger(__name__)


class ChatService:
    """Service for handling chat interactions."""

    def __init__(
        self,
        rag_engine: Optional[RAGEngine] = None,
        db: Optional[SQLiteManager] = None,
    ):
        """Initialize chat service.

        Args:
            rag_engine: RAG engine instance
            db: Database manager
        """
        self.rag_engine = rag_engine or RAGEngine()
        self.db = db or SQLiteManager()

    async def process_query(
        self,
        query: str,
        folder_id: Optional[str],
        user_id: str,
        conversation_id: Optional[str] = None,
    ) -> Dict:
        """Process a chat query.

        Args:
            query: User query
            folder_id: Optional folder ID to search
            user_id: User ID
            conversation_id: Optional conversation ID

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
                conversation_id, "user", query, user_id
            )

            # Process query with RAG
            if folder_id:
                result = await self.rag_engine.query(query, folder_id)
            else:
                # TODO: Handle queries without folder context
                result = {"response": "No folder specified", "citations": []}

            # Store assistant response
            await self.db.store_message(
                conversation_id, "assistant", result["response"], user_id
            )

            return {
                "response": result["response"],
                "citations": result["citations"],
                "conversation_id": conversation_id,
            }
        except Exception as e:
            logger.error("Chat query processing failed", error=str(e))
            raise

