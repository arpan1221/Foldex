"""LangChain conversation memory and context management."""

from typing import Optional, List, Dict, Any
import structlog

try:
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.schema import BaseMessage, HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback for newer LangChain versions
    try:
        from langchain.memory import ConversationBufferWindowMemory
        from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        ConversationBufferWindowMemory = None
        BaseMessage = None
        HumanMessage = None
        AIMessage = None

from app.config.settings import settings
from app.core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)


class ConversationMemory:
    """Manages conversation memory using LangChain's ConversationBufferWindowMemory.

    Provides multi-turn conversation context with configurable window size
    and proper message history management.
    """

    def __init__(
        self,
        conversation_id: str,
        k: int = 10,
        return_messages: bool = True,
    ):
        """Initialize conversation memory.

        Args:
            conversation_id: Unique identifier for the conversation
            k: Number of conversation turns to keep in memory (window size)
            return_messages: Whether to return message objects or strings

        Raises:
            ProcessingError: If LangChain is not available
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        self.conversation_id = conversation_id
        self.k = k
        self.return_messages = return_messages

        # Initialize LangChain memory
        self.memory = ConversationBufferWindowMemory(
            k=k,
            return_messages=return_messages,
            memory_key="chat_history",
            input_key="question",
            output_key="answer",
        )

        logger.debug(
            "Initialized conversation memory",
            conversation_id=conversation_id,
            window_size=k,
        )

    def add_user_message(self, message: str) -> None:
        """Add user message to conversation history.

        Args:
            message: User message text
        """
        try:
            self.memory.chat_memory.add_user_message(message)
            logger.debug(
                "Added user message to memory",
                conversation_id=self.conversation_id,
                message_length=len(message),
            )
        except Exception as e:
            logger.error(
                "Failed to add user message to memory",
                conversation_id=self.conversation_id,
                error=str(e),
            )
            raise

    def add_ai_message(self, message: str) -> None:
        """Add AI response to conversation history.

        Args:
            message: AI response text
        """
        try:
            self.memory.chat_memory.add_ai_message(message)
            logger.debug(
                "Added AI message to memory",
                conversation_id=self.conversation_id,
                message_length=len(message),
            )
        except Exception as e:
            logger.error(
                "Failed to add AI message to memory",
                conversation_id=self.conversation_id,
                error=str(e),
            )
            raise

    def get_memory_variables(self) -> Dict[str, Any]:
        """Get memory variables for chain input.

        Returns:
            Dictionary with memory variables (chat_history, etc.)
        """
        try:
            return self.memory.load_memory_variables({})
        except Exception as e:
            logger.error(
                "Failed to load memory variables",
                conversation_id=self.conversation_id,
                error=str(e),
            )
            return {"chat_history": []}

    def clear(self) -> None:
        """Clear conversation memory."""
        try:
            self.memory.clear()
            logger.debug(
                "Cleared conversation memory",
                conversation_id=self.conversation_id,
            )
        except Exception as e:
            logger.error(
                "Failed to clear memory",
                conversation_id=self.conversation_id,
                error=str(e),
            )

    def get_history(self) -> List[BaseMessage]:
        """Get conversation history as message objects.

        Returns:
            List of BaseMessage objects
        """
        try:
            if self.return_messages:
                return self.memory.chat_memory.messages
            else:
                # Convert string history to messages if needed
                history = self.memory.chat_memory.messages
                return history if isinstance(history, list) else []
        except Exception as e:
            logger.error(
                "Failed to get conversation history",
                conversation_id=self.conversation_id,
                error=str(e),
            )
            return []

    def get_history_string(self) -> str:
        """Get conversation history as formatted string.

        Returns:
            Formatted conversation history string
        """
        try:
            variables = self.get_memory_variables()
            chat_history = variables.get("chat_history", [])
            
            if isinstance(chat_history, str):
                return chat_history
            elif isinstance(chat_history, list):
                # Format list of messages
                formatted = []
                for msg in chat_history:
                    if isinstance(msg, HumanMessage):
                        formatted.append(f"Human: {msg.content}")
                    elif isinstance(msg, AIMessage):
                        formatted.append(f"Assistant: {msg.content}")
                    else:
                        formatted.append(str(msg))
                return "\n".join(formatted)
            else:
                return str(chat_history)
        except Exception as e:
            logger.error(
                "Failed to format conversation history",
                conversation_id=self.conversation_id,
                error=str(e),
            )
            return ""

    def get_memory(self) -> ConversationBufferWindowMemory:
        """Get the underlying LangChain memory object.

        Returns:
            ConversationBufferWindowMemory instance
        """
        return self.memory


class MemoryManager:
    """Manages multiple conversation memories."""

    def __init__(self):
        """Initialize memory manager."""
        self.memories: Dict[str, ConversationMemory] = {}

    def get_or_create_memory(
        self,
        conversation_id: str,
        k: int = 10,
    ) -> ConversationMemory:
        """Get existing memory or create new one.

        Args:
            conversation_id: Unique conversation identifier
            k: Window size for new memories

        Returns:
            ConversationMemory instance
        """
        if conversation_id not in self.memories:
            self.memories[conversation_id] = ConversationMemory(
                conversation_id=conversation_id,
                k=k,
            )
            logger.info(
                "Created new conversation memory",
                conversation_id=conversation_id,
            )

        return self.memories[conversation_id]

    def get_memory(self, conversation_id: str) -> Optional[ConversationMemory]:
        """Get existing memory.

        Args:
            conversation_id: Conversation identifier

        Returns:
            ConversationMemory instance or None if not found
        """
        return self.memories.get(conversation_id)

    def clear_memory(self, conversation_id: str) -> None:
        """Clear memory for a conversation.

        Args:
            conversation_id: Conversation identifier
        """
        if conversation_id in self.memories:
            self.memories[conversation_id].clear()
            logger.info(
                "Cleared conversation memory",
                conversation_id=conversation_id,
            )

    def delete_memory(self, conversation_id: str) -> None:
        """Delete memory for a conversation.

        Args:
            conversation_id: Conversation identifier
        """
        if conversation_id in self.memories:
            del self.memories[conversation_id]
            logger.info(
                "Deleted conversation memory",
                conversation_id=conversation_id,
            )

    def clear_all(self) -> None:
        """Clear all conversation memories."""
        self.memories.clear()
        logger.info("Cleared all conversation memories")


# Global memory manager instance
_memory_manager: Optional[MemoryManager] = None


def get_memory_manager() -> MemoryManager:
    """Get global memory manager instance.

    Returns:
        MemoryManager instance
    """
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = MemoryManager()
    return _memory_manager

