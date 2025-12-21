"""LangChain-based context window and memory management."""

from typing import Optional, Dict, Any, List
import structlog

try:
    from langchain.memory import (
        ConversationBufferMemory,
        ConversationSummaryBufferMemory,
        ConversationTokenBufferMemory,
        ConversationBufferWindowMemory,
    )
    from langchain.memory.vectorstore import VectorStoreRetrieverMemory
    from langchain_core.memory import BaseMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.memory import (
            ConversationBufferMemory,
            ConversationSummaryBufferMemory,
            ConversationTokenBufferMemory,
            ConversationBufferWindowMemory,
        )
        from langchain.memory.vectorstore import VectorStoreRetrieverMemory
        from langchain.schema import BaseMemory
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        ConversationBufferMemory = None
        ConversationSummaryBufferMemory = None
        ConversationTokenBufferMemory = None
        ConversationBufferWindowMemory = None
        VectorStoreRetrieverMemory = None
        BaseMemory = None

from app.core.exceptions import ProcessingError
from app.rag.llm_chains import OllamaLLM

logger = structlog.get_logger(__name__)


class MemoryManager:
    """Manages different types of LangChain memory for conversations."""

    def __init__(self, llm: Optional[OllamaLLM] = None):
        """Initialize memory manager.

        Args:
            llm: Optional Ollama LLM instance for summary memory
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        self.llm = llm or OllamaLLM()
        self.memories: Dict[str, BaseMemory] = {}
        self.logger = structlog.get_logger(__name__)

    def create_buffer_memory(
        self,
        conversation_id: str,
        return_messages: bool = True,
    ) -> ConversationBufferMemory:
        """Create conversation buffer memory.

        Args:
            conversation_id: Conversation identifier
            return_messages: Whether to return messages as objects

        Returns:
            ConversationBufferMemory instance
        """
        try:
            memory = ConversationBufferMemory(
                return_messages=return_messages,
                memory_key="chat_history",
            )

            self.memories[conversation_id] = memory

            logger.debug("Created buffer memory", conversation_id=conversation_id)

            return memory

        except Exception as e:
            logger.error("Failed to create buffer memory", error=str(e))
            raise ProcessingError(f"Failed to create buffer memory: {str(e)}") from e

    def create_summary_buffer_memory(
        self,
        conversation_id: str,
        max_token_limit: int = 2000,
        return_messages: bool = True,
    ) -> ConversationSummaryBufferMemory:
        """Create conversation summary buffer memory.

        Args:
            conversation_id: Conversation identifier
            max_token_limit: Maximum token limit before summarizing
            return_messages: Whether to return messages as objects

        Returns:
            ConversationSummaryBufferMemory instance
        """
        try:
            memory = ConversationSummaryBufferMemory(
                llm=self.llm.get_llm(),
                max_token_limit=max_token_limit,
                return_messages=return_messages,
                memory_key="chat_history",
            )

            self.memories[conversation_id] = memory

            logger.debug(
                "Created summary buffer memory",
                conversation_id=conversation_id,
                max_token_limit=max_token_limit,
            )

            return memory

        except Exception as e:
            logger.error("Failed to create summary buffer memory", error=str(e))
            raise ProcessingError(
                f"Failed to create summary buffer memory: {str(e)}"
            ) from e

    def create_token_buffer_memory(
        self,
        conversation_id: str,
        max_token_limit: int = 2000,
        return_messages: bool = True,
    ) -> ConversationTokenBufferMemory:
        """Create conversation token buffer memory.

        Args:
            conversation_id: Conversation identifier
            max_token_limit: Maximum token limit
            return_messages: Whether to return messages as objects

        Returns:
            ConversationTokenBufferMemory instance
        """
        try:
            memory = ConversationTokenBufferMemory(
                llm=self.llm.get_llm(),
                max_token_limit=max_token_limit,
                return_messages=return_messages,
                memory_key="chat_history",
            )

            self.memories[conversation_id] = memory

            logger.debug(
                "Created token buffer memory",
                conversation_id=conversation_id,
                max_token_limit=max_token_limit,
            )

            return memory

        except Exception as e:
            logger.error("Failed to create token buffer memory", error=str(e))
            raise ProcessingError(
                f"Failed to create token buffer memory: {str(e)}"
            ) from e

    def create_window_memory(
        self,
        conversation_id: str,
        k: int = 10,
        return_messages: bool = True,
    ) -> ConversationBufferWindowMemory:
        """Create conversation buffer window memory.

        Args:
            conversation_id: Conversation identifier
            k: Number of conversation turns to keep
            return_messages: Whether to return messages as objects

        Returns:
            ConversationBufferWindowMemory instance
        """
        try:
            memory = ConversationBufferWindowMemory(
                k=k,
                return_messages=return_messages,
                memory_key="chat_history",
            )

            self.memories[conversation_id] = memory

            logger.debug(
                "Created window memory",
                conversation_id=conversation_id,
                k=k,
            )

            return memory

        except Exception as e:
            logger.error("Failed to create window memory", error=str(e))
            raise ProcessingError(f"Failed to create window memory: {str(e)}") from e

    def create_vector_store_memory(
        self,
        conversation_id: str,
        retriever: Any,
        return_messages: bool = True,
    ) -> VectorStoreRetrieverMemory:
        """Create vector store retriever memory.

        Args:
            conversation_id: Conversation identifier
            retriever: LangChain retriever instance
            return_messages: Whether to return messages as objects

        Returns:
            VectorStoreRetrieverMemory instance
        """
        try:
            memory = VectorStoreRetrieverMemory(
                retriever=retriever,
                return_messages=return_messages,
                memory_key="chat_history",
            )

            self.memories[conversation_id] = memory

            logger.debug(
                "Created vector store memory",
                conversation_id=conversation_id,
            )

            return memory

        except Exception as e:
            logger.error("Failed to create vector store memory", error=str(e))
            raise ProcessingError(
                f"Failed to create vector store memory: {str(e)}"
            ) from e

    def get_memory(self, conversation_id: str) -> Optional[BaseMemory]:
        """Get memory for conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Memory instance or None
        """
        return self.memories.get(conversation_id)

    def clear_memory(self, conversation_id: str) -> None:
        """Clear memory for conversation.

        Args:
            conversation_id: Conversation identifier
        """
        if conversation_id in self.memories:
            memory = self.memories[conversation_id]
            if hasattr(memory, "clear"):
                memory.clear()
            del self.memories[conversation_id]

            logger.debug("Cleared memory", conversation_id=conversation_id)


class ContextWindowManager:
    """Manages context window for conversations."""

    def __init__(self, max_tokens: int = 2000):
        """Initialize context window manager.

        Args:
            max_tokens: Maximum tokens in context window
        """
        self.max_tokens = max_tokens
        self.logger = structlog.get_logger(__name__)

    def estimate_tokens(self, text: str) -> int:
        """Estimate token count for text.

        Args:
            text: Text to estimate

        Returns:
            Estimated token count
        """
        # Simple estimation: ~4 characters per token
        return len(text) // 4

    def fits_in_context(
        self,
        query: str,
        context: str,
        history: Optional[str] = None,
    ) -> bool:
        """Check if content fits in context window.

        Args:
            query: Current query
            context: Context documents
            history: Optional conversation history

        Returns:
            True if fits, False otherwise
        """
        query_tokens = self.estimate_tokens(query)
        context_tokens = self.estimate_tokens(context)
        history_tokens = self.estimate_tokens(history) if history else 0

        total_tokens = query_tokens + context_tokens + history_tokens

        return total_tokens <= self.max_tokens

    def truncate_context(
        self,
        context: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Truncate context to fit in window.

        Args:
            context: Context to truncate
            max_tokens: Maximum tokens (defaults to self.max_tokens)

        Returns:
            Truncated context
        """
        max_tokens = max_tokens or self.max_tokens

        # Estimate and truncate
        estimated_tokens = self.estimate_tokens(context)
        if estimated_tokens <= max_tokens:
            return context

        # Truncate proportionally
        ratio = max_tokens / estimated_tokens
        target_length = int(len(context) * ratio)

        # Try to truncate at sentence boundary
        truncated = context[:target_length]
        last_period = truncated.rfind('.')
        if last_period > target_length * 0.8:  # If period is reasonably close
            truncated = truncated[:last_period + 1]

        return truncated + "..."

