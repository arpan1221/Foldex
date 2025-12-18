"""LangChain memory management and cleanup."""

from typing import Optional, Dict, Any
import structlog
import gc
import time
import asyncio

try:
    from langchain_core.memory import BaseMemory
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.schema import BaseMemory
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        BaseMemory = None

from app.core.exceptions import ProcessingError
from app.rag.context_management import MemoryManager

logger = structlog.get_logger(__name__)


class MemoryOptimizer:
    """Optimizer for LangChain memory management."""

    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        max_memory_age: int = 3600,
        cleanup_interval: int = 300,
    ):
        """Initialize memory optimizer.

        Args:
            memory_manager: Optional memory manager
            max_memory_age: Maximum age of memory in seconds
            cleanup_interval: Cleanup interval in seconds
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        self.memory_manager = memory_manager or MemoryManager()
        self.max_memory_age = max_memory_age
        self.cleanup_interval = cleanup_interval
        self.memory_timestamps: Dict[str, float] = {}
        self.logger = structlog.get_logger(__name__)

    def register_memory(self, conversation_id: str) -> None:
        """Register memory with timestamp.

        Args:
            conversation_id: Conversation identifier
        """
        self.memory_timestamps[conversation_id] = time.time()
        logger.debug("Registered memory", conversation_id=conversation_id)

    def cleanup_old_memories(self) -> int:
        """Cleanup old memories.

        Returns:
            Number of memories cleaned up
        """
        try:
            current_time = time.time()
            cleaned_count = 0

            for conversation_id, timestamp in list(self.memory_timestamps.items()):
                age = current_time - timestamp
                if age > self.max_memory_age:
                    # Clear memory
                    self.memory_manager.clear_memory(conversation_id)
                    del self.memory_timestamps[conversation_id]
                    cleaned_count += 1

            if cleaned_count > 0:
                logger.info(
                    "Cleaned up old memories",
                    cleaned_count=cleaned_count,
                )

            return cleaned_count

        except Exception as e:
            logger.error("Memory cleanup failed", error=str(e))
            return 0

    def optimize_memory_usage(
        self,
        conversation_id: str,
        max_messages: Optional[int] = None,
    ) -> None:
        """Optimize memory usage for conversation.

        Args:
            conversation_id: Conversation identifier
            max_messages: Maximum number of messages to keep
        """
        try:
            memory = self.memory_manager.get_memory(conversation_id)
            if not memory:
                return

            # If memory has chat_memory, optimize it
            if hasattr(memory, "chat_memory"):
                messages = memory.chat_memory.messages
                if max_messages and len(messages) > max_messages:
                    # Keep only recent messages
                    recent_messages = messages[-max_messages:]
                    memory.chat_memory.clear()
                    for msg in recent_messages:
                        memory.chat_memory.add_message(msg)

                    logger.debug(
                        "Optimized memory usage",
                        conversation_id=conversation_id,
                        kept_messages=max_messages,
                    )

        except Exception as e:
            logger.error("Memory optimization failed", error=str(e))

    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get memory statistics.

        Returns:
            Dictionary with memory statistics
        """
        try:
            active_memories = len(self.memory_timestamps)
            oldest_memory_age = 0.0

            if self.memory_timestamps:
                current_time = time.time()
                oldest_timestamp = min(self.memory_timestamps.values())
                oldest_memory_age = current_time - oldest_timestamp

            return {
                "active_memories": active_memories,
                "oldest_memory_age": oldest_memory_age,
                "max_memory_age": self.max_memory_age,
            }

        except Exception as e:
            logger.error("Failed to get memory statistics", error=str(e))
            return {}


class MemoryCleanupScheduler:
    """Scheduler for periodic memory cleanup."""

    def __init__(
        self,
        optimizer: MemoryOptimizer,
        cleanup_interval: int = 300,
    ):
        """Initialize memory cleanup scheduler.

        Args:
            optimizer: Memory optimizer
            cleanup_interval: Cleanup interval in seconds
        """
        self.optimizer = optimizer
        self.cleanup_interval = cleanup_interval
        self.running = False
        self.logger = structlog.get_logger(__name__)

    async def start(self) -> None:
        """Start cleanup scheduler."""
        self.running = True
        logger.info("Started memory cleanup scheduler")

        while self.running:
            try:
                await asyncio.sleep(self.cleanup_interval)
                if self.running:
                    cleaned = self.optimizer.cleanup_old_memories()
                    if cleaned > 0:
                        # Force garbage collection
                        gc.collect()

            except Exception as e:
                logger.error("Cleanup scheduler error", error=str(e))

    def stop(self) -> None:
        """Stop cleanup scheduler."""
        self.running = False
        logger.info("Stopped memory cleanup scheduler")


class MemoryPool:
    """Pool for managing multiple memory instances."""

    def __init__(
        self,
        memory_manager: Optional[MemoryManager] = None,
        max_pool_size: int = 100,
    ):
        """Initialize memory pool.

        Args:
            memory_manager: Optional memory manager
            max_pool_size: Maximum pool size
        """
        self.memory_manager = memory_manager or MemoryManager()
        self.max_pool_size = max_pool_size
        self.pool: Dict[str, BaseMemory] = {}
        self.access_times: Dict[str, float] = {}
        self.logger = structlog.get_logger(__name__)

    def get_memory(
        self,
        conversation_id: str,
        create_if_missing: bool = True,
    ) -> Optional[BaseMemory]:
        """Get memory from pool.

        Args:
            conversation_id: Conversation identifier
            create_if_missing: Whether to create if missing

        Returns:
            Memory instance or None
        """
        try:
            # Check if in pool
            if conversation_id in self.pool:
                self.access_times[conversation_id] = time.time()
                return self.pool[conversation_id]

            # Create if missing
            if create_if_missing:
                memory = self.memory_manager.get_memory(conversation_id)
                if not memory:
                    # Create buffer memory as default
                    memory = self.memory_manager.create_buffer_memory(conversation_id)
                if memory:
                    self._add_to_pool(conversation_id, memory)
                    return memory

            return None

        except Exception as e:
            logger.error("Failed to get memory from pool", error=str(e))
            return None

    def _add_to_pool(
        self,
        conversation_id: str,
        memory: BaseMemory,
    ) -> None:
        """Add memory to pool.

        Args:
            conversation_id: Conversation identifier
            memory: Memory instance
        """
        # Check pool size
        if len(self.pool) >= self.max_pool_size:
            # Remove least recently used
            self._evict_lru()

        self.pool[conversation_id] = memory
        self.access_times[conversation_id] = time.time()

    def _evict_lru(self) -> None:
        """Evict least recently used memory."""
        if not self.access_times:
            return

        # Find LRU
        lru_id = min(self.access_times.items(), key=lambda x: x[1])[0]

        # Remove from pool
        if lru_id in self.pool:
            del self.pool[lru_id]
        if lru_id in self.access_times:
            del self.access_times[lru_id]

        logger.debug("Evicted LRU memory", conversation_id=lru_id)

    def clear_pool(self) -> None:
        """Clear memory pool."""
        self.pool.clear()
        self.access_times.clear()
        logger.info("Cleared memory pool")

    def get_pool_statistics(self) -> Dict[str, Any]:
        """Get pool statistics.

        Returns:
            Dictionary with pool statistics
        """
        return {
            "pool_size": len(self.pool),
            "max_pool_size": self.max_pool_size,
            "utilization": len(self.pool) / self.max_pool_size if self.max_pool_size > 0 else 0.0,
        }

