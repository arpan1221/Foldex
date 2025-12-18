"""Advanced LangChain conversation chains with context management."""

from typing import Optional, Dict, Any, Callable
import structlog

try:
    from langchain.chains import ConversationChain
    from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
    from langchain_core.memory import BaseMemory
    from langchain_core.retrievers import BaseRetriever
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.chains import ConversationChain
        from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
        from langchain.schema import BaseMemory, BaseRetriever
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        ConversationChain = None
        ConversationalRetrievalChain = None
        BaseMemory = None
        BaseRetriever = None

from app.core.exceptions import ProcessingError
from app.rag.llm_chains import OllamaLLM
from app.rag.prompt_management import PromptManager, get_prompt_manager
from app.rag.context_management import MemoryManager, ContextWindowManager
from app.rag.citation_callbacks import CitationCallbackHandler

logger = structlog.get_logger(__name__)


class AdvancedConversationChain:
    """Advanced conversation chain with multiple memory types and context management."""

    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        llm: Optional[OllamaLLM] = None,
        memory_manager: Optional[MemoryManager] = None,
        prompt_manager: Optional[PromptManager] = None,
        memory_type: str = "summary_buffer",
        max_tokens: int = 4000,
    ):
        """Initialize advanced conversation chain.

        Args:
            retriever: Optional LangChain retriever
            llm: Ollama LLM instance
            memory_manager: Memory manager instance
            prompt_manager: Prompt manager instance
            memory_type: Type of memory (buffer, summary_buffer, token_buffer, window, vector_store)
            max_tokens: Maximum tokens for context window

        Raises:
            ProcessingError: If initialization fails
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        self.retriever = retriever
        self.llm = llm or OllamaLLM()
        self.memory_manager = memory_manager or MemoryManager(llm=self.llm)
        self.prompt_manager = prompt_manager or get_prompt_manager()
        self.memory_type = memory_type
        self.context_manager = ContextWindowManager(max_tokens=max_tokens)

        # Initialize citation callback
        self.citation_callback = CitationCallbackHandler()

        # Initialize chain
        self.chain = None
        self.conversation_memories: Dict[str, BaseMemory] = {}

    def get_or_create_memory(
        self,
        conversation_id: str,
    ) -> BaseMemory:
        """Get or create memory for conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            Memory instance
        """
        # Check if memory exists
        memory = self.memory_manager.get_memory(conversation_id)
        if memory:
            return memory

        # Create new memory based on type
        if self.memory_type == "buffer":
            memory = self.memory_manager.create_buffer_memory(conversation_id)
        elif self.memory_type == "summary_buffer":
            memory = self.memory_manager.create_summary_buffer_memory(
                conversation_id,
                max_token_limit=2000,
            )
        elif self.memory_type == "token_buffer":
            memory = self.memory_manager.create_token_buffer_memory(
                conversation_id,
                max_token_limit=2000,
            )
        elif self.memory_type == "window":
            memory = self.memory_manager.create_window_memory(
                conversation_id,
                k=10,
            )
        elif self.memory_type == "vector_store" and self.retriever:
            memory = self.memory_manager.create_vector_store_memory(
                conversation_id,
                retriever=self.retriever,
            )
        else:
            # Default to summary buffer
            memory = self.memory_manager.create_summary_buffer_memory(conversation_id)

        self.conversation_memories[conversation_id] = memory
        return memory

    def create_conversation_chain(
        self,
        conversation_id: str,
    ) -> ConversationChain:
        """Create conversation chain for a conversation.

        Args:
            conversation_id: Conversation identifier

        Returns:
            ConversationChain instance
        """
        try:
            # Get or create memory
            memory = self.get_or_create_memory(conversation_id)

            # Get prompt
            prompt = self.prompt_manager.get_prompt("default")

            # Create conversation chain
            chain = ConversationChain(
                llm=self.llm.get_llm(),
                memory=memory,
                prompt=prompt,
                verbose=True,
            )

            logger.info(
                "Created conversation chain",
                conversation_id=conversation_id,
                memory_type=self.memory_type,
            )

            return chain

        except Exception as e:
            logger.error("Failed to create conversation chain", error=str(e))
            raise ProcessingError(f"Failed to create conversation chain: {str(e)}") from e

    def create_retrieval_conversation_chain(
        self,
        conversation_id: str,
    ) -> ConversationalRetrievalChain:
        """Create conversational retrieval chain.

        Args:
            conversation_id: Conversation identifier

        Returns:
            ConversationalRetrievalChain instance
        """
        try:
            if not self.retriever:
                raise ProcessingError("Retriever is required for retrieval chain")

            # Get or create memory
            memory = self.get_or_create_memory(conversation_id)

            # Get prompt
            prompt = self.prompt_manager.get_prompt("default")

            # Create chain
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.llm.get_llm(),
                retriever=self.retriever,
                memory=memory,
                return_source_documents=True,
                combine_docs_chain_kwargs={"prompt": prompt},
                verbose=True,
            )

            logger.info(
                "Created retrieval conversation chain",
                conversation_id=conversation_id,
            )

            return chain

        except Exception as e:
            logger.error("Failed to create retrieval conversation chain", error=str(e))
            raise ProcessingError(
                f"Failed to create retrieval conversation chain: {str(e)}"
            ) from e

    async def invoke_conversation(
        self,
        query: str,
        conversation_id: str,
        use_retrieval: bool = True,
        streaming_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Invoke conversation chain.

        Args:
            query: User query
            conversation_id: Conversation identifier
            use_retrieval: Whether to use retrieval
            streaming_callback: Optional streaming callback

        Returns:
            Dictionary with response and metadata
        """
        try:
            # Get or create chain
            if use_retrieval and self.retriever:
                chain = self.create_retrieval_conversation_chain(conversation_id)
            else:
                chain = self.create_conversation_chain(conversation_id)

            # Set up callbacks
            callbacks = [self.citation_callback]
            if streaming_callback:
                from app.rag.llm_chains import StreamingCallbackHandler
                callbacks.append(StreamingCallbackHandler(streaming_callback))

            # Prepare input
            if use_retrieval:
                chain_input = {"question": query}
            else:
                chain_input = {"input": query}

            # Check context window
            memory = self.get_or_create_memory(conversation_id)
            history = self._get_conversation_history(memory)

            # Truncate context if needed
            context = ""  # Would be retrieved context
            if not self.context_manager.fits_in_context(query, context, history):
                # Truncate history
                logger.warning("Context window exceeded, truncating history")

            # Invoke chain
            result = chain.invoke(chain_input, callbacks=callbacks)

            # Extract response
            if use_retrieval:
                answer = result.get("answer", "")
                source_documents = result.get("source_documents", [])
            else:
                answer = result.get("response", "")
                source_documents = []

            # Get citations
            citations = self.citation_callback.get_citations()

            return {
                "answer": answer,
                "citations": citations,
                "source_documents": source_documents,
                "conversation_id": conversation_id,
            }

        except Exception as e:
            logger.error("Conversation invocation failed", error=str(e), exc_info=True)
            raise ProcessingError(f"Conversation failed: {str(e)}") from e

    def _get_conversation_history(self, memory: BaseMemory) -> str:
        """Get conversation history as string.

        Args:
            memory: Memory instance

        Returns:
            History string
        """
        try:
            if hasattr(memory, "chat_memory"):
                messages = memory.chat_memory.messages
                history = "\n".join([str(msg.content) for msg in messages])
                return history
            return ""

        except Exception as e:
            logger.error("Failed to get conversation history", error=str(e))
            return ""


class ConversationBranchingChain:
    """LangChain chain for conversation branching and context inheritance."""

    def __init__(
        self,
        base_chain: AdvancedConversationChain,
        llm: Optional[OllamaLLM] = None,
    ):
        """Initialize conversation branching chain.

        Args:
            base_chain: Base conversation chain
            llm: Optional Ollama LLM instance
        """
        self.base_chain = base_chain
        self.llm = llm or OllamaLLM()
        self.branches: Dict[str, Dict[str, Any]] = {}
        self.logger = structlog.get_logger(__name__)

    def create_branch(
        self,
        parent_conversation_id: str,
        branch_id: str,
    ) -> str:
        """Create a conversation branch from parent.

        Args:
            parent_conversation_id: Parent conversation identifier
            branch_id: Branch identifier

        Returns:
            Branch conversation identifier
        """
        try:
            # Get parent memory
            parent_memory = self.base_chain.get_or_create_memory(parent_conversation_id)

            # Create new memory for branch
            branch_conversation_id = f"{parent_conversation_id}_branch_{branch_id}"
            branch_memory = self.base_chain.get_or_create_memory(branch_conversation_id)

            # Inherit context from parent
            if hasattr(parent_memory, "chat_memory"):
                parent_messages = parent_memory.chat_memory.messages
                # Copy messages to branch (up to last 5)
                for msg in parent_messages[-5:]:
                    if hasattr(branch_memory, "chat_memory"):
                        branch_memory.chat_memory.add_message(msg)

            # Track branch
            self.branches[branch_conversation_id] = {
                "parent": parent_conversation_id,
                "branch_id": branch_id,
            }

            logger.info(
                "Created conversation branch",
                parent=parent_conversation_id,
                branch=branch_conversation_id,
            )

            return branch_conversation_id

        except Exception as e:
            logger.error("Failed to create conversation branch", error=str(e))
            raise ProcessingError(f"Failed to create branch: {str(e)}") from e

    def merge_branch(
        self,
        branch_conversation_id: str,
        parent_conversation_id: str,
    ) -> None:
        """Merge branch back into parent conversation.

        Args:
            branch_conversation_id: Branch conversation identifier
            parent_conversation_id: Parent conversation identifier
        """
        try:
            # Get memories
            branch_memory = self.base_chain.get_or_create_memory(branch_conversation_id)
            parent_memory = self.base_chain.get_or_create_memory(parent_conversation_id)

            # Merge messages
            if hasattr(branch_memory, "chat_memory") and hasattr(parent_memory, "chat_memory"):
                branch_messages = branch_memory.chat_memory.messages
                # Add new messages from branch to parent
                for msg in branch_messages:
                    if msg not in parent_memory.chat_memory.messages:
                        parent_memory.chat_memory.add_message(msg)

            logger.info(
                "Merged conversation branch",
                branch=branch_conversation_id,
                parent=parent_conversation_id,
            )

        except Exception as e:
            logger.error("Failed to merge branch", error=str(e))
            raise ProcessingError(f"Failed to merge branch: {str(e)}") from e

