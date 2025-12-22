"""LangChain conversation and QA chains with Ollama integration."""

from typing import Optional, List, Dict, Any, Callable, AsyncIterator
import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

try:
    from langchain_community.llms import Ollama
    from langchain_community.chat_models import ChatOllama
    from langchain.chains import ConversationalRetrievalChain
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.schema import LLMResult
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback for older LangChain versions
    try:
        from langchain_community.llms import Ollama
        from langchain_community.chat_models import ChatOllama
        from langchain.chains import ConversationalRetrievalChain
        from langchain.callbacks.base import BaseCallbackHandler
        from langchain.schema import LLMResult
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        # Final fallback
        try:
            from langchain.llms import Ollama
            from langchain.chat_models import ChatOllama
            from langchain.chains import ConversationalRetrievalChain
            from langchain.callbacks.base import BaseCallbackHandler
            from langchain.schema import LLMResult
            LANGCHAIN_AVAILABLE = True
        except ImportError:
            LANGCHAIN_AVAILABLE = False
            Ollama = None
            ChatOllama = None
            ConversationalRetrievalChain = None
            BaseCallbackHandler = None
            LLMResult = None

from app.config.settings import settings
from app.core.exceptions import ProcessingError
from app.rag.memory import ConversationMemory
from app.rag.prompt_management import PromptManager, get_prompt_manager
from app.rag.vector_store import LangChainVectorStore
from app.rag.stream_sanitizer import ThinkTagStreamFilter

logger = structlog.get_logger(__name__)


if not LANGCHAIN_AVAILABLE or BaseCallbackHandler is None:
    # Create a dummy class if LangChain is not available
    class StreamingCallbackHandler:
        def __init__(self, callback=None):
            self.callback = callback
            self.tokens = []
else:
    class StreamingCallbackHandler(BaseCallbackHandler):
        """Callback handler for streaming LLM responses."""

        def __init__(self, callback: Optional[Callable[[str], None]] = None):
            """Initialize streaming callback handler.

            Args:
                callback: Optional callback function to receive tokens
            """
            super().__init__()
            self.callback = callback
            self.tokens: List[str] = []
            self._filter = ThinkTagStreamFilter()

        def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any) -> None:
            """Log when LLM starts."""
            logger.info("LLM generation started", prompts_count=len(prompts))

        def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[Any]], **kwargs: Any) -> None:
            """Log when Chat Model starts."""
            logger.info("Chat Model generation started", messages_count=len(messages))

        def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
            """Handle new token from LLM."""
            if not token:
                return

            safe = self._filter.process(token)
            if not safe:
                return

            self.tokens.append(safe)
            if self.callback:
                try:
                    self.callback(safe)
                except Exception as e:
                    logger.error("Streaming callback error", error=str(e), exc_info=True)

        def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
            """Handle LLM response completion.

            Args:
                response: LLM result
                **kwargs: Additional arguments
            """
            logger.debug("LLM response completed", token_count=len(self.tokens))

        def get_full_response(self) -> str:
            """Get full accumulated response.

            Returns:
                Complete response string
            """
            return "".join(self.tokens)


class OllamaLLM:
    """LangChain Ollama LLM wrapper with error handling and retries."""

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.3,
        timeout: Optional[int] = None,
    ):
        """Initialize Ollama LLM.

        Args:
            model: Ollama model name (default: from settings)
            base_url: Ollama base URL (default: from settings)
            temperature: Sampling temperature (0.3 optimized for Llama 3.2:3b factual responses)
            timeout: Request timeout in seconds

        Raises:
            ProcessingError: If LangChain is not available or initialization fails
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain langchain-community"
            )

        self.model_name = model or settings.OLLAMA_MODEL
        self.base_url = base_url or settings.OLLAMA_BASE_URL
        self.temperature = temperature
        self.timeout = timeout or settings.OLLAMA_TIMEOUT

        # Verify Ollama connection before initializing
        self._verify_ollama_connection()

        try:
            logger.info(
                "Initializing ChatOllama",
                model=self.model_name,
                base_url=self.base_url,
            )

            # Initialize ChatOllama with streaming enabled
            self.llm = ChatOllama(
                model=self.model_name,
                base_url=self.base_url,
                temperature=self.temperature,
                timeout=self.timeout,
                num_ctx=2048,  # Context window optimized for Llama 3.2:3b (2K tokens)
                num_predict=settings.OLLAMA_NUM_PREDICT,  # Maximum tokens to generate (prevents response truncation)
                streaming=True,  # Enable token-by-token streaming
                keep_alive=settings.OLLAMA_KEEP_ALIVE,  # Keep model loaded in memory
            )

            logger.info("ChatOllama initialized successfully")
        except Exception as e:
            error_msg = str(e)
            # Provide helpful error messages for common issues
            if "404" in error_msg or "not found" in error_msg.lower():
                if "model" in error_msg.lower():
                    raise ProcessingError(
                        f"Ollama model '{self.model_name}' not found. "
                        f"Please pull it with: ollama pull {self.model_name}\n"
                        f"Ollama base URL: {self.base_url}"
                    ) from e
                else:
                    raise ProcessingError(
                        f"Ollama endpoint not found (404). "
                        f"Please ensure Ollama is running at {self.base_url}\n"
                        f"Start Ollama with: ollama serve"
                    ) from e
            elif "connection" in error_msg.lower() or "refused" in error_msg.lower():
                raise ProcessingError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    f"Please ensure Ollama is running.\n"
                    f"Start Ollama with: ollama serve"
                ) from e
            else:
                logger.error(
                    "Failed to initialize ChatOllama",
                    model=self.model_name,
                    base_url=self.base_url,
                    error=error_msg,
                    exc_info=True,
                    )
                raise ProcessingError(
                    f"Failed to initialize ChatOllama: {error_msg}\n"
                    f"Model: {self.model_name}, Base URL: {self.base_url}"
                ) from e

    def _verify_ollama_connection(self) -> None:
        """Verify Ollama is accessible and model exists.
        
        Raises:
            ProcessingError: If Ollama is not accessible or model not found
        """
        try:
            try:
                import httpx
            except ImportError:
                logger.warning("httpx not available, skipping Ollama connection verification")
                return
            
            # Check if Ollama is running
            try:
                response = httpx.get(
                    f"{self.base_url}/api/tags",
                    timeout=5
                )
                if response.status_code != 200:
                    raise ProcessingError(
                        f"Ollama returned status {response.status_code} at {self.base_url}. "
                        f"Please ensure Ollama is running."
                    )
            except httpx.ConnectError:
                raise ProcessingError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    f"Please ensure Ollama is running.\n"
                    f"Start Ollama with: ollama serve\n"
                    f"If running locally (not Docker), set OLLAMA_BASE_URL=http://localhost:11434"
                )
            except httpx.TimeoutException:
                raise ProcessingError(
                    f"Timeout connecting to Ollama at {self.base_url}. "
                    f"Please ensure Ollama is running and accessible."
                )
            
            # Check if model exists
            models_data = response.json()
            available_models = [m.get("name", "") for m in models_data.get("models", [])]
            
            # Check for exact match first
            exact_match = self.model_name in available_models
            
            if exact_match:
                # Use exact model name
                actual_model = self.model_name
                logger.info(
                    "Ollama connection verified (exact match)",
                    model=actual_model,
                    base_url=self.base_url,
                    available_models_count=len(available_models)
                )
            else:
                # Look for models with same base name (e.g., qwen2.5 matches qwen2.5:7b or qwen2.5:3b)
                model_base = self.model_name.split(":")[0]
                matching_models = [m for m in available_models if m.startswith(model_base + ":") or m == model_base]
                
                if matching_models:
                    # Use the first matching model (prefer the one we asked for if it exists, otherwise use first match)
                    actual_model = matching_models[0]
                    logger.warning(
                        "Exact model not found, using similar model",
                        required=self.model_name,
                        using=actual_model,
                        available_models=available_models
                    )
                    # Update to use the actual model name
                    self.model_name = actual_model
                    logger.info(
                        "Ollama connection verified (using similar model)",
                        model=self.model_name,
                        base_url=self.base_url,
                        available_models_count=len(available_models)
                    )
                else:
                    available_str = ", ".join(available_models) if available_models else "None"
                    raise ProcessingError(
                        f"Ollama model '{self.model_name}' not found.\n"
                        f"Available models: {available_str}\n"
                        f"Please pull the model with: ollama pull {self.model_name}"
                    )
            
        except ProcessingError:
            raise
        except Exception as e:
            logger.warning(
                "Failed to verify Ollama connection, proceeding anyway",
                error=str(e)
            )
            # Don't fail initialization if verification fails, let the actual call fail

    def get_llm(self) -> ChatOllama:
        """Get the underlying LangChain ChatOllama instance.

        Returns:
            ChatOllama instance
        """
        return self.llm


class ConversationalRAGChain:
    """LangChain ConversationalRetrievalChain for context-aware RAG responses."""

    def __init__(
        self,
        vector_store: LangChainVectorStore,
        llm: Optional[OllamaLLM] = None,
        memory: Optional[ConversationMemory] = None,
        prompt_manager: Optional[PromptManager] = None,
        retriever_k: int = 5,
    ):
        """Initialize conversational RAG chain.

        Args:
            vector_store: LangChain vector store for retrieval
            llm: Ollama LLM instance
            memory: Conversation memory instance
            prompt_manager: Prompt manager instance
            retriever_k: Number of documents to retrieve

        Raises:
            ProcessingError: If initialization fails
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        self.vector_store = vector_store
        self.llm = llm or OllamaLLM()
        self.memory = memory
        self.prompt_manager = prompt_manager or get_prompt_manager()
        self.retriever_k = retriever_k

        # Create retriever from vector store
        try:
            self.retriever = self.vector_store.vector_store.as_retriever(
                search_kwargs={"k": self.retriever_k}
            )
            logger.debug("Created retriever from vector store", k=self.retriever_k)
        except Exception as e:
            logger.error("Failed to create retriever", error=str(e))
            raise ProcessingError(f"Failed to create retriever: {str(e)}") from e

        # Initialize chain
        self.chain = None
        self._initialize_chain()

    def _initialize_chain(self) -> None:
        """Initialize ConversationalRetrievalChain."""
        try:
            # Get default prompt
            prompt = self.prompt_manager.get_prompt("default")

            # Create chain with memory
            # Note: ConversationalRetrievalChain handles memory internally
            chain_kwargs = {
                "llm": self.llm.get_llm(),
                "retriever": self.retriever,
                "return_source_documents": True,
                "verbose": True,
            }

            # Add memory if available
            if self.memory:
                chain_kwargs["memory"] = self.memory.get_memory()

            # Add prompt to combine_docs_chain_kwargs
            chain_kwargs["combine_docs_chain_kwargs"] = {"prompt": prompt}

            self.chain = ConversationalRetrievalChain.from_llm(**chain_kwargs)

            logger.info("Initialized ConversationalRetrievalChain")
        except Exception as e:
            logger.error(
                "Failed to initialize ConversationalRetrievalChain",
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(
                f"Failed to initialize chain: {str(e)}"
            ) from e

    async def invoke(
        self,
        question: str,
        query_type: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, Any]:
        """Invoke chain with question.

        Args:
            question: User question
            query_type: Optional query type for prompt selection
            streaming_callback: Optional callback for streaming tokens

        Returns:
            Dictionary with answer and source documents

        Raises:
            ProcessingError: If chain invocation fails
        """
        try:
            # Classify query type if not provided
            if not query_type:
                query_type = self.prompt_manager.classify_query_type(question)

            # Update prompt if query type is different
            if query_type != "default":
                prompt = self.prompt_manager.get_prompt(query_type)
                # Reinitialize chain with new prompt
                chain_kwargs = {
                    "llm": self.llm.get_llm(),
                    "retriever": self.retriever,
                    "return_source_documents": True,
                    "verbose": True,
                    "combine_docs_chain_kwargs": {"prompt": prompt},
                }
                if self.memory:
                    chain_kwargs["memory"] = self.memory.get_memory()
                self.chain = ConversationalRetrievalChain.from_llm(**chain_kwargs)

            # Set up streaming if callback provided
            callbacks = []
            if streaming_callback:
                callbacks.append(StreamingCallbackHandler(streaming_callback))

            # Prepare input
            chain_input = {"question": question}

            # Invoke chain with retry mechanism (synchronous but wrapped for async compatibility)
            result = self._invoke_with_retry(chain_input, callbacks=callbacks)

            # Update memory if available
            if self.memory:
                self.memory.add_user_message(question)
                if "answer" in result:
                    self.memory.add_ai_message(result["answer"])

            logger.info(
                "Chain invocation completed",
                question_length=len(question),
                answer_length=len(result.get("answer", "")),
                source_count=len(result.get("source_documents", [])),
            )

            return {
                "answer": result.get("answer", ""),
                "source_documents": result.get("source_documents", []),
                "query_type": query_type,
            }

        except Exception as e:
            logger.error(
                "Chain invocation failed",
                question=question[:100],
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(f"Chain invocation failed: {str(e)}") from e

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        reraise=True,
    )
    def _invoke_with_retry(
        self, chain_input: Dict[str, Any], callbacks: Optional[List] = None
    ) -> Dict[str, Any]:
        """Invoke chain with retry mechanism.

        Args:
            chain_input: Chain input dictionary
            callbacks: Optional list of callbacks

        Returns:
            Chain result dictionary

        Raises:
            ProcessingError: If all retries fail or chain is not initialized
        """
        if self.chain is None:
            raise ProcessingError("Chain is not initialized")
        try:
            return self.chain.invoke(chain_input, callbacks=callbacks)
        except (ConnectionError, TimeoutError) as e:
            logger.warning(
                "Chain invocation failed, will retry",
                error=str(e),
                error_type=type(e).__name__,
            )
            raise
        except Exception as e:
            error_msg = str(e)
            error_str = error_msg.lower()
            
            # Check for 404 errors (model not found, endpoint not found)
            if "404" in error_msg or "not found" in error_str:
                if "model" in error_str:
                    raise ProcessingError(
                        f"Ollama model not found (404). "
                        f"Please ensure the model '{self.model_name}' is available.\n"
                        f"Pull it with: ollama pull {self.model_name}"
                    ) from e
                else:
                    raise ProcessingError(
                        f"Ollama endpoint not found (404). "
                        f"Please ensure Ollama is running at {self.base_url}\n"
                        f"Start Ollama with: ollama serve"
                    ) from e
            
            # Check for connection errors
            if "connection" in error_str or "refused" in error_str or "unreachable" in error_str:
                raise ProcessingError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    f"Please ensure Ollama is running.\n"
                    f"Start Ollama with: ollama serve\n"
                    f"If running locally (not Docker), set OLLAMA_BASE_URL=http://localhost:11434"
                ) from e
            
            logger.error(
                "Chain invocation failed with unexpected error",
                error=error_msg,
                error_type=type(e).__name__,
            )
            raise ProcessingError(f"Chain invocation failed: {error_msg}") from e

    async def stream(
        self, question: str, query_type: Optional[str] = None
    ) -> AsyncIterator[str]:
        """Stream chain response.

        Args:
            question: User question
            query_type: Optional query type

        Yields:
            Response tokens as strings
        """
        # Note: Full streaming support requires async chain implementation
        # This is a simplified version that collects tokens via callback
        try:
            tokens = []

            await self.invoke(
                question=question,
                query_type=query_type,
                streaming_callback=lambda token: tokens.append(token),
            )

            # Yield tokens as they were collected
            for token in tokens:
                yield token

        except Exception as e:
            logger.error("Streaming failed", error=str(e))
            raise ProcessingError(f"Streaming failed: {str(e)}") from e


