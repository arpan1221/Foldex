"""Improved async streaming for LLM responses to reduce TTFT.

This module provides async-native streaming implementation that bypasses
callback overhead and provides faster token delivery.
"""

import asyncio
from typing import AsyncIterator, Optional, Dict, Any, List
import structlog

try:
    from langchain_community.chat_models import ChatOllama
    from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.chat_models import ChatOllama
        from langchain.schema import HumanMessage, SystemMessage, AIMessage
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        ChatOllama = None
        HumanMessage = None
        SystemMessage = None
        AIMessage = None

from app.core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)


class AsyncStreamingLLM:
    """Async-native streaming LLM wrapper for reduced TTFT.

    This provides direct async streaming without callback overhead,
    resulting in faster token delivery to the client.
    """

    def __init__(self, llm: Optional[ChatOllama] = None):
        """Initialize async streaming LLM.

        Args:
            llm: ChatOllama instance
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain langchain-community"
            )

        self.llm = llm

    async def stream_with_context(
        self,
        query: str,
        context: str,
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream LLM response with context.

        This provides direct async streaming for minimal TTFT.

        Args:
            query: User query
            context: Retrieved context documents
            system_prompt: Optional system prompt

        Yields:
            Response tokens as strings
        """
        if not self.llm:
            raise ProcessingError("LLM not initialized")

        try:
            # Build messages
            messages = []

            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))

            # Combine context and query
            user_message = f"""Context Documents:
{context}

Question: {query}

Answer:"""

            messages.append(HumanMessage(content=user_message))

            # Stream response using native async streaming
            logger.debug("Starting async LLM streaming", query_length=len(query))

            async for chunk in self.llm.astream(messages):
                # Extract content from chunk
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
                elif isinstance(chunk, str):
                    yield chunk

        except Exception as e:
            logger.error("Async streaming failed", error=str(e))
            raise ProcessingError(f"Async streaming failed: {str(e)}") from e

    async def stream_simple(
        self,
        prompt: str,
    ) -> AsyncIterator[str]:
        """Stream LLM response for simple prompt.

        Args:
            prompt: Complete prompt text

        Yields:
            Response tokens as strings
        """
        if not self.llm:
            raise ProcessingError("LLM not initialized")

        try:
            # Stream response
            async for chunk in self.llm.astream(prompt):
                if hasattr(chunk, "content") and chunk.content:
                    yield chunk.content
                elif isinstance(chunk, str):
                    yield chunk

        except Exception as e:
            logger.error("Simple async streaming failed", error=str(e))
            raise ProcessingError(f"Async streaming failed: {str(e)}") from e


class BufferedAsyncStream:
    """Buffered async stream for smoother token delivery.

    Buffers tokens to reduce network overhead while maintaining
    low latency.
    """

    def __init__(
        self,
        source: AsyncIterator[str],
        buffer_size: int = 3,
        flush_interval: float = 0.1,
    ):
        """Initialize buffered stream.

        Args:
            source: Source async iterator
            buffer_size: Number of tokens to buffer
            flush_interval: Max seconds to wait before flushing
        """
        self.source = source
        self.buffer_size = buffer_size
        self.flush_interval = flush_interval

    async def stream(self) -> AsyncIterator[str]:
        """Stream buffered tokens.

        Yields:
            Buffered token strings
        """
        buffer: List[str] = []
        last_flush = asyncio.get_event_loop().time()

        try:
            async for token in self.source:
                buffer.append(token)

                current_time = asyncio.get_event_loop().time()
                should_flush = (
                    len(buffer) >= self.buffer_size
                    or (current_time - last_flush) >= self.flush_interval
                )

                if should_flush:
                    # Flush buffer
                    if buffer:
                        yield "".join(buffer)
                        buffer = []
                        last_flush = current_time

            # Flush remaining
            if buffer:
                yield "".join(buffer)

        except Exception as e:
            logger.error("Buffered streaming failed", error=str(e))
            raise


class StreamingOrchestrator:
    """Orchestrates streaming with retrieval and TTFT optimization.

    Coordinates retrieval, context optimization, and streaming
    to minimize TTFT.
    """

    def __init__(
        self,
        streaming_llm: AsyncStreamingLLM,
        use_buffering: bool = True,
        buffer_size: int = 3,
    ):
        """Initialize streaming orchestrator.

        Args:
            streaming_llm: Async streaming LLM instance
            use_buffering: Enable token buffering
            buffer_size: Buffer size for batching tokens
        """
        self.streaming_llm = streaming_llm
        self.use_buffering = use_buffering
        self.buffer_size = buffer_size

    async def stream_with_retrieval(
        self,
        query: str,
        retrieved_docs: List[Any],
        system_prompt: Optional[str] = None,
    ) -> AsyncIterator[str]:
        """Stream response with retrieved documents.

        Args:
            query: User query
            retrieved_docs: Retrieved context documents
            system_prompt: Optional system prompt

        Yields:
            Response tokens
        """
        # Format context
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            if hasattr(doc, "page_content"):
                content = doc.page_content
            elif hasattr(doc, "content"):
                content = doc.content
            else:
                content = str(doc)

            context_parts.append(f"[Document {i}]:\n{content}\n")

        context = "\n\n".join(context_parts)

        # Stream response
        source = self.streaming_llm.stream_with_context(
            query=query,
            context=context,
            system_prompt=system_prompt,
        )

        if self.use_buffering:
            # Use buffered streaming
            buffered = BufferedAsyncStream(
                source=source,
                buffer_size=self.buffer_size,
            )
            async for chunk in buffered.stream():
                yield chunk
        else:
            # Direct streaming
            async for token in source:
                yield token


async def create_streaming_response(
    llm: ChatOllama,
    query: str,
    context_docs: List[Any],
    system_prompt: Optional[str] = None,
    use_buffering: bool = True,
) -> AsyncIterator[str]:
    """Create a streaming response with optimal TTFT.

    This is a convenience function for creating streaming responses
    with all optimizations enabled.

    Args:
        llm: ChatOllama instance
        query: User query
        context_docs: Retrieved context documents
        system_prompt: Optional system prompt
        use_buffering: Enable token buffering

    Yields:
        Response tokens
    """
    streaming_llm = AsyncStreamingLLM(llm=llm)

    orchestrator = StreamingOrchestrator(
        streaming_llm=streaming_llm,
        use_buffering=use_buffering,
    )

    async for chunk in orchestrator.stream_with_retrieval(
        query=query,
        retrieved_docs=context_docs,
        system_prompt=system_prompt,
    ):
        yield chunk
