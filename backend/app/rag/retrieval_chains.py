"""LangChain RetrievalQA chains with custom retrievers."""

from typing import Optional, Dict, Any, List, Callable
import re
import json
import structlog

try:
    from langchain.chains import RetrievalQA
    from langchain.chains.question_answering import load_qa_chain
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.runnables import RunnablePassthrough, RunnableParallel
    from langchain_core.output_parsers import StrOutputParser
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback for older LangChain versions
    try:
        from langchain.chains import RetrievalQA
        from langchain.chains.question_answering import load_qa_chain
        from langchain.schema import BaseRetriever, Document
        from langchain_core.runnables import RunnablePassthrough, RunnableParallel
        from langchain_core.output_parsers import StrOutputParser
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        RetrievalQA = None
        load_qa_chain = None
        BaseRetriever = None
        Document = None
        RunnablePassthrough = None
        RunnableParallel = None
        StrOutputParser = None

from app.core.exceptions import ProcessingError
from app.rag.prompt_management import PromptManager, get_prompt_manager
from app.rag.llm_chains import OllamaLLM
from app.rag.ttft_optimization import get_ttft_optimizer
from app.rag.inline_citation_system import get_citation_extractor
from app.rag.granular_citations import get_granular_citation_extractor
from app.services.debug_service import DebugMetrics

logger = structlog.get_logger(__name__)


def clean_response(text: str) -> str:
    """Clean LLM response by removing thinking tags and unwanted content.

    Args:
        text: Raw LLM response

    Returns:
        Cleaned response text
    """
    if not text:
        return text

    # Remove query understanding JSON objects that might leak from classification
    # Pattern: {...} at start of text, possibly repeated
    # Try to remove JSON objects at the beginning
    text_stripped = text.strip()
    json_removed = False
    
    # Look for JSON objects at the start (query understanding format)
    while text_stripped.startswith('{'):
        try:
            # Try to find the end of the JSON object
            brace_count = 0
            end_idx = -1
            for i, char in enumerate(text_stripped):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end_idx = i + 1
                        break
            
            if end_idx > 0:
                # Check if it looks like query understanding JSON
                json_str = text_stripped[:end_idx]
                try:
                    parsed = json.loads(json_str)
                    # Check if it has query understanding fields
                    if "query_type" in parsed and "confidence" in parsed:
                        # This is query understanding JSON, remove it
                        text_stripped = text_stripped[end_idx:].strip()
                        json_removed = True
                        continue
                except json.JSONDecodeError:
                    pass
            break
        except Exception:
            break
    
    if json_removed:
        text = text_stripped

    # If a thinking block is present, drop any preface before it to avoid leaking
    # meta-reasoning like "Hmm, the user wants..." that some models emit.
    low = text.lower()
    think_idx = low.find("<think>")
    antml_idx = low.find("<antml:thinking>")
    candidates = [i for i in (think_idx, antml_idx) if i != -1]
    if candidates:
        text = text[min(candidates) :]

    # Remove thinking tags and their content
    # Match <think>...</think> or </think> tags
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'</?think>', '', text, flags=re.IGNORECASE)
    text = re.sub(r'</?antml:thinking>', '', text, flags=re.IGNORECASE)

    # Remove excessive markdown formatting that wasn't requested
    # Only if the response starts with it (likely unwanted structure)
    if text.strip().startswith('#'):
        lines = text.strip().split('\n')
        # Remove leading header if it's the only header
        if lines and lines[0].startswith('#') and sum(1 for line in lines if line.startswith('#')) == 1:
            text = '\n'.join(lines[1:]).strip()

    # Remove common meta-commentary phrases
    meta_phrases = [
        r'Based on the (?:context|document|file|provided information)',
        r'According to the (?:context|document|file|provided information)',
        r'The (?:context|document|file) (?:states|mentions|says|indicates)',
        r'From the (?:context|document|file|provided information)',
        r'In (?:the )?(?:context|document|file)',
    ]
    for phrase in meta_phrases:
        text = re.sub(phrase + r',?\s*', '', text, flags=re.IGNORECASE)

    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
    text = text.strip()

    return text


class RetrievalQAChain:
    """LangChain RetrievalQA chain with custom retriever and prompt management."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: Optional[OllamaLLM] = None,
        prompt_manager: Optional[PromptManager] = None,
        chain_type: str = "stuff",
        return_source_documents: bool = True,
        enable_ttft_optimization: bool = True,
    ):
        """Initialize RetrievalQA chain.

        Args:
            retriever: LangChain retriever instance
            llm: Ollama LLM instance
            prompt_manager: Prompt manager instance
            chain_type: Chain type ("stuff", "map_reduce", "refine", "map_rerank")
            return_source_documents: Whether to return source documents
            enable_ttft_optimization: Enable TTFT optimizations

        Raises:
            ProcessingError: If LangChain is not available or initialization fails
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        self.retriever = retriever
        self.llm = llm or OllamaLLM()
        self.prompt_manager = prompt_manager or get_prompt_manager()
        self.chain_type = chain_type
        self.return_source_documents = return_source_documents
        self.enable_ttft_optimization = enable_ttft_optimization

        # TTFT optimization
        self.ttft_optimizer = get_ttft_optimizer() if enable_ttft_optimization else None

        # Initialize chain
        self.chain = None
        self._initialize_chain()

    def _initialize_chain(self) -> None:
        """Initialize retrieval chain using LCEL for better streaming."""
        try:
            # Get default prompt
            prompt = self.prompt_manager.get_prompt("default")

            # Create format_docs function with TTFT optimization
            def format_docs_factory(query_context=None):
                """Factory to create format_docs with query context."""
                def format_docs(docs):
                    # Apply TTFT context optimization if enabled
                    optimized_docs = docs
                    if self.ttft_optimizer and query_context:
                        optimized_docs = self.ttft_optimizer.optimize_context(docs, query_context)
                    # Use prompt manager's context formatting for consistent, file-grouped output
                    return self.prompt_manager.format_context(list(optimized_docs))
                return format_docs

            # Define the LCEL chain
            # We use RunnableParallel to get both the answer and the source documents
            self.chain = RunnableParallel({
                "context": self.retriever | format_docs_factory(),
                "question": RunnablePassthrough(),
                "source_documents": self.retriever
            }) | {
                "answer": (
                    RunnablePassthrough.assign(
                        context=lambda x: x["context"],
                        question=lambda x: x["question"]
                    ) | prompt | self.llm.get_llm() | StrOutputParser()
                ),
                "source_documents": lambda x: x["source_documents"]
            }

            logger.info(
                "Initialized LCEL retrieval chain",
                chain_type="lcel",
                ttft_optimization=self.enable_ttft_optimization,
            )

        except Exception as e:
            logger.error(
                "Failed to initialize LCEL retrieval chain",
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(
                f"Failed to initialize retrieval chain: {str(e)}"
            ) from e

    async def invoke(
        self,
        query: str,
        query_type: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        citations_callback: Optional[Callable[[list], None]] = None,
        callbacks: Optional[List] = None,
        debug_metrics: Optional[DebugMetrics] = None,
    ) -> Dict[str, Any]:
        """Invoke chain with query.

        Args:
            query: User query
            query_type: Optional query type for prompt selection
            streaming_callback: Optional callback for streaming tokens
            status_callback: Optional callback for status updates
            citations_callback: Optional callback for progressive citations
            callbacks: Optional list of LangChain callbacks for observability
            debug_metrics: Optional debug metrics collector

        Returns:
            Dictionary with answer and source documents

        Raises:
            ProcessingError: If chain invocation fails
        """
        try:
            # TTFT Optimization: Create optimized format_docs function
            def format_docs(docs):
                # Check for empty retrieval
                if not docs or len(docs) == 0:
                    logger.warning(
                        "Retriever returned empty results - LLM will have no context",
                        query=query[:100],
                        retriever_type=type(self.retriever).__name__,
                    )
                    # Return a message that indicates no context, but don't break the chain
                    # The LLM should handle this gracefully
                    return "No relevant documents were retrieved for this query. Please answer based on general knowledge or indicate that you don't have specific information about this query."
                
                # Apply context optimization if enabled
                if self.ttft_optimizer:
                    docs = self.ttft_optimizer.optimize_context(docs, query)
                    logger.debug(
                        "Applied TTFT context optimization",
                        original_count=len(docs),
                        optimized_count=len(docs)
                    )
                # Use prompt manager's context formatting for consistent, file-grouped output
                return self.prompt_manager.format_context(list(docs))

            # Update prompt if query type is different
            if query_type and query_type != "default":
                prompt = self.prompt_manager.get_prompt(query_type)

                # Reinitialize chain with new prompt
                self.chain = RunnableParallel({
                    "context": self.retriever | format_docs,
                    "question": RunnablePassthrough(),
                    "source_documents": self.retriever
                }) | {
                    "answer": (
                        RunnablePassthrough.assign(
                            context=lambda x: x["context"],
                            question=lambda x: x["question"]
                        ) | prompt | self.llm.get_llm() | StrOutputParser()
                    ),
                    "source_documents": lambda x: x["source_documents"]
                }

            # Set up streaming if callback provided
            callback_list = callbacks or []
            if streaming_callback:
                from app.rag.llm_chains import StreamingCallbackHandler
                callback_list.append(StreamingCallbackHandler(streaming_callback))
                logger.info("Streaming callback registered", callback_count=len(callback_list))
            else:
                # This is expected behavior for non-streaming requests - use debug level
                logger.debug("No streaming callback provided - using non-streaming mode")

            # Send "Retrieving context..." status before retrieval
            if status_callback:
                status_callback("Retrieving context...")

            # Get query embedding for debug if enabled
            if debug_metrics:
                try:
                    # Try multiple paths to access embeddings
                    embeddings = None
                    
                    # Path 1: Direct vector_store on retriever (AdaptiveRetriever)
                    if hasattr(self.retriever, "vector_store") and hasattr(self.retriever.vector_store, "embeddings"):
                        embeddings = self.retriever.vector_store.embeddings
                    # Path 2: Through _vector_retriever (AdaptiveRetriever's internal retriever)
                    elif hasattr(self.retriever, "_vector_retriever") and hasattr(self.retriever._vector_retriever, "vector_store"):
                        if hasattr(self.retriever._vector_retriever.vector_store, "embeddings"):
                            embeddings = self.retriever._vector_retriever.vector_store.embeddings
                    # Path 3: Direct vectorstore attribute (legacy)
                    elif hasattr(self.retriever, "vectorstore") and hasattr(self.retriever.vectorstore, "embeddings"):
                        embeddings = self.retriever.vectorstore.embeddings
                    
                    if embeddings and hasattr(embeddings, "embed_query"):
                        query_embedding = embeddings.embed_query(query)
                        debug_metrics.set_query_embedding(query_embedding)
                    else:
                        logger.debug("Could not find embeddings for debug - skipping query embedding")
                except Exception as e:
                    logger.warning("Failed to get query embedding for debug", error=str(e), exc_info=True)

            # Start LLM timing if debug enabled
            if debug_metrics:
                debug_metrics.start_llm()

            # Invoke chain using astream_events for proper token streaming
            if self.chain is None:
                raise ProcessingError("Chain is not initialized")

            final_result = {
                "answer": "",
                "source_documents": []
            }

            citations_sent = False
            context_formatted = False
            llm_started = False

            # Use astream_events for proper token-level streaming
            # This API properly fires on_llm_new_token events
            try:
                async for event in self.chain.astream_events(
                    query, 
                    config={"callbacks": callback_list},
                    version="v1"
                ):
                    event_type = event.get("event")
                    
                    # Handle retriever events (source documents)
                    if event_type == "on_retriever_end":
                        documents = event.get("data", {}).get("output", [])
                        # Ensure documents is a list
                        if isinstance(documents, list) and documents:
                            if not final_result["source_documents"]:
                                final_result["source_documents"] = documents
                            else:
                                # Merge if we already have some
                                final_result["source_documents"].extend(documents)
                        elif not isinstance(documents, list) and documents:
                            # If it's not a list, try to convert
                            logger.warning("Retriever output is not a list", type=type(documents).__name__)
                            # Convert to list if it's a single document or other iterable
                            try:
                                if hasattr(documents, 'metadata') or hasattr(documents, 'page_content'):
                                    # It's a single Document object
                                    doc_list = [documents]
                                elif isinstance(documents, dict):
                                    # Dict might contain documents under various keys
                                    if "documents" in documents:
                                        doc_list = documents["documents"] if isinstance(documents["documents"], list) else []
                                    elif "results" in documents:
                                        doc_list = documents["results"] if isinstance(documents["results"], list) else []
                                    elif "source_documents" in documents:
                                        doc_list = documents["source_documents"] if isinstance(documents["source_documents"], list) else []
                                    else:
                                        # If dict has values that are Documents, extract them
                                        doc_list = [v for v in documents.values() if hasattr(v, 'metadata') or hasattr(v, 'page_content')]
                                        if not doc_list:
                                            doc_list = []
                                elif isinstance(documents, tuple):
                                    # Try to convert tuple
                                    doc_list = list(documents) if documents else []
                                else:
                                    doc_list = []
                                
                                if doc_list and not final_result["source_documents"]:
                                    final_result["source_documents"] = doc_list
                                
                                # Collect debug metrics for retrieved chunks
                                if debug_metrics and doc_list:
                                    for doc in doc_list:
                                        metadata = doc.metadata if hasattr(doc, "metadata") else {}
                                        score = metadata.get("relevance_score") or metadata.get("score")
                                        debug_metrics.add_retrieved_chunk(doc, score=score, metadata=metadata)
                                
                                # Send progressive citations
                                if not citations_sent and citations_callback and doc_list:
                                    temp_citations = []
                                    for doc in doc_list:
                                        metadata = doc.metadata if hasattr(doc, "metadata") else {}
                                        temp_citations.append({
                                        "file_id": metadata.get("file_id"),
                                        "file_name": metadata.get("file_name", "Unknown"),
                                        "chunk_id": metadata.get("chunk_id"),
                                        "page_number": metadata.get("page_number"),
                                        "chunk_index": metadata.get("chunk_index"),
                                        "start_time": metadata.get("start_time"),
                                        "end_time": metadata.get("end_time"),
                                        "relevance_score": metadata.get("relevance_score"),
                                        "content_preview": (
                                            doc.page_content[:200]
                                            if hasattr(doc, "page_content")
                                            else str(doc)[:200]
                                        ),
                                        "source": metadata.get("source"),
                                        "file_path": metadata.get("file_path"),
                                        "mime_type": metadata.get("mime_type"),
                                        "metadata": metadata,
                                    })
                                
                                if citations_callback:
                                    try:
                                        citations_callback(temp_citations)
                                        logger.info("Progressive citations sent", count=len(temp_citations))
                                    except Exception as e:
                                        logger.warning("Failed to send progressive citations", error=str(e))
                                citations_sent = True
                                
                                # Send "Generating response..." status
                                if status_callback:
                                    status_callback("Generating response...")
                            except Exception as convert_error:
                                logger.warning(
                                    "Failed to convert retriever output to list",
                                    error=str(convert_error),
                                    type=type(documents).__name__
                                )
                    
                    # Handle LLM start event
                    elif event_type == "on_chat_model_start":
                        if not llm_started:
                            llm_started = True
                            logger.debug("LLM generation started via astream_events")
                    
                    # Handle LLM token streaming (this is the key event!)
                    elif event_type == "on_chat_model_stream":
                        chunk_data = event.get("data", {}).get("chunk")
                        if chunk_data:
                            # Extract token from chunk
                            token = ""
                            if hasattr(chunk_data, "content"):
                                token = chunk_data.content
                            elif isinstance(chunk_data, dict) and "content" in chunk_data:
                                token = chunk_data["content"]
                            
                            if token:
                                final_result["answer"] += token
                                # Explicitly call streaming callback to ensure it's invoked
                                # The StreamingCallbackHandler should also fire, but this ensures it
                                if streaming_callback:
                                    try:
                                        streaming_callback(token)
                                        logger.debug("Streamed token via explicit callback", token_length=len(token))
                                    except Exception as e:
                                        logger.warning("Streaming callback error", error=str(e))
                    
                    # Also handle on_llm_new_token for compatibility
                    elif event_type == "on_llm_new_token":
                        token = event.get("data", {}).get("token", "")
                        if token:
                            final_result["answer"] += token
                            if streaming_callback:
                                try:
                                    streaming_callback(token)
                                except Exception as e:
                                    logger.warning("Streaming callback error", error=str(e))
                    
                    # Handle chain end (final output)
                    elif event_type == "on_chain_end":
                        output = event.get("data", {}).get("output", {})
                        
                        # Handle different output structures
                        if isinstance(output, dict):
                            # Store context for debug if available
                            if not context_formatted and "context" in output:
                                context_formatted = True
                                if debug_metrics:
                                    context_text = output["context"]
                                    token_count = len(context_text) // 4
                                    debug_metrics.set_final_context(context_text, token_count)
                            
                            # Capture final answer if not streamed
                            if "answer" in output and not final_result["answer"]:
                                final_result["answer"] = output["answer"]
                            
                            # Capture source documents if not already captured
                            if "source_documents" in output:
                                docs = output["source_documents"]
                                # Ensure it's a list
                                if isinstance(docs, list):
                                    if not final_result["source_documents"]:
                                        final_result["source_documents"] = docs
                                    else:
                                        # Merge if we already have some
                                        final_result["source_documents"].extend(docs)
                                elif isinstance(docs, dict):
                                    # Handle dict case - try to extract documents
                                    logger.warning("source_documents is a dict in on_chain_end", keys=list(docs.keys())[:5])
                                    # This will be handled in the final cleanup step
                        elif isinstance(output, list):
                            # Sometimes output is a list of documents
                            if not final_result["source_documents"]:
                                final_result["source_documents"] = output
                            
            except Exception as stream_error:
                logger.warning(
                    "astream_events failed, falling back to astream",
                    error=str(stream_error)
                )
                # Fallback to original astream approach
                async for chunk in self.chain.astream(query, config={"callbacks": callback_list}):
                    if "context" in chunk and not context_formatted:
                        if debug_metrics:
                            context_text = chunk["context"]
                            token_count = len(context_text) // 4
                            debug_metrics.set_final_context(context_text, token_count)
                        context_formatted = True

                    if "source_documents" in chunk and chunk["source_documents"]:
                        final_result["source_documents"].extend(chunk["source_documents"])
                        
                        if debug_metrics:
                            for doc in chunk["source_documents"]:
                                metadata = doc.metadata if hasattr(doc, "metadata") else {}
                                score = metadata.get("relevance_score") or metadata.get("score")
                                debug_metrics.add_retrieved_chunk(doc, score=score, metadata=metadata)

                        if not citations_sent and citations_callback:
                            temp_citations = []
                            for doc in chunk["source_documents"]:
                                metadata = doc.metadata if hasattr(doc, "metadata") else {}
                                temp_citations.append({
                                    "file_id": metadata.get("file_id"),
                                    "file_name": metadata.get("file_name", "Unknown"),
                                    "chunk_id": metadata.get("chunk_id"),
                                    "page_number": metadata.get("page_number"),
                                    "chunk_index": metadata.get("chunk_index"),
                                    "start_time": metadata.get("start_time"),
                                    "end_time": metadata.get("end_time"),
                                    "relevance_score": metadata.get("relevance_score"),
                                    "content_preview": (
                                        doc.page_content[:200]
                                        if hasattr(doc, "page_content")
                                        else str(doc)[:200]
                                    ),
                                    "source": metadata.get("source"),
                                    "file_path": metadata.get("file_path"),
                                    "mime_type": metadata.get("mime_type"),
                                    "metadata": metadata,
                                })

                            citations_callback(temp_citations)
                            citations_sent = True

                            if status_callback:
                                status_callback("Generating response...")

                    if "answer" in chunk:
                        token = chunk["answer"]
                        final_result["answer"] += token
                        # Explicitly call streaming callback in fallback path
                        if streaming_callback:
                            try:
                                streaming_callback(token)
                            except Exception as e:
                                logger.warning("Streaming callback error in fallback", error=str(e))

            # End LLM timing if debug enabled
            if debug_metrics:
                debug_metrics.end_llm()

            # Clean the response to remove thinking tags and unwanted content
            cleaned_answer = clean_response(final_result["answer"])

            # Store LLM response for debug
            if debug_metrics:
                debug_metrics.set_llm_response(final_result["answer"], cleaned_answer)

            # Ensure source_documents is a list
            source_docs = final_result.get("source_documents", [])
            if not isinstance(source_docs, list):
                logger.warning(
                    "source_documents is not a list, converting",
                    type=type(source_docs).__name__,
                    value=str(source_docs)[:200] if source_docs else "None"
                )
                # Try to extract documents from dict structure
                if isinstance(source_docs, dict):
                    # Maybe it's wrapped in a dict - try common keys
                    if "source_documents" in source_docs:
                        source_docs = source_docs["source_documents"]
                    elif "documents" in source_docs:
                        source_docs = source_docs["documents"]
                    else:
                        # Try to find any list values in the dict
                        for key, value in source_docs.items():
                            if isinstance(value, list) and value:
                                # Check if it looks like documents
                                if len(value) > 0 and (hasattr(value[0], 'page_content') or hasattr(value[0], 'metadata')):
                                    source_docs = value
                                    logger.info(f"Found source_documents in dict key: {key}")
                                    break
                
                # Final check - if still not a list, set to empty
                if not isinstance(source_docs, list):
                    source_docs = []
            
            # Extract granular citations from the response
            granular_extractor = get_granular_citation_extractor(min_quote_similarity=0.5)
            try:
                cleaned_answer, granular_citations = granular_extractor.extract_citations_with_quotes(
                    cleaned_answer,
                    source_docs,
                    include_context=True
                )

                # Format citations for UI
                used_citations = []
                for citation in granular_citations:
                    formatted = granular_extractor.format_citation_for_ui(
                        citation,
                        format_type="detailed"
                    )
                    # Merge with original citation data
                    used_citations.append({**citation, **formatted})

                logger.info(
                    "Granular citations extracted",
                    citation_count=len(used_citations),
                    avg_confidence=sum(c.get("quote_confidence", 0) for c in used_citations) / len(used_citations) if used_citations else 0,
                )
            except Exception as citation_error:
                logger.error(
                    "Granular citation extraction failed, falling back to basic",
                    error=str(citation_error),
                    source_docs_count=len(source_docs),
                    exc_info=True
                )
                # Fallback to basic citation extraction
                citation_extractor = get_citation_extractor()
                try:
                    cleaned_answer, used_citations = citation_extractor.extract_used_citations(
                        cleaned_answer,
                        source_docs
                    )
                except Exception:
                    used_citations = []

            logger.info(
                "Chain streaming completed",
                query_length=len(query),
                answer_length=len(cleaned_answer),
                source_count=len(source_docs),
                cited_sources=len(used_citations),
            )

            return {
                "answer": cleaned_answer,
                "source_documents": source_docs,
                "citations": used_citations,  # Add extracted citations
                "query_type": query_type,
            }

        except Exception as e:
            # Record error in debug metrics if enabled
            if debug_metrics:
                debug_metrics.set_error(str(e))
                if debug_metrics.llm_start_time and not debug_metrics.llm_end_time:
                    debug_metrics.end_llm()
                if debug_metrics.retrieval_start_time and not debug_metrics.retrieval_end_time:
                    debug_metrics.end_retrieval()
            
            logger.error(
                "Chain invocation failed",
                query=query[:100],
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(f"Chain invocation failed: {str(e)}") from e

    def get_chain(self) -> RetrievalQA:
        """Get the underlying LangChain RetrievalQA chain.

        Returns:
            RetrievalQA chain instance
        """
        return self.chain


class HybridRetrievalChain:
    """Hybrid retrieval chain combining multiple retrieval strategies."""

    def __init__(
        self,
        retriever: BaseRetriever,
        llm: Optional[OllamaLLM] = None,
        prompt_manager: Optional[PromptManager] = None,
        chain_type: str = "stuff",
    ):
        """Initialize hybrid retrieval chain.

        Args:
            retriever: Hybrid retriever instance
            llm: Ollama LLM instance
            prompt_manager: Prompt manager instance
            chain_type: Chain type
        """
        self.retrieval_qa_chain = RetrievalQAChain(
            retriever=retriever,
            llm=llm,
            prompt_manager=prompt_manager,
            chain_type=chain_type,
        )

    async def invoke(
        self,
        query: str,
        query_type: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
        status_callback: Optional[Callable[[str], None]] = None,
        citations_callback: Optional[Callable[[list], None]] = None,
        callbacks: Optional[List] = None,
        debug_metrics: Optional[DebugMetrics] = None,
    ) -> Dict[str, Any]:
        """Invoke hybrid retrieval chain.

        Args:
            query: User query
            query_type: Optional query type
            streaming_callback: Optional streaming callback
            status_callback: Optional callback for status updates
            citations_callback: Optional callback for progressive citations
            callbacks: Optional list of LangChain callbacks for observability
            debug_metrics: Optional debug metrics collector

        Returns:
            Dictionary with answer and sources
        """
        return await self.retrieval_qa_chain.invoke(
            query=query,
            query_type=query_type,
            streaming_callback=streaming_callback,
            status_callback=status_callback,
            citations_callback=citations_callback,
            callbacks=callbacks,
            debug_metrics=debug_metrics,
        )

    def _extract_citations(self, source_documents: List[Document]) -> List[Dict[str, Any]]:
        """Extract citation information from source documents.

        Args:
            source_documents: List of source Document objects

        Returns:
            List of citation dictionaries with metadata
        """
        citations = []

        for doc in source_documents:
            metadata = doc.metadata if hasattr(doc, "metadata") else {}

            citation = {
                "file_id": metadata.get("file_id"),
                "file_name": metadata.get("file_name", "Unknown"),
                "chunk_id": metadata.get("chunk_id"),
                "page_number": metadata.get("page_number"),
                "chunk_index": metadata.get("chunk_index"),
                "start_time": metadata.get("start_time"),
                "end_time": metadata.get("end_time"),
                "content_preview": doc.page_content[:200] if hasattr(doc, "page_content") else str(doc)[:200],
                "metadata": metadata,
            }

            citations.append(citation)

        return citations

