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
from app.config.settings import settings
from app.rag.prompt_management import PromptManager, get_prompt_manager
from app.rag.llm_chains import OllamaLLM
from app.rag.ttft_optimization import get_ttft_optimizer
from app.rag.inline_citation_system import get_citation_extractor
from app.rag.granular_citations import get_granular_citation_extractor
from app.services.debug_service import DebugMetrics

logger = structlog.get_logger(__name__)


def clean_response(text: str) -> str:
    """Clean LLM response by removing thinking tags and unwanted content.

    Enhanced cleaning for file-specific chats and general queries to ensure
    clean, professional responses without thinking tags, meta-commentary, or
    unwanted artifacts.

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
    think_simple_idx = low.find("<think>")
    candidates = [i for i in (think_idx, antml_idx, think_simple_idx) if i != -1]
    if candidates:
        text = text[min(candidates) :]

    # Remove thinking tags and their content (comprehensive pattern matching)
    # Match various thinking tag formats
    thinking_patterns = [
        r'<think>.*?</think>',
        r'<think>.*?</think>',
        r'<antml:thinking>.*?</antml:thinking>',
        r'<reasoning>.*?</reasoning>',
        r'</?think[^>]*>',
        r'</?antml:thinking[^>]*>',
        r'</?redacted_reasoning[^>]*>',
    ]
    for pattern in thinking_patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)

    # Remove unwanted prefatory phrases that models sometimes add
    unwanted_prefixes = [
        r'^I\'m (?:happy to help|ready to assist|here to help).*?[.!?]\s*',
        r'^I don\'t see (?:a )?(?:specific )?(?:question|context).*?[.!?]\s*',
        r'^However, I don\'t see.*?[.!?]\s*',
        r'^Unfortunately, (?:I don\'t|there is no).*?[.!?]\s*',
        r'^I\'m ready to help.*?[.!?]\s*',
        r'^Please provide.*?[.!?]\s*',
    ]
    for prefix in unwanted_prefixes:
        text = re.sub(prefix, '', text, flags=re.IGNORECASE | re.MULTILINE)

    # Remove excessive markdown formatting that wasn't requested
    # Only if the response starts with it (likely unwanted structure)
    if text.strip().startswith('#'):
        lines = text.strip().split('\n')
        # Remove leading header if it's the only header
        if lines and lines[0].startswith('#') and sum(1 for line in lines if line.startswith('#')) == 1:
            text = '\n'.join(lines[1:]).strip()

    # Remove common meta-commentary phrases (enhanced list)
    meta_phrases = [
        r'Based on the (?:context|document|file|provided information|information provided)',
        r'According to the (?:context|document|file|provided information|information provided)',
        r'The (?:context|document|file) (?:states|mentions|says|indicates|shows|reveals)',
        r'From the (?:context|document|file|provided information|information provided)',
        r'In (?:the )?(?:context|document|file|provided information)',
        r'As (?:mentioned|stated|indicated|shown) in the (?:context|document|file)',
        r'Looking at the (?:context|document|file|information)',
        r'According to this (?:context|document|file)',
    ]
    for phrase in meta_phrases:
        # Remove phrase with optional comma/colon and following whitespace
        text = re.sub(phrase + r'[,:]?\s*', '', text, flags=re.IGNORECASE)

    # Remove redundant "Regarding X:" prefixes that might be added
    # This can happen in file-specific chats where the system adds context
    text = re.sub(r'^Regarding [^:]+:\s*', '', text, flags=re.IGNORECASE | re.MULTILINE)

    # Remove common closing phrases that are unnecessary
    unwanted_closings = [
        r'\.?\s*Please let me know if you need (?:any )?(?:more )?(?:additional )?(?:information|help|assistance|clarification)\.?$',
        r'\.?\s*Feel free to ask if you have (?:any )?(?:more )?(?:additional )?(?:questions|queries)\.?$',
        r'\.?\s*I hope this helps\.?$',
        r'\.?\s*Let me know if you need anything else\.?$',
    ]
    for closing in unwanted_closings:
        text = re.sub(closing, '', text, flags=re.IGNORECASE | re.MULTILINE)

    # Clean up extra whitespace and normalize line breaks
    text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Max 2 consecutive newlines
    text = re.sub(r'[ \t]+', ' ', text)  # Normalize spaces
    text = re.sub(r' +\n', '\n', text)  # Remove trailing spaces on lines
    text = re.sub(r'\n +', '\n', text)  # Remove leading spaces on lines
    text = text.strip()

    # Remove repetitive/repeating text patterns (common LLM failure mode)
    # Enhanced duplicate detection for sentences and paragraphs
    lines = text.split('\n')
    cleaned_lines = []
    seen_sentences: set[str] = set()
    seen_line_signatures = set()
    
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            cleaned_lines.append(line)
            continue
            
        # Split line into sentences for better duplicate detection
        sentences = re.split(r'[.!?]+', line_stripped)
        cleaned_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence or len(sentence) < 10:
                continue
                
            # Normalize sentence (remove extra spaces, lowercase for comparison)
            sentence_normalized = re.sub(r'\s+', ' ', sentence.lower())
            
            # Check for exact duplicate sentences
            if sentence_normalized in seen_sentences:
                continue
                
            # Check for near-duplicates (80% similarity threshold)
            is_duplicate = False
            for seen in seen_sentences:
                # Simple similarity check: count common words
                seen_words = set(seen.split())
                current_words = set(sentence_normalized.split())
                if len(seen_words) > 0 and len(current_words) > 0:
                    common = seen_words & current_words
                    similarity = len(common) / max(len(seen_words), len(current_words))
                    if similarity > 0.8 and abs(len(seen_words) - len(current_words)) < 5:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                seen_sentences.add(sentence_normalized)
                cleaned_sentences.append(sentence)
                # Keep only last 50 sentences to avoid memory bloat
                if len(seen_sentences) > 50:
                    seen_sentences = set(list(seen_sentences)[-50:])
        
        # Reconstruct line from cleaned sentences
        if cleaned_sentences:
            cleaned_line = '. '.join(cleaned_sentences) + '.' if cleaned_sentences else ''
            # Also check line-level duplicates (first 100 chars as signature)
            line_sig = cleaned_line.lower()[:100]
            if line_sig not in seen_line_signatures:
                seen_line_signatures.add(line_sig)
                cleaned_lines.append(cleaned_line)
                # Keep only last 30 line signatures
                if len(seen_line_signatures) > 30:
                    seen_line_signatures = set(list(seen_line_signatures)[-30:])
        elif not line_stripped:  # Keep empty lines
            cleaned_lines.append(line)
    
    text = '\n'.join(cleaned_lines)

    # Remove garbled/corrupted text patterns first (text with merged words)
    # Pattern: words merged together like "Basedthe" or "documentthat"
    # Fix common merges
    text = re.sub(r'(\w+)([A-Z][a-z]+)', r'\1 \2', text)  # Split merged words
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)  # Split camelCase merges
    text = re.sub(r'(\w)([.!?])(\w)', r'\1\2 \3', text)  # Add space after punctuation
    
    # Remove repetitive word sequences (garbled text pattern)
    # Pattern: repeated words like "After analyzing After analyzing"
    text = re.sub(r'\b(\w+)(\s+\1){2,}\b', r'\1', text, flags=re.IGNORECASE)
    
    # Remove garbled repetitive phrases (enhanced detection)
    # Pattern: "After analyzing all three files After analyzing all three files"
    words = text.split()
    cleaned_words = []
    i = 0
    window_size = 5  # Check for 5-word sequence repeats
    
    while i < len(words):
        if i >= window_size:  # Need some context
            # Check for repeating sequences of 3-5 words
            for seq_len in range(window_size, 2, -1):  # Try longer sequences first (5, 4, 3)
                if i + seq_len <= len(words):
                    current_seq = ' '.join(words[i:i+seq_len]).lower()
                    # Check last 30 words for this sequence
                    lookback_start = max(0, i - 30)
                    recent_text = ' '.join(words[lookback_start:i]).lower()
                    
                    if current_seq in recent_text:
                        # Found repetition, skip this sequence
                        i += seq_len
                        break
            else:
                # No repetition found, add current word
                cleaned_words.append(words[i])
                i += 1
        else:
            # Not enough context yet, add word
            cleaned_words.append(words[i])
            i += 1
    
    text = ' '.join(cleaned_words)
    
    # Remove asterisks which are often corruption markers
    text = re.sub(r'\*{2,}', '', text)  # Remove multiple asterisks
    text = re.sub(r'\*\s*\*', '', text)  # Remove asterisk pairs with spaces
    
    # Be VERY conservative with word corruption fixes - only fix obvious issues
    # Don't remove words that might be valid (e.g., technical terms, proper nouns)
    # Only fix clear corruption patterns like asterisks or obvious jumbles
    
    # Remove corrupted/jumbled text patterns
    # Pattern: text with excessive punctuation, parentheses, or special characters mixed in
    # Examples: "DKSH Smolla Indian ),(Mum Ganbaiesh, arIndpania)"
    # This pattern suggests garbled output
    sentences = re.split(r'[.!?]+', text)
    cleaned_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Check for excessive special characters that suggest garbling
        special_char_ratio = len(re.findall(r'[^\w\s]', sentence)) / max(len(sentence), 1)
        
        # Check for suspicious patterns: parentheses mid-word, excessive commas
        # Also check for word boundary corruption (letters bleeding between words)
        has_suspicious_pattern = (
            re.search(r'\w+[(),]\w+', sentence) or  # Parentheses/comma inside words
            re.search(r'[(),]{2,}', sentence) or  # Multiple consecutive punctuation
            re.search(r'\w{10,}', sentence) or  # Very long words (might be corrupted concatenation)
            special_char_ratio > 0.15  # More than 15% special characters
        )
        
        # Check for excessive word repetition within sentence (garbled pattern)
        words_in_sent = sentence.split()
        if len(words_in_sent) > 5:
            word_counts: dict[str, int] = {}
            for w in words_in_sent:
                word_counts[w.lower()] = word_counts.get(w.lower(), 0) + 1
            max_repeats = max(word_counts.values()) if word_counts else 1
            # If a word appears more than 30% of the time, it's likely garbled
            if max_repeats > len(words_in_sent) * 0.3:
                has_suspicious_pattern = True
        
        # Only skip if it's clearly garbled AND short (don't skip legitimate technical text)
        if has_suspicious_pattern and len(sentence.split()) < 10:
            continue
        
        cleaned_sentences.append(sentence)
    
    text = '. '.join(cleaned_sentences) if cleaned_sentences else text

    # Final pass: remove any remaining empty lines at start/end
    lines = text.split('\n')
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    text = '\n'.join(lines)

    # Final cleanup of excessive repetition
    # Remove paragraphs that are mostly duplicated
    paragraphs = text.split('\n\n')
    unique_paragraphs = []
    seen_paragraphs: set[str] = set()
    
    for para in paragraphs:
        para_stripped = para.strip()
        if not para_stripped:
            continue
            
        para_normalized = re.sub(r'\s+', ' ', para_stripped.lower())
        para_sig = para_normalized[:150]  # First 150 chars as signature
        
        # Check for exact duplicates
        if para_sig and para_sig not in seen_paragraphs:
            # Also check for near-duplicates (very similar paragraphs)
            is_near_duplicate = False
            for seen_sig in seen_paragraphs:
                # Calculate simple similarity
                common_chars = sum(1 for a, b in zip(para_sig, seen_sig) if a == b)
                similarity = common_chars / max(len(para_sig), len(seen_sig), 1)
                if similarity > 0.85:  # 85% similarity threshold
                    is_near_duplicate = True
                    break
            
            if not is_near_duplicate:
                seen_paragraphs.add(para_sig)
                unique_paragraphs.append(para_stripped)
                # Keep only last 40 paragraph signatures
                if len(seen_paragraphs) > 40:
                    seen_paragraphs = set(list(seen_paragraphs)[-40:])
    
    text = '\n\n'.join(unique_paragraphs)

    # Detect repeated opening text blocks (common in corrupted responses)
    # Pattern: "The literature review... The literature review..."
    # BUT: Be very conservative - only truncate if we're CERTAIN it's a duplicate
    # Don't truncate if text is short (might be cutting off valid content)
    if len(text) > 1000:  # Only check for long responses
        # Try to find if text starts with a repeated block
        # Check first 500 chars against rest of text
        first_block = text[:500].lower().strip()
        remaining_text = text[500:].lower().strip()
        
        # Look for first_block appearing early in remaining_text
        # This indicates the response was duplicated
        if len(first_block) > 200 and len(remaining_text) > 200:  # Need substantial blocks
            # Check if first 300 chars of remaining_text match first 300 chars of first_block
            # Use longer signature for more confidence
            first_block_sig = first_block[:300]
            remaining_sig = remaining_text[:300]
            
            # Calculate similarity - require HIGH confidence (90%+) before truncating
            matching_chars = sum(1 for a, b in zip(first_block_sig, remaining_sig) if a == b)
            similarity = matching_chars / max(len(first_block_sig), len(remaining_sig), 1)
            
            # Only truncate if VERY high similarity (90%+) AND we can find a good break point
            if similarity > 0.9:  # 90% similarity - very high confidence required
                # Find a sentence boundary near 500 chars to truncate cleanly
                truncate_point = 500
                # Look for sentence end within 100 chars of 500
                for i in range(500, min(600, len(text))):
                    if text[i] in '.!?':
                        truncate_point = i + 1
                        break
                
                # Only truncate if we found a good break point
                if truncate_point < len(text):
                    text = text[:truncate_point].strip()
                    logger.debug("Detected repeated opening block, truncated to first occurrence", 
                               original_length=len(text), truncated_length=truncate_point)
    
    # Final aggressive cleanup: remove any remaining obvious duplicates at start
    # Sometimes the LLM repeats the entire response
    # BUT: Be very conservative - only remove if we're CERTAIN
    sentences = re.split(r'[.!?]+', text)
    if len(sentences) > 8:  # Only check if we have MANY sentences (8+)
        # Check if first half of sentences is very similar to second half
        mid_point = len(sentences) // 2
        first_half = ' '.join(sentences[:mid_point]).lower()
        second_half = ' '.join(sentences[mid_point:]).lower()
        
        # If second half starts very similarly to first half, it might be a duplicate
        if len(first_half) > 200 and len(second_half) > 200:  # Need substantial halves
            first_words = first_half.split()[:50]  # First 50 words (more thorough check)
            second_words = second_half.split()[:50]
            
            # Check if first 50 words match - require HIGH confidence
            matching_words = sum(1 for a, b in zip(first_words, second_words) if a == b)
            word_similarity = matching_words / max(len(first_words), len(second_words), 1)
            
            # Only remove if VERY high similarity (90%+) - be conservative
            if word_similarity > 0.9:  # 90% of words match = very likely duplicate
                # Second half is a repeat, keep only first half
                text = '. '.join(sentences[:mid_point]) + '.'
                logger.debug("Detected repeated response halves, kept only first half",
                           original_sentences=len(sentences), kept_sentences=mid_point)
    
    # One more pass: remove duplicate sentences at the very beginning
    # Pattern: "Based on the provided resume, X. Based on the provided resume, X."
    text = re.sub(r'^(.*?)(\.\s*)\1(\.)', r'\1\2', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Remove repeated opening phrases more aggressively
    # Pattern: "The literature review of X is... The literature review of X is..."
    # Match longer opening phrases that repeat
    opening_phrase_patterns = [
        r'^(The literature review[^.!?]{0,200}?)(\.\s+)\1',
        r'^(Based on[^.!?]{0,200}?)(\.\s+)\1',
        r'^(The main[^.!?]{0,200}?)(\.\s+)\1',
        r'^(Based on the [^.!?]{0,200}?)(\.\s+)\1',
        r'^(According to the [^.!?]{0,200}?)(\.\s+)\1',
        r'^(The document[^.!?]{0,200}?)(\.\s+)\1',
    ]
    for pattern in opening_phrase_patterns:
        text = re.sub(pattern, r'\1\2', text, flags=re.IGNORECASE | re.DOTALL)
    
    # Final pass: Remove any remaining obvious duplicate phrases
    # Check for patterns like "Based on the README.md file, X. Based on the README.md file, X."
    duplicate_phrase_pattern = r'^(.{50,300}?)(\.\s+)(\1)(\.)'
    while re.search(duplicate_phrase_pattern, text, flags=re.IGNORECASE | re.DOTALL):
        text = re.sub(duplicate_phrase_pattern, r'\1\2', text, flags=re.IGNORECASE | re.DOTALL)

    # Final cleanup: normalize whitespace one more time
    text = re.sub(r'\s+', ' ', text)  # Normalize all whitespace to single spaces
    text = re.sub(r'\n\s*\n', '\n\n', text)  # Normalize line breaks
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
            # Get default prompt (file_id will be checked in invoke method)
            prompt = self.prompt_manager.get_prompt("default")

            # Create format_docs function with TTFT optimization
            # Note: folder_summary is not available at initialization time, will be handled in invoke method
            def format_docs_factory(query_context=None):
                """Factory to create format_docs with query context."""
                def format_docs(docs):
                    # Apply TTFT context optimization if enabled
                    optimized_docs = docs
                    if self.ttft_optimizer and query_context:
                        optimized_docs = self.ttft_optimizer.optimize_context(docs, query_context)
                    # Use prompt manager's context formatting for consistent, file-grouped output
                    # Folder summary will be injected in invoke method's format_docs
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
        folder_summary: Optional[Dict[str, Any]] = None,
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
            folder_summary: Optional folder summary dictionary to inject into context

        Returns:
            Dictionary with answer and source documents

        Raises:
            ProcessingError: If chain invocation fails
        """
        try:
            # Check if file_id filter is set (file-specific chat) - do this early so format_docs can use it
            file_id = None
            if hasattr(self.retriever, "_file_id_filter"):
                file_id = getattr(self.retriever, "_file_id_filter", None)
                if file_id:
                    logger.debug("Detected file-specific query, using file-specific prompt", file_id=file_id)

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
                # Include folder summary if available (skip for file-specific queries)
                # file_id is captured from outer scope
                return self.prompt_manager.format_context(list(docs), folder_summary=folder_summary, file_id=file_id)

            # Update prompt if query type is different or file_id is present
            if (query_type and query_type != "default") or file_id:
                prompt = self.prompt_manager.get_prompt(query_type or "default", file_id=file_id)
            else:
                prompt = self.prompt_manager.get_prompt("default")
            
            # Reinitialize chain with prompt (always reinitialize to ensure format_docs closure captures file_id)
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
            # ChatOllama has native streaming (streaming=True), so StreamingCallbackHandler
            # will handle token streaming via on_llm_new_token callbacks
            callback_list = callbacks or []
            stream_handler = None
            if streaming_callback:
                from app.rag.llm_chains import StreamingCallbackHandler
                stream_handler = StreamingCallbackHandler(streaming_callback)
                callback_list.append(stream_handler)
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
                    
                    # Skip token streaming events - ChatOllama has native streaming (streaming=True)
                    # StreamingCallbackHandler will handle token streaming via on_llm_new_token callbacks
                    # We'll get the full answer from the handler at the end to avoid duplicates
                    # DO NOT manually accumulate tokens here - this causes duplicate handling
                    
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
                            # Only use this if we don't have a handler answer (handler takes precedence)
                            if "answer" in output:
                                output_answer = output["answer"]
                                # Use output answer if it's longer than what we have (more complete)
                                if output_answer and len(output_answer) > len(final_result.get("answer", "")):
                                    final_result["answer"] = output_answer
                                    logger.debug("Using on_chain_end answer", length=len(output_answer))
                                elif not final_result.get("answer"):
                                    # If we have no answer yet, use output answer
                                    final_result["answer"] = output_answer
                                    logger.debug("Using on_chain_end answer (no previous answer)", length=len(output_answer))
                            
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
                        # In fallback astream path, chunks may contain answer segments
                        # StreamingCallbackHandler should still handle token-level streaming via callbacks
                        # Only accumulate here if handler didn't accumulate it
                        answer_segment = chunk["answer"]
                        if answer_segment and not stream_handler:
                            # No handler, so we need to accumulate manually
                            final_result["answer"] += answer_segment
                        # Note: If stream_handler exists, it accumulates tokens via callbacks
                        # and we'll get the full answer from handler.get_full_response() later

            # End LLM timing if debug enabled
            if debug_metrics:
                debug_metrics.end_llm()

            # Get the final answer - prefer StreamingCallbackHandler's accumulated response
            # since ChatOllama streams natively and the handler properly accumulates tokens
            if stream_handler and hasattr(stream_handler, 'get_full_response'):
                handler_answer = stream_handler.get_full_response()
                if handler_answer:
                    # Use handler's answer (properly accumulated without duplicates)
                    final_result["answer"] = handler_answer
                    logger.debug("Using StreamingCallbackHandler accumulated response", length=len(handler_answer))
            
            # Clean the response to remove thinking tags and unwanted content
            # Post-processing is disabled by default for testing (can be enabled via ENABLE_RESPONSE_POST_PROCESSING)
            if settings.ENABLE_RESPONSE_POST_PROCESSING:
                cleaned_answer = clean_response(final_result["answer"])
                logger.debug("Response post-processing enabled, cleaned answer", 
                           original_length=len(final_result["answer"]),
                           cleaned_length=len(cleaned_answer))
            else:
                cleaned_answer = final_result["answer"]
                logger.debug("Response post-processing disabled, using raw answer")

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
        folder_summary: Optional[Dict[str, Any]] = None,
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
            folder_summary: Optional folder summary dictionary to inject into context

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
            folder_summary=folder_summary,
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

