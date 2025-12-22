"""Lightweight document-level summarization for enhanced chunking.

Generates brief document summaries that are included in chunk metadata
to improve retrieval quality for generic queries.
"""

from typing import Dict, Any, Optional, List
import asyncio
import structlog
from app.rag.llm_chains import OllamaLLM

logger = structlog.get_logger(__name__)


class DocumentSummarizer:
    """Generates lightweight document summaries for chunk enhancement."""
    
    def __init__(self, llm: Optional[OllamaLLM] = None):
        """Initialize document summarizer.
        
        Args:
            llm: Optional LLM instance (uses lightweight model by default)
        """
        self.llm = llm or OllamaLLM(model="llama3.2:1b", temperature=0.1)
        self.lightweight_llm = self.llm.get_llm()
    
    async def generate_summary(
        self,
        content: str,
        file_name: str,
        file_type: Optional[str] = None,
        max_length: int = 200,
    ) -> str:
        """Generate brief document summary.
        
        Args:
            content: Document content (can be truncated for large docs)
            file_name: Name of the file
            file_type: Optional file type (PDF, Code, etc.)
            max_length: Maximum summary length in characters
            
        Returns:
            Brief summary (1-2 sentences, max 200 chars)
        """
        try:
            # Truncate content if too long (use first 2000 chars for summary)
            content_preview = content[:2000].strip()
            if len(content) > 2000:
                content_preview += "..."
            
            file_type_desc = f" ({file_type})" if file_type else ""
            
            prompt = f"""Summarize this document in exactly one sentence (max {max_length} characters).

File: {file_name}{file_type_desc}

Content preview:
{content_preview}

One-sentence summary (be specific about what the document contains or discusses):"""
            
            response = await asyncio.wait_for(
                self.lightweight_llm.ainvoke(prompt),
                timeout=10.0
            )
            
            summary = response.content.strip() if hasattr(response, "content") else str(response).strip()
            
            # Ensure summary is within max length
            if len(summary) > max_length:
                summary = summary[:max_length-3] + "..."
            
            return summary if summary else f"{file_name} contains relevant content."
            
        except asyncio.TimeoutError:
            logger.warning("Document summary generation timed out", file_name=file_name)
            return f"{file_name} contains relevant content."
        except Exception as e:
            logger.warning(
                "Document summary generation failed, using fallback",
                file_name=file_name,
                error=str(e)
            )
            return f"{file_name} contains relevant content."
    
    def extract_key_terms(self, content: str, max_terms: int = 5) -> List[str]:
        """Extract key terms from document content.
        
        Args:
            content: Document content
            max_terms: Maximum number of terms to extract
            
        Returns:
            List of key terms
        """
        import re
        from collections import Counter
        
        # Extract meaningful words (3+ chars, capitalized or all caps)
        words = re.findall(r'\b[A-Z][a-z]{2,}\b|\b[A-Z]{2,}\b', content)
        
        # Filter common words
        common_words = {
            'The', 'This', 'That', 'These', 'Those', 'There', 'They',
            'From', 'With', 'Have', 'Has', 'Had', 'Will', 'Would',
            'Should', 'Could', 'Can', 'May', 'Might', 'Must',
        }
        
        word_counts = Counter(w for w in words if w not in common_words)
        
        return [term for term, _ in word_counts.most_common(max_terms)]
    
    async def generate_document_context(
        self,
        content: str,
        file_name: str,
        file_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Generate complete document context for chunking.
        
        Args:
            content: Document content
            file_name: Name of the file
            file_type: Optional file type
            
        Returns:
            Dictionary with summary, key_terms, and theme
        """
        summary = await self.generate_summary(content, file_name, file_type)
        key_terms = self.extract_key_terms(content)
        
        return {
            "document_summary": summary,
            "document_key_terms": key_terms,
            "document_theme": key_terms[0] if key_terms else None,
        }


# Global instance
_document_summarizer: Optional[DocumentSummarizer] = None


def get_document_summarizer() -> DocumentSummarizer:
    """Get global document summarizer instance.
    
    Returns:
        DocumentSummarizer instance
    """
    global _document_summarizer
    
    if _document_summarizer is None:
        _document_summarizer = DocumentSummarizer()
    
    return _document_summarizer

