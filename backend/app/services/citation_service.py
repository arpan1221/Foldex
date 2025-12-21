"""LangChain-powered citation management service."""

from typing import Optional, Dict, Any, List
import structlog

from app.core.exceptions import ProcessingError
from app.rag.citation_chains import CitationChain, ConversationalCitationChain
from app.rag.source_tracking import SourceTracker
from app.rag.citation_callbacks import CitationCallbackHandler
from app.rag.llm_chains import OllamaLLM
from app.rag.prompt_management import PromptManager, get_prompt_manager

try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.schema import BaseRetriever, Document
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        BaseRetriever = None
        Document = None

logger = structlog.get_logger(__name__)


class CitationFormatter:
    """Formats citations in simple style."""

    @staticmethod
    def format_simple(
        citation: Dict[str, Any],
    ) -> str:
        """Format citation in simple style.

        Args:
            citation: Citation dictionary

        Returns:
            Simple-formatted citation string
        """
        file_name = citation.get("file_name", "Unknown")
        page_number = citation.get("page_number")
        chunk_id = citation.get("chunk_id")

        parts = [file_name]
        if page_number:
            parts.append(f"p. {page_number}")
        if chunk_id:
            parts.append(f"chunk {chunk_id[:8]}")

        return f"[{', '.join(parts)}]"


class CitationService:
    """Service for managing citations with LangChain integration."""

    def __init__(
        self,
        retriever: Optional[BaseRetriever] = None,
        llm: Optional[OllamaLLM] = None,
        prompt_manager: Optional[PromptManager] = None,
    ):
        """Initialize citation service.

        Args:
            retriever: Optional LangChain retriever
            llm: Optional Ollama LLM instance
            prompt_manager: Optional prompt manager
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        self.retriever = retriever
        self.llm = llm or OllamaLLM()
        self.prompt_manager = prompt_manager or get_prompt_manager()

        # Initialize components
        self.source_tracker = SourceTracker()
        self.citation_formatter = CitationFormatter()

        # Initialize chains
        self.citation_chain = None
        self.conversational_chain = None

        if retriever:
            self.citation_chain = CitationChain(
                retriever=retriever,
                llm=self.llm,
                prompt_manager=self.prompt_manager,
            )

    async def generate_citations(
        self,
        question: str,
        query_type: Optional[str] = None,
        citation_style: str = "simple",
    ) -> Dict[str, Any]:
        """Generate citations for a question.

        Args:
            question: User question
            query_type: Optional query type
            citation_style: Citation style (always uses simple format)

        Returns:
            Dictionary with answer and formatted citations

        Raises:
            ProcessingError: If citation generation fails
        """
        try:
            if not self.citation_chain:
                raise ProcessingError("Citation chain not initialized")

            # Invoke chain
            result = await self.citation_chain.invoke(
                question=question,
                query_type=query_type,
            )

            # Format citations
            formatted_citations = []
            for citation in result.get("citations", []):
                formatted = self._format_citation(citation, citation_style)
                formatted_citations.append({
                    **citation,
                    "formatted": formatted,
                })

            return {
                "answer": result.get("answer", ""),
                "citations": formatted_citations,
                "source_documents": result.get("source_documents", []),
            }

        except Exception as e:
            logger.error(
                "Citation generation failed",
                error=str(e),
                exc_info=True,
            )
            raise ProcessingError(f"Citation generation failed: {str(e)}") from e

    def _format_citation(
        self,
        citation: Dict[str, Any],
        style: str,
    ) -> str:
        """Format citation in simple style.

        Args:
            citation: Citation dictionary
            style: Citation style (ignored, always uses simple)

        Returns:
            Formatted citation string
        """
        return self.citation_formatter.format_simple(citation)

    def assess_citation_confidence(
        self,
        citations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """Assess confidence and reliability of citations.

        Args:
            citations: List of citation dictionaries

        Returns:
            Dictionary with confidence assessment
        """
        try:
            if not citations:
                return {
                    "average_confidence": 0.0,
                    "high_confidence_count": 0,
                    "medium_confidence_count": 0,
                    "low_confidence_count": 0,
                    "reliability_scores": [],
                }

            reliability_scores = [
                citation.get("reliability", 0.5) for citation in citations
            ]

            average_confidence = sum(reliability_scores) / len(reliability_scores)

            high_confidence = sum(1 for score in reliability_scores if score >= 0.7)
            medium_confidence = sum(
                1 for score in reliability_scores if 0.4 <= score < 0.7
            )
            low_confidence = sum(1 for score in reliability_scores if score < 0.4)

            return {
                "average_confidence": average_confidence,
                "high_confidence_count": high_confidence,
                "medium_confidence_count": medium_confidence,
                "low_confidence_count": low_confidence,
                "reliability_scores": reliability_scores,
            }

        except Exception as e:
            logger.error("Citation confidence assessment failed", error=str(e))
            return {
                "average_confidence": 0.0,
                "high_confidence_count": 0,
                "medium_confidence_count": 0,
                "low_confidence_count": 0,
                "reliability_scores": [],
            }

    def filter_citations_by_confidence(
        self,
        citations: List[Dict[str, Any]],
        min_confidence: float = 0.5,
    ) -> List[Dict[str, Any]]:
        """Filter citations by confidence threshold.

        Args:
            citations: List of citation dictionaries
            min_confidence: Minimum confidence threshold

        Returns:
            Filtered list of citations
        """
        return [
            citation
            for citation in citations
            if citation.get("reliability", 0.0) >= min_confidence
        ]

    def set_retriever(self, retriever: BaseRetriever) -> None:
        """Set retriever for citation service.

        Args:
            retriever: LangChain retriever instance
        """
        self.retriever = retriever
        self.citation_chain = CitationChain(
            retriever=retriever,
            llm=self.llm,
            prompt_manager=self.prompt_manager,
        )
        logger.info("Updated citation chain with new retriever")


# Global citation service instance
_citation_service: Optional[CitationService] = None


def get_citation_service(
    retriever: Optional[BaseRetriever] = None,
    llm: Optional[OllamaLLM] = None,
) -> CitationService:
    """Get global citation service instance.

    Args:
        retriever: Optional retriever instance
        llm: Optional LLM instance

    Returns:
        CitationService instance
    """
    global _citation_service
    if _citation_service is None:
        _citation_service = CitationService(retriever=retriever, llm=llm)
    elif retriever and _citation_service.retriever != retriever:
        _citation_service.set_retriever(retriever)
    return _citation_service

