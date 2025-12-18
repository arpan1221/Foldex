"""LangChain evaluation chains for response quality assessment."""

from typing import Optional, Dict, Any, List, Tuple
import structlog

try:
    from langchain.evaluation import CriteriaEvalChain, PairwiseStringEvalChain
    from langchain.evaluation.criteria import Criteria
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    CriteriaEvalChain = None
    PairwiseStringEvalChain = None
    LLMChain = None
    PromptTemplate = None
    Criteria = None

from app.core.exceptions import ProcessingError
from app.rag.llm_chains import OllamaLLM

logger = structlog.get_logger(__name__)


class ResponseEvaluator:
    """LangChain-based response quality evaluator."""

    def __init__(self, llm: Optional[OllamaLLM] = None):
        """Initialize response evaluator.

        Args:
            llm: Optional Ollama LLM instance
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        self.llm = llm or OllamaLLM()
        self.evaluators: Dict[str, Any] = {}
        self._initialize_evaluators()

    def _initialize_evaluators(self) -> None:
        """Initialize LangChain evaluators."""
        try:
            llm_instance = self.llm.get_llm()

            # Initialize CriteriaEvalChain for relevance
            self.evaluators["relevance"] = CriteriaEvalChain.from_llm(
                llm=llm_instance,
                criteria=Criteria.RELEVANCE,
            )

            # Initialize CriteriaEvalChain for accuracy
            self.evaluators["accuracy"] = CriteriaEvalChain.from_llm(
                llm=llm_instance,
                criteria=Criteria.CORRECTNESS,
            )

            # Initialize PairwiseEvalChain for comparison
            self.evaluators["pairwise"] = PairwiseStringEvalChain.from_llm(
                llm=llm_instance,
            )

            logger.info("Initialized LangChain evaluators")

        except Exception as e:
            logger.error("Failed to initialize evaluators", error=str(e))
            # Continue without evaluators if initialization fails

    async def evaluate_relevance(
        self,
        query: str,
        response: str,
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate response relevance to query.

        Args:
            query: User query
            response: Generated response
            context: Optional context documents

        Returns:
            Dictionary with relevance evaluation
        """
        try:
            if "relevance" not in self.evaluators:
                return {"score": 0.5, "reasoning": "Evaluator not available"}

            evaluator = self.evaluators["relevance"]

            # Prepare input
            eval_input = {
                "input": query,
                "output": response,
            }
            if context:
                eval_input["reference"] = context

            # Evaluate
            result = evaluator.evaluate_strings(**eval_input)

            return {
                "score": self._parse_score(result),
                "reasoning": result.get("reasoning", ""),
                "value": result.get("value", ""),
            }

        except Exception as e:
            logger.error("Relevance evaluation failed", error=str(e))
            return {"score": 0.5, "reasoning": f"Evaluation failed: {str(e)}"}

    async def evaluate_accuracy(
        self,
        query: str,
        response: str,
        reference: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Evaluate response accuracy.

        Args:
            query: User query
            response: Generated response
            reference: Optional reference answer

        Returns:
            Dictionary with accuracy evaluation
        """
        try:
            if "accuracy" not in self.evaluators:
                return {"score": 0.5, "reasoning": "Evaluator not available"}

            evaluator = self.evaluators["accuracy"]

            # Prepare input
            eval_input = {
                "input": query,
                "output": response,
            }
            if reference:
                eval_input["reference"] = reference

            # Evaluate
            result = evaluator.evaluate_strings(**eval_input)

            return {
                "score": self._parse_score(result),
                "reasoning": result.get("reasoning", ""),
                "value": result.get("value", ""),
            }

        except Exception as e:
            logger.error("Accuracy evaluation failed", error=str(e))
            return {"score": 0.5, "reasoning": f"Evaluation failed: {str(e)}"}

    async def evaluate_pairwise(
        self,
        response1: str,
        response2: str,
        query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compare two responses using pairwise evaluation.

        Args:
            response1: First response
            response2: Second response
            query: Optional query for context

        Returns:
            Dictionary with comparison results
        """
        try:
            if "pairwise" not in self.evaluators:
                return {"preferred": "response1", "score": 0.5}

            evaluator = self.evaluators["pairwise"]

            # Evaluate
            result = evaluator.evaluate_string_pairs(
                prediction=response1,
                prediction_b=response2,
                input=query or "",
            )

            preferred = "response1" if result.get("score", 0) > 0 else "response2"

            return {
                "preferred": preferred,
                "score": abs(result.get("score", 0)),
                "reasoning": result.get("reasoning", ""),
            }

        except Exception as e:
            logger.error("Pairwise evaluation failed", error=str(e))
            return {"preferred": "response1", "score": 0.5}

    def _parse_score(self, result: Dict[str, Any]) -> float:
        """Parse score from evaluation result.

        Args:
            result: Evaluation result dictionary

        Returns:
            Score as float (0.0 to 1.0)
        """
        try:
            value = result.get("value", "")
            if isinstance(value, str):
                if "YES" in value.upper() or "CORRECT" in value.upper():
                    return 1.0
                elif "NO" in value.upper() or "INCORRECT" in value.upper():
                    return 0.0
                else:
                    return 0.5

            score = result.get("score", 0.5)
            if isinstance(score, (int, float)):
                return float(score)

            return 0.5

        except Exception as e:
            logger.error("Failed to parse score", error=str(e))
            return 0.5


class CitationAccuracyEvaluator:
    """Custom evaluator for citation accuracy."""

    def __init__(self, llm: Optional[OllamaLLM] = None):
        """Initialize citation accuracy evaluator.

        Args:
            llm: Optional Ollama LLM instance
        """
        self.llm = llm or OllamaLLM()
        self.logger = structlog.get_logger(__name__)

    async def evaluate_citation_accuracy(
        self,
        response: str,
        citations: List[Dict[str, Any]],
        source_documents: List[Any],
    ) -> Dict[str, Any]:
        """Evaluate accuracy of citations in response.

        Args:
            response: Generated response
            citations: List of citations
            source_documents: List of source documents

        Returns:
            Dictionary with citation accuracy evaluation
        """
        try:
            # Check if citations match source documents
            citation_file_ids = {c.get("file_id") for c in citations if c.get("file_id")}
            source_file_ids = set()

            for doc in source_documents:
                if hasattr(doc, "metadata"):
                    file_id = doc.metadata.get("file_id")
                    if file_id:
                        source_file_ids.add(file_id)

            # Calculate match rate
            if source_file_ids:
                match_rate = len(citation_file_ids.intersection(source_file_ids)) / len(source_file_ids)
            else:
                match_rate = 0.0

            # Check citation completeness
            citation_count = len(citations)
            expected_citations = min(len(source_documents), max(1, len(response) // 200))
            completeness = min(citation_count / expected_citations, 1.0) if expected_citations > 0 else 0.0

            # Overall accuracy
            accuracy = (match_rate + completeness) / 2.0

            return {
                "accuracy": accuracy,
                "match_rate": match_rate,
                "completeness": completeness,
                "citation_count": citation_count,
                "source_count": len(source_documents),
            }

        except Exception as e:
            logger.error("Citation accuracy evaluation failed", error=str(e))
            return {
                "accuracy": 0.5,
                "match_rate": 0.5,
                "completeness": 0.5,
            }


class SourceDiversityEvaluator:
    """Custom evaluator for source diversity."""

    def __init__(self):
        """Initialize source diversity evaluator."""
        self.logger = structlog.get_logger(__name__)

    def evaluate_source_diversity(
        self,
        source_documents: List[Any],
    ) -> Dict[str, Any]:
        """Evaluate diversity of sources.

        Args:
            source_documents: List of source documents

        Returns:
            Dictionary with diversity metrics
        """
        try:
            if not source_documents:
                return {
                    "diversity_score": 0.0,
                    "unique_files": 0,
                    "total_sources": 0,
                }

            # Count unique files
            unique_files = set()
            file_types = set()

            for doc in source_documents:
                if hasattr(doc, "metadata"):
                    file_id = doc.metadata.get("file_id")
                    if file_id:
                        unique_files.add(file_id)

                    mime_type = doc.metadata.get("mime_type", "")
                    if mime_type:
                        file_types.add(mime_type)

            # Calculate diversity score
            total_sources = len(source_documents)
            unique_count = len(unique_files)
            diversity_score = unique_count / total_sources if total_sources > 0 else 0.0

            # Type diversity
            type_diversity = len(file_types) / max(total_sources, 1)

            return {
                "diversity_score": diversity_score,
                "type_diversity": type_diversity,
                "unique_files": unique_count,
                "total_sources": total_sources,
                "file_types": list(file_types),
            }

        except Exception as e:
            logger.error("Source diversity evaluation failed", error=str(e))
            return {
                "diversity_score": 0.0,
                "unique_files": 0,
                "total_sources": 0,
            }


class ResponseCoherenceEvaluator:
    """Custom evaluator for response coherence and completeness."""

    def __init__(self, llm: Optional[OllamaLLM] = None):
        """Initialize coherence evaluator.

        Args:
            llm: Optional Ollama LLM instance
        """
        self.llm = llm or OllamaLLM()
        self.logger = structlog.get_logger(__name__)

    async def evaluate_coherence(
        self,
        response: str,
    ) -> Dict[str, Any]:
        """Evaluate response coherence.

        Args:
            response: Generated response

        Returns:
            Dictionary with coherence evaluation
        """
        try:
            # Simple heuristics for coherence
            coherence_score = 0.5

            # Check for complete sentences
            sentences = response.split('.')
            complete_sentences = [s for s in sentences if len(s.strip()) > 10]
            if complete_sentences:
                coherence_score += 0.2

            # Check for reasonable length
            if 50 <= len(response) <= 2000:
                coherence_score += 0.2

            # Check for structure (paragraphs, lists, etc.)
            if '\n' in response or '-' in response or '*' in response:
                coherence_score += 0.1

            coherence_score = min(coherence_score, 1.0)

            return {
                "coherence_score": coherence_score,
                "sentence_count": len(complete_sentences),
                "response_length": len(response),
            }

        except Exception as e:
            logger.error("Coherence evaluation failed", error=str(e))
            return {"coherence_score": 0.5}

    async def evaluate_completeness(
        self,
        query: str,
        response: str,
    ) -> Dict[str, Any]:
        """Evaluate response completeness.

        Args:
            query: User query
            response: Generated response

        Returns:
            Dictionary with completeness evaluation
        """
        try:
            # Simple heuristic: response should be substantial
            query_length = len(query)
            response_length = len(response)

            # Response should be at least as long as query
            if response_length >= query_length:
                completeness = min(response_length / (query_length * 3), 1.0)
            else:
                completeness = response_length / (query_length * 2)

            return {
                "completeness_score": completeness,
                "query_length": query_length,
                "response_length": response_length,
            }

        except Exception as e:
            logger.error("Completeness evaluation failed", error=str(e))
            return {"completeness_score": 0.5}

