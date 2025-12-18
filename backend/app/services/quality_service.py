"""LangChain-based quality assurance pipeline."""

from typing import Optional, Dict, Any, List
import structlog

from app.core.exceptions import ProcessingError
from app.rag.evaluation_chains import (
    ResponseEvaluator,
    CitationAccuracyEvaluator,
    SourceDiversityEvaluator,
    ResponseCoherenceEvaluator,
)
from app.rag.validation_chains import (
    FactChecker,
    ConsistencyValidator,
    SelfCorrectionChain,
)
from app.rag.quality_callbacks import QualityMetricsCallbackHandler, ResponseQualityTracker
from app.rag.llm_chains import OllamaLLM

logger = structlog.get_logger(__name__)


class QualityService:
    """Service for comprehensive quality assurance using LangChain evaluation."""

    def __init__(self, llm: Optional[OllamaLLM] = None):
        """Initialize quality service.

        Args:
            llm: Optional Ollama LLM instance
        """
        self.llm = llm or OllamaLLM()

        # Initialize evaluators
        try:
            self.response_evaluator = ResponseEvaluator(llm=self.llm)
            self.citation_evaluator = CitationAccuracyEvaluator(llm=self.llm)
            self.source_diversity_evaluator = SourceDiversityEvaluator()
            self.coherence_evaluator = ResponseCoherenceEvaluator(llm=self.llm)

            # Initialize validators
            self.fact_checker = FactChecker(llm=self.llm)
            self.consistency_validator = ConsistencyValidator(llm=self.llm)
            self.self_correction = SelfCorrectionChain(llm=self.llm)

            # Initialize trackers
            self.quality_tracker = ResponseQualityTracker()
            self.metrics_callback = QualityMetricsCallbackHandler()

            logger.info("Initialized quality service")

        except Exception as e:
            logger.error("Failed to initialize quality service", error=str(e))
            raise ProcessingError(f"Failed to initialize quality service: {str(e)}") from e

    async def evaluate_response_quality(
        self,
        query: str,
        response: str,
        citations: List[Dict[str, Any]],
        source_documents: List[Any],
        context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Comprehensive response quality evaluation.

        Args:
            query: User query
            response: Generated response
            citations: List of citations
            source_documents: List of source documents
            context: Optional context documents

        Returns:
            Dictionary with comprehensive quality assessment
        """
        try:
            logger.info("Evaluating response quality", query_length=len(query))

            # Run all evaluations
            evaluations = {}

            # Relevance evaluation
            relevance = await self.response_evaluator.evaluate_relevance(
                query=query,
                response=response,
                context=context,
            )
            evaluations["relevance"] = relevance

            # Accuracy evaluation
            accuracy = await self.response_evaluator.evaluate_accuracy(
                query=query,
                response=response,
                reference=context,
            )
            evaluations["accuracy"] = accuracy

            # Citation accuracy
            citation_accuracy = await self.citation_evaluator.evaluate_citation_accuracy(
                response=response,
                citations=citations,
                source_documents=source_documents,
            )
            evaluations["citation_accuracy"] = citation_accuracy

            # Source diversity
            source_diversity = self.source_diversity_evaluator.evaluate_source_diversity(
                source_documents=source_documents,
            )
            evaluations["source_diversity"] = source_diversity

            # Coherence
            coherence = await self.coherence_evaluator.evaluate_coherence(response)
            evaluations["coherence"] = coherence

            # Completeness
            completeness = await self.coherence_evaluator.evaluate_completeness(
                query=query,
                response=response,
            )
            evaluations["completeness"] = completeness

            # Calculate overall quality score
            overall_score = self._calculate_overall_quality(evaluations)

            # Track quality
            self.quality_tracker.track_response(
                response=response,
                citations=citations,
                source_documents=source_documents,
            )

            result = {
                "overall_quality": overall_score,
                "evaluations": evaluations,
                "recommendations": self._generate_improvement_recommendations(evaluations),
            }

            logger.info(
                "Quality evaluation completed",
                overall_quality=overall_score,
            )

            return result

        except Exception as e:
            logger.error("Quality evaluation failed", error=str(e), exc_info=True)
            raise ProcessingError(f"Quality evaluation failed: {str(e)}") from e

    async def validate_response(
        self,
        response: str,
        source_documents: List[Any],
    ) -> Dict[str, Any]:
        """Validate response facts and consistency.

        Args:
            response: Generated response
            source_documents: List of source documents

        Returns:
            Dictionary with validation results
        """
        try:
            logger.info("Validating response")

            # Extract claims from response (simplified)
            claims = self._extract_claims(response)

            # Fact-check each claim
            fact_check_results = []
            for claim in claims[:5]:  # Limit to 5 claims
                # Get relevant source context
                source_context = self._get_relevant_context(claim, source_documents)

                if source_context:
                    fact_check = await self.fact_checker.check_fact(
                        claim=claim,
                        source_context=source_context,
                    )
                    fact_check_results.append(fact_check)

            # Calculate validation score
            if fact_check_results:
                supported_count = sum(1 for r in fact_check_results if r.get("supported"))
                validation_score = supported_count / len(fact_check_results)
            else:
                validation_score = 0.5

            return {
                "validation_score": validation_score,
                "fact_check_results": fact_check_results,
                "claims_checked": len(fact_check_results),
            }

        except Exception as e:
            logger.error("Response validation failed", error=str(e))
            return {
                "validation_score": 0.5,
                "fact_check_results": [],
                "claims_checked": 0,
                "error": str(e),
            }

    async def improve_response_quality(
        self,
        query: str,
        response: str,
        citations: List[Dict[str, Any]],
        source_documents: List[Any],
    ) -> Dict[str, Any]:
        """Improve response quality using self-correction.

        Args:
            query: User query
            response: Original response
            citations: List of citations
            source_documents: List of source documents

        Returns:
            Dictionary with improved response
        """
        try:
            logger.info("Improving response quality")

            # Evaluate quality first
            quality_assessment = await self.evaluate_response_quality(
                query=query,
                response=response,
                citations=citations,
                source_documents=source_documents,
            )

            # Identify quality issues
            quality_issues = self._identify_quality_issues(quality_assessment)

            if not quality_issues:
                # No issues found, return original
                return {
                    "original_response": response,
                    "improved_response": response,
                    "improvement_applied": False,
                    "reason": "No quality issues identified",
                }

            # Get source context
            source_context = self._build_source_context(source_documents)

            # Apply self-correction
            improvement_result = await self.self_correction.improve_response(
                original_response=response,
                quality_issues=quality_issues,
                source_context=source_context,
            )

            return {
                **improvement_result,
                "quality_assessment": quality_assessment,
                "issues_identified": quality_issues,
            }

        except Exception as e:
            logger.error("Response improvement failed", error=str(e))
            return {
                "original_response": response,
                "improved_response": response,
                "improvement_applied": False,
                "error": str(e),
            }

    def _calculate_overall_quality(
        self,
        evaluations: Dict[str, Any],
    ) -> float:
        """Calculate overall quality score from evaluations.

        Args:
            evaluations: Dictionary with evaluation results

        Returns:
            Overall quality score (0.0 to 1.0)
        """
        try:
            scores = []

            # Relevance (weight: 0.2)
            if "relevance" in evaluations:
                scores.append(evaluations["relevance"].get("score", 0.5) * 0.2)

            # Accuracy (weight: 0.2)
            if "accuracy" in evaluations:
                scores.append(evaluations["accuracy"].get("score", 0.5) * 0.2)

            # Citation accuracy (weight: 0.2)
            if "citation_accuracy" in evaluations:
                scores.append(evaluations["citation_accuracy"].get("accuracy", 0.5) * 0.2)

            # Source diversity (weight: 0.15)
            if "source_diversity" in evaluations:
                scores.append(evaluations["source_diversity"].get("diversity_score", 0.5) * 0.15)

            # Coherence (weight: 0.15)
            if "coherence" in evaluations:
                scores.append(evaluations["coherence"].get("coherence_score", 0.5) * 0.15)

            # Completeness (weight: 0.1)
            if "completeness" in evaluations:
                scores.append(evaluations["completeness"].get("completeness_score", 0.5) * 0.1)

            overall = sum(scores) if scores else 0.5
            return min(overall, 1.0)

        except Exception as e:
            logger.error("Failed to calculate overall quality", error=str(e))
            return 0.5

    def _generate_improvement_recommendations(
        self,
        evaluations: Dict[str, Any],
    ) -> List[str]:
        """Generate improvement recommendations based on evaluations.

        Args:
            evaluations: Dictionary with evaluation results

        Returns:
            List of improvement recommendations
        """
        recommendations = []

        # Check relevance
        relevance_score = evaluations.get("relevance", {}).get("score", 0.5)
        if relevance_score < 0.6:
            recommendations.append("Improve response relevance to the query")

        # Check accuracy
        accuracy_score = evaluations.get("accuracy", {}).get("score", 0.5)
        if accuracy_score < 0.6:
            recommendations.append("Improve response accuracy and factual correctness")

        # Check citation accuracy
        citation_accuracy = evaluations.get("citation_accuracy", {}).get("accuracy", 0.5)
        if citation_accuracy < 0.6:
            recommendations.append("Improve citation accuracy and completeness")

        # Check source diversity
        diversity_score = evaluations.get("source_diversity", {}).get("diversity_score", 0.5)
        if diversity_score < 0.5:
            recommendations.append("Increase source diversity for better coverage")

        # Check coherence
        coherence_score = evaluations.get("coherence", {}).get("coherence_score", 0.5)
        if coherence_score < 0.6:
            recommendations.append("Improve response coherence and structure")

        # Check completeness
        completeness_score = evaluations.get("completeness", {}).get("completeness_score", 0.5)
        if completeness_score < 0.6:
            recommendations.append("Provide more complete and detailed response")

        return recommendations

    def _identify_quality_issues(
        self,
        quality_assessment: Dict[str, Any],
    ) -> List[str]:
        """Identify specific quality issues from assessment.

        Args:
            quality_assessment: Quality assessment dictionary

        Returns:
            List of quality issues
        """
        issues = []
        evaluations = quality_assessment.get("evaluations", {})

        # Low relevance
        if evaluations.get("relevance", {}).get("score", 0.5) < 0.6:
            issues.append("Response is not sufficiently relevant to the query")

        # Low accuracy
        if evaluations.get("accuracy", {}).get("score", 0.5) < 0.6:
            issues.append("Response may contain inaccuracies")

        # Poor citations
        if evaluations.get("citation_accuracy", {}).get("accuracy", 0.5) < 0.6:
            issues.append("Citations are incomplete or inaccurate")

        # Low diversity
        if evaluations.get("source_diversity", {}).get("diversity_score", 0.5) < 0.5:
            issues.append("Response relies on too few sources")

        # Poor coherence
        if evaluations.get("coherence", {}).get("coherence_score", 0.5) < 0.6:
            issues.append("Response lacks coherence and structure")

        return issues

    def _extract_claims(self, response: str) -> List[str]:
        """Extract factual claims from response.

        Args:
            response: Response text

        Returns:
            List of claims
        """
        # Simple extraction: split by sentences
        import re
        sentences = re.split(r'[.!?]+', response)
        claims = [s.strip() for s in sentences if len(s.strip()) > 20]
        return claims[:10]  # Limit to 10 claims

    def _get_relevant_context(
        self,
        claim: str,
        source_documents: List[Any],
    ) -> Optional[str]:
        """Get relevant context for a claim.

        Args:
            claim: Claim to find context for
            source_documents: List of source documents

        Returns:
            Relevant context string or None
        """
        try:
            # Simple matching: find document with claim keywords
            claim_lower = claim.lower()
            claim_words = set(claim_lower.split())

            for doc in source_documents:
                if hasattr(doc, "page_content"):
                    content = doc.page_content.lower()
                    content_words = set(content.split())
                    overlap = claim_words.intersection(content_words)
                    if len(overlap) >= len(claim_words) * 0.3:  # 30% overlap
                        return doc.page_content[:500]  # Return first 500 chars

            # Return first document if no match
            if source_documents and hasattr(source_documents[0], "page_content"):
                return source_documents[0].page_content[:500]

            return None

        except Exception as e:
            logger.error("Failed to get relevant context", error=str(e))
            return None

    def _build_source_context(self, source_documents: List[Any]) -> str:
        """Build source context string from documents.

        Args:
            source_documents: List of source documents

        Returns:
            Combined source context
        """
        try:
            contexts = []
            for doc in source_documents[:5]:  # Limit to 5 documents
                if hasattr(doc, "page_content"):
                    contexts.append(doc.page_content[:300])  # First 300 chars

            return "\n\n".join(contexts)

        except Exception as e:
            logger.error("Failed to build source context", error=str(e))
            return ""

    def get_quality_metrics(self) -> Dict[str, Any]:
        """Get collected quality metrics.

        Returns:
            Dictionary with quality metrics
        """
        return {
            "callback_metrics": self.metrics_callback.get_metrics(),
            "average_quality": self.quality_tracker.get_average_quality(),
        }


# Global quality service instance
_quality_service: Optional[QualityService] = None


def get_quality_service(llm: Optional[OllamaLLM] = None) -> QualityService:
    """Get global quality service instance.

    Args:
        llm: Optional LLM instance

    Returns:
        QualityService instance
    """
    global _quality_service
    if _quality_service is None:
        _quality_service = QualityService(llm=llm)
    return _quality_service

