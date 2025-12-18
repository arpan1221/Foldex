"""LangChain fact-checking and consistency validation chains."""

from typing import Optional, Dict, Any, List
import structlog

try:
    from langchain.chains import LLMChain
    from langchain.prompts import PromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    LLMChain = None
    PromptTemplate = None

from app.core.exceptions import ProcessingError
from app.rag.llm_chains import OllamaLLM

logger = structlog.get_logger(__name__)


class FactChecker:
    """LangChain-based fact-checking chain."""

    def __init__(self, llm: Optional[OllamaLLM] = None):
        """Initialize fact checker.

        Args:
            llm: Optional Ollama LLM instance
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        self.llm = llm or OllamaLLM()
        self.fact_check_chain = None
        self._initialize_chain()

    def _initialize_chain(self) -> None:
        """Initialize fact-checking chain."""
        try:
            fact_check_prompt = PromptTemplate(
                input_variables=["claim", "source_context"],
                template="""You are a fact-checker. Evaluate whether the following claim is supported by the source context.

Claim: {claim}

Source Context: {source_context}

Evaluate:
1. Is the claim directly supported by the source context? (YES/NO)
2. Is the claim partially supported? (YES/NO)
3. Is the claim contradicted by the source context? (YES/NO)
4. What is your confidence level? (HIGH/MEDIUM/LOW)

Provide your evaluation in the following format:
SUPPORTED: [YES/NO]
PARTIAL: [YES/NO]
CONTRADICTED: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [Your reasoning]
""",
            )

            self.fact_check_chain = LLMChain(
                llm=self.llm.get_llm(),
                prompt=fact_check_prompt,
            )

            logger.info("Initialized fact-checking chain")

        except Exception as e:
            logger.error("Failed to initialize fact-checking chain", error=str(e))
            raise ProcessingError(f"Failed to initialize fact checker: {str(e)}") from e

    async def check_fact(
        self,
        claim: str,
        source_context: str,
    ) -> Dict[str, Any]:
        """Check if a claim is supported by source context.

        Args:
            claim: Claim to verify
            source_context: Source context for verification

        Returns:
            Dictionary with fact-check results
        """
        try:
            if self.fact_check_chain is None:
                raise ProcessingError("Fact-check chain not initialized")

            result = self.fact_check_chain.invoke({
                "claim": claim,
                "source_context": source_context,
            })

            # Parse result
            evaluation = self._parse_fact_check_result(result.get("text", ""))

            return {
                "claim": claim,
                "supported": evaluation.get("supported", False),
                "partial": evaluation.get("partial", False),
                "contradicted": evaluation.get("contradicted", False),
                "confidence": evaluation.get("confidence", "MEDIUM"),
                "reasoning": evaluation.get("reasoning", ""),
            }

        except Exception as e:
            logger.error("Fact-checking failed", error=str(e))
            return {
                "claim": claim,
                "supported": False,
                "partial": False,
                "contradicted": False,
                "confidence": "LOW",
                "reasoning": f"Fact-checking failed: {str(e)}",
            }

    def _parse_fact_check_result(self, result_text: str) -> Dict[str, Any]:
        """Parse fact-check result text.

        Args:
            result_text: Result text from LLM

        Returns:
            Parsed evaluation dictionary
        """
        try:
            evaluation = {
                "supported": False,
                "partial": False,
                "contradicted": False,
                "confidence": "MEDIUM",
                "reasoning": result_text,
            }

            result_upper = result_text.upper()

            # Parse supported
            if "SUPPORTED: YES" in result_upper:
                evaluation["supported"] = True
            elif "SUPPORTED: NO" in result_upper:
                evaluation["supported"] = False

            # Parse partial
            if "PARTIAL: YES" in result_upper:
                evaluation["partial"] = True

            # Parse contradicted
            if "CONTRADICTED: YES" in result_upper:
                evaluation["contradicted"] = True

            # Parse confidence
            if "CONFIDENCE: HIGH" in result_upper:
                evaluation["confidence"] = "HIGH"
            elif "CONFIDENCE: LOW" in result_upper:
                evaluation["confidence"] = "LOW"

            return evaluation

        except Exception as e:
            logger.error("Failed to parse fact-check result", error=str(e))
            return {
                "supported": False,
                "partial": False,
                "contradicted": False,
                "confidence": "MEDIUM",
                "reasoning": result_text,
            }


class ConsistencyValidator:
    """LangChain-based consistency validation chain."""

    def __init__(self, llm: Optional[OllamaLLM] = None):
        """Initialize consistency validator.

        Args:
            llm: Optional Ollama LLM instance
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        self.llm = llm or OllamaLLM()
        self.consistency_chain = None
        self._initialize_chain()

    def _initialize_chain(self) -> None:
        """Initialize consistency validation chain."""
        try:
            consistency_prompt = PromptTemplate(
                input_variables=["response1", "response2"],
                template="""You are a consistency validator. Compare two responses to determine if they are consistent with each other.

Response 1: {response1}

Response 2: {response2}

Evaluate:
1. Are the responses consistent? (YES/NO)
2. Are there contradictions? (YES/NO)
3. What is the consistency level? (HIGH/MEDIUM/LOW)

Provide your evaluation:
CONSISTENT: [YES/NO]
CONTRADICTIONS: [YES/NO]
LEVEL: [HIGH/MEDIUM/LOW]
REASONING: [Your reasoning]
""",
            )

            self.consistency_chain = LLMChain(
                llm=self.llm.get_llm(),
                prompt=consistency_prompt,
            )

            logger.info("Initialized consistency validation chain")

        except Exception as e:
            logger.error("Failed to initialize consistency chain", error=str(e))
            raise ProcessingError(f"Failed to initialize validator: {str(e)}") from e

    async def validate_consistency(
        self,
        response1: str,
        response2: str,
    ) -> Dict[str, Any]:
        """Validate consistency between two responses.

        Args:
            response1: First response
            response2: Second response

        Returns:
            Dictionary with consistency validation results
        """
        try:
            if self.consistency_chain is None:
                raise ProcessingError("Consistency chain not initialized")

            result = self.consistency_chain.invoke({
                "response1": response1,
                "response2": response2,
            })

            # Parse result
            evaluation = self._parse_consistency_result(result.get("text", ""))

            return {
                "consistent": evaluation.get("consistent", False),
                "contradictions": evaluation.get("contradictions", False),
                "level": evaluation.get("level", "MEDIUM"),
                "reasoning": evaluation.get("reasoning", ""),
            }

        except Exception as e:
            logger.error("Consistency validation failed", error=str(e))
            return {
                "consistent": False,
                "contradictions": True,
                "level": "LOW",
                "reasoning": f"Validation failed: {str(e)}",
            }

    def _parse_consistency_result(self, result_text: str) -> Dict[str, Any]:
        """Parse consistency validation result.

        Args:
            result_text: Result text from LLM

        Returns:
            Parsed evaluation dictionary
        """
        try:
            evaluation = {
                "consistent": False,
                "contradictions": False,
                "level": "MEDIUM",
                "reasoning": result_text,
            }

            result_upper = result_text.upper()

            # Parse consistent
            if "CONSISTENT: YES" in result_upper:
                evaluation["consistent"] = True
            elif "CONSISTENT: NO" in result_upper:
                evaluation["consistent"] = False

            # Parse contradictions
            if "CONTRADICTIONS: YES" in result_upper:
                evaluation["contradictions"] = True

            # Parse level
            if "LEVEL: HIGH" in result_upper:
                evaluation["level"] = "HIGH"
            elif "LEVEL: LOW" in result_upper:
                evaluation["level"] = "LOW"

            return evaluation

        except Exception as e:
            logger.error("Failed to parse consistency result", error=str(e))
            return {
                "consistent": False,
                "contradictions": True,
                "level": "MEDIUM",
                "reasoning": result_text,
            }


class SelfCorrectionChain:
    """LangChain chain for self-correction and improvement."""

    def __init__(self, llm: Optional[OllamaLLM] = None):
        """Initialize self-correction chain.

        Args:
            llm: Optional Ollama LLM instance
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        self.llm = llm or OllamaLLM()
        self.correction_chain = None
        self._initialize_chain()

    def _initialize_chain(self) -> None:
        """Initialize self-correction chain."""
        try:
            correction_prompt = PromptTemplate(
                input_variables=["original_response", "quality_issues", "source_context"],
                template="""You are a response quality improver. Given an original response, quality issues, and source context, generate an improved response.

Original Response: {original_response}

Quality Issues: {quality_issues}

Source Context: {source_context}

Generate an improved response that:
1. Addresses the quality issues
2. Better aligns with the source context
3. Maintains accuracy and coherence
4. Includes proper citations

Improved Response:
""",
            )

            self.correction_chain = LLMChain(
                llm=self.llm.get_llm(),
                prompt=correction_prompt,
            )

            logger.info("Initialized self-correction chain")

        except Exception as e:
            logger.error("Failed to initialize correction chain", error=str(e))
            raise ProcessingError(f"Failed to initialize correction chain: {str(e)}") from e

    async def improve_response(
        self,
        original_response: str,
        quality_issues: List[str],
        source_context: str,
    ) -> Dict[str, Any]:
        """Improve response based on quality issues.

        Args:
            original_response: Original response
            quality_issues: List of quality issues identified
            source_context: Source context for improvement

        Returns:
            Dictionary with improved response and metadata
        """
        try:
            if self.correction_chain is None:
                raise ProcessingError("Correction chain not initialized")

            issues_text = "\n".join(f"- {issue}" for issue in quality_issues)

            result = self.correction_chain.invoke({
                "original_response": original_response,
                "quality_issues": issues_text,
                "source_context": source_context,
            })

            improved_response = result.get("text", original_response)

            return {
                "original_response": original_response,
                "improved_response": improved_response,
                "issues_addressed": quality_issues,
                "improvement_applied": True,
            }

        except Exception as e:
            logger.error("Response improvement failed", error=str(e))
            return {
                "original_response": original_response,
                "improved_response": original_response,
                "issues_addressed": [],
                "improvement_applied": False,
                "error": str(e),
            }

