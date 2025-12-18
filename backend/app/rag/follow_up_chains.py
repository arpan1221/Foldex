"""LangChain chains for generating follow-up questions and query clarification."""

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


class FollowUpQuestionGenerator:
    """LangChain chain for generating follow-up questions."""

    def __init__(self, llm: Optional[OllamaLLM] = None):
        """Initialize follow-up question generator.

        Args:
            llm: Optional Ollama LLM instance
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        self.llm = llm or OllamaLLM()
        self.follow_up_chain = None
        self._initialize_chain()

    def _initialize_chain(self) -> None:
        """Initialize follow-up question generation chain."""
        try:
            follow_up_prompt = PromptTemplate(
                input_variables=["conversation_history", "last_response", "available_context"],
                template="""Based on the conversation history and the last response, generate 3-5 relevant follow-up questions that would help the user explore the topic further or clarify their understanding.

Conversation History:
{conversation_history}

Last Response:
{last_response}

Available Context Topics:
{available_context}

Generate follow-up questions that:
1. Build on the information provided
2. Explore related aspects
3. Seek clarification on ambiguous points
4. Dive deeper into specific topics

Format your response as a numbered list of questions, one per line:
1. [Question 1]
2. [Question 2]
3. [Question 3]
""",
            )

            self.follow_up_chain = LLMChain(
                llm=self.llm.get_llm(),
                prompt=follow_up_prompt,
            )

            logger.info("Initialized follow-up question generation chain")

        except Exception as e:
            logger.error("Failed to initialize follow-up chain", error=str(e))
            raise ProcessingError(
                f"Failed to initialize follow-up generator: {str(e)}"
            ) from e

    async def generate_follow_ups(
        self,
        conversation_history: List[Dict[str, str]],
        last_response: str,
        available_context: Optional[str] = None,
    ) -> List[str]:
        """Generate follow-up questions.

        Args:
            conversation_history: List of conversation messages
            last_response: Last response from assistant
            available_context: Optional context about available information

        Returns:
            List of follow-up questions
        """
        try:
            if self.follow_up_chain is None:
                raise ProcessingError("Follow-up chain not initialized")

            # Format conversation history
            history_text = self._format_conversation_history(conversation_history)

            # Generate follow-ups
            result = self.follow_up_chain.invoke({
                "conversation_history": history_text,
                "last_response": last_response,
                "available_context": available_context or "General information available",
            })

            # Parse questions from result
            questions = self._parse_follow_up_questions(result.get("text", ""))

            logger.debug(
                "Generated follow-up questions",
                question_count=len(questions),
            )

            return questions

        except Exception as e:
            logger.error("Follow-up generation failed", error=str(e))
            return []

    def _format_conversation_history(
        self,
        history: List[Dict[str, str]],
    ) -> str:
        """Format conversation history for prompt.

        Args:
            history: List of conversation messages

        Returns:
            Formatted history string
        """
        formatted = []
        for msg in history[-5:]:  # Last 5 messages
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted.append(f"{role.capitalize()}: {content}")

        return "\n".join(formatted)

    def _parse_follow_up_questions(self, text: str) -> List[str]:
        """Parse follow-up questions from LLM output.

        Args:
            text: LLM output text

        Returns:
            List of questions
        """
        import re

        questions = []

        # Try to extract numbered questions
        pattern = r'^\d+[\.\)]\s*(.+)$'
        for line in text.split('\n'):
            match = re.match(pattern, line.strip())
            if match:
                question = match.group(1).strip()
                if question and question.endswith('?'):
                    questions.append(question)

        # If no numbered questions, try to extract questions ending with ?
        if not questions:
            for line in text.split('\n'):
                line = line.strip()
                if line.endswith('?') and len(line) > 10:
                    questions.append(line)

        return questions[:5]  # Limit to 5 questions


class QueryClarificationChain:
    """LangChain chain for query clarification."""

    def __init__(self, llm: Optional[OllamaLLM] = None):
        """Initialize query clarification chain.

        Args:
            llm: Optional Ollama LLM instance
        """
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        self.llm = llm or OllamaLLM()
        self.clarification_chain = None
        self._initialize_chain()

    def _initialize_chain(self) -> None:
        """Initialize query clarification chain."""
        try:
            clarification_prompt = PromptTemplate(
                input_variables=["query", "conversation_history"],
                template="""Analyze the following query and determine if it needs clarification. If the query is ambiguous, unclear, or could be interpreted multiple ways, generate clarifying questions.

Query: {query}

Conversation History:
{conversation_history}

Determine:
1. Is the query clear and unambiguous? (YES/NO)
2. What aspects might need clarification?
3. What information is missing?

If clarification is needed, generate 2-3 specific clarifying questions.

Format:
NEEDS_CLARIFICATION: [YES/NO]
CLARIFYING_QUESTIONS:
1. [Question 1]
2. [Question 2]
REASONING: [Why clarification is needed]
""",
            )

            self.clarification_chain = LLMChain(
                llm=self.llm.get_llm(),
                prompt=clarification_prompt,
            )

            logger.info("Initialized query clarification chain")

        except Exception as e:
            logger.error("Failed to initialize clarification chain", error=str(e))
            raise ProcessingError(
                f"Failed to initialize clarification chain: {str(e)}"
            ) from e

    async def check_clarification_needed(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
    ) -> Dict[str, Any]:
        """Check if query needs clarification.

        Args:
            query: User query
            conversation_history: Optional conversation history

        Returns:
            Dictionary with clarification assessment
        """
        try:
            if self.clarification_chain is None:
                raise ProcessingError("Clarification chain not initialized")

            # Format history
            history_text = ""
            if conversation_history:
                history_text = "\n".join([
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                    for msg in conversation_history[-3:]
                ])

            # Check clarification
            result = self.clarification_chain.invoke({
                "query": query,
                "conversation_history": history_text or "No previous conversation",
            })

            # Parse result
            clarification = self._parse_clarification_result(result.get("text", ""))

            return clarification

        except Exception as e:
            logger.error("Clarification check failed", error=str(e))
            return {
                "needs_clarification": False,
                "clarifying_questions": [],
                "reasoning": f"Clarification check failed: {str(e)}",
            }

    def _parse_clarification_result(self, text: str) -> Dict[str, Any]:
        """Parse clarification result.

        Args:
            text: LLM output text

        Returns:
            Parsed clarification dictionary
        """
        import re

        clarifying_questions: List[str] = []
        clarification: Dict[str, Any] = {
            "needs_clarification": False,
            "clarifying_questions": clarifying_questions,
            "reasoning": text,
        }

        text_upper = text.upper()

        # Check if clarification needed
        if "NEEDS_CLARIFICATION: YES" in text_upper:
            clarification["needs_clarification"] = True

        # Extract questions
        questions_section = False
        for line in text.split('\n'):
            if "CLARIFYING_QUESTIONS" in line.upper():
                questions_section = True
                continue

            if questions_section:
                match = re.match(r'^\d+[\.\)]\s*(.+)$', line.strip())
                if match:
                    question = match.group(1).strip()
                    if question:
                        clarifying_questions.append(question)
                elif line.strip().startswith("REASONING"):
                    questions_section = False

        return clarification

