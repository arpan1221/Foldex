"""LangChain prompt templates and prompt engineering for different query types."""

from typing import Optional, Dict, Any
import structlog

try:
    from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
    from langchain_core.prompts import PromptTemplate as CorePromptTemplate
    from langchain_core.prompts import ChatPromptTemplate as CoreChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback for older LangChain versions
    try:
        from langchain.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        PromptTemplate = None
        ChatPromptTemplate = None
        MessagesPlaceholder = None

from app.core.exceptions import ProcessingError

logger = structlog.get_logger(__name__)


class PromptManager:
    """Manages LangChain prompt templates for different query types."""

    def __init__(self):
        """Initialize prompt manager."""
        if not LANGCHAIN_AVAILABLE:
            raise ProcessingError(
                "LangChain is not installed. Install with: pip install langchain"
            )

        self.prompts = self._initialize_prompts()
        logger.info("Initialized prompt manager")

    def _initialize_prompts(self) -> Dict[str, Any]:
        """Initialize all prompt templates.

        Returns:
            Dictionary of prompt templates
        """
        prompts = {}

        # Factual Query Prompt - for direct questions with specific answers
        prompts["factual"] = self._create_factual_prompt()

        # Synthesis Prompt - for questions requiring information synthesis
        prompts["synthesis"] = self._create_synthesis_prompt()

        # Relationship Prompt - for questions about connections and relationships
        prompts["relationship"] = self._create_relationship_prompt()

        # Default Prompt - general purpose
        prompts["default"] = self._create_default_prompt()

        return prompts

    def _create_factual_prompt(self) -> PromptTemplate:
        """Create prompt template for factual queries.

        Returns:
            PromptTemplate for factual queries (RetrievalQA compatible)
        """
        template = """You are a helpful AI assistant that answers questions based on the provided context documents.

Context Documents:
{context}

Question: {question}

Instructions:
- Answer the question directly and factually based ONLY on the provided context
- If the answer is not in the context, say "I don't have enough information to answer this question based on the provided documents."
- Cite specific documents or sections when possible
- Be concise and accurate

Answer:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def _create_synthesis_prompt(self) -> PromptTemplate:
        """Create prompt template for synthesis queries.

        Returns:
            PromptTemplate for synthesis queries (RetrievalQA compatible)
        """
        template = """You are a helpful AI assistant that synthesizes information from multiple documents to answer complex questions.

Context Documents:
{context}

Question: {question}

Instructions:
- Synthesize information from multiple context documents to provide a comprehensive answer
- Identify patterns, themes, and connections across documents
- Provide a well-structured answer that integrates information from different sources
- Cite the relevant documents that contributed to your synthesis
- If information is contradictory across documents, acknowledge the differences
- Be thorough but organized in your response

Answer:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def _create_relationship_prompt(self) -> PromptTemplate:
        """Create prompt template for relationship queries.

        Returns:
            PromptTemplate for relationship queries (RetrievalQA compatible)
        """
        template = """You are a helpful AI assistant that identifies relationships and connections between concepts, entities, and documents.

Context Documents:
{context}

Question: {question}

Instructions:
- Identify relationships, connections, and patterns between entities mentioned in the context
- Explain how different concepts, people, or documents relate to each other
- Map out dependencies, influences, or causal relationships when applicable
- Provide clear explanations of how things are connected
- Cite specific documents that mention these relationships

Answer:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def _create_default_prompt(self) -> PromptTemplate:
        """Create default prompt template for general queries.

        Returns:
            PromptTemplate for general queries (RetrievalQA compatible)
        """
        template = """You are a helpful AI assistant that answers questions based on provided context documents.

Context Documents:
{context}

Question: {question}

Instructions:
- Answer the question based on the provided context documents
- Be helpful, accurate, and cite sources when possible
- If you don't have enough information, say so clearly
- Format your response in a clear and readable way

Answer:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def get_prompt(
        self, query_type: str = "default"
    ) -> PromptTemplate:
        """Get prompt template for query type.

        Args:
            query_type: Type of query (factual, synthesis, relationship, default)

        Returns:
            PromptTemplate instance

        Raises:
            ProcessingError: If query type is not supported
        """
        if query_type not in self.prompts:
            logger.warning(
                "Unknown query type, using default",
                query_type=query_type,
            )
            query_type = "default"

        return self.prompts[query_type]

    def classify_query_type(self, query: str) -> str:
        """Classify query type based on content.

        Args:
            query: User query text

        Returns:
            Query type (factual, synthesis, relationship, default)
        """
        query_lower = query.lower()

        # Relationship indicators
        relationship_keywords = [
            "relationship",
            "related",
            "connection",
            "link",
            "compare",
            "difference",
            "similar",
            "associate",
            "correlate",
            "depend",
            "influence",
            "cause",
            "effect",
        ]

        # Synthesis indicators
        synthesis_keywords = [
            "summarize",
            "synthesize",
            "overall",
            "general",
            "all documents",
            "across",
            "multiple",
            "combine",
            "integrate",
            "pattern",
            "theme",
        ]

        # Factual indicators
        factual_keywords = [
            "what is",
            "who is",
            "when did",
            "where is",
            "how many",
            "how much",
            "which",
            "name",
            "list",
        ]

        # Check for relationship queries
        if any(keyword in query_lower for keyword in relationship_keywords):
            return "relationship"

        # Check for synthesis queries
        if any(keyword in query_lower for keyword in synthesis_keywords):
            return "synthesis"

        # Check for factual queries
        if any(keyword in query_lower for keyword in factual_keywords):
            return "factual"

        # Default to general
        return "default"

    def format_context(self, documents: list) -> str:
        """Format retrieved documents as context string.

        Args:
            documents: List of document objects (LangChain Documents or DocumentChunks)

        Returns:
            Formatted context string
        """
        context_parts = []

        for i, doc in enumerate(documents, 1):
            # Handle both LangChain Documents and DocumentChunks
            if hasattr(doc, "page_content"):
                content = doc.page_content
                metadata = doc.metadata if hasattr(doc, "metadata") else {}
            elif hasattr(doc, "content"):
                content = doc.content
                metadata = doc.metadata if hasattr(doc, "metadata") else {}
            else:
                content = str(doc)
                metadata = {}

            file_name = metadata.get("file_name", "Unknown")
            page_number = metadata.get("page_number")
            chunk_index = metadata.get("chunk_index")

            # Build citation
            citation_parts = [f"Document {i}"]
            if file_name:
                citation_parts.append(f"from '{file_name}'")
            if page_number is not None:
                citation_parts.append(f"page {page_number}")
            if chunk_index is not None:
                citation_parts.append(f"chunk {chunk_index}")

            citation = " ".join(citation_parts)

            context_parts.append(f"[{citation}]:\n{content}\n")

        return "\n\n".join(context_parts)


# Global prompt manager instance
_prompt_manager: Optional["PromptManager"] = None


def get_prompt_manager() -> "PromptManager":
    """Get global prompt manager instance.

    Returns:
        PromptManager instance
    """
    global _prompt_manager
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    return _prompt_manager

