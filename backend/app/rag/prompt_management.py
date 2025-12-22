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

        # File-specific Prompt - for queries about a single file
        prompts["file_specific"] = self._create_file_specific_prompt()

        return prompts

    def _create_factual_prompt(self) -> PromptTemplate:
        """Create prompt template for factual queries.

        Returns:
            PromptTemplate for factual queries (RetrievalQA compatible)
        """
        template = """Answer the question using the provided context.

Context:
{context}

Question: {question}

Instructions:
- Provide a comprehensive answer that fully addresses the question
- For simple questions (e.g., "What is X?"), a brief answer is appropriate
- For detailed questions (e.g., "Explain X", "Describe Y", "Summarize Z"), provide a thorough explanation
- Use information from all relevant document chunks
- Add citations [1], [2] after each claim
- No thinking tags or meta-commentary
- Adapt response length to the complexity and scope of the question

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
        template = """Analyze ALL documents in the context and provide a comprehensive synthesis.

Context:
{context}

Question: {question}

Instructions:
- Analyze patterns, themes, and connections across ALL files mentioned in the context
- For similarity queries (e.g., "are there similarities across files?"), explicitly compare files and identify:
  * Common themes, topics, or concepts shared across multiple files
  * Similar approaches, methods, or techniques used in different files
  * Overlapping entities, people, or organizations mentioned
  * Complementary or related content between files
- Be thorough and specific - cite exact examples from files
- Discuss both similarities AND differences when relevant
- Group similar files together in your analysis
- Use citations [1], [2] after each claim referencing specific files
- Be concise but comprehensive - avoid repetition
- No thinking tags, meta-commentary, or repeating phrases
- If no similarities exist, state that clearly

Answer directly and clearly:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def _create_relationship_prompt(self) -> PromptTemplate:
        """Create prompt template for relationship queries.

        Returns:
            PromptTemplate for relationship queries (RetrievalQA compatible)
        """
        template = """Analyze the files in the context and identify relationships, similarities, and connections.

Context:
{context}

Question: {question}

Instructions:
- For similarity queries (e.g., "are there similarities across files?"), explicitly identify:
  * What themes, topics, or concepts appear in multiple files
  * Which files share similar content, approaches, or focus
  * Common entities, methods, or ideas across files
  * How files complement or relate to each other
- Compare files directly and state similarities clearly
- Be specific - cite exact examples from the files
- Group similar files together in your response
- If similarities exist, describe them clearly with file references
- If no clear similarities exist, state that explicitly
- Add citations [1], [2] after each claim referencing specific files
- Be concise and direct - avoid repetition or garbled text
- No thinking tags, meta-commentary, or repeating phrases
- Write clearly and stop when you've answered the question

Answer directly:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def _create_file_specific_prompt(self) -> PromptTemplate:
        """Create prompt template for file-specific queries (when user chats with a single file).

        Returns:
            PromptTemplate for file-specific queries (RetrievalQA compatible)
        """
        template = """Answer the question using ONLY the information from the provided context chunks.

Context (from the file):
{context}

Question: {question}

Instructions:
- Use ONLY the information from the context chunks above (they are all from the same file)
- Provide a clear, comprehensive answer that fully addresses the question
- Be specific and cite exact examples from the context using [1], [2] citations
- Write clearly and concisely - avoid repetition or repeating the same information
- Stop when you've fully answered the question
- Do NOT repeat the question or rephrase it unnecessarily
- Do NOT repeat the same information multiple times in different ways
- No thinking tags, meta-commentary, or garbled/repetitive text

CITATION REQUIREMENTS:
- Add inline citations like [1], [2] immediately after claims from the context
- Only cite sources you actually used from the context above
- Multiple citations for one claim: [1][2]

Answer directly and clearly:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def _create_default_prompt(self) -> PromptTemplate:
        """Create default prompt template for general queries.

        Returns:
            PromptTemplate for general queries (RetrievalQA compatible)
        """
        template = """Answer the question using information from ALL files in the context.

Context:
{context}

Question: {question}

Instructions:
- Provide a complete and thorough answer that fully addresses the question
- Use information from ALL files in the context, not just one
- For similarity/comparison queries, explicitly identify and describe:
  * Common themes, topics, or concepts across files
  * Similar approaches, methods, or content in different files
  * How files relate to each other
- Adapt your response length to match the query complexity
- Be specific and cite exact examples from files
- Add citations [1], [2] after each claim referencing specific files
- Be concise - avoid repetition or repeating the same information multiple times
- Write clearly and stop when the question is fully answered
- No thinking tags, meta-commentary, or garbled/repetitive text

CITATION REQUIREMENTS:
- Add inline citations like [1], [2] immediately after claims from sources
- Only cite sources you actually used from the context above
- Multiple citations for one claim: [1][2]

Answer directly and clearly:"""

        return PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )

    def get_prompt(
        self, query_type: str = "default", file_id: Optional[str] = None
    ) -> PromptTemplate:
        """Get prompt template for query type.

        Args:
            query_type: Type of query (factual, synthesis, relationship, default, file_specific)
            file_id: Optional file_id - if provided, use file_specific prompt

        Returns:
            PromptTemplate instance

        Raises:
            ProcessingError: If query type is not supported
        """
        # If file_id is provided, use file-specific prompt
        if file_id:
            logger.debug("Using file-specific prompt", file_id=file_id)
            return self.prompts.get("file_specific", self.prompts["default"])
        
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

        # Relationship indicators (includes similarity queries)
        relationship_keywords = [
            "relationship",
            "related",
            "connection",
            "link",
            "compare",
            "difference",
            "similar",
            "similarities",
            "similarity",
            "similar to",
            "same",
            "common",
            "shared",
            "overlap",
            "associate",
            "correlate",
            "depend",
            "influence",
            "cause",
            "effect",
            "in common",
            "have in common",
        ]

        # Synthesis indicators (cross-file analysis)
        synthesis_keywords = [
            "summarize",
            "synthesize",
            "overall",
            "general",
            "all documents",
            "all files",
            "across",
            "across files",
            "across documents",
            "multiple",
            "multiple files",
            "combine",
            "integrate",
            "pattern",
            "patterns",
            "theme",
            "themes",
            "together",
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

    def format_folder_summary(self, folder_summary: Optional[Dict[str, Any]]) -> str:
        """Format folder summary for inclusion in prompt context.
        
        Args:
            folder_summary: Folder summary dictionary from database
            
        Returns:
            Formatted folder summary string, or empty string if None
        """
        if not folder_summary:
            return ""
        
        lines = []
        lines.append("=== FOLDER OVERVIEW ===")
        
        # Master summary (contains cross-file similarities)
        if folder_summary.get("summary"):
            lines.append(f"\n{folder_summary['summary']}")
        
        # File type distribution
        file_type_dist = folder_summary.get("file_type_distribution", {})
        if file_type_dist:
            type_parts = [f"{count} {ftype}" for ftype, count in sorted(file_type_dist.items())]
            lines.append(f"\nFile Types: {', '.join(type_parts)}")
        
        # Cross-file relationships (CRITICAL for similarity queries)
        relationship_summary = folder_summary.get("relationship_summary", [])
        if relationship_summary and len(relationship_summary) > 0:
            lines.append("\n--- Cross-File Relationships & Similarities ---")
            for rel in relationship_summary[:5]:  # Top 5 relationships
                source = rel.get('source_file', 'File')
                target = rel.get('target_file', 'File')
                rel_type = rel.get('relationship_type', 'related')
                confidence = rel.get('confidence', 0)
                lines.append(f"• {source} ↔ {target}: {rel_type} (confidence: {confidence:.0%})")
        
        # Common themes across files
        insights = folder_summary.get("insights") or {}
        top_themes = insights.get("top_themes", []) if isinstance(insights, dict) else []
        if top_themes:
            lines.append(f"\nCommon Themes Across Files: {', '.join(top_themes)}")
        
        # Top entities (brief)
        entity_summary = folder_summary.get("entity_summary") or {}
        top_entities = entity_summary.get("top_entities", []) if isinstance(entity_summary, dict) else []
        if top_entities:
            entity_names = [e.get("entity", "") for e in top_entities[:5]]
            if entity_names:
                lines.append(f"Key Entities: {', '.join(entity_names)}")
        
        lines.append("\n" + "=" * 25 + "\n")
        
        return "\n".join(lines)

    def format_context(
        self, 
        documents: list, 
        numbered: bool = True, 
        include_relationships: bool = True,
        folder_summary: Optional[Dict[str, Any]] = None,
        file_id: Optional[str] = None
    ) -> str:
        """Format retrieved documents as context string with optional numbering for citations.

        Args:
            documents: List of document objects (LangChain Documents or DocumentChunks)
            numbered: If True, number each chunk for inline citations
            include_relationships: If True, include document relationship analysis
            folder_summary: Optional folder summary dictionary to prepend to context
            file_id: Optional file_id - if provided, skip folder summary (file-specific query)

        Returns:
            Formatted context string with optionally numbered chunks, grouped by file
        """
        # Format folder summary if provided (skip for file-specific queries)
        folder_summary_text = ""
        if folder_summary and not file_id:
            folder_summary_text = self.format_folder_summary(folder_summary)
        
        if not documents:
            return folder_summary_text

        # Group documents by file
        from collections import defaultdict
        file_groups = defaultdict(list)

        for idx, doc in enumerate(documents, 1):
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
            start_time = metadata.get("start_time")
            end_time = metadata.get("end_time")

            # Build source reference
            location_parts = []
            if page_number is not None:
                location_parts.append(f"p.{page_number}")
            if start_time is not None and end_time is not None:
                location_parts.append(f"{start_time:.1f}s-{end_time:.1f}s")

            location = f", {', '.join(location_parts)}" if location_parts else ""

            # Store with metadata for grouping
            file_groups[file_name].append({
                "idx": idx,
                "content": content,
                "location": location,
            })

        # Build formatted context grouped by file
        context_parts = []
        
        # Prepend folder summary if provided
        if folder_summary_text:
            context_parts.append(folder_summary_text)

        # Add header with file count and relationships
        unique_files = len(file_groups)
        if unique_files > 1:
            file_list = ", ".join(f'"{name}"' for name in sorted(file_groups.keys()))
            context_parts.append(f"IMPORTANT: This context contains excerpts from {unique_files} different files: {file_list}")
            context_parts.append("You MUST analyze all of these files in your response.")

            # Add document relationship analysis if enabled
            if include_relationships:
                from app.knowledge_graph.document_relationships import get_document_relationship_detector
                detector = get_document_relationship_detector()
                relationship_data = detector.detect_document_relationships(documents)

                # Add shared themes if found
                shared_themes = relationship_data.get("shared_themes", [])
                if shared_themes:
                    top_themes = [t["theme"] for t in shared_themes[:5]]
                    context_parts.append(f"Shared themes across files: {', '.join(top_themes)}")

                # Add document topics
                document_topics = relationship_data.get("document_topics", {})
                if document_topics:
                    topic_summary = []
                    for file_name in sorted(document_topics.keys())[:3]:  # Top 3 files
                        topics = document_topics[file_name][:3]  # Top 3 topics per file
                        if topics:
                            topic_summary.append(f"{file_name}: {', '.join(topics)}")
                    if topic_summary:
                        context_parts.append("Document topics: " + " | ".join(topic_summary))

            context_parts.append("")  # Blank line separator

        # Add chunks grouped by file
        for file_name in sorted(file_groups.keys()):
            chunks = file_groups[file_name]

            # Add file separator
            if unique_files > 1:
                context_parts.append(f"--- File: {file_name} ({len(chunks)} excerpt{'s' if len(chunks) != 1 else ''}) ---")

            for chunk in chunks:
                if numbered:
                    source_ref = f"[{chunk['idx']}] {file_name}{chunk['location']}"
                else:
                    source_ref = f"{file_name}{chunk['location']}"

                context_parts.append(f"{source_ref}:\n{chunk['content']}\n")

        return "\n".join(context_parts).strip()


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

