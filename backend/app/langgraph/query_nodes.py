"""Individual LangGraph nodes for query analysis and processing."""

from typing import Dict, Any, List
import structlog
import re

try:
    from langgraph.graph import StateGraph
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False
    StateGraph = None

from app.langgraph.query_state import QueryState, QueryStateManager
from app.models.documents import DocumentChunk

logger = structlog.get_logger(__name__)


class QueryNodes:
    """Collection of LangGraph nodes for query processing."""

    def __init__(self, llm: Any = None):
        """Initialize query nodes.

        Args:
            llm: Optional LLM for query analysis
        """
        self.llm = llm
        self.state_manager = QueryStateManager()

    def parse_query_node(self, state: QueryState) -> QueryState:
        """Node: Parse and normalize query.

        Args:
            state: Current workflow state

        Returns:
            Updated state with parsed query
        """
        try:
            logger.debug("Parsing query", query_length=len(state["query"]))

            query = state["query"].strip()

            # Normalize query
            query = re.sub(r'\s+', ' ', query)  # Normalize whitespace
            query = query.strip()

            # Extract basic keywords
            keywords = self._extract_keywords(query)

            updates = {
                "query": query,
                "keywords": keywords,
                "workflow_step": "parsed",
                "routing_path": "parse_query",
            }

            return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Query parsing failed", error=str(e), exc_info=True)
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "parse_failed",
                    "error_message": f"Query parsing failed: {str(e)}",
                },
            )

    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query.

        Args:
            query: Query string

        Returns:
            List of keywords
        """
        # Remove stop words and extract meaningful terms
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "should", "could", "may", "might", "must", "can", "what",
            "when", "where", "who", "why", "how", "which", "this", "that",
        }

        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]

        return keywords

    def detect_intent_node(self, state: QueryState) -> QueryState:
        """Node: Detect query intent and type.

        Args:
            state: Current workflow state

        Returns:
            Updated state with detected intent
        """
        try:
            logger.debug("Detecting query intent", query=state["query"][:100])

            query = state["query"].lower()

            # Intent detection patterns
            intent = None
            query_type = None

            # Factual queries: "what is", "who is", "define", "explain"
            if any(phrase in query for phrase in ["what is", "who is", "define", "explain", "tell me about"]):
                intent = "factual"
                query_type = "definition"

            # Relational queries: "how are", "related to", "connected", "relationship"
            elif any(phrase in query for phrase in ["how are", "related to", "connected", "relationship", "link between"]):
                intent = "relational"
                query_type = "relationship"

            # Temporal queries: "when", "before", "after", "timeline", "sequence"
            elif any(phrase in query for phrase in ["when", "before", "after", "timeline", "sequence", "order"]):
                intent = "temporal"
                query_type = "temporal_query"

            # Synthesis queries: "summarize", "overview", "main points", "key findings"
            elif any(phrase in query for phrase in ["summarize", "overview", "main points", "key findings", "summary"]):
                intent = "synthesis"
                query_type = "summary"

            # Comparison queries: "compare", "difference", "versus", "vs", "similar"
            elif any(phrase in query for phrase in ["compare", "difference", "versus", " vs ", "similar", "different"]):
                intent = "comparison"
                query_type = "comparison"

            # Default to factual if unclear
            if intent is None:
                intent = "factual"
                query_type = "general"

            updates = {
                "query_intent": intent,
                "query_type": query_type,
                "workflow_step": "intent_detected",
                "routing_path": "detect_intent",
            }

            logger.debug(
                "Query intent detected",
                intent=intent,
                query_type=query_type,
            )

            return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Intent detection failed", error=str(e), exc_info=True)
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "intent_detection_failed",
                    "error_message": f"Intent detection failed: {str(e)}",
                },
            )

    def extract_entities_node(self, state: QueryState) -> QueryState:
        """Node: Extract entities from query.

        Args:
            state: Current workflow state

        Returns:
            Updated state with extracted entities
        """
        try:
            logger.debug("Extracting entities from query")

            query = state["query"]
            entities = []

            # Extract potential entities using patterns
            # Person names (capitalized words)
            person_pattern = r'\b[A-Z][a-z]+ [A-Z][a-z]+\b'
            persons = re.findall(person_pattern, query)
            for person in set(persons):
                entities.append({
                    "text": person,
                    "type": "PERSON",
                    "confidence": 0.7,
                })

            # Dates
            date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b|\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\b'
            dates = re.findall(date_pattern, query, re.IGNORECASE)
            for date in set(dates):
                entities.append({
                    "text": date if isinstance(date, str) else " ".join(date),
                    "type": "DATE",
                    "confidence": 0.8,
                })

            # Organizations
            org_pattern = r'\b[A-Z][a-zA-Z]+ (Inc|Corp|LLC|Ltd|Company|Corporation)\b'
            orgs = re.findall(org_pattern, query)
            for org in set(orgs):
                entities.append({
                    "text": org if isinstance(org, str) else " ".join(org),
                    "type": "ORGANIZATION",
                    "confidence": 0.75,
                })

            # File/document references
            file_pattern = r'\b[A-Za-z0-9_-]+\.(pdf|doc|docx|txt|py|js|java|cpp)\b'
            files = re.findall(file_pattern, query, re.IGNORECASE)
            for file_match in set(files):
                entities.append({
                    "text": file_match,
                    "type": "FILE",
                    "confidence": 0.9,
                })

            updates = {
                "entities": entities,
                "workflow_step": "entities_extracted",
                "routing_path": "extract_entities",
            }

            logger.debug(
                "Entities extracted",
                entity_count=len(entities),
            )

            return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Entity extraction failed", error=str(e), exc_info=True)
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "entity_extraction_failed",
                    "error_message": f"Entity extraction failed: {str(e)}",
                },
            )

    def retrieve_vector_node(self, state: QueryState) -> QueryState:
        """Node: Retrieve documents using vector similarity.

        Args:
            state: Current workflow state

        Returns:
            Updated state with retrieved chunks
        """
        try:
            logger.debug("Retrieving documents using vector similarity")

            # This would integrate with vector store
            # For now, return empty list as placeholder
            retrieved_chunks: List[DocumentChunk] = []

            updates = {
                "retrieved_chunks": retrieved_chunks,
                "selected_retrieval_strategies": "vector",
                "workflow_step": "vector_retrieved",
                "routing_path": "retrieve_vector",
                "stats": {"retrieval_count": len(retrieved_chunks)},
            }

            return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Vector retrieval failed", error=str(e), exc_info=True)
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "vector_retrieval_failed",
                    "error_message": f"Vector retrieval failed: {str(e)}",
                },
            )

    def retrieve_keyword_node(self, state: QueryState) -> QueryState:
        """Node: Retrieve documents using keyword search.

        Args:
            state: Current workflow state

        Returns:
            Updated state with retrieved chunks
        """
        try:
            logger.debug("Retrieving documents using keyword search")

            # This would integrate with keyword search
            # For now, return empty list as placeholder
            retrieved_chunks: List[DocumentChunk] = []

            # Append to existing retrieved chunks
            existing_chunks = state["retrieved_chunks"]
            all_chunks = existing_chunks + retrieved_chunks

            updates = {
                "retrieved_chunks": retrieved_chunks,  # Only new chunks
                "selected_retrieval_strategies": "keyword",
                "workflow_step": "keyword_retrieved",
                "routing_path": "retrieve_keyword",
                "stats": {"retrieval_count": len(all_chunks)},
            }

            return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Keyword retrieval failed", error=str(e), exc_info=True)
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "keyword_retrieval_failed",
                    "error_message": f"Keyword retrieval failed: {str(e)}",
                },
            )

    def traverse_graph_node(self, state: QueryState) -> QueryState:
        """Node: Traverse knowledge graph for relationships.

        Args:
            state: Current workflow state

        Returns:
            Updated state with graph traversal results
        """
        try:
            logger.debug("Traversing knowledge graph")

            # This would integrate with knowledge graph
            # For now, return empty list as placeholder
            graph_results: List[Dict[str, Any]] = []

            updates = {
                "graph_traversal_results": graph_results,
                "selected_retrieval_strategies": "graph",
                "workflow_step": "graph_traversed",
                "routing_path": "traverse_graph",
            }

            return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Graph traversal failed", error=str(e), exc_info=True)
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "graph_traversal_failed",
                    "error_message": f"Graph traversal failed: {str(e)}",
                },
            )

    def assemble_context_node(self, state: QueryState) -> QueryState:
        """Node: Assemble context from retrieved documents.

        Args:
            state: Current workflow state

        Returns:
            Updated state with assembled context
        """
        try:
            logger.debug("Assembling context from retrieved documents")

            # Combine all retrieved chunks
            all_chunks = state["retrieved_chunks"]
            graph_results = state["graph_traversal_results"]

            # Build context documents
            context_documents = []

            for chunk in all_chunks:
                context_documents.append({
                    "chunk_id": chunk.chunk_id,
                    "file_id": chunk.file_id,
                    "content": chunk.content[:500],  # Truncate for context
                    "metadata": chunk.metadata,
                })

            # Add graph traversal results
            for result in graph_results:
                context_documents.append({
                    "type": "graph_relationship",
                    "data": result,
                })

            updates = {
                "context_documents": context_documents,
                "workflow_step": "context_assembled",
                "routing_path": "assemble_context",
            }

            logger.debug(
                "Context assembled",
                document_count=len(context_documents),
            )

            return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Context assembly failed", error=str(e), exc_info=True)
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "context_assembly_failed",
                    "error_message": f"Context assembly failed: {str(e)}",
                },
            )

    def synthesize_answer_node(self, state: QueryState) -> QueryState:
        """Node: Synthesize final answer from context.

        Args:
            state: Current workflow state

        Returns:
            Updated state with final answer
        """
        try:
            logger.debug("Synthesizing answer")

            query = state["query"]
            context_documents = state["context_documents"]
            query_intent = state["query_intent"]

            # Build context text
            context_text = "\n\n".join([
                doc.get("content", "") if isinstance(doc, dict) else str(doc)
                for doc in context_documents[:10]  # Limit context size
            ])

            # Generate answer based on intent
            if query_intent == "synthesis":
                answer = self._generate_synthesis_answer(query, context_text)
            elif query_intent == "comparison":
                answer = self._generate_comparison_answer(query, context_text)
            elif query_intent == "relational":
                answer = self._generate_relational_answer(query, context_text)
            elif query_intent == "temporal":
                answer = self._generate_temporal_answer(query, context_text)
            else:
                answer = self._generate_factual_answer(query, context_text)

            # Extract citations
            citations = self._extract_citations(context_documents)

            # Calculate confidence
            confidence = self._calculate_confidence(context_documents, answer)

            updates = {
                "final_answer": answer,
                "citations": citations,
                "confidence_score": confidence,
                "workflow_step": "answer_synthesized",
                "routing_path": "synthesize_answer",
            }

            logger.info(
                "Answer synthesized",
                answer_length=len(answer),
                citation_count=len(citations),
                confidence=confidence,
            )

            return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Answer synthesis failed", error=str(e), exc_info=True)
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "synthesis_failed",
                    "error_message": f"Answer synthesis failed: {str(e)}",
                },
            )

    def _generate_factual_answer(self, query: str, context: str) -> str:
        """Generate factual answer.

        Args:
            query: User query
            context: Context documents

        Returns:
            Answer string
        """
        # Placeholder - would use LLM in production
        if context:
            return f"Based on the documents: {context[:200]}..."
        return "I couldn't find relevant information to answer your question."

    def _generate_synthesis_answer(self, query: str, context: str) -> str:
        """Generate synthesis answer.

        Args:
            query: User query
            context: Context documents

        Returns:
            Answer string
        """
        # Placeholder - would use LLM in production
        if context:
            return f"Summary: {context[:300]}..."
        return "I couldn't find information to summarize."

    def _generate_comparison_answer(self, query: str, context: str) -> str:
        """Generate comparison answer.

        Args:
            query: User query
            context: Context documents

        Returns:
            Answer string
        """
        # Placeholder - would use LLM in production
        if context:
            return f"Comparison: {context[:300]}..."
        return "I couldn't find information to compare."

    def _generate_relational_answer(self, query: str, context: str) -> str:
        """Generate relational answer.

        Args:
            query: User query
            context: Context documents

        Returns:
            Answer string
        """
        # Placeholder - would use LLM in production
        if context:
            return f"Relationship: {context[:300]}..."
        return "I couldn't find relationship information."

    def _generate_temporal_answer(self, query: str, context: str) -> str:
        """Generate temporal answer.

        Args:
            query: User query
            context: Context documents

        Returns:
            Answer string
        """
        # Placeholder - would use LLM in production
        if context:
            return f"Timeline: {context[:300]}..."
        return "I couldn't find temporal information."

    def _extract_citations(self, context_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract citations from context documents.

        Args:
            context_documents: List of context documents

        Returns:
            List of citation dictionaries
        """
        citations = []

        for doc in context_documents:
            if isinstance(doc, dict) and "chunk_id" in doc:
                citations.append({
                    "file_id": doc.get("file_id"),
                    "chunk_id": doc.get("chunk_id"),
                    "file_name": doc.get("metadata", {}).get("file_name", "Unknown"),
                    "page_number": doc.get("metadata", {}).get("page_number"),
                })

        return citations

    def _calculate_confidence(
        self,
        context_documents: List[Dict[str, Any]],
        answer: str,
    ) -> float:
        """Calculate confidence score for answer.

        Args:
            context_documents: Context documents
            answer: Generated answer

        Returns:
            Confidence score (0.0 to 1.0)
        """
        if not context_documents:
            return 0.0

        if not answer or len(answer) < 10:
            return 0.0

        # Simple confidence based on context availability
        doc_count = len(context_documents)
        base_confidence = min(doc_count / 5.0, 1.0)  # Max confidence with 5+ docs

        return float(base_confidence)

    def error_handler_node(self, state: QueryState) -> QueryState:
        """Node: Handle errors and attempt recovery.

        Args:
            state: Current workflow state

        Returns:
            Updated state with error handling
        """
        try:
            logger.warning(
                "Error handler invoked",
                workflow_step=state["workflow_step"],
                error_message=state.get("error_message"),
            )

            # Try to continue with available context
            if state["retrieved_chunks"] or state["context_documents"]:
                updates = {
                    "workflow_step": "error_recovered",
                    "error_message": None,
                }
                return self.state_manager.update_state(state, updates)
            else:
                # No context available, end workflow
                updates = {
                    "workflow_step": "failed",
                    "final_answer": "I encountered an error processing your query. Please try rephrasing it.",
                }
                return self.state_manager.update_state(state, updates)

        except Exception as e:
            logger.error("Error handler failed", error=str(e))
            return self.state_manager.update_state(
                state,
                {
                    "workflow_step": "critical_failure",
                    "error_message": f"Error handler failed: {str(e)}",
                    "final_answer": "A critical error occurred. Please try again later.",
                },
            )

