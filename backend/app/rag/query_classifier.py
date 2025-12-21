"""Query classification system for adaptive retrieval strategies.

Uses lightweight LLM (Ollama) to categorize user queries into distinct types
requiring different retrieval approaches.
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import json
import re
import structlog

from app.rag.llm_chains import OllamaLLM
from app.rag.content_type_detector import ContentTypeDetector, ContentType

logger = structlog.get_logger(__name__)


class QueryType(str, Enum):
    """Query categories requiring different retrieval strategies."""
    
    FACTUAL_SPECIFIC = "FACTUAL_SPECIFIC"  # Questions about specific files/content types
    FACTUAL_GENERAL = "FACTUAL_GENERAL"    # Broad overview questions
    RELATIONSHIP = "RELATIONSHIP"          # Patterns/connections across documents
    COMPARISON = "COMPARISON"              # Comparing entities/documents
    ENTITY_SEARCH = "ENTITY_SEARCH"        # Finding all mentions of something
    TEMPORAL = "TEMPORAL"                  # Time-based queries


@dataclass
class QueryUnderstanding:
    """Complete understanding of a user query."""
    
    query_type: QueryType
    confidence: float  # 0.0 to 1.0
    content_type: Optional[str] = None  # Detected content type (audio, image, etc.)
    entities: Optional[List[str]] = None  # Extracted entities (proper nouns, quoted phrases)
    file_references: Optional[List[str]] = None  # Referenced file names
    explanation: str = ""  # Human-readable explanation
    
    def __post_init__(self):
        """Initialize empty lists if None."""
        if self.entities is None:
            object.__setattr__(self, "entities", [])
        if self.file_references is None:
            object.__setattr__(self, "file_references", [])


# Classification prompt template
CLASSIFICATION_PROMPT = """You are a query classifier for a document search system. Analyze the user query and classify it into one of these categories:

1. FACTUAL_SPECIFIC - Questions about specific files or content types
   Examples: "What's in the audio file?", "Describe the screenshot", "What does LICENSE say?", "List all colleges in the csv file"
   
2. FACTUAL_GENERAL - Broad overview questions
   Examples: "What is this folder about?", "Summarize the contents"
   
3. RELATIONSHIP - Questions seeking patterns/connections across documents
   Examples: "What are common themes?", "Find patterns across files", "How do these relate?"
   
4. COMPARISON - Questions comparing entities/documents
   Examples: "Compare X and Y", "Difference between A and B", "X versus Y"
   
5. ENTITY_SEARCH - Questions seeking all mentions of something
   Examples: "Find all mentions of X", "Where is Y discussed?", "Every reference to Z"
   
6. TEMPORAL - Time-based queries
   Examples: "What changed recently?", "Latest updates", "Most recent files"

Content type detection:
- "csv file" or "csv" → content_type: "text" (CSV files are text/data files)
- "pdf" or "document" or "resume" → content_type: "document"
- "audio" or "wav" → content_type: "audio"
- "image" or "screenshot" → content_type: "image"
- "code" or "script" → content_type: "code"

File reference detection (CRITICAL):
- When user mentions a file type (e.g., "csv file"), match files with that extension from available_files
- Example: "colleges in the csv file" → file_references should include all .csv files from available_files
- Example: "data in tax_delinquent.csv" → file_references: ["tax_delinquent.csv"]
- Always match against available_files list - use exact filename matches when possible

Entities: Extract key terms, proper nouns, quoted phrases mentioned in the query.

IMPORTANT: Respond with ONLY valid JSON. No markdown code blocks, no explanations before or after, just the raw JSON object.

{{
    "query_type": "FACTUAL_SPECIFIC",
    "confidence": 0.9,
    "content_type": "text",
    "entities": ["colleges", "names"],
    "file_references": ["tax_delinquent.csv"],
    "explanation": "Query asks for specific data from CSV file"
}}

User query: "{query}"

Available files: {available_files}

JSON:"""


class QueryClassifier:
    """Classify queries into types using lightweight LLM."""
    
    # Content type keywords for fallback detection
    CONTENT_TYPE_KEYWORDS = {
        "audio": ["audio", "sound", "recording", "whisper", "transcription", "wav", "mp3"],
        "image": ["image", "screenshot", "picture", "photo", "png", "jpg", "jpeg"],
        "video": ["video", "movie", "clip", "mp4", "mov"],
        "document": ["document", "pdf", "doc", "docx", "text", "file"],
        "code": ["code", "script", "program", "function", "class", "python"],
        "text": ["text", "csv", "tsv", "data", "table"],
    }
    
    def __init__(
        self,
        available_files: Optional[List[Any]] = None,
        llm: Optional[OllamaLLM] = None,
        use_llm: bool = True,
    ):
        """Initialize query classifier.
        
        Args:
            available_files: Optional list of file metadata dicts or Document objects with 'file_name' in metadata
                for file reference detection
            llm: Optional OllamaLLM instance (creates new one if not provided)
            use_llm: Whether to use LLM for classification (True) or fallback to pattern matching
        """
        # Normalize available_files from Document objects to dicts
        file_dicts = []
        for item in (available_files or []):
            if hasattr(item, 'metadata'):
                # LangChain Document object - extract metadata
                file_dicts.append(item.metadata)
            elif isinstance(item, dict):
                # Already a dictionary
                file_dicts.append(item)
        self.available_files = file_dicts  # Store normalized dicts
        self._file_names = {f.get("file_name", "").lower() for f in file_dicts if f.get("file_name")}
        self.use_llm = use_llm
        self.llm: Optional[OllamaLLM] = None
        self._content_type_detector = ContentTypeDetector()  # Enhanced detector
        
        if use_llm:
            try:
                # Use llama3.2:1b for query classification - lighter and faster
                llm_instance = llm or OllamaLLM(model="llama3.2:1b", temperature=0.1)
                self.llm = llm_instance
                logger.info("Query classifier initialized with LLM", model=self.llm.model_name if self.llm else "unknown")
            except Exception as e:
                logger.warning("Failed to initialize LLM for query classification, using pattern fallback", error=str(e))
                self.use_llm = False
                self.llm = None
    
    async def classify(self, query: str) -> QueryUnderstanding:
        """Classify query and extract context.
        
        Args:
            query: User query string
            
        Returns:
            QueryUnderstanding with type, confidence, and extracted context
        """
        query = query.strip()
        
        if not query:
            return QueryUnderstanding(
                query_type=QueryType.FACTUAL_GENERAL,
                confidence=0.0,
                explanation="Empty query - defaulting to general factual query",
            )
        
        if self.use_llm and self.llm:
            try:
                return await self._classify_with_llm(query)
            except Exception as e:
                logger.warning("LLM classification failed, using fallback", error=str(e))
                return self._classify_with_patterns(query)
        else:
            return self._classify_with_patterns(query)
    
    def classify_sync(self, query: str) -> QueryUnderstanding:
        """Synchronous version of classify (uses pattern matching).
        
        Args:
            query: User query string
            
        Returns:
            QueryUnderstanding with type, confidence, and extracted context
        """
        return self._classify_with_patterns(query)
    
    async def _classify_with_llm(self, query: str) -> QueryUnderstanding:
        """Classify query using LLM.
        
        Args:
            query: User query string
            
        Returns:
            QueryUnderstanding with classification results
        """
        # Prepare available files list for prompt
        file_list = ", ".join([f.get("file_name", "") for f in self.available_files[:10]])  # Limit to 10 files
        if not file_list:
            file_list = "none"
        
        # Format prompt
        prompt = CLASSIFICATION_PROMPT.format(
            query=query,
            available_files=file_list,
        )
        
        if not self.llm:
            raise ValueError("LLM not initialized")
        
        try:
            # Invoke LLM with non-streaming call to avoid classification JSON being streamed
            # Get the underlying ChatOllama and temporarily disable streaming for classification
            llm_instance = self.llm.get_llm()
            
            # Temporarily disable streaming if it's a ChatOllama instance
            original_streaming = None
            if hasattr(llm_instance, 'streaming'):
                original_streaming = llm_instance.streaming
                llm_instance.streaming = False
            
            try:
                # Ensure NO callbacks are active during classification to prevent JSON streaming
                # Use config with empty callbacks list to prevent any streaming callbacks
                response = await llm_instance.ainvoke(
                    prompt,
                    config={"callbacks": []}  # Explicitly disable all callbacks
                )
            finally:
                # Restore original streaming setting
                if original_streaming is not None:
                    llm_instance.streaming = original_streaming
            
            # Extract response text
            if hasattr(response, 'content'):
                response_text = response.content
            elif isinstance(response, str):
                response_text = response
            else:
                response_text = str(response)
            
            # Parse JSON from response (may have markdown code blocks)
            json_text = self._extract_json(response_text)
            
            if json_text:
                result = json.loads(json_text)
                return self._parse_llm_result(result, query)
            else:
                logger.warning("No JSON found in LLM response, using pattern fallback", response=response_text[:100])
                return self._classify_with_patterns(query)
                
        except json.JSONDecodeError as e:
            logger.warning("Failed to parse LLM response as JSON, using pattern fallback", error=str(e))
            return self._classify_with_patterns(query)
        except Exception as e:
            logger.error("LLM classification error", error=str(e), exc_info=True)
            return self._classify_with_patterns(query)
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from LLM response (may be wrapped in markdown code blocks).
        
        Args:
            text: LLM response text
            
        Returns:
            Extracted JSON string or None
        """
        # Try to find JSON code block
        json_block = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_block:
            return json_block.group(1)
        
        # Try to find JSON object directly
        json_obj = re.search(r'\{.*\}', text, re.DOTALL)
        if json_obj:
            return json_obj.group(0)
        
        return None
    
    def _parse_llm_result(self, result: Dict, query: str) -> QueryUnderstanding:
        """Parse LLM classification result into QueryUnderstanding.
        
        Args:
            result: Parsed JSON from LLM
            query: Original query (for validation)
            
        Returns:
            QueryUnderstanding object
        """
        try:
            # Validate and parse query_type
            query_type_str = result.get("query_type", "FACTUAL_GENERAL")
            try:
                query_type = QueryType(query_type_str)
            except ValueError:
                logger.warning(f"Invalid query_type from LLM: {query_type_str}, defaulting to FACTUAL_GENERAL")
                query_type = QueryType.FACTUAL_GENERAL
            
            # Parse confidence
            confidence = float(result.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))  # Clamp to 0-1
            
            # Parse content_type - use enhanced detector if LLM didn't provide it or validation fails
            content_type = result.get("content_type")
            if content_type and content_type not in self.CONTENT_TYPE_KEYWORDS:
                # Validate content_type, fallback to detector if invalid
                content_type = None
            
            # If LLM didn't detect content_type, use enhanced detector
            if not content_type:
                detection = self._content_type_detector.detect(query, available_files=self.available_files)
                if detection["content_type"] != ContentType.ANY:
                    content_type = detection["content_type"].value
                    # Use detector's confidence if it's higher than LLM's
                    if detection["confidence"] > confidence:
                        confidence = detection["confidence"]
            
            # Parse entities
            entities = result.get("entities", [])
            if not isinstance(entities, list):
                entities = []
            entities = [str(e) for e in entities if e]
            
            # Parse file_references
            file_refs = result.get("file_references", [])
            if not isinstance(file_refs, list):
                file_refs = []
            file_refs = [str(f) for f in file_refs if f]
            
            # Validate file references against available files
            validated_file_refs = []
            for ref in file_refs:
                ref_lower = ref.lower()
                # Find matching file (case-insensitive)
                for file_info in self.available_files:
                    file_name = file_info.get("file_name", "")
                    if ref_lower == file_name.lower() or ref_lower in file_name.lower() or file_name.lower() in ref_lower:
                        validated_file_refs.append(file_name)
                        break
            
            # Parse explanation
            explanation = result.get("explanation", "")
            if not explanation:
                explanation = f"{query_type.value.replace('_', ' ').title()} query ({int(confidence * 100)}% confidence)"
            
            return QueryUnderstanding(
                query_type=query_type,
                confidence=confidence,
                content_type=content_type,
                entities=entities,
                file_references=validated_file_refs,
                explanation=explanation,
            )
            
        except Exception as e:
            logger.error("Failed to parse LLM result", error=str(e), result=result)
            return self._classify_with_patterns(query)
    
    def _classify_with_patterns(self, query: str) -> QueryUnderstanding:
        """Fallback pattern-based classification.
        
        Args:
            query: User query string
            
        Returns:
            QueryUnderstanding with classification results
        """
        query_lower = query.lower()
        
        # Simple pattern matching as fallback
        if any(kw in query_lower for kw in ["compare", "versus", "vs", "difference", "contrast"]):
            query_type = QueryType.COMPARISON
            confidence = 0.8
        elif any(kw in query_lower for kw in ["common", "themes", "patterns", "across", "relate"]):
            query_type = QueryType.RELATIONSHIP
            confidence = 0.7
        elif any(kw in query_lower for kw in ["find all", "every mention", "all references", "where is"]):
            query_type = QueryType.ENTITY_SEARCH
            confidence = 0.7
        elif any(kw in query_lower for kw in ["recent", "latest", "changed", "updated"]):
            query_type = QueryType.TEMPORAL
            confidence = 0.7
        elif any(kw in query_lower for kw in ["audio", "screenshot", "image", "pdf", "csv", "file"]) or self._detect_file_references(query):
            query_type = QueryType.FACTUAL_SPECIFIC
            confidence = 0.8
        elif any(kw in query_lower for kw in ["about", "overview", "summarize", "folder"]):
            query_type = QueryType.FACTUAL_GENERAL
            confidence = 0.6
        else:
            query_type = QueryType.FACTUAL_GENERAL
            confidence = 0.5
        
        # Extract content type
        content_type = self._detect_content_type(query_lower)
        
        # Extract entities
        entities = self._extract_entities(query)
        
        # Detect file references
        file_refs = self._detect_file_references(query)
        
        return QueryUnderstanding(
            query_type=query_type,
            confidence=confidence,
            content_type=content_type,
            entities=entities,
            file_references=file_refs,
            explanation=f"{query_type.value.replace('_', ' ').title()} query (pattern-based, {int(confidence * 100)}% confidence)",
        )
    
    def _detect_content_type(self, query_lower: str) -> Optional[str]:
        """Detect content type from query using enhanced detector."""
        detection = self._content_type_detector.detect(query_lower, available_files=self.available_files)
        if detection["content_type"] != ContentType.ANY:
            return detection["content_type"].value
        return None
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract entities (proper nouns, quoted phrases) from query."""
        entities: List[str] = []
        
        # Extract quoted strings
        quoted = re.findall(r'"([^"]+)"|\'([^\']+)\'', query)
        for q in quoted:
            if q[0] or q[1]:
                entities.append(str(q[0] or q[1]))
        
        # Extract capitalized words (potential proper nouns)
        capitalized_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b'
        capitalized = re.findall(capitalized_pattern, query)
        entities.extend(capitalized)
        
        # Remove duplicates and filter out common words
        common_words = {"The", "This", "That", "What", "Which", "Where", "When", "How", "Why"}
        filtered_entities = [e for e in set(entities) if e not in common_words and len(e) > 2]
        
        return filtered_entities
    
    def _detect_file_references(self, query: str) -> List[str]:
        """Detect file name references in query.
        
        Matches file names by priority:
        1. Exact filename match (case-insensitive) - highest priority
        2. Extension-based match (e.g., "csv file" matches .csv files) - high priority
        3. Base name match (without extension)
        4. Word-level match (e.g., "resume" matches "RESUME_GANESH_ARPAN...pdf") - lower priority
        """
        if not self._file_names:
            return []
        
        references = []
        query_lower = query.lower()
        # Extract words from query (split on whitespace and punctuation)
        query_words = set(re.findall(r'\b\w+\b', query_lower))
        
        # Priority 1: Check for extension-based matches first (e.g., "csv file" should match .csv files)
        extension_matches = []
        for ext in [".csv", ".tsv", ".pdf", ".txt", ".docx", ".png", ".jpg", ".wav", ".mp3"]:
            ext_name = ext[1:]  # Remove the dot
            if f"{ext_name} file" in query_lower or f"{ext_name} files" in query_lower:
                # Match files with this extension
                for file_name in self._file_names:
                    if file_name.endswith(ext):
                        original_file = next(
                            (f.get("file_name") for f in self.available_files 
                             if f.get("file_name", "").lower() == file_name and f.get("file_name")),
                            None
                        )
                        if original_file and original_file not in extension_matches:
                            extension_matches.append(original_file)
        
        if extension_matches:
            return extension_matches  # Return extension matches (highest priority)
        
        # Priority 2: Check for exact filename or base name matches
        for file_name in self._file_names:
            base_name = file_name.rsplit('.', 1)[0] if '.' in file_name else file_name
            
            # Method 1: Exact filename or base name match
            if file_name in query_lower or base_name in query_lower:
                matched = True
            else:
                # Method 2: Check if any significant word from query appears in filename
                # Extract words from filename (split on underscores, hyphens, dots)
                filename_words = set(re.findall(r'\b\w+\b', base_name.lower()))
                # Match if any query word appears in filename (but skip very short words like "a", "an", "the")
                # Special handling for common terms like "resume", "cv", "pdf"
                significant_query_words = {w for w in query_words if len(w) >= 2}  # Lower threshold to catch "cv", "pdf"
                matched = bool(significant_query_words & filename_words)
            
            if matched:
                # Find original case file name
                original_file = next(
                    (f.get("file_name") for f in self.available_files 
                     if f.get("file_name", "").lower() == file_name and f.get("file_name")),
                    None
                )
                if original_file and original_file not in references:
                    references.append(original_file)
        
        return references
    
    def update_available_files(self, available_files: List[Any]) -> None:
        """Update available files for file reference detection.
        
        Args:
            available_files: List of file metadata dicts or Document objects with 'file_name' in metadata
        """
        # Convert Document objects to dicts by extracting metadata
        file_dicts = []
        for item in available_files:
            if hasattr(item, 'metadata'):
                # LangChain Document object - extract metadata
                file_dicts.append(item.metadata)
            elif isinstance(item, dict):
                # Already a dictionary
                file_dicts.append(item)
        
        self.available_files = file_dicts
        self._file_names = {f.get("file_name", "").lower() for f in file_dicts if f.get("file_name")}


# Global classifier instance
_classifier: Optional[QueryClassifier] = None


def get_query_classifier(
    available_files: Optional[List[Any]] = None,
    llm: Optional[OllamaLLM] = None,
    use_llm: bool = True,
) -> QueryClassifier:
    """Get global query classifier instance.
    
    Args:
        available_files: Optional list of file metadata dicts
        llm: Optional OllamaLLM instance
        use_llm: Whether to use LLM for classification
        
    Returns:
        QueryClassifier instance
    """
    global _classifier
    if _classifier is None:
        _classifier = QueryClassifier(available_files=available_files, llm=llm, use_llm=use_llm)
    elif available_files is not None:
        _classifier.update_available_files(available_files)
    return _classifier
