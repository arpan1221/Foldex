"""Named Entity Recognition (NER) for entity extraction."""

from typing import List, Dict
import re
import structlog

logger = structlog.get_logger(__name__)

# Try to import spaCy, fallback to regex-only if not available
try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None


class EntityExtractor:
    """Extracts entities from document chunks using spaCy and regex."""

    def __init__(self):
        """Initialize entity extractor."""
        self.nlp = None
        if SPACY_AVAILABLE:
            try:
                # Try to load spaCy model
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for entity extraction")
            except OSError:
                logger.warning(
                    "spaCy model 'en_core_web_sm' not found. "
                    "Install with: python -m spacy download en_core_web_sm. "
                    "Falling back to regex-only extraction."
                )
                self.nlp = None
        else:
            logger.warning(
                "spaCy not available. Using regex-only entity extraction. "
                "Install with: pip install spacy && python -m spacy download en_core_web_sm"
            )

    def _calculate_confidence(self, entity_text: str, entity_type: str) -> float:
        """Calculate confidence score for an entity.
        
        Args:
            entity_text: Entity text
            entity_type: Entity type
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        # Higher confidence for longer entities and specific types
        base_confidence = 0.7
        
        if len(entity_text) > 3:
            base_confidence += 0.1
        if len(entity_text) > 10:
            base_confidence += 0.1
            
        # Code entities are usually more reliable
        if entity_type in ["FUNCTION", "CLASS", "MODULE", "VARIABLE"]:
            base_confidence += 0.1
            
        return min(base_confidence, 1.0)

    def _extract_code_entities(self, text: str) -> List[Dict]:
        """Extract code-specific entities using regex.
        
        Args:
            text: Text to extract entities from
            
        Returns:
            List of entity dictionaries
        """
        entities = []
        
        # Functions: def functionName( or function functionName(
        for match in re.finditer(r'\b(?:def|function)\s+(\w+)\s*\(', text):
            entities.append({
                "text": match.group(1),
                "type": "FUNCTION",
                "start_char": match.start(1),
                "end_char": match.end(1),
                "confidence": self._calculate_confidence(match.group(1), "FUNCTION"),
            })
        
        # Classes: class ClassName
        for match in re.finditer(r'\bclass\s+(\w+)', text):
            entities.append({
                "text": match.group(1),
                "type": "CLASS",
                "start_char": match.start(1),
                "end_char": match.end(1),
                "confidence": self._calculate_confidence(match.group(1), "CLASS"),
            })
        
        # Imports: import X, from X import Y
        for match in re.finditer(r'\b(?:import|from)\s+(\w+(?:\.\w+)*)', text):
            entities.append({
                "text": match.group(1),
                "type": "MODULE",
                "start_char": match.start(1),
                "end_char": match.end(1),
                "confidence": self._calculate_confidence(match.group(1), "MODULE"),
            })
        
        # Variables: const/let/var variableName = or variableName:
        for match in re.finditer(r'\b(?:const|let|var)\s+(\w+)\s*[=:]', text):
            entities.append({
                "text": match.group(1),
                "type": "VARIABLE",
                "start_char": match.start(1),
                "end_char": match.end(1),
                "confidence": self._calculate_confidence(match.group(1), "VARIABLE"),
            })
        
        # File references: "file.py" or 'file.py' or file.py
        for match in re.finditer(r'["\']?([\w\-_]+\.(?:py|js|ts|tsx|jsx|md|txt|json|yaml|yml))["\']?', text):
            entities.append({
                "text": match.group(1),
                "type": "FILE",
                "start_char": match.start(1),
                "end_char": match.end(1),
                "confidence": 0.9,  # File names are usually reliable
            })
        
        return entities

    async def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities from text.

        Args:
            text: Text to extract entities from

        Returns:
            List of entity dictionaries with type, text, start, end, and confidence
        """
        try:
            logger.debug("Extracting entities", text_length=len(text))
            entities = []
            
            # Use spaCy for standard NER if available
            if self.nlp:
                doc = self.nlp(text)
                for ent in doc.ents:
                    entities.append({
                        "text": ent.text,
                        "type": ent.label_,
                        "start_char": ent.start_char,
                        "end_char": ent.end_char,
                        "confidence": self._calculate_confidence(ent.text, ent.label_),
                    })
            
            # Always extract code entities using regex
            code_entities = self._extract_code_entities(text)
            entities.extend(code_entities)
            
            # Deduplicate entities (same text and type at same position)
            seen = set()
            unique_entities = []
            for entity in entities:
                key = (entity["text"], entity["type"], entity["start_char"])
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(entity)
            
            logger.debug(
                "Entity extraction completed",
                entity_count=len(unique_entities),
                spacy_used=self.nlp is not None,
            )
            
            return unique_entities
            
        except Exception as e:
            logger.error("Entity extraction failed", error=str(e), exc_info=True)
            return []

