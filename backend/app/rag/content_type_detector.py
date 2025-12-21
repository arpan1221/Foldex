"""Enhanced content type detection from user queries.

Detects content types, file references, multi-type queries, and exclusion patterns.
Integrates with query classifier to provide intelligent filtering.
"""

import re
from typing import List, Dict, Optional, Union, Any
from enum import Enum
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


class ContentType(str, Enum):
    """Content types that can be detected from queries."""

    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"  # PDF, text, markdown
    CODE = "code"
    IMAGE = "image"
    TEXT = "text"  # CSV, TSV, plain text data
    ANY = "any"  # No specific type detected


@dataclass
class ContentTypeDetection:
    """Structured detection result."""
    
    filter_type: str  # "file_type" | "file_id" | "multi_type" | "exclude" | "none"
    filter_value: Union[str, List[str], Dict[str, Any], None]
    content_type: Optional[ContentType] = None
    content_types: Optional[List[ContentType]] = None  # For multi-type
    confidence: float = 0.0
    chromadb_where_clause: Optional[Dict[str, Any]] = None
    explanation: str = ""
    excluded_files: Optional[List[str]] = None  # For exclusion filters


class ContentTypeDetector:
    """
    Enhanced content type detector with confidence scoring, multi-type support,
    explicit file detection, and exclusion patterns.
    
    Uses keyword patterns - NO LLM needed.
    Fast, deterministic, and works offline.
    """

    def __init__(self):
        """Initialize content type detector with expanded keyword patterns."""
        # Expanded keyword patterns with confidence weights
        self.patterns = {
            ContentType.AUDIO: {
                "keywords": {
                    "audio": 1.0,  # Explicit keyword = high confidence
                    "recording": 1.0,
                    "sound": 0.9,
                    "voice": 0.9,
                    "speech": 0.9,
                    "said": 0.6,  # Implicit reference = lower confidence
                    "mentioned": 0.6,
                    "spoken": 0.7,
                    "talk": 0.6,
                    "conversation": 0.8,
                    "interview": 0.9,
                    "podcast": 0.95,
                    "call": 0.8,
                    "meeting audio": 0.95,
                    "voice memo": 0.9,
                    "transcription": 0.95,
                    "transcript": 0.95,
                    "listen": 0.7,
                    "heard": 0.6,
                    "speaker": 0.7,
                    "audio file": 1.0,
                },
                "file_extensions": {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".mp4", ".wma", ".aac"},
                "file_types": {"audio"},
            },
            ContentType.VIDEO: {
                "keywords": {
                    "video": 1.0,
                    "visual": 0.7,
                    "showed": 0.6,
                    "screen": 0.7,
                    "footage": 0.9,
                    "clip": 0.8,
                    "movie": 0.9,
                    "film": 0.9,
                },
                "file_extensions": {".mp4", ".mov", ".avi", ".mkv", ".webm", ".flv"},
                "file_types": {"video"},
            },
            ContentType.DOCUMENT: {
                "keywords": {
                    "paper": 0.9,
                    "document": 0.8,
                    "pdf": 1.0,
                    "resume": 1.0,
                    "cv": 1.0,
                    "article": 0.9,
                    "report": 0.9,
                    "publication": 0.9,
                    "page": 0.6,
                    "section": 0.6,
                    "chapter": 0.7,
                    "paragraph": 0.6,
                    "written": 0.7,
                    "text": 0.5,  # Generic, lower confidence
                    "reads": 0.6,
                    "states": 0.6,
                    "according to": 0.6,
                    "says in": 0.6,
                    "mentions in": 0.6,
                },
                "file_extensions": {".pdf", ".txt", ".md", ".doc", ".docx", ".rtf", ".odt"},
                "file_types": {"pdf", "text", "markdown", "document"},
            },
            ContentType.IMAGE: {
                "keywords": {
                    "image": 1.0,
                    "screenshot": 1.0,
                    "picture": 0.9,
                    "photo": 0.9,
                    "photograph": 0.9,
                    "graphic": 0.8,
                    "diagram": 0.8,
                    "chart": 0.8,
                    "visualization": 0.8,
                    "visual": 0.7,
                    "png": 1.0,
                    "jpg": 1.0,
                    "jpeg": 1.0,
                    "gif": 1.0,
                },
                "file_extensions": {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".svg", ".webp", ".tiff"},
                "file_types": {"image"},
            },
            ContentType.CODE: {
                "keywords": {
                    "code": 1.0,
                    "function": 0.8,
                    "class": 0.8,
                    "method": 0.8,
                    "implementation": 0.9,
                    "algorithm": 0.8,
                    "script": 0.9,
                    "program": 0.9,
                    "repository": 0.8,
                    "source code": 1.0,
                    "codebase": 0.9,
                    "source": 0.7,
                },
                "file_extensions": {".py", ".js", ".java", ".cpp", ".ts", ".go", ".rs", ".rb", ".php", ".cs"},
                "file_types": {"code"},
            },
            ContentType.TEXT: {
                "keywords": {
                    "csv": 1.0,
                    "csv file": 1.0,
                    "csv files": 1.0,
                    "csv data": 1.0,
                    "tsv": 1.0,
                    "data": 0.7,
                    "table": 0.8,
                    "spreadsheet": 0.9,
                    "tabular": 0.8,
                    "delimited": 0.7,
                    "comma-separated": 1.0,
                },
                "file_extensions": {".csv", ".tsv", ".txt", ".tab"},
                "file_types": {"text", "csv"},
            },
        }

    def _normalize_available_files(self, available_files: Optional[List[Any]]) -> Optional[List[Dict]]:
        """Normalize available_files from Document objects to dicts.
        
        Args:
            available_files: List of file dicts or Document objects
            
        Returns:
            List of file dicts with metadata extracted
        """
        if not available_files:
            return None
        
        normalized = []
        for item in available_files:
            if hasattr(item, 'metadata'):
                # LangChain Document object - extract metadata
                normalized.append(item.metadata)
            elif isinstance(item, dict):
                # Already a dictionary
                normalized.append(item)
        
        return normalized
    
    def detect(
        self, query: str, available_files: Optional[List[Any]] = None
    ) -> Dict[str, Any]:
        """
        Enhanced content type detection with confidence scoring and advanced filtering.

        Args:
            query: User's question
            available_files: Optional list of file dicts or Document objects with 'file_name' and optionally 'file_id'

        Returns:
            {
                'content_type': ContentType,
                'confidence': float (0-1),
                'metadata_filter': dict for ChromaDB filtering,
                'explanation': str,
                'filter_type': str,  # "file_type" | "file_id" | "multi_type" | "exclude" | "none"
                'filter_value': Union[str, List[str], Dict],
            }
        """
        query_lower = query.lower()
        
        # Normalize available_files from Document objects to dicts
        normalized_files = self._normalize_available_files(available_files)
        
        # 1. Check for exclusion patterns (e.g., "except README", "excluding", "not in")
        exclusion_result = self._detect_exclusion(query_lower, normalized_files)
        if exclusion_result:
            return self._detection_to_dict(exclusion_result)
        
        # 2. Check for explicit file references (highest priority, 100% confidence)
        explicit_file_result = self._detect_explicit_file(query_lower, normalized_files)
        if explicit_file_result:
            return self._detection_to_dict(explicit_file_result)
        
        # 3. Check for file extensions in query (100% confidence)
        extension_result = self._detect_file_extension(query_lower)
        if extension_result:
            return self._detection_to_dict(extension_result)
        
        # 4. Check for multi-type queries (e.g., "compare audio and PDF")
        multi_type_result = self._detect_multi_type(query_lower)
        if multi_type_result:
            return self._detection_to_dict(multi_type_result)
        
        # 5. Score based on keyword matches with confidence weighting
        scores: Dict[ContentType, float] = {ct: 0.0 for ct in ContentType}
        
        for content_type, patterns in self.patterns.items():
            max_score = 0.0
            for keyword, weight in patterns["keywords"].items():
                if keyword in query_lower:
                    # Use keyword weight directly, take maximum if multiple keywords match
                    max_score = max(max_score, weight)
            
            scores[content_type] = max_score
        
        # 6. Determine winner
        max_score = max(scores.values())
        
        if max_score == 0:
            # No specific type detected
            logger.debug("No specific content type detected")
            detection = ContentTypeDetection(
                filter_type="none",
                filter_value=None,
                content_type=ContentType.ANY,
                confidence=0.0,
                chromadb_where_clause=None,
                explanation="No specific content type detected - searching all files",
            )
            return self._detection_to_dict(detection)
        
        # Get content type with highest score
        detected_type = max(scores.items(), key=lambda x: x[1])[0]
        
        # Build metadata filter
        file_types = self.patterns[detected_type]["file_types"]
        metadata_filter = {"file_type": {"$in": list(file_types)}}
        
        logger.info(
            "Content type detected",
            content_type=detected_type.value,
            confidence=max_score,
            file_types=list(file_types),
        )
        
        detection = ContentTypeDetection(
            filter_type="file_type",
            filter_value=detected_type.value,
            content_type=detected_type,
            confidence=max_score,
            chromadb_where_clause=metadata_filter,
            explanation=f"Detected {detected_type.value} query (confidence: {max_score:.0%})",
        )
        return self._detection_to_dict(detection)
    
    def _detect_explicit_file(
        self, query_lower: str, available_files: Optional[List[Dict]]
    ) -> Optional[ContentTypeDetection]:
        """
        Detect explicit file references in query.
        
        Returns file_id filter with 100% confidence if file is found.
        Handles both exact filename matches and extension-based patterns (e.g., "csv file").
        
        Args:
            query_lower: Lowercase query
            available_files: List of available files with 'file_name' and optionally 'file_id'
            
        Returns:
            ContentTypeDetection with file_id filter if found, None otherwise
        """
        if not available_files:
            return None
        
        # First, check for extension-based patterns (e.g., "csv file", "the csv", "csv files")
        # This handles queries like "which colleges are in the csv file"
        extension_patterns = {
            "csv": [".csv", ".tsv"],
            "pdf": [".pdf"],
            "txt": [".txt"],
            "docx": [".docx", ".doc"],
            "png": [".png"],
            "jpg": [".jpg", ".jpeg"],
            "wav": [".wav"],
            "mp3": [".mp3"],
            "html": [".html", ".htm"],
        }
        
        for ext_name, extensions in extension_patterns.items():
            # Check for patterns like "csv file", "the csv", "csv files", "this csv"
            patterns = [f"{ext_name} file", f"{ext_name} files", f"the {ext_name}", f"this {ext_name}"]
            if any(pattern in query_lower for pattern in patterns):
                # Find all files with matching extensions
                matching_files = []
                for file_info in available_files:
                    file_name = file_info.get("file_name", "")
                    if not file_name:
                        continue
                    
                    # Check if file has matching extension
                    for ext in extensions:
                        if file_name.lower().endswith(ext):
                            matching_files.append(file_info)
                            break
                
                if matching_files:
                    # If only one matching file, use file_id filter (preferred)
                    if len(matching_files) == 1:
                        file_info = matching_files[0]
                        file_id = file_info.get("file_id")
                        file_name = file_info.get("file_name", "")
                        
                        if file_id:
                            where_clause = {"file_id": file_id}
                            filter_value = file_id
                            filter_type = "file_id"
                        else:
                            where_clause = {"file_name": file_name}
                            filter_value = file_name
                            filter_type = "file_name"
                        
                        # Determine content type from extension
                        content_type = ContentType.TEXT if ext_name == "csv" else ContentType.DOCUMENT
                        
                        logger.debug(
                            "Extension-based file reference detected (single file)",
                            file_name=file_name,
                            extension=ext_name,
                            content_type=content_type.value,
                        )
                        
                        return ContentTypeDetection(
                            filter_type=filter_type,
                            filter_value=filter_value,
                            content_type=content_type,
                            confidence=1.0,
                            chromadb_where_clause=where_clause,
                            explanation=f"Extension-based reference to {ext_name} file: '{file_name}'",
                        )
                    else:
                        # Multiple matching files - use file_name with $in operator
                        file_names = [f.get("file_name") for f in matching_files if f.get("file_name")]
                        if file_names:
                            content_type = ContentType.TEXT if ext_name == "csv" else ContentType.DOCUMENT
                            
                            logger.debug(
                                "Extension-based file reference detected (multiple files)",
                                extension=ext_name,
                                file_count=len(file_names),
                                content_type=content_type.value,
                            )
                            
                            return ContentTypeDetection(
                                filter_type="file_name",
                                filter_value=file_names,
                                content_type=content_type,
                                confidence=0.9,  # Slightly lower since multiple files match
                                chromadb_where_clause={"file_name": {"$in": file_names}},
                                explanation=f"Extension-based reference to {ext_name} files: {len(file_names)} files",
                            )
        
        # Fallback: Check for exact filename or base name matches
        for file_info in available_files:
            file_name = file_info.get("file_name", "")
            file_id = file_info.get("file_id")
            
            if not file_name:
                continue
            
            file_name_lower = file_name.lower()
            base_name = file_name_lower.rsplit('.', 1)[0] if '.' in file_name_lower else file_name_lower
            
            # Check if file name or base name (without extension) is in query
            if file_name_lower in query_lower or base_name in query_lower:
                # Determine content type from extension
                file_ext = "." + file_name.split(".")[-1] if "." in file_name else ""
                
                content_type = None
                for ct, patterns in self.patterns.items():
                    if file_ext in patterns["file_extensions"]:
                        content_type = ct
                        break
                
                if not content_type:
                    # Default to document if unknown extension
                    content_type = ContentType.DOCUMENT
                
                # Build filter - prefer file_id if available, otherwise file_name
                # Pass values directly (not wrapped in {"$eq": ...}) - ChromaDB handles simple equality directly
                if file_id:
                    where_clause = {"file_id": file_id}
                    filter_value = file_id
                else:
                    where_clause = {"file_name": file_name}
                    filter_value = file_name
                
                logger.debug(
                    "Explicit file reference detected",
                    file_name=file_name,
                    content_type=content_type.value,
                    file_id=file_id,
                )
                
                return ContentTypeDetection(
                    filter_type="file_id" if file_id else "file_name",
                    filter_value=filter_value,
                    content_type=content_type,
                    confidence=1.0,  # 100% - explicit reference
                    chromadb_where_clause=where_clause,
                    explanation=f"Explicit reference to file '{file_name}'",
                )
        
        return None
    
    def _detect_file_extension(self, query_lower: str) -> Optional[ContentTypeDetection]:
        """
        Detect file extensions in query (100% confidence).
        
        Args:
            query_lower: Lowercase query
            
        Returns:
            ContentTypeDetection if extension found, None otherwise
        """
        # Match file extensions (e.g., .wav, .pdf, .png)
        extension_pattern = r'\.([a-z0-9]{2,5})\b'
        matches = re.findall(extension_pattern, query_lower)
        
        if not matches:
            return None
        
        # Check each extension against known patterns
        for ext_match in matches:
            ext = "." + ext_match.lower()
            for content_type, patterns in self.patterns.items():
                if ext in patterns["file_extensions"]:
                    file_types = patterns["file_types"]
                    where_clause = {"file_type": {"$in": list(file_types)}}
                    
                    logger.debug(
                        "File extension detected",
                        extension=ext,
                        content_type=content_type.value,
                    )
                    
                    return ContentTypeDetection(
                        filter_type="file_type",
                        filter_value=content_type.value,
                        content_type=content_type,
                        confidence=1.0,  # 100% - explicit extension
                        chromadb_where_clause=where_clause,
                        explanation=f"File extension '{ext}' detected (100% confidence)",
                    )
        
        return None
    
    def _detect_multi_type(self, query_lower: str) -> Optional[ContentTypeDetection]:
        """
        Detect queries about multiple content types (e.g., "compare audio and PDF").
        
        Args:
            query_lower: Lowercase query
            
        Returns:
            ContentTypeDetection with multi_type filter if found, None otherwise
        """
        # Look for comparison/combination patterns
        comparison_keywords = ["and", "versus", "vs", "compare", "between"]
        has_comparison = any(kw in query_lower for kw in comparison_keywords)
        
        if not has_comparison:
            return None
        
        detected_types = []
        detected_type_names = []
        
        # Check for each content type
        for content_type, patterns in self.patterns.items():
            # Check keywords (high confidence only)
            for keyword, weight in patterns["keywords"].items():
                if weight >= 0.8 and keyword in query_lower:
                    detected_types.append(content_type)
                    detected_type_names.append(content_type.value)
                    break
            # Also check extensions
            for ext in patterns["file_extensions"]:
                if ext in query_lower:
                    if content_type not in detected_types:
                        detected_types.append(content_type)
                        detected_type_names.append(content_type.value)
                    break
        
        if len(detected_types) >= 2:
            # Multiple types detected - combine file_types
            all_file_types = set()
            for ct in detected_types:
                all_file_types.update(self.patterns[ct]["file_types"])
            
            where_clause = {"file_type": {"$in": list(all_file_types)}}
            
            logger.debug(
                "Multi-type query detected",
                types=detected_type_names,
            )
            
            return ContentTypeDetection(
                filter_type="multi_type",
                filter_value=detected_type_names,
                content_type=None,  # Multiple types
                content_types=detected_types,
                confidence=0.9,  # High confidence for explicit comparisons
                chromadb_where_clause=where_clause,
                explanation=f"Multi-type query detected: {', '.join(detected_type_names)}",
            )
        
        return None
    
    def _detect_exclusion(
        self, query_lower: str, available_files: Optional[List[Dict]]
    ) -> Optional[ContentTypeDetection]:
        """
        Detect exclusion patterns (e.g., "except README", "excluding X", "not in Y").
        
        Args:
            query_lower: Lowercase query
            available_files: List of available files
            
        Returns:
            ContentTypeDetection with exclude filter if found, None otherwise
        """
        exclusion_patterns = [
            r"except\s+(\w+(?:\.[\w]+)?)",  # "except README" or "except file.txt"
            r"excluding\s+(\w+(?:\.[\w]+)?)",
            r"not\s+in\s+(\w+(?:\.[\w]+)?)",
            r"exclude\s+(\w+(?:\.[\w]+)?)",
        ]
        
        excluded_files = []
        
        for pattern in exclusion_patterns:
            matches = re.finditer(pattern, query_lower)
            for match in matches:
                excluded_ref = match.group(1).strip()
                excluded_files.append(excluded_ref)
        
        if not excluded_files or not available_files:
            return None
        
        # Match excluded references to actual files
        matched_files = []
        for excluded_ref in excluded_files:
            excluded_lower = excluded_ref.lower()
            for file_info in available_files:
                file_name = file_info.get("file_name", "")
                file_name_lower = file_name.lower()
                base_name = file_name_lower.rsplit('.', 1)[0] if '.' in file_name_lower else file_name_lower
                
                if excluded_lower == file_name_lower or excluded_lower == base_name or excluded_lower in file_name_lower:
                    matched_files.append(file_name)
                    break
        
        if matched_files:
            # Build exclusion filter (ChromaDB uses $ne for "not equal")
            # For multiple files, we'd use $nin (not in)
            if len(matched_files) == 1:
                where_clause = {"file_name": {"$ne": matched_files[0]}}
            else:
                where_clause = {"file_name": {"$nin": matched_files}}
            
            logger.debug(
                "Exclusion pattern detected",
                excluded_files=matched_files,
            )
            
            return ContentTypeDetection(
                filter_type="exclude",
                filter_value={"file_names": matched_files},
                confidence=0.95,  # High confidence for explicit exclusion
                chromadb_where_clause=where_clause,
                explanation=f"Exclusion pattern detected - excluding: {', '.join(matched_files)}",
                excluded_files=matched_files,
            )
        
        return None
    
    def _detection_to_dict(self, detection: ContentTypeDetection) -> Dict[str, Any]:
        """Convert ContentTypeDetection to dict format for backward compatibility."""
        result = {
            "content_type": detection.content_type if detection.content_type else ContentType.ANY,
            "confidence": detection.confidence,
            "metadata_filter": detection.chromadb_where_clause,
            "explanation": detection.explanation,
            "filter_type": detection.filter_type,
            "filter_value": detection.filter_value,
        }
        
        if detection.content_types:
            result["content_types"] = [ct.value for ct in detection.content_types]
        
        if detection.excluded_files:
            result["excluded_files"] = detection.excluded_files
        
        return result
