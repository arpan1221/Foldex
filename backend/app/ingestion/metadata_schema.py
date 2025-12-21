"""Standardized metadata schemas for different file types.

Provides consistent metadata structure across all processors to enable:
- Precise filtering by content type
- Accurate citation generation
- Clear distinction between actual content and mentions
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from enum import Enum
import structlog

logger = structlog.get_logger(__name__)


class FileType(str, Enum):
    """Supported file types for content identification."""

    PDF = "pdf"
    TEXT = "text"
    MARKDOWN = "markdown"
    CODE = "code"
    AUDIO = "audio"
    VIDEO = "video"
    IMAGE = "image"
    HTML = "html"
    DOCUMENT = "document"  # Generic office documents (DOCX, XLSX, PPTX)
    UNKNOWN = "unknown"


class ChunkType(str, Enum):
    """Types of content chunks for precise filtering."""

    # Document chunks
    DOCUMENT_SECTION = "document_section"
    DOCUMENT_SUMMARY = "document_summary"
    DOCUMENT_TABLE = "document_table"
    DOCUMENT_TITLE = "document_title"
    DOCUMENT_LIST_ITEM = "document_list_item"

    # Audio chunks
    AUDIO_TRANSCRIPTION = "audio_transcription"
    AUDIO_SUMMARY = "audio_summary"

    # Code chunks
    CODE_FUNCTION = "code_function"
    CODE_CLASS = "code_class"
    CODE_MODULE = "code_module"

    # Image chunks
    IMAGE_OCR_TEXT = "image_ocr_text"
    IMAGE_OCR_GROUPED = "image_ocr_grouped"  # Spatially grouped OCR elements
    IMAGE_SUMMARY = "image_summary"  # Overall image description

    # Generic
    GENERIC_TEXT = "generic_text"


class MetadataBuilder:
    """Build standardized metadata for chunks with type-specific fields."""

    SCHEMA_VERSION = "1.0"

    @staticmethod
    def base_metadata(
        file_id: str,
        file_name: str,
        file_type: FileType,
        chunk_type: ChunkType,
        chunk_id: str,
        drive_url: str = "",
    ) -> Dict[str, Any]:
        """
        Create base metadata present in ALL chunks.

        This is the MINIMUM metadata for any chunk. All processors must
        include these fields to enable consistent filtering and citation.

        Args:
            file_id: Unique file identifier
            file_name: Name of the source file
            file_type: Type of file (PDF, audio, text, etc.)
            chunk_type: Type of chunk (transcription, section, etc.)
            chunk_id: Unique chunk identifier
            drive_url: Google Drive URL (optional)

        Returns:
            Dictionary with base metadata fields
        """
        return {
            # === File Identification ===
            "file_id": file_id,
            "file_name": file_name,
            "file_type": file_type.value,
            "drive_url": drive_url,
            # === Chunk Identification ===
            "chunk_id": chunk_id,
            "chunk_type": chunk_type.value,
            # === Ingestion Info ===
            "ingestion_date": datetime.utcnow().isoformat(),
            "schema_version": MetadataBuilder.SCHEMA_VERSION,
        }

    @staticmethod
    def pdf_metadata(
        base: Dict[str, Any],
        page_number: int,
        total_pages: int,
        section: str = "",
        authors: Optional[List[str]] = None,
        title: str = "",
    ) -> Dict[str, Any]:
        """
        PDF-specific metadata.

        Enables queries like:
        - "What's on page 5 of the paper?"
        - "Find the methodology section"

        Args:
            base: Base metadata dictionary
            page_number: Page number (1-indexed)
            total_pages: Total number of pages
            section: Section name or heading
            authors: List of author names
            title: Document title

        Returns:
            Dictionary with PDF-specific metadata added
        """
        return {
            **base,
            # === PDF-Specific ===
            "page_number": page_number,
            "total_pages": total_pages,
            "section": section,
            "authors": authors or [],
            "document_title": title,
            # === Search Hints ===
            "searchable_location": f"page {page_number}",
            "citation_format": f"{base['file_name']}, p.{page_number}",
        }

    @staticmethod
    def audio_metadata(
        base: Dict[str, Any],
        start_time: float,
        end_time: float,
        segment_index: int,
        total_segments: int,
        speaker: Optional[str] = None,
        confidence: Optional[float] = None,
        language: str = "en",
    ) -> Dict[str, Any]:
        """
        Audio-specific metadata.

        Enables queries like:
        - "What was said at 5:30?"
        - "Summarize the first 10 minutes"

        Args:
            base: Base metadata dictionary
            start_time: Start time in seconds
            end_time: End time in seconds
            segment_index: Segment index (0-indexed)
            total_segments: Total number of segments
            speaker: Speaker identifier (if available)
            confidence: Transcription confidence score
            language: Detected language code

        Returns:
            Dictionary with audio-specific metadata added
        """
        duration = end_time - start_time
        return {
            **base,
            # === Audio-Specific ===
            "start_time": start_time,
            "end_time": end_time,
            "duration": duration,
            "segment_index": segment_index,
            "total_segments": total_segments,
            "speaker": speaker,
            "confidence": confidence,
            "language": language,
            # === Display Formats ===
            "timestamp_display": MetadataBuilder._format_time(start_time),
            "time_range": f"{MetadataBuilder._format_time(start_time)} - {MetadataBuilder._format_time(end_time)}",
            # === Search Hints ===
            "searchable_location": f"at {MetadataBuilder._format_time(start_time)}",
            "citation_format": f"{base['file_name']} [{MetadataBuilder._format_time(start_time)}]",
        }

    @staticmethod
    def text_metadata(
        base: Dict[str, Any],
        line_start: int,
        line_end: int,
        section: str = "",
    ) -> Dict[str, Any]:
        """
        Text/Markdown file metadata.

        Args:
            base: Base metadata dictionary
            line_start: Starting line number
            line_end: Ending line number
            section: Section name or heading

        Returns:
            Dictionary with text-specific metadata added
        """
        return {
            **base,
            # === Text-Specific ===
            "line_start": line_start,
            "line_end": line_end,
            "section": section,
            # === Search Hints ===
            "searchable_location": f"lines {line_start}-{line_end}",
            "citation_format": f"{base['file_name']}, lines {line_start}-{line_end}",
        }

    @staticmethod
    def code_metadata(
        base: Dict[str, Any],
        language: str,
        function_name: str = "",
        class_name: str = "",
        line_start: int = 0,
        line_end: int = 0,
    ) -> Dict[str, Any]:
        """
        Code file metadata.

        Args:
            base: Base metadata dictionary
            language: Programming language
            function_name: Function name (if applicable)
            class_name: Class name (if applicable)
            line_start: Starting line number
            line_end: Ending line number

        Returns:
            Dictionary with code-specific metadata added
        """
        location = f"{class_name}.{function_name}" if class_name and function_name else function_name or class_name
        return {
            **base,
            # === Code-Specific ===
            "language": language,
            "function_name": function_name,
            "class_name": class_name,
            "line_start": line_start,
            "line_end": line_end,
            # === Search Hints ===
            "searchable_location": location or f"lines {line_start}-{line_end}",
            "citation_format": f"{base['file_name']}:{line_start}",
        }

    @staticmethod
    def image_metadata(
        base: Dict[str, Any],
        page_number: Optional[int] = None,
        coordinates: Optional[Dict[str, Any]] = None,
        is_summary: bool = False,
        grouped_elements_count: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Image-specific metadata (OCR results).

        Args:
            base: Base metadata dictionary
            page_number: Page number if applicable (for multi-page images)
            coordinates: OCR bounding box coordinates
            is_summary: Whether this is a summary chunk describing the image
            grouped_elements_count: Number of OCR elements grouped together

        Returns:
            Dictionary with image-specific metadata added
        """
        metadata = {**base}
        if page_number is not None:
            metadata["page_number"] = page_number
        if coordinates:
            # Store coordinates as string representation for compatibility
            metadata["ocr_coordinates"] = str(coordinates)
        if is_summary:
            metadata["is_image_summary"] = True
        if grouped_elements_count is not None:
            metadata["grouped_elements_count"] = grouped_elements_count
        
        # Search hints
        if is_summary:
            metadata["searchable_location"] = "image summary"
            metadata["citation_format"] = f"{base['file_name']} (image summary)"
        elif coordinates:
            metadata["searchable_location"] = "image content"
            metadata["citation_format"] = f"{base['file_name']} (image)"
        else:
            metadata["searchable_location"] = "image content"
            metadata["citation_format"] = f"{base['file_name']}"
        
        return metadata

    @staticmethod
    def document_metadata(
        base: Dict[str, Any],
        page_number: Optional[int] = None,
        section: str = "",
        element_type: str = "",
    ) -> Dict[str, Any]:
        """
        Generic document metadata (Office docs, HTML, etc.).

        Args:
            base: Base metadata dictionary
            page_number: Page number if applicable
            section: Section name or heading
            element_type: Type of document element (Title, NarrativeText, etc.)

        Returns:
            Dictionary with document-specific metadata added
        """
        metadata = {**base}
        if page_number is not None:
            metadata["page_number"] = page_number
        if section:
            metadata["section"] = section
        if element_type:
            metadata["element_type"] = element_type  # Preserve for backward compatibility
        
        # Search hints
        location_parts = []
        if page_number:
            location_parts.append(f"page {page_number}")
        if section:
            location_parts.append(f"section: {section}")
        metadata["searchable_location"] = ", ".join(location_parts) if location_parts else "document content"
        metadata["citation_format"] = f"{base['file_name']}" + (f", p.{page_number}" if page_number else "")
        
        return metadata

    @staticmethod
    def _format_time(seconds: float) -> str:
        """
        Format seconds as MM:SS or HH:MM:SS.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)

        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    @staticmethod
    def merge_metadata(
        base: Dict[str, Any], additional: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Merge additional metadata into base metadata.

        Useful for preserving existing metadata while adding standardized fields.

        Args:
            base: Base metadata dictionary
            additional: Additional metadata to merge (optional)

        Returns:
            Merged metadata dictionary
        """
        if not additional:
            return base

        # Merge additional metadata, but don't overwrite base fields
        merged = {**base}
        for key, value in additional.items():
            # Skip None values
            if value is None:
                continue
            # Convert lists to strings for compatibility
            if isinstance(value, list):
                merged[key] = ", ".join(str(v) for v in value) if value else None
            # Preserve other types
            elif isinstance(value, (str, int, float, bool)):
                merged[key] = value
            else:
                merged[key] = str(value)

        return merged

