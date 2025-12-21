"""Inline citation system that tracks which chunks were actually used in the response."""

import re
from typing import List, Dict, Any, Optional, Tuple
import structlog

try:
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.schema import Document
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        Document = None

logger = structlog.get_logger(__name__)


class InlineCitationExtractor:
    """
    Extracts and manages inline citations from LLM responses.
    
    Uses LLM to identify which chunks were actually referenced,
    then formats citations inline like: "Stock trading uses MDPs [1]"
    """

    def __init__(self):
        """Initialize citation extractor."""
        self.citation_pattern = re.compile(r'\[(\d+)\]')

    def create_citation_prompt_suffix(self, source_documents: List[Document]) -> str:
        """
        Create a prompt suffix that instructs the LLM to cite sources.
        
        Args:
            source_documents: List of retrieved documents
            
        Returns:
            Prompt suffix with numbered sources
        """
        if not source_documents:
            return ""
        
        # Number each source document
        sources_text = "\n\nAVAILABLE SOURCES (cite these using [1], [2], etc.):\n"
        for idx, doc in enumerate(source_documents, 1):
            metadata = doc.metadata if hasattr(doc, "metadata") else {}
            file_name = metadata.get("file_name", "Unknown")
            page_num = metadata.get("page_number")
            chunk_idx = metadata.get("chunk_index", "")
            
            # Create source reference
            source_ref = f"[{idx}] {file_name}"
            if page_num:
                source_ref += f", p.{page_num}"
            if chunk_idx:
                source_ref += f", chunk {chunk_idx}"
            
            # Add preview of content
            content_preview = doc.page_content[:150] if hasattr(doc, "page_content") else ""
            if len(content_preview) > 0:
                sources_text += f"{source_ref}\n  Preview: {content_preview}...\n"
        
        return sources_text

    def create_citation_instruction(self) -> str:
        """
        Create instructions for the LLM to cite sources inline.
        
        Returns:
            Citation instructions
        """
        return """
CITATION REQUIREMENTS:
- Add inline citations like [1], [2] immediately after claims from sources
- Only cite sources you actually used from the AVAILABLE SOURCES list
- Multiple citations for one claim: [1][2]
- Do NOT invent citation numbers - only use the provided source numbers
- If you don't use a source, don't cite it
- Example: "The system uses optimization techniques [1] to improve performance [2]."
"""

    def extract_used_citations(
        self,
        response: str,
        source_documents: List[Document]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Extract which sources were actually cited in the response.
        
        Args:
            response: LLM response text with inline citations
            source_documents: Original list of source documents
            
        Returns:
            Tuple of (response text, list of used citations)
        """
        # Find all citation numbers in the response
        cited_numbers = set()
        for match in self.citation_pattern.finditer(response):
            try:
                num = int(match.group(1))
                if 1 <= num <= len(source_documents):
                    cited_numbers.add(num)
            except ValueError:
                continue
        
        # Build citation list for only the cited sources
        used_citations = []
        for num in sorted(cited_numbers):
            doc_idx = num - 1  # Convert to 0-based index
            if doc_idx < len(source_documents):
                doc = source_documents[doc_idx]
                metadata = doc.metadata if hasattr(doc, "metadata") else {}
                
                # Build Google Drive URL
                google_drive_url = self._get_drive_url(metadata)
                
                # Extract page number for display
                page_number = metadata.get("page_number")
                page_display = f"p.{page_number}" if page_number else None
                
                citation = {
                    "citation_number": num,
                    "file_id": metadata.get("file_id") or "",  # Ensure file_id is always present
                    "file_name": metadata.get("file_name", "Unknown"),
                    "chunk_id": metadata.get("chunk_id"),
                    "page_number": page_number,
                    "page_display": page_display,  # Add page_display for consistency
                    "chunk_index": metadata.get("chunk_index"),
                    "start_time": metadata.get("start_time"),
                    "end_time": metadata.get("end_time"),
                    "content_preview": (
                        doc.page_content[:200]
                        if hasattr(doc, "page_content")
                        else ""
                    ),
                    "source": metadata.get("source"),
                    "file_path": metadata.get("file_path"),
                    "mime_type": metadata.get("mime_type"),
                    "google_drive_url": google_drive_url,  # Always include URL (can be None)
                    "metadata": metadata,
                }
                used_citations.append(citation)
        
        logger.info(
            "Extracted inline citations",
            total_sources=len(source_documents),
            cited_sources=len(used_citations),
            citation_numbers=sorted(cited_numbers),
        )
        
        return response, used_citations

    def _get_drive_url(self, metadata: Dict[str, Any]) -> Optional[str]:
        """
        Extract or construct Google Drive URL from metadata.
        
        Args:
            metadata: Document metadata
            
        Returns:
            Google Drive URL if available
        """
        # Check if URL is already in metadata
        if "google_drive_url" in metadata:
            return metadata["google_drive_url"]
        
        # Construct URL from file_id
        file_id = metadata.get("file_id")
        if file_id:
            return f"https://drive.google.com/file/d/{file_id}/view"
        
        return None

    def deduplicate_citations_by_file(
        self,
        citations: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Deduplicate citations from the same source file.
        
        Keeps the first citation from each file and aggregates page numbers.
        
        Args:
            citations: List of citation dictionaries
            
        Returns:
            Deduplicated list of citations
        """
        file_citations: Dict[str, Dict[str, Any]] = {}
        
        for citation in citations:
            file_id = citation.get("file_id")
            if not file_id:
                continue
            
            if file_id not in file_citations:
                # First citation from this file
                file_citations[file_id] = citation.copy()
                file_citations[file_id]["page_numbers"] = []
                file_citations[file_id]["citation_numbers"] = []
            
            # Add page number if available
            page_num = citation.get("page_number")
            if page_num and page_num not in file_citations[file_id]["page_numbers"]:
                file_citations[file_id]["page_numbers"].append(page_num)
            
            # Track citation numbers
            cite_num = citation.get("citation_number")
            if cite_num:
                file_citations[file_id]["citation_numbers"].append(cite_num)
        
        # Format page numbers for display
        deduplicated = []
        for citation in file_citations.values():
            if citation["page_numbers"]:
                citation["page_numbers"].sort()
                citation["page_display"] = ", ".join(f"p.{p}" for p in citation["page_numbers"])
            else:
                citation["page_display"] = None
            
            deduplicated.append(citation)
        
        logger.debug(
            "Deduplicated citations",
            original_count=len(citations),
            deduplicated_count=len(deduplicated),
        )
        
        return deduplicated

    def format_citation_for_display(
        self,
        citation: Dict[str, Any],
        include_url: bool = True
    ) -> str:
        """
        Format a citation for display.
        
        Args:
            citation: Citation dictionary
            include_url: Whether to include Google Drive URL
            
        Returns:
            Formatted citation string
        """
        file_name = citation.get("file_name", "Unknown")
        page_display = citation.get("page_display")
        
        # Basic format: "filename.pdf"
        formatted = file_name
        
        # Add page numbers if available
        if page_display:
            formatted += f", {page_display}"
        
        # Add URL if requested
        if include_url:
            drive_url = citation.get("google_drive_url")
            if drive_url:
                formatted += f" ({drive_url})"
        
        return formatted


def create_citation_aware_prompt(
    base_prompt: str,
    source_documents: List[Document],
    extractor: Optional[InlineCitationExtractor] = None
) -> str:
    """
    Enhance a prompt with citation instructions and numbered sources.
    
    Args:
        base_prompt: Original prompt template
        source_documents: Retrieved source documents
        extractor: Citation extractor instance
        
    Returns:
        Enhanced prompt with citation instructions
    """
    if not extractor:
        extractor = InlineCitationExtractor()
    
    # Add citation instructions
    citation_instruction = extractor.create_citation_instruction()
    
    # Add numbered sources
    sources_text = extractor.create_citation_prompt_suffix(source_documents)
    
    # Combine
    enhanced_prompt = base_prompt + citation_instruction + sources_text
    
    return enhanced_prompt


# Global instance
_citation_extractor: Optional[InlineCitationExtractor] = None


def get_citation_extractor() -> InlineCitationExtractor:
    """Get the global citation extractor instance."""
    global _citation_extractor
    if _citation_extractor is None:
        _citation_extractor = InlineCitationExtractor()
    return _citation_extractor

