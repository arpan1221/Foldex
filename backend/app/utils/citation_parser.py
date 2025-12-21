"""
Citation parser for inline citation markers in LLM responses.

Converts [cid:chunk_id] markers to HTML citation links.
"""

import re
from typing import Dict, List, Optional, Tuple, Any
import structlog

logger = structlog.get_logger(__name__)


def parse_inline_citations(
    text: str,
    chunk_map: Dict[str, Dict[str, Any]],
    citation_number_map: Optional[Dict[str, int]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Parse [cid:X] markers and replace with HTML citation links.
    
    Args:
        text: Response text with [cid:X] markers
        chunk_map: Dict mapping chunk_id -> metadata
        citation_number_map: Optional pre-computed citation number mapping
    
    Returns:
        Tuple of (parsed_text, used_citations_list)
    """
    if not text:
        return text, []
    
    # Pattern to match [cid:chunk_id]
    pattern = r'\[cid:([a-zA-Z0-9_-]+)\]'
    
    # Find all citation markers
    matches = list(re.finditer(pattern, text))
    
    if not matches:
        return text, []
    
    # Build citation number mapping if not provided
    if citation_number_map is None:
        citation_number_map = {}
        citation_counter = 1
        seen_chunk_ids = set()
        
        for match in matches:
            chunk_id = match.group(1)
            if chunk_id not in seen_chunk_ids:
                citation_number_map[chunk_id] = citation_counter
                citation_counter += 1
                seen_chunk_ids.add(chunk_id)
    
    # Collect used citations
    used_citations = []
    used_chunk_ids = set()
    
    def replace_citation(match):
        chunk_id = match.group(1)
        
        # Look up chunk metadata
        if chunk_id not in chunk_map:
            logger.warning("Citation chunk_id not found in chunk_map", chunk_id=chunk_id)
            return f'<sup class="citation-missing">[?]</sup>'
        
        meta = chunk_map[chunk_id]
        
        # Get citation number
        citation_num = citation_number_map.get(chunk_id, 1)
        
        # Track used citations
        if chunk_id not in used_chunk_ids:
            used_chunk_ids.add(chunk_id)
            # Ensure all required fields are present
            file_id = meta.get("file_id", "")
            file_name = meta.get("file_name", "Unknown")
            
            used_citations.append({
                "chunk_id": chunk_id,
                "citation_number": citation_num,
                "file_id": file_id,  # REQUIRED by frontend
                "file_name": file_name,
                "page_number": meta.get("page_number"),
                "page_display": f"p.{meta.get('page_number')}" if meta.get("page_number") else None,
                "google_drive_url": meta.get("url") or meta.get("drive_url") or meta.get("google_drive_url"),
                "mime_type": meta.get("mime_type"),  # Optional but helpful
                "section": meta.get("section", ""),
                "content_preview": meta.get("content_preview", "")[:200],
                "start_time": meta.get("start_time"),  # For audio/video
                "end_time": meta.get("end_time"),  # For audio/video
            })
        
        # Generate HTML link
        file_name = meta.get("file_name", "Unknown")
        page = meta.get("page_number", "")
        url = meta.get("url") or meta.get("drive_url") or meta.get("google_drive_url") or "#"
        
        # Build citation text
        citation_text = f"[{citation_num}]"  # Format: [1], [2], etc.
        
        citation_html = (
            f'<sup>'
            f'<a href="{url}" target="_blank" rel="noopener noreferrer" '
            f'class="citation-link" '
            f'title="{file_name}{f", p.{page}" if page else ""}">'
            f'{citation_text}'
            f'</a>'
            f'</sup>'
        )
        
        return citation_html
    
    # Replace all citations
    parsed = re.sub(pattern, replace_citation, text)
    
    # Sort citations by citation number
    used_citations.sort(key=lambda x: x["citation_number"])
    
    return parsed, used_citations


def extract_used_citations(text: str) -> List[str]:
    """Extract list of chunk_ids actually used in response.
    
    Args:
        text: Response text with [cid:X] markers
    
    Returns:
        List of unique chunk IDs
    """
    pattern = r'\[cid:([a-zA-Z0-9_-]+)\]'
    matches = re.findall(pattern, text)
    return list(set(matches))  # Deduplicate


def clean_citation_markers(text: str) -> str:
    """Remove citation markers from text (for plain text display).
    
    Args:
        text: Text with [cid:X] markers
    
    Returns:
        Text with markers removed
    """
    pattern = r'\[cid:([a-zA-Z0-9_-]+)\]'
    return re.sub(pattern, '', text)

