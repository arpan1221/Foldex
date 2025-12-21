"""
Metadata extraction for documents.

Extracts titles, authors, sections, and entities from PDFs and text files.
Uses spaCy NER for robust entity extraction with defined scope.
"""

from typing import Dict, Any, List, Optional, Set
import re
import structlog

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    fitz = None

# spaCy for Named Entity Recognition
try:
    import spacy
    from spacy.language import Language
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None
    Language = None

logger = structlog.get_logger(__name__)


class MetadataExtractor:
    """
    Extract metadata and entities from documents (PDFs, text files).
    
    Uses spaCy NER for robust entity extraction with defined scope:
    - PERSON: Authors, contributors, creators
    - ORG: Organizations, institutions, companies  
    - GPE: Geographic locations (cities, countries)
    - DATE: Publication dates, time references
    - PRODUCT: Software, tools, systems mentioned
    - WORK_OF_ART: Referenced papers, books, articles
    """

    def __init__(self, use_ner: bool = True):
        """
        Initialize metadata extractor.
        
        Args:
            use_ner: Whether to use spaCy NER for entity extraction (default: True)
        """
        self.use_ner = use_ner and SPACY_AVAILABLE
        self.nlp: Optional[Language] = None
        
        # Initialize spaCy model
        if self.use_ner:
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logger.info("Loaded spaCy model for NER", model="en_core_web_sm")
            except OSError:
                logger.warning(
                    "spaCy model not found",
                    install_command="python -m spacy download en_core_web_sm"
                )
                self.use_ner = False
        
        # Define entity extraction scope
        self.entity_scope = {
            "PERSON": "authors",          # Authors, contributors, creators
            "ORG": "organizations",       # Organizations, institutions
            "GPE": "locations",           # Geographic locations
            "DATE": "dates",              # Publication dates, time references
            "PRODUCT": "products",        # Software, tools, systems
            "WORK_OF_ART": "references",  # Referenced works
        }

    def extract_pdf_metadata(
        self,
        pdf_doc: Any,  # fitz.Document
        file_path: str,
    ) -> Dict[str, Any]:
        """
        Extract metadata from PDF document.

        Args:
            pdf_doc: PyMuPDF document object
            file_path: Path to PDF file

        Returns:
            Dictionary with extracted metadata
        """
        metadata: Dict[str, Any] = {
            "document_title": None,
            "authors": [],
            "creation_date": None,
            "subject": None,
        }

        try:
            # Get PDF metadata
            pdf_metadata = pdf_doc.metadata

            if pdf_metadata:
                # Extract title
                if pdf_metadata.get("title"):
                    metadata["document_title"] = pdf_metadata["title"].strip()

                # Extract author
                if pdf_metadata.get("author"):
                    authors_str = pdf_metadata["author"].strip()
                    # Split by common separators
                    authors = self._parse_authors(authors_str)
                    metadata["authors"] = authors

                # Extract creation date
                if pdf_metadata.get("creationDate"):
                    metadata["creation_date"] = pdf_metadata["creationDate"]

                # Extract subject
                if pdf_metadata.get("subject"):
                    metadata["subject"] = pdf_metadata["subject"].strip()

            # If no title in metadata, try to extract from first page
            if not metadata.get("document_title") and len(pdf_doc) > 0:
                first_page = pdf_doc[0]
                first_page_text = first_page.get_text()

                if first_page_text:
                    title = self._extract_title_from_text(first_page_text)
                    if title:
                        metadata["document_title"] = title

            # Always try to extract authors from first page if not in metadata
            # (moved outside title check so it runs even if title exists in PDF metadata)
            if len(pdf_doc) > 0:
                first_page = pdf_doc[0]
                first_page_text = first_page.get_text()
                
                if first_page_text:
                    authors_list = metadata.get("authors")
                    if not authors_list or (isinstance(authors_list, list) and len(authors_list) == 0):
                        authors = self._extract_authors_from_text(first_page_text)
                        if authors:
                            metadata["authors"] = authors

            # Fallback: use filename as title
            if not metadata["document_title"]:
                import os
                filename = os.path.basename(file_path)
                # Remove extension and clean up
                title = os.path.splitext(filename)[0]
                title = title.replace("_", " ").replace("-", " ")
                metadata["document_title"] = title

            logger.debug(
                "Extracted PDF metadata",
                title=metadata["document_title"],
                authors=metadata["authors"],
            )

        except Exception as e:
            logger.warning(
                "Failed to extract PDF metadata",
                file_path=file_path,
                error=str(e),
            )

        return metadata

    def extract_text_metadata(
        self,
        content: str,
        file_path: str,
        is_markdown: bool = False,
    ) -> Dict[str, Any]:
        """
        Extract metadata from text content.

        Args:
            content: Text content
            file_path: Path to file
            is_markdown: Whether content is Markdown

        Returns:
            Dictionary with extracted metadata
        """
        metadata: Dict[str, Any] = {
            "document_title": None,
        }

        try:
            # For Markdown, extract title from first heading
            if is_markdown:
                title = self._extract_markdown_title(content)
                if title:
                    metadata["document_title"] = title

            # If no title found, try to extract from first line
            if not metadata.get("document_title"):
                lines = content.strip().split("\n")
                if lines:
                    first_line = lines[0].strip()
                    # Use first line if it looks like a title (short, no period at end)
                    if len(first_line) < 100 and not first_line.endswith("."):
                        metadata["document_title"] = first_line

            # Fallback: use filename
            if not metadata.get("document_title"):
                import os
                filename = os.path.basename(file_path)
                title = os.path.splitext(filename)[0]
                title = title.replace("_", " ").replace("-", " ")
                metadata["document_title"] = title

        except Exception as e:
            logger.warning(
                "Failed to extract text metadata",
                file_path=file_path,
                error=str(e),
            )

        return metadata

    def detect_section(self, text: str, page_number: int) -> Optional[str]:
        """
        Detect section name from text content using generic patterns.
        
        Works for any document type - not limited to academic papers.

        Args:
            text: Text content
            page_number: Page number

        Returns:
            Section name if detected, None otherwise
        """
        try:
            # Check first few lines for section headers
            lines = text.strip().split("\n")[:10]

            for line in lines:
                line_clean = line.strip()
                
                if not line_clean or len(line_clean) < 3:
                    continue

                # Pattern 1: ALL CAPS headings (common in many document types)
                # e.g., "EXECUTIVE SUMMARY", "CHAPTER 1", "OVERVIEW"
                if line_clean.isupper() and len(line_clean.split()) <= 6:
                    return line_clean.title()

                # Pattern 2: Numbered sections (works for any topic)
                # e.g., "1. Introduction", "2.1 Market Analysis", "Chapter 3: Results"
                numbered_patterns = [
                    r"^\s*(\d+\.?\d*)\s+(.+)$",  # "1. Title" or "1.1 Title"
                    r"^\s*(?:Chapter|Section|Part)\s+(\d+)[:\-\s]+(.+)$",  # "Chapter 1: Title"
                    r"^\s*([IVXLCDM]+)\.\s+(.+)$",  # Roman numerals: "I. Title"
                ]
                
                for pattern in numbered_patterns:
                    match = re.match(pattern, line_clean, re.IGNORECASE)
                    if match:
                        # Return the title part (last group)
                        title = match.groups()[-1].strip()
                        if title and len(title) < 100:  # Reasonable title length
                            return title

                # Pattern 3: Lines ending with colon (often section headers)
                # e.g., "Key Findings:", "Summary:", "Next Steps:"
                if line_clean.endswith(":") and len(line_clean.split()) <= 6:
                    return line_clean[:-1].strip()

                # Pattern 4: Short lines with title case (likely headers)
                # e.g., "Market Overview", "Technical Specifications"
                words = line_clean.split()
                if (2 <= len(words) <= 6 and 
                    sum(1 for w in words if w and w[0].isupper()) >= len(words) * 0.7 and
                    not line_clean.endswith((".", "!", "?"))):  # Not a sentence
                    return line_clean

                # Pattern 5: Markdown-style headers
                # e.g., "# Title", "## Subtitle"
                if line_clean.startswith("#"):
                    header_text = re.sub(r"^#+\s*", "", line_clean).strip()
                    if header_text:
                        return header_text

            # No section detected - return None (don't assume anything)
            return None

        except Exception as e:
            logger.debug("Section detection failed", error=str(e))
            return None

    def _extract_title_from_text(self, text: str) -> Optional[str]:
        """Extract title from first page text using generic heuristics.
        
        Works for any document type - business reports, technical docs, papers, etc.

        Args:
            text: First page text

        Returns:
            Extracted title or None
        """
        try:
            lines = [line.strip() for line in text.split("\n") if line.strip()]

            if not lines:
                return None

            # Strategy 1: Look for centered text (often indicates title)
            # Check if line has significant leading/trailing whitespace
            raw_lines = text.split("\n")[:15]
            for i, raw_line in enumerate(raw_lines):
                if not raw_line.strip():
                    continue
                
                leading_spaces = len(raw_line) - len(raw_line.lstrip())
                line_clean = raw_line.strip()
                
                # Centered text (lots of leading spaces) that's substantial
                if leading_spaces > 10 and 15 < len(line_clean) < 150:
                    # Skip if it looks like metadata
                    if not any(keyword in line_clean.lower() for keyword in 
                              ["page", "doi:", "arxiv:", "http", "@", "email"]):
                        return line_clean

            # Strategy 2: Find the longest line in first 10 lines (often the title)
            candidates = []
            for i, line in enumerate(lines[:10]):
                # Skip very short lines
                if len(line) < 15:
                    continue

                # Skip lines with common non-title patterns
                skip_patterns = [
                    "page", "doi:", "arxiv:", "http", "www.", "@",
                    "email", "tel:", "fax:", "©", "copyright",
                    "all rights reserved", "confidential"
                ]
                if any(pattern in line.lower() for pattern in skip_patterns):
                    continue

                # Skip lines that are clearly sentences (end with period)
                if line.endswith(".") and len(line) > 50:
                    continue

                candidates.append((i, line, len(line)))

            if candidates:
                # Sort by length (longest first)
                candidates.sort(key=lambda x: x[2], reverse=True)
                
                # Get the longest line
                _, title, _ = candidates[0]
                
                # Verify it's not too long (titles are usually < 150 chars)
                if len(title) < 150:
                    return title

            # Strategy 3: First line that looks like a title
            # (title case, reasonable length, no sentence-ending punctuation)
            for line in lines[:8]:
                if 15 < len(line) < 150:
                    words = line.split()
                    # Check if majority of words are capitalized (title case)
                    capitalized = sum(1 for w in words if w and w[0].isupper())
                    if capitalized >= len(words) * 0.6:
                        # Not a sentence (doesn't end with period, question mark, etc.)
                        if not line.endswith((".", "!", "?")):
                            return line

            # Fallback: return first substantial line that's not too long
            for line in lines[:5]:
                if 20 < len(line) < 150:
                    return line

            return None

        except Exception as e:
            logger.debug("Title extraction failed", error=str(e))
            return None

    def extract_entities(
        self,
        text: str,
        entity_types: Optional[List[str]] = None,
        max_length: int = 3000
    ) -> Dict[str, List[str]]:
        """
        Extract named entities from text using spaCy NER.
        
        Args:
            text: Text to extract entities from
            entity_types: List of entity types to extract (default: all in scope)
            max_length: Maximum text length to process (default: 3000 chars)
        
        Returns:
            Dictionary mapping entity types to lists of extracted entities
            Example: {"PERSON": ["John Doe", "Jane Smith"], "ORG": ["MIT", "Google"]}
        """
        if not self.use_ner or not self.nlp:
            logger.debug("NER not available, skipping entity extraction")
            return {}
        
        # Default to all entity types in scope
        if entity_types is None:
            entity_types = list(self.entity_scope.keys())
        
        try:
            # Process text (limit length for performance)
            doc = self.nlp(text[:max_length])
            
            # Extract entities by type
            entities: Dict[str, Set[str]] = {etype: set() for etype in entity_types}
            
            for ent in doc.ents:
                if ent.label_ in entity_types:
                    # Clean and normalize entity text
                    entity_text = ent.text.strip()
                    if entity_text and len(entity_text) > 1:
                        entities[ent.label_].add(entity_text)
            
            # Convert sets to sorted lists
            result = {
                etype: sorted(list(entities[etype]))
                for etype in entity_types
                if entities[etype]
            }
            
            logger.debug(
                "Extracted entities",
                entity_counts={k: len(v) for k, v in result.items()}
            )
            
            return result
            
        except Exception as e:
            logger.warning("Entity extraction failed", error=str(e))
            return {}
    
    def _extract_authors_from_text(self, text: str) -> List[str]:
        """Extract authors/creators from first page text.
        
        Uses spaCy NER (PERSON entities) with fallback to pattern matching.
        Generic approach - works for papers, reports, documents, etc.

        Args:
            text: First page text

        Returns:
            List of author names
        """
        try:
            # Strategy 1: Use spaCy NER to extract PERSON entities
            if self.use_ner and self.nlp:
                authors = self._extract_authors_with_ner(text)
                if authors:
                    logger.debug(f"Extracted {len(authors)} authors using NER")
                    return authors
                logger.debug("No authors found with NER, falling back to patterns")
            
            # Strategy 2: Fallback to pattern matching (original logic)
            lines = [line.strip() for line in text.split("\n") if line.strip()]

            # Strategy 1: Look for explicit author labels
            author_labels = [
                r"^(?:by|author|authors|written by|prepared by|created by|contributor)[:\s]+(.+)$",
                r"^(.+?)\s*(?:\(author\)|\(authors\))$",
            ]
            
            for line in lines[:20]:
                for pattern in author_labels:
                    match = re.match(pattern, line, re.IGNORECASE)
                    if match:
                        author_text = match.group(1).strip()
                        if author_text:
                            return self._parse_authors(author_text)

            # Strategy 2: Look for name-like patterns with separators
            # (works for papers, business docs, reports)
            for i, line in enumerate(lines[:15]):
                # Skip very long lines (likely not author names)
                if len(line) > 120:
                    continue

                # Look for patterns like "John Doe, Jane Smith"
                # or "John Doe and Jane Smith"
                if any(sep in line for sep in [",", " and ", " & "]):
                    # Check if line contains name-like patterns
                    if self._looks_like_author_line(line):
                        authors = self._parse_authors(line)
                        if authors:
                            return authors

            # Strategy 3: Look for individual names on separate lines
            # (common in academic papers where each author has affiliation below)
            potential_authors = []
            title_words: Set[str] = set()  # Track words from the title to avoid false positives
            
            # First, try to identify the title (usually first 1-3 lines)
            for line in lines[:3]:
                if len(line) > 15:  # Substantial line
                    title_words.update(word.lower() for word in line.split())
            
            for i, line in enumerate(lines[:30]):
                # Skip very long or very short lines
                if len(line) < 5 or len(line) > 80:
                    continue
                
                # Skip lines with common non-author patterns
                skip_patterns = [
                    "http", "www.", "@", ".com", ".org", ".edu",
                    "email", "tel:", "phone:", "fax:", "abstract",
                    "department", "institute", "university", "college",
                    "india", "usa", "uk", "china", "japan", "germany",
                    "mumbai", "delhi", "london", "new york", "beijing",
                    "based", "using", "system", "control", "learning"  # Common title words
                ]
                if any(pattern in line.lower() for pattern in skip_patterns):
                    continue
                
                # Skip if line appears to be part of the title
                # (shares significant words with title)
                line_words = set(word.lower() for word in line.split())
                if title_words and len(line_words & title_words) >= len(line_words) * 0.5:
                    continue
                
                # Check if line looks like a name (2-4 words, mostly capitalized)
                words = line.split()
                if 2 <= len(words) <= 4:
                    # Count capitalized words
                    capitalized = sum(1 for w in words if w and w[0].isupper())
                    if capitalized >= len(words) * 0.8:  # At least 80% capitalized
                        # Verify it's not a title or section header
                        if not line.isupper() and not line.endswith(":"):
                            potential_authors.append(line)
            
            # Return if we found 2-10 potential authors (reasonable range)
            if 2 <= len(potential_authors) <= 10:
                return potential_authors[:10]  # Cap at 10 authors

            return []

        except Exception as e:
            logger.debug("Author extraction failed", error=str(e))
            return []
    
    def _extract_authors_with_ner(self, text: str) -> List[str]:
        """
        Extract authors using spaCy NER (PERSON entities).
        
        Filters PERSON entities to identify likely authors by:
        - Position in document (first 30% of text)
        - Proximity to title
        - Exclusion of common non-author patterns
        
        Args:
            text: First page text
        
        Returns:
            List of author names
        """
        if not self.nlp:
            return []
        
        try:
            # Process first portion of text (where authors typically appear)
            author_section_text = text[:2000]  # First 2000 chars
            doc = self.nlp(author_section_text)
            
            # Extract PERSON entities
            person_entities = [
                ent.text.strip()
                for ent in doc.ents
                if ent.label_ == "PERSON"
            ]
            
            if not person_entities:
                return []
            
            # Filter out false positives
            filtered_authors = []
            for person in person_entities:
                # Skip very short names (likely incomplete)
                if len(person) < 4:
                    continue
                
                # Skip names that appear to be part of citations or references
                # (e.g., "Smith et al.", "Jones (2020)")
                if any(pattern in person.lower() for pattern in ["et al", "fig.", "table"]):
                    continue
                
                # Skip if it's part of a URL or email
                person_lower = person.lower()
                if any(pattern in person_lower for pattern in ["http", "www.", "@", ".com"]):
                    continue
                
                # Skip common title words that might be misidentified
                if person_lower in ["abstract", "introduction", "conclusion", "references"]:
                    continue
                
                filtered_authors.append(person)
            
            # Deduplicate while preserving order
            seen = set()
            unique_authors = []
            for author in filtered_authors:
                if author not in seen:
                    seen.add(author)
                    unique_authors.append(author)
            
            # Return reasonable number of authors (2-10 is typical)
            if 2 <= len(unique_authors) <= 10:
                return unique_authors[:10]
            
            # If we have 1 author or more than 10, be more selective
            # Look for authors in first 1000 chars only
            if len(unique_authors) > 10:
                early_doc = self.nlp(text[:1000])
                early_persons = [
                    ent.text.strip()
                    for ent in early_doc.ents
                    if ent.label_ == "PERSON" and len(ent.text.strip()) >= 4
                ]
                if 2 <= len(early_persons) <= 10:
                    return early_persons[:10]
            
            # Return what we have, capped at 10
            return unique_authors[:10] if unique_authors else []
            
        except Exception as e:
            logger.debug("NER author extraction failed", error=str(e))
            return []

    def _looks_like_author_line(self, line: str) -> bool:
        """Check if line looks like it contains author/creator names.
        
        Generic check - works for any document type.

        Args:
            line: Text line

        Returns:
            True if line looks like author names
        """
        if len(line) > 150:
            return False

        # Skip lines with URLs, emails, or common non-author patterns
        skip_patterns = [
            "http", "www.", "@", ".com", ".org", ".edu",
            "email", "tel:", "phone:", "fax:",
            "abstract", "summary", "overview",
            "page ", "doi:", "isbn:"
        ]
        if any(pattern in line.lower() for pattern in skip_patterns):
            return False

        # Check for capital letters (names typically have capitals)
        capital_count = sum(1 for c in line if c.isupper())
        if capital_count < 2:
            return False

        # Check for separators (multiple names)
        if not any(sep in line for sep in [",", " and ", " & "]):
            return False

        # Check if it has reasonable word count (2-10 words for names)
        word_count = len(line.split())
        if not (2 <= word_count <= 10):
            return False

        return True

    def _parse_authors(self, authors_str: str) -> List[str]:
        """Parse author string into list of names.

        Args:
            authors_str: String containing author names

        Returns:
            List of author names
        """
        try:
            # Split by common separators
            authors = []

            # Try splitting by comma first
            if "," in authors_str:
                parts = authors_str.split(",")
                for part in parts:
                    # Check for 'and' in each part
                    if " and " in part.lower():
                        sub_parts = re.split(r"\s+and\s+", part, flags=re.IGNORECASE)
                        authors.extend([p.strip() for p in sub_parts if p.strip()])
                    else:
                        name = part.strip()
                        if name:
                            authors.append(name)
            # Try splitting by 'and'
            elif " and " in authors_str.lower():
                parts = re.split(r"\s+and\s+", authors_str, flags=re.IGNORECASE)
                authors = [p.strip() for p in parts if p.strip()]
            # Try splitting by '&'
            elif "&" in authors_str:
                parts = authors_str.split("&")
                authors = [p.strip() for p in parts if p.strip()]
            else:
                # Single author
                authors = [authors_str.strip()]

            # Clean up author names
            cleaned_authors = []
            for author in authors:
                # Remove common suffixes/prefixes
                author = re.sub(r"\s*\(.*?\)\s*", "", author)  # Remove parentheses
                author = re.sub(r"\s*\[.*?\]\s*", "", author)  # Remove brackets
                author = author.strip()

                # Only keep if it looks like a name (has at least 2 words)
                if len(author.split()) >= 2 and len(author) < 50:
                    cleaned_authors.append(author)

            return cleaned_authors[:10]  # Limit to 10 authors

        except Exception as e:
            logger.debug("Author parsing failed", error=str(e))
            return []

    def _extract_markdown_title(self, content: str) -> Optional[str]:
        """Extract title from Markdown content.

        Args:
            content: Markdown content

        Returns:
            Title if found
        """
        try:
            # Look for first H1 heading
            lines = content.split("\n")
            for line in lines[:20]:  # Check first 20 lines
                line = line.strip()
                if line.startswith("# "):
                    title = line[2:].strip()
                    if title:
                        return title

            return None

        except Exception as e:
            logger.debug("Markdown title extraction failed", error=str(e))
            return None

