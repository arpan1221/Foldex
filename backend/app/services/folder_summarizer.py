"""Post-ingestion folder summarization service.

This service generates comprehensive folder-level summaries and insights
after file ingestion completes. It creates a "meta-understanding" layer
that enhances RAG query context.
"""

from typing import Dict, Any, List, Optional
import asyncio
import time
from collections import Counter
import structlog
import re

from app.database.sqlite_manager import SQLiteManager
from app.rag.llm_chains import OllamaLLM
from app.api.v1.websocket import manager
from app.core.exceptions import ProcessingError
from app.utils.datetime_utils import get_eastern_time
from app.knowledge_graph.graph_builder import GraphBuilder

logger = structlog.get_logger(__name__)


class FolderSummarizer:
    """Generates comprehensive folder-level summaries and insights."""

    def __init__(
        self,
        db: Optional[SQLiteManager] = None,
        llm: Optional[OllamaLLM] = None,
    ):
        """Initialize folder summarizer.

        Args:
            db: Database manager instance
            llm: LLM instance for generating summaries
        """
        self.db = db or SQLiteManager()
        self.llm = llm or OllamaLLM()
        # Lightweight LLM for fast summaries (llama3.2:1b)
        self.lightweight_llm = OllamaLLM(model="llama3.2:1b", temperature=0.1).get_llm()

    async def generate_folder_summary(
        self,
        folder_id: str,
        send_progress: bool = True,
    ) -> Dict[str, Any]:
        """Generate comprehensive folder summary.

        This is the main entry point called after knowledge graph building.
        Implements a 6-step workflow to create folder-level understanding.

        Args:
            folder_id: Folder ID to summarize
            send_progress: Whether to send WebSocket progress updates

        Returns:
            Dictionary containing all summary components

        Raises:
            ProcessingError: If summarization fails
        """
        total_start = time.time()
        try:
            logger.info("Starting folder summarization", folder_id=folder_id)

            if send_progress:
                try:
                    await manager.send_message(
                        folder_id,
                        {
                            "type": "learning_started",
                            "folder_id": folder_id,
                            "message": "Learning your folder...",
                            "progress": 0.0,
                        },
                    )
                    logger.debug("Sent learning_started WebSocket message", folder_id=folder_id)
                except Exception as e:
                    logger.warning(
                        "Failed to send learning_started WebSocket message",
                        folder_id=folder_id,
                        error=str(e),
                    )

            # Update learning status
            await self.db.update_folder_learning_status(folder_id, "learning_in_progress")

            # Step 1: Collect file inventory (20% progress)
            logger.info("Collecting file inventory", folder_id=folder_id)
            file_inventory = await self._collect_file_inventory(folder_id)
            logger.info(
                "File inventory collected",
                folder_id=folder_id,
                total_files=file_inventory['total_files'],
            )
            if send_progress:
                try:
                    await manager.send_message(
                        folder_id,
                        {
                            "type": "summary_progress",
                            "folder_id": folder_id,
                            "message": f"Analyzed {file_inventory['total_files']} files...",
                            "progress": 0.2,
                        },
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to send summary_progress WebSocket message (step 1)",
                        folder_id=folder_id,
                        error=str(e),
                    )

            # Step 2: Generate per-file summaries (40% progress)
            logger.info("Generating per-file summaries", folder_id=folder_id)
            file_summaries = await self._generate_file_summaries(folder_id, file_inventory)
            logger.info(
                "Per-file summaries generated",
                folder_id=folder_id,
                summary_count=len(file_summaries),
            )
            if send_progress:
                try:
                    await manager.send_message(
                        folder_id,
                        {
                            "type": "summary_progress",
                            "folder_id": folder_id,
                            "message": "Summarizing individual files...",
                            "progress": 0.4,
                        },
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to send summary_progress WebSocket message (step 2)",
                        folder_id=folder_id,
                        error=str(e),
                    )

            # Step 3: Extract entity summary (60% progress)
            entity_summary = await self._extract_entity_summary(folder_id)
            if send_progress:
                try:
                    await manager.send_message(
                        folder_id,
                        {
                            "type": "summary_progress",
                            "folder_id": folder_id,
                            "message": "Identifying key entities and themes...",
                            "progress": 0.6,
                        },
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to send summary_progress WebSocket message (step 3)",
                        folder_id=folder_id,
                        error=str(e),
                    )

            # Step 4: Analyze cross-file relationships (80% progress)
            relationship_summary = await self._analyze_relationships(folder_id)
            if send_progress:
                try:
                    await manager.send_message(
                        folder_id,
                        {
                            "type": "summary_progress",
                            "folder_id": folder_id,
                            "message": "Analyzing document relationships...",
                            "progress": 0.8,
                        },
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to send summary_progress WebSocket message (step 4)",
                        folder_id=folder_id,
                        error=str(e),
                    )

            # Step 5: Generate master summary (90% progress)
            # Use lightweight LLM (llama3.2:1b) for fast summary generation
            top_entities = [e["entity"] for e in entity_summary.get("top_entities", [])[:5]]
            master_summary = await self._generate_lightweight_summary(
                file_inventory, top_entities, include_subfolder_info=True
            )
            if send_progress:
                try:
                    await manager.send_message(
                        folder_id,
                        {
                            "type": "summary_progress",
                            "folder_id": folder_id,
                            "message": "Creating folder overview...",
                            "progress": 0.9,
                        },
                    )
                except Exception as e:
                    logger.warning(
                        "Failed to send summary_progress WebSocket message (step 5)",
                        folder_id=folder_id,
                        error=str(e),
                    )

            # Step 6: Determine folder capabilities
            capabilities = await self._determine_capabilities(
                file_inventory,
                entity_summary,
                relationship_summary,
            )

            # Get graph statistics from entity and relationship summaries
            graph_stats = self._get_graph_statistics(entity_summary, relationship_summary)

            # Compile complete summary
            summary_data = {
                "summary": master_summary,
                "file_type_distribution": file_inventory["type_distribution"],
                "entity_summary": entity_summary,
                "relationship_summary": relationship_summary,
                "capabilities": capabilities,
                "graph_statistics": graph_stats,
                "insights": {
                    "total_files": file_inventory["total_files"],
                    "unique_file_types": len(file_inventory["type_distribution"]),
                    "top_themes": [t["theme"] for t in entity_summary.get("top_themes", [])[:5]],
                    "key_relationships": len(relationship_summary),
                    "subfolder_count": file_inventory.get("subfolder_count", 0),
                },
                "learning_completed_at": get_eastern_time(),
            }

            # Store in database
            await self._store_summary(folder_id, summary_data)

            # Update learning status
            await self.db.update_folder_learning_status(folder_id, "learning_complete")
            
            # Build knowledge graph in background (non-blocking)
            # This runs asynchronously and doesn't block the response
            asyncio.create_task(self._build_knowledge_graph_background(folder_id, summary_data))

            # Send completion message
            if send_progress:
                try:
                    await manager.send_message(
                        folder_id,
                        {
                            "type": "summary_complete",
                            "folder_id": folder_id,
                            "message": "Folder learning complete!",
                            "progress": 1.0,
                            "summary_data": {
                                "total_files": summary_data["insights"]["total_files"],
                                "unique_file_types": summary_data["insights"]["unique_file_types"],
                                "top_themes": summary_data["insights"]["top_themes"],
                            },
                        },
                    )
                    logger.debug("Sent summary_complete WebSocket message", folder_id=folder_id)
                except Exception as e:
                    logger.warning(
                        "Failed to send summary_complete WebSocket message",
                        folder_id=folder_id,
                        error=str(e),
                    )

            total_elapsed = time.time() - total_start
            logger.info(
                "Folder summarization completed",
                folder_id=folder_id,
                total_files=file_inventory["total_files"],
                capabilities_count=len(capabilities),
                total_elapsed_seconds=round(total_elapsed, 2),
            )

            return summary_data

        except Exception as e:
            logger.error(
                "Folder summarization failed",
                folder_id=folder_id,
                error=str(e),
                error_type=type(e).__name__,
                exc_info=True,
            )

            # Update status to failed
            try:
                await self.db.update_folder_learning_status(folder_id, "learning_failed")
            except Exception as db_error:
                logger.error(
                    "Failed to update learning status in database",
                    folder_id=folder_id,
                    error=str(db_error),
                )

            if send_progress:
                try:
                    await manager.send_message(
                        folder_id,
                        {
                            "type": "summary_error",
                            "folder_id": folder_id,
                            "error": str(e),
                            "message": f"Failed to learn folder: {str(e)}",
                        },
                    )
                except Exception as ws_error:
                    logger.error(
                        "Failed to send summary_error WebSocket message",
                        folder_id=folder_id,
                        error=str(ws_error),
                    )

            raise ProcessingError(f"Folder summarization failed: {str(e)}") from e

    async def _collect_file_inventory(self, folder_id: str) -> Dict[str, Any]:
        """Collect file inventory with type distribution, including subfolders.

        Args:
            folder_id: Folder ID (root folder)

        Returns:
            Dictionary with total_files, type_distribution, files list, and subfolder info
        """
        # Get files including subfolders
        files = await self.db.get_files_by_folder(folder_id, include_subfolders=True)
        
        # Get subfolder information
        subfolders = await self.db.get_subfolders(folder_id)

        type_distribution = Counter()
        file_details = []

        for file in files:
            mime_type = file.get("mime_type", "unknown")
            file_name = file.get("file_name", "")
            file_type = self._categorize_file_type(mime_type, file_name)
            type_distribution[file_type] += 1

            file_details.append({
                "file_id": file.get("file_id"),
                "file_name": file_name,
                "file_type": file_type,
                "mime_type": mime_type,
            })

        return {
            "total_files": len(files),
            "type_distribution": dict(type_distribution),
            "files": file_details,
            "subfolder_count": len(subfolders),
            "subfolders": [
                {
                    "folder_id": sf["folder_id"],
                    "folder_name": sf["folder_name"],
                    "file_count": sf.get("file_count", 0),
                }
                for sf in subfolders
            ],
        }

    def _categorize_file_type(self, mime_type: str, file_name: Optional[str] = None) -> str:
        """Categorize MIME type into specific file type category.
        
        Keep file types separate - CSV, PDF, MD should NOT be clubbed together.
        This ensures accurate file type distribution in folder summaries.

        Args:
            mime_type: MIME type string
            file_name: Optional file name for extension-based detection

        Returns:
            Specific file type category
        """
        mime_lower = mime_type.lower()
        
        # Check file extension if mime type is too generic
        if file_name:
            file_name_lower = file_name.lower()
            if file_name_lower.endswith('.csv'):
                return "CSV"
            elif file_name_lower.endswith(('.tsv', '.tab')):
                return "TSV"
            elif file_name_lower.endswith(('.md', '.markdown')):
                return "Markdown"
            elif file_name_lower.endswith('.pdf'):
                return "PDF"
            elif file_name_lower.endswith(('.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs', '.rb', '.php', '.cs', '.swift', '.kt')):
                return "Code"
            elif file_name_lower.endswith(('.html', '.htm')):
                return "HTML"
            elif file_name_lower.endswith(('.json', '.xml', '.yaml', '.yml')):
                return "Data"
        
        # MIME type-based detection (more specific first)
        if mime_lower == "text/csv" or "csv" in mime_lower:
            return "CSV"
        elif mime_lower == "text/tab-separated-values" or "tsv" in mime_lower:
            return "TSV"
        elif mime_lower == "text/markdown" or "markdown" in mime_lower:
            return "Markdown"
        elif "pdf" in mime_lower:
            return "PDF"
        elif "audio" in mime_lower:
            return "Audio"
        elif "image" in mime_lower:
            return "Image"
        elif "video" in mime_lower:
            return "Video"
        elif "text/html" in mime_lower or "html" in mime_lower:
            return "HTML"
        elif "application/json" in mime_lower or "json" in mime_lower:
            return "Data"
        elif "text/plain" in mime_lower or (mime_lower.startswith("text/") and "csv" not in mime_lower and "markdown" not in mime_lower):
            return "Text"
        elif "python" in mime_lower or "javascript" in mime_lower or "x-python" in mime_lower or "x-javascript" in mime_lower:
            return "Code"
        elif "document" in mime_lower or "msword" in mime_lower or "wordprocessingml" in mime_lower:
            return "Document"
        elif "spreadsheet" in mime_lower or "excel" in mime_lower or "officedocument.spreadsheetml" in mime_lower:
            return "Spreadsheet"
        elif "presentation" in mime_lower or "powerpoint" in mime_lower:
            return "Presentation"
        else:
            return "Other"

    async def _generate_file_summaries(
        self,
        folder_id: str,
        file_inventory: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate brief summaries for sample of files.

        OPTIMIZED: Batch query all chunks at once instead of N+1 queries.
        Uses lightweight content extraction (no LLM calls).

        Args:
            folder_id: Folder ID
            file_inventory: File inventory data

        Returns:
            List of file summaries
        """
        import time
        start_time = time.time()
        
        summaries = []
        # Process ALL files to ensure every file gets at least one line in summary
        files_to_summarize = file_inventory["files"]
        file_ids = {f["file_id"]: f for f in files_to_summarize}

        try:
            # OPTIMIZED: Get all chunks for folder in ONE query instead of N queries
            # Get up to 30 chunks (2 per file for 15 files)
            all_chunks = await self.db.get_chunks_by_folder(folder_id, limit=1000, include_subfolders=True)
            
            # Group chunks by file_id - use up to 3 chunks per file for better context
            # Prioritize earlier chunks as they often contain headers/titles/introductions
            chunks_by_file: Dict[str, List[str]] = {}  # Store content strings only
            chunk_counts_by_file: Dict[str, int] = {}  # Track chunk count per file
            
            for chunk in all_chunks:
                # Handle DocumentChunk objects
                if hasattr(chunk, 'file_id'):
                    file_id = chunk.file_id
                    content = chunk.content if hasattr(chunk, 'content') else ""
                elif isinstance(chunk, dict):
                    file_id = chunk.get("file_id") or chunk.get("metadata", {}).get("file_id")
                    content = chunk.get("content", "")
                else:
                    continue
                
                if not file_id or file_id not in file_ids or not content:
                    continue
                
                # Initialize if needed
                if file_id not in chunks_by_file:
                    chunks_by_file[file_id] = []
                    chunk_counts_by_file[file_id] = 0
                
                # Use up to 3 chunks per file (increased from 2 for better context)
                # This ensures we capture more content from each file
                if chunk_counts_by_file[file_id] < 3:
                    chunks_by_file[file_id].append(content)
                    chunk_counts_by_file[file_id] += 1

            # Generate summaries from grouped chunks using lightweight LLM
            # Ensure EVERY file gets a summary (even if no chunks found)
            for file_id, file_info in file_ids.items():
                chunks = chunks_by_file.get(file_id, [])
                
                # If no chunks found, create a simple summary based on file type and name
                if not chunks:
                    file_type_desc = file_info['file_type'].lower()
                    summary_text = f"{file_info['file_name']} is a {file_type_desc} file in this folder."
                    summaries.append({
                        "file_id": file_id,
                        "file_name": file_info["file_name"],
                        "file_type": file_info["file_type"],
                        "summary": summary_text,
                    })
                    continue

                # Combine chunks intelligently - use up to 800 chars for better context
                # This allows for more comprehensive file summaries
                combined_content = " ".join(chunks)[:800].strip()
                
                if not combined_content:
                    continue

                # Generate summary using lightweight LLM
                try:
                    # Create a more descriptive prompt that guides the LLM better
                    file_type_description = {
                        "CSV": "CSV data file",
                        "TSV": "TSV data file",
                        "PDF": "PDF document",
                        "Markdown": "Markdown file",
                        "Text": "text file",
                        "Code": "code file",
                        "Document": "document",
                        "Spreadsheet": "spreadsheet",
                        "HTML": "HTML file",
                        "Data": "data file",
                        "Audio": "audio file",
                        "Image": "image file",
                    }.get(file_info['file_type'], file_info['file_type'].lower() + " file")
                    
                    summary_prompt = f"""File: {file_info['file_name']} ({file_type_description}).

Content preview:
{combined_content}

Provide a one-sentence summary describing what this file contains or discusses:"""
                    
                    response = await asyncio.wait_for(
                        self.lightweight_llm.ainvoke(summary_prompt),
                        timeout=6.0  # Increased timeout for more reliable completion
                    )
                    summary_text = response.content.strip() if hasattr(response, "content") else str(response).strip()
                    
                    # Clean up summary (remove quotes, prefixes like "Summary:", etc.)
                    summary_text = re.sub(r'^(Summary|File|This file):\s*', '', summary_text, flags=re.IGNORECASE).strip()
                    summary_text = summary_text.strip('"\'')
                    
                    if not summary_text or len(summary_text) < 10:
                        # Fallback: create a simple descriptive summary
                        preview = combined_content[:100].replace('\n', ' ').strip()
                        summary_text = f"{file_info['file_name']} contains {file_type_description.lower()} content: {preview}..."
                except asyncio.TimeoutError:
                    logger.warning(
                        "LLM summary generation timed out for file, using fallback",
                        file_id=file_id,
                        file_name=file_info["file_name"],
                        timeout_seconds=6.0
                    )
                    # Fallback: create a descriptive summary from content
                    preview = combined_content[:100].replace('\n', ' ').strip()
                    file_type_desc = file_info['file_type'].lower()
                    summary_text = f"{file_info['file_name']} is a {file_type_desc} file containing: {preview}..."
                except Exception as e:
                    error_msg = str(e) if str(e) else f"{type(e).__name__}"
                    logger.warning(
                        "Failed to generate LLM summary for file, using fallback",
                        file_id=file_id,
                        file_name=file_info["file_name"],
                        error=error_msg,
                        error_type=type(e).__name__,
                        exc_info=False  # Don't log full traceback for expected fallbacks
                    )
                    # Fallback: create a descriptive summary from content
                    preview = combined_content[:100].replace('\n', ' ').strip()
                    file_type_desc = file_info['file_type'].lower()
                    summary_text = f"{file_info['file_name']} is a {file_type_desc} file containing: {preview}..."

                summaries.append({
                    "file_id": file_id,
                    "file_name": file_info["file_name"],
                    "file_type": file_info["file_type"],
                    "summary": summary_text,
                })
            
            elapsed = time.time() - start_time
            logger.info(
                "File summaries generated",
                folder_id=folder_id,
                summary_count=len(summaries),
                elapsed_seconds=round(elapsed, 2),
            )

        except Exception as e:
            logger.error(
                "Failed to generate file summaries",
                folder_id=folder_id,
                error=str(e),
                exc_info=True,
            )

        return summaries

    async def _extract_entity_summary(self, folder_id: str) -> Dict[str, Any]:
        """Extract top entities and themes from folder chunks.

        Args:
            folder_id: Folder ID

        Returns:
            Dictionary with top_entities and top_themes
        """
        try:
            # Get chunks for the folder
            chunks = await self.db.get_chunks_by_folder(folder_id, limit=100, include_subfolders=True)
            
            if not chunks:
                return {
                    "top_entities": [],
                    "top_themes": [],
                }
            
            # Extract entities (keywords/important terms) from chunk content
            entity_counts = Counter()
            theme_keywords = Counter()
            
            # Common entity patterns
            entity_patterns = [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized phrases (potential proper nouns)
                r'\b\d{4}\b',  # Years
                r'\b[A-Z]{2,}\b',  # Acronyms
            ]
            
            # Theme keywords
            theme_keyword_groups = {
                'research': ['research', 'study', 'analysis', 'investigation', 'findings', 'data', 'results'],
                'technical': ['system', 'algorithm', 'implementation', 'architecture', 'design', 'code', 'function'],
                'business': ['business', 'market', 'company', 'customer', 'revenue', 'profit', 'strategy'],
                'academic': ['paper', 'thesis', 'publication', 'journal', 'article', 'reference', 'citation'],
                'process': ['process', 'workflow', 'procedure', 'method', 'approach', 'technique'],
            }
            
            for chunk in chunks:
                content = chunk.content.lower()
                
                # Extract entities (capitalized phrases that appear multiple times)
                for pattern in entity_patterns:
                    matches = re.findall(pattern, chunk.content)
                    for match in matches:
                        if len(match) > 2 and match.isupper() or (match[0].isupper() and len(match.split()) <= 3):
                            entity_counts[match] += 1
                
                # Count theme keywords
                for theme, keywords in theme_keyword_groups.items():
                    for keyword in keywords:
                        if keyword in content:
                            theme_keywords[theme] += 1
            
            # Get top entities (excluding very common words)
            top_entities = []
            common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'way', 'use', 'man', 'say', 'she', 'use'}
            
            for entity, count in entity_counts.most_common(20):
                entity_lower = entity.lower()
                if count >= 2 and entity_lower not in common_words and len(entity) > 2:
                    top_entities.append({
                        "entity": entity,
                        "count": count,
                    })
            
            # Get top themes
            top_themes = []
            for theme, count in theme_keywords.most_common(5):
                if count > 0:
                    top_themes.append({
                        "theme": theme,
                        "count": count,
                    })
            
            return {
                "top_entities": top_entities[:10],  # Top 10 entities
                "top_themes": top_themes,
            }
            
        except Exception as e:
            logger.error(
                "Failed to extract entity summary",
                folder_id=folder_id,
                error=str(e),
                exc_info=True,
            )
        return {
            "top_entities": [],
            "top_themes": [],
        }

    async def _analyze_relationships(self, folder_id: str) -> List[Dict[str, Any]]:
        """Analyze key cross-file relationships based on content similarity.

        Args:
            folder_id: Folder ID

        Returns:
            List of relationship summaries
        """
        try:
            # Get chunks grouped by file
            chunks = await self.db.get_chunks_by_folder(folder_id, limit=500, include_subfolders=True)
            
            if not chunks:
                return []
            
            # Group chunks by file_id
            chunks_by_file: Dict[str, List[str]] = {}
            file_names: Dict[str, str] = {}
            
            for chunk in chunks:
                file_id = chunk.file_id
                if file_id not in chunks_by_file:
                    chunks_by_file[file_id] = []
                    file_names[file_id] = chunk.metadata.get("file_name", file_id)
                
                chunks_by_file[file_id].append(chunk.content.lower())
            
            # Extract keywords from each file
            file_keywords: Dict[str, set] = {}
            for file_id, contents in chunks_by_file.items():
                keywords = set()
                all_text = " ".join(contents)
                
                # Extract meaningful words (3+ chars, not common words)
                words = re.findall(r'\b[a-z]{3,}\b', all_text)
                common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'way', 'use', 'man', 'say', 'she', 'use', 'this', 'that', 'with', 'have', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'}
                
                word_counts = Counter(words)
                for word, count in word_counts.most_common(50):
                    if word not in common_words:
                        keywords.add(word)
                
                file_keywords[file_id] = keywords
            
            # Find relationships based on keyword overlap
            relationships = []
            file_ids = list(chunks_by_file.keys())
            
            for i, file_id1 in enumerate(file_ids):
                for file_id2 in file_ids[i+1:]:
                    keywords1 = file_keywords.get(file_id1, set())
                    keywords2 = file_keywords.get(file_id2, set())
                    
                    # Calculate overlap
                    overlap = keywords1.intersection(keywords2)
                    if len(overlap) >= 5:  # At least 5 shared keywords
                        overlap_ratio = len(overlap) / max(len(keywords1), len(keywords2), 1)
                        
                        if overlap_ratio > 0.1:  # At least 10% overlap
                            relationships.append({
                                "source_file": file_names.get(file_id1, file_id1),
                                "target_file": file_names.get(file_id2, file_id2),
                                "relationship_type": "content_similarity",
                                "confidence": min(overlap_ratio * 2, 1.0),  # Scale to 0-1
                                "shared_keywords": len(overlap),
                            })
            
            # Sort by confidence
            relationships.sort(key=lambda x: x["confidence"], reverse=True)
            
            return relationships[:10]  # Return top 10 relationships
            
        except Exception as e:
            logger.error(
                "Failed to analyze relationships",
                folder_id=folder_id,
                error=str(e),
                exc_info=True,
            )
            return []

    async def _generate_lightweight_summary(
        self, 
        file_inventory: Dict[str, Any], 
        top_entities: List[str],
        include_subfolder_info: bool = False,
    ) -> str:
        """Generate summary with lightweight model - 5 second max.
        
        Args:
            file_inventory: File inventory data (may include subfolder_count and subfolders)
            top_entities: List of top entity strings
            include_subfolder_info: Whether to include subfolder information in summary
            
        Returns:
            One sentence summary
        """
        # Ultra-short prompt for 1B model
        entity_text = ", ".join(top_entities[:3]) if top_entities else "various topics"
        
        # Include subfolder information if available
        subfolder_info = ""
        if include_subfolder_info and file_inventory.get("subfolder_count", 0) > 0:
            subfolder_names = [sf["folder_name"] for sf in file_inventory.get("subfolders", [])[:5]]
            if subfolder_names:
                subfolder_info = f" Organized in {file_inventory['subfolder_count']} subfolders including: {', '.join(subfolder_names)}."
        
        prompt = f"""Folder: {file_inventory['total_files']} files{subfolder_info}. Main topics: {entity_text}. 
        
One sentence summary:"""
        
        try:
            # Use lightweight model with short timeout
            response = await asyncio.wait_for(
                self.lightweight_llm.ainvoke(prompt),  # Llama 3.2 1B
                timeout=5.0  # Much shorter timeout
            )
            summary = response.content.strip() if hasattr(response, "content") else str(response).strip()
            return summary if summary else self._generate_fast_summary(file_inventory, [])
        except Exception as e:
            logger.warning("Lightweight summary generation failed, using fallback", error=str(e))
            # Fallback to template
            return self._generate_fast_summary(file_inventory, [])

    def _generate_fast_summary(
        self,
        file_inventory: Dict[str, Any],
        file_summaries: List[Dict[str, Any]],
    ) -> str:
        """Generate fast folder summary without LLM call.
        
        Args:
            file_inventory: File inventory data (may include subfolder_count and subfolders)
            file_summaries: Per-file summaries
            
        Returns:
            Folder summary string
        """
        type_summary = ", ".join([
            f"{count} {ftype}" for ftype, count in file_inventory['type_distribution'].items()
        ])
        
        file_names = ", ".join([f['file_name'] for f in file_inventory['files'][:10]])
        if len(file_inventory['files']) > 10:
            file_names += f", and {len(file_inventory['files']) - 10} more"
        
        summary = f"This folder contains {file_inventory['total_files']} documents ({type_summary}). Files include: {file_names}."
        
        # Add subfolder information if available
        if file_inventory.get("subfolder_count", 0) > 0:
            subfolder_names = [sf["folder_name"] for sf in file_inventory.get("subfolders", [])[:5]]
            if subfolder_names:
                summary += f" Organized in {file_inventory['subfolder_count']} subfolders: {', '.join(subfolder_names)}."
            else:
                summary += f" Organized in {file_inventory['subfolder_count']} subfolder{'s' if file_inventory['subfolder_count'] > 1 else ''}."
        
        return summary

    async def _generate_master_summary(
        self,
        folder_id: str,
        file_inventory: Dict[str, Any],
        file_summaries: List[Dict[str, Any]],
        entity_summary: Dict[str, Any],
        relationship_summary: List[Dict[str, Any]],
    ) -> str:
        """Generate master folder-level summary using LLM.

        Args:
            folder_id: Folder ID
            file_inventory: File inventory data
            file_summaries: Per-file summaries
            entity_summary: Entity summary data
            relationship_summary: Relationship summary data

        Returns:
            Master folder summary string
        """
        # Build file type summary
        type_summary = ", ".join([
            f"{count} {ftype}" for ftype, count in file_inventory['type_distribution'].items()
        ])

        # OPTIMIZED: Use shorter, more focused prompt for faster LLM response
        # Limit file list to 10 files to reduce prompt size
        limited_file_list = ", ".join([
            f['file_name'] for f in file_inventory["files"][:10]
        ])
        
        master_prompt = f"""Summarize this folder ({file_inventory['total_files']} files, types: {type_summary}) in 2-3 sentences. Files: {limited_file_list}

Summary:"""

        try:
            # Use timeout to prevent hanging
            response = await asyncio.wait_for(
                self.llm.get_llm().ainvoke(master_prompt),
                timeout=30.0  # 30 second timeout
            )
            summary = response.content if hasattr(response, "content") else str(response)
            summary_text = summary.strip()
            
            # Ensure summary is not empty
            if not summary_text or len(summary_text) < 20:
                raise ValueError("Summary too short or empty")
            
            return summary_text
        except asyncio.TimeoutError:
            logger.warning("Master summary generation timed out", folder_id=folder_id)
            return f"This folder contains {file_inventory['total_files']} documents ({type_summary})."
        except Exception as e:
            logger.error("Failed to generate master summary", folder_id=folder_id, error=str(e))
            # Return fallback summary
            return f"This folder contains {file_inventory['total_files']} documents including {type_summary}."

    async def _determine_capabilities(
        self,
        file_inventory: Dict[str, Any],
        entity_summary: Dict[str, Any],
        relationship_summary: List[Dict[str, Any]],
    ) -> List[str]:
        """Determine what types of questions can be answered from this folder.

        Args:
            file_inventory: File inventory data
            entity_summary: Entity summary data
            relationship_summary: Relationship summary data

        Returns:
            List of capability descriptions
        """
        capabilities = []

        # Based on file types
        type_dist = file_inventory["type_distribution"]
        if "PDF" in type_dist:
            capabilities.append("Answer questions about written documents and research papers")
        if "Audio" in type_dist:
            capabilities.append("Provide information from audio transcripts")
        if "Code" in type_dist:
            capabilities.append("Explain code structure and implementation details")
        if "Spreadsheet" in type_dist:
            capabilities.append("Analyze data from spreadsheets")

        # Based on entities
        if entity_summary.get("top_entities"):
            capabilities.append("Discuss specific entities, people, or concepts mentioned in documents")

        # Based on relationships
        if relationship_summary:
            capabilities.append("Compare and contrast information across multiple documents")
            capabilities.append("Trace connections and relationships between files")

        # General capabilities
        if file_inventory["total_files"] > 1:
            capabilities.append("Synthesize information from multiple sources")

        return capabilities

    def _get_graph_statistics(
        self, 
        entity_summary: Dict[str, Any], 
        relationship_summary: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get knowledge graph statistics from entity and relationship summaries.

        Args:
            entity_summary: Entity summary dictionary with top_entities
            relationship_summary: List of relationship dictionaries

        Returns:
            Graph statistics dictionary with node_count, edge_count, relationship_types
        """
        try:
            # Get top entities (entities are nodes in the graph)
            top_entities = entity_summary.get("top_entities", [])
            
            # Count unique relationship types
            relationship_types = set()
            for rel in relationship_summary:
                rel_type = rel.get("relationship_type", "unknown")
                relationship_types.add(rel_type)
            
            # Node count = number of unique entities
            node_count = len(top_entities)
            
            # Edge count = number of relationships
            edge_count = len(relationship_summary)
            
            logger.debug(
                "Graph statistics calculated",
                node_count=node_count,
                edge_count=edge_count,
                relationship_types_count=len(relationship_types),
            )
            
            return {
                "node_count": node_count,
                "edge_count": edge_count,
                "relationship_types": len(relationship_types),
            }
        except Exception as e:
            logger.error(
                "Failed to calculate graph statistics",
                error=str(e),
                exc_info=True,
            )
            return {
                "node_count": 0,
                "edge_count": 0,
                "relationship_types": 0,
            }

    async def _build_knowledge_graph_background(
        self,
        folder_id: str,
        summary_data: Dict[str, Any],
    ) -> None:
        """Build knowledge graph in background after summarization.
        
        This runs asynchronously and doesn't block. Users can continue chatting
        while the graph is being built. A WebSocket notification is sent when complete.
        
        Args:
            folder_id: Folder ID
            summary_data: Summary data containing entity and relationship information
        """
        try:
            logger.info("Starting background knowledge graph construction", folder_id=folder_id)
            
            # Send initial status update
            try:
                await manager.send_message(
                    folder_id,
                    {
                        "type": "graph_building",
                        "folder_id": folder_id,
                        "message": "Building knowledge graph...",
                    },
                )
            except Exception as e:
                logger.warning("Failed to send graph_building WebSocket message", error=str(e))
            
            # Get chunks for the folder
            chunks = await self.db.get_chunks_by_folder(folder_id, limit=500, include_subfolders=True)
            
            if not chunks:
                logger.warning("No chunks found for graph construction", folder_id=folder_id)
                return
            
            # Get relationship summary from summary_data
            relationship_summary = summary_data.get("relationship_summary", [])
            
            # Build graph using GraphBuilder
            graph_builder = GraphBuilder()
            graph = await graph_builder.build_graph(
                chunks=chunks,
                relationships=relationship_summary,
            )
            
            # Convert graph to JSON format for storage and visualization
            if hasattr(graph, "to_json"):
                # FoldexKnowledgeGraph instance
                graph_json = graph.to_json()
            else:
                # NetworkX graph - convert to JSON format
                import networkx as nx
                nodes = []
                for node, data in graph.nodes(data=True):
                    nodes.append({
                        "id": str(node),
                        "label": data.get("label", str(node)),
                        "type": data.get("type", "unknown"),
                        "node_type": data.get("node_type", "entity"),
                    })
                
                links = []
                for source, target, data in graph.edges(data=True):
                    links.append({
                        "source": str(source),
                        "target": str(target),
                        "relation": data.get("relation", "related"),
                    })
                
                graph_json = {
                    "nodes": nodes,
                    "links": links,
                    "stats": {
                        "node_count": len(nodes),
                        "link_count": len(links),
                        "document_count": len([
                            n for n in nodes if n.get("node_type") == "document"
                        ]),
                    },
                }
            
            # Store graph data as JSON string (not pickle, for easier frontend access)
            import json
            graph_data_json = json.dumps(graph_json).encode('utf-8')
            await self.db.store_knowledge_graph(folder_id, graph_data_json)
            
            logger.info(
                "Knowledge graph built and stored",
                folder_id=folder_id,
                node_count=graph_json.get("stats", {}).get("node_count", 0),
                link_count=graph_json.get("stats", {}).get("link_count", 0),
            )
            
            # Send completion notification
            try:
                await manager.send_message(
                    folder_id,
                    {
                        "type": "graph_complete",
                        "folder_id": folder_id,
                        "message": "Knowledge graph ready! Click to visualize relationships.",
                        "graph_stats": graph_json.get("stats", {}),
                    },
                )
            except Exception as e:
                logger.warning("Failed to send graph_complete WebSocket message", error=str(e))
                
        except Exception as e:
            logger.error(
                "Background knowledge graph construction failed",
                folder_id=folder_id,
                error=str(e),
                exc_info=True,
            )
            # Send error notification (non-blocking)
            try:
                await manager.send_message(
                    folder_id,
                    {
                        "type": "graph_error",
                        "folder_id": folder_id,
                        "message": "Failed to build knowledge graph",
                        "error": str(e),
                    },
                )
            except Exception:
                pass  # Ignore WebSocket errors

    async def _store_summary(self, folder_id: str, summary_data: Dict[str, Any]) -> None:
        """Store summary data in database.

        Args:
            folder_id: Folder ID
            summary_data: Complete summary data dictionary
        """
        await self.db.update_folder_summary(
            folder_id=folder_id,
            summary=summary_data["summary"],
            insights=summary_data["insights"],
            file_type_distribution=summary_data["file_type_distribution"],
            entity_summary=summary_data["entity_summary"],
            relationship_summary=summary_data["relationship_summary"],
            capabilities=summary_data["capabilities"],
            graph_statistics=summary_data["graph_statistics"],
            learning_completed_at=summary_data["learning_completed_at"],
        )


def get_folder_summarizer() -> FolderSummarizer:
    """Get global folder summarizer instance.

    Returns:
        FolderSummarizer instance
    """
    return FolderSummarizer()
