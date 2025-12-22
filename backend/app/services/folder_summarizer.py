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
            # Pass file_summaries to enable LLM-based theme extraction
            entity_summary = await self._extract_entity_summary(folder_id, file_summaries=file_summaries)
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
            # Use comprehensive LLM summary that includes cross-file similarities
            master_summary_raw = await self._generate_master_summary(
                folder_id,
                file_inventory,
                file_summaries,
                entity_summary,
                relationship_summary,
            )
            # Post-process summary to clean up formatting and remove redundant content
            master_summary = self._cleanup_summary(master_summary_raw)
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

            # Send completion message IMMEDIATELY (before starting background tasks)
            # This ensures the frontend can navigate to chat right away
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
                    logger.info("Sent summary_complete WebSocket message", folder_id=folder_id)
                except Exception as e:
                    logger.error(
                        "Failed to send summary_complete WebSocket message",
                        folder_id=folder_id,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
            
            # Knowledge graph building is now user-initiated only
            # Users click "Build knowledge graph" button in sidebar to trigger it
            # No automatic background building after summarization

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
            # Get all chunks to enable diverse sampling
            all_chunks = await self.db.get_chunks_by_folder(folder_id, limit=1000, include_subfolders=True)
            
            # Group chunks by file_id - use diverse sampling strategy
            # Store full chunk objects for better sampling
            chunks_by_file: Dict[str, List[Dict[str, Any]]] = {}  # Store chunk dicts with content and index
            
            for idx, chunk in enumerate(all_chunks):
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
                    
                chunks_by_file[file_id].append({
                    "content": content,
                    "index": idx,
                })
            
            # Now apply diverse sampling strategy per file
            # Sample from beginning, middle, and end if file has many chunks
            sampled_chunks_by_file: Dict[str, List[str]] = {}
            for file_id, chunk_list in chunks_by_file.items():
                total_chunks = len(chunk_list)
                sampled = []
                
                if total_chunks <= 5:
                    # Small files: take all chunks
                    sampled = [chunk["content"] for chunk in chunk_list]
                elif total_chunks <= 10:
                    # Medium files: take first 2, middle 2, last 1
                    sampled.append(chunk_list[0]["content"])  # First
                    sampled.append(chunk_list[1]["content"] if total_chunks > 1 else chunk_list[0]["content"])
                    mid_point = total_chunks // 2
                    sampled.append(chunk_list[mid_point]["content"])
                    sampled.append(chunk_list[mid_point + 1]["content"] if mid_point + 1 < total_chunks else chunk_list[mid_point]["content"])
                    sampled.append(chunk_list[-1]["content"])  # Last
                else:
                    # Large files: take first 2, sample 3 from middle, last 1 (total 6)
                    sampled.append(chunk_list[0]["content"])  # First chunk (often intro/header)
                    sampled.append(chunk_list[1]["content"] if total_chunks > 1 else chunk_list[0]["content"])
                    
                    # Sample 3 chunks from middle (at 1/4, 1/2, 3/4 positions)
                    quarter = total_chunks // 4
                    mid = total_chunks // 2
                    three_quarter = (3 * total_chunks) // 4
                    sampled.append(chunk_list[quarter]["content"])
                    sampled.append(chunk_list[mid]["content"])
                    sampled.append(chunk_list[three_quarter]["content"])
                    
                    sampled.append(chunk_list[-1]["content"])  # Last chunk (often conclusion)
                
                # Remove duplicates while preserving order
                seen = set()
                unique_sampled = []
                for content in sampled:
                    content_hash = hash(content[:100])  # Hash first 100 chars to detect near-duplicates
                    if content_hash not in seen:
                        seen.add(content_hash)
                        unique_sampled.append(content)
                
                sampled_chunks_by_file[file_id] = unique_sampled

            # Prepare file data for batch summarization
            # Build structured file information for single LLM call
            file_data_list = []
            file_type_descriptions = {
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
            }
            
            for file_id, file_info in file_ids.items():
                chunks = sampled_chunks_by_file.get(file_id, [])
                
                # Combine chunks - use up to 2000 chars per file for better context
                # This gives LLM more actual content to work with
                combined_content = " ".join(chunks)[:2000].strip() if chunks else ""
                
                file_type_desc = file_type_descriptions.get(
                    file_info['file_type'], 
                    file_info['file_type'].lower() + " file"
                )
                
                file_data_list.append({
                    "file_id": file_id,
                    "file_name": file_info["file_name"],
                    "file_type": file_info["file_type"],
                    "file_type_desc": file_type_desc,
                    "content": combined_content,
                    "has_content": bool(combined_content),
                })
            
            # Generate all summaries in a single batch LLM call
            try:
                # Build batch prompt with all files
                batch_prompt_parts = [
                    "You are a document summarization assistant. Your task is to provide one-sentence summaries for each file listed below.",
                    "",
                    "INSTRUCTIONS:",
                    "1. For EACH file listed, provide exactly ONE sentence describing what the file contains or discusses",
                    "2. Be specific and factual - reference actual content from the file preview",
                    "3. Avoid vague phrases like 'contains information' or 'is about various topics'",
                    "4. Do NOT include any meta-commentary, explanations, or additional text",
                    "5. Do NOT repeat words like 'CRITICAL' or 'IMPORTANT'",
                    "",
                    "REQUIRED OUTPUT FORMAT (use this EXACT format for each file):",
                    "FILE: <exact file name as shown above>",
                    "SUMMARY: <one sentence only>",
                    "",
                    "IMPORTANT: You must provide a summary for ALL files. Do not skip any files.",
                    "",
                    "Files to summarize:",
                    "",
                ]
                
                # Add each file to the prompt
                for idx, file_data in enumerate(file_data_list, 1):
                    batch_prompt_parts.append(f"--- File {idx} ---")
                    batch_prompt_parts.append(f"File Name: {file_data['file_name']}")
                    batch_prompt_parts.append(f"File Type: {file_data['file_type_desc']}")
                    if file_data['has_content']:
                        batch_prompt_parts.append(f"Content Preview: {file_data['content']}")
                    else:
                        batch_prompt_parts.append("Content Preview: [No content extracted]")
                    batch_prompt_parts.append("")
                
                batch_prompt_parts.append("")
                batch_prompt_parts.append("Now provide summaries for ALL files above using the format:")
                batch_prompt_parts.append("FILE: <file name>")
                batch_prompt_parts.append("SUMMARY: <one sentence>")
                batch_prompt_parts.append("")
                batch_prompt_parts.append("Start your response with the first FILE: line. Do not include any introductory text.")
                batch_prompt = "\n".join(batch_prompt_parts)
                
                # Call LLM once for all files
                logger.info(
                    "Generating batch file summaries",
                    folder_id=folder_id,
                    file_count=len(file_data_list),
                )
                
                response = await asyncio.wait_for(
                    self.lightweight_llm.ainvoke(batch_prompt),
                    timeout=90.0  # Extended timeout (> 1 minute) for batch processing
                )
                response_text = response.content.strip() if hasattr(response, "content") else str(response).strip()
                
                # Check for LLM refusal or meta-commentary
                refusal_indicators = [
                    "i can help you",
                    "i notice",
                    "however i notice",
                    "i'll be happy",
                    "if you provide",
                    "for now",
                    "based on the single",
                    "i can help",
                    "i notice that there is only",
                ]
                response_lower = response_text.lower()
                has_refusal = any(indicator in response_lower for indicator in refusal_indicators)
                
                if has_refusal:
                    logger.warning(
                        "LLM refusal detected in batch summary response, using fallback",
                        folder_id=folder_id,
                        response_preview=response_text[:200],
                    )
                    # Skip parsing and use fallback summaries
                    parsed_summaries = {}
                else:
                    # Parse batch response to extract individual summaries
                    parsed_summaries = self._parse_batch_summaries(response_text, file_data_list)
                
                # Use parsed summaries, fallback to generated ones for missing files
                for file_data in file_data_list:
                    file_id = file_data["file_id"]
                    parsed_summary = parsed_summaries.get(file_id)
                    
                    if parsed_summary:
                        # Clean up parsed summary one more time to remove any remaining "CRITICAL"
                        summary_text = re.sub(r'\b(CRITICAL|IMPORTANT)\b', '', parsed_summary, flags=re.IGNORECASE).strip()
                        summary_text = re.sub(r'\s+', ' ', summary_text).strip()
                        # If cleaned summary is too short or invalid, use fallback
                        if not summary_text or len(summary_text) < 10:
                            if file_data["has_content"]:
                                preview = file_data["content"][:100].replace('\n', ' ').strip()
                                summary_text = f"{file_data['file_name']} is a {file_data['file_type_desc'].lower()} containing relevant content. The file includes: {preview}..."
                            else:
                                file_type_desc = file_data['file_type'].lower()
                                summary_text = f"{file_data['file_name']} is a {file_type_desc} file in this folder. The file's content could not be extracted for summarization."
                    elif file_data["has_content"]:
                        # Fallback: create a two-sentence descriptive summary from content
                        preview = file_data["content"][:100].replace('\n', ' ').strip()
                        summary_text = f"{file_data['file_name']} is a {file_data['file_type_desc'].lower()} containing relevant content. The file includes: {preview}..."
                    else:
                        # No content available
                        file_type_desc = file_data['file_type'].lower()
                        summary_text = f"{file_data['file_name']} is a {file_type_desc} file in this folder. The file's content could not be extracted for summarization."

                    summaries.append({
                        "file_id": file_id,
                        "file_name": file_data["file_name"],
                        "file_type": file_data["file_type"],
                        "summary": summary_text,
                    })
                    
            except asyncio.TimeoutError:
                logger.warning(
                    "Batch LLM summary generation timed out, using fallbacks",
                    folder_id=folder_id,
                    file_count=len(file_data_list),
                    timeout_seconds=90.0  # Extended timeout (> 1 minute)
                )
                # Fallback: create simple summaries for all files
                for file_data in file_data_list:
                    if file_data["has_content"]:
                        preview = file_data["content"][:100].replace('\n', ' ').strip()
                        summary_text = f"{file_data['file_name']} is a {file_data['file_type_desc'].lower()} file with content relevant to the folder. The file contains: {preview}..."
                    else:
                        file_type_desc = file_data['file_type'].lower()
                        summary_text = f"{file_data['file_name']} is a {file_type_desc} file in this folder. The file's content could not be extracted for summarization."
                    
                    summaries.append({
                        "file_id": file_data["file_id"],
                        "file_name": file_data["file_name"],
                        "file_type": file_data["file_type"],
                        "summary": summary_text,
                    })
            except Exception as e:
                error_msg = str(e) if str(e) else f"{type(e).__name__}"
                logger.warning(
                    "Batch LLM summary generation failed, using fallbacks",
                    folder_id=folder_id,
                    file_count=len(file_data_list),
                    error=error_msg,
                    error_type=type(e).__name__,
                    exc_info=False
                )
                # Fallback: create simple summaries for all files
                for file_data in file_data_list:
                    if file_data["has_content"]:
                        preview = file_data["content"][:100].replace('\n', ' ').strip()
                        summary_text = f"{file_data['file_name']} is a {file_data['file_type_desc'].lower()} file with content relevant to the folder. The file contains: {preview}..."
                    else:
                        file_type_desc = file_data['file_type'].lower()
                        summary_text = f"{file_data['file_name']} is a {file_type_desc} file in this folder. The file's content could not be extracted for summarization."
                    
                    summaries.append({
                        "file_id": file_data["file_id"],
                        "file_name": file_data["file_name"],
                        "file_type": file_data["file_type"],
                    "summary": summary_text,
                })
            
            elapsed = time.time() - start_time
            logger.info(
                "File summaries generated (batch mode)",
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

    def _parse_batch_summaries(self, response_text: str, file_data_list: List[Dict[str, Any]]) -> Dict[str, str]:
        """Parse batch LLM response to extract individual file summaries.
        
        Args:
            response_text: Raw LLM response containing multiple file summaries
            file_data_list: List of file data dictionaries with file_id and file_name
            
        Returns:
            Dictionary mapping file_id to summary text
        """
        parsed = {}
        
        if not response_text:
            return parsed
        
        # Create mapping of file names to file IDs for lookup
        name_to_id = {data["file_name"]: data["file_id"] for data in file_data_list}
        
        # Try to parse structured format: "FILE: <name>" followed by "SUMMARY: <text>"
        lines = response_text.split('\n')
        current_file_name = None
        current_summary_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                # Empty line - if we have accumulated summary, save it
                if current_file_name and current_summary_lines:
                    summary_text = ' '.join(current_summary_lines).strip()
                    if summary_text and current_file_name in name_to_id:
                        file_id = name_to_id[current_file_name]
                        # Clean up summary (remove quotes, prefixes, etc.)
                        summary_text = re.sub(r'^(Summary|File|This file|CRITICAL|IMPORTANT):\s*', '', summary_text, flags=re.IGNORECASE).strip()
                        summary_text = summary_text.strip('"\'')
                        summary_text = re.sub(r'\b(CRITICAL|IMPORTANT)\b', '', summary_text, flags=re.IGNORECASE).strip()
                        summary_text = re.sub(r'\s+', ' ', summary_text).strip()
                        if summary_text and len(summary_text) > 10:
                            parsed[file_id] = summary_text
                    current_file_name = None
                    current_summary_lines = []
                continue
            
            # Check for "FILE:" marker
            if line.upper().startswith('FILE:'):
                # Save previous summary if exists
                if current_file_name and current_summary_lines:
                    summary_text = ' '.join(current_summary_lines).strip()
                    if summary_text and current_file_name in name_to_id:
                        file_id = name_to_id[current_file_name]
                        summary_text = re.sub(r'^(Summary|File|This file|CRITICAL|IMPORTANT):\s*', '', summary_text, flags=re.IGNORECASE).strip()
                        summary_text = summary_text.strip('"\'')
                        summary_text = re.sub(r'\b(CRITICAL|IMPORTANT)\b', '', summary_text, flags=re.IGNORECASE).strip()
                        summary_text = re.sub(r'\s+', ' ', summary_text).strip()
                        if summary_text and len(summary_text) > 10:
                            parsed[file_id] = summary_text
                
                # Extract file name (remove "FILE:" prefix and clean)
                file_name_part = line[5:].strip().strip(':').strip()
                # Remove any trailing "CRITICAL" or other noise words
                file_name_part = re.sub(r'\s+(CRITICAL|IMPORTANT)$', '', file_name_part, flags=re.IGNORECASE).strip()
                
                # Try to match against known file names (improved fuzzy match)
                matched = False
                # First try exact match (case-insensitive)
                for known_name in name_to_id.keys():
                    if file_name_part.lower() == known_name.lower():
                        current_file_name = known_name
                        matched = True
                        break
                
                # Then try substring match
                if not matched:
                    for known_name in name_to_id.keys():
                        if file_name_part.lower() in known_name.lower() or known_name.lower() in file_name_part.lower():
                            current_file_name = known_name
                            matched = True
                            break
                
                # Last resort: try to find file name within the line
                if not matched:
                    for known_name in name_to_id.keys():
                        if known_name in file_name_part or file_name_part in known_name:
                            current_file_name = known_name
                            matched = True
                            break
                
                # If still no match, try word-by-word matching
                if not matched and file_name_part:
                    words = file_name_part.split()
                    for word in words:
                        for known_name in name_to_id.keys():
                            if word in known_name or known_name.startswith(word):
                                current_file_name = known_name
                                matched = True
                                break
                        if matched:
                            break
                
                current_summary_lines = []
            
            # Check for "SUMMARY:" marker
            elif line.upper().startswith('SUMMARY:'):
                summary_part = line[8:].strip()
                # Remove "CRITICAL" or other noise from summary
                summary_part = re.sub(r'\b(CRITICAL|IMPORTANT)\b', '', summary_part, flags=re.IGNORECASE).strip()
                summary_part = re.sub(r'\s+', ' ', summary_part).strip()
                if summary_part:
                    current_summary_lines.append(summary_part)
            
            # Accumulate summary lines (content after FILE: or SUMMARY: markers)
            elif current_file_name:
                # Skip lines that are just noise words
                if line.upper() not in ['CRITICAL', 'IMPORTANT', 'TASK:', 'REQUIRED:', 'FORMAT:']:
                    # Remove "CRITICAL" if it appears in the line
                    cleaned_line = re.sub(r'\b(CRITICAL|IMPORTANT)\b', '', line, flags=re.IGNORECASE).strip()
                    cleaned_line = re.sub(r'\s+', ' ', cleaned_line).strip()
                    if cleaned_line:
                        current_summary_lines.append(cleaned_line)
        
        # Handle last summary if response ends without empty line
        if current_file_name and current_summary_lines:
            summary_text = ' '.join(current_summary_lines).strip()
            if summary_text and current_file_name in name_to_id:
                file_id = name_to_id[current_file_name]
                summary_text = re.sub(r'^(Summary|File|This file|CRITICAL|IMPORTANT):\s*', '', summary_text, flags=re.IGNORECASE).strip()
                summary_text = summary_text.strip('"\'')
                summary_text = re.sub(r'\b(CRITICAL|IMPORTANT)\b', '', summary_text, flags=re.IGNORECASE).strip()
                summary_text = re.sub(r'\s+', ' ', summary_text).strip()
                if summary_text and len(summary_text) > 10:
                    parsed[file_id] = summary_text
        
        # If structured parsing didn't work well, try alternative patterns
        # Look for patterns like "filename: summary" or numbered lists
        if len(parsed) < len(file_data_list) * 0.5:  # Less than 50% parsed
            logger.debug(
                "Structured parsing yielded few results, trying alternative patterns",
                parsed_count=len(parsed),
                total_files=len(file_data_list),
            )
            # Alternative: look for file names in text followed by summaries
            for file_data in file_data_list:
                if file_data["file_id"] in parsed:
                    continue  # Already parsed
                
                file_name = file_data["file_name"]
                # Try to find file name in response
                name_pos = response_text.find(file_name)
                if name_pos != -1:
                    # Extract text after file name (next 200 chars or until next file name/newline)
                    excerpt_start = name_pos + len(file_name)
                    excerpt_end = excerpt_start + 300
                    # Find next file name or end of response
                    for other_data in file_data_list:
                        if other_data["file_id"] != file_data["file_id"]:
                            other_pos = response_text.find(other_data["file_name"], excerpt_start)
                            if other_pos != -1 and other_pos < excerpt_end:
                                excerpt_end = other_pos
                    
                    excerpt = response_text[excerpt_start:excerpt_end].strip()
                    # Clean up excerpt
                    excerpt = re.sub(r'^(Summary|File|This file|:|-)\s*', '', excerpt, flags=re.IGNORECASE).strip()
                    # Take first two sentences if available
                    sentences = re.split(r'[.!?]+', excerpt)
                    if len(sentences) >= 2:
                        summary_text = '. '.join(sentences[:2]).strip()
                        if summary_text:
                            summary_text = summary_text.strip('"\'')
                            if len(summary_text) > 20:  # Valid summary
                                parsed[file_data["file_id"]] = summary_text
        
        logger.debug(
            "Parsed batch summaries",
            parsed_count=len(parsed),
            total_files=len(file_data_list),
        )
        
        return parsed

    async def _extract_entity_summary(
        self, 
        folder_id: str,
        file_summaries: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """Extract top entities and themes from folder content using LLM analysis.
        
        Uses LLM to extract meaningful, content-specific themes instead of generic keyword matching.

        Args:
            folder_id: Folder ID
            file_summaries: Optional list of file summaries for better theme extraction

        Returns:
            Dictionary with top_entities and top_themes
        """
        try:
            # Get chunks for entity extraction (for proper nouns, acronyms, etc.)
            chunks = await self.db.get_chunks_by_folder(folder_id, limit=100, include_subfolders=True)
            
            if not chunks:
                return {
                    "top_entities": [],
                    "top_themes": [],
                }
            
            # Extract entities (proper nouns, acronyms, years) using pattern matching
            entity_counts = Counter()
            
            # Common entity patterns
            entity_patterns = [
                r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b',  # Capitalized phrases (potential proper nouns)
                r'\b\d{4}\b',  # Years
                r'\b[A-Z]{2,}\b',  # Acronyms
            ]
            
            for chunk in chunks:
                # Extract entities (capitalized phrases that appear multiple times)
                for pattern in entity_patterns:
                    matches = re.findall(pattern, chunk.content)
                    for match in matches:
                        if len(match) > 2 and (match.isupper() or (match[0].isupper() and len(match.split()) <= 3)):
                            entity_counts[match] += 1
            
            # Get top entities (excluding very common words)
            top_entities = []
            common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'way', 'use', 'man', 'say', 'she', 'use', 'This', 'The', 'That', 'These', 'Those'}
            
            for entity, count in entity_counts.most_common(20):
                entity_lower = entity.lower()
                if count >= 2 and entity_lower not in common_words and len(entity) > 2:
                    top_entities.append({
                        "entity": entity,
                        "count": count,
                    })
            
            # Use LLM to extract meaningful, content-specific themes from file summaries
            top_themes = []
            if file_summaries and len(file_summaries) > 0:
                try:
                    # Build prompt with file summaries
                    file_summary_text = "\n".join([
                        f"- {fs['file_name']}: {fs['summary']}"
                        for fs in file_summaries
                    ])
                    
                    theme_prompt = f"""Analyze the following file summaries and identify 3-5 SPECIFIC themes or topics that appear across these files. 

IMPORTANT: 
- Extract SPECIFIC themes based on actual content (e.g., "machine learning algorithms", "environmental policy", "urban planning")
- DO NOT use generic themes like "research", "technical", "academic", "business", "process"
- Focus on what the files are actually about, not their format or type
- If files have very different themes, list them separately

File Summaries:
{file_summary_text}

Provide themes as a simple comma-separated list of 3-5 specific topics (e.g., "deep reinforcement learning, stock trading strategies, financial algorithms" or "environmental systems, urban planning, data analysis"). 
Themes only (no explanations):"""

                    response = await asyncio.wait_for(
                        self.llm.get_llm().ainvoke(theme_prompt),
                        timeout=30.0
                    )
                    theme_response = response.content.strip() if hasattr(response, "content") else str(response).strip()
                    
                    # Parse themes from response
                    if theme_response:
                        # Split by comma, clean up
                        themes_raw = [t.strip() for t in theme_response.split(',')]
                        # Filter out generic themes
                        generic_themes = {'research', 'technical', 'academic', 'business', 'process', 'analysis', 'study', 'data', 'information', 'content', 'document'}
                        for theme in themes_raw[:5]:  # Limit to 5 themes
                            theme_lower = theme.lower()
                            # Skip if it's a generic theme or too short
                            if theme_lower not in generic_themes and len(theme) > 3:
                                top_themes.append({
                                    "theme": theme.strip(),
                                    "count": 1,  # All themes are equally important
                                })
                except Exception as e:
                    logger.warning(
                        "LLM theme extraction failed, falling back to empty themes",
                        folder_id=folder_id,
                        error=str(e),
                    )
            
            # If no themes extracted, return empty list (better than generic themes)
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

        # Build comprehensive file summary list - include all files with their summaries
        # Ensure the master summary mentions each file at least once
        file_summary_lines = []
        for file_summary in file_summaries:
            file_summary_lines.append(f"- {file_summary['file_name']}: {file_summary['summary']}")
        
        # If we have too many files, include at least the first 15 files with summaries
        if len(file_summary_lines) > 15:
            file_summary_text = "\n".join(file_summary_lines[:15])
            remaining_count = len(file_summary_lines) - 15
            file_summary_text += f"\n... and {remaining_count} more files."
        else:
            file_summary_text = "\n".join(file_summary_lines)
        
        # Build relationship summary text for context
        relationship_text = ""
        if relationship_summary and len(relationship_summary) > 0:
            relationship_lines = []
            for rel in relationship_summary[:5]:  # Top 5 relationships
                relationship_lines.append(
                    f"- {rel.get('source_file', 'File')} and {rel.get('target_file', 'File')} "
                    f"share {rel.get('relationship_type', 'content similarity')} "
                    f"(confidence: {rel.get('confidence', 0):.0%})"
                )
            relationship_text = "\n\nCross-file Relationships:\n" + "\n".join(relationship_lines)
        
        # Build entity/themes summary for context (only if we have meaningful themes)
        themes_text = ""
        if entity_summary and entity_summary.get('top_themes') and len(entity_summary['top_themes']) > 0:
            themes = [t.get('theme', '') for t in entity_summary['top_themes'] if t.get('theme')]
            if themes:  # Only include if we have actual themes (not generic ones)
                themes_text = f"\n\nKey Themes: {', '.join(themes)}"
        
        master_prompt = f"""You are analyzing a folder containing {file_inventory['total_files']} files. Your task is to create a concise, informative summary that synthesizes the ACTUAL CONTENT from these files.

CRITICAL: You must write a summary based ONLY on the file summaries provided below. Do NOT include meta-commentary like "I can help you" or "I notice that there is only one file summary provided". Write the summary directly.

Folder Overview:
- Total Files: {file_inventory['total_files']}
- File Types: {type_summary}
{themes_text}
{relationship_text}

File Summaries (each summarizes actual content from that file):
{file_summary_text}

CRITICAL INSTRUCTIONS - Read carefully:
1. **Use ONLY the actual content described in the file summaries above** - do NOT make generic statements
2. **Avoid repetitive phrases** like:
   - "contains information about"
   - "is related to various topics"
   - "contributes to the folder's theme"
   - "is part of a collection"
   - "appears to be focused on"
3. **Be specific and factual** - reference actual content, topics, or entities mentioned in the file summaries
4. **Synthesize meaningfully** - if files share topics, state WHAT those topics are (don't just say "they share topics")

Write 2-3 concise paragraphs that:
- Paragraph 1: State what this folder actually contains based on the file summaries. If files have DIFFERENT themes/topics, clearly state that (e.g., "This folder contains papers on diverse topics including X, Y, and Z" rather than forcing a unified theme). If they share a theme, be specific about what that theme is.
- Paragraph 2: If files share common topics, describe those specific connections. If files have different focuses, briefly mention the range of topics. Be concrete - cite actual subject matter from the summaries.
- Paragraph 3: Mention any notable individual files if they stand out, or summarize how the files relate (or don't relate) to each other. Don't force connections that don't exist.

EXAMPLE OF GOOD SUMMARY:
"This folder contains research documents about deep reinforcement learning for automated stock trading. The papers focus on modifying DRL algorithms to incorporate short selling and thresholding controls to optimize profit and minimize losses. Several documents discuss the application of turbulence as a safety mechanism during extreme market events. The audio files contain transcribed interviews about related trading strategies."

EXAMPLE OF BAD SUMMARY (avoid this):
"This folder contains a diverse collection of files related to research and technical documentation. The files share common themes and exhibit content similarities. Each file contributes to the folder's overall purpose of research and analysis."

Now write the summary using ONLY the actual content described above:"""

        try:
            # Use timeout to prevent hanging
            response = await asyncio.wait_for(
                self.llm.get_llm().ainvoke(master_prompt),
                timeout=90.0  # Extended timeout (> 1 minute)
            )
            summary = response.content if hasattr(response, "content") else str(response)
            summary_text = summary.strip()
            
            # Check for LLM refusal or meta-commentary in master summary
            refusal_indicators = [
                "i can help you",
                "i notice",
                "however i notice",
                "i'll be happy",
                "if you provide",
                "for now",
                "based on the single",
                "i can help",
                "i notice that there is only",
            ]
            summary_lower = summary_text.lower()
            has_refusal = any(indicator in summary_lower for indicator in refusal_indicators)
            
            if has_refusal:
                logger.warning(
                    "LLM refusal detected in master summary, using fallback",
                    folder_id=folder_id,
                    summary_preview=summary_text[:200],
                )
                # Generate fallback summary from file summaries
                summary_parts = []
                summary_parts.append(f"This folder contains {file_inventory['total_files']} files ({type_summary}).")
                if file_summaries:
                    summary_parts.append("Files include:")
                    for fs in file_summaries[:5]:
                        summary_parts.append(f"- {fs['file_name']}: {fs['summary']}")
                    if len(file_summaries) > 5:
                        summary_parts.append(f"... and {len(file_summaries) - 5} more files.")
                summary_text = " ".join(summary_parts)
            
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

    def _cleanup_summary(self, summary: str) -> str:
        """Post-process summary text to improve readability and remove redundant content.
        
        Args:
            summary: Raw summary text from LLM
            
        Returns:
            Cleaned summary text
        """
        if not summary:
            return summary
        
        text = summary.strip()
        
        # Remove common markdown headers/prefixes that add no value
        prefixes_to_remove = [
            r'^\*\*Folder Summary:\*\*\s*',
            r'^Folder Summary:\s*',
            r'^\*\*Summary:\*\*\s*',
            r'^Summary:\s*',
            r'^##\s+Summary\s*',
            r'^#\s+Summary\s*',
        ]
        for pattern in prefixes_to_remove:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Remove excessive markdown formatting
        # Replace multiple asterisks with nothing (preserve single asterisks for emphasis)
        text = re.sub(r'\*\*\*\*+', '', text)
        
        # Clean up bullet point formatting issues
        # Normalize bullet points (handle both - and *)
        text = re.sub(r'^\s*[-*•]\s+', '* ', text, flags=re.MULTILINE)
        
        # Remove redundant phrases that don't add value
        redundant_phrases = [
            (r'\bUpon examining the contents of the folder, it becomes apparent that\s+', ''),
            (r'\bIt should be noted that\s+', ''),
            (r'\bIt is worth mentioning that\s+', ''),
            (r'\bIn conclusion,\s+', ''),
            (r'\bTo summarize,\s+', ''),
            (r'\bOverall,\s+the\b', 'The'),
            (r'\bOverall,\s+this\b', 'This'),
            (r'\bThe overall purpose of this folder appears to be\b', 'This folder is'),
            (r'\bsuggesting that\s+it may be\b', 'and may be'),
            (r'\bindicating a strong connection\b', 'sharing'),
            (r'\bsuggesting a close relationship\b', 'with'),
        ]
        for pattern, replacement in redundant_phrases:
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Fix common formatting issues
        # Remove extra spaces after periods
        text = re.sub(r'\.\s{2,}', '. ', text)
        # Normalize multiple newlines to double newline max
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove spaces before punctuation
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        # Fix spacing after punctuation
        text = re.sub(r'([.,;:!?])([^\s\n])', r'\1 \2', text)
        
        # Remove similarity score references if they're too verbose
        # This pattern matches "similarity score: X%" and removes the verbose context
        text = re.sub(r'\s*\(similarity score:\s*\d+%\)', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\s*similarity score:\s*\d+%', '', text, flags=re.IGNORECASE)
        # Remove references to "appears to be" when followed by "focused on" or similar
        text = re.sub(r'\bappears to be focused on\b', 'focuses on', text, flags=re.IGNORECASE)
        text = re.sub(r'\bappears to be an?\b', 'is a', text, flags=re.IGNORECASE)
        
        # Clean up paragraph breaks - ensure paragraphs are separated by double newline
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        text = '\n\n'.join(paragraphs)
        
        # Remove leading/trailing whitespace from each line but preserve paragraph structure
        lines = text.split('\n')
        cleaned_lines = []
        for line in lines:
            cleaned_line = line.strip()
            if cleaned_line:  # Only add non-empty lines
                cleaned_lines.append(cleaned_line)
            elif cleaned_lines and cleaned_lines[-1]:  # Preserve paragraph breaks
                cleaned_lines.append('')
        text = '\n'.join(cleaned_lines)
        
        # Remove redundant section headers that add no value
        redundant_sections = [
            r'\*\*Folder Summary:\*\*\s*',
            r'\*\*Cross-file Relationships and Similarities:\*\*\s*',
            r'\*\*Key Relationships and Connections:\*\*\s*',
            r'\*\*Collective Knowledge and Theme:\*\*\s*',
            r'\*\*Common Entities, Methods, or Concepts:\*\*\s*',
            r'Cross-file Relationships and Similarities:\s*',
            r'Key Relationships and Connections:\s*',
            r'Collective Knowledge and Theme:\s*',
            r'Common Entities, Methods, or Concepts:\s*',
        ]
        for pattern in redundant_sections:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Clean up redundant introductory phrases
        # Remove "The individual files within the folder are interconnected through various means:" type phrases
        text = re.sub(r'The individual files within the folder are interconnected through various means:\s*', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Individual files within the folder are interconnected through various means:\s*', '', text, flags=re.IGNORECASE)
        
        # Final cleanup: remove any remaining excessive whitespace
        text = re.sub(r' {2,}', ' ', text)  # Multiple spaces to single space
        text = text.strip()
        
        # Remove any remaining empty paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 10]
        text = '\n\n'.join(paragraphs)
        
        return text

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
                        "type": "building_graph",
                        "folder_id": folder_id,
                        "message": "Building knowledge graph in background...",
                    },
                )
            except Exception as e:
                logger.warning("Failed to send building_graph WebSocket message", error=str(e))
            
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
            # Check if it's a database initialization error
            from app.core.exceptions import DatabaseError
            if isinstance(e, DatabaseError) and "not initialized" in str(e):
                logger.warning(
                    "Database not initialized for background graph construction, skipping",
                    folder_id=folder_id,
                )
                return
            
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
