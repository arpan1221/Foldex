"""Main orchestrator for folder processing with WebSocket updates and error recovery."""

from typing import List, Optional, Dict, Any
import asyncio
import structlog
from datetime import datetime

from app.core.exceptions import ProcessingError, AuthenticationError
from app.services.google_drive import GoogleDriveService
from app.services.document_processor import DocumentProcessor
from app.services.knowledge_graph import KnowledgeGraphService
from app.database.sqlite_manager import SQLiteManager
from app.rag.vector_store import LangChainVectorStore
from app.api.v1.websocket import manager
from app.utils.cache import FileCache, Cache

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


class ProcessingStatus:
    """Processing status tracking."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class FolderProcessor:
    """Orchestrates folder processing workflow with progress tracking and error recovery."""

    def __init__(
        self,
        drive_service: Optional[GoogleDriveService] = None,
        doc_processor: Optional[DocumentProcessor] = None,
        kg_service: Optional[KnowledgeGraphService] = None,
        db: Optional[SQLiteManager] = None,
        vector_store: Optional[LangChainVectorStore] = None,
    ):
        """Initialize folder processor.

        Args:
            drive_service: Google Drive service
            doc_processor: Document processor
            kg_service: Knowledge graph service
            db: Database manager
            vector_store: LangChain vector store manager
        """
        self.drive_service = drive_service or GoogleDriveService()
        self.doc_processor = doc_processor or DocumentProcessor()
        self.kg_service = kg_service or KnowledgeGraphService()
        self.db = db or SQLiteManager()
        self.vector_store = vector_store or LangChainVectorStore()
        self.file_cache = FileCache()
        self.status_cache = Cache()
        self._processing_tasks: Dict[str, asyncio.Task] = {}

    async def process_folder_async(
        self, folder_id: str, user_id: str, google_token: Optional[str] = None
    ) -> None:
        """Process folder asynchronously with progress updates and error recovery.

        Args:
            folder_id: Google Drive folder ID
            user_id: User ID
            google_token: Google OAuth2 access token

        Raises:
            ProcessingError: If processing fails
            AuthenticationError: If authentication fails
        """
        task_key = f"{folder_id}_{user_id}"
        
        # Check if already processing
        if task_key in self._processing_tasks:
            logger.warning("Folder already being processed", folder_id=folder_id)
            return

        try:
            logger.info("Starting folder processing", folder_id=folder_id, user_id=user_id)

            # Update status in database
            await self._update_processing_status(folder_id, ProcessingStatus.PROCESSING, user_id)

            # Notify via WebSocket
            await manager.send_message(
                folder_id,
                {
                    "type": "processing_started",
                    "folder_id": folder_id,
                    "message": "Starting folder processing...",
                    "progress": 0.0,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

            # Validate token
            if not google_token:
                raise AuthenticationError("Google access token required for folder processing")

            # Get folder metadata (name) from Google Drive
            try:
                folder_metadata = await self.drive_service.get_file_metadata(folder_id, google_token)
                folder_name = folder_metadata.get("name", f"Folder {folder_id[:8]}")
            except Exception as e:
                logger.warning("Failed to get folder name, using default", folder_id=folder_id, error=str(e))
                folder_name = f"Folder {folder_id[:8]}"

            # Store folder in database with explicit root_folder_id
            await self.db.store_folder(
                folder_id=folder_id,
                user_id=user_id,
                folder_name=folder_name,
                file_count=0,
                status="processing",
                parent_folder_id=None,  # Explicitly set to None for root
                root_folder_id=folder_id,  # Explicitly set root_folder_id to itself
            )

            # List files from Google Drive (recursively including all subfolders)
            # Create a token refresh callback to refresh token during long operations
            async def refresh_token_callback():
                """Refresh Google token if needed during folder traversal."""
                from app.api.v1.folders import get_user_google_token
                try:
                    new_token = await get_user_google_token(user_id, None)
                    logger.debug("Token refreshed during folder traversal", user_id=user_id)
                    return new_token
                except Exception as e:
                    logger.warning("Failed to refresh token during traversal", error=str(e))
                    return google_token  # Return original token if refresh fails
            
            # Get hierarchical folder structure
            folder_structure = await self.drive_service.list_folder_files(
                folder_id, google_token, user_id, recursive=True
            )
            # Handle both dict and list return types
            if isinstance(folder_structure, dict):
                files = folder_structure.get("files", [])
                subfolders = folder_structure.get("subfolders", [])
            else:
                # Legacy list format
                files = folder_structure if isinstance(folder_structure, list) else []
                subfolders = []
            total_files = len(files)
            
            # Build folder path map for files
            folder_path_map = self._build_folder_path_map(subfolders, folder_name)
            
            # Add path information to files
            for file_info in files:
                parent_folder_id = file_info.get("parent_folder_id", folder_id)
                if parent_folder_id in folder_path_map:
                    file_info["path"] = f"{folder_path_map[parent_folder_id]}/{file_info.get('name', 'Unknown')}"
                else:
                    file_info["path"] = file_info.get("name", "Unknown")
            
            logger.info(
                "Files collected from folder",
                folder_id=folder_id,
                total_files=total_files,
                total_subfolders=len(subfolders),
                recursive=True
            )
            
            # Send folder structure information
            await manager.send_message(
                folder_id,
                {
                    "type": "folder_structure",
                    "folder_id": folder_id,
                    "folder_name": folder_name,
                    "total_files": total_files,
                    "total_subfolders": len(subfolders),
                    "subfolders": self._flatten_subfolders(subfolders),
                    "message": f"Found {total_files} files in {len(subfolders)} subfolders",
                },
            )
            
            # Store subfolders in database with hierarchy
            await self._store_folder_hierarchy(folder_id, user_id, subfolders)

            if total_files == 0:
                await manager.send_message(
                    folder_id,
                    {
                        "type": "processing_complete",
                        "folder_id": folder_id,
                        "message": "No files found in folder",
                        "files_processed": 0,
                        "total_files": 0,
                        "progress": 1.0,
                    },
                )
                await self._update_processing_status(folder_id, ProcessingStatus.COMPLETED, user_id)
                # Update folder record
                await self.db.store_folder(
                    folder_id=folder_id,
                    user_id=user_id,
                    folder_name=folder_name,
                    file_count=0,
                    status="completed",
                    parent_folder_id=None,  # Explicitly set to None for root
                    root_folder_id=folder_id,  # Explicitly set root_folder_id to itself
                )
                return

            await manager.send_message(
                folder_id,
                {
                    "type": "files_detected",
                    "folder_id": folder_id,
                    "total_files": total_files,
                    "message": f"Found {total_files} files to process",
                },
            )

            # Process files concurrently with intelligent limits
            # Different limits for different file types:
            # - Small text files: 10 concurrent
            # - PDFs/Documents: 5 concurrent
            # - Audio/Video: 2 concurrent (larger files)
            
            # Categorize files by type for optimized processing
            small_files = []  # Text, markdown, small docs
            medium_files = []  # PDFs, Office docs
            large_files = []  # Audio, video, large files
            
            for file_info in files:
                mime_type = file_info.get("mimeType", "")
                file_size_raw = file_info.get("size", 0)
                
                # Convert size to int (Google Drive API sometimes returns strings)
                try:
                    file_size = int(file_size_raw) if file_size_raw else 0
                except (ValueError, TypeError):
                    file_size = 0
                    logger.warning(
                        "Invalid file size, defaulting to 0",
                        file_id=file_info.get("id"),
                        size_value=file_size_raw,
                    )
                
                # Categorize by MIME type and size
                if mime_type.startswith(("text/", "application/json")) or file_size < 100_000:  # < 100KB
                    small_files.append(file_info)
                elif mime_type.startswith(("audio/", "video/")) or file_size > 10_000_000:  # > 10MB
                    large_files.append(file_info)
                else:
                    medium_files.append(file_info)
            
            logger.info(
                "Files categorized for processing",
                small_files=len(small_files),
                medium_files=len(medium_files),
                large_files=len(large_files),
            )
            
            # Create semaphores for each category
            small_semaphore = asyncio.Semaphore(10)  # 10 concurrent small files
            medium_semaphore = asyncio.Semaphore(5)  # 5 concurrent medium files
            large_semaphore = asyncio.Semaphore(2)   # 2 concurrent large files
            
            processed_count = 0
            failed_files: List[Dict[str, Any]] = []
            processing_lock = asyncio.Lock()

            async def process_single_file(file_info: Dict[str, Any], file_index: int, semaphore: asyncio.Semaphore) -> None:
                """Process a single file with progress tracking.
                
                Args:
                    file_info: File metadata
                    file_index: Index in the file list
                    semaphore: Semaphore to control concurrency for this file type
                """
                nonlocal processed_count
                file_id = file_info.get("id")
                file_name = file_info.get("name", "Unknown")
                file_parent_folder_id = file_info.get("parent_folder_id", folder_id)
                file_path = file_info.get("path", file_name)  # Relative path in folder hierarchy
                
                async with semaphore:
                    try:
                        # Get current progress count (lock only for reading)
                        async with processing_lock:
                            current_count = processed_count

                        # Send file start notification (no lock - WebSocket is thread-safe)
                        await manager.send_message(
                            folder_id,
                            {
                                "type": "file_processing",
                                "folder_id": folder_id,
                                "file_id": file_id,
                                "file_name": file_name,
                                "file_path": file_path,
                                "parent_folder_id": file_parent_folder_id,
                                "file_index": file_index + 1,
                                "files_processed": current_count,  # Files completed so far
                                "total_files": total_files,
                                "message": f"Processing {file_path}...",
                            },
                        )

                        # Process file with retry logic
                        await self._process_file_with_retry(
                            file_info, file_parent_folder_id, user_id, google_token, max_retries=2
                        )

                        # Update count (lock only for increment)
                        async with processing_lock:
                            processed_count += 1
                            current_count = processed_count
                            progress = processed_count / total_files if total_files > 0 else 0

                        # Send progress update (no lock - WebSocket is thread-safe)
                        await manager.send_message(
                            folder_id,
                            {
                                "type": "file_processed",
                                "folder_id": folder_id,
                                "file_id": file_id,
                                "file_name": file_name,
                                "file_path": file_path,
                                "parent_folder_id": file_parent_folder_id,
                                "files_processed": current_count,
                                "total_files": total_files,
                                "progress": progress,
                                "message": f"Processed {file_path} ({current_count}/{total_files})",
                            },
                        )

                    except Exception as e:
                        logger.error(
                            "File processing failed",
                            file_id=file_id,
                            file_name=file_name,
                            error=str(e),
                            exc_info=True,
                        )

                        # Update count and failed files list (lock for shared state)
                        async with processing_lock:
                            processed_count += 1
                            current_count = processed_count

                            failed_files.append({
                                "file_id": file_id,
                                "file_name": file_name,
                                "file_path": file_path,
                                "error": str(e),
                            })

                        # Send error notification (no lock - WebSocket is thread-safe)
                        await manager.send_message(
                            folder_id,
                            {
                                "type": "file_error",
                                "folder_id": folder_id,
                                "file_id": file_id,
                                "file_name": file_name,
                                "file_path": file_path,
                                "parent_folder_id": file_parent_folder_id,
                                "files_processed": current_count,  # Include failed files in count
                                "total_files": total_files,
                                "error": str(e),
                                "message": f"Failed to process {file_path}: {str(e)}",
                            },
                        )

            # Process all files concurrently with category-specific semaphores
            # Process in order: small files first (fastest), then medium, then large
            # This provides better perceived performance
            tasks = []
            
            # Small files (fast processing)
            for index, file_info in enumerate(small_files):
                tasks.append(process_single_file(file_info, index, small_semaphore))
            
            # Medium files
            offset = len(small_files)
            for index, file_info in enumerate(medium_files):
                tasks.append(process_single_file(file_info, offset + index, medium_semaphore))
            
            # Large files (slowest)
            offset = len(small_files) + len(medium_files)
            for index, file_info in enumerate(large_files):
                tasks.append(process_single_file(file_info, offset + index, large_semaphore))
            
            logger.info(
                "Starting concurrent file processing",
                total_tasks=len(tasks),
                small_concurrent=10,
                medium_concurrent=5,
                large_concurrent=2,
            )
            
            # Gather all tasks - exceptions are returned as results, not raised
            # This allows other files to continue processing even if one fails
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Log any exceptions that occurred (they're already handled in process_single_file)
            for idx, result in enumerate(results):
                if isinstance(result, Exception):
                    # This shouldn't happen since process_single_file catches all exceptions
                    # But log it just in case
                    logger.warning(
                        "Unexpected exception in file processing task",
                        task_index=idx,
                        error_type=type(result).__name__,
                        error=str(result),
                    )

            # Send completion notification IMMEDIATELY after chunking
            # This allows users to start chatting right away
            status_message = f"Processing completed: {processed_count}/{total_files} files processed"
            if failed_files:
                status_message += f", {len(failed_files)} failed"

            completion_message = {
                "type": "processing_complete",
                "folder_id": folder_id,
                "files_processed": processed_count,
                "total_files": total_files,
                "failed_files": len(failed_files),
                "progress": 1.0,
                "message": status_message,
                "timestamp": datetime.utcnow().isoformat(),
            }
            
            # Send completion message multiple times to ensure delivery
            for attempt in range(3):
                try:
                    await manager.send_message(folder_id, completion_message)
                    logger.info(
                        "Completion message sent",
                        folder_id=folder_id,
                        attempt=attempt + 1,
                    )
                    break
                except Exception as e:
                    if attempt == 2:
                        logger.warning(
                            "Failed to send completion message after 3 attempts",
                            folder_id=folder_id,
                            error=str(e),
                        )
                    await asyncio.sleep(0.1)

            # Chunking complete - file-specific chat is now available
            # Summarization and knowledge graph building are now user-initiated actions
            # No automatic background tasks - users control when to run these

            # Update status in database
            await self._update_processing_status(
                folder_id,
                ProcessingStatus.COMPLETED if processed_count > 0 else ProcessingStatus.FAILED,
                user_id,
                {
                    "files_processed": processed_count,
                    "total_files": total_files,
                    "failed_files": failed_files,
                },
            )

            # Update folder record with final file count
            await self.db.store_folder(
                folder_id=folder_id,
                user_id=user_id,
                folder_name=folder_name,
                file_count=processed_count,
                status="completed" if processed_count > 0 else "failed",
                parent_folder_id=None,  # Explicitly set to None for root
                root_folder_id=folder_id,  # Explicitly set root_folder_id to itself
            )

            logger.info(
                "Folder processing completed",
                folder_id=folder_id,
                files_processed=processed_count,
                total_files=total_files,
                failed_count=len(failed_files),
            )

        except AuthenticationError:
            await self._update_processing_status(folder_id, ProcessingStatus.FAILED, user_id)
            await manager.send_message(
                folder_id,
                {
                    "type": "processing_error",
                    "folder_id": folder_id,
                    "error": "Authentication failed",
                    "message": "Google access token expired or invalid",
                },
            )
            raise
        except Exception as e:
            logger.error("Folder processing failed", folder_id=folder_id, error=str(e), exc_info=True)
            
            await self._update_processing_status(folder_id, ProcessingStatus.FAILED, user_id)
            
            await manager.send_message(
                folder_id,
                {
                    "type": "processing_error",
                    "folder_id": folder_id,
                    "error": str(e),
                    "message": f"Processing failed: {str(e)}",
                },
            )
            raise ProcessingError(f"Failed to process folder: {str(e)}") from e
        finally:
            # Clean up task tracking
            if task_key in self._processing_tasks:
                del self._processing_tasks[task_key]

    async def _process_file_with_retry(
        self,
        file_info: Dict[str, Any],
        folder_id: str,
        user_id: str,
        google_token: str,
        max_retries: int = 2,
    ) -> None:
        """Process a file with retry logic.

        Args:
            file_info: File metadata from Google Drive
            folder_id: Parent folder ID
            user_id: User ID
            google_token: Google OAuth2 access token
            max_retries: Maximum number of retry attempts

        Raises:
            ProcessingError: If processing fails after retries
        """
        file_id = file_info.get("id")
        mime_type = file_info.get("mimeType")

        if not file_id:
            raise ProcessingError("File ID is required")
        
        if not mime_type:
            raise ProcessingError("MIME type is required")

        # Check if file type is supported
        if not self.drive_service.is_supported_file_type(mime_type):
            logger.info("Skipping unsupported file type", file_id=file_id, mime_type=mime_type)
            return

        last_error = None
        for attempt in range(max_retries + 1):
            try:
                # Download file (with caching)
                file_path = await self.drive_service.download_file(
                    file_id, google_token, user_id, mime_type
                )

                # Process document - no progress callback to avoid blocking WebSocket
                # Progress is already tracked at file-level (file_processing/file_processed messages)
                chunks = await self.doc_processor.process_file(
                    file_path, file_info, progress_callback=None
                )

                # Store in database
                await self.db.store_file_metadata(file_info, folder_id)
                await self.db.store_chunks(chunks)

                # Convert chunks to LangChain Documents and store in vector store
                # Run vector store operations in background to avoid blocking progress updates
                if LANGCHAIN_AVAILABLE and chunks:
                    documents = []
                    for chunk in chunks:
                        # Create LangChain Document with metadata
                        doc = Document(
                            page_content=chunk.content,
                            metadata={
                                "chunk_id": chunk.chunk_id,
                                "file_id": chunk.file_id,
                                "folder_id": folder_id,
                                **chunk.metadata
                            }
                        )
                        documents.append(doc)
                    
                    # Add documents to vector store in background (non-blocking)
                    # This allows progress updates to continue immediately
                    async def add_to_vector_store():
                        try:
                            await self.vector_store.add_documents(documents)
                            logger.debug(
                                "Added documents to vector store",
                                file_id=file_id,
                                document_count=len(documents)
                            )
                        except Exception as e:
                            # Log error but don't crash - vector store is optional for progress
                            logger.error(
                                "Failed to add documents to vector store (non-fatal)",
                                file_id=file_id,
                                error=str(e),
                                exc_info=True,
                            )
                        except BaseException as e:
                            # Catch all exceptions including asyncio.CancelledError
                            logger.warning(
                                "Vector store background task cancelled or failed",
                                file_id=file_id,
                                error_type=type(e).__name__,
                            )
                    
                    # Create background task (don't await - let it run in background)
                    # Errors are handled internally and won't crash the main process
                    asyncio.create_task(add_to_vector_store())

                # Success - return immediately (vector store operations run in background)
                return

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    logger.warning(
                        "File processing attempt failed, retrying",
                        file_id=file_id,
                        attempt=attempt + 1,
                        error=str(e),
                    )
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(
                        "File processing failed after retries",
                        file_id=file_id,
                        attempts=attempt + 1,
                        error=str(e),
                    )

        raise ProcessingError(f"Failed to process file after {max_retries + 1} attempts: {str(last_error)}")

    async def _process_file(
        self, file_info: dict, folder_id: str, user_id: str, google_token: str
    ) -> None:
        """Process a single file (legacy method, use _process_file_with_retry).

        Args:
            file_info: File metadata from Google Drive
            folder_id: Parent folder ID
            user_id: User ID
            google_token: Google OAuth2 access token
        """
        await self._process_file_with_retry(file_info, folder_id, user_id, google_token)

    async def _update_processing_status(
        self,
        folder_id: str,
        status: str,
        user_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Update processing status in database.

        Args:
            folder_id: Folder ID
            status: Processing status
            user_id: User ID
            metadata: Optional status metadata
        """
        try:
            # Actually update the database status
            await self.db.update_folder_status(
                folder_id=folder_id,
                status=status,
            )
            
            logger.info(
                "Processing status updated",
                folder_id=folder_id,
                status=status,
                user_id=user_id,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning("Failed to update processing status", folder_id=folder_id, error=str(e))

    async def _store_folder_hierarchy(
        self, root_folder_id: str, user_id: str, subfolders: List[Dict[str, Any]]
    ) -> None:
        """Store folder hierarchy in database.

        Args:
            root_folder_id: Root folder ID
            user_id: User ID
            subfolders: List of subfolder metadata with nested structure
        """
        async def store_subfolder_recursive(subfolder_data: Dict[str, Any]) -> None:
            """Recursively store subfolder and its children."""
            folder_id = subfolder_data.get("folder_id")
            folder_name = subfolder_data.get("folder_name", "Unknown")
            parent_folder_id = subfolder_data.get("parent_folder_id")
            file_count = subfolder_data.get("file_count", 0)
            
            if not folder_id:
                logger.warning("Skipping subfolder without folder_id", subfolder_data=subfolder_data)
                return
            
            # CRITICAL: Prevent root folder from being stored as a subfolder
            if folder_id == root_folder_id:
                logger.warning(
                    "Attempted to store root folder as subfolder, skipping",
                    folder_id=folder_id,
                    root_folder_id=root_folder_id
                )
                return
            
            # Store this subfolder
            await self.db.store_folder(
                folder_id=folder_id,
                user_id=user_id,
                folder_name=folder_name,
                file_count=file_count,
                status="completed",  # Subfolders are considered completed when stored
                parent_folder_id=parent_folder_id,
                root_folder_id=root_folder_id,  # Always set to root
            )
            
            # Recursively store nested subfolders
            nested_subfolders = subfolder_data.get("subfolders", [])
            for nested_subfolder in nested_subfolders:
                await store_subfolder_recursive(nested_subfolder)
        
        # Store all subfolders recursively
        for subfolder in subfolders:
            try:
                # Double-check: skip if this is the root folder
                if subfolder.get("folder_id") == root_folder_id:
                    logger.warning(
                        "Skipping root folder in subfolders list",
                        root_folder_id=root_folder_id
                    )
                    continue
                    
                await store_subfolder_recursive(subfolder)
                logger.debug(
                    "Stored subfolder hierarchy",
                    subfolder_id=subfolder.get("folder_id"),
                    subfolder_name=subfolder.get("folder_name")
                )
            except Exception as e:
                logger.warning(
                    "Failed to store subfolder",
                    subfolder_id=subfolder.get("folder_id"),
                    error=str(e)
                )
                # Continue storing other subfolders even if one fails
                continue

    def _build_folder_path_map(
        self, subfolders: List[Dict[str, Any]], root_folder_name: str
    ) -> Dict[str, str]:
        """Build a map of folder_id -> folder_path for quick lookup.
        
        Args:
            subfolders: List of subfolder metadata with nested structure
            root_folder_name: Name of the root folder
            
        Returns:
            Dictionary mapping folder_id to folder path
        """
        folder_map = {}
        
        def process_subfolder(subfolder: Dict[str, Any], parent_path: str = "") -> None:
            """Recursively process subfolders to build path map."""
            folder_id = subfolder.get("folder_id")
            folder_name = subfolder.get("folder_name", "Unknown")
            
            if parent_path:
                current_path = f"{parent_path}/{folder_name}"
            else:
                current_path = f"{root_folder_name}/{folder_name}"
            
            if folder_id:
                folder_map[folder_id] = current_path
            
            # Process nested subfolders
            nested_subfolders = subfolder.get("subfolders", [])
            for nested_subfolder in nested_subfolders:
                process_subfolder(nested_subfolder, current_path)
        
        for subfolder in subfolders:
            process_subfolder(subfolder)
        
        return folder_map
    
    def _flatten_subfolders(
        self, subfolders: List[Dict[str, Any]], parent_path: str = ""
    ) -> List[Dict[str, Any]]:
        """Flatten subfolder hierarchy for WebSocket messages.
        
        Args:
            subfolders: List of subfolder metadata with nested structure
            parent_path: Parent folder path
            
        Returns:
            Flattened list of subfolder information
        """
        flattened = []
        
        for subfolder in subfolders:
            folder_id = subfolder.get("folder_id")
            folder_name = subfolder.get("folder_name", "Unknown")
            current_path = f"{parent_path}/{folder_name}" if parent_path else folder_name
            
            flattened.append({
                "folder_id": folder_id,
                "folder_name": folder_name,
                "folder_path": current_path,
                "parent_folder_id": subfolder.get("parent_folder_id"),
                "file_count": subfolder.get("file_count", 0),
                "subfolder_count": subfolder.get("subfolder_count", 0),
            })
            
            # Recursively flatten nested subfolders
            nested_subfolders = subfolder.get("subfolders", [])
            if nested_subfolders:
                flattened.extend(self._flatten_subfolders(nested_subfolders, current_path))
        
        return flattened

    async def _background_learning_task(self, folder_id: str, user_id: str) -> None:
        """Background task for knowledge graph building and folder summarization.
        
        This runs asynchronously after chunking completes, allowing users to
        start chatting immediately while learning happens in the background.
        
        Args:
            folder_id: Folder ID
            user_id: User ID
        """
        try:
            # Optional: Build knowledge graph (can be disabled for speed)
            # Skip for now to prioritize speed - can be enabled later if needed
            # if self.kg_service:
            #     try:
            #         await manager.send_message(
            #             folder_id,
            #             {
            #                 "type": "building_graph",
            #                 "folder_id": folder_id,
            #                 "message": "Building knowledge graph...",
            #             },
            #         )
            #         await self.kg_service.build_graph(folder_id)
            #         await manager.send_message(
            #             folder_id,
            #             {
            #                 "type": "graph_complete",
            #                 "folder_id": folder_id,
            #                 "message": "Knowledge graph built successfully",
            #             },
            #         )
            #     except Exception as e:
            #         logger.error("Knowledge graph build failed", folder_id=folder_id, error=str(e))

            # Generate folder summary (POST-INGESTION SUMMARIZATION)
            try:
                from app.services.folder_summarizer import get_folder_summarizer

                logger.info("Starting background folder summarization", folder_id=folder_id)

                summarizer = get_folder_summarizer()
                summary_data = await summarizer.generate_folder_summary(
                    folder_id=folder_id,
                    send_progress=True  # WebSocket updates enabled
                )

                logger.info(
                    "Background folder summarization completed",
                    folder_id=folder_id,
                    capabilities_count=len(summary_data.get("capabilities", [])),
                )

            except Exception as e:
                logger.error(
                    "Background folder summarization failed",
                    folder_id=folder_id,
                    error=str(e),
                    exc_info=True,
                )

        except Exception as e:
            logger.error(
                "Background learning task failed",
                folder_id=folder_id,
                error=str(e),
                exc_info=True,
            )

    def cancel_processing(self, folder_id: str, user_id: str) -> bool:
        """Cancel ongoing processing for a folder.

        Args:
            folder_id: Folder ID
            user_id: User ID

        Returns:
            True if cancellation was successful
        """
        task_key = f"{folder_id}_{user_id}"
        if task_key in self._processing_tasks:
            task = self._processing_tasks[task_key]
            if not task.done():
                task.cancel()
                logger.info("Processing cancelled", folder_id=folder_id, user_id=user_id)
                return True
        return False
