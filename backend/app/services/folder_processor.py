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
from app.database.vector_store import VectorStore
from app.api.v1.websocket import manager
from app.utils.cache import FileCache, Cache

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
        vector_store: Optional[VectorStore] = None,
    ):
        """Initialize folder processor.

        Args:
            drive_service: Google Drive service
            doc_processor: Document processor
            kg_service: Knowledge graph service
            db: Database manager
            vector_store: Vector store manager
        """
        self.drive_service = drive_service or GoogleDriveService()
        self.doc_processor = doc_processor or DocumentProcessor()
        self.kg_service = kg_service or KnowledgeGraphService()
        self.db = db or SQLiteManager()
        self.vector_store = vector_store or VectorStore()
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

            # List files from Google Drive
            files = await self.drive_service.list_folder_files(folder_id, google_token, user_id)
            total_files = len(files)

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

            # Process files with error recovery
            processed_count = 0
            failed_files: List[Dict[str, Any]] = []

            for file_index, file_info in enumerate(files):
                try:
                    file_id = file_info.get("id")
                    file_name = file_info.get("name", "Unknown")

                    # Send file start notification
                    await manager.send_message(
                        folder_id,
                        {
                            "type": "file_processing",
                            "folder_id": folder_id,
                            "file_id": file_id,
                            "file_name": file_name,
                            "file_index": file_index + 1,
                            "total_files": total_files,
                            "message": f"Processing {file_name}...",
                        },
                    )

                    # Process file with retry logic
                    await self._process_file_with_retry(
                        file_info, folder_id, user_id, google_token, max_retries=2
                    )

                    processed_count += 1

                    # Send progress update
                    progress = processed_count / total_files if total_files > 0 else 0
                    await manager.send_message(
                        folder_id,
                        {
                            "type": "file_processed",
                            "folder_id": folder_id,
                            "file_id": file_id,
                            "file_name": file_name,
                            "files_processed": processed_count,
                            "total_files": total_files,
                            "progress": progress,
                            "message": f"Processed {file_name} ({processed_count}/{total_files})",
                        },
                    )

                except Exception as e:
                    logger.error(
                        "File processing failed",
                        file_id=file_info.get("id"),
                        file_name=file_info.get("name"),
                        error=str(e),
                        exc_info=True,
                    )

                    failed_files.append({
                        "file_id": file_info.get("id"),
                        "file_name": file_info.get("name"),
                        "error": str(e),
                    })

                    # Send error notification but continue processing
                    await manager.send_message(
                        folder_id,
                        {
                            "type": "file_error",
                            "folder_id": folder_id,
                            "file_id": file_info.get("id"),
                            "file_name": file_info.get("name"),
                            "error": str(e),
                            "message": f"Failed to process {file_info.get('name')}: {str(e)}",
                        },
                    )

            # Build knowledge graph if files were processed
            if processed_count > 0:
                try:
                    await manager.send_message(
                        folder_id,
                        {
                            "type": "building_graph",
                            "folder_id": folder_id,
                            "message": "Building knowledge graph...",
                        },
                    )

                    await self.kg_service.build_graph(folder_id)

                    await manager.send_message(
                        folder_id,
                        {
                            "type": "graph_complete",
                            "folder_id": folder_id,
                            "message": "Knowledge graph built successfully",
                        },
                    )
                except Exception as e:
                    logger.error("Knowledge graph build failed", folder_id=folder_id, error=str(e))
                    # Don't fail entire processing if graph build fails

            # Send completion notification
            status_message = f"Processing completed: {processed_count}/{total_files} files processed"
            if failed_files:
                status_message += f", {len(failed_files)} failed"

            await manager.send_message(
                folder_id,
                {
                    "type": "processing_complete",
                    "folder_id": folder_id,
                    "files_processed": processed_count,
                    "total_files": total_files,
                    "failed_files": len(failed_files),
                    "progress": 1.0,
                    "message": status_message,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            )

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

                # Process document with progress callback
                def progress_callback(progress: float):
                    # File-level progress can be sent here if needed
                    pass

                chunks = await self.doc_processor.process_file(
                    file_path, file_info, progress_callback=progress_callback
                )

                # Store in database
                await self.db.store_file_metadata(file_info, folder_id)
                await self.db.store_chunks(chunks)

                # Store embeddings
                await self.vector_store.add_chunks(chunks)

                # Success - return
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
            # TODO: Implement status storage in database
            # For now, just log it
            logger.info(
                "Processing status updated",
                folder_id=folder_id,
                status=status,
                user_id=user_id,
                metadata=metadata,
            )
        except Exception as e:
            logger.warning("Failed to update processing status", folder_id=folder_id, error=str(e))

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
