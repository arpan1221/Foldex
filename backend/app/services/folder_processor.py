"""Main orchestrator for folder processing."""

from typing import List, Optional
import structlog

from app.core.exceptions import ProcessingError
from app.services.google_drive import GoogleDriveService
from app.services.document_processor import DocumentProcessor
from app.services.knowledge_graph import KnowledgeGraphService
from app.database.sqlite_manager import SQLiteManager
from app.database.vector_store import VectorStore
from app.api.v1.websocket import manager

logger = structlog.get_logger(__name__)


class FolderProcessor:
    """Orchestrates folder processing workflow."""

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

    async def process_folder_async(
        self, folder_id: str, user_id: str
    ) -> None:
        """Process folder asynchronously with progress updates.

        Args:
            folder_id: Google Drive folder ID
            user_id: User ID
        """
        try:
            logger.info("Starting folder processing", folder_id=folder_id, user_id=user_id)

            # Notify via WebSocket
            await manager.send_message(
                folder_id,
                {"type": "processing_started", "folder_id": folder_id},
            )

            # Download files from Google Drive
            files = await self.drive_service.list_folder_files(folder_id, user_id)

            # Process each file
            processed_count = 0
            for file_info in files:
                try:
                    await self._process_file(file_info, folder_id, user_id)
                    processed_count += 1

                    # Send progress update
                    await manager.send_message(
                        folder_id,
                        {
                            "type": "file_processed",
                            "file_id": file_info["id"],
                            "file_name": file_info["name"],
                            "progress": processed_count / len(files),
                        },
                    )
                except Exception as e:
                    logger.error(
                        "File processing failed",
                        file_id=file_info["id"],
                        error=str(e),
                    )

            # Build knowledge graph
            await self.kg_service.build_graph(folder_id)

            # Send completion notification
            await manager.send_message(
                folder_id,
                {
                    "type": "processing_complete",
                    "folder_id": folder_id,
                    "files_processed": processed_count,
                },
            )

            logger.info(
                "Folder processing completed",
                folder_id=folder_id,
                files_processed=processed_count,
            )
        except Exception as e:
            logger.error("Folder processing failed", folder_id=folder_id, error=str(e))
            await manager.send_message(
                folder_id,
                {"type": "processing_error", "error": str(e)},
            )
            raise ProcessingError(f"Failed to process folder: {str(e)}")

    async def _process_file(
        self, file_info: dict, folder_id: str, user_id: str
    ) -> None:
        """Process a single file.

        Args:
            file_info: File metadata from Google Drive
            folder_id: Parent folder ID
            user_id: User ID
        """
        # Download file
        file_path = await self.drive_service.download_file(
            file_info["id"], user_id
        )

        # Process document
        chunks = await self.doc_processor.process_file(file_path, file_info)

        # Store in database
        await self.db.store_file_metadata(file_info, folder_id)
        await self.db.store_chunks(chunks)

        # Store embeddings
        await self.vector_store.add_chunks(chunks)

