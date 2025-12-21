"""API endpoints for folder summary/knowledge base."""

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any
import structlog
import traceback

from app.database.sqlite_manager import SQLiteManager
from app.services.folder_summarizer import get_folder_summarizer
from app.core.exceptions import DatabaseError
from app.config.settings import settings
from sqlalchemy import select
from app.models.database import FolderRecord

logger = structlog.get_logger(__name__)
router = APIRouter(prefix="/folders", tags=["folder-summary"])


def get_db() -> SQLiteManager:
    """Get database manager dependency."""
    return SQLiteManager()


@router.get("/{folder_id}/summary")
async def get_folder_summary(
    folder_id: str,
    db: SQLiteManager = Depends(get_db),
) -> Dict[str, Any]:
    """Get folder summary and learning status.

    Args:
        folder_id: Folder ID
        db: Database manager

    Returns:
        Folder summary data

    Raises:
        HTTPException: If folder not found or retrieval fails

    Example:
        >>> GET /api/v1/folders/abc123/summary
        {
            "folder_id": "abc123",
            "folder_name": "Research Papers",
            "summary": "This folder contains...",
            "learning_status": "learning_complete",
            "capabilities": [...],
            ...
        }
    """
    try:
        # Get summary
        summary_data = await db.get_folder_summary(folder_id)

        if not summary_data:
            # Return pending status if no summary exists yet
            return JSONResponse(
                status_code=status.HTTP_200_OK,
                content={
                    "folder_id": folder_id,
                    "folder_name": None,
                    "summary": None,
                    "learning_status": "learning_pending",
                    "insights": None,
                    "file_type_distribution": None,
                    "entity_summary": None,
                    "relationship_summary": None,
                    "capabilities": None,
                    "graph_statistics": None,
                    "learning_completed_at": None,
                }
            )

        return JSONResponse(status_code=status.HTTP_200_OK, content=summary_data)

    except DatabaseError as e:
        logger.error(
            "Database error getting folder summary",
            folder_id=folder_id,
            error=str(e),
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Failed to retrieve folder summary",
                "error_type": "DatabaseError",
                "detail": str(e) if not settings.is_production else "Internal server error",
            },
        )
    except Exception as e:
        logger.error(
            "Unexpected error getting folder summary",
            folder_id=folder_id,
            error=str(e),
            exc_info=True,
            traceback=traceback.format_exc(),
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Failed to retrieve folder summary",
                "error_type": type(e).__name__,
                "detail": str(e) if not settings.is_production else "Internal server error",
            },
        )


@router.post("/{folder_id}/summary/generate")
async def generate_folder_summary(
    folder_id: str,
    background_tasks: BackgroundTasks,
    db: SQLiteManager = Depends(get_db),
) -> Dict[str, str]:
    """Generate folder summary (user-initiated).

    This endpoint triggers folder summarization which enables general queries
    across files. Summary generation may take over 1 minute for large folders.

    Args:
        folder_id: Folder ID
        background_tasks: FastAPI background tasks
        db: Database manager

    Returns:
        Status message

    Raises:
        HTTPException: If folder not found

    Example:
        >>> POST /api/v1/folders/abc123/summary/generate
        {
            "message": "Folder summary generation started",
            "folder_id": "abc123",
            "status": "learning_in_progress"
        }
    """
    try:
        # Verify folder exists by checking FolderRecord
        async with db._get_db_manager().get_session() as session:
            folder_stmt = select(FolderRecord).where(FolderRecord.folder_id == folder_id)
            folder_result = await session.execute(folder_stmt)
            folder = folder_result.scalar_one_or_none()
            
            if not folder:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Folder not found",
                )

        # Trigger summarization in background with extended timeout
        summarizer = get_folder_summarizer()

        async def generate_task():
            """Background task to generate summary."""
            try:
                await summarizer.generate_folder_summary(
                    folder_id=folder_id, send_progress=True
                )
            except Exception as e:
                logger.error(
                    "Background summary generation failed",
                    folder_id=folder_id,
                    error=str(e),
                )

        background_tasks.add_task(generate_task)

        return {
            "message": "Folder summary generation started",
            "folder_id": folder_id,
            "status": "learning_in_progress",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "Failed to start folder summary generation", folder_id=folder_id, error=str(e)
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start folder summary generation: {str(e)}",
        )
