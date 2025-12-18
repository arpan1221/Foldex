"""Folder processing API endpoints."""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from pydantic import BaseModel
from typing import Optional

from app.api.deps import get_current_user
from app.core.exceptions import ValidationError
from app.core.security import sanitize_folder_id
from app.services.folder_processor import FolderProcessor

router = APIRouter()


class ProcessFolderRequest(BaseModel):
    """Request model for folder processing."""

    folder_id: str


class ProcessingResponse(BaseModel):
    """Response model for folder processing."""

    folder_id: str
    status: str
    message: str


@router.post("/process", response_model=ProcessingResponse)
async def process_folder(
    request: ProcessFolderRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
) -> ProcessingResponse:
    """Process a Google Drive folder for indexing.

    Args:
        request: Folder processing request
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user

    Returns:
        Processing response with status

    Raises:
        HTTPException: If validation or processing fails
    """
    try:
        folder_id = sanitize_folder_id(request.folder_id)
        processor = FolderProcessor()

        # Start background processing
        background_tasks.add_task(
            processor.process_folder_async, folder_id, current_user.get("sub")
        )

        return ProcessingResponse(
            folder_id=folder_id,
            status="processing",
            message="Folder processing started",
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )


@router.get("/{folder_id}/status")
async def get_folder_status(
    folder_id: str, current_user: dict = Depends(get_current_user)
) -> dict:
    """Get processing status for a folder.

    Args:
        folder_id: Google Drive folder ID
        current_user: Current authenticated user

    Returns:
        Folder processing status
    """
    # TODO: Implement status retrieval
    return {"folder_id": folder_id, "status": "unknown"}

