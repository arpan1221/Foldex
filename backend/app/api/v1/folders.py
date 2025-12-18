"""Folder processing API endpoints with background tasks."""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from typing import List, Optional
import structlog

from app.api.deps import get_current_user, get_db_session
from app.core.exceptions import ValidationError, ProcessingError, AuthenticationError
from app.core.security import sanitize_folder_id
from app.services.google_drive import GoogleDriveService
from app.services.folder_processor import FolderProcessor
from app.models.documents import (
    ProcessFolderRequest,
    ProcessFolderResponse,
    FolderStatusResponse,
    FileMetadata,
    FolderMetadata,
)
from app.database.sqlite_manager import SQLiteManager
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)
router = APIRouter()


async def get_user_google_token(user_id: str, db: AsyncSession) -> str:
    """Get user's Google access token from database.

    Args:
        user_id: User identifier
        db: Database session

    Returns:
        Google OAuth2 access token

    Raises:
        AuthenticationError: If token not found
    """
    # TODO: Store and retrieve Google tokens from database
    # For now, this is a placeholder - tokens should be stored
    # when user authenticates and refreshed as needed
    raise AuthenticationError("Google access token not available. Please re-authenticate.")


@router.post("/process", response_model=ProcessFolderResponse, status_code=status.HTTP_202_ACCEPTED)
async def process_folder(
    request: ProcessFolderRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> ProcessFolderResponse:
    """Process a Google Drive folder for indexing.

    Validates folder ID, lists files, and starts background processing.

    Args:
        request: Folder processing request
        background_tasks: FastAPI background tasks
        current_user: Current authenticated user
        db: Database session

    Returns:
        Processing response with status

    Raises:
        HTTPException: If validation or processing fails
    """
    try:
        user_id = current_user.get("user_id")
        if not user_id:
            raise AuthenticationError("User ID not found")

        # Extract folder ID from URL if provided
        folder_id = request.folder_id
        if request.folder_url:
            drive_service = GoogleDriveService()
            parsed_id = drive_service.parse_folder_url(request.folder_url)
            if parsed_id:
                folder_id = parsed_id
            else:
                raise ValidationError("Invalid folder URL format")

        # Sanitize and validate folder ID
        folder_id = sanitize_folder_id(folder_id)

        # Get user's Google access token
        # TODO: Retrieve from database or session
        google_token = current_user.get("google_access_token")
        if not google_token:
            raise AuthenticationError("Google access token required. Please re-authenticate.")

        # Initialize services
        drive_service = GoogleDriveService()
        processor = FolderProcessor()

        # List files to get count
        try:
            files = await drive_service.list_folder_files(folder_id, google_token, user_id)
            files_detected = len(files)
        except Exception as e:
            logger.error("Failed to list folder files", folder_id=folder_id, error=str(e))
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to access folder: {str(e)}",
            )

        # Start background processing
        background_tasks.add_task(
            processor.process_folder_async,
            folder_id,
            user_id,
            google_token,
        )

        logger.info(
            "Folder processing started",
            folder_id=folder_id,
            user_id=user_id,
            files_detected=files_detected,
        )

        return ProcessFolderResponse(
            folder_id=folder_id,
            status="processing",
            message="Folder processing started",
            files_detected=files_detected,
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e)
        )
    except Exception as e:
        logger.error("Unexpected error in process_folder", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during folder processing",
        )


@router.get("/{folder_id}/status", response_model=FolderStatusResponse)
async def get_folder_status(
    folder_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> FolderStatusResponse:
    """Get processing status for a folder.

    Args:
        folder_id: Google Drive folder ID
        current_user: Current authenticated user
        db: Database session

    Returns:
        Folder processing status

    Raises:
        HTTPException: If folder not found or access denied
    """
    try:
        folder_id = sanitize_folder_id(folder_id)
        user_id = current_user.get("user_id")

        # Get status from database
        db_manager = SQLiteManager()
        file_count = await db_manager.get_file_count_by_folder(folder_id)

        # TODO: Implement proper status tracking
        # For now, return basic status
        status_value = "completed" if file_count > 0 else "pending"

        return FolderStatusResponse(
            folder_id=folder_id,
            status=status_value,
            files_processed=file_count,
            total_files=file_count,
            progress=1.0 if file_count > 0 else 0.0,
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to get folder status", folder_id=folder_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve folder status",
        )


@router.get("/{folder_id}/files", response_model=List[FileMetadata])
async def get_folder_files(
    folder_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> List[FileMetadata]:
    """Get list of files in a processed folder.

    Args:
        folder_id: Google Drive folder ID
        current_user: Current authenticated user
        db: Database session

    Returns:
        List of file metadata

    Raises:
        HTTPException: If folder not found or access denied
    """
    try:
        folder_id = sanitize_folder_id(folder_id)
        user_id = current_user.get("user_id")

        # Get files from database
        db_manager = SQLiteManager()
        files = await db_manager.get_files_by_folder(folder_id)

        return [
            FileMetadata(
                file_id=file["file_id"],
                file_name=file["file_name"],
                mime_type=file.get("mime_type", "application/octet-stream"),
                size=file.get("size", 0),
                folder_id=file.get("folder_id", folder_id),
                created_at=file.get("created_at"),
                modified_at=file.get("modified_at"),
            )
            for file in files
        ]

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to get folder files", folder_id=folder_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve folder files",
        )


@router.get("", response_model=List[FolderMetadata])
async def get_user_folders(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> List[FolderMetadata]:
    """Get list of folders processed by the current user.

    Args:
        current_user: Current authenticated user
        db: Database session

    Returns:
        List of folder metadata

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        user_id = current_user.get("user_id")

        # TODO: Implement folder listing from database
        # For now, return empty list
        return []

    except Exception as e:
        logger.error("Failed to get user folders", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve folders",
        )


@router.get("/{folder_id}/metadata", response_model=FolderMetadata)
async def get_folder_metadata(
    folder_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> FolderMetadata:
    """Get metadata for a processed folder.

    Args:
        folder_id: Google Drive folder ID
        current_user: Current authenticated user
        db: Database session

    Returns:
        Folder metadata

    Raises:
        HTTPException: If folder not found
    """
    try:
        folder_id = sanitize_folder_id(folder_id)
        user_id = current_user.get("user_id")

        # Get Google access token
        google_token = current_user.get("google_access_token")
        if not google_token:
            raise AuthenticationError("Google access token required")

        # Get folder metadata from Google Drive
        drive_service = GoogleDriveService()
        folder_info = await drive_service.get_folder_metadata(folder_id, google_token)

        # Get file count from database
        db_manager = SQLiteManager()
        file_count = await db_manager.get_file_count_by_folder(folder_id)

        return FolderMetadata(
            folder_id=folder_id,
            folder_name=folder_info.get("name", folder_id),
            file_count=file_count,
            total_size=0,  # TODO: Calculate total size
            created_at=folder_info.get("createdTime"),
            modified_at=folder_info.get("modifiedTime"),
            web_view_link=folder_info.get("webViewLink"),
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except AuthenticationError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to get folder metadata", folder_id=folder_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve folder metadata",
        )
