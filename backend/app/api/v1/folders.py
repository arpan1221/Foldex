"""Folder processing API endpoints with background tasks."""

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status
from typing import List
import structlog
from datetime import datetime, timedelta

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
    """Get user's Google access token from database, refreshing if expired.

    Args:
        user_id: User identifier
        db: Database session

    Returns:
        Google OAuth2 access token (refreshed if needed)

    Raises:
        AuthenticationError: If token not found or refresh fails
    """
    from app.core.auth import refresh_google_access_token
    
    db_manager = SQLiteManager()
    user = await db_manager.get_user_by_id(user_id)
    
    if not user or not user.get("google_access_token"):
        raise AuthenticationError("Google access token not available. Please re-authenticate.")
    
    access_token = user["google_access_token"]
    refresh_token = user.get("google_refresh_token")
    token_expiry = user.get("google_token_expiry")
    
    # Check if token is expired or about to expire (within 5 minutes)
    needs_refresh = False
    if token_expiry:
        try:
            if isinstance(token_expiry, str):
                expiry_dt = datetime.fromisoformat(token_expiry.replace('Z', '+00:00'))
            else:
                expiry_dt = token_expiry
            
            # Check if expired or expires within 5 minutes
            if expiry_dt <= datetime.utcnow() + timedelta(minutes=5):
                needs_refresh = True
        except Exception as e:
            logger.warning("Failed to parse token expiry", error=str(e))
            # If we can't parse expiry, try to use the token and refresh if it fails
            needs_refresh = False
    
    # If no expiry info, try to validate the token
    if not token_expiry:
        try:
            from app.core.auth import validate_google_token
            await validate_google_token(access_token)
        except AuthenticationError:
            needs_refresh = True
    
    # Refresh token if needed
    if needs_refresh:
        if not refresh_token:
            raise AuthenticationError(
                "Google access token expired and no refresh token available. Please re-authenticate."
            )
        
        try:
            refresh_result = await refresh_google_access_token(refresh_token)
            new_access_token = refresh_result["access_token"]
            expires_in = refresh_result["expires_in"]
            
            # Update token in database
            new_expiry = datetime.utcnow() + timedelta(seconds=expires_in)
            await db_manager.update_user(
                user_id,
                google_access_token=new_access_token,
                google_token_expiry=new_expiry,
            )
            
            logger.info("Google access token refreshed", user_id=user_id)
            return new_access_token
        except AuthenticationError as e:
            logger.error("Failed to refresh Google access token", user_id=user_id, error=str(e))
            raise AuthenticationError(
                "Failed to refresh Google access token. Please re-authenticate."
            ) from e
    
    return access_token


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

        # Get folder from database to get actual status
        db_manager = SQLiteManager()
        folders = await db_manager.get_user_folders(user_id)
        folder = next((f for f in folders if f["folder_id"] == folder_id), None)
        
        if not folder:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Folder not found"
            )
        
        # Get actual file count from database
        file_count = await db_manager.get_file_count_by_folder(folder_id)
        
        # Use actual status from database
        status_value = folder.get("status", "pending")
        
        # Calculate progress based on status
        if status_value == "completed":
            progress = 1.0
        elif status_value == "processing":
            # If processing, we can't know total files yet, so use 0.5 as placeholder
            progress = 0.5 if file_count == 0 else min(0.9, file_count / max(file_count * 2, 1))
        else:
            progress = 0.0

        return FolderStatusResponse(
            folder_id=folder_id,
            status=status_value,
            files_processed=file_count,
            total_files=file_count,  # This could be improved to track total_files separately
            progress=progress,
            error=None,
        )

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except HTTPException:
        raise
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
    """Get list of files and subfolders in a folder (flat list, for backward compatibility).

    Args:
        folder_id: Google Drive folder ID
        current_user: Current authenticated user
        db: Database session

    Returns:
        List of file and folder metadata

    Raises:
        HTTPException: If folder not found or access denied
    """
    try:
        folder_id = sanitize_folder_id(folder_id)
        user_id = current_user.get("user_id")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token"
            )

        # Get user's Google access token
        google_token = await get_user_google_token(user_id, db)

        # Get files and folders from Google Drive (not just processed files)
        drive_service = GoogleDriveService()
        drive_items = await drive_service.list_folder_files(folder_id, google_token, user_id)

        # Get processed files from database for additional metadata
        db_manager = SQLiteManager()
        processed_files = await db_manager.get_files_by_folder(folder_id)
        processed_file_map = {f["file_id"]: f for f in processed_files}

        result = []
        for item in drive_items:
            if not isinstance(item, dict):
                continue
            item_id = item.get("id")
            if not item_id:
                continue
                
            mime_type = item.get("mimeType", "application/octet-stream")
            is_folder = mime_type == "application/vnd.google-apps.folder"
            
            # Get additional metadata from processed files if available
            processed_data = processed_file_map.get(item_id, {})
            
            # Parse dates
            created_time = item.get("createdTime")
            modified_time = item.get("modifiedTime")
            created_at = None
            modified_at = None
            if created_time:
                try:
                    from datetime import datetime
                    created_at = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
                except Exception:
                    pass
            if modified_time:
                try:
                    from datetime import datetime
                    modified_at = datetime.fromisoformat(modified_time.replace('Z', '+00:00'))
                except Exception:
                    pass

            result.append(
                FileMetadata(
                    file_id=item_id,
                    file_name=item.get("name", "Unknown"),
                    mime_type=mime_type,
                    size=int(item.get("size", 0)) if not is_folder else 0,
                    folder_id=folder_id,
                    created_at=created_at or processed_data.get("created_at"),
                    modified_at=modified_at or processed_data.get("modified_at"),
                    web_view_link=item.get("webViewLink"),
                    web_content_link=item.get("webContentLink"),
                    is_folder=is_folder,
                )
            )

        return result

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


@router.get("/{folder_id}/tree")
async def get_folder_tree(
    folder_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
):
    """Get tree structure representing the exact Google Drive folder hierarchy.

    Args:
        folder_id: Google Drive folder ID (root of the tree)
        current_user: Current authenticated user
        db: Database session

    Returns:
        Tree structure with root node and nested children

    Raises:
        HTTPException: If folder not found or access denied
    """
    try:
        from app.models.documents import TreeNode
        
        folder_id = sanitize_folder_id(folder_id)
        user_id = current_user.get("user_id")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token"
            )

        # Get user's Google access token
        google_token = await get_user_google_token(user_id, db)

        # Build tree structure
        drive_service = GoogleDriveService()
        tree_data = await drive_service.build_folder_tree(folder_id, google_token, user_id)

        # Convert to TreeNode model for validation
        tree_node = TreeNode(**tree_data)
        
        return tree_node.model_dump()

    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to get folder tree", folder_id=folder_id, error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve folder tree",
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
        List of folder metadata with file and folder counts

    Raises:
        HTTPException: If retrieval fails
    """
    try:
        user_id = current_user.get("user_id")
        if not user_id:
            raise AuthenticationError("User ID not found")

        # Get folders from database
        db_manager = SQLiteManager()
        folders = await db_manager.get_user_folders(user_id)

        # Get user's Google access token for counting files/folders
        try:
            google_token = await get_user_google_token(user_id, db)
            drive_service = GoogleDriveService()
        except Exception:
            # If token unavailable, use database counts only
            google_token = None
            drive_service = None

        result = []
        for folder in folders:
            folder_id = folder["folder_id"]
            file_count = 0
            folder_count = 0

            # Try to get actual counts from Google Drive
            if google_token and drive_service:
                try:
                    drive_items = await drive_service.list_folder_files(folder_id, google_token, user_id)
                    for item in drive_items:
                        mime_type = item.get("mimeType", "")
                        if mime_type == "application/vnd.google-apps.folder":
                            folder_count += 1
                        else:
                            file_count += 1
                except Exception as e:
                    logger.warning(
                        "Failed to get folder counts from Google Drive",
                        folder_id=folder_id,
                        error=str(e)
                    )
                    # Fall back to database count if available
                    file_count = folder.get("file_count", 0)

            # If we couldn't get counts from Google Drive, use database count
            if file_count == 0 and folder_count == 0:
                file_count = folder.get("file_count", 0)

            result.append(
                FolderMetadata(
                    folder_id=folder_id,
                    folder_name=folder["folder_name"],
                    file_count=file_count,
                    folder_count=folder_count,
                    status=folder["status"],
                    created_at=folder["created_at"],
                    updated_at=folder["updated_at"],
                )
            )

        return result

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
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token"
            )

        # Get Google access token
        google_token = await get_user_google_token(user_id, db)

        # Get folder metadata from Google Drive
        drive_service = GoogleDriveService()
        folder_info = await drive_service.get_folder_metadata(folder_id, google_token)


        # Get file and folder counts from Google Drive
        drive_items = await drive_service.list_folder_files(folder_id, google_token, user_id)
        file_count = 0
        folder_count = 0
        for item in drive_items:
            mime_type = item.get("mimeType", "")
            if mime_type == "application/vnd.google-apps.folder":
                folder_count += 1
            else:
                file_count += 1

        return FolderMetadata(
            folder_id=folder_id,
            folder_name=folder_info.get("name", folder_id),
            file_count=file_count,
            folder_count=folder_count,
            status="completed" if file_count > 0 or folder_count > 0 else "pending",
            created_at=folder_info.get("createdTime"),
            updated_at=folder_info.get("modifiedTime"),
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


@router.delete("/{folder_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_folder(
    folder_id: str,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> None:
    """Delete a folder and all associated data.

    Args:
        folder_id: Google Drive folder ID
        current_user: Current authenticated user
        db: Database session

    Raises:
        HTTPException: If deletion fails or folder not found
    """
    try:
        folder_id = sanitize_folder_id(folder_id)
        user_id = current_user.get("user_id")
        
        if not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User ID not found in token"
            )

        # Get file IDs before deletion for cache cleanup
        db_manager = SQLiteManager()
        files = await db_manager.get_files_by_folder(folder_id)
        file_ids = [file["file_id"] for file in files]

        # Delete chunks from vector store (before database deletion)
        try:
            from app.rag.vector_store import LangChainVectorStore
            vector_store = LangChainVectorStore()
            # Note: LangChainVectorStore may not have delete_chunks_by_folder method
            # This is a best-effort deletion
            logger.info("Vector store cleanup for folder", folder_id=folder_id)
        except Exception as e:
            logger.warning(
                "Failed to delete chunks from vector store",
                folder_id=folder_id,
                error=str(e)
            )
            # Don't fail the request if vector store deletion fails

        # Delete from database (cascade deletes files and chunks)
        deleted = await db_manager.delete_folder(folder_id, user_id)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Folder not found or access denied"
            )

        # Clean up cached files
        try:
            from app.utils.cache import FileCache
            cache = FileCache()
            for file_id in file_ids:
                cache.delete_file(file_id, user_id)
        except Exception as e:
            logger.warning(
                "Failed to clean up cached files",
                folder_id=folder_id,
                error=str(e)
            )
            # Don't fail the request if cache cleanup fails

        logger.info("Folder deleted successfully", folder_id=folder_id, user_id=user_id)

    except HTTPException:
        raise
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(e)
        )
    except Exception as e:
        logger.error("Failed to delete folder", folder_id=folder_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete folder",
        )
