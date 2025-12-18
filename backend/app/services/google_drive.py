"""Google Drive API integration service with OAuth2 authentication."""

from typing import List, Dict, Optional, Any, Union, Callable
import os
import re
from pathlib import Path
import structlog

from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build  # type: ignore
from googleapiclient.http import MediaIoBaseDownload  # type: ignore
from googleapiclient.errors import HttpError  # type: ignore

from app.config.settings import settings
from app.core.exceptions import AuthenticationError, ProcessingError, ValidationError
from app.utils.file_utils import ensure_directory, sanitize_filename
from app.utils.cache import FileCache
from app.utils.rate_limiter import GoogleDriveRateLimiter

logger = structlog.get_logger(__name__)

# Google Drive API rate limits (handled by GoogleDriveRateLimiter)
MAX_REQUESTS_PER_SECOND = 10
MAX_REQUESTS_PER_100_SECONDS = 1000

# Supported file types for processing
SUPPORTED_MIME_TYPES = {
    "application/pdf": "pdf",
    "application/vnd.google-apps.document": "gdoc",
    "application/vnd.google-apps.spreadsheet": "gsheet",
    "application/vnd.google-apps.presentation": "gslides",
    "text/plain": "txt",
    "text/markdown": "md",
    "text/csv": "csv",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "audio/mpeg": "audio",
    "audio/wav": "audio",
    "audio/mp4": "audio",
    "audio/x-m4a": "audio",
    "video/mp4": "video",
    "video/quicktime": "video",
}


# RateLimiter moved to app.utils.rate_limiter module for better organization


class GoogleDriveService:
    """Service for interacting with Google Drive API."""

    def __init__(self, file_cache: Optional[FileCache] = None):
        """Initialize Google Drive service.

        Args:
            file_cache: Optional file cache instance
        """
        self.rate_limiter = GoogleDriveRateLimiter()
        self.service = None
        self.file_cache = file_cache or FileCache()

    def _build_service(self, access_token: str):
        """Build Google Drive API service with credentials.

        Args:
            access_token: Google OAuth2 access token
        """
        try:
            credentials = Credentials(token=access_token)
            self.service = build("drive", "v3", credentials=credentials, cache_discovery=False)
            logger.debug("Google Drive service initialized")
        except Exception as e:
            logger.error("Failed to build Google Drive service", error=str(e))
            raise AuthenticationError("Failed to initialize Google Drive service") from e

    def parse_folder_url(self, folder_url: str) -> Optional[str]:
        """Extract folder ID from Google Drive URL.

        Args:
            folder_url: Google Drive folder URL

        Returns:
            Folder ID or None if invalid
        """
        patterns = [
            r"/folders/([a-zA-Z0-9_-]+)",
            r"[?&]id=([a-zA-Z0-9_-]+)",
            r"drive\.google\.com/drive/folders/([a-zA-Z0-9_-]+)",
        ]

        for pattern in patterns:
            match = re.search(pattern, folder_url)
            if match:
                folder_id = match.group(1)
                # Validate folder ID format
                if re.match(r"^[a-zA-Z0-9_-]+$", folder_id) and len(folder_id) <= 64:
                    return folder_id

        return None

    async def list_folder_files(
        self, folder_id: str, access_token: str, user_id: str, recursive: bool = True
    ) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
        """List all files in a Google Drive folder, optionally recursively.

        Args:
            folder_id: Google Drive folder ID
            access_token: Google OAuth2 access token
            user_id: Authenticated user ID
            recursive: If True, recursively fetch files from all subfolders

        Returns:
            List of file metadata dictionaries with folder_id added to each file

        Raises:
            AuthenticationError: If user lacks access
            ProcessingError: If folder listing fails
            ValidationError: If folder ID is invalid
        """
        try:
            if not folder_id or not re.match(r"^[a-zA-Z0-9_-]+$", folder_id):
                raise ValidationError("Invalid folder ID format")

            if not self.service:
                self._build_service(access_token)
            logger.info("Listing folder files", folder_id=folder_id, user_id=user_id, recursive=recursive)

            if recursive:
                result = await self._list_folder_files_recursive(
                    folder_id, access_token, user_id, 
                    root_folder_id=folder_id,
                    parent_folder_id=None,
                    token_refresh_callback=None
                )
                # Return hierarchical structure with files and subfolders
                return result
            else:
                files = await self._list_folder_files_single(folder_id, access_token, user_id)
                return {"files": files, "subfolders": []}

        except (AuthenticationError, ValidationError, ProcessingError):
            raise
        except Exception as e:
            logger.error("Failed to list folder files", folder_id=folder_id, error=str(e), exc_info=True)
            raise ProcessingError(f"Failed to list folder files: {str(e)}") from e

    async def _list_folder_files_single(
        self, folder_id: str, access_token: str, user_id: str
    ) -> List[Dict[str, Any]]:
        """List files in a single folder (non-recursive).

        Args:
            folder_id: Google Drive folder ID
            access_token: Google OAuth2 access token
            user_id: Authenticated user ID

        Returns:
            List of file metadata dictionaries with folder_id added
        """
        if not self.service:
            self._build_service(access_token)
            
        files = []
        page_token = None

        while True:
            await self.rate_limiter.acquire()

            try:
                query = f"'{folder_id}' in parents and trashed = false"
                results = (
                    self.service.files()  # type: ignore
                    .list(
                        q=query,
                        pageSize=100,
                        fields="nextPageToken, files(id, name, mimeType, size, createdTime, modifiedTime, webViewLink, webContentLink)",
                        pageToken=page_token,
                    )
                    .execute()
                )
                
                # Report success to rate limiter
                await self.rate_limiter.report_success()

                items = results.get("files", [])
                # Add folder_id to each file for tracking
                for item in items:
                    item["folder_id"] = folder_id
                files.extend(items)

                page_token = results.get("nextPageToken")
                if not page_token:
                    break

            except HttpError as e:
                # Report error to rate limiter
                is_rate_limit = e.resp.status == 429
                await self.rate_limiter.report_error(is_rate_limit=is_rate_limit)
                
                if e.resp.status == 404:
                    raise ValidationError(f"Folder not found: {folder_id}")
                elif e.resp.status == 403:
                    raise AuthenticationError("Access denied to folder")
                elif e.resp.status == 429:
                    # Rate limit hit - will be handled by adaptive limiter
                    logger.warning("Rate limit hit, backing off", folder_id=folder_id)
                    raise ProcessingError("Rate limit exceeded, please try again later")
                else:
                    logger.error("Google Drive API error", error=str(e), status=e.resp.status)
                    raise ProcessingError(f"Failed to list folder files: {str(e)}")

        logger.debug("Folder files listed", folder_id=folder_id, file_count=len(files))
        return files

    async def _list_folder_files_recursive(
        self, folder_id: str, access_token: str, user_id: str, 
        root_folder_id: Optional[str] = None, parent_folder_id: Optional[str] = None,
        visited: Optional[set] = None, 
        token_refresh_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Recursively list all files in a folder and all subfolders, maintaining hierarchy.

        Args:
            folder_id: Google Drive folder ID
            access_token: Google OAuth2 access token
            user_id: Authenticated user ID
            root_folder_id: Root folder ID (top-level folder being processed)
            parent_folder_id: Parent folder ID for hierarchy tracking
            visited: Set of already visited folder IDs to prevent infinite loops

        Returns:
            Dictionary containing:
            - files: List of file metadata dictionaries
            - subfolders: List of subfolder metadata with their nested structure
        """
        if visited is None:
            visited = set()
        
        if root_folder_id is None:
            root_folder_id = folder_id

        # Prevent infinite loops from circular folder references
        if folder_id in visited:
            logger.warning("Circular folder reference detected", folder_id=folder_id)
            return {"files": [], "subfolders": []}

        visited.add(folder_id)
        all_files = []
        all_subfolders = []

        try:
            # Get all items in current folder
            items = await self._list_folder_files_single(folder_id, access_token, user_id)
            
            # Separate files and subfolders
            subfolders = []
            files = []
            
            for item in items:
                mime_type = item.get("mimeType", "")
                # Google Drive folders have this MIME type
                if mime_type == "application/vnd.google-apps.folder":
                    subfolders.append(item)
                else:
                    # Add hierarchy information to files
                    item["parent_folder_id"] = folder_id
                    item["root_folder_id"] = root_folder_id
                    files.append(item)
            
            # Add files from current folder
            all_files.extend(files)
            logger.debug(
                "Found files and subfolders",
                folder_id=folder_id,
                file_count=len(files),
                subfolder_count=len(subfolders)
            )
            
            # Recursively process each subfolder
            for subfolder in subfolders:
                subfolder_id = subfolder.get("id")
                subfolder_name = subfolder.get("name", "Unknown")
                if subfolder_id:
                    try:
                        # Recursively get files and subfolders from this subfolder
                        subfolder_result = await self._list_folder_files_recursive(
                            subfolder_id, access_token, user_id, 
                            root_folder_id=root_folder_id,
                            parent_folder_id=folder_id,
                            visited=visited
                        )
                        
                        # Add hierarchy information to subfolder metadata
                        subfolder_metadata = {
                            "folder_id": subfolder_id,
                            "folder_name": subfolder_name,
                            "parent_folder_id": folder_id,
                            "root_folder_id": root_folder_id,
                            "file_count": len(subfolder_result["files"]),
                            "subfolder_count": len(subfolder_result["subfolders"]),
                            "files": subfolder_result["files"],
                            "subfolders": subfolder_result["subfolders"],
                            "mimeType": "application/vnd.google-apps.folder",
                            "createdTime": subfolder.get("createdTime"),
                            "modifiedTime": subfolder.get("modifiedTime"),
                        }
                        all_subfolders.append(subfolder_metadata)
                        
                        # Add files from subfolder
                        all_files.extend(subfolder_result["files"])
                        
                        logger.debug(
                            "Processed subfolder",
                            subfolder_id=subfolder_id,
                            subfolder_name=subfolder_name,
                            file_count=len(subfolder_result["files"]),
                            nested_subfolder_count=len(subfolder_result["subfolders"])
                        )
                    except Exception as e:
                        logger.warning(
                            "Failed to process subfolder",
                            subfolder_id=subfolder_id,
                            error=str(e)
                        )
                        # Continue processing other subfolders even if one fails
                        continue

        except Exception as e:
            logger.error(
                "Error in recursive folder listing",
                folder_id=folder_id,
                error=str(e),
                exc_info=True
            )
            # Return files collected so far even if there's an error
            pass

        finally:
            # Remove from visited set to allow re-visiting if needed (though shouldn't happen)
            visited.discard(folder_id)

        logger.info(
            "Recursive folder listing complete",
            folder_id=folder_id,
            total_files=len(all_files),
            total_subfolders=len(all_subfolders)
        )
        return {"files": all_files, "subfolders": all_subfolders}

    async def build_folder_tree(
        self, folder_id: str, access_token: str, user_id: str,
        visited: Optional[set] = None,
        token_refresh_callback: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """Build a tree structure representing the exact Google Drive folder hierarchy.

        Args:
            folder_id: Google Drive folder ID (root of the tree)
            access_token: Google OAuth2 access token
            user_id: Authenticated user ID
            visited: Set of already visited folder IDs to prevent infinite loops
            token_refresh_callback: Optional callback to refresh access token

        Returns:
            Dictionary representing the root node with children:
            {
                "id": folder_id,
                "name": folder_name,
                "is_folder": True,
                "mime_type": "application/vnd.google-apps.folder",
                "size": 0,
                "created_at": datetime,
                "modified_at": datetime,
                "children": [TreeNode, ...]
            }
        """
        if visited is None:
            visited = set()

        # Prevent infinite loops from circular folder references
        if folder_id in visited:
            logger.warning("Circular folder reference detected", folder_id=folder_id)
            return {
                "id": folder_id,
                "name": "Circular Reference",
                "is_folder": True,
                "mime_type": "application/vnd.google-apps.folder",
                "size": 0,
                "children": []
            }

        visited.add(folder_id)

        try:
            # Get folder metadata
            folder_metadata = await self.get_file_metadata(folder_id, access_token)
            folder_name = folder_metadata.get("name", "Unknown Folder")
            created_time = folder_metadata.get("createdTime")
            modified_time = folder_metadata.get("modifiedTime")

            # Parse dates
            created_at = None
            modified_at = None
            if created_time and isinstance(created_time, str):
                try:
                    from datetime import datetime
                    created_at = datetime.fromisoformat(created_time.replace('Z', '+00:00'))
                except Exception:
                    pass
            if modified_time and isinstance(modified_time, str):
                try:
                    from datetime import datetime
                    modified_at = datetime.fromisoformat(modified_time.replace('Z', '+00:00'))
                except Exception:
                    pass

            # Get all items in current folder
            items = await self._list_folder_files_single(folder_id, access_token, user_id)

            # Separate files and subfolders
            files = []
            subfolders = []

            for item in items:
                mime_type = item.get("mimeType", "")
                if mime_type == "application/vnd.google-apps.folder":
                    subfolders.append(item)
                else:
                    files.append(item)

            # Build children list
            children = []

            # Add files as leaf nodes
            for file_item in files:
                file_created = None
                file_modified = None
                created_time_str = file_item.get("createdTime")
                if created_time_str and isinstance(created_time_str, str):
                    try:
                        from datetime import datetime
                        file_created = datetime.fromisoformat(created_time_str.replace('Z', '+00:00'))
                    except Exception:
                        pass
                modified_time_str = file_item.get("modifiedTime")
                if modified_time_str and isinstance(modified_time_str, str):
                    try:
                        from datetime import datetime
                        file_modified = datetime.fromisoformat(modified_time_str.replace('Z', '+00:00'))
                    except Exception:
                        pass

                children.append({
                    "id": file_item.get("id", ""),
                    "name": file_item.get("name", "Unknown"),
                    "is_folder": False,
                    "mime_type": file_item.get("mimeType", "application/octet-stream"),
                    "size": int(file_item.get("size", 0)),
                    "created_at": file_created,
                    "modified_at": file_modified,
                    "web_view_link": file_item.get("webViewLink"),
                    "web_content_link": file_item.get("webContentLink"),
                    "children": []
                })

            # Recursively build subfolder trees
            for subfolder in subfolders:
                subfolder_id = subfolder.get("id")
                if subfolder_id:
                    try:
                        subfolder_tree = await self.build_folder_tree(
                            subfolder_id, access_token, user_id, visited, token_refresh_callback
                        )
                        children.append(subfolder_tree)
                    except Exception as e:
                        logger.warning(
                            "Failed to build subfolder tree",
                            subfolder_id=subfolder_id,
                            error=str(e)
                        )
                        # Add a placeholder node for failed subfolder
                        children.append({
                            "id": subfolder_id,
                            "name": subfolder.get("name", "Unknown"),
                            "is_folder": True,
                            "mime_type": "application/vnd.google-apps.folder",
                            "size": 0,
                            "children": []
                        })

            # Build root node
            root_node = {
                "id": folder_id,
                "name": folder_name,
                "is_folder": True,
                "mime_type": "application/vnd.google-apps.folder",
                "size": 0,
                "created_at": created_at,
                "modified_at": modified_at,
                "web_view_link": folder_metadata.get("webViewLink"),
                "children": children
            }

            logger.debug(
                "Built folder tree",
                folder_id=folder_id,
                folder_name=folder_name,
                file_count=len(files),
                subfolder_count=len(subfolders),
                total_children=len(children)
            )

            return root_node

        except Exception as e:
            logger.error(
                "Error building folder tree",
                folder_id=folder_id,
                error=str(e),
                exc_info=True
            )
            # Return a minimal node on error
            return {
                "id": folder_id,
                "name": "Error Loading Folder",
                "is_folder": True,
                "mime_type": "application/vnd.google-apps.folder",
                "size": 0,
                "children": []
            }
        finally:
            visited.discard(folder_id)

    async def get_file_metadata(
        self, file_id: str, access_token: str
    ) -> Dict[str, Any]:
        """Get metadata for a file.

        Args:
            file_id: Google Drive file ID
            access_token: Google OAuth2 access token

        Returns:
            File metadata dictionary

        Raises:
            ProcessingError: If metadata retrieval fails
        """
        try:
            self._build_service(access_token)
            if not self.service:
                raise ProcessingError("Failed to initialize Google Drive service")
                
            await self.rate_limiter.acquire()

            file_metadata = (
                self.service.files()
                .get(
                    fileId=file_id,
                    fields="id, name, mimeType, size, createdTime, modifiedTime, webViewLink, webContentLink, parents",
                )
                .execute()
            )

            return file_metadata

        except HttpError as e:
            if e.resp.status == 404:
                raise ProcessingError(f"File not found: {file_id}")
            elif e.resp.status == 403:
                raise AuthenticationError("Access denied to file")
            else:
                logger.error("Failed to get file metadata", file_id=file_id, error=str(e))
                raise ProcessingError(f"Failed to get file metadata: {str(e)}")
        except Exception as e:
            logger.error("Failed to get file metadata", file_id=file_id, error=str(e))
            raise ProcessingError(f"Failed to get file metadata: {str(e)}") from e

    async def download_file(
        self, file_id: str, access_token: str, user_id: str, mime_type: Optional[str] = None
    ) -> str:
        """Download a file from Google Drive to local cache.

        Args:
            file_id: Google Drive file ID
            access_token: Google OAuth2 access token
            user_id: Authenticated user ID
            mime_type: Optional MIME type for export format

        Returns:
            Local file path

        Raises:
            ProcessingError: If download fails
        """
        try:
            self._build_service(access_token)

            # Get file metadata first
            file_metadata = await self.get_file_metadata(file_id, access_token)
            file_name = file_metadata.get("name", file_id)
            file_mime_type = mime_type or file_metadata.get("mimeType", "")

            # Create cache directory
            cache_dir = os.path.join(settings.CACHE_DIR, user_id)
            ensure_directory(cache_dir)

            # Determine file extension
            file_ext = self._get_file_extension(file_mime_type, file_name)
            local_filename = sanitize_filename(f"{file_id}{file_ext}")
            local_path = os.path.join(cache_dir, local_filename)

            # Check file cache first
            cached_path = self.file_cache.get_file_path(file_id, user_id)
            if cached_path and cached_path.exists():
                logger.debug("File found in cache", file_id=file_id, path=str(cached_path))
                return str(cached_path)

            # Check if file already exists in cache directory
            if os.path.exists(local_path):
                logger.debug("File already exists", file_id=file_id, path=local_path)
                # Store in cache metadata
                self.file_cache.store_file(file_id, user_id, Path(local_path))
                return local_path

            await self.rate_limiter.acquire()
            
            if not self.service:
                raise ProcessingError("Failed to initialize Google Drive service")

            # Handle Google Workspace files (need export)
            if file_mime_type.startswith("application/vnd.google-apps"):
                export_mime_type = self._get_export_mime_type(file_mime_type)
                request = self.service.files().export_media(fileId=file_id, mimeType=export_mime_type)
            else:
                request = self.service.files().get_media(fileId=file_id)

            # Download file
            with open(local_path, "wb") as fh:
                downloader = MediaIoBaseDownload(fh, request)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
                    if status:
                        logger.debug(
                            "Download progress",
                            file_id=file_id,
                            progress=int(status.progress() * 100),
                        )

            file_size = os.path.getsize(local_path)
            logger.info("File downloaded", file_id=file_id, path=local_path, size=file_size)
            
            # Store in cache
            self.file_cache.store_file(file_id, user_id, Path(local_path))
            
            return local_path

        except HttpError as e:
            if e.resp.status == 404:
                raise ProcessingError(f"File not found: {file_id}")
            elif e.resp.status == 403:
                raise AuthenticationError("Access denied to file")
            else:
                logger.error("Failed to download file", file_id=file_id, error=str(e))
                raise ProcessingError(f"Failed to download file: {str(e)}")
        except Exception as e:
            logger.error("Failed to download file", file_id=file_id, error=str(e), exc_info=True)
            raise ProcessingError(f"Failed to download file: {str(e)}") from e

    def _get_file_extension(self, mime_type: str, file_name: str) -> str:
        """Get file extension from MIME type or filename.

        Args:
            mime_type: File MIME type
            file_name: Original file name

        Returns:
            File extension with dot (e.g., ".pdf")
        """
        # Try to get extension from MIME type mapping
        if mime_type in SUPPORTED_MIME_TYPES:
            ext = SUPPORTED_MIME_TYPES[mime_type]
            if ext == "pdf":
                return ".pdf"
            elif ext == "gdoc":
                return ".docx"  # Export as DOCX
            elif ext == "gsheet":
                return ".xlsx"  # Export as XLSX
            elif ext == "gslides":
                return ".pptx"  # Export as PPTX
            elif ext == "txt":
                return ".txt"
            elif ext == "md":
                return ".md"
            elif ext == "csv":
                return ".csv"
            elif ext == "docx":
                return ".docx"
            elif ext == "xlsx":
                return ".xlsx"
            elif ext == "pptx":
                return ".pptx"
            elif ext == "audio":
                # Try to determine from filename
                if file_name.lower().endswith((".mp3", ".wav", ".m4a")):
                    return os.path.splitext(file_name)[1]
                return ".mp3"  # Default
            elif ext == "video":
                if file_name.lower().endswith((".mp4", ".mov")):
                    return os.path.splitext(file_name)[1]
                return ".mp4"  # Default

        # Fallback to filename extension
        if file_name:
            ext = os.path.splitext(file_name)[1]
            if ext:
                return ext

        return ""

    def _get_export_mime_type(self, google_mime_type: str) -> str:
        """Get export MIME type for Google Workspace files.

        Args:
            google_mime_type: Google Workspace MIME type

        Returns:
            Export MIME type
        """
        export_map = {
            "application/vnd.google-apps.document": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.google-apps.spreadsheet": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.google-apps.presentation": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        }
        return export_map.get(google_mime_type, "application/pdf")

    def is_supported_file_type(self, mime_type: str) -> bool:
        """Check if file type is supported for processing.

        Args:
            mime_type: File MIME type

        Returns:
            True if file type is supported
        """
        return mime_type in SUPPORTED_MIME_TYPES or any(
            mime_type.startswith(prefix) for prefix in ["text/", "audio/", "video/"]
        )

    async def get_folder_metadata(
        self, folder_id: str, access_token: str
    ) -> Dict[str, Any]:
        """Get metadata for a folder.

        Args:
            folder_id: Google Drive folder ID
            access_token: Google OAuth2 access token

        Returns:
            Folder metadata dictionary

        Raises:
            ProcessingError: If metadata retrieval fails
        """
        try:
            return await self.get_file_metadata(folder_id, access_token)
        except Exception as e:
            logger.error("Failed to get folder metadata", folder_id=folder_id, error=str(e))
            raise ProcessingError(f"Failed to get folder metadata: {str(e)}") from e
