"""Google Drive API integration service with OAuth2 authentication."""

from typing import List, Dict, Optional, Any
import os
import re
import tempfile
import asyncio
from datetime import datetime, timedelta
import structlog

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError
import httpx
import io

from app.config.settings import settings
from app.core.exceptions import AuthenticationError, ProcessingError, ValidationError
from app.utils.file_utils import ensure_directory, sanitize_filename, get_file_hash
from app.utils.cache import FileCache

logger = structlog.get_logger(__name__)

# Google Drive API rate limits
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


class RateLimiter:
    """Simple rate limiter for API calls."""

    def __init__(self, max_calls: int, time_window: float):
        """Initialize rate limiter.

        Args:
            max_calls: Maximum number of calls allowed
            time_window: Time window in seconds
        """
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = []

    async def acquire(self):
        """Acquire permission to make an API call."""
        now = datetime.utcnow()
        # Remove old calls outside the time window
        self.calls = [call_time for call_time in self.calls if (now - call_time).total_seconds() < self.time_window]

        # If we've hit the limit, wait
        if len(self.calls) >= self.max_calls:
            oldest_call = min(self.calls)
            wait_time = self.time_window - (now - oldest_call).total_seconds() + 0.1
            if wait_time > 0:
                await asyncio.sleep(wait_time)
                # Clean up again after waiting
                now = datetime.utcnow()
                self.calls = [call_time for call_time in self.calls if (now - call_time).total_seconds() < self.time_window]

        self.calls.append(datetime.utcnow())


class GoogleDriveService:
    """Service for interacting with Google Drive API."""

    def __init__(self, file_cache: Optional[FileCache] = None):
        """Initialize Google Drive service.

        Args:
            file_cache: Optional file cache instance
        """
        self.rate_limiter = RateLimiter(MAX_REQUESTS_PER_SECOND, 1.0)
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
        self, folder_id: str, access_token: str, user_id: str
    ) -> List[Dict[str, Any]]:
        """List all files in a Google Drive folder.

        Args:
            folder_id: Google Drive folder ID
            access_token: Google OAuth2 access token
            user_id: Authenticated user ID

        Returns:
            List of file metadata dictionaries

        Raises:
            AuthenticationError: If user lacks access
            ProcessingError: If folder listing fails
            ValidationError: If folder ID is invalid
        """
        try:
            if not folder_id or not re.match(r"^[a-zA-Z0-9_-]+$", folder_id):
                raise ValidationError("Invalid folder ID format")

            self._build_service(access_token)
            logger.info("Listing folder files", folder_id=folder_id, user_id=user_id)

            files = []
            page_token = None

            while True:
                await self.rate_limiter.acquire()

                try:
                    query = f"'{folder_id}' in parents and trashed = false"
                    results = (
                        self.service.files()
                        .list(
                            q=query,
                            pageSize=100,
                            fields="nextPageToken, files(id, name, mimeType, size, createdTime, modifiedTime, webViewLink, webContentLink)",
                            pageToken=page_token,
                        )
                        .execute()
                    )

                    items = results.get("files", [])
                    files.extend(items)

                    page_token = results.get("nextPageToken")
                    if not page_token:
                        break

                except HttpError as e:
                    if e.resp.status == 404:
                        raise ValidationError(f"Folder not found: {folder_id}")
                    elif e.resp.status == 403:
                        raise AuthenticationError("Access denied to folder")
                    else:
                        logger.error("Google Drive API error", error=str(e), status=e.resp.status)
                        raise ProcessingError(f"Failed to list folder files: {str(e)}")

            logger.info("Folder files listed", folder_id=folder_id, file_count=len(files))
            return files

        except (AuthenticationError, ValidationError, ProcessingError):
            raise
        except Exception as e:
            logger.error("Failed to list folder files", folder_id=folder_id, error=str(e), exc_info=True)
            raise ProcessingError(f"Failed to list folder files: {str(e)}") from e

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
