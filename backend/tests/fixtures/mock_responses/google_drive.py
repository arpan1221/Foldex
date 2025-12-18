"""Mock Google Drive API responses for testing."""

# Sample folder listing response
FOLDER_FILES_RESPONSE = {
    "files": [
        {
            "id": "1abc123xyz",
            "name": "Sample Document.pdf",
            "mimeType": "application/pdf",
            "size": "245760",
            "createdTime": "2024-01-15T10:30:00.000Z",
            "modifiedTime": "2024-01-20T14:22:00.000Z",
            "webViewLink": "https://drive.google.com/file/d/1abc123xyz/view",
            "webContentLink": "https://drive.google.com/uc?id=1abc123xyz",
            "parents": ["folder_123"],
        },
        {
            "id": "2def456uvw",
            "name": "Meeting Notes.txt",
            "mimeType": "text/plain",
            "size": "8192",
            "createdTime": "2024-01-16T09:15:00.000Z",
            "modifiedTime": "2024-01-18T16:45:00.000Z",
            "webViewLink": "https://drive.google.com/file/d/2def456uvw/view",
            "webContentLink": "https://drive.google.com/uc?id=2def456uvw",
            "parents": ["folder_123"],
        },
        {
            "id": "3ghi789rst",
            "name": "README.md",
            "mimeType": "text/markdown",
            "size": "4096",
            "createdTime": "2024-01-10T08:00:00.000Z",
            "modifiedTime": "2024-01-19T11:30:00.000Z",
            "webViewLink": "https://drive.google.com/file/d/3ghi789rst/view",
            "webContentLink": "https://drive.google.com/uc?id=3ghi789rst",
            "parents": ["folder_123"],
        },
    ],
    "nextPageToken": None,
}

# Sample file metadata response
FILE_METADATA_RESPONSE = {
    "id": "1abc123xyz",
    "name": "Sample Document.pdf",
    "mimeType": "application/pdf",
    "size": "245760",
    "createdTime": "2024-01-15T10:30:00.000Z",
    "modifiedTime": "2024-01-20T14:22:00.000Z",
    "webViewLink": "https://drive.google.com/file/d/1abc123xyz/view",
    "webContentLink": "https://drive.google.com/uc?id=1abc123xyz",
    "parents": ["folder_123"],
}

# Sample folder metadata response
FOLDER_METADATA_RESPONSE = {
    "id": "folder_123",
    "name": "Test Folder",
    "mimeType": "application/vnd.google-apps.folder",
    "createdTime": "2024-01-01T00:00:00.000Z",
    "modifiedTime": "2024-01-20T14:22:00.000Z",
}

# Sample error response
ERROR_RESPONSE_404 = {
    "error": {
        "errors": [
            {
                "domain": "global",
                "reason": "notFound",
                "message": "File not found: 1abc123xyz",
            }
        ],
        "code": 404,
        "message": "File not found: 1abc123xyz",
    }
}

ERROR_RESPONSE_403 = {
    "error": {
        "errors": [
            {
                "domain": "global",
                "reason": "insufficientPermissions",
                "message": "Insufficient permissions to access file",
            }
        ],
        "code": 403,
        "message": "Insufficient permissions to access file",
    }
}

# Sample OAuth token response
OAUTH_TOKEN_RESPONSE = {
    "access_token": "ya29.mock_access_token_12345",
    "expires_in": 3600,
    "refresh_token": "1//mock_refresh_token_67890",
    "scope": "https://www.googleapis.com/auth/drive.readonly",
    "token_type": "Bearer",
}

# Sample user info response
USER_INFO_RESPONSE = {
    "id": "123456789",
    "email": "test@example.com",
    "verified_email": True,
    "name": "Test User",
    "picture": "https://lh3.googleusercontent.com/a/test",
    "given_name": "Test",
    "family_name": "User",
}

