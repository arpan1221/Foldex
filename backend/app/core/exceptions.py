"""Custom exception classes for Foldex application."""


class FoldexException(Exception):
    """Base exception for all Foldex errors."""

    def __init__(self, message: str, status_code: int = 500):
        """Initialize exception.

        Args:
            message: Error message
            status_code: HTTP status code
        """
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


class AuthenticationError(FoldexException):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed"):
        """Initialize authentication error."""
        super().__init__(message, status_code=401)


class ValidationError(FoldexException):
    """Raised when input validation fails."""

    def __init__(self, message: str = "Validation failed"):
        """Initialize validation error."""
        super().__init__(message, status_code=400)


class ProcessingError(FoldexException):
    """Raised when document processing fails."""

    def __init__(self, message: str, file_path: str | None = None):
        """Initialize processing error.

        Args:
            message: Error message
            file_path: Optional file path that caused the error
        """
        self.file_path = file_path
        super().__init__(message, status_code=500)


class DocumentProcessingError(ProcessingError):
    """Raised when a specific document fails to process."""

    def __init__(self, file_path: str, reason: str):
        """Initialize document processing error.

        Args:
            file_path: Path to the file that failed
            reason: Reason for failure
        """
        self.file_path = file_path
        self.reason = reason
        super().__init__(
            f"Failed to process {file_path}: {reason}",
            file_path=file_path,
        )


class DatabaseError(FoldexException):
    """Raised when database operations fail."""

    def __init__(self, message: str = "Database operation failed"):
        """Initialize database error."""
        super().__init__(message, status_code=500)


class VectorStoreError(FoldexException):
    """Raised when vector store operations fail."""

    def __init__(self, message: str = "Vector store operation failed"):
        """Initialize vector store error."""
        super().__init__(message, status_code=500)

