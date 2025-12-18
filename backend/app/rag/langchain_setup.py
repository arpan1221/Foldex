"""LangChain document loaders and text splitters for intelligent chunking."""

from typing import List, Optional, Dict, Any
from pathlib import Path
import structlog

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document
    from langchain_community.document_loaders import (
        PyPDFLoader,
        TextLoader,
        UnstructuredFileLoader,
    )
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback for older LangChain versions
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain.schema import Document
        from langchain.document_loaders import (
            PyPDFLoader,
            TextLoader,
            UnstructuredFileLoader,
        )
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        RecursiveCharacterTextSplitter = None
        Document = None
        PyPDFLoader = None
        TextLoader = None
        UnstructuredFileLoader = None

from app.config.settings import settings
from app.core.exceptions import DocumentProcessingError

logger = structlog.get_logger(__name__)


class LangChainTextSplitter:
    """Wrapper for LangChain's RecursiveCharacterTextSplitter with custom configuration."""

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[List[str]] = None,
    ):
        """Initialize text splitter.

        Args:
            chunk_size: Maximum size of chunks (default: from settings)
            chunk_overlap: Overlap between chunks (default: from settings)
            separators: List of separators to use for splitting (default: LangChain defaults)
        """
        if not LANGCHAIN_AVAILABLE:
            raise DocumentProcessingError(
                "", "LangChain is not installed. Install with: pip install langchain"
            )

        self.chunk_size = chunk_size or settings.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or settings.CHUNK_OVERLAP

        # Default separators prioritize semantic boundaries
        default_separators = [
            "\n\n",  # Paragraph breaks
            "\n",  # Line breaks
            ". ",  # Sentence endings
            " ",  # Word boundaries
            "",  # Character boundaries
        ]

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=separators or default_separators,
            length_function=len,
            is_separator_regex=False,
        )

        logger.debug(
            "Initialized LangChain text splitter",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def split_text(self, text: str) -> List[str]:
        """Split text into chunks.

        Args:
            text: Text to split

        Returns:
            List of text chunks
        """
        return self.splitter.split_text(text)

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks while preserving metadata.

        Args:
            documents: List of LangChain Document objects

        Returns:
            List of chunked Document objects with preserved metadata
        """
        return self.splitter.split_documents(documents)

    def create_documents(
        self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[Document]:
        """Create Document objects from texts with optional metadata.

        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dictionaries (one per text)

        Returns:
            List of Document objects
        """
        return self.splitter.create_documents(texts, metadatas=metadatas)


class LangChainDocumentLoader:
    """Factory for creating LangChain document loaders based on file type."""

    @staticmethod
    def get_loader(file_path: str, mime_type: Optional[str] = None) -> Any:
        """Get appropriate LangChain loader for file type.

        Args:
            file_path: Path to file
            mime_type: Optional MIME type for faster detection

        Returns:
            LangChain document loader instance

        Raises:
            DocumentProcessingError: If LangChain not available or unsupported file type
        """
        if not LANGCHAIN_AVAILABLE:
            raise DocumentProcessingError(
                file_path, "LangChain is not installed. Install with: pip install langchain"
            )

        path = Path(file_path)
        extension = path.suffix.lower()

        # Determine file type
        if mime_type:
            if mime_type == "application/pdf":
                return PyPDFLoader(file_path)
            elif mime_type.startswith("text/"):
                return TextLoader(file_path, encoding="utf-8")
        elif extension == ".pdf":
            return PyPDFLoader(file_path)
        elif extension in [".txt", ".md", ".markdown", ".csv"]:
            return TextLoader(file_path, encoding="utf-8")
        else:
            # Fallback to unstructured loader
            try:
                return UnstructuredFileLoader(file_path)
            except Exception as e:
                logger.warning(
                    "Unstructured loader failed, trying text loader",
                    file_path=file_path,
                    error=str(e),
                )
                return TextLoader(file_path, encoding="utf-8")

    @staticmethod
    async def load_document(file_path: str, mime_type: Optional[str] = None) -> List[Document]:
        """Load document using appropriate LangChain loader.

        Args:
            file_path: Path to file
            mime_type: Optional MIME type

        Returns:
            List of LangChain Document objects

        Raises:
            DocumentProcessingError: If loading fails
        """
        try:
            loader = LangChainDocumentLoader.get_loader(file_path, mime_type)
            documents = loader.load()

            # Add file metadata to each document
            file_metadata = {
                "source": str(file_path),
                "file_path": str(file_path),
            }

            for doc in documents:
                if doc.metadata:
                    doc.metadata.update(file_metadata)
                else:
                    doc.metadata = file_metadata

            logger.debug(
                "Loaded document with LangChain",
                file_path=file_path,
                document_count=len(documents),
            )

            return documents

        except Exception as e:
            logger.error(
                "Failed to load document with LangChain",
                file_path=file_path,
                error=str(e),
                exc_info=True,
            )
            raise DocumentProcessingError(file_path, f"Failed to load document: {str(e)}") from e


def create_text_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> LangChainTextSplitter:
    """Factory function to create a text splitter instance.

    Args:
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks

    Returns:
        LangChainTextSplitter instance
    """
    return LangChainTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def is_langchain_available() -> bool:
    """Check if LangChain is available.

    Returns:
        True if LangChain is installed
    """
    return LANGCHAIN_AVAILABLE

