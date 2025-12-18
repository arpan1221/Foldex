"""LangChain-based document processors for intelligent chunking and processing."""

from typing import List, Optional, Callable, Dict, Any
from pathlib import Path
import uuid
import structlog

from app.processors.base import BaseProcessor
from app.models.documents import DocumentChunk
from app.core.exceptions import DocumentProcessingError
from app.rag.langchain_setup import (
    LangChainDocumentLoader,
    LangChainTextSplitter,
    create_text_splitter,
    is_langchain_available,
)

try:
    from langchain_core.documents import Document
    LANGCHAIN_AVAILABLE = True
except ImportError:
    # Fallback for older LangChain versions
    try:
        from langchain.schema import Document
        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        Document = None

logger = structlog.get_logger(__name__)


class LangChainPDFProcessor(BaseProcessor):
    """PDF processor using LangChain document loaders and text splitters."""

    def __init__(self):
        """Initialize LangChain PDF processor."""
        super().__init__()
        if not is_langchain_available():
            logger.warning("LangChain not available, PDF processing will fail")
        self.text_splitter = create_text_splitter()

    async def can_process(self, file_path: str, mime_type: Optional[str] = None) -> bool:
        """Check if file is a PDF.

        Args:
            file_path: Path to file
            mime_type: Optional MIME type

        Returns:
            True if file is PDF
        """
        if mime_type:
            return mime_type == "application/pdf"
        return file_path.lower().endswith(".pdf")

    async def process(
        self,
        file_path: str,
        file_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[DocumentChunk]:
        """Process PDF using LangChain loaders and splitters.

        Args:
            file_path: Path to PDF file
            file_id: Optional file identifier
            metadata: Optional file metadata
            progress_callback: Optional progress callback

        Returns:
            List of document chunks with page metadata

        Raises:
            DocumentProcessingError: If processing fails
        """
        if not is_langchain_available():
            raise DocumentProcessingError(
                file_path, "LangChain is not installed"
            )

        if not Path(file_path).exists():
            raise DocumentProcessingError(file_path, "File not found")

        try:
            file_id = file_id or str(uuid.uuid4())
            metadata = metadata or {}

            logger.info("Processing PDF with LangChain", file_path=file_path)

            # Load PDF documents with LangChain
            if progress_callback:
                progress_callback(0.1)

            documents = await LangChainDocumentLoader.load_document(
                file_path, mime_type="application/pdf"
            )

            if progress_callback:
                progress_callback(0.3)

            # Split documents into chunks while preserving metadata
            chunked_documents = self.text_splitter.split_documents(documents)

            if progress_callback:
                progress_callback(0.7)

            # Convert LangChain documents to DocumentChunk objects
            chunks = []
            for idx, doc in enumerate(chunked_documents):
                chunk_id = f"{file_id}_chunk_{idx}"

                # Extract page number from metadata if available
                page_number = doc.metadata.get("page", None)
                if page_number is None:
                    # Try to extract from source metadata
                    source = doc.metadata.get("source", "")
                    if "page" in source.lower():
                        # Extract page number from source string
                        try:
                            page_number = int(
                                source.split("page")[-1].split(".")[0].strip()
                            )
                        except (ValueError, IndexError):
                            page_number = None

                chunk_metadata = {
                    "chunk_index": idx,
                    "page_number": page_number,
                    "file_name": metadata.get("file_name", Path(file_path).name),
                    "mime_type": "application/pdf",
                    **doc.metadata,
                }

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=doc.page_content,
                    file_id=file_id,
                    metadata=chunk_metadata,
                )

                chunks.append(chunk)

            if progress_callback:
                progress_callback(1.0)

            logger.info(
                "PDF processed with LangChain",
                file_path=file_path,
                chunk_count=len(chunks),
            )

            return chunks

        except Exception as e:
            logger.error(
                "PDF processing failed with LangChain",
                file_path=file_path,
                error=str(e),
                exc_info=True,
            )
            raise DocumentProcessingError(file_path, str(e)) from e

    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions.

        Returns:
            List of supported extensions
        """
        return [".pdf"]


class LangChainTextProcessor(BaseProcessor):
    """Text file processor using LangChain document loaders and text splitters."""

    def __init__(self):
        """Initialize LangChain text processor."""
        super().__init__()
        if not is_langchain_available():
            logger.warning("LangChain not available, text processing will fail")
        self.text_splitter = create_text_splitter()

    async def can_process(self, file_path: str, mime_type: Optional[str] = None) -> bool:
        """Check if file is a text file.

        Args:
            file_path: Path to file
            mime_type: Optional MIME type

        Returns:
            True if file is text
        """
        text_extensions = [".txt", ".md", ".markdown", ".csv", ".json", ".xml"]
        if mime_type:
            return mime_type.startswith("text/")
        return any(file_path.lower().endswith(ext) for ext in text_extensions)

    async def process(
        self,
        file_path: str,
        file_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[DocumentChunk]:
        """Process text file using LangChain loaders and splitters.

        Args:
            file_path: Path to text file
            file_id: Optional file identifier
            metadata: Optional file metadata
            progress_callback: Optional progress callback

        Returns:
            List of document chunks

        Raises:
            DocumentProcessingError: If processing fails
        """
        if not is_langchain_available():
            raise DocumentProcessingError(
                file_path, "LangChain is not installed"
            )

        if not Path(file_path).exists():
            raise DocumentProcessingError(file_path, "File not found")

        try:
            file_id = file_id or str(uuid.uuid4())
            metadata = metadata or {}

            logger.info("Processing text file with LangChain", file_path=file_path)

            # Determine MIME type
            mime_type = metadata.get("mime_type", "text/plain")

            if progress_callback:
                progress_callback(0.2)

            # Load text document with LangChain
            documents = await LangChainDocumentLoader.load_document(
                file_path, mime_type=mime_type
            )

            if progress_callback:
                progress_callback(0.4)

            # Split documents into chunks
            chunked_documents = self.text_splitter.split_documents(documents)

            if progress_callback:
                progress_callback(0.8)

            # Convert LangChain documents to DocumentChunk objects
            chunks = []
            for idx, doc in enumerate(chunked_documents):
                chunk_id = f"{file_id}_chunk_{idx}"

                chunk_metadata = {
                    "chunk_index": idx,
                    "file_name": metadata.get("file_name", Path(file_path).name),
                    "mime_type": mime_type,
                    **doc.metadata,
                }

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=doc.page_content,
                    file_id=file_id,
                    metadata=chunk_metadata,
                )

                chunks.append(chunk)

            if progress_callback:
                progress_callback(1.0)

            logger.info(
                "Text file processed with LangChain",
                file_path=file_path,
                chunk_count=len(chunks),
            )

            return chunks

        except Exception as e:
            logger.error(
                "Text processing failed with LangChain",
                file_path=file_path,
                error=str(e),
                exc_info=True,
            )
            raise DocumentProcessingError(file_path, str(e)) from e

    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions.

        Returns:
            List of supported extensions
        """
        return [".txt", ".md", ".markdown", ".csv", ".json", ".xml"]


class LangChainAudioProcessor(BaseProcessor):
    """Audio processor that creates LangChain documents from transcripts."""

    def __init__(self):
        """Initialize LangChain audio processor."""
        super().__init__()
        if not is_langchain_available():
            logger.warning("LangChain not available, audio processing will fail")
        self.text_splitter = create_text_splitter()

    async def can_process(self, file_path: str, mime_type: Optional[str] = None) -> bool:
        """Check if file is an audio file.

        Args:
            file_path: Path to file
            mime_type: Optional MIME type

        Returns:
            True if file is audio
        """
        audio_extensions = [".mp3", ".wav", ".m4a", ".flac", ".ogg"]
        if mime_type:
            return mime_type.startswith("audio/")
        return any(file_path.lower().endswith(ext) for ext in audio_extensions)

    async def process(
        self,
        file_path: str,
        file_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[DocumentChunk]:
        """Process audio file by creating LangChain documents from transcript.

        Note: This processor expects a transcript file to exist alongside the audio.
        For full audio processing, integrate with Whisper in the calling code.

        Args:
            file_path: Path to audio file
            file_id: Optional file identifier
            metadata: Optional file metadata (should include transcript if available)
            progress_callback: Optional progress callback

        Returns:
            List of document chunks with temporal metadata

        Raises:
            DocumentProcessingError: If processing fails
        """
        if not is_langchain_available():
            raise DocumentProcessingError(
                file_path, "LangChain is not installed"
            )

        try:
            file_id = file_id or str(uuid.uuid4())
            metadata = metadata or {}

            logger.info("Processing audio file with LangChain", file_path=file_path)

            # Get transcript from metadata or try to load from file
            transcript = metadata.get("transcript", "")
            if not transcript:
                # Try to load transcript from .txt file with same name
                transcript_path = Path(file_path).with_suffix(".txt")
                if transcript_path.exists():
                    with open(transcript_path, "r", encoding="utf-8") as f:
                        transcript = f.read()

            if not transcript:
                raise DocumentProcessingError(
                    file_path, "No transcript available for audio file"
                )

            if progress_callback:
                progress_callback(0.3)

            # Create LangChain document from transcript
            langchain_doc = Document(
                page_content=transcript,
                metadata={
                    "source": str(file_path),
                    "file_path": str(file_path),
                    "file_name": metadata.get("file_name", Path(file_path).name),
                    "mime_type": metadata.get("mime_type", "audio/mpeg"),
                },
            )

            # Split transcript into chunks by speech segments
            chunked_documents = self.text_splitter.split_documents([langchain_doc])

            if progress_callback:
                progress_callback(0.7)

            # Convert to DocumentChunk objects with temporal metadata
            chunks = []
            for idx, doc in enumerate(chunked_documents):
                chunk_id = f"{file_id}_chunk_{idx}"

                # Extract timestamp information if available in metadata
                start_time = metadata.get("start_times", {}).get(idx, None)
                end_time = metadata.get("end_times", {}).get(idx, None)

                chunk_metadata = {
                    "chunk_index": idx,
                    "file_name": metadata.get("file_name", Path(file_path).name),
                    "mime_type": metadata.get("mime_type", "audio/mpeg"),
                    "start_time": start_time,
                    "end_time": end_time,
                    **doc.metadata,
                }

                chunk = DocumentChunk(
                    chunk_id=chunk_id,
                    content=doc.page_content,
                    file_id=file_id,
                    metadata=chunk_metadata,
                )

                chunks.append(chunk)

            if progress_callback:
                progress_callback(1.0)

            logger.info(
                "Audio file processed with LangChain",
                file_path=file_path,
                chunk_count=len(chunks),
            )

            return chunks

        except Exception as e:
            logger.error(
                "Audio processing failed with LangChain",
                file_path=file_path,
                error=str(e),
                exc_info=True,
            )
            raise DocumentProcessingError(file_path, str(e)) from e

    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions.

        Returns:
            List of supported extensions
        """
        return [".mp3", ".wav", ".m4a", ".flac", ".ogg"]

