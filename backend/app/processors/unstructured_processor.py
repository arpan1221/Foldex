"""Unstructured.io processor for advanced document parsing with OCR support.

This processor uses Unstructured.io to extract structured content from
various document formats including PDFs, Office documents, and images.
Supports multiple processing strategies and OCR for scanned documents.
"""

from typing import List, Optional, Callable, Dict, Any, Union
from pathlib import Path
import time
import uuid
import structlog

try:
    from unstructured.partition.auto import partition
    from unstructured.chunking.title import chunk_by_title
    from unstructured.chunking.basic import chunk_elements
    UNSTRUCTURED_AVAILABLE = True
except ImportError:
    UNSTRUCTURED_AVAILABLE = False
    partition = None
    chunk_by_title = None
    chunk_elements = None

from app.processors.base import BaseProcessor
from app.models.documents import DocumentChunk
from app.core.exceptions import DocumentProcessingError
from app.config.settings import settings
from app.ingestion.metadata_schema import MetadataBuilder, FileType, ChunkType

logger = structlog.get_logger(__name__)


class UnstructuredProcessor(BaseProcessor):
    """Processor using Unstructured.io for advanced document parsing.
    
    Supports multiple processing strategies:
    - fast: Quick processing with basic extraction
    - hi_res: High-resolution processing for complex layouts
    - ocr_only: OCR-only mode for scanned documents
    
    Handles PDFs, Office documents, images, and other formats supported
    by Unstructured.io with local-first processing.
    """
    
    def __init__(
        self,
        strategy: str = "fast",
        enable_ocr: bool = True,
        chunking_strategy: str = "title",
    ):
        """Initialize Unstructured processor.
        
        Args:
            strategy: Processing strategy ("fast", "hi_res", "ocr_only")
            enable_ocr: Enable OCR for images and scanned documents
            chunking_strategy: Chunking strategy ("title", "basic", "none")
            
        Raises:
            DocumentProcessingError: If Unstructured.io is not available
        """
        super().__init__()
        
        if not UNSTRUCTURED_AVAILABLE:
            raise DocumentProcessingError(
                "",
                "Unstructured.io is not installed. Install with: pip install unstructured[all-docs]"
            )
        
        self.strategy = strategy
        self.enable_ocr = enable_ocr and settings.ENABLE_OCR
        self.chunking_strategy = chunking_strategy
        
        # Processing statistics
        self._stats: Dict[str, Any] = {
            "total_elements": 0,
            "processing_time_seconds": 0.0,
            "file_size_bytes": 0,
            "elements_by_type": {},
        }
        
        logger.info(
            "Initialized UnstructuredProcessor",
            strategy=strategy,
            enable_ocr=self.enable_ocr,
            chunking_strategy=chunking_strategy,
        )
    
    async def can_process(self, file_path: str, mime_type: Optional[str] = None) -> bool:
        """Check if processor can handle the file type.
        
        Args:
            file_path: Path to the file
            mime_type: Optional MIME type for faster detection
            
        Returns:
            True if processor can handle the file
        """
        if not UNSTRUCTURED_AVAILABLE:
            return False
        
        path_obj = Path(file_path)
        extension = path_obj.suffix.lower()
        
        # Supported extensions for Unstructured.io
        supported_extensions = {
            # Documents
            ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
            ".odt", ".ods", ".odp", ".rtf",
            # Text and structured data
            ".txt", ".md", ".html", ".htm", ".xml", ".csv",
            # Images (for OCR)
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp",
        }
        
        if extension in supported_extensions:
            return True
        
        # Check MIME type
        if mime_type:
            supported_mime_prefixes = [
                "application/pdf",
                "application/msword",
                "application/vnd.openxmlformats-officedocument",
                "application/vnd.oasis.opendocument",
                "text/",
                "image/",
            ]
            return any(mime_type.startswith(prefix) for prefix in supported_mime_prefixes)
        
        return False
    
    async def process(
        self,
        file_path: str,
        file_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[DocumentChunk]:
        """Process document using Unstructured.io and return chunks.
        
        Args:
            file_path: Path to the file to process
            file_id: Optional file identifier for chunk IDs
            metadata: Optional file metadata to include in chunks
            progress_callback: Optional callback for progress updates (0.0 to 1.0)
            
        Returns:
            List of document chunks with structured metadata
            
        Raises:
            DocumentProcessingError: If processing fails
        """
        if not UNSTRUCTURED_AVAILABLE:
            raise DocumentProcessingError(
                file_path,
                "Unstructured.io is not installed"
            )
        
        start_time = time.time()
        path_obj = Path(file_path)
        
        # Reset statistics
        self._stats = {
            "total_elements": 0,
            "processing_time_seconds": 0.0,
            "file_size_bytes": 0,
            "elements_by_type": {},
        }
        
        try:
            # Validate file exists
            if not path_obj.exists():
                raise DocumentProcessingError(file_path, "File not found")
            
            # Get file size
            file_size = path_obj.stat().st_size
            self._stats["file_size_bytes"] = file_size
            
            logger.info(
                "Processing document with Unstructured.io",
                file_path=file_path,
                file_id=file_id,
                strategy=self.strategy,
                enable_ocr=self.enable_ocr,
                file_size_bytes=file_size,
            )
            
            if progress_callback:
                progress_callback(0.1)
            
            # Prepare partition parameters
            partition_kwargs: Dict[str, Any] = {
                "strategy": self.strategy,
            }
            
            # Add OCR settings if enabled
            if self.enable_ocr:
                partition_kwargs["ocr_languages"] = ["eng"]  # Default to English
                # For images and PDFs, enable OCR
                if path_obj.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"]:
                    partition_kwargs["strategy"] = "ocr_only"
                elif path_obj.suffix.lower() == ".pdf" and self.strategy == "ocr_only":
                    partition_kwargs["strategy"] = "ocr_only"
            
            if progress_callback:
                progress_callback(0.2)
            
            # Partition document
            try:
                elements = partition(str(path_obj), **partition_kwargs)
            except Exception as e:
                # Handle corrupted files
                if "corrupted" in str(e).lower() or "invalid" in str(e).lower():
                    raise DocumentProcessingError(
                        file_path,
                        f"File appears to be corrupted: {str(e)}"
                    )
                # Handle OCR failures
                if "ocr" in str(e).lower() or "tesseract" in str(e).lower():
                    # For images, OCR is required - cannot fallback
                    is_image = path_obj.suffix.lower() in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"]
                    if is_image:
                        raise DocumentProcessingError(
                            file_path,
                            f"OCR is required for image files but failed: {str(e)}"
                        )
                    
                    logger.warning(
                        "OCR processing failed, attempting without OCR",
                        file_path=file_path,
                        error=str(e),
                    )
                    # Retry without OCR for non-image files
                    partition_kwargs.pop("ocr_languages", None)
                    if partition_kwargs.get("strategy") == "ocr_only":
                        partition_kwargs["strategy"] = "fast"
                    try:
                        elements = partition(str(path_obj), **partition_kwargs)
                    except Exception as retry_error:
                        raise DocumentProcessingError(
                            file_path,
                            f"OCR processing failed and fallback failed: {str(retry_error)}"
                        )
                else:
                    raise DocumentProcessingError(
                        file_path,
                        f"Unstructured.io processing failed: {str(e)}"
                    )
            
            if progress_callback:
                progress_callback(0.6)
            
            # Convert elements to chunks
            chunks = await self._elements_to_chunks(
                elements=elements,
                file_path=file_path,
                file_id=file_id or str(uuid.uuid4()),
                metadata=metadata or {},
                progress_callback=progress_callback,
            )
            
            # Update statistics
            processing_time = time.time() - start_time
            self._stats["total_elements"] = len(elements)
            self._stats["processing_time_seconds"] = processing_time
            
            # Count elements by type
            for element in elements:
                element_type = getattr(element, "category", "unknown")
                self._stats["elements_by_type"][element_type] = (
                    self._stats["elements_by_type"].get(element_type, 0) + 1
                )
            
            if progress_callback:
                progress_callback(1.0)
            
            logger.info(
                "Document processing completed",
                file_path=file_path,
                file_id=file_id,
                chunk_count=len(chunks),
                element_count=len(elements),
                processing_time_seconds=round(processing_time, 2),
            )
            
            return chunks
            
        except DocumentProcessingError:
            raise
        except Exception as e:
            logger.error(
                "Unstructured.io processing failed",
                file_path=file_path,
                error=str(e),
                exc_info=True,
            )
            raise DocumentProcessingError(
                file_path,
                f"Unstructured.io processing failed: {str(e)}"
            ) from e
    
    def _determine_file_type(self, file_path: str, mime_type: Optional[str] = None) -> FileType:
        """Determine FileType from file path and mime type.
        
        Args:
            file_path: Path to file
            mime_type: Optional MIME type
            
        Returns:
            FileType enum value
        """
        path_obj = Path(file_path)
        extension = path_obj.suffix.lower()
        
        # Check by extension first
        if extension in [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp"]:
            return FileType.IMAGE
        elif extension == ".html" or extension == ".htm":
            return FileType.HTML
        elif extension in [".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".ods", ".odp"]:
            return FileType.DOCUMENT
        elif extension == ".pdf":
            return FileType.PDF
        elif extension in [".txt", ".csv"]:
            return FileType.TEXT
        elif extension in [".md", ".markdown"]:
            return FileType.MARKDOWN
        
        # Check by MIME type if provided
        if mime_type:
            if mime_type.startswith("image/"):
                return FileType.IMAGE
            elif mime_type.startswith("text/html"):
                return FileType.HTML
            elif mime_type.startswith("application/vnd.openxmlformats-officedocument") or \
                 mime_type.startswith("application/msword") or \
                 mime_type.startswith("application/vnd.oasis.opendocument"):
                return FileType.DOCUMENT
            elif mime_type == "application/pdf":
                return FileType.PDF
            elif mime_type.startswith("text/"):
                return FileType.TEXT
        
        return FileType.UNKNOWN
    
    def _element_type_to_chunk_type(self, element_type: str, file_type: FileType) -> ChunkType:
        """Map Unstructured.io element type to ChunkType.
        
        Args:
            element_type: Unstructured.io element category
            file_type: FileType of the source file
            
        Returns:
            ChunkType enum value
        """
        # For images, use image-specific chunk types
        if file_type == FileType.IMAGE:
            element_lower = element_type.lower()
            if "title" in element_lower or "heading" in element_lower:
                return ChunkType.IMAGE_OCR_TEXT
            else:
                return ChunkType.IMAGE_OCR_TEXT
        
        # For documents, map element types
        element_lower = element_type.lower()
        if "title" in element_lower or "heading" in element_lower:
            return ChunkType.DOCUMENT_TITLE
        elif "table" in element_lower:
            return ChunkType.DOCUMENT_TABLE
        elif "list" in element_lower:
            return ChunkType.DOCUMENT_LIST_ITEM
        elif "summary" in element_lower:
            return ChunkType.DOCUMENT_SUMMARY
        else:
            return ChunkType.DOCUMENT_SECTION
    
    def _group_ocr_elements_spatially(
        self, 
        elements: List[Any], 
        max_distance: float = 50.0
    ) -> List[List[Any]]:
        """Group OCR elements by spatial proximity.
        
        Elements are grouped if they are within max_distance pixels of each other
        (using bounding box coordinates). This creates more meaningful chunks
        from OCR results.
        
        Args:
            elements: List of Unstructured.io elements with coordinates
            max_distance: Maximum distance (in pixels) for grouping
            
        Returns:
            List of element groups
        """
        if not elements:
            return []
        
        groups: List[List[Any]] = []
        
        for element in elements:
            # Get coordinates if available
            coords = None
            if hasattr(element, "metadata") and element.metadata:
                if hasattr(element.metadata, "coordinates"):
                    coords = element.metadata.coordinates
                elif hasattr(element.metadata, "get"):
                    coords = element.metadata.get("coordinates")
            
            if not coords:
                # No coordinates, add as standalone group
                groups.append([element])
                continue
            
            # Extract bounding box center and size
            try:
                if hasattr(coords, "points"):
                    points = coords.points
                    if points and len(points) >= 2:
                        # Calculate center and approximate size
                        x_coords = [p[0] for p in points]
                        y_coords = [p[1] for p in points]
                        center_x = sum(x_coords) / len(x_coords)
                        center_y = sum(y_coords) / len(y_coords)
                        width = max(x_coords) - min(x_coords)
                        
                        # Try to find a nearby group
                        added_to_group = False
                        for group in groups:
                            # Check if any element in group is nearby
                            for group_elem in group:
                                group_coords = None
                                if hasattr(group_elem, "metadata") and group_elem.metadata:
                                    if hasattr(group_elem.metadata, "coordinates"):
                                        group_coords = group_elem.metadata.coordinates
                                    elif hasattr(group_elem.metadata, "get"):
                                        group_coords = group_elem.metadata.get("coordinates")
                                
                                if group_coords and hasattr(group_coords, "points"):
                                    group_points = group_coords.points
                                    if group_points and len(group_points) >= 2:
                                        g_x_coords = [p[0] for p in group_points]
                                        g_y_coords = [p[1] for p in group_points]
                                        g_center_x = sum(g_x_coords) / len(g_x_coords)
                                        g_center_y = sum(g_y_coords) / len(g_y_coords)
                                        
                                        # Calculate distance
                                        distance = ((center_x - g_center_x) ** 2 + (center_y - g_center_y) ** 2) ** 0.5
                                        
                                        # Also check if elements are on same horizontal/vertical line
                                        if distance <= max_distance or \
                                           (abs(center_y - g_center_y) < max_distance and abs(center_x - g_center_x) < width + max(max(g_x_coords) - min(g_x_coords), 0)):
                                            group.append(element)
                                            added_to_group = True
                                            break
                            
                            if added_to_group:
                                break
                        
                        if not added_to_group:
                            groups.append([element])
                    else:
                        groups.append([element])
                else:
                    groups.append([element])
            except Exception:
                # If coordinate parsing fails, add as standalone
                groups.append([element])
        
        return groups
    
    def _create_image_summary_chunk(
        self,
        grouped_elements: List[List[Any]],
        file_path: str,
        file_id: str,
        metadata: Dict[str, Any],
        chunks: List[DocumentChunk],
    ) -> None:
        """Create a summary chunk describing the overall image content.
        
        Args:
            grouped_elements: List of element groups from OCR
            file_path: Source file path
            file_id: File identifier
            metadata: Base metadata dictionary
            chunks: List to append the summary chunk to
        """
        # Collect all text from elements
        all_texts = []
        for group in grouped_elements:
            for element in group:
                text = str(getattr(element, "text", "")).strip()
                if text:
                    all_texts.append(text)
        
        if not all_texts:
            return
        
        # Create a summary description
        summary_parts = ["Image content from " + Path(file_path).name + ":"]
        summary_parts.extend(all_texts[:20])  # Limit to first 20 text elements
        if len(all_texts) > 20:
            summary_parts.append(f"... and {len(all_texts) - 20} more text elements")
        
        summary_text = " ".join(summary_parts)
        
        # Generate chunk ID (use -1 to indicate summary chunk)
        chunk_id = self._generate_chunk_id(file_id, -1)
        
        # Determine file type
        file_type = self._determine_file_type(file_path, metadata.get("mime_type"))
        
        # Get file name and drive URL
        file_name = metadata.get("file_name", Path(file_path).name)
        drive_url = metadata.get("drive_url", metadata.get("web_view_link", ""))
        
        # Build base metadata
        base_meta = MetadataBuilder.base_metadata(
            file_id=file_id,
            file_name=file_name,
            file_type=file_type,
            chunk_type=ChunkType.IMAGE_SUMMARY,
            chunk_id=chunk_id,
            drive_url=drive_url,
        )
        
        # Add image-specific metadata
        final_metadata = MetadataBuilder.image_metadata(
            base=base_meta,
            is_summary=True,
            grouped_elements_count=len(grouped_elements),
        )
        
        # Merge with additional metadata
        final_metadata = MetadataBuilder.merge_metadata(final_metadata, metadata)
        
        # Create chunk
        chunk = DocumentChunk(
            chunk_id=chunk_id,
            content=summary_text,
            file_id=file_id,
            metadata=final_metadata,
            embedding=None,
        )
        
        chunks.append(chunk)
    
    def _enhance_short_content(
        self, 
        content: str, 
        file_name: str, 
        file_type: FileType, 
        chunk_type: ChunkType
    ) -> str:
        """Enhance short content chunks with context for better semantic matching.
        
        Args:
            content: Original chunk content
            file_name: Name of source file
            file_type: Type of file
            chunk_type: Type of chunk
            
        Returns:
            Enhanced content with context
        """
        # If content is already long enough, return as-is
        if len(content) > 100:
            return content
        
        # Add context based on file and chunk type
        if file_type == FileType.IMAGE and chunk_type == ChunkType.IMAGE_OCR_TEXT:
            return f"Text from image {file_name}: {content}"
        elif file_type == FileType.AUDIO and chunk_type == ChunkType.AUDIO_TRANSCRIPTION:
            return f"Audio transcription from {file_name}: {content}"
        elif file_type == FileType.AUDIO and chunk_type == ChunkType.AUDIO_SUMMARY:
            return f"Audio summary from {file_name}: {content}"
        else:
            return content
    
    async def _elements_to_chunks(
        self,
        elements: List[Any],
        file_path: str,
        file_id: str,
        metadata: Dict[str, Any],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[DocumentChunk]:
        """Convert Unstructured.io elements to DocumentChunk objects.
        
        Args:
            elements: List of Unstructured.io element objects
            file_path: Source file path
            file_id: File identifier
            metadata: Base metadata dictionary
            progress_callback: Optional progress callback
            
        Returns:
            List of DocumentChunk objects
        """
        chunks: List[DocumentChunk] = []
        
        # Determine file type
        file_type = self._determine_file_type(file_path, metadata.get("mime_type"))
        is_image = file_type == FileType.IMAGE
        
        # For images, group OCR elements spatially before processing
        if is_image:
            element_groups = self._group_ocr_elements_spatially(elements, max_distance=50.0)
            
            # Create summary chunk for image
            self._create_image_summary_chunk(
                grouped_elements=element_groups,
                file_path=file_path,
                file_id=file_id,
                metadata=metadata,
                chunks=chunks,
            )
            
            # Process each group as a combined chunk
            chunk_index = 0
            for group_idx, element_group in enumerate(element_groups):
                # Combine text from all elements in group
                combined_text = " ".join(
                    str(getattr(elem, "text", "")).strip() 
                    for elem in element_group 
                    if str(getattr(elem, "text", "")).strip()
                )
                
                if not combined_text:
                    continue
                
                # Use first element's type and metadata
                first_element = element_group[0]
                element_type = getattr(first_element, "category", "unknown")
                element_metadata = first_element.metadata if hasattr(first_element, "metadata") else None
                
                # Process as grouped chunk
                await self._create_chunk_from_text(
                    text=combined_text,
                    element_type=element_type,
                    element_metadata=element_metadata,
                    file_path=file_path,
                    file_id=file_id,
                    metadata=metadata,
                    element_index=group_idx,
                    chunk_index=chunk_index,
                    sub_chunk_index=None,
                    total_sub_chunks=1,
                    chunks=chunks,
                    file_type=file_type,
                    is_grouped=True,
                    grouped_count=len(element_group),
                )
                chunk_index += 1
        else:
            # For non-image files, process elements normally
            chunk_index = 0
            for idx, element in enumerate(elements):
                try:
                    # Extract element properties
                    element_text = str(getattr(element, "text", "")).strip()
                    if not element_text:
                        continue
                    
                    element_type = getattr(element, "category", "unknown")
                    
                    # Split large element text into smaller chunks if needed
                    # DocumentChunk has max_length=50000, so we split at 45000 to be safe
                    MAX_CHUNK_SIZE = 45000
                    if len(element_text) > MAX_CHUNK_SIZE:
                        # Split by newlines if possible (for CSV, TSV, etc.)
                        text_parts = []
                        if "\n" in element_text:
                            lines = element_text.split("\n")
                            current_chunk = ""
                            for line in lines:
                                if len(current_chunk) + len(line) + 1 > MAX_CHUNK_SIZE and current_chunk:
                                    text_parts.append(current_chunk)
                                    current_chunk = line + "\n"
                                else:
                                    current_chunk += line + "\n"
                            if current_chunk:
                                text_parts.append(current_chunk)
                        else:
                            # No newlines, split by character count
                            for i in range(0, len(element_text), MAX_CHUNK_SIZE):
                                text_parts.append(element_text[i:i + MAX_CHUNK_SIZE])
                        
                        # Process each text part as a separate chunk
                        for part_idx, text_part in enumerate(text_parts):
                            if not text_part.strip():
                                continue
                            
                            await self._create_chunk_from_text(
                                text=text_part,
                                element_type=element_type,
                                element_metadata=element.metadata if hasattr(element, "metadata") else None,
                                file_path=file_path,
                                file_id=file_id,
                                metadata=metadata,
                                element_index=idx,
                                chunk_index=chunk_index,
                                sub_chunk_index=part_idx,
                                total_sub_chunks=len(text_parts),
                                chunks=chunks,
                                file_type=file_type,
                                is_grouped=False,
                                grouped_count=1,
                            )
                            chunk_index += 1
                    else:
                        # Element fits in one chunk
                        await self._create_chunk_from_text(
                            text=element_text,
                            element_type=element_type,
                            element_metadata=element.metadata if hasattr(element, "metadata") else None,
                            file_path=file_path,
                            file_id=file_id,
                            metadata=metadata,
                            element_index=idx,
                            chunk_index=chunk_index,
                            sub_chunk_index=None,
                            total_sub_chunks=1,
                            chunks=chunks,
                            file_type=file_type,
                            is_grouped=False,
                            grouped_count=1,
                        )
                        chunk_index += 1
                        
                except Exception as e:
                    logger.warning(
                        "Failed to convert element to chunk",
                        element_index=idx,
                        file_path=file_path,
                        error=str(e),
                    )
                    continue
        
        return chunks
    
    async def _create_chunk_from_text(
        self,
        text: str,
        element_type: str,
        element_metadata: Optional[Any],
        file_path: str,
        file_id: str,
        metadata: Dict[str, Any],
        element_index: int,
        chunk_index: int,
        sub_chunk_index: Optional[int],
        total_sub_chunks: int,
        chunks: List[DocumentChunk],
        file_type: FileType,
        is_grouped: bool = False,
        grouped_count: int = 1,
    ) -> None:
        """Create a DocumentChunk from text, handling metadata extraction.
        
        Args:
            text: Text content for the chunk (must be <= 50000 chars)
            element_type: Type of element (category)
            element_metadata: Raw element metadata object
            file_path: Source file path
            file_id: File identifier
            metadata: Base metadata dictionary
            element_index: Index of the source element
            chunk_index: Index of this chunk (across all chunks)
            sub_chunk_index: Index within split chunks (None if not split)
            total_sub_chunks: Total number of sub-chunks (1 if not split)
            chunks: List to append the created chunk to
        """
        path_obj = Path(file_path)
        
        # Get file name and drive URL
        file_name = metadata.get("file_name", path_obj.name)
        drive_url = metadata.get("drive_url", metadata.get("web_view_link", ""))
        
        # Generate chunk ID (include sub-chunk index if split)
        # For sub-chunks, create a compound index: element_index * 1000 + sub_chunk_index
        if sub_chunk_index is not None:
            compound_index = element_index * 1000 + sub_chunk_index
            chunk_id = self._generate_chunk_id(file_id, compound_index)
        else:
            chunk_id = self._generate_chunk_id(file_id, element_index)
        
        # Map element type to chunk type
        chunk_type = self._element_type_to_chunk_type(element_type, file_type)
        
        # Get element metadata - convert ElementMetadata object to dict
        element_metadata_dict = {}
        page_number = None
        coordinates = None
        if element_metadata:
            try:
                # Try to convert ElementMetadata to dict
                if hasattr(element_metadata, "__dict__"):
                    element_metadata_dict = element_metadata.__dict__.copy()
                elif hasattr(element_metadata, "dict"):
                    element_metadata_dict = element_metadata.dict()
                else:
                    # Try dict() constructor
                    element_metadata_dict = dict(element_metadata)
            except Exception:
                # If conversion fails, try accessing attributes directly
                element_metadata_dict = {}
                if hasattr(element_metadata, "page_number"):
                    page_number = element_metadata.page_number
                if hasattr(element_metadata, "coordinates"):
                    coordinates = element_metadata.coordinates
            
            # Extract page_number and coordinates from metadata dict if available
            if page_number is None:
                page_number = element_metadata_dict.get("page_number")
            if coordinates is None:
                coordinates = element_metadata_dict.get("coordinates")
        
        # Build base metadata using MetadataBuilder
        base_meta = MetadataBuilder.base_metadata(
            file_id=file_id,
            file_name=file_name,
            file_type=file_type,
            chunk_type=chunk_type,
            chunk_id=chunk_id,
            drive_url=drive_url,
        )
        
        # Add type-specific metadata
        if file_type == FileType.IMAGE:
            final_metadata = MetadataBuilder.image_metadata(
                base=base_meta,
                page_number=page_number,
                coordinates=coordinates,
                is_summary=False,
                grouped_elements_count=grouped_count if is_grouped else None,
            )
        elif file_type == FileType.PDF:
            # For PDFs processed by UnstructuredProcessor, use document metadata
            final_metadata = MetadataBuilder.document_metadata(
                base=base_meta,
                page_number=page_number,
                section=element_metadata_dict.get("section", ""),
                element_type=element_type,
            )
        else:
            # For other document types (DOCX, HTML, etc.)
            final_metadata = MetadataBuilder.document_metadata(
                base=base_meta,
                page_number=page_number,
                section=element_metadata_dict.get("section", ""),
                element_type=element_type,
            )
        
        # Add sub-chunk info if this was split
        if sub_chunk_index is not None and total_sub_chunks > 1:
            final_metadata["chunk_index"] = sub_chunk_index
            final_metadata["total_chunks"] = total_sub_chunks
        
        # Merge element-specific metadata (only simple types, preserving standardized fields)
        if element_metadata_dict:
            filtered_metadata = self._filter_metadata_for_chunks(element_metadata_dict)
            # Add filtered metadata, but don't overwrite standardized fields
            for k, v in filtered_metadata.items():
                if k not in ["text", "category", "page_number", "file_id", "file_name"]:
                    if k not in final_metadata:  # Don't overwrite standardized fields
                        final_metadata[k] = v
        
        # Preserve source_file and source_path for backward compatibility
        final_metadata["source_file"] = str(path_obj.name)
        final_metadata["source_path"] = str(path_obj)
        final_metadata["element_type"] = element_type  # Preserve for backward compatibility
        
        # Enhance short content with context
        enhanced_text = self._enhance_short_content(text, file_name, file_type, chunk_type)
        
        # Create DocumentChunk
        chunk = DocumentChunk(
            chunk_id=chunk_id,
            content=enhanced_text,
            file_id=file_id,
            metadata=final_metadata,
            embedding=None,
        )
        
        chunks.append(chunk)
                
    def _filter_metadata_for_chunks(self, metadata: Dict[str, Any]) -> Dict[str, Union[str, int, float, bool]]:
        """Filter metadata to only include simple types compatible with DocumentChunk.
        
        DocumentChunk metadata must be Dict[str, Union[str, int, float, bool]].
        This method converts complex types (lists, sets, dicts) to strings or excludes them.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Filtered metadata with only simple types
        """
        filtered: Dict[str, Union[str, int, float, bool]] = {}
        
        for key, value in metadata.items():
            # Skip private/internal attributes
            if key.startswith("_"):
                continue
        
            # Accept simple types directly
            if isinstance(value, (str, int, float, bool)):
                filtered[key] = value
            elif isinstance(value, (list, tuple, set, frozenset)):
                # Convert sequences to comma-separated string
                try:
                    value_list = list(value)[:10]  # Limit to first 10 items
                    filtered[key] = ", ".join(str(item) for item in value_list)
                except Exception:
                    pass  # Skip if conversion fails
            elif value is None:
                continue  # Skip None values
            else:
                # Convert other types to string
                try:
                    filtered[key] = str(value)[:500]  # Limit string length
                except Exception:
                    pass  # Skip if conversion fails
        
        return filtered
    
    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions.
        
        Returns:
            List of supported extensions
        """
        return [
            # Documents
            ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
            ".odt", ".ods", ".odp", ".rtf",
            # Text
            ".txt", ".md", ".html", ".htm", ".xml",
            # Images (for OCR)
            ".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp",
        ]
    
    def get_supported_mime_types(self) -> List[str]:
        """Get supported MIME types.
        
        Returns:
            List of supported MIME types
        """
        return [
            # PDF
            "application/pdf",
            # Microsoft Office
            "application/msword",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            # OpenDocument
            "application/vnd.oasis.opendocument.text",
            "application/vnd.oasis.opendocument.spreadsheet",
            "application/vnd.oasis.opendocument.presentation",
            # Text and structured data
            "text/plain", "text/markdown", "text/html", "text/xml", "text/csv",
            # Images
            "image/jpeg", "image/png", "image/gif", "image/bmp", "image/tiff", "image/webp",
        ]
    
    async def process_document(
        self,
        file_path: Path,
        file_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> List[DocumentChunk]:
        """Process a document and return chunks (convenience method).
        
        This is a convenience wrapper around the process() method that
        accepts a Path object and provides a simpler interface.
        
        Args:
            file_path: Path to the file to process
            file_id: Optional file identifier for chunk IDs
            metadata: Optional file metadata to include in chunks
            
        Returns:
            List of document chunks with structured metadata
            
        Raises:
            DocumentProcessingError: If processing fails
            
        Example:
            >>> processor = UnstructuredProcessor(strategy="hi_res", enable_ocr=True)
            >>> chunks = await processor.process_document(
            ...     Path("document.pdf"),
            ...     file_id="file_123",
            ...     metadata={"folder_id": "folder_456"}
            ... )
            >>> print(f"Extracted {len(chunks)} chunks")
        """
        return await self.process(
            file_path=str(file_path),
            file_id=file_id,
            metadata=metadata,
        )
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """Get processing statistics from last operation.
        
        Returns:
            Dictionary with processing statistics:
            - total_elements: Number of elements extracted
            - processing_time_seconds: Time taken to process
            - file_size_bytes: Size of processed file
            - elements_by_type: Count of elements by type
            
        Example:
            >>> processor = UnstructuredProcessor()
            >>> chunks = await processor.process("document.pdf")
            >>> stats = processor.get_processing_stats()
            >>> print(f"Processed {stats['total_elements']} elements in {stats['processing_time_seconds']:.2f}s")
        """
        return self._stats.copy()

