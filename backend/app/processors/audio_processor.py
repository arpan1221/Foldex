"""Audio file processor using Whisper for transcription.

Supports audio files (.m4a, .mp3, .wav, .flac, .ogg) and video files with
audio tracks (.mp4). Transcribes audio to text with timestamps for
citation-driven architecture.
"""

from typing import List, Optional, Callable, Dict, Any
import structlog
from pathlib import Path
import os
import tempfile
import subprocess

from app.processors.base import BaseProcessor
from app.models.documents import DocumentChunk
from app.core.exceptions import DocumentProcessingError
from app.config.settings import settings
from app.ingestion.metadata_schema import MetadataBuilder, FileType, ChunkType

logger = structlog.get_logger(__name__)

# Try to import Whisper
WHISPER_AVAILABLE = False
FASTER_WHISPER_AVAILABLE = False
whisper = None
WhisperModel = None

try:
    import whisper
    WHISPER_AVAILABLE = True
except ImportError:
    pass

# Try faster-whisper if regular whisper not available
if not WHISPER_AVAILABLE:
    try:
        from faster_whisper import WhisperModel
        FASTER_WHISPER_AVAILABLE = True
    except ImportError:
        pass


class AudioProcessor(BaseProcessor):
    """Processor for audio files using Whisper transcription."""

    def __init__(self):
        """Initialize audio processor."""
        self.model_name = settings.WHISPER_MODEL
        self.device = settings.WHISPER_DEVICE
        self.whisper_model = None
        self._model_lock = None
        
        # Initialize lock for thread safety
        import asyncio
        self._model_lock = asyncio.Lock()
    
    def _load_model(self):
        """Load Whisper model (lazy loading)."""
        if self.whisper_model is not None:
            return self.whisper_model
        
        if FASTER_WHISPER_AVAILABLE:
            logger.info("Loading faster-whisper model", model=self.model_name, device=self.device)
            self.whisper_model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type="int8" if self.device == "cpu" else "float16",
            )
            logger.info("faster-whisper model loaded successfully")
        elif WHISPER_AVAILABLE:
            logger.info("Loading openai-whisper model", model=self.model_name, device=self.device)
            self.whisper_model = whisper.load_model(self.model_name, device=self.device)
            logger.info("openai-whisper model loaded successfully")
        else:
            logger.warning(
                "Whisper not available. Install with: pip install openai-whisper or pip install faster-whisper"
            )
            self.whisper_model = None
        
        return self.whisper_model

    async def can_process(self, file_path: str, mime_type: Optional[str] = None) -> bool:
        """Check if file is an audio file or video file with audio track.

        Args:
            file_path: Path to file
            mime_type: Optional MIME type for faster detection

        Returns:
            True if file is audio or video with audio track
        """
        # Check MIME type first if provided
        if mime_type:
            audio_mime_types = self.get_supported_mime_types()
            if mime_type in audio_mime_types:
                return True
            # Also check for video MIME types that may contain audio
            if mime_type.startswith("video/"):
                return True
        
        # Fall back to extension check
        audio_extensions = self.get_supported_extensions()
        return any(file_path.lower().endswith(ext) for ext in audio_extensions)

    def _extract_audio_from_video(self, video_path: str) -> str:
        """Extract audio track from video file using ffmpeg.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Path to extracted audio file (temporary)
            
        Raises:
            DocumentProcessingError: If extraction fails
        """
        try:
            # Create temporary audio file
            temp_dir = Path(tempfile.gettempdir()) / "foldex_audio"
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            audio_path = temp_dir / f"{Path(video_path).stem}.wav"
            
            logger.info(
                "Extracting audio from video",
                video_path=video_path,
                audio_path=str(audio_path),
            )
            
            # Use ffmpeg to extract audio
            cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-vn",  # No video
                "-acodec", "pcm_s16le",  # PCM 16-bit
                "-ar", "16000",  # Sample rate for Whisper
                "-ac", "1",  # Mono
                "-y",  # Overwrite output
                str(audio_path),
            ]
            
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True,
                    timeout=300,  # 5 minute timeout
                )
                logger.info("Audio extraction completed", audio_path=str(audio_path))
                return str(audio_path)
            except subprocess.TimeoutExpired:
                raise DocumentProcessingError(
                    video_path,
                    "Audio extraction timed out (video file may be too large)"
                )
            except subprocess.CalledProcessError as e:
                error_output = e.stderr.lower()
                if "ffmpeg" in error_output and "not found" in error_output:
                    raise DocumentProcessingError(
                        video_path,
                        "ffmpeg not found. Install with: brew install ffmpeg (macOS) or "
                        "apt-get install ffmpeg (Linux)"
                    )
                elif "invalid" in error_output or "corrupted" in error_output:
                    raise DocumentProcessingError(
                        video_path,
                        f"Video file appears to be corrupted or invalid: {e.stderr}"
                    )
                else:
                    raise DocumentProcessingError(
                        video_path,
                        f"Audio extraction failed: {e.stderr}"
                    )
            except FileNotFoundError:
                raise DocumentProcessingError(
                    video_path,
                    "ffmpeg not found. Install with: brew install ffmpeg (macOS) or "
                    "apt-get install ffmpeg (Linux)"
                )
        except DocumentProcessingError:
            raise
        except Exception as e:
            logger.error(
                "Audio extraction failed",
                video_path=video_path,
                error=str(e),
                exc_info=True,
            )
            raise DocumentProcessingError(
                video_path,
                f"Audio extraction failed: {str(e)}"
            ) from e
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as HH:MM:SS or MM:SS.
        
        Args:
            seconds: Time in seconds
            
        Returns:
            Formatted timestamp string
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"

    async def process(
        self,
        file_path: str,
        file_id: Optional[str] = None,
        metadata: Optional[dict] = None,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> List[DocumentChunk]:
        """Process audio file with Whisper, chunk by speech segments.
        
        Creates chunks for each segment with timestamps and a summary chunk
        with the full transcription for citation-driven architecture.

        Args:
            file_path: Path to audio file or video file with audio track
            file_id: Optional file identifier for chunk IDs
            metadata: Optional file metadata to include in chunks
            progress_callback: Optional callback for progress updates (0.0 to 1.0)

        Returns:
            List of document chunks with temporal metadata (summary chunk first, then segments)

        Raises:
            DocumentProcessingError: If processing fails
        """
        temp_audio_path = None
        try:
            logger.info("Processing audio file", file_path=file_path, file_id=file_id)
            
            # Handle video files - extract audio track
            file_ext = Path(file_path).suffix.lower()
            actual_audio_path = file_path
            
            if file_ext == ".mp4":
                logger.info("Detected video file, extracting audio track", file_path=file_path)
                temp_audio_path = self._extract_audio_from_video(file_path)
                actual_audio_path = temp_audio_path
                if progress_callback:
                    progress_callback(0.1)
            
            # Load model if not already loaded
            async with self._model_lock:
                model = self._load_model()
            
            if model is None:
                raise DocumentProcessingError(
                    file_path,
                    "Whisper model not available. Install with: pip install openai-whisper or faster-whisper"
                )
            
            if progress_callback:
                progress_callback(0.2)
            
            # Transcribe audio
            import asyncio
            loop = asyncio.get_event_loop()
            
            full_text = ""
            segment_list = []
            detected_language = "en"
            
            if FASTER_WHISPER_AVAILABLE:
                # faster-whisper API
                segments, info = await loop.run_in_executor(
                    None,
                    lambda: model.transcribe(
                        str(actual_audio_path),
                        language="en",
                        task="transcribe",
                    )
                )
                
                # Convert segments to list
                full_text_parts = []
                for idx, segment in enumerate(segments):
                    segment_list.append({
                        "id": idx,
                        "start": segment.start,
                        "end": segment.end,
                        "text": segment.text.strip(),
                        "confidence": getattr(segment, "probability", None),
                    })
                    full_text_parts.append(segment.text.strip())
                
                full_text = " ".join(full_text_parts)
                detected_language = info.language if hasattr(info, "language") else "en"
                
            else:
                # openai-whisper API
                result = await loop.run_in_executor(
                    None,
                    lambda: model.transcribe(
                        str(actual_audio_path),
                        task="transcribe",
                        language="en",
                        verbose=False,
                    )
                )
                segment_list = result.get("segments", [])
                full_text = result.get("text", "").strip()
                detected_language = result.get("language", "en")
            
            if progress_callback:
                progress_callback(0.7)
            
            if not segment_list:
                logger.warning("No segments found in audio file", file_path=file_path)
                # Return a single chunk indicating no transcription
                file_name = metadata.get("file_name", Path(file_path).name) if metadata else Path(file_path).name
                drive_url = metadata.get("drive_url", metadata.get("web_view_link", "")) if metadata else ""
                
                base_meta = MetadataBuilder.base_metadata(
                    file_id=file_id or "unknown",
                    file_name=file_name,
                    file_type=FileType.AUDIO,
                    chunk_type=ChunkType.AUDIO_SUMMARY,
                    chunk_id=self._generate_chunk_id(file_id or "unknown", "summary"),
                    drive_url=drive_url,
                )
                
                return [DocumentChunk(
                    chunk_id=base_meta["chunk_id"],
                    content=f"[Audio file processed but no speech segments detected in {file_name}]",
                    file_id=base_meta["file_id"],
                    metadata=MetadataBuilder.merge_metadata(
                        base_meta,
                        {
                            "processing_status": "no_segments",
                            **(metadata or {}),
                        }
                    ),
                    embedding=None,
                )]
            
            # Prepare file metadata
            file_name = metadata.get("file_name", Path(file_path).name) if metadata else Path(file_path).name
            drive_url = metadata.get("drive_url", metadata.get("web_view_link", "")) if metadata else ""
            audio_format = file_ext if file_ext != ".mp4" else "mp4_audio"
            total_duration = segment_list[-1].get("end", 0.0) if segment_list else 0.0
            
            # Create summary chunk (full transcription) - FIRST chunk
            chunks: List[DocumentChunk] = []
            
            if full_text.strip():
                base_meta = MetadataBuilder.base_metadata(
                    file_id=file_id or "unknown",
                    file_name=file_name,
                    file_type=FileType.AUDIO,
                    chunk_type=ChunkType.AUDIO_SUMMARY,
                    chunk_id=self._generate_chunk_id(file_id or "unknown", "full"),
                    drive_url=drive_url,
                )
                
                summary_metadata = MetadataBuilder.merge_metadata(
                    base_meta,
                    {
                        "audio_format": audio_format,
                        "total_segments": len(segment_list),
                        "total_duration": total_duration,
                        "language": detected_language,
                        "source": "whisper_transcription",
                        **(metadata or {}),
                    }
                )
                
                summary_chunk = DocumentChunk(
                    chunk_id=base_meta["chunk_id"],
                    content=f"Full transcription of {file_name}:\n\n{full_text}",
                    file_id=base_meta["file_id"],
                    metadata=summary_metadata,
                    embedding=None,
                )
                chunks.append(summary_chunk)
            
            # Create chunks from segments with timestamps
            for segment in segment_list:
                segment_text = segment.get("text", "").strip()
                if not segment_text:
                    continue
                
                start_time = segment.get("start", 0.0)
                end_time = segment.get("end", 0.0)
                segment_id = segment.get("id", len(chunks))
                
                # Build base metadata
                base_meta = MetadataBuilder.base_metadata(
                    file_id=file_id or "unknown",
                    file_name=file_name,
                    file_type=FileType.AUDIO,
                    chunk_type=ChunkType.AUDIO_TRANSCRIPTION,
                    chunk_id=self._generate_chunk_id(file_id or "unknown", f"segment_{segment_id}"),
                    drive_url=drive_url,
                )
                
                # Add audio-specific metadata
                audio_meta = MetadataBuilder.audio_metadata(
                    base=base_meta,
                    start_time=start_time,
                    end_time=end_time,
                    segment_index=segment_id,
                    total_segments=len(segment_list),
                    speaker=None,  # Could be extracted from speaker diarization if available
                    confidence=segment.get("confidence"),
                    language=detected_language,
                )
                
                # Merge with additional metadata
                final_metadata = MetadataBuilder.merge_metadata(
                    audio_meta,
                    {
                        "audio_format": audio_format,
                        "source": "whisper_transcription",
                        **(metadata or {}),
                    }
                )
                
                # Enhance short content with context for better semantic matching
                enhanced_content = segment_text
                if len(segment_text) <= 100:
                    enhanced_content = f"Audio transcription from {file_name}: {segment_text}"
                
                chunk = DocumentChunk(
                    chunk_id=base_meta["chunk_id"],
                    content=enhanced_content,
                    file_id=base_meta["file_id"],
                    metadata=final_metadata,
                    embedding=None,
                )
                chunks.append(chunk)
                
                # Update progress
                if progress_callback:
                    progress = 0.7 + (len(chunks) / len(segment_list)) * 0.3
                    if asyncio.iscoroutinefunction(progress_callback):
                        await progress_callback(progress)
                    else:
                        progress_callback(progress)
            
            # Clean up temporary audio file if created
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                    logger.debug("Cleaned up temporary audio file", path=temp_audio_path)
                except Exception as e:
                    logger.warning("Failed to clean up temporary audio file", path=temp_audio_path, error=str(e))
            
            logger.info(
                "Audio transcription completed",
                file_path=file_path,
                segment_count=len(segment_list),
                chunk_count=len(chunks),
                language=detected_language,
            )
            
            return chunks
            
        except DocumentProcessingError:
            raise
        except Exception as e:
            logger.error("Audio processing failed", file_path=file_path, error=str(e), exc_info=True)
            raise DocumentProcessingError(file_path, str(e)) from e
        finally:
            # Ensure cleanup of temporary file even on error
            if temp_audio_path and os.path.exists(temp_audio_path):
                try:
                    os.remove(temp_audio_path)
                except Exception:
                    pass

    def get_supported_extensions(self) -> List[str]:
        """Get supported file extensions.

        Returns:
            List of audio extensions (including .mp4 for video with audio tracks)
        """
        return [".mp3", ".wav", ".m4a", ".mp4", ".flac", ".ogg"]

    def get_supported_mime_types(self) -> List[str]:
        """Get supported MIME types.

        Returns:
            List of audio MIME types (including video/mp4 for video with audio tracks)
        """
        return [
            "audio/mpeg",
            "audio/mp3",
            "audio/wav",
            "audio/x-wav",
            "audio/m4a",
            "audio/x-m4a",
            "audio/mp4",
            "audio/flac",
            "audio/ogg",
            "audio/vorbis",
            "video/mp4",  # Video files with audio tracks
        ]

