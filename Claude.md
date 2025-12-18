# CLAUDE.md - Foldex Development Guidelines

## Project Overview

**Foldex** is a local-first multimodal RAG system that transforms Google Drive folders into intelligent conversation interfaces. Users authenticate with Google Drive, paste a folder link, and chat with an AI assistant that understands relationships across text documents, PDFs, audio files, and code.

### Core Architecture Principles

1. **Local-First**: All processing, storage, and LLM inference happens locally
2. **RAG-Centric**: Vector database + Knowledge Graph + Hybrid retrieval at the core
3. **Multimodal**: Audio (Whisper) + Text (various formats) + Code processing
4. **Citation-Driven**: Every AI response must include precise source citations
5. **Real-time**: WebSocket updates for processing progress and chat

## Code Style & Standards

### Python Backend Standards

#### Type Hints & Pydantic
```python
# REQUIRED: All functions must have complete type hints
from typing import List, Dict, Optional, Union
from pydantic import BaseModel, Field

class DocumentChunk(BaseModel):
    chunk_id: str = Field(..., description="Unique chunk identifier")
    content: str = Field(..., min_length=1, max_length=2000)
    file_id: str = Field(..., description="Source file identifier")
    metadata: Dict[str, Union[str, int, float]] = Field(default_factory=dict)
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")

async def process_document(
    file_path: str, 
    processor_type: str
) -> List[DocumentChunk]:
    """Process document and return chunks with metadata."""
    pass
```

#### Error Handling Pattern
```python
# REQUIRED: Use custom exceptions with detailed context
from app.core.exceptions import ProcessingError, AuthenticationError

class DocumentProcessingError(ProcessingError):
    """Raised when document processing fails."""
    def __init__(self, file_path: str, reason: str):
        self.file_path = file_path
        self.reason = reason
        super().__init__(f"Failed to process {file_path}: {reason}")

# REQUIRED: All async functions must handle errors gracefully
async def process_pdf(file_path: str) -> List[DocumentChunk]:
    try:
        # Processing logic here
        pass
    except FileNotFoundError:
        raise DocumentProcessingError(file_path, "File not found")
    except Exception as e:
        logger.error(f"Unexpected error processing {file_path}: {str(e)}")
        raise DocumentProcessingError(file_path, f"Unexpected error: {type(e).__name__}")
```

#### Logging Standards
```python
import structlog

# REQUIRED: Use structured logging throughout
logger = structlog.get_logger(__name__)

async def index_folder(folder_id: str, user_id: str) -> None:
    logger.info("Starting folder indexing", 
                folder_id=folder_id, 
                user_id=user_id)
    
    try:
        # Processing logic
        logger.info("Folder indexing completed", 
                   folder_id=folder_id, 
                   chunk_count=chunk_count,
                   processing_time_seconds=elapsed)
    except Exception as e:
        logger.error("Folder indexing failed",
                    folder_id=folder_id,
                    error=str(e),
                    error_type=type(e).__name__)
        raise
```

#### Database Patterns
```python
# REQUIRED: Use dependency injection for database connections
from app.database.sqlite_manager import SQLiteManager
from app.database.vector_store import VectorStore

class DocumentService:
    def __init__(self, db: SQLiteManager, vector_store: VectorStore):
        self.db = db
        self.vector_store = vector_store
    
    async def store_document_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Store chunks in both SQL and vector databases."""
        async with self.db.transaction():
            # Store metadata in SQLite
            for chunk in chunks:
                await self.db.insert_chunk_metadata(chunk)
            
            # Store embeddings in ChromaDB
            await self.vector_store.add_chunks(chunks)
```

### TypeScript Frontend Standards

#### Component Structure
```typescript
// REQUIRED: Use functional components with proper TypeScript
import React, { useState, useEffect } from 'react';
import { ChatMessage, FileMetadata } from '../services/types';

interface ChatInterfaceProps {
  folderId: string;
  onNewMessage: (message: ChatMessage) => void;
  isProcessing: boolean;
}

export const ChatInterface: React.FC<ChatInterfaceProps> = ({
  folderId,
  onNewMessage,
  isProcessing
}) => {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState<string>('');
  
  // Component logic here
  
  return (
    <div className="chat-interface">
      {/* JSX here */}
    </div>
  );
};
```

#### API Service Pattern
```typescript
// REQUIRED: Centralized API service with proper error handling
class APIService {
  private baseURL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:8000';
  
  async processFolder(folderId: string): Promise<ProcessingStatus> {
    try {
      const response = await fetch(`${this.baseURL}/api/v1/folders/process`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${this.getAuthToken()}`
        },
        body: JSON.stringify({ folder_id: folderId })
      });
      
      if (!response.ok) {
        throw new APIError(response.status, await response.text());
      }
      
      return await response.json();
    } catch (error) {
      console.error('API Error:', error);
      throw error;
    }
  }
}
```

## Testing Requirements

### Backend Testing Standards

#### Unit Tests
```python
# REQUIRED: Comprehensive unit tests for all services
import pytest
from unittest.mock import AsyncMock, patch
from app.services.document_processor import DocumentProcessor
from app.models.documents import DocumentChunk

class TestDocumentProcessor:
    @pytest.fixture
    async def processor(self):
        return DocumentProcessor()
    
    @pytest.mark.asyncio
    async def test_process_pdf_success(self, processor, sample_pdf_path):
        """Test successful PDF processing."""
        chunks = await processor.process_pdf(sample_pdf_path)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, DocumentChunk) for chunk in chunks)
        assert all(chunk.content.strip() for chunk in chunks)
        assert all(chunk.file_id for chunk in chunks)
    
    @pytest.mark.asyncio
    async def test_process_pdf_file_not_found(self, processor):
        """Test PDF processing with non-existent file."""
        with pytest.raises(DocumentProcessingError):
            await processor.process_pdf("/non/existent/file.pdf")
```

#### Integration Tests
```python
# REQUIRED: End-to-end workflow tests
import pytest
from httpx import AsyncClient
from app.main import app

@pytest.mark.asyncio
async def test_folder_processing_workflow():
    """Test complete folder processing workflow."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # 1. Authenticate
        auth_response = await client.post("/api/v1/auth/token", 
                                         json={"access_token": "test_token"})
        assert auth_response.status_code == 200
        
        # 2. Process folder
        process_response = await client.post("/api/v1/folders/process",
                                           json={"folder_id": "test_folder_id"})
        assert process_response.status_code == 202
        
        # 3. Query processed folder
        query_response = await client.post("/api/v1/chat/query",
                                         json={"query": "test query"})
        assert query_response.status_code == 200
        assert "citations" in query_response.json()
```

### Frontend Testing Standards

```typescript
// REQUIRED: Component testing with React Testing Library
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { ChatInterface } from '../ChatInterface';

describe('ChatInterface', () => {
  it('should send message when form is submitted', async () => {
    const mockOnNewMessage = jest.fn();
    
    render(
      <ChatInterface 
        folderId="test_folder" 
        onNewMessage={mockOnNewMessage}
        isProcessing={false}
      />
    );
    
    const input = screen.getByPlaceholderText('Type your message...');
    const submitButton = screen.getByRole('button', { name: /send/i });
    
    fireEvent.change(input, { target: { value: 'Test message' } });
    fireEvent.click(submitButton);
    
    await waitFor(() => {
      expect(mockOnNewMessage).toHaveBeenCalledWith(
        expect.objectContaining({
          content: 'Test message',
          type: 'user'
        })
      );
    });
  });
});
```

## Component Specifications

### Document Processors

#### Base Processor Interface
```python
from abc import ABC, abstractmethod
from typing import List
from app.models.documents import DocumentChunk

class BaseProcessor(ABC):
    """Base interface for all document processors."""
    
    @abstractmethod
    async def can_process(self, file_path: str) -> bool:
        """Check if processor can handle the file type."""
        pass
    
    @abstractmethod
    async def process(self, file_path: str) -> List[DocumentChunk]:
        """Extract content and return structured chunks."""
        pass
    
    @abstractmethod
    def get_supported_extensions(self) -> List[str]:
        """Return list of supported file extensions."""
        pass
```

#### Processor Implementation Requirements
```python
class AudioProcessor(BaseProcessor):
    """REQUIRED: Must handle chunking by speech segments, not fixed time."""
    
    async def process(self, file_path: str) -> List[DocumentChunk]:
        # 1. Transcribe with Whisper (include timestamps)
        # 2. Segment by speaker turns or natural pauses  
        # 3. Create chunks with temporal metadata
        # 4. Extract speaker information if available
        pass

class PDFProcessor(BaseProcessor):
    """REQUIRED: Must preserve document structure and page references."""
    
    async def process(self, file_path: str) -> List[DocumentChunk]:
        # 1. Extract text with page numbers
        # 2. Identify sections/headers if possible
        # 3. Maintain reading order
        # 4. Include page/section metadata for citations
        pass
```

### RAG Engine Specifications

#### Retrieval Strategy
```python
class HybridRetriever:
    """REQUIRED: Must implement multiple retrieval strategies."""
    
    async def retrieve(self, query: str, k: int = 10) -> List[DocumentChunk]:
        # 1. Semantic similarity search (vector)
        # 2. Keyword/BM25 search for exact matches
        # 3. Knowledge graph traversal for relationships
        # 4. Temporal filtering if query includes time references
        # 5. Rerank using cross-encoder model
        pass
```

#### Citation Engine Requirements
```python
class CitationEngine:
    """REQUIRED: Must provide precise source attribution."""
    
    def generate_citations(self, 
                          response: str, 
                          source_chunks: List[DocumentChunk]
                         ) -> List[Citation]:
        # 1. Map response claims to specific chunks
        # 2. Include file name, page/timestamp, confidence
        # 3. Support multiple citation formats
        # 4. Handle overlapping sources
        pass
```

### Knowledge Graph Specifications

```python
class RelationshipDetector:
    """REQUIRED: Must detect multiple relationship types."""
    
    RELATIONSHIP_TYPES = {
        'entity_overlap': 'Documents mention same entities',
        'temporal': 'Documents created in sequence',
        'cross_reference': 'Document A references Document B',
        'topical_similarity': 'Documents share themes',
        'implementation_gap': 'Code doesn't match specifications'
    }
    
    async def detect_relationships(self, 
                                 chunks: List[DocumentChunk]
                                ) -> List[Relationship]:
        # Implement detection logic for each relationship type
        pass
```

## Performance Requirements

### Processing Benchmarks
- **Folder Indexing**: < 2 minutes for 20 documents (mixed types)
- **Query Response**: < 15 seconds for complex cross-document queries
- **Audio Transcription**: Real-time processing (1 minute audio = 1 minute processing)
- **Memory Usage**: < 4GB peak during indexing
- **Storage Efficiency**: < 10MB additional data per 1MB source content

### Optimization Rules
```python
# REQUIRED: Use async/await for all I/O operations
async def process_files_concurrently(file_paths: List[str]) -> List[DocumentChunk]:
    tasks = [process_single_file(path) for path in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    return [r for r in results if not isinstance(r, Exception)]

# REQUIRED: Implement caching for expensive operations  
@cache(ttl=3600)  # 1 hour cache
async def generate_embeddings(text: str) -> List[float]:
    return await embedding_model.encode(text)

# REQUIRED: Batch operations where possible
async def store_embeddings_batch(chunks: List[DocumentChunk]) -> None:
    embeddings = await asyncio.gather(*[
        generate_embeddings(chunk.content) for chunk in chunks
    ])
    await vector_store.add_batch(chunks, embeddings)
```

## Development Workflow

### Git Commit Standards
```bash
# REQUIRED: Use conventional commits
feat: add audio transcription with speaker detection
fix: resolve memory leak in PDF processing
docs: update API documentation for citation format
test: add integration tests for knowledge graph
refactor: optimize embedding generation pipeline
```

### Code Review Checklist
- [ ] All functions have type hints and docstrings
- [ ] Error handling covers edge cases
- [ ] Tests cover new functionality
- [ ] Performance impact considered
- [ ] Memory usage reasonable
- [ ] Logging includes relevant context
- [ ] Citations work for new content types

### Local Development Setup
```bash
# REQUIRED: Environment setup script
./scripts/setup.sh                    # Install dependencies
python scripts/download_models.py     # Download ML models
docker-compose up -d                  # Start local services
pytest backend/tests/                 # Run backend tests
npm test                              # Run frontend tests
```

## Security & Privacy

### Data Handling Rules
```python
# REQUIRED: Never log sensitive data
logger.info("Processing folder", 
           folder_id=folder_id,          # OK to log
           user_id=user_id[:8] + "...")  # Truncate sensitive IDs

# FORBIDDEN: Do not log file contents or user queries
logger.debug("File content", content=file_content)  # ❌ NEVER
logger.info("User query", query=user_query)         # ❌ NEVER

# REQUIRED: Sanitize all user inputs
def sanitize_folder_id(folder_id: str) -> str:
    if not re.match(r'^[a-zA-Z0-9_-]+$', folder_id):
        raise ValidationError("Invalid folder ID format")
    return folder_id[:64]  # Limit length
```

### Local Storage Requirements
- All data must be stored locally (no cloud services except Google Drive API)
- User can delete all data with single command
- No persistent storage of auth tokens beyond session
- Clear data retention policy in UI

## Documentation Requirements

### API Documentation
```python
# REQUIRED: Comprehensive docstrings for all public functions
async def process_folder(folder_id: str, user_id: str) -> ProcessingResult:
    """
    Process a Google Drive folder for intelligent search.
    
    Args:
        folder_id: Google Drive folder identifier
        user_id: Authenticated user identifier
        
    Returns:
        ProcessingResult containing:
            - total_files: Number of files processed
            - chunk_count: Number of text chunks created
            - processing_time: Time taken in seconds
            - supported_files: List of successfully processed files
            - unsupported_files: List of files that couldn't be processed
            
    Raises:
        AuthenticationError: If user lacks folder access
        ProcessingError: If folder processing fails
        ValidationError: If folder_id is invalid
        
    Example:
        >>> result = await process_folder("1abc123xyz", "user_789")
        >>> print(f"Processed {result.chunk_count} chunks")
    """
```

### README Requirements
- Clear setup instructions for local development
- Demo video link and instructions
- Architecture overview with diagrams
- Performance characteristics and limitations
- Supported file types and processing capabilities

### CHANGELOG Requirements
- **REQUIRED**: Maintain CHANGELOG.md in root directory
- **REQUIRED**: Update for every major change:
  - New features or enhancements
  - Bug fixes affecting user experience
  - Breaking changes or API modifications
  - Security updates
  - Performance improvements
  - Significant dependency updates
- **Format**: Use date-based entries with categories (Added, Changed, Fixed, Security, etc.)
- **Style**: Follow [Keep a Changelog](https://keepachangelog.com/) format
- **Timing**: Update changelog before committing major changes

This document serves as the source of truth for all code generation. When implementing any component, refer back to these standards to ensure consistency and quality.