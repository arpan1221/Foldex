# Changelog

All notable changes to the Foldex project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **⚡ TTFT (Time-To-First-Token) Optimization System** (2025-12-18)
  - Comprehensive TTFT optimization to reduce AI response latency
  - **Prompt Caching**: LRU cache for system prompts with configurable TTL (default 1 hour)
    - Reduces repeated prompt processing overhead
    - MD5-based cache keys for query types and folders
    - Cache hit/miss statistics tracking
    - Maximum cache size: 100 entries (configurable)
  - **Context Window Optimization**: Intelligent context size reduction
    - Truncates documents to 1500 characters while preserving meaning
    - Sentence-boundary aware truncation
    - Limits total context to 6000 characters (configurable)
    - Reduces tokens sent to LLM without sacrificing quality
  - **Model Pre-Warming**: Startup warmup to eliminate cold-start latency
    - Automatically loads model into memory on application startup
    - Runs simple inference to prepare model for first user request
    - Configurable via `ENABLE_MODEL_WARMUP` setting
    - Logs warmup time for monitoring
  - **Improved Async Streaming**: Native async streaming without callback overhead
    - `AsyncStreamingLLM` class for direct async token streaming
    - `BufferedAsyncStream` for smoother token delivery
    - `StreamingOrchestrator` for coordinated retrieval and streaming
    - Reduces latency compared to callback-based streaming
  - **Configuration Settings** (backend/app/config/settings.py):
    - `ENABLE_TTFT_OPTIMIZATION`: Master toggle (default: True)
    - `ENABLE_PROMPT_CACHE`: Enable prompt caching (default: True)
    - `PROMPT_CACHE_SIZE`: Cache size (default: 100)
    - `PROMPT_CACHE_TTL`: Cache TTL in seconds (default: 3600)
    - `MAX_CONTEXT_CHARS`: Max context window (default: 6000)
    - `ENABLE_MODEL_WARMUP`: Pre-warm model (default: True)
  - **New Modules**:
    - `backend/app/rag/ttft_optimization.py`: Core optimization classes
    - `backend/app/rag/async_streaming.py`: Async streaming implementation
    - `backend/app/startup/warmup.py`: Startup warmup orchestration
  - **Integration Points**:
    - Integrated into RetrievalQA chains via `enable_ttft_optimization` parameter
    - Automatic warmup in FastAPI lifespan manager (backend/app/main.py)
    - Context optimization applied in `format_docs` during retrieval
  - **Performance Impact**:
    - Expected 30-50% reduction in TTFT for cached prompts
    - 20-30% reduction in TTFT from context optimization
    - Eliminates 1-3 second cold-start delay with model warmup

### Fixed
- **Token-by-Token Streaming for Chat Responses**
  - Upgraded LangChain from 0.1.0 to 0.1.20 for proper Ollama streaming support
  - Upgraded langchain-community from 0.0.10 to 0.0.38 with full callback propagation
  - Upgraded langchain-core from 0.1.7 to 0.1.52 for compatibility
  - Added `verbose=True` to RetrievalQA chains to enable callback propagation through chain layers
  - StreamingCallbackHandler now properly receives tokens from Ollama LLM
  - Chat responses stream token-by-token via Server-Sent Events (SSE) to frontend
  - Fixed issue where older LangChain versions didn't support streaming callbacks with Ollama
  - Improved user experience with real-time response generation
- **Chat Query RAG Initialization**
  - Fixed "NoneType object is not subscriptable" error when querying chat
  - Added missing `rank-bm25` package required for BM25 retriever in hybrid search
  - Fixed "Missing some input keys: {'chat_history'}" error in chain invocation
  - Chain now receives empty chat_history parameter for new conversations
  - Fixed BM25Retriever deprecation warning by importing from langchain-community
  - RAG service now initializes properly for chat queries
- **Folder Processing Infinite Loop**
  - Fixed critical infinite processing loop where folders appeared to never finish processing
  - `useFolderProcessor` now correctly sets `isProcessing` to false when processing completes
  - Fixed navigation timer in FolderInput that was resetting on every status update
  - Processing now correctly completes and auto-navigates to chat after 3 seconds
  - Data now appears in sidebar immediately after processing without requiring page refresh
- **Processing Progress Display**
  - Fixed file count display showing "0 of 2" instead of actual progress (e.g., "50 of 137")
  - FolderUploadInterface now uses backend-provided `total_files` and `files_processed` for accurate progress
  - Progress bar and percentage now reflect actual processing status in real-time
  - Added `files_processed` field to all WebSocket messages (`file_processing`, `file_error`) for consistent progress tracking
  - Progress now updates in real-time as each file starts processing, not just when completed
  - Failed files are correctly counted in progress total to maintain accurate completion tracking
- **WebSocket Connection Management**
  - Fixed WebSocket disconnections (code 1012) during long file processing
  - Added server-side keepalive pings every 10 seconds to prevent timeout (reduced for large folders)
  - Configured Uvicorn with proper WebSocket timeout settings (20s ping interval, 30s timeout, 75s keepalive)
  - **Removed serialization lock from WebSocket sends** - allows concurrent message delivery for large folders
  - Messages now sent in parallel without blocking each other (274 messages no longer serialized)
  - **Fixed auto-reload disconnections during file downloads** - excluded data directory from file watcher
  - Server no longer restarts when files are downloaded to cache during processing
  - WebSocket now remains stable during processing of 100+ file folders
  - Connection properly cleaned up on component unmount or navigation
  - Concurrent keepalive and message handling tasks prevent blocking
- **Sidebar Folder Hierarchy Display**
  - Fixed root folder appearing as its own subfolder in the sidebar tree view
  - Root folder contents now display directly at the root level of the tree instead of being nested under the folder name
  - Improved user experience by eliminating redundant nesting when viewing folder contents
- **Dependency Version Compatibility**
  - Updated PyTorch requirement from 2.1.0 to >=2.2.0,<3.0.0 for compatibility with available versions
  - PyTorch 2.1.0 is no longer available on PyPI, updated to use latest compatible version

### Added
- **Local Development Utilities**
  - Enhanced setup.sh script with colored output and comprehensive checks
  - Automatic .env file generation with secure SECRET_KEY
  - Database initialization in setup script
  - Enhanced download_models.py with progress indicators and error handling
  - Ollama connection checking in model downloader
  - Docker Compose configuration with ChromaDB service
  - Health checks for all Docker services
  - Volume management for persistent data
  - Complete README.md with setup instructions
  - .env.example template with all configuration options
  - Prerequisites checking (Python, Node.js, Docker)
  - Troubleshooting guide in README
  - Performance benchmarks documentation
  - Development workflow instructions

- **Comprehensive Testing Foundation**
  - Backend pytest configuration with async fixtures
  - Database fixtures with temporary SQLite and ChromaDB instances
  - Vector store fixtures for embedding tests
  - Mock Google Drive API responses
  - Sample test documents (PDF, text, Markdown)
  - Unit tests for document processors (PDF, text, base)
  - DocumentProcessor orchestrator tests
  - Frontend React Testing Library setup
  - Component tests for GoogleAuth, LoadingSpinner, ErrorDisplay
  - Test utilities and helpers (custom render, mocks)
  - Vitest configuration with coverage
  - Test fixtures for common scenarios
  - Mock services and progress callbacks
  - Sample file generators for testing

- **Complete Frontend-Backend API Integration**
  - Comprehensive API service with all endpoints (auth, folders, chat, system)
  - Centralized API state management hook (useAPI)
  - Specialized hooks: useAuth, useFolders, useChat
  - APIException class for typed error handling
  - Automatic token refresh with retry logic
  - ErrorDisplay component for consistent error UI
  - LoadingOverlay component for loading states
  - Optimistic UI updates for chat messages
  - Network failure handling with retry mechanisms
  - Error boundaries at component and app level
  - Health check integration on app mount
  - Proper TypeScript interfaces for all API responses
  - Request/response interceptors for auth and errors
  - Graceful error recovery throughout the app

- **Background Processing Infrastructure**
  - Enhanced FolderProcessor with error recovery and retry logic
  - WebSocket endpoints with JWT authentication
  - ConnectionManager with connection tracking and metadata
  - FileCache for downloaded file management with TTL
  - Cache utility for key-value storage with expiration
  - Frontend WebSocketService with auto-reconnection
  - Keep-alive ping/pong mechanism
  - Processing status persistence and tracking
  - Task cancellation support
  - Exponential backoff for retries
  - Progress tracking with detailed updates
  - Error recovery with failed file tracking
  - WebSocket message routing and handling
  - Connection state management
  - Automatic cleanup of expired cache entries

- **Base File Processing System**
  - BaseProcessor interface with async processing support
  - PDFProcessor using PyMuPDF for text extraction
  - TextProcessor for plain text and Markdown files
  - DocumentProcessor orchestrator with file type routing
  - Progress tracking callbacks for processing updates
  - Smart chunking with sentence/paragraph awareness
  - Metadata preservation in document chunks
  - Page number tracking for PDF citations
  - Markdown structure preservation
  - Encoding detection for text files
  - File size validation before processing
  - Comprehensive error handling with detailed messages
  - Support for multiple text encodings (UTF-8, Latin-1, CP1252)
  - Chunk overlap for context preservation
  - File type detection by extension and MIME type

- **Google Drive File Access Services**
  - GoogleDriveService with OAuth2 integration
  - Folder URL parsing and validation
  - File enumeration with pagination support
  - File download with caching
  - Metadata extraction for files and folders
  - Rate limiting for API calls (10 req/sec, 1000 req/100sec)
  - Support for Google Workspace files (Docs, Sheets, Slides)
  - File type detection and validation
  - Comprehensive error handling for API failures
  - Background task processing for folder indexing
  - Folder processing endpoints with status tracking
  - File metadata models with Pydantic validation
  - File utilities for validation and metadata extraction
  - MIME type detection and file type classification

- **Google Drive Authentication System**
  - Google OAuth2 flow with Drive scope validation
  - JWT token creation and verification (access and refresh tokens)
  - User management with database persistence
  - Token refresh endpoint for extended sessions
  - Comprehensive error handling for auth failures
  - Secure session management utilities
  - UserRecord database model for user storage
  - API endpoints: /token, /refresh, /me, /logout
  - Automatic user creation/update on authentication
  - Email validation and verification status tracking
  - Google token validation with scope checking
  - HTTP interceptors for automatic token refresh

- **Main Application Layout & Routing**
  - App.tsx with React Router setup and protected routes
  - MainLayout component with responsive grid system
  - Sidebar component with folder navigation and search
  - ErrorBoundary component for error handling
  - ProtectedRoute component for authentication guards
  - LoadingSpinner reusable component
  - API service updates with getUserFolders method
  - Automatic 401 error handling with redirect to login
  - Responsive sidebar with collapse functionality
  - Folder list with active state indicators
  - Search functionality for folders
  - Empty states with helpful messages

- **Knowledge Graph Visualization**
  - KnowledgeGraph component with React Flow integration
  - Interactive graph display with custom node styling
  - Multiple layout algorithms (hierarchical, force-directed, circular)
  - RelationshipView sidebar with expandable relationship cards
  - GraphControls component for filtering and layout selection
  - useGraphVisualization hook for graph data management
  - File type filtering with visual indicators
  - Relationship type filtering with color-coded edges
  - Node click handlers for interaction
  - MiniMap and Controls for navigation
  - Graph info overlay showing node and edge counts
  - Responsive design following Figma wireframe
  - Dark theme with gradient node styling

- **Chat Interface Components**
  - ChatInterface component with message history and auto-scroll
  - MessageBubble component with user/assistant styling following Figma design
  - CitationDisplay component with hover effects and file references
  - InputArea component with auto-resize textarea and keyboard shortcuts
  - Typing indicators with animated dots
  - Welcome message for empty chat state
  - Enhanced date formatting (today, yesterday, relative dates)
  - Keyboard shortcuts (Enter to send, Shift+Enter for new line, '/' to focus)
  - Responsive layout matching Figma wireframe design
  - Dark gradient theme with proper message styling

- **Folder Input & Processing Interface**
  - FolderInput component with Google Drive URL validation and real-time feedback
  - FileOverview component with file type detection, icons, and size display
  - ProcessingStatus component with WebSocket real-time progress updates
  - Enhanced useFolderProcessor hook with WebSocket integration and state management
  - URL validation supporting multiple Google Drive URL formats
  - File type detection with visual indicators and badges
  - Progress bars with gradient animations
  - Error states with user-friendly messages
  - Responsive design following Figma wireframe patterns

### Added (Previous)
- Initial repository structure with complete backend and frontend scaffolding
- Backend FastAPI application with API routes for authentication, folders, chat, and WebSocket
- Frontend React application with TypeScript and Tailwind CSS
- Configuration management with Pydantic settings and environment variable support
- Structured logging with structlog for backend
- Custom exception handling system with FoldexException base class
- Document processors for PDF, Audio (Whisper), Text, and Code files
- RAG engine components: embeddings, hybrid retriever, reranker, and LLM interface
- Knowledge graph system with entity extraction and relationship detection
- Database layer with SQLite manager and ChromaDB vector store
- WebSocket support for real-time processing updates
- Frontend components: authentication, chat interface, folder processing, visualization
- Custom React hooks: useAuth, useChat, useFolderProcessor
- API service layer with TypeScript types
- Tailwind CSS configuration with Foldex brand colors
- Docker Compose setup for local development
- Setup scripts for environment initialization and model downloads
- Test structure with unit, integration, and E2E test directories
- Monitoring utilities for performance metrics
- Comprehensive type definitions for API communication

### Changed
- N/A (initial release)

### Deprecated
- N/A (initial release)

### Removed
- N/A (initial release)

### Fixed
- N/A (initial release)

### Security
- N/A (initial release)

---

## [0.1.0] - 2024-12-XX

### Added
- CHANGELOG.md tracking system for all major project changes
- Development workflow rules requiring changelog maintenance
- Documentation standards updated to include changelog requirements

### Added (Configuration & Infrastructure)
- **Configuration System**
  - Pydantic v2 settings with environment variable support
  - Field validators for DEBUG, CORS_ORIGINS, and SECRET_KEY
  - Auto-directory creation on startup
  - Development/production mode detection

- **Logging System**
  - Structured logging with structlog
  - JSON output for production, pretty console for development
  - Request logging middleware
  - Context variable support

- **FastAPI Application**
  - Lifespan management for startup/shutdown events
  - CORS middleware configuration
  - Comprehensive exception handlers
  - Health check and root endpoints
  - Request/response logging

- **Frontend Configuration**
  - Vite configuration with proxy setup
  - Path aliases for cleaner imports
  - Build optimization with code splitting
  - Tailwind CSS with custom Foldex brand colors
  - Comprehensive TypeScript type definitions

- **Documentation**
  - CHANGELOG.md for tracking all major changes
  - Updated development rules to require changelog maintenance

---

## Categories

Changes are categorized as:
- **Added**: New features
- **Changed**: Changes in existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security improvements or vulnerability fixes

---

## Notes

- Dates follow ISO 8601 format (YYYY-MM-DD)
- Breaking changes are marked with ⚠️
- Security updates are marked with 🔒
- Performance improvements are marked with ⚡

