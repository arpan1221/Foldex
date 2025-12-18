# Changelog

All notable changes to the Foldex project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

