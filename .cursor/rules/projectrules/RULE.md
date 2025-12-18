---
alwaysApply: true
---
# Foldex Cursor AI Rules

## Project Context
You are working on **Foldex**, a local-first multimodal RAG system that transforms Google Drive folders into intelligent conversation interfaces. This is a 5-day take-home assignment showcasing advanced AI capabilities.

## Core Architecture Understanding
- **Backend**: FastAPI + SQLite + ChromaDB + Local LLMs (Ollama)
- **Frontend**: React + TypeScript + Tailwind CSS
- **Processing**: Audio (Whisper) + Text + PDF + Code analysis
- **RAG Engine**: Hybrid retrieval (semantic + keyword + knowledge graph)
- **Storage**: Local-first, no cloud services except Google Drive API

## Code Generation Rules

### Python Backend Standards
- **ALWAYS** use complete type hints with pydantic models
- **ALWAYS** implement proper async/await patterns for I/O
- **ALWAYS** include structured logging with contextual information
- **ALWAYS** use custom exceptions with detailed error context
- **NEVER** log sensitive data (file contents, user queries, auth tokens)
- **REQUIRED**: Follow dependency injection pattern for services
- **REQUIRED**: Include docstrings with Args, Returns, Raises, Examples

### TypeScript Frontend Standards  
- **ALWAYS** use functional components with proper TypeScript interfaces
- **ALWAYS** implement proper error boundaries and loading states
- **ALWAYS** use centralized API service with proper error handling
- **REQUIRED**: Follow React hook patterns (useState, useEffect, custom hooks)
- **REQUIRED**: Use Tailwind CSS for styling, no inline styles

### File Processing Requirements
- **Audio files**: Use Whisper for transcription, chunk by speech segments
- **PDF files**: Preserve page numbers and document structure for citations
- **Text files**: Maintain original formatting and metadata
- **Code files**: Parse with tree-sitter, preserve function/class boundaries
- **All processors**: Must implement BaseProcessor interface

### RAG System Requirements
- **Embeddings**: Use sentence-transformers locally (all-mpnet-base-v2)
- **Vector DB**: ChromaDB with proper collection management
- **Retrieval**: Hybrid approach (semantic + keyword + graph traversal)
- **Citations**: Every AI response must include precise source attribution
- **Knowledge Graph**: NetworkX + SQLite for relationship storage

### Testing Requirements
- **Unit Tests**: pytest for backend, Jest/RTL for frontend
- **Coverage**: Minimum 80% code coverage for core services
- **Integration**: End-to-end workflow tests for critical paths
- **Fixtures**: Use realistic test data, not minimal examples
- **Mocking**: Mock external APIs (Google Drive, OpenAI) in tests

## Performance Constraints
- **Folder Indexing**: < 2 minutes for 20 mixed-type documents
- **Query Response**: < 15 seconds for complex cross-document analysis  
- **Memory Usage**: < 4GB peak during processing
- **Audio Processing**: Target real-time transcription speed
- **Startup Time**: < 30 seconds for full system initialization

## Security & Privacy Rules
- **Local Storage Only**: All user data remains on local machine
- **No Logging**: Never log file contents, user queries, or personal info
- **Input Validation**: Sanitize all user inputs, especially file paths
- **Auth Handling**: Use Google OAuth2, no persistent token storage
- **Error Messages**: Don't expose internal system details to users

## Code Quality Standards
- **Error Handling**: Graceful degradation, never crash the system
- **Async Operations**: Use asyncio.gather for concurrent processing
- **Resource Management**: Proper cleanup of file handles and connections
- **Caching**: Implement intelligent caching for expensive operations
- **Monitoring**: Include performance metrics and health checks

## File Structure Adherence
```
backend/app/
├── services/          # Business logic (orchestrators)
├── processors/        # Document processing (PDF, audio, etc.)
├── rag/              # RAG engine components
├── knowledge_graph/   # Graph construction and queries
├── models/           # Pydantic data models
├── api/              # FastAPI route handlers
└── database/         # Storage abstraction layers

frontend/src/
├── components/       # React components by feature
├── services/         # API clients and utilities
├── hooks/           # Custom React hooks
└── utils/           # Pure functions and helpers
```

## Documentation Requirements
- **API Endpoints**: OpenAPI/Swagger documentation
- **Component Props**: JSDoc comments for all React components
- **Service Methods**: Comprehensive docstrings with examples
- **README Updates**: Keep setup instructions current
- **Architecture Docs**: Update when adding new components

## When Implementing New Features
1. **Check CLAUDE.md** for relevant patterns and interfaces
2. **Follow established file structure** and naming conventions
3. **Include comprehensive error handling** with custom exception types
4. **Add tests first** (TDD approach preferred)
5. **Update documentation** for any new APIs or components
6. **Consider performance impact** and add monitoring if needed

## Common Patterns to Follow

### Service Layer Pattern
```python
class DocumentService:
    def __init__(self, db: SQLiteManager, vector_store: VectorStore):
        self.db = db
        self.vector_store = vector_store
        self.logger = structlog.get_logger(__name__)
    
    async def process_document(self, file_path: str) -> ProcessingResult:
        # Implementation with proper error handling and logging
```

### API Endpoint Pattern  
```python
@router.post("/folders/{folder_id}/process")
async def process_folder(
    folder_id: str,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user)
) -> ProcessingResponse:
    # Validation, service call, proper response
```

### React Component Pattern
```typescript
interface ComponentProps {
  // Explicit prop types
}

export const Component: React.FC<ComponentProps> = ({ prop1, prop2 }) => {
  // Hooks, state management, error handling
  return <div>...</div>;
};
```

## Development Workflow
- **Commit Messages**: Use conventional commits (feat:, fix:, docs:)
- **Branch Strategy**: Feature branches for each major component  
- **Code Review**: Self-review against CLAUDE.md checklist before commits
- **Testing**: Run tests before any commits
- **Performance**: Monitor memory usage during development

## AI Assistant Behavior
- **Be Specific**: Generate complete, working code, not pseudocode
- **Include Tests**: Provide unit tests with any new functionality
- **Think Performance**: Consider resource usage in all implementations
- **Explain Decisions**: Comment non-obvious code choices
- **Stay Consistent**: Follow established patterns rather than inventing new ones

Remember: This is a take-home assignment meant to showcase production-quality code and architecture. Every component should demonstrate professional software development practices.