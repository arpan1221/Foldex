foldex/
├── README.md
├── CLAUDE.md                          # Development guidelines for AI assistance
├── .gitignore
├── .env.example
├── requirements.txt
├── package.json
├── docker-compose.yml
│
├── frontend/                          # React web interface
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── auth/
│   │   │   │   ├── GoogleAuth.tsx
│   │   │   │   └── AuthContext.tsx
│   │   │   ├── chat/
│   │   │   │   ├── ChatInterface.tsx
│   │   │   │   ├── MessageBubble.tsx
│   │   │   │   └── CitationDisplay.tsx
│   │   │   ├── folder/
│   │   │   │   ├── FolderInput.tsx
│   │   │   │   ├── FileOverview.tsx
│   │   │   │   └── ProcessingStatus.tsx
│   │   │   └── visualization/
│   │   │       ├── KnowledgeGraph.tsx
│   │   │       └── RelationshipView.tsx
│   │   ├── hooks/
│   │   │   ├── useAuth.ts
│   │   │   ├── useChat.ts
│   │   │   └── useFolderProcessor.ts
│   │   ├── services/
│   │   │   ├── api.ts
│   │   │   ├── websocket.ts
│   │   │   └── types.ts
│   │   ├── utils/
│   │   │   ├── formatters.ts
│   │   │   └── validators.ts
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   ├── tsconfig.json
│   ├── vite.config.ts
│   └── tailwind.config.js
│
├── backend/                           # FastAPI backend
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI app entry point
│   │   ├── config/
│   │   │   ├── __init__.py
│   │   │   ├── settings.py            # App configuration
│   │   │   └── logging.py             # Logging setup
│   │   ├── api/
│   │   │   ├── __init__.py
│   │   │   ├── deps.py                # Dependency injection
│   │   │   └── v1/
│   │   │       ├── __init__.py
│   │   │       ├── auth.py            # Auth endpoints
│   │   │       ├── folders.py         # Folder processing endpoints
│   │   │       ├── chat.py            # Chat/query endpoints
│   │   │       └── websocket.py       # Real-time updates
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── auth.py                # Authentication logic
│   │   │   ├── exceptions.py          # Custom exceptions
│   │   │   └── security.py            # Security utilities
│   │   ├── services/                  # Business logic layer
│   │   │   ├── __init__.py
│   │   │   ├── folder_processor.py    # Main orchestrator
│   │   │   ├── google_drive.py        # Drive API integration
│   │   │   ├── document_processor.py  # File extraction
│   │   │   ├── knowledge_graph.py     # Graph construction
│   │   │   ├── rag_engine.py         # RAG core logic
│   │   │   └── chat_service.py        # Query processing
│   │   ├── models/                    # Data models
│   │   │   ├── __init__.py
│   │   │   ├── auth.py
│   │   │   ├── documents.py
│   │   │   ├── chat.py
│   │   │   └── database.py
│   │   ├── database/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # Database base classes
│   │   │   ├── sqlite_manager.py      # SQLite operations
│   │   │   └── vector_store.py        # ChromaDB interface
│   │   ├── processors/                # Document processing
│   │   │   ├── __init__.py
│   │   │   ├── base.py                # Base processor interface
│   │   │   ├── pdf_processor.py       # PDF extraction
│   │   │   ├── audio_processor.py     # Whisper integration
│   │   │   ├── text_processor.py      # Text/Markdown
│   │   │   └── code_processor.py      # Source code parsing
│   │   ├── rag/                      # RAG system components
│   │   │   ├── __init__.py
│   │   │   ├── embeddings.py         # Local embedding service
│   │   │   ├── retriever.py          # Hybrid retrieval logic
│   │   │   ├── reranker.py           # Cross-encoder reranking
│   │   │   └── llm_interface.py      # Local LLM integration
│   │   ├── knowledge_graph/
│   │   │   ├── __init__.py
│   │   │   ├── entity_extractor.py   # NER processing
│   │   │   ├── relationship_detector.py # Relationship mapping
│   │   │   ├── graph_builder.py      # NetworkX graph construction
│   │   │   └── graph_queries.py      # Graph traversal logic
│   │   └── utils/
│   │       ├── __init__.py
│   │       ├── file_utils.py         # File handling utilities
│   │       ├── text_utils.py         # Text processing helpers
│   │       └── cache.py              # Caching utilities
│   ├── tests/
│   │   ├── __init__.py
│   │   ├── conftest.py               # Pytest configuration
│   │   ├── unit/                     # Unit tests
│   │   │   ├── test_processors/
│   │   │   ├── test_rag/
│   │   │   └── test_services/
│   │   ├── integration/              # Integration tests
│   │   │   ├── test_api/
│   │   │   └── test_e2e/
│   │   └── fixtures/                 # Test data
│   │       ├── sample_documents/
│   │       └── mock_responses/
│   ├── requirements/
│   │   ├── base.txt                  # Core dependencies
│   │   ├── dev.txt                   # Development dependencies
│   │   └── test.txt                  # Testing dependencies
│   ├── alembic/                      # Database migrations (if needed)
│   ├── Dockerfile
│   └── pytest.ini
│
├── data/                             # Local data storage
│   ├── cache/                        # Temporary file cache
│   ├── vector_db/                    # ChromaDB persistence
│   ├── knowledge_graphs/             # Saved graph data
│   └── sessions/                     # User session data
│
├── models/                           # Local ML models
│   ├── embeddings/                   # sentence-transformers cache
│   ├── whisper/                      # Whisper model cache
│   └── llm/                          # Local LLM models (Ollama)
│
├── docs/                            # Documentation
│   ├── api/                         # API documentation
│   ├── architecture/                # System design docs
│   ├── deployment/                  # Deployment guides
│   └── demo/                        # Demo materials
│
├── scripts/                         # Utility scripts
│   ├── setup.sh                     # Environment setup
│   ├── download_models.py           # Download required models
│   ├── migrate_db.py                # Database utilities
│   └── demo_data_generator.py       # Generate demo folder
│
├── monitoring/                      # Performance monitoring
│   ├── metrics.py                   # Performance metrics
│   └── logging_config.yml           # Structured logging
│
└── deployment/                      # Deployment configuration
    ├── docker/
    ├── nginx/
    └── systemd/