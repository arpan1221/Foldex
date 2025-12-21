# Foldex

**Foldex** is a local-first multimodal RAG (Retrieval-Augmented Generation) system that transforms Google Drive folders into intelligent conversation interfaces. Ask questions, find files, and get insights from your documents using AI.

## Features

- 🔐 **Google Drive Integration**: Authenticate and process folders from Google Drive
- 📄 **Multimodal Processing**: Supports PDFs, text files, Markdown, audio (Whisper), and code
- 🧠 **Intelligent RAG**: Hybrid retrieval with semantic search, keyword matching, and knowledge graphs
- 💬 **Conversational Interface**: Chat with your documents with precise citations
- 📊 **Knowledge Graph Visualization**: Interactive graph showing document relationships
- 🔒 **Local-First**: All processing and storage happens locally on your machine
- ⚡ **Real-time Updates**: WebSocket-based progress tracking for folder processing

## Architecture

- **Backend**: FastAPI + SQLite + ChromaDB + Local LLMs (Ollama)
- **Frontend**: React + TypeScript + Tailwind CSS
- **Processing**: Audio (Whisper) + Text + PDF + Code analysis
- **RAG Engine**: Hybrid retrieval (semantic + keyword + knowledge graph)
- **Storage**: Local-first, no cloud services except Google Drive API

## Prerequisites

- **Python 3.10+**
- **Node.js 18+** and npm
- **Docker** (optional, for ChromaDB and Ollama)
- **Ollama** (for local LLM inference)
- **Google OAuth2 Credentials** (for Google Drive access)

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd Foldex
```

### 2. Run Setup Script

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

This will:
- Create Python virtual environment
- Install backend and frontend dependencies
- Create necessary directories
- Generate `.env` file with default settings

### 3. Configure Environment

Edit `.env` file and add your Google OAuth2 credentials:

```bash
GOOGLE_CLIENT_ID=your-client-id
GOOGLE_CLIENT_SECRET=your-client-secret
```

**Optional: LangSmith Observability**

For production monitoring and debugging, configure LangSmith tracing:

```bash
# Get your API key from https://smith.langchain.com
LANGCHAIN_API_KEY=your-langsmith-api-key
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=foldex
```

This enables full observability of the LangGraph multi-document synthesis pipeline.

### 4. Download ML Models

```bash
source venv/bin/activate
python scripts/download_models.py
```

This downloads:
- Sentence transformer model for embeddings
- Whisper model for audio transcription

### 5. Start Ollama (for Local LLM)

```bash
# Install Ollama: https://ollama.ai
ollama serve

# In another terminal, pull the model:
ollama pull llama3.2
```

### 6. Start Development Servers

**Option A: Manual Start**

```bash
# Terminal 1: Backend
cd backend
source ../venv/bin/activate
uvicorn app.main:app --reload

# Terminal 2: Frontend
cd frontend
npm run dev
```

**Option B: Docker Compose**

```bash
docker-compose up -d
```

This starts:
- ChromaDB (vector database) on port 8001
- Backend API on port 8000
- Frontend on port 3000
- Ollama on port 11434

### 7. Access the Application

Open your browser and navigate to:
- **Frontend**: http://localhost:3000
- **Backend API Docs**: http://localhost:8000/api/docs

## Development Setup

### Backend Development

```bash
# Activate virtual environment
source venv/bin/activate

# Install development dependencies
cd backend
pip install -r requirements/dev.txt

# Run tests
pytest tests/ --cov=app

# Run with hot reload
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development

```bash
cd frontend

# Install dependencies
npm install

# Run development server
npm run dev

# Run tests
npm test

# Build for production
npm run build
```

## Project Structure

```
Foldex/
├── backend/                 # FastAPI backend
│   ├── app/
│   │   ├── api/             # API routes
│   │   ├── services/        # Business logic
│   │   ├── processors/      # Document processors
│   │   ├── rag/             # RAG engine
│   │   ├── knowledge_graph/ # Knowledge graph
│   │   └── database/        # Database layer
│   ├── tests/               # Backend tests
│   └── requirements/        # Python dependencies
├── frontend/                 # React frontend
│   ├── src/
│   │   ├── components/      # React components
│   │   ├── hooks/           # Custom hooks
│   │   ├── services/        # API clients
│   │   └── utils/           # Utilities
│   └── package.json
├── scripts/                  # Setup and utility scripts
├── data/                     # Local data storage
├── models/                   # ML models
├── docker-compose.yml        # Docker services
└── README.md
```

## Configuration

### Environment Variables

Key environment variables (see `.env.example` for full list):

- `GOOGLE_CLIENT_ID`: Google OAuth2 client ID
- `GOOGLE_CLIENT_SECRET`: Google OAuth2 client secret
- `OLLAMA_MODEL`: Local LLM model name (default: `llama3.2`)
- `EMBEDDING_MODEL`: Embedding model (default: `sentence-transformers/all-mpnet-base-v2`)
- `SECRET_KEY`: JWT signing key (generate with `openssl rand -hex 32`)

### Database

Foldex uses:
- **SQLite** for metadata storage (default: `./data/foldex.db`)
- **ChromaDB** for vector embeddings (default: `./data/vector_db`)

Both are automatically initialized on first run.

## Usage

### 1. Authenticate with Google Drive

1. Click "Sign in with Google" on the landing page
2. Grant permissions to access Google Drive
3. You'll be redirected back to the application

### 2. Process a Folder

1. Paste a Google Drive folder URL
2. Click "Process Folder"
3. Monitor progress via WebSocket updates
4. Wait for processing to complete

### 3. Chat with Your Documents

1. Navigate to the chat interface
2. Ask questions about your documents
3. View citations and source references
4. Explore the knowledge graph visualization

## API Documentation

Interactive API documentation is available at:
- **Swagger UI**: http://localhost:8000/api/docs
- **ReDoc**: http://localhost:8000/api/redoc

## Testing

### Backend Tests

```bash
cd backend
source ../venv/bin/activate
pytest tests/ -v
pytest tests/ --cov=app --cov-report=html
```

### Frontend Tests

```bash
cd frontend
npm test
npm test -- --coverage
```

## Troubleshooting

### Common Issues

**1. Port Already in Use**
```bash
# Change ports in .env or docker-compose.yml
BACKEND_PORT=8001
FRONTEND_PORT=3001
```

**2. Ollama Connection Failed**
```bash
# Ensure Ollama is running
ollama serve

# Check if model is available
ollama list
```

**3. Google OAuth Errors**
- Verify `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` in `.env`
- Ensure redirect URI matches: `http://localhost:3000/auth/callback`
- Check OAuth consent screen configuration in Google Cloud Console

**4. Model Download Fails**
```bash
# Manually download models
python scripts/download_models.py

# Or set custom model paths in .env
EMBEDDING_MODEL=your-custom-model
```

**5. Database Errors**
```bash
# Reset database (WARNING: Deletes all data)
rm -rf data/foldex.db data/vector_db
# Restart application to recreate
```

## Performance

- **Folder Indexing**: < 2 minutes for 20 documents (mixed types)
- **Query Response**: < 15 seconds for complex cross-document queries
- **Memory Usage**: < 4GB peak during indexing
- **Storage**: < 10MB additional data per 1MB source content

### Performance Optimizations

Foldex includes several optimizations to reduce latency and improve response times:

#### 1. Model Keep-Alive Configuration

The LLM model (llama3.2:3b) is kept loaded in memory to eliminate cold-start latency:

```bash
# In docker-compose.yml or .env
OLLAMA_KEEP_ALIVE=-1  # Keep model loaded indefinitely
```

This ensures the model is always ready for inference, reducing first-token latency from ~5-10 seconds to <1 second.

#### 2. Query Embedding Cache

An intelligent LRU cache stores query embeddings to avoid redundant embedding generation:

- **Cache Size**: 1000 queries (configurable via `EMBEDDING_CACHE_MAX_SIZE`)
- **TTL**: 1 hour (configurable via `EMBEDDING_CACHE_TTL`)
- **Hit Rate**: Typically 30-50% for repeated queries
- **Latency Reduction**: ~200-500ms per cached query

The cache automatically:
- Normalizes queries (lowercase, strip whitespace) for better hit rates
- Evicts least-recently-used entries when full
- Expires entries after TTL
- Provides statistics via API endpoint

#### 3. Automatic Model Warmup

On startup, the application automatically:
- Pre-loads the LLM model into memory
- Initializes TTFT (Time-To-First-Token) optimizer
- Warms up embedding models
- Prepares caches and connection pools

This eliminates cold-start delays for the first user request.

#### 4. Manual Warmup Endpoints

For fine-grained control, use these API endpoints:

```bash
# Warmup LLM model
curl -X POST http://localhost:8000/api/v1/warmup/model

# Initialize TTFT optimizer
curl -X POST http://localhost:8000/api/v1/warmup/ttft

# Get cache statistics
curl http://localhost:8000/api/v1/warmup/cache/stats

# Clear embedding cache
curl -X POST http://localhost:8000/api/v1/warmup/cache/clear

# Cleanup expired cache entries
curl -X POST http://localhost:8000/api/v1/warmup/cache/cleanup
```

#### Configuration

Optimize performance by adjusting these settings in `.env`:

```bash
# Model Keep-Alive
OLLAMA_KEEP_ALIVE=-1              # -1 = keep loaded, 0 = unload immediately

# Embedding Cache
EMBEDDING_CACHE_ENABLED=true      # Enable/disable cache
EMBEDDING_CACHE_MAX_SIZE=1000     # Maximum cached queries
EMBEDDING_CACHE_TTL=3600          # Cache TTL in seconds (1 hour)

# Model Warmup
ENABLE_MODEL_WARMUP=true          # Warmup on startup
ENABLE_TTFT_OPTIMIZATION=true     # Enable TTFT optimizer
```

#### Performance Monitoring

Monitor cache performance via the stats endpoint:

```bash
curl http://localhost:8000/api/v1/warmup/cache/stats
```

Response:
```json
{
  "status": "success",
  "cache_stats": {
    "size": 245,
    "max_size": 1000,
    "hits": 1523,
    "misses": 782,
    "hit_rate": 66.08,
    "ttl_seconds": 3600
  }
}
```

#### Expected Latency Improvements

With all optimizations enabled:

- **First Query (Cold Start)**: ~8-12 seconds → ~2-4 seconds
- **Subsequent Queries (Warm)**: ~5-8 seconds → ~1-3 seconds
- **Cached Queries**: ~5-8 seconds → ~0.5-2 seconds
- **Model Load Time**: ~5-10 seconds → 0 seconds (pre-loaded)

## Contributing

1. Follow the coding standards in `Claude.md`
2. Write tests for new features
3. Update `CHANGELOG.md` for significant changes
4. Use conventional commits

## License

[Your License Here]

## Support

For issues and questions, please open an issue on GitHub.

---

**Built with ❤️ for local-first AI document processing**

