#!/bin/bash

# Foldex Development Environment Setup Script
set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

print_info() { echo -e "${BLUE}ℹ${NC} $1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }

echo -e "${BLUE}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           Foldex Development Environment Setup               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Check prerequisites
print_info "Checking prerequisites..."

# Check Python
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed"
    exit 1
fi
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
print_success "Python $PYTHON_VERSION found"

# Check Node.js
if ! command -v node &> /dev/null; then
    print_error "Node.js is required but not installed"
    exit 1
fi
NODE_VERSION=$(node --version)
print_success "Node.js $NODE_VERSION found"

# Check npm
if ! command -v npm &> /dev/null; then
    print_error "npm is required but not installed"
    exit 1
fi
NPM_VERSION=$(npm --version)
print_success "npm $NPM_VERSION found"

# Check Docker (optional)
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | awk '{print $3}' | tr -d ',')
    print_success "Docker $DOCKER_VERSION found"
    DOCKER_AVAILABLE=true
else
    print_warning "Docker not found (optional for local services)"
    DOCKER_AVAILABLE=false
fi

echo ""

# Create virtual environment
print_info "Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    print_success "Virtual environment created"
else
    print_info "Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate
print_success "Virtual environment activated"

# Install backend dependencies
print_info "Installing backend dependencies..."
cd backend
pip install --upgrade pip --quiet
pip install -r requirements/base.txt --quiet
if [ -f "requirements/dev.txt" ]; then
    pip install -r requirements/dev.txt --quiet
fi
print_success "Backend dependencies installed"
cd ..

# Install frontend dependencies
print_info "Installing frontend dependencies..."
cd frontend
npm install --silent
print_success "Frontend dependencies installed"
cd ..

# Create data directories
print_info "Creating data directories..."
mkdir -p data/{cache,vector_db,knowledge_graphs,sessions}
print_success "Data directories created"

# Create model directories
print_info "Creating model directories..."
mkdir -p models/{embeddings,whisper,llm}
print_success "Model directories created"

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_info "Creating .env file from template..."
    cat > .env << EOF
# Foldex Environment Configuration
APP_ENV=development
DEBUG=true
LOG_LEVEL=INFO

# Server Configuration
BACKEND_HOST=0.0.0.0
BACKEND_PORT=8000
FRONTEND_PORT=3000

# Database Configuration
DATABASE_PATH=./data/foldex.db
VECTOR_DB_PATH=./data/vector_db
CACHE_DIR=./data/cache
KNOWLEDGE_GRAPH_DIR=./data/knowledge_graphs
SESSIONS_DIR=./data/sessions

# Google Drive API (Required for production)
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REDIRECT_URI=http://localhost:3000/auth/callback

# Local LLM Configuration (Ollama)
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
OLLAMA_TIMEOUT=300

# Embedding Model
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=32

# Whisper Configuration
WHISPER_MODEL=base
WHISPER_DEVICE=cpu

# Security
SECRET_KEY=$(openssl rand -hex 32)
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Performance
MAX_CONCURRENT_PROCESSING=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_FILE_SIZE_MB=100
EOF
    print_success ".env file created"
else
    print_info ".env file already exists"
fi

# Initialize database
print_info "Initializing database..."
cd backend
python3 -c "
import asyncio
from app.database.base import initialize_database

async def init():
    await initialize_database()
    print('Database initialized successfully')

asyncio.run(init())
" 2>/dev/null || print_warning "Database initialization skipped (will be created on first run)"
cd ..

echo ""
print_success "Setup complete!"
echo ""
echo -e "${GREEN}Next steps:${NC}"
echo "  1. Configure Google OAuth credentials in .env (if using Google Drive)"
echo "  2. Download ML models: ${BLUE}python scripts/download_models.py${NC}"
echo "  3. Start Ollama (if using local LLM): ${BLUE}ollama serve${NC}"
echo "  4. Start backend: ${BLUE}cd backend && source ../venv/bin/activate && uvicorn app.main:app --reload${NC}"
echo "  5. Start frontend: ${BLUE}cd frontend && npm run dev${NC}"
echo ""
echo -e "${YELLOW}Or use Docker Compose:${NC}"
echo "  ${BLUE}docker-compose up -d${NC}"
echo ""
