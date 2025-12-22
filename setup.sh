#!/bin/bash

# Foldex Docker Setup Script
# This script sets up the entire Foldex pipeline using Docker Compose

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Helper functions
print_info() { echo -e "${BLUE}ℹ${NC} $1"; }
print_success() { echo -e "${GREEN}✓${NC} $1"; }
print_warning() { echo -e "${YELLOW}⚠${NC} $1"; }
print_error() { echo -e "${RED}✗${NC} $1"; }
print_step() { echo -e "\n${CYAN}▶${NC} $1"; }

# Banner
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           Foldex Docker Setup Script                        ║"
echo "║           Complete Pipeline Setup                            ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Step 1: Check prerequisites
print_step "Step 1: Checking prerequisites..."

# Check Docker
if ! command -v docker &> /dev/null; then
    print_error "Docker is required but not installed"
    echo "Please install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi
DOCKER_VERSION=$(docker --version | awk '{print $3}' | tr -d ',')
print_success "Docker $DOCKER_VERSION found"

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    print_error "Docker Compose is required but not installed"
    echo "Please install Docker Compose from: https://docs.docker.com/compose/install/"
    exit 1
fi

# Check which compose command to use
if docker compose version &> /dev/null; then
    COMPOSE_CMD="docker compose"
    print_success "Docker Compose (plugin) found"
else
    COMPOSE_CMD="docker-compose"
    COMPOSE_VERSION=$($COMPOSE_CMD --version | awk '{print $3}' | tr -d ',')
    print_success "Docker Compose $COMPOSE_VERSION found"
fi

# Check if Docker daemon is running
if ! docker info &> /dev/null; then
    print_error "Docker daemon is not running"
    echo "Please start Docker Desktop or the Docker daemon"
    exit 1
fi
print_success "Docker daemon is running"

# Step 2: Create necessary directories
print_step "Step 2: Creating data directories..."

mkdir -p data/{cache,vector_db,knowledge_graphs,sessions}
mkdir -p models/{embeddings,whisper,llm}
print_success "Data directories created"

# Step 3: Create .env file if it doesn't exist
print_step "Step 3: Setting up environment configuration..."

if [ ! -f ".env" ]; then
    print_info "Creating .env file from template..."
    
    # Generate a secure secret key
    SECRET_KEY=$(openssl rand -hex 32 2>/dev/null || python3 -c "import secrets; print(secrets.token_hex(32))" 2>/dev/null || echo "7d41abbf1e7f065fa34d9b8bac82c7fc554a0edf1fbeb417a3aac9932c05a33c")
    
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
# Get credentials from: https://console.cloud.google.com/apis/credentials
GOOGLE_CLIENT_ID=
GOOGLE_CLIENT_SECRET=
GOOGLE_REDIRECT_URI=http://localhost:3000/auth/callback

# Local LLM Configuration (Ollama)
OLLAMA_BASE_URL=http://ollama:11434
OLLAMA_MODEL=llama3.2:3b
OLLAMA_TIMEOUT=300
OLLAMA_KEEP_ALIVE=-1

# Embedding Model
EMBEDDING_MODEL=nomic-embed-text:latest
EMBEDDING_TYPE=ollama
EMBEDDING_DEVICE=cpu
EMBEDDING_BATCH_SIZE=32

# Reranking Configuration
USE_RERANKING=true
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

# Whisper Configuration
WHISPER_MODEL=base
WHISPER_DEVICE=cpu

# Security
SECRET_KEY=${SECRET_KEY}
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Performance
MAX_CONCURRENT_PROCESSING=4
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
MAX_FILE_SIZE_MB=100

# CORS Configuration
CORS_ORIGINS=["http://localhost:3000","http://localhost:5173"]

# ChromaDB Configuration
CHROMADB_HOST=chromadb
CHROMADB_PORT=8000
EOF
    print_success ".env file created"
    print_warning "Please configure GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env for Google Drive integration"
else
    print_info ".env file already exists, skipping creation"
fi

# Step 4: Pull Docker images
print_step "Step 4: Pulling Docker images..."

print_info "Pulling ChromaDB image..."
docker pull chromadb/chroma:latest || print_warning "Failed to pull ChromaDB image (will pull during compose)"

print_info "Pulling Ollama image..."
docker pull ollama/ollama:latest || print_warning "Failed to pull Ollama image (will pull during compose)"

print_success "Docker images ready"

# Step 5: Build containers
print_step "Step 5: Building containers..."

print_info "Building backend container..."
$COMPOSE_CMD build backend || {
    print_error "Failed to build backend container"
    exit 1
}

print_info "Building frontend container..."
$COMPOSE_CMD build frontend || {
    print_error "Failed to build frontend container"
    exit 1
}

print_success "Containers built successfully"

# Step 6: Start services
print_step "Step 6: Starting services..."

print_info "Starting ChromaDB..."
$COMPOSE_CMD up -d chromadb

# Wait for ChromaDB to be healthy
print_info "Waiting for ChromaDB to be ready..."
MAX_WAIT=60
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if docker exec foldex-chromadb timeout 1 bash -c '</dev/tcp/localhost/8000' 2>/dev/null; then
        print_success "ChromaDB is ready"
        break
    fi
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
    echo -n "."
done
echo ""

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    print_warning "ChromaDB health check timeout, but continuing..."
fi

print_info "Starting Ollama..."
$COMPOSE_CMD up -d ollama

# Wait for Ollama to be ready
print_info "Waiting for Ollama to be ready..."
MAX_WAIT=60
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if docker exec foldex-ollama timeout 1 bash -c '</dev/tcp/localhost/11434' 2>/dev/null; then
        print_success "Ollama is ready"
        break
    fi
    sleep 2
    WAIT_COUNT=$((WAIT_COUNT + 2))
    echo -n "."
done
echo ""

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    print_warning "Ollama health check timeout, but continuing..."
fi

print_info "Starting backend and frontend..."
$COMPOSE_CMD up -d backend frontend

print_success "All services started"

# Step 7: Pull and warm up Ollama models
print_step "Step 7: Setting up Ollama models..."

print_info "Pulling LLM model (llama3.2:3b)..."
docker exec foldex-ollama ollama pull llama3.2:3b || {
    print_warning "Failed to pull llama3.2:3b, trying alternative..."
    docker exec foldex-ollama ollama pull llama3.2 || print_warning "Model pull failed, you may need to pull manually"
}

print_info "Pulling embedding model (nomic-embed-text:latest)..."
docker exec foldex-ollama ollama pull nomic-embed-text:latest || {
    print_warning "Failed to pull nomic-embed-text:latest, trying alternative..."
    docker exec foldex-ollama ollama pull nomic-embed-text || print_warning "Embedding model pull failed, you may need to pull manually"
}

# Warm up the model
print_info "Warming up LLM model (this may take a minute)..."
docker exec foldex-ollama ollama run llama3.2:3b "Hello, this is a warmup request." > /dev/null 2>&1 || {
    print_warning "Model warmup failed, but this is not critical"
}

print_success "Ollama models ready"

# Step 8: Wait for backend to be healthy
print_step "Step 8: Verifying services..."

print_info "Waiting for backend to be ready..."
MAX_WAIT=120
WAIT_COUNT=0
while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    if curl -f http://localhost:8000/health > /dev/null 2>&1; then
        print_success "Backend is healthy"
        break
    fi
    sleep 3
    WAIT_COUNT=$((WAIT_COUNT + 3))
    echo -n "."
done
echo ""

if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    print_warning "Backend health check timeout"
    print_info "Check logs with: $COMPOSE_CMD logs backend"
    print_info "Backend may still be starting (this can take 30-60 seconds)"
else
    print_success "Backend is responding"
fi

# Check frontend
print_info "Checking frontend..."
sleep 10
if command -v curl &> /dev/null; then
    if curl -f http://localhost:3000 > /dev/null 2>&1; then
        print_success "Frontend is responding"
    else
        print_warning "Frontend may still be starting (this is normal, can take 30-60 seconds)"
        print_info "Check logs with: $COMPOSE_CMD logs frontend"
    fi
else
    if docker ps | grep -q foldex-frontend; then
        print_success "Frontend container is running"
        print_info "Frontend may take 30-60 seconds to fully start"
    else
        print_warning "Frontend container not found"
    fi
fi

# Step 9: Display status
print_step "Step 9: Service status"

echo ""
echo -e "${CYAN}Service Status:${NC}"
$COMPOSE_CMD ps

echo ""
echo -e "${CYAN}Container Logs (last 10 lines):${NC}"
echo -e "${YELLOW}Backend:${NC}"
docker logs --tail 10 foldex-backend 2>&1 | head -10 || true
echo ""
echo -e "${YELLOW}Frontend:${NC}"
docker logs --tail 10 foldex-frontend 2>&1 | head -10 || true

# Final summary
echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║                    Setup Complete!                           ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${CYAN}Access your application:${NC}"
echo -e "  ${BLUE}Frontend:${NC}  http://localhost:3000"
echo -e "  ${BLUE}Backend API:${NC}  http://localhost:8000/api/docs"
echo -e "  ${BLUE}ChromaDB:${NC}  http://localhost:8001"
echo ""
echo -e "${CYAN}Useful commands:${NC}"
echo -e "  ${BLUE}View logs:${NC}        $COMPOSE_CMD logs -f [service]"
echo -e "  ${BLUE}Stop services:${NC}     $COMPOSE_CMD down"
echo -e "  ${BLUE}Restart services:${NC}  $COMPOSE_CMD restart [service]"
echo -e "  ${BLUE}View status:${NC}      $COMPOSE_CMD ps"
echo ""
echo -e "${YELLOW}⚠ Important:${NC}"
echo -e "  1. Configure GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET in .env for Google Drive integration"
echo -e "  2. Models are kept loaded in memory (OLLAMA_KEEP_ALIVE=-1) for optimal performance"
echo -e "  3. First query may take longer as models initialize"
echo ""
echo -e "${GREEN}Happy coding! 🚀${NC}"
