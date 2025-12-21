#!/bin/bash
# Foldex startup script with model warmup and service initialization

set -e

echo "=========================================="
echo "Starting Foldex Services"
echo "=========================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Check if Ollama is running
echo -e "${YELLOW}Checking Ollama service...${NC}"
if ! pgrep -x "ollama" > /dev/null; then
    echo -e "${YELLOW}Starting Ollama...${NC}"
    ollama serve &
    sleep 3
    echo -e "${GREEN}Ollama started${NC}"
else
    echo -e "${GREEN}Ollama is already running${NC}"
fi

# 2. Warmup model
echo -e "${YELLOW}Warming up model...${NC}"
cd "$(dirname "$0")/.."
python scripts/warmup_model.py
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Model warmed up successfully${NC}"
else
    echo -e "${YELLOW}Model warmup failed (continuing anyway)${NC}"
fi

# 3. Check if Redis is running (optional)
if command -v redis-cli &> /dev/null; then
    echo -e "${YELLOW}Checking Redis service...${NC}"
    if ! redis-cli ping > /dev/null 2>&1; then
        echo -e "${YELLOW}Redis not running (optional - caching will use in-memory)${NC}"
    else
        echo -e "${GREEN}Redis is running${NC}"
    fi
else
    echo -e "${YELLOW}Redis not installed (optional - caching will use in-memory)${NC}"
fi

# 4. Start FastAPI backend
echo -e "${YELLOW}Starting FastAPI backend...${NC}"
cd backend
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    echo -e "${YELLOW}Warning: Virtual environment not found${NC}"
fi

# Start backend in background
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo -e "${GREEN}Backend started (PID: $BACKEND_PID)${NC}"

# 5. Start React frontend (optional)
if [ -d "../frontend" ]; then
    echo -e "${YELLOW}Starting React frontend...${NC}"
    cd ../frontend
    npm start &
    FRONTEND_PID=$!
    echo -e "${GREEN}Frontend started (PID: $FRONTEND_PID)${NC}"
else
    echo -e "${YELLOW}Frontend directory not found, skipping${NC}"
fi

echo ""
echo "=========================================="
echo -e "${GREEN}All services started!${NC}"
echo "=========================================="
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"
echo ""
echo "To stop services, press Ctrl+C or run:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo ""

# Wait for user interrupt
wait

