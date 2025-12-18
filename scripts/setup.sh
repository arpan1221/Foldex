#!/bin/bash

# Foldex setup script
set -e

echo "Setting up Foldex development environment..."

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install backend dependencies
echo "Installing backend dependencies..."
cd backend
pip install --upgrade pip
pip install -r requirements/base.txt
cd ..

# Install frontend dependencies
echo "Installing frontend dependencies..."
cd frontend
npm install
cd ..

# Create data directories
echo "Creating data directories..."
mkdir -p data/cache
mkdir -p data/vector_db
mkdir -p data/knowledge_graphs
mkdir -p data/sessions

# Create model directories
echo "Creating model directories..."
mkdir -p models/embeddings
mkdir -p models/whisper
mkdir -p models/llm

echo "Setup complete!"
echo ""
echo "Next steps:"
echo "1. Copy .env.example to .env and configure your settings"
echo "2. Run 'python scripts/download_models.py' to download ML models"
echo "3. Start the backend: 'npm run dev:backend'"
echo "4. Start the frontend: 'npm run dev:frontend'"

