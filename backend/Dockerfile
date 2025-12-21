FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including Tesseract OCR for image processing)
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    tesseract-ocr \
    libtesseract-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements/ requirements/
RUN pip install --no-cache-dir -r requirements/base.txt

# Download spaCy model (after spacy is installed)
RUN python -m spacy download en_core_web_sm || \
    pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1-py3-none-any.whl || \
    echo "Warning: spaCy model installation failed, but continuing..."

# Download NLTK data (required by unstructured)
RUN python -m nltk.downloader punkt punkt_tab averaged_perceptron_tagger averaged_perceptron_tagger_eng wordnet stopwords || echo "Warning: NLTK data download failed, but continuing..."

# Copy application code
COPY app/ app/

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

