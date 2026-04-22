FROM python:3.11-slim

# Install system dependencies for OCR and PDF rendering
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        tesseract-ocr \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy and install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and data
COPY src/ src/
COPY main.py .
COPY tests/ tests/
COPY data/ data/

# Ensure output directories exist
RUN mkdir -p outputs/embeddings outputs/plots outputs/models outputs/results

CMD ["python", "main.py"]
