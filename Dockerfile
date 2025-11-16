FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps (for some pip packages)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy app code (excluding large files via .dockerignore)
COPY . /app

# Create data directories for transcripts and ChromaDB
RUN mkdir -p /app/data/chroma_db /app/transcripts

# Debug: Verify transcripts were copied
RUN echo "=== Checking transcripts ===" && \
    ls -la /app/transcripts/ | head -10 && \
    echo "Total transcript files: $(find /app/transcripts -name '*.md' | wc -l)"

# Set to Ask-Ads-Marketing deploy profile
ENV WIKI_VAULT_CONFIG=/app/configs/ask-ads-marketing-deploy.yaml
ENV PYTHONPATH=/app

# Railway provides PORT environment variable
EXPOSE 8000

# Initialize data directories and start the Remote MCP Server
# Note: Railway volume should be mounted at /data for persistent ChromaDB storage
# If volume exists with ChromaDB, it will be used. Otherwise, ingestion runs on first deploy.
# Set FORCE_REINDEX=true environment variable to force re-ingestion
CMD mkdir -p /app/static && \
    if [ -d "/data" ]; then \
      echo "Using Railway volume at /data for persistent storage"; \
      mkdir -p /data/chroma_db; \
      ln -sfn /data/chroma_db /app/data/chroma_db; \
    else \
      echo "No volume found - using ephemeral storage (will rebuild on each deploy)"; \
      mkdir -p /app/data/chroma_db; \
    fi && \
    if [ "$FORCE_REINDEX" = "true" ]; then \
      echo "FORCE_REINDEX=true detected - clearing old database and re-ingesting..."; \
      rm -rf /app/data/chroma_db/*; \
      python3 scripts/ingest.py; \
    elif [ ! -f /app/data/chroma_db/chroma.sqlite3 ]; then \
      echo "ChromaDB not found - running ingestion (this takes ~10 minutes)..."; \
      python3 scripts/ingest.py; \
    else \
      echo "ChromaDB found - skipping ingestion"; \
    fi && \
    uvicorn remote_mcp_server:app --host 0.0.0.0 --port ${PORT:-8000}
