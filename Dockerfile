# JuriX Backend — Production Dockerfile
# Python 3.11 — PostgreSQL native (no Redis/Meilisearch/Celery)

FROM python:3.11-slim

LABEL maintainer="JuriX Team <support@jurix.cm>"
LABEL description="JuriX v3.0 Backend API"
LABEL version="3.0.0"

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency file first (for Docker layer caching)
COPY requirements.txt ./

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create required directories
RUN mkdir -p /app/models/fasttext /app/logs /app/data

# Download fastText language identification model
RUN if [ ! -f /app/models/fasttext/lid.176.bin ]; then \
        echo "Downloading fastText model..." && \
        wget -q https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin \
             -O /app/models/fasttext/lid.176.bin || \
        echo "fastText model download failed - will skip language detection"; \
    fi

# Non-root user for security
RUN useradd -m -u 1000 -s /bin/bash jurix && \
    chown -R jurix:jurix /app

USER jurix

# Runtime env vars
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    ML_MODELS_PATH=/app/models \
    LOG_LEVEL=INFO

# Railway injects PORT automatically
CMD uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --workers 1 \
    --log-level info
