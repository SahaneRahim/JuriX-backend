# JuriX Backend — Production Dockerfile
# Python 3.11 avec support ML (scikit-learn, spaCy, fastText)

FROM python:3.11-slim

LABEL maintainer="JuriX Team <support@jurix.cm>"
LABEL description="JuriX v2.1 Backend API"
LABEL version="2.1.0"

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

# Upgrade pip and install dependencies blazingly fast
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Download spaCy models
RUN python -m spacy download fr_core_news_sm --quiet || echo "fr model download failed" && \
    python -m spacy download en_core_web_sm --quiet || echo "en model download failed"

# Create required directories
RUN mkdir -p /app/models/spacy /app/models/fasttext /app/logs /app/data

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

# Railway injects the PORT env var automatically
# Use 1 worker; scale up depending on Railway plan
CMD uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --workers 1 \
    --log-level info
