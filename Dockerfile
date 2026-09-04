# JuriX Backend — Production Dockerfile
# Python 3.11 — PostgreSQL native (no Redis/Meilisearch/Celery)

FROM python:3.11-slim

LABEL maintainer="JuriX Team <support@jurix.cm>"
LABEL description="JuriX v3.0 Backend API"
LABEL version="3.0.0"

# Dependances systeme.
#
# tesseract-ocr et poppler-utils ne sont PAS optionnels :
#   - poppler fournit pdftoppm, dont depend pdf2image ; sans lui,
#     GET /laws/{id}/page/{n} — la visionneuse par images du front — repond 500
#     en production alors qu'il fonctionne en developpement ;
#   - tesseract-ocr et ses dictionnaires francais/anglais servent au repli OCR
#     et au diagnostic de la couche texte des PDF scannes.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    curl \
    git \
    wget \
    ca-certificates \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-fra \
    tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency file first (for Docker layer caching)
COPY requirements.txt ./

# Upgrade pip and install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copie du code applicatif.
# Ce que ce COPY embarque est filtre par .dockerignore : sans lui, le fichier
# .env — donc les cles d'API — se retrouvait dans une couche de l'image, lisible
# par quiconque peut la telecharger.
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
    LOG_LEVEL=INFO \
    TESSERACT_PATH=/usr/bin/tesseract

# Les migrations sont appliquees au demarrage : le schema du produit vit dans
# alembic (search_vector, index GIN, declencheurs, extensions), pas dans
# Base.metadata — un conteneur qui demarre sans les avoir jouees repond 500 sur
# toute recherche.
# Railway injects PORT automatically
CMD alembic upgrade head && uvicorn app.main:app \
    --host 0.0.0.0 \
    --port ${PORT:-8000} \
    --workers 1 \
    --log-level info
