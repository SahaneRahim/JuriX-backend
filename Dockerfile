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
COPY requirements.txt constraints.txt ./

# constraints.txt fige chaque version sur celle du venv de developpement, ou la
# suite de tests passe (scripts/figer_versions.sh). Sans lui, l'image prenait
# les dernieres versions du jour de sa construction, jamais testees ensemble :
# SQLAlchemy 2.1 l'a fait planter au demarrage.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Utilisateur sans privileges, et modele fastText de detection de langue
# (126 Mo), AVANT la copie du code : cette couche reste alors en cache d'une
# mise a jour a l'autre. Placee apres, chaque modification du code
# retelechargeait le modele, puis le chown le recopiait dans une couche de
# plus ; et un telechargement rate donnait, sans erreur, une image sans
# detection de langue. Un echec fait desormais echouer la construction :
# `docker compose up --build` s'arrete, et l'ancienne version continue de
# tourner.
RUN useradd -m -u 1000 -s /bin/bash jurix && \
    mkdir -p /app/models/fasttext /app/logs /app/data && \
    wget -q https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin \
         -O /app/models/fasttext/lid.176.bin && \
    chown -R jurix:jurix /app

# Copie du code applicatif.
# Ce que ce COPY embarque est filtre par .dockerignore : sans lui, le fichier
# .env — donc les cles d'API — se retrouvait dans une couche de l'image, lisible
# par quiconque peut la telecharger. --chown evite un `chown -R` apres coup,
# qui recopiait tout /app dans une nouvelle couche.
COPY --chown=jurix:jurix . .

USER jurix

# Runtime env vars
# ML_MODELS_PATH et LOG_LEVEL ont ete retirees : aucun code ne les lisait.
# language_detector.py resout lui-meme le chemin du modele, et la journalisation
# est configuree dans app/main.py. Des variables d'environnement que personne ne
# lit donnent l'illusion d'un reglage.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    TESSERACT_PATH=/usr/bin/tesseract

# Les migrations sont appliquees au demarrage : le schema du produit vit dans
# alembic (search_vector, index GIN, declencheurs, extensions), pas dans
# Base.metadata — un conteneur qui demarre sans les avoir jouees repond 500 sur
# toute recherche.
# Railway injects PORT automatically.
#
# Forme JSON avec `exec` : en forme shell, docker lance `/bin/sh -c "..."` et
# c'est le SHELL qui recoit le SIGTERM de `docker stop`. uvicorn, lui, ne le
# voit jamais et se fait tuer au bout du delai de grace — donc le lifespan
# n'atteint jamais son `finally`, ou vivent la fermeture du pool de connexions
# et l'arret de la tache de purge des caches. `exec` fait remplacer le shell
# par uvicorn, qui recoit alors le signal directement.
#
# Le shell reste necessaire pour `&&` et pour l'expansion de ${PORT}.
#
# --no-access-log : le journal d'acces d'uvicorn ecrit l'URL entiere, chaine de
# requete comprise. Or le WebSocket de l'envoi par lots porte le jeton
# d'administration dans l'URL (?token=...) : il finissait en clair dans
# `docker compose logs`. L'application journalise elle-meme ce qui compte.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --log-level info --no-access-log"]
