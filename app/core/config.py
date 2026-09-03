"""Configuration application - Toutes les variables d'environnement."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Settings de l'application."""

    # App
    APP_NAME: str = "JuriX API"
    VERSION: str = "2.1.0"
    DEBUG: bool = True
    ENVIRONMENT: str = "development"

    # Database
    DATABASE_URL: str = "postgresql+asyncpg://jurix:jurix_dev_password_change_in_prod@localhost:5432/jurix_db"

    # Recherche et cache assures par PostgreSQL (tsvector, pg_trgm, query_cache).

    # Gemini API (LLM for RAG)
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-3-flash"
    GEMINI_EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    # Dimension demandee a l'API (output_dimensionality). Le modele sort 3072
    # nativement, mais pgvector n'indexe pas au-dela de 2000 : a 1536 l'index
    # HNSW redevient possible. Doit rester egal a la dimension declaree sur
    # Article.embedding et a celle de la migration e4f5a6b7c8d9.
    EMBEDDING_DIM: int = 1536

    # LlamaParse (PDF extraction OCR)
    LLAMA_CLOUD_API_KEY: str = ""
    # Tier de parsing : fast(1cr) | cost_effective(3cr) | agentic(10cr) | agentic_plus(45cr)
    # cost_effective est le meilleur rapport qualite/prix mesure sur le corpus prc.cm
    LLAMA_PARSE_TIER: str = "cost_effective"
    # Cache OCR par sha256 — evite de repayer l'extraction d'un fichier deja traite
    OCR_CACHE_DIR: str = "./data/ocr_cache"

    # CORS — comma-separated list of extra allowed origins for production
    # Example: https://jurix.vercel.app,https://www.jurix.cm
    ALLOWED_ORIGINS: str = ""

    # Security
    SECRET_KEY: str = "dev_secret_key_change_in_production_with_openssl_rand_hex_32"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # QODO_API_KEY, ZEROSTEP_API_KEY et CORS_ORIGINS ont ete retires : les deux
    # premiers n'etaient lus nulle part, et main.py lit ALLOWED_ORIGINS, pas
    # CORS_ORIGINS — cette liste, qui contenait un joker "*", ne s'appliquait a
    # rien tout en donnant a lire le contraire.

    # Upload limits
    MAX_UPLOAD_SIZE: int = 1073741824  # 1GB in bytes for batch upload

    # OCR — chemin du binaire tesseract. Vide par defaut : sous Linux et dans
    # l'image Docker il est sur le PATH. Le defaut precedent etait un chemin
    # Windows, et n'etait de toute facon lu par personne.
    TESSERACT_PATH: str = ""

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields from .env


settings = Settings()
