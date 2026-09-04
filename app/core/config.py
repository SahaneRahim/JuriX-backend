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
    # Dimension demandee a l'API (output_dimensionality) : la sortie native du
    # modele, sans troncature. Le plafond de 2000 dimensions de pgvector ne
    # s'applique qu'au type `vector` ; l'index est pose sur une expression
    # `halfvec(3072)`, qui monte a 4000 (migration f5a6b7c8d9e0). On garde donc
    # la pleine precision en stockage et un index utilisable.
    # Doit rester egal a la dimension declaree sur Article.embedding.
    EMBEDDING_DIM: int = 3072

    # Delai maximal d'un appel Gemini, en SECONDES. Sans lui, le client
    # n'impose AUCUN delai : une prise reseau qui ne repond plus bloque
    # indefiniment, sans exception, donc sans declencher la moindre reprise.
    # Observe en conditions reelles : une ingestion figee 40 minutes sur un
    # appel d'embeddings, processus vivant, zero UC consommee.
    GEMINI_TIMEOUT_S: int = 120

    # ---- Re-ranking des chunks (app/services/reranker.py) ----
    # Etage 1 : traits lexicaux, sans reseau ni dependance. Quelques
    # millisecondes, actif par defaut.
    RERANK_ENABLED: bool = True
    # Etage 2 : notation des meilleurs chunks par Gemini. Ajoute un appel
    # facture et 400 a 900 ms sur le chemin critique, d'ou le defaut a False :
    # a n'activer qu'apres l'avoir vu gagner sur le lot d'evaluation tenu a
    # l'ecart.
    RERANK_LLM_ENABLED: bool = False
    RERANK_LLM_TOP_N: int = 20
    RERANK_LLM_TIMEOUT_S: float = 4.0

    # ---- Fusion hybride ----
    # Valeurs par defaut NON calibrees sur ce corpus : RRF_K = 60 vient du
    # papier d'origine sur des runs TREC. A remplacer par les valeurs issues
    # de scripts/eval/run_eval.py, en citant le fichier de run en commentaire.
    RRF_K: int = 60
    TEXT_WEIGHT: float = 0.4
    SEMANTIC_WEIGHT: float = 0.6

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
