"""
Service de génération d'embeddings vectoriels pour recherche sémantique.

Ce service utilise Gemini API pour générer des embeddings 3072-dim
multilingues (FR/EN) avec cache PostgreSQL pour optimiser les performances.

Architecture:
- Model: models/gemini-embedding-001 (Gemini API)
- Dimensions: 3072 (compatible pgvector)
- Cache: Table embedding_cache PostgreSQL avec TTL 7 jours (remplace Redis)
- Performance: <300ms single, <2s batch(10)

Author: JuriX Team
Version: 3.0.0 (PostgreSQL cache - no Redis)
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import google.generativeai as genai
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingServiceError(Exception):
    """Exception levée lors d'erreurs de génération d'embeddings."""
    pass


class EmbeddingService:
    """
    Service de génération d'embeddings vectoriels pour recherche sémantique.

    Utilise Gemini API avec cache PostgreSQL (table embedding_cache).
    Supporte FR/EN, dimensions 3072, normalisation L2 automatique.

    Attributes:
        EMBEDDING_MODEL: Nom du modèle Gemini
        EMBEDDING_DIM: Dimension des embeddings (3072)
        CACHE_TTL_SECONDS: Durée de vie cache (7 jours)
        use_cache: Flag activation cache
    """

    EMBEDDING_MODEL = settings.GEMINI_EMBEDDING_MODEL
    EMBEDDING_DIM = 3072
    CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 jours
    MAX_TEXT_LENGTH = 10000
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds

    def __init__(
        self,
        use_cache: bool = True,
        api_key: Optional[str] = None,
    ):
        """
        Initialise le service d'embeddings.

        Configure Gemini API et optionnellement le cache PostgreSQL.

        Args:
            use_cache: Active/désactive le cache PostgreSQL
            api_key: Clé API Gemini (si None, utilise settings.GEMINI_API_KEY)

        Raises:
            EmbeddingServiceError: Si Gemini API ne peut pas être initialisée
        """
        logger.info("🚀 Initialisation EmbeddingService (Gemini API)...")

        # Initialisation Gemini API
        try:
            api_key = api_key or settings.GEMINI_API_KEY
            if not api_key:
                raise EmbeddingServiceError("GEMINI_API_KEY non configurée")
            genai.configure(api_key=api_key)
            logger.info(f"✅ Gemini API configurée (model: {self.EMBEDDING_MODEL})")
        except Exception as e:
            logger.error(f"❌ Échec configuration Gemini API: {e}")
            raise EmbeddingServiceError(f"Impossible de configurer Gemini API: {e}") from e

        # Cache PostgreSQL (connexion synchrone pour les tasks Celery/background)
        self.use_cache = use_cache
        self._sync_engine = None
        self._sync_session_factory = None

        if use_cache:
            try:
                sync_url = settings.DATABASE_URL.replace("+asyncpg", "")
                self._sync_engine = create_engine(sync_url, pool_pre_ping=True, pool_size=2)
                self._sync_session_factory = sessionmaker(bind=self._sync_engine)
                logger.info("✅ EmbeddingService cache PostgreSQL activé")
            except Exception as e:
                logger.warning(f"⚠️ Cache PostgreSQL non disponible, désactivé: {e}")
                self.use_cache = False

        logger.info(
            f"✅ EmbeddingService initialisé "
            f"(dim={self.EMBEDDING_DIM}, cache={self.use_cache}, provider=Gemini)"
        )

    # ==================== PUBLIC API ====================

    def generate_embedding(self, text: str, normalize: bool = True) -> np.ndarray:
        """
        Génère l'embedding pour un texte unique via Gemini API.

        Vérifie d'abord le cache PostgreSQL, génère l'embedding si nécessaire,
        puis le stocke dans le cache pour utilisation future.

        Args:
            text: Texte à encoder
            normalize: Applique normalisation L2 (True recommandé pour cosine)

        Returns:
            Embedding numpy array de dimension (3072,)

        Raises:
            ValueError: Si texte vide ou trop long (>10000 chars)
            EmbeddingServiceError: Si génération échoue
        """
        assert isinstance(text, str), "text must be a string"
        assert isinstance(normalize, bool), "normalize must be a boolean"

        start_time = time.time()
        self._validate_text(text)
        cache_key = self._get_cache_key(text)

        # Tentative récupération cache PostgreSQL
        if self.use_cache:
            cached_embedding = self._get_from_pg_cache(cache_key)
            if cached_embedding is not None:
                elapsed = (time.time() - start_time) * 1000
                logger.debug(f"🔍 Cache HIT ({elapsed:.1f}ms): {cache_key[:20]}...")
                return cached_embedding

        # Génération embedding via Gemini API avec retry
        for attempt in range(self.MAX_RETRIES):
            try:
                result = genai.embed_content(
                    model=self.EMBEDDING_MODEL,
                    content=text,
                    task_type="retrieval_document",
                )
                embedding = np.array(result["embedding"], dtype=np.float32)

                if embedding.shape[0] != self.EMBEDDING_DIM:
                    raise EmbeddingServiceError(
                        f"Dimension incorrecte: {embedding.shape[0]} != {self.EMBEDDING_DIM}"
                    )

                if normalize:
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm

                # Stockage dans cache PostgreSQL
                if self.use_cache:
                    self._store_in_pg_cache(cache_key, embedding)

                elapsed = (time.time() - start_time) * 1000
                logger.debug(f"⏱️ Embedding généré en {elapsed:.1f}ms (dim={embedding.shape[0]})")
                return embedding

            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(f"⚠️ Tentative {attempt + 1} échouée, retry: {e}")
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"❌ Erreur génération embedding après {self.MAX_RETRIES} tentatives: {e}")
                    raise EmbeddingServiceError(f"Échec génération embedding: {e}") from e

        # Fallback in case loop exits without returning or raising
        raise EmbeddingServiceError("Impossible de générer l'embedding (tentatives invalides).")

    def generate_batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = 20,
        normalize: bool = True,
        max_retries: int = 3,
        retry_delay: float = 5.0,
    ) -> List[np.ndarray]:
        """
        Génère les embeddings pour un batch de textes (cache-aware).

        Args:
            texts: Liste de textes à encoder
            batch_size: Taille max des chunks pour l'API
            normalize: Applique normalisation L2
            max_retries: Nombre max de tentatives par chunk
            retry_delay: Délai entre tentatives

        Returns:
            Liste d'embeddings numpy array (ordre préservé)
        """
        assert texts, "La liste de textes ne peut pas être vide"
        start_time = time.time()

        for idx, text in enumerate(texts):
            try:
                self._validate_text(text)
            except ValueError as e:
                raise ValueError(f"Texte invalide à l'index {idx}: {e}") from e

        logger.info(f"🔄 Génération batch: {len(texts)} textes...")

        # Phase 1: Check cache PostgreSQL
        embeddings_dict, texts_to_generate, indices_to_generate = self._check_batch_cache(texts)
        cache_hits = len(texts) - len(texts_to_generate)
        logger.info(f"📊 Cache: {cache_hits} hits, {len(texts_to_generate)} misses")

        # Phase 2: Generate missing embeddings in chunks
        if texts_to_generate:
            total = len(texts_to_generate)
            chunk_count = (total + batch_size - 1) // batch_size
            logger.info(f"📦 Processing {total} texts in {chunk_count} chunks of max {batch_size}")

            for chunk_idx in range(0, total, batch_size):
                chunk_texts = texts_to_generate[chunk_idx:chunk_idx + batch_size]
                chunk_indices = indices_to_generate[chunk_idx:chunk_idx + batch_size]
                current_chunk = (chunk_idx // batch_size) + 1

                self._generate_chunk_with_retry(
                    chunk_texts, chunk_indices, embeddings_dict,
                    current_chunk, chunk_count, normalize, max_retries, retry_delay,
                )

                if chunk_idx + batch_size < total:
                    time.sleep(0.5)

        # Phase 3: Reconstitute original order
        embeddings_list = [embeddings_dict[i] for i in range(len(texts))]

        elapsed = (time.time() - start_time) * 1000
        logger.info(
            f"✅ Batch généré en {elapsed:.0f}ms "
            f"({len(texts)} embeddings, {cache_hits} from cache)"
        )
        return embeddings_list

    def similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calcule la similarité cosine entre deux embeddings."""
        if emb1.shape != (self.EMBEDDING_DIM,):
            raise ValueError(f"emb1 dimension incorrecte: {emb1.shape} != ({self.EMBEDDING_DIM},)")
        if emb2.shape != (self.EMBEDDING_DIM,):
            raise ValueError(f"emb2 dimension incorrecte: {emb2.shape} != ({self.EMBEDDING_DIM},)")
        return max(0.0, min(1.0, float(np.dot(emb1, emb2))))

    def health_check(self) -> Dict[str, Any]:
        """Vérifie l'état de santé du service."""
        status_info: Dict[str, Any] = {
            "service": "EmbeddingService",
            "provider": "Gemini API",
            "model": self.EMBEDDING_MODEL,
            "dimensions": self.EMBEDDING_DIM,
            "cache_enabled": self.use_cache,
            "cache_backend": "PostgreSQL" if self.use_cache else "disabled",
            "status": "healthy",
        }

        try:
            test_emb = self.generate_embedding("test", normalize=False)
            if test_emb.shape[0] != self.EMBEDDING_DIM:
                status_info["status"] = "degraded"
                status_info["api_error"] = f"Dimension incorrecte: {test_emb.shape[0]}"
        except Exception as e:
            status_info["status"] = "unhealthy"
            status_info["api_error"] = str(e)

        return status_info

    # ==================== POSTGRESQL CACHE (private) ====================

    def _get_pg_session(self):
        """Retourne une session synchrone PostgreSQL."""
        if self._sync_session_factory is None:
            return None
        return self._sync_session_factory()

    def _get_from_pg_cache(self, text_hash: str) -> Optional[np.ndarray]:
        """
        Récupère un embedding du cache PostgreSQL (table embedding_cache).

        Args:
            text_hash: Clé de cache (hash SHA-256 du texte)

        Returns:
            Embedding numpy array ou None si absent/expiré
        """
        if not self.use_cache or self._sync_session_factory is None:
            return None

        try:
            from sqlalchemy import text as sql_text
            session = self._get_pg_session()
            if session is None:
                return None
            with session:
                result = session.execute(
                    sql_text(
                        "SELECT embedding_json FROM embedding_cache "
                        "WHERE text_hash = :key AND expires_at > now()"
                    ),
                    {"key": text_hash},
                )
                row = result.fetchone()
                if row:
                    embedding_list = json.loads(row[0])
                    return np.array(embedding_list, dtype=np.float32)
        except Exception as e:
            logger.debug(f"⚠️ Cache PG read failed ({text_hash[:16]}...): {e}")

        return None

    def _store_in_pg_cache(self, text_hash: str, embedding: np.ndarray) -> None:
        """
        Stocke un embedding dans la table embedding_cache PostgreSQL.

        Args:
            text_hash: Clé de cache
            embedding: Embedding numpy array à stocker
        """
        if not self.use_cache or self._sync_session_factory is None:
            return

        try:
            from sqlalchemy import text as sql_text
            expires_at = datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(seconds=self.CACHE_TTL_SECONDS)
            embedding_json = json.dumps(embedding.tolist())

            session = self._get_pg_session()
            if session is None:
                return
            with session:
                session.execute(
                    sql_text(
                        "INSERT INTO embedding_cache (text_hash, embedding_json, expires_at) "
                        "VALUES (:key, :data, :exp) "
                        "ON CONFLICT (text_hash) DO UPDATE "
                        "SET embedding_json = :data, expires_at = :exp"
                    ),
                    {"key": text_hash, "data": embedding_json, "exp": expires_at},
                )
                session.commit()
        except Exception as e:
            logger.debug(f"⚠️ Cache PG write failed ({text_hash[:16]}...): {e}")

    # ==================== BATCH CACHE HELPERS ====================

    def _check_batch_cache(self, texts: List[str]):
        """Check cache for all texts, return dict of hits and lists of misses."""
        embeddings_dict = {}
        texts_to_generate = []
        indices_to_generate = []

        for idx, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if self.use_cache:
                cached_emb = self._get_from_pg_cache(cache_key)
                if cached_emb is not None:
                    embeddings_dict[idx] = cached_emb
                    continue
            texts_to_generate.append(text)
            indices_to_generate.append(idx)

        return embeddings_dict, texts_to_generate, indices_to_generate

    def _generate_chunk_with_retry(
        self, chunk_texts, chunk_indices, embeddings_dict,
        current_chunk, chunk_count, normalize, max_retries, retry_delay,
    ):
        """Generate embeddings for a single chunk with retry logic."""
        logger.info(f"  📝 Chunk {current_chunk}/{chunk_count}: {len(chunk_texts)} texts...")

        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model=self.EMBEDDING_MODEL,
                    content=chunk_texts,
                    task_type="retrieval_document",
                )
                new_embeddings = [np.array(emb, dtype=np.float32) for emb in result["embedding"]]

                if normalize:
                    new_embeddings = [
                        emb / np.linalg.norm(emb) if np.linalg.norm(emb) > 0 else emb
                        for emb in new_embeddings
                    ]

                for idx, text, embedding in zip(chunk_indices, chunk_texts, new_embeddings):
                    embeddings_dict[idx] = embedding
                    if self.use_cache:
                        self._store_in_pg_cache(self._get_cache_key(text), embedding)

                logger.info(f"  ✅ Chunk {current_chunk}/{chunk_count} completed")
                return

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    logger.warning(
                        f"  ⚠️ Chunk {current_chunk} failed (attempt {attempt + 1}): {e}. "
                        f"Retry in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    raise EmbeddingServiceError(
                        f"Échec génération batch après {max_retries} tentatives: {e}"
                    ) from e

    # ==================== PRIVATE HELPERS ====================

    def _validate_text(self, text: str) -> None:
        """Valide le texte d'entrée."""
        if not text or not text.strip():
            raise ValueError("Le texte ne peut pas être vide")
        if len(text) > self.MAX_TEXT_LENGTH:
            raise ValueError(
                f"Texte trop long: {len(text)} chars > {self.MAX_TEXT_LENGTH} max"
            )

    def _preprocess_text(self, text: str) -> str:
        """Prétraite le texte pour normalisation cache."""
        return text.strip()

    def _get_cache_key(self, text: str) -> str:
        """Génère la clé de cache SHA-256 pour un texte."""
        normalized = self._preprocess_text(text).lower()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:64]
