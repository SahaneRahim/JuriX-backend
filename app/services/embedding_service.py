"""
Service de génération d'embeddings vectoriels pour recherche sémantique.

Ce service utilise Gemini API pour générer des embeddings 3072-dim
multilingues (FR/EN) avec cache Redis pour optimiser les performances.

Architecture:
- Model: models/gemini-embedding-001 (Gemini API)
- Dimensions: 3072 (compatible pgvector)
- Cache: Redis avec TTL 7 jours
- Performance: <300ms single, <2s batch(10)

Usage:
    service = EmbeddingService()

    # Single embedding
    emb = service.generate_embedding("Article 1er du Code civil")
    # np.ndarray shape (3072,)

    # Batch embeddings
    embs = service.generate_batch_embeddings([
        "Article 1er...",
        "Article 2..."
    ])
    # List[np.ndarray]

    # Similarity
    score = service.similarity(emb1, emb2)  # 0.0 to 1.0

Performance cible:
- Single: <300ms
- Batch (10): <2s
- Batch (32): <6s
- Cache hit: <10ms

Author: JuriX Team
Version: 2.0.0 (Gemini API)
"""

import hashlib
import json
import logging
import time
from typing import Any, Dict, List, Optional

import numpy as np
import redis
import google.generativeai as genai
from app.core.config import settings

logger = logging.getLogger(__name__)


class EmbeddingServiceError(Exception):
    """Exception levée lors d'erreurs de génération d'embeddings."""
    pass


class EmbeddingService:
    """
    Service de génération d'embeddings vectoriels pour recherche sémantique.

    Utilise Gemini API avec modèle gemini-embedding-001 optimisé pour:
    - Recherche sémantique dans textes juridiques
    - Support français et anglais
    - Embeddings de dimension 3072 (compatible pgvector)
    - Cache Redis pour performances optimales

    Caractéristiques:
    - Model: models/gemini-embedding-001 (Gemini)
    - Normalisation L2 automatique (cosine similarity)
    - Batch processing cache-aware
    - Retry logic pour robustesse
    - Graceful degradation si Redis indisponible

    Performance:
    - Single embedding: <300ms
    - Batch (10 embeddings): <2s
    - Cache hit: <10ms

    Attributes:
        EMBEDDING_MODEL: Nom du modèle Gemini
        EMBEDDING_DIM: Dimension des embeddings (3072)
        CACHE_TTL_SECONDS: Durée de vie cache (7 jours)
        redis_client: Client Redis (optionnel)
        use_cache: Flag activation cache
    """

    # Configuration constantes
    EMBEDDING_MODEL = settings.GEMINI_EMBEDDING_MODEL
    EMBEDDING_DIM = 3072
    CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 jours
    MAX_TEXT_LENGTH = 10000  # Limite sécurité
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds

    def __init__(
        self,
        redis_url: Optional[str] = None,
        use_cache: bool = True,
        api_key: Optional[str] = None
    ):
        """
        Initialise le service d'embeddings.

        Configure Gemini API et initialise optionnellement la connexion 
        Redis pour le cache.

        Args:
            redis_url: URL connexion Redis (ex: "redis://localhost:6379/0").
                      Si None, utilise settings.REDIS_URL par défaut.
            use_cache: Active/désactive le cache Redis
            api_key: Clé API Gemini (si None, utilise settings.GEMINI_API_KEY)

        Raises:
            EmbeddingServiceError: Si Gemini API ne peut pas être initialisée

        Example:
            >>> # Avec cache (défaut)
            >>> service = EmbeddingService()
            >>> 
            >>> # Sans cache
            >>> service = EmbeddingService(use_cache=False)
            >>> 
            >>> # Redis custom
            >>> service = EmbeddingService(
            ...     redis_url="redis://:password@localhost:6379/1"
            ... )
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
            raise EmbeddingServiceError(
                f"Impossible de configurer Gemini API: {e}"
            ) from e

        # Initialisation cache Redis
        self.use_cache = use_cache
        self.redis_client = None

        if use_cache:
            try:
                redis_url = redis_url or settings.REDIS_URL
                self.redis_client = redis.from_url(
                    redis_url,
                    decode_responses=False,  # Binary pour np.array
                    socket_connect_timeout=2,
                    socket_timeout=2
                )
                # Test connexion
                self.redis_client.ping()
                logger.info(f"✅ Cache Redis connecté: {redis_url}")
            except Exception as e:
                logger.warning(
                    f"⚠️  Redis non disponible, cache désactivé: {e}"
                )
                self.redis_client = None
                self.use_cache = False
        else:
            logger.info("ℹ️  Cache Redis désactivé")

        logger.info(
            f"✅ EmbeddingService initialisé "
            f"(dim={self.EMBEDDING_DIM}, cache={self.use_cache}, provider=Gemini)"
        )

    def generate_embedding(
        self,
        text: str,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Génère l'embedding pour un texte unique via Gemini API.

        Vérifie d'abord le cache Redis, génère l'embedding si nécessaire,
        puis le stocke dans le cache pour utilisation future.

        Args:
            text: Texte à encoder (article de loi, question, etc.)
            normalize: Applique normalisation L2 (True recommandé pour cosine)

        Returns:
            Embedding numpy array de dimension (3072,), normalisé si demandé

        Raises:
            ValueError: Si texte vide ou trop long (>10000 chars)
            EmbeddingServiceError: Si génération échoue

        Performance:
            - Cache hit: <10ms
            - Cache miss: <300ms (Gemini API)

        Example:
            >>> service = EmbeddingService()
            >>> emb = service.generate_embedding(
            ...     "Article 1er du Code civil camerounais"
            ... )
            >>> emb.shape
            (3072,)
            >>> np.linalg.norm(emb)  # Normalized
            1.0
        """
        assert isinstance(text, str), "text must be a string"
        assert isinstance(normalize, bool), "normalize must be a boolean"

        start_time = time.time()

        # Validation
        self._validate_text(text)

        # Prétraitement pour cache key
        cache_key = self._get_cache_key(text)

        # Tentative récupération cache
        if self.use_cache and self.redis_client:
            cached_embedding = self._get_from_cache(cache_key)
            if cached_embedding is not None:
                elapsed = (time.time() - start_time) * 1000
                logger.debug(f"🔍 Cache HIT ({elapsed:.1f}ms): {cache_key[:20]}...")
                return cached_embedding

        # Génération embedding via Gemini API
        for attempt in range(self.MAX_RETRIES):
            try:
                result = genai.embed_content(
                    model=self.EMBEDDING_MODEL,
                    content=text,
                    task_type="retrieval_document"
                )
                
                embedding = np.array(result['embedding'], dtype=np.float32)

                # Validation dimension
                if embedding.shape[0] != self.EMBEDDING_DIM:
                    raise EmbeddingServiceError(
                        f"Dimension incorrecte: {embedding.shape[0]} != {self.EMBEDDING_DIM}"
                    )

                # Normalisation L2 si demandée
                if normalize:
                    norm = np.linalg.norm(embedding)
                    if norm > 0:
                        embedding = embedding / norm

                # Stockage cache
                if self.use_cache and self.redis_client:
                    self._store_in_cache(cache_key, embedding)

                elapsed = (time.time() - start_time) * 1000
                logger.debug(
                    f"⏱️  Embedding généré en {elapsed:.1f}ms "
                    f"(dim={embedding.shape[0]}, norm={normalize})"
                )

                return embedding

            except Exception as e:
                if attempt < self.MAX_RETRIES - 1:
                    logger.warning(f"⚠️  Tentative {attempt + 1} échouée, retry: {e}")
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"❌ Erreur génération embedding après {self.MAX_RETRIES} tentatives: {e}")
                    raise EmbeddingServiceError(
                        f"Échec génération embedding: {e}"
                    ) from e

    def generate_batch_embeddings(
        self,
        texts: List[str],
        batch_size: int = 20,
        normalize: bool = True,
        max_retries: int = 3,
        retry_delay: float = 5.0
    ) -> List[np.ndarray]:
        """
        Génère les embeddings pour un batch de textes (cache-aware).

        Pipeline: validate → check cache → generate missing → reconstitute.

        Args:
            texts: Liste de textes à encoder
            batch_size: Taille max des chunks pour l'API
            normalize: Applique normalisation L2
            max_retries: Nombre max de tentatives par chunk
            retry_delay: Délai entre tentatives

        Returns:
            Liste d'embeddings numpy array (ordre préservé)

        Raises:
            ValueError: Si liste vide ou textes invalides
            EmbeddingServiceError: Si génération échoue
        """
        assert texts, "La liste de textes ne peut pas être vide"
        start_time = time.time()

        # Validate all texts
        for idx, text in enumerate(texts):
            try:
                self._validate_text(text)
            except ValueError as e:
                raise ValueError(f"Texte invalide à l'index {idx}: {e}") from e

        logger.info(f"🔄 Génération batch: {len(texts)} textes...")

        # Phase 1: Check cache
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
                    current_chunk, chunk_count, normalize, max_retries, retry_delay
                )

                # Rate limit delay between chunks
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

    def _check_batch_cache(self, texts: List[str]):
        """Check cache for all texts, return dict of hits and lists of misses."""
        embeddings_dict = {}
        texts_to_generate = []
        indices_to_generate = []

        for idx, text in enumerate(texts):
            cache_key = self._get_cache_key(text)
            if self.use_cache and self.redis_client:
                cached_emb = self._get_from_cache(cache_key)
                if cached_emb is not None:
                    embeddings_dict[idx] = cached_emb
                    continue
            texts_to_generate.append(text)
            indices_to_generate.append(idx)

        return embeddings_dict, texts_to_generate, indices_to_generate

    def _generate_chunk_with_retry(
        self, chunk_texts, chunk_indices, embeddings_dict,
        current_chunk, chunk_count, normalize, max_retries, retry_delay
    ):
        """Generate embeddings for a single chunk with retry logic."""
        logger.info(f"  📝 Chunk {current_chunk}/{chunk_count}: {len(chunk_texts)} texts...")

        for attempt in range(max_retries):
            try:
                result = genai.embed_content(
                    model=self.EMBEDDING_MODEL,
                    content=chunk_texts,
                    task_type="retrieval_document"
                )
                new_embeddings = [
                    np.array(emb, dtype=np.float32) for emb in result['embedding']
                ]

                if normalize:
                    new_embeddings = [
                        emb / np.linalg.norm(emb) if np.linalg.norm(emb) > 0 else emb
                        for emb in new_embeddings
                    ]

                # Store in dict and cache
                for idx, text, embedding in zip(chunk_indices, chunk_texts, new_embeddings):
                    embeddings_dict[idx] = embedding
                    if self.use_cache and self.redis_client:
                        cache_key = self._get_cache_key(text)
                        self._store_in_cache(cache_key, embedding)

                logger.info(f"  ✅ Chunk {current_chunk}/{chunk_count} completed")
                return  # Success

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (attempt + 1)
                    logger.warning(
                        f"  ⚠️ Chunk {current_chunk} failed (attempt {attempt + 1}/{max_retries}): {e}. "
                        f"Retrying in {wait_time}s..."
                    )
                    time.sleep(wait_time)
                else:
                    logger.error(f"❌ Chunk {current_chunk} failed after {max_retries} attempts: {e}")
                    raise EmbeddingServiceError(
                        f"Échec génération batch embeddings après {max_retries} tentatives: {e}"
                    ) from e

    def similarity(
        self,
        emb1: np.ndarray,
        emb2: np.ndarray
    ) -> float:
        """
        Calcule la similarité cosine entre deux embeddings.

        Args:
            emb1: Premier embedding (3072,)
            emb2: Deuxième embedding (3072,)

        Returns:
            Score de similarité entre 0.0 et 1.0
            (1.0 = identiques, 0.0 = orthogonaux)

        Raises:
            ValueError: Si dimensions incorrectes ou non-matching

        Example:
            >>> service = EmbeddingService()
            >>> emb1 = service.generate_embedding("Article 1er")
            >>> emb2 = service.generate_embedding("Article premier")
            >>> score = service.similarity(emb1, emb2)
            >>> score > 0.8  # Très similaires
            True
        """
        # Validation dimensions
        if emb1.shape != (self.EMBEDDING_DIM,):
            raise ValueError(
                f"emb1 dimension incorrecte: {emb1.shape} != ({self.EMBEDDING_DIM},)"
            )
        if emb2.shape != (self.EMBEDDING_DIM,):
            raise ValueError(
                f"emb2 dimension incorrecte: {emb2.shape} != ({self.EMBEDDING_DIM},)"
            )

        # Cosine similarity (dot product si normalisés)
        similarity_score = float(np.dot(emb1, emb2))

        # Clamp entre 0 et 1 (erreurs arrondis)
        return max(0.0, min(1.0, similarity_score))

    def health_check(self) -> Dict[str, Any]:
        """
        Vérifie l'état de santé du service.

        Teste:
        - Disponibilité de Gemini API (génération test)
        - Connectivité Redis si cache activé
        - Configuration actuelle

        Returns:
            Dictionnaire avec status et métadonnées

        Example:
            >>> service = EmbeddingService()
            >>> health = service.health_check()
            >>> health['status']
            'healthy'
            >>> health['dimensions']
            3072
        """
        status_info = {
            "service": "EmbeddingService",
            "provider": "Gemini API",
            "model": self.EMBEDDING_MODEL,
            "dimensions": self.EMBEDDING_DIM,
            "cache_enabled": self.use_cache,
            "status": "healthy"
        }

        # Test Gemini API
        try:
            test_emb = self.generate_embedding("test", normalize=False)
            if test_emb.shape[0] != self.EMBEDDING_DIM:
                status_info["status"] = "degraded"
                status_info["api_error"] = f"Dimension incorrecte: {test_emb.shape[0]}"
        except Exception as e:
            status_info["status"] = "unhealthy"
            status_info["api_error"] = str(e)

        # Test cache Redis
        if self.use_cache and self.redis_client:
            try:
                self.redis_client.ping()
                status_info["cache_status"] = "connected"
            except Exception as e:
                status_info["cache_status"] = "disconnected"
                status_info["cache_error"] = str(e)
        else:
            status_info["cache_status"] = "disabled"

        return status_info

    # ==================== MÉTHODES PRIVÉES ====================

    def _validate_text(self, text: str) -> None:
        """
        Valide le texte d'entrée.

        Args:
            text: Texte à valider

        Raises:
            ValueError: Si texte vide ou trop long
        """
        if not text or not text.strip():
            raise ValueError("Le texte ne peut pas être vide")

        if len(text) > self.MAX_TEXT_LENGTH:
            raise ValueError(
                f"Texte trop long: {len(text)} chars > {self.MAX_TEXT_LENGTH} max"
            )

    def _preprocess_text(self, text: str) -> str:
        """
        Prétraite le texte pour normalisation cache.

        Args:
            text: Texte brut

        Returns:
            Texte normalisé (strip whitespace)
        """
        return text.strip()

    def _get_cache_key(self, text: str) -> str:
        """
        Génère la clé de cache pour un texte.

        Utilise MD5 hash du texte normalisé pour:
        - Clés compactes (32 chars vs texte complet)
        - Déterminisme (même texte = même clé)
        - Risque collision négligeable (<1M textes)

        Args:
            text: Texte à hasher

        Returns:
            Clé cache format "embedding:gemini:{md5_hash}"
        """
        normalized = self._preprocess_text(text).lower()
        text_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()
        return f"embedding:gemini:{text_hash}"

    def _get_from_cache(self, cache_key: str) -> Optional[np.ndarray]:
        """
        Récupère un embedding du cache Redis.

        Args:
            cache_key: Clé de cache

        Returns:
            Embedding si trouvé, None sinon
        """
        try:
            cached_data = self.redis_client.get(cache_key)
            if cached_data:
                # Désérialisation JSON -> numpy array
                embedding_list = json.loads(cached_data)
                return np.array(embedding_list, dtype=np.float32)
        except Exception as e:
            logger.warning(f"⚠️  Erreur lecture cache {cache_key[:20]}: {e}")

        return None

    def _store_in_cache(
        self,
        cache_key: str,
        embedding: np.ndarray
    ) -> None:
        """
        Stocke un embedding dans le cache Redis.

        Args:
            cache_key: Clé de cache
            embedding: Embedding à stocker
        """
        try:
            # Sérialisation numpy -> JSON
            embedding_list = embedding.tolist()
            cached_data = json.dumps(embedding_list)

            # Stockage avec TTL
            self.redis_client.setex(
                cache_key,
                self.CACHE_TTL_SECONDS,
                cached_data
            )
        except Exception as e:
            logger.warning(f"⚠️  Erreur écriture cache {cache_key[:20]}: {e}")
