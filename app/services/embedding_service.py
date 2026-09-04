"""
Service de génération d'embeddings vectoriels pour recherche sémantique.

Ce service utilise Gemini API pour générer des embeddings 3072-dim
multilingues (FR/EN) avec cache PostgreSQL pour optimiser les performances.

Architecture:
- Model: models/gemini-embedding-001 (Gemini API)
- Dimensions: 3072 (settings.EMBEDDING_DIM ; index HNSW via une expression halfvec)
- Cache: Table embedding_cache PostgreSQL avec TTL 7 jours (cache en base)
- Performance: <300ms single, <2s batch(10)

Author: JuriX Team
Version: 3.0.0 (cache PostgreSQL)
"""

import asyncio
import hashlib
import json
import logging
import time
from functools import lru_cache
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from google import genai
from google.genai import types
from app.core.config import settings
from app.core.database import SyncSessionLocal

logger = logging.getLogger(__name__)


def _is_daily_quota_error(error: Exception) -> bool:
    """
    Reconnait un depassement de quota JOURNALIER.

    Google renvoie 429 aussi bien pour un quota par minute que par jour ; seul
    le quotaId les distingue. Le premier se rejoue avec profit apres quelques
    secondes, le second jamais avant sa reinitialisation (minuit, heure du
    Pacifique).
    """
    message = str(error)
    if "429" not in message and "RESOURCE_EXHAUSTED" not in message:
        return False
    return "PerDay" in message or "per day" in message.lower()


class QuotaExhaustedError(Exception):
    """
    Quota JOURNALIER epuise.

    Distinguee des autres erreurs parce qu'elle ne se rejoue pas : un quota
    "PerDay" ne se recharge pas en trente secondes, et reessayer ne fait que
    bruler du temps avant d'echouer quand meme. Le delai suggere par Google
    (retryDelay ~30 s) ne vaut que pour les quotas par MINUTE.
    """


class EmbeddingServiceError(Exception):
    """Exception levée lors d'erreurs de génération d'embeddings."""
    pass


class EmbeddingService:
    """
    Service de génération d'embeddings vectoriels pour recherche sémantique.

    Utilise Gemini API avec cache PostgreSQL (table embedding_cache).
    Supporte FR/EN, normalisation L2 automatique.

    Dimension : settings.EMBEDDING_DIM (3072), demandée à l'API via
    output_dimensionality. C'est la sortie native du modèle ; le plafond
    d'indexation de 2000 dimensions de pgvector est contourné par un index sur
    une expression halfvec, pas par une troncature du vecteur.

    Attributes:
        EMBEDDING_MODEL: Nom du modèle Gemini
        EMBEDDING_DIM: Dimension des embeddings (settings.EMBEDDING_DIM)
        CACHE_TTL_SECONDS: Durée de vie cache (7 jours)
        use_cache: Flag activation cache
    """

    EMBEDDING_MODEL = settings.GEMINI_EMBEDDING_MODEL
    EMBEDDING_DIM = settings.EMBEDDING_DIM
    # Dimension native du modele. En dessous, l'API tronque le vecteur
    # (Matryoshka) et NE le renormalise PAS : c'est a nous de le faire.
    NATIVE_EMBEDDING_DIM = 3072
    # Version de la cle de cache. A incrementer si la facon de construire les
    # vecteurs change sans que le modele ni la dimension ne changent.
    # v3 : la normalisation est devenue inconditionnelle, donc le vecteur
    # produit pour un meme (modele, dimension, tache, texte) a change. C'est
    # exactement le cas que cette constante existe pour couvrir.
    CACHE_KEY_VERSION = "v3"
    CACHE_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 jours
    MAX_TEXT_LENGTH = 10000
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds

    # Types de tâche Gemini. La recherche est asymétrique : un article indexé
    # et une question d'utilisateur ne doivent pas être encodés de la même
    # façon, sinon la pertinence se dégrade.
    TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
    TASK_QUERY = "RETRIEVAL_QUERY"

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
            # SDK google-genai (le meme que gemini_service.py). L'ancien
            # google-generativeai et celui-ci ne peuvent pas cohabiter : le
            # projet declarait l'ancien alors que gemini_service.py importe
            # le nouveau, ce qui empechait l'application de demarrer.
            # http_options impose un delai maximal. Sans lui le client attend
            # indefiniment : une ingestion est restee figee 40 minutes sur un
            # appel d'embeddings, le processus endormi sur une lecture de
            # prise reseau. Aucune exception n'etant levee, la boucle de
            # reprise juste en dessous ne se declenchait jamais.
            self.client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(
                    timeout=settings.GEMINI_TIMEOUT_S * 1000  # en millisecondes
                ),
            )
            logger.info(f"✅ Gemini API configurée (model: {self.EMBEDDING_MODEL})")
        except Exception as e:
            logger.error(f"❌ Échec configuration Gemini API: {e}")
            raise EmbeddingServiceError(f"Impossible de configurer Gemini API: {e}") from e

        # Cache PostgreSQL (connexion synchrone pour les taches de fond).
        # Le service ne POSSEDE plus de moteur : il emprunte celui de
        # app.core.database. Un create_engine par instance restait ouvert pour
        # toute la duree du processus, et chaque loi traitee en creait un.
        self.use_cache = use_cache
        self._sync_session_factory = SyncSessionLocal if use_cache else None

        logger.info(
            f"✅ EmbeddingService initialisé "
            f"(dim={self.EMBEDDING_DIM}, cache={self.use_cache}, provider=Gemini)"
        )

    # ==================== PUBLIC API ====================

    def generate_embedding(
        self,
        text: str,
        normalize: bool = True,
        task_type: str = TASK_DOCUMENT,
    ) -> np.ndarray:
        """
        Génère l'embedding pour un texte unique via Gemini API.

        Vérifie d'abord le cache PostgreSQL, génère l'embedding si nécessaire,
        puis le stocke dans le cache pour utilisation future.

        Args:
            text: Texte à encoder
            normalize: Applique normalisation L2 (True recommandé pour cosine)
            task_type: TASK_DOCUMENT pour indexer un article,
                       TASK_QUERY pour encoder une question d'utilisateur.
                       La recherche est asymétrique : utiliser le même type des
                       deux côtés dégrade la pertinence.

        Returns:
            Embedding numpy array de dimension (EMBEDDING_DIM,)

        Raises:
            ValueError: Si texte vide ou trop long (>10000 chars)
            EmbeddingServiceError: Si génération échoue
        """
        assert isinstance(text, str), "text must be a string"
        assert isinstance(normalize, bool), "normalize must be a boolean"

        start_time = time.time()
        self._validate_text(text)
        cache_key = self._cache_key(text, task_type)

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
                result = self.client.models.embed_content(
                    model=self.EMBEDDING_MODEL,
                    contents=text,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.EMBEDDING_DIM,
                    ),
                )
                embedding = np.array(result.embeddings[0].values, dtype=np.float32)

                if embedding.shape[0] != self.EMBEDDING_DIM:
                    raise EmbeddingServiceError(
                        f"Dimension incorrecte: {embedding.shape[0]} != {self.EMBEDDING_DIM}"
                    )

                embedding = self._normalize(embedding, normalize)

                # Stockage dans cache PostgreSQL
                if self.use_cache:
                    self._store_in_pg_cache(cache_key, embedding)

                elapsed = (time.time() - start_time) * 1000
                logger.debug(f"⏱️ Embedding généré en {elapsed:.1f}ms (dim={embedding.shape[0]})")
                return embedding

            except Exception as e:
                if _is_daily_quota_error(e):
                    raise QuotaExhaustedError(
                        f"Quota journalier d'embeddings epuise: {e}"
                    ) from e

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
        # ValueError et non assert : les assertions sautent sous python -O,
        # la validation disparaitrait alors silencieusement en production.
        if not texts:
            raise ValueError("La liste de textes ne peut pas être vide")
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
                    cached = np.array(json.loads(row[0]), dtype=np.float32)
                    # Garde de dimension : une entree ecrite sous une autre
                    # configuration (1536 avant la bascule) doit etre ignoree,
                    # pas servie. Sinon la colonne vector rejette
                    # l'insertion, ou pire, la comparaison est silencieusement
                    # fausse.
                    if cached.ndim != 1 or cached.shape[0] != self.EMBEDDING_DIM:
                        logger.debug(
                            f"⚠️ Cache PG: dimension {cached.shape} ignorée "
                            f"(attendu {self.EMBEDDING_DIM})"
                        )
                        return None
                    return cached
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
            # Même convention de clé que generate_embedding (le lot n'encode
            # que des documents).
            cache_key = self._cache_key(text, self.TASK_DOCUMENT)
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
                result = self.client.models.embed_content(
                    model=self.EMBEDDING_MODEL,
                    contents=chunk_texts,
                    config=types.EmbedContentConfig(
                        task_type=self.TASK_DOCUMENT,
                        output_dimensionality=self.EMBEDDING_DIM,
                    ),
                )

                # L'API peut renvoyer MOINS de vecteurs que de textes. Sans ce
                # controle, zip() perdait silencieusement la queue du lot, puis
                # embeddings_dict[idx] levait un KeyError bien plus loin, sans
                # rapport apparent avec la cause.
                if len(result.embeddings) != len(chunk_texts):
                    raise EmbeddingServiceError(
                        f"Réponse incomplète: {len(result.embeddings)} vecteurs "
                        f"pour {len(chunk_texts)} textes"
                    )

                new_embeddings = [
                    np.array(emb.values, dtype=np.float32) for emb in result.embeddings
                ]

                # Garde de dimension, absente de ce chemin alors que le chemin
                # unitaire l'avait : un lot mal dimensionne allait directement
                # en base et cassait l'insertion dans la colonne vector.
                bad = [e.shape[0] for e in new_embeddings if e.shape[0] != self.EMBEDDING_DIM]
                if bad:
                    raise EmbeddingServiceError(
                        f"Dimension incorrecte dans le lot: {bad[0]} != {self.EMBEDDING_DIM}"
                    )

                new_embeddings = [self._normalize(emb, normalize) for emb in new_embeddings]

                for idx, text, embedding in zip(chunk_indices, chunk_texts, new_embeddings):
                    embeddings_dict[idx] = embedding
                    if self.use_cache:
                        # Même clé qu'en lecture (_check_batch_cache), sinon le
                        # cache n'aurait jamais de hit.
                        self._store_in_pg_cache(
                            self._cache_key(text, self.TASK_DOCUMENT), embedding
                        )

                logger.info(f"  ✅ Chunk {current_chunk}/{chunk_count} completed")
                return

            except Exception as e:
                # Un quota JOURNALIER ne se recharge pas pendant la boucle :
                # insister ne fait que perdre trois fois le delai avant
                # d'echouer de toute facon. On leve tout de suite, avec un type
                # distinct pour que l'appelant puisse s'arreter proprement.
                if _is_daily_quota_error(e):
                    raise QuotaExhaustedError(
                        f"Quota journalier d'embeddings epuise: {e}"
                    ) from e

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

    def _cache_key(self, text: str, task_type: str) -> str:
        """
        Clé de cache SHA-256.

        Contient le modèle, la DIMENSION et le type de tâche, pas seulement le
        texte : sans la dimension, un vecteur écrit sous une autre dimension
        serait resservi tel quel, et sans le type de tâche un
        même texte encodé comme document ou comme question partagerait une
        entrée alors que les vecteurs diffèrent.

        Le texte n'est PLUS mis en minuscules. Gemini est sensible à la casse
        et le français juridique la porte ("Article PREMIER" n'est pas
        "article premier") : minusculer confondait des textes distincts sous
        une seule clé.
        """
        payload = "\x00".join([
            self.CACHE_KEY_VERSION,
            self.EMBEDDING_MODEL,
            str(self.EMBEDDING_DIM),
            task_type,
            self._preprocess_text(text),
        ])
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:64]

    def _normalize(self, embedding: np.ndarray, requested: bool = True) -> np.ndarray:
        """
        Normalisation L2, INCONDITIONNELLE.

        Le parametre `requested` est conserve pour les appelants existants mais
        volontairement ignore. Trois raisons :

        1. Tous les consommateurs supposent une norme de 1. similarity() est un
           np.dot nu, et le score `1 - distance` borne a [0, 1] ne vaut que sur
           la sphere unite. Cette garantie ne doit pas dependre d'un reglage.
        2. Cela ne coute rien. A la dimension native, l'API renvoie deja un
           vecteur unitaire : l'operation est alors un no-op de quelques
           microsecondes. Sous la dimension native, la troncature Matryoshka
           sort avec une norme quelconque et la normalisation est obligatoire.
        3. Aucun appelant n'a besoin du vecteur brut : le seul
           normalize=False du code est health_check(), qui ne regarde que la
           forme.
        """
        norm = float(np.linalg.norm(embedding))
        return embedding / norm if norm > 0 else embedding

    async def generate_embedding_async(
        self,
        text: str,
        task_type: str = TASK_QUERY,
    ) -> np.ndarray:
        """
        Version asynchrone de generate_embedding.

        Le client Gemini est synchrone, le backoff est un time.sleep et les
        lectures/écritures de cache passent par psycopg2 : appelée telle quelle
        depuis une coroutine, generate_embedding gèle la boucle d'événements
        pour tout l'aller-retour réseau. Tout appel depuis du code async doit
        passer par ici.

        task_type vaut TASK_QUERY par défaut : le seul appelant asynchrone est
        la recherche sémantique, qui encode une question.
        """
        return await asyncio.to_thread(self.generate_embedding, text, True, task_type)


@lru_cache()
def get_embedding_service() -> "EmbeddingService":
    """
    Instance unique d'EmbeddingService.

    La fabrique vit ICI et non dans app/core/dependencies.py : il en existait
    deux, celle des dependances FastAPI et le singleton prive de SearchService,
    donc jusqu'a deux instances par processus. Elles n'ont plus de moteur en
    propre depuis que le cache emprunte SyncSessionLocal, mais deux clients
    Gemini et deux journaux d'initialisation restaient inutiles. La placer ici
    evite le cycle d'import : dependencies.py importe deja SearchService.
    """
    logger.info("📦 Création du singleton EmbeddingService")
    return EmbeddingService(use_cache=True)
