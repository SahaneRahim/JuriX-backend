"""
Service de génération d'embeddings vectoriels pour recherche sémantique.

Le service ne parle plus directement a un modele : il delegue a un FOURNISSEUR
interchangeable, et garde pour lui tout le reste — validation, cache
PostgreSQL, gardes de dimension, normalisation, reprises, ordre des lots.

POURQUOI CETTE SEPARATION. Le modele d'embeddings est le composant que JuriX
changera le plus souvent : gratuit en local pour le lancement, meilleur et
payant quand les moyens le permettront. Le changer doit etre un reglage, pas
un chantier. Un fournisseur ne sait qu'une chose — transformer des textes en
vecteurs — et laisse remonter ses exceptions brutes.

Architecture:
- Fournisseur: Gemini API (models/gemini-embedding-001)
- Dimensions: settings.EMBEDDING_DIM
- Cache: Table embedding_cache PostgreSQL avec TTL 7 jours (cache en base)

Author: JuriX Team
Version: 4.0.0 (fournisseurs interchangeables)
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timedelta, timezone
from functools import lru_cache
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


# ==================== FOURNISSEURS ====================


class GeminiProvider:
    """
    Embeddings par l'API Gemini.

    CONTRAT D'UN FOURNISSEUR, que tout autre doit respecter :

    - `name`        : identifiant court ("gemini")
    - `label`       : libelle affiche par health_check
    - `model`       : nom du modele
    - `fingerprint` : identite EXACTE de ce qui produit les vecteurs. Elle entre
                      dans la cle de cache : deux fournisseurs, ou deux
                      variantes d'un meme modele, ne doivent jamais se servir
                      leurs vecteurs. A la meme dimension, rien d'autre ne les
                      distingue.
    - `native_dim`  : dimension maximale que le modele sait produire
    - `retryable`   : un echec merite-t-il une nouvelle tentative ? Vrai pour
                      une API distante, faux pour un modele local, ou rejouer
                      ne ferait que rejouer la meme erreur
    - `inter_batch_delay_s` : pause entre deux lots, pour menager un quota
    - `embed(texts, task_type, dim)` : rend EXACTEMENT un vecteur float32 par
                      texte, de longueur `dim`, normalise ou non. Le service
                      verifie le nombre et la dimension, et normalise : le
                      fournisseur n'a pas a s'en charger.
    """

    name = "gemini"
    label = "Gemini API"
    native_dim = 3072
    retryable = True
    inter_batch_delay_s = 0.5

    def __init__(self, api_key: Optional[str] = None):
        api_key = api_key or settings.GEMINI_API_KEY
        if not api_key:
            raise EmbeddingServiceError("GEMINI_API_KEY non configurée")
        self.model = settings.GEMINI_EMBEDDING_MODEL
        # SDK google-genai (le meme que gemini_service.py). L'ancien
        # google-generativeai et celui-ci ne peuvent pas cohabiter : le
        # projet declarait l'ancien alors que gemini_service.py importe
        # le nouveau, ce qui empechait l'application de demarrer.
        # http_options impose un delai maximal. Sans lui le client attend
        # indefiniment : une ingestion est restee figee 40 minutes sur un
        # appel d'embeddings, le processus endormi sur une lecture de
        # prise reseau. Aucune exception n'etant levee, la boucle de
        # reprise juste en dessous ne se declenchait jamais.
        #
        # `genai.Client` est appele PAR ATTRIBUT DU MODULE, et ce n'est pas un
        # detail de style : les tests remplacent
        # `app.services.embedding_service.genai.Client`. Un
        # `from google.genai import Client` contournerait la doublure, et les
        # tests appelleraient la vraie API — payante.
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=settings.GEMINI_TIMEOUT_S * 1000  # en millisecondes
            ),
        )

    @property
    def fingerprint(self) -> str:
        return f"{self.name}|{self.model}"

    def embed(self, texts: List[str], task_type: str, dim: int) -> List[np.ndarray]:
        # `self.client.models` est relu a CHAQUE appel, jamais memorise a la
        # construction : un test remplace `embed_content` apres coup.
        result = self.client.models.embed_content(
            model=self.model,
            # Un texte seul part en chaine, un lot en liste : c'est la forme
            # d'appel historique, conservee a l'identique.
            contents=texts[0] if len(texts) == 1 else texts,
            config=types.EmbedContentConfig(
                task_type=task_type,
                output_dimensionality=dim,
            ),
        )
        return [np.array(emb.values, dtype=np.float32) for emb in result.embeddings]


def _construire_fournisseur(api_key: Optional[str] = None) -> GeminiProvider:
    """Fournisseur par defaut, selon la configuration."""
    return GeminiProvider(api_key=api_key)


class EmbeddingService:
    """
    Service de génération d'embeddings vectoriels pour recherche sémantique.

    Delegue la production des vecteurs a un fournisseur (voir GeminiProvider),
    avec cache PostgreSQL (table embedding_cache). Supporte FR/EN,
    normalisation L2 inconditionnelle.

    Dimension : settings.EMBEDDING_DIM, demandee au fournisseur a CHAQUE appel
    et jamais figee en lui — scripts/eval/validate_slicing.py change la
    dimension d'une instance apres coup.

    Attributes:
        EMBEDDING_MODEL: Nom du modèle du fournisseur par defaut
        EMBEDDING_DIM: Dimension des embeddings (settings.EMBEDDING_DIM)
        CACHE_TTL_SECONDS: Durée de vie cache (7 jours)
        use_cache: Flag activation cache
    """

    EMBEDDING_MODEL = settings.GEMINI_EMBEDDING_MODEL
    EMBEDDING_DIM = settings.EMBEDDING_DIM
    # Version de la cle de cache. A incrementer si la facon de construire les
    # vecteurs change sans que l'empreinte du fournisseur ni la dimension ne
    # changent.
    # v3 : la normalisation est devenue inconditionnelle, donc le vecteur
    # produit pour un meme (modele, dimension, tache, texte) a change.
    # v4 : la cle porte l'EMPREINTE du fournisseur de l'instance, et non plus le
    # nom du modele Gemini fige a l'import. Sans cela, deux fournisseurs a la
    # meme dimension se resserviraient leurs vecteurs pendant toute la duree de
    # vie du cache, et la garde de dimension ne verrait rien.
    CACHE_KEY_VERSION = "v4"
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
        provider: Optional[Any] = None,
    ):
        """
        Initialise le service d'embeddings.

        Args:
            use_cache: Active/désactive le cache PostgreSQL
            api_key: Clé API Gemini (si None, utilise settings.GEMINI_API_KEY)
            provider: Fournisseur a utiliser. None construit celui de la
                configuration ; un test y injecte sa doublure.

        Raises:
            EmbeddingServiceError: Si le fournisseur ne peut pas être initialisé,
                ou si la dimension configuree depasse ce qu'il sait produire
        """
        if provider is None:
            try:
                provider = _construire_fournisseur(api_key)
            except Exception as e:
                logger.error(f"❌ Échec configuration Gemini API: {e}")
                raise EmbeddingServiceError(f"Impossible de configurer Gemini API: {e}") from e
        self._provider = provider

        # Refuse a la construction ce qui echouerait au premier appel. Sans ce
        # controle, une dimension trop grande ne se revele qu'apres trois
        # tentatives, et la recherche hybride avale l'erreur sans bruit.
        if self.EMBEDDING_DIM > provider.native_dim:
            raise EmbeddingServiceError(
                f"EMBEDDING_DIM={self.EMBEDDING_DIM} depasse la dimension native "
                f"de {provider.label} ({provider.native_dim})"
            )

        # Cache PostgreSQL (connexion synchrone pour les taches de fond).
        # Le service ne POSSEDE plus de moteur : il emprunte celui de
        # app.core.database. Un create_engine par instance restait ouvert pour
        # toute la duree du processus, et chaque loi traitee en creait un.
        self.use_cache = use_cache
        self._sync_session_factory = SyncSessionLocal if use_cache else None

        logger.info(
            f"✅ EmbeddingService initialisé "
            f"(dim={self.EMBEDDING_DIM}, cache={self.use_cache}, provider={provider.label})"
        )

    @property
    def client(self):
        """
        Client du fournisseur, s'il en a un. Conserve parce que des appelants
        et des tests le lisent : le delai du client Gemini, ou le remplacement
        de `client.models.embed_content` par une doublure.
        """
        return getattr(self._provider, "client", None)

    @property
    def provider(self):
        """Fournisseur de cette instance."""
        return self._provider

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

        # Génération par le fournisseur, avec reprises s'il en vaut la peine
        for attempt in range(self.MAX_RETRIES):
            try:
                embedding = self._provider.embed([text], task_type, self.EMBEDDING_DIM)[0]

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

                if attempt < self.MAX_RETRIES - 1 and self._provider.retryable:
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

                if chunk_idx + batch_size < total and self._provider.inter_batch_delay_s:
                    time.sleep(self._provider.inter_batch_delay_s)

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
            "provider": self._provider.label,
            "model": self._provider.model,
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
                new_embeddings = self._provider.embed(
                    chunk_texts, self.TASK_DOCUMENT, self.EMBEDDING_DIM
                )

                # Un fournisseur peut renvoyer MOINS de vecteurs que de textes.
                # Sans ce controle, zip() perdait silencieusement la queue du
                # lot, puis embeddings_dict[idx] levait un KeyError bien plus
                # loin, sans rapport apparent avec la cause.
                if len(new_embeddings) != len(chunk_texts):
                    raise EmbeddingServiceError(
                        f"Réponse incomplète: {len(new_embeddings)} vecteurs "
                        f"pour {len(chunk_texts)} textes"
                    )

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

                if attempt < max_retries - 1 and self._provider.retryable:
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

        Contient l'EMPREINTE du fournisseur, la DIMENSION et le type de tâche,
        pas seulement le texte : sans l'empreinte, deux fournisseurs a la meme
        dimension se resserviraient leurs vecteurs ; sans la dimension, un
        vecteur écrit sous une autre dimension serait resservi tel quel ; sans
        le type de tâche, un même texte encodé comme document ou comme question
        partagerait une entrée alors que les vecteurs diffèrent.

        L'empreinte est celle du fournisseur de L'INSTANCE, pas une constante
        de classe figee a l'import.

        Le texte n'est PLUS mis en minuscules. Les modeles sont sensibles à la
        casse et le français juridique la porte ("Article PREMIER" n'est pas
        "article premier") : minusculer confondait des textes distincts sous
        une seule clé.
        """
        payload = "\x00".join([
            self.CACHE_KEY_VERSION,
            self._provider.fingerprint,
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
