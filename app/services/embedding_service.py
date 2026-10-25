"""
Service de génération d'embeddings vectoriels pour recherche sémantique.

Le service ne parle plus directement a un modele : il delegue a un FOURNISSEUR
interchangeable, et garde pour lui tout le reste — validation, cache
PostgreSQL, gardes de dimension, normalisation, reprises, ordre des lots.

POURQUOI CETTE SEPARATION. Le modele d'embeddings est le composant que JuriX
changera le plus souvent : il a deja change trois fois (Gemini, EmbeddingGemma
en local, Voyage). Le changer doit etre un reglage, pas un chantier. Un
fournisseur ne sait qu'une chose — transformer des textes en vecteurs — et
laisse remonter ses exceptions.

Architecture:
- Fournisseurs: l'API Voyage AI (defaut, VoyageProvider) ou l'API Gemini
  (GeminiProvider), selon settings.EMBEDDING_PROVIDER
- Dimensions: settings.EMBEDDING_DIM (1024)
- Cache: Table embedding_cache PostgreSQL avec TTL 7 jours (cache en base)

Author: JuriX Team
Version: 5.0.0 (Voyage AI)
"""

import asyncio
import hashlib
import json
import logging
import re
import threading
import time
from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from functools import lru_cache, partial
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

from app.core.config import settings
from app.core.database import SyncSessionLocal

logger = logging.getLogger(__name__)

# Fils dedies aux QUESTIONS. Hors de l'executeur par defaut d'asyncio
# (min(32, coeurs + 4) fils, 6 sur la VM de production), que l'ingestion
# occupe un fil par document : un fournisseur lent, dont les appels survivent a
# l'echeance (voir generate_embedding_async), saturerait cet executeur et
# ferait attendre tout le reste. Ici, il ne sature que ses quatre fils, et les
# questions suivantes retombent sur le plein texte a l'echeance.
_EXECUTEUR_QUESTIONS = ThreadPoolExecutor(max_workers=4, thread_name_prefix="embed-question")

# Vecteurs de QUESTIONS gardes en memoire par une instance (cache LRU).
# Une recherche de 1 Ko de vecteur ne merite pas mieux : 512 questions, 2 Mo.
VECTEURS_QUESTIONS_MAX = 512


class _VecteursRecents:
    """
    Les derniers vecteurs de questions, en memoire du processus : un LRU borne,
    partage entre fils (l'appel au fournisseur rend la main sur un fil
    d'executeur, la lecture se fait dans la boucle).

    POURQUOI EN MEMOIRE, et non dans la table embedding_cache. Les questions y
    passaient par le moteur SYNCHRONE (psycopg2), et ce chemin ne servait
    jamais en production : app/core/database.py derive son URL de DATABASE_URL
    en retirant seulement « +asyncpg », et garde la query string. Avec le
    pooler, DATABASE_URL porte « ?prepared_statement_cache_size=0 », option
    que psycopg2 refuse avant tout acces reseau (« invalid dsn: invalid
    connection option », verifie) ; l'erreur etait avalee au niveau DEBUG.
    Bilan mesure : embedding_cache sans une ligne inseree, et en local, ou la
    connexion reussit, quatre allers-retours avant chaque appel au
    fournisseur. La reparer ferait payer ces quatre allers-retours en
    production aussi, sur chaque question nouvelle, pour ne servir que les
    questions reposees au-dela des cinq minutes de query_cache.

    Les vecteurs gardes sont en lecture seule : un appelant qui en ecrirait un
    corromprait la question suivante ; il recoit plutot une erreur.
    """

    def __init__(self, taille: int):
        self.taille = taille
        self._entrees: "OrderedDict[str, np.ndarray]" = OrderedDict()
        self._verrou = threading.Lock()

    def lire(self, cle: str) -> Optional[np.ndarray]:
        with self._verrou:
            vecteur = self._entrees.get(cle)
            if vecteur is not None:
                self._entrees.move_to_end(cle)
            return vecteur

    def garder(self, cle: str, vecteur: np.ndarray) -> np.ndarray:
        """Garde une COPIE en lecture seule de `vecteur`, et la rend."""
        copie = np.array(vecteur, dtype=np.float32)
        copie.flags.writeable = False
        with self._verrou:
            self._entrees[cle] = copie
            self._entrees.move_to_end(cle)
            while len(self._entrees) > self.taille:
                self._entrees.popitem(last=False)
        return copie

    def __len__(self) -> int:
        with self._verrou:
            return len(self._entrees)


def motif_court(erreur: BaseException) -> str:
    """
    « HTTP 429 », « delai depasse »... : jamais le message, qui peut porter le
    detail d'un refus, voire la cle. Pour GET /search/health, route publique.
    """
    if isinstance(erreur, TimeoutError):
        return "delai depasse"
    message = str(erreur)
    code = re.search(r"HTTP (\d{3})", message)
    if code:
        return f"HTTP {code.group(1)}"
    if "Pas de vecteur pour la question" in message or "delai depasse" in message:
        return "delai depasse"
    if "injoignable" in message:
        return "injoignable"
    return type(erreur.__cause__ or erreur).__name__


class SuiviSemantique:
    """
    Issue des volets semantiques recents : chaque appel au fournisseur pour une
    question, et chaque echec de la requete vectorielle. Une question servie
    par le cache ne compte pas : elle ne dit rien du fournisseur.

    POURQUOI. La sonde du demarrage ne voit ni une cle revoquee ensuite, ni un
    compte bride (3 requetes par minute sans moyen de paiement) : chaque
    question retombait sur le plein texte, sans autre trace que le journal.
    Un TAUX sur une fenetre glissante, et non des echecs consecutifs : sous un
    bridage, succes et refus alternent ; et un etat qui ne vieillit pas
    laisserait la sante « degraded » une nuit entiere apres une coupure d'une
    minute.
    """

    FENETRE_S = 15 * 60
    MINIMUM = 3
    TAUX_DEGRADE = 0.3

    def __init__(self):
        self._evenements = deque(maxlen=200)
        self._verrou = threading.Lock()
        # Issue de la sonde du demarrage : (motif ou None, horodatage ISO,
        # instant monotone). Un echec passager garde le service ; la sante le
        # montre jusqu'au premier appel reussi qui suit.
        self._sonde: Optional[tuple] = None

    def noter_sonde(self, erreur: Optional[BaseException]) -> None:
        from datetime import datetime, timezone

        motif = None if erreur is None else motif_court(erreur)
        quand = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._verrou:
            self._sonde = (motif, quand, time.monotonic())

    def noter(self, erreur: Optional[BaseException]) -> None:
        from datetime import datetime, timezone

        motif = None if erreur is None else motif_court(erreur)
        quand = datetime.now(timezone.utc).isoformat(timespec="seconds")
        with self._verrou:
            self._evenements.append((time.monotonic(), motif, quand))

    def vider(self) -> None:
        with self._verrou:
            self._evenements.clear()
            self._sonde = None

    def etat(self) -> Dict[str, Any]:
        limite = time.monotonic() - self.FENETRE_S
        with self._verrou:
            tous = list(self._evenements)
            sonde = self._sonde
        recents = [e for e in tous if e[0] >= limite]
        echecs = [e for e in recents if e[1] is not None]
        dernier = echecs[-1] if echecs else None
        # Sonde ratee au demarrage, et aucun appel reussi depuis : rien ne dit
        # que le volet semantique fonctionne.
        sonde_en_echec = bool(sonde and sonde[0] is not None) and not any(
            e[1] is None and e[0] >= sonde[2] for e in tous
        )
        return {
            "fenetre_min": self.FENETRE_S // 60,
            "appels": len(recents),
            "echecs": len(echecs),
            "derniere_erreur": dernier[1] if dernier else None,
            "derniere_erreur_le": dernier[2] if dernier else None,
            "sonde_demarrage": None if sonde is None else {"erreur": sonde[0], "le": sonde[1]},
            "degrade": sonde_en_echec or (
                len(echecs) >= self.MINIMUM and len(echecs) / len(recents) >= self.TAUX_DEGRADE
            ),
        }


suivi_semantique = SuiviSemantique()


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


class EmbeddingDefinitiveError(EmbeddingServiceError):
    """
    Echec qu'un nouvel essai reproduirait : cle refusee, requete invalide,
    tache ou dimension inconnues du fournisseur.

    Jamais rejouee par le service. Sous-classe d'EmbeddingServiceError : les
    appelants qui attrapent celle-ci (process_law, regenerate_embeddings)
    l'attrapent aussi. Au demarrage, elle ecarte le service
    (search_service.precharger_les_embeddings) ; une panne passagere, non.

    `code` : le code HTTP du refus, None pour un refus avant tout envoi. 400
    et 422 visent la requete (un texte), les autres la configuration (cle,
    droits, modele) : un traitement par lots peut sauter un lot refuse, pas
    continuer avec une cle refusee.
    """

    def __init__(self, message: str, code: Optional[int] = None):
        super().__init__(message)
        self.code = code

    @property
    def vise_la_configuration(self) -> bool:
        return self.code not in (400, 422)


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
                      fournisseur n'a pas a s'en charger. Une erreur qu'un
                      nouvel essai reproduirait est une
                      EmbeddingDefinitiveError, jamais rejouee.

    Membres FACULTATIFS, lus par getattr :

    - `dimensions_permises` : les seules dimensions acceptees, quand le modele
                      n'en accepte pas d'autres que celles-la (Voyage)
    - `max_caracteres` : texte le plus long accepte (defaut :
                      EmbeddingService.MAX_TEXT_LENGTH)
    - `delai_question_s` : delai d'un appel pour une question ; le service en
                      fait l'echeance dure de generate_embedding_async
    - `prechauffer(dim)` : sonde appelee au demarrage de l'API
    """

    name = "gemini"
    label = "Gemini API"
    native_dim = 3072
    retryable = True
    inter_batch_delay_s = 0.5
    # La limite d'avant Voyage, gardee pour ce fournisseur : 2 048 jetons de
    # contexte chez gemini-embedding-001.
    max_caracteres = 10_000

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
        try:
            return self._embed(texts, task_type, dim)
        except genai_errors.ClientError as e:
            # 429 se rejoue, et _is_daily_quota_error lit son message. Les
            # autres 4xx sont definitifs ; une cle invalide arrive en 400
            # (API_KEY_INVALID) mais vise la configuration, pas la requete.
            if e.code == 429:
                raise
            cle_refusee = e.code in (401, 403) or "API_KEY_INVALID" in str(e)
            raise EmbeddingDefinitiveError(
                f"Gemini HTTP {e.code} : {str(e)[:200]}", code=401 if cle_refusee else e.code
            ) from None

    def _embed(self, texts: List[str], task_type: str, dim: int) -> List[np.ndarray]:
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


VOYAGE_URL = "https://api.voyageai.com/v1/embeddings"

# Transport HTTP des appels a Voyage. None : le reseau. Les tests y posent un
# httpx.MockTransport (doublure, ou garde qui interdit tout appel reel). Lu a
# CHAQUE appel, il s'applique aussi au singleton deja construit.
transport_http: Optional[Any] = None

# Taches du service traduites en input_type de Voyage. Voyage ajoute lui-meme
# son gabarit (« Represent the query for retrieving supporting documents: »,
# « Represent the document for retrieval: ») : aucun prefixe cote client, il
# serait double. Un input_type absent est accepte par l'API mais encode SANS
# gabarit, dans un espace voisin et sans erreur : d'ou le refus d'une tache
# inconnue.
_VOYAGE_INPUT_TYPES = {"RETRIEVAL_QUERY": "query", "RETRIEVAL_DOCUMENT": "document"}

# Codes qu'un nouvel essai reproduirait : requete, cle ou droits a corriger.
# 429 et 5xx, eux, se rejouent apres une pause (docs.voyageai.com, error-codes).
_VOYAGE_CODES_DEFINITIFS = frozenset({400, 401, 403, 404, 422})


def empreinte_voyage(modele: str) -> str:
    """
    Empreinte des vecteurs Voyage d'un modele.

    Fonction de module : le chargement des vecteurs precalcules
    (scripts/regenerate_embeddings.py --depuis-lots) l'ecrit en base sans
    construire de fournisseur, donc sans cle. Elle doit rester EXACTEMENT celle
    du fournisseur : un seul caractere d'ecart, et les vecteurs charges passent
    pour etrangers — /search/health en « degraded », et une regeneration
    complete qui les recalculerait tous.
    """
    return f"voyage|{modele}"


class VoyageProvider:
    """
    Embeddings par l'API Voyage AI (voyage-4-large par defaut).

    Respecte le contrat decrit sur GeminiProvider. Choisi par mesure sur le
    jeu relu de 102 questions et tout le corpus (MRR@20 0,758 contre 0,537
    pour EmbeddingGemma en francais, 0,680 contre 0,537 en anglais).

    Le corps de requete reprend celui des vecteurs du corpus
    (data/mesure_voyage/vectoriser_voyage.py) : memes input_type, dimension
    1024, sortie float et troncature cote serveur. En changer un seul element
    changerait l'espace des questions sans la moindre erreur. Les deux derniers
    sont les defauts de l'API ; ils sont envoyes quand meme, pour qu'un
    changement de defaut chez Voyage ne change rien ici.

    Aucun appel reseau a la construction : construire ne coute rien, et ne
    depend ni du reseau ni du quota (tests, --dry-run).
    """

    name = "voyage"
    label = "Voyage AI"
    native_dim = 2048
    dimensions_permises = (256, 512, 1024, 2048)
    retryable = True
    # Palier 1 (carte enregistree) : 2 000 requetes et 3 M jetons par minute,
    # pour toute l'organisation. Sans carte, 3 requetes et 10 000 jetons par
    # minute : inutilisable pour le site.
    inter_batch_delay_s = 0.0

    # Une requete accepte 1 000 textes et 120 000 jetons pour voyage-4-large.
    # Marges : 900 textes et 90 000 jetons ESTIMES a 2,6 caracteres par jeton
    # (mesure sur le corpus : 3,1). Ce sont les reglages qui ont vectorise les
    # 32 303 morceaux du corpus sans un refus. Un tableau de chiffres compte
    # plus de jetons par caractere : un refus « trop de jetons » coupe alors le
    # lot en deux (_envoyer).
    TEXTES_PAR_REQUETE = 900
    JETONS_PAR_REQUETE = 90_000
    CAR_PAR_JETON = 2.6
    # Texte le plus long accepte : 30 000 jetons estimes, la coupe appliquee
    # aux vecteurs du corpus. Au-dela de 32 000 jetons reels, Voyage tronque
    # lui-meme (truncation). Le plus long morceau du corpus fait 15 323
    # caracteres.
    max_caracteres = 78_000

    # Duree de vie d'une connexion inactive, en secondes. 5 s par defaut chez
    # httpx : entre deux questions du chat, la connexion serait presque
    # toujours refermee, et chaque question repaierait la poignee de main.
    CONNEXION_INACTIVE_S = 60.0

    def __init__(self, api_key: Optional[str] = None):
        # strip : un espace ou un saut de ligne recopie avec la cle rendait
        # l'en-tete Authorization illegal.
        cle = (api_key or settings.VOYAGE_API_KEY or "").strip()
        if not cle:
            raise EmbeddingServiceError("VOYAGE_API_KEY non configurée")
        self._cle = cle
        self.model = settings.VOYAGE_EMBEDDING_MODEL
        # Client HTTP GARDE d'un appel a l'autre, construit au premier appel.
        # Un client par appel rouvrait TCP et TLS a chaque question : mesure
        # depuis le poste de developpement, 1,14 s de mediane contre 0,42 s
        # avec une connexion reutilisee. httpx.Client se partage entre fils.
        self._client_http: Optional[httpx.Client] = None
        self._transport_du_client: Optional[Any] = None
        self._verrou_client = threading.Lock()

    @property
    def fingerprint(self) -> str:
        return empreinte_voyage(self.model)

    @property
    def delai_question_s(self) -> float:
        """Delai d'un appel pour une question, relu a chaque acces."""
        return settings.VOYAGE_TIMEOUT_QUESTION_S

    def embed(self, texts: List[str], task_type: str, dim: int) -> List[np.ndarray]:
        input_type = _VOYAGE_INPUT_TYPES.get(task_type)
        if input_type is None:
            raise EmbeddingDefinitiveError(f"task_type inconnu pour Voyage : {task_type}")
        if dim not in self.dimensions_permises:
            raise EmbeddingDefinitiveError(
                f"Dimension {dim} refusee par Voyage (permises : {self.dimensions_permises})"
            )

        # Une question attend l'utilisateur : delai court, au-dela duquel la
        # recherche repond par le plein texte seul. Un lot de documents peut
        # peser 90 000 jetons : delai long. La connexion, elle, est toujours
        # courte.
        if input_type == "query":
            delai = settings.VOYAGE_TIMEOUT_QUESTION_S
        else:
            delai = settings.VOYAGE_TIMEOUT_DOCUMENT_S
        timeout = httpx.Timeout(delai, connect=min(2.0, delai))

        client = self._client()
        vecteurs: List[np.ndarray] = []
        for lot in self._lots(texts):
            vecteurs.extend(self._envoyer(client, lot, input_type, dim, timeout))
        return vecteurs

    def fermer(self) -> None:
        """Ferme la connexion gardee (arret de l'API, fin d'un test reel)."""
        with self._verrou_client:
            if self._client_http is not None:
                self._client_http.close()
            self._client_http = None
            self._transport_du_client = None

    def _client(self) -> httpx.Client:
        """
        Le client garde, reconstruit si `transport_http` a change : la garde
        reseau des tests le remplace a chaque test, et un client construit
        avant la garde garderait l'ancien transport — le vrai reseau.
        """
        transport = transport_http
        with self._verrou_client:
            if self._client_http is None or self._transport_du_client is not transport:
                if self._client_http is not None:
                    self._client_http.close()
                self._client_http = httpx.Client(
                    transport=transport,
                    limits=httpx.Limits(keepalive_expiry=self.CONNEXION_INACTIVE_S),
                )
                self._transport_du_client = transport
            return self._client_http

    def prechauffer(self, dim: int) -> None:
        """
        Vectorise une question d'un mot (1 a 3 jetons), MAINTENANT.

        Appele au demarrage de l'API : une cle refusee, un modele ou une
        dimension inconnus se voient la, pas a la premiere question, ou la
        recherche hybride avalerait l'erreur et repondrait en plein texte seul.
        """
        vecteur = self.embed(["prechauffage"], "RETRIEVAL_QUERY", dim)[0]
        if vecteur.shape != (dim,):
            raise EmbeddingDefinitiveError(f"Voyage rend {vecteur.shape}, attendu ({dim},)")

    def _estimer(self, texte: str) -> int:
        return int(len(texte) / self.CAR_PAR_JETON) + 1

    def _lots(self, texts: List[str]):
        """Lots d'au plus TEXTES_PAR_REQUETE textes et JETONS_PAR_REQUETE jetons estimes."""
        lot: List[str] = []
        jetons = 0
        for texte in texts:
            j = self._estimer(texte)
            if lot and (jetons + j > self.JETONS_PAR_REQUETE or len(lot) >= self.TEXTES_PAR_REQUETE):
                yield lot
                lot, jetons = [], 0
            lot.append(texte)
            jetons += j
        if lot:
            yield lot

    def _envoyer(
        self, client: httpx.Client, lot: List[str], input_type: str, dim: int, timeout: httpx.Timeout,
    ) -> List[np.ndarray]:
        corps = {
            "input": lot,
            "model": self.model,
            "input_type": input_type,
            "output_dimension": dim,
            "output_dtype": "float",
            "truncation": True,
        }
        try:
            r = client.post(
                VOYAGE_URL, headers={"Authorization": f"Bearer {self._cle}"}, json=corps, timeout=timeout,
            )
        except httpx.HTTPError as e:
            # Delai depasse, DNS, connexion refusee : passager, donc rejoue par
            # le service pour un lot de documents. Cle masquee, et sans chainer
            # l'exception d'origine : h11 recopie un en-tete refuse (cle suivie
            # d'un espace...) dans son message, et une trace l'afficherait.
            # Une lenteur n'est pas une panne reseau : le motif les distingue,
            # pour qu'on releve le delai plutot que de chercher une coupure.
            message = str(e).replace(self._cle, "***")
            nature = "delai depasse" if isinstance(e, httpx.TimeoutException) else "injoignable"
            raise EmbeddingServiceError(
                f"Voyage {nature} : {type(e).__name__}: {message}"
            ) from None

        if r.status_code == 200:
            try:
                donnees = sorted(r.json()["data"], key=lambda e: e["index"])
                vecteurs = [np.asarray(e["embedding"], dtype=np.float32) for e in donnees]
            except (ValueError, KeyError, TypeError) as e:
                raise EmbeddingServiceError(f"Réponse Voyage illisible : {type(e).__name__}") from e
            if len(vecteurs) != len(lot):
                raise EmbeddingServiceError(
                    f"Réponse incomplète: {len(vecteurs)} vecteurs pour {len(lot)} textes"
                )
            return vecteurs

        detail = self._detail(r)
        if r.status_code == 400 and len(lot) > 1 and "token" in detail.lower():
            # Lot plus lourd que prevu (chiffres, tableaux) : coupe en deux.
            moitie = len(lot) // 2
            return (
                self._envoyer(client, lot[:moitie], input_type, dim, timeout)
                + self._envoyer(client, lot[moitie:], input_type, dim, timeout)
            )
        # Le code HTTP figure dans le message : regenerate_embeddings ne
        # reattend un lot que s'il y lit « 429 ».
        message = f"Voyage HTTP {r.status_code} : {detail}"
        if r.status_code in _VOYAGE_CODES_DEFINITIFS:
            raise EmbeddingDefinitiveError(message, code=r.status_code)
        raise EmbeddingServiceError(message)

    def _detail(self, r: httpx.Response) -> str:
        """Motif d'un refus, court, et jamais porteur de la cle."""
        try:
            corps = r.json()
            detail = str(corps.get("detail") or "") if isinstance(corps, dict) else ""
        except ValueError:
            detail = ""
        # Masquer AVANT de couper : coupee d'abord, une cle a cheval sur la
        # limite laissait passer son debut.
        return (detail or r.text or "").replace(self._cle, "***")[:200]


def _construire_fournisseur(api_key: Optional[str] = None):
    """
    Fournisseur par defaut, selon settings.EMBEDDING_PROVIDER. `api_key` est
    la cle du fournisseur configure.
    """
    if settings.EMBEDDING_PROVIDER == "voyage":
        return VoyageProvider(api_key=api_key)
    return GeminiProvider(api_key=api_key)


class EmbeddingService:
    """
    Service de génération d'embeddings vectoriels pour recherche sémantique.

    Delegue la production des vecteurs a un fournisseur (contrat decrit sur
    GeminiProvider), avec cache PostgreSQL (table embedding_cache). Supporte
    FR/EN, normalisation L2 inconditionnelle.

    Dimension : settings.EMBEDDING_DIM, demandee au fournisseur a CHAQUE appel
    et jamais figee en lui — scripts/eval/validate_slicing.py change la
    dimension d'une instance apres coup.

    Le modele n'est PAS un attribut de classe : il depend du fournisseur de
    chaque instance (service.provider.model). Un attribut fige a l'import
    mentait des qu'un fournisseur etait injecte.

    Attributes:
        EMBEDDING_DIM: Dimension des embeddings (settings.EMBEDDING_DIM)
        CACHE_TTL_SECONDS: Durée de vie cache (7 jours)
        use_cache: Flag activation cache
    """

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
    # Texte le plus long accepte, par defaut. Un fournisseur qui declare
    # `max_caracteres` impose le sien a l'instance (voir __init__) : lire
    # MAX_TEXT_LENGTH sur l'instance, jamais sur la classe.
    MAX_TEXT_LENGTH = 78_000
    # Tentatives pour un DOCUMENT. Une question n'en fait qu'une : elle attend
    # un utilisateur, et un echec passager ne lui coute que le volet semantique
    # de sa recherche (repli sur le plein texte).
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # seconds

    # Types de tâche. La recherche est asymétrique : un article indexé et une
    # question d'utilisateur ne doivent pas être encodés de la même façon,
    # sinon la pertinence se dégrade. Chaque fournisseur les traduit dans son
    # vocabulaire (input_type de Voyage, task_type de Gemini).
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
            api_key: Clé du fournisseur configuré (si None, celle des settings)
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
                fournisseur = settings.EMBEDDING_PROVIDER
                logger.error(f"❌ Échec configuration du fournisseur « {fournisseur} »: {e}")
                raise EmbeddingServiceError(
                    f"Impossible de configurer le fournisseur d'embeddings « {fournisseur} »: {e}"
                ) from e
        self._provider = provider

        # Refuse a la construction ce qui echouerait au premier appel. Sans ce
        # controle, une dimension trop grande ne se revele qu'apres trois
        # tentatives, et la recherche hybride avale l'erreur sans bruit.
        if self.EMBEDDING_DIM > provider.native_dim:
            raise EmbeddingServiceError(
                f"EMBEDDING_DIM={self.EMBEDDING_DIM} depasse la dimension native "
                f"de {provider.label} ({provider.native_dim})"
            )
        # Voyage n'accepte que quelques dimensions : 768 passerait le controle
        # ci-dessus et n'echouerait qu'au premier appel, en HTTP 400, que la
        # recherche hybride avalerait.
        permises = getattr(provider, "dimensions_permises", None)
        if permises and self.EMBEDDING_DIM not in permises:
            raise EmbeddingServiceError(
                f"EMBEDDING_DIM={self.EMBEDDING_DIM} n'est pas une dimension de "
                f"{provider.label} (permises : {', '.join(map(str, permises))})"
            )
        self.MAX_TEXT_LENGTH = getattr(provider, "max_caracteres", self.MAX_TEXT_LENGTH)

        # Cache PostgreSQL (connexion synchrone pour les taches de fond).
        # Le service ne POSSEDE plus de moteur : il emprunte celui de
        # app.core.database. Un create_engine par instance restait ouvert pour
        # toute la duree du processus, et chaque loi traitee en creait un.
        self.use_cache = use_cache
        self._sync_session_factory = SyncSessionLocal if use_cache else None
        # Les questions, elles, ont leur cache en memoire : voir _VecteursRecents.
        self._vecteurs_questions = _VecteursRecents(VECTEURS_QUESTIONS_MAX)

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

    def fermer(self) -> None:
        """Ferme la connexion gardee par le fournisseur, s'il en garde une."""
        fermer = getattr(self._provider, "fermer", None)
        if fermer is not None:
            fermer()

    def prechauffer(self) -> None:
        """
        Sonde le fournisseur maintenant, s'il en a une : pour Voyage, une
        question d'un mot qui verifie la cle, le modele, la dimension et la
        sortie reseau. Sans effet pour un fournisseur sans sonde (Gemini).

        Appel direct, sans cache ni reprise : une entree en cache masquerait
        une cle revoquee, et une reprise retarderait le demarrage.
        """
        prechauffer = getattr(self._provider, "prechauffer", None)
        if prechauffer is not None:
            prechauffer(self.EMBEDDING_DIM)

    # ==================== PUBLIC API ====================

    def generate_embedding(
        self,
        text: str,
        normalize: bool = True,
        task_type: str = TASK_DOCUMENT,
    ) -> np.ndarray:
        """
        Génère l'embedding pour un texte unique, par le fournisseur.

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
            ValueError: Si texte vide ou trop long (> MAX_TEXT_LENGTH)
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

        embedding = self._produire(text, normalize, task_type)

        # Stockage dans cache PostgreSQL
        if self.use_cache:
            self._store_in_pg_cache(cache_key, embedding)

        elapsed = (time.time() - start_time) * 1000
        logger.debug(f"⏱️ Embedding généré en {elapsed:.1f}ms (dim={embedding.shape[0]})")
        return embedding

    def _produire(self, text: str, normalize: bool, task_type: str) -> np.ndarray:
        """
        Le vecteur d'un texte deja valide, par le fournisseur : ni lecture ni
        ecriture du cache, reprises s'il en vaut la peine.

        Une question n'a droit qu'a un essai : trois essais et leurs pauses
        (1 s puis 2 s) la feraient attendre plus de dix secondes avant le
        repli sur le plein texte.
        """
        essais = 1 if task_type == self.TASK_QUERY else self.MAX_RETRIES
        for attempt in range(essais):
            try:
                embedding = self._provider.embed([text], task_type, self.EMBEDDING_DIM)[0]

                if embedding.shape[0] != self.EMBEDDING_DIM:
                    raise EmbeddingServiceError(
                        f"Dimension incorrecte: {embedding.shape[0]} != {self.EMBEDDING_DIM}"
                    )

                return self._normalize(embedding, normalize)

            except EmbeddingDefinitiveError as e:
                # Relancee TELLE QUELLE : son type dit a l'appelant qu'il est
                # inutile de recommencer.
                logger.error(f"❌ Erreur définitive du fournisseur d'embeddings: {e}")
                raise
            except Exception as e:
                if _is_daily_quota_error(e):
                    raise QuotaExhaustedError(
                        f"Quota journalier d'embeddings epuise: {e}"
                    ) from e

                if attempt < essais - 1 and self._provider.retryable:
                    logger.warning(f"⚠️ Tentative {attempt + 1} échouée, retry: {e}")
                    time.sleep(self.RETRY_DELAY * (attempt + 1))
                else:
                    logger.error(f"❌ Erreur génération embedding après {attempt + 1} tentative(s): {e}")
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
            # Comme une question : un essai, delai court. Sur le chemin des
            # documents, un fournisseur muet tiendrait la sonde six minutes.
            test_emb = self.generate_embedding("test", normalize=False, task_type=self.TASK_QUERY)
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

            except EmbeddingDefinitiveError:
                # Rejouer un refus (cle, requete) ne ferait que le repeter.
                raise
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

        Les fournisseurs sont synchrones et le backoff est un time.sleep :
        appelée telle quelle depuis une coroutine, generate_embedding gèle la
        boucle d'événements pour tout l'aller-retour réseau. Tout appel depuis
        du code async doit passer par ici.

        task_type vaut TASK_QUERY par défaut : le seul appelant asynchrone est
        la recherche sémantique, qui encode une question.

        Une QUESTION deja vue (meme texte, au blanc de bord pres, meme modele,
        meme dimension) est servie par le cache en memoire de l'instance
        (_VecteursRecents), dans la boucle : ni fil, ni fournisseur, ni base.
        Les questions ne passent plus par la table embedding_cache : voir
        _VecteursRecents pour la raison. Une question servie ainsi ne compte
        pas dans suivi_semantique : elle ne dit rien du fournisseur.

        Pour une question nouvelle, si le fournisseur declare
        `delai_question_s`, l'attente est BORNEE a ce delai plus une seconde :
        au-dela, EmbeddingServiceError, et la recherche hybride repond par le
        plein texte seul. httpx borne la connexion et chaque lecture, mais pas
        la resolution DNS : seule cette echeance garantit le delai. Le fil,
        lui, ne s'interrompt pas : il finit son appel en arriere-plan (borne
        par les delais httpx), puis libere sa place. Les questions ont leurs
        propres fils (_EXECUTEUR_QUESTIONS).
        """
        if task_type != self.TASK_QUERY:
            return await asyncio.to_thread(self.generate_embedding, text, True, task_type)

        self._validate_text(text)
        cle = self._cache_key(text, task_type) if self.use_cache else None
        if cle is not None:
            deja_vue = self._vecteurs_questions.lire(cle)
            if deja_vue is not None:
                return deja_vue

        boucle = asyncio.get_running_loop()
        delai = getattr(self._provider, "delai_question_s", None)
        echeance = delai + 1.0 if delai else None
        try:
            async with asyncio.timeout(echeance):
                vecteur = await boucle.run_in_executor(
                    _EXECUTEUR_QUESTIONS, partial(self._produire, text, True, task_type)
                )
        except TimeoutError as e:
            suivi_semantique.noter(e)
            raise EmbeddingServiceError(
                f"Pas de vecteur pour la question en {echeance:.1f} s ({self._provider.label})"
            ) from e
        except Exception as e:
            suivi_semantique.noter(e)
            raise
        suivi_semantique.noter(None)
        if cle is not None:
            return self._vecteurs_questions.garder(cle, vecteur)
        return vecteur


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
