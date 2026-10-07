"""
GroqService : client de l'API Groq, pour les decisions courtes en JSON.

Deux usages : le classement d'intention de chaque message du chat
(intent_classifier.py) et le classement des lois dans les 14 domaines
(legal_domain_classifier.py), au fil de l'ingestion ou par lots.

CE QUI A CHANGE, ET POURQUOI. La premiere version envoyait ses requetes sans
rien compter et rendait None sur toute erreur. Sur le palier gratuit — 30
requetes et 8 000 jetons par minute, 1 000 requetes et 200 000 jetons par
jour, PAR MODELE — le reclassement recevait 89 refus sur 122 requetes, et un
quota epuise ressemblait a une reponse vide : le classement rangeait alors la
loi en Droit Administratif, sans le dire.

Desormais :
- chaque appel passe par le limiteur de son modele (app/core/limiteur.py), qui
  fait attendre AVANT d'envoyer ;
- un 429 est respecte (Retry-After) puis l'appel est rejoue ; une attente de
  plus de GROQ_ATTENTE_MAX_429_S, ou un message de quota JOURNALIER, ferme le
  modele jusqu'a l'echeance et leve GroqQuotaError ;
- les 5xx et les pannes reseau sont rejoues 3 fois, puis GroqIndisponibleError ;
- une reponse inexploitable leve GroqReponseInvalideError ;
- la sortie est contrainte par un schema JSON strict, et la reflexion des
  modeles est coupee ou reduite : elle consommerait le budget de jetons.

Author: JuriX Team
"""

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Optional

import httpx

from app.core.config import settings
from app.core.limiteur import AttenteTropLongue, Limiteur, Reservation, lire_duree

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODELES_URL = "https://api.groq.com/openai/v1/models"

# Transport HTTP de tous les appels sortants. None : le reseau. Les tests y
# posent un httpx.MockTransport (doublure, ou garde qui interdit tout appel
# reel). Lu a CHAQUE appel, il s'applique aussi au singleton deja construit.
transport_http: Optional[Any] = None

# Un appel = au plus ESSAIS envois (429 courts, 5xx, pannes reseau compris).
ESSAIS = 4
DELAI_5XX_S = 1.5

# Le message d'un 429 de quota journalier : « Limit 200000, Used 199532,
# Requested 1024 ... tokens per day (TPD) ». Aucun en-tete ne le porte.
_QUOTA_DU_JOUR = re.compile(r"per day|\(TPD\)|\(RPD\)", re.IGNORECASE)

# Les raisons du limiteur qui disent « quota », par opposition a « debit ».
_RAISONS_DE_QUOTA = {"requetes/jour", "jetons/jour", "quota Groq"}

# Duree de vie de la sonde de sante : /health peut etre appele souvent.
SANTE_CACHE_S = 60.0


class GroqServiceError(Exception):
    """Erreur de l'API Groq."""


class GroqLimiteError(GroqServiceError):
    """
    L'appel n'a pas pu partir dans le delai accepte (limiteur ou 429).

    `retry_after` : secondes avant qu'un appel ait une chance d'aboutir.
    """

    def __init__(self, message: str, *, retry_after: float, raison: str = ""):
        super().__init__(message)
        self.retry_after = retry_after
        self.raison = raison


class GroqQuotaError(GroqLimiteError):
    """Quota du modele epuise : plus aucun appel avant `retry_after`."""


class GroqIndisponibleError(GroqServiceError):
    """Panne passagere : 5xx ou reseau, apres tous les essais."""


class GroqReponseInvalideError(GroqServiceError):
    """Reponse recue mais inexploitable : JSON invalide, ou coupee."""


@dataclass(frozen=True)
class ReponseGroq:
    """L'objet JSON rendu, et ce qu'il a coute."""

    donnees: Dict[str, Any]
    jetons: int
    modele: str


# ==================== LIMITEURS (UN PAR MODELE) ====================

_LIMITEURS: Dict[str, Limiteur] = {}
_VERROU_LIMITEURS = threading.Lock()


def limiteur_groq(modele: str) -> Limiteur:
    """
    Le limiteur du modele, partage par tout le processus.

    Un par modele et non un par service : chez Groq, les quotas sont propres
    a chaque modele, et le chat comme le pipeline passent par le meme modele.
    """
    with _VERROU_LIMITEURS:
        limiteur = _LIMITEURS.get(modele)
        if limiteur is None:
            limiteur = Limiteur(
                f"groq:{modele}",
                requetes_par_minute=settings.GROQ_REQUETES_PAR_MINUTE,
                rafale=settings.GROQ_RAFALE,
                jetons_par_minute=settings.GROQ_JETONS_PAR_MINUTE,
                jetons_entree_par_minute=settings.GROQ_JETONS_ENTREE_PAR_MINUTE,
                requetes_par_jour=settings.GROQ_REQUETES_PAR_JOUR,
                jetons_par_jour=settings.GROQ_JETONS_PAR_JOUR,
            )
            _LIMITEURS[modele] = limiteur
        return limiteur


def estimer_jetons(texte: str) -> int:
    """
    Jetons d'un texte, estimes LARGE : 3 caracteres par jeton.

    Le francais tourne autour de 3,5 a 4. Surestimer fait attendre un peu trop
    tot ; sous-estimer fait partir des requetes qui reviennent en 429. La
    reservation est corrigee par l'usage reel des la reponse.
    """
    return len(texte) // 3 + 1


def parametres_de_raisonnement(modele: str) -> Dict[str, Any]:
    """
    Coupe ou reduit la reflexion : elle est facturee sur max_completion_tokens.

    qwen3 accepte « none ». gpt-oss ne descend pas sous « low », et
    include_reasoning=False retire le raisonnement de la reponse.
    """
    if modele.startswith("qwen/"):
        return {"reasoning_effort": "none"}
    if modele.startswith("openai/gpt-oss"):
        return {"reasoning_effort": "low", "include_reasoning": False}
    return {}


def _message_d_erreur(reponse: httpx.Response) -> str:
    try:
        corps = reponse.json()
    except ValueError:
        return reponse.text[:300]
    erreur = corps.get("error") if isinstance(corps, dict) else None
    if isinstance(erreur, dict):
        return str(erreur.get("message") or erreur)[:300]
    return str(corps)[:300]


def _entier(valeur: Optional[str]) -> Optional[int]:
    try:
        return int(float(valeur)) if valeur is not None else None
    except ValueError:
        return None


class GroqService:
    """Client de l'API Groq. Sans etat propre : les limiteurs sont par modele."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
        *,
        limiteur_pour=limiteur_groq,
    ):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model_name = model_name or settings.GROQ_MODEL
        self.timeout = timeout or settings.GROQ_TIMEOUT_S
        # Injectable : les tests fournissent des limiteurs sur horloge factice.
        self._limiteur_pour = limiteur_pour
        self._sante: Optional[tuple] = None

        if not self.api_key:
            raise GroqServiceError("GROQ_API_KEY non configurée. Renseignez-la dans .env")

        logger.info(f"✅ GroqService initialisé: model={self.model_name}")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "JuriX-Backend/1.0",
        }

    def limiteur(self, modele: Optional[str] = None) -> Limiteur:
        return self._limiteur_pour(modele or self.model_name)

    # ------------------------------------------------------------ requete

    def _corps(
        self,
        modele: str,
        systeme: str,
        message: str,
        schema: Optional[Dict[str, Any]],
        nom_schema: str,
        max_jetons: int,
    ) -> Dict[str, Any]:
        if schema is None:
            format_de_reponse: Dict[str, Any] = {"type": "json_object"}
        else:
            format_de_reponse = {
                "type": "json_schema",
                "json_schema": {"name": nom_schema, "strict": True, "schema": schema},
            }
        return {
            "model": modele,
            "messages": [
                {"role": "system", "content": systeme},
                {"role": "user", "content": message},
            ],
            "temperature": 0,
            "max_completion_tokens": max_jetons,
            "response_format": format_de_reponse,
            **parametres_de_raisonnement(modele),
        }

    def _limite(self, erreur: AttenteTropLongue, modele: str) -> GroqLimiteError:
        """Traduit un refus du limiteur : quota si l'attente le dit."""
        quota = (
            erreur.raison in _RAISONS_DE_QUOTA
            or erreur.attente > settings.GROQ_ATTENTE_MAX_429_S
        )
        classe = GroqQuotaError if quota else GroqLimiteError
        return classe(
            f"Groq ({modele}) : {erreur}", retry_after=erreur.attente, raison=erreur.raison
        )

    def _recaler(self, limiteur: Limiteur, entetes: httpx.Headers) -> None:
        """Les en-tetes x-ratelimit-* voient aussi les appels des autres processus."""
        limiteur.recaler(
            requetes_restantes_jour=_entier(entetes.get("x-ratelimit-remaining-requests")),
            reprise_requetes_s=lire_duree(entetes.get("x-ratelimit-reset-requests")),
            jetons_restants_minute=_entier(entetes.get("x-ratelimit-remaining-tokens")),
        )

    def _interpreter(
        self,
        modele: str,
        reponse: httpx.Response,
        reservation: Reservation,
        essai: int,
        debut: float,
    ) -> Optional[ReponseGroq]:
        """
        La reponse, ou None s'il faut renvoyer (le limiteur a deja ete
        repousse), ou une exception. Commun aux chemins sync et async.
        """
        limiteur = self.limiteur(modele)
        self._recaler(limiteur, reponse.headers)
        code = reponse.status_code

        if code == 200:
            corps = reponse.json()
            usage = corps.get("usage") or {}
            if usage.get("total_tokens") is not None:
                reservation.corriger(usage["total_tokens"], usage.get("prompt_tokens"))
            choix = corps["choices"][0]
            fin = choix.get("finish_reason")
            contenu = (choix.get("message") or {}).get("content") or ""
            logger.info(
                "🧭 Groq %s : fin=%s jetons=%s duree=%.2fs",
                modele, fin, usage.get("total_tokens"), time.monotonic() - debut,
            )
            if fin == "length":
                raise GroqReponseInvalideError(
                    f"Reponse de {modele} coupee : max_completion_tokens atteint"
                )
            try:
                donnees = json.loads(contenu)
            except json.JSONDecodeError as e:
                raise GroqReponseInvalideError(
                    f"JSON invalide de {modele} : {contenu[:200]!r}"
                ) from e
            if not isinstance(donnees, dict):
                raise GroqReponseInvalideError(f"Objet JSON attendu de {modele}")
            return ReponseGroq(donnees, int(usage.get("total_tokens") or reservation.jetons), modele)

        message = _message_d_erreur(reponse)
        if code == 429:
            attente = lire_duree(reponse.headers.get("retry-after"))
            if attente is None:
                attente = 2.0 * essai
            if _QUOTA_DU_JOUR.search(message) or attente > settings.GROQ_ATTENTE_MAX_429_S:
                limiteur.repousser(attente, "quota Groq")
                raise GroqQuotaError(
                    f"Quota Groq epuise pour {modele} : {message}",
                    retry_after=attente, raison="quota Groq",
                )
            if essai >= ESSAIS:
                limiteur.repousser(attente, "429")
                raise GroqLimiteError(
                    f"Groq ({modele}) refuse encore apres {ESSAIS} essais : {message}",
                    retry_after=attente, raison="429",
                )
            limiteur.repousser(attente, "429")
            return None

        if code >= 500:
            if essai >= ESSAIS:
                raise GroqIndisponibleError(f"Groq ({modele}) HTTP {code} : {message}")
            limiteur.repousser(DELAI_5XX_S * 2 ** (essai - 1), f"HTTP {code}")
            return None

        # 400 (requete refusee, schema compris), 401 (cle), 413 (trop long) :
        # renvoyer ne changerait rien.
        raise GroqServiceError(f"Groq ({modele}) HTTP {code} : {message}")

    def _echec_reseau(self, modele: str, essai: int, erreur: Exception) -> None:
        if essai >= ESSAIS:
            raise GroqIndisponibleError(f"Groq ({modele}) injoignable : {erreur!r}") from erreur
        logger.warning("🧭 Groq %s : %r (essai %d/%d)", modele, erreur, essai, ESSAIS)
        self.limiteur(modele).repousser(DELAI_5XX_S * 2 ** (essai - 1), "reseau")

    def _preparer(self, modele, systeme, message, schema, nom_schema, max_jetons):
        corps = self._corps(modele, systeme, message, schema, nom_schema, max_jetons)
        entree = estimer_jetons(systeme + message + (json.dumps(schema) if schema else ""))
        return corps, entree, entree + max_jetons

    async def completer_json(
        self,
        *,
        systeme: str,
        message: str,
        schema: Optional[Dict[str, Any]] = None,
        nom_schema: str = "reponse",
        max_jetons: int = 256,
        modele: Optional[str] = None,
        attente_max: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> ReponseGroq:
        """
        Un objet JSON conforme a `schema` (mode strict), ou une exception typee.

        `attente_max` borne l'attente imposee par le limiteur (defaut :
        GROQ_ATTENTE_MAX_429_S) ; au-dela, GroqLimiteError sans rien envoyer.
        """
        modele = modele or self.model_name
        attente_max = settings.GROQ_ATTENTE_MAX_429_S if attente_max is None else attente_max
        corps, entree, total = self._preparer(
            modele, systeme, message, schema, nom_schema, max_jetons
        )
        limiteur = self.limiteur(modele)

        async with httpx.AsyncClient(
            timeout=timeout or self.timeout, transport=transport_http
        ) as client:
            for essai in range(1, ESSAIS + 1):
                try:
                    reservation = await limiteur.attendre(
                        total, attente_max, jetons_entree=entree
                    )
                except AttenteTropLongue as e:
                    raise self._limite(e, modele) from e
                debut = time.monotonic()
                try:
                    reponse = await client.post(GROQ_API_URL, headers=self._headers(), json=corps)
                except httpx.HTTPError as e:
                    self._echec_reseau(modele, essai, e)
                    continue
                resultat = self._interpreter(modele, reponse, reservation, essai, debut)
                if resultat is not None:
                    return resultat
        raise GroqIndisponibleError(f"Groq ({modele}) : aucun essai n'a abouti")

    def completer_json_sync(
        self,
        *,
        systeme: str,
        message: str,
        schema: Optional[Dict[str, Any]] = None,
        nom_schema: str = "reponse",
        max_jetons: int = 256,
        modele: Optional[str] = None,
        attente_max: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> ReponseGroq:
        """Version synchrone de `completer_json`, pour le pipeline et les scripts."""
        modele = modele or self.model_name
        attente_max = settings.GROQ_ATTENTE_MAX_429_S if attente_max is None else attente_max
        corps, entree, total = self._preparer(
            modele, systeme, message, schema, nom_schema, max_jetons
        )
        limiteur = self.limiteur(modele)

        with httpx.Client(timeout=timeout or self.timeout, transport=transport_http) as client:
            for essai in range(1, ESSAIS + 1):
                try:
                    reservation = limiteur.attendre_sync(
                        total, attente_max, jetons_entree=entree
                    )
                except AttenteTropLongue as e:
                    raise self._limite(e, modele) from e
                debut = time.monotonic()
                try:
                    reponse = client.post(GROQ_API_URL, headers=self._headers(), json=corps)
                except httpx.HTTPError as e:
                    self._echec_reseau(modele, essai, e)
                    continue
                resultat = self._interpreter(modele, reponse, reservation, essai, debut)
                if resultat is not None:
                    return resultat
        raise GroqIndisponibleError(f"Groq ({modele}) : aucun essai n'a abouti")

    # ------------------------------------------------------------- sante

    async def health_check(self) -> Dict[str, Any]:
        """
        Le modele est-il servi, et la cle acceptee ? Sans rien generer.

        `GET /models/{modele}` ne consomme aucun jeton. Le resultat est garde
        SANTE_CACHE_S secondes : une sonde appelee en boucle ne doit pas
        entamer le quota de requetes.
        """
        maintenant = time.monotonic()
        if self._sante and maintenant - self._sante[0] < SANTE_CACHE_S:
            return self._sante[1]
        try:
            async with httpx.AsyncClient(timeout=5.0, transport=transport_http) as client:
                reponse = await client.get(
                    f"{GROQ_MODELES_URL}/{self.model_name}", headers=self._headers()
                )
            if reponse.status_code == 200:
                resultat = {"status": "healthy", "model": self.model_name}
            else:
                resultat = {
                    "status": "unhealthy",
                    "model": self.model_name,
                    "error": f"HTTP {reponse.status_code} : {_message_d_erreur(reponse)}",
                }
        except httpx.HTTPError as e:
            resultat = {"status": "unhealthy", "model": self.model_name, "error": repr(e)}
        self._sante = (maintenant, resultat)
        return resultat


@lru_cache(maxsize=1)
def get_groq_service() -> GroqService:
    """Singleton cached instance de GroqService."""
    return GroqService()
