"""
MistralService - Client for Mistral AI API (LLM for RAG Chat).

Replaces GeminiService using Mistral's OpenAI-compatible REST API via httpx.AsyncClient.
Provides async methods for:
- generate(): Single response generation
- generate_stream(): Streaming response generation (SSE)
- health_check(): API connectivity check

CE QUI COUPAIT LES REPONSES, ET CE QUI LE REMPLACE :

- Une fin « length » (budget de jetons atteint) n'etait pas signalee en flux :
  la reponse s'arretait au milieu d'une phrase, sans mention. Les fins sont
  desormais normalisees dans le vocabulaire du reste du code (« STOP »,
  « MAX_TOKENS », « ERROR »), et une reponse coupee — par le budget, ou par
  un flux rompu apres les premiers mots — est COMPLETEE : une seule suite est
  demandee, le texte deja produit passe en prefixe de l'assistant
  (`"prefix": true`). La mention « reponse interrompue » ne reste que si la
  suite echoue.
- Un 429 (debit) devenait « quota epuise, revenez demain », sans nouvel
  essai. Le limiteur espace desormais les departs (30 requetes par minute
  pour ministral-14b) ; un 429 est rejoue (Retry-After, sinon attente
  croissante), 4 essais et 20 s au plus, puis le modele de secours, puis
  MistralOverloadedError : « service sature, reessayez ». MistralQuotaError ne
  dit plus que le quota MENSUEL.
- 60 s ne suffisaient pas a une comparaison de 8 192 jetons : le delai suit
  le budget demande.
- Chaque appel journalise modele, fin, jetons et duree : la prochaine coupure
  se lira dans les journaux.
"""

import json
import logging
import random
import re
import threading
import time
from functools import lru_cache
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

import httpx

from app.core.config import settings
from app.core.limiteur import AttenteTropLongue, Limiteur, lire_duree

logger = logging.getLogger(__name__)

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
MISTRAL_MODELES_URL = "https://api.mistral.ai/v1/models"

# Transport HTTP de tous les appels sortants. None : le reseau. Les tests y
# posent un httpx.MockTransport (doublure, ou garde qui interdit tout appel
# reel). Lu a CHAQUE appel, il s'applique aussi au singleton deja construit.
transport_http: Optional[Any] = None

# Envois au plus par appel (429, 5xx et pannes de connexion compris).
ESSAIS = 4

# Fins de Mistral, traduites dans le vocabulaire commun (celui de Gemini, que
# rag_service lit) : « length » et « model_length » sont des coupures.
FINS = {
    "stop": "STOP",
    "length": "MAX_TOKENS",
    "model_length": "MAX_TOKENS",
    "error": "ERROR",
    "tool_calls": "STOP",
}
# Flux rompu apres les premiers mots, et suite impossible.
FIN_INTERROMPUE = "INTERROMPUE"
FINS_INCOMPLETES = frozenset({"MAX_TOKENS", "ERROR", FIN_INTERROMPUE})

# Un 429 qui parle du mois est le quota mensuel, pas le debit.
_QUOTA_MENSUEL = re.compile(r"month|mensuel|quota", re.IGNORECASE)

# Mots-cles que le mode strict de Mistral ne documente pas : retires plutot
# que de risquer un refus de tout le schema.
_MOTS_CLES_RETIRES = frozenset({"minItems", "maxItems"})

SANTE_CACHE_S = 60.0


class MistralServiceError(Exception):
    """Base exception for Mistral service errors."""
    pass


class MistralQuotaError(MistralServiceError):
    """Quota MENSUEL du fournisseur epuise : inutile de reessayer avant le mois prochain."""
    pass


class MistralOverloadedError(MistralServiceError):
    """Saturation passagere (429 de debit, 5xx, attente trop longue) : reessayer."""
    pass


# ==================== LIMITEURS (UN PAR MODELE) ====================

_LIMITEURS: Dict[str, Limiteur] = {}
_VERROU_LIMITEURS = threading.Lock()


def limiteur_mistral(modele: str) -> Limiteur:
    """Le limiteur du modele, partage par tout le processus."""
    with _VERROU_LIMITEURS:
        limiteur = _LIMITEURS.get(modele)
        if limiteur is None:
            par_seconde = (
                settings.MISTRAL_REQUETES_PAR_SECONDE_SECOURS
                if modele == settings.MISTRAL_MODEL_SECOURS
                else settings.MISTRAL_REQUETES_PAR_SECONDE
            )
            limiteur = Limiteur(
                f"mistral:{modele}",
                requetes_par_minute=par_seconde * 60,
                concurrence=settings.MISTRAL_CONCURRENCE,
            )
            _LIMITEURS[modele] = limiteur
        return limiteur


def schema_strict(schema: Any) -> Any:
    """
    Le schema, tel que le mode strict l'exige : a chaque objet, tous les champs
    requis et aucun autre accepte. Sans `additionalProperties: false`, Mistral
    ignorait le schema, et la comparaison revenait sous d'autres cles.
    """
    if isinstance(schema, list):
        return [schema_strict(element) for element in schema]
    if not isinstance(schema, dict):
        return schema
    resultat = {
        cle: schema_strict(valeur)
        for cle, valeur in schema.items()
        if cle not in _MOTS_CLES_RETIRES
    }
    if resultat.get("type") == "object" and isinstance(resultat.get("properties"), dict):
        resultat["additionalProperties"] = False
        resultat["required"] = list(resultat["properties"])
    return resultat


def en_boucle(texte: str) -> bool:
    """
    Le texte tourne-t-il en rond ? Ses 80 derniers caracteres y figurent deja
    au moins trois fois. Demander une suite a un texte en boucle prolongerait
    la boucle.
    """
    if len(texte) < 400:
        return False
    fin = texte[-80:].strip()
    return bool(fin) and texte.count(fin) >= 3


class _SansEcho:
    """
    Retire l'echo du prefixe d'une suite, qu'il arrive ou non.

    Avec `"prefix": true`, la reponse peut recommencer par le texte deja
    produit. Tant que ce qui arrive en est le debut, rien n'est rendu ; des
    que ca diverge, ou que le prefixe entier est passe, tout le reste l'est.
    """

    def __init__(self, prefixe: str):
        self.prefixe = prefixe
        self.tampon = ""
        self.tranche = False

    def filtrer(self, morceau: str) -> str:
        if self.tranche:
            return morceau
        self.tampon += morceau
        if self.prefixe.startswith(self.tampon):
            return ""
        self.tranche = True
        if self.tampon.startswith(self.prefixe):
            return self.tampon[len(self.prefixe):]
        return self.tampon


def _message_d_erreur(contenu: bytes) -> str:
    texte = contenu.decode(errors="replace")
    try:
        corps = json.loads(texte)
    except ValueError:
        return texte[:300]
    if isinstance(corps, dict):
        return str(corps.get("message") or corps.get("detail") or corps)[:300]
    return str(corps)[:300]


class MistralService:
    """
    Client asynchrone pour l'API Mistral AI.
    """

    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 2048

    SYSTEM_INSTRUCTION = """You are an expert legal assistant specializing in Cameroonian law (JuriX).

CORE RULES:
1. Answer ONLY based on legal documents provided in the context
2. ALWAYS cite articles with exact references (e.g., "Article 5 of the Constitution", "Article 161 du Code Pénal")
3. If information is not in the context, say: "Je ne trouve pas cette information dans les documents disponibles."
4. Adapt language style to user persona (citizen=simple, lawyer=technical, student=educational)
5. Respond in French by default, or English if user writes in English

RESPONSE FORMAT:
- Be concise but complete
- Use clear paragraphs
- End with "Sources:" listing cited documents

FORBIDDEN:
- Never invent laws, articles, or legal interpretations
- No personalized legal advice
- No speculation on court decisions"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
        *,
        limiteur_pour: Callable[[str], Limiteur] = limiteur_mistral,
    ):
        self.api_key = api_key or settings.MISTRAL_API_KEY
        self.model_name = model_name or settings.MISTRAL_MODEL
        self.timeout = timeout or settings.MISTRAL_TIMEOUT_S
        # Injectable : les tests fournissent des limiteurs sur horloge factice.
        self._limiteur_pour = limiteur_pour
        self._sante: Optional[Tuple[float, Dict[str, Any]]] = None

        if not self.api_key:
            raise MistralServiceError(
                "MISTRAL_API_KEY non configurée. Renseignez-la dans .env"
            )

        logger.info(f"✅ MistralService initialisé: model={self.model_name}")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "JuriX-Backend/1.0",
        }

    def limiteur(self, modele: Optional[str] = None) -> Limiteur:
        return self._limiteur_pour(modele or self.model_name)

    def _secours(self, modele: str) -> Optional[str]:
        secours = settings.MISTRAL_MODEL_SECOURS
        return secours if secours and secours != modele else None

    def delai_hors_flux(self, max_tokens: int) -> float:
        """Une generation longue n'est pas une panne : le delai suit le budget."""
        return max(self.timeout, 20.0 + max_tokens / 40.0)

    # ------------------------------------------------------------- envoi

    def _attente_apres(self, code: int, contenu: bytes, entetes: httpx.Headers, essai: int) -> float:
        """
        L'attente avant de renvoyer, ou une exception si renvoyer ne sert a
        rien (quota mensuel, cle refusee, requete invalide).
        """
        message = _message_d_erreur(contenu)
        if code == 429:
            if _QUOTA_MENSUEL.search(message):
                raise MistralQuotaError(f"Quota mensuel Mistral epuise : {message}")
            attente = lire_duree(entetes.get("retry-after"))
            if attente is None:
                attente = 2.0 ** (essai - 1) + random.uniform(0, 0.5)
            return attente
        if code in (500, 502, 503, 504):
            return 2.0 ** (essai - 1) + random.uniform(0, 0.5)
        if code == 402:
            raise MistralQuotaError(f"Quota Mistral epuise (HTTP 402) : {message}")
        raise MistralServiceError(f"Erreur API Mistral (HTTP {code}): {message}")

    def _place(self, modele: str, debut: float):
        """La place du limiteur, dans ce qui reste du budget d'attente."""
        limiteur = self.limiteur(modele)
        restant = settings.MISTRAL_ATTENTE_MAX_S - (limiteur.maintenant() - debut)
        return limiteur.creneau(attente_max=max(0.0, restant))

    def _abandonner(self, limiteur: Limiteur, debut: float, essai: int, attente: float) -> bool:
        return essai == ESSAIS or (
            limiteur.maintenant() - debut + attente > settings.MISTRAL_ATTENTE_MAX_S
        )

    async def _envoyer(
        self,
        corps: Dict[str, Any],
        *,
        timeout: float,
        secours_permis: bool = True,
        debut: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        POST hors flux, avec les nouveaux essais. Rend le JSON de la reponse.

        `debut` : instant du premier essai. Le modele de secours le recoit,
        si bien que les 20 s d'attente sont un budget pour l'APPEL, pas par
        modele.
        """
        modele = corps["model"]
        limiteur = self.limiteur(modele)
        debut = limiteur.maintenant() if debut is None else debut
        cause = "saturation"
        async with httpx.AsyncClient(timeout=timeout, transport=transport_http) as client:
            for essai in range(1, ESSAIS + 1):
                try:
                    async with self._place(modele, debut):
                        reponse = await client.post(
                            MISTRAL_API_URL, headers=self._headers(), json=corps
                        )
                except AttenteTropLongue as e:
                    cause = f"attente {e.attente:.0f} s ({e.raison})"
                    break
                except httpx.TimeoutException as e:
                    # Une generation qui depasse son delai ne se relance pas :
                    # elle couterait le meme temps une seconde fois.
                    raise MistralOverloadedError(
                        f"Délai d'attente dépassé avec Mistral ({timeout:.0f} s)"
                    ) from e
                except httpx.TransportError as e:
                    attente, cause = 2.0 ** (essai - 1), f"reseau {e!r}"
                else:
                    if reponse.status_code == 200:
                        return reponse.json()
                    attente = self._attente_apres(
                        reponse.status_code, reponse.content, reponse.headers, essai
                    )
                    cause = f"HTTP {reponse.status_code}"
                # Le delai impose est inscrit AVANT de decider d'abandonner :
                # sinon un Retry-After plus long que le budget etait perdu, et
                # l'appel suivant repartait aussitot vers le modele sanctionne.
                limiteur.repousser(attente, cause)
                if self._abandonner(limiteur, debut, essai, attente):
                    break
                logger.warning(
                    "⚠️ Mistral %s : %s, nouvel essai dans %.1f s (%d/%d)",
                    modele, cause, attente, essai, ESSAIS,
                )

        secours = self._secours(modele) if secours_permis else None
        if secours:
            logger.warning("⚠️ Mistral %s sature (%s) : modele de secours %s", modele, cause, secours)
            return await self._envoyer(
                {**corps, "model": secours}, timeout=timeout, secours_permis=False, debut=debut
            )
        raise MistralOverloadedError(
            f"Le service Mistral est momentanément saturé ({cause}). Réessayez dans un instant."
        )

    async def _flux(
        self,
        corps: Dict[str, Any],
        *,
        secours_permis: bool = True,
        debut: Optional[float] = None,
    ) -> AsyncIterator[Tuple[str, Optional[str], Optional[Dict[str, Any]], str]]:
        """
        POST en flux. Rend des evenements (texte, fin brute, usage, modele).

        Les nouveaux essais n'ont lieu qu'AVANT le premier octet : une fois le
        flux ouvert, une rupture remonte a l'appelant, qui demande une suite.
        `debut` : comme pour `_envoyer`.
        """
        modele = corps["model"]
        limiteur = self.limiteur(modele)
        debut = limiteur.maintenant() if debut is None else debut
        cause = "saturation"
        # MISTRAL_TIMEOUT_S au plus entre deux morceaux, pas pour la reponse.
        delais = httpx.Timeout(connect=10.0, read=self.timeout, write=30.0, pool=10.0)
        async with httpx.AsyncClient(timeout=delais, transport=transport_http) as client:
            for essai in range(1, ESSAIS + 1):
                try:
                    async with self._place(modele, debut):
                        requete = client.build_request(
                            "POST", MISTRAL_API_URL, headers=self._headers(), json=corps
                        )
                        try:
                            reponse = await client.send(requete, stream=True)
                        except httpx.TimeoutException as e:
                            raise MistralOverloadedError(
                                f"Mistral n'a pas commencé à répondre en {self.timeout:.0f} s"
                            ) from e
                        except httpx.TransportError as e:
                            attente, cause = 2.0 ** (essai - 1), f"reseau {e!r}"
                        else:
                            try:
                                if reponse.status_code == 200:
                                    async for evenement in self._lire_flux(reponse, modele):
                                        yield evenement
                                    return
                                contenu = await reponse.aread()
                                attente = self._attente_apres(
                                    reponse.status_code, contenu, reponse.headers, essai
                                )
                                cause = f"HTTP {reponse.status_code}"
                            finally:
                                await reponse.aclose()
                except AttenteTropLongue as e:
                    cause = f"attente {e.attente:.0f} s ({e.raison})"
                    break
                limiteur.repousser(attente, cause)
                if self._abandonner(limiteur, debut, essai, attente):
                    break
                logger.warning(
                    "⚠️ Mistral %s (flux) : %s, nouvel essai dans %.1f s (%d/%d)",
                    modele, cause, attente, essai, ESSAIS,
                )

        secours = self._secours(modele) if secours_permis else None
        if secours:
            logger.warning("⚠️ Mistral %s sature (%s) : modele de secours %s", modele, cause, secours)
            async for evenement in self._flux(
                {**corps, "model": secours}, secours_permis=False, debut=debut
            ):
                yield evenement
            return
        raise MistralOverloadedError(
            f"Le service Mistral est momentanément saturé ({cause}). Réessayez dans un instant."
        )

    @staticmethod
    async def _lire_flux(reponse: httpx.Response, modele: str):
        async for ligne in reponse.aiter_lines():
            if not ligne or not ligne.startswith("data: "):
                continue
            donnees = ligne[6:].strip()
            if donnees == "[DONE]":
                return
            try:
                morceau = json.loads(donnees)
            except json.JSONDecodeError:
                continue
            choix = (morceau.get("choices") or [{}])[0]
            texte = (choix.get("delta") or {}).get("content") or ""
            yield (
                texte, choix.get("finish_reason"), morceau.get("usage"),
                morceau.get("model") or modele,
            )

    # ------------------------------------------------------------ corps

    def _corps(
        self,
        messages: List[Dict[str, Any]],
        *,
        temperature: float,
        max_tokens: int,
        flux: bool = False,
        format_de_reponse: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        corps: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if flux:
            corps["stream"] = True
        if format_de_reponse:
            corps["response_format"] = format_de_reponse
        return corps

    @staticmethod
    def _format_de_reponse(
        response_mime_type: Optional[str], response_schema: Optional[Dict]
    ) -> Optional[Dict[str, Any]]:
        if response_schema:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "reponse",
                    "schema": schema_strict(response_schema),
                    "strict": True,
                },
            }
        if response_mime_type == "application/json":
            return {"type": "json_object"}
        return None

    @staticmethod
    def _journal(modele, fin, usage, debut, suite=False) -> None:
        usage = usage or {}
        logger.info(
            "🧠 Mistral %s%s : fin=%s jetons=%s+%s duree=%.1fs",
            modele, " (suite)" if suite else "", fin,
            usage.get("prompt_tokens"), usage.get("completion_tokens"),
            time.monotonic() - debut,
        )

    # -------------------------------------------------------- generation

    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        response_mime_type: Optional[str] = None,
        response_schema: Optional[Dict] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Génère une réponse complète depuis Mistral.

        Rend {"response": texte, "fin": fin normalisee}, et "tronquee": True si
        la reponse reste coupee. Une reponse coupee en texte libre est
        completee une fois ; une reponse JSON ne l'est pas (un JSON recolle par
        une suite n'est pas fiable) : l'appelant decide.
        """
        messages = [
            {"role": "system", "content": system or self.SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ]
        format_de_reponse = self._format_de_reponse(response_mime_type, response_schema)
        debut = time.monotonic()
        try:
            donnees = await self._envoyer(
                self._corps(
                    messages, temperature=temperature, max_tokens=max_tokens,
                    format_de_reponse=format_de_reponse,
                ),
                timeout=self.delai_hors_flux(max_tokens),
            )
        except MistralServiceError:
            raise
        except Exception as e:
            logger.error(f"❌ Erreur inattendue Mistral generate: {e!r}")
            raise MistralServiceError(f"Erreur de génération Mistral: {e}") from e

        choix = donnees["choices"][0]
        texte = (choix.get("message") or {}).get("content") or ""
        fin = FINS.get(choix.get("finish_reason"), "UNKNOWN")
        self._journal(donnees.get("model") or self.model_name, fin, donnees.get("usage"), debut)

        if fin == "MAX_TOKENS" and format_de_reponse is None and texte and not en_boucle(texte):
            suite, fin = await self._suite(messages, texte, temperature)
            texte += suite

        resultat: Dict[str, Any] = {"response": texte, "fin": fin}
        if fin in FINS_INCOMPLETES:
            resultat["tronquee"] = True
        return resultat

    async def _suite(
        self, messages: List[Dict[str, Any]], texte: str, temperature: float
    ) -> Tuple[str, str]:
        """La suite d'une reponse coupee (hors flux), et la fin qui en resulte."""
        debut = time.monotonic()
        jetons = settings.MISTRAL_JETONS_SUITE
        try:
            donnees = await self._envoyer(
                self._corps(
                    [*messages, {"role": "assistant", "content": texte, "prefix": True}],
                    temperature=temperature, max_tokens=jetons,
                ),
                timeout=self.delai_hors_flux(jetons),
            )
        except MistralServiceError as e:
            logger.warning("⚠️ Suite de reponse impossible : %s", e)
            return "", "MAX_TOKENS"
        choix = donnees["choices"][0]
        contenu = (choix.get("message") or {}).get("content") or ""
        fin = FINS.get(choix.get("finish_reason"), "UNKNOWN")
        self._journal(
            donnees.get("model") or self.model_name, fin, donnees.get("usage"), debut, suite=True
        )
        return (contenu[len(texte):] if contenu.startswith(texte) else contenu), fin

    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        fin: Optional[Callable[[str], None]] = None,
        usage: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """
        Génère une réponse en streaming (SSE) depuis Mistral.

        `fin` recoit la fin normalisee : « STOP », « MAX_TOKENS » si la reponse
        reste coupee malgre la suite, FIN_INTERROMPUE si le flux s'est rompu
        et que la suite a echoue. `usage` recoit le decompte de jetons du
        premier flux et le modele qui a repondu (mesures, evaluation).
        """
        messages = [
            {"role": "system", "content": system or self.SYSTEM_INSTRUCTION},
            {"role": "user", "content": prompt},
        ]
        produit: List[str] = []
        brute: Optional[str] = None
        usage_flux = None
        modele = self.model_name
        rupture: Optional[Exception] = None
        debut = time.monotonic()

        try:
            async for texte, raison, morceau_usage, modele in self._flux(
                self._corps(messages, temperature=temperature, max_tokens=max_tokens, flux=True)
            ):
                if raison:
                    brute = raison
                if morceau_usage:
                    usage_flux = morceau_usage
                if texte:
                    produit.append(texte)
                    yield texte
        except MistralServiceError:
            raise
        except Exception as e:
            # Lecture rompue : delai entre deux morceaux, connexion coupee.
            if not produit:
                logger.error(f"❌ Erreur streaming Mistral: {e!r}")
                raise MistralServiceError(f"Erreur streaming Mistral: {e}") from e
            rupture = e

        fin_normalisee = FIN_INTERROMPUE if rupture else FINS.get(brute, "UNKNOWN")
        self._journal(
            modele, f"{FIN_INTERROMPUE} ({rupture!r})" if rupture else fin_normalisee,
            usage_flux, debut,
        )

        if usage is not None:
            usage({**(usage_flux or {}), "modele": modele})

        texte = "".join(produit)
        if fin_normalisee in ("MAX_TOKENS", FIN_INTERROMPUE) and texte and not en_boucle(texte):
            async for nouveau in self._suite_en_flux(messages, texte, temperature):
                if isinstance(nouveau, tuple):
                    fin_normalisee = nouveau[0]
                else:
                    yield nouveau

        if fin is not None:
            fin(fin_normalisee)

    async def _suite_en_flux(self, messages, texte: str, temperature: float):
        """
        Rend les morceaux NOUVEAUX de la suite, puis un tuple (fin,) si elle a
        abouti. Un echec n'est que journalise : la fin d'origine reste.
        """
        debut = time.monotonic()
        filtre = _SansEcho(texte)
        fin_suite, usage, modele = None, None, self.model_name
        try:
            async for morceau, raison, morceau_usage, modele in self._flux(
                self._corps(
                    [*messages, {"role": "assistant", "content": texte, "prefix": True}],
                    temperature=temperature, max_tokens=settings.MISTRAL_JETONS_SUITE,
                    flux=True,
                )
            ):
                if raison:
                    fin_suite = FINS.get(raison, "UNKNOWN")
                if morceau_usage:
                    usage = morceau_usage
                nouveau = filtre.filtrer(morceau) if morceau else ""
                if nouveau:
                    yield nouveau
        except Exception as e:
            logger.warning("⚠️ Suite de reponse impossible : %r", e)
            return
        self._journal(modele, fin_suite, usage, debut, suite=True)
        if fin_suite:
            yield (fin_suite,)

    # ------------------------------------------------------------- sante

    async def health_check(self) -> Dict[str, Any]:
        """
        La cle est-elle acceptee, et le modele servi ? Sans rien generer.

        `GET /v1/models` ne coute aucun jeton. Le resultat est garde
        SANTE_CACHE_S secondes : /rag/health peut etre interroge souvent.
        """
        maintenant = time.monotonic()
        if self._sante and maintenant - self._sante[0] < SANTE_CACHE_S:
            return self._sante[1]
        try:
            async with httpx.AsyncClient(timeout=5.0, transport=transport_http) as client:
                reponse = await client.get(MISTRAL_MODELES_URL, headers=self._headers())
            if reponse.status_code == 200:
                # Un alias (« -latest ») est servi sans figurer comme identifiant.
                modeles = set()
                for modele in reponse.json().get("data", []):
                    modeles.add(modele.get("id"))
                    modeles.update(modele.get("aliases") or [])
                resultat = {
                    "status": "healthy" if self.model_name in modeles else "degraded",
                    "model": self.model_name,
                }
                if self.model_name not in modeles:
                    resultat["error"] = "modele absent de /v1/models"
            else:
                resultat = {
                    "status": "unhealthy",
                    "model": self.model_name,
                    "error": f"HTTP {reponse.status_code} : {_message_d_erreur(reponse.content)}",
                }
        except httpx.HTTPError as e:
            resultat = {"status": "unhealthy", "error": repr(e), "model": self.model_name}
        self._sante = (maintenant, resultat)
        return resultat


@lru_cache(maxsize=1)
def get_mistral_service() -> MistralService:
    """Singleton cached instance de MistralService."""
    return MistralService()
