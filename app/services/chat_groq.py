"""
Redaction des reponses du chat : Groq d'abord, puis le fournisseur habituel.

POURQUOI. Mesure du 07/11/2026, memes extraits pour tous les modeles, juge a
l'aveugle mistral-large-4 : sur 24 questions en francais et 12 en anglais,
qwen3.8-27b (Groq) obtient 4,25 et 5,00 de justesse sur 5, ministral-14b
(Mistral, modele du chat jusque-la) 2,44 et 3,00. ministral-14b niait la SARL
unipersonnelle en citant l'article 309 de l'AUSCGIE, qui l'autorise en toutes
lettres, et envoyait les salaries devant le conseil de prud'hommes, juridiction
francaise.

LE PALIER GRATUIT DE GROQ est etroit : 8 000 jetons par minute (7 000 en
entree pour qwen), 1 000 requetes et environ 200 000 jetons par jour, PAR
MODELE. Un prompt du chat fait de 2 800 a 7 300 jetons. D'ou la cascade :
- un prompt qui ne tient pas dans la minute n'est pas envoye (il reviendrait
  en 413) ;
- une attente de plus de CHAT_GROQ_ATTENTE_MAX_S, un 429, un quota epuise ou
  une panne font passer au modele suivant, sans nouvel essai ;
- en bout de chaine, le service de LLM_PROVIDER, avec ses propres essais et
  son modele de secours.
Une fois le premier morceau lu, la reponse reste celle de ce modele : une
rupture donne une reponse interrompue, que le chat signale.

Seuls `generate` et `generate_stream` passent par la cascade. Tout le reste
(sante, limiteur, jetons d'entree...) est celui du service de repli.

Author: JuriX Team
"""

import json
import logging
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Tuple

import httpx

from app.core.config import settings
from app.core.limiteur import AttenteTropLongue, lire_duree
from app.services import groq_service
from app.services.groq_service import (
    _QUOTA_DU_JOUR,
    GROQ_API_URL,
    _entier,
    _message_d_erreur,
    limiteur_groq,
)
from app.services.mistral_service import FIN_INTERROMPUE, FINS

logger = logging.getLogger(__name__)


class GroqIndisponiblePourLeChat(Exception):
    """Ce modele ne peut pas rediger cette reponse maintenant : passer au suivant."""


def modeles_du_chat() -> List[str]:
    return [m.strip() for m in (settings.CHAT_GROQ_MODELES or "").split(",") if m.strip()]


def parametres_de_reflexion(modele: str) -> Dict[str, Any]:
    """
    La reflexion reste ACTIVE (c'est avec elle que la mesure a ete faite),
    mais hors de la reponse : le lecteur ne doit voir que le texte final.
    """
    if modele.startswith("openai/gpt-oss"):
        return {"include_reasoning": False}
    return {"reasoning_format": "hidden"}


def jetons_du_prompt(texte: str) -> int:
    """
    Jetons d'un prompt du chat, a 3,5 caracteres par jeton. Mesure du
    07/11/2026 sur les prompts reels : de 3,2 a 4,2 en francais, de 3,4 a 3,9
    en anglais. `estimer_jetons` (3 caracteres) ne sous-estime jamais, mais il
    ecartait de Groq des prompts qui y tenaient. Une sous-estimation coute
    peu : Groq repond 413 tout de suite, et la cascade passe au suivant.
    """
    return len(texte) * 2 // 7 + 1


def tient_dans_la_minute(jetons_entree: int, max_jetons: int) -> bool:
    """Un prompt plus gros que le plafond par minute reviendrait en 413."""
    plafond_entree = settings.GROQ_JETONS_ENTREE_PAR_MINUTE or None
    plafond_total = settings.GROQ_JETONS_PAR_MINUTE or None
    if plafond_entree is not None and jetons_entree > plafond_entree:
        return False
    if plafond_total is not None and jetons_entree + max_jetons > plafond_total:
        return False
    return True


class RedactionGroq:
    """Une reponse du chat par UN modele Groq, en flux ou d'un bloc."""

    def __init__(self, modele: str, *, api_key: Optional[str] = None, limiteur_pour=limiteur_groq):
        self.modele = modele
        self.api_key = api_key or settings.GROQ_API_KEY
        self._limiteur_pour = limiteur_pour

    def _corps(self, prompt: str, systeme: str, temperature: float, flux: bool) -> Dict[str, Any]:
        corps = {
            "model": self.modele,
            "messages": [
                {"role": "system", "content": systeme},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_completion_tokens": settings.CHAT_GROQ_MAX_JETONS,
            **parametres_de_reflexion(self.modele),
        }
        if flux:
            corps["stream"] = True
        return corps

    def _reserver(self, prompt: str, systeme: str):
        entree = jetons_du_prompt(systeme + prompt)
        max_jetons = settings.CHAT_GROQ_MAX_JETONS
        if not tient_dans_la_minute(entree, max_jetons):
            raise GroqIndisponiblePourLeChat(
                f"prompt trop long pour {self.modele} ({entree} jetons estimes)"
            )
        limiteur = self._limiteur_pour(self.modele)
        try:
            reservation = limiteur.reserver(
                entree + max_jetons, settings.CHAT_GROQ_ATTENTE_MAX_S, jetons_entree=entree
            )
        except AttenteTropLongue as e:
            raise GroqIndisponiblePourLeChat(f"{self.modele} : {e}") from e
        return limiteur, reservation

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "JuriX-Backend/1.0",
        }

    @staticmethod
    def _recaler(limiteur, entetes: httpx.Headers) -> None:
        limiteur.recaler(
            requetes_restantes_jour=_entier(entetes.get("x-ratelimit-remaining-requests")),
            reprise_requetes_s=lire_duree(entetes.get("x-ratelimit-reset-requests")),
            jetons_restants_minute=_entier(entetes.get("x-ratelimit-remaining-tokens")),
        )

    def _refus(self, limiteur, code: int, message: str, entetes: httpx.Headers) -> GroqIndisponiblePourLeChat:
        """Un refus de Groq : le limiteur en tient compte, puis on passe au suivant."""
        if code == 429:
            attente = lire_duree(entetes.get("retry-after")) or 10.0
            quota = bool(_QUOTA_DU_JOUR.search(message))
            limiteur.repousser(attente, "quota Groq" if quota else "429")
        elif code >= 500:
            limiteur.repousser(2.0, f"HTTP {code}")
        return GroqIndisponiblePourLeChat(f"{self.modele} HTTP {code} : {message[:200]}")

    def _delais(self) -> httpx.Timeout:
        return httpx.Timeout(connect=10.0, read=settings.CHAT_GROQ_TIMEOUT_S, write=30.0, pool=10.0)

    async def generer(self, prompt: str, systeme: str, temperature: float) -> Dict[str, Any]:
        limiteur, reservation = self._reserver(prompt, systeme)
        if reservation.attente > 0:
            await limiteur._dormir_async(reservation.attente)
        debut = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._delais(), transport=groq_service.transport_http) as client:
                reponse = await client.post(
                    GROQ_API_URL, headers=self._headers(),
                    json=self._corps(prompt, systeme, temperature, flux=False),
                )
        except httpx.HTTPError as e:
            limiteur.repousser(2.0, "reseau")
            raise GroqIndisponiblePourLeChat(f"{self.modele} injoignable : {e!r}") from e
        self._recaler(limiteur, reponse.headers)
        if reponse.status_code != 200:
            raise self._refus(limiteur, reponse.status_code, _message_d_erreur(reponse), reponse.headers)
        donnees = reponse.json()
        usage = donnees.get("usage") or {}
        if usage.get("total_tokens") is not None:
            reservation.corriger(usage["total_tokens"], usage.get("prompt_tokens"))
        choix = (donnees.get("choices") or [{}])[0]
        texte = ((choix.get("message") or {}).get("content") or "").strip()
        if not texte:
            raise GroqIndisponiblePourLeChat(f"{self.modele} : reponse vide")
        fin = FINS.get(choix.get("finish_reason"), "UNKNOWN")
        logger.info(
            "🧭 Chat Groq %s : fin=%s jetons=%s duree=%.2fs",
            self.modele, fin, usage.get("total_tokens"), time.monotonic() - debut,
        )
        resultat: Dict[str, Any] = {"response": texte, "fin": fin, "modele": self.modele}
        if fin != "STOP":
            resultat["tronquee"] = True
        return resultat

    async def flux(self, prompt: str, systeme: str, temperature: float) -> AsyncIterator[Tuple[str, Optional[str], Optional[Dict[str, Any]]]]:
        """
        Rend (texte, fin brute, usage). Avant le premier morceau, tout echec
        leve GroqIndisponiblePourLeChat ; apres, l'exception d'origine.
        """
        limiteur, reservation = self._reserver(prompt, systeme)
        if reservation.attente > 0:
            await limiteur._dormir_async(reservation.attente)
        async with httpx.AsyncClient(timeout=self._delais(), transport=groq_service.transport_http) as client:
            requete = client.build_request(
                "POST", GROQ_API_URL, headers=self._headers(),
                json=self._corps(prompt, systeme, temperature, flux=True),
            )
            try:
                reponse = await client.send(requete, stream=True)
            except httpx.HTTPError as e:
                limiteur.repousser(2.0, "reseau")
                raise GroqIndisponiblePourLeChat(f"{self.modele} injoignable : {e!r}") from e
            try:
                self._recaler(limiteur, reponse.headers)
                if reponse.status_code != 200:
                    contenu = await reponse.aread()
                    try:
                        message = str((json.loads(contenu).get("error") or {}).get("message") or contenu[:200])
                    except (ValueError, AttributeError):
                        message = contenu[:200].decode("utf-8", "replace")
                    raise self._refus(limiteur, reponse.status_code, message, reponse.headers)
                async for ligne in reponse.aiter_lines():
                    if not ligne.startswith("data: "):
                        continue
                    donnees = ligne[6:].strip()
                    if donnees == "[DONE]":
                        return
                    try:
                        morceau = json.loads(donnees)
                    except json.JSONDecodeError:
                        continue
                    usage = (morceau.get("x_groq") or {}).get("usage") or morceau.get("usage")
                    if usage and usage.get("total_tokens") is not None:
                        reservation.corriger(usage["total_tokens"], usage.get("prompt_tokens"))
                    choix = (morceau.get("choices") or [{}])[0]
                    texte = (choix.get("delta") or {}).get("content") or ""
                    yield texte, choix.get("finish_reason"), usage
            finally:
                await reponse.aclose()


class ChatEnCascade:
    """
    Le service de generation du chat : les modeles Groq de CHAT_GROQ_MODELES
    dans l'ordre, puis `repli` (le service de LLM_PROVIDER).
    """

    def __init__(self, repli: Any, modeles: Optional[List[str]] = None, *, redaction_pour: Callable[[str], RedactionGroq] = RedactionGroq):
        self.repli = repli
        self.modeles = modeles_du_chat() if modeles is None else modeles
        self._redaction_pour = redaction_pour

    def __getattr__(self, nom: str) -> Any:
        return getattr(self.repli, nom)

    async def generate(self, prompt: str, system: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 2048, **kwargs) -> Dict[str, Any]:
        if system and not kwargs.get("response_schema") and not kwargs.get("response_mime_type"):
            for modele in self.modeles:
                try:
                    return await self._redaction_pour(modele).generer(prompt, system, temperature)
                except GroqIndisponiblePourLeChat as e:
                    logger.warning("⚠️ Chat : %s, modele suivant", e)
        return await self.repli.generate(
            prompt=prompt, system=system, temperature=temperature, max_tokens=max_tokens, **kwargs
        )

    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        fin: Optional[Callable[[str], None]] = None,
        usage: Optional[Callable[[Dict[str, Any]], None]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        for modele in (self.modeles if system else []):
            produit = False
            brute: Optional[str] = None
            usage_flux: Optional[Dict[str, Any]] = None
            rupture: Optional[Exception] = None
            try:
                async for texte, raison, morceau_usage in self._redaction_pour(modele).flux(prompt, system, temperature):
                    if raison:
                        brute = raison
                    if morceau_usage:
                        usage_flux = morceau_usage
                    if texte:
                        produit = True
                        yield texte
            except GroqIndisponiblePourLeChat as e:
                if not produit:
                    logger.warning("⚠️ Chat : %s, modele suivant", e)
                    continue
                rupture = e
            except (httpx.HTTPError, ValueError) as e:
                if not produit:
                    logger.warning("⚠️ Chat : %s rompu avant le premier morceau (%r), modele suivant", modele, e)
                    continue
                rupture = e
            if not produit:
                logger.warning("⚠️ Chat : %s n'a rien rendu, modele suivant", modele)
                continue
            fin_normalisee = FIN_INTERROMPUE if rupture else FINS.get(brute, FIN_INTERROMPUE)
            logger.info("🧭 Chat Groq %s (flux) : fin=%s", modele, fin_normalisee)
            if usage is not None:
                usage({**(usage_flux or {}), "modele": modele})
            if fin is not None:
                fin(fin_normalisee)
            return
        async for morceau in self.repli.generate_stream(
            prompt=prompt, system=system, temperature=temperature, max_tokens=max_tokens,
            fin=fin, usage=usage, **kwargs,
        ):
            yield morceau


def service_du_chat(repli: Any) -> Any:
    """
    `repli` enveloppe dans la cascade Groq, si elle est configuree. Un double
    de test (ni Mistral ni Gemini) est rendu tel quel : les tests du chat ne
    doivent jamais appeler Groq.
    """
    from app.services.gemini_service import GeminiService
    from app.services.mistral_service import MistralService

    if repli is None or not settings.GROQ_API_KEY or not modeles_du_chat():
        return repli
    if not isinstance(repli, (MistralService, GeminiService)):
        return repli
    return ChatEnCascade(repli)
