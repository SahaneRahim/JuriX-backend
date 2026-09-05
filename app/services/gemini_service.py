"""
GeminiService - Client for Google Gemini API (LLM for RAG).

Uses the new google-genai SDK (replaces deprecated google-generativeai).

Provides async methods for:
- generate(): Single response generation
- generate_stream(): Streaming response generation  
- health_check(): API connectivity check
"""

import asyncio
import functools
import logging
import re
from functools import lru_cache
from typing import AsyncIterator, Dict, Optional

from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)


# Budget de la sonde de sante. Doit laisser passer la phase de reflexion des
# modeles a raisonnement, sinon la sonde mesure son propre budget et non l'API.
# Budget de la sonde de sante. Doit laisser passer la phase de reflexion des
# modeles a raisonnement, sinon la sonde mesure son propre budget et non l'API.
HEALTH_CHECK_MAX_TOKENS = 512

# Nombre de tentatives sur une saturation passagere du fournisseur (503).
OVERLOAD_MAX_ATTEMPTS = 3
OVERLOAD_BASE_DELAY_S = 1.5


def _visible_text(response) -> str:
    """
    Texte destine a l'utilisateur, reflexion interne exclue.

    Les modeles a raisonnement renvoient leurs etapes de reflexion comme des
    `parts` portant `thought=True`. `response.text` les concatene avec la
    reponse : quand le budget de jetons s'epuise pendant la reflexion, ce qui
    remonte a l'ecran est un morceau de monologue interne. Observe en
    production sur une question de suivi : la reponse commencait par
    « *Self-Correction during drafting:* The user said "Et ce chiffre"… ».

    Se rabat sur `response.text` si la structure en parts est absente : mieux
    vaut une reponse un peu bavarde que pas de reponse.
    """
    try:
        parts = response.candidates[0].content.parts or []
    except (AttributeError, IndexError, TypeError):
        return response.text or ""

    visible = [p.text for p in parts if p.text and not getattr(p, "thought", False)]
    if visible:
        return "".join(visible)
    return response.text or ""


def _finish_reason(response) -> str:
    """Raison d'arret du modele, sous forme lisible, ou 'UNKNOWN'."""
    try:
        reason = response.candidates[0].finish_reason
    except (AttributeError, IndexError, TypeError):
        return "UNKNOWN"
    return getattr(reason, "name", str(reason)) if reason is not None else "UNKNOWN"


_RETRY_DELAY = re.compile(r"'retryDelay':\s*'(\d+)s'")


def _is_quota_exhausted(error: Exception) -> bool:
    """
    Vrai pour un depassement de quota (429), a distinguer d'une saturation.

    Le palier gratuit plafonne a 20 requetes de generation par jour et par
    modele (`GenerateRequestsPerDayPerProjectPerModel-FreeTier`). Ce n'est pas
    passager : le reessayer en boucle ne fait que bruler des appels.
    """
    if getattr(error, "code", None) == 429:
        return True
    text = str(error)
    return "429" in text and "RESOURCE_EXHAUSTED" in text


def retry_after_seconds(error: Exception, defaut: int = 60) -> int:
    """Delai conseille par le fournisseur, en secondes."""
    match = _RETRY_DELAY.search(str(error))
    return int(match.group(1)) if match else defaut


def _is_overloaded(error: Exception) -> bool:
    """
    Vrai pour une saturation passagere du fournisseur.

    Mesure sur le palier gratuit : « 503 UNAVAILABLE. This model is currently
    experiencing high demand. » Ce n'est ni une erreur de configuration ni un
    depassement de quota — cela se retente.
    """
    if getattr(error, "code", None) == 503:
        return True
    text = str(error)
    return "503" in text and ("UNAVAILABLE" in text or "high demand" in text)


class GeminiServiceError(Exception):
    """Base exception for Gemini service errors."""
    pass


class GeminiQuotaError(GeminiServiceError):
    """
    Quota du fournisseur epuise (429).

    Distincte de la saturation : une saturation se retente dans la seconde, un
    quota journalier ne se retente pas avant le lendemain. Les confondre
    faisait bruler des appels pour rien et affichait « erreur interne » la ou
    la cause est connue et explicable.
    """


class GeminiOverloadedError(GeminiServiceError):
    """
    Saturation passagere du fournisseur (503), apres epuisement des reprises.

    Sous-classe de GeminiServiceError pour que tout appelant existant continue
    de l'attraper, mais nommee a part : elle appelle un message « reessaie »,
    pas un message « le service est casse ».
    """

class GeminiService:
    """
    Client for Google Gemini API.
    
    Uses the new google-genai SDK for LLM operations.
    """
    
    # Default generation config
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 1000
    
    # System instruction for legal assistant
    SYSTEM_INSTRUCTION = """You are an expert legal assistant specializing in Cameroonian law (JuriX).

CORE RULES:
1. Answer ONLY based on legal documents provided in the context
2. ALWAYS cite articles with exact references (e.g., "Article 5 of the Constitution")
3. If information is not in the context, say: "Je ne trouve pas cette information dans les documents disponibles."
4. Adapt language style to user persona (citizen=simple, lawyer=technical, student=educational)
5. Respond in French by default, or English if user writes in English

RESPONSE FORMAT:
- Be concise but complete (max 300 words unless complex topic)
- Use clear paragraphs
- End with "Sources:" listing cited documents

FORBIDDEN:
- Never invent laws, articles, or legal interpretations
- No personalized legal advice
- No speculation on court decisions"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        """
        Initialize Gemini service.
        
        Args:
            api_key: Gemini API key (defaults to settings.GEMINI_API_KEY)
            model_name: Model to use (defaults to settings.GEMINI_MODEL)
        """
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.GEMINI_MODEL
        
        if not self.api_key:
            raise GeminiServiceError(
                "GEMINI_API_KEY not configured. Set it in .env"
            )
        
        # Initialize client with API key
        # Meme delai que le service d'embeddings : sans http_options, un appel
        # de generation peut bloquer la boucle d'evenements indefiniment.
        self.client = genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(timeout=settings.GEMINI_TIMEOUT_S * 1000),
        )
        
        logger.info(f"✅ GeminiService initialized: model={self.model_name}")
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        response_mime_type: Optional[str] = None,
        response_schema: Optional[Dict] = None,
        **kwargs
    ) -> Dict:
        """
        Generate a response from Gemini.

        Args:
            prompt: User prompt/question with context
            system: Optional system prompt override
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            response_mime_type: "application/json" pour une sortie structuree
            response_schema: schema JSON impose a la reponse

        Ces deux derniers parametres sont EXPLICITES et non laisses a **kwargs :
        tout ce qui tombait dans kwargs etait silencieusement ignore, si bien
        qu'un appelant demandant du JSON structure recevait de la prose sans
        aucun signal.

        Returns:
            Dict with 'response' key containing generated text
        """
        try:
            logger.debug(f"🤖 Generating response (temp={temperature}, max={max_tokens})")
            
            # Use provided system or default
            system_instruction = system or self.SYSTEM_INSTRUCTION
            
            # Create generation config
            config_kwargs = {
                "temperature": temperature,
                "max_output_tokens": max_tokens,
                "system_instruction": system_instruction,
            }
            if response_mime_type:
                config_kwargs["response_mime_type"] = response_mime_type
            if response_schema:
                config_kwargs["response_schema"] = response_schema

            config = types.GenerateContentConfig(**config_kwargs)
            
            # Deporte dans un thread : le client google-genai est SYNCHRONE.
            # Appele directement depuis cette coroutine, il gelait la boucle
            # d'evenements pendant tout l'aller-retour avec le modele — soit
            # plusieurs secondes a chaque question posee au RAG, pendant
            # lesquelles le serveur ne traitait plus aucune autre requete.
            # Reprise sur saturation passagere. Mesure sur le palier gratuit :
            # « 503 UNAVAILABLE. This model is currently experiencing high
            # demand. » revient par salves de quelques secondes. Sans cette
            # boucle, une salve remonte en 500 a l'utilisateur alors qu'un
            # second essai deux secondes plus tard aboutit.
            last_error: Optional[Exception] = None
            for attempt in range(1, OVERLOAD_MAX_ATTEMPTS + 1):
                try:
                    response = await asyncio.to_thread(
                        functools.partial(
                            self.client.models.generate_content,
                            model=self.model_name,
                            contents=prompt,
                            config=config,
                        )
                    )
                    break
                except Exception as err:
                    # Un 429 n'est pas une saturation : le retenter brule des
                    # appels sur un quota deja epuise.
                    if not _is_overloaded(err) or attempt == OVERLOAD_MAX_ATTEMPTS:
                        raise
                    last_error = err
                    delai = OVERLOAD_BASE_DELAY_S * (2 ** (attempt - 1))
                    logger.warning(
                        f"⚠️ Gemini sature (essai {attempt}/{OVERLOAD_MAX_ATTEMPTS}), "
                        f"nouvelle tentative dans {delai:.1f}s"
                    )
                    await asyncio.sleep(delai)

            texte = _visible_text(response)
            if texte.strip():
                logger.info(f"✅ Generated {len(texte)} chars")
                return {"response": texte}

            # Une reponse vide a une raison, et elle est exploitable. La jeter
            # pour un message generique privait l'appelant du seul indice
            # disponible — notamment MAX_TOKENS, qui se corrige en augmentant le
            # budget plutot qu'en accusant le modele.
            finish = _finish_reason(response)
            logger.warning(f"⚠️ Reponse vide de Gemini (finish_reason={finish})")
            if finish == "MAX_TOKENS":
                raise GeminiServiceError(
                    "Reponse tronquee : le budget de jetons a ete epuise avant "
                    "la premiere phrase."
                )
            raise GeminiServiceError(f"Reponse vide du modele (finish_reason={finish})")

        except GeminiServiceError:
            raise
        except Exception as e:
            if _is_quota_exhausted(e):
                delai = retry_after_seconds(e)
                logger.warning(f"⚠️ Quota Gemini epuise, reprise conseillee dans {delai}s")
                raise GeminiQuotaError(
                    f"Quota de generation epuise. Reessaie dans {delai} secondes."
                ) from e
            if _is_overloaded(e):
                logger.warning(f"⚠️ Gemini sature apres {OVERLOAD_MAX_ATTEMPTS} essais: {e}")
                raise GeminiOverloadedError(
                    "Le service de generation est momentanement sature. "
                    "Reessaie dans quelques instants."
                ) from e
            logger.error(f"❌ Gemini generation error: {e}")
            raise GeminiServiceError(f"Erreur de génération: {str(e)}") from e
    
    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **kwargs
    ) -> AsyncIterator[str]:
        """
        Stream a response from Gemini.
        
        Yields text chunks as they are generated.
        """
        try:
            logger.debug(f"🤖 Streaming response (temp={temperature}, max={max_tokens})")
            
            system_instruction = system or self.SYSTEM_INSTRUCTION
            
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction
            )
            
            # client.aio : la surface ASYNCHRONE native de google-genai.
            # La version synchrone bloquait la boucle d'evenements a chaque
            # morceau recu — sur une reponse de plusieurs secondes, le serveur
            # ne traitait plus aucune autre requete pendant tout le flux, ce qui
            # annule l'interet meme du streaming. Un deport par thread ne
            # convient pas ici : il faudrait un pont de file d'attente pour
            # reinjecter chaque morceau dans la boucle, la ou le client async
            # fait exactement cela nativement.
            response_stream = await self.client.aio.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config=config
            )

            produced = 0
            async for chunk in response_stream:
                if chunk.text:
                    produced += 1
                    yield chunk.text

            if produced == 0:
                # Arrive quand max_output_tokens est trop serre pour un modele
                # a raisonnement : le budget est consomme avant le premier
                # caractere de reponse. Le client recoit un flux vide, sans
                # erreur — sans cette trace, le silence est indechiffrable.
                logger.warning(
                    f"⚠️ Flux vide (max_tokens={max_tokens}) : budget "
                    f"probablement epuise par le raisonnement du modele"
                )

            logger.info(f"✅ Streaming complete ({produced} morceaux)")
            
        except Exception as e:
            logger.error(f"❌ Gemini streaming error: {e}")
            raise GeminiServiceError(f"Erreur de streaming: {str(e)}")
    
    async def health_check(self) -> Dict:
        """
        Check if Gemini API is accessible.
        
        Returns:
            Dict with 'status' key ('healthy' or 'unhealthy')
        """
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents="Reponds simplement: OK",
                # 10 jetons rendaient CETTE SONDE FAUSSE. Les modeles a
                # raisonnement consomment leur budget en reflexion interne avant
                # d'emettre le moindre caractere : `response.text` revenait vide
                # avec finish_reason=MAX_TOKENS, et la sonde concluait « en
                # panne » sur une API parfaitement joignable. Mesure sur trois
                # modeles (gemini-3-flash-preview, gemini-3.5-flash,
                # gemini-flash-latest) : les trois echouaient a 10 jetons, les
                # trois repondaient a 512.
                config=types.GenerateContentConfig(max_output_tokens=HEALTH_CHECK_MAX_TOKENS),
            )

            if response.text:
                return {"status": "healthy", "model": self.model_name}

            # Une reponse vide n'est pas la meme chose selon la raison. Budget
            # epuise = l'API a repondu, donc elle est joignable ; c'est le
            # budget de la sonde qui est en cause, pas le service.
            finish = _finish_reason(response)
            if finish == "MAX_TOKENS":
                return {
                    "status": "healthy",
                    "model": self.model_name,
                    "note": "budget de la sonde epuise en reflexion, API joignable",
                }
            return {
                "status": "unhealthy",
                "model": self.model_name,
                "reason": f"reponse vide (finish_reason={finish})",
            }

        except Exception as e:
            # Une saturation passagere n'est pas une panne du service. La
            # distinguer evite de declarer le produit hors service pour une
            # minute de charge chez le fournisseur.
            if _is_overloaded(e):
                logger.warning(f"⚠️ Gemini sature: {e}")
                return {
                    "status": "degraded",
                    "model": self.model_name,
                    "reason": "modele momentanement sature (503)",
                }
            logger.error(f"❌ Health check failed: {e}")
            return {"status": "unhealthy", "model": self.model_name, "reason": str(e)}


# Singleton instance
_gemini_service: Optional[GeminiService] = None


@lru_cache()
def get_gemini_service() -> GeminiService:
    """
    Singleton factory for GeminiService.
    """
    global _gemini_service
    if _gemini_service is None:
        _gemini_service = GeminiService()
    return _gemini_service


