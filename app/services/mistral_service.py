"""
MistralService - Client for Mistral AI API (LLM for RAG Chat).

Replaces GeminiService using Mistral's OpenAI-compatible REST API via httpx.AsyncClient.
Provides async methods for:
- generate(): Single response generation
- generate_stream(): Streaming response generation (SSE)
- health_check(): API connectivity check
"""

import asyncio
import json
import logging
from functools import lru_cache
from typing import Any, AsyncIterator, Callable, Dict, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

MISTRAL_API_URL = "https://api.mistral.ai/v1/chat/completions"
OVERLOAD_MAX_ATTEMPTS = 3
OVERLOAD_BASE_DELAY_S = 1.5


class MistralServiceError(Exception):
    """Base exception for Mistral service errors."""
    pass


class MistralQuotaError(MistralServiceError):
    """Quota ou Rate limit du fournisseur épuisé (429)."""
    pass


class MistralOverloadedError(MistralServiceError):
    """Saturation passagère du fournisseur (503 / 502)."""
    pass


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
    ):
        self.api_key = api_key or settings.MISTRAL_API_KEY
        self.model_name = model_name or settings.MISTRAL_MODEL
        self.timeout = timeout or settings.MISTRAL_TIMEOUT_S

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
        """Génère une réponse complète depuis Mistral."""
        system_instruction = system or self.SYSTEM_INSTRUCTION
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if response_mime_type == "application/json" or response_schema:
            payload["response_format"] = {"type": "json_object"}

        for attempt in range(1, OVERLOAD_MAX_ATTEMPTS + 1):
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    resp = await client.post(
                        MISTRAL_API_URL, headers=self._headers(), json=payload
                    )

                if resp.status_code == 200:
                    data = resp.json()
                    choice = data["choices"][0]
                    content = choice["message"]["content"] or ""
                    finish_reason = choice.get("finish_reason")
                    res: Dict[str, Any] = {"response": content}
                    if finish_reason == "length":
                        res["tronquee"] = True
                    return res

                if resp.status_code == 429:
                    raise MistralQuotaError(
                        "Quota ou débit Mistral dépassé (429). Réessayez dans un instant."
                    )

                if resp.status_code in (502, 503, 504):
                    if attempt == OVERLOAD_MAX_ATTEMPTS:
                        raise MistralOverloadedError(
                            "Le service Mistral est momentanément saturé (503)."
                        )
                    delai = OVERLOAD_BASE_DELAY_S * (2 ** (attempt - 1))
                    logger.warning(
                        f"⚠️ Mistral saturé (essai {attempt}/{OVERLOAD_MAX_ATTEMPTS}), retry dans {delai:.1f}s"
                    )
                    await asyncio.sleep(delai)
                    continue

                error_detail = resp.text
                try:
                    error_json = resp.json()
                    error_detail = error_json.get("message") or error_detail
                except Exception:
                    pass
                raise MistralServiceError(
                    f"Erreur API Mistral (HTTP {resp.status_code}): {error_detail}"
                )

            except (MistralQuotaError, MistralOverloadedError, MistralServiceError):
                raise
            except httpx.TimeoutException as e:
                if attempt == OVERLOAD_MAX_ATTEMPTS:
                    raise MistralServiceError(f"Délai d'attente dépassé avec Mistral: {e}") from e
                await asyncio.sleep(OVERLOAD_BASE_DELAY_S)
            except Exception as e:
                logger.error(f"❌ Erreur inattendue Mistral generate: {e}")
                raise MistralServiceError(f"Erreur de génération Mistral: {e}") from e

        raise MistralServiceError("Échec des tentatives de génération Mistral.")

    async def generate_stream(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        fin: Optional[Callable[[str], None]] = None,
        **kwargs,
    ) -> AsyncIterator[str]:
        """Génère une réponse en streaming (SSE) depuis Mistral."""
        system_instruction = system or self.SYSTEM_INSTRUCTION
        messages = [
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": prompt},
        ]

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        dernier_finish_reason = "UNKNOWN"
        produced = 0

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                async with client.stream(
                    "POST", MISTRAL_API_URL, headers=self._headers(), json=payload
                ) as response:
                    if response.status_code == 429:
                        raise MistralQuotaError("Quota de génération Mistral dépassé (429).")
                    if response.status_code in (502, 503, 504):
                        raise MistralOverloadedError("Service Mistral momentanément saturé (503).")
                    if response.status_code != 200:
                        error_body = await response.aread()
                        raise MistralServiceError(
                            f"Erreur streaming Mistral (HTTP {response.status_code}): {error_body.decode(errors='replace')}"
                        )

                    async for line in response.aiter_lines():
                        if not line or not line.startswith("data: "):
                            continue
                        raw_data = line[6:].strip()
                        if raw_data == "[DONE]":
                            break
                        try:
                            chunk_json = json.loads(raw_data)
                            choices = chunk_json.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            content = delta.get("content")
                            reason = choices[0].get("finish_reason")
                            if reason:
                                dernier_finish_reason = reason
                            if content:
                                produced += 1
                                yield content
                        except json.JSONDecodeError:
                            continue

            if fin is not None:
                fin(dernier_finish_reason)

            logger.info(f"✅ Streaming Mistral terminé ({produced} morceaux reçus)")

        except (MistralQuotaError, MistralOverloadedError, MistralServiceError):
            raise
        except Exception as e:
            logger.error(f"❌ Erreur streaming Mistral: {e}")
            raise MistralServiceError(f"Erreur streaming Mistral: {e}") from e

    async def health_check(self) -> Dict[str, Any]:
        """Sonde de santé de l'API Mistral."""
        try:
            res = await self.generate("Reponds: OK", max_tokens=10)
            return {
                "status": "healthy" if res.get("response") else "degraded",
                "model": self.model_name,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "model": self.model_name}


@lru_cache(maxsize=1)
def get_mistral_service() -> MistralService:
    """Singleton cached instance de MistralService."""
    return MistralService()
