"""
GroqService - Client ultra-rapide pour Groq API (Routage et Décisions Système 1).

Optimisé pour les décisions instantanées (< 100 ms) avec format JSON strict garanti.
Utilisé prioritairement par app/services/intent_classifier.py.
"""

import asyncio
import json
import logging
from functools import lru_cache
from typing import Any, Dict, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqServiceError(Exception):
    """Base exception for Groq service errors."""
    pass


class GroqService:
    """
    Client asynchrone pour l'API Groq (LPU d'inférence ultra-rapide).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.api_key = api_key or settings.GROQ_API_KEY
        self.model_name = model_name or settings.GROQ_MODEL
        self.timeout = timeout or settings.GROQ_TIMEOUT_S

        if not self.api_key:
            raise GroqServiceError(
                "GROQ_API_KEY non configurée. Renseignez-la dans .env"
            )

        logger.info(f"✅ GroqService initialisé: model={self.model_name}")

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "JuriX-Backend/1.0",
        }

    async def classify_intent_json(
        self,
        prompt: str,
        system: str,
        timeout: Optional[float] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Envoie un prompt de classification avec contrainte JSON stricte.
        Renvoie le dictionnaire JSON parsé ou None en cas d'erreur.
        """
        actual_timeout = timeout or self.timeout
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 120,
            "response_format": {"type": "json_object"},
        }

        try:
            async with httpx.AsyncClient(timeout=actual_timeout) as client:
                resp = await client.post(
                    GROQ_API_URL, headers=self._headers(), json=payload
                )

            if resp.status_code == 200:
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                if content:
                    return json.loads(content)
                return None

            logger.warning(
                f"🧭 Groq classification error HTTP {resp.status_code}: {resp.text}"
            )
            return None

        except Exception as e:
            logger.warning(f"🧭 Exception lors de l'appel Groq: {e}")
            return None

    def _build_domain_payload(
        self,
        title: str,
        text: str = "",
        doc_type: Optional[str] = None,
        canonical_domains: Optional[list] = None,
    ) -> Dict[str, Any]:
        from app.services.legal_domain_classifier import CANONICAL_DOMAINS

        domains_list = canonical_domains or list(CANONICAL_DOMAINS)
        domains_formatted = "\n".join(f"- {d}" for d in domains_list)

        system_prompt = (
            "Tu es un haut magistrat et expert juriste du droit camerounais.\n"
            "Ta mission est de classifier avec rigueur chaque texte juridique dans la catégorie officielle la plus exacte.\n\n"
            f"Domaines canoniques officiels :\n{domains_formatted}\n\n"
            "Consignes strictes :\n"
            "- \"categorie\" DOIT être exactement l'un des domaines canoniques de la liste ci-dessus.\n"
            "- \"categories_secondaires\" : liste (max 2) d'autres domaines pertinents parmi la liste, ou [].\n"
            "- \"confiance\" : nombre décimal entre 0.50 et 1.00.\n"
            "- \"justification\" : une phrase synthétique expliquant la qualification juridique retenue.\n"
            "Réponds UNIQUEMENT en JSON sous ce format :\n"
            "{\n"
            '  "categorie": "nom_exact",\n'
            '  "confiance": 0.95,\n'
            '  "categories_secondaires": ["nom_exact"],\n'
            '  "justification": "..."\n'
            "}"
        )

        user_content = f"Titre : {title.strip()}"
        if doc_type:
            user_content += f"\nType d'acte : {doc_type}"
        if text:
            clean_text = " ".join(text[:1500].split())
            user_content += f"\nExtrait du texte : {clean_text}"

        return {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            "temperature": 0.0,
            "max_tokens": 250,
            "response_format": {"type": "json_object"},
        }

    async def classify_legal_domain_async(
        self,
        title: str,
        text: str = "",
        doc_type: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """Classification asynchrone d'un texte juridique via Groq/Qwen avec retry sur 429."""
        payload = self._build_domain_payload(title, text, doc_type)
        actual_timeout = timeout or self.timeout

        try:
            async with httpx.AsyncClient(timeout=actual_timeout) as client:
                for attempt in range(max_retries):
                    resp = await client.post(
                        GROQ_API_URL, headers=self._headers(), json=payload
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        content = data["choices"][0]["message"]["content"]
                        if content:
                            return json.loads(content)
                        return None
                    elif resp.status_code == 429:
                        wait_header = resp.headers.get("retry-after")
                        wait_time = float(wait_header) if wait_header else (2.0 * (attempt + 1))
                        wait_time = max(wait_time, 2.0)
                        logger.info(
                            f"🧭 Groq Rate limit (429) - attente de {wait_time:.1f}s (tentative {attempt+1}/{max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                    else:
                        logger.warning(f"🧭 Groq HTTP {resp.status_code}: {resp.text}")
                        return None
            return None
        except Exception as e:
            logger.warning(f"🧭 Erreur classification Groq: {e}")
            return None

    def classify_legal_domain_sync(
        self,
        title: str,
        text: str = "",
        doc_type: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: int = 5,
    ) -> Optional[Dict[str, Any]]:
        """Classification synchrone d'un texte juridique via Groq/Qwen avec retry sur 429."""
        import time

        payload = self._build_domain_payload(title, text, doc_type)
        actual_timeout = timeout or self.timeout

        try:
            with httpx.Client(timeout=actual_timeout) as client:
                for attempt in range(max_retries):
                    resp = client.post(
                        GROQ_API_URL, headers=self._headers(), json=payload
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        content = data["choices"][0]["message"]["content"]
                        if content:
                            return json.loads(content)
                        return None
                    elif resp.status_code == 429:
                        wait_header = resp.headers.get("retry-after")
                        wait_time = float(wait_header) if wait_header else (2.0 * (attempt + 1))
                        wait_time = max(wait_time, 2.0)
                        logger.info(
                            f"🧭 Groq Rate limit (429) sync - attente de {wait_time:.1f}s (tentative {attempt+1}/{max_retries})"
                        )
                        time.sleep(wait_time)
                    else:
                        logger.warning(f"🧭 Groq HTTP {resp.status_code}: {resp.text}")
                        return None
            return None
        except Exception as e:
            logger.warning(f"🧭 Erreur classification Groq synchrone: {e}")
            return None

    async def health_check(self) -> Dict[str, Any]:
        """Sonde de santé Groq."""
        try:
            res = await self.classify_intent_json(
                prompt="bonjour",
                system="Reponds uniquement en JSON: {\"status\": \"ok\"}",
                timeout=3.0,
            )
            return {
                "status": "healthy" if res else "degraded",
                "model": self.model_name,
            }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e), "model": self.model_name}


@lru_cache(maxsize=1)
def get_groq_service() -> GroqService:
    """Singleton cached instance de GroqService."""
    return GroqService()
