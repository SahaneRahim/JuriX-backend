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
from functools import lru_cache
from typing import AsyncIterator, Dict, Optional

from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)


class GeminiServiceError(Exception):
    """Base exception for Gemini service errors."""
    pass


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
            response = await asyncio.to_thread(
                functools.partial(
                    self.client.models.generate_content,
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
            )
            
            # Extract text from response
            if response.text:
                logger.info(f"✅ Generated {len(response.text)} chars")
                return {"response": response.text}
            else:
                logger.warning("⚠️ Empty response from Gemini")
                return {"response": "Je ne peux pas répondre à cette question."}
                
        except Exception as e:
            logger.error(f"❌ Gemini generation error: {e}")
            raise GeminiServiceError(f"Erreur de génération: {str(e)}")
    
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
            # Simple test generation
            response = self.client.models.generate_content(
                model=self.model_name,
                contents="Say 'OK' in one word.",
                config=types.GenerateContentConfig(max_output_tokens=10)
            )
            
            if response.text:
                return {"status": "healthy", "model": self.model_name}
            else:
                return {"status": "unhealthy", "reason": "Empty response"}
                
        except Exception as e:
            logger.error(f"❌ Health check failed: {e}")
            return {"status": "unhealthy", "reason": str(e)}


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


def clear_gemini_service_cache():
    """Clear the singleton instance (for testing)."""
    global _gemini_service
    _gemini_service = None
    get_gemini_service.cache_clear()
    logger.info("🗑️ GeminiService cache cleared")
