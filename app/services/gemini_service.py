"""
GeminiService - Client for Google Gemini API (LLM for RAG).

Uses the new google-genai SDK (replaces deprecated google-generativeai).

Provides async methods for:
- generate(): Single response generation
- generate_stream(): Streaming response generation  
- health_check(): API connectivity check
"""

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
        self.client = genai.Client(api_key=self.api_key)
        
        logger.info(f"✅ GeminiService initialized: model={self.model_name}")
    
    async def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        **kwargs
    ) -> Dict:
        """
        Generate a response from Gemini.
        
        Args:
            prompt: User prompt/question with context
            system: Optional system prompt override
            temperature: Generation temperature (0.0-1.0)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Dict with 'response' key containing generated text
        """
        try:
            logger.debug(f"🤖 Generating response (temp={temperature}, max={max_tokens})")
            
            # Use provided system or default
            system_instruction = system or self.SYSTEM_INSTRUCTION
            
            # Create generation config
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                system_instruction=system_instruction
            )
            
            # Generate content (sync call wrapped for async compatibility)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
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
            
            # Stream content
            response_stream = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            
            for chunk in response_stream:
                if chunk.text:
                    yield chunk.text
                    
            logger.info("✅ Streaming complete")
            
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
