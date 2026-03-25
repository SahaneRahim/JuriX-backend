"""
Service LlamaParse pour extraction de texte de haute qualité depuis PDFs juridiques.

Ce service utilise LlamaParse (LlamaIndex Cloud) pour:
- Extraction propre sans tampons/signatures
- Compréhension de la structure des articles
- Nettoyage automatique des artefacts OCR
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Instructions personnalisées pour parser les documents juridiques camerounais
LEGAL_DOCUMENT_PARSING_INSTRUCTIONS = """
You are parsing official Cameroonian legal documents (laws, decrees, ordinances, constitutional texts).

## WHAT TO EXTRACT (IMPORTANT):
- All articles with their numbers (Article 1, Article 2, Article premier, etc.)
- All article content/text in full
- Section headers (LIVRE, CHAPITRE, SECTION, TITRE)
- Preamble text if present
- Any numbered or lettered subsections within articles

## WHAT TO COMPLETELY IGNORE AND EXCLUDE:

### 1. OFFICIAL STAMPS (Tampons officiels):
- Rectangular stamps with "PRESIDENCE DE LA REPUBLIQUE" / "PRESIDENCY OF THE REPUBLIC"
- Stamps containing "SECRETARIAT GENERAL"
- "SERVICE DU FICHIER LEGISLATIF ET REGLEMENTAIRE" stamps
- "LEGISLATIVE AND STATUTORY AFFAIRS CARD INDEX SERVICE" stamps
- "COPIE CERTIFIEE CONFORME" / "CERTIFIED TRUE COPY" stamps (any color: black, red/pink, blue)
- Any rectangular border stamps with official text

### 2. OFFICIAL SEALS (Sceaux ronds):
- Round seals with "REPUBLIQUE DU CAMEROUN" around the edge
- Seals with "LE PRESIDENT" or "THE PRESIDENT" text
- Seals with stars and national emblems
- Seals containing "PAIX TRAVAIL PATRIE" or "PEACE WORK FATHERLAND"
- Any circular official emblems

### 3. SIGNATURES:
- Handwritten signatures (cursive scribbles)
- The signature of "Paul BIYA" or any president's signature
- Paraphs and initials
- Any handwritten marks or annotations

### 4. DATE STAMPS:
- "Yaoundé, le [DATE]" location/date headers
- Date stamps like "25 FEV 2021", "12 JUIL 2016"
- Any administrative date markings

### 5. HEADERS AND FOOTERS:
- Page numbers
- Running headers/footers
- "PAIX - TRAVAIL - PATRIE" motto

### 6. SCAN ARTIFACTS:
- Blurry or partial text from stamps
- OCR garbage characters (e.g., "ET 2£ SLE", "Ei 5", "SS---")
- Text that appears faded, partial, or illegible
- Random characters from poor scanning

## OUTPUT FORMAT:
- Preserve the legal structure exactly
- Use clear article separation
- Keep French language as-is (do not translate)
- Format section headers in UPPERCASE
- Each article should start with "Article X" or "ARTICLE X" on its own line
- Paragraph text should follow directly

## EXAMPLE OUTPUT:
CHAPITRE I
DISPOSITIONS GÉNÉRALES

Article 1
La présente loi régit les conditions d'exercice de la profession...

Article 2
Au sens de la présente loi, on entend par:
a) Professionnel: toute personne physique ou morale...
b) Activité: tout acte exercé...
"""


class LlamaParseError(Exception):
    """Exception levée lors d'erreurs LlamaParse."""
    pass


class LlamaParseService:
    """
    Service d'extraction de texte de haute qualité via LlamaParse.
    
    LlamaParse utilise des LLMs pour comprendre et extraire le contenu
    des documents de manière intelligente, ignorant les éléments
    non pertinents comme les tampons et signatures.
    """

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialise le service LlamaParse.
        
        Args:
            api_key: Clé API LlamaCloud. Si non fournie, utilise
                     settings.LLAMA_CLOUD_API_KEY puis LLAMA_CLOUD_API_KEY env var.
        """
        if api_key:
            self.api_key = api_key
        else:
            try:
                from app.core.config import settings
                self.api_key = settings.LLAMA_CLOUD_API_KEY or os.getenv("LLAMA_CLOUD_API_KEY")
            except Exception:
                self.api_key = os.getenv("LLAMA_CLOUD_API_KEY")

        self._parser = None
        self._initialized = False
        
        if not self.api_key:
            logger.warning("LLAMA_CLOUD_API_KEY not set. LlamaParse will not be available.")
        else:
            logger.info(f"✅ LlamaParseService configured with key ...{self.api_key[-6:]}")
    
    async def _ensure_initialized(self):
        """Initialise le parser LlamaParse si nécessaire."""
        if self._initialized:
            return
        
        if not self.api_key:
            raise LlamaParseError("LLAMA_CLOUD_API_KEY is not configured")
        
        try:
            from llama_parse import LlamaParse
            
            self._parser = LlamaParse(
                api_key=self.api_key,
                result_type="markdown",
                parsing_instruction=LEGAL_DOCUMENT_PARSING_INSTRUCTIONS,
                language="fr",
                verbose=False,
            )
            self._initialized = True
            logger.info("LlamaParse service initialized successfully")
            
        except ImportError:
            raise LlamaParseError(
                "llama-parse package not installed. Run: pip install llama-parse"
            )
        except Exception as e:
            raise LlamaParseError(f"Failed to initialize LlamaParse: {e}")
    
    async def extract_text(
        self,
        pdf_path: Path,
        timeout_seconds: int = 300,
    ) -> str:
        """
        Extrait le texte d'un PDF juridique via LlamaParse.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            timeout_seconds: Timeout maximum (défaut: 5 minutes)
        
        Returns:
            Texte extrait et nettoyé
        
        Raises:
            LlamaParseError: Si l'extraction échoue
        """
        await self._ensure_initialized()
        
        if not pdf_path.exists():
            raise LlamaParseError(f"PDF file not found: {pdf_path}")
        
        try:
            logger.info(f"Starting LlamaParse extraction for: {pdf_path.name}")
            
            # LlamaParse is sync, run in thread pool
            loop = asyncio.get_event_loop()
            documents = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._parser.load_data(str(pdf_path))
                ),
                timeout=timeout_seconds
            )
            
            if not documents:
                logger.warning(f"No content extracted from: {pdf_path.name}")
                return ""
            
            # Combine all document chunks
            full_text = "\n\n".join(doc.text for doc in documents if doc.text)
            
            # Post-processing cleanup
            cleaned_text = self._post_process(full_text)
            
            logger.info(
                f"LlamaParse extraction complete: {len(cleaned_text)} chars from {pdf_path.name}"
            )
            
            return cleaned_text
            
        except asyncio.TimeoutError:
            raise LlamaParseError(
                f"Extraction timeout after {timeout_seconds}s for {pdf_path.name}"
            )
        except Exception as e:
            logger.error(f"LlamaParse extraction failed: {e}")
            raise LlamaParseError(f"Extraction failed: {e}")
    
    def _post_process(self, text: str) -> str:
        """
        Nettoyage post-extraction.
        
        Même avec des instructions, certains artefacts peuvent persister.
        Ce post-processing finalise le nettoyage des tampons camerounais.
        """
        assert isinstance(text, str), "text must be a string"
        assert len(text) < 10_000_000, "text is unreasonably large (>10MB)"

        import re
        
        if not text:
            return ""
        
        cleaned = text
        
        # Remove any remaining page markers
        cleaned = re.sub(r"<<?\s*PAGE\s*:?\s*\d+\s*>>?", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\[?\s*Page\s*\d+\s*\]?", "", cleaned, flags=re.IGNORECASE)
        
        # ========================================
        # OFFICIAL STAMPS (Tampons officiels)
        # ========================================
        stamp_patterns = [
            # Presidency stamps
            r"PRESIDENCE\s+DE\s+LA\s+REPUB[A-Z]*\.?",
            r"PRESIDENCY\s+OF\s+THE\s+REPUBLIC",
            # Secretariat stamps
            r"SECRETARIAT\s+GEN[A-Z]*\s*\"?",
            # Legislative service stamps
            r"SERVICE\s+DU\s+FICHIER\s+LEGISLATIF[\s\w]*",
            r"LEGISLATIVE\s+AND\s+STATUTORY\s+AFFAIRS[\s\w]*",
            # Certified copy stamps
            r"COPIE\s+CERTIFIEE\s+[A-Z\s]*",
            r"CERTIFIED\s+TRUE\s+COPY",
            # Republic
            r"REPUBLIQUE\s+DU\s+CAMEROUN",
            r"REPUBLIC\s+OF\s+CAMEROON",
        ]
        
        # ========================================
        # OFFICIAL SEALS (Sceaux ronds)
        # ========================================
        seal_patterns = [
            r"LE\s+PR[EÉ]SIDENT",
            r"THE\s+PRESIDENT",
            r"PAIX\s*[-–—]?\s*TRAVAIL\s*[-–—]?\s*PATRIE",
            r"PEACE\s*[-–—]?\s*WORK\s*[-–—]?\s*FATHERLAND",
        ]
        
        # ========================================
        # SIGNATURES
        # ========================================
        signature_patterns = [
            r"Paul\s+BIYA",
            r"PAUL\s+BIYA",
            r"LE\s+PR[EÉ]SIDENT\s+DE\s+LA\s+R[EÉ]PUBLIQUE\s*,?",
        ]
        
        # ========================================
        # DATE STAMPS
        # ========================================
        date_patterns = [
            r"Yaound[eé],?\s+le\s+\d{1,2}\s+[A-Z]{3,4}\.?\s+\d{4}",
            r"\d{1,2}\s+(JAN|FEV|MARS?|AVR|MAI|JUIN?|JUIL?|AO[UÛ]T?|SEPT?|OCT|NOV|DEC)\.?\s+\d{4}",
        ]
        
        # Apply all patterns
        all_patterns = stamp_patterns + seal_patterns + signature_patterns + date_patterns
        for pattern in all_patterns:
            cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
        
        # Remove lines of only dashes/underscores (stamp borders)
        cleaned = re.sub(r"^[-–—_=]{3,}\s*$", "", cleaned, flags=re.MULTILINE)
        
        # Remove isolated short lines that look like stamp fragments
        cleaned = re.sub(r"^[A-Z]{1,3}\s+\d*[£€$]?\s*[A-Z]{0,4}\s*[A-Za-z]{0,5}$", "", cleaned, flags=re.MULTILINE)
        
        # Remove lines with only stars (from seals)
        cleaned = re.sub(r"^[\s\*★☆]+$", "", cleaned, flags=re.MULTILINE)
        
        # Normalize whitespace
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        cleaned = re.sub(r"[ \t]+", " ", cleaned)
        
        return cleaned.strip()
    
    async def extract_articles(
        self,
        pdf_path: Path,
        timeout_seconds: int = 300,
    ) -> list[dict]:
        """
        Extrait et parse les articles d'un PDF juridique.
        
        Args:
            pdf_path: Chemin vers le fichier PDF
            timeout_seconds: Timeout maximum
        
        Returns:
            Liste de dictionnaires avec:
            - number: Numéro de l'article
            - title: Titre/en-tête de l'article
            - content: Contenu de l'article
        """
        full_text = await self.extract_text(pdf_path, timeout_seconds)
        
        if not full_text:
            return []
        
        return self._parse_articles(full_text)
    
    def _parse_articles(self, text: str) -> list[dict]:
        """Parse le texte en articles structurés."""
        import re
        
        articles = []
        
        # Pattern pour détecter les articles
        article_pattern = re.compile(
            r"^(Article\s+(\d+|premier|[IVX]+))\s*[-:.]?\s*(.*)$",
            re.IGNORECASE | re.MULTILINE
        )
        
        # Split by article markers
        parts = article_pattern.split(text)
        
        current_preamble = parts[0].strip() if parts else ""
        
        # Add preamble as first "article" if exists
        if current_preamble:
            articles.append({
                "number": "0",
                "title": "Préambule",
                "content": current_preamble,
            })
        
        # Process article matches
        i = 1
        while i < len(parts) - 2:
            full_header = parts[i].strip()
            number = parts[i + 1].strip()
            title_line = parts[i + 2].strip() if i + 2 < len(parts) else ""
            content = parts[i + 3].strip() if i + 3 < len(parts) else ""
            
            # Normalize number
            if number.lower() == "premier":
                number = "1"
            
            articles.append({
                "number": number,
                "title": title_line or f"Article {number}",
                "content": content,
            })
            
            i += 4
        
        return articles
    
    def is_available(self) -> bool:
        """Vérifie si LlamaParse est disponible."""
        return bool(self.api_key)
    
    async def health_check(self) -> dict:
        """Vérifie l'état du service."""
        status = {
            "service": "llama_parse",
            "available": self.is_available(),
            "api_key_configured": bool(self.api_key),
        }
        
        if self.api_key:
            try:
                await self._ensure_initialized()
                status["initialized"] = True
                status["status"] = "healthy"
            except Exception as e:
                status["initialized"] = False
                status["status"] = "error"
                status["error"] = str(e)
        else:
            status["status"] = "unavailable"
        
        return status


# Singleton instance
_llama_parse_service: Optional[LlamaParseService] = None


def get_llama_parse_service() -> LlamaParseService:
    """Get or create the LlamaParseService singleton instance."""
    global _llama_parse_service
    if _llama_parse_service is None:
        _llama_parse_service = LlamaParseService()
    return _llama_parse_service
