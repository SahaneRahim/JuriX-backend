"""
LlamaParseService - Extraction OCR des documents juridiques via LlamaParse v2.

Remplace l'extraction par couche texte (pdfplumber/pypdf), inexploitable sur le
corpus prc.cm : 21% des documents n'ont aucune couche texte utilisable, et les
79% restants sont pollués par le filigrane www.prc.cm et les cachets officiels.

Banc d'essai sur 6 pages representatives (verite terrain relevee visuellement) :

    moteur                      prose    tableau budgetaire
    couche texte actuelle        20%           0/9
    tesseract 5.5 local          74%           0/9
    mistral OCR                 100%           8/9   (chiffre faux, reproductible)
    llamaparse cost_effective   100%           9/9

Architecture :
- API REST v2 : POST /parse/upload -> poll GET /parse/{id} -> markdown
- Cache disque par sha256 : une page n'est jamais payee deux fois
- Retry exponentiel : 1 echec reseau observe sur 6 appels au banc
- Decoupe automatique des PDF volumineux (5 fichiers >50 Mo dans le corpus)
- Marqueurs <<PAGE:n>> pour que text_chunker renseigne Article.page_number
- Nettoyage des cachets officiels (regle 7 du chunking)

Tiers disponibles (credits/page, 1000 credits = 1,25 $) :
    fast            1    relit la couche texte -> 50% de rappel, inutilisable ici
    cost_effective  3    recommande : 100% de rappel, 9/9 sur les tableaux
    agentic        10    aucun gain mesure sur ce corpus
    agentic_plus   45    aucun gain mesure sur ce corpus

Author: JuriX Team
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)


class LlamaParseError(Exception):
    """Erreur d'extraction LlamaParse."""
    pass


# ==================== NETTOYAGE DES CACHETS ====================

# Vocabulaire du cachet "COPIE CERTIFIEE CONFORME" appose sur chaque page du
# corpus prc.cm. LlamaParse le restitue en texte (contrairement a Mistral OCR
# qui l'ignore), il faut donc le retirer avant chunking sinon il pollue le FTS
# et les embeddings de chaque document.
_STAMP_VOCAB = re.compile(
    r"(copie\s+certifi\w*\s+conforme"
    r"|certified\s+true\s+copy"
    r"|service\s+du\s+fichier\s+l[ée]gislatif"
    r"|legislative\s+and\s+statutory\s+affairs"
    r"|presidency\s+of\s+the\s+republic"
    r"|secr[ée]tariat[\s-]+g[ée]n[ée]ral"
    r"|pr[ée]sidence\s+de\s+la\s+r[ée]publique)",
    re.IGNORECASE,
)

# Fragments du filigrane diagonal "www.prc.cm" eclate par l'OCR
_WATERMARK = re.compile(
    r"^\s*(?:w\s*){1,3}$|^\s*\.?\s*(?:p\s*r\s*c|c\s*m|prc\.cm|www\.prc\.cm)\s*\.?\s*$",
    re.IGNORECASE,
)

# Balisage HTML inline emis par LlamaParse (les <table> sont conserves : la
# structure tabulaire porte du sens, cf. les annexes budgetaires).
# <sup>/<sub> sont critiques : "ARTICLE 1<sup>ER</sup>" doit redevenir
# "ARTICLE 1ER" pour que ARTICLE_PATTERNS de text_chunker le reconnaisse.
_INLINE_TAGS = re.compile(
    r"</?(?:u|mark|b|i|em|strong|span|sup|sub|small)\b[^>]*>", re.IGNORECASE
)


def _strip_stamp_blocks(text: str) -> str:
    """
    Retire les blocs de cachet officiel du markdown.

    Un bloc = au moins 2 lignes consecutives (blancs autorises) appartenant au
    vocabulaire du cachet. Le seuil de 2 evite de supprimer un en-tete legitime :
    beaucoup de documents portent "PRESIDENCE DE LA REPUBLIQUE" comme autorite
    emettrice, ce qui est du contenu, pas du tampon.

    Args:
        text: Markdown brut renvoye par LlamaParse

    Returns:
        Markdown nettoye
    """
    lines = text.splitlines()
    drop = [False] * len(lines)
    i = 0

    while i < len(lines):
        if not _STAMP_VOCAB.search(lines[i]):
            i += 1
            continue

        # Cas 1 : cachet entier restitue sur une seule ligne, par ex.
        # "[signature: PRESIDENCE ... COPIE CERTIFIEE CONFORME CERTIFIED TRUE COPY]"
        # ou "stamp: ..." / "logo: ...". Deux phrases distinctes du vocabulaire sur
        # la meme ligne ne peuvent pas etre du texte juridique legitime.
        if len(set(m.group(0).lower() for m in _STAMP_VOCAB.finditer(lines[i]))) >= 2:
            drop[i] = True
            i += 1
            continue

        # Etendre le bloc tant qu'on reste dans le vocabulaire du cachet
        j = i
        hits = 0
        while j < len(lines):
            stripped = lines[j].strip(" >*|-\t")
            if _STAMP_VOCAB.search(lines[j]):
                hits += 1
                j += 1
            elif not stripped:
                j += 1
            else:
                break

        if hits >= 2:
            for k in range(i, j):
                drop[k] = True
        i = max(j, i + 1)

    kept = [ln for ln, d in zip(lines, drop) if not d]
    kept = [ln for ln in kept if not _WATERMARK.match(ln)]
    cleaned = "\n".join(kept)
    cleaned = _INLINE_TAGS.sub("", cleaned)
    return re.sub(r"\n{4,}", "\n\n\n", cleaned).strip()


class LlamaParseService:
    """
    Client LlamaParse v2 pour l'extraction des PDF juridiques scannes.

    Usage:
        >>> service = LlamaParseService()
        >>> if service.is_available():
        ...     text = await service.extract_text(Path("decret.pdf"))
    """

    BASE_URL = "https://api.cloud.llamaindex.ai/api/v2/parse"
    DEFAULT_TIER = "cost_effective"

    # Decoupe defensive : seuil conservateur, en-deca de la limite annoncee.
    # Le corpus contient 5 fichiers >50 Mo (max observe : 124,6 Mo / 48 pages).
    MAX_FILE_MB = 45
    SPLIT_PAGES = 100

    MAX_RETRIES = 4
    RETRY_BASE_DELAY = 3.0
    POLL_INTERVAL = 3.0
    POLL_TIMEOUT = 900.0
    UPLOAD_TIMEOUT = 300.0

    def __init__(
        self,
        api_key: Optional[str] = None,
        tier: Optional[str] = None,
        cache_dir: Optional[Path] = None,
    ):
        """
        Initialise le service.

        Args:
            api_key: Cle LlamaCloud (defaut: settings.LLAMA_CLOUD_API_KEY)
            tier: Tier de parsing (defaut: settings.LLAMA_PARSE_TIER)
            cache_dir: Repertoire du cache sha256 (defaut: ./data/ocr_cache)
        """
        self.api_key = api_key or settings.LLAMA_CLOUD_API_KEY
        self.tier = tier or getattr(settings, "LLAMA_PARSE_TIER", self.DEFAULT_TIER)

        self.cache_dir = Path(
            cache_dir or getattr(settings, "OCR_CACHE_DIR", "./data/ocr_cache")
        )
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"⚠️ Cache OCR indisponible ({e}), desactive")
            self.cache_dir = None

        if self.api_key:
            logger.info(f"✅ LlamaParseService pret (tier={self.tier})")
        else:
            logger.warning("⚠️ LLAMA_CLOUD_API_KEY absente — LlamaParse desactive")

    # ==================== API PUBLIQUE ====================

    def is_available(self) -> bool:
        """Le service est-il configure ?"""
        return bool(self.api_key)

    async def extract_text(self, file_path: Path) -> str:
        """
        Extrait le texte complet d'un PDF, avec marqueurs de page.

        Point d'entree utilise par app/tasks/process_law.py.

        Args:
            file_path: Chemin du PDF

        Returns:
            Markdown nettoye, pages separees par <<PAGE:n>>

        Raises:
            LlamaParseError: Si l'extraction echoue apres tous les retries
        """
        pages = await self.extract_pages(file_path)
        if not pages:
            raise LlamaParseError(f"Aucune page extraite de {file_path.name}")

        parts = [f"<<PAGE:{i}>>\n{md}" for i, md in enumerate(pages, start=1) if md.strip()]
        text = "\n\n".join(parts)
        logger.info(
            f"✅ LlamaParse: {len(pages)} page(s), {len(text)} caracteres "
            f"depuis {file_path.name}"
        )
        return text

    async def extract_pages(self, file_path: Path) -> List[str]:
        """
        Extrait le markdown page par page (cache-aware).

        Args:
            file_path: Chemin du PDF

        Returns:
            Liste de markdown, un element par page, dans l'ordre
        """
        assert file_path is not None, "file_path requis"
        if not self.is_available():
            raise LlamaParseError("LLAMA_CLOUD_API_KEY non configuree")
        if not file_path.exists():
            raise LlamaParseError(f"Fichier introuvable: {file_path}")

        cache_key = self._sha256(file_path)
        cached = self._read_cache(cache_key)
        if cached is not None:
            logger.info(f"🎯 Cache OCR HIT: {file_path.name} ({len(cached)} pages)")
            return cached

        chunks = self._split_if_needed(file_path)
        pages: List[str] = []
        try:
            for idx, chunk_path in enumerate(chunks, start=1):
                if len(chunks) > 1:
                    logger.info(f"📄 Segment {idx}/{len(chunks)}: {chunk_path.name}")
                pages.extend(await self._parse_one(chunk_path))
        finally:
            for c in chunks:
                if c != file_path:
                    c.unlink(missing_ok=True)

        pages = [_strip_stamp_blocks(p) for p in pages]
        self._write_cache(cache_key, pages)
        return pages

    async def health_check(self) -> Dict[str, Any]:
        """Verifie la configuration et la joignabilite de l'API."""
        if not self.is_available():
            return {"service": "LlamaParse", "status": "unconfigured", "tier": self.tier}
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.get(
                    f"{self.BASE_URL}/00000000-0000-0000-0000-000000000000",
                    headers=self._headers(),
                )
            reachable = r.status_code in (200, 401, 403, 404, 422)
            return {
                "service": "LlamaParse",
                "status": "healthy" if reachable else "degraded",
                "tier": self.tier,
                "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            }
        except Exception as e:
            return {"service": "LlamaParse", "status": "unreachable", "error": str(e)}

    # ==================== APPEL API ====================

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    async def _parse_one(self, file_path: Path) -> List[str]:
        """Soumet un fichier, attend le job, renvoie le markdown par page."""
        last_error: Optional[Exception] = None

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                job_id = await self._upload(file_path)
                return await self._poll(job_id)
            except LlamaParseError:
                raise
            except Exception as e:
                last_error = e
                if attempt < self.MAX_RETRIES:
                    delay = self.RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        f"⚠️ Tentative {attempt}/{self.MAX_RETRIES} echouee "
                        f"({type(e).__name__}: {e}), retry dans {delay:.0f}s"
                    )
                    await asyncio.sleep(delay)

        raise LlamaParseError(
            f"Echec apres {self.MAX_RETRIES} tentatives sur {file_path.name}: {last_error}"
        )

    async def _upload(self, file_path: Path) -> str:
        """Soumet le fichier et renvoie l'id du job."""
        config = json.dumps({"tier": self.tier, "version": "latest"})

        async with httpx.AsyncClient(timeout=self.UPLOAD_TIMEOUT) as client:
            with file_path.open("rb") as fh:
                resp = await client.post(
                    f"{self.BASE_URL}/upload",
                    headers=self._headers(),
                    files={"file": (file_path.name, fh, "application/pdf")},
                    data={"configuration": config},
                )

        if resp.status_code in (401, 403):
            raise LlamaParseError(f"Cle LlamaCloud refusee (HTTP {resp.status_code})")
        if resp.status_code not in (200, 201):
            raise RuntimeError(f"upload HTTP {resp.status_code}: {resp.text[:300]}")

        payload = resp.json()
        job_id = (payload.get("job") or {}).get("id") or payload.get("id")
        if not job_id:
            raise RuntimeError(f"Pas de job id dans la reponse: {str(payload)[:300]}")
        return job_id

    async def _poll(self, job_id: str) -> List[str]:
        """Interroge le job jusqu'a completion, renvoie le markdown par page."""
        started = time.monotonic()

        async with httpx.AsyncClient(timeout=120.0) as client:
            while True:
                if time.monotonic() - started > self.POLL_TIMEOUT:
                    raise LlamaParseError(
                        f"Timeout ({self.POLL_TIMEOUT:.0f}s) sur le job {job_id}"
                    )
                await asyncio.sleep(self.POLL_INTERVAL)

                resp = await client.get(
                    f"{self.BASE_URL}/{job_id}",
                    headers=self._headers(),
                    params={"expand": "markdown,markdown_full"},
                )
                if resp.status_code != 200:
                    raise RuntimeError(f"poll HTTP {resp.status_code}: {resp.text[:300]}")

                data = resp.json()
                job = data.get("job") or {}
                status = job.get("status") or data.get("status")

                if status in ("COMPLETED", "SUCCESS"):
                    return self._pages_from(data)
                if status in ("FAILED", "CANCELLED", "ERROR"):
                    raise LlamaParseError(
                        f"Job {status}: {job.get('error_message') or 'raison inconnue'}"
                    )

    @staticmethod
    def _pages_from(data: Dict[str, Any]) -> List[str]:
        """Normalise la reponse v2 en liste de markdown par page."""
        md = data.get("markdown")

        if isinstance(md, list):
            return [
                (p.get("markdown", "") if isinstance(p, dict) else str(p)) for p in md
            ]
        if isinstance(md, str) and md.strip():
            return [md]

        pages = data.get("pages")
        if isinstance(pages, list):
            return [
                (p.get("markdown", "") if isinstance(p, dict) else str(p)) for p in pages
            ]

        full = data.get("markdown_full")
        return [full] if isinstance(full, str) and full.strip() else []

    # ==================== DECOUPE DES GROS FICHIERS ====================

    def _split_if_needed(self, file_path: Path) -> List[Path]:
        """
        Decoupe un PDF trop volumineux en segments temporaires.

        Returns:
            [file_path] si aucune decoupe n'est necessaire, sinon les segments.
        """
        size_mb = file_path.stat().st_size / 1_000_000
        if size_mb <= self.MAX_FILE_MB:
            return [file_path]

        try:
            from pypdf import PdfReader, PdfWriter
        except ImportError:
            logger.warning(
                f"⚠️ pypdf absent, {file_path.name} ({size_mb:.0f} Mo) envoye tel quel"
            )
            return [file_path]

        reader = PdfReader(str(file_path))
        total = len(reader.pages)
        logger.info(
            f"✂️ {file_path.name}: {size_mb:.0f} Mo / {total} pages -> "
            f"segments de {self.SPLIT_PAGES} pages"
        )

        parts: List[Path] = []
        for start in range(0, total, self.SPLIT_PAGES):
            writer = PdfWriter()
            for page in reader.pages[start:start + self.SPLIT_PAGES]:
                writer.add_page(page)
            part = file_path.with_name(f"{file_path.stem}__part{start // self.SPLIT_PAGES}.pdf")
            with part.open("wb") as fh:
                writer.write(fh)
            parts.append(part)

        return parts

    # ==================== CACHE SHA256 ====================

    @staticmethod
    def _sha256(file_path: Path) -> str:
        h = hashlib.sha256()
        with file_path.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    def _cache_path(self, key: str) -> Optional[Path]:
        return self.cache_dir / f"{key}.json" if self.cache_dir else None

    def _read_cache(self, key: str) -> Optional[List[str]]:
        path = self._cache_path(key)
        if not path or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("tier") != self.tier:
                return None
            return payload.get("pages")
        except Exception as e:
            logger.warning(f"⚠️ Cache OCR illisible ({e}), ignore")
            return None

    def _write_cache(self, key: str, pages: List[str]) -> None:
        path = self._cache_path(key)
        if not path:
            return
        try:
            path.write_text(
                json.dumps(
                    {"tier": self.tier, "pages": pages, "cached_at": time.time()},
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
        except OSError as e:
            logger.warning(f"⚠️ Ecriture cache OCR echouee ({e})")


@lru_cache()
def get_llama_parse_service() -> LlamaParseService:
    """Singleton LlamaParseService."""
    return LlamaParseService()


def clear_llama_parse_service_cache() -> None:
    """Force la recreation du singleton (tests, rotation de cle)."""
    get_llama_parse_service.cache_clear()
