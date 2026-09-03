"""
Utility functions for OCR operations.

Provides helpers for:
- PDF text detection
- PDF to image conversion
- OCR text cleaning and post-processing
- Text quality assessment

Author: JuriX Development Team
Date: 2026-01-11
"""

import logging
import re
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)


def has_embedded_text(pdf_path: Path, min_text_length: int = 50) -> Tuple[bool, float]:
    """
    Check if PDF has embedded text (native PDF).

    Args:
        pdf_path: Path to PDF file
        min_text_length: Minimum text length to consider as "has text"

    Returns:
        Tuple of (has_text, text_coverage_ratio)
        - has_text: True if PDF has meaningful embedded text
        - text_coverage_ratio: Ratio of pages with text (0.0-1.0)

    Example:
        >>> has_text, coverage = has_embedded_text(Path("doc.pdf"))
        >>> has_text
        True
        >>> coverage
        0.95
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)
        total_pages = len(reader.pages)
        pages_with_text = 0
        total_text_length = 0

        for page in reader.pages:
            text = page.extract_text()
            text_length = len(text.strip())
            total_text_length += text_length

            if text_length >= min_text_length:
                pages_with_text += 1

        text_coverage = pages_with_text / total_pages if total_pages > 0 else 0.0
        has_text = total_text_length >= min_text_length

        return has_text, text_coverage

    except Exception as e:
        logger.warning(f"Failed to check embedded text: {e}")
        return False, 0.0


def has_images(pdf_path: Path) -> bool:
    """
    Check if PDF contains images.

    Args:
        pdf_path: Path to PDF file

    Returns:
        True if PDF contains images

    Example:
        >>> has_images(Path("scanned.pdf"))
        True
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(pdf_path)

        # Check first few pages for images
        pages_to_check = min(3, len(reader.pages))

        for i in range(pages_to_check):
            page = reader.pages[i]
            if "/XObject" in page.get("/Resources", {}):
                x_object = page["/Resources"]["/XObject"].get_object()
                for obj in x_object:
                    if x_object[obj]["/Subtype"] == "/Image":
                        return True

        return False

    except Exception as e:
        logger.warning(f"Failed to check for images: {e}")
        return False


def convert_pdf_to_images(pdf_path: Path, dpi: int = 300) -> List:
    """
    Convert PDF pages to images for OCR processing.

    Args:
        pdf_path: Path to PDF file
        dpi: Resolution for conversion (default: 300)

    Returns:
        List of PIL Image objects

    Raises:
        ImportError: If pdf2image not available
        Exception: If conversion fails

    Example:
        >>> images = convert_pdf_to_images(Path("doc.pdf"), dpi=300)
        >>> len(images)
        45
    """
    try:
        from pdf2image import convert_from_path

        images = convert_from_path(
            str(pdf_path),
            dpi=dpi,
            fmt="png",
            thread_count=2,  # Parallel processing
        )

        logger.info(f"Converted {len(images)} pages to images at {dpi} DPI")
        return images

    except (ImportError, Exception) as e:
        logger.warning(f"pdf2image failed ({e}), trying pypdf image extraction fallback...")
        try:
            return convert_pdf_to_images_pypdf(pdf_path)
        except Exception as fallback_error:
            logger.error(f"Fallback image extraction failed: {fallback_error}")
            raise e

def convert_pdf_to_images_pypdf(pdf_path: Path) -> List:
    """
    Extract images from PDF using pypdf (Fallback if poppler missing).
    Best for scanned PDFs where each page is an image.
    """
    from pypdf import PdfReader
    from PIL import Image
    import io

    reader = PdfReader(pdf_path)
    images = []

    for i, page in enumerate(reader.pages):
        count = 0
        for image_file_object in page.images:
            try:
                # pypdf returns ImageFile objects containing data
                image_data = image_file_object.data
                image = Image.open(io.BytesIO(image_data))
                images.append(image)
                count += 1
            except Exception as img_err:
                logger.warning(f"Failed to extract image {count} from page {i}: {img_err}")
    
    if not images:
        raise Exception("No images found in PDF using pypdf extraction")
        
    logger.info(f"Extracted {len(images)} images from PDF using pypdf")
    return images


def clean_ocr_text(text: str) -> str:
    """
    Clean and post-process OCR text for RAG indexing.

    Removes stamps, signatures, headers, page numbers, watermarks,
    and formatting artifacts.

    Args:
        text: Raw OCR text

    Returns:
        Clean text optimized for RAG indexing
    """
    if not text:
        return ""

    # 1. Normalize whitespace (preserve newlines for structure)
    text = re.sub(r"[ \t]+", " ", text)

    # 2. Fix OCR character replacements
    text = _fix_ocr_characters(text)

    # 3. Remove polluting content (stamps, headers, signatures, dates)
    text = _remove_administrative_content(text)

    # 4. Remove page markers, watermarks, reference numbers
    text = _remove_page_markers_and_watermarks(text)

    # 5. Fix formatting
    text = _fix_ocr_formatting(text)


    return text


def _fix_ocr_characters(text: str) -> str:
    """Fix common OCR character misreplacements."""
    replacements = {
        "\u2019": "'", "\u2018": "'",
        "\u00ab": '"', "\u00bb": '"',
        "\u2014": "-", "\u2013": "-", "\u2026": "...",
        "\ufb01": "fi", "\ufb02": "fl",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _remove_administrative_content(text: str) -> str:
    """Remove stamps, headers, signatures, and dates from OCR text."""
    stamp_patterns = [
        r"COPIE\s+CERTIFIEE\s+CONFORME", r"POUR\s+COPIE\s+CONFORME",
        r"COPIE\s+CONFORME", r"CERTIFIE\s+CONFORME",
        r"AMPLIATIONS?\s*:", r"AMPLIATION",
        r"CERTIFIED\s+TRUE\s+COPY", r"CERTIFIED\s+COPY",
        r"TRUE\s+COPY", r"FOR\s+CERTIFIED\s+COPY",
    ]
    header_patterns = [
        r"PRESIDENCE\s+DE\s+LA\s+REPUBLIQUE", r"PRESIDENCY\s+OF\s+THE\s+REPUBLIC",
        r"SECRETARIAT\s+GENERAL",
        r"SERVICE\s+DU\s+FICHIER\s+LEGISLATIF\s+ET\s+REGLEMENTAIRE",
        r"LEGISLATIVE\s+AND\s+STATUTORY\s+AFFAIRS\s+CARD\s+INDEX\s+SERVICE",
        r"REPUBLIQUE\s+DU\s+CAMEROUN", r"REPUBLIC\s+OF\s+CAMEROON",
        r"PAIX\s*[-\u2013]\s*TRAVAIL\s*[-\u2013]\s*PATRIE",
        r"PEACE\s*[-\u2013]\s*WORK\s*[-\u2013]\s*FATHERLAND",
        r"JOURNAL\s+OFFICIEL", r"OFFICIAL\s+GAZETTE",
    ]
    signature_patterns = [
        r"Le\s+Pr[e\u00e9]sident\s+de\s+la\s+R[e\u00e9]publique[^.]*\.?",
        r"Le\s+Premier\s+Ministre[^.]*\.?",
        r"Le\s+Ministre\s+d['\u2019]?[Ee\u00c9\u00e9]tat[^.]*\.?",
        r"Le\s+Ministre[^.]*\.?",
        r"Le\s+Secr[e\u00e9]taire\s+G[e\u00e9]n[e\u00e9]ral[^.]*\.?",
        r"Le\s+Directeur[^.]*\.?",
        r"LE\s+PRESIDENT\s+DE\s+LA\s+REPUBLIQUE[^.]*\.?",
        r"(?:Sign[e\u00e9]|Signature)\s*:?\s*[^.\\\n]*",
        r"\(se?\)\s+[A-Z][a-z\u00c0-\u00ff]+(?:\s+[A-Z][a-z\u00c0-\u00ff]+)*",
    ]
    date_patterns = [
        r"Yaound[e\u00e9],?\s+le\s+\d{1,2}\s+\w+\s+\d{4}",
        r"Fait\s+[\u00e0a]\s+Yaound[e\u00e9][^.]*\.",
        r"Fait\s+le\s+\d{1,2}\s+\w+\s+\d{4}",
        r"Done\s+at\s+Yaound[e\u00e9][^.]*\.",
    ]
    for pattern in stamp_patterns + header_patterns + signature_patterns + date_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text


def _remove_page_markers_and_watermarks(text: str) -> str:
    """Remove page numbers, watermarks, and reference numbers."""
    page_patterns = [
        r"^\s*[-\u2013]\s*\d+\s*[-\u2013]\s*$",
        r"^\s*Page\s+\d+\s*(?:sur|of|/)?\s*\d*\s*$",
        r"^\s*\d+\s*/\s*\d+\s*$",
        r"^\s*\d{1,3}\s*$",
    ]
    for pattern in page_patterns:
        text = re.sub(pattern, "", text, flags=re.MULTILINE | re.IGNORECASE)

    watermark_patterns = [
        r"CONFIDENTIEL", r"CONFIDENTIAL",
        r"NE\s+PAS\s+DIFFUSER", r"DO\s+NOT\s+DISTRIBUTE",
        r"DRAFT", r"BROUILLON",
        r"INTERNAL\s+USE\s+ONLY", r"USAGE\s+INTERNE",
    ]
    ref_patterns = [
        r"R[e\u00e9]f[.:]?\s*N[\u00b0o]?\s*[\d/-]+",
        r"N[\u00b0o]?\s*[\d]+/[A-Z]+/[A-Z]+",
        r"Dossier\s+N[\u00b0o]?\s*:?\s*[\w/-]+",
    ]
    for pattern in watermark_patterns + ref_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text


def _fix_ocr_formatting(text: str) -> str:
    """Fix spacing, line breaks, and trailing whitespace."""
    text = re.sub(r" +([.,;:!?])", r"\1", text)
    text = re.sub(r"([.,;:!?])([A-Za-z\u00c0-\u00ff])", r"\1 \2", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s+$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" +$", "", text, flags=re.MULTILINE)
    return text.strip()





def merge_text_blocks(blocks: List[str]) -> str:
    """
    Merge multiple text blocks (e.g., from different pages).

    Args:
        blocks: List of text blocks

    Returns:
        Merged text with proper spacing

    Example:
        >>> blocks = ["Article 1. Text...", "Article 2. More text..."]
        >>> merge_text_blocks(blocks)
        'Article 1. Text...\\n\\nArticle 2. More text...'
    """
    if not blocks:
        return ""

    # Clean each block
    cleaned_blocks = [clean_ocr_text(block) for block in blocks if block.strip()]

    # Join with double newline
    merged = "\n\n".join(cleaned_blocks)

    return merged


def assess_text_quality(text: str) -> float:
    """
    Assess quality of OCR text.

    Heuristics:
    - Ratio of alphabetic characters
    - Presence of common words
    - Sentence structure

    Args:
        text: OCR text to assess

    Returns:
        Quality score (0.0-1.0)

    Example:
        >>> assess_text_quality("Article 1. La présente loi...")
        0.95
        >>> assess_text_quality("@#$% garbage !!!")
        0.1
    """
    if not text or len(text) < 10:
        return 0.0

    score = 0.0

    # 1. Alphabetic character ratio (40% weight)
    alpha_chars = sum(c.isalpha() for c in text)
    total_chars = len(text)
    alpha_ratio = alpha_chars / total_chars if total_chars > 0 else 0
    score += alpha_ratio * 0.4

    # 2. Word ratio (30% weight)
    words = text.split()
    if words:
        # Check for reasonable word lengths
        reasonable_words = sum(1 for w in words if 2 <= len(w) <= 20)
        word_ratio = reasonable_words / len(words)
        score += word_ratio * 0.3

    # 3. Common French/English words (30% weight)
    common_words = {
        "le",
        "la",
        "les",
        "de",
        "et",
        "un",
        "une",
        "the",
        "a",
        "an",
        "of",
        "and",
        "article",
        "loi",
        "law",
    }
    text_lower = text.lower()
    found_common = sum(1 for word in common_words if word in text_lower)
    common_ratio = min(found_common / 5, 1.0)  # At least 5 common words = 1.0
    score += common_ratio * 0.3

    return min(score, 1.0)


def estimate_ocr_time(page_count: int, dpi: int = 300) -> int:
    """
    Estimate OCR processing time.

    Args:
        page_count: Number of pages
        dpi: DPI setting

    Returns:
        Estimated time in seconds

    Example:
        >>> estimate_ocr_time(45, dpi=300)
        18  # ~0.4s per page
    """
    # Base time per page (seconds)
    base_time_per_page = 0.4

    # DPI multiplier
    dpi_multiplier = dpi / 300

    # Total estimate
    estimated_seconds = int(page_count * base_time_per_page * dpi_multiplier)

    return max(estimated_seconds, 1)


def split_text_into_pages(text: str, page_markers: List[str] = None) -> List[str]:
    """
    Split OCR text back into pages if markers present.

    Args:
        text: Full OCR text
        page_markers: List of page marker patterns

    Returns:
        List of text per page

    Example:
        >>> text = "Page 1 text\\n--- PAGE BREAK ---\\nPage 2 text"
        >>> split_text_into_pages(text, ["--- PAGE BREAK ---"])
        ['Page 1 text', 'Page 2 text']
    """
    if not page_markers:
        # Default markers
        page_markers = ["\f", "--- PAGE BREAK ---", "\n\n\n"]

    pages = [text]

    for marker in page_markers:
        if marker in text:
            pages = text.split(marker)
            break

    return [page.strip() for page in pages if page.strip()]
