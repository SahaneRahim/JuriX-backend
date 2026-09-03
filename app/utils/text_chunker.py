"""
Text chunking utilities for legal document processing.

Extracts articles from Cameroonian legal documents following
common patterns: Article X, Art. X, Section X, etc.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class ArticleExtractionError(Exception):
    """Raised when article extraction fails."""
    pass


@dataclass
class ExtractedArticle:
    """Represents a single extracted article."""
    number: str
    title: Optional[str]
    content: str
    position: int
    parent_id: Optional[str]
    section: Optional[str]
    word_count: int
    char_count: int
    page_number: Optional[int] = None  # PDF page number (1-indexed) for navigation


# Pattern constants - COMPREHENSIVE support for French and English variants
# French variants: Article 1, Article 1er, Article premier, Article première, Article deuxième, etc.
# English variants: Section 1, Section one, Section first, Article 1, Article one, etc.
# NOTE: le groupe capturant englobe la numerotation hierarchique complete
# ((\d+(?:\.\d+)*)). Auparavant le '(?:\.\d+)*' etait HORS du groupe :
# 'Article 1.1' et 'Article 1.2' etaient tous deux captures comme '1' et
# fusionnaient avec l'article 1 — une erreur de citation sur une base juridique.
ARTICLE_PATTERNS = [
    # === FRENCH PATTERNS ===
    # Article + number (1, 2, 3...) OR ordinals (1er, 1ère, 2ème, 3ème...) OR words (premier, première, deuxième...)
    r'(?:^|\n)\s*Article\s+(?:' +
        r'(\d+(?:\.\d+)*)(?:er|ère|ème)?' +  # Article 1, Article 1er, Article 2ème, Article 1.1
        r'|' +
        r'(premier|première|deuxième|second|seconde|troisième|quatrième|cinquième|sixième|septième|huitième|neuvième|dixième)' +  # Article premier, Article deuxième...
    r')\s*[.:\-–]?\s*',
    
    # Art. (abbreviation) + number OR ordinals
    r'(?:^|\n)\s*Art\.?\s+(?:' +
        r'(\d+(?:\.\d+)*)(?:er|ère|ème)?' +  # Art. 1, Art. 1er, Art. 2ème
        r'|' +
        r'(premier|première|deuxième|second|seconde|troisième|quatrième|cinquième|sixième|septième|huitième|neuvième|dixième)' +  # Art. premier
    r')\s*[.:\-–]?\s*',
    
    # === ENGLISH PATTERNS ===
    # Section + number OR words (one, first, second, third...)
    r'(?:^|\n)\s*Section\s+(?:' +
        r'(\d+(?:\.\d+)*)' +  # Section 1, Section 1.1
        r'|' +
        r'(one|first|two|second|three|third|four|fourth|five|fifth|six|sixth|seven|seventh|eight|eighth|nine|ninth|ten|tenth)' +  # Section one, Section first
    r')\s*[.:\-–]?\s*',
    
    # Article (English style) + number OR words
    r'(?:^|\n)\s*Article\s+(?:' +
        r'(\d+(?:\.\d+)*)' +  # Article 1 (English)
        r'|' +
        r'(one|first|two|second|three|third|four|fourth|five|fifth|six|sixth|seven|seventh|eight|eighth|nine|ninth|ten|tenth)' +  # Article one (English)
    r')\s*[.:\-–]?\s*',
    
    # Sec. (abbreviation, English) + number
    r'(?:^|\n)\s*Sec\.?\s+(\d+(?:\.\d+)*)\s*[.:\-–]?\s*',

    # === NUMEROTATION CODIFIEE (Code Général des Impôts, CGI) ===
    # "Article L 94 septies.-", "Article L 94", "Art. M 12 bis"
    # Rencontre dans les lois de finances qui modifient le CGI.
    r'(?:^|\n)\s*Art(?:icle|\.)?\s+([A-Z]\s*\d+(?:\s+(?:bis|ter|quater|quinquies|'
    r'sexies|septies|octies|novies|decies))?)\s*[.:\-–]?\s*',

    # === ORDINAUX COMPOSES (français) ===
    # "ARTICLE QUATRE-VINGT-SIXIÈME", "Article trente-et-unième"
    # La liste explicite ci-dessus s'arrête à "dixième" ; les lois de finances
    # numérotent leurs articles en toutes lettres bien au-delà.
    r'(?:^|\n)\s*Article\s+((?:[A-Za-zÀ-ÿ]+-)+[A-Za-zÀ-ÿ]*(?:ièmes?|èmes?|iemes?|emes?))'
    r'\s*[.:\-–]?\s*',
]

# Pattern for préambule/preamble detection - multiple patterns for flexibility
PREAMBLE_PATTERNS = [
    r'(?:^|\n)\s*(PRÉAMBULE|PREAMBULE|PREAMBLE)\s*[.:]?\s*',
    r'(?:^|\n)\s*Le\s+peuple\s+camerounais',  # Constitution camerounaise
    r'(?:^|\n)\s*The\s+people\s+of\s+Cameroon',  # English version
    r'(?:^|\n)\s*Nous,\s+peuple',  # Other constitutions
    r'(?:^|\n)\s*We,\s+the\s+people',  # English generic
]

# Patterns indicating end of legal basis / start of substantive content
LEGAL_BASIS_END_PATTERNS = [
    r'(?:^|\n)\s*(PRÉAMBULE|PREAMBULE|PREAMBLE)\s*[.:]?',
    r'(?:^|\n)\s*Le\s+peuple\s+camerounais',
    r'(?:^|\n)\s*The\s+people\s+of\s+Cameroon',
    r'(?:^|\n)\s*(DÉCIDE|DECIDE|DÉCRÈTE|DECRETE|ARRÊTE|ARRETE)\s*[.:]?',
    r'(?:^|\n)\s*(TITRE\s+(?:PREMIER|I|1)|TITLE\s+(?:ONE|I|1))\s*[.:\-–]?',
]

TITLE_PATTERN = r'(?:^|\n)\s*Article\s+\d+\s*[.:]?\s*([^\n]+?)(?:\n|$)'

# Section patterns - TITRE and CHAPITRE that appear between articles
#
# ATTENTION : l'alternative de chiffres romains utilisait 'V?I{0,3}', qui peut
# matcher la CHAINE VIDE. 'TITRE\s+' suffisait alors a declencher une detection
# de section : un article intitule 'Titre niveau 1' etait pris pour un en-tete,
# et son contenu coupe a zero caractere puis ecarte comme 'trop court'.
# Autrement dit, tout article dont le titre commence par Titre/Chapitre etait
# silencieusement PERDU. Remplace par [IVXLC]{1,7}, qui ne peut pas etre vide.
# These define the section/chapter for subsequent articles
SECTION_PATTERNS = [
    # TITRE PREMIER - DE L'ÉTAT, TITRE I, TITRE 1, etc.
    r'(?:^|\n)\s*(TITRE\s+(?:PREMIER|PREMI[ÈE]RE|DEUXI[ÈE]ME|TROISI[ÈE]ME|QUATRI[ÈE]ME|CINQUI[ÈE]ME|SIXI[ÈE]ME|SEPTI[ÈE]ME|HUITI[ÈE]ME|NEUVI[ÈE]ME|DIXI[ÈE]ME|[IVXLC]{1,7}|\d+)\s*[.:\-–]?\s*[^\n]*)',
    # CHAPITRE PREMIER, CHAPITRE I, CHAPITRE 1, etc.
    r'(?:^|\n)\s*(CHAPITRE\s+(?:PREMIER|PREMI[ÈE]RE|DEUXI[ÈE]ME|TROISI[ÈE]ME|QUATRI[ÈE]ME|CINQUI[ÈE]ME|SIXI[ÈE]ME|SEPTI[ÈE]ME|HUITI[ÈE]ME|NEUVI[ÈE]ME|DIXI[ÈE]ME|[IVXLC]{1,7}|\d+)\s*[.:\-–]?\s*[^\n]*)',
    # PART ONE, PART I, PART 1 (English)
    r'(?:^|\n)\s*(PART\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|[IVXLC]{1,7}|\d+)\s*[.:\-–]?\s*[^\n]*)',
    # CHAPTER ONE, CHAPTER I, CHAPTER 1 (English)
    r'(?:^|\n)\s*(CHAPTER\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|[IVXLC]{1,7}|\d+)\s*[.:\-–]?\s*[^\n]*)',
]


def normalize_article_number(number: str) -> str:
    """
    Normalize article/section numbers to standard format.
    
    Converts ALL variants to numeric format:
    - French: 'premier', 'première' -> '1', 'deuxième', 'second' -> '2', etc.
    - English: 'one', 'first' -> '1', 'two', 'second' -> '2', etc.
    - Ordinals: '1er', '1ère', '2ème' -> '1', '2', etc.
    - Special: 'PRÉAMBULE', 'PREAMBULE', 'PREAMBLE' -> 'PREAMBULE'
    """
    if not number:
        return number
    
    lower = number.lower().strip()
    
    # === FRENCH WORD-TO-NUMBER MAPPING ===
    french_numbers = {
        'premier': '1', 'première': '1',
        'deuxième': '2', 'second': '2', 'seconde': '2',
        'troisième': '3',
        'quatrième': '4',
        'cinquième': '5',
        'sixième': '6',
        'septième': '7',
        'huitième': '8',
        'neuvième': '9',
        'dixième': '10',
    }
    
    # === ENGLISH WORD-TO-NUMBER MAPPING ===
    english_numbers = {
        'one': '1', 'first': '1',
        'two': '2', 'second': '2',
        'three': '3', 'third': '3',
        'four': '4', 'fourth': '4',
        'five': '5', 'fifth': '5',
        'six': '6', 'sixth': '6',
        'seven': '7', 'seventh': '7',
        'eight': '8', 'eighth': '8',
        'nine': '9', 'ninth': '9',
        'ten': '10', 'tenth': '10',
    }
    
    # Check French word numbers
    if lower in french_numbers:
        return french_numbers[lower]
    
    # Check English word numbers
    if lower in english_numbers:
        return english_numbers[lower]
    
    # Handle preamble
    if lower in ['préambule', 'preambule', 'preamble']:
        return 'PREAMBULE'
    
    # Remove French ordinal suffixes (1er, 1ère, 2ème, 3ème...)
    if lower.endswith(('er', 'ère', 'ème')):
        # Extract just the number part
        number_clean = re.sub(r'(er|ère|ème)$', '', lower)
        if number_clean.replace('.', '').isdigit():
            return number_clean
    
    # Return as-is for numeric values (1, 2, 3, 1.1, 2.3, etc.)
    return number


def _make_chunk(
    number: str, title: Optional[str], content: str,
    position: int, section: Optional[str],
    page_number: int, parent_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a standardized chunk dict. Centralizes chunk creation logic."""
    assert content, "Chunk content must not be empty"
    assert isinstance(position, int) and position >= 0, "Position must be non-negative int"
    return {
        'number': number,
        'title': title,
        'content': content,
        'position': position,
        'parent_id': parent_id,
        'section': section,
        'word_count': len(content.split()),
        'char_count': len(content),
        'page_number': page_number,
    }


def _update_page(text: str, page_marker_pattern: re.Pattern, current_page: int) -> int:
    """Extract page number from text if present, else return current page."""
    page_match = page_marker_pattern.search(text)
    if page_match:
        return int(page_match.group(1))
    return current_page


def _extract_pre_article_chunks(
    processed_text: str, pattern: re.Pattern,
    page_marker_pattern: re.Pattern, current_page: int,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Extract legal basis and preamble chunks from pre-article text.

    Returns:
        Tuple of (chunks, next_position, current_page)
    """
    assert processed_text, "Processed text must not be empty"
    assert pattern is not None, "Article pattern must be provided"

    chunks: List[Dict[str, Any]] = []
    position = 0
    matches = list(pattern.finditer(processed_text))

    if not matches:
        return chunks, position, current_page

    pre_article_text = processed_text[:matches[0].start()].strip()
    if not pre_article_text:
        return chunks, position, current_page

    # Find earliest preamble marker
    preamble_match = _find_earliest_preamble(pre_article_text)

    if preamble_match:
        chunks, position, current_page = _split_legal_basis_and_preamble(
            pre_article_text, preamble_match, page_marker_pattern, current_page
        )
    else:
        chunks, position, current_page = _classify_pre_article_text(
            pre_article_text, page_marker_pattern, current_page
        )

    return chunks, position, current_page


def _find_earliest_preamble(pre_article_text: str) -> Optional[re.Match]:
    """Find the earliest preamble pattern match in pre-article text."""
    best_match = None
    best_pos = None
    for preamble_pattern in PREAMBLE_PATTERNS:
        match = re.search(preamble_pattern, pre_article_text, re.IGNORECASE | re.MULTILINE)
        if match and (best_pos is None or match.start() < best_pos):
            best_pos = match.start()
            best_match = match
    return best_match


def _split_legal_basis_and_preamble(
    pre_article_text: str, preamble_match: re.Match,
    page_marker_pattern: re.Pattern, current_page: int,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Split pre-article text into legal basis and preamble chunks."""
    chunks: List[Dict[str, Any]] = []
    position = 0

    legal_basis_text = pre_article_text[:preamble_match.start()].strip()
    preamble_text = pre_article_text[preamble_match.start():].strip()

    # Add legal basis chunk (if substantial)
    if legal_basis_text and len(legal_basis_text) > 20:
        current_page = _update_page(legal_basis_text, page_marker_pattern, current_page)
        clean = page_marker_pattern.sub('', legal_basis_text).strip()
        chunks.append(_make_chunk('LEGAL_BASIS', 'Base légale', clean, position, None, current_page))
        position += 1

    # Add preamble chunk
    if preamble_text:
        current_page = _update_page(preamble_text, page_marker_pattern, current_page)
        clean = page_marker_pattern.sub('', preamble_text).strip()
        chunks.append(_make_chunk('PREAMBULE', 'Préambule', clean, position, None, current_page))
        position += 1

    return chunks, position, current_page


def _classify_pre_article_text(
    pre_article_text: str, page_marker_pattern: re.Pattern, current_page: int,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """Classify pre-article text as either preamble or legal basis."""
    is_preamble = any(
        re.match(p, pre_article_text.strip(), re.IGNORECASE)
        for p in PREAMBLE_PATTERNS[1:]
    )

    current_page = _update_page(pre_article_text, page_marker_pattern, current_page)
    clean = page_marker_pattern.sub('', pre_article_text).strip()

    number = 'PREAMBULE' if is_preamble else 'LEGAL_BASIS'
    title = 'Préambule' if is_preamble else 'Base légale'

    chunk = _make_chunk(number, title, clean, 0, None, current_page)
    return [chunk], 1, current_page


def _extract_article_chunks(
    processed_text: str, pattern: re.Pattern,
    page_marker_pattern: re.Pattern, start_position: int,
    current_page: int, min_article_length: int,
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Extract article chunks from text using detected pattern.

    Returns:
        Tuple of (chunks, next_position, current_page)
    """
    assert processed_text, "Text must not be empty"
    assert min_article_length >= 0, "min_article_length must be non-negative"

    chunks: List[Dict[str, Any]] = []
    position = start_position
    raw_articles = _split_by_pattern_with_sections(processed_text, pattern)

    for number, content, section in raw_articles:
        current_page = _update_page(content, page_marker_pattern, current_page)
        clean_content = page_marker_pattern.sub('', content).strip()
        normalized_number = normalize_article_number(number)
        parent_id = _get_parent_id(normalized_number)

        # _extract_title existait mais n'etait appelee nulle part : le titre
        # etait passe en dur a None, donc AUCUN article n'avait jamais de titre.
        # C'est une perte directe pour les citations affichees a l'utilisateur
        # et pour le contexte envoye au modele.
        title = _extract_title(clean_content)
        clean_content = _clean_article_content(clean_content, False, title)
        char_count = len(clean_content)

        if char_count >= min_article_length:
            chunks.append(_make_chunk(
                normalized_number, title, clean_content,
                position, section, current_page, parent_id,
            ))
            position += 1
        else:
            logger.warning(
                f"⚠️  Article {normalized_number} trop court "
                f"({char_count} chars), ignoré"
            )

    return chunks, position, current_page


def _extract_paragraph_chunks(
    processed_text: str, page_marker_pattern: re.Pattern,
    current_page: int, min_article_length: int,
) -> List[Dict[str, Any]]:
    """
    Extract paragraph chunks from documents without articles.

    Returns:
        List of chunk dicts
    """
    assert processed_text, "Text must not be empty"
    assert min_article_length >= 0, "min_article_length must be non-negative"

    chunks: List[Dict[str, Any]] = []
    position = 0
    paragraphs = re.split(r'\n\s*\n', processed_text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    if len(paragraphs) > 1:
        for i, paragraph in enumerate(paragraphs, start=1):
            current_page = _update_page(paragraph, page_marker_pattern, current_page)
            clean_para = page_marker_pattern.sub('', paragraph).strip()
            if len(clean_para) >= min_article_length:
                chunks.append(_make_chunk(
                    f'PARA_{i}', f'Paragraphe {i}', clean_para,
                    position, None, current_page,
                ))
                position += 1
    else:
        logger.info("📄 Texte continu - stockage en un seul chunk")
        current_page = _update_page(processed_text, page_marker_pattern, current_page)
        clean_text = page_marker_pattern.sub('', processed_text).strip()
        chunks.append(_make_chunk(
            'FULL_TEXT', 'Document complet', clean_text,
            0, None, current_page,
        ))

    return chunks


def extract_articles(
    text: str,
    min_article_length: int = 10,
    preserve_formatting: bool = False,
    strict: bool = True
) -> List[Dict[str, Any]]:
    """
    Extract articles from legal document text with COMPLETE content preservation.

    Delegates to helper functions for each extraction phase:
    1. Pre-article content (legal basis, preamble)
    2. Articles (with section tracking)
    3. Paragraphs (for non-article documents)

    Args:
        text: Full legal document text
        min_article_length: Minimum characters per chunk
        preserve_formatting: Keep original whitespace/formatting
        strict: Raise errors vs warnings for validation failures

    Returns:
        List of chunk dicts with standard keys (number, title, content, etc.)

    Raises:
        ValueError: If text is empty or too large
        ArticleExtractionError: If no content could be extracted
    """
    # 1. Validate input
    assert isinstance(text, str), "text must be a string"
    if not text or not text.strip():
        raise ValueError("Le texte ne peut pas être vide")
    if len(text) > 5_000_000:
        raise ValueError(f"Texte trop volumineux ({len(text)} chars, max 5M)")

    # 2. Preprocess and detect pattern
    processed_text = _preprocess_text(text, preserve_formatting)
    pattern = _detect_article_pattern(processed_text)
    page_marker_pattern = re.compile(r'<<PAGE:(\d+)>>')

    # 3. Extract chunks based on document structure
    if pattern:
        logger.info("📋 Document avec articles détecté")
        pre_chunks, position, page = _extract_pre_article_chunks(
            processed_text, pattern, page_marker_pattern, 1
        )
        article_chunks, _, _ = _extract_article_chunks(
            processed_text, pattern, page_marker_pattern,
            position, page, min_article_length
        )
        chunks = pre_chunks + article_chunks
    else:
        logger.info("📄 Document sans articles - extraction par paragraphes")
        chunks = _extract_paragraph_chunks(
            processed_text, page_marker_pattern, 1, min_article_length
        )

    # 4. Final validation
    if not chunks:
        raise ArticleExtractionError("Aucun contenu extrait du document")

    logger.info(f"✅ Extracted {len(chunks)} chunks (complete content preservation)")
    return chunks


def _validate_input(text: str, strict: bool) -> None:
    """Validate input text."""
    if not text or not text.strip():
        raise ValueError("Le texte ne peut pas être vide")

    # Removed minimum length requirement to support smaller documents
    # if len(text) < 200:
    #     raise ValueError(f"Texte trop court ({len(text)} chars, minimum 200)")

    if len(text) > 5_000_000:
        raise ValueError(
            f"Texte trop volumineux ({len(text)} chars, max 5M)"
        )


def _preprocess_text(text: str, preserve_formatting: bool) -> str:
    """Preprocess text for article extraction."""
    if preserve_formatting:
        return text

    # Normalize line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Remove excessive blank lines (keep structure)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Normalize spaces (but preserve line breaks)
    lines = text.split('\n')
    lines = [re.sub(r' {2,}', ' ', line.strip()) for line in lines]
    text = '\n'.join(lines)

    return text


def _detect_article_pattern(text: str) -> re.Pattern:
    """Detect which article pattern is most common."""
    pattern_counts = {}

    for pattern_str in ARTICLE_PATTERNS:
        pattern = re.compile(pattern_str, re.IGNORECASE | re.MULTILINE)
        matches = pattern.findall(text)
        pattern_counts[pattern_str] = len(matches)

    # Use most common pattern
    best_pattern_str = max(pattern_counts, key=pattern_counts.get)

    if pattern_counts[best_pattern_str] == 0:
        logger.warning("⚠️ Aucun pattern d'article détecté.")
        return None

    logger.info(
        f"📋 Detected pattern: {best_pattern_str} "
        f"({pattern_counts[best_pattern_str]} matches)"
    )

    return re.compile(best_pattern_str, re.IGNORECASE | re.MULTILINE)



def _split_by_pattern(text: str, pattern: re.Pattern) -> List[Tuple[str, str]]:
    """Split text by article pattern (legacy, without section tracking)."""
    articles = []
    matches = list(pattern.finditer(text))

    for i, match in enumerate(matches):
        number = None
        for group_idx in range(1, len(match.groups()) + 1):
            group_value = match.group(group_idx)
            if group_value is not None:
                number = group_value
                break
        
        if number is None:
            continue

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        articles.append((number, content))

    return articles


def _split_by_pattern_with_sections(text: str, pattern: re.Pattern) -> List[Tuple[str, str, Optional[str]]]:
    """
    Split text by article pattern WITH section tracking.
    
    Detects TITRE and CHAPITRE headers between articles and associates
    each article with its current section.
    
    Returns:
        List of tuples: (article_number, content, section_header)
    """
    articles = []
    
    # Compile section patterns.
    # SANS re.IGNORECASE : le mot-cle doit etre en capitales, comme dans les
    # textes du corpus ("CHAPITRE I - DISPOSITIONS GENERALES"). Avec
    # IGNORECASE, un titre d'article commencant par "Titre 1" ou "Chapitre 2"
    # etait pris pour un en-tete de section — et plus bas, tout le contenu qui
    # suit un en-tete trouve DANS un article est coupe. Un article dont le
    # titre ressemblait a une section perdait donc la totalite de son texte,
    # silencieusement. Le compromis est asymetrique : rater un en-tete en
    # minuscules ne coute qu'une metadonnee de section, le confondre avec un
    # titre coute l'article entier.
    section_pattern = re.compile(
        '|'.join(f'({p})' for p in SECTION_PATTERNS),
        re.MULTILINE
    )
    
    # Find all article matches
    article_matches = list(pattern.finditer(text))
    
    # Track current section
    current_section = None
    
    for i, match in enumerate(article_matches):
        # Extract article number
        number = None
        for group_idx in range(1, len(match.groups()) + 1):
            group_value = match.group(group_idx)
            if group_value is not None:
                number = group_value
                break
        
        if number is None:
            logger.warning(f"⚠️  Could not extract article number from match: {match.group(0)}")
            continue
        
        # Check for section headers BEFORE this article
        # Look in the text between previous article end and this article start
        if i == 0:
            # First article - look from start of text
            search_start = 0
        else:
            # Look from end of previous article
            search_start = article_matches[i - 1].end()
        
        search_end = match.start()
        between_text = text[search_start:search_end]
        
        # Find section headers in between text
        section_matches = list(section_pattern.finditer(between_text))
        if section_matches:
            # Use the LAST section header found (closest to this article)
            last_section_match = section_matches[-1]
            # Extract the matched section text (first non-None group)
            for group_idx in range(1, len(last_section_match.groups()) + 1):
                group_value = last_section_match.group(group_idx)
                if group_value:
                    current_section = group_value.strip()
                    logger.info(f"📑 Section détectée: {current_section}")
                    break
        
        # Article content (from this match to next match or end)
        start = match.end()
        end = article_matches[i + 1].start() if i + 1 < len(article_matches) else len(text)
        content = text[start:end].strip()
        
        # Strip section headers from content (they belong between articles, not in content)
        # Find first section header in content and cut before it
        section_in_content = section_pattern.search(content)
        if section_in_content:
            # Cut content before the section header
            content = content[:section_in_content.start()].strip()
        
        articles.append((number, content, current_section))
    
    return articles


def _extract_title(content: str) -> Optional[str]:
    """
    Extract article title if present.
    
    Titles are typically on first line after article number.
    Example: "Article 1. Dispositions générales\nLa présente loi..."
    
    Returns None if:
    - First line starts with paragraph number like (1), (2)
    - First line is too long (>100 chars) or too short (<5 chars)
    - First line doesn't start with uppercase
    """
    lines = content.split('\n', 1)
    if len(lines) == 0:
        return None

    first_line = lines[0].strip()
    
    # Ignore if line starts with paragraph number like (1), (2), 1), 2), etc.
    if re.match(r'^\(?\d+\)?[.\s)]', first_line):
        return None
    
    # Ignore if line starts with dash, bullet, or other list markers
    if re.match(r'^[-•*–—]\s*', first_line):
        return None
    
    # Ignore if line looks like a sentence continuation (starts with lowercase)
    if first_line and first_line[0].islower():
        return None

    # If first line is short (<100 chars) and not empty, likely title
    if 5 <= len(first_line) <= 100:
        # Check if it looks like a title (short, capitalized)
        if first_line[0].isupper():
            return first_line.rstrip('.')

    return None


def _clean_article_content(content: str, preserve_formatting: bool, title: Optional[str]) -> str:
    """Clean article content."""
    if preserve_formatting:
        return content.strip()

    # Remove title if extracted separately
    if title:
        # Remove first line if it matches the title
        lines = content.split('\n')
        if len(lines) > 1:
            first_line = lines[0].strip().rstrip('.')
            if first_line == title:
                content = '\n'.join(lines[1:])

    return content.strip()


def _get_parent_id(number: str) -> Optional[str]:
    """Determine parent article for hierarchical numbering."""
    # Example: "1.1" → parent="1", "1.2.3" → parent="1.2"

    if '.' not in number:
        return None

    parts = number.split('.')
    if len(parts) > 1:
        return '.'.join(parts[:-1])

    return None
