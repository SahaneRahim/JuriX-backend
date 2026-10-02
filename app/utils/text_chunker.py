"""
Text chunking utilities for legal document processing.

Extracts articles from Cameroonian legal documents following
common patterns: Article X, Art. X, Section X, etc.
"""

import bisect
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ArticleExtractionError(Exception):
    """Raised when article extraction fails."""
    pass


# Pattern constants - COMPREHENSIVE support for French and English variants
# French variants: Article 1, Article 1er, Article premier, Article première, Article deuxième, etc.
# English variants: Section 1, Section one, Section first, Article 1, Article one, etc.
# NOTE: le groupe capturant englobe la numerotation hierarchique complete
# ((\d+(?:\.\d+)*)). Auparavant le '(?:\.\d+)*' etait HORS du groupe :
# 'Article 1.1' et 'Article 1.2' etaient tous deux captures comme '1' et
# fusionnaient avec l'article 1 — une erreur de citation sur une base juridique.
# Prefixe commun a tous les motifs de marqueur d'article.
#
# `(?:^|\n)` ancre en debut de ligne, puis `[#>\-\*_]{0,4}` laisse passer la
# decoration markdown produite par LlamaParse. Le corpus ecrit `**ARTICLE 1er.**-`
# et `**Article 3** :` : l'ancien prefixe `(?:^|\n)\s*Article` echouait sur les
# deux, les `**` s'intercalant entre le saut de ligne et le mot.
#
# `\s` est PROSCRIT ici : il avale les sauts de ligne et ferait correspondre un
# « article 5 » cite en plein paragraphe, ce qui couperait un article en deux a
# chaque renvoi interne.
#
# Mesure sur les 27 lois : 8 avaient au moins un numero reconnu, contre 27
# apres ce changement ; 77 marqueurs contre 378. Sur le seul Code Minier,
# 41 articles indexes contre 193 reellement presents.
_MARKER_PREFIX = r'(?:^|\n)[ \t]*[#>\-\*_]{0,4}[ \t]*'

# Ce qui separe le numero d'article de son texte : « .- », « - », « : », « . ».
_SEPARATEUR_ARTICLE = r'[ \t]*(?:\.?[ \t]*[-–—]|:|\.)'

# Suffixe de « 1er » tel que l'OCR le rend : « 1er », « 1e », « 1r », « 1ºr », « 1° ».
_SUFFIXE_ORDINAL = r'(?:er|ère|ème|e|r|º[ \t]*r|°[ \t]*r|°)?'

# Le mot « Article » tel que l'OCR le rend : « ARTIiCLE », « ARTICLÈ »,
# « ARTlCLE », « ARTICLES 170.- ». Mesure sur le Code minier extrait par
# Docling : 3 articles sur 200 perdaient leur marqueur, et leur texte se
# fondait dans l'article precedent. Le pluriel n'est admis que suivi d'un
# numero ET d'un separateur : « Articles 7 et 8 de la loi » est un renvoi.
_MOT_ARTICLE = (
    r'Art[iíìl1]{1,2}cl[eèéê]'
    r'(?:s(?=[ \t_]*\d+[^\s\d]{0,3}' + _SEPARATEUR_ARTICLE + r'))?'
)

# Docling numerote lui-meme les items de liste : « 2. ARTICLE 41.- », et
# parfois « . ARTICLE 31.- ». Le prefixe est admis A CONDITION qu'un
# separateur suive le numero : « 2. Article 12 de la loi » est une
# enumeration, pas un en-tete. Mesure sur les sorties Docling du banc : 8
# articles dans 3 documents fusionnaient avec le precedent.
_PREFIXE_LISTE = (
    r'(?:(?:\d{1,3}[.)]|\.)[ \t]*'
    r'(?=(?:' + _MOT_ARTICLE + r'|Art\.)[ \t_]*\d+(?:er|ère|ème|e|r)?' + _SEPARATEUR_ARTICLE + r'))?'
)

# Ordinaux en toutes lettres, accentues ou non : l'OCR rend les capitales sans
# accent (« ARTICLE DEUXIEME »), et les lois de finances numerotent bien
# au-dela de « dixieme ». 17 articles de la loi de finances 2016 etaient
# perdus, fondus dans le precedent.
_ORDINAUX = (
    r'premier|premi[èe]re|deuxi[èe]me|second|seconde|troisi[èe]me|quatri[èe]me'
    r'|cinqui[èe]me|sixi[èe]me|septi[èe]me|huiti[èe]me|neuvi[èe]me|dixi[èe]me'
    r'|onzi[èe]me|douzi[èe]me|treizi[èe]me|quatorzi[èe]me|quinzi[èe]me|seizi[èe]me'
    r'|vingti[èe]me|trenti[èe]me|quaranti[èe]me|cinquanti[èe]me|soixanti[èe]me|centi[èe]me'
)

# « Section N » est un ARTICLE dans les textes anglais (« Section 1: This law
# ... »), mais une SUBDIVISION dans les textes francais (« Section 1 : Des
# dispositions generales »), entre le chapitre et les articles. Applique a un
# texte francais, ce motif fabriquait des pseudo-articles : verifie, un code
# numerote 1, 2, 3... ressortait en « 1, 5, 6, 2, 7 ». Ces deux motifs ne
# servent donc qu'aux textes anglais (voir _semantique_anglaise).
_SECTION_ARTICLE = (
    _MARKER_PREFIX + r'Section[ \t]*(?:' +
        r'(\d+(?:\.\d+)*)' +  # Section 1, Section 1.1
        r'|' +
        r'(one|first|two|second|three|third|four|fourth|five|fifth|six|sixth|seven|seventh|eight|eighth|nine|ninth|ten|tenth)' +  # Section one, Section first
    r')\s*[.:\-–]?\s*'
)
_SEC_ARTICLE = _MARKER_PREFIX + r'Sec\.?[ \t]*(\d+(?:\.\d+)*)\s*[.:\-–]?[ \t]*'
_MOTIFS_ARTICLE_ANGLAIS = (_SECTION_ARTICLE, _SEC_ARTICLE)

ARTICLE_PATTERNS = [
    # === FRENCH PATTERNS ===
    # Article + numero (1, 1er, 1.1) OU ordinal en lettres (premier, deuxieme...).
    # « [ \t_]* » : RapidOCR lit le soulignement du mot ARTICLE comme « _ »
    # (« ARTICLE_1er.- », « Article_ 1 : ») ; 12 marqueurs du banc etaient
    # perdus, leurs articles fondus dans la base legale.
    _MARKER_PREFIX + _PREFIXE_LISTE + _MOT_ARTICLE + r'[ \t_]*(?:' +
        r'(\d+(?:\.\d+)*)' + _SUFFIXE_ORDINAL +
        r'|' +
        r'(' + _ORDINAUX + r')' +
    r')\s*[.:\-–]?\s*',

    # Art. (abbreviation) + number OR ordinals
    _MARKER_PREFIX + _PREFIXE_LISTE + r'Art\.?[ \t_]*(?:' +
        r'(\d+(?:\.\d+)*)' + _SUFFIXE_ORDINAL +
        r'|' +
        r'(' + _ORDINAUX + r')' +
    r')\s*[.:\-–]?\s*',
    
    # === ENGLISH PATTERNS ===
    # Section + number OR words (one, first, second, third...)
    _SECTION_ARTICLE,
    
    # Article (English style) + number OR words
    _MARKER_PREFIX + r'Article[ \t]*(?:' +
        r'(\d+(?:\.\d+)*)' +  # Article 1 (English)
        r'|' +
        r'(one|first|two|second|three|third|four|fourth|five|fifth|six|sixth|seven|seventh|eight|eighth|nine|ninth|ten|tenth)' +  # Article one (English)
    r')\s*[.:\-–]?\s*',
    
    # Sec. (abbreviation, English) + number
    _SEC_ARTICLE,

    # === NUMEROTATION CODIFIEE (Code Général des Impôts, CGI) ===
    # "Article L 94 septies.-", "Article L 94", "Art. M 12 bis"
    # Rencontre dans les lois de finances qui modifient le CGI.
    # La lettre de codification est une VRAIE majuscule ((?-i:...)) : sous
    # IGNORECASE, le « s » de « Articles 413 a 419 » passait pour elle, et le
    # chatbot citait « Article s 413 ».
    _MARKER_PREFIX + r'Art(?:icles?|\.)?[ \t]*((?-i:[A-Z])[ \t]*\d+(?:[ \t]+(?:bis|ter|quater|quinquies|'
    r'sexies|septies|octies|novies|decies))?)\s*[.:\-–]?\s*',

    # === ORDINAUX COMPOSES (français) ===
    # "ARTICLE QUATRE-VINGT-SIXIÈME", "Article trente-et-unième"
    # La liste explicite ci-dessus s'arrête à "dixième" ; les lois de finances
    # numérotent leurs articles en toutes lettres bien au-delà.
    _MARKER_PREFIX + r'Article[ \t]*((?:[A-Za-zÀ-ÿ]+-)+[A-Za-zÀ-ÿ]*(?:ièmes?|èmes?|iemes?|emes?))'
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
# Numero d'une subdivision. Le chiffre romain est lu sans egard a la casse
# (« TITRE il » pour « TITRE II »), le mot-cle, lui, reste en capitales. Il peut
# etre colle au mot-cle (« CHAPITREII », « SECTIONI ») : l'OCR mange l'espace.
_NUMERO_SUBDIVISION = (
    r'(?:PREMIER|PREMI[ÈE]RE|UNIQUE|DEUXI[ÈE]ME|TROISI[ÈE]ME|QUATRI[ÈE]ME|CINQUI[ÈE]ME'
    r'|SIXI[ÈE]ME|SEPTI[ÈE]ME|HUITI[ÈE]ME|NEUVI[ÈE]ME|DIXI[ÈE]ME|(?i:[IVXLC]{1,7})|\d+)\b'
)

SECTION_PATTERNS = [
    # TITRE PREMIER - DE L'ÉTAT, TITRE I, TITRE 1, TITREII, TITRE il, etc.
    r'(?:^|\n)\s*(TITRE[ \t]*' + _NUMERO_SUBDIVISION + r'[ \t]*[.:\-–]?[ \t]*[^\n]*)',
    # CHAPITRE PREMIER, CHAPITRE I, CHAPITRE 1, CHAPITREII, etc.
    r'(?:^|\n)\s*(CHAPITRE[ \t]*' + _NUMERO_SUBDIVISION + r'[ \t]*[.:\-–]?[ \t]*[^\n]*)',
    # PART ONE, PART I, PART 1 (English)
    r'(?:^|\n)\s*(PART\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|[IVXLC]{1,7}|\d+)\s*[.:\-–]?\s*[^\n]*)',
    # CHAPTER ONE, CHAPTER I, CHAPTER 1 (English)
    r'(?:^|\n)\s*(CHAPTER\s+(?:ONE|TWO|THREE|FOUR|FIVE|SIX|SEVEN|EIGHT|NINE|TEN|[IVXLC]{1,7}|\d+)\s*[.:\-–]?\s*[^\n]*)',
]

# Sous-division francaise « Section 1 : Des ... », « SECTION II - ... ».
# Ajoutee aux en-tetes des textes francais seulement. Exige un separateur ou
# une fin de ligne apres le numero : « Section 2 du chapitre 3 est modifiee »,
# en debut de ligne dans un article, n'est pas un en-tete et ne doit pas le
# couper.
_SECTION_FRANCAISE = (
    r'(?:^|\n)\s*((?:SOUS-SECTION|SECTION|Section|PARAGRAPHE)[ \t]*' + _NUMERO_SUBDIVISION +
    # separateur puis titre, ou titre en capitales sur la meme ligne
    # (« SECTION II DES DÉFINITIONS »), ou fin de ligne
    r'[ \t]*(?:[.:\-–][^\n]*|[^\na-zà-ÿ]*)(?=\n|$))'
)

# Marqueur de page pose par l'extracteur : `<<PAGE:n>>`, n = page physique.
_PAGE_MARKER = re.compile(r'<<PAGE:(\d+)>>')


class _PageIndex:
    """
    Page physique de chaque position du texte, d'apres les marqueurs.

    La page d'un chunk se lisait sur le PREMIER marqueur trouve DANS son
    contenu. Or le marqueur qui ouvre une page tombe a la fin du chunk qui la
    precede : le dernier article de chaque page recevait la page SUIVANTE
    (« Art.1 p.1 <<PAGE:2>> Art.2 » donnait 2 a l'article 1), et cette page
    fausse partait aussi dans embed_text. La page d'un chunk est desormais
    celle du dernier marqueur place AVANT son debut : la convention des
    citations, un article se cite a la page ou il commence.
    """

    def __init__(self, text: str):
        self._debuts: List[int] = []
        self._pages: List[int] = []
        for m in _PAGE_MARKER.finditer(text):
            self._debuts.append(m.start())
            self._pages.append(int(m.group(1)))

    def page(self, position: int) -> int:
        i = bisect.bisect_right(self._debuts, position) - 1
        return self._pages[i] if i >= 0 else 1


def _premier_contenu(text: str, debut: int, fin: int) -> int:
    """Position du premier caractere qui n'est ni un blanc ni un marqueur de page."""
    i = debut
    while i < fin:
        if text[i].isspace():
            i += 1
            continue
        marqueur = _PAGE_MARKER.match(text, i)
        if marqueur:
            i = marqueur.end()
            continue
        return i
    return debut


def _sans_marqueurs(text: str) -> str:
    return _PAGE_MARKER.sub('', text).strip()


# Valeurs des mots de nombre, sans accent. « second » vaut 2.
_NOMBRES = {
    "un": 1, "une": 1, "premier": 1, "premiere": 1, "deux": 2, "second": 2, "seconde": 2,
    "trois": 3, "quatre": 4, "cinq": 5, "six": 6, "sept": 7, "huit": 8, "neuf": 9,
    "dix": 10, "onze": 11, "douze": 12, "treize": 13, "quatorze": 14, "quinze": 15,
    "seize": 16, "vingt": 20, "trente": 30, "quarante": 40, "cinquante": 50,
    "soixante": 60, "cent": 100,
}
# Radical d'un ordinal -> mot de nombre : « cinquieme » -> cinq, « neuvieme » -> neuf
_RADICAUX = {"cinqu": "cinq", "neuv": "neuf", "un": "un"}

_NOMBRES_ANGLAIS = {
    'one': '1', 'first': '1', 'two': '2', 'second': '2', 'three': '3', 'third': '3',
    'four': '4', 'fourth': '4', 'five': '5', 'fifth': '5', 'six': '6', 'sixth': '6',
    'seven': '7', 'seventh': '7', 'eight': '8', 'eighth': '8', 'nine': '9', 'ninth': '9',
    'ten': '10', 'tenth': '10',
}


def _sans_accents(texte: str) -> str:
    import unicodedata

    return "".join(
        c for c in unicodedata.normalize("NFKD", texte) if not unicodedata.combining(c)
    )


def _ordinal_en_nombre(mot: str) -> Optional[int]:
    """
    Ordinal francais en toutes lettres -> nombre, ou None.

    « deuxieme » -> 2, « trente-et-unieme » -> 31, « quatre-vingt-dix-
    septieme » -> 97, « cent-deuxieme » -> 102. Sans accent, insensible a la
    casse. Un mot inconnu rend None : l'appelant garde alors le texte.
    """
    jetons = [j for j in re.split(r"[-\s]+", _sans_accents(mot).lower()) if j and j != "et"]
    if not jetons:
        return None
    dernier = jetons[-1]
    if dernier in ("premier", "premiere", "second", "seconde"):
        jetons[-1] = dernier
    else:
        radical = re.sub(r"iemes?$|emes?$", "", dernier)
        if radical == dernier:
            return None
        # « seizieme » -> seize, « trentieme » -> trente : le e final tombe
        if radical not in _NOMBRES and radical + "e" in _NOMBRES:
            radical += "e"
        jetons[-1] = _RADICAUX.get(radical, radical)
    total = 0
    for jeton in jetons:
        valeur = _NOMBRES.get(jeton.rstrip("s"))
        if valeur is None:
            return None
        if valeur == 100:
            total = (total or 1) * 100
        elif valeur == 20 and total % 100 == 4:      # quatre-vingt
            total += 76
        else:
            total += valeur
    return total or None


def normalize_article_number(number: str) -> str:
    """
    Normalize article/section numbers to standard format.

    Converts ALL variants to numeric format:
    - French: 'premier', 'deuxieme', 'SEIZIEME', 'quatre-vingt-dix-septieme' -> '1', '2', '16', '97'
    - English: 'one', 'first' -> '1', 'two', 'second' -> '2', etc.
    - Ordinals: '1er', '1ère', '2ème' -> '1', '2', etc.
    - Special: 'PRÉAMBULE', 'PREAMBULE', 'PREAMBLE' -> 'PREAMBULE'
    """
    if not number:
        return number

    lower = number.lower().strip()

    if lower in _NOMBRES_ANGLAIS:
        return _NOMBRES_ANGLAIS[lower]

    # Handle preamble
    if lower in ['préambule', 'preambule', 'preamble']:
        return 'PREAMBULE'

    # Remove French ordinal suffixes (1er, 1ère, 2ème, 3ème...)
    if lower.endswith(('er', 'ère', 'ème')):
        number_clean = re.sub(r'(er|ère|ème)$', '', lower)
        if number_clean.replace('.', '').isdigit():
            return number_clean

    # Ordinal en toutes lettres, simple ou compose
    if re.fullmatch(r"[a-zà-ÿ\s-]+", lower):
        valeur = _ordinal_en_nombre(lower)
        if valeur is not None:
            return str(valeur)

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


def _extract_pre_article_chunks(
    processed_text: str, pattern: re.Pattern, pages: _PageIndex,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Extract legal basis and preamble chunks from pre-article text.

    Returns:
        Tuple of (chunks, next_position)
    """
    assert processed_text, "Processed text must not be empty"
    assert pattern is not None, "Article pattern must be provided"

    premier = pattern.search(processed_text)
    if not premier:
        return [], 0

    fin = premier.start()
    # La vacuite se juge SANS les marqueurs de page. Un texte qui s'ouvre sur
    # « <<PAGE:1>>\nArticle 1er.- » laissait le marqueur seul devant l'article :
    # non vide avant nettoyage, vide apres, et _make_chunk levait
    # AssertionError — avalee plus haut, la loi restait SANS AUCUN article.
    if not _sans_marqueurs(processed_text[:fin]):
        return [], 0

    preamble_match = _find_earliest_preamble(processed_text[:fin])
    if preamble_match:
        chunks = _split_legal_basis_and_preamble(processed_text, fin, preamble_match, pages)
    else:
        chunks = _classify_pre_article_text(processed_text, fin, pages)
    return chunks, len(chunks)


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
    text: str, fin: int, preamble_match: re.Match, pages: _PageIndex,
) -> List[Dict[str, Any]]:
    """Split pre-article text into legal basis and preamble chunks."""
    chunks: List[Dict[str, Any]] = []
    coupure = preamble_match.start()

    # Base legale, si substantielle
    legal_basis = _sans_marqueurs(text[:coupure])
    if len(legal_basis) > 20:
        page = pages.page(_premier_contenu(text, 0, coupure))
        chunks.append(_make_chunk('LEGAL_BASIS', 'Base légale', legal_basis, len(chunks), None, page))

    preambule = _sans_marqueurs(text[coupure:fin])
    if preambule:
        page = pages.page(_premier_contenu(text, coupure, fin))
        chunks.append(_make_chunk('PREAMBULE', 'Préambule', preambule, len(chunks), None, page))

    return chunks


def _classify_pre_article_text(text: str, fin: int, pages: _PageIndex) -> List[Dict[str, Any]]:
    """Classify pre-article text as either preamble or legal basis."""
    clean = _sans_marqueurs(text[:fin])
    if not clean:
        return []
    is_preamble = any(re.match(p, clean, re.IGNORECASE) for p in PREAMBLE_PATTERNS[1:])

    number = 'PREAMBULE' if is_preamble else 'LEGAL_BASIS'
    title = 'Préambule' if is_preamble else 'Base légale'
    page = pages.page(_premier_contenu(text, 0, fin))
    return [_make_chunk(number, title, clean, 0, None, page)]


def _extract_article_chunks(
    processed_text: str, pattern: re.Pattern, pages: _PageIndex,
    start_position: int, min_article_length: int, english: bool,
) -> List[Dict[str, Any]]:
    """Extract article chunks from text using detected pattern."""
    assert processed_text, "Text must not be empty"
    assert min_article_length >= 0, "min_article_length must be non-negative"

    chunks: List[Dict[str, Any]] = []
    position = start_position
    raw_articles = _split_by_pattern_with_sections(processed_text, pattern, english)

    for number, content, section, debut in raw_articles:
        page = pages.page(_premier_contenu(processed_text, debut, len(processed_text)))
        clean_content = _sans_marqueurs(content)
        normalized_number = normalize_article_number(number)
        parent_id = _get_parent_id(normalized_number)

        # _extract_title existait mais n'etait appelee nulle part : le titre
        # etait passe en dur a None, donc AUCUN article n'avait jamais de titre.
        # C'est une perte directe pour les citations affichees a l'utilisateur
        # et pour le contexte envoye au modele.
        title = _extract_title(clean_content)
        clean_content = _clean_article_content(clean_content, False, title)
        char_count = len(clean_content)

        if clean_content and char_count >= min_article_length:
            chunks.append(_make_chunk(
                normalized_number, title, clean_content,
                position, section, page, parent_id,
            ))
            position += 1
        else:
            logger.warning(
                f"⚠️  Article {normalized_number} trop court "
                f"({char_count} chars), ignoré"
            )

    return chunks


def _extract_paragraph_chunks(
    processed_text: str, pages: _PageIndex, min_article_length: int,
) -> List[Dict[str, Any]]:
    """Extract paragraph chunks from documents without articles."""
    assert processed_text, "Text must not be empty"
    assert min_article_length >= 0, "min_article_length must be non-negative"

    # Paragraphes avec leur position, pour lire la page de chacun. Un
    # paragraphe reduit a un marqueur de page n'en est pas un.
    paragraphs: List[Tuple[int, str]] = []
    debut = 0
    for separateur in re.finditer(r'\n\s*\n', processed_text):
        paragraphs.append((debut, processed_text[debut:separateur.start()]))
        debut = separateur.end()
    paragraphs.append((debut, processed_text[debut:]))
    paragraphs = [(d, p) for d, p in paragraphs if _sans_marqueurs(p)]

    chunks: List[Dict[str, Any]] = []
    if len(paragraphs) > 1:
        for i, (debut, paragraph) in enumerate(paragraphs, start=1):
            clean_para = _sans_marqueurs(paragraph)
            if len(clean_para) >= min_article_length:
                page = pages.page(_premier_contenu(processed_text, debut, debut + len(paragraph)))
                chunks.append(_make_chunk(
                    f'PARA_{i}', f'Paragraphe {i}', clean_para,
                    len(chunks), None, page,
                ))
    elif paragraphs:
        logger.info("📄 Texte continu - stockage en un seul chunk")
        page = pages.page(_premier_contenu(processed_text, 0, len(processed_text)))
        chunks.append(_make_chunk(
            'FULL_TEXT', 'Document complet', _sans_marqueurs(processed_text),
            0, None, page,
        ))

    return chunks


def _semantique_anglaise(text: str, language: Optional[str]) -> bool:
    """
    « Section N » est-il un article (texte anglais) ou une subdivision ?

    La langue, quand l'appelant la connait, tranche. Sinon : un texte qui porte
    au moins un marqueur « Article N » ou « Art. N » est traite en francais.
    """
    if language:
        return language.lower().startswith("en")
    for motif in ARTICLE_PATTERNS[:2]:
        if re.search(motif, text, re.IGNORECASE | re.MULTILINE):
            return False
    return True


def extract_articles(
    text: str,
    min_article_length: int = 10,
    preserve_formatting: bool = False,
    strict: bool = True,
    language: Optional[str] = None,
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
        language: Langue du document ("fr", "en"), si connue. Decide du role
            de « Section N » : article en anglais, subdivision en francais.

    Returns:
        List of chunk dicts with standard keys (number, title, content, etc.)

    Raises:
        ValueError: If text is empty or too large
        ArticleExtractionError: If no content could be extracted
    """
    # 1. Validate input
    assert isinstance(text, str), "text must be a string"
    if not text or not _sans_marqueurs(text):
        raise ValueError("Le texte ne peut pas être vide")
    if len(text) > 5_000_000:
        raise ValueError(f"Texte trop volumineux ({len(text)} chars, max 5M)")

    # 2. Preprocess and detect pattern
    processed_text = _preprocess_text(text, preserve_formatting)
    english = _semantique_anglaise(processed_text, language)
    pattern = _detect_article_pattern(processed_text, english)
    pages = _PageIndex(processed_text)

    # 3. Extract chunks based on document structure
    if pattern:
        logger.info("📋 Document avec articles détecté")
        pre_chunks, position = _extract_pre_article_chunks(processed_text, pattern, pages)
        article_chunks = _extract_article_chunks(
            processed_text, pattern, pages, position, min_article_length, english
        )
        chunks = pre_chunks + article_chunks
    else:
        logger.info("📄 Document sans articles - extraction par paragraphes")
        chunks = _extract_paragraph_chunks(processed_text, pages, min_article_length)

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


# Part du meilleur motif au-dela de laquelle un motif secondaire est conserve.
# 20 % : assez bas pour rattraper les documents qui melangent deux conventions
# ("Article 1er" puis "Art. 2", frequent dans les lois de finances), assez haut
# pour ne pas retenir un motif qui ne touche qu'une ligne isolee — un faux
# positif coupe un article en deux, ce qui coute plus cher qu'un motif manque.
_SECONDARY_PATTERN_RATIO = 0.20


def _detect_article_pattern(text: str, english: bool = True) -> Optional[re.Pattern]:
    """
    Compose l'expression de detection des articles a partir du texte.

    Auparavant un SEUL motif etait retenu, le plus frequent (`max()`). Un
    document melangeant deux conventions perdait toutes les occurrences du motif
    perdant : elles n'etaient pas detectees comme articles, donc absorbees dans
    le chunk precedent ou, si elles precedaient le premier article reconnu, dans
    LEGAL_BASIS. Reproduit sur un document ouvrant par "Article 1.-" puis
    poursuivant en "Art. 2.-", "Art. 3.-", "Art. 4.-" : l'article 1 disparaissait
    en tant qu'article.

    Tous les motifs atteignant _SECONDARY_PATTERN_RATIO du meilleur sont donc
    fusionnes en une alternance. Chaque branche garde ses propres groupes de
    capture et l'appelant lit le premier groupe non nul, donc l'alternance ne
    change pas la lecture du numero.

    Dans un texte francais, « Section N » est une subdivision : ses motifs ne
    sont pas candidats (voir _SECTION_ARTICLE).
    """
    candidats = [
        p for p in ARTICLE_PATTERNS if english or p not in _MOTIFS_ARTICLE_ANGLAIS
    ]
    counts = {
        pattern_str: len(
            re.compile(pattern_str, re.IGNORECASE | re.MULTILINE).findall(text)
        )
        for pattern_str in candidats
    }

    best = max(counts.values())
    if best == 0:
        logger.warning("⚠️ Aucun pattern d'article détecté.")
        return None

    seuil = max(1, best * _SECONDARY_PATTERN_RATIO)
    retenus = [p for p in candidats if counts[p] >= seuil]

    logger.info(
        f"📋 {len(retenus)} motif(s) retenu(s) sur {len(candidats)}, "
        f"{sum(counts[p] for p in retenus)} occurrence(s)"
    )

    # Les branches sont deja parenthesees et sans ancrage mutuellement exclusif ;
    # l'alternance les essaie dans l'ordre de ARTICLE_PATTERNS.
    return re.compile("|".join(retenus), re.IGNORECASE | re.MULTILINE)


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


def _renvoi_en_liste(match: re.Match) -> bool:
    """
    Une puce, puis un numero SANS separateur : un renvoi, pas un marqueur.

    « - article 12 de la loi n° 2016/017 ; » dans une liste d'abrogations
    coupait l'article en cours et inventait un article 12. Un marqueur que
    Docling rend en element de liste porte toujours son separateur :
    « - ARTICLE 12.- (1) ... ».
    """
    texte = match.string
    debut = match.start() + len(match.group(0)) - len(match.group(0).lstrip("\r\n"))
    fin = texte.find("\n", debut)
    ligne = texte[debut:fin if fin != -1 else len(texte)]
    # Une puce suivie d'un blanc : « **Article 3** : » est une emphase
    puce = re.match(r"[ \t]*[-*•][ \t]+", ligne)
    if not puce:
        return False
    return not re.match(
        r"\S+[ \t_]*(?:\d+[^\s\d]{0,3}|[A-Za-zÀ-ÿ]{4,})[ \t]*[.:\-–—]", ligne[puce.end():]
    )


def _split_by_pattern_with_sections(
    text: str, pattern: re.Pattern, english: bool = True
) -> List[Tuple[str, str, Optional[str], int]]:
    """
    Split text by article pattern WITH section tracking.

    Detects TITRE and CHAPITRE headers between articles and associates
    each article with its current section. In French texts, « Section N »
    sub-divisions are section headers too.

    Returns:
        List of tuples: (article_number, content, section_header, start), where
        `start` is the position of the chunk in `text`, used to read its page.
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
    motifs_section = SECTION_PATTERNS if english else SECTION_PATTERNS + [_SECTION_FRANCAISE]
    section_pattern = re.compile(
        '|'.join(f'({p})' for p in motifs_section),
        re.MULTILINE
    )
    
    # Find all article matches
    article_matches = [m for m in pattern.finditer(text) if not _renvoi_en_liste(m)]
    
    # Track current section
    current_section = None

    # Numerotation des chapeaux de section emis (cf. plus bas). Compteur propre
    # pour ne pas entrer en collision avec la numerotation des articles.
    section_seq = 1

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

        # Un en-tete de section trouve DANS le contenu marque la fin de
        # l'article : ce qui suit appartient a la nouvelle section.
        section_in_content = section_pattern.search(text, start, end)
        chapeau = ""
        if section_in_content:
            chapeau = text[section_in_content.end():end].strip()
            content = text[start:section_in_content.start()].strip()
        else:
            content = text[start:end].strip()

        articles.append((number, content, current_section, match.start()))

        # Le chapeau de section etait purement SUPPRIME : le texte situe entre
        # l'en-tete TITRE/CHAPITRE et l'article suivant n'etait rattache a
        # personne — ni a l'article precedent, borne par cet en-tete, ni au
        # suivant, dont le contenu ne commence qu'a son propre motif. Sur les
        # codes, ou chaque titre s'ouvre par un paragraphe de portee, et sur les
        # annexes introduites par un CHAPITRE, la fin du document disparaissait
        # entierement et sans trace.
        #
        # Il est emis comme chunk distinct plutot que colle a un article : il
        # n'appartient a aucun des deux, et le rattacher fausserait la citation.
        # Le prefixe SECTION_ suit la convention des chunks non-articles deja en
        # place (LEGAL_BASIS, PARA_n) ; chunk_refiner le classe ensuite.
        if chapeau:
            articles.append((
                f"SECTION_{section_seq}",
                chapeau,
                section_in_content.group(0).strip(),
                section_in_content.end(),
            ))
            section_seq += 1

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
    """
    Nettoie le contenu d'un article.

    Le titre est COPIE dans le champ `title`, jamais retire du contenu.

    Il l'etait auparavant, et c'etait une perte visible pour l'utilisateur.
    `_extract_title` accepte toute premiere ligne de 5 a 100 caracteres
    commencant par une majuscule : sur du texte extrait, la premiere PHRASE
    normative d'un article remplit tres souvent ce critere. Elle quittait alors
    le contenu pour un champ que rien n'affiche — aucun endpoint ne renvoie
    `Article.title` (`LawDetailResponse` n'est utilise nulle part, et
    `ArticleResponse` ne porte pas le champ), et `search_service` construit ses
    extraits depuis `content` seul. Cas reproduit : "Monsieur X, matricule
    765 609-Y", premiere ligne d'une liste nominative, disparaissait de
    l'application.

    Dupliquer le titre dans les deux champs coute quelques dizaines d'octets par
    article ; le retirer coutait une phrase de texte juridique.
    """
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
