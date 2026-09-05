"""
Reconnaissance d'une reference d'article dans une requete.

« article 35 du code minier » doit ouvrir le Code Minier sur son article 35.
Cela suppose trois choses : reconnaitre le numero, reconnaitre le document, et
normaliser le numero sous la forme reellement stockee.

CE MODULE EXISTE PARCE QUE LA LOGIQUE ETAIT ECRITE HUIT FOIS. Un inventaire du
depot trouve des motifs « article N » dans rag_service (trois : detection dans
la question, CITATION_REGEX sur la reponse du modele, normalisation),
search_service, reranker, legal_domain_classifier, chunk_refiner et
text_chunker — tous legerement differents. Celui du RAG refuse « art.35 » faute
d'espace ; celui du reranker est le seul a accepter le prefixe L/R/D. Ces ecarts
ne se voient jamais a la lecture, seulement a l'usage.

Author: JuriX Team
"""

import re
from dataclasses import dataclass
from typing import Optional

# Ordinaux ecrits en toutes lettres. `text_chunker.normalize_article_number`
# ecrit "1" en base pour « Article premier » : sans cette table, une requete
# « article premier du code minier » produirait la cible "PREMIER", qui ne
# correspond a aucune ligne.
_ORDINAL_WORDS = {
    "PREMIER": "1", "PREMIERE": "1", "PREMIÈRE": "1", "1ER": "1", "1ÈRE": "1",
    "1ERE": "1", "FIRST": "1",
    "DEUXIEME": "2", "DEUXIÈME": "2", "SECOND": "2", "SECONDE": "2",
    "TROISIEME": "3", "TROISIÈME": "3", "THIRD": "3",
}

_LEADING = re.compile(r"^(ARTICLE|ART\.?|SECTION)\s*", re.IGNORECASE)

# Motifs essayes DANS L'ORDRE. Les deux premiers exigent une mention de
# document ; les deux derniers acceptent un numero seul. L'ordre compte : sans
# lui, « article 35 du code minier » serait capte par le motif « numero seul »,
# qui jetterait « du code minier ».
_PATTERNS = (
    # « article 35 du code minier », « art.35 de la constitution »
    re.compile(
        r"\bart(?:icle|\.)?\s*([LRD]\s*)?(\d+(?:[-.]\d+)*(?:\s*(?:bis|ter|quater))?)"
        r"\s+(?:de\s+la\s+|de\s+l['’]\s*|du\s+|de\s+|des\s+)(.+)",
        re.IGNORECASE,
    ),
    # « article premier de la constitution »
    re.compile(
        r"\bart(?:icle|\.)?\s+(premier|première|1er|1ère)"
        r"\s+(?:de\s+la\s+|de\s+l['’]\s*|du\s+|de\s+|des\s+)(.+)",
        re.IGNORECASE,
    ),
    # « article 35 » seul — aucune mention de document
    re.compile(
        r"\bart(?:icle|\.)?\s*([LRD]\s*)?(\d+(?:[-.]\d+)*)\s*$", re.IGNORECASE
    ),
    re.compile(r"\bart(?:icle|\.)?\s+(premier|première|1er|1ère)\s*$", re.IGNORECASE),
)


@dataclass(frozen=True)
class ArticleReference:
    """Numero demande, et mention du document si elle est presente."""

    number: str
    doc_hint: str = ""

    @property
    def has_hint(self) -> bool:
        return bool(self.doc_hint.strip())


def normalize_number(number: Optional[str]) -> str:
    """
    Ramene un numero d'article a la forme stockee dans `articles.number`.

    « Article 1er », « premier », « 1ER » et « 1 » designent le meme article.
    """
    if not number:
        return ""
    cleaned = _LEADING.sub("", str(number).strip().upper())
    cleaned = cleaned.strip(" .:-")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return _ORDINAL_WORDS.get(cleaned, cleaned)


def parse_reference(query: str) -> Optional[ArticleReference]:
    """
    Extrait la reference d'article d'une requete, ou None.

    >>> parse_reference("article 35 du code minier")
    ArticleReference(number='35', doc_hint='code minier')
    >>> parse_reference("art.35")
    ArticleReference(number='35', doc_hint='')
    >>> parse_reference("le regime des permis de recherche") is None
    True
    """
    if not query:
        return None
    text = query.strip()

    for index, pattern in enumerate(_PATTERNS):
        match = pattern.search(text)
        if not match:
            continue
        groups = match.groups()
        if index == 0:
            prefix, number, hint = groups
            raw = f"{prefix.strip()} {number}".strip() if prefix else number
            return ArticleReference(normalize_number(raw), hint.strip(" .,;:"))
        if index == 1:
            number, hint = groups
            return ArticleReference(normalize_number(number), hint.strip(" .,;:"))
        if index == 2:
            prefix, number = groups
            raw = f"{prefix.strip()} {number}".strip() if prefix else number
            return ArticleReference(normalize_number(raw))
        return ArticleReference(normalize_number(groups[0]))

    return None
