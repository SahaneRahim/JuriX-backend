"""
Primitives lexicales partagees : mots vides, repli des accents, tokenisation.

Ces trois fonctions sont utilisees a la fois par le service RAG (extraction de
mots-cles) et par le re-ranking (traits lexicaux). Les regrouper ici evite deux
listes de mots vides qui divergeront.

Author: JuriX Team
"""

import re
import unicodedata
from typing import Set

# Mots vides francais et anglais. Deplaces depuis RAGService.STOPWORDS, dont ils
# etaient l'unique definition.
STOPWORDS: Set[str] = {
    # Français
    "le", "la", "les", "un", "une", "des", "de", "du", "d", "l",
    "ce", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes",
    "son", "sa", "ses", "notre", "nos", "votre", "vos", "leur", "leurs",
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "me", "te", "se", "lui", "y", "en", "qui", "que", "quoi", "dont", "où",
    "et", "ou", "mais", "donc", "or", "ni", "car", "si", "quand", "comme",
    "à", "au", "aux", "avec", "sans", "sous", "sur", "dans", "par", "pour",
    "est", "sont", "être", "avoir", "fait", "faire", "dit", "dire",
    "c", "qu", "n", "s", "m", "t",
    "moi", "toi", "soi", "eux",
    "ne", "pas", "plus", "moins", "très", "bien", "mal",
    "tout", "tous", "toute", "toutes", "autre", "autres",
    "quel", "quelle", "quels", "quelles",
    "comment", "pourquoi", "combien",
    "explique", "expliquer", "donne", "donner", "dis", "parle", "parler",
    # English
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "can",
    "i", "you", "he", "she", "it", "we", "they", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
    "and", "or", "but", "if", "then", "so", "because",
    "of", "to", "in", "on", "at", "by", "for", "with", "about", "from",
    "explain", "tell", "give", "show", "describe",
}

_WORD = re.compile(r"\w+", re.UNICODE)


def fold_accents(text: str) -> str:
    """
    Replie les accents et met en minuscules.

    Les questions d'utilisateurs sont ecrites sans accents aussi souvent
    qu'avec : "responsabilite" doit rencontrer "responsabilité".
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def content_words(text: str, min_length: int = 3) -> Set[str]:
    """Mots de contenu, accents replies, mots vides et mots courts retires."""
    if not text:
        return set()
    return {
        word
        for word in _WORD.findall(fold_accents(text))
        if len(word) > min_length and word not in STOPWORDS
    }


def jaccard(left: Set[str], right: Set[str]) -> float:
    """Recoupement de Jaccard. 0.0 si l'un des deux ensembles est vide."""
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
