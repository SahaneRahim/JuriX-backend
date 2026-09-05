"""
Nettoyage du markdown produit par l'extraction de PDF.

Ces regles ne dependent d'AUCUN fournisseur : elles decrivent ce que le corpus
prc.cm contient et qu'il ne faut pas indexer. Elles vivaient dans
`llama_parse_service.py` et auraient disparu avec lui ; elles ont ete calibrees
sur ce corpus et servent tout autant a l'extraction par Gemini.

Trois regles, chacune payee par une observation :

- Le cachet « COPIE CERTIFIEE CONFORME » est appose sur CHAQUE page. Restitue en
  texte, il pollue le plein texte et les vecteurs de tous les documents.
- Le filigrane diagonal `www.prc.cm` ressort eclate caractere par caractere.
- `ARTICLE 1<sup>ER</sup>` doit redevenir `ARTICLE 1ER`, faute de quoi
  `text_chunker.ARTICLE_PATTERNS` ne reconnait pas le marqueur — mais `<table>`
  n'est JAMAIS retire : les annexes budgetaires portent leur sens dans leur
  structure tabulaire.

Author: JuriX Team
"""

import re

# Vocabulaire du cachet "COPIE CERTIFIEE CONFORME" appose sur chaque page du
# corpus prc.cm. L'extracteur le restitue en texte, il faut donc le retirer avant chunking sinon il pollue le FTS
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

# Balisage HTML inline emis par l'extracteur (les <table> sont conserves : la
# structure tabulaire porte du sens, cf. les annexes budgetaires).
# <sup>/<sub> sont critiques : "ARTICLE 1<sup>ER</sup>" doit redevenir
# "ARTICLE 1ER" pour que ARTICLE_PATTERNS de text_chunker le reconnaisse.
_INLINE_TAGS = re.compile(
    r"</?(?:u|mark|b|i|em|strong|span|sup|sub|small)\b[^>]*>", re.IGNORECASE
)


def strip_stamp_blocks(text: str) -> str:
    """
    Retire les blocs de cachet officiel du markdown.

    Un bloc = au moins 2 lignes consecutives (blancs autorises) appartenant au
    vocabulaire du cachet. Le seuil de 2 evite de supprimer un en-tete legitime :
    beaucoup de documents portent "PRESIDENCE DE LA REPUBLIQUE" comme autorite
    emettrice, ce qui est du contenu, pas du tampon.

    Args:
        text: Markdown brut d'extraction

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
