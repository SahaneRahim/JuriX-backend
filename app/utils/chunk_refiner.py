"""
Raffinage des chunks pour le RAG juridique camerounais.

Couche de post-traitement appliquee APRES text_chunker.extract_articles().
Le chunker existant decoupe correctement par article ; ce module regle ce qui
se passe autour, et qui determine la qualite des reponses du chatbot.

Constat qui motive ce module — composition mesuree du corpus prc.cm (1883 docs) :

    nominatif (listes de noms, avancements)   52,6 %
    ratification / emprunt (1-3 articles)     19,4 %
    NORMATIF (contenu juridique reel)         28,0 %

Un chunking naif produit donc l'essentiel de ses chunks a partir des 52,6 % qui
ne repondront jamais a une question de droit — et comme toutes les listes de noms
s'embeddent au meme endroit de l'espace vectoriel, elles saturent le top-k de
chaque recherche semantique.

Regles appliquees :
  R2  contextualisation      en-tete document prepose a chaque chunk (embed_text)
  R3  visas hors index       LEGAL_BASIS sorti de l'index + graphe de citations
  R4  listes nominatives     1 chunk normatif + entrees en table separee
  R5  tableaux entiers       jamais coupes au milieu d'une ligne
  R6  tailles                decoupe aux alineas, boilerplate hors index
  R7  deduplication          chunks identiques fusionnes

Principe directeur : RIEN N'EST SUPPRIME. Les chunks ecartes du vectoriel
gardent `embed=False` et restent cherchables en FTS exact. Sur une base
juridique, ne jamais perdre de contenu — seulement le hierarchiser.

Author: JuriX Team
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ==================== SEUILS ====================

# ~800 tokens. Au-dela on coupe aux alineas (1), (2), (3)...
TARGET_MAX_CHARS = 3000
# En-deca, un chunk isole n'est pas repondable seul
MIN_CHARS = 120
# Proportion de lignes "nominatives" a partir de laquelle un tableau est un roster
ROSTER_ROW_RATIO = 0.6
ROSTER_MIN_ROWS = 5


# ==================== MOTIFS ====================

# Matricule administratif camerounais : "765 609-Y", "699 536-N"
_MATRICULE = re.compile(r"\b\d{3}\s?\d{3}\s?-\s?[A-Z]\b")

# Nom en capitales : au moins deux mots, apostrophes et tirets admis
_CAPS_NAME = re.compile(r"^[A-ZÀ-Þ][A-ZÀ-Þ'’\-\.]*(?:\s+[A-ZÀ-Þ][A-ZÀ-Þ'’\-\.]*){1,6}$")

# Tableaux HTML produits par LlamaParse
_TABLE_BLOCK = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
_TR_BLOCK = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_TD_CELL = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_ANY_TAG = re.compile(r"<[^>]+>")

# Liste nominative en texte brut : "20. ADAMA ADAM 765 645-M"
_PLAIN_ROSTER_LINE = re.compile(
    r"^\s*(\d{1,4})\s*[.)-]\s+(.{3,80}?)\s+(\d{3}\s?\d{3}\s?-\s?[A-Z])\s*$"
)

# Vrai numero d'article : "1", "1er", "12.3", "L 94 septies", "QUATRE-VINGT-SIXIEME".
# Exclut PARA_n / LEGAL_BASIS / PREAMBULE, qui sont des replis internes du chunker.
_REAL_ARTICLE_NUMBER = re.compile(
    r"^(?:\d+(?:er|ere|eme|ème|ère)?(?:\.\d+)*"
    r"|[A-Z]\s*\d+(?:\s+\w+)?"
    r"|(?:[A-Za-zÀ-ÿ]+-)+[A-Za-zÀ-ÿ]*(?:ièmes?|èmes?|iemes?|emes?))$",
    re.IGNORECASE,
)

# Alineas numerotes d'un article : "(1)", "(2)" en debut de ligne
_ALINEA = re.compile(r"(?=^\s*\((\d{1,2})\)\s)", re.MULTILINE)

# Visas : "Vu la Constitution ;", "Vu le decret n° 2011/412 du 09 decembre 2011"
_VISA_LINE = re.compile(r"^\s*Vu\s+(.{5,300}?)\s*;?\s*$", re.IGNORECASE | re.MULTILINE)
# Reference d'un texte cite dans un visa
_CITED_REF = re.compile(
    r"\b(loi|d[ée]cret|arr[êe]t[ée]|ordonnance|d[ée]cision|circulaire)\s+"
    r"(?:constitutionnelle\s+)?n[°o]\s*([\d./\-]+)",
    re.IGNORECASE,
)

# Articles d'execution, presents dans quasiment chaque texte du corpus.
# Ils restent en base et en FTS, mais hors index vectoriel : indexes, ils
# produisent des milliers de quasi-doublons qui ecrasent la similarite cosinus.
_BOILERPLATE = re.compile(
    r"(sera enregistr[ée]|publi[ée]\s+(?:selon|au)\s+(?:la\s+proc[ée]dure|Journal)"
    r"|ins[ée]r[ée]\s+au\s+Journal\s+Officiel"
    r"|shall be registered|published in the Official Gazette"
    r"|abrog[ée]e?s?\s+toutes\s+dispositions\s+ant[ée]rieures"
    r"|entre\s+en\s+vigueur\s+[àa]\s+compter\s+de\s+la\s+date\s+de\s+sa\s+signature)",
    re.IGNORECASE,
)


# ==================== STRUCTURES ====================


@dataclass
class DocumentContext:
    """Metadonnees du document, issues de la passe d'extraction page 1."""

    reference: str
    title: str
    doc_type: Optional[str] = None
    date: Optional[str] = None
    category: Optional[str] = None
    language: str = "fr"

    def header(self, page: Optional[int] = None, section: Optional[str] = None) -> str:
        """
        Construit l'en-tete prepose a chaque chunk (regle R2).

        Sans lui, "Article 3.- La depense resultant des presentes dispositions
        sera imputee sur le budget de l'Etat" est un chunk orphelin : ni le
        lecteur ni l'embedding ne savent de quelle depense il s'agit.
        """
        lines = [f"{self.reference}" if self.reference else ""]
        if self.date:
            lines[0] = f"{lines[0]} du {self.date}".strip()
        if self.title:
            lines.append(self.title)
        meta = [m for m in (self.category, self.language, section) if m]
        if page:
            meta.append(f"page {page}")
        if meta:
            lines.append(" · ".join(str(m) for m in meta))
        return "\n".join(ln for ln in lines if ln).strip()


@dataclass
class RosterEntry:
    """Une ligne de liste nominative, stockee hors index vectoriel."""

    article_number: str
    position: int
    name: str
    identifier: Optional[str] = None
    rank: Optional[str] = None


@dataclass
class RefinedChunks:
    """Resultat du raffinage."""

    chunks: List[Dict[str, Any]] = field(default_factory=list)
    legal_basis: Optional[str] = None
    citations: List[str] = field(default_factory=list)
    roster: List[RosterEntry] = field(default_factory=list)
    stats: Dict[str, Any] = field(default_factory=dict)

    @property
    def embeddable(self) -> List[Dict[str, Any]]:
        """Chunks a vectoriser (les seuls qui coutent des appels API)."""
        return [c for c in self.chunks if c.get("embed")]


# ==================== NORMALISATION AVANT CHUNKING ====================

# Emphase markdown en debut de ligne : "**ARTICLE 1ER**:" empeche
# ARTICLE_PATTERNS de reconnaitre l'article (le motif attend "Article" en
# debut de ligne, pas "**Article").
_MD_EMPHASIS = re.compile(r"(?<![\w*_])(\*{1,3}|_{1,3})(?=\S)(.+?)(?<=\S)\1(?![\w*_])")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)
_SPACED_CAPS = re.compile(r"\b((?:[A-ZÀ-Þ]\s){2,}[A-ZÀ-Þ])\b")


def normalize_for_chunking(text: str) -> str:
    """
    Prepare le markdown OCR pour text_chunker.extract_articles().

    A appeler AVANT extract_articles(). Sans cette passe, l'extraction echoue
    et retombe sur un decoupage par paragraphes (PARA_1, PARA_2...), ce qui
    fait perdre le numero d'article — donc toute possibilite de citation.

    Trois normalisations, toutes constatees sur des sorties LlamaParse reelles :
      - "**ARTICLE 1ER**:"        -> "ARTICLE 1ER:"     (emphase markdown)
      - "# A R R Ê T E:"          -> "ARRÊTE:"          (titre + lettres espacees)
      - "Article 1<sup>ER</sup>"  -> deja traite en amont par le service OCR

    Args:
        text: Markdown issu de l'OCR

    Returns:
        Texte normalise, marqueurs <<PAGE:n>> et tableaux <table> preserves
    """
    if not text:
        return text
    out = _MD_HEADING.sub("", text)
    out = _MD_EMPHASIS.sub(r"\2", out)
    # "A R R Ê T E" -> "ARRÊTE" (l'OCR restitue l'interlettrage des titres)
    out = _SPACED_CAPS.sub(lambda m: m.group(1).replace(" ", ""), out)
    return out


# ==================== HELPERS TABLEAUX ====================


def _cell_text(html: str) -> str:
    """Texte nu d'une cellule HTML."""
    return _ANY_TAG.sub("", html).replace("&nbsp;", " ").strip()


def _table_rows(table_html: str, skip_header: bool = True) -> List[List[str]]:
    """
    Extrait les lignes d'un tableau HTML sous forme de listes de cellules.

    Args:
        table_html: Bloc <table>...</table>
        skip_header: Ignore les lignes d'en-tete (<th>), sinon "Nom"/"Indice"
                     seraient comptes comme des personnes du roster.
    """
    rows = []
    for tr in _TR_BLOCK.findall(table_html):
        if skip_header and re.search(r"<th\b", tr, re.IGNORECASE):
            continue
        cells = [_cell_text(td) for td in _TD_CELL.findall(tr)]
        if any(cells):
            rows.append(cells)
    return rows


def _table_to_text(table_html: str) -> str:
    """
    Aplatit un tableau HTML en texte delimite, pour l'embedding.

    Le balisage <table>/<tr>/<td> n'apporte rien a un vecteur et dilue le
    signal semantique. On garde le HTML dans `content` (affichage, citation)
    et on vectorise cette version texte.
    """
    lines = []
    for tr in _TR_BLOCK.findall(table_html):
        cells = [_cell_text(td) for td in _TD_CELL.findall(tr)]
        if any(cells):
            lines.append(" | ".join(cells))
    return "\n".join(lines)


def _looks_nominative(rows: List[List[str]]) -> bool:
    """
    Le tableau est-il une liste de personnes ?

    Critere : au moins ROSTER_ROW_RATIO des lignes portent soit un matricule,
    soit un nom entierement en capitales de 2 mots ou plus.
    """
    if len(rows) < ROSTER_MIN_ROWS:
        return False

    hits = 0
    for cells in rows:
        joined = " ".join(cells)
        if _MATRICULE.search(joined):
            hits += 1
            continue
        if any(_CAPS_NAME.match(c) for c in cells if len(c) > 4):
            hits += 1
    return hits / len(rows) >= ROSTER_ROW_RATIO


def _parse_roster_rows(
    rows: List[List[str]], article_number: str, start: int = 0
) -> List[RosterEntry]:
    """Convertit les lignes d'un tableau nominatif en RosterEntry."""
    entries: List[RosterEntry] = []
    for cells in rows:
        joined = " ".join(cells)
        mat = _MATRICULE.search(joined)
        identifier = mat.group(0).strip() if mat else None

        name = None
        for c in cells:
            if len(c) > 4 and _CAPS_NAME.match(c):
                name = c
                break
        if not name:
            # repli : la cellule la plus longue qui n'est ni le rang ni le matricule
            candidates = [
                c for c in cells
                if len(c) > 4 and not _MATRICULE.fullmatch(c.strip())
                and not re.fullmatch(r"\d{1,4}\s*[.)-]?", c.strip())
            ]
            name = max(candidates, key=len) if candidates else None
        if not name:
            continue

        rank = next(
            (c.strip(" .)-") for c in cells if re.fullmatch(r"\d{1,4}\s*[.)-]?", c.strip())),
            None,
        )
        entries.append(
            RosterEntry(
                article_number=article_number,
                position=start + len(entries),
                name=name.strip(),
                identifier=identifier,
                rank=rank,
            )
        )
    return entries


def _parse_plain_roster(text: str, article_number: str) -> Tuple[List[RosterEntry], str]:
    """
    Variante texte brut : "20. ADAMA ADAM 765 645-M" sur des lignes successives.

    Returns:
        (entrees, texte sans les lignes nominatives)
    """
    entries: List[RosterEntry] = []
    kept: List[str] = []
    for line in text.splitlines():
        m = _PLAIN_ROSTER_LINE.match(line)
        if m:
            entries.append(
                RosterEntry(
                    article_number=article_number,
                    position=len(entries),
                    name=m.group(2).strip(),
                    identifier=m.group(3).strip(),
                    rank=m.group(1),
                )
            )
        else:
            kept.append(line)

    if len(entries) < ROSTER_MIN_ROWS:
        return [], text
    return entries, "\n".join(kept)


# ==================== REGLES ====================


def _apply_roster_rule(
    chunk: Dict[str, Any]
) -> Tuple[Dict[str, Any], List[RosterEntry]]:
    """
    R4 — Effondre une liste nominative en un seul chunk normatif.

    Un arrete nommant 904 inspecteurs ne doit pas produire 904 vecteurs :
    ils sont semantiquement indiscernables et satureraient toutes les
    recherches. On garde l'enveloppe juridique (qui, promu a quoi, a quel
    indice, a compter de quand) et on renvoie les personnes en table a part,
    cherchables en FTS exact.
    """
    content = chunk["content"]
    number = str(chunk.get("number", ""))
    entries: List[RosterEntry] = []
    stripped = content

    # Variante tableau HTML (sortie LlamaParse standard)
    for table_html in _TABLE_BLOCK.findall(content):
        rows = _table_rows(table_html)
        if not _looks_nominative(rows):
            continue
        body = [r for r in rows if not _MATRICULE.search(" ".join(r)) or True]
        entries.extend(_parse_roster_rows(body, number, start=len(entries)))
        stripped = stripped.replace(table_html, f"[[ROSTER:{len(entries)}]]")

    # Variante texte brut
    if not entries:
        entries, candidate = _parse_plain_roster(content, number)
        if entries:
            stripped = candidate + f"\n\n[[ROSTER:{len(entries)}]]"

    if not entries:
        return chunk, []

    placeholder = f"[{len(entries)} personnes concernées — liste nominative complète en annexe]"
    chunk = dict(chunk)
    chunk["content"] = re.sub(r"\[\[ROSTER:\d+\]\]", placeholder, stripped).strip()
    chunk["kind"] = "roster"
    chunk["roster_count"] = len(entries)
    chunk["char_count"] = len(chunk["content"])
    chunk["word_count"] = len(chunk["content"].split())
    return chunk, entries


def _split_long_article(
    chunk: Dict[str, Any], target_max_chars: int = TARGET_MAX_CHARS
) -> List[Dict[str, Any]]:
    """
    R5/R6 — Decoupe un article trop long, sans jamais casser un tableau.

    Coupe prioritairement aux alineas (1), (2), (3), en repetant l'en-tete
    de l'article dans chaque morceau pour qu'il reste citable seul.
    """
    content = chunk["content"]
    if len(content) <= target_max_chars:
        return [chunk]

    # R5 : un tableau ne se coupe pas. Si le chunk en contient un, on le laisse
    # entier meme au-dela de la cible : une ligne isolee ("31 | 180 |
    # EDUCATION PRESCOLAIRE | 31 915 303") ne repond a aucune question.
    if _TABLE_BLOCK.search(content):
        chunk = dict(chunk)
        chunk["kind"] = chunk.get("kind") or "table"
        chunk["oversized"] = True
        return [chunk]

    parts = [p.strip() for p in _ALINEA.split(content) if p and p.strip()]
    if len(parts) < 2:
        return [chunk]

    head = parts[0] if not parts[0].lstrip().startswith("(") else ""
    body = parts[1:] if head else parts
    prefix = (head[:200].strip() + "\n") if head else ""

    out: List[Dict[str, Any]] = []
    for i, part in enumerate(body):
        piece = dict(chunk)
        piece["content"] = (prefix + part).strip() if i > 0 else (head + "\n" + part).strip()
        piece["number"] = f"{chunk['number']}.{i + 1}"
        piece["parent_id"] = str(chunk["number"])
        piece["char_count"] = len(piece["content"])
        piece["word_count"] = len(piece["content"].split())
        out.append(piece)
    return out or [chunk]


def _extract_citations(legal_basis: str) -> List[str]:
    """
    R3 — Parse les visas en graphe de citations.

    "Vu le decret n° 2011/412 du 09 decembre 2011 portant reorganisation..."
    devient une arete du graphe. C'est une fonctionnalite produit ("quels
    textes s'appuient sur cette loi ?") et ca ne coute rien puisqu'on a
    deja le texte.
    """
    refs: List[str] = []
    for visa in _VISA_LINE.findall(legal_basis):
        for kind, num in _CITED_REF.findall(visa):
            ref = f"{kind.strip().lower()} n° {num.strip()}"
            if ref not in refs:
                refs.append(ref)
    return refs


def _dedupe(chunks: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """R7 — Fusionne les chunks au contenu strictement identique."""
    seen: Dict[str, int] = {}
    out: List[Dict[str, Any]] = []
    removed = 0
    for c in chunks:
        key = hashlib.sha256(
            re.sub(r"\s+", " ", c["content"]).strip().lower().encode("utf-8")
        ).hexdigest()
        if key in seen:
            removed += 1
            continue
        seen[key] = 1
        out.append(c)
    return out, removed


# ==================== POINT D'ENTREE ====================


def refine(
    chunks: List[Dict[str, Any]],
    context: DocumentContext,
    target_max_chars: int = TARGET_MAX_CHARS,
) -> RefinedChunks:
    """
    Applique les regles R2-R7 aux chunks bruts de text_chunker.

    Args:
        chunks: Sortie de extract_articles()
        context: Metadonnees du document (reference, titre, date, categorie)
        target_max_chars: Taille cible avant decoupe aux alineas

    Returns:
        RefinedChunks — `.embeddable` donne les seuls chunks a vectoriser
    """
    assert isinstance(chunks, list), "chunks doit etre une liste"
    assert context is not None, "context requis"

    result = RefinedChunks()
    working: List[Dict[str, Any]] = []

    for raw in chunks:
        chunk = dict(raw)
        number = str(chunk.get("number", ""))

        # Le motif d'article consomme "Article 5." mais laisse le tiret de
        # "Article 5.-" en tete du contenu. Cosmetique, mais ce tiret orphelin
        # se retrouverait dans chaque citation affichee a l'utilisateur.
        chunk["content"] = re.sub(r"^\s*[-–—.:]\s*", "", chunk["content"]).strip()
        chunk["char_count"] = len(chunk["content"])
        chunk["word_count"] = len(chunk["content"].split())

        # R3 — visas hors index vectoriel, convertis en citations.
        #
        # Attention : text_chunker etiquette LEGAL_BASIS TOUT le texte precedant
        # le premier article. Sur une page qui ne commence pas par un article
        # — cas courant au milieu d'un document — c'est la suite de l'article
        # precedent, donc du contenu juridique reel. L'exclure de l'index ferait
        # disparaitre silencieusement le debut de chaque page de la recherche.
        # On ne classe en base legale que si des visas "Vu ..." sont presents.
        if number == "LEGAL_BASIS":
            if _VISA_LINE.search(chunk["content"]):
                result.legal_basis = chunk["content"]
                result.citations = _extract_citations(chunk["content"])
                chunk["kind"] = "legal_basis"
                chunk["embed"] = False
            else:
                chunk["kind"] = "continuation"
                chunk["embed"] = True
            working.append(chunk)
            continue

        if number == "PREAMBULE":
            chunk["kind"] = "preamble"
            chunk["embed"] = True
            working.append(chunk)
            continue

        # R4 — listes nominatives
        chunk, entries = _apply_roster_rule(chunk)
        if entries:
            result.roster.extend(entries)

        # R6 — articles d'execution : conserves, mais hors vectoriel
        if _BOILERPLATE.search(chunk["content"]) and len(chunk["content"]) < 600:
            chunk["kind"] = "boilerplate"
            chunk["embed"] = False
            working.append(chunk)
            continue

        chunk.setdefault("kind", "article")
        chunk["embed"] = True

        # R5/R6 — decoupe des articles trop longs
        working.extend(_split_long_article(chunk, target_max_chars))

    # R6 — chunks trop courts : conserves, mais hors vectoriel
    for chunk in working:
        if chunk.get("embed") and len(chunk["content"]) < MIN_CHARS:
            chunk["embed"] = False
            chunk["kind"] = "fragment"

    # R7 — deduplication
    working, removed = _dedupe(working)

    # R2 — contextualisation : embed_text = en-tete + contenu
    for position, chunk in enumerate(working):
        chunk["position"] = position
        if not chunk.get("embed"):
            chunk["embed_text"] = None
            continue

        header = context.header(
            page=chunk.get("page_number"), section=chunk.get("section")
        )

        # Ne labelliser "Article X" que si X est un vrai numero d'article.
        # PARA_n est le repli par paragraphes de text_chunker : l'annoncer
        # comme un article ferait citer au chatbot des references inexistantes.
        number = str(chunk.get("number") or "")
        label = f"Article {number}" if _REAL_ARTICLE_NUMBER.match(number) else ""

        # Les tableaux sont vectorises en texte delimite, pas en HTML brut
        body = _TABLE_BLOCK.sub(lambda m: _table_to_text(m.group(0)), chunk["content"])

        chunk["embed_text"] = "\n".join(
            part for part in (header, "———", label, body) if part
        ).strip()

    result.chunks = working
    result.stats = {
        "chunks_in": len(chunks),
        "chunks_out": len(working),
        "embeddable": len(result.embeddable),
        "roster_entries": len(result.roster),
        "citations": len(result.citations),
        "duplicates_removed": removed,
        "kinds": {
            k: sum(1 for c in working if c.get("kind") == k)
            for k in sorted({c.get("kind", "article") for c in working})
        },
    }

    logger.info(
        f"🔧 Raffinage {context.reference}: {len(chunks)} → {len(working)} chunks, "
        f"{len(result.embeddable)} a vectoriser "
        f"({len(result.roster)} personnes, {len(result.citations)} citations)"
    )
    return result
