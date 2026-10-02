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
  R4  listes nominatives     liste gardee au contenu, resumee au vecteur,
                             entrees aussi en table separee
  R5  tableaux entiers       jamais coupes au milieu d'une ligne
  R6  tailles                decoupe aux alineas, boilerplate et signature
                             hors index
  R7  deduplication          chunks identiques fusionnes

Principe directeur : RIEN N'EST SUPPRIME. Les chunks ecartes du vectoriel
gardent `embed=False` et restent cherchables en FTS exact. Sur une base
juridique, ne jamais perdre de contenu — seulement le hierarchiser.

Author: JuriX Team
"""

import hashlib
import html
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
# Seuil d'un VRAI article : « Article 4.- Le decret n° 2001/041 est abroge. »
# est court mais normatif, et l'ecarter du vectoriel le rendait introuvable par
# le sens. En deca de ce seuil, l'article n'a plus de texte (OCR vide).
MIN_CHARS_ARTICLE = 30
# Proportion de lignes "nominatives" a partir de laquelle un tableau est un roster
ROSTER_ROW_RATIO = 0.6
ROSTER_MIN_ROWS = 5


# ==================== MOTIFS ====================

# Matricule administratif camerounais : "765 609-Y", "699 536-N", et le format
# actuel sans tiret "0570984M", "583397-B".
_MATRICULE = re.compile(r"\b(?:\d{3}\s?\d{3}\s?-\s?[A-Z]|\d{6,7}\s?-?\s?[A-Z])\b")

# Montant : « 12 512 163 », « 1.200.000 ». Une ligne de tableau qui en porte
# deux est une ligne budgetaire, pas une personne.
_MONTANT = re.compile(r"\b\d{1,3}(?:[ .\u202f]\d{3})+\b")

# Civilite : signal d'une personne dans une cellule
_CIVILITE = re.compile(r"\b(?:M\.|MM\.|Mme|Mlle|Mr|Mrs|Dr|Monsieur|Madame)\s", re.IGNORECASE)

# En-tete de tableau nominatif : « N° | Nom | Matricule »
_ENTETE_TABLEAU = re.compile(
    r"\b(?:n[°ºo]|nom|noms|pr[ée]noms?|matricules?|grades?|indices?|corps|fonctions?|rang)\b",
    re.IGNORECASE,
)

# Nom en capitales : au moins deux mots, apostrophes et tirets admis
_CAPS_NAME = re.compile(r"^[A-ZÀ-Þ][A-ZÀ-Þ'’\-\.]*(?:\s+[A-ZÀ-Þ][A-ZÀ-Þ'’\-\.]*){1,6}$")

# Tableaux HTML produits par LlamaParse
_TABLE_BLOCK = re.compile(r"<table\b.*?</table>", re.IGNORECASE | re.DOTALL)
_TR_BLOCK = re.compile(r"<tr\b[^>]*>(.*?)</tr>", re.IGNORECASE | re.DOTALL)
_TD_CELL = re.compile(r"<t[dh]\b[^>]*>(.*?)</t[dh]>", re.IGNORECASE | re.DOTALL)
_ANY_TAG = re.compile(r"<[^>]+>")

# Tableaux markdown « a barres » : la forme qu'ecrivent Docling et la consigne
# Gemini. Deux lignes consecutives au moins, chacune ouverte et fermee par une
# barre. Le detecteur ne connaissait que le HTML de LlamaParse : une liste
# nominative rendue en tableau markdown echappait a R4, et un tableau long
# etait coupe au milieu par R5.
_MD_TABLE_BLOCK = re.compile(r"(?:^[ \t]*\|.*\|[ \t]*(?:\n|$)){2,}", re.MULTILINE)
_MD_SEPARATOR_ROW = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{0,}:?\s*\|?\s*$")

# Liste nominative en texte brut. Trois formes reelles du corpus :
#   "20. ADAMA ADAM 765 645-M"                     rang, nom, matricule
#   "2447 DAWE KOLWE RICHARD 598 330-A"            rang SANS ponctuation
#   "12. NGO BIYONG Marie 501 303-Z anc. cons."    texte APRES le matricule
# L'ancien motif exigeait la ponctuation apres le rang et la fin de ligne apres
# le matricule : sur 13 listes de 5 personnes ou plus mesurees dans un
# echantillon du corpus, 2 seulement etaient reconnues. Une puce (« - »),
# forme des listes Docling, est toleree en tete.
_PLAIN_ROSTER_LINE = re.compile(
    r"^\s*(?:[-*•]\s+)?(\d{1,4})\s*[.)\-]?\s+(.{3,80}?)\s+"
    r"(\d{3}\s?\d{3}\s?-\s?[A-Z])\b(.{0,60})$"
)

# Liste SANS matricule : « 1. NGONO Marie Claire », « 181 ABENA ESSOMBA Paul ».
# Signature : un rang, puis un nom de famille en capitales. Ce motif est plus
# large que le precedent ; il ne vaut liste nominative que dans un texte de
# nomination (voir _NOMINATION_CUE) et quand ces lignes font l'essentiel du
# chunk (ROSTER_ROW_RATIO).
_NUMBERED_NAME_LINE = re.compile(
    r"^\s*(?:[-*•]\s+)?(\d{1,4})\s*[.)\-]?\s+"
    r"((?:M\.|MM\.|Mme|Mlle|Mr|Mrs|Dr)?\s*[A-ZÀ-Þ][A-ZÀ-Þ'’\-]+"
    r"(?:\s+[A-ZÀ-Þa-zà-ÿ][A-Za-zÀ-ÿ'’\-\.]*){0,6})\s*[,;.]?\s*$"
)
# Liste a puces SANS rang ni matricule : « - TEMGOUA NOUMBO HONORINE ». Meme
# garde que la precedente : contexte de nomination, et part des lignes.
_BULLET_NAME_LINE = re.compile(
    r"^\s*[-*•]\s+((?:M\.|MM\.|Mme|Mlle)?\s*[A-ZÀ-Þ][A-ZÀ-Þ'’\-]+(?:\s+[A-ZÀ-Þ][A-ZÀ-Þ'’\-]+){1,6})\s*[,;.]?\s*$"
)

_NOMINATION_CUE = re.compile(
    r"\b(nomm[ée]s?|nomination|promu|promus|promotion|admis|avancements?"
    r"|dont\s+les\s+noms\s+suivent|ci-apr[èe]s\s+d[ée]sign[ée]s|appointed|promoted)\b",
    re.IGNORECASE,
)

# Vrai numero d'article : "1", "1er", "12.3", "L 94 septies", "QUATRE-VINGT-SIXIEME".
# Exclut PARA_n / LEGAL_BASIS / PREAMBULE, qui sont des replis internes du chunker.
_REAL_ARTICLE_NUMBER = re.compile(
    r"^(?:\d+(?:er|ere|eme|ème|ère)?(?:\.\d+)*"
    r"|[A-Z]\s*\d+(?:\s+\w+)?"
    r"|(?:[A-Za-zÀ-ÿ]+-)+[A-Za-zÀ-ÿ]*(?:ièmes?|èmes?|iemes?|emes?))$",
    re.IGNORECASE,
)

# Pseudo-numeros que le decoupeur donne au texte hors articles : preambule,
# visas, chapeau de section, repli par paragraphes, signature, annexe — et
# leurs morceaux (« ANNEXE.2 »). Ce ne sont pas des numeros d'article : ils ne
# s'affichent ni ne se citent comme tels. Un seul motif, partage par la
# recherche, le chat et l'assemblage du contexte : chacun tenait sa propre
# liste, et aucune ne connaissait SECTION_n ni PARA_n.
PSEUDO_NUMERO = re.compile(
    r"^(?:PREAMBULE|LEGAL_BASIS|SIGNATURE|ANNEXE|SECTION_\d+|PARA_\d+)(?:\.\d+)*$",
    re.IGNORECASE,
)


def est_pseudo_numero(numero: Optional[str]) -> bool:
    """Le numero est-il un pseudo-numero du decoupeur ?"""
    return bool(numero) and bool(PSEUDO_NUMERO.match(str(numero).strip()))


# Alineas numerotes d'un article : "(1)", "(2)" en debut de ligne, eventuellement
# precedes d'une puce (Docling rend un alinea « (1) » en element de liste).
#
# AUCUN groupe capturant : re.split insere dans son resultat chaque groupe
# capture. L'ancien motif « (?=^\s*\((\d{1,2})\)\s) » intercalait donc les
# numeros entre les morceaux : fragments « Chapeau.\n1 » en alternance avec les
# alineas, et l'alinea (2) etiquete « Article 9.4 ».
_ALINEA = re.compile(r"(?=^[ \t]*(?:[-*•][ \t]*)?\(\d{1,2}\)\s)", re.MULTILINE)
_DEBUT_ALINEA = re.compile(r"^\s*(?:[-*•]\s*)?\(")

# Visas : "Vu la Constitution ;", "Vu le decret n° 2011/412 du 09 decembre 2011"
# Docling rend les visas en items de liste : « - Vu le decret n° ... ».
_VISA_LINE = re.compile(
    r"^\s*(?:[-*•]\s*)?Vu\s+(.{5,300}?)\s*[;,.]?\s*$", re.IGNORECASE | re.MULTILINE
)
# Reference d'un texte cite dans un visa
_CITED_REF = re.compile(
    r"\b(loi|d[ée]cret|arr[êe]t[ée]|ordonnance|d[ée]cision|circulaire)\s+"
    r"(?:constitutionnelle\s+)?n[°ºo]\s*([\d./\-]+)",
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
    r"|entre\s+en\s+vigueur\s+[àa]\s+compter\s+de\s+la\s+date\s+de\s+sa\s+signature"
    # Formule de promulgation, en tete de chaque loi
    r"|(?:l[’']Assembl[ée]e\s+Nationale|le\s+Parlement)\s+a\s+d[ée]lib[ée]r[ée]\s+et\s+adopt[ée]"
    r"|Parliament\s+has\s+deliberated\s+and\s+adopted)",
    re.IGNORECASE,
)

# Un mot en minuscules : sans lui, un chunk n'est fait que de titres en
# capitales (« CHAPITRE II DES MINES / SECTION I DES PERMIS ») ou de restes
# du cachet. Il ne repond a aucune question.
_MOT_EN_MINUSCULES = re.compile(r"[a-zà-ÿ]{3,}")


# Debut du bloc de signature qui suit la formule d'execution. Sur les decrets
# d'une page, Docling place le titre de l'acte APRES la signature : sans
# coupure, signature et titre se collaient au dernier article, qui depassait
# le seuil des formules d'execution et partait dans l'index vectoriel.
_SIGNATURE = re.compile(
    r"^\W*(?:\d{1,2}\s+[A-Za-zÀ-ÿ]{3,9}\.?\s+\d{4}\s+)?"
    r"(?:YAOUND[ÉE]|Fait\s+[àa]\s+[A-ZÀ-Þ][\wÀ-ÿ\-]+)\s*,?\s*(?:le|the)\b"
    r"|^\W*LE\s*PR[ÉE]SIDENT\s*DE\s*LA\s*R[ÉE]PUBLIQUE\s*,?\s*$"
    r"|^\W*(?:THE\s*)?PRESIDENT\s*OF\s*THE\s*REPUBLIC\s*,?\s*$"
    r"|^\W*POUR\s+LE\s+PR[ÉE]SIDENT",
    re.IGNORECASE | re.MULTILINE,
)

# Debut d'une annexe apres la signature : elle redevient un chunk a part
# entiere, vectorise. Sans cette borne, une annexe (bareme, liste de
# beneficiaires) partait avec la signature hors de l'index vectoriel.
_DEBUT_ANNEXE = re.compile(
    r"^[ \t]*(?:\|[ \t]*)?(?:ANNEXES?|ANNEX|TABLEAU|[EÉ]TAT|LISTE)\b",
    re.IGNORECASE | re.MULTILINE,
)

# Ligne d'une liste nominative : rangee de tableau, puce, ou rang suivi d'un
# nom en capitales
_LIGNE_DE_LISTE = re.compile(r"^\s*(?:\||[-*•]\s|\d{1,4}\s*[.)\-]?\s+[A-ZÀ-Þ])")


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
# Commentaires HTML : `<!-- image -->` et `<!-- page break -->` de Docling.
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
# Alinea rendu en element de liste par Docling : « - (1) Le ... » -> « (1) Le ... »
_ALINEA_EN_LISTE = re.compile(r"^([ \t]*)[-*•][ \t]+(\(\d{1,2}\)\s)", re.MULTILINE)


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
    out = _HTML_COMMENT.sub("", text)
    # Echappements HTML d'un export markdown (« &amp; », « &#x27; ») : sans
    # ce decodage ils partent tels quels dans le contenu et les vecteurs.
    out = html.unescape(out)
    out = _MD_HEADING.sub("", out)
    out = _MD_EMPHASIS.sub(r"\2", out)
    # "A R R Ê T E" -> "ARRÊTE" (l'OCR restitue l'interlettrage des titres)
    out = _SPACED_CAPS.sub(lambda m: m.group(1).replace(" ", ""), out)
    out = _ALINEA_EN_LISTE.sub(r"\1\2", out)
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


def _md_table_rows(table_md: str, skip_header: bool = True) -> List[List[str]]:
    """
    Lignes d'un tableau markdown, sous forme de listes de cellules.

    La ligne de separation `|---|` est ecartee ; avec skip_header, les lignes
    qui la precedent aussi — l'en-tete « N° | Nom | Matricule » ne doit pas
    compter pour une personne.
    """
    lignes = [ln for ln in table_md.splitlines() if ln.strip()]
    separateur = next((i for i, ln in enumerate(lignes) if _MD_SEPARATOR_ROW.match(ln)), None)
    if separateur is not None:
        entete = lignes[:separateur]
        # Docling met en « en-tete » la premiere ligne de DONNEES quand le
        # tableau n'en a pas : « | 1. NDOUNGA EVINA JEAN | 607 027-V | ». La
        # sauter perdait une personne par page de liste. Elle n'est sautee que
        # si elle ressemble a un en-tete : ni matricule, ni rang, et des mots
        # comme Nom ou Matricule.
        vrai_entete = all(
            not _MATRICULE.search(ln)
            and not re.match(r"\s*\|?\s*\d{1,4}\s*[.)-]", ln)
            and _ENTETE_TABLEAU.search(ln)
            for ln in entete
        )
        if skip_header and vrai_entete:
            lignes = lignes[separateur + 1:]
        else:
            lignes = entete + lignes[separateur + 1:]
    rows = []
    for ligne in lignes:
        cells = [c.strip() for c in ligne.strip().strip("|").split("|")]
        if any(cells):
            rows.append(cells)
    return rows


def _md_table_to_text(table_md: str) -> str:
    """Tableau markdown aplati pour l'embedding, sans la ligne de separation."""
    return "\n".join(
        " | ".join(cells) for cells in _md_table_rows(table_md, skip_header=False)
    )


def _contains_table(content: str) -> bool:
    return bool(_TABLE_BLOCK.search(content) or _MD_TABLE_BLOCK.search(content))


def _looks_nominative(rows: List[List[str]], contexte: str = "") -> bool:
    """
    Le tableau est-il une liste de personnes ?

    Critere : au moins ROSTER_ROW_RATIO des lignes portent soit un matricule,
    soit un nom entierement en capitales de 2 mots ou plus — et le tableau
    porte un SIGNAL DE PERSONNE : des matricules, des civilites, ou un texte de
    nomination autour. Sans ce signal, les tableaux budgetaires passaient :
    « DEVELOPPEMENT DU PRESCOLAIRE » est un nom en capitales, et une loi de
    finances perdait ses montants dans une « liste de 56 personnes ». Une ligne
    qui porte deux montants n'est jamais une personne.
    """
    if len(rows) < ROSTER_MIN_ROWS:
        return False

    hits = matricules = civilites = 0
    for cells in rows:
        joined = " ".join(cells)
        if len(_MONTANT.findall(joined)) >= 2:
            continue
        if _MATRICULE.search(joined):
            hits += 1
            matricules += 1
            continue
        if _CIVILITE.search(joined):
            civilites += 1
        if any(_CAPS_NAME.match(c) or _nom_colle(c) for c in cells if len(c) > 4):
            hits += 1
    signal = matricules >= 0.3 * len(rows) or civilites or _NOMINATION_CUE.search(contexte)
    return bool(signal) and hits / len(rows) >= ROSTER_ROW_RATIO


def _nom_colle(cellule: str) -> bool:
    """« MBINCHOBLAISEAMBE » : un nom en capitales que l'OCR a colle en un mot."""
    c = re.sub(r"^\s*\d{1,4}\s*[.)-]\s*", "", cellule.strip())
    return bool(re.fullmatch(r"[A-ZÀ-Þ'’\-]{8,60}", c))


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
        rank_in_cell = None
        for c in cells:
            # « 2. ELOUNDOU ESSOMBA ALBERT » : rang et nom dans la meme cellule
            m = re.match(r"\s*(\d{1,4})\s*[.)-]\s+(.+)$", c)
            candidat = m.group(2).strip() if m else c
            if len(candidat) > 4 and (_CAPS_NAME.match(candidat) or _nom_colle(candidat)):
                name = candidat
                rank_in_cell = m.group(1) if m else None
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

        rank = rank_in_cell or next(
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
    Variante texte brut : une personne par ligne.

    Deux passes. D'abord les lignes qui portent un matricule : signature sure,
    5 suffisent. Sinon, les lignes « rang + nom en capitales », plus ambigues :
    elles ne valent liste que dans un texte de nomination, et si elles font au
    moins ROSTER_ROW_RATIO des lignes du chunk — une enumeration de ministeres
    dans un article normatif ne doit pas disparaitre du contenu.

    Returns:
        (entrees, texte sans les lignes nominatives)
    """
    lignes = text.splitlines()

    def _collecte(motif, avec_matricule: bool) -> Tuple[List[RosterEntry], List[str]]:
        entries: List[RosterEntry] = []
        kept: List[str] = []
        for line in lignes:
            m = motif.match(line)
            if m:
                entries.append(
                    RosterEntry(
                        article_number=article_number,
                        position=len(entries),
                        name=m.group(2).strip(),
                        identifier=m.group(3).strip() if avec_matricule else None,
                        rank=m.group(1),
                    )
                )
            else:
                kept.append(line)
        return entries, kept

    entries, kept = _collecte(_PLAIN_ROSTER_LINE, avec_matricule=True)
    if len(entries) >= ROSTER_MIN_ROWS:
        return entries, "\n".join(kept)

    if not _NOMINATION_CUE.search(text):
        return [], text
    non_vides = sum(1 for ln in lignes if ln.strip())
    entries, kept = _collecte(_NUMBERED_NAME_LINE, avec_matricule=False)
    if len(entries) >= ROSTER_MIN_ROWS and len(entries) >= ROSTER_ROW_RATIO * non_vides:
        return entries, "\n".join(kept)

    # Puces sans rang : « - TEMGOUA NOUMBO HONORINE »
    puces: List[RosterEntry] = []
    reste: List[str] = []
    for line in lignes:
        m = _BULLET_NAME_LINE.match(line)
        if m:
            puces.append(RosterEntry(article_number=article_number, position=len(puces),
                                     name=m.group(1).strip()))
        else:
            reste.append(line)
    if len(puces) >= ROSTER_MIN_ROWS and len(puces) >= ROSTER_ROW_RATIO * non_vides:
        return puces, "\n".join(reste)
    return [], text


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
    resume = content

    # Variante tableau : HTML (LlamaParse) ou markdown a barres (Docling)
    tables = [(t, _table_rows(t)) for t in _TABLE_BLOCK.findall(content)]
    tables += [(m.group(0), _md_table_rows(m.group(0))) for m in _MD_TABLE_BLOCK.finditer(content)]
    for table, rows in tables:
        if not _looks_nominative(rows, content):
            continue
        lignes = _parse_roster_rows(rows, number, start=len(entries))
        entries.extend(lignes)
        # Un decompte PAR tableau : le total global repete deux fois trompait.
        resume = resume.replace(
            table.rstrip("\n"), f"[{len(lignes)} personnes — liste nominative]"
        )

    # Variante texte brut
    if not entries:
        entries, candidate = _parse_plain_roster(content, number)
        if entries:
            resume = candidate + f"\n\n[{len(entries)} personnes — liste nominative]"

    if not entries:
        return chunk, []

    # Le CONTENU garde la liste : c'est lui qu'on affiche, qu'on cite et que
    # le plein texte indexe. La liste n'etait enregistree nulle part ailleurs —
    # les noms disparaissaient de l'application, remplaces par une « annexe »
    # qui n'existait pas. Seul le texte VECTORISE prend la forme resumee : 904
    # inspecteurs ne doivent pas faire 904 vecteurs, ni saturer la recherche.
    chunk = dict(chunk)
    chunk["embed_body"] = resume.strip()
    chunk["kind"] = "roster"
    chunk["roster_count"] = len(entries)
    return chunk, entries


def _split_roster(
    chunk: Dict[str, Any], target_max_chars: int = TARGET_MAX_CHARS
) -> List[Dict[str, Any]]:
    """
    R4/R5 — Decoupe une longue liste nominative entre ses lignes.

    La liste reste dans le contenu, pour l'affichage, la citation et le plein
    texte. Mais entiere, une liste de 904 personnes fait un chunk de 40 000
    caracteres dont le contexte du chat ne lit que les 6 000 premiers : un nom
    de la fin, retrouve par le plein texte, n'arrivait jamais au modele.

    Chaque morceau repete le chapeau de l'article, tronque, pour dire de quelle
    liste il s'agit. Seul le premier est vectorise, avec le resume de la liste
    (`embed_body`) : 904 noms ne font toujours qu'un vecteur. Une ligne n'est
    jamais coupee ; un tableau HTML ne se coupe pas du tout.
    """
    content = chunk["content"]
    if len(content) <= target_max_chars or _TABLE_BLOCK.search(content):
        return [chunk]

    lignes = content.splitlines()
    debut = next((i for i, ln in enumerate(lignes) if _LIGNE_DE_LISTE.match(ln)), len(lignes))
    chapeau = "\n".join(lignes[:debut]).strip()
    prefixe = chapeau[:200].strip()

    morceaux: List[List[str]] = []
    taille = len(chapeau)
    for ligne in lignes[debut:]:
        if morceaux and morceaux[-1] and taille + len(ligne) + 1 > target_max_chars:
            morceaux.append([])
            taille = len(prefixe)
        if not morceaux:
            morceaux.append([])
        morceaux[-1].append(ligne)
        taille += len(ligne) + 1
    if len(morceaux) < 2:
        return [chunk]

    out: List[Dict[str, Any]] = []
    for i, morceau in enumerate(morceaux):
        piece = dict(chunk)
        tete = chapeau if i == 0 else prefixe
        piece["content"] = (tete + "\n" + "\n".join(morceau)).strip()
        piece["number"] = f"{chunk['number']}.{i + 1}"
        piece["parent_id"] = str(chunk["number"])
        piece["char_count"] = len(piece["content"])
        piece["word_count"] = len(piece["content"].split())
        if i > 0:
            piece["embed"] = False
            piece.pop("embed_body", None)
        out.append(piece)
    return out


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

    # Une liste nominative se coupe entre ses lignes, tableau ou non : chaque
    # ligne y est une personne, complete en elle-meme.
    if chunk.get("kind") == "roster":
        return _split_roster(chunk, target_max_chars)

    # R5 : un tableau ne se coupe pas. Si le chunk en contient un, on le laisse
    # entier meme au-dela de la cible : une ligne isolee ("31 | 180 |
    # EDUCATION PRESCOLAIRE | 31 915 303") ne repond a aucune question.
    #
    # La nature « table » n'etait jamais posee : refine() fixe "article" par
    # defaut AVANT cet appel, et `get("kind") or "table"` gardait donc
    # "article". Un roster garde sa nature.
    if _contains_table(content):
        chunk = dict(chunk)
        if chunk.get("kind") in (None, "article"):
            chunk["kind"] = "table"
        chunk["oversized"] = True
        return [chunk]

    parts = [p.strip() for p in _ALINEA.split(content) if p and p.strip()]
    if len(parts) < 2:
        # Sans alineas numerotes, aux paragraphes, puis aux items de liste.
        # Un article de definitions de 18 000 caracteres restait d'un bloc :
        # tronque a 10 000 caracteres puis a 2048 jetons, la fin n'etait
        # jamais vectorisee.
        parts = [p.strip() for p in re.split(r"\n\s*\n", content) if p.strip()]
        if len(parts) < 2:
            parts = [p.strip() for p in re.split(r"\n(?=\s*[-*•]\s)", content) if p.strip()]
        if len(parts) < 2:
            return [chunk]
        head, body = "", parts
    else:
        head = parts[0] if not _DEBUT_ALINEA.match(parts[0]) else ""
        body = parts[1:] if head else parts

    # Regroupement glouton jusqu'a la taille cible : pas de morceau minuscule
    # a chaque alinea court, pas de morceau geant non plus.
    morceaux: List[str] = []
    for part in body:
        if morceaux and len(morceaux[-1]) + len(part) + 1 <= target_max_chars:
            morceaux[-1] += "\n" + part
        else:
            morceaux.append(part)

    # Le chapeau est repete en tete de chaque morceau pour qu'il reste citable
    # seul, mais tronque : colle entier au premier, un long chapeau faisait un
    # premier morceau a peine plus petit que l'article.
    prefix = (head[:200].strip() + "\n") if head else ""
    out: List[Dict[str, Any]] = []
    for i, morceau in enumerate(morceaux):
        piece = dict(chunk)
        piece["content"] = (prefix + morceau).strip()
        piece["number"] = f"{chunk['number']}.{i + 1}"
        piece["parent_id"] = str(chunk["number"])
        piece["char_count"] = len(piece["content"])
        piece["word_count"] = len(piece["content"].split())
        out.append(piece)
    return out or [chunk]


def _detacher_signature(chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    R6 — Detache du dernier article la signature et ce qui la suit.

    Le decoupeur colle au dernier article tout ce qui le suit : lieu et date,
    « LE PRESIDENT DE LA REPUBLIQUE », le nom du signataire — et, sur les
    decrets d'une page, le titre de l'acte, que Docling lit APRES la signature.
    La formule d'execution depassait alors le seuil de 600 caracteres, n'etait
    plus reconnue, et partait dans l'index vectoriel avec la signature.

    Retourne l'article, puis un chunk SIGNATURE hors vectoriel, puis, si une
    annexe suit, un chunk ANNEXE traite comme un article. Rien n'est supprime.
    """
    content = chunk["content"]
    m = _SIGNATURE.search(content)
    if not m or not content[:m.start()].strip():
        return [chunk]

    reste = content[m.start():]
    annexe = _DEBUT_ANNEXE.search(reste, m.end() - m.start())
    morceaux = [
        (content[:m.start()], {}),
        (reste[:annexe.start()] if annexe else reste,
         {"number": "SIGNATURE", "kind": "signature", "embed": False}),
    ]
    if annexe:
        morceaux.append((reste[annexe.start():], {"number": "ANNEXE"}))

    out: List[Dict[str, Any]] = []
    for texte, champs in morceaux:
        if not texte.strip():
            continue
        piece = dict(chunk)
        piece.update(champs)
        if champs:
            piece["parent_id"] = str(chunk.get("number", ""))
        piece["content"] = texte.strip()
        piece["char_count"] = len(piece["content"])
        piece["word_count"] = len(piece["content"].split())
        out.append(piece)
    return out


def _extract_citations(legal_basis: str) -> List[str]:
    """
    R3 — Parse les visas en graphe de citations.

    "Vu le decret n° 2011/412 du 09 decembre 2011 portant reorganisation..."
    devient une arete du graphe. C'est une fonctionnalite produit ("quels
    textes s'appuient sur cette loi ?") et ca ne coute rien puisqu'on a
    deja le texte.
    """
    refs: List[str] = []
    visas = [v for bloc in _VISA_LINE.findall(legal_basis) for v in re.split(r";\s*Vu\b", bloc)]
    for visa in visas:
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
            elif _BOILERPLATE.search(chunk["content"]) and len(chunk["content"]) < 600:
                # Titre de la loi et formule de promulgation, sans visas
                chunk["kind"] = "boilerplate"
                chunk["embed"] = False
            elif not _MOT_EN_MINUSCULES.search(chunk["content"]):
                chunk["kind"] = "fragment"
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

        for piece in _detacher_signature(chunk):
            if piece.get("kind") == "signature":
                working.append(piece)
                continue

            # R4 — listes nominatives
            piece, entries = _apply_roster_rule(piece)
            if entries:
                result.roster.extend(entries)

            # R6 — articles d'execution : conserves, mais hors vectoriel
            if _BOILERPLATE.search(piece["content"]) and len(piece["content"]) < 600:
                piece["kind"] = "boilerplate"
                piece["embed"] = False
                working.append(piece)
                continue

            piece.setdefault("kind", "annexe" if piece.get("number") == "ANNEXE" else "article")
            piece["embed"] = True

            # Chapeau de section ou repli par paragraphes fait de seuls titres
            # en capitales : conserve, hors vectoriel. Un tableau en capitales
            # (annexe budgetaire) reste vectorise.
            if (
                str(piece.get("number", "")).upper().startswith(("SECTION_", "PARA_"))
                and not _MOT_EN_MINUSCULES.search(piece["content"])
                and not _contains_table(piece["content"])
            ):
                piece["kind"] = "fragment"
                piece["embed"] = False
                working.append(piece)
                continue

            # R5/R6 — decoupe des articles trop longs
            working.extend(_split_long_article(piece, target_max_chars))

    # R6 — chunks trop courts : conserves, mais hors vectoriel. Un vrai article
    # a son propre seuil, bien plus bas : court ne veut pas dire vide de sens.
    for chunk in working:
        seuil = (
            MIN_CHARS_ARTICLE
            if _REAL_ARTICLE_NUMBER.match(str(chunk.get("number") or ""))
            else MIN_CHARS
        )
        if chunk.get("embed") and len(chunk["content"]) < seuil:
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

        # Les tableaux sont vectorises en texte delimite, pas en HTML brut ni
        # avec la ligne de separation markdown. Une liste nominative l'est
        # sous sa forme resumee (R4).
        body = chunk.get("embed_body") or chunk["content"]
        body = _TABLE_BLOCK.sub(lambda m: _table_to_text(m.group(0)), body)
        body = _MD_TABLE_BLOCK.sub(lambda m: _md_table_to_text(m.group(0)) + "\n", body)

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
