"""
Nettoyage du markdown produit par l'extraction de PDF.

Ces regles ne dependent d'AUCUN fournisseur : elles decrivent ce que le corpus
prc.cm contient et qu'il ne faut pas indexer. Elles vivaient dans
`llama_parse_service.py` et auraient disparu avec lui ; elles ont ete calibrees
sur ce corpus et servent tout autant a l'extraction par Gemini.

Trois regles, chacune payee par une observation :

- Le cachet « COPIE CERTIFIEE CONFORME » est appose sur CHAQUE page. Restitue en
  texte, il pollue le plein texte et les vecteurs de tous les documents.
- Le filigrane diagonal `www.prc.cm` ressort eclate caractere par caractere, ou
  en tableau d'une cellule.
- `ARTICLE 1<sup>ER</sup>` doit redevenir `ARTICLE 1ER`, faute de quoi
  `text_chunker.ARTICLE_PATTERNS` ne reconnait pas le marqueur — mais `<table>`
  n'est JAMAIS retire : les annexes budgetaires portent leur sens dans leur
  structure tabulaire.

`nettoyer_markdown` y ajoute la mise a plat des marqueurs que Docling decore :
le texte du document s'affiche en texte brut, et le sommaire de la page de
lecture n'y reconnait « Article » et « CHAPITRE » qu'en debut de ligne.

Author: JuriX Team
"""

import re

# Expressions PROPRES au cachet « COPIE CERTIFIEE CONFORME » appose sur chaque
# page du corpus prc.cm : elles n'apparaissent dans aucun texte juridique.
#
# Le cachet est toujours en CAPITALES. La recherche respecte donc la casse :
# « une copie certifiee conforme de l'acte de naissance », piece a fournir dans
# un dossier de candidature, n'est jamais prise pour lui.
#
# Les variantes tolerees ont ete relevees sur les sorties Docling du corpus.
# L'OCR colle les mots (« COPIECERTIFIEECONFORME », « CERTIFIEDTRUECOPY ») et
# confond des lettres (« APFAIRS », « CARD IBDEX », « REGLENENTAIRE »,
# « BERVICE ») : avec les anciens motifs, a espaces obligatoires et sans
# confusion admise, le cachet restait dans un tiers des pages.
_STAMP_SPECIFIC_MOTIF = (
    r"C[O0]P[IL1T]E\s*C[ET]R[TI1][I1]F[I1][A-ZÉ]{0,3}\s*C[O0]N[FP][A-Z]{1,2}[MN][EF]"
    r"|CER[TI1][I1]F[I1][A-Z]{1,2}\s*[TY]R?UE\s*[CGS][O0][PR]Y"
    r"|\w{0,3}[VT]ICE\s*D[UO]\s*\w?I[CG][HM]IE[RN]"
    r"(?:\s*L[A-ZÉ]{3,6}ATIF)?(?:\s*ET\s*R[EÉ][A-ZÉ]{2,10}AIRE)?"
    r"|(?:L[A-Z]{2,6}ATIVE\s*A[A-Z]{2}\s*)?[STB][TY]A[TY]UT[O0C]R[YT]\s*A[FP]{2}AI[RBP]S"
    r"|CAR[DOGS]\s*[A-Z]{1,2}[DB]EX\s*[SB]ER[VT]I[CS]E"
)

# Vocabulaire du cachet : ses expressions propres, plus des expressions qui
# sont AUSSI des autorites legitimes (« Présidence de la République »,
# « Secrétariat général ») et ne sont retirees qu'a cote d'une expression
# propre. L'extracteur restitue le cachet en texte : il faut le retirer avant
# le decoupage, sinon il pollue le plein texte et les vecteurs de chaque page.
_STAMP_VOCAB = re.compile(
    rf"(?-i:{_STAMP_SPECIFIC_MOTIF})"
    r"|pr[ée][sog]idence\s*de\s*la\s*r[ée]pu\w{4,6}"
    r"|pr\w{2}idency\s*o[ef]\s*the\s*repu\w{3,4}"
    r"|\w{1,2}cr[ée]tariat[\s-]*\w{1,2}n[ée]ra[il]",
    re.IGNORECASE,
)

# Fragments du filigrane diagonal "www.prc.cm" eclate par l'OCR
_WATERMARK = re.compile(
    r"^\s*(?:w\s*){1,3}$|^\s*\.?\s*(?:p\s*r\s*c|c\s*m|prc\.cm|www\.prc\.cm)\s*\.?\s*$",
    re.IGNORECASE,
)

# Lambeaux du cachet que l'OCR a ecorches au-dela des motifs ci-dessus :
# « CERTIFIEDYRUECOPY », « CERTIFIED TRUE/COPY », « FICHIER » seul sur sa
# ligne. Une ligne COURTE, en capitales, qui n'est qu'un mot propre au cachet
# ou qui commence par l'une de ses expressions n'est que du cachet.
_FRAGMENT_DE_CACHET = re.compile(
    r"^[\W_]*(?:CER[TI1][I1]F[I1]E[ED]|[STB][TY]A[TY]UT[O0C]R[YT]|\w?I[CG][HM]IER)[\W_]*$"
    r"|^[\W_]*(?:C[O0]P[IL1T]E\W*C[ET]RT|CER[TI1][I1]F[I1][A-Z]{1,2}\W*(?:[TY]|C[O0]N)"
    r"|L[A-Z]{2,6}ATIVE\s*A[A-Z]D\s*[STB]|CAR[DS]\s*IN[DB]EX|\w{0,3}[VT]ICE\s*D[UO]\s*\w?I[CG])"
    r"[A-Z\W_]{0,40}$"
)

# Ligne de separation d'un tableau markdown : « |----|---| »
_SEPARATEUR_TABLEAU = re.compile(r"^\s*\|\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)*\|?\s*$")

# Balisage HTML inline emis par l'extracteur (les <table> sont conserves : la
# structure tabulaire porte du sens, cf. les annexes budgetaires).
# <sup>/<sub> sont critiques : "ARTICLE 1<sup>ER</sup>" doit redevenir
# "ARTICLE 1ER" pour que ARTICLE_PATTERNS de text_chunker le reconnaisse.
_INLINE_TAGS = re.compile(
    r"</?(?:u|mark|b|i|em|strong|span|sup|sub|small)\b[^>]*>", re.IGNORECASE
)


_STAMP_SPECIFIC = re.compile(_STAMP_SPECIFIC_MOTIF)
# Etiquettes qu'un extracteur pose devant un cachet decrit : « [signature: ...] »
_STAMP_CUES = re.compile(r"\b(?:signature|stamp|logo|cachet|sceau)\b", re.IGNORECASE)
# En deca de ce nombre de caracteres de mot, une fois le cachet ote, la ligne
# n'etait que le cachet.
_RESTE_MINIMAL = 25


def _reste_utile(ligne: str) -> int:
    sans = _STAMP_CUES.sub("", _STAMP_VOCAB.sub(" ", ligne))
    return len(re.sub(r"[\W_]+", "", sans))


def _oter_le_cachet(ligne: str) -> str:
    """
    Ote d'une ligne le cachet, et lui seul.

    Une ligne qui n'est que le cachet disparait. Une ligne ou le cachet s'est
    fondu dans le texte — l'OCR en pleine page mele parfois le tampon a la
    derniere ligne d'un article — garde son texte : on n'en retire que les
    suites d'expressions du vocabulaire qui contiennent une expression propre
    au cachet. « Le Secretaire general de la Presidence de la Republique est
    charge de l'execution » y survit donc intact.
    """
    if _reste_utile(ligne) < _RESTE_MINIMAL:
        return ""
    matches = list(_STAMP_VOCAB.finditer(ligne))
    # Suites contigues d'expressions, separees seulement par des blancs ou de
    # la ponctuation
    groupes: list = []
    for m in matches:
        if groupes and not re.search(r"\w", ligne[groupes[-1][-1].end():m.start()]):
            groupes[-1].append(m)
        else:
            groupes.append([m])
    for groupe in reversed(groupes):
        if any(_STAMP_SPECIFIC.search(m.group(0)) for m in groupe):
            ligne = ligne[:groupe[0].start()] + " " + ligne[groupe[-1].end():]
    return re.sub(r"[ \t]{2,}", " ", ligne).strip()


# Titres markdown : « ## CHAPITRE II DES MINES ».
_TITRE_MD = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE)

# Le mot « Article » tel que l'OCR le rend : « ARTIiCLE », « ARTICLÈ »,
# « ARTICIE », « ARTCLE », « RTICLE », « ARTTICLE », « Artice »,
# « ARTICLES 170.- ». Meme tolerance que text_chunker._MOT_ARTICLE.
_MOT_ARTICLE = r"[AÀ]?R[TL]{1,2}[IÍÌL1]{0,2}C{1,2}[IL1]?[ELTOÈÉÊ][ER]?S?"
# Un vrai marqueur : le mot, un numero (ou PREMIER), puis un separateur
_SUITE_DE_MARQUEUR = r"[ \t_]*(?:\d+|PREMIER)\S{0,4}?[ \t]*[.:\-–]"

# Ce que Docling et l'OCR posent devant un marqueur, en debut de ligne :
# puce ou numero de liste (« - ARTICLE 12.- », « 2. ARTICLE 41.- »,
# « . ARTICLE 31.- », « 1 Article 1er : »), reste du filigrane www.prc.cm
# (« ww ARTICLE 1er.- »), lettre ou symbole parasite du cachet
# (« A ARTiCLE 1er.- », « ■ ARTiCLE 1ºr.- »). Il n'est retire que devant un
# VRAI marqueur, numero puis separateur : un renvoi (« - article 12 de la
# loi n° ... ») reste un element de liste.
_PREFIXE_DE_MARQUEUR = re.compile(
    r"^[ \t]*(?:(?:[-*•]|\d{1,3}[.)]?|\.|_+|[^\w\s]{1,2}|[A-Za-z]|w{2,3}|prc|cm|www\.prc\.cm)"
    r"[ \t./]+){1,4}"
    rf"(?={_MOT_ARTICLE}{_SUITE_DE_MARQUEUR})",
    re.IGNORECASE | re.MULTILINE,
)
# Mot coupe par l'OCR : « ARTI CLE1er.- »
_MOT_COUPE = re.compile(
    r"\b(A[ \t]?R[ \t]?T[ \t]?I[ \t]?C[ \t]?L[ \t]?E)(?=[ \t_]*(?:\d|[lI]e?r\b|PREMIER))",
    re.IGNORECASE,
)
# « Article ler.- » : le chiffre 1 lu comme un l
_PREMIER_EN_L = re.compile(rf"\b({_MOT_ARTICLE}[ \t_]*)[lI](?=(?:e?r|ère)\b)", re.IGNORECASE)
# Mot ecorche en tete d'un vrai marqueur, remis d'aplomb
_MOT_ECORCHE = re.compile(
    rf"^([ \t]*)({_MOT_ARTICLE})(?={_SUITE_DE_MARQUEUR})", re.IGNORECASE | re.MULTILINE
)
# « ARTICLE_1er » : souligne lu par l'OCR entre le mot et le numero
_SOULIGNE_DE_MARQUEUR = re.compile(r"\b(ARTICLE)_+(?=\d|PREMIER)", re.IGNORECASE)


def _mot_canonique(m: re.Match) -> str:
    mot = m.group(2)
    if mot.upper() == "ARTICLE":
        return m.group(0)
    return m.group(1) + ("ARTICLE" if mot[:3].isupper() else "Article")


def _mot_recolle(m: re.Match) -> str:
    mot = re.sub(r"[ \t]", "", m.group(1))
    return mot if mot == m.group(1) else mot + " "


def normaliser_marqueurs(text: str) -> str:
    """
    Met a plat les titres markdown et les marqueurs d'article decores.

    Mesure sur les sorties Docling du corpus : un marqueur d'article sur cinq
    arrivait en titre (« ## ARTICLE 2.- »), en element de liste
    (« - ARTICLE 12.- »), numerote (« 2. ARTICLE 41.- »), precede d'un reste
    du cachet (« ■ ARTiCLE 1ºr.- ») ou ecorche (« ARTIiCLE 81.- »,
    « ARTI CLE1er.- », « Article ler.- »). Le sommaire de la page de lecture,
    qui cherche « Article » en debut de ligne, ne les voyait pas, et le
    decoupeur collait leur texte a l'article precedent — ou, pour l'article
    1er, aux visas, hors de l'index vectoriel.
    """
    text = _TITRE_MD.sub("", text)
    text = _MOT_COUPE.sub(_mot_recolle, text)
    text = _PREMIER_EN_L.sub(r"\g<1>1", text)
    text = _PREFIXE_DE_MARQUEUR.sub("", text)
    text = _MOT_ECORCHE.sub(_mot_canonique, text)
    text = re.sub(r"(?m)^([ \t]*ARTICLE)(?=\d)", r"\1 ", text, flags=re.IGNORECASE)
    return _SOULIGNE_DE_MARQUEUR.sub(r"\1 ", text)


def nettoyer_markdown(text: str) -> str:
    """Nettoyage complet a la lecture : cachet, filigrane, balises, marqueurs."""
    return normaliser_marqueurs(strip_stamp_blocks(text))


def _ligne_de_filigrane(ligne: str) -> bool:
    """
    Ligne qui n'est que du filigrane, nue ou en tableau, ou un lambeau du cachet.

    Docling rend parfois le filigrane diagonal en tableau d'une cellule :
    « | www.prc.cm | » suivi de sa ligne de separation. Il passait tel quel dans
    le contenu, et un article long qui le contenait etait pris pour un tableau,
    donc jamais decoupe.
    """
    if "|" not in ligne:
        return bool(_WATERMARK.match(ligne) or _FRAGMENT_DE_CACHET.match(ligne.strip()))
    cellules = [c.strip() for c in ligne.strip().strip("|").split("|")]
    pleines = [c for c in cellules if c]
    return bool(pleines) and all(_WATERMARK.match(c) for c in pleines)


def _sans_separateur_orphelin(lignes: list) -> list:
    """Retire une ligne de separation de tableau qui n'a plus de rangee voisine."""

    def rangee(i: int) -> bool:
        return (
            0 <= i < len(lignes)
            and lignes[i].lstrip().startswith("|")
            and not _SEPARATEUR_TABLEAU.match(lignes[i])
        )

    return [
        ln for i, ln in enumerate(lignes)
        if not (_SEPARATEUR_TABLEAU.match(ln) and not rangee(i - 1) and not rangee(i + 1))
    ]


def strip_stamp_blocks(text: str) -> str:
    """
    Retire le cachet officiel du markdown, sans emporter le texte voisin.

    Une ligne est traitee si elle porte une expression propre au cachet, ou si
    elle appartient a un bloc d'au moins 2 lignes consecutives (blancs admis)
    du vocabulaire, dont une expression propre. Le seuil de 2 et l'exigence
    d'une expression propre evitent de supprimer un en-tete legitime :
    beaucoup de documents portent « PRESIDENCE DE LA REPUBLIQUE » et
    « SECRETARIAT GENERAL » comme autorite emettrice, ce qui est du contenu,
    pas du tampon.

    Auparavant, toute ligne portant deux expressions distinctes du vocabulaire
    etait supprimee ENTIERE. Verifie : « Article 2.-… COPIE CERTIFIEE CONFORME
    CERTIFIED TRUE COPY… » disparaissait avec son article, et la formule
    d'execution « Le Secretariat General de la Presidence de la Republique est
    charge… » aussi, alors qu'elle ne contient aucun cachet.

    Args:
        text: Markdown brut d'extraction

    Returns:
        Markdown nettoye
    """
    lines = text.splitlines()
    a_traiter = [bool(_STAMP_SPECIFIC.search(ln)) for ln in lines]

    i = 0
    while i < len(lines):
        if not _STAMP_VOCAB.search(lines[i]):
            i += 1
            continue
        # Etendre le bloc tant qu'on reste dans le vocabulaire du cachet
        j = i
        hits = 0
        propre = False
        while j < len(lines):
            stripped = lines[j].strip(" >*|-\t")
            if _STAMP_VOCAB.search(lines[j]):
                hits += 1
                propre = propre or bool(_STAMP_SPECIFIC.search(lines[j]))
                j += 1
            elif not stripped:
                j += 1
            else:
                break
        if hits >= 2 and propre:
            for k in range(i, j):
                if _STAMP_VOCAB.search(lines[k]):
                    a_traiter[k] = True
        i = max(j, i + 1)

    kept = []
    for ligne, traiter in zip(lines, a_traiter):
        if traiter:
            ligne = _oter_le_cachet(ligne)
            if not ligne:
                continue
        kept.append(ligne)
    kept = _sans_separateur_orphelin([ln for ln in kept if not _ligne_de_filigrane(ln)])
    cleaned = "\n".join(kept)
    cleaned = _INLINE_TAGS.sub("", cleaned)
    return re.sub(r"\n{4,}", "\n\n\n", cleaned).strip()
