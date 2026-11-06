"""
Les mentions d'articles dans la prose du blog : TOUTES, et non seulement la
forme « l'article X du Code Y » que lit la lecture des citations du chat.

POURQUOI CE MODULE. Revue du tour 1 (30/10/2026) : le blog confiait sa
relecture à lecture_des_citations, dont CITATION_REGEX ne reconnaît que
« article N » ou « art. N » suivi d'un nom de texte. Passaient sans aucun
avertissement, jusqu'au public : « les articles 286 et 312 du Code pénal »
(exactement l'invention de l'essai réel du 29/10), « les articles 312 à
315 », « l'article premier du Code pénal », « l'article 312. » ou « Selon
l'article 312, … » sans nom de texte, « Le Code pénal, en son article 312 »,
et toute forme anglaise « Section 312 of the Penal Code ». Le lecteur lisait
un numéro d'article qu'aucun extrait ne portait, sans carte pour le vérifier.

CE QUE FAIT CE MODULE. Un filet PRUDENT, indépendant de la grammaire du chat :
1. mentions_d_articles : chaque mention d'un ou de plusieurs articles
   (pluriels, énumérations « 286 et 312 », intervalles « 312 à 315 »,
   « premier », alinéas, « art. », « Section », nom du texte après les
   numéros ou avant eux : « Le Code pénal, en son article 312 »), français et
   anglais ;
2. numeros_non_portes : les numéros qu'aucun extrait gardé ne porte. Sans nom
   de texte, le numéro seul doit être celui d'un extrait. Avec un nom, c'est
   la lecture des citations du chat (lire_les_citations, qui fait foi pour
   reconnaître un nom de texte) qui juge, sur une citation au singulier
   recomposée pour chaque numéro : « l'article 286 du Code pénal ». Un renvoi
   (« du même code », « de cette loi ») prend le nom de la mention précédente
   du même champ.
DANS LE DOUTE, SIGNALER : une mention signalée à tort coûte une relecture ; une
mention inventée qui passe arrive au lecteur.

Deux autres filets, pour les mêmes raisons (restes de la revue qui peuvent
montrer un droit faux au lecteur) :
- textes_nommes_non_portes : un texte nommé sans numéro d'article (« le Code
  de la famille », « la Constitution ») qu'aucun extrait ne porte ;
- droit_etranger : le droit d'un autre pays (« le code civil français », « la
  loi française », « la Cour de cassation ») : le prompt l'interdit, l'essai
  réel en avait repris un préavis.

Ce module ne lit que du texte et des extraits : la base n'y est jamais lue.
"""

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Set, Tuple

from app.schemas.search import ChunkResult
from app.services.article_reference import normalize_number
from app.services.lecture_des_citations import (
    etendue_du_nom,
    lire_les_citations,
    nature_nommee,
)

# ==================== LA MENTION ====================

# Le mot qui annonce un article. « Section » : l'article des textes anglais
# (« Section 312 of the Penal Code ») ; en français, « la section 2 » désigne
# une partie d'un texte, pas un article (_SECTION_FRANCAISE l'écarte).
# « art 312 », sans point : seulement devant un chiffre (« l'art de… » n'en est pas).
_MOT = r"(?P<mot>articles?|arts?\.|arts?(?=\s*\d)|sections?)"
# Un numéro : « 312 », « 1er », « premier », « 12 bis », « L 104 », « 94-2 ».
# Jamais suivi de « ° » (« 2° » est un point d'énumération) ni de « / »
# (« 2016/007 » est le numéro d'un texte).
_SUFFIXE = r"(?:\s*(?:bis|ter|quater|quinquies|sexies|septies|octies|nonies|decies)\b)"
_NUMERO = (
    r"(?:premi[eè]re?\b|1\s*(?:er|ère|re)\b"
    r"|(?:(?<!\w)(?-i:[LRD])\.?\s?)?\d+(?:\s*(?:er|ère|re)\b)?" + _SUFFIXE + r"?(?:[-.]\d+)*)"
    r"(?!\s*°)(?![\w/])"
)
# Ce qui précise l'article sans en être un : « (2) », « alinéa 2 »,
# « al. 1er », « paragraphe 3 », « alinéas 1 et 2 ». Lu, puis ignoré.
_ALINEA = (
    r"(?:\s*\(\s*\d+\s*\)"
    r"|\s*,?\s*(?:alin[ée]as?|al\.|paragraphes?|paragraphs?|para\.|§|sub-?sections?)\s*"
    r"(?:\(?\d+\)?(?:\s*(?:er|ère))?|premier|première)"
    r"(?:\s*(?:,|et|à|and|to)\s*\(?\d+\)?)*,?)"
)
# Entre deux numéros d'une énumération : « 286 et 312 », « 312 à 315 »,
# « 1737 et l'article 1736 », « 5 and 6 ». Le second numéro n'est pas une
# quantité : « l'article 312, 3 ans de prison » ne cite pas l'article 3.
_ENTRE = (
    r"(?P<sep>\s*(?:,|;|&|\bet\b|\bou\b|\bà\b|\bau\b|\band\b|\bor\b|\bto\b|\bthrough\b)\s*)"
    r"(?:(?:à|au|aux|de|du|des)\s+)?(?:l['’]\s*)?(?:(?:articles?|arts?\.|sections?)\s*)?"
)
_PAS_UNE_QUANTITE = (
    r"(?!\s*(?:ans?\b|années?\b|mois\b|jours?\b|semaines?\b|heures?\b|minutes?\b|%"
    r"|pour\s*cent|francs?\b|fcfa\b|f\b|euros?\b|dollars?\b|fois\b|personnes?\b"
    r"|conditions?\b|cas\b|millions?\b|milliards?\b|mille\b|years?\b|months?\b|days?\b))"
)
_ET_SUIVANTS = r"(?:\s*(?:et\s+suivants?|et\s+ss?\.|and\s+following|et\s+seq\.))?"
_MENTION = re.compile(
    r"(?<!\w)" + _MOT + r"\s*(?P<numeros>" + _NUMERO + _ALINEA + r"?"
    r"(?:" + _ENTRE + _NUMERO + _PAS_UNE_QUANTITE + _ALINEA + r"?)*)" + _ET_SUIVANTS,
    re.IGNORECASE,
)
_UN_NUMERO = re.compile(_NUMERO, re.IGNORECASE)
_UN_ALINEA = re.compile(_ALINEA, re.IGNORECASE)
_INTERVALLE = re.compile(r"\s*(?:à|au|to|through)\s*$", re.IGNORECASE)
# « la section 2 du chapitre 3 » : une partie d'un texte français.
_SECTION_FRANCAISE = re.compile(
    r"(?:\b(?:la|les|une|cette|ces|de\s+la|à\s+la|chaque|même)\s+)$", re.IGNORECASE
)

# Le nom du texte APRÈS les numéros : « du Code pénal », « de la Loi N°2016/007 »,
# « of the Penal Code ». Un point n'arrête pas « Law No. 92/007 ».
_LIAISON = re.compile(
    r"(?:\s*,)?\s+(?P<liaison>du|des|de\s+la|de\s+l['’]|de|d['’]|of\s+the|of)\s*"
    # Le nom s'arrête avant la citation suivante (« … de la loi n° 2016/007 et
    # l'article 6 du décret »), comme le groupe 3 de CITATION_REGEX.
    r"(?P<nom>[^\W\d_](?:(?!\s+(?:et|ou|and|or)\s+(?:l['’]\s*)?(?:art(?:icles?|\.)|sections?))"
    r"(?:[^.,;:!?()\[\]\n«»\"“”*_]|\.(?=\s*\d)))*)",
    re.IGNORECASE,
)
# Le nom AVANT : « Le Code pénal, en son article 312 », « the Penal Code, in
# its section 312 ». Le nom est la fin de la proposition qui précède.
#
# En minuscule, le mot de nature compte aussi, derrière un déterminant : « Le
# décret d'application, en son article 40, … » (revue du tour 2, sonde des
# mentions : la seule forme sur 15 qui passait). Sans le nom, le numéro était
# jugé contre tous les extraits, et l'article 40 du Code du travail le
# couvrait.
_EN_SON = re.compile(r",?\s*(?:en|dans|à|in|under|at)\s+(?:son|ses|sa|its)\s*$", re.IGNORECASE)
_NOM_AVANT = re.compile(
    r"(?:(?P<determinant>\b(?i:le|la|les|l['’]|the)\s*)?"
    r"(?P<nom>(?:Code|Constitution|Acte\s+uniforme|Actes\s+uniformes|Loi|D[ée]cret|Ordonnance"
    r"|Arr[êe]t[ée]|Trait[ée]|Convention|Charte|Statut|Law|Act|Decree|Ordinance|Treaty"
    r"|(?!(?:Le|La|Les|The)\b)(?:[A-Z][a-z]+\s+){1,3}Code)\b[^.;:!?()\n]*?)"
    r"|(?P<determinant_minuscule>\b(?i:le|la|les|l['’]|ce|cet|cette|the)\s*)"
    r"(?P<nom_minuscule>(?i:code|constitution|acte\s+uniforme|loi|d[ée]cret|ordonnance"
    r"|arr[êe]t[ée]|trait[ée]|convention|charte|statut)\b[^.;:!?()\n]*?))\s*$"
)
# Un renvoi au texte d'une mention précédente.
_RENVOI = re.compile(
    r"^(?:(?:ce|cet|cette|ces|this|these|the|le|la|les)\s+)?"
    r"(?:m[êe]mes?|ledit|ladite|lesdits|lesdites|dudit|desdits|desdites|said|same"
    r"|pr[ée]sente?s?)\b"
    r"|^(?:ce|cet|cette|ces|this|these)\s+[^\W\d_]+"
    r"|\b(?:pr[ée]cit[ée]e?s?|susvis[ée]e?s?|susmentionn[ée]e?s?|aforementioned)\b",
    re.IGNORECASE,
)
_NATURE_SEULE = re.compile(
    r"^(?:(?:le|la|les|l['’]|the)\s*)?(?:code|loi|d[ée]cret|ordonnance|arr[êe]t[ée]|texte"
    r"|law|act|decree|code)s?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MentionDArticle:
    """Une mention d'un ou de plusieurs articles dans un texte."""

    debut: int
    fin: int
    texte: str
    """La mention telle qu'elle est écrite (« les articles 286 et 312 du Code pénal »)."""
    numeros: Tuple[str, ...]
    """Les numéros cités, normalisés (« 1 » pour « premier »), intervalles dépliés."""
    nom: Optional[str] = None
    """Le nom du texte, tel qu'écrit ; None si la mention n'en nomme aucun."""
    liaison: str = "du"
    renvoi: bool = False
    """Le nom renvoie au texte d'une mention précédente (« du même code »)."""


def _numeros_de(passage: str) -> Tuple[str, ...]:
    """Les numéros d'une énumération, alinéas ignorés, intervalles dépliés (30 au plus)."""
    sans_alineas = _UN_ALINEA.sub(" ", passage)
    numeros: List[str] = []
    precedent: Optional[re.Match] = None
    for trouve in _UN_NUMERO.finditer(sans_alineas):
        numero = normalize_number(re.sub(r"\s+", " ", trouve.group()))
        entre = sans_alineas[precedent.end():trouve.start()] if precedent else ""
        if (
            precedent is not None and _INTERVALLE.search(entre)
            and numeros and numeros[-1].isdigit() and numero.isdigit()
            and 0 < int(numero) - int(numeros[-1]) <= 30
        ):
            numeros.extend(str(n) for n in range(int(numeros[-1]) + 1, int(numero)))
        if numero and numero not in numeros:
            numeros.append(numero)
        precedent = trouve
    return tuple(numeros)


# Verbes anglais que etendue_du_nom ne connaît pas : « Articles 5 and 6 of the
# Labour Code apply » ne nomme que le « Labour Code ».
_VERBE_ANGLAIS = re.compile(
    r"\s+(?:apply|applies|provide|punish|protect|protects|state|require|allow|prohibit|cover"
    r"|define|guarantee|grant|set|say|govern|regulate|impose|give|lay)\b.*$",
    re.IGNORECASE | re.DOTALL,
)


def _nom_apres(texte: str, fin: int) -> Tuple[Optional[str], str, int]:
    """(nom, liaison, fin de la mention) : le nom du texte qui suit les numéros, s'il y en a un."""
    trouve = _LIAISON.match(texte, fin)
    if trouve is None:
        return None, "du", fin
    brut = trouve.group("nom")
    longueur = etendue_du_nom(brut)
    if longueur <= 0:
        return None, "du", fin
    nom = _VERBE_ANGLAIS.sub("", brut[:longueur]).strip()
    if not nom:
        return None, "du", fin
    return nom, " ".join(trouve.group("liaison").split()), trouve.start("nom") + len(nom)


def _nom_avant(texte: str, debut: int) -> Tuple[Optional[str], str]:
    """Le nom écrit AVANT « en son article » (« Le Code pénal, en son article 312 »)."""
    avant = texte[:debut]
    tournure = _EN_SON.search(avant)
    if tournure is None:
        return None, "du"
    proposition = re.split(r"[.;:!?\n]", avant[:tournure.start()])[-1]
    trouve = _NOM_AVANT.search(proposition)
    if trouve is None:
        return None, "du"
    nom = (trouve.group("nom") or trouve.group("nom_minuscule")).strip()
    determinant = (
        trouve.group("determinant") or trouve.group("determinant_minuscule") or ""
    ).strip().lower()
    if determinant == "the" or re.match(r"[A-Z][a-z]+\s+Code\b|Law|Act|Decree", nom):
        return nom, "of the"
    if determinant in ("la",):
        return nom, "de la"
    if determinant in ("l'", "l’"):
        return nom, "de l'"
    return nom, "du"


def mentions_d_articles(texte: Optional[str]) -> List[MentionDArticle]:
    """Toutes les mentions d'articles du texte, dans l'ordre."""
    texte = texte or ""
    mentions: List[MentionDArticle] = []
    for trouve in _MENTION.finditer(texte):
        if trouve.group("mot").lower().startswith("section") and _SECTION_FRANCAISE.search(
            texte[max(0, trouve.start() - 12):trouve.start()]
        ):
            continue
        numeros = _numeros_de(trouve.group("numeros"))
        if not numeros:
            continue
        nom, liaison, fin = _nom_apres(texte, trouve.end())
        if nom is None:
            nom, liaison = _nom_avant(texte, trouve.start())
        renvoi = bool(nom) and (bool(_RENVOI.search(nom)) or bool(_NATURE_SEULE.match(nom)))
        mentions.append(MentionDArticle(
            debut=trouve.start(), fin=fin,
            texte=" ".join(texte[trouve.start():fin].split()),
            numeros=numeros, nom=nom, liaison=liaison, renvoi=renvoi,
        ))
    return mentions


# ==================== CE QU'AUCUN EXTRAIT NE PORTE ====================


def _numeros_lus(extraits: Sequence[ChunkResult]) -> Set[str]:
    """Les numéros d'articles des extraits, « 55 » compris pour le morceau « 55.1 »."""
    numeros: Set[str] = set()
    for extrait in extraits:
        numero = normalize_number(extrait.number or "")
        if numero:
            numeros.add(numero)
            base = re.fullmatch(r"(\d+)\.\d+", numero)
            if base:
                numeros.add(base.group(1))
    return numeros


def _numero_propre(extrait: Optional[ChunkResult]) -> str:
    """Le numéro de l'article dont l'extrait est le texte (« 55 » pour le morceau « 55.1 »)."""
    numero = normalize_number((extrait.number if extrait is not None else "") or "")
    base = re.fullmatch(r"(\d+)\.\d+", numero)
    return base.group(1) if base else numero


def _porte_sous_ce_nom(numero: str, liaison: str, nom: str, extraits: Sequence[ChunkResult]) -> bool:
    """
    Un extrait porte-t-il l'article `numero` du texte `nom` ? La lecture des
    citations du chat en juge, sur la citation recomposée au singulier : elle
    seule sait reconnaître un nom de texte (intitulé, nature, numéro, sigle).

    Et l'extrait qui la porte doit ÊTRE cet article. Dans le blog, chaque
    extrait est UN article (un passage) ; la lecture du chat, elle, rattache
    aussi une citation au morceau qui porte l'en-tête de l'article dans son
    texte. Revue du tour 2 : l'article 1 du décret 2021/597 CITE « «ARTICLE
    2.- (nouveau) … » » d'un autre décret, et « l'article 2 du Décret
    N°2021/597 » passait pour porté ; or l'article 2 de ce décret est « Le
    présent décret sera enregistré… ».
    """
    liaison = "de l'" if liaison in ("de l'", "de l’", "d'", "d’") else liaison
    separateur = "" if liaison.endswith(("'", "’")) else " "
    mots = nom.split()
    # Le nom lu peut emporter la suite de la phrase (« du Code civil règlent le
    # congé ») : etendue_du_nom ne connaît pas tous les verbes. Un nom qu'aucun
    # texte lu ne porte est relu sans ses derniers mots, tant qu'il nomme
    # encore un texte : « Code civil » est reconnu, l'article jugé sous ce nom.
    # Jamais plus court que le mot de nature et le mot qui le distingue.
    for longueur in range(len(mots), 0, -1):
        essai = " ".join(mots[:longueur])
        if longueur < len(mots) and (nature_nommee(essai) is None or longueur < 2):
            break
        phrase = f"L'article {numero} {liaison}{separateur}{essai}."
        occurrences = [o for o in lire_les_citations(phrase, extraits) if o.forme == "article"]
        if not occurrences:
            return False
        if not any(o.raison == "texte_non_consulte" for o in occurrences):
            return not any(o.introuvable for o in occurrences) and all(
                _numero_propre(o.extrait) == numero for o in occurrences
            )
    return False


def numeros_non_portes(
    texte: Optional[str], extraits: Sequence[ChunkResult]
) -> List[Tuple[MentionDArticle, Tuple[str, ...]]]:
    """
    Chaque mention du texte dont un numéro au moins n'est porté par aucun
    extrait, avec ces numéros-là.
    """
    lus = _numeros_lus(extraits)
    resultat: List[Tuple[MentionDArticle, Tuple[str, ...]]] = []
    # Le nom de la dernière mention qui en écrit un : l'antécédent d'un renvoi.
    antecedent: Optional[Tuple[str, str]] = None
    for mention in mentions_d_articles(texte):
        nom, liaison = mention.nom, mention.liaison
        if nom and mention.renvoi:
            nom, liaison = antecedent if antecedent else (None, "du")
        elif nom and nature_nommee(nom) is None and not re.search(r"\d", nom):
            # « de la sorte », « of the contract » : ce qui suit ne nomme aucun texte.
            nom = None
        elif nom:
            antecedent = (nom, liaison)
        manquants = tuple(
            numero for numero in mention.numeros
            if not (_porte_sous_ce_nom(numero, liaison, nom, extraits) if nom else numero in lus)
        )
        if manquants:
            resultat.append((mention, manquants))
    return resultat


# ==================== TEXTES NOMMÉS SANS ARTICLE ====================

# Un texte nommé par sa nature en majuscule : « le Code de la famille », « la
# Constitution », « l'Acte uniforme relatif au droit commercial général ».
# Une loi en minuscule (« la loi pénale s'impose à tous ») n'en nomme aucun.
_TEXTE_NOMME = re.compile(
    r"(?<!\w)(?P<nom>(?:Code|Constitution|Acte\s+uniforme|Actes\s+uniformes|Charte"
    r"|Penal\s+Code|Labour\s+Code|Civil\s+Code|Criminal\s+Procedure\s+Code|Uniform\s+Act)"
    # Un nom s'arrête avant « et » / « ou » : « le Code pénal et l'Acte uniforme
    # sur les sûretés » nomme deux textes, jugés chacun.
    r"(?:(?!\s(?:et|ou|and|or)\s)(?:[^.,;:!?()\[\]\n«»\"“”*_]|\.(?=\s*\d)))*)"
)


def textes_nommes_non_portes(texte: Optional[str], extraits: Sequence[ChunkResult]) -> List[str]:
    """
    Les textes nommés (sans numéro d'article) qu'aucun extrait ne porte. La
    lecture des citations du chat en juge sur une citation recomposée avec un
    numéro qu'aucun texte n'a : « texte non consulté » veut dire qu'aucun
    extrait n'est de ce texte.
    """
    texte = texte or ""
    occupes = [(m.debut, m.fin) for m in mentions_d_articles(texte)]
    noms: List[str] = []
    for trouve in _TEXTE_NOMME.finditer(texte):
        if any(debut <= trouve.start() < fin for debut, fin in occupes):
            continue
        brut = trouve.group("nom")
        nom = brut[:etendue_du_nom(brut)].strip()
        if not nom or nom in noms:
            continue
        phrase = f"L'article 999999 du {nom}."
        occurrences = [o for o in lire_les_citations(phrase, extraits) if o.forme == "article"]
        if any(o.raison == "texte_non_consulte" for o in occurrences):
            noms.append(nom)
    return noms


# ==================== DROIT D'UN AUTRE PAYS ====================

_PAYS = (
    r"(?:fran[çc]ais(?:e|es)?|belges?|s[ée]n[ée]galais(?:e|es)?|ivoirien(?:ne)?s?"
    r"|gabonais(?:e|es)?|nig[ée]rian(?:e|es)?|am[ée]ricain(?:e|es)?|britanniques?"
    r"|de\s+France|de\s+Belgique|du\s+S[ée]n[ée]gal|de\s+C[ôo]te\s+d['’]Ivoire|du\s+Gabon"
    r"|du\s+Nig[ée]ria|French|Belgian|Senegalese|Nigerian|American|British)"
)
_DROIT_ETRANGER = re.compile(
    r"(?<![\w’'])(?:(?:droit|lois?|codes?|code\s+(?:civil|p[ée]nal|du\s+travail|de\s+commerce)"
    r"|jurisprudence|l[ée]gislation|r[ée]glementation)\s+(?:[^\W\d_]+\s+){0,2}" + _PAYS + r"\b"
    r"|" + _PAYS + r"\s+(?:law|code|civil\s+code|penal\s+code|labour\s+code|legislation|courts?)\b"
    r"|Cour\s+de\s+cassation|Conseil\s+d['’][ÉE]tat\s+fran[çc]ais)",
    re.IGNORECASE,
)


def droit_etranger(texte: Optional[str]) -> List[str]:
    """Les mentions du droit d'un autre pays, telles qu'écrites."""
    vues: List[str] = []
    for trouve in _DROIT_ETRANGER.finditer(texte or ""):
        mention = " ".join(trouve.group().split())
        if mention not in vues:
            vues.append(mention)
    return vues


def en_toutes_lettres(numeros: Iterable[str]) -> str:
    """« l'article 312 », « les articles 286 et 312 »."""
    liste = list(numeros)
    if len(liste) == 1:
        return f"l'article {liste[0]}"
    return f"les articles {', '.join(liste[:-1])} et {liste[-1]}"
