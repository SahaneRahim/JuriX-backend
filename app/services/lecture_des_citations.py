"""
Lecture des citations d'une reponse du chat : ou elles sont dans le texte, ce
qu'elles designent, et celles qu'aucun texte consulte ne porte.

POURQUOI CE MODULE. `RAGService._extract_citations` ne rendait que des sources
validees et dedoublonnees, sans leur place dans la reponse, et une citation
que rien ne portait etait jetee sans un mot (« if chunk is None: continue ») :
le lecteur voyait « l'article 312 du Code pénal » sans source ni alerte. La
lecture rend desormais chaque OCCURRENCE : sa position, l'extrait qui la
porte, ou la raison pour laquelle aucun ne la porte. Les sources d'avant en
sont tirees, une par article, avec les memes regles de rattachement.

CE QUI COMPTE COMME CITATION. Deux formes :
- « l'article X du Code Y » (CITATION_REGEX), la grammaire que les prompts
  imposent ;
- un texte designe par son numero, « le décret N°2018/9387/CAB/PM »,
  « l'arrêté n° 016/MTPS/SG/CJ du 26 mai 1993 ». Le modele en nomme sans
  article, et c'etait precisement ce qui s'affichait sans lien ni alerte.
Un code nomme sans numero ni article (« le Code du travail prévoit… ») n'en
est pas une : le signaler a chaque mention serait du bruit.

Les primitives de lecture d'un numero d'article et CITATION_REGEX vivaient
dans rag_service.py ; elles sont ici pour que la lecture se teste seule, et
rag_service les reexporte sous leurs noms d'origine.
"""

import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from typing import (
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from app.schemas.search import ChunkResult
from app.services.liens_entre_textes import LIENS
from app.services.reperes_textes import dates_de_l_acte, en_tete_de_l_intitule, numero_de_l_acte
from app.utils.chunk_refiner import est_pseudo_numero
from app.utils.titre_officiel import numero_du_titre

# ==================== NUMEROS D'ARTICLES ====================

# Formes ordinales rencontrees dans le corpus, ramenees a leur rang.
_ORDINAL_WORDS = {
    "PREMIER": "1", "PREMIERE": "1", "PREMIÈRE": "1", "1ER": "1", "1ÈRE": "1",
    "FIRST": "1", "DEUXIEME": "2", "DEUXIÈME": "2", "SECOND": "2", "SECONDE": "2",
    "TROISIEME": "3", "TROISIÈME": "3",
}


def _citable_article_number(number: Optional[str]) -> Optional[str]:
    """Numero d'article citable, ou None pour un pseudo-numero."""
    if not number or est_pseudo_numero(number):
        return None
    return number


def _normalize_article_number(number: str) -> str:
    """
    Normalise un numero d'article pour comparaison.

    Les numeros circulent sous des formes multiples — "1", "1er", "PREMIER",
    "Article 1" — et etaient compares tantot par egalite de chaines, tantot par
    inclusion. L'inclusion faisait correspondre "1" a "10", "11", "100" : la
    mauvaise citation etait alors rattachee a la reponse.
    """
    if not number:
        return ""
    cleaned = str(number).strip().upper()
    cleaned = re.sub(r"^(ARTICLE|ART\.?|SECTION)\s+", "", cleaned)
    cleaned = cleaned.strip(" .:-")
    return _ORDINAL_WORDS.get(cleaned, cleaned)


def _numero_de_base(number: Optional[str]) -> Optional[str]:
    """
    « 55 » pour le morceau « 55.2 » : le decoupage range un article trop long
    sous « 55.1 », « 55.2 »… (chunk_refiner), et 5 961 morceaux pointes du
    corpus n'ont aucun « 55 » a cote d'eux. Sans cela, « l'article 55 du Code
    électoral » ne retrouvait jamais l'article qu'il cite, et serait signale
    comme introuvable.
    """
    if not number or not re.fullmatch(r"\s*\d+\.\d+\s*", str(number)):
        return None
    return _normalize_article_number(str(number).split(".")[0])


def article_compte(number: Optional[str]) -> Optional[str]:
    """
    L'article auquel appartient un morceau, pour COMPTER les articles trouves :
    « 55.1 » et « 55.2 » font un seul article 55, « SECTION_9.26 » la section
    9. None pour une ligne niveau loi, qui n'est pas un article.
    """
    if not number or not str(number).strip():
        return None
    return _normalize_article_number(re.sub(r"\.\d+\s*$", "", str(number).strip()))


# Couvre du / de la / de l' / des / de, et la numerotation composee
# (161, 1er, 94-2, L 94 septies). L'ancienne version ne connaissait que
# "de la " : "article 161 du Code OHADA", la formulation la plus courante
# en francais juridique, n'etait JAMAIS reconnue — les reponses du chatbot
# sortaient donc sans aucune source.
CITATION_REGEX = (
    # « art. » : les prompts l'interdisent, le modele l'ecrit quand meme,
    # surtout entre parentheses (« (art. 15 du Décret N°2019/143) ») —
    # mesure du 2026-10-11, un tiers des citations illisibles de l'avocat.
    # Illisibles, elles ne devenaient pas des sources. \b : « départ. 12 »
    # n'est pas une citation.
    r"\b[Aa]rt(?:icle|\.)\s*"
    r"([LRD]\s*)?(\d+(?:[-.]\d+)*(?:\s*(?:er|bis|ter|quater|septies))?)"
    # Paragraphe a l'anglaise, colle au numero : « Article 33(2) of… ».
    r"(?:\(\d+\))?"
    # Un alinea glisse entre le numero et le texte : « l'article 5, alinéa 1,
    # du Décret N°2021/743 », « alinéa 1er », « alinéa premier »,
    # « paragraphe 2 ». Lu puis ignore : la source est l'article.
    r"(?:,?\s+(?:alin[ée]a|al\.|paragraphe?)\s*(?:\d+(?:\s*er)?|premier),?)?"
    # `of the` / `of` : sans eux, AUCUNE reponse anglaise n'a jamais produit
    # une seule citation extraite. « According to Article 161 of the OHADA
    # Code » echouait sur la contrainte de majuscule du groupe suivant, qui
    # butait sur le « of » minuscule. Toutes les sources anglaises affichees
    # jusqu'ici venaient donc du repli `_create_sources_from_results`,
    # c'est-a-dire n'etaient pas des citations mais des devinettes.
    r"\s+(?:du\s+|de\s+la\s+|de\s+l['’]\s*|des\s+|de\s+|of\s+the\s+|of\s+)?"
    # Le nom du texte en gras ou en italique : « l'article 337 de la **Loi N°
    # 2016/007 portant Code Pénal** ». Sans ce groupe, le texte cite devenait
    # « de la » : aucun nom, et l'article 337, pourtant lu par le modele,
    # etait signale introuvable (mesure du 2026-10-28 sur 323 reponses
    # d'evaluation, 3 fausses alertes). NON CAPTURANT : les tests des prompts
    # et scripts/eval lisent les groupes 2 et 3.
    r"(?:\*{1,3}|_{1,3})?"
    # Le texte cite s'arrete a la ponctuation, a une parenthese, et avant
    # la citation suivante : les prompts demandent « l'article 5 du Code
    # Minier et l'article 6 du Code Minier », et ce groupe avalait la
    # seconde, qui ne devenait jamais une source.
    r"([A-ZÀ-Ý](?:(?!\s+(?:et|ou|and|or)\s+(?:l['’]\s*)?art(?:icle|\.))[^,\.;()])+)"
)

# Ce qui suit « article 37 » dans le texte d'une citation, quand la citation
# s'arrete a sa designation : le mot de liaison et une marque d'emphase.
_LIAISON_FINALE = re.compile(
    r"(?:[\s,]+(?:du|de\s+la|de\s+l['’]|des|de|of\s+the|of))?[\s,*_]*$", re.IGNORECASE
)


def _fin_de_la_designation(texte: str, citation: "re.Match") -> int:
    """
    Fin de « article 37 » (alinea compris), sans le mot de liaison qui suit
    ni une marque d'emphase : la citation qui ne nomme aucun texte (« L'article
    37 précise que… ») ou dont le nom est en gras s'arrete la.

    POURQUOI JAMAIS DE MARQUE D'EMPHASE DANS UNE CITATION. Le front relit le
    markdown pour l'afficher ; une citation qui couvrirait le « ** » ouvrant
    d'un gras sans son fermant l'obligerait a couper ce gras en deux.
    """
    avant = texte[citation.start():citation.start(3)]
    return citation.start() + _LIAISON_FINALE.search(avant).start()


# Un « _ » qui ouvre un italique : precede d'un blanc ou d'une ponctuation,
# suivi d'une lettre ou d'un chiffre (« la loi _n° 92/007_ »).
_SOULIGNE_OUVRANT = re.compile(r"(?<!\w)_(?=\w)")
_EMPHASE = re.compile(r"[*_]")


def _avant_l_emphase(texte: str, citation: "re.Match", fin: int) -> int:
    """
    `fin`, ramenee avant la premiere marque d'emphase du passage cite, sans le
    mot de liaison qui la precede, et jamais en deca de la designation.

    « l'article 35 de la loi _n° 92/007_ » soulignait « article 35 de la loi
    _n° 92/007 » : un « _ » ouvrant sans son fermant, que le front ne peut pas
    couper (revue du 2026-10-28). Le passage devient « article 35 de la loi »,
    comme un nom en gras s'arrete a « article 35 » : un passage cite ne
    contient jamais « * » ni « _ ».
    """
    marque = _EMPHASE.search(texte, citation.start(), fin)
    if marque is None:
        return fin
    avant = texte[citation.start():marque.start()]
    return max(
        citation.start() + _LIAISON_FINALE.search(avant).start(),
        _fin_de_la_designation(texte, citation),
    )


# ==================== NOMS DE TEXTES ====================

# « N° 92/007 », « n°92/007 », « Nº 92/007 », « No. 2016/007 » : une seule forme.
_NUMERO_DE_TEXTE = re.compile(r"(?<!\w)n\s*(?:°|º|o\.?)\s*(?=\d)")


def normaliser_le_nom(texte: Optional[str]) -> str:
    """
    Forme de comparaison d'un nom de texte : minuscules, sans accents,
    apostrophes et blancs unifies, « N° 92/007 » ecrit « n°92/007 ».

    Le modele ecrit « Code penal », la base « CODE PÉNAL » ; il ecrit « Loi n°
    2016/007 », la base « Loi N°2016/007 ». Sans forme commune, le meme texte
    passait pour deux, et sa citation restait sans source.
    """
    decompose = unicodedata.normalize("NFD", texte or "")
    plat = "".join(c for c in decompose if not unicodedata.combining(c)).casefold()
    plat = plat.replace("’", "'").replace("‘", "'")
    plat = _NUMERO_DE_TEXTE.sub("n°", plat)
    return " ".join(plat.split())


# Mots qui relient les mots d'un nom sans le designer (formes normalisees).
_LIAISONS = frozenset({
    "du", "de", "des", "la", "le", "les", "l", "d", "et", "a", "au", "aux", "en", "sur",
    "par", "portant", "relatif", "relative", "relatifs", "relatives", "n°",
    "of", "the", "and", "on", "to", "relating",
})
# Mots qui ne designent aucun texte a eux seuls : « Code » nomme tous les codes,
# « Acte uniforme » les dix actes OHADA.
_GENERIQUES = frozenset({
    "code", "codes", "loi", "lois", "decret", "decrets", "arrete", "arretes", "ordonnance",
    "ordonnances", "acte", "actes", "uniforme", "uniformes", "traite", "texte", "textes",
    "titre", "chapitre", "section", "decision", "circulaire", "ohada", "general", "generale",
    "law", "laws", "decree", "act", "order", "ordinance", "uniform", "treaty", "text",
})


def _mots_du_nom(nom_normalise: str) -> List[str]:
    return [mot for mot in re.split(r"[\s']+", nom_normalise) if mot]


def _prefixe_significatif(prefixe: str) -> bool:
    """Un debut de mention qui peut, seul, designer un texte."""
    mots = _mots_du_nom(prefixe)
    if not mots or mots[-1] in _LIAISONS:
        return False
    return any(mot not in _LIAISONS and mot not in _GENERIQUES for mot in mots)


def _prefixes(mention: str) -> List[Tuple[int, str]]:
    """
    Les debuts de la mention qui peuvent nommer un texte, du plus long au plus
    court : (longueur dans la mention, forme normalisee).

    POURQUOI DES DEBUTS. Le texte cite par CITATION_REGEX court jusqu'a la
    ponctuation : « l'article 37 du Code du travail prévoit que le salarié… »
    lui donne « Code du travail prévoit que le salarié ». Exige en entier dans
    le titre, il ne nommait aucune loi, et la citation se rattachait au premier
    article 37 venu, de n'importe quel texte. La mention entiere reste
    candidate, comme avant ; ses debuts doivent porter un mot qui designe
    (« travail »), pas seulement « Code ».
    """
    mots = list(re.finditer(r"\S+", mention))
    rendus = []
    for nombre in range(len(mots), 0, -1):
        fin = mots[nombre - 1].end()
        prefixe = normaliser_le_nom(mention[:fin])
        if nombre == len(mots) or _prefixe_significatif(prefixe):
            rendus.append((fin, prefixe))
    return rendus


def _nomme(nom_normalise: str, prefixe: str) -> bool:
    """Le nom contient-il ce debut de mention, en mots entiers ?"""
    if not prefixe:
        return False
    return re.search(r"(?<!\w)" + re.escape(prefixe) + r"(?!\w)", nom_normalise) is not None


def _reconnaitre(mention: str, noms: Dict[int, Tuple[str, str]]) -> Tuple[int, Set[int]]:
    """
    (longueur de la mention qui nomme un texte consulte, textes ainsi nommes).

    Le plus long debut reconnu l'emporte : « Code général des impôts » ne doit
    pas s'arreter a un « Code général » qu'un autre titre contiendrait.
    """
    for fin, prefixe in _prefixes(mention):
        lois = {
            law_id for law_id, (titre, reference) in noms.items()
            if _nomme(titre, prefixe) or _nomme(reference, prefixe)
        }
        if lois:
            return fin, lois
    return 0, set()


# Ce qui ne fait jamais partie du nom d'un texte. Le texte cite court jusqu'a
# la ponctuation forte : il emportait un retour a la ligne et le titre markdown
# suivant, ou les « ** » d'une citation en gras, et ne nommait plus rien. Un
# « _ » colle a la fin d'un mot ferme un italique (« _Code pénal_ prévoit »).
_HORS_DU_NOM = re.compile(r"[\r\n*«»\"“”:!?\[\]|]|_{2}|(?<=\w)_(?!\w)|\s[–—-]\s")


def _mention_utile(mention: str) -> str:
    coupe = _HORS_DU_NOM.search(mention)
    return (mention[:coupe.start()] if coupe else mention).rstrip()


# Mots qui enchainent la phrase apres le nom d'un texte : le verbe dont la
# citation est le sujet, ou la proposition qui s'y accroche. En minuscules
# AVEC accents : « a » (verbe) n'est pas « à » (« relatif à »).
_SUITES = frozenset({
    "dispose", "disposent", "prévoit", "prévoient", "précise", "précisent", "stipule",
    "stipulent", "énonce", "énoncent", "indique", "indiquent", "fixe", "fixent", "définit",
    "définissent", "punit", "punissent", "réprime", "sanctionne", "impose", "imposent",
    "oblige", "obligent", "permet", "permettent", "interdit", "interdisent", "exige",
    "exigent", "garantit", "consacre", "reconnaît", "reconnait", "établit", "mentionne",
    "accorde", "confère", "rappelle", "ajoute", "organise", "régit", "encadre", "prescrit",
    "autorise", "dit", "est", "sont", "était", "sera", "seront", "a", "ont", "peut",
    "peuvent", "doit", "doivent", "reste", "demeure", "modifié", "modifiée", "modifiés",
    "abrogé", "abrogée", "complété", "complétée", "remplacé", "remplacée",
    "qui", "que", "dont", "où", "lequel", "laquelle", "lesquels", "lesquelles", "lorsque",
    "quand", "si", "car", "mais", "donc", "ainsi", "alors", "également", "aussi",
    "notamment", "toutefois", "cependant", "néanmoins", "pour", "selon", "afin", "dans",
    "avec", "sans", "sous", "entre", "contre", "depuis", "pendant", "après", "avant",
    "lors", "il", "elle", "ils", "elles", "on",
    # Jamais dans le nom d'un texte : « l'article 312 du Code pénal ne
    # s'applique pas » soulignait « Code pénal ne ».
    "ne", "se", "y", "lui", "leur",
    "provides", "states", "says", "sets", "requires", "prohibits", "allows", "is", "are",
    "was", "were", "may", "shall", "must", "can", "which", "that", "where", "when", "if",
    "who", "under", "for", "with", "as", "does", "do", "did", "has", "have", "will",
    "would", "should", "could", "not", "also",
})
_ELISIONS_DE_SUITE = frozenset({"qu", "s", "n"})  # « qu'il », « s'applique », « n'est »
_MOTS_DU_NOM_AU_PLUS = 12


def _enchaine_la_phrase(mot: str) -> bool:
    """Le mot continue la phrase au lieu de nommer un texte : verbe, conjonction…"""
    brut = mot.casefold()
    elision = re.split(r"['’]", brut, maxsplit=1)[0] if re.search(r"['’]", brut) else None
    return brut in _SUITES or elision in _ELISIONS_DE_SUITE


def ne_nomme_rien(mention: str) -> bool:
    """
    « L'article 37 précise que… » : apres le numero, la phrase continue. Le
    premier mot n'etait pas controle, et la citation devenait « article 37
    précise » ; pire, « est » (« L'article 5 est abrogé ») designait la loi
    dont le titre contient « de l'Est ».
    """
    premier = re.match(r"\S+", mention)
    return premier is None or _enchaine_la_phrase(premier.group())


def etendue_du_nom(mention: str) -> int:
    """
    Longueur du nom de texte en tete de la mention, quand aucun texte consulte
    ne le reconnait : jusqu'au premier mot qui enchaine la phrase, sans finir
    sur un mot de liaison. Une estimation, pour souligner la citation dans le
    texte ; jamais pour la rattacher a une source. 0 : la mention ne nomme rien.
    """
    gardes = []
    for mot in list(re.finditer(r"\S+", mention))[:_MOTS_DU_NOM_AU_PLUS]:
        if _enchaine_la_phrase(mot.group()):
            break
        gardes.append(mot)
    while gardes and normaliser_le_nom(gardes[-1].group()) in _LIAISONS:
        gardes.pop()
    return gardes[-1].end() if gardes else 0


# « de la même loi », « dudit décret » : le texte de la citation precedente.
# Sans antecedent, le numero seul departageait, et l'article 132 « de la même
# loi » pouvait devenir celui d'une autre loi consultee.
_ANAPHORE = re.compile(
    r"(?:m[êe]mes?|ledit|ladite|lesdits|lesdites|dudit|desdits|desdites|said|same)\b",
    re.IGNORECASE,
)


@lru_cache(maxsize=256)
def _entete_d_article(voulu: str) -> "re.Pattern":
    """
    L'en-tete de l'article `voulu` dans le texte d'un morceau : « ARTICLE 142
    (nouveau).- », « Art. 6 : », « Article premier.- ».

    POURQUOI. Une loi modificative est souvent rangee sous un pseudo-numero :
    la loi 2026/003 porte son article 142 dans le morceau « SECTION_1 », et
    « l'article 142 de la Loi N°2026/003 », que le modele avait lu, etait
    signale introuvable (4 fausses alertes sur 323 reponses d'evaluation).
    Strict a dessein : un separateur apres le numero, et pas d'apostrophe
    avant, pour que « prévus à l'article 4 ci-dessus » ne passe pas pour
    l'article 4.

    Deux formes reelles s'y ajoutent (revue du 2026-10-28) : le point double
    de l'OCR (« Art..5- (1) », fin du morceau « 2 » de la loi 90/053) et
    l'alinea reecrit d'une loi modificative (« ARTICLE 6. (2) (nouveau) : »,
    loi 2008/001). Sans elles, une citation que le numero propre rattache
    desormais au bon texte y etait dite non consultee.
    """
    numero = r"(?:1|premi[eè]re?)" if voulu == "1" else re.escape(voulu)
    return re.compile(
        r"(?<![\w’'])art(?:icle)?\.{0,2}\s*" + numero
        + r"(?:\s*(?:er|ère|re|e))?(?![\w.]?\d)(?:\.?\s*\(\d+\))?"
        r"\s*(?:\((?:nouveau|nouvelle|new)\)\s*)?"
        r"(?:\.\s*[-–—]|[-–—:])",
        re.IGNORECASE,
    )


def porte_l_entete(contenu: Optional[str], numero: Optional[str]) -> bool:
    """
    Le morceau porte-t-il l'en-tete de l'article `numero` ?

    Public : l'apercu d'un article (GET /laws/{id}/articles/{numero}) ne rend un
    morceau « englobant » qu'a cette preuve, celle-la meme qui rattache la
    citation du chat a ce morceau (_resoudre_article). Un couple incoherent
    (article 999, morceau de l'article 35.1) etait rendu comme si le morceau
    contenait l'article.
    """
    voulu = _normalize_article_number(numero or "")
    return bool(voulu) and _entete_d_article(voulu).search(contenu or "") is not None


# ==================== TEXTES DESIGNES PAR LEUR NUMERO ====================

# « le décret N°2018/9387/CAB/PM », « l'arrêté n° 016/MTPS/SG/CJ du 26 mai
# 1993 », « Law No. 2016/007 of 12 July 2016 ». Le numero commence par un
# chiffre et finit sur une lettre ou un chiffre : « 92/007. » rend « 92/007 »,
# et « _le décret n° 94/199_ » (italique) rend « 94/199 ». Un « _ » y etait
# avale (« 94/199_ ») : le numero n'etait plus celui du decret lu, la
# citation etait signalee a tort, et le passage cite portait un « _ » que le
# front ne peut pas couper (revue du 2026-10-28). Le mois de la date, en
# lettres seulement, pour la meme raison.
TEXTE_NUMEROTE_REGEX = re.compile(
    r"\b(?P<nature>lois?|d[ée]crets?|arr[êe]t[ée]s?|ordonnances?|d[ée]cisions?|circulaires?"
    r"|laws?|decrees?|orders?|ordinances?|circulars?)"
    r"(?:\s+(?:organique|présidentiel(?:le)?|conjoint|ministériel(?:le)?|organic|presidential))?"
    r"\s+(?:n\s*[°º]|no\.?|num[ée]ro|number)\s*"
    r"(?P<numero>\d(?:[^\W_]|[/\-])*[^\W_]|\d)"
    r"(?P<date>\s+(?:du|of)\s+\d{1,2}(?:er|st|nd|rd|th)?\s+[^\W\d_]+\.?,?\s+\d{4})?",
    re.IGNORECASE,
)

# Une nature, et celles qui la traduisent : un titre en anglais (« LAW No.
# 2016/007 ») repond a « la loi n° 2016/007 ».
_NATURES_EQUIVALENTES = {
    "loi": ("loi", "law"),
    "law": ("loi", "law"),
    "decret": ("decret", "decree"),
    "decree": ("decret", "decree"),
    "arrete": ("arrete", "order"),
    "ordonnance": ("ordonnance", "ordinance", "order"),
    "ordinance": ("ordonnance", "ordinance", "order"),
    "order": ("arrete", "ordonnance", "order", "ordinance"),
    "decision": ("decision",),
    "circulaire": ("circulaire", "circular"),
    "circular": ("circulaire", "circular"),
}


def _natures(mot: str) -> Tuple[str, ...]:
    nature = normaliser_le_nom(mot)
    if nature not in _NATURES_EQUIVALENTES and nature.endswith("s"):
        nature = nature[:-1]
    return _NATURES_EQUIVALENTES.get(nature, (nature,))


def _forme_des_numeros(texte: str) -> str:
    """« 92-007 » et « 92/007 » : le meme numero."""
    return texte.replace("-", "/")


# Les mots de nature, sous la forme de normaliser_le_nom.
_MOTS_DE_NATURE = frozenset(mot for mots in _NATURES_EQUIVALENTES.values() for mot in mots)


class NumeroPropre(NamedTuple):
    """Le numero d'un texte et sa nature, lus en tete de SON intitule."""

    numero: str
    """Forme de comparaison : minuscules, sans accents, « - » ecrit « / »."""
    nature: str
    """Premier mot de nature de l'en-tete (« loi », « decret », « law »…)."""


@lru_cache(maxsize=4096)
def numero_propre(titre: Optional[str]) -> Optional[NumeroPropre]:
    """
    Le numero de CE texte et sa nature ; None si son intitule n'en porte pas
    (codes, traites, noms de fichier).

    Le numero est celui des lignes de liste (titre_officiel.numero_du_titre),
    qui repare l'OCR (« N°202l/345 » se lit 2021/345), ou a defaut le premier
    « N° » de l'en-tete (« la Loi N°2008/001 », « Loi organique N°… »,
    « Constitution (Loi N°96/06 … », « LAW No. 2016/007 … »). Jamais un numero
    de l'OBJET : « Décret N°2015/407 … portant application de la loi n°97/021 »
    est le decret 2015/407, pas la loi 97/021. La nature est le premier mot de
    nature de l'en-tete, pour la meme raison.
    """
    if not titre:
        return None
    numero = numero_du_titre(titre) or numero_de_l_acte(titre)
    if not numero:
        return None
    mots = re.findall(r"[a-z]+", normaliser_le_nom(en_tete_de_l_intitule(titre)))
    nature = next((mot for mot in mots if mot in _MOTS_DE_NATURE), None)
    if nature is None:
        return None
    return NumeroPropre(_forme_des_numeros(normaliser_le_nom(numero)), nature)


def porte_le_numero(nom: Optional[str], nature: str, numero: str) -> bool:
    """
    Le titre est-il celui du texte de cette nature et de ce numero ?

    Compare au numero PROPRE du texte (numero_propre). La recherche se faisait
    dans tout l'intitule : sur 323 vraies reponses, 18 citations etaient liees
    a un texte qui ne faisait que CITER le numero (« la loi n°90/053 » a la loi
    2021/022 « modifiant … la loi n° 90/053 », « la loi n°97/021 », absente du
    corpus, au decret 2015/407 qui l'applique), et un intitule abime par l'OCR
    (« N°202l/345 ») faisait signaler a tort une citation exacte (revue du
    2026-10-28).

    La nature compte : une loi et un decret peuvent porter le meme numero.
    """
    propre = numero_propre(nom)
    return (
        propre is not None
        and propre.numero == _forme_des_numeros(normaliser_le_nom(numero))
        and propre.nature in _natures(nature)
    )


# Mois d'une date ecrite par le modele (« du 19 décembre 1990 », « of 12 July
# 2016 »), sous la forme de normaliser_le_nom ; abreviations comprises.
_MOIS_ECRITS = {
    **dict.fromkeys(("janvier", "janv", "january", "jan"), 1),
    **dict.fromkeys(("fevrier", "fevr", "fev", "february", "feb"), 2),
    **dict.fromkeys(("mars", "march", "mar"), 3),
    **dict.fromkeys(("avril", "avr", "april", "apr"), 4),
    **dict.fromkeys(("mai", "may"), 5),
    **dict.fromkeys(("juin", "june", "jun"), 6),
    **dict.fromkeys(("juillet", "juil", "july", "jul"), 7),
    **dict.fromkeys(("aout", "august", "aug"), 8),
    **dict.fromkeys(("septembre", "sept", "sep", "september"), 9),
    **dict.fromkeys(("octobre", "oct", "october"), 10),
    **dict.fromkeys(("novembre", "nov", "november"), 11),
    **dict.fromkeys(("decembre", "dec", "december"), 12),
}


def date_ecrite(texte: Optional[str]) -> Optional[date]:
    """La date du groupe `date` de TEXTE_NUMEROTE_REGEX ; None si elle ne se lit pas."""
    trouve = re.search(
        r"(\d{1,2})(?:er|st|nd|rd|th)?\s+([a-z]+)\.?,?\s+(\d{4})", normaliser_le_nom(texte)
    )
    mois = _MOIS_ECRITS.get(trouve.group(2)) if trouve else None
    if mois is None:
        return None
    try:
        return date(int(trouve.group(3)), mois, int(trouve.group(1)))
    except ValueError:
        return None


def de_cette_date(ecrite: Optional[date], jour: Optional[date], titre: Optional[str]) -> bool:
    """
    La date ecrite dans la citation est-elle celle du texte ? Comparee a sa date
    d'acte (reperes_textes.dates_de_l_acte), ou a sa seule annee quand le jour
    n'est pas sur. Rien d'ecrit, ou rien de connu : rien a contredire.

    « le décret n° 94/199 du 14 août 1992 » n'est pas le decret du 7 octobre
    1994 : le lier ouvrait un texte que la phrase ne nomme pas.
    """
    if ecrite is None:
        return True
    dates = dates_de_l_acte(jour, titre)
    if dates.date_signature is not None:
        return dates.date_signature == ecrite
    return dates.annee is None or dates.annee == ecrite.year


def designe_ce_texte(citation: "re.Match", titre: Optional[str], jour: Optional[date]) -> bool:
    """
    La citation (TEXTE_NUMEROTE_REGEX) designe-t-elle le texte de cet intitule
    et de cette date : sa nature, son numero propre et, si elle est ecrite, sa
    date ?
    """
    return porte_le_numero(titre, citation.group("nature"), citation.group("numero")) and (
        de_cette_date(date_ecrite(citation.group("date")), jour, titre)
    )


# ==================== OCCURRENCES ====================


@dataclass(frozen=True)
class TexteNomme:
    """
    Un texte que le modele ne lit pas mais qu'une ligne de statut lui nomme :
    « Statut : article modifié par la Loi N°2008/001 du 14 avril 2008 ».

    Le prompt demande de nommer le texte qui change une regle : le signaler
    comme introuvable punirait le modele d'avoir obei. Il est relie au texte
    en base par sa reference, ou laisse sans lien s'il manque au corpus.
    """

    titre: str
    reference: Optional[str]
    date: Optional[date] = None
    """Connue pour le texte qui touche (par_date) ; pas pour celui qu'il touche."""


@dataclass(frozen=True)
class Occurrence:
    """Une citation de la reponse : sa place, et ce qui la porte."""

    debut: int
    """Indice Python du premier caractere cite."""
    fin: int
    """Indice Python qui suit le dernier caractere cite."""
    forme: str
    """« article » (CITATION_REGEX) ou « texte » (texte designe par son numero)."""
    extrait: Optional[ChunkResult] = None
    """L'extrait consulte qui porte la citation ; None si aucun."""
    article: Optional[str] = None
    """Numero d'article a afficher et a ouvrir ; None : le texte entier."""
    cle: Optional[Tuple[int, str]] = None
    """(loi, numero cite normalise) : la cle de dedoublonnage des sources d'avant."""
    texte_nomme: Optional[TexteNomme] = None
    raison: Optional[str] = None
    """Pour une citation introuvable : article_non_consulte | texte_non_consulte."""

    @property
    def introuvable(self) -> bool:
        return self.raison is not None


class _Resolution(NamedTuple):
    extrait: Optional[ChunkResult]
    identique: bool
    """Le numero de l'extrait est celui de la citation (et non « 55.1 », « SECTION_1 »)."""
    etendue: int
    """Longueur du nom dans la mention ; 0 : la citation s'arrete a sa designation."""
    raison: Optional[str]
    designe: Optional[FrozenSet[int]]
    """
    Les textes que la citation designe, antecedent de « la même loi » qui suit.
    Vide : un texte non consulte. None : elle ne nomme rien, l'antecedent reste.
    """


def _resoudre_article(
    voulu: str,
    mention: str,
    extraits: Sequence[ChunkResult],
    noms: Dict[int, Tuple[str, str]],
    antecedent: Optional[FrozenSet[int]] = None,
) -> _Resolution:
    """
    Ce que designe « l'article `voulu` » suivi de `mention`.

    Ordre des regles, du plus sur au moins sur :
    1. le texte nomme est consulte — pour « la même loi », celui de la citation
       precedente ; pour « la loi n° 90/053 », celui dont c'est le numero
       PROPRE (designe_ce_texte), jamais un texte dont l'intitule ne fait que
       le citer : son article de ce numero, sinon un morceau de cet article
       (« 55.1 »), sinon le morceau qui en porte l'en-tete (« SECTION_1 »
       d'une loi modificative), sinon sa ligne niveau loi ; sinon RIEN —
       jamais l'article de meme numero d'une AUTRE loi presente ;
    2. la mention designe un texte par son numero, que rien de consulte ne
       porte, ou renvoie a une citation precedente qui n'en designait aucun :
       rien ;
    3. le texte n'est reconnu nulle part (le modele l'a nomme autrement, ou ne
       l'a pas nomme : « L'article 37 précise… ») : le premier extrait qui
       porte ce numero, comme avant, avec les replis de la regle 1 (morceau,
       puis en-tete). Rien : « article_non_consulte » si la citation ne nomme
       aucun texte, « texte_non_consulte » si elle en nomme un.
    """
    anaphore = _ANAPHORE.match(mention) is not None
    if ne_nomme_rien(mention):
        etendue, lois, designe = 0, frozenset(), None
    elif anaphore:
        etendue = etendue_du_nom(mention)
        if antecedent is not None and not antecedent:
            return _Resolution(None, False, etendue, "texte_non_consulte", frozenset())
        lois = antecedent or frozenset()
        designe = lois or None
    elif (numerote := TEXTE_NUMEROTE_REGEX.match(mention)) is not None:
        # « l'article 6 de la Loi N°2008/001 » : la loi 2008/001, et non la
        # loi 2026/002, dont l'intitule « modifiant … la loi N°2008/001 »
        # contient le meme nom (revue du 2026-10-28). Le nom souligne peut
        # aller au-dela du numero, mais dans l'intitule de CE texte seulement.
        lois = frozenset(
            e.law_id for e in extraits
            if designe_ce_texte(numerote, e.law_title, e.publication_date)
        )
        fin = _reconnaitre(mention, {i: noms[i] for i in lois if i in noms})[0] if lois else 0
        etendue, designe = max(numerote.end(), fin), lois
    else:
        etendue, lois = _reconnaitre(mention, noms)
        lois = frozenset(lois)
        designe = lois
    if lois:
        candidats = [e for e in extraits if e.law_id in lois]
        trouve = _l_article_parmi(voulu, candidats)
        if trouve is not None:
            return _Resolution(trouve[0], trouve[1], etendue, None, designe)
        entier = next((e for e in candidats if not e.number), None)
        if entier is not None:
            return _Resolution(entier, False, etendue, None, designe)
        return _Resolution(None, False, etendue, "article_non_consulte", designe)

    if designe is not None and not anaphore:
        numerote = TEXTE_NUMEROTE_REGEX.match(mention)
        if numerote:
            # Le nom s'arrete au numero (et a sa date) : « fixant les
            # modalités… » qui suit n'est pas a souligner.
            return _Resolution(None, False, numerote.end(), "texte_non_consulte", frozenset())
        etendue = etendue_du_nom(mention)
    trouve = _l_article_parmi(voulu, extraits)
    if trouve is not None:
        return _Resolution(
            trouve[0], trouve[1], etendue, None,
            None if designe is None else frozenset({trouve[0].law_id}),
        )
    # Sans nom (« L'article 35 précise… »), aucun texte n'est en cause : c'est
    # l'article qui manque aux extraits lus. « texte_non_consulte » disait le
    # contraire de la citation (revue du 2026-10-28).
    if designe is None:
        return _Resolution(None, False, etendue, "article_non_consulte", None)
    return _Resolution(None, False, etendue, "texte_non_consulte", frozenset())


def _l_article_parmi(
    voulu: str, extraits: Sequence[ChunkResult]
) -> Optional[Tuple[ChunkResult, bool]]:
    """
    (extrait, numero identique) de l'article `voulu` parmi `extraits` : son
    morceau de ce numero, sinon un morceau de cet article (« 55.1 »), sinon le
    morceau qui en porte l'en-tete (« SECTION_1 » d'une loi modificative). Le
    premier dans l'ordre du contexte, a chaque etape.

    Une seule suite de replis pour un texte nomme (regle 1) et pour une
    citation sans nom (regle 3) : « L'article 35 précise… », avec les
    morceaux 35.1 et 35.2 du Code du travail lus, etait signale introuvable
    alors que sa forme nommee se rattachait.
    """
    identique = next(
        (e for e in extraits if e.number and _normalize_article_number(e.number) == voulu), None
    )
    if identique is not None:
        return identique, True
    partie = next((e for e in extraits if _numero_de_base(e.number) == voulu), None)
    if partie is not None:
        return partie, False
    entete = _entete_d_article(voulu)
    porteur = next((e for e in extraits if e.number and entete.search(e.content or "")), None)
    if porteur is not None:
        return porteur, False
    return None


def _textes_nommes_par_les_statuts(extraits: Iterable[ChunkResult]) -> List[TexteNomme]:
    """
    Les textes que les lignes de statut des extraits nomment : celui qui touche
    un extrait (par_titre), et celui qu'un extrait touche (texte_nom).

    Large a dessein : il ne sert qu'a NE PAS signaler une citation, jamais a
    rattacher une citation a un autre texte que celui de son numero.
    """
    references = {e.reference for e in extraits if e.reference}
    nommes: List[TexteNomme] = []
    for lien in LIENS:
        if lien.texte in references:
            nommes.append(TexteNomme(lien.par_titre, lien.par_reference, lien.par_date))
        if lien.par_reference and lien.par_reference in references:
            nommes.append(TexteNomme(lien.texte_nom, lien.texte))
    return nommes


def lire_les_citations(texte: str, extraits: Sequence[ChunkResult]) -> List[Occurrence]:
    """
    Toutes les citations de la reponse, dans l'ordre du texte.

    `extraits` : ce que le modele a lu. Une citation qu'aucun ne porte est
    rendue avec sa raison ; elle n'est plus jetee.
    """
    texte = texte or ""
    noms = {}
    for extrait in extraits:
        noms.setdefault(
            extrait.law_id,
            (normaliser_le_nom(extrait.law_title), normaliser_le_nom(extrait.reference)),
        )

    # Les textes designes par leur numero, rattaches d'avance : ce sont des
    # citations, et aussi l'antecedent possible d'un « article 5 de la même loi ».
    # Au texte dont c'est le numero propre, et a lui seul (designe_ce_texte).
    numerotes = [
        (match, next(
            (e for e in extraits if designe_ce_texte(match, e.law_title, e.publication_date)),
            None,
        ))
        for match in TEXTE_NUMEROTE_REGEX.finditer(texte)
    ]

    occurrences: List[Occurrence] = []
    # Ce que chaque citation d'article a lu, nom compris : son nom n'est pas
    # une autre citation, meme quand elle s'arrete avant lui (nom en gras).
    lues: List[Tuple[int, int]] = []
    antecedent: Optional[FrozenSet[int]] = None
    lus = 0
    for match in re.finditer(CITATION_REGEX, texte, re.IGNORECASE):
        while lus < len(numerotes) and numerotes[lus][0].end() <= match.start():
            porteur = numerotes[lus][1]
            antecedent = frozenset() if porteur is None else frozenset({porteur.law_id})
            lus += 1
        # Groupes : 1 = prefixe de code (L/R/D, optionnel), 2 = numero,
        # 3 = texte cite.
        prefixe = (match.group(1) or "").strip()
        numero_cite = f"{prefixe} {match.group(2)}".strip()
        voulu = _normalize_article_number(match.group(2))
        # Un italique ouvert dans le nom (« la loi _n° 92/007_ ») ne le coupe
        # pas : lu comme un blanc, de meme longueur, pour que les positions
        # tiennent. Le « _ » fermant, lui, arrete le nom (_mention_utile).
        mention = _SOULIGNE_OUVRANT.sub(" ", _mention_utile(match.group(3)))
        resolution = _resoudre_article(voulu, mention, extraits, noms, antecedent)
        # Le nom tel que la resolution l'a lu, et non toute la mention, qui
        # court jusqu'a la ponctuation : « la Constitution a été modifiée par
        # la Loi N°2008/001 » nomme aussi la loi 2008/001, citation a part.
        lues.append((match.start(), match.start(3) + max(resolution.etendue, 0)))
        if resolution.designe is not None:
            antecedent = resolution.designe
        extrait = resolution.extrait
        if resolution.etendue <= 0 or texte[match.start(3) - 1] in "*_":
            fin = _fin_de_la_designation(texte, match)
        else:
            fin = _avant_l_emphase(texte, match, match.start(3) + resolution.etendue)
        if extrait is None:
            occurrences.append(Occurrence(match.start(), fin, "article", raison=resolution.raison))
            continue
        if not extrait.number:
            article = numero_cite
        elif resolution.identique:
            article = _citable_article_number(extrait.number) or numero_cite
        else:
            # Un morceau (« 55.1 », « SECTION_1 ») : la page du texte connait
            # l'article 55, pas le decoupage en base.
            article = numero_cite
        occurrences.append(Occurrence(
            match.start(), fin, "article", extrait=extrait, article=article,
            cle=(extrait.law_id, voulu),
        ))

    nommes: Optional[List[TexteNomme]] = None
    for match, extrait in numerotes:
        if any(debut < match.end() and match.start() < fin for debut, fin in lues):
            continue  # le nom du texte d'une citation d'article
        if extrait is not None:
            occurrences.append(Occurrence(
                match.start(), match.end(), "texte", extrait=extrait,
                cle=(extrait.law_id, ""),
            ))
            continue
        if nommes is None:
            nommes = _textes_nommes_par_les_statuts(extraits)
        connu = next((n for n in nommes if designe_ce_texte(match, n.titre, n.date)), None)
        if connu is not None:
            occurrences.append(Occurrence(
                match.start(), match.end(), "texte", texte_nomme=connu,
            ))
            continue
        occurrences.append(Occurrence(
            match.start(), match.end(), "texte", raison="texte_non_consulte",
        ))

    return sorted(occurrences, key=lambda o: o.debut)


# ==================== INDICES DU FRONT ====================


def indices_utf16(texte: str) -> Callable[[int], int]:
    """
    Convertit un indice Python (point de code) en indice JavaScript (unite de
    code UTF-16) dans `texte`.

    Le front decoupe la reponse avec `String.prototype.slice`, qui compte en
    UTF-16 : un emoji en tete de reponse decalait d'un cran chaque citation qui
    le suit, et le lien tombait a cote de la formule.
    """
    if all(ord(c) <= 0xFFFF for c in texte):
        return lambda indice: indice
    cumul = [0]
    for caractere in texte:
        cumul.append(cumul[-1] + (2 if ord(caractere) > 0xFFFF else 1))
    return lambda indice: cumul[indice]
