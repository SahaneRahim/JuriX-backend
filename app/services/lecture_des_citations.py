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

UNE CITATION QUI NOMME UN TEXTE N'EST RATTACHEE QU'A CE TEXTE. Incident du
2026-10-29 en production : a « Quelles indemnités en cas de licenciement
abusif ? », la recherche n'avait rendu que des tableaux de lois de finances ;
« l'article 76 du Code du travail », que le modele citait de memoire, etait
devenu un lien vers le morceau « 76.37 » de la loi de finances 2023, et cette
loi s'affichait sous « Articles cités ». Un nom de texte qu'aucun texte
consulte ne porte laissait la citation au premier article de meme numero,
de n'importe quel texte. Elle est desormais signalee (texte_non_consulte),
que le nom porte une nature (« Code du travail », « Labour Code »), un
numero (« Law No. 92/007 »), un sigle (« du CT ») ou seulement des mots
qu'aucun intitule lu ne porte (« de la législation du travail »). Un nom
coupe par une virgule ou une preposition (« la loi relative à la lutte |
contre le blanchiment », « le décret portant création | , organisation et
fonctionnement des Centres Hospitaliers Régionaux ») exige aussi les mots de
sa suite de l'intitule qui continue le nom : il ne designe jamais un autre
texte lu.

DANS LE DOUTE, SIGNALER PLUTOT QUE LIER. Une citation signalee est
acceptable ; un lien vers un autre texte ne l'est pas.
- Une citation sans nom (« L'article 76 précise… »), ou qui ne dit que la
  nature (« de la loi ») : dans le texte que la derniere citation designe, et
  lui seul (« Voir l'article 76 du Code du travail. L'article 76 prévoit
  aussi… » : signalee). Jamais dans un tableau de credits d'une loi de
  finances, meme ce texte-la.
- RIEN NE DESIGNE SON TEXTE (ni nom, ni citation avant elle : « L'article 76
  prévoit… » en tete de reponse, « du code », « de cette loi », « of this
  Law ») : liee seulement si un seul texte lu (de cette nature) porte
  l'article, que l'extrait en porte l'en-tete (« Article 76.- ») et qu'il
  n'est pas un tableau de credits (_sans_antecedent). Sinon signalee. Le
  numero seul ne dit pas quel texte le modele cite de memoire : rejoue sur
  les lignes reelles de l'incident, il donnait le morceau 76.37, et
  l'article 76 du Code pénal quand le Code du travail etait lu sans le sien
  (revue du 2026-10-29, tour final). « de la loi », « du décret », « of the
  Law » sans aucune citation avant : signalee.
- Un nom ecrit en passant (« Le Code du travail prévoit… L'article 76… »,
  « … (Code du travail) » dans sa proposition, jusqu'a la virgule), ou un
  texte designe par son seul numero dans une phrase precedente (« La loi n°
  2022/020 ne traite pas du licenciement. L'article 76… »), ne choisit
  jamais l'article : il ne peut que signaler. Le meme article cite ailleurs
  AVEC le nom d'un autre texte le signale aussi ; il ne le confirme jamais.
- Plusieurs textes lus du meme nom portent l'article, et rien ne dit lequel :
  signalee.

CE QUI RESTE. Les vraies reponses n'ont presque pas de citation sans
antecedent : une seule des 754 citations des 323 reponses d'evaluation
(« l'article 57 de ce décret », juste), desormais signalee. Le contenu des
morceaux ne porte pas leur en-tete (le decoupage la range dans le numero),
si bien qu'une telle citation n'est presque jamais liee : une fausse alerte,
voulue. Un texte que le modele nomme autrement que son intitule (« Code des
Finances Publiques » pour le régime financier de l'État) est signale, meme lu.

UN COUT BORNE. La lecture tourne dans la boucle d'evenements de l'API, une
fois par reponse. Une reponse degeneree (32 000 caracteres sans ponctuation,
des centaines de noms) coutait jusqu'a 11 s. Les noms en prose relus par
citation (_NOMS_RELUS_AU_PLUS), la fenetre d'un nom (_FENETRE_D_UN_NOM), les
noms lus (_NOMS_LUS_AU_PLUS) et les citations lues (_CITATIONS_LUES_AU_PLUS)
sont bornes, bien au-dela des vraies reponses ; au-dela, une citation est
signalee, jamais liee.

Les primitives de lecture d'un numero d'article et CITATION_REGEX vivaient
dans rag_service.py ; elles sont ici pour que la lecture se teste seule, et
rag_service les reexporte sous leurs noms d'origine.
"""

import re
import unicodedata
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from itertools import islice
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
from app.services.search_service import est_un_tableau_de_programmes
from app.utils.chunk_refiner import est_pseudo_numero
from app.utils.titre_officiel import numero_du_titre

# est_un_tableau_de_programmes : le detecteur des tableaux de credits de la
# recherche, lu tel quel. La lecture ne lie jamais une citation sans nom a ce que
# la recherche ecarte deja comme n'etant pas du droit (incident du 2026-10-29).

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


@lru_cache(maxsize=4096)
def _normalize_article_number(number: str) -> str:
    """
    Normalise un numero d'article pour comparaison.

    En cache : chaque citation compare son numero a celui de chaque extrait lu,
    et 1 500 citations de numeros differents normalisaient 33 000 fois les 22
    memes numeros (revue du 2026-10-29, tour final).

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


@lru_cache(maxsize=4096)
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

# Le groupe 3 de CITATION_REGEX, lu a une autre place : apres une enumeration
# (« point 8, du Décret … ») ou en tete d'un nom ecrit dans la phrase.
_MENTION = re.compile(
    r"(?:\*{1,3}|_{1,3})?"
    r"([A-ZÀ-Ý](?:(?!\s+(?:et|ou|and|or)\s+(?:l['’]\s*)?art(?:icle|\.))[^,\.;()])+)",
    re.IGNORECASE,
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


# Les caracteres deja classes, et ceux d'entre eux qui sont des signes combinants
# (unicodedata.combining), a retirer : la table de str.translate.
_CARACTERES_CLASSES: Set[str] = set()
_COMBINANTS: Dict[int, None] = {}


def normaliser_le_nom(texte: Optional[str]) -> str:
    """
    Forme de comparaison d'un nom de texte : minuscules, sans accents,
    apostrophes et blancs unifies, « N° 92/007 » ecrit « n°92/007 ».

    Le modele ecrit « Code penal », la base « CODE PÉNAL » ; il ecrit « Loi n°
    2016/007 », la base « Loi N°2016/007 ». Sans forme commune, le meme texte
    passait pour deux, et sa citation restait sans source.
    """
    texte = texte or ""
    if texte.isascii():
        # Ni accent ni signe combinant : NFD ne change rien. Le chemin le plus
        # frequent de _prefixes, appele pour chaque debut de chaque nom.
        plat = texte.casefold()
    else:
        decompose = unicodedata.normalize("NFD", texte)
        # Les signes combinants retires par str.translate, et non caractere par
        # caractere en Python : chaque debut d'un nom accentue en passait les
        # caracteres un a un, le plus cher de la lecture de 500 noms inconnus
        # (revue du 2026-10-29, tour final). Chaque caractere nouveau est classe
        # une fois (unicodedata.combining), comme avant.
        for caractere in set(decompose) - _CARACTERES_CLASSES:
            if unicodedata.combining(caractere):
                _COMBINANTS[ord(caractere)] = None
            _CARACTERES_CLASSES.add(caractere)
        plat = decompose.translate(_COMBINANTS).casefold()
    plat = plat.replace("’", "'").replace("‘", "'")
    plat = _NUMERO_DE_TEXTE.sub("n°", plat)
    return " ".join(plat.split())


# Mots qui relient les mots d'un nom sans le designer (formes normalisees).
# « no » : le nom anglais coupe au point de « Law No. 92/007 ».
_LIAISONS = frozenset({
    "du", "de", "des", "la", "le", "les", "l", "d", "et", "a", "au", "aux", "en", "sur",
    "par", "portant", "relatif", "relative", "relatifs", "relatives", "n°", "no",
    "of", "the", "and", "on", "to", "relating",
})
# Mots qui ne designent aucun texte a eux seuls : « Code » nomme tous les codes,
# « Acte uniforme » les dix actes OHADA, « article » une partie de n'importe lequel.
_GENERIQUES = frozenset({
    "code", "codes", "loi", "lois", "decret", "decrets", "arrete", "arretes", "ordonnance",
    "ordonnances", "acte", "actes", "uniforme", "uniformes", "traite", "texte", "textes",
    "titre", "chapitre", "section", "decision", "circulaire", "ohada", "general", "generale",
    "article", "articles", "art",
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


# Les mots d'une mention que _prefixes essaie au plus. Le nom le plus long
# des 754 citations reelles en compte 18 (« Traité portant révision du Traité
# relatif à l'harmonisation du droit des affaires en Afrique de 2008 »). Sans
# borne, chaque debut d'une mention sans ponctuation etait normalise et
# cherche dans chaque intitule : O(W²), 11 s pour un nom de 8 000 caracteres,
# dans la boucle d'evenements de l'API (revue du 2026-10-29, tour 3).
_MOTS_D_UN_DEBUT_AU_PLUS = 25


def _prefixes(mention: str) -> List[Tuple[int, str, str]]:
    """
    Les debuts de la mention qui peuvent nommer un texte, du plus long au plus
    court : (longueur dans la mention, forme normalisee, forme sans liaison
    (_sans_de_des_du)).

    POURQUOI DES DEBUTS. Le texte cite par CITATION_REGEX court jusqu'a la
    ponctuation : « l'article 37 du Code du travail prévoit que le salarié… »
    lui donne « Code du travail prévoit que le salarié ». Exige en entier dans
    le titre, il ne nommait aucune loi, et la citation se rattachait au premier
    article 37 venu, de n'importe quel texte. La mention entiere (ses
    _MOTS_D_UN_DEBUT_AU_PLUS premiers mots) reste candidate, comme avant ; ses
    debuts doivent porter un mot qui designe (« travail »), pas seulement
    « Code ».
    """
    mots = list(islice(_MOT.finditer(mention), _MOTS_D_UN_DEBUT_AU_PLUS))
    if not mots:
        return []
    # La mention normalisee une fois, ses debuts lus mot a mot : normaliser
    # chaque debut coutait le plus cher de la lecture d'un nom inconnu (500
    # citations de noms uniques, revue du 2026-10-29, tour final). Le resultat
    # est le meme tant que chaque mot reste un mot : sinon (« N° 92/007 » ecrit
    # « n°92/007 », un mot fait de signes combinants), chaque debut est normalise.
    jetons = normaliser_le_nom(mention[:mots[-1].end()]).split(" ")
    if len(jetons) != len(mots):
        return [
            (mots[nombre - 1].end(), prefixe, _sans_de_des_du(prefixe))
            for nombre in range(len(mots), 0, -1)
            for prefixe in (normaliser_le_nom(mention[:mots[nombre - 1].end()]),)
            if nombre == len(mots) or _prefixe_significatif(prefixe)
        ]
    # La forme sans liaison aussi, mot a mot : « des », « du » ne sont des mots
    # entiers qu'entre deux blancs, ou dans un mot (« statut-du »), pareil seuls.
    canoniques = [_sans_de_des_du(jeton) for jeton in jetons]
    # _prefixe_significatif, debut par debut : le dernier mot du debut, et s'il
    # en contient un qui designe.
    significatifs = []
    dernier, designe = None, False
    for jeton in jetons:
        for mot in _mots_du_nom(jeton):
            dernier = mot
            designe = designe or (mot not in _LIAISONS and mot not in _GENERIQUES)
        significatifs.append(dernier is not None and dernier not in _LIAISONS and designe)
    return [
        (mots[nombre - 1].end(), " ".join(jetons[:nombre]), " ".join(canoniques[:nombre]))
        for nombre in range(len(mots), 0, -1)
        if nombre == len(mots) or significatifs[nombre - 1]
    ]


_DE_DES_DU = frozenset({"de", "des", "du"})


@lru_cache(maxsize=4096)
def _motif_du_nom(prefixe: str) -> "re.Pattern":
    """
    Le debut de mention, en mots entiers, « de », « des » et « du »
    interchangeables : la loi de finances 2026 s'intitule « … portant loi DES
    Finances … ». « la loi de finances » ne la nommait pas ; lue avec celle de
    2023, la citation allait a la loi de 2023 (sondes du 2026-10-29, tour 3).
    """
    mots = (r"(?:de|des|du)" if mot in _DE_DES_DU else re.escape(mot) for mot in prefixe.split(" "))
    return re.compile(r"(?<!\w)" + " ".join(mots) + r"(?!\w)")


# « des » et « du », mots entiers : « de » dans la forme sans liaison (_sans_de_des_du).
_DES_DU = re.compile(r"(?<!\w)(?:des|du)(?!\w)")


@lru_cache(maxsize=8192)
def _sans_de_des_du(texte: str) -> str:
    """
    Le texte, « des » et « du » ecrits « de » : la forme ou un debut de mention
    que _motif_du_nom trouve dans un intitule est forcement une sous-chaine.
    """
    return _DES_DU.sub("de", texte)


def _nomme(nom_normalise: str, prefixe: str) -> bool:
    """
    Le nom contient-il ce debut de mention, en mots entiers ?

    Une sous-chaine d'abord, sans motif : chaque debut d'un nom inconnu
    compilait son motif pour chaque intitule lu, et 500 citations de noms
    uniques prenaient 1,7 s (revue du 2026-10-29, tour final). Le motif ne
    tranche que les debuts que l'intitule contient deja.
    """
    if not prefixe:
        return False
    if _sans_de_des_du(prefixe) not in _sans_de_des_du(nom_normalise):
        return False
    return _motif_du_nom(prefixe).search(nom_normalise) is not None


# Les codes du corpus nommes en anglais, et leur nom francais (formes de
# normaliser_le_nom). Le prompt anglais demande « Article 35 of the Labour
# Code » ; les intitules sont francais (« … portant Code du travail »). Sans
# eux, une reponse anglaise ne reconnaitrait aucun code consulte, et chacune
# de ses citations serait signalee. Les plus longs d'abord.
_NOMS_ANGLAIS = (
    ("code of criminal procedure", "code de procedure penale"),
    ("criminal procedure code", "code de procedure penale"),
    ("code of military justice", "code de justice militaire"),
    ("military justice code", "code de justice militaire"),
    ("public procurement code", "code des marches publics"),
    ("public contracts code", "code des marches publics"),
    ("family benefits code", "code des prestations familiales"),
    ("nationality code", "code de la nationalite"),
    ("labour code", "code du travail"),
    ("labor code", "code du travail"),
    ("penal code", "code penal"),
    ("criminal code", "code penal"),
    ("civil code", "code civil"),
    ("electoral code", "code electoral"),
    ("mining code", "code minier"),
    ("petroleum code", "code petrolier"),
    ("insurance code", "code des assurances"),
    ("uniform act", "acte uniforme"),
)


_NOMS_ANGLAIS_MOTIFS = tuple(
    (re.compile(r"(?<!\w)" + re.escape(anglais) + r"(?!\w)"), francais)
    for anglais, francais in _NOMS_ANGLAIS
)
assert all("code" in anglais or "uniform act" in anglais for anglais, _ in _NOMS_ANGLAIS)


# « OHADA Code », « Penal Code » : en anglais, le mot qui distingue le code le
# precede ; en francais, il le suit (« Code OHADA », « Code pénal »).
_CODE_A_L_ANGLAISE = re.compile(r"((?:[^\W_]+ ){1,2})code")


@lru_cache(maxsize=1024)
def _en_francais(prefixe: str) -> str:
    """
    La forme francaise d'un nom de code ecrit en anglais : « labour code » ->
    « code du travail », « ohada code » -> « code ohada ». Tout autre nom,
    inchange. Une forme de plus a chercher dans les intitules, jamais une
    forme de moins.
    """
    # Chaque nom anglais de la table contient « code » ou « uniform act » :
    # sans eux, rien ne change, et 18 substitutions par debut de nom coutaient
    # le plus cher de la lecture d'un nom inconnu (revue du tour final).
    if "code" not in prefixe and "uniform act" not in prefixe:
        return prefixe
    for motif, francais in _NOMS_ANGLAIS_MOTIFS:
        prefixe = motif.sub(francais, prefixe)
    inverse = _CODE_A_L_ANGLAISE.fullmatch(prefixe)
    return "code " + inverse.group(1).strip() if inverse else prefixe


def _reconnaitre(mention: str, noms: Dict[int, Tuple[str, str]]) -> Tuple[int, Set[int]]:
    """(longueur de la mention qui nomme un texte consulte, textes ainsi nommes)."""
    fin, lois, _ = _reconnaitre_en_detail(mention, noms)
    return fin, lois


def _reconnaitre_en_detail(
    mention: str, noms: Dict[int, Tuple[str, str]], complement: Optional["_Complement"] = None
) -> Tuple[int, Set[int], Set[int]]:
    """
    (longueur de la mention qui nomme un texte consulte, textes ainsi nommes,
    ceux d'entre eux dont l'intitule ne fait que CITER ce nom).

    Le plus long debut reconnu l'emporte : « Code général des impôts » ne doit
    pas s'arreter a un « Code général » qu'un autre titre contiendrait. Un
    debut que le titre continue autrement que le nom ne compte pas (_diverge) ;
    le nom ecrit en anglais est aussi cherche sous sa forme francaise.

    LE PLUS LONG DEBUT ECARTE N'EST PAS REMPLACE PAR UN PLUS COURT. « Acte
    uniforme relatif au droit des sociétés | commerciales », l'Acte uniforme
    des sociétés COOPÉRATIVES et celui de l'arbitrage lus : le plus long debut
    present, celui des coopératives, est ecarte ; le debut plus court
    « Acte uniforme relatif au droit », que l'arbitrage porte aussi, rattachait
    la citation a l'arbitrage (revue du 2026-10-29). Le nom est alors lu
    mot a mot (_par_ses_mots), ou signale.

    `complement` : ce qui continue le nom au-dela d'une virgule ou d'une
    preposition (_complement_du_nom). Un intitule qui continue lui aussi le
    nom doit en porter les mots (_porte_le_complement).
    """
    fin_du_nom = etendue_du_nom(mention)
    # Un debut ne peut etre dans un intitule (ou une reference) que s'il est une
    # sous-chaine de sa forme sans liaison (_nomme) : inutile de chercher les
    # autres (une phrase de 700 « le Code … » sans ponctuation coutait 0,8 s,
    # 2026-10-29 ; 500 noms uniques, 1,7 s au tour final).
    connus = _connus(tuple(nom for paire in noms.values() for nom in paire))
    for fin, prefixe, canonique in _prefixes(mention):
        francais = _en_francais(prefixe)
        formes = {prefixe} if canonique in connus else set()
        if francais != prefixe and _sans_de_des_du(francais) in connus:
            formes.add(francais)
        if not formes:
            continue
        suite = normaliser_le_nom(mention[fin:fin_du_nom])
        lois, cites, ecarte = set(), set(), False
        for law_id, (titre, reference) in noms.items():
            if any(_nomme(reference, forme) for forme in formes):
                lois.add(law_id)
                continue
            portees = [forme for forme in formes if _nomme(titre, forme)]
            if not portees:
                continue
            if all(_diverge(titre, forme, suite) for forme in portees):
                ecarte = True
                continue
            if not _porte_le_complement(titre, prefixe, portees, complement):
                ecarte = True
                continue
            lois.add(law_id)
            if not any(
                _son_propre_nom(titre, forme) and _de_sa_nature(forme, titre) for forme in portees
            ):
                cites.add(law_id)
        if lois:
            return _jusqu_au_sigle(mention, fin, fin_du_nom, [noms[i][0] for i in lois]), lois, cites
        if ecarte:
            return 0, set(), set()
    return 0, set(), set()


@lru_cache(maxsize=256)
def _connus(noms: Tuple[str, ...]) -> str:
    """Les intitules et references lus, sans liaison (_sans_de_des_du), un par ligne."""
    return "\n".join(_sans_de_des_du(nom) for nom in noms)


def _de_sa_nature(prefixe: str, titre: str) -> bool:
    """
    Un nom qui ouvre sur une nature (« Convention des Nations Unies sur… »,
    « Code du travail ») est-il de la nature du texte lui-meme (_natures_propres) ?
    La loi « autorisant le Président de la République à ratifier la Convention
    des Nations Unies sur … » porte le nom de la convention sans renvoi que
    _RENVOI_DANS_L_INTITULE reconnaisse : son propre article 5 passait pour
    celui de la convention (sondes du 2026-10-29, tour 3).
    """
    mots = [mot for mot in _mots_du_nom(prefixe) if mot not in _DETERMINANTS]
    if not mots or mots[0] not in _NATURES_DE_TEXTE:
        return True
    return bool(frozenset(_natures(mots[0])) & _natures_propres(titre))


# Dans un intitule, le renvoi a un AUTRE texte : « modifiant … DE LA LOI
# n°2012/001 … portant code électoral », « relevant DU CODE du Travail »,
# « portant ratification DU TRAITÉ … ». Ce qui suit nomme ce texte-la. Une
# revision le reecrit en entier : « Loi N°96/06 … portant révision de la
# Constitution » est la Constitution.
_RENVOI_DANS_L_INTITULE = re.compile(
    r"(?<!revision )(?<!refonte )(?<!\w)(?:"
    r"(?:de la|de l'|du|des|a la|a l'|au|aux|par la|par le|par l')\s?"
    r"(?:lois?|decrets?|ordonnances?|arretes?|codes?|actes? uniformes?|traites?|constitution"
    r"|conventions?|chartes?|protocoles?|accords?)(?!\w)"
    r"|(?:la|le|l')\s?(?:loi|decret|ordonnance|arrete)\s?n°"
    r")"
)


@lru_cache(maxsize=4096)
def _debut_du_renvoi(titre: str) -> Optional[int]:
    """
    Ou commence, dans l'intitule normalise, le renvoi a un autre texte
    (_RENVOI_DANS_L_INTITULE) ; None : aucun. Un motif lourd, cherche pour
    chaque nom lu dans chaque intitule lu : une fois par intitule suffit.
    """
    renvoi = _RENVOI_DANS_L_INTITULE.search(titre)
    return renvoi.start() if renvoi is not None else None


def _son_propre_nom(titre: str, prefixe: str) -> bool:
    """
    L'intitule porte-t-il ce nom comme le sien, et non dans le renvoi a un autre
    texte ? « Loi N°2012/001 … portant Code électoral » : oui. « Loi N°2026/003
    … modifiant … de la loi N°2012/001 … portant code électoral » : non.
    """
    porte = _motif_du_nom(prefixe).search(titre)
    renvoi = _debut_du_renvoi(titre)
    return porte is not None and (renvoi is None or porte.start() < renvoi)


# Le morceau d'un texte modificatif qui reecrit l'article : « (nouveau).- (1)
# L'empêchement définitif… » (loi 2026/003, morceau « 145 », l'article 145 du
# Code électoral modifie).
_ARTICLE_REECRIT = re.compile(r"[\s«\"“'*_]*\((?:nouveau|nouvelle|new)\)", re.IGNORECASE)


def _reecrit_l_article(extrait: ChunkResult, voulu: str) -> bool:
    """
    Le morceau d'un texte qui ne fait que CITER le texte nomme porte-t-il
    l'article `voulu` de ce texte ? Seulement s'il le reecrit : le morceau
    « 145 » de la loi 2026/003, « (nouveau).- … », est l'article 145 du Code
    électoral modifie ; son morceau « SECTION_1 » porte « ARTICLE 142
    (nouveau).- ». Son propre article 2 (« La présente loi sera
    enregistrée… ») n'est pas l'article 2 du Code électoral, ni l'article 2
    du « Décret … relevant du Code du Travail » celui du Code du travail.
    """
    numero = _normalize_article_number(extrait.number or "")
    if (numero == voulu or _numero_de_base(extrait.number) == voulu) and _ARTICLE_REECRIT.match(
        extrait.content or ""
    ):
        return True
    # L'en-tete seul peut etre celui d'un article du texte modificatif lui-meme
    # (« Article 2.- La présente loi sera enregistrée… ») : il faut « (nouveau) ».
    return any(
        re.search(r"nouve|new", entete.group(), re.IGNORECASE)
        for entete in _entete_d_article(voulu).finditer(extrait.content or "")
    )


def _jusqu_au_sigle(mention: str, fin: int, fin_du_nom: int, titres: Sequence[str]) -> int:
    """
    `fin`, prolongee sur le sigle qui suit le nom reconnu quand il abrege la
    suite de l'intitule : « la réorganisation | de l'APZE » est souligne en
    entier, comme avant.
    """
    for mot in re.finditer(r"[^\W_]+", mention[fin:fin_du_nom]):
        forme = normaliser_le_nom(mot.group())
        if forme in _GENITIFS:
            continue
        if len(forme) >= 2 and mot.group().isupper() and any(
            _initiales(re.findall(r"[^\W_]+", titre[porte.end():])).startswith(forme)
            for titre in titres
            for porte in _motif_du_nom(normaliser_le_nom(mention[:fin])).finditer(titre)
        ):
            return fin + mot.end()
        break
    return fin


# Ce qui relie un nom a son complement : « droit | des sociétés », « fonction
# publique | de l'État ». _diverge les saute pour comparer les noms eux-memes.
_GENITIFS = frozenset({"de", "du", "des", "d", "la", "le", "les", "l", "of", "the"})
# Mots en -ant / -ing qui ne sont pas des participes.
_NON_PARTICIPES = frozenset({
    "enfant", "enfants", "commercant", "commercants", "representant", "representants",
    "habitant", "habitants", "montant", "montants", "etudiant", "etudiants", "ressortissant",
    "ressortissants", "exploitant", "exploitants", "occupant", "occupants", "intervenant",
    "intervenants", "independant", "mining", "accounting", "banking", "trafficking", "housing",
    "building", "fishing", "hunting", "shipping", "planning", "leasing", "marketing", "financing",
})


def _participe(mot: str) -> bool:
    """
    « concernant », « fixant », « governing » : la phrase continue, pas le nom.
    Lu par _diverge seulement : « Décret … fixant les modalités » ouvre aussi
    des intitules, et couper le nom la le reduirait a « décret ».
    """
    return len(mot) >= 6 and mot.endswith(("ant", "ing")) and mot not in _NON_PARTICIPES


def _racine(mot: str) -> str:
    """« commerciales » et « commercial », « civile » et « civil » : un seul mot."""
    if len(mot) > 3 and mot.endswith("s"):
        mot = mot[:-1]
    if len(mot) > 3 and mot.endswith("e"):
        mot = mot[:-1]
    return mot


def _nom_qui_suit(mots: Sequence[str]) -> Optional[str]:
    """
    Le mot qui continue un nom, complement compris (« des | sociétés », « à |
    l'École », « au | Secrétariat ») : None si ce qui suit n'en est pas un —
    une date, une virgule, « modifié », « camerounais », un verbe, une
    preposition (« sur l'indemnité »).

    « à », « au », « aux » relient aussi un nom a son complement : « décret
    portant nomination de responsables | à l'École Internationale… » contre
    l'intitule lu « … de responsables | au Secrétariat Général… » ne
    divergeait pas (« à » et « au » ne distinguent rien), et la citation
    allait a cet autre decret (balayage de la revue du 2026-10-29, tour final).
    """
    for mot in mots:
        if mot in _GENITIFS or mot in _A_AU_AUX:
            continue
        return mot if _distinctif(mot) and not _participe(mot) else None
    return None


# « to » : le meme relais en anglais (« Uniform Act relating to | Contracts… »).
# Sauter un relais ne peut qu'ajouter un ecart, donc retirer un lien, jamais
# en creer un.
_A_AU_AUX = frozenset({"a", "au", "aux", "to"})


def _initiales(mots: Sequence[str]) -> str:
    return "".join(mot[0] for mot in mots if mot not in _LIAISONS and mot[0].isalpha())


def _diverge(titre: str, prefixe: str, suite: str) -> bool:
    """
    Le titre continue-t-il le nom AUTREMENT que la mention, partout ou il le
    porte ? « Acte uniforme portant organisation | des sûretés » contre
    l'intitule « Acte uniforme portant organisation | des procédures
    collectives… » : meme debut, deux textes. Le plus long debut reconnu
    rattachait la citation des sûretés a l'Acte uniforme des procedures
    collectives, et « Code de procédure | civile » au Code de procédure
    pénale.

    Il faut DEUX noms differents, un de chaque cote, complement compris
    (« droit | commercial général » contre « droit | de l'arbitrage »). Ce
    que la phrase ajoute au nom (« Code civil | concernant la force
    obligatoire », « Code du travail | sur l'indemnité ») ne contredit rien,
    ni ce que le titre ajoute (« | du 15 décembre 2010 », « | , modifiée et
    complétée », « | applicable au Cameroun ») : la premiere version comptait
    tout ecart, et ecartait le bon texte (revue du 2026-10-29). Un sigle
    n'ecarte pas le nom qu'il abrege (« | de l'APZE » contre « | de l'Agence
    de Promotion des Zones Economiques »).
    """
    nom = _nom_qui_suit(re.findall(r"[^\W_]+", suite))
    if nom is None:
        return False
    for porte in _motif_du_nom(prefixe).finditer(titre):
        reste = re.findall(r"[^\W_]+", titre[porte.end():])
        autre = _nom_qui_suit(reste)
        if autre is None or _racine(autre) == _racine(nom):
            return False
        if _initiales(reste).startswith(nom):
            return False
    return True


def _sans_separateurs(texte: str) -> str:
    """Le texte plie, sans tirets, espaces ni « _ » : pour un premier tri rapide."""
    return re.sub(r"[\s_-]+", "", texte.casefold())


def _reconnaitre_la_reference(
    nom: str, noms: Dict[int, Tuple[str, str]]
) -> Tuple[int, Set[int]]:
    """
    (fin de la reference dans le nom, textes qui la portent) : « l'article
    1.15 de la Loi PRC-8842 ». Le modele recopie la reference que l'en-tete de
    chaque extrait lui donne ; « loi » seul ne nomme rien, et la citation,
    juste, etait rattachee par son seul numero (vraie reponse d'evaluation).
    """
    tete = nom[:etendue_du_nom(nom)]
    plie = _sans_separateurs(tete)
    fin, lois = 0, set()
    for law_id, (_, reference) in noms.items():
        # La reference (normalisee) doit au moins etre dans le nom : 22 motifs
        # par nom lu, sinon, pour des noms qui n'en recopient presque jamais.
        if not reference or _sans_separateurs(reference) not in plie:
            continue
        # Tirets, espaces et « _ » interchangeables : le modele recopie
        # « OHADA-AUSCGIE-2014 » en « Loi OHADA AUSCGIE 2014 », et la citation
        # juste etait signalee « non retrouvee » (chat du 08/11/2026).
        motif = r"[\s_-]+".join(re.escape(m) for m in re.split(r"[\s_-]+", reference) if m)
        trouve = re.search(r"(?<![\w-])" + motif + r"(?![\w-])", tete, re.IGNORECASE)
        if trouve:
            fin, lois = max(fin, trouve.end()), lois | {law_id}
    return fin, lois


# Ce qui ne fait jamais partie du nom d'un texte. Le texte cite court jusqu'a
# la ponctuation forte : il emportait un retour a la ligne et le titre markdown
# suivant, ou les « ** » d'une citation en gras, et ne nommait plus rien. Un
# « _ » colle a la fin d'un mot ferme un italique (« _Code pénal_ prévoit »).
_HORS_DU_NOM = re.compile(r"[\r\n*«»\"“”:!?\[\]|]|_{2}|(?<=\w)_(?!\w)|\s[–—-]\s")


def _mention_utile(mention: str) -> str:
    coupe = _HORS_DU_NOM.search(mention)
    return (mention[:coupe.start()] if coupe else mention).rstrip()


def _preparer(brute: str) -> str:
    """
    La mention telle que la lecture la compare. Un italique ouvert dans le nom
    (« la loi _n° 92/007_ ») ne le coupe pas : lu comme un blanc, de meme
    longueur, pour que les positions tiennent. Le « _ » fermant, lui, arrete
    le nom (_mention_utile).
    """
    utile = _mention_utile(brute)
    return _SOULIGNE_OUVRANT.sub(" ", utile) if "_" in utile else utile


# Mots qui enchainent la phrase apres le nom d'un texte : le verbe dont la
# citation est le sujet, ou la proposition qui s'y accroche. En minuscules
# AVEC accents : « a » (verbe) n'est pas « à » (« relatif à »), « arrête »
# (verbe) n'est pas « arrêté ».
#
# Pas de participe (« fixant », « autorisant », « concernant ») : ils ouvrent
# des intitules (« Décret … fixant les modalités… », « Loi … autorisant le
# Président de la République à ratifier… »), et le nom coupe la ne serait
# plus que « décret », que n'importe quel decret lu porte. _diverge les lit a
# part (_participe).
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
    # Pas « on » : en anglais, il est DANS le nom (« Law on Freedom of
    # Association », « Uniform Act on Arbitration »), que ce pronom coupait a
    # « Law », que toute loi lue porte (revue du 2026-10-29).
    "lors", "il", "elle", "ils", "elles",
    # Jamais dans le nom d'un texte : « l'article 312 du Code pénal ne
    # s'applique pas » soulignait « Code pénal ne ».
    "ne", "se", "y", "lui", "leur",
    # « du Statut général de la fonction publique tel que modifié », « de
    # l'Acte uniforme … détermine les conditions » : la suite etait lue comme
    # la fin du nom, et le bon texte ecarte (revue du 2026-10-29).
    "tel", "telle", "tels", "telles", "détermine", "déterminent", "énumère", "énumèrent",
    "détaille", "détaillent", "pose", "posent", "vise", "visent", "concerne", "concernent",
    "arrête", "arrêtent", "institue", "instituent", "instaure", "instaurent", "crée",
    "créent", "renvoie", "renvoient", "consacrent", "garantissent", "affirme", "affirment",
    "rappellent", "ajoutent", "complète", "complètent", "modifie", "modifient", "abroge",
    "abrogent", "remplace", "remplacent", "applique", "appliquent", "décrit", "décrivent",
    "prescrivent", "encadrent", "organisent", "régissent", "reconnaissent", "accordent",
    "confèrent", "exclut", "excluent", "admet", "admettent", "requiert", "requièrent",
    "soumet", "soumettent", "subordonne", "subordonnent", "protège", "protègent",
    "sanctionnent", "répriment", "qualifie", "qualifient", "comporte", "comportent",
    "contient", "contiennent", "comprend", "comprennent", "fait", "font", "rend", "rendent",
    "donne", "donnent", "ouvre", "ouvrent", "fournit", "fournissent", "habilite",
    "habilitent", "confie", "confient", "attribue", "attribuent", "distingue", "distinguent",
    "provides", "states", "says", "sets", "requires", "prohibits", "allows", "is", "are",
    "was", "were", "may", "shall", "must", "can", "which", "that", "where", "when", "if",
    # Pas « with » non plus : « Law on Persons with Disabilities ».
    "who", "under", "for", "as", "does", "do", "did", "has", "have", "will",
    "would", "should", "could", "not", "also",
    # « Article 75 of the same Code too » : « too » etait lu comme le mot qui
    # distingue le code, et la citation signalee (sondes tenues a l'ecart, tour
    # final). Aucun intitule du corpus, en francais, ne les porte.
    "too", "either",
    "lists", "lays", "governs", "establishes", "defines", "specifies", "stipulates",
    "grants", "guarantees", "gives", "contains", "covers", "deals", "regulates", "imposes",
    "punishes", "permits", "forbids", "obliges", "entitles", "authorizes", "authorises",
    "creates", "determines", "describes", "outlines", "applies", "remains", "makes",
    "treats", "refers", "mentions", "indicates", "confirms", "adds", "recognizes",
    "recognises", "prescribes", "enumerates", "puts",
})
_ELISIONS_DE_SUITE = frozenset({"qu", "s", "n"})  # « qu'il », « s'applique », « n'est »
_MOTS_DU_NOM_AU_PLUS = 12


@lru_cache(maxsize=8192)
def _enchaine_la_phrase(mot: str) -> bool:
    """Le mot continue la phrase au lieu de nommer un texte : verbe, conjonction…"""
    brut = mot.casefold()
    if brut in _SUITES:
        return True
    elision = _APOSTROPHE.split(brut, maxsplit=1)
    return len(elision) > 1 and elision[0] in _ELISIONS_DE_SUITE


_APOSTROPHE = re.compile(r"['’]")
_MOT = re.compile(r"\S+")


def ne_nomme_rien(mention: str) -> bool:
    """
    « L'article 37 précise que… » : apres le numero, la phrase continue. Le
    premier mot n'etait pas controle, et la citation devenait « article 37
    précise » ; pire, « est » (« L'article 5 est abrogé ») designait la loi
    dont le titre contient « de l'Est ».
    """
    premier = re.match(r"\S+", mention)
    return premier is None or _enchaine_la_phrase(premier.group())


# En cache, petit : la meme mention y passe cinq fois par citation, et une
# mention peut etre longue (jusqu'a la ponctuation).
@lru_cache(maxsize=1024)
def etendue_du_nom(mention: str) -> int:
    """
    Longueur du nom de texte en tete de la mention, quand aucun texte consulte
    ne le reconnait : jusqu'au premier mot qui enchaine la phrase, sans finir
    sur un mot de liaison. Une estimation, pour souligner la citation dans le
    texte ; jamais pour la rattacher a une source. 0 : la mention ne nomme rien.
    """
    gardes = []
    for mot in islice(_MOT.finditer(mention), _MOTS_DU_NOM_AU_PLUS):
        if _enchaine_la_phrase(mot.group()):
            break
        gardes.append(mot)
    while gardes and normaliser_le_nom(gardes[-1].group()) in _LIAISONS:
        gardes.pop()
    return gardes[-1].end() if gardes else 0


# « de la même loi », « dudit décret » : le texte de la citation precedente.
# Sans antecedent, le numero seul departageait, et l'article 132 « de la même
# loi » pouvait devenir celui d'une autre loi consultee.
#
# Le demonstratif (« de cette loi », « de ce même décret ») et l'adjectif qui
# renvoie (« la présente loi », « le décret précité ») disent la meme chose.
# Ils ne comptaient pas : « l'article 11 de cette loi », qui suit la Loi
# N°2014/028 nommee dans la phrase, n'etait rattache que par son numero
# (vraies reponses d'evaluation, 4 cas). Ils sont lus comme « même ».
_ANAPHORE = re.compile(
    r"(?:(?:ce|cet|cette|ces|this|these|the)\s+)?"
    r"(?:m[êe]mes?|ledit|ladite|lesdits|lesdites|dudit|desdits|desdites|said|same"
    r"|pr[ée]sente?s?)\b"
    # « ce qui », « ce que » : un pronom, pas un texte.
    r"|(?:ce|cet|cette|ces|this|these)\s+(?=[^\W\d_])(?!(?:qui|que|qu['’]|dont|quoi)\b)"
    # « le décret précité », « le décret cité », « la loi mentionnée ».
    r"|[^\W\d_]+\s+(?:pr[ée]cit[ée]e?s?|cit[ée]e?s?|susvis[ée]e?s?|vis[ée]e?s?"
    r"|susmentionn[ée]e?s?|mentionn[ée]e?s?|susdite?s?|aforementioned|above-mentioned"
    r"|mentioned|cited)\b",
    re.IGNORECASE,
)
# « dudit décret » : le mot de liaison est dans le renvoi.
_RENVOI_CONTRACTE = re.compile(r"d(?:udit|esdite?s)\b", re.IGNORECASE)
# « de notre Code du travail », « of our Labour Code » : le possessif precede le nom.
_POSSESSIF = re.compile(r"(?:notre|votre|nos|vos|our|your)\s+(?=[^\W\d_])", re.IGNORECASE)

# Le mot de liaison entre le numero et le nom (« l'article 37 DU Code… »,
# « Article 35 OF the Labour Code »). Sans lui, ce qui suit le numero est le
# verbe de la phrase : « L'article 37 traite du licenciement », « L'article 5
# arrête la liste », « Article 37 orders the reinstatement » — « traite »,
# « arrête », « orders » ne sont pas un Traité, un Arrêté, des Orders, et ces
# citations, que la production rattachait, etaient signalees (revue du
# 2026-10-29).
_LIE_AU_NOM = re.compile(r"(?<![\w’'])(?:du|des?|of)(?![\w’'])", re.IGNORECASE)

# Ce qui separe le numero d'un article du nom de son texte quand plusieurs
# articles, ou une subdivision, partagent ce nom : « l'article 43 et l'article
# 45 du Décret N°2019/143 », « l'article 6 et 9 de la Loi N°2026/002 »,
# « l'article 88, alinéa 1, point 8, du Décret … ». Le texte cite de
# CITATION_REGEX commence alors au « et », ou s'arrete a la virgule : le nom
# n'etait pas reconnu, et la citation ne tenait qu'a son numero (vraies
# reponses d'evaluation, 9 cas). Il est lu apres ce passage, dans le texte
# lui-meme. La revue du 2026-10-29 a montre les autres formes, ou le numero
# seul rattachait encore « l'article 76 … du Code du travail », non consulte,
# au morceau 76.37 d'une loi de finances : « ainsi que l'article 77 »,
# « et à l'article 77 », « ou de l'article 77 », « alinéas 1 et 2 »,
# « premier alinéa », « précité », « combiné avec l'article 77 », « as well
# as Article 77 », « subparagraph 2 ». Et les qualificatifs de l'article
# (« l'article 76 in fine du Code du travail », « l'article 76 modifié du
# … », « Article 76 as amended of the … ») : la citation passait pour une
# citation sans nom, et allait au morceau 76.37 (sondes du 2026-10-29, tour 3).
_NUMERO_ENUMERE = (
    r"[LRD]?\s*\d+(?:[-.]\d+)*(?:\s*(?:er|bis|ter|quater|septies))?(?:\(\d+\))?"
)
_AVANT_LE_NOM = re.compile(
    r"(?:(?:"
    r"(?:et|ou|ainsi\s+que|de\s+m[êe]me\s+que|combin[ée]e?s?\s+avec|lue?s?\s+avec|and|or"
    r"|as\s+well\s+as|together\s+with|read\s+with|in\s+conjunction\s+with|to)"
    r"\s+(?:(?:à|au|aux|de|du|des|with|to|of)\s+)?(?:l['’]\s*)?(?:art(?:icles?|\.)\s*)?"
    + _NUMERO_ENUMERE +
    r"|et\s+suivants?|et\s+ss?\.|and\s+following|et\s+seq\."
    r"|(?:premier|première|1er|1re|deuxième|second|seconde|troisième|dernier|dernière)"
    r"\s+(?:alin[ée]a|paragraphe|tiret|point)"
    r"|(?:alin[ée]as?|al\.|paragraphes?|points?|tirets?|sub-?paragraphs?|paragraphs?|para\."
    r"|items?|clauses?)\s*(?:\(?\d+\)?|\([a-z]\)|premier|1er)"
    r"(?:\s*(?:,|et|à|and|to)\s*(?:\(?\d+\)?|\([a-z]\)))*"
    r"|pr[ée]cit[ée]e?s?|susvis[ée]e?s?|susmentionn[ée]e?s?|nouveau|nouvelle|ancien(?:ne)?"
    r"|in\s+fine|modifi[ée]e?s?|r[ée]vis[ée]e?s?|compl[ée]t[ée]e?s?|abrog[ée]e?s?"
    r"|as\s+(?:amended|revised)|amended|revised|repealed"
    r")(?:\s*,)?\s+)+"
    r"(?:du|de\s+la|de\s+l['’]|des|de|of\s+the|of)(?:\s+|(?<=['’]))",
    re.IGNORECASE,
)

_ET_SUIVANTS = re.compile(r"(?:et\s+suivants?|and\s+following)(?![\w’'])|et\s+ss?\.", re.I)

# Mots qui disent la nature d'un texte, sous la forme de normaliser_le_nom.
# « titre », « chapitre », « section » n'en sont pas : ils designent une
# partie d'un texte, pas un texte. « règlement » non plus : aucun texte du
# corpus n'en est un, et c'est le mot qui distingue une loi de reglement.
# Nature, il n'etait pas exige : « l'article 76 de la loi portant loi de
# règlement … » allait a la loi de finances lue (revue du 2026-10-29, tour 3).
_NATURES_DE_TEXTE = frozenset({
    "code", "codes", "loi", "lois", "decret", "decrets", "arrete", "arretes",
    "ordonnance", "ordonnances", "acte", "actes", "traite", "traites", "constitution",
    "decision", "decisions", "circulaire", "circulaires", "instruction", "instructions",
    "convention", "conventions", "accord", "accords", "statut", "statuts",
    "charte", "protocole",
    "law", "laws", "act", "acts", "decree", "decrees", "order", "orders", "ordinance",
    "ordinances", "treaty", "treaties", "agreement", "agreements", "regulation",
    "regulations", "statute", "statutes", "charter", "circular", "circulars", "protocol",
})
_DETERMINANTS = frozenset({"la", "le", "les", "l", "the"})
# Prepositions anglaises absentes de _LIAISONS et de _SUITES : « Article 5 in
# order to » ne nomme aucun texte.
_PREPOSITIONS_ANGLAISES = frozenset({"in", "by", "at", "from", "into", "within"})


@lru_cache(maxsize=1024)
def nature_nommee(nom: str) -> Optional[str]:
    """
    La nature du texte que le nom designe, ou None s'il n'en designe aucun.

    « Code du travail » -> « code », « Labour Code » -> « code », « OHADA
    Uniform Act » -> « act », « Loi PRC-8842 » -> « loi ». Le mot de nature est
    en tete du nom, avant tout mot de liaison : « point 8 », « et suivants »,
    « Les associations obéissent à deux régimes » (« Article 5 Les
    associations… ») ne designent aucun texte, et gardent le rattachement au
    numero seul.
    """
    mots = _mots_du_nom(normaliser_le_nom(nom[:etendue_du_nom(nom)]))
    while mots and mots[0] in _DETERMINANTS:
        mots.pop(0)
    for mot in mots:
        if mot in _LIAISONS or mot in _PREPOSITIONS_ANGLAISES:
            return None
        if mot in _NATURES_DE_TEXTE:
            return mot
    return None


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
    return bool(voulu) and _porte_l_entete_de(voulu, contenu or "")


# ==================== TEXTES DESIGNES PAR LEUR NUMERO ====================

# « le décret N°2018/9387/CAB/PM », « l'arrêté n° 016/MTPS/SG/CJ du 26 mai
# 1993 », « Law No. 2016/007 of 12 July 2016 ». Le numero commence par un
# chiffre et finit sur une lettre ou un chiffre : « 92/007. » rend « 92/007 »,
# et « _le décret n° 94/199_ » (italique) rend « 94/199 ». Un « _ » y etait
# avale (« 94/199_ ») : le numero n'etait plus celui du decret lu, la
# citation etait signalee a tort, et le passage cite portait un « _ » que le
# front ne peut pas couper (revue du 2026-10-28). Le mois de la date, en
# lettres seulement, pour la meme raison.
_NATURE_NUMEROTEE = (
    r"\b(?P<nature>lois?|d[ée]crets?|arr[êe]t[ée]s?|ordonnances?|d[ée]cisions?|circulaires?"
    r"|laws?|decrees?|orders?|ordinances?|circulars?)"
    r"(?:\s+(?:organique|présidentiel(?:le)?|conjoint|ministériel(?:le)?|organic|presidential))?"
)
_DATE_ECRITE = r"(?P<date>\s+(?:du|of)\s+\d{1,2}(?:er|st|nd|rd|th)?\s+[^\W\d_]+\.?,?\s+\d{4})?"
TEXTE_NUMEROTE_REGEX = re.compile(
    _NATURE_NUMEROTEE
    + r"\s+(?:n\s*[°º]|no\.?|num[ée]ro|number)\s*"
    r"(?P<numero>\d(?:[^\W_]|[/\-])*[^\W_]|\d)"
    + _DATE_ECRITE,
    re.IGNORECASE,
)

# « l'article 36 de l'Ordonnance 2025/002 », « l'article 21 de la Loi
# 2017/015 », « l'article 5 du décret 2021/745 » : le numero sans « n° ». Le
# nom n'etait pas lu comme un numero, et la citation ne tenait qu'au numero
# d'article (vraies reponses d'evaluation, 3 cas) ; elle aurait sinon ete
# signalee. Lu seulement dans le nom d'une citation d'article : un texte cite
# seul reste reconnu a son « n° », comme avant. Une barre ou un tiret est
# exige : « la loi 2 » n'est pas un numero de texte.
_TEXTE_NUMEROTE_SANS_SIGNE = re.compile(
    _NATURE_NUMEROTEE + r"\s+(?P<numero>\d[^\W_]*(?:[/\-][^\W_]+)+)" + _DATE_ECRITE,
    re.IGNORECASE,
)


def _designation_numerotee(texte: str, debut: int = 0) -> Optional["re.Match"]:
    """
    Le texte que le nom qui commence a `debut` designe par son numero, avec ou
    sans « n° » ; None sinon.

    Lu dans la REPONSE, pas dans le nom coupe : CITATION_REGEX arrete le nom au
    point, et « Article 35 of Law No. 92/007 » ne nommait plus que « Law No ».
    Lu comme un texte de nature « law » sans numero, il etait signale, loi lue
    ou non (revue du 2026-10-29) : c'est pourtant la forme que le prompt
    anglais demande (« Article 15 of Decree No. 2019/143 »).
    """
    return TEXTE_NUMEROTE_REGEX.match(texte, debut) or _TEXTE_NUMEROTE_SANS_SIGNE.match(
        texte, debut
    )


# Une nature, et celles qui la traduisent : un titre en anglais (« LAW No.
# 2016/007 ») repond a « la loi n° 2016/007 ».
_NATURES_NUMEROTEES = {
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
# Les autres natures, pour un nom ecrit en anglais : les intitules du corpus
# sont tous francais. « Article 9 of the Uniform Act », « Article 31 of the
# Treaty » ne trouvaient aucun texte de leur nature, meme lu (revue du
# 2026-10-29).
_NATURES_EQUIVALENTES = {
    **_NATURES_NUMEROTEES,
    "acte": ("acte", "act"),
    "act": ("acte", "act"),
    "traite": ("traite", "treaty"),
    "treaty": ("traite", "treaty"),
    "treatie": ("traite", "treaty"),
    "statut": ("statut", "statute"),
    "statute": ("statut", "statute"),
    "charte": ("charte", "charter"),
    "charter": ("charte", "charter"),
    "reglement": ("reglement", "regulation"),
    "regulation": ("reglement", "regulation"),
    "accord": ("accord", "agreement"),
    "agreement": ("accord", "agreement"),
    "protocole": ("protocole", "protocol"),
    "protocol": ("protocole", "protocol"),
}


def _natures(mot: str) -> Tuple[str, ...]:
    nature = normaliser_le_nom(mot)
    if nature not in _NATURES_EQUIVALENTES and nature.endswith("s"):
        nature = nature[:-1]
    return _NATURES_EQUIVALENTES.get(nature, (nature,))


def _forme_des_numeros(texte: str) -> str:
    """« 92-007 » et « 92/007 » : le meme numero."""
    return texte.replace("-", "/")


# Les mots de nature d'un texte numerote, sous la forme de normaliser_le_nom.
_MOTS_DE_NATURE = frozenset(mot for mots in _NATURES_NUMEROTEES.values() for mot in mots)


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


# ==================== TEXTES DESIGNES PAR LEUR NATURE ====================


@dataclass
class _Lecture:
    """Ce que la lecture d'une reponse consulte a chaque citation."""

    texte: str
    sans_italiques: str
    """Le texte, chaque « _ » ouvrant un italique lu comme un blanc (_preparer)."""
    extraits: Sequence[ChunkResult]
    noms: Dict[int, Tuple[str, str]]
    """(intitule, reference) de chaque texte lu, sous la forme de normaliser_le_nom."""
    premiers: Dict[int, ChunkResult] = field(default_factory=dict)
    """Le premier extrait de chaque texte lu : sa date et son intitule."""
    _fins_de_phrase: Optional[List[int]] = None
    _noms_lus: Dict[Tuple, Optional["_Nom"]] = field(default_factory=dict)
    """Ce que _identifier a deja lu dans cette reponse (_identifier)."""
    noms_identifies: int = 0
    """Les noms lus par _identifier hors cache et hors numero (_NOMS_LUS_AU_PLUS)."""
    articles_trouves: Dict[Tuple, "_Resolution"] = field(default_factory=dict)
    """
    L'article de ce numero dans ces textes (_dans_ces_textes, _sans_antecedent),
    deja cherche : la meme reponse cite souvent le meme article, et chaque
    recherche relit chaque extrait (600 « L'article 76 » : 6 600 lectures).
    """
    _par_numero_propre: Optional[Dict[str, List[ChunkResult]]] = None

    def designes_par(self, citation: "re.Match") -> List[ChunkResult]:
        """
        Les extraits dont le texte est celui que la citation (TEXTE_NUMEROTE_REGEX)
        designe (designe_ce_texte), dans l'ordre du contexte. Seuls les textes de
        ce numero propre sont compares : chaque texte numerote d'une reponse
        relisait l'intitule de chaque extrait (400 textes numerotes, 22 extraits).
        """
        if self._par_numero_propre is None:
            self._par_numero_propre = {}
            for extrait in self.extraits:
                propre = numero_propre(extrait.law_title)
                if propre is not None:
                    self._par_numero_propre.setdefault(propre.numero, []).append(extrait)
        numero = _forme_des_numeros(normaliser_le_nom(citation.group("numero")))
        return [
            e for e in self._par_numero_propre.get(numero, ())
            if designe_ce_texte(citation, e.law_title, e.publication_date)
        ]

    def phrase_finie_entre(self, debut: int, fin: int) -> bool:
        """
        Une phrase finit-elle entre `debut` et `fin` (_FIN_DE_PHRASE) ? Les fins
        de phrase sont cherchees une fois par reponse : chaque citation sans nom
        relirait sinon tout le texte depuis le dernier texte numerote.
        """
        if self._fins_de_phrase is None:
            self._fins_de_phrase = [m.start() for m in _FIN_DE_PHRASE.finditer(self.texte)]
        rang = bisect_left(self._fins_de_phrase, debut)
        return rang < len(self._fins_de_phrase) and self._fins_de_phrase[rang] < fin

    @classmethod
    def de(cls, texte: str, extraits: Sequence[ChunkResult]) -> "_Lecture":
        noms: Dict[int, Tuple[str, str]] = {}
        premiers: Dict[int, ChunkResult] = {}
        for extrait in extraits:
            if extrait.law_id not in noms:
                noms[extrait.law_id] = (
                    normaliser_le_nom(extrait.law_title), normaliser_le_nom(extrait.reference),
                )
                premiers[extrait.law_id] = extrait
        # Sans « _ », le texte meme : la substitution parcourait 32 000 caracteres
        # pour rien (2 ms, revue du tour final).
        sans_italiques = _SOULIGNE_OUVRANT.sub(" ", texte) if "_" in texte else texte
        return cls(texte, sans_italiques, extraits, noms, premiers)


def _textes_de_cette_nature(nature: str, noms: Dict[int, Tuple[str, str]]) -> Set[int]:
    """
    Les textes consultes de cette nature, ou d'une nature qui la traduit
    (« law » : « loi ») : « de ce code » vise un des codes lus (_natures_propres).
    """
    formes = frozenset(_natures(nature))
    return {law_id for law_id, (titre, _) in noms.items() if formes & _natures_propres(titre)}


# Ce qui, apres « portant », mene a l'objet de l'intitule : « portant RÉVISION
# DE LA Constitution », « portant Code du travail ».
_VERS_L_OBJET = frozenset({
    "revision", "refonte", "de", "du", "des", "la", "le", "les", "l", "un", "une", "nouveau",
    "nouvelle",
})


@lru_cache(maxsize=4096)
def _natures_propres(titre_normalise: str) -> FrozenSet[str]:
    """
    Les natures du texte lui-meme, traductions comprises : celle qui ouvre son
    intitule, et celle de l'instrument qu'il porte (« Loi … portant Code du
    travail » : une loi et un code ; « Décret … portant Code des Marchés
    Publics » ; « Loi … portant révision de la Constitution »).

    POURQUOI PAS TOUT MOT DE NATURE DE L'INTITULE. 187 decrets du corpus
    « ratifiant l'Accord de prêt… », des lois « autorisant … à ratifier la
    Convention… », la loi « relative à la lutte contre … la traite des
    personnes » portent un autre mot de nature dans leur objet. Apres un de
    ces decrets, « l'article 12 de la loi » ou « du code » allait a son propre
    article 12, parce que son intitule cite « la loi n°2019/008 … portant
    Code Pétrolier » (sondes du 2026-10-29, tour 3).
    """
    # Avant le renvoi a un autre texte : « … d'application DE LA LOI n°2019/008
    # … portant Code Pétrolier » ne fait pas du decret un code.
    renvoi = _RENVOI_DANS_L_INTITULE.search(titre_normalise)
    mots = re.findall(r"[^\W_]+", titre_normalise[:renvoi.start()] if renvoi else titre_normalise)
    propres = set()
    tete = next(
        (mot for mot in mots or re.findall(r"[^\W_]+", titre_normalise) if mot in _NATURES_DE_TEXTE),
        None,
    )
    if tete is not None:
        propres.add(tete)
    for rang, mot in enumerate(mots):
        if mot not in ("portant", "instituant"):
            continue
        objet = next((m for m in mots[rang + 1:] if m not in _VERS_L_OBJET), None)
        if objet in _NATURES_DE_TEXTE:
            propres.add(objet)
    return frozenset(forme for mot in propres for forme in _natures(mot))


@lru_cache(maxsize=4096)
def _mots_du_titre(titre_normalise: str) -> FrozenSet[str]:
    return frozenset(re.findall(r"[^\W_]+", titre_normalise))


@lru_cache(maxsize=4096)
def _racines_du_titre(titre_normalise: str) -> FrozenSet[str]:
    return frozenset(_racine(mot) for mot in _mots_du_titre(titre_normalise))


@lru_cache(maxsize=4096)
def _initiales_du_titre(titre_normalise: str) -> str:
    return _initiales(re.findall(r"[^\W_]+", titre_normalise))


# Un sigle ecrit dans le nom : « et du GIE », « de l'APZE ».
_SIGLE_DANS_LE_NOM = re.compile(r"(?<![\w’'])[A-Z][A-Z0-9]{1,7}(?![\w’'])")


def _porte_ces_mots(
    titre: str, exigences: Sequence[FrozenSet[str]], sigles: FrozenSet[str]
) -> bool:
    """
    L'intitule porte-t-il chacun de ces mots ? Un sigle du nom est porte par
    les initiales des mots de l'intitule : « GIE », « groupement d'intérêt
    économique ».
    """
    racines = _racines_du_titre(titre)
    for exigence in exigences:
        if exigence & racines:
            continue
        sigle = next((s for s in sigles if _racine(s) in exigence), None)
        if sigle is None or sigle not in _initiales_du_titre(titre):
            return False
    return True


def _sigles_du_nom(nom: str) -> FrozenSet[str]:
    # Trois lettres au moins : deux initiales se rencontrent par hasard.
    return frozenset(
        normaliser_le_nom(sigle) for sigle in _SIGLE_DANS_LE_NOM.findall(nom)
        if len(sigle) >= 3 and normaliser_le_nom(sigle) not in _GENERIQUES
    )


# Une annee dans un nom : « le Traité de 1993 ». Pas le debut d'un numero de
# texte : « 2021 » de « décret 2021/745 » n'est pas une date.
_ANNEE_ECRITE = re.compile(r"(?<![\d/-])(?:18|19|20)\d{2}(?![\d/-])")
# L'annee qui suit un nom reconnu : « le Code minier de 2016 », « la loi de
# finances 2023 », « … pour l'exercice 2023 ».
_ANNEE_APRES_LE_NOM = re.compile(
    r"\s*,?\s*(?:(?:de|du|of)\s+)?"
    r"(?:(?:pour|for)\s+(?:(?:l['’]\s*|the\s+)?(?:exercice|(?:fiscal|financial|budget)\s+year)\s+)?)?"
    r"(?P<annee>(?:18|19|20)\d{2})(?![\d/-])",
    re.IGNORECASE,
)
# Mots d'un nom qui ne distinguent aucun texte du corpus : « Code du travail
# camerounais », « Acte uniforme révisé », « Code civil applicable au
# Cameroun ».
_SANS_PORTEE = frozenset({
    "camerounais", "camerounaise", "camerounaises", "cameroun", "cameroon", "cameroonian",
    "republique", "revise", "revisee", "nouveau", "nouvelle", "ancien", "ancienne", "actuel",
    "actuelle", "vigueur", "applicable", "applicables", "annote", "annotee", "edition",
    "consolide", "consolidee",
    "republic", "new", "former", "current", "amended", "revised", "consolidated",
})
# Mots-outils, francais et anglais, absents des listes precedentes : « Law on
# Freedom of Association in Cameroon » ne demande pas « in » a l'intitule,
# « Law laying down… » pas « down », « notre Code du travail » pas « notre ».
_MOTS_OUTILS = frozenset({
    "in", "by", "at", "from", "into", "within", "an", "its", "their", "his", "her", "this",
    "these", "those", "such", "any", "all", "each", "every", "other", "whose", "number",
    "numero", "num", "ce", "cet", "cette", "ces", "son", "sa", "ses", "un", "une",
    "down", "up", "out", "about", "upon", "via", "per", "thereof", "herein", "hereof",
    "notre", "votre", "nos", "vos", "our", "your", "meme", "memes", "lay",
})


# Les mots qui enchainent la phrase (_SUITES), sous la forme de
# normaliser_le_nom : « pour », « selon », « prévoit » ne distinguent aucun
# texte (« la loi de finances pour l'exercice 2023 »).
_SUITES_NORMALISEES = frozenset(normaliser_le_nom(mot) for mot in _SUITES)


def _distinctif(mot: str) -> bool:
    """Un mot (forme de normaliser_le_nom) qui distingue un texte d'un autre de meme nature."""
    return (
        len(mot) > 1 and not any(c.isdigit() for c in mot)
        and mot not in _LIAISONS and mot not in _GENERIQUES and mot not in _NATURES_DE_TEXTE
        and mot not in _DETERMINANTS and mot not in _SANS_PORTEE and mot not in _MOIS_ECRITS
        and mot not in _SUITES_NORMALISEES and mot not in _MOTS_OUTILS
    )


# Les noms anglais des textes du corpus : les intitules sont francais, le
# prompt anglais demande « Article X of the Y Law ». « Article 5 of the
# Consumer Protection Law », « Article 120 of the General Civil Service
# Statute » ne trouvaient aucun intitule, et etaient signales, texte et
# article lus (revue du 2026-10-29). Expressions d'abord, les plus longues en
# tete, puis mot a mot ; un mot sans traduction est cherche tel quel.
_EXPRESSIONS_ANGLAISES = (
    ("general rules and regulations of the public service", "statut general fonction publique"),
    ("general rules and regulations of the civil service", "statut general fonction publique"),
    ("decentralised local authorities", "collectivites territoriales decentralisees"),
    ("decentralized local authorities", "collectivites territoriales decentralisees"),
    ("transparency and good governance", "transparence bonne gouvernance"),
    ("carriage of goods by road", "transport marchandises route"),
    ("simplified recovery procedures", "procedures simplifiees recouvrement"),
    ("economic interest group", "groupement interet economique"),
    ("persons with disabilities", "personnes handicapees"),
    ("trafficking in persons", "traite personnes"),
    ("collective proceedings", "procedures collectives"),
    ("criminal procedure", "procedure penale"),
    ("public procurement", "marches publics"),
    ("public contracts", "marches publics"),
    ("family benefits", "prestations familiales"),
    ("civil service", "fonction publique"),
    ("public service", "fonction publique"),
    ("civil status", "etat civil"),
    ("land tenure", "regime foncier"),
    ("state lands", "regime domanial"),
    ("disabled persons", "personnes handicapees"),
    ("human trafficking", "traite personnes"),
    ("occupational diseases", "maladies professionnelles"),
    ("industrial accidents", "accidents travail"),
    ("work accidents", "accidents travail"),
    ("supreme court", "cour supreme"),
    ("administrative courts", "tribunaux administratifs"),
    ("constitutional council", "conseil constitutionnel"),
    ("public establishments", "etablissements publics"),
    ("public finance", "finances publiques"),
    ("local taxation", "fiscalite locale"),
    ("electronic commerce", "commerce electronique"),
    ("electronic communications", "communications electroniques"),
    ("general commercial law", "droit commercial general"),
    ("commercial companies", "societes commerciales"),
    ("cooperative societies", "societes cooperatives"),
    ("cooperative companies", "societes cooperatives"),
    ("military justice", "justice militaire"),
    ("framework law", "loi cadre"),
    ("regional and local authorities", "collectivites territoriales decentralisees"),
    ("local and regional authorities", "collectivites territoriales decentralisees"),
    ("social security", "securite sociale"),
    ("social insurance", "prevoyance sociale"),
    ("money laundering", "blanchiment"),
    ("political parties", "partis politiques"),
    ("trade unions", "syndicats"),
    ("intellectual property", "propriete intellectuelle"),
    ("industrial property", "propriete industrielle"),
    ("neighbouring rights", "droits voisins"),
    ("higher education", "enseignement superieur"),
    ("urban planning", "urbanisme"),
    ("town planning", "urbanisme"),
    ("personal data", "donnees personnel"),
    ("human rights", "droits homme"),
    # Revue du 2026-10-29, tour 3 : intitules OHADA et lois lus, signales a tort.
    ("general rules and regulations", "statut general"),
    ("measures of execution", "voies execution"),
    ("discharge of debts", "apurement passif"),
    ("business law", "droit affaires"),
)
_MOTS_ANGLAIS = {
    "labour": ("travail",), "labor": ("travail",), "employment": ("travail",),
    "finance": ("finances",), "financial": ("financier", "financiere"),
    "consumer": ("consommateur",), "consumers": ("consommateur",),
    "arbitration": ("arbitrage",), "penal": ("penal",), "criminal": ("penal",),
    "commercial": ("commercial",), "company": ("societes",), "companies": ("societes",),
    "cooperative": ("cooperatives",), "cooperatives": ("cooperatives",),
    "security": ("suretes", "securite"), "securities": ("suretes",), "sureties": ("suretes",),
    "secured": ("suretes",), "proceedings": ("procedures",), "insolvency": ("collectives",),
    "recovery": ("recouvrement",), "enforcement": ("execution",), "accounting": ("comptable",),
    "mining": ("minier",), "petroleum": ("petrolier",), "oil": ("petrolier",),
    "insurance": ("assurances",), "nationality": ("nationalite",), "military": ("militaire",),
    "procurement": ("marches",), "family": ("familiales",), "benefits": ("prestations",),
    "freedom": ("liberte",), "terrorism": ("terrorisme",), "environment": ("environnement",),
    "environmental": ("environnement",), "health": ("sante",), "competition": ("concurrence",),
    "electronic": ("electronique",), "cybersecurity": ("cybersecurite",),
    "cybercrime": ("cybercriminalite",), "electricity": ("electricite",), "water": ("eau",),
    "land": ("foncier",), "investment": ("investissement",), "investments": ("investissement",),
    "disabled": ("handicapees",), "disability": ("handicapees",), "persons": ("personnes",),
    "judicial": ("judiciaire",), "organization": ("organisation",), "supreme": ("supreme",),
    "court": ("cour",), "courts": ("tribunaux",), "administrative": ("administratifs",),
    "constitutional": ("constitutionnel",), "council": ("conseil",),
    "decentralised": ("decentralisees",), "decentralized": ("decentralisees",),
    "local": ("locale", "locales"), "authorities": ("collectivites",),
    "territorial": ("territoriales",), "transparency": ("transparence",),
    "governance": ("gouvernance",), "establishments": ("etablissements",),
    "public": ("public", "publique", "publics", "publiques"), "private": ("prive", "privee"),
    "tax": ("fiscalite", "impots"), "taxation": ("fiscalite",), "pensions": ("pensions",),
    "pension": ("pensions",), "social": ("sociale",), "goods": ("marchandises",),
    "road": ("route", "routiere"), "carriage": ("transport",), "status": ("statut",),
    "freedoms": ("liberte",), "associations": ("association",),
    "medicines": ("medicament",), "medicine": ("medicament",), "copyright": ("auteur",),
    "governmental": ("gouvernementales",), "management": ("gestion",),
    "organisations": ("organisations",), "organizations": ("organisations",),
    "liberty": ("liberte",), "rights": ("droits",), "child": ("enfant",),
    "children": ("enfant",), "women": ("femme",), "property": ("propriete",),
    "communication": ("communication",), "orientation": ("orientation",),
    "repression": ("repression",), "suppression": ("repression",), "acts": ("actes",),
    # Revue du tour 3 : « Article 5 of the Law governing the Electricity
    # Sector », « … on Private Investment Incentives » etaient signales, la
    # loi lue. Un mot anglais sans traduction est cherche tel quel dans des
    # intitules francais : il ne s'y trouve pas, et la citation est signalee,
    # jamais rattachee a un autre texte.
    "sector": ("secteur",), "sectors": ("secteur",), "incentive": ("incitations",),
    "incentives": ("incitations",), "state": ("etat",), "entities": ("entites",),
    "regional": ("regionales", "regions"), "regions": ("regions",),
    "decentralization": ("decentralisation",), "councils": ("conseils",),
    "municipalities": ("communes",), "traditional": ("traditionnelles",),
    "chiefdoms": ("chefferies",), "forest": ("forets", "forestier"),
    "forests": ("forets", "forestier"), "forestry": ("forets", "forestier"),
    "wildlife": ("faune",), "fisheries": ("peche",), "fishing": ("peche",),
    "energy": ("energie",), "renewable": ("renouvelables",), "gas": ("gaz",),
    "banking": ("bancaire", "banques"), "bank": ("banque",), "banks": ("banques",),
    "customs": ("douanes", "douanier"), "income": ("revenu",),
    "budgetary": ("budgetaire",), "press": ("presse",), "political": ("politiques",),
    "parties": ("partis",), "unions": ("syndicats", "syndicales"),
    "retirement": ("retraite",), "servants": ("fonctionnaires",), "army": ("armee",),
    "defence": ("defense",), "penitentiary": ("penitentiaire",),
    "judiciary": ("judiciaire",), "tribunals": ("tribunaux",), "assembly": ("assemblee",),
    "senate": ("senat",), "housing": ("logement",), "planning": ("urbanisme", "planification"),
    "tourism": ("tourisme",), "livestock": ("elevage",), "food": ("alimentaire",),
    "pharmacy": ("pharmacie",), "medical": ("medicale", "medical"), "hospital": ("hopital",),
    "hospitals": ("hopitaux",), "higher": ("superieur",), "sports": ("sportives", "sport"),
    "intellectual": ("intellectuelle",), "industrial": ("industrielle",),
    "neighbouring": ("voisins",), "shipping": ("marine", "maritime"),
    "postal": ("postal", "poste"), "digital": ("numerique",), "data": ("donnees",),
    "personal": ("personnel", "personnelles"), "marriage": ("mariage",),
    "minors": ("mineurs",), "refugees": ("refugies",), "foreigners": ("etrangers",),
    "citizenship": ("nationalite",), "standards": ("normes",), "quality": ("qualite",),
    "statistics": ("statistique",), "research": ("recherche",),
    "enterprises": ("entreprises",), "laundering": ("blanchiment",), "work": ("travail",),
    "trafficking": ("traite",), "smuggling": ("trafic",), "traffic": ("circulation",),
    "activity": ("activite",), "activities": ("activites",),
    # Revue du 2026-10-29, tour 3 : « Law on the Fight against Trafficking in
    # Persons », « Anti-Trafficking Law », « Treaty on the Harmonisation of
    # Business Law in Africa »… etaient signales, le texte lu.
    "fight": ("lutte",), "against": ("contre",), "anti": ("lutte", "contre"),
    "discharge": ("apurement",), "debts": ("passif",), "debt": ("passif", "dette"),
    "business": ("affaires",), "africa": ("afrique",), "harmonization": ("harmonisation",),
    "contracts": ("contrats",), "contract": ("contrat", "contrats"),
    "economic": ("economique",), "interest": ("interet",), "group": ("groupement", "groupe"),
    "groups": ("groupement", "groupe"),
}
# Mots qui ne nomment jamais un texte, meme apres « de » : « l'article 5 de
# l'annexe », « l'article 12 des dispositions générales », « l'article 5 du
# projet de loi ». « législation », « réglementation », « droit » nomment
# une matiere : ce sont les mots qui SUIVENT qui disent le texte (« de la
# législation du travail »).
_SANS_NOM = frozenset(_racine(mot) for mot in (
    "disposition", "dispositions", "maniere", "facon", "fait", "part", "sorte", "livre",
    "partie", "annexe", "alinea", "paragraphe", "point", "tiret", "cas", "ensemble", "suite",
    "plus", "premier", "premiere", "deuxieme", "second", "troisieme", "dernier", "projet",
    "proposition", "legislation", "reglementation", "droit", "matiere", "regle", "regles",
    "principe", "termes", "terme", "sens", "application", "vertu", "base", "cadre",
))


def _gerondif(mot: str) -> bool:
    """« governing », « establishing » : rien qu'un intitule francais porte."""
    return len(mot) >= 6 and mot.endswith("ing") and mot not in _NON_PARTICIPES


@lru_cache(maxsize=1024)
def _exigences(nom_normalise: str) -> Tuple[FrozenSet[str], ...]:
    """
    Ce que le nom exige d'un intitule pour le designer : pour chacun de ses
    mots distinctifs, les racines qui le disent (un mot anglais et sa
    traduction francaise, « commercial » et « commerciales »).
    """
    texte = nom_normalise
    for anglais, francais in _EXPRESSIONS_ANGLAISES:
        if anglais in texte:
            # « Economic Interest Groups » : l'expression, au pluriel.
            texte = re.sub(r"(?<!\w)" + re.escape(anglais) + r"s?(?!\w)", francais, texte)
    exigences: List[FrozenSet[str]] = []
    for mot in re.findall(r"[^\W_]+", texte):
        traductions = _MOTS_ANGLAIS.get(mot)
        if traductions is not None:
            exigence = frozenset(_racine(forme) for forme in (mot, *traductions))
        elif _distinctif(mot) and not _gerondif(mot):
            exigence = frozenset({_racine(mot)})
        else:
            continue
        if exigence not in exigences:
            exigences.append(exigence)
    return tuple(exigences)


# L'exercice d'une loi de finances ou de reglement, dans son intitule normalise.
_EXERCICE = re.compile(r"(?<!\w)exercice(?: budgetaire)? (?P<annee>(?:18|19|20)\d{2})(?!\d)")


def _de_cette_annee(
    law_id: int,
    annee: int,
    formes: Iterable[str],
    lecture: _Lecture,
    ecrite: Optional[date] = None,
    inconnue: bool = False,
) -> bool:
    """
    Le texte lu est-il celui de cette annee : son annee d'acte, l'exercice
    d'une loi de finances (« pour l'exercice 2023 », votee en 2022), ou la
    date que son intitule donne au nom (« Constitution du 02 juin 1972 ») ?

    Jamais un nombre quelconque de l'intitule : « la loi de 2012 » n'est pas
    la loi de 2026 qui modifie « la loi N°2012/001 du 19 avril 2012 ».
    `inconnue` : ce que vaut un texte sans date en base.

    UN TEXTE DONT L'INTITULE DIT SON EXERCICE EST CELUI DE CETTE ANNEE-LA, ET
    D'ELLE SEULE. « la loi de finances 2022 », « … pour l'exercice 2022 »
    allaient a la loi de finances pour l'exercice 2023, votee en 2022 (revue
    du 2026-10-29, tour 3). Une date ecrite en entier (« du 27 décembre
    2022 ») reste comparee a la date de l'acte.
    """
    extrait = lecture.premiers.get(law_id)
    dates = dates_de_l_acte(extrait.publication_date, extrait.law_title) if extrait else None
    if ecrite is not None and dates is not None and dates.date_signature not in (None, ecrite):
        return False
    titre = lecture.noms.get(law_id, ("", ""))[0]
    exercice = _EXERCICE.search(titre)
    renvoi = _debut_du_renvoi(titre)
    if ecrite is None and exercice is not None and (renvoi is None or exercice.start() < renvoi):
        return int(exercice.group("annee")) == annee
    if dates is not None and dates.annee == annee:
        return True
    if re.search(rf"(?<!\w)exercice(?: budgetaire)? {annee}(?!\d)", titre):
        return True
    for forme in formes:
        if re.search(
            rf"(?<!\w){re.escape(forme)} (?:du |de |of )?\d{{1,2}}(?:er)? [a-z]+ {annee}(?!\d)", titre
        ):
            return True
    return inconnue and (dates is None or dates.annee is None)


# Le seul texte de sa nature : « la Constitution » le designe sans autre mot.
_NATURES_UNIQUES = frozenset({"constitution"})
# Les mots d'un nom qui ne dit que la nature d'un texte : « la loi », « le
# Code », « l'Acte uniforme », « the Law ».
_MOTS_D_UNE_NATURE = _NATURES_DE_TEXTE | _LIAISONS | _DETERMINANTS | {"uniforme", "uniformes"}


class _Nom(NamedTuple):
    """Ce qu'un nom designe."""

    etendue: int
    """Longueur du nom dans la mention ; 0 : rien a souligner apres « article N »."""
    lois: Optional[FrozenSet[int]]
    """Les textes lus qu'il designe ; vide : un texte non consulte ; None : aucun texte."""
    natures: FrozenSet[str] = frozenset()
    """Les natures que le nom ecrit (« code », « loi »), pour un renvoi qui suit."""
    precis: bool = True
    """Le nom distingue un texte (« Code du travail ») ; False : une nature seule (« la loi »)."""
    cites: FrozenSet[int] = frozenset()
    """Ceux de `lois` dont l'intitule ne fait que citer le nom (_reecrit_l_article)."""
    vague: bool = False
    """Rien ne designe le texte : un renvoi sans antecedent (« de ce code », aucun code nomme)."""


def _par_ses_mots(
    nom: str, nature: str, lecture: _Lecture, complement: Optional["_Complement"] = None
) -> _Nom:
    """
    Ce que designe un nom de cette nature qu'aucun titre ne contient d'un seul
    tenant : un titre lu qui porte sa nature, CHACUN de ses mots distinctifs
    (traduits s'il est anglais) et, s'il en ecrit une, son annee.

    - « l'article 9 de l'Acte uniforme portant organisation des sûretés » :
      l'intitule est « Acte uniforme RÉVISÉ portant organisation des
      sûretés » ; trois Actes uniformes du corpus portent ce « révisé », que
      l'usage omet ;
    - « l'article 31 du Traité de 1993 » (vraie reponse d'evaluation sur la
      CCJA) : les traites de 1993 et de 2008 etaient lus, et le numero seul
      donnait l'article 31 du traite de 2008 ;
    - « l'article 35 de la loi du 14 août 1992 » ;
    - « Article 120 of the General Civil Service Statute », « Article 76 of
      the 2023 Finance Law » (l'exercice 2023, vote en 2022).

    Sans mot distinctif ni date (« le Traité OHADA », « l'Acte uniforme
    OHADA »), le nom ne designe qu'une nature : les textes lus de cette
    nature, comme « l'article 5 du décret » vise deja les decrets lus ; aucun
    s'il n'y en a pas, jamais un texte d'une autre nature.

    Un nom coupe par une preposition (`complement`, _complement_du_nom) n'est
    pas une nature seule : « de la loi contre la corruption » exige
    « corruption », « Law for the Protection of Consumers » « protection »
    et « consommateur ». Apres une virgule, seule une enumeration continue le
    nom (« portant création, organisation et fonctionnement des… »), et
    seulement d'un intitule qui le continue (_porte_le_complement_des_mots) :
    « de la loi, l'employeur doit… » reste une nature seule.
    """
    etendue = etendue_du_nom(nom)
    # « the Finance Law for 2023 » : « for » arrete le nom, pas son annee.
    apres = _ANNEE_APRES_LE_NOM.match(nom, etendue) if etendue else None
    if apres is not None and not _ANNEE_ECRITE.search(nom[:etendue]):
        etendue = apres.end()
    tete = nom[:etendue]
    exigences = _exigences(normaliser_le_nom(tete))
    annee = _ANNEE_ECRITE.search(tete)
    formes = _natures(nature)
    coupe_par_une_preposition = complement is not None and complement.fin is not None
    if not exigences and annee is None and not coupe_par_une_preposition:
        lois = frozenset(_textes_de_cette_nature(nature, lecture.noms))
        return _Nom(etendue, lois, frozenset(formes), nature in _NATURES_UNIQUES)
    if coupe_par_une_preposition:
        # Le nom souligne comprend son complement : « article 37 de la loi
        # contre la corruption », et non « article 37 de la loi ».
        etendue = max(etendue, complement.fin)
    ecrite = date_ecrite(tete[:annee.end()]) if annee else None
    sigles = _sigles_du_nom(tete)
    lois, cites = set(), set()
    for law_id, (titre, _) in lecture.noms.items():
        if not any(forme in _mots_du_titre(titre) for forme in formes):
            continue
        if not _porte_ces_mots(titre, exigences, sigles):
            continue
        if not _porte_le_complement_des_mots(titre, complement):
            continue
        if annee is not None and not _de_cette_annee(
            law_id, int(annee.group()), formes, lecture, ecrite
        ):
            continue
        lois.add(law_id)
        # « Décret … fixant les modalités d'application de la loi … portant
        # Code Pétrolier » porte la nature et les mots du Code pétrolier, mais
        # dans son renvoi : il ne fait que le citer (_reecrit_l_article).
        if not frozenset(formes) & _natures_propres(titre):
            cites.add(law_id)
    return _Nom(etendue, frozenset(lois), frozenset(formes), cites=frozenset(cites))



def _a_cette_annee(
    mention: str, fin: int, lois: Set[int], lecture: _Lecture
) -> Tuple[int, Set[int]]:
    """
    (fin du nom, textes) quand une annee suit le nom reconnu : « le Code
    minier de 2016 », les codes de 2016 et de 2023 lus, est celui de 2016, et
    seul le code de 2023 lu, il n'est pas consulte. Le plus long debut ne
    departageait pas deux versions d'un texte, et la seconde prenait la
    citation de la premiere.
    """
    ecrite = _ANNEE_APRES_LE_NOM.match(mention, fin)
    if ecrite is None:
        return fin, lois
    prefixe = normaliser_le_nom(mention[:fin])
    formes = (prefixe, _en_francais(prefixe))
    annee = int(ecrite.group("annee"))
    return ecrite.end(), {
        law_id for law_id in lois
        if _de_cette_annee(law_id, annee, formes, lecture, inconnue=True)
    }


# Un sigle en tete du nom : « l'article 76 du CT », « de l'AUDCG ».
_SIGLE = re.compile(r"(?:(?:la|le|les|l['’]|the)\s*)?([A-Z][A-Z0-9]{1,9})(?![\w’'-])")
# Les sigles des codes : ceux des Actes uniformes sont dans leurs references
# (« OHADA-AUDCG-2010 »).
_SIGLES = {
    "ct": "code du travail",
    "cp": "code penal",
    "cpp": "code de procedure penale",
    "cc": "code civil",
    "cmp": "code des marches publics",
    "cgi": "code general des impots",
    "cjm": "code de justice militaire",
    "cpf": "code des prestations familiales",
    "cgctd": "code general des collectivites territoriales decentralisees",
}
_CHIFFRE_ROMAIN = re.compile(r"[ivxlcdm]+")


def _par_le_sigle(mention: str, lecture: _Lecture) -> Optional[_Nom]:
    """
    Ce que designe un sigle : le texte lu dont la reference le porte
    (« OHADA-AUDCG-2010 »), ou le code qu'il abrege. None : un sigle inconnu,
    que la suite lit comme un mot (_par_les_mots).
    """
    trouve = _SIGLE.match(mention)
    if trouve is None:
        return None
    sigle = trouve.group(1).casefold()
    if sigle in _LIAISONS or sigle in _GENERIQUES or _CHIFFRE_ROMAIN.fullmatch(sigle):
        return None
    lois = frozenset(
        law_id for law_id, (_, reference) in lecture.noms.items()
        if sigle in reference.split("-")
    )
    if lois:
        return _Nom(trouve.end(), lois)
    developpe = _SIGLES.get(sigle)
    if developpe is None:
        return None
    # Le sigle se lit comme le nom qu'il abrege, texte qui ne fait que le citer
    # compris : « du CT » n'est pas le « Décret … relevant du Code du Travail ».
    _, lois, cites = _reconnaitre_en_detail(developpe, lecture.noms)
    if not lois:
        lois = set(_par_ses_mots(developpe, "code", lecture).lois or ())
    return _Nom(trouve.end(), frozenset(lois), frozenset({"code"}), cites=frozenset(cites))


def _par_les_mots(
    mention: str, lecture: _Lecture, complement: Optional["_Complement"] = None
) -> Optional[_Nom]:
    """
    Un nom sans nature (« de la législation du travail », « du droit du
    travail », « du contrat de travail ») : les textes lus dont l'intitule
    porte CHACUN de ses mots distinctifs, et ceux du complement qui le
    continue (_par_ses_mots). Aucun : un texte non consulte. Il n'a pas de
    mot distinctif (« du texte », « de l'annexe ») : None, il ne nomme rien.

    POURQUOI. Ces noms gardaient le rattachement au numero seul : « l'article
    76 de la législation du travail », Code du travail non consulte, allait
    encore au morceau 76.37 d'une loi de finances (revue du 2026-10-29).
    """
    etendue = etendue_du_nom(mention)
    exigences = [
        exigence for exigence in _exigences(normaliser_le_nom(mention[:etendue]))
        if not exigence & _SANS_NOM
    ]
    if not exigences:
        return None
    if complement is not None and complement.fin is not None:
        etendue = max(etendue, complement.fin)
    sigles = _sigles_du_nom(mention[:etendue])
    lois = frozenset(
        law_id for law_id, (titre, _) in lecture.noms.items()
        if _porte_ces_mots(titre, exigences, sigles)
        and _porte_le_complement_des_mots(titre, complement)
    )
    return _Nom(etendue, lois)


# ==================== NOMS COUPES PAR UNE VIRGULE OU UNE PREPOSITION ====================

# Les prepositions de _SUITES qui peuvent ouvrir le complement d'un nom de
# texte : « la loi relative à la lutte CONTRE le blanchiment des capitaux »,
# « le décret portant nomination de responsables DANS les services
# extérieurs… », « the Law FOR the Protection of Consumers ». Sur les 2 287
# intitules du corpus, 334 portent « dans », 304 « pour », 121 « avec », 103
# « entre » et 12 « contre » (titres_dev.tsv, 2026-10-29). « a » : le « à »
# sans accent de 9 intitules (« portant nomination de responsables a l'Ecole
# Internationale… »), que _SUITES lit comme le verbe et qui coupait le nom a
# « … de responsables », que d'autres decrets lus portent (balayage, tour
# final). Suivi d'un groupe nominal seulement : « a pour objet » reste le verbe.
_PREPOSITIONS_DU_NOM = frozenset({
    "pour", "contre", "dans", "sous", "avec", "sans", "entre", "for", "under", "a",
})
# Ce qui ouvre un groupe nominal apres la preposition : un determinant, ou une
# majuscule. « pour calculer », « for 2023 » ne continuent aucun nom.
_GROUPE_NOMINAL = re.compile(
    r"\s+(?:(?i:le|la|les|l['’]|un|une|des|du|au|aux|ce|cet|cette|ces|certains|certaines"
    r"|plusieurs|chaque|tout|toute|tous|toutes|son|sa|ses|leur|leurs|the|a|an|this|these"
    r"|those|all|any|each|every|some|its|their)(?:\s+|(?<=['’]))|(?=[A-ZÀ-Ý]))"
)
# Apres une virgule, le nom ne continue que par un nom nu, en minuscules : «
# portant création, ORGANISATION et fonctionnement… ». Un determinant, un
# pronom, une majuscule ouvrent la phrase : « Selon l'article 9 de la loi
# portant régime financier de l'État, le ministre… », « …, qui dispose… ».
_NOM_NU = re.compile(r"\s*(?P<mot>[^\W\d_]+)(?![\w’'])")
# Un complement ecrit apres une virgule s'arrete a la ponctuation suivante,
# comme le nom (CITATION_REGEX) ; borne, pour ne jamais relire la reponse.
_JUSQU_A_LA_PONCTUATION = re.compile(r"[^,.;()\n]{0,400}")
# Ce qui, dans un intitule, relie un nom a sa suite sans la designer : « …
# de la lutte | CONTRE le trafic », « … portant création | , organisation ».
_RELAIS_DU_TITRE = (
    _GENITIFS | _LIAISONS | {"ou", "et"} | {normaliser_le_nom(mot) for mot in _PREPOSITIONS_DU_NOM}
)


class _Complement(NamedTuple):
    """Ce qui continue un nom au-dela de sa coupure, et ce qu'il exige d'un intitule."""

    exigences: Tuple[FrozenSet[str], ...]
    sigles: FrozenSet[str]
    avant: str
    """Le dernier mot du nom avant la coupure, sous la forme de normaliser_le_nom."""
    fin: Optional[int]
    """Fin du complement dans la mention ; None : il suit une virgule, hors de la mention."""


def _complement_du_nom(texte: str, debut: int, mention: str) -> Optional[_Complement]:
    """
    Le complement qui continue le nom de la mention au-dela de sa coupure, et
    les mots qu'il exige ; None : rien ne le continue. `debut` : la place de
    la mention dans la reponse.

    - Une preposition de _SUITES arrete le nom (etendue_du_nom) : « la loi
      relative à la lutte | contre le blanchiment des capitaux » ;
    - CITATION_REGEX arrete la mention a la virgule : « le décret portant
      création | , organisation et fonctionnement des Centres Hospitaliers
      Régionaux ».

    POURQUOI. Le nom coupe designait un AUTRE texte lu : la loi relative a la
    lutte contre la traite des personnes pour « … | contre le blanchiment »,
    le decret des infrastructures sportives pour celui des Centres
    Hospitaliers Regionaux, toute loi lue pour « de la loi | contre la
    corruption », lue comme une nature seule. Sur les intitules du corpus
    cites par leur nom d'usage, 46 mauvais liens sur 4 464 (revue du
    2026-10-29, tour 3).
    """
    fin_du_nom = etendue_du_nom(mention)
    if fin_du_nom == 0 or _ANNEE_APRES_LE_NOM.match(mention, fin_du_nom):
        # « la loi de finances pour l'exercice 2023 » : une annee, lue a part.
        return None
    avant = normaliser_le_nom(mention[:fin_du_nom].split()[-1])
    reste = mention[fin_du_nom:]
    preposition = re.match(r"\s+([^\W\d_]+)", reste)
    if preposition is not None and preposition.group(1).casefold() in _PREPOSITIONS_DU_NOM:
        groupe = _GROUPE_NOMINAL.match(reste, preposition.end())
        if groupe is None:
            return None
        debut_du_complement = fin_du_nom + groupe.end()
        suite = mention[debut_du_complement:]
    elif not reste.strip() and texte[debut + len(mention):debut + len(mention) + 1] == ",":
        debut_du_complement = None
        suite = _preparer(_JUSQU_A_LA_PONCTUATION.match(texte, debut + len(mention) + 1).group())
        apres = re.match(r"\s*([^\W\d_]+)", suite)
        groupe = (
            _GROUPE_NOMINAL.match(suite, apres.end())
            if apres is not None and apres.group(1).casefold() in _PREPOSITIONS_DU_NOM else None
        )
        nu = _NOM_NU.match(suite)
        if groupe is not None:
            # Une preposition et un nom, apres la virgule : « le décret habilitant
            # à signer un Accord de Prêt, pour le financement du Programme… ».
            # Le nom coupe la designait un autre decret d'emprunt lu (balayage de
            # la revue du 2026-10-29, tour final). Le complement n'est exige que
            # d'un intitule qui continue le nom (_porte_le_complement_des_mots).
            suite = suite[groupe.end():]
        elif (
            nu is None or not nu.group("mot").islower()
            or normaliser_le_nom(nu.group("mot")) in _GENITIFS | _LIAISONS | _MOTS_OUTILS
            or _enchaine_la_phrase(nu.group("mot"))
        ):
            return None
        else:
            suite = suite.lstrip()
    else:
        return None
    etendue = etendue_du_nom(suite)
    exigences = _exigences(normaliser_le_nom(suite[:etendue])) if etendue else ()
    if not exigences or (debut_du_complement is None and len(exigences) < 2):
        # Apres une virgule, un mot seul (« …de la fonction publique, oui. »)
        # est la phrase ; la suite d'un nom est une enumeration : « création,
        # organisation et fonctionnement des Centres Hospitaliers… ».
        return None
    fin = None if debut_du_complement is None else debut_du_complement + etendue
    return _Complement(exigences, _sigles_du_nom(suite[:etendue]), avant, fin)


def _titre_continue(titre: str, portees: Iterable[str]) -> bool:
    """
    L'intitule continue-t-il le nom au-dela du debut que la mention en porte ?
    « … relative à la lutte | contre le trafic … », « … portant création | ,
    organisation … », « … portant création | de la Faculté … », « …
    portant nomination de responsables | au Ministère … » : oui. « …
    portant Code du travail | », « … portant loi de finances | de la
    République du Cameroun pour l'exercice 2023 », « … Code électoral | ,
    modifiée et complétée … » : non, ce qui suit ne distingue aucun texte.
    """
    for forme in portees:
        for porte in _motif_du_nom(forme).finditer(titre):
            suivant = next(
                (mot for mot in re.findall(r"[^\W_]+", titre[porte.end():porte.end() + 200])
                 if mot not in _RELAIS_DU_TITRE),
                None,
            )
            if suivant is not None and _distinctif(suivant) and not _participe(suivant):
                return True
    return False


def _porte_le_complement(
    titre: str, prefixe: str, portees: Iterable[str], complement: Optional[_Complement]
) -> bool:
    """
    L'intitule, qui porte le debut `prefixe` de la mention, porte-t-il aussi le
    complement qui continue le nom (_reconnaitre_en_detail) ?

    Il ne l'exige que s'il continue lui-meme le nom a cet endroit
    (_titre_continue). « Code du travail | pour les salariés » ne demande
    rien : l'intitule s'arrete au nom, la suite est la phrase. Une nature
    seule (« la loi | contre la corruption ») non plus : _par_ses_mots lit sa
    preposition.
    """
    if complement is None or all(mot in _MOTS_D_UNE_NATURE for mot in _mots_du_nom(prefixe)):
        return True
    if not _titre_continue(titre, portees):
        return True
    return _porte_ces_mots(titre, complement.exigences, complement.sigles)


def _porte_le_complement_des_mots(titre: str, complement: Optional[_Complement]) -> bool:
    """
    Pour un nom lu mot a mot (_par_ses_mots, _par_les_mots) : l'intitule
    porte-t-il les mots du complement ? Celui d'une preposition est toujours
    exige ; celui d'une virgule, si l'intitule continue le nom apres son
    dernier mot (« portant création | de la Faculté… »).
    """
    if complement is None:
        return True
    if complement.fin is None and not _titre_continue(titre, [complement.avant]):
        return True
    return _porte_ces_mots(titre, complement.exigences, complement.sigles)


def _identifier(mention: str, debut: int, lecture: _Lecture, en_prose: bool = False) -> Optional[_Nom]:
    """
    Ce que designe le nom qui commence a `debut` dans la reponse, `mention`
    etant ce nom lu jusqu'a la ponctuation. None : il ne nomme aucun texte.

    Dans l'ordre : son numero (« la loi n° 92/007 », « Law No. 92/007 ») ; un
    debut que porte un intitule lu (« Code du travail »), et l'annee qui le
    suit ; la reference recopiee de l'en-tete (« PRC-8842 ») ; un sigle ; sa
    nature et ses mots ; ses mots seuls. `en_prose` : un nom ecrit dans la
    phrase, qui n'a pas de mot de liaison pour dire que c'est un nom.

    EN CACHE PAR REPONSE. Une reponse repete ses noms (« le Code du travail »
    a chaque phrase), et la lecture d'un nom est le plus cher de la lecture :
    400 citations du meme nom coupe par une virgule prenaient 0,5 s (revue du
    2026-10-29, tour final). La cle est tout ce que la lecture du nom consulte :
    la mention, le texte designe par son numero s'il y en a un, sinon ce que
    _complement_du_nom lit apres elle : le caractere qui la suit et, si c'est
    une virgule, la proposition qui suit.
    """
    numerote = _designation_numerotee(lecture.sans_italiques, debut)
    if numerote is not None:
        cle = (mention, en_prose, numerote.group(0), numerote.group("nature"),
               numerote.group("numero"), numerote.group("date"))
    else:
        apres = debut + len(mention)
        suivant = lecture.texte[apres:apres + 1]
        proposition = (
            _JUSQU_A_LA_PONCTUATION.match(lecture.texte, apres + 1).group()
            if suivant == "," else None
        )
        cle = (mention, en_prose, None, suivant, proposition)
    if cle not in lecture._noms_lus:
        if numerote is None and lecture.noms_identifies >= _NOMS_LUS_AU_PLUS:
            # Budget epuise : un nom que la lecture ne verifie plus est celui
            # d'un texte non consulte. Il ne peut que signaler sa citation, ou
            # celle qui le suit (_sans_nom, _renvoi), jamais en lier une.
            lecture._noms_lus[cle] = _Nom(etendue_du_nom(mention), frozenset())
        else:
            if numerote is None:
                lecture.noms_identifies += 1
            lecture._noms_lus[cle] = _identifier_sans_cache(
                mention, debut, lecture, en_prose, numerote
            )
    return lecture._noms_lus[cle]


# Les noms qu'une reponse fait lire au plus (_identifier, hors cache et hors
# numero) : citations nommees, noms ecrits en prose, renvois devant un nom.
# Chacun coute jusqu'a 0,6 ms quand il est long et inconnu ; 500 citations de
# noms uniques prenaient 1,7 s, dans la boucle d'evenements de l'API (revue du
# 2026-10-29, tour final). Les 323 vraies reponses en lisent au plus 5. Au-dela,
# le nom est celui d'un texte non consulte : la citation est signalee.
_NOMS_LUS_AU_PLUS = 32


def _identifier_sans_cache(
    mention: str,
    debut: int,
    lecture: _Lecture,
    en_prose: bool,
    numerote: Optional["re.Match"],
) -> Optional[_Nom]:
    if numerote is not None:
        # « l'article 6 de la Loi N°2008/001 » : la loi 2008/001, et non la
        # loi 2026/002, dont l'intitule « modifiant … la loi N°2008/001 »
        # contient le meme nom (revue du 2026-10-28).
        lois = frozenset(e.law_id for e in lecture.designes_par(numerote))
        # Le nom s'arrete au numero (et a sa date) : « fixant les modalités… »
        # qui suit n'est pas a souligner, sauf dans l'intitule de CE texte.
        etendue = numerote.end() - debut
        if lois:
            etendue = max(etendue, _reconnaitre(mention, {i: lecture.noms[i] for i in lois})[0])
        return _Nom(etendue, lois, frozenset(_natures(numerote.group("nature"))))
    complement = _complement_du_nom(lecture.texte, debut, mention)
    fin, lois, cites = _reconnaitre_en_detail(mention, lecture.noms, complement)
    if lois:
        # « de la loi », « du décret », « de l'Acte uniforme » : le nom entier
        # est dans l'intitule de chaque texte lu de cette nature, mais il ne
        # distingue aucun d'eux. Une nature seule, que _resoudre_article lit
        # comme « de cette loi » ; une annee qui suit la rend precise (« le
        # Code de 2016 »). « Code OHADA » reste un nom : « OHADA » n'est pas
        # une nature, et c'est l'intitule entier d'un texte.
        seule = all(
            mot in _MOTS_D_UNE_NATURE for mot in _mots_du_nom(normaliser_le_nom(mention[:fin]))
        )
        avant_l_annee = fin
        fin, lois = _a_cette_annee(mention, fin, lois, lecture)
        nature = nature_nommee(mention)
        natures = frozenset(_natures(nature)) if nature else frozenset()
        precis = not (
            seule and nature is not None and nature not in _NATURES_UNIQUES
            and fin == avant_l_annee
        )
        if lois or fin == avant_l_annee or nature is None:
            return _Nom(fin, frozenset(lois), natures, precis, frozenset(cites & lois))
        # L'annee ecarte chaque texte qui porte le nom d'un seul tenant : il
        # est relu mot a mot, annee comprise. Un intitule peut dire le meme nom
        # autrement, et c'est alors le texte de cette annee.
        return _par_ses_mots(mention, nature, lecture, complement)
    fin, lois = _reconnaitre_la_reference(mention, lecture.noms)
    if lois:
        return _Nom(fin, frozenset(lois))
    sigle = _par_le_sigle(mention, lecture)
    if sigle is not None:
        return sigle
    nature = nature_nommee(mention)
    if nature is not None:
        return _par_ses_mots(mention, nature, lecture, complement)
    if en_prose:
        return None
    return _par_les_mots(mention, lecture, complement)


# ==================== CE QUE LA REPONSE VIENT DE NOMMER ====================


class _Designation(NamedTuple):
    """Un texte que la reponse a designe : une citation, un numero, une phrase."""

    lois: FrozenSet[int]
    """Les textes lus ; vide : un texte que rien de consulte ne porte."""
    natures: FrozenSet[str]
    """Les natures que son nom ecrit."""
    fin: int
    """Fin du nom dans la reponse."""
    cites: FrozenSet[int] = frozenset()
    """Ceux de `lois` dont l'intitule ne fait que citer le nom."""
    par_numero: bool = False
    """
    Un texte designe par son seul numero, en passant (« La loi n° 2022/020 ne
    traite pas du licenciement. »), et non par une citation d'article : il
    n'est pas l'antecedent d'une citation sans nom (_sans_nom).
    """


# Un nom de texte ecrit dans la phrase, hors citation : « Le Code du travail
# prévoit… », « la Loi de finances 2023 », « The Labour Code… ». « un décret
# spécifique », « le statut domanial du bien », « le décret cité » ne sont pas
# des textes, et les lire comme tels faisait signaler « l'article 55 du même
# décret », juste (vraies reponses d'evaluation, 4 cas) : un indefini, un
# renvoi ou un nom imprecis ne designent rien (_NomsEnProse). « Traité »,
# « Arrêté » avec leur accent : « traite », « arrête » sont des verbes.
#
# Les minuscules aussi (tour 3) : « Dans le code du travail, l'article 76
# prévoit… », « … selon la loi de 1992. L'article 76 en fixe le montant »
# allaient au morceau 76.37 des tableaux de l'incident. Un nom ecrit en
# passant ne fait plus que confirmer ou signaler (_sans_nom) ; les 323 vraies
# reponses sont lues a l'identique (2026-10-29).
_NOM_EN_PROSE = re.compile(
    r"(?<![\w’'-])(?:[Cc](?:ode|ODE)S?|[Ll](?:oi|OI)S?(?:-[Cc]adre)?|[Dd][ée]crets?|DÉCRETS?"
    r"|[Oo]rdonnances?|ORDONNANCES?|[Aa]rr[êe]tés?|ARRÊTÉS?|[Aa]ctes?\s+[Uu]niformes?"
    r"|ACTES?\s+UNIFORMES?|[Tt]raités?|TRAITÉS?|[Cc]onstitution|CONSTITUTION)(?![\w’'-])"
    r"|(?<![\w’'-])(?:[A-Z][\w’'-]*\s+){0,3}"
    r"(?:Code|Law|Act|Decree|Treaty|Constitution|Statute|Ordinance)(?![\w’'-])"
)
# Ce qui, devant un nom, dit qu'il n'en est pas un : un indefini (« un
# Décret », « d'une Loi », « any Law ») ou un demonstratif, qui renvoie.
_AVANT_UN_NOM_COMMUN = re.compile(
    r"(?<![\w’'])(?:un|une|d['’]\s*un|d['’]\s*une|des|autres?|chaque|tout|toute|aucun|aucune"
    r"|certains?|certaines?|plusieurs|ce|cet|cette|ces|a|an|any|some|other|each|every|this"
    r"|these|that|those)\s+(?:\*{1,3}|_{1,3})?$",
    re.IGNORECASE,
)


class _NomsEnProse:
    """
    Les noms de texte ecrits dans la reponse hors citation (_NOM_EN_PROSE),
    chacun lu UNE fois, a la demande. Un nom precis seulement : « la loi »,
    « le code » ne designent rien.

    POURQUOI UNE FOIS. Chaque citation relisait tous les noms ecrits depuis le
    dernier texte designe : une citation sans nom ne deplace pas ce point, et
    chaque nom repassait par _identifier, en O(W²) sur sa mention. 80 « la Loi
    camerounaise applicable » suivis de 100 « L'article N précise. » (4,7 k
    caracteres) prenaient 189 s, et une reponse d'allure naturelle de 18,5 k
    caracteres 3,9 s, dans la boucle d'evenements de l'API, donc pour tous
    les utilisateurs (revue du 2026-10-29, tour 3). La lecture d'un nom ne
    depend plus de la citation qui la demande : il s'arrete a la citation qui
    le suit.
    """

    def __init__(self, lecture: _Lecture, debuts_des_citations: Sequence[int]):
        self._lecture = lecture
        self._candidats_lus: Optional[List["re.Match"]] = None
        self._debuts_lus: Optional[List[int]] = None
        self._citations = debuts_des_citations
        self._lus: Dict[int, Optional[_Designation]] = {}

    @property
    def _candidats(self) -> List["re.Match"]:
        # Cherches a la premiere demande : la plupart des reponses n'ont aucune
        # citation sans nom, et ce balayage etait le plus cher de leur lecture.
        if self._candidats_lus is None:
            self._candidats_lus = list(_NOM_EN_PROSE.finditer(self._lecture.texte))
        return self._candidats_lus

    @property
    def _debuts(self) -> List[int]:
        if self._debuts_lus is None:
            self._debuts_lus = [candidat.start() for candidat in self._candidats]
        return self._debuts_lus

    def _lire(self, rang: int) -> Optional[_Designation]:
        if rang not in self._lus:
            self._lus[rang] = self._designation(self._candidats[rang])
        return self._lus[rang]

    def _designation(self, candidat: "re.Match") -> Optional[_Designation]:
        texte = self._lecture.texte
        if _AVANT_UN_NOM_COMMUN.search(texte, max(0, candidat.start() - 16), candidat.start()):
            return None
        # « Under The Labour Code » : le nom commence apres les mots de la phrase.
        tete = next(
            (mot for mot in re.finditer(r"\S+", candidat.group())
             if not _enchaine_la_phrase(mot.group())),
            None,
        )
        if tete is None:
            return None
        suivante = bisect_right(self._citations, candidat.start())
        fin = self._citations[suivante] if suivante < len(self._citations) else len(texte)
        # Borne : rien au-dela des _MOTS_D_UN_DEBUT_AU_PLUS premiers mots ne sert,
        # et _MENTION courait jusqu'a la ponctuation suivante pour chaque nom,
        # O(n²) sur une phrase sans ponctuation (11,9 s, revue du tour final).
        debut_du_nom = candidat.start() + tete.start()
        fin = min(fin, debut_du_nom + _FENETRE_D_UN_NOM)
        trouve = _MENTION.match(texte, debut_du_nom, fin)
        if trouve is None:
            return None
        # Rien au-dela ne sert a reconnaitre un nom (_prefixes) : une phrase sans
        # ponctuation n'est pas relue en entier pour chaque nom qu'elle contient.
        brute = trouve.group(1)
        mots = list(islice(re.finditer(r"\S+", brute), _MOTS_D_UN_DEBUT_AU_PLUS))
        mention = _preparer(brute[:mots[-1].end()] if mots else brute)
        if not mention or _ANAPHORE.match(mention) is not None:
            return None
        nom = _identifier(mention, trouve.start(1), self._lecture, en_prose=True)
        if nom is None or nom.lois is None or not nom.precis:
            return None
        return _Designation(nom.lois, nom.natures, trouve.start(1) + nom.etendue, nom.cites)

    def dernier(self, debut: int, fin: int) -> Optional[_Designation]:
        """
        Le dernier nom precis qui commence dans [debut, fin), parmi les
        _NOMS_RELUS_AU_PLUS derniers noms ecrits avant `fin` ; None : aucun.

        POURQUOI UNE BORNE. « la loi », « le code » ne designent rien et
        n'arretent pas la lecture a reculons : une boucle « le code et la loi
        et le décret » de 32 000 caracteres, apres une citation nommee, faisait
        relire ses 3 000 noms a la citation sans nom qui la suit (8 s, revue du
        tour final). Les vraies reponses ecrivent au plus 15 noms en tout (323
        reponses d'evaluation, 2026-10-29) : la borne ne les touche pas. Au-dela,
        le nom n'est plus lu ; il ne pouvait que signaler la citation.
        """
        rang = bisect_left(self._debuts, fin) - 1
        arret = max(-1, rang - _NOMS_RELUS_AU_PLUS)
        while rang > arret and self._debuts[rang] >= debut:
            designation = self._lire(rang)
            if designation is not None:
                return designation
            rang -= 1
        return None

    def premier(self, debut: int, fin: int) -> Optional[_Designation]:
        """
        Le premier nom precis qui commence dans [debut, fin), parmi les
        _NOMS_RELUS_AU_PLUS premiers ; None : aucun. Borne pour la meme raison
        que `dernier` : la fin d'une proposition sans ponctuation peut en
        contenir des milliers.
        """
        rang = bisect_left(self._debuts, debut)
        arret = min(len(self._debuts), rang + _NOMS_RELUS_AU_PLUS)
        while rang < arret and self._debuts[rang] < fin:
            designation = self._lire(rang)
            if designation is not None:
                return designation
            rang += 1
        return None


_PAS_ENCORE_LU = object()
# Les noms ecrits en prose relus au plus pour une citation (_NomsEnProse.dernier,
# .premier). Les 323 vraies reponses en ecrivent au plus 15 en tout.
_NOMS_RELUS_AU_PLUS = 16
# La part de la reponse lue pour un nom ecrit en prose : ses
# _MOTS_D_UN_DEBUT_AU_PLUS premiers mots tiennent en 400 caracteres (le plus
# long passage cite des 754 citations reelles, numero et nom, en fait 166).
_FENETRE_D_UN_NOM = 400


class _Precedents:
    """
    Ce que la reponse designe autour d'une citation :
    - `citation` : la derniere citation qui designe un texte, ou le dernier
      texte designe par son numero ;
    - `prose()` : le dernier nom ecrit dans la phrase depuis (« Le Code du
      travail prévoit… ») ;
    - `dans_la_suite()` : le premier texte nomme dans la fin de sa
      proposition (« … (Code du travail) »).
    """

    def __init__(
        self,
        lecture: _Lecture,
        noms: _NomsEnProse,
        citation: Optional[_Designation],
        debut: int,
        suite: Tuple[int, int],
    ):
        self._lecture = lecture
        self._noms = noms
        self.citation = citation
        self.debut = debut
        """Debut de la citation dans la reponse."""
        self._suite = suite
        self._prose = _PAS_ENCORE_LU

    def prose(self) -> Optional[_Designation]:
        if self._prose is _PAS_ENCORE_LU:
            depuis = self.citation.fin if self.citation is not None else 0
            self._prose = self._noms.dernier(depuis, self.debut)
        return self._prose

    def recents(self) -> List[_Designation]:
        """Du plus recent au plus ancien : le nom ecrit dans la phrase, puis la citation."""
        return [d for d in (self.prose(), self.citation) if d is not None]

    def dans_la_suite(self) -> Optional[_Designation]:
        """
        Le premier texte nomme dans la fin de la proposition : un nom ecrit
        (« (Code du travail) ») ou un texte designe par son numero (« (loi n°
        92/007) ») ; None : aucun.
        """
        debut, fin = self._suite
        numerote = TEXTE_NUMEROTE_REGEX.search(self._lecture.texte, debut, fin)
        prose = self._noms.premier(debut, numerote.start() if numerote else fin)
        if prose is not None or numerote is None:
            return prose
        return _Designation(
            frozenset(e.law_id for e in self._lecture.designes_par(numerote)),
            frozenset(_natures(numerote.group("nature"))),
            numerote.end(),
        )


# Deux natures d'acte differentes (une loi, un decret) ou deux instruments
# differents (un code, un traite) ne designent jamais le meme texte ; une loi
# peut porter un code.
_NATURES_D_ACTE = frozenset({
    "loi", "law", "decret", "decree", "ordonnance", "ordinance", "arrete", "order",
    "decision", "circulaire", "circular", "instruction",
})


def _compatibles(formes: FrozenSet[str], natures: FrozenSet[str]) -> bool:
    if not natures or formes & natures:
        return True
    return (formes <= _NATURES_D_ACTE) != (natures <= _NATURES_D_ACTE)


def _de_cette_nature(
    designation: _Designation, formes: FrozenSet[str], lecture: _Lecture
) -> Optional[_Designation]:
    """
    La designation restreinte a ses textes de cette nature (« de ce code »
    apres « le Code du travail », « de la loi » apres la loi qui porte ce
    code) ; None : elle est d'une autre nature. Un texte non consulte de
    nature compatible la garde : « l'article 74 du Code du travail […]
    l'article 76 de cette loi » ne va pas a une loi de finances.
    """
    if not designation.lois:
        return designation if _compatibles(formes, designation.natures) else None
    if formes & designation.natures:
        return designation
    ces = frozenset(
        law_id for law_id in designation.lois
        if formes & _natures_propres(lecture.noms.get(law_id, ("", ""))[0])
    )
    return designation._replace(lois=ces, cites=designation.cites & ces) if ces else None


# Un demonstratif en tete du nom : « de ce Code du travail », « de cette loi ».
_DEMONSTRATIF = re.compile(r"(?:ce|cet|cette|ces|this|these)\s+", re.IGNORECASE)
# Les autres mots de renvoi qu'un nom peut suivre (la premiere forme de
# _ANAPHORE) : « dudit Code du travail », « de la présente loi de finances ».
_ANAPHORE_DEVANT_UN_NOM = re.compile(
    r"(?:(?:ce|cet|cette|ces|this|these|the)\s+)?"
    r"(?:m[êe]mes?|ledit|ladite|lesdits|lesdites|dudit|desdits|desdites|said|same"
    r"|pr[ée]sente?s?)\s+",
    re.IGNORECASE,
)


def _renvoi(mention: str, debut: int, lecture: _Lecture, precedents: _Precedents) -> _Nom:
    """
    Ce que designe « de la même loi », « de ce code », « du décret précité » :
    le dernier texte nomme qui peut l'etre, citation ou nom ecrit dans la
    phrase — le demonstratif le designe. `debut` : la place de la mention
    dans la reponse. « de la loi » seule, sans demonstratif, est lue par
    _sans_nom : un nom ecrit en passant n'y choisit pas l'article.

    Un renvoi qui dit sa nature ne vise qu'un texte de cette nature : « Le
    préavis est fixé par l'article 5 du Décret N°2021/745. […] l'article 37
    de ce code » cherchait l'article 37 dans le decret (revue du
    2026-10-29). Le decret ecarte, ce sont les codes lus — ou le code que la
    phrase vient de nommer (_de_cette_nature).

    Un demonstratif devant un NOM (« de cette loi de finances », « de ce
    Code du travail », « de ce décret n° 2015/406 ») designe ce nom, et non
    le dernier texte cite : « l'article 76 du Code du travail, puis l'article
    76 de cette loi de finances » signalait la seconde, la loi de finances
    lue (sondes du 2026-10-29, tour 3). Parmi les textes du nom, celui que la
    reponse vient de citer, s'il en est.
    """
    demonstratif = _DEMONSTRATIF.match(mention)
    en_prose = False
    if demonstratif is None or _ANAPHORE.match(mention, demonstratif.end()) is not None:
        # « dudit Code du travail », « du même Code du travail », « of the said
        # Labour Code » : un nom aussi. Lu comme un renvoi de nature (« code »),
        # sans antecedent, il allait au seul code lu qui porte le numero, Code
        # du travail lu ou non (sondes du tour final). Un nom seulement, comme
        # en prose : « du même texte non plus » n'en est pas un (_par_les_mots
        # y lisait « non » et le signalait).
        demonstratif = _ANAPHORE_DEVANT_UN_NOM.match(mention)
        en_prose = True
    if demonstratif is not None:
        nom = _identifier(
            mention[demonstratif.end():], debut + demonstratif.end(), lecture, en_prose
        )
        if nom is not None and nom.lois is not None and nom.precis:
            lois = next(
                (d.lois & nom.lois for d in precedents.recents() if d.lois & nom.lois), nom.lois
            )
            return nom._replace(
                etendue=demonstratif.end() + nom.etendue, lois=lois, cites=nom.cites & lois
            )
    etendue = etendue_du_nom(mention)
    nature = nature_nommee(mention)
    recents = precedents.recents()
    if nature is None:
        # « du même texte » : le dernier texte nomme, ou rien (le numero seul).
        if recents:
            dernier = recents[0]
            return _Nom(etendue, dernier.lois, dernier.natures, cites=dernier.cites)
        return _Nom(etendue, None)
    formes = frozenset(_natures(nature))
    for designation in recents:
        retenue = _de_cette_nature(designation, formes, lecture)
        if retenue is not None:
            return _Nom(etendue, retenue.lois, designation.natures, cites=retenue.cites)
    # Sans antecedent de sa nature, « de ce code » designe encore un code : un
    # des codes consultes, jamais l'article de meme numero d'une loi de finances.
    # Rien ne l'a nomme : la meme regle qu'une citation sans nom (`vague`).
    return _Nom(
        etendue, frozenset(_textes_de_cette_nature(nature, lecture.noms)), formes, vague=True
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
    raison: Optional[str]
    designe: Optional[FrozenSet[int]] = None
    """
    Les textes que la citation designe, antecedent de « la même loi » qui suit.
    Vide : un texte non consulte. None : elle ne nomme rien, l'antecedent reste.
    """
    natures: FrozenSet[str] = frozenset()
    debut_du_nom: Optional[int] = None
    """Debut du nom dans la reponse."""
    fin_du_nom: Optional[int] = None
    """Fin du nom dans la reponse ; None : la citation s'arrete a « article N »."""
    cites: FrozenSet[int] = frozenset()
    """Ceux de `designe` dont l'intitule ne fait que citer le nom."""
    vague: bool = False
    """
    La citation ne nomme pas son texte (_sans_nom, renvoi sans antecedent) :
    le meme article cite ailleurs AVEC le nom d'un autre texte la signale
    (_signaler_l_article_nomme_ailleurs). Il ne la lie jamais.
    """


def _resoudre_article(
    voulu: str, citation: "re.Match", lecture: _Lecture, precedents: _Precedents
) -> _Resolution:
    """
    Ce que designe « l'article `voulu` » de cette citation.

    Ordre des regles, du plus sur au moins sur :
    1. le texte nomme est consulte — pour « la même loi », « ce code », le
       dernier texte nomme de cette nature (_renvoi) ; pour
       « cette loi de finances », ce nom ; pour « la loi n° 90/053 » (ou
       « la loi 90/053 », « Law No. 90/053 »), celui dont c'est le numero
       PROPRE (designe_ce_texte), jamais un texte dont l'intitule ne fait que
       le citer : son article de ce numero, sinon un morceau de cet article
       (« 55.1 »), sinon le morceau qui en porte l'en-tete (« SECTION_1 »
       d'une loi modificative), sinon sa ligne niveau loi ; sinon RIEN —
       jamais l'article de meme numero d'une AUTRE loi presente ;
    2. la mention nomme un texte — par son numero, par un nom (« Code du
       travail », « Labour Code », « Acte uniforme… »), un sigle, des mots
       (« la législation du travail ») ou en renvoyant a un texte nomme avant
       — que rien de consulte ne porte : rien (« texte_non_consulte »). C'est
       l'incident du 2026-10-29 : « l'article 76 du Code du travail », Code du
       travail non consulte, etait rattache au morceau « 76.37 » d'une loi de
       finances ;
    3. la mention ne nomme aucun texte (« L'article 37 précise… », « l'article
       6 et suivants », « L'article 37 traite du licenciement » : pas de mot
       de liaison), ou seulement sa nature (« de la loi ») : _sans_nom ; et
       sans antecedent, comme un renvoi sans antecedent (« de cette loi »),
       _sans_antecedent.
    """
    texte = lecture.texte
    # « et l'article 45 du Décret N°2019/143 », « point 8, du Décret … » : le
    # nom qui suit est celui du texte de CET article aussi.
    saut = _AVANT_LE_NOM.match(texte, citation.start(3))
    if saut is not None:
        trouve = _MENTION.match(texte, saut.end())
        if trouve is None:
            return _sans_nom(voulu, lecture, precedents)
        debut, mention, lie = trouve.start(1), _preparer(trouve.group(1)), True
    else:
        debut, mention = citation.start(3), _preparer(citation.group(3))
        lie = (
            _LIE_AU_NOM.search(texte, citation.end(2), citation.start(3)) is not None
            or _RENVOI_CONTRACTE.match(mention) is not None
            # « - Article 76 Code du travail : … », dans une liste : sans « du »,
            # mais un nom (une majuscule, une nature), pas un verbe. Le lire
            # comme une citation sans nom le rendait au morceau 76.37.
            or (mention[:1].isupper() and nature_nommee(mention) is not None)
        )
    if not lie or ne_nomme_rien(mention):
        resolution = _sans_nom(voulu, lecture, precedents)
        # « (article 6 et suivants) » : souligne en entier, comme avant.
        suivants = _ET_SUIVANTS.match(texte, citation.start(3))
        if suivants is None:
            return resolution
        return resolution._replace(debut_du_nom=citation.start(3), fin_du_nom=suivants.end())
    # « de notre Code du travail » : aucun intitule ne porte le possessif, et la
    # citation etait signalee, Code du travail lu (sondes du 2026-10-29, tour 3).
    possessif = _POSSESSIF.match(mention)
    if possessif is not None:
        debut, mention = debut + possessif.end(), mention[possessif.end():]
    if _ANAPHORE.match(mention) is not None:
        nom = _renvoi(mention, debut, lecture, precedents)
    else:
        nom = _identifier(mention, debut, lecture)
        if nom is None:
            nom = _Nom(etendue_du_nom(mention), None)
        elif nom.lois is not None and not nom.precis:
            # Une nature seule (« de la loi », « du décret », « of the Law ») :
            # une citation sans nom, restreinte a cette nature. « Voir l'article
            # 74 du Code du travail. […] l'article 76 de la loi » allait a toute
            # loi lue, donc au morceau 76.37 de la loi de finances : l'incident,
            # sans le nom (sondes du 2026-10-29, tour 3).
            return _sans_nom(voulu, lecture, precedents, nom.natures)._replace(
                natures=nom.natures, debut_du_nom=debut,
                fin_du_nom=debut + nom.etendue if nom.etendue > 0 else None,
            )
        elif nom.lois is not None and len(nom.lois) > 1:
            # Plusieurs textes lus portent ce nom (« la loi de finances », trois
            # lois de finances lues ; le Code minier de 2016 et celui de 2023) :
            # celui que la reponse vient de designer, s'il en est un. « l'article
            # 6 de la loi de finances 2026 […] l'article 5 de la loi de finances »
            # allait a la loi de 2023, premiere dans le contexte.
            recente = next(
                (d.lois & nom.lois for d in precedents.recents() if d.lois & nom.lois), None
            )
            if recente is not None:
                nom = nom._replace(lois=recente, cites=nom.cites & recente)
            elif _plusieurs_porteurs(voulu, nom.lois - nom.cites, lecture.extraits):
                # Rien ne dit lequel : « l'article 5 du Code minier », les codes de
                # 2016 et de 2023 lus, chacun avec son article 5. Le premier dans le
                # contexte etait pris ; dans le doute, la citation est signalee
                # (aucune des 754 citations reelles, 2026-10-29). Un texte qui ne
                # fait que citer le nom (la loi qui modifie le Code électoral) ne
                # compte pas : il n'en porte que les articles qu'il reecrit.
                fin_du_nom = debut + nom.etendue if nom.etendue > 0 else None
                return _Resolution(
                    None, False, "article_non_consulte", nom.lois, nom.natures, debut,
                    fin_du_nom, nom.cites,
                )
    fin_du_nom = debut + nom.etendue if nom.etendue > 0 else None
    if nom.lois is None:
        # Un nom qui ne designe rien (« du texte », « du même texte » sans
        # rien avant) : comme une citation sans nom, souligne jusqu'au nom.
        return _sans_nom(voulu, lecture, precedents)._replace(
            debut_du_nom=debut, fin_du_nom=fin_du_nom
        )
    if not nom.lois:
        # La phrase jusqu'a sa fin, pas le nom coupe a la virgule : « du Décret
        # habilitant à signer un Accord de Prêt, pour le financement du
        # Programme Bâtir les Écoles » ne designe pas un autre decret
        # « habilitant à signer un Accord de prêt ».
        fin_de_phrase = re.compile(r"[;:!?\n]|\.(?:\s|$)").search(texte, debut)
        rapproche = _rapprocher(
            texte[debut:fin_de_phrase.start() if fin_de_phrase else len(texte)], voulu, lecture
        )
        if rapproche is not None:
            return rapproche._replace(natures=nom.natures, debut_du_nom=debut, fin_du_nom=fin_du_nom)
        return _Resolution(
            None, False, "texte_non_consulte", frozenset(), nom.natures, debut, fin_du_nom,
            vague=nom.vague,
        )
    if nom.vague:
        # « l'article 76 de cette loi », « of this Law », « du présent code »,
        # « dudit code », sans aucun texte de cette nature nomme avant : rien ne
        # designe le texte, comme pour une citation sans nom (_sans_antecedent).
        # Sur les lignes reelles de l'incident, ces renvois allaient au morceau
        # 76.37 de la loi de finances 2023, et avec le Code du travail et le Code
        # pénal lus, a l'article 76 du Code pénal (revue du 2026-10-29, tour final).
        resolution = _sans_antecedent(voulu, lecture, frozenset(nom.lois - nom.cites))
        # Ce qu'elle designe pour la citation suivante : le texte lie, ou un
        # texte inconnu. Jamais tous les textes de cette nature : « Article 77
        # provides… » apres « Article 76 of this Law » les aurait repris, et
        # rattache par le numero seul.
        designe = (
            frozenset({resolution.extrait.law_id}) if resolution.extrait is not None else frozenset()
        )
        return resolution._replace(
            designe=designe, natures=nom.natures, debut_du_nom=debut, fin_du_nom=fin_du_nom,
            vague=True,
        )
    return _dans_ces_textes_lus(voulu, lecture, nom.lois, nom.cites)._replace(
        natures=nom.natures, debut_du_nom=debut, fin_du_nom=fin_du_nom, cites=nom.cites,
    )


def _sans_nom(
    voulu: str,
    lecture: _Lecture,
    precedents: _Precedents,
    formes: Optional[FrozenSet[str]] = None,
) -> _Resolution:
    """
    Une citation qui ne nomme aucun texte (« L'article 37 précise… »), ou
    seulement sa nature (`formes` : « l'article 37 de la loi », « du décret »,
    « of the Law ») — regle 3 :

    1. la derniere citation designe un texte (de cette nature, pour une
       nature seule) : la citation est lue dans ce texte, et lui seul. « Voir
       l'article 76 du Code du travail. L'article 76 prévoit aussi… », Code du
       travail non consulte, allait au morceau 76.37 d'une loi de finances
       (revue du 2026-10-29) : elle est signalee comme la premiere. Lu sans
       cet article, le texte ne le prete pas a un autre : article_non_consulte.
       Une citation sans nom n'est jamais liee a un tableau de credits, meme
       dans le texte de la citation precedente ;
    2. sinon, RIEN NE DESIGNE SON TEXTE : _sans_antecedent, qui ne la lie que
       si un seul texte lu porte l'article, que l'extrait en porte l'en-tete
       et qu'il n'est pas un tableau de credits ; sinon elle est signalee.
       Une nature d'acte seule (« de la loi », « du décret ») sans aucune
       citation avant : signalee. Un texte designe par son seul numero, en
       passant (« La loi n° 2022/020 ne traite pas du licenciement. »), n'est
       pas l'antecedent d'une citation sans nom : il ne fait que la signaler,
       comme un nom ecrit dans la phrase ;
    3. un nom ecrit dans la phrase depuis la derniere citation (« Le Code du
       travail prévoit des dommages. L'article 76 précise… »), un texte nomme
       dans la fin de sa proposition (« … couvre les dommages (Code du
       travail) »), et la derniere citation quand ce nom la contredit, ne
       CHOISISSENT jamais l'article : ils ne peuvent que signaler la citation
       (texte non consulte, ou un autre texte que celui de l'article retenu).

    POURQUOI 2. La production liait ces citations au premier article de ce
    numero dans le contexte, et le tour precedent au seul texte lu qui le
    porte. Rejouee sur les lignes reelles de jurix_dev, la seconde regle
    donnait encore l'affichage de l'incident : « L'article 76 prévoit des
    dommages-intérêts. », « de cette loi », « of this Law » allaient au
    morceau 76.37 de la loi de finances 2023 quand seuls ses tableaux etaient
    lus, et a l'article 76 du Code pénal (« Exécution de la loi ») quand le
    Code du travail etait lu sans son article 76 (revue du 2026-10-29, tour
    final). Le numero seul ne dit pas quel texte le modele cite de memoire.

    POURQUOI 3. Lus comme le texte de la citation (tour 3), ces noms
    rattachaient « L'article 37 fixe le préavis, que la Loi de finances 2023
    ne modifie pas. » et « La Loi de finances 2023 ne traite pas du
    licenciement. L'article 37 fixe la durée du préavis. » a l'article 37 de
    la loi de finances. Un nom ecrit en passant n'est pas une citation : il
    ne peut qu'empecher un lien, jamais en creer un (la phrase peut dire que
    ce texte ne traite PAS du sujet). Dans le doute, signaler plutot que lier.
    """
    def restreinte(designation: Optional[_Designation]) -> Optional[_Designation]:
        if designation is None or formes is None:
            return designation
        return _de_cette_nature(designation, formes, lecture)

    def rendue(resolution: _Resolution) -> _Resolution:
        # Une citation sans nom ne designe rien : l'antecedent reste. Une nature
        # seule designe le texte lu, ou un texte non consulte : « Selon l'article
        # 5 de l'Acte uniforme, … L'article 5 précise… », aucun Acte uniforme
        # lu, signale aussi la seconde (revue du 2026-10-29).
        designe = None
        if formes is not None and resolution.extrait is not None:
            designe = frozenset({resolution.extrait.law_id})
        elif formes is not None and resolution.raison == "texte_non_consulte":
            designe = frozenset()
        return resolution._replace(designe=designe, vague=True)

    signale = _Resolution(None, False, "texte_non_consulte")
    brute = precedents.citation
    # Le numero d'un texte ecrit dans une phrase precedente ne choisit pas le
    # texte d'une citation sans nom : « La loi n° 2022/020 ne traite pas du
    # licenciement. L'article 76 prévoit des dommages-intérêts. » allait au
    # morceau 76.37 de cette loi, et « La Loi N° 2016/007 portant Code pénal ne
    # traite pas du licenciement. L'article 76 prévoit… » a l'article 76 du Code
    # pénal (revue du 2026-10-29, tour final). Dans la meme phrase, il l'introduit
    # (« D'après la Loi N° 2016/007 …, l'article 337 précise », « le Décret
    # N°2015/406 (article 6 et suivants) », vraies reponses d'evaluation). Une
    # nature seule (« de l'arrêté ») ou un renvoi le gardent : ils le designent
    # (« article 1.19 de l'arrêté » apres « l'Arrêté N°0218/CAB/PR »).
    en_passant = (
        brute
        if formes is None and brute is not None and brute.par_numero
        and lecture.phrase_finie_entre(brute.fin, precedents.debut)
        else None
    )
    citation = None if en_passant is not None else restreinte(brute)
    if citation is not None:
        prose = restreinte(precedents.prose())
        if prose is None or prose.lois & citation.lois:
            if not citation.lois:
                return rendue(signale)
            resolution = _dans_ces_textes_lus(voulu, lecture, citation.lois, citation.cites)
            if (
                formes is None and resolution.extrait is not None
                and est_un_tableau_de_programmes(resolution.extrait)
            ):
                # Un tableau de credits AE/CP n'est jamais l'article d'une
                # citation qui ne nomme pas son texte, meme lue dans le texte de
                # la citation precedente : « L'article 76 de la loi de finances
                # 2023 ne concerne que le budget. L'article 76 prévoit des
                # dommages-intérêts. » liait la seconde au morceau 76.37
                # (sondes de la revue du tour final). Une citation qui NOMME la
                # loi de finances garde son lien.
                return rendue(_Resolution(None, False, "article_non_consulte"))
        else:
            # Le nom ecrit depuis contredit la citation precedente : le numero
            # seul, que la lecture des indices ci-dessous signale toujours (les
            # deux textes ne peuvent pas porter le meme extrait). Le nom est
            # celui d'un texte non consulte : signalee sans chercher le numero,
            # comme le ferait la lecture des indices (texte_non_consulte).
            if not prose.lois:
                return rendue(signale)
            extraits = lecture.extraits
            if formes is not None:
                ces = _textes_de_ces_natures(formes, lecture)
                if not ces:
                    return rendue(signale)
                extraits = [e for e in extraits if e.law_id in ces]
            resolution = _par_le_numero_seul(voulu, extraits)
        indices: Tuple[Optional[_Designation], ...] = (prose, citation)
    else:
        if formes is not None and brute is None and formes <= _NATURES_D_ACTE:
            # « l'article 76 de la loi », « de la loi camerounaise », « of the
            # Law », aucune citation avant : rien ne designe une loi parmi des
            # milliers. La production la liait a la premiere loi lue qui porte
            # ce numero ; rejouee sur les lignes reelles de l'incident, au
            # morceau 76.37 de la loi de finances 2023 (tour final). Signalee.
            # Aucune des 754 citations des 323 vraies reponses n'a cette forme.
            return rendue(signale)
        lois = None
        if formes is not None:
            lois = _textes_de_ces_natures(formes, lecture)
            if not lois:
                # « l'article 76 du code », aucun code lu : c'est le texte qui manque.
                return rendue(signale)
        resolution = _sans_antecedent(voulu, lecture, lois)
        if resolution.extrait is None:
            # Deja signalee : les noms ecrits autour ne changeraient que la
            # raison, que le front n'affiche pas. Ne pas les lire borne le cout
            # d'une reponse faite de citations sans nom (revue du tour final).
            return rendue(resolution)
        indices = (restreinte(precedents.prose()), restreinte(en_passant))
    for indice in (*indices, restreinte(precedents.dans_la_suite())):
        if indice is None:
            continue
        if not indice.lois:
            return rendue(signale)
        if resolution.extrait is not None and not _designe_l_extrait(
            indice, resolution.extrait, voulu
        ):
            # Le texte designe autour d'elle n'est pas celui de l'article retenu.
            # Une raison pour le front, qui ne les distingue pas a l'affichage :
            # le texte designe a ete consulte, et ne donne pas cet article.
            return rendue(_Resolution(None, False, "article_non_consulte"))
    return rendue(resolution)


def _textes_de_ces_natures(formes: FrozenSet[str], lecture: _Lecture) -> FrozenSet[int]:
    """Les textes lus de ces natures (_natures_propres) : « du code », les codes lus."""
    return frozenset(
        law_id for law_id, (titre, _) in lecture.noms.items() if formes & _natures_propres(titre)
    )


def _sans_antecedent(
    voulu: str, lecture: _Lecture, lois: Optional[FrozenSet[int]] = None
) -> _Resolution:
    cle = ("sans antecedent", voulu, lois)
    if cle not in lecture.articles_trouves:
        lecture.articles_trouves[cle] = _sans_antecedent_sans_cache(voulu, lecture, lois)
    return lecture.articles_trouves[cle]


def _sans_antecedent_sans_cache(
    voulu: str, lecture: _Lecture, lois: Optional[FrozenSet[int]] = None
) -> _Resolution:
    """
    L'article `voulu` d'une citation que rien ne designe : ni nom, ni citation
    avant elle (« L'article 76 prévoit… » en tete de reponse, « l'article 76 du
    code », « de cette loi », « of this Law » sans antecedent). `lois` : les
    textes lus qu'elle peut designer (sa nature) ; None : tout texte lu.

    Liee seulement si les trois tiennent :
    - un seul de ces textes porte l'article (_l_article_parmi) : plusieurs le
      portent, et rien ne dit lequel ;
    - un extrait de ce texte en porte l'EN-TETE (« Article 76.- ») : il commence
      bien cet article. Ni un morceau « 76.37 » d'un tableau, ni un morceau
      « 76 » dont le contenu ne dit pas qu'il l'est : le decoupage range l'en-tete
      dans le numero, et le contenu seul ne prouve rien (sur un echantillon de
      6 527 morceaux de jurix_dev, aucun ne porte l'en-tete de son propre
      article, 2026-10-29) ;
    - cet extrait n'est pas un tableau de credits d'une loi de finances.
    Sinon : article_non_consulte, sans lien.

    POURQUOI SI STRICT. Le numero seul ne dit pas quel texte le modele cite de
    memoire, et les vraies reponses n'ont pas ce chemin : sur les 323 reponses
    d'evaluation, une seule citation n'a pas d'antecedent (« l'article 57 de ce
    décret »). En revanche, sur les lignes reelles de l'incident, c'est le lien
    faux de l'incident lui-meme (revue du 2026-10-29, tour final).
    """
    extraits = [e for e in lecture.extraits if lois is None or e.law_id in lois]
    porteurs: Set[int] = set()
    for extrait in extraits:
        if extrait.law_id in porteurs or _l_article_parmi(voulu, (extrait,)) is None:
            continue
        porteurs.add(extrait.law_id)
        if len(porteurs) > 1:
            return _Resolution(None, False, "article_non_consulte")
    if not porteurs:
        return _Resolution(None, False, "article_non_consulte")
    (seul,) = porteurs
    entete = next(
        (
            e for e in extraits
            if e.law_id == seul and e.number and _porte_l_entete_de(voulu, e.content or "")
        ),
        None,
    )
    if entete is None or est_un_tableau_de_programmes(entete):
        return _Resolution(None, False, "article_non_consulte")
    return _Resolution(entete, _normalize_article_number(entete.number) == voulu, None)


def _plusieurs_porteurs(
    voulu: str, lois: FrozenSet[int], extraits: Sequence[ChunkResult]
) -> bool:
    """
    Plusieurs de ces textes portent-ils l'article `voulu` (_l_article_parmi) ?

    Un seul passage sur les extraits : un texte porte l'article des qu'un de
    ses extraits le porte. Texte par texte, chaque citation relisait chaque
    extrait de chaque texte lu : 600 citations sans nom, 22 extraits longs,
    1,7 s contre 30 ms en production (tour final).
    """
    porteurs: Set[int] = set()
    for extrait in extraits:
        if extrait.law_id not in lois or extrait.law_id in porteurs:
            continue
        if _l_article_parmi(voulu, (extrait,)) is not None:
            porteurs.add(extrait.law_id)
            if len(porteurs) > 1:
                return True
    return False


def _designe_l_extrait(designation: _Designation, extrait: ChunkResult, voulu: str) -> bool:
    """
    L'extrait est-il un article du texte designe ? D'un texte qui ne fait que
    citer le nom (« Décret … relevant du Code du Travail »), seulement un
    morceau qui reecrit l'article (_reecrit_l_article).
    """
    if extrait.law_id not in designation.lois:
        return False
    return extrait.law_id not in designation.cites or _reecrit_l_article(extrait, voulu)


# Mots qui ne distinguent aucun texte : natures, liaisons, formules des
# intitules. Le reste (« travail », « AUSCGIE », « foncier », « OHADA »)
# designe un texte.
_MOTS_COMMUNS = frozenset("""
    loi lois law laws acte actes act acts uniforme uniformes revise code codes
    decret decrets decree arrete arretes order ordonnance ordonnances
    decision circulaire instruction traite convention reglement texte textes
    portant relatif relative relatifs relatives modifiant completant fixant
    the and of on du de des la le les l d au aux et en sur par pour dans
    n no nr numero camerounais camerounaise cameroun cameroon republique
""".split())


# Mots des formules d'intitule : presents dans des centaines de textes, ils ne
# designent aucun d'eux (« Décret portant création », « loi de finances »).
_MOTS_DE_FORMULE = frozenset("""
    creation organisation fonctionnement nomination application modalites
    habilitant ratification autorisant approbation attributions finances
    exercice general generale generaux statut statuts regime dispositions
""".split())


# Mots de liaison que _enchaine_la_phrase arrete, mais qui continuent un nom
# (« la loi relative à la lutte contre la corruption », « Accord de Prêt,
# pour le financement du Programme Bâtir les Écoles »).
_PREPOSITIONS_DU_NOM = frozenset("""
    pour contre avec sans sur sous entre par en a à au aux de du des et ou
    of for on and against with
""".split())


def _nom_sans_la_suite(mention: str) -> str:
    """La mention jusqu'au mot qui enchaine la phrase (« … AUSCGIE précise »)."""
    mots = []
    for mot in _MOT.findall(mention):
        brut = mot.strip(".,;").casefold()
        if brut not in _PREPOSITIONS_DU_NOM and _enchaine_la_phrase(brut):
            break
        mots.append(mot)
    return " ".join(mots)


def _mots_distinctifs(texte: str) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """(mots distinctifs, annees) d'un nom, plie et sans accents (normaliser_le_nom)."""
    mots = re.findall(r"[a-z0-9]+", normaliser_le_nom(texte))
    annees = frozenset(m for m in mots if re.fullmatch(r"(1[89]|20)\d\d", m))
    distinctifs = frozenset(
        m for m in mots
        if m not in _MOTS_COMMUNS and m not in _MOTS_DE_FORMULE and m not in annees
        and not m.isdigit() and len(m) >= 3
    )
    return distinctifs, annees


def _rapprocher(nom: str, voulu: str, lecture: "_Lecture") -> Optional["_Resolution"]:
    """
    Dernier recours avant « texte_non_consulte » : un nom que la lecture ne
    reconnait pas, mais dont TOUS les mots distinctifs sont dans l'intitule
    ou la reference d'UN seul texte lu, qui porte cet article. Le modele deforme le nom qu'il recopie (« la Loi
    OHADA AUSCGIE 2014 », « le Code camerounais du travail de 1992 »,
    « l'Acte uniforme sur les sociétés ») ; la citation, juste, etait
    signalee « non retrouvée » (chat du 08/11/2026).

    Jamais quand le nom contredit le texte : une annee absente de sa
    reference, de son intitule et de sa date, un mot distinctif absent
    (« l'article 76 du Code du travail » face a la seule loi de finances lue),
    ou plusieurs textes lus aussi proches.
    """
    if lecture.noms_identifies >= _NOMS_LUS_AU_PLUS:
        return None
    distinctifs, annees = _mots_distinctifs(_nom_sans_la_suite(nom))
    if not distinctifs:
        return None
    retenus = []
    for law_id, extrait in lecture.premiers.items():
        mots_du_texte, annees_du_texte = _mots_distinctifs(
            f"{extrait.law_title} {re.sub(r'[_-]+', ' ', extrait.reference or '')}"
        )
        # L'annee de l'intitule prime : « loi de finances pour l'exercice
        # 2023 », votee en 2022, n'est pas « la loi de finances 2022 ».
        # Sans annee dans l'intitule, celles de la reference et de la date.
        _, annees_du_titre = _mots_distinctifs(extrait.law_title)
        if annees_du_titre:
            annees_du_texte = annees_du_titre
        elif extrait.publication_date:
            annees_du_texte |= {str(extrait.publication_date.year)}
        # TOUS les mots distinctifs du nom : un seul mot commun rattachait
        # « l'Acte uniforme relatif au droit des sociétés commerciales », non
        # lu, a l'Acte uniforme relatif au droit commercial general, ou « le
        # Code de procédure civile » au Code de procédure pénale.
        if not distinctifs <= mots_du_texte or (annees and not annees & annees_du_texte):
            continue
        retenus.append((len(distinctifs), law_id))
    if not retenus:
        return None
    retenus.sort(reverse=True)
    if len(retenus) > 1 and retenus[0][0] == retenus[1][0]:
        return None
    resolution = _dans_ces_textes_lus(voulu, lecture, frozenset({retenus[0][1]}))
    return resolution if resolution.extrait is not None else None


def _dans_ces_textes_lus(
    voulu: str, lecture: _Lecture, lois: FrozenSet[int], cites: FrozenSet[int] = frozenset()
) -> _Resolution:
    """_dans_ces_textes parmi les extraits lus, en cache pour la reponse (articles_trouves)."""
    cle = ("dans ces textes", voulu, lois, cites)
    if cle not in lecture.articles_trouves:
        lecture.articles_trouves[cle] = _dans_ces_textes(voulu, lecture.extraits, lois, cites)
    return lecture.articles_trouves[cle]


def _dans_ces_textes(
    voulu: str,
    extraits: Sequence[ChunkResult],
    lois: FrozenSet[int],
    cites: FrozenSet[int] = frozenset(),
) -> _Resolution:
    """
    L'article `voulu` parmi les extraits de ces textes, et d'eux seuls (regle 1).
    D'un texte qui ne fait que citer le nom (`cites`), seul un morceau qui
    reecrit l'article compte (_reecrit_l_article), et jamais sa ligne niveau
    loi : le « Décret … relevant du Code du Travail » n'est pas le Code du
    travail.
    """
    candidats = [
        e for e in extraits
        if e.law_id in lois and (e.law_id not in cites or _reecrit_l_article(e, voulu))
    ]
    trouve = _l_article_parmi(voulu, candidats)
    if trouve is not None:
        return _Resolution(trouve[0], trouve[1], None, lois, cites=cites)
    entier = next((e for e in candidats if not e.number and e.law_id not in cites), None)
    if entier is not None:
        return _Resolution(entier, False, None, lois, cites=cites)
    # Aucun texte lu ne porte ce nom comme le sien : c'est lui qui manque.
    raison = "texte_non_consulte" if lois <= cites else "article_non_consulte"
    return _Resolution(None, False, raison, lois, cites=cites)


def _par_le_numero_seul(voulu: str, extraits: Sequence[ChunkResult]) -> _Resolution:
    """
    Le premier extrait qui porte ce numero, de n'importe quel texte. Il ne lie
    plus rien seul (_sans_antecedent) : quand un nom ecrit dans la phrase
    contredit la citation precedente (_sans_nom), il ne sert qu'a la raison
    de l'alerte que la lecture des indices pose toujours.
    """
    trouve = _l_article_parmi(voulu, extraits)
    if trouve is not None:
        return _Resolution(trouve[0], trouve[1], None)
    # Sans nom (« L'article 35 précise… »), aucun texte n'est en cause : c'est
    # l'article qui manque aux extraits lus. « texte_non_consulte » disait le
    # contraire de la citation (revue du 2026-10-28).
    return _Resolution(None, False, "article_non_consulte")


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
    # Un seul passage, dans l'ordre du contexte : le premier morceau de ce
    # numero l'emporte des qu'il est vu ; sinon le premier morceau de l'article,
    # sinon le premier qui en porte l'en-tete (cherche tant qu'aucun morceau de
    # l'article n'est vu). Trois passages coutaient 66 comparaisons par
    # citation sur 22 extraits (revue du 2026-10-29, tour final).
    partie = porteur = None
    for extrait in extraits:
        numero = extrait.number
        if not numero:
            continue
        if _normalize_article_number(numero) == voulu:
            return extrait, True
        if partie is None:
            if _numero_de_base(numero) == voulu:
                partie = extrait
            elif porteur is None and _porte_l_entete_de(voulu, extrait.content or ""):
                porteur = extrait
    if partie is not None:
        return partie, False
    if porteur is not None:
        return porteur, False
    return None


# Le debut d'un en-tete d'article (_entete_d_article), jusqu'a son numero, et ce
# qui suit le numero. Separes pour ne compiler aucun motif par numero : 1 500
# citations de numeros differents compilaient 1 500 motifs, 1,4 s de la lecture
# d'une reponse adverse (revue du 2026-10-29, tour final).
_DEBUT_D_ENTETE = re.compile(r"(?<![\w’'])art(?:icle)?\.{0,2}\s*", re.IGNORECASE)
_SUITE_D_ENTETE = re.compile(
    r"(?:\s*(?:er|ère|re|e))?(?![\w.]?\d)(?:\.?\s*\(\d+\))?"
    r"\s*(?:\((?:nouveau|nouvelle|new)\)\s*)?"
    r"(?:\.\s*[-–—]|[-–—:])",
    re.IGNORECASE,
)


@lru_cache(maxsize=1024)
def _numeros_d_entete(contenu: str) -> Tuple[int, ...]:
    """
    Les places du contenu ou commence le numero d'un en-tete possible (apres
    « Article », « Art. », « Art.. »). La plupart des morceaux n'en ont aucune :
    le decoupage range l'en-tete dans le numero. En cache par contenu : chaque
    citation le demande a chaque extrait lu.
    """
    return tuple(debut.end() for debut in _DEBUT_D_ENTETE.finditer(contenu))


def _porte_l_entete_de(voulu: str, contenu: str) -> bool:
    """
    L'extrait porte-t-il l'en-tete de l'article `voulu` (_entete_d_article) ?

    Le meme motif, lu en deux temps : le debut d'en-tete est deterministe (aucun
    numero ne commence par une lettre, un point ou un blanc), le numero est le
    texte meme de `voulu`, et la suite ne depend pas du numero. Seul l'article
    premier (« Article premier.- ») garde son motif, compile une fois.
    """
    if voulu == "1":
        return _entete_d_article("1").search(contenu) is not None
    cible = voulu.casefold()
    longueur = len(voulu)
    return any(
        contenu[debut:debut + longueur].casefold() == cible
        and _SUITE_D_ENTETE.match(contenu, debut + longueur) is not None
        for debut in _numeros_d_entete(contenu)
    )


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


# La fin de la proposition d'une citation : point (pas celui de « No. »,
# « art. », « cf. »), point-virgule, virgule, retour a la ligne. Sans la
# virgule, « L'article 37 fixe le préavis, que la Loi de finances 2023 ne
# modifie pas » et « L'article 37 précise la durée du préavis, et le Code
# pénal punit par ailleurs la fraude » prenaient le texte nomme apres la
# virgule pour celui de la citation (revue du 2026-10-29, tour 3).
# La ponctuation en tete du motif (et les exceptions verifiees apres elle) : le
# moteur saute alors d'une ponctuation a l'autre au lieu d'essayer chaque
# caractere (5 ms sur 32 000 caracteres, revue du tour final).
_FIN_DE_PROPOSITION = re.compile(
    r"[.!?;](?<![Nn]o[.!?;])(?<![Aa]rt[.!?;])(?<!cf[.!?;])(?=\s|$)|[,\n]"
)
# La fin d'une phrase : la meme ponctuation, sans la virgule.
_FIN_DE_PHRASE = re.compile(r"[.!?;](?<![Nn]o[.!?;])(?<![Aa]rt[.!?;])(?<!cf[.!?;])(?=\s|$)|\n")


def _signaler_l_article_nomme_ailleurs(
    occurrences: List[Occurrence], lectures: Sequence[Tuple[str, _Resolution, int]]
) -> None:
    """
    Une citation qui ne nomme pas son texte (`vague`), quand la reponse cite
    ailleurs le MEME article avec le nom d'un autre texte : signalee, en place.

    « **En bref** : l'article 76 prévoit des dommages-intérêts. **En détail** :
    l'article 76 du Code du travail… », avec les seuls tableaux de l'incident :
    la premiere allait au morceau 76.37 de la loi de finances, sous « Articles
    cités », la seconde etait signalee (revue du 2026-10-29, tour 3). Le nom
    ecrit pour le meme numero, avant ou apres, est la meilleure indication du
    texte vise ; il ne fait que signaler, jamais lier ailleurs.

    Il ne confirme pas non plus : une citation sans nom que _sans_antecedent
    signale le reste, meme si le meme article est nomme plus loin (« **En
    bref** : l'article 35… **En détail** : l'article 35 du Code du travail »).
    Une fausse alerte, voulue : la confirmation liait « l'article 76 » au
    morceau 76.37 des qu'une autre phrase nommait la loi de finances pour son
    article 76, et aucune des 323 vraies reponses ne s'en servait (revue du
    2026-10-29, tour final).

    Les noms d'un meme numero sont dedoublonnes : 300 citations nommees du meme
    article et 300 citations sans nom se comparaient deux a deux.
    """
    nommes: Dict[str, Dict[Tuple[FrozenSet[int], FrozenSet[int]], _Designation]] = {}
    for voulu, resolution, _ in lectures:
        if not resolution.vague and resolution.designe is not None:
            nommes.setdefault(voulu, {}).setdefault(
                (resolution.designe, resolution.cites),
                _Designation(resolution.designe, resolution.natures, 0, resolution.cites),
            )
    for voulu, resolution, rang in lectures:
        occurrence = occurrences[rang]
        if not resolution.vague or occurrence.extrait is None:
            continue
        autre = next(
            (
                d for d in nommes.get(voulu, {}).values()
                if not _designe_l_extrait(d, occurrence.extrait, voulu)
            ),
            None,
        )
        if autre is not None:
            occurrences[rang] = Occurrence(
                occurrence.debut, occurrence.fin, "article",
                raison="texte_non_consulte" if not autre.lois else "article_non_consulte",
            )


# Les citations d'article qu'une reponse fait lire au plus. Les 323 vraies
# reponses en ont au plus 10 ; 1 500 citations sans nom dans 32 000 caracteres
# (une sortie degeneree) coutaient 90 ms, dans la boucle d'evenements de l'API
# (revue du 2026-10-29, tour final). Au-dela, chacune est signalee sans lien.
_CITATIONS_LUES_AU_PLUS = 300


class _PassagesLus:
    """
    Les passages lus par les citations d'article (citation et nom), dans l'ordre
    du texte : un texte numerote qui en chevauche un est le nom d'une citation,
    pas une citation a part. Le plus loin qu'ils vont, debut par debut, pour une
    reponse en O(log n) : la comparaison a chaque passage coutait O(n²) sur 400
    textes numerotes et 400 citations.
    """

    def __init__(self) -> None:
        self._debuts: List[int] = []
        self._plus_loin: List[int] = []

    def ajouter(self, debut: int, fin: int) -> None:
        self._debuts.append(debut)
        self._plus_loin.append(max(fin, self._plus_loin[-1]) if self._plus_loin else fin)

    def chevauche(self, debut: int, fin: int) -> bool:
        """Un passage lu commence-t-il avant `fin` et finit-il apres `debut` ?"""
        avant = bisect_left(self._debuts, fin)
        return avant > 0 and self._plus_loin[avant - 1] > debut


def lire_les_citations(texte: str, extraits: Sequence[ChunkResult]) -> List[Occurrence]:
    """
    Toutes les citations de la reponse, dans l'ordre du texte.

    `extraits` : ce que le modele a lu. Une citation qu'aucun ne porte est
    rendue avec sa raison ; elle n'est plus jetee.
    """
    texte = texte or ""
    lecture = _Lecture.de(texte, extraits)

    # Les textes designes par leur numero, rattaches d'avance : ce sont des
    # citations, et aussi l'antecedent possible d'un « article 5 de la même loi ».
    # Au texte dont c'est le numero propre, et a lui seul (designe_ce_texte).
    numerotes = [
        (match, next(iter(lecture.designes_par(match)), None))
        for match in TEXTE_NUMEROTE_REGEX.finditer(texte)
    ]

    occurrences: List[Occurrence] = []
    # Ce que chaque citation d'article a lu, nom compris : son nom n'est pas
    # une autre citation, meme quand elle s'arrete avant lui (nom en gras).
    lues = _PassagesLus()
    # (numero, resolution, rang de l'occurrence) de chaque citation d'article.
    lectures: List[Tuple[str, _Resolution, int]] = []
    derniere: Optional[_Designation] = None
    lus = 0
    citations = list(re.finditer(CITATION_REGEX, texte, re.IGNORECASE))
    noms_en_prose = _NomsEnProse(lecture, [citation.start() for citation in citations])
    for rang, match in enumerate(citations):
        while lus < len(numerotes) and numerotes[lus][0].end() <= match.start():
            numerote, porteur = numerotes[lus]
            lus += 1
            # Le numero ecrit DANS le nom d'une citation (« l'article 5 de la
            # Loi N°2021/022 modifiant la loi n° 90/053 ») : la citation a deja
            # designe son texte, ce n'est pas un autre antecedent.
            if lues.chevauche(numerote.start(), numerote.end()):
                continue
            derniere = _Designation(
                frozenset() if porteur is None else frozenset({porteur.law_id}),
                frozenset(_natures(numerote.group("nature"))),
                numerote.end(),
                par_numero=True,
            )
        if rang >= _CITATIONS_LUES_AU_PLUS:
            # Au-dela du budget, signalee sans etre lue (_CITATIONS_LUES_AU_PLUS).
            lues.ajouter(match.start(), match.end())
            occurrences.append(Occurrence(
                match.start(), _fin_de_la_designation(texte, match), "article",
                raison="article_non_consulte",
            ))
            continue
        # Groupes : 1 = prefixe de code (L/R/D, optionnel), 2 = numero,
        # 3 = texte cite.
        prefixe = (match.group(1) or "").strip()
        numero_cite = f"{prefixe} {match.group(2)}".strip()
        voulu = _normalize_article_number(match.group(2))
        # La fin de sa proposition, sans la citation suivante : elle peut
        # nommer apres coup le texte d'une citation sans nom (_sans_nom).
        suivante = citations[rang + 1].start() if rang + 1 < len(citations) else len(texte)
        point = _FIN_DE_PROPOSITION.search(texte, match.end(2), suivante)
        suite = (match.end(2), point.start() if point is not None else suivante)
        resolution = _resoudre_article(
            voulu, match, lecture,
            _Precedents(lecture, noms_en_prose, derniere, match.start(), suite),
        )
        # Le nom tel que la resolution l'a lu, et non toute la mention, qui
        # court jusqu'a la ponctuation : « la Constitution a été modifiée par
        # la Loi N°2008/001 » nomme aussi la loi 2008/001, citation a part.
        fin_lue = resolution.fin_du_nom if resolution.fin_du_nom is not None else match.start(3)
        lues.ajouter(match.start(), fin_lue)
        if resolution.designe is not None:
            derniere = _Designation(
                resolution.designe, resolution.natures, fin_lue, resolution.cites
            )
        extrait = resolution.extrait
        if resolution.fin_du_nom is None or texte[resolution.debut_du_nom - 1] in "*_":
            fin = _fin_de_la_designation(texte, match)
        else:
            fin = _avant_l_emphase(texte, match, resolution.fin_du_nom)
        lectures.append((voulu, resolution, len(occurrences)))
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
    _signaler_l_article_nomme_ailleurs(occurrences, lectures)

    nommes: Optional[List[TexteNomme]] = None
    for match, extrait in numerotes:
        if lues.chevauche(match.start(), match.end()):
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
