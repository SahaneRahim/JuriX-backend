"""
Les reperes d'un texte : sa nature, son numero d'acte et ses dates.

POURQUOI CE MODULE. Le statut juridique d'un texte nomme d'autres textes
(« modifié par la loi n° 2026/002 du 14 avril 2026 »). Le front compose cette
mention lui-meme, dans la langue de l'interface (le site est bilingue) : il lui
faut des pieces separees, pas une phrase francaise toute faite. La nature et la
date sont en base pour les textes du corpus ; le numero d'acte n'existe que dans
l'intitule, et les textes absents du corpus (« la Loi n° 2012/015 du 21 décembre
2012 ») n'ont que l'intitule saisi dans le releve des liens.

LES DATES. `laws.publication_date` n'est pas une date de publication : c'est la
date que l'intitule donne a l'acte, lue a l'ingestion (scripts/ingest_corpus.py,
_infer_date), d'ou le « Signé le 28 décembre 2021 » de la page. Deux cas
l'empechent d'etre une date de signature, et `dates_de_l_acte` les isole pour
que le front ne puisse pas les afficher comme telle : le 1er janvier de
remplissage (seule l'annee est sure) et l'intitule qui ne date que la
publication au Journal officiel (deux Actes uniformes OHADA).

Fonctions pures, sans base : testables sur les intitules reels du corpus.
"""

import re
from datetime import date
from typing import NamedTuple, Optional

from app.utils.titre_officiel import date_du_texte

# Le numero se lit AVANT l'objet de l'acte : « Décret N°82/241 du 24 juin 1982
# modifiant […] du décret n°77/245 » porte deux numeros, et seul le premier est
# le sien. Meme liste de verbes que scripts/ingest_corpus.py (_DEBUT_OBJET), qui
# lit la date de signature au meme endroit pour la meme raison.
_DEBUT_OBJET = re.compile(
    r"\b(portant|modifiant|compl[ée]tant|fixant|ratifiant|habilitant|autorisant"
    r"|abrogeant|accordant|nommant|relatif|relative|instituant|cr[ée]ant|approuvant"
    r"|rendant|d[ée]clarant|convoquant|mettant|prorogeant|r[ée]gissant"
    r"|r[ée]glementant|organisant|d[ée]finissant|instaurant|attribuant)\b",
    re.IGNORECASE,
)

# « N°2026/002 », « N° 2013/271 », « No. 2020/556 », « n°0827/CAB/PR »,
# « N°69/LF/18 », « N°67-LF-7 », « N°83-19 », « Arrêté N°0967 ». Le numero
# commence par un chiffre : « LOI N° DU » (numero laisse en blanc a l'OCR) n'en
# donne aucun.
_NUMERO = re.compile(
    r"\bn(?:\s*[°º]|o\.|o\s)\s*\.?\s*([0-9][0-9A-Za-z]*(?:\s*[/-]\s*[0-9A-Za-z]+)*)",
    re.IGNORECASE,
)

# « Loi 2019/023 du 24 décembre 2019 » : le numero suit la nature, sans « N° ».
_NUMERO_SANS_SIGNE = re.compile(
    r"^\W*(?:loi(?:-cadre)?|d[ée]cret|ordonnance|arr[êe]t[ée]|d[ée]cision|circulaire)\s+"
    r"((?:19|20)?\d{2}\s*/\s*\d+)\b",
    re.IGNORECASE,
)

# Nature d'un texte absent du corpus, lue sur le premier mot de son intitule
# (« la Loi n° 2012/015 », « l'Ordonnance N°77/2 », « le Décret N°2005/481 »).
# Les valeurs sont celles de laws.type (app/schemas/law.py), accents compris :
# un texte du corpus et un texte absent se decrivent de la meme facon.
_NATURES = (
    ("loi", "loi"),
    ("decret", "décret"),
    ("ordonnance", "ordonnance"),
    ("arrete", "arrêté"),
    ("decision", "décision"),
    ("circulaire", "circulaire"),
    ("instruction", "instruction"),
    ("acte uniforme", "acte uniforme"),
)

# Espace exige apres « la », « le », « les » : « Lettre … » garde son « Le ».
_ARTICLE_DEFINI = re.compile(r"^\s*(?:(?:la|le|les)\s+|l['’]\s*)", re.IGNORECASE)


def _replier(texte: str) -> str:
    """Minuscules sans accents : « Décret » et « DECRET » se comparent."""
    import unicodedata

    decompose = unicodedata.normalize("NFKD", texte.lower())
    return "".join(c for c in decompose if not unicodedata.combining(c))


def numero_de_l_acte(titre: Optional[str]) -> Optional[str]:
    """
    Le numero de l'acte, tel qu'ecrit dans son intitule, espaces retires.

    >>> numero_de_l_acte("Décret N°2021/745 du 28 décembre 2021 fixant …")
    '2021/745'
    >>> numero_de_l_acte("Loi N° 2017/015 du 12 juillet 2017 modifiant … la loi N° 2013/004")
    '2017/015'
    >>> numero_de_l_acte("Acte uniforme révisé portant organisation des sûretés") is None
    True
    """
    if not titre:
        return None
    objet = _DEBUT_OBJET.search(titre)
    tete = titre[: objet.start()] if objet else titre
    trouve = _NUMERO.search(tete) or _NUMERO_SANS_SIGNE.search(_ARTICLE_DEFINI.sub("", tete))
    if not trouve:
        return None
    return re.sub(r"\s+", "", trouve.group(1))


def nature_depuis_intitule(titre: Optional[str]) -> str:
    """
    Nature d'un texte d'apres le premier mot de son intitule ; « autre » a defaut.

    Sert aux textes ABSENTS du corpus, connus par leur seul intitule. Pour un
    texte du corpus, laws.type fait foi.
    """
    replie = _replier(_ARTICLE_DEFINI.sub("", titre or "")).lstrip(" \t*#-_")
    for debut, nature in _NATURES:
        if replie.startswith(debut):
            return nature
    return "autre"


class DatesDeLActe(NamedTuple):
    """Les dates d'un acte, chacune sous un nom qui dit ce qu'elle est."""

    date_signature: Optional[date]
    """Date de l'acte (« Signé le … ») ; None si l'intitule ne la donne pas au jour pres."""
    annee: Optional[int]
    """Annee de la date en base, sure meme quand le jour ne l'est pas."""
    date_publication_jo: Optional[date]
    """Date de publication au Journal officiel, quand c'est la seule que l'intitule donne."""


# « Acte uniforme portant organisation des procédures simplifiées de
# recouvrement […], révisé, publié au Journal officiel de l'OHADA du 15
# novembre 2023 » : la seule date de l'intitule est celle du Journal officiel,
# et l'ingestion l'a prise pour la date de l'acte. Ecrire « Signé le 15
# novembre 2023 » serait faux.
_PUBLIE_AU_JO = re.compile(r"\bpubli[ée]e?s?\s+au\s+journal\s+officiel\b", re.IGNORECASE)


def dates_de_l_acte(jour: Optional[date], titre: Optional[str]) -> DatesDeLActe:
    """
    Les dates d'un acte d'apres `laws.publication_date` (`jour`) et son intitule.

    Un 1er janvier que l'intitule ne mentionne pas est une date de
    remplissage : l'ingestion (_infer_date) l'ecrit quand l'intitule ne donne
    ni jour ni mois. Seule l'annee est alors sure. La regle est UNE, celle de
    titre_officiel.date_du_texte : la page d'un texte, ses lignes de liste
    (app/schemas/reperes.py), les textes nommes par un statut, le chat et le
    contexte du modele (prompts._date_du_texte) la partagent. Deux copies
    avaient deja diverge (« janvier » ici, « janv » abrege la-bas).
    """
    if jour is None:
        return DatesDeLActe(None, None, None)
    if _PUBLIE_AU_JO.search(titre or ""):
        return DatesDeLActe(None, jour.year, jour)
    if date_du_texte(jour, titre) is None:
        return DatesDeLActe(None, jour.year, None)
    return DatesDeLActe(jour, jour.year, None)


def intitule_sans_article(titre: str) -> str:
    """
    « la Loi n° 2012/015 du 21 décembre 2012 » -> « Loi n° 2012/015 du 21 décembre 2012 ».

    Le releve des liens ecrit ses intitules pour une phrase (« modifié par la
    Loi … ») ; affiches seuls, ils commencent par la nature.
    """
    sans = _ARTICLE_DEFINI.sub("", titre or "").strip()
    return sans[:1].upper() + sans[1:]
