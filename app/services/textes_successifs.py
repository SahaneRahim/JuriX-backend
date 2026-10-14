"""
Ce qu'un texte plus recent du corpus a change dans un texte plus ancien.

POURQUOI CE MODULE. Le corpus garde chaque texte dans sa version publiee : le
Code electoral de 2012 tel quel, et a cote la loi de 2026 qui en reecrit des
articles. La recherche remonte souvent l'ancien article, et le chat le citait
comme la regle en vigueur. Archiver l'ancien texte n'est pas la reponse : la
plupart de ses articles restent applicables.

Chaque extrait envoye au modele porte donc la date de son texte et, quand un
texte plus recent l'a touche, une ligne de statut qui nomme ce texte et dit ce
qui en decoule (« Pour ce qui a changé, ce texte fait foi », « L'article ne
s'applique plus »). Le prompt (prompts._FORME_FR) demande de suivre ces lignes,
et dit ce que la date ne departage pas : deux textes DIFFERENTS. Le Code de
justice militaire (2017) est plus recent que le Code penal (2016) ; il ne
regit pas pour autant une agression sexuelle commise par un civil.

D'OU VIENNENT LES LIENS. De app/services/liens_entre_textes.py, ou chacun
porte sa preuve : la disposition qui modifie, abroge ou remplace, relue en
base. Un lien douteux n'y est pas saisi : une ligne de statut fausse egare le
modele plus surement qu'une ligne absente.

LE DECOUPAGE EN BASE N'EST PAS LE TEXTE. Un article est parfois coupe en
morceaux (« 55.1 », « 55.2 »), range sous un pseudo-numero (« SECTION_2 »),
ou colle a la fin de l'article precedent. Une compilation comme le Code civil
reprend les memes numeros dans plusieurs titres. D'ou `section`, `morceaux` et
`sauf` ; scripts/maintenance/verifier_liens_textes.py controle en base, apres
chaque ingestion, que chaque lien designe bien les morceaux voulus.
"""

import re
from datetime import date
from typing import Dict, Iterable, List, NamedTuple, Optional, Tuple

from app.services.liens_entre_textes import LIENS, Lien
from app.utils.chunk_refiner import est_pseudo_numero


class Nature(NamedTuple):
    participe: str
    participe_pluriel: str
    verbe: str


# Ce qu'un texte plus recent fait a un texte plus ancien, ou aux articles
# listes. « abroge_partiellement » : seulement en partie, pour ce que precise
# la portee (une matiere, les dispositions contraires). « complete » : un
# alinea ou un article s'ajoute, l'existant reste en vigueur.
NATURES = {
    "modifie": Nature("modifié", "modifiés", "modifie"),
    "complete": Nature("complété", "complétés", "complète"),
    "abroge": Nature("abrogé", "abrogés", "abroge"),
    "abroge_partiellement": Nature("abrogé en partie", "abrogés en partie", "abroge en partie"),
    "remplace": Nature("remplacé", "remplacés", "remplace"),
}

# Ce que la ligne de statut dit au modele, selon la nature et ce qu'elle vise.
# La consequence est dans la ligne elle-meme : le prompt n'a pas a enumerer
# chaque libelle, et un libelle nouveau ne peut pas rester sans consigne.
CONSEQUENCES = {
    ("modifie", "article"): "Pour ce qui a changé, ce texte fait foi.",
    ("modifie", "texte"): "L'extrait peut encore s'appliquer, sous réserve de cette modification.",
    ("complete", "article"): "L'article reste en vigueur.",
    ("complete", "texte"): "Le texte reste en vigueur.",
    ("abroge", "article"): "L'article ne s'applique plus.",
    ("abroge", "texte"): "Le texte ne s'applique plus.",
    ("abroge_partiellement", "article"): "Hors de ce que vise ce texte, l'article s'applique encore.",
    ("abroge_partiellement", "texte"): "L'extrait peut encore s'appliquer, sous réserve de cette abrogation.",
    ("remplace", "article"): "L'article ne s'applique plus.",
    ("remplace", "texte"): "Le texte ne s'applique plus.",
}


_PLAGE = re.compile(r"(\d+)\s*à\s*(\d+)")
_ORDINAL = re.compile(r"(\d+)\s*(?:er|re|ère|e|ème)")

# Les morceaux d'un texte modificatif qui ne disent rien de ce qu'il change :
# la ligne « Ce texte modifie … » n'y a pas sa place.
_HORS_DISPOSITIF = {"PREAMBULE", "LEGAL_BASIS", "SIGNATURE", "ANNEXE"}


def _numero_de_base(numero: Optional[str]) -> Optional[str]:
    """
    Le numero d'article sous lequel un texte modificatif le designe.

    « 55.2 » est le deuxieme morceau de l'article 55 (voir chunk_refiner) et
    « 1er » s'ecrit « 1 » dans une liste d'articles. Un pseudo-numero
    (preambule, visas, section) n'est pas un article : aucun lien ne le vise.
    """
    if not numero or est_pseudo_numero(numero):
        return None
    base = str(numero).strip().lower().split(".")[0].strip()
    ordinal = _ORDINAL.fullmatch(base)
    return ordinal.group(1) if ordinal else (base or None)


def _vise(designation: str, base: str) -> bool:
    plage = _PLAGE.fullmatch(designation.strip())
    if plage:
        return base.isdigit() and int(plage.group(1)) <= int(base) <= int(plage.group(2))
    return _numero_de_base(designation) == base


def _designe(designations: Tuple[str, ...], numero: Optional[str]) -> bool:
    """
    Aucune designation : tout le texte. Sinon, l'article doit en faire partie,
    ou le morceau etre designe tel quel (« SECTION_1 » porte l'article 142 de
    la loi 2026/003 : seul `par_articles` en a besoin).
    """
    if not designations:
        return True
    if numero is not None and str(numero).strip() in designations:
        return True
    # « 67.3 » designe ce morceau seulement, pas ses voisins 67.1 et 67.2.
    base = _numero_de_base(numero)
    return base is not None and any(_vise(d, base) for d in designations if "." not in d)


def _normalise(texte: Optional[str]) -> str:
    return re.sub(r"\s+", " ", texte or "").strip().casefold()


def _dans_la_section(lien: Lien, section: Optional[str]) -> bool:
    return lien.section is None or _normalise(section).startswith(_normalise(lien.section))


def _contient(contenu: Optional[str], repere: str) -> bool:
    return _normalise(repere) in _normalise(contenu)


def _article_range_ailleurs(lien: Lien, numero: Optional[str], contenu: Optional[str]) -> Optional[str]:
    """L'article touche que le morceau `numero` contient, si son repere y est."""
    if numero is None:
        return None
    for morceau, article, repere in lien.morceaux:
        if morceau == str(numero).strip() and _contient(contenu, repere):
            return article
    return None


def _excepte(lien: Lien, numero: Optional[str], contenu: Optional[str]) -> bool:
    return any(
        _numero_de_base(morceau) == _numero_de_base(numero) and _contient(contenu, repere)
        for morceau, repere in lien.sauf
    )


def _ou_le_trouver(lien: Lien) -> str:
    return (
        f" ({lien.par_reference})" if lien.par_reference
        else " (texte absent des documents de JuriX)"
    )


def _liste(articles: Tuple[str, ...]) -> str:
    if len(articles) == 1 and not _PLAGE.fullmatch(articles[0]):
        return f"article {articles[0]}"
    return f"articles {', '.join(articles)}"


def _suite(lien: Lien, consequence: str = "") -> str:
    morceaux = []
    if lien.portee:
        morceaux.append(f" : {lien.portee}.")
    elif consequence:
        morceaux.append(".")
    if consequence:
        morceaux.append(f" {consequence}")
    return "".join(morceaux)


class LigneDeStatut(NamedTuple):
    """Une ligne, en deux temps : ce qui l'identifie, puis le detail."""

    tete: str
    suite: str

    def __str__(self) -> str:
        return self.tete + self.suite


def _statut(lien: Lien, article_contenu: Optional[str] = None) -> LigneDeStatut:
    nature = NATURES[lien.nature]
    vise = "article" if lien.articles else "texte"
    consequence = CONSEQUENCES[(lien.nature, vise)]
    if article_contenu:
        sujet = f"cet extrait contient tout ou partie de l'article {article_contenu},"
        consequence += " Le reste de l'extrait n'est pas concerné."
    else:
        sujet = vise
    return LigneDeStatut(
        f"Statut : {sujet} {nature.participe} par {lien.par_titre}{_ou_le_trouver(lien)}",
        _suite(lien, consequence),
    )


# Une ligne « niveau loi » porte tout le texte : elle nomme les textes plus
# recents qui en touchent des articles, sans leur portee. Le Code civil en
# compte une trentaine : detaillees, ces lignes pesaient 18 000 caracteres,
# pour des articles que l'extrait, tronque, ne montrait meme pas.
_TEXTES_NOMMES_AU_PLUS = 4


def _statut_du_texte_entier(liens: List[Lien]) -> LigneDeStatut:
    noms = []
    for lien in sorted(liens, key=lambda lien: lien.par_date, reverse=True):
        nom = f"{lien.par_titre}{_ou_le_trouver(lien)}"
        if nom not in noms:
            noms.append(nom)
    reste = len(noms) - _TEXTES_NOMMES_AU_PLUS
    suite = " : " + " ; ".join(noms[:_TEXTES_NOMMES_AU_PLUS])
    if reste > 0:
        suite += f" ; et {reste} autre{'s' if reste > 1 else ''}"
    return LigneDeStatut(
        "Statut : des articles de ce texte ont été modifiés ou abrogés depuis", suite + "."
    )


def _ce_que_ce_texte_change(lien: Lien) -> LigneDeStatut:
    cible = lien.texte_nom
    # Les articles d'un titre de la compilation du Code civil suivent souvent
    # la numerotation d'un Acte uniforme : sans le titre, « articles 3 à 27 »
    # semblerait designer ceux du Code.
    if lien.section and "titre" not in cible.lower():
        cible += f" ({lien.section.rstrip('. ')})"
    if lien.articles:
        cible += f", {_liste(lien.articles)}"
    return LigneDeStatut(f"Ce texte {NATURES[lien.nature].verbe} {cible}", _suite(lien))


def lignes_en_deux(
    reference: Optional[str],
    numero: Optional[str],
    contenu: Optional[str] = None,
    section: Optional[str] = None,
    liens: Iterable[Lien] = None,
) -> List[LigneDeStatut]:
    """
    Les lignes a placer sous l'en-tete d'un extrait.

    Pour un extrait d'un texte touche : son statut, avec le texte qui le
    touche. Pour un extrait du texte qui touche : ce qu'il change. Le modele
    voit ainsi le lien des deux cotes, que la recherche ait remonte l'ancien
    article, le nouveau, ou les deux. `numero` None : une ligne « niveau loi »,
    dont le contenu est le texte entier.
    """
    if not reference:
        return []
    statuts: List[Tuple[Lien, LigneDeStatut]] = []
    articles_touches: List[Lien] = []
    inverses: List[LigneDeStatut] = []
    for lien in LIENS if liens is None else liens:
        if lien.texte == reference:
            if numero is None:
                if lien.articles:
                    articles_touches.append(lien)
                else:
                    statuts.append((lien, _statut(lien)))
            elif not _dans_la_section(lien, section):
                continue
            elif _designe(lien.articles, numero) and not _excepte(lien, numero, contenu):
                statuts.append((lien, _statut(lien)))
            elif (article := _article_range_ailleurs(lien, numero, contenu)) is not None:
                statuts.append((lien, _statut(lien, article)))
        elif lien.par_reference == reference:
            if numero is None or (
                str(numero).strip().upper() not in _HORS_DISPOSITIF
                and _designe(lien.par_articles, numero)
            ):
                inverses.append(_ce_que_ce_texte_change(lien))
    fin = _derniere_fin(lien for lien, _ in statuts)
    if fin is not None:
        statuts = [(lien, ligne) for lien, ligne in statuts if lien.par_date >= fin]
        articles_touches = [lien for lien in articles_touches if lien.par_date >= fin]
    lignes = [ligne for _, ligne in statuts]
    if articles_touches:
        lignes.append(_statut_du_texte_entier(articles_touches))
    return lignes + inverses


def _derniere_fin(liens: Iterable[Lien]) -> Optional[date]:
    """
    Date du dernier texte qui abroge ou remplace l'extrait. Ce qui l'avait
    modifie avant n'a plus rien a dire : sous un article de la loi 2013/004,
    « modifié par la loi de 2017, ce texte fait foi » contredisait « remplacé
    par l'ordonnance de 2025, le texte ne s'applique plus ».
    """
    fins = [lien.par_date for lien in liens if lien.nature in ("abroge", "remplace")]
    return max(fins) if fins else None


def lignes_de_statut(
    reference: Optional[str],
    numero: Optional[str],
    contenu: Optional[str] = None,
    section: Optional[str] = None,
    liens: Iterable[Lien] = None,
) -> List[str]:
    """`lignes_en_deux`, chaque ligne d'un tenant."""
    return [str(ligne) for ligne in lignes_en_deux(reference, numero, contenu, section, liens)]


def abreger_les_repetitions(
    lignes: List[LigneDeStatut], deja_vues: Dict[str, int], index: int
) -> List[str]:
    """
    Une ligne deja donnee sous un extrait precedent n'est repetee qu'en tete :
    huit extraits d'un meme traite portaient huit fois le meme paragraphe.
    """
    rendues = []
    for ligne in lignes:
        complete = str(ligne)
        if ligne.suite and complete in deja_vues:
            rendues.append(f"{ligne.tete} (détails sous l'extrait [{deja_vues[complete]}])")
        else:
            deja_vues.setdefault(complete, index)
            rendues.append(complete)
    return rendues
