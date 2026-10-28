"""
Recherche affinee : filtres, tri et effectifs calcules par le serveur, sur
TOUT l'ensemble candidat d'une requete, avant la pagination.

POURQUOI. Le front filtrait par nature et triait par date dans le navigateur,
sur la seule page affichee : des la page 2, effectifs et ordre etaient faux,
et aucun filtre n'affichait d'effectif. Filtrer une page ne dit rien des
autres.

L'ENSEMBLE CANDIDAT. La recherche ne classe pas tout le corpus : elle classe
un nombre borne de morceaux (titres, articles en plein texte, plus proches
voisins semantiques), fusionnes puis regroupes par loi. Le parcours classique
recalculait cet ensemble a chaque page, avec un budget qui grandit avec
l'offset : le total et l'ordre changeaient donc d'une page a l'autre. Ici,
l'ensemble est calcule UNE fois par requete, avec un budget fixe (voir
SearchService.FENETRE_ENSEMBLE), sans dependre ni de la page, ni du tri, ni
des filtres de ce module. Ce qu'il contient est exactement ce que la
pagination parcourt.

FILTRES PUIS PAGE. Les filtres de `refine` s'appliquent a l'ensemble entier ;
le tri ensuite ; la page en dernier. Les effectifs suivent la regle des
facettes disjonctives : l'effectif d'une valeur tient compte de tous les
filtres actifs SAUF celui de sa propre facette. Cocher « Lois (12) » donne
donc 12 resultats, et les autres onglets gardent leurs effectifs. Les
`filters` historiques de SearchRequest, eux, restent des filtres SQL : ils
delimitent l'ensemble candidat lui-meme, comme avant.

Fonctions pures, sans base : testees sans PostgreSQL.
"""

import hashlib
import json
import re
import unicodedata
from collections import Counter
from datetime import date
from typing import Dict, List, Optional, Sequence, Tuple

from pydantic import BaseModel, Field

from app.schemas.search import (
    FacetteAnnee,
    FacetteCategorie,
    FacetteValeur,
    SearchFacets,
    SearchFilters,
    SearchRefine,
    SearchResult,
)
from app.services.text_features import STOPWORDS

TRIS = ("pertinence", "date_desc", "date_asc")
TRI_PAR_DEFAUT = "pertinence"


class EnsembleCandidat(BaseModel):
    """
    L'ensemble candidat d'une requete, dans l'ordre de pertinence, tel que
    query_cache le garde : les lois classees (contenu reduit a l'apercu) et la
    navigation directe, qui ne dependent que de la requete.
    """

    resultats: List[SearchResult] = Field(default_factory=list)
    target_law_id: Optional[int] = None
    target_article: Optional[str] = None
    direct_navigation: bool = False


def replier_nature(valeur: Optional[str]) -> str:
    """
    « Décret », « decret », « acte_uniforme » et « Acte uniforme » se
    comparent : les natures arrivent par une URL (`?nature=decret`), sans
    accents ni espaces, et sont rangees en base avec (« acte uniforme »).
    """
    if not valeur:
        return ""
    decompose = unicodedata.normalize("NFKD", valeur.lower())
    sans_accents = "".join(c for c in decompose if not unicodedata.combining(c))
    return re.sub(r"[\s_-]+", " ", sans_accents).strip()


# ==================== FILTRES ====================


def _dans_la_periode(jour: Optional[date], refine: SearchRefine) -> bool:
    if refine.year_from is None and refine.year_to is None and (
        refine.date_from is None and refine.date_to is None
    ):
        return True
    # Un texte sans date ne peut pas etre dit « de la periode ».
    if jour is None:
        return False
    if refine.year_from is not None and jour.year < refine.year_from:
        return False
    if refine.year_to is not None and jour.year > refine.year_to:
        return False
    if refine.date_from is not None and jour < refine.date_from:
        return False
    if refine.date_to is not None and jour > refine.date_to:
        return False
    return True


def retenu(resultat: SearchResult, refine: Optional[SearchRefine], sauf: str = "") -> bool:
    """
    Vrai si le resultat passe les filtres de `refine`, celui de la facette
    `sauf` excepte (« types », « categories » ou « dates »).
    """
    if refine is None:
        return True
    if sauf != "types" and refine.types:
        if replier_nature(resultat.type) not in {replier_nature(t) for t in refine.types}:
            return False
    if sauf != "categories" and refine.category_ids:
        if resultat.category_id not in refine.category_ids:
            return False
    if sauf != "dates" and not _dans_la_periode(resultat.publication_date, refine):
        return False
    return True


# ==================== EFFECTIFS ====================


def effectifs(resultats: Sequence[SearchResult], refine: Optional[SearchRefine]) -> SearchFacets:
    """
    Effectifs par nature, par domaine et par annee, sur l'ensemble candidat.

    Exhaustifs : chaque resultat qui passe les autres filtres est compte une
    fois dans chaque facette, y compris sans domaine (id null) ou sans date
    (annee null). Une valeur absente d'une facette a un effectif nul.
    """
    natures: Counter = Counter()
    for r in resultats:
        if retenu(r, refine, sauf="types"):
            natures[r.type] += 1
    domaines: Counter = Counter()
    noms: Dict[Optional[int], Optional[str]] = {}
    for r in resultats:
        if retenu(r, refine, sauf="categories"):
            domaines[r.category_id] += 1
            noms.setdefault(r.category_id, r.category_name)
    annees: Counter = Counter()
    for r in resultats:
        if retenu(r, refine, sauf="dates"):
            annees[r.publication_date.year if r.publication_date else None] += 1

    return SearchFacets(
        # Effectif decroissant, puis libelle : deux appels identiques rendent
        # le meme ordre.
        types=[
            FacetteValeur(value=v, count=n)
            for v, n in sorted(natures.items(), key=lambda kv: (-kv[1], kv[0]))
        ],
        categories=[
            FacetteCategorie(id=c, name=noms.get(c), count=n)
            for c, n in sorted(
                domaines.items(), key=lambda kv: (-kv[1], noms.get(kv[0]) or "", kv[0] or 0)
            )
        ],
        # Annee decroissante, l'annee inconnue a la fin.
        years=[
            FacetteAnnee(value=a, count=n)
            for a, n in sorted(annees.items(), key=lambda kv: (kv[0] is None, -(kv[0] or 0)))
        ],
    )


# ==================== TRI ET PAGE ====================


def trier(resultats: List[SearchResult], tri: str) -> List[SearchResult]:
    """
    Tri stable : a date egale, l'ordre de pertinence est garde. Les textes sans
    date passent apres les autres, dans les deux sens.
    """
    if tri == "date_desc":
        return sorted(
            resultats,
            key=lambda r: (r.publication_date is None, -(r.publication_date or date.min).toordinal()),
        )
    if tri == "date_asc":
        return sorted(
            resultats,
            key=lambda r: (r.publication_date is None, (r.publication_date or date.min).toordinal()),
        )
    return list(resultats)


def affiner(
    ensemble: Sequence[SearchResult],
    refine: Optional[SearchRefine],
    tri: Optional[str],
    offset: int,
    limit: int,
) -> Tuple[List[SearchResult], int, SearchFacets]:
    """
    Filtre l'ensemble candidat, le trie, puis decoupe la page.

    Returns:
        (la page, le total apres filtres, les effectifs)
    """
    retenus = [r for r in ensemble if retenu(r, refine)]
    retenus = trier(retenus, tri or TRI_PAR_DEFAUT)
    return retenus[offset:offset + limit], len(retenus), effectifs(ensemble, refine)


# ==================== CLE DE CACHE DE L'ENSEMBLE ====================


def cle_de_l_ensemble(
    version: str,
    query: str,
    mode: str,
    semantic_query: Optional[str],
    filters: Optional[SearchFilters],
    fenetre: int,
) -> str:
    """
    Cle de query_cache de l'ensemble candidat.

    Ni la page, ni le tri, ni `refine` n'y entrent : changer d'onglet ou de
    page relit l'ensemble au lieu de relancer la recherche. Les `filters`
    historiques et la fenetre, si : ils changent l'ensemble lui-meme.
    """
    brut = json.dumps(
        {
            "v": version,
            "ensemble": fenetre,
            "mode": mode,
            "q": query,
            "semantique": semantic_query if mode != "text" else None,
            "filtres": filters.model_dump(mode="json") if filters else None,
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(brut.encode()).hexdigest()[:64]


# ==================== EXTRAITS SURLIGNES ====================

# Taille de l'extrait rendu, du meme ordre que ts_headline (deux fragments de
# 20 a 40 mots) ; au-dela du debut, un texte de loi n'est plus lu pour
# chercher le passage : un article tient en quelques milliers de caracteres.
LONGUEUR_EXTRAIT = 300
LECTURE_MAX = 20_000
_MOTS_VIDES = {
    "".join(c for c in unicodedata.normalize("NFKD", m) if not unicodedata.combining(c))
    for m in STOPWORDS
}


def _replier(mot: str) -> str:
    decompose = unicodedata.normalize("NFKD", mot.lower())
    return "".join(c for c in decompose if not unicodedata.combining(c))


def _table_de_repli() -> Dict[int, str]:
    """
    Minuscule sans accent, caractere pour caractere : « É » → « e ».

    str.translate garde la longueur du texte, donc les positions : on cherche
    dans le texte replie et on marque le texte d'origine aux memes indices.
    Replier mot a mot (_replier) coutait 5,7 ms sur les 20 000 premiers
    caracteres du Code du travail, meme sans aucune correspondance ; ici, une
    seule passe en C. Seuls les caracteres qui se replient en UN caractere
    sont dans la table (« œ » reste « œ », comme avec _replier).
    """
    table: Dict[int, str] = {}
    for code in range(0x41, 0x250):
        replie = _replier(chr(code))
        if len(replie) == 1 and replie != chr(code):
            table[code] = replie
    return table


_REPLI = _table_de_repli()


def racines_de_la_requete(requete: str) -> List[str]:
    """
    Les racines a surligner : chaque mot de la requete, replie, sans mots vides
    ni mot exclu (« -mot »), tronque pour couvrir ses variantes.

    Pas de racinisation francaise disponible ici : une troncature a
    max(6, n - 4) lettres la remplace. « licenciement » donne « licencie »
    (licencié, licenciements), « indemnité » « indemn » (indemnités) ;
    « commercial » donne « commer », qui evite « comme ». Un mot de six
    lettres ou moins est garde entier.
    """
    racines: List[str] = []
    for brut in re.finditer(r"(-?)(\w+)", requete):
        if brut.group(1):
            continue
        mot = _replier(brut.group(2))
        if len(mot) < 3 or mot in _MOTS_VIDES or mot.isdigit():
            continue
        racine = mot[: max(6, len(mot) - 4)] if len(mot) > 6 else mot
        if racine not in racines:
            racines.append(racine)
    return racines


def surligner(contenu: Optional[str], requete: str, longueur: int = LONGUEUR_EXTRAIT) -> Optional[str]:
    """
    Un extrait du contenu autour des mots de la requete, marques de <mark>,
    au format des extraits de ts_headline (texte brut et balises <mark> ; le
    front echappe tout le reste). None si aucun mot n'y figure : l'extrait
    existant reste alors tel quel.

    La fenetre retenue est celle qui contient le plus de racines DIFFERENTES :
    « indemnité de licenciement » prefere le passage qui porte les deux mots a
    celui qui repete l'un des deux.
    """
    if not contenu:
        return None
    racines = racines_de_la_requete(requete)
    if not racines:
        return None
    texte = contenu[:LECTURE_MAX]
    # Un mot entier qui COMMENCE par une racine : « licencie » marque
    # « licenciement ». Cherche dans le texte replie, de meme longueur, puis
    # marque aux memes indices dans le texte d'origine. L'alternance garde
    # l'ordre des racines : la premiere qui convient l'emporte.
    motif = re.compile(
        r"(?<!\w)(?:" + "|".join(f"({re.escape(racine)})" for racine in racines) + r")\w*"
    )
    # (debut, fin, indice de racine)
    trouves = [(m.start(), m.end(), m.lastindex - 1) for m in motif.finditer(texte.translate(_REPLI))]
    if not trouves:
        return None

    # Fenetre qui couvre le plus de racines distinctes, puis le plus de mots.
    amorce = longueur // 5
    meilleure, score_max = 0, (-1, -1)
    for debut, _, _ in trouves:
        depart = max(0, debut - amorce)
        dedans = [t for t in trouves if t[0] >= depart and t[1] <= depart + longueur]
        score = (len({t[2] for t in dedans}), len(dedans))
        if score > score_max:
            meilleure, score_max = depart, score
    depart = meilleure
    fin = min(len(texte), depart + longueur)
    # Pas de mot coupe aux bords.
    if depart > 0:
        espace = texte.find(" ", depart)
        depart = espace + 1 if 0 <= espace < fin else depart
    if fin < len(texte):
        espace = texte.rfind(" ", depart, fin)
        fin = espace if espace > depart else fin

    morceaux: List[str] = []
    curseur = depart
    for d, f, _ in trouves:
        if d < depart or f > fin:
            continue
        morceaux.append(texte[curseur:d])
        morceaux.append(f"<mark>{texte[d:f]}</mark>")
        curseur = f
    morceaux.append(texte[curseur:fin])
    extrait = "".join(morceaux).strip()
    if depart > 0:
        extrait = "… " + extrait
    if fin < len(contenu):
        extrait += " …"
    return extrait
