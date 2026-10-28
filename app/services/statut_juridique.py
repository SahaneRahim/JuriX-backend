"""
Le statut juridique d'un texte : ce que d'autres textes lui ont fait, et ce
qu'il a fait a d'autres. Sert la page de lecture (GET /laws/{id}) et l'apercu
d'un article (GET /laws/{id}/articles/{numero}).

POURQUOI. La page de lecture affichait « EN VIGUEUR » sous chaque article, sans
aucune donnee derriere ; le chat, lui, connaissait les 67 liens prouves du
releve. Presenter comme applicable un texte abroge est l'erreur la plus grave
pour un service juridique. La page dit desormais ce qui est PROUVE, date du
releve, et rien d'autre.

D'OU VIENT LE STATUT. Des liens de liens_entre_textes : LIENS (l'etat actuel,
que suit le chat) et LIENS_HISTORIQUES (liens anterieurs, depasses depuis ou
deja integres au texte affiche). Rien n'est deduit des dates ni devine des
intitules : un texte sans lien est un texte dont JuriX n'a RELEVE aucune
modification, pas un texte intact.

LA MEME REGLE QUE LE CHAT. Un article est marque par
textes_successifs.morceau_touche, la regle des lignes de statut : sections
d'une compilation, article range sous un autre numero, exceptions. La page et
le chat ne peuvent pas se contredire sur un meme morceau.

COUT. Le releve est en memoire. Les textes lies (une soixantaine) sont lus en
UNE requete, gardee dix minutes ; un texte sans lien ne coute aucune requete.
Seuls les morceaux dont la decision depend d'un repere (`morceaux`, `sauf`)
sont relus, jamais le texte entier. La regle n'est appliquee qu'aux morceaux
qu'une liste d'articles peut designer (_IndexDesMorceaux) : quelques
millisecondes pour le Code civil, au lieu de 450.
"""

import logging
import re
import time
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Set, Tuple

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.law import Article, Law
from app.schemas.statut import (
    Changement,
    StatutDeLArticle,
    StatutJuridique,
    TexteLie,
    VersionEnBase,
)
from app.services.article_reference import normalize_number
from app.services.liens_entre_textes import (
    LIENS,
    LIENS_HISTORIQUES,
    RELEVE_LE,
    Lien,
    LienHistorique,
)
from app.services.reperes_textes import (
    dates_de_l_acte,
    intitule_sans_article,
    nature_depuis_intitule,
    numero_de_l_acte,
)
from app.services.textes_successifs import (
    NATURES,
    _contient,
    _numero_de_base,
    morceau_touche,
    porte_le_changement,
    reperes_a_lire,
)
from app.utils.chunk_refiner import est_pseudo_numero
from app.utils.titre_officiel import numero_du_titre

logger = logging.getLogger(__name__)

# Ce qui met fin a ce qu'un texte plus ancien avait ecrit : meme regle que
# textes_successifs._derniere_fin, qui fait taire, sous un extrait, ce qui
# l'avait modifie avant son abrogation.
_FINS = ("abroge", "remplace")

_ORDRE_DES_TYPES = ("abroge", "remplace", "abroge_partiellement", "modifie", "complete")

_LIBELLES = {
    "reecrit_en_entier": "réécrit en entier",
    **{nature: valeurs.participe for nature, valeurs in NATURES.items()},
}

# « réécrit en entier ; … », « réécrits en entier ; … » : la portee du releve le
# dit en tete. Ailleurs (« seul l'alinéa 4 est réécrit »), elle dit l'etendue
# elle-meme, et nature_modification reste « modifie ».
_REECRIT_EN_ENTIER = re.compile(r"^\s*réécrits?\s+en\s+entier\b", re.IGNORECASE)

# « les articles modifiés ne sont pas connus » : un lien sans liste d'articles
# n'est pas toujours un lien sur tout le texte.
_ARTICLES_INCONNUS = re.compile(r"ne\s+sont\s+pas\s+connus", re.IGNORECASE)

_PLAGE = re.compile(r"(\d+)\s*à\s*(\d+)")
# Une plage plus large serait une faute de saisie, pas un titre de code.
_PLAGE_MAX = 3000


class InfoTexte(NamedTuple):
    """Ce que la base dit d'un texte nomme par un lien."""

    id: int
    reference: str
    type: str
    titre: str
    date: Optional[date]
    status: str


class MorceauDuTexte(NamedTuple):
    """Un article en base, tel que le statut le lit. `contenu` None : non charge."""

    id: int
    numero: Optional[str]
    section: Optional[str]
    contenu: Optional[str] = None


@dataclass(frozen=True)
class _Entree:
    """Un lien vu depuis un texte."""

    lien: Lien
    sens: str
    historique: Optional[LienHistorique]


# ============================================================================
# Les liens d'un texte
# ============================================================================


def _entrees(
    reference: str,
    liens: Iterable[Lien],
    historiques: Iterable[LienHistorique],
) -> List[_Entree]:
    entrees = []
    for lien, historique in [(lien, None) for lien in liens] + [(h.lien, h) for h in historiques]:
        if lien.texte == reference:
            entrees.append(_Entree(lien, "modifie_par", historique))
        elif lien.par_reference and lien.par_reference == reference:
            entrees.append(_Entree(lien, "modifie", historique))
    return entrees


def _bases(designations: Sequence[str]) -> Optional[Set[str]]:
    """Numeros de base designes, plages developpees ; None : tout le texte."""
    if not designations:
        return None
    bases: Set[str] = set()
    for designation in designations:
        plage = _PLAGE.fullmatch(designation.strip())
        if plage and int(plage.group(2)) - int(plage.group(1)) <= _PLAGE_MAX:
            bases.update(str(n) for n in range(int(plage.group(1)), int(plage.group(2)) + 1))
        else:
            base = _numero_de_base(designation)
            if base:
                bases.add(base)
    return bases


def _depasse_par(lien: Lien, liens: Sequence[Lien]) -> Optional[Lien]:
    """
    Le dernier lien de LIENS qui abroge ou remplace, depuis, tout ce que `lien`
    avait touche. La loi 2017/015 reecrit des articles de la loi 2013/004, que
    l'ordonnance 2025/002 remplace en entier : ses reecritures relevent de
    l'historique.
    """
    candidats = []
    touches = _bases(lien.articles)
    for autre in liens:
        if (
            autre is lien
            or autre.texte != lien.texte
            or autre.nature not in _FINS
            or autre.par_date <= lien.par_date
        ):
            continue
        if autre.section is not None and autre.section != lien.section:
            continue
        couverts = _bases(autre.articles)
        if couverts is None or (touches is not None and touches <= couverts):
            candidats.append(autre)
    return max(candidats, key=lambda autre: autre.par_date) if candidats else None


# ============================================================================
# Construction du statut
# ============================================================================


def _cle_de_base(base: Optional[str]) -> Optional[str]:
    """« 0012 » et « 12 » sous la meme cle : l'index rend un peu trop, la regle tranche."""
    return str(int(base)) if base and base.isdigit() else base


class _IndexDesMorceaux:
    """
    Les morceaux d'un texte, retrouves par leur numero exact ou par leur numero
    de base.

    POURQUOI. Le Code civil compte 2 134 morceaux et 29 liens : passer chaque
    lien sur chaque morceau (morceau_touche normalise a chaque appel les deux
    intitules de section compares) coutait 450 ms a chaque affichage de la
    page. L'index ne rend que les morceaux qu'une liste d'articles PEUT
    designer, un sur-ensemble ; la decision reste a la regle du chat
    (morceau_touche, porte_le_changement, reperes_a_lire), appliquee a ces
    seuls morceaux. Un test verifie sur tous les liens du releve qu'il ne
    perd rien.
    """

    def __init__(self, morceaux: Sequence[MorceauDuTexte]):
        self.morceaux = list(morceaux)
        self._par_numero: Dict[str, List[int]] = {}
        self._par_base: Dict[Optional[str], List[int]] = {}
        self._sans_numero: List[int] = []
        for rang, morceau in enumerate(self.morceaux):
            if morceau.numero is None:
                self._sans_numero.append(rang)
                continue
            numero = str(morceau.numero).strip()
            self._par_numero.setdefault(numero, []).append(rang)
            self._par_base.setdefault(_cle_de_base(_numero_de_base(numero)), []).append(rang)

    def _rendre(self, rangs: Set[int]) -> List[MorceauDuTexte]:
        # Un morceau sans numero n'est designe par aucune liste, mais la regle
        # l'accepte parfois (porte_le_changement le traite en ligne « niveau
        # loi ») : il reste candidat, et la regle decide. Ordre du texte.
        return [self.morceaux[rang] for rang in sorted(rangs.union(self._sans_numero))]

    def designes(self, designations: Sequence[str], exacts: Iterable[str] = ()) -> List[MorceauDuTexte]:
        """
        Les morceaux que `designations` peut designer selon _designe (numero
        tel quel, numero de base, plage « 34 à 101 »), plus ceux dont le numero
        est dans `exacts` (les `morceaux` d'un lien). Sans designation : tous.
        """
        if not designations:
            return list(self.morceaux)
        rangs: Set[int] = set()
        for numero in list(designations) + list(exacts):
            rangs.update(self._par_numero.get(numero, ()))
        for designation in designations:
            # « 67.3 » ne designe que ce morceau-la (_designe) : deja vu ci-dessus.
            if "." in designation:
                continue
            plage = _PLAGE.fullmatch(designation.strip())
            if plage:
                debut, fin = int(plage.group(1)), int(plage.group(2))
                if fin - debut > _PLAGE_MAX:
                    return list(self.morceaux)
                for numero in range(debut, fin + 1):
                    rangs.update(self._par_base.get(str(numero), ()))
            else:
                base = _numero_de_base(designation)
                if base is not None:
                    rangs.update(self._par_base.get(_cle_de_base(base), ()))
        return self._rendre(rangs)

    def a_reperes(self, liens: Iterable[Lien]) -> List[MorceauDuTexte]:
        """Les morceaux que reperes_a_lire peut retenir : `morceaux` par numero, `sauf` par base."""
        rangs: Set[int] = set()
        for lien in liens:
            for numero, _, _ in lien.morceaux:
                rangs.update(self._par_numero.get(numero, ()))
            for numero, _ in lien.sauf:
                rangs.update(self._par_base.get(_cle_de_base(_numero_de_base(numero)), ()))
        return self._rendre(rangs)




def texte_lie(
    reference: Optional[str],
    intitule: str,
    jour: Optional[date],
    infos: Dict[str, InfoTexte],
) -> TexteLie:
    """
    Un texte nomme par un lien, decrit par la base s'il y est (`infos`), sinon
    par l'intitule et la date du releve.
    """
    info = infos.get(reference) if reference else None
    if info is not None:
        publie = info.status == "published"
        return TexteLie(
            id=info.id if publie else None,
            reference=info.reference,
            nature=info.type,
            # La lecture des lignes de liste (app/schemas/reperes.py) : un texte
            # du corpus porte le meme numero dans une liste et sous le statut
            # d'un autre. Elle repare aussi l'OCR (« N°202l/345 », « N1078 »).
            numero=numero_du_titre(info.titre),
            titre=info.titre,
            **dates_de_l_acte(info.date, info.titre)._asdict(),
            consultable=publie,
        )
    # Absent de la base : l'intitule du releve, ecrit pour une phrase, et la
    # date que le releve a lue dans le texte qui le cite. Jamais dans une liste,
    # il peut garder la lecture plus large de numero_de_l_acte en second.
    titre = intitule_sans_article(intitule)
    return TexteLie(
        nature=nature_depuis_intitule(titre),
        numero=numero_du_titre(titre) or numero_de_l_acte(titre),
        titre=titre,
        **dates_de_l_acte(jour, titre)._asdict(),
    )


def _nature_modification(lien: Lien) -> str:
    if lien.nature == "modifie" and _REECRIT_EN_ENTIER.match(lien.portee or ""):
        return "reecrit_en_entier"
    return lien.nature


def _etendue(lien: Lien) -> str:
    if lien.articles:
        return "articles"
    if _ARTICLES_INCONNUS.search(lien.portee or ""):
        return "inconnue"
    return "texte"


def cle_article(designation: Optional[str]) -> Optional[str]:
    """
    Cle de `par_article` : le numero normalise comme le fait le front
    (normaliserNumeroArticle) et article_reference.normalize_number, en
    majuscules, « 1er » -> « 1 », sans suffixe de morceau (« 55.2 » -> « 55 »).
    None pour un pseudo-numero (« SECTION_2 ») : ce n'est pas un article.
    """
    if not designation or est_pseudo_numero(designation):
        return None
    base = _numero_de_base(designation)
    return normalize_number(base) if base else None


def _cles(designations: Sequence[str]) -> List[str]:
    cles: List[str] = []
    for designation in designations:
        plage = _PLAGE.fullmatch(designation.strip())
        if plage and int(plage.group(2)) - int(plage.group(1)) <= _PLAGE_MAX:
            valeurs = [str(n) for n in range(int(plage.group(1)), int(plage.group(2)) + 1)]
        else:
            valeurs = [designation]
        for valeur in valeurs:
            cle = cle_article(valeur)
            if cle and cle not in cles:
                cles.append(cle)
    return cles


class _Construit(NamedTuple):
    statut: StatutJuridique
    changements: Dict[str, Changement]


def _construire(
    reference: Optional[str],
    infos: Dict[str, InfoTexte],
    morceaux: Sequence[MorceauDuTexte],
    liens: Sequence[Lien],
    historiques: Sequence[LienHistorique],
    releve_le: date,
) -> _Construit:
    entrees = _entrees(reference, liens, historiques) if reference else []

    # Ordre d'affichage : du texte le plus ancien au plus recent, comme se lit
    # un historique, et les changements d'un meme texte cote a cote (le front
    # en fait une ligne par texte). Les identifiants suivent cet ordre.
    groupes: Dict[Tuple[str, str], List[_Entree]] = {}
    for entree in entrees:
        if entree.sens == "modifie_par":
            cle = ("modifie_par", entree.lien.par_reference or entree.lien.par_titre)
        else:
            cle = ("modifie", entree.lien.texte)
        groupes.setdefault(cle, []).append(entree)

    def _date_du_groupe(cle_et_entrees) -> Tuple:
        (sens, cle), membres = cle_et_entrees
        if sens == "modifie_par":
            return (membres[0].lien.par_date, cle)
        info = infos.get(cle)
        return (info.date if info and info.date else date.min, cle)

    tous: Dict[str, Changement] = {}
    paires: List[Tuple[_Entree, Changement]] = []
    rendus = {"modifie_par": [], "modifie": []}
    compteur = 0
    for (sens, _), membres in sorted(groupes.items(), key=_date_du_groupe):
        changements = []
        for entree in membres:
            compteur += 1
            lien, historique = entree.lien, entree.historique
            if sens == "modifie_par":
                autre = texte_lie(lien.par_reference, lien.par_titre, lien.par_date, infos)
            else:
                autre = texte_lie(lien.texte, lien.texte_nom, None, infos)
            depasse_par = None
            if historique is not None and historique.depasse_par_titre:
                depasse_par = texte_lie(
                    historique.depasse_par_reference,
                    historique.depasse_par_titre,
                    historique.depasse_le,
                    infos,
                )
            elif historique is None and (fin := _depasse_par(lien, liens)) is not None:
                depasse_par = texte_lie(fin.par_reference, fin.par_titre, fin.par_date, infos)
            nature_modification = _nature_modification(lien)
            changement = Changement(
                id=f"c{compteur}",
                sens=sens,
                type=lien.nature,
                nature_modification=nature_modification,
                libelle=_LIBELLES[nature_modification],
                texte=autre,
                etendue=_etendue(lien),
                articles_cibles=list(lien.articles),
                section=lien.section,
                articles_source=list(lien.par_articles),
                portee=lien.portee or None,
                depasse=depasse_par is not None,
                depasse_par=depasse_par,
                integre_au_texte=bool(historique and historique.integre_au_texte),
            )
            changements.append(changement)
            tous[changement.id] = changement
            paires.append((entree, changement))
        rendus[sens].extend(changements)

    par_article: Dict[str, List[str]] = {}
    par_morceau: Dict[str, List[str]] = {}
    index = _IndexDesMorceaux(morceaux)
    for entree, changement in paires:
        lien = entree.lien
        if entree.sens == "modifie_par":
            if not lien.articles:
                # Tout le texte, ou des articles inconnus : la page l'affiche en
                # tete, pas sous chacun des articles.
                continue
            cles = _cles(lien.articles)
            candidats = index.designes(lien.articles, [numero for numero, _, _ in lien.morceaux])
            vises = [m for m in candidats if morceau_touche(lien, m.numero, m.contenu, m.section)[0]]
        else:
            if not lien.par_articles:
                continue
            cles = _cles(lien.par_articles)
            vises = [m for m in index.designes(lien.par_articles) if porte_le_changement(lien, m.numero)]
        for cle in cles:
            par_article.setdefault(cle, []).append(changement.id)
        for morceau in vises:
            par_morceau.setdefault(str(morceau.id), []).append(changement.id)

    actuels = [c for c in rendus["modifie_par"] if not c.depasse and not c.integre_au_texte]
    fins = [c for c in actuels if c.etendue == "texte" and c.type in _FINS]
    if fins:
        # La date du lien, et non celle du texte affiche : date_signature est
        # nulle pour un 1er janvier de remplissage ou une date de Journal
        # officiel, et l'ordre en serait faux.
        date_du_lien = {changement.id: entree.lien.par_date for entree, changement in paires}
        derniere = max(fins, key=lambda c: date_du_lien[c.id])
        etat, abroge_par = derniere.type, derniere.texte
    else:
        etat, abroge_par = ("modifie" if actuels else "aucune_modification_connue"), None
    types = [t for t in _ORDRE_DES_TYPES if any(c.type == t for c in actuels)]

    integres: List[TexteLie] = []
    for changement in rendus["modifie_par"]:
        if changement.integre_au_texte and changement.texte not in integres:
            integres.append(changement.texte)

    statut = StatutJuridique(
        releve_le=releve_le,
        etat=etat,
        types=types,
        abroge=abroge_par is not None,
        abroge_par=abroge_par,
        version_en_base=VersionEnBase(textes_integres=integres) if integres else None,
        modifie_par=rendus["modifie_par"],
        modifie=rendus["modifie"],
        par_article=par_article,
        par_morceau=par_morceau,
    )
    return _Construit(statut, tous)


def construire_statut(
    reference: Optional[str],
    infos: Dict[str, InfoTexte],
    morceaux: Sequence[MorceauDuTexte] = (),
    liens: Sequence[Lien] = LIENS,
    historiques: Sequence[LienHistorique] = LIENS_HISTORIQUES,
    releve_le: date = RELEVE_LE,
) -> StatutJuridique:
    """Le statut d'un texte. Fonction pure : la base est lue par l'appelant."""
    return _construire(reference, infos, morceaux, liens, historiques, releve_le).statut


def construire_statut_d_un_article(
    reference: Optional[str],
    infos: Dict[str, InfoTexte],
    morceaux: Sequence[MorceauDuTexte],
    liens: Sequence[Lien] = LIENS,
    historiques: Sequence[LienHistorique] = LIENS_HISTORIQUES,
    releve_le: date = RELEVE_LE,
) -> StatutDeLArticle:
    """
    Le statut d'un article, compose de `morceaux` (contenu charge).

    Les changements qui le touchent ou qu'il porte, plus ceux qui portent sur
    tout le texte : sous un article de la loi 2013/004, « remplacée par
    l'ordonnance 2025/002 » doit se lire meme si aucune liste ne le nomme.
    """
    construit = _construire(reference, infos, morceaux, liens, historiques, releve_le)
    statut = construit.statut
    retenus = [c.id for c in statut.modifie_par if c.etendue != "articles"]
    for morceau in morceaux:
        for identifiant in statut.par_morceau.get(str(morceau.id), []):
            if identifiant not in retenus:
                retenus.append(identifiant)
    ordre = list(construit.changements)
    return StatutDeLArticle(
        releve_le=statut.releve_le,
        etat_du_texte=statut.etat,
        version_en_base=statut.version_en_base,
        changements=[construit.changements[i] for i in sorted(retenus, key=ordre.index)],
    )


# ============================================================================
# Lectures en base
# ============================================================================

# Dix minutes, comme la liste des categories (app/api/routes/categories.py) :
# les textes lies ne changent qu'a l'ingestion. Un seul processus sert l'API.
_DUREE_CACHE_S = 600.0
_cache_infos: Optional[Tuple[float, Dict[str, InfoTexte]]] = None


def invalider_cache_statut() -> None:
    """Oublie les textes lies en cache : la prochaine lecture repasse par la base."""
    global _cache_infos
    _cache_infos = None


def _references_liees() -> List[str]:
    references: Set[str] = set()
    for lien in list(LIENS) + [h.lien for h in LIENS_HISTORIQUES]:
        references.add(lien.texte)
        if lien.par_reference:
            references.add(lien.par_reference)
    references.update(h.depasse_par_reference for h in LIENS_HISTORIQUES if h.depasse_par_reference)
    return sorted(references)


def reference_liee(reference: Optional[str]) -> bool:
    """Le texte figure-t-il dans un lien ? Sinon, son statut ne coute aucune lecture."""
    return bool(reference) and reference in _REFERENCES_LIEES


_REFERENCES_LIEES = frozenset(_references_liees())


async def infos_des_textes_lies(db: AsyncSession) -> Dict[str, InfoTexte]:
    """Les textes nommes par un lien, en UNE requete, gardes dix minutes."""
    global _cache_infos
    entree = _cache_infos
    maintenant = time.monotonic()
    if entree is not None and maintenant - entree[0] < _DUREE_CACHE_S:
        return entree[1]
    lignes = (
        await db.execute(
            select(Law.id, Law.reference, Law.type, Law.title, Law.publication_date, Law.status)
            .where(Law.reference.in_(_references_liees()))
        )
    ).all()
    infos = {
        ligne.reference: InfoTexte(
            ligne.id, ligne.reference, ligne.type, ligne.title, ligne.publication_date, ligne.status
        )
        for ligne in lignes
    }
    _cache_infos = (maintenant, infos)
    return infos


def morceaux_etablis(reference: Optional[str]) -> List[Tuple[str, str, str]]:
    """
    (morceau, article, repere) etablis par le releve pour ce texte : un article
    dont une partie est rangee sous un autre numero (la fin de l'article 6 de la
    Constitution est le morceau SECTION_2).
    """
    return [triplet for lien in _liens_du_texte(reference or "") for triplet in lien.morceaux]


def contient_le_repere(contenu: Optional[str], repere: str) -> bool:
    """Meme comparaison que les lignes de statut : casse et espaces ignores."""
    return _contient(contenu, repere)


def _liens_du_texte(reference: str) -> List[Lien]:
    return [
        lien for lien in list(LIENS) + [h.lien for h in LIENS_HISTORIQUES]
        if lien.texte == reference and lien.articles
    ]


async def contenus_a_lire(
    db: AsyncSession, reference: Optional[str], morceaux: Sequence[MorceauDuTexte]
) -> List[MorceauDuTexte]:
    """
    Les morceaux, avec le contenu de ceux dont la decision depend d'un repere.

    Une requete au plus, sur quelques lignes : jamais le texte entier.
    """
    if not reference:
        return list(morceaux)
    liens = _liens_du_texte(reference)
    a_lire = [
        m.id for m in _IndexDesMorceaux(morceaux).a_reperes(liens)
        if m.contenu is None and any(reperes_a_lire(lien, m.numero) for lien in liens)
    ]
    if not a_lire:
        return list(morceaux)
    contenus = dict(
        (await db.execute(select(Article.id, Article.content).where(Article.id.in_(a_lire)))).all()
    )
    return [m._replace(contenu=contenus.get(m.id, m.contenu)) for m in morceaux]


async def statut_du_texte(
    db: AsyncSession, reference: Optional[str], articles: Sequence
) -> StatutJuridique:
    """
    Le statut d'un texte pour GET /laws/{id}. `articles` : ses lignes `articles`
    (id, number, section), deja chargees pour le sommaire.
    """
    if not reference_liee(reference):
        return construire_statut(reference, {})
    infos = await infos_des_textes_lies(db)
    morceaux = [MorceauDuTexte(a.id, a.number, a.section) for a in articles]
    morceaux = await contenus_a_lire(db, reference, morceaux)
    return construire_statut(reference, infos, morceaux)


async def statut_d_un_article(
    db: AsyncSession, reference: Optional[str], morceaux: Sequence[MorceauDuTexte]
) -> StatutDeLArticle:
    """Le statut d'un article pour GET /laws/{id}/articles/{numero} ; `morceaux` avec contenu."""
    infos = await infos_des_textes_lies(db) if reference_liee(reference) else {}
    return construire_statut_d_un_article(reference, infos, morceaux)
