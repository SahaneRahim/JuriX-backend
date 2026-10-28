"""
Le statut prouve d'une source du chat : ce qu'un texte plus recent du corpus a
fait de l'article ou du texte cite, d'apres app/services/liens_entre_textes.py.

POURQUOI CE MODULE. Le modele lit deja ces liens, en lignes de statut sous
chaque extrait (textes_successifs.lignes_en_deux). L'interface, elle, ne
recevait rien : une source s'affichait sans dire qu'un texte plus recent avait
reecrit l'article cite. Ce module rend les MEMES liens, structures, pour
qu'elle puisse ecrire « Modifié par la Loi N°… du … » avec un lien vers ce
texte, et, quand aucun lien n'est connu, la mention neutre « version
d'origine, non consolidée » plutot qu'un « En vigueur » que rien ne prouve.

MEME SELECTION QUE LES LIGNES DU MODELE. La boucle reprend pas a pas le cote
« statut » de lignes_en_deux : section, articles designes, exceptions,
articles ranges sous un autre numero, puis l'oubli de ce qui precede la
derniere abrogation. Elle vit ici et non dans textes_successifs pour ne pas
remanier un module que d'autres chantiers touchent ; en contrepartie,
tests/test_services/test_statut_des_sources.py confronte les deux sur chaque
lien du corpus : l'interface ne doit jamais dire autre chose que ce que le
modele a lu.
"""

from typing import Dict, Iterable, List, Optional

from app.schemas.rag import LienProuve, StatutProuve
from app.services.liens_entre_textes import LIENS, Lien
from app.services.textes_successifs import (
    NATURES,
    _article_range_ailleurs,
    _dans_la_section,
    _derniere_fin,
    _designe,
    _excepte,
    lignes_de_statut,
)

# Du plus fort au plus faible : l'etat affiche est celui du lien le plus fort.
# Apres le filtre de _derniere_fin, une abrogation ou un remplacement ne
# cotoie plus que des liens posterieurs.
ORDRE_DES_ETATS = ("abroge", "remplace", "abroge_partiellement", "modifie", "complete")


def liens_qui_touchent(
    reference: Optional[str],
    numero: Optional[str],
    contenu: Optional[str] = None,
    section: Optional[str] = None,
    liens: Iterable[Lien] = None,
) -> List[Lien]:
    """
    Les textes plus recents qui touchent cet extrait, comme lignes_en_deux les
    choisit pour le modele.

    `numero` None : le texte entier. Tous ses liens sont rendus, ceux qui ne
    visent que certains articles compris (chacun porte sa liste `articles`) :
    une interface qui cite le texte doit pouvoir dire que des articles en ont
    change, comme la ligne « niveau loi » le dit au modele.
    """
    if not reference:
        return []
    retenus: List[Lien] = []
    for lien in LIENS if liens is None else liens:
        if lien.texte != reference:
            continue
        if numero is None:
            retenus.append(lien)
        elif not _dans_la_section(lien, section):
            continue
        elif _designe(lien.articles, numero) and not _excepte(lien, numero, contenu):
            retenus.append(lien)
        elif _article_range_ailleurs(lien, numero, contenu) is not None:
            retenus.append(lien)
    # Ce qui avait modifie l'extrait avant sa derniere abrogation n'a plus rien
    # a dire. Pour le texte entier, seule une abrogation du TEXTE fait date :
    # l'abrogation d'un article n'efface pas ce qui touche les autres.
    fin = _derniere_fin(
        lien for lien in retenus if numero is not None or not lien.articles
    )
    if fin is not None:
        retenus = [lien for lien in retenus if lien.par_date >= fin]
    return retenus


def etat_prouve(liens: List[Lien], numero: Optional[str]) -> Optional[str]:
    """
    L'etat le plus fort que les liens prouvent, ou None.

    Pour le texte entier, seuls comptent les liens qui le visent tout entier :
    « abrogé » sur un texte dont un seul article l'est serait faux.
    """
    natures = {lien.nature for lien in liens if numero is not None or not lien.articles}
    return next((nature for nature in ORDRE_DES_ETATS if nature in natures), None)


def references_a_resoudre(liens: Iterable[Lien]) -> List[str]:
    """Les references en base des textes qui touchent, pour leur identifiant."""
    return sorted({lien.par_reference for lien in liens if lien.par_reference})


def statut_prouve(
    reference: Optional[str],
    numero: Optional[str],
    contenu: Optional[str] = None,
    section: Optional[str] = None,
    ids: Optional[Dict[str, int]] = None,
) -> StatutProuve:
    """
    Le statut a joindre a une source : etat, liens et lignes du modele.

    `ids` (reference -> identifiant en base) sert le lien vers le texte qui
    touche ; un texte absent du corpus, ou une reference non resolue, garde
    `par_law_id` a None. Sans aucun lien, le statut est VIDE (etat None) : ce
    n'est pas « en vigueur », c'est « rien de prouve », et l'interface le dit.
    """
    ids = ids or {}
    liens = liens_qui_touchent(reference, numero, contenu, section)
    return StatutProuve(
        etat=etat_prouve(liens, numero),
        liens=[
            LienProuve(
                nature=lien.nature,
                libelle=NATURES[lien.nature].participe,
                par_titre=lien.par_titre,
                par_date=lien.par_date,
                par_reference=lien.par_reference,
                par_law_id=ids.get(lien.par_reference) if lien.par_reference else None,
                articles=list(lien.articles),
                par_articles=list(lien.par_articles),
                portee=lien.portee or None,
            )
            for lien in sorted(liens, key=lambda lien: lien.par_date, reverse=True)
        ],
        lignes=lignes_de_statut(reference, numero, contenu, section),
    )
