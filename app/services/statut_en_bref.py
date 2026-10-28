"""
Le statut d'un texte en une ligne de liste : combien de textes plus recents
l'ont modifie, complete, abroge ou remplace, et s'il est abroge en entier.

POURQUOI. Les listes (recherche, domaines) n'affirmaient rien du statut d'un
texte, ou pire, la page de lecture affichait « En vigueur » sans donnee
derriere. Seuls les liens PROUVES de app/services/liens_entre_textes.py disent
quelque chose : chacun porte la disposition qui le fonde. Cette ligne en est
le resume, assez leger pour chaque resultat : « Abrogé en partie par 11 textes
et complété par 1 ». Le detail (quels articles, par quel texte, avec quelle
portee) reste a la page du texte.

AUCUN LIEN CONNU NE VEUT PAS DIRE « EN VIGUEUR ». `statut_en_bref` rend alors
None : le releve couvre 23 textes du corpus, pas les 2 287. Le client doit
s'abstenir d'afficher un statut, ou dire « version d'origine, non consolidée ».

Les donnees sont lues dans liens_entre_textes.LIENS, sans y rien changer ; le
resume est calcule une fois, a l'import.

LE RESUME DE LA PAGE. La ligne resume ce que la page du texte (statut
juridique de GET /laws/{id}) donne pour ACTUEL : un lien depasse depuis par
une abrogation ou un remplacement n'y compte pas. La loi 2013/004 se lisait
« modifiée, complétée et remplacée » en liste, « remplacée » sur sa page, ou
les reecritures de 2017 sont de l'historique.
"""

from collections import defaultdict
from datetime import date
from typing import Dict, Iterable, Optional

from pydantic import BaseModel, Field

from app.services.liens_entre_textes import LIENS, Lien
from app.services.statut_juridique import lien_qui_depasse

# Natures qui mettent fin au texte entier quand le lien ne vise aucun article.
_FIN_DU_TEXTE = ("abroge", "remplace")


class StatutEnBref(BaseModel):
    """Resume des liens prouves qui touchent un texte, pour une ligne de liste."""

    abroge: bool = Field(
        ...,
        description=(
            "Vrai si un texte plus recent abroge ou remplace le texte ENTIER "
            "(lien prouve, sans liste d'articles) : le texte ne s'applique plus"
        ),
    )
    textes: int = Field(
        ..., ge=1, description="Nombre de textes plus recents, distincts, qui le touchent"
    )
    natures: Dict[str, int] = Field(
        ...,
        description=(
            "Nombre de textes distincts par nature de lien : modifie, complete, "
            "abroge, abroge_partiellement, remplace"
        ),
    )
    date_dernier_texte: date = Field(
        ..., description="Date du plus recent des textes qui le touchent"
    )


def _qui(lien: Lien) -> str:
    """Le texte qui touche, identifie par sa reference, ou son intitule s'il manque au corpus."""
    return lien.par_reference or lien.par_titre


def resumer(liens: Iterable[Lien]) -> Dict[str, StatutEnBref]:
    """Le resume de chaque texte touche, par reference en base, liens depasses exclus."""
    liens = list(liens)
    par_texte: Dict[str, list] = defaultdict(list)
    for lien in liens:
        if lien_qui_depasse(lien, liens) is not None:
            continue
        par_texte[lien.texte].append(lien)
    resumes: Dict[str, StatutEnBref] = {}
    for reference, ses_liens in par_texte.items():
        par_nature: Dict[str, set] = defaultdict(set)
        for lien in ses_liens:
            par_nature[lien.nature].add(_qui(lien))
        resumes[reference] = StatutEnBref(
            abroge=any(
                lien.nature in _FIN_DU_TEXTE and not lien.articles for lien in ses_liens
            ),
            textes=len({_qui(lien) for lien in ses_liens}),
            # Ordre stable, pour que deux reponses identiques se comparent.
            natures={nature: len(qui) for nature, qui in sorted(par_nature.items())},
            date_dernier_texte=max(lien.par_date for lien in ses_liens),
        )
    return resumes


_RESUMES: Dict[str, StatutEnBref] = resumer(LIENS)


def statut_en_bref(reference: Optional[str]) -> Optional[StatutEnBref]:
    """Le resume des liens prouves qui touchent ce texte, ou None si aucun n'est connu."""
    if not reference:
        return None
    return _RESUMES.get(reference)
