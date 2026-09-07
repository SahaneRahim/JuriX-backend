"""
Schemas du mode COMPARAISON.

Le contrat central est celui de `ComparisonCell` : une cellule ne peut pas
exister sans dire d'ou elle vient. `sources` est obligatoire — vide signifie
« le corpus ne repond pas », jamais « je n'ai pas cite ». C'est ce qui empeche
le modele de combler un trou par de la prose, et c'est la raison d'etre du mode.

Author: JuriX Team
"""

from typing import List, Optional

from pydantic import BaseModel, Field, StringConstraints, field_validator
from typing_extensions import Annotated

# Axes de comparaison par defaut. IMPOSES, et non laisses au modele : sinon il
# choisit les criteres sur lesquels il a de la matiere, ce qui donne une grille
# flatteuse et inutile. Un critere sans reponse est une information.
CRITERES_PAR_DEFAUT = [
    "Objet et definition",
    "Conditions d'octroi",
    "Autorite competente",
    "Duree et renouvellement",
    "Obligations du titulaire",
    "Cession et transmission",
    "Retrait, suspension ou annulation",
]

# Mention d'absence, PAR LANGUE. Elle est injectee dans le prompt systeme, qui
# ordonne au modele de l'ecrire mot pour mot : une constante francaise unique
# faisait rendre « Non trouve… » au milieu d'une comparaison anglaise.
MENTIONS_ABSENCE = {
    "fr": "Non trouvé dans les textes consultés",
    "en": "Not found in the texts consulted",
}
MENTION_ABSENCE = MENTIONS_ABSENCE["fr"]


def mention_absence(language: str) -> str:
    return MENTIONS_ABSENCE.get(language, MENTIONS_ABSENCE["fr"])


class ComparisonRequest(BaseModel):
    """Deux sujets a comparer, et rien de plus."""

    subject_a: str = Field(..., min_length=2, max_length=200)
    subject_b: str = Field(..., min_length=2, max_length=200)
    language: str = Field("fr", description="Langue de la reponse : fr ou en")
    criteria: Optional[List[Annotated[str, StringConstraints(max_length=120)]]] = Field(
        None,
        max_length=12,
        description="Axes de comparaison. Par defaut, les sept axes canoniques.",
    )
    law_id: Optional[int] = Field(
        None, description="Restreindre la comparaison a un seul document"
    )
    top_k: int = Field(8, ge=3, le=20, description="Articles recuperes par sujet")

    @field_validator("language")
    @classmethod
    def valider_langue(cls, v: str) -> str:
        if v.lower() not in {"fr", "en"}:
            raise ValueError(f"Language must be 'fr' or 'en'. Got: {v}")
        return v.lower()

    @field_validator("subject_a", "subject_b")
    @classmethod
    def valider_sujet(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Subject must not be blank")
        return v.strip()

    @field_validator("criteria")
    @classmethod
    def valider_criteres(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        if v is None:
            return None
        nettoyes = [c.strip() for c in v if c and c.strip()]
        if not nettoyes:
            raise ValueError("Criteria must not be empty when provided")
        # Un critere est un LIBELLE, pas un paragraphe. Sans borne par element
        # ni interdiction du retour a la ligne, ce champ devient une entree de
        # prompt libre : il est recopie tel quel en queue du prompt, entre deux
        # lignes d'instruction — la position la plus favorable a une injection —
        # sur une route publique et sans limiteur de debit. Douze criteres de
        # 50 000 caracteres pesaient douze fois le corpus lui-meme.
        for c in nettoyes:
            if "\n" in c or "\r" in c:
                raise ValueError("A criterion must be a single line")
        return nettoyes


class SourceRef(BaseModel):
    """
    Un article cite par une cellule, resolu jusqu'a son identifiant.

    Le modele ne rend qu'un NUMERO ; le service le rapproche des articles
    reellement recuperes pour retrouver l'identifiant, le titre de la loi et le
    texte. Un numero que le modele invente ne se rapproche de rien et
    n'apparait donc jamais ici — c'est le premier filtre anti-invention.
    """

    article_id: Optional[int] = None
    law_id: int
    law_title: str
    reference: str
    number: str
    article_title: Optional[str] = None
    page_number: Optional[int] = None
    content: str = Field("", description="Texte integral, deplie sous la cellule")


class ComparisonCell(BaseModel):
    """Une case de la grille : ce qui est dit, et ou c'est ecrit."""

    value: str
    sources: List[SourceRef] = Field(default_factory=list)


class ComparisonRow(BaseModel):
    criterion: str
    a: ComparisonCell
    b: ComparisonCell


class ComparisonResponse(BaseModel):
    subject_a: str
    subject_b: str
    language: str
    rows: List[ComparisonRow]
    key_differences: List[str] = Field(default_factory=list)
    blind_spots: List[str] = Field(
        default_factory=list,
        description="Ce que les textes consultes ne permettent pas de trancher",
    )
    articles_a: List[SourceRef] = Field(
        default_factory=list, description="Articles recuperes pour le sujet A"
    )
    articles_b: List[SourceRef] = Field(default_factory=list)
    unmatched_citations: List[str] = Field(
        default_factory=list,
        description=(
            "Numeros cites par le modele qui ne correspondent a aucun article "
            "recupere. Toujours affiche : c'est le signal d'une citation inventee."
        ),
    )
    retrieval_time_ms: int = 0
    generation_time_ms: int = 0
