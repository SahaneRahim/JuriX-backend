"""
Schemas de GET /api/v1/corpus : les chiffres du corpus, pour l'accueil du chat
et la page « Aucun résultat » de la recherche (app/services/chiffres_du_corpus.py).
"""

from datetime import date, datetime
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from app.schemas.statut import TexteLie


class SourceDuCorpus(BaseModel):
    """Une famille de textes du corpus, par provenance."""

    code: Literal["prc", "ohada", "cima", "autres"]
    libelle: Dict[str, str] = Field(..., description="Libellé par langue : {fr, en}")
    textes: int


class NatureDuCorpus(BaseModel):
    nature: str = Field(..., description="Nature de l'acte, comme laws.type (loi, décret…)")
    textes: int


class CorpusResponse(BaseModel):
    """
    Les chiffres du corpus.

    Trois dates, trois sens, a ne pas confondre :
    - `dernier_ajout_le` : jour du dernier texte AJOUTE a JuriX (ingestion) ;
    - `texte_le_plus_recent.date_signature` : le texte le plus recent SIGNE ;
    - `liens_releves_le` : jour du releve des modifications entre textes.
    """

    textes: int = Field(..., description="Textes publiés dans JuriX")
    articles: int = Field(
        ...,
        description="Articles de ces textes : morceaux de type article, un seul par article "
        "découpé en plusieurs morceaux",
    )
    textes_par_nature: List[NatureDuCorpus] = Field(
        default_factory=list, description="Du plus fréquent au plus rare"
    )
    texte_le_plus_recent: Optional[TexteLie] = Field(
        None, description="Le texte publié dont la date d'acte est la plus récente"
    )
    dernier_ajout_le: Optional[date] = Field(
        None, description="Jour du dernier texte ajouté à JuriX (date technique)"
    )
    liens_releves_le: date = Field(
        ..., description="Jour du relevé des modifications entre textes (statut_juridique)"
    )
    sources: List[SourceDuCorpus] = Field(default_factory=list)
    libelle_sources: Dict[str, str] = Field(
        ..., description="Périmètre en quelques mots, par langue : « textes officiels camerounais et OHADA »"
    )
    calcule_le: datetime = Field(..., description="Heure du calcul (UTC) ; le résultat est gardé en mémoire")
