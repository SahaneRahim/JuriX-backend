"""
Liste paginee des textes : GET /api/v1/laws/?page=N.

Une page legere pour les listes (domaines, tous les textes) : ni texte, ni
champs d'administration, mais les reperes d'une ligne dense (objet en clair,
numero, date, statut prouve) et le VRAI total, pour afficher « 1 277 textes ·
20 affichés » et un lien « ?page=2 » qui marche sans JavaScript.
"""

from datetime import date
from typing import List, Literal, Optional

from pydantic import BaseModel, Field

from app.schemas.reperes import ReperesDuTexte

TriDesListes = Literal["date_desc", "date_asc"]


class LawListLightItem(ReperesDuTexte):
    """Une ligne de liste, sans le texte."""

    id: int
    reference: str
    title: str = Field(..., description="Intitule officiel complet")
    type: str = Field(..., description="Nature de l'acte (décret, loi, arrêté…)")
    language: Optional[str] = None
    status: str
    category_id: Optional[int] = None
    category_name: Optional[str] = None
    publication_date: Optional[date] = Field(
        None,
        description=(
            "Date en base, lue dans l'intitule. Le 1er janvier peut etre une date "
            "de remplissage : preferer `date_signature`, ou `annee`"
        ),
    )
    article_count: int = Field(0, ge=0, description="Nombre d'articles du texte")


class LawListPage(BaseModel):
    """Une page de la liste, et le total des textes qui passent les filtres."""

    items: List[LawListLightItem] = Field(default_factory=list)
    total: int = Field(..., ge=0, description="Textes qui passent les filtres, toutes pages")
    page: int = Field(..., ge=1, description="Page rendue, a partir de 1")
    page_size: int = Field(..., ge=1, description="Taille de page demandee")
    pages: int = Field(..., ge=0, description="Nombre de pages (0 si aucun texte)")
    sort: TriDesListes = Field(..., description="Tri applique")
