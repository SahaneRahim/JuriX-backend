"""
Annuaire des institutions (contacts verifies, avec leur source).

Route publique et sans base : la liste vit dans app/services/annuaire.py.
"""
from typing import Optional

from fastapi import APIRouter, Response

from app.services.annuaire import REGIONS, THEMES, lister

router = APIRouter()


@router.get("")
async def annuaire(response: Response, theme: Optional[str] = None, region: Optional[str] = None):
    # Change seulement a un deploiement : une heure en cache chez le client et le CDN.
    response.headers["Cache-Control"] = "public, max-age=3600"
    return {"themes": THEMES, "regions": REGIONS, "fiches": lister(theme, region)}
