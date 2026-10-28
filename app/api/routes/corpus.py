"""
GET /api/v1/corpus : les chiffres du corpus, pour l'accueil du chat (« Je
réponds à partir de 2 275 textes officiels camerounais et OHADA… ») et la page
« Aucun résultat » de la recherche.

Route publique et legere : le calcul est garde dix minutes en memoire
(app/services/chiffres_du_corpus.py), et la reponse peut etre gardee par le
navigateur et par un cache partage.
"""

from fastapi import APIRouter, Depends, Response
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from app.schemas.corpus import CorpusResponse
from app.services.chiffres_du_corpus import chiffres_du_corpus

router = APIRouter(tags=["corpus"])

# public : la reponse ne depend ni du compte ni d'un cookie. max-age=300 : la
# page du chat la redemande a chaque visite, et depuis le Cameroun chaque appel
# coute 0,5 a 1 s. s-maxage=600 : la duree du cache memoire, qu'un cache
# partage ne doit pas depasser. stale-while-revalidate : un compte en retard est
# sans consequence, une attente ne l'est pas. Meme raisonnement que
# CACHE_CONTROL_CATEGORIES.
CACHE_CONTROL_CORPUS = "public, max-age=300, s-maxage=600, stale-while-revalidate=3600"


# Sans barre oblique finale aussi, hors du schema : meme raison que GET /laws
# (une redirection 307 vers http:// derriere le proxy coutait un aller-retour).
@router.get("", response_model=CorpusResponse)
@router.get("/", response_model=CorpusResponse, include_in_schema=False)
async def get_corpus(response: Response, db: AsyncSession = Depends(get_db)) -> CorpusResponse:
    """
    Nombre de textes et d'articles, natures, sources, texte le plus récent.

    Trois dates à ne pas confondre : `dernier_ajout_le` (dernier texte ajouté à
    JuriX), `texte_le_plus_recent.date_signature` (texte le plus récent signé)
    et `liens_releves_le` (relevé des modifications entre textes).
    """
    chiffres = await chiffres_du_corpus(db)
    response.headers["Cache-Control"] = CACHE_CONTROL_CORPUS
    return chiffres
