"""Point d'entrée FastAPI."""

import asyncio
import logging
import re
import time
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from starlette.datastructures import MutableHeaders

from app.api.routes import (
    admin,
    analytics,
    articles,
    auth,
    batch_upload,
    categories,
    classifier,
    comparison,
    language,
    laws,
    ocr,
    personas,
    rag,
    search,
    upload,
)
from app.core.config import settings
from app.core.database import AsyncSession, close_db, get_db
from app.services.search_service import precharger_les_embeddings

logger = logging.getLogger(__name__)

# Les journaux de l'application partaient au niveau INFO vers des loggers sans
# gestionnaire : uvicorn ne configure que les siens, et Python ne laisse passer
# que WARNING par defaut. Le journal d'un appel Mistral ou Groq (raison de fin,
# jetons, duree) n'apparaissait donc nulle part, ni en developpement ni dans
# `docker compose logs api`. propagate reste vrai : la capture de pytest
# continue de les voir.
_journal_app = logging.getLogger("app")
if not _journal_app.handlers:
    _sortie = logging.StreamHandler()
    _sortie.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    _journal_app.addHandler(_sortie)
    _journal_app.setLevel(logging.INFO)


class _MasqueJetonDansURL(logging.Filter):
    """
    uvicorn journalise chaque WebSocket avec son URL entiere
    (« "WebSocket /api/v1/batch-upload/ws/...?token=..." [accepted] »), par son
    journal d'erreurs que --no-access-log ne coupe pas. Le jeton
    d'administration finissait en clair dans `docker compose logs api`.
    """

    _JETON = re.compile(r"(token=)[^&\s\"]+")

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.args, tuple):
            record.args = tuple(
                self._JETON.sub(r"\1***", a) if isinstance(a, str) else a for a in record.args
            )
        return True


logging.getLogger("uvicorn.error").addFilter(_MasqueJetonDansURL())

# Valeur de repli declaree dans config.py. Si elle survit hors developpement,
# les JWT seraient signes avec une clé publiée dans le dépôt.
_DEV_SECRET_KEY = "dev_secret_key_change_in_production_with_openssl_rand_hex_32"

# Cles qu'aucune instance hors developpement ne doit garder : toutes sont
# publiees dans le depot (valeur de developpement, modeles des fichiers
# d'exemple). Un modele oublie dans le .env de production passait le
# garde-fou, qui ne connaissait que la premiere.
_CLES_PUBLIEES = frozenset({
    _DEV_SECRET_KEY,
    "a-generer",
    "generate-a-strong-random-key-here",
    "change-me",
    "changeme",
    "secret",
})
_LONGUEUR_MIN_SECRET = 32


def _secret_key_refusee(cle: str) -> str:
    """La raison de refuser cette SECRET_KEY hors developpement, ou "" si elle convient."""
    if cle in _CLES_PUBLIEES:
        return "c'est une valeur publiee dans le depot"
    if len(cle) < _LONGUEUR_MIN_SECRET:
        return f"elle est trop courte ({len(cle)} caracteres, {_LONGUEUR_MIN_SECRET} au moins)"
    return ""


# Intervalle de purge : desormais `settings.CACHE_CLEANUP_INTERVAL_S`, lu A
# CHAQUE TOUR de boucle et non a l'import. Une constante de module est figee au
# chargement, donc ni reglable par l'environnement ni modifiable par un test.
#
# Le cache de recherche vit cinq minutes ; passer plus souvent ne libererait
# rien de plus. Effet de bord utile en production : cette boucle touche la base
# regulierement, ce qui empeche un hebergeur gratuit de mettre le projet en
# pause pour inactivite.
STATS_AGGREGATION_INTERVAL_S = 24 * 60 * 60


async def _agreger_les_statistiques() -> None:
    """
    Agregation quotidienne des statistiques par persona.

    `PersonaService.aggregate_daily_stats` existait, etait testee, et son
    docstring annoncait « appelee par une tache Celery chaque nuit ». Il n'y a
    jamais eu de Celery dans ce depot : la table `persona_stats` restait vide et
    les routes /personas/* qui la lisent rendaient des zeros. Meme motif que la
    purge des caches ci-dessous : une promesse de traitement de fond non tenue
    est pire qu'aucune promesse.
    """
    from app.core.database import AsyncSessionLocal
    from app.services.persona_service import PersonaService

    while True:
        try:
            await asyncio.sleep(STATS_AGGREGATION_INTERVAL_S)
            async with AsyncSessionLocal() as session:
                await PersonaService(session).aggregate_daily_stats()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"⚠️ Agregation des statistiques impossible: {e}")


async def _purger_les_caches() -> None:
    """Boucle de menage. Ne doit jamais interrompre le service."""
    from app.core.database import AsyncSessionLocal
    from app.services.postgres_search_service import cleanup_expired_cache

    while True:
        try:
            await asyncio.sleep(settings.CACHE_CLEANUP_INTERVAL_S)
            async with AsyncSessionLocal() as session:
                supprimees = await cleanup_expired_cache(session)
            if supprimees:
                logger.info(f"🧹 {supprimees} entrees de cache expirees supprimees")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.warning(f"⚠️ Purge des caches impossible: {e}")



@asynccontextmanager
async def lifespan(app: FastAPI):
    """Cycle de vie de l'application."""
    raison = _secret_key_refusee(settings.SECRET_KEY)
    if settings.ENVIRONMENT != "development" and raison:
        raise RuntimeError(
            f"SECRET_KEY refusée alors que ENVIRONMENT={settings.ENVIRONMENT!r} : "
            f"{raison}. Générez-en une avec `openssl rand -hex 32` et placez-la "
            "dans .env — sinon n'importe qui peut forger un jeton d'administration."
        )
    logger.info(f"🚀 {settings.APP_NAME} v{settings.VERSION} ({settings.ENVIRONMENT})")

    # Fournisseur d'embeddings sonde AVANT d'accepter des requetes : une cle
    # absente ou refusee se voit ici, pas a la premiere question. Un echec
    # n'empeche pas le demarrage — plein texte, chat et comptes fonctionnent
    # sans embeddings — mais il est journalise et la sante le montre. Un echec
    # passager (reseau pas encore pret) garde le service : chaque question
    # retentera. Hors de la boucle : c'est un appel HTTPS synchrone. Borne a
    # 20 s : httpx ne borne pas la resolution DNS, et un DNS muet au
    # redemarrage de la VM ne doit pas retenir l'API.
    try:
        await asyncio.wait_for(asyncio.to_thread(precharger_les_embeddings), timeout=20)
    except TimeoutError as e:
        logger.warning("⚠️ Sonde du fournisseur d'embeddings sans reponse en 20 s : demarrage sans elle")
        # La sante le montre ; la sonde, si elle finit, ecrira son issue.
        from app.services.embedding_service import suivi_semantique

        suivi_semantique.noter_sonde(e)

    # Purge des caches expires, en tache de fond.
    #
    # `cleanup_expired_cache` existait, etait importee par search_service, et
    # n'etait APPELEE NULLE PART. `query_cache` et `embedding_cache` portent une
    # colonne `expires_at` que rien n'appliquait : les deux tables grossissaient
    # sans fin. Une colonne d'expiration non appliquee est pire que pas de
    # colonne du tout — elle laisse croire que le menage est fait.
    tache_menage = asyncio.create_task(_purger_les_caches())
    tache_stats = asyncio.create_task(_agreger_les_statistiques())
    try:
        yield
    finally:
        for tache in (tache_menage, tache_stats):
            tache.cancel()
            with suppress(asyncio.CancelledError):
                await tache
        # La connexion gardee vers le fournisseur d'embeddings (Voyage).
        from app.services import search_service

        fournisseur = getattr(search_service._embedding_service_instance, "provider", None)
        fermer = getattr(fournisseur, "fermer", None)
        if fermer is not None:
            fermer()
        # close_db() existait mais n'etait jamais appele : le pool de
        # connexions asyncpg n'etait jamais libere a l'arret. Dans le `finally`
        # pour qu'il s'execute meme si l'arret vient d'une exception.
        await close_db()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    debug=settings.DEBUG,
    lifespan=lifespan,
)

# CORS — allow localhost in dev, Vercel domain in prod
# Set ALLOWED_ORIGINS in your env for production (comma-separated)
_extra_origins = [o.strip().rstrip('/') for o in settings.ALLOWED_ORIGINS.split(",") if o.strip()]
_cors_origins = [
    "http://localhost:5173",
    "http://localhost:4173",
    "http://127.0.0.1:5173",
    "https://juri-x-frontend.vercel.app",
] + _extra_origins

# Duree de vie de la reponse a une requete preliminaire (OPTIONS), en secondes.
# Starlette en annonce 600 par defaut : un telephone refaisait donc le
# prealable d'une recherche toutes les dix minutes — 0,94 a 1,2 s mesurees
# avant meme que le POST parte. 7200 est le plafond de Chrome (deux heures) ;
# Firefox accepte plus, mais Chrome ramenerait toute valeur superieure a 7200.
CORS_MAX_AGE_S = 7200

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    # Lisible par le code du front (fetch), et pas seulement dans les outils
    # de developpement : la duree serveur d'une recherche y devient mesurable.
    expose_headers=["Server-Timing"],
    max_age=CORS_MAX_AGE_S,
)


class DureeServeur:
    """
    En-tete `Server-Timing: app;dur=<ms>` sur chaque reponse HTTP.

    POURQUOI. Le `search_time_ms` du tableau de bord s'arrete avant les
    ecritures qui suivent la recherche et avant la construction de la reponse :
    mesure, il laissait de cote 1,1 a 4,9 s selon la taille du resultat. Cette
    duree-ci court de l'arrivee de la requete jusqu'a l'envoi des en-tetes.
    Pour une reponse JSON, le corps est deja serialise a ce moment-la : c'est
    tout le travail du serveur. Pour un flux, c'est le delai avant le premier
    octet. Les outils de developpement du navigateur l'affichent dans l'onglet
    Timing, et curl le montre sans rien installer.

    Intergiciel ASGI pur, et non BaseHTTPMiddleware : celui-ci fait passer
    chaque corps de reponse par une file supplementaire, et casse le flux des
    reponses longues. Ici, seul le message de debut de reponse est touche.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            # WebSocket (envoi par lots) et cycle de vie : rien a mesurer.
            await self.app(scope, receive, send)
            return

        debut = time.perf_counter()

        async def envoyer(message):
            if message["type"] == "http.response.start":
                duree_ms = (time.perf_counter() - debut) * 1000
                MutableHeaders(scope=message).append("Server-Timing", f"app;dur={duree_ms:.1f}")
            await send(message)

        await self.app(scope, receive, envoyer)


# Ajoute APRES CORS, donc place AUTOUR de lui : la duree compte aussi les
# reponses aux requetes preliminaires, que CORS sert sans atteindre les routes.
app.add_middleware(DureeServeur)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "JuriX API v2.1",
        "status": "running",
        "version": settings.VERSION
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/health/db")
async def health_db(db: AsyncSession = Depends(get_db)):
    """Sonde legere pour reinitialiser la mise en veille Neon sans charger le CPU."""
    await db.execute(text("SELECT 1"))
    return {"status": "healthy", "database": "connected"}


# Include routers with global /api/v1 prefix
app.include_router(auth.router, prefix="/api/v1/auth", tags=["auth"])
app.include_router(language.router, prefix="/api/v1/language", tags=["language"])
app.include_router(classifier.router, prefix="/api/v1/classifier", tags=["classifier"])
app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(categories.router, prefix="/api/v1/categories", tags=["categories"])
app.include_router(articles.router, prefix="/api/v1/articles", tags=["articles"])
app.include_router(personas.router, prefix="/api/v1/personas", tags=["personas"])
app.include_router(rag.router, prefix="/api/v1/rag", tags=["rag", "chatbot"])
app.include_router(comparison.router, prefix="/api/v1/compare", tags=["compare"])
app.include_router(upload.router, prefix="/api/v1/upload", tags=["upload"])
app.include_router(ocr.router, prefix="/api/v1/ocr", tags=["ocr"])
app.include_router(laws.router, prefix="/api/v1/laws", tags=["laws"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["analytics"])
app.include_router(admin.router, prefix="/api/v1/admin", tags=["admin"])
app.include_router(batch_upload.router, prefix="/api/v1/batch-upload", tags=["batch"])
# Trigger reload

