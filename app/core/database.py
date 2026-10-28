"""
Gestion des sessions de base de données asynchrones.

Ce module configure SQLAlchemy 2.0 avec sessions asynchrones pour PostgreSQL.
Utilise le pattern AsyncSession pour toutes les opérations DB.

Usage dans routes FastAPI:
    @router.get("/laws")
    async def list_laws(db: AsyncSession = Depends(get_db)):
        result = await db.execute(select(Law))
        return result.scalars().all()

Author: JuriX Team
"""

import logging
import time
from typing import AsyncGenerator

from fastapi import HTTPException
from sqlalchemy import create_engine, event, exc, text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from sqlalchemy.util import await_only

from app.core import mesure_sql
from app.core.config import settings

logger = logging.getLogger(__name__)


# Base class for all SQLAlchemy models
class Base(DeclarativeBase):
    """Base class pour tous les modèles SQLAlchemy."""
    pass


# ---------------------------------------------------------------------------
# Connexions mortes : verification a la sortie du pool, sans pool_pre_ping
# ---------------------------------------------------------------------------
# pool_pre_ping coutait QUATRE allers-retours a chaque emprunt d'une connexion :
# derriere un pooler en mode transaction, SQLAlchemy entoure son sondage d'une
# transaction explicite (BEGIN, ";" prepare puis execute, ROLLBACK — le cache
# d'instructions etant coupe, preparer coute un aller-retour). Mesure en
# production : 4 des 8 allers-retours de GET /categories, 8 par recherche
# (deux emprunts), a 83 ms chacun entre Railway (Virginie) et le pooler
# (Irlande). Le payer a CHAQUE emprunt protegeait contre un cas rare.
#
# Ce qui tue reellement une connexion du pool, chez nous :
#   - Neon (cible) suspend le calcul apres 5 min sans requete — les connexions
#     inactives ne l'en empechent pas — et ferme alors toutes les connexions ;
#   - un redemarrage du pooler (Supavisor, PgBouncer de Neon), qui les ferme
#     aussi ;
#   - plus rarement, un equipement reseau qui oublie une connexion inactive.
# Ni Supavisor (client_idle_timeout = 0 par defaut) ni le PgBouncer de Neon
# (aucun delai documente) ne ferment d'eux-memes une connexion client inactive.
# Il n'y a donc pas de delai a « devancer » avec pool_recycle.
#
# Les deux premiers cas FERMENT la connexion : asyncpg le voit sans rien
# envoyer (is_closed()), la boucle d'evenements ayant recu la fin de flux. Le
# troisieme ne se voit qu'en parlant au serveur. D'ou, a chaque emprunt :
#   1. connexion fermee -> remplacee, ZERO aller-retour ;
#   2. inactive depuis plus de PING_APRES_INACTIVITE_S -> un « SELECT 1 » en
#      protocole simple : UN aller-retour, sans instruction preparee ni
#      transaction, donc sans risque derriere un pooler en mode transaction ;
#      borne par DELAI_PING_S, la ou pool_pre_ping pouvait attendre la fin des
#      retransmissions TCP sur une connexion devenue muette ;
#   3. sinon, rien : les emprunts d'une meme requete, ou de requetes
#      rapprochees, ne paient plus aucun sondage.
# Un echec leve DisconnectionError : le pool jette la connexion et en ouvre
# une neuve, de facon transparente — le mecanisme meme de pool_pre_ping.
#
# Ce qui reste non couvert, comme avec pool_pre_ping : une connexion qui meurt
# pendant qu'elle sert. La requete echoue alors une fois, et le pool est
# invalide.
PING_APRES_INACTIVITE_S = 60.0
DELAI_PING_S = 5.0
_RENDUE_A = "jurix_rendue_a"


def _noter_le_retour(dbapi_connection, connection_record) -> None:
    """Horodate le retour au pool : base de la decision de sonder."""
    connection_record.info[_RENDUE_A] = time.monotonic()


def _verifier_a_la_sortie(dbapi_connection, connection_record, connection_proxy) -> None:
    """
    Ecarte une connexion morte avant de la preter (voir plus haut).

    Appelee dans le greenlet de SQLAlchemy : await_only y execute une
    coroutine asyncpg. connection_record.info est vide pour une connexion
    neuve — le pool le vide a chaque reconnexion — : elle n'est pas sondee.
    """
    connexion = connection_record.driver_connection
    if connexion is None or connexion.is_closed():
        raise exc.DisconnectionError("connexion fermee par le serveur pendant son inactivite")

    rendue_a = connection_record.info.get(_RENDUE_A)
    if rendue_a is None or time.monotonic() - rendue_a < PING_APRES_INACTIVITE_S:
        return
    # Compte dans Server-Timing (ping=) : un aller-retour de plus pour la
    # requete qui emprunte cette connexion.
    mesure_sql.noter_un_sondage()
    try:
        # execute() sans argument : protocole simple, un seul aller-retour.
        await_only(connexion.execute("SELECT 1", timeout=DELAI_PING_S))
    except Exception as erreur:
        raise exc.DisconnectionError(
            f"sondage d'une connexion inactive en echec ({type(erreur).__name__})"
        ) from erreur


def brancher_verification_des_connexions(moteur_synchrone) -> None:
    """Pose la verification sur le pool d'un moteur asyncpg (engine.sync_engine)."""
    event.listen(moteur_synchrone, "checkin", _noter_le_retour)
    event.listen(moteur_synchrone, "checkout", _verifier_a_la_sortie)


# Create async engine
engine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG if hasattr(settings, 'DEBUG') else False,  # Log SQL queries in debug mode
    # Caches d'instructions COUPES, et ils le restent : un pooler en mode
    # transaction peut executer la preparation et l'execution sur deux
    # connexions serveur differentes.
    connect_args={
        "statement_cache_size": 0,
        "prepared_statement_cache_size": 0,
    },
    # 5 + 5, et non 20 + 10. L'ancien dimensionnement demandait jusqu'a 30
    # connexions asynchrones (plus 10 synchrones) pour un service qui n'en
    # utilise jamais trois. Derriere un pooler externe — le mode transaction de
    # Supavisor, par exemple — le multiplexage est deja fait en amont, et un
    # pool local surdimensionne ne fait que consommer le quota de connexions.
    pool_size=5,
    max_overflow=5,
    # Remplace par _verifier_a_la_sortie (voir plus haut).
    pool_pre_ping=False,
    # LIFO : la connexion rendue en dernier ressort la premiere. Les emprunts
    # rapproches retombent sur une connexion tout juste utilisee, donc sans
    # sondage ; les autres vieillissent a l'ecart au lieu d'etre sondees a
    # tour de role (FIFO).
    pool_use_lifo=True,
    # -1 et non 3600 : recycler par l'AGE fermait chaque heure une connexion
    # saine, et la requete suivante payait la reconnexion — une vingtaine
    # d'allers-retours avec TLS, environ 1,5 s depuis la Virginie. Aucun des
    # deux poolers n'imposant de duree de vie, cela ne protegeait de rien que
    # la verification a la sortie ne couvre deja. Avec Neon, la suspension
    # ferme de toute facon les connexions apres 5 min sans requete.
    pool_recycle=-1,
)
brancher_verification_des_connexions(engine.sync_engine)
# Instructions, transactions et connexions de chaque requete, pour l'en-tete
# Server-Timing (voir app/core/mesure_sql.py).
mesure_sql.brancher(engine.sync_engine)

# Create async session factory
AsyncSessionLocal = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Don't expire objects after commit
    autoflush=False,  # Manual flush control
    autocommit=False,  # Manual commit control
)


# ---------------------------------------------------------------------------
# Moteur SYNCHRONE partage
# ---------------------------------------------------------------------------
# Le pipeline de traitement (app/tasks/process_law.py) et le cache
# d'embeddings tournent dans un thread d'executor, hors boucle d'evenements :
# ils ont besoin d'un moteur synchrone. Onze appels a create_engine() etaient
# dissemines dans le code, un par fonction ou par instance, aucun n'appelait
# jamais dispose() : chaque loi traitee laissait un pool de connexions ouvert
# jusqu'a l'arret du processus.
#
# Les consommateurs doivent utiliser SyncSessionLocal(), JAMAIS sync_engine
# directement : c'est ce qui permet de rebrancher toute l'application sur une
# autre base avec sessionmaker.configure(bind=...), ce dont les tests dependent.
#
# LA CHAINE DE REQUETE EST RETIREE, comme dans alembic/env.py. L'URL de
# production porte des options propres a asyncpg
# (?prepared_statement_cache_size=0, documentee dans .env.example), que libpq
# refuse avant meme d'ouvrir une connexion :
#     invalid dsn: invalid URI query parameter
# Le cache d'embeddings, qui passe par ce moteur, n'a donc JAMAIS atteint la
# base en production (embedding_cache : zero insertion, aucune requete dans
# pg_stat_statements), et l'erreur etait avalee au niveau DEBUG. Rien d'utile
# n'est perdu : le TLS se regle par PGSSLMODE, que libpq lit comme asyncpg, et
# a defaut libpq tente TLS d'abord (sslmode=prefer).
SYNC_DATABASE_URL = settings.DATABASE_URL.replace("+asyncpg", "").split("?", 1)[0]

sync_engine = create_engine(
    SYNC_DATABASE_URL,
    pool_size=5,
    max_overflow=5,
    # Garde ici : psycopg2 ne voit qu'en parlant au serveur qu'une connexion a
    # ete fermee (pas de boucle d'evenements qui recoive la fin de flux), et
    # son sondage ne coute qu'un aller-retour (SELECT 1 en autocommit).
    pool_pre_ping=True,
    pool_use_lifo=True,
    # Meme raisonnement que pour le moteur asynchrone : le sondage couvre les
    # connexions mortes, recycler par l'age ne faisait que reconnecter.
    pool_recycle=-1,
    future=True,
)


# Une erreur du moteur synchrone se voyait mal : le cache d'embeddings
# l'avale au niveau DEBUG, et c'est ainsi qu'une URL refusee par psycopg2 a pu
# desactiver ce cache en production sans que rien ne le dise. Chaque cause
# distincte est desormais journalisee UNE fois en WARNING, sans inonder le
# journal si elle se repete a chaque recherche. Les echecs du sondage ne
# comptent pas : le pool les repare seul.
_CAUSES_SIGNALEES: set = set()
_CAUSES_SIGNALEES_MAX = 20


def _signaler_une_fois(contexte) -> None:
    if contexte.is_pre_ping:
        return
    erreur = contexte.original_exception
    # Premiere ligne seulement : libpq ne cite jamais le mot de passe, mais une
    # ligne DETAIL peut citer des valeurs.
    premiere_ligne = (str(erreur).strip().splitlines() or [""])[0][:200]
    cause = f"{type(erreur).__name__}: {premiere_ligne}"
    if cause in _CAUSES_SIGNALEES or len(_CAUSES_SIGNALEES) >= _CAUSES_SIGNALEES_MAX:
        return
    _CAUSES_SIGNALEES.add(cause)
    logger.warning(
        f"⚠️ Moteur synchrone (psycopg2) en echec — {cause}. Le cache d'embeddings "
        "et le pipeline d'ingestion en dependent. Signale une seule fois par cause."
    )


def brancher_signalement_des_erreurs(moteur_synchrone) -> None:
    """Pose _signaler_une_fois sur un moteur psycopg2."""
    event.listen(moteur_synchrone, "handle_error", _signaler_une_fois)


brancher_signalement_des_erreurs(sync_engine)

SyncSessionLocal = sessionmaker(
    bind=sync_engine,
    expire_on_commit=False,
    autoflush=False,
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency pour obtenir une session DB asynchrone.

    Crée une nouvelle session pour chaque requête et garantit
    sa fermeture propre même en cas d'erreur.

    Yields:
        AsyncSession: Session de base de données asynchrone

    Example:
        >>> @router.get("/laws")
        >>> async def list_laws(db: AsyncSession = Depends(get_db)):
        ...     result = await db.execute(select(Law))
        ...     return result.scalars().all()
    """
    async with AsyncSessionLocal() as session:
        try:
            logger.debug("📦 Nouvelle session DB créée")
            yield session
        except HTTPException:
            # Une HTTPException n'est pas une erreur de base : c'est une
            # reponse deliberee de la route. La journaliser en « Erreur session
            # DB » avec sa trace complete noyait les vraies erreurs SQL sous des
            # 404 et des 429 parfaitement normaux. On annule quand meme la
            # transaction, sans mentir sur la cause.
            await session.rollback()
            raise
        except Exception as e:
            logger.error(f"❌ Erreur session DB: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()
            logger.debug("🗑️  Session DB fermée")




# init_db() et drop_db() ont ete retirees : elles appelaient
# Base.metadata.create_all/drop_all, et n'etaient appelees par personne. Le
# schema appartient a Alembic — deux sources de verite pour une meme structure
# finissent toujours par diverger.


async def close_db() -> None:
    """
    Ferme le moteur de base de données.

    Ferme toutes les connexions actives et libère les ressources.
    Doit être appelé lors de l'arrêt de l'application.

    Example:
        >>> # Dans le shutdown event de FastAPI
        >>> @app.on_event("shutdown")
        >>> async def shutdown():
        ...     await close_db()
    """
    await engine.dispose()
    close_sync_db()
    logger.info("✅ Moteur de base de données fermé")


def close_sync_db() -> None:
    """Ferme le moteur synchrone partage. Appele par close_db()."""
    sync_engine.dispose()
    logger.info("✅ Moteur synchrone fermé")


async def health_check_db() -> bool:
    """
    Vérifie la santé de la connexion à la base de données.

    Returns:
        bool: True si la connexion fonctionne, False sinon

    Example:
        >>> is_healthy = await health_check_db()
        >>> if not is_healthy:
        ...     logger.error("DB connection failed")
    """
    try:
        async with AsyncSessionLocal() as session:
            # text() obligatoire : SQLAlchemy 2.0 refuse une chaine brute
            # (ObjectNotExecutableError). L'exception etait avalee par le except
            # ci-dessous, donc health_check_db renvoyait TOUJOURS False et
            # GET /api/v1/admin/system declarait la base en panne en permanence.
            await session.execute(text("SELECT 1"))
            logger.debug("✅ DB health check OK")
            return True
    except Exception as e:
        logger.error(f"❌ DB health check failed: {e}")
        return False
