import os
import sys
from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool
from sqlalchemy.engine import make_url

from alembic import context

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
#
# disable_existing_loggers=False, et ce n'est pas cosmetique. Par defaut,
# fileConfig DESACTIVE tout logger deja cree. Or la suite de tests lance
# `alembic upgrade head` DANS son propre processus (tests/conftest.py), apres
# avoir importe app.* : tous les loggers de l'application devenaient muets
# pour le reste de la session. Un test qui verifie un journal echouait alors
# seulement dans la suite complete, jamais isole — et tout avertissement
# emis pendant les tests disparaissait. Le conteneur, lui, n'est pas touche :
# il lance alembic dans un processus distinct avant uvicorn.
if config.config_file_name is not None:
    fileConfig(config.config_file_name, disable_existing_loggers=False)

# add your model's MetaData object here
# for 'autogenerate' support
from app.core.database import Base
from app.models import (
    Article,
    Category,
    Conversation,
    EmailToken,
    Law,
    Message,
    User,
)

target_metadata = Base.metadata

# La base visee vient de DATABASE_URL, et de NULLE PART AILLEURS.
#
# Un repli sur settings.DATABASE_URL — donc sur le .env — avait ete ajoute pour
# eviter de taper la variable. Il faisait migrer en silence la base que le .env
# designe ce jour-la : demain, celle de production chez Neon. alembic.ini porte
# volontairement `placeholder://not-used` pour qu'un oubli echoue ; l'echec
# dit desormais quoi faire.
#
# L'application parle asyncpg ; Alembic a besoin d'un pilote synchrone.
_db_url = os.environ.get("DATABASE_URL", "")
if not _db_url:
    raise RuntimeError(
        "DATABASE_URL absente de l'environnement : Alembic ne lit pas le .env, "
        "la base visee doit etre nommee a chaque commande. Par exemple :\n"
        "  DATABASE_URL=postgresql://jurix:jurix@localhost:5433/jurix_dev alembic upgrade head"
    )

# Alembic needs a sync driver: swap asyncpg with psycopg2
_db_url = _db_url.replace("postgresql+asyncpg://", "postgresql://")
_db_url = _db_url.replace("postgres://", "postgresql://")  # parfois donne en postgres:// nu

# LA CHAINE DE REQUETE EST RETIREE, et ce n'est pas une precaution de style.
#
# Echanger le schema sans toucher aux parametres laissait passer a psycopg2
# des options qu'il ne connait pas. Mesure :
#
#     postgresql://...?prepared_statement_cache_size=0
#     -> ProgrammingError: invalid dsn: invalid URI query parameter
#
# Or ce parametre est INDISPENSABLE cote application avec un pooler en mode
# transaction, qui ne supporte pas les instructions preparees. Une seule URL
# devait donc servir deux pilotes qui n'acceptent pas les memes options.
#
# Le conteneur lance `alembic upgrade head` a chaque demarrage : sans ce
# nettoyage, l'image ne demarre pas du tout en production.
#
# Rien d'utile n'est perdu : le TLS se regle par PGSSLMODE, variable
# d'environnement lue nativement par libpq comme par asyncpg.
_db_url = _db_url.split("?", 1)[0]

config.set_main_option("sqlalchemy.url", _db_url)

# La cible est affichee AVANT toute migration, mot de passe masque : une
# erreur de base se voit avant d'avoir des consequences.
print(
    f"Alembic -> {make_url(_db_url).render_as_string(hide_password=True)}",
    file=sys.stderr,
)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
