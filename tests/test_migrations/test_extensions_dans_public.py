"""
Les extensions PostgreSQL doivent resider dans le schema `public`.

CE QUE CE FICHIER PROTEGE, ET POURQUOI CE N'EST PAS THEORIQUE. La migration
`a6b7c8d9e0f1` definit `immutable_unaccent` par un corps qui NOMME le schema :

    SELECT public.unaccent('public.unaccent'::regdictionary, $1)

Ce nommage explicite est correct — il rend la fonction insensible au
`search_path`, ce qu'exige `IMMUTABLE`. Mais il impose une condition : que
l'extension soit bel et bien dans `public`. Or plusieurs hebergeurs infogeres
installent les extensions dans un schema dedie (`extensions`, le plus souvent).
Sur une telle base, la migration echoue, ou pire, la fonction se cree et ne
tombe en erreur qu'a la premiere recherche accentuee — donc en production, sur
une requete d'utilisateur.

`pg_trgm` a le meme besoin : `gin_trgm_ops` est nomme sans qualification dans
`c2d3e4f5a6b7`, et sa resolution passe par le `search_path`.

Le remede tient en une ligne, a executer AVANT la premiere migration :

    CREATE EXTENSION IF NOT EXISTS unaccent WITH SCHEMA public;

Ces tests sont la pour que l'oubli se voie ici, et non a la premiere recherche.

Usage:
    pytest tests/test_migrations/test_extensions_dans_public.py -v
"""

import pytest
from sqlalchemy import text

EXTENSIONS = ("vector", "pg_trgm", "unaccent")


@pytest.mark.asyncio
async def test_les_trois_extensions_sont_installees(db_session):
    presentes = {
        r[0]
        for r in (
            await db_session.execute(
                text("SELECT extname FROM pg_extension WHERE extname = ANY(:noms)"),
                {"noms": list(EXTENSIONS)},
            )
        ).all()
    }

    assert presentes == set(EXTENSIONS), f"extensions manquantes : {set(EXTENSIONS) - presentes}"


@pytest.mark.parametrize("extension", EXTENSIONS)
@pytest.mark.asyncio
async def test_l_extension_est_dans_public(db_session, extension):
    """Voir le docstring du module : `immutable_unaccent` nomme `public` en dur."""
    schema = (
        await db_session.execute(
            text(
                "SELECT n.nspname FROM pg_extension e "
                "JOIN pg_namespace n ON n.oid = e.extnamespace "
                "WHERE e.extname = :nom"
            ),
            {"nom": extension},
        )
    ).scalar_one_or_none()

    assert schema == "public", (
        f"l'extension {extension} est dans le schema {schema!r} et non 'public'. "
        f"Rejouer : CREATE EXTENSION IF NOT EXISTS {extension} WITH SCHEMA public;"
    )


@pytest.mark.asyncio
async def test_immutable_unaccent_replie_bien_les_accents(db_session):
    """
    Le test qui prouve que la chaine complete fonctionne.

    Il ne verifie pas un catalogue mais le RESULTAT : c'est lui qui echouerait
    si la fonction existait tout en pointant vers un dictionnaire introuvable.
    """
    resultat = (
        await db_session.execute(text("SELECT immutable_unaccent('Société Générale à Yaoundé')"))
    ).scalar_one()

    assert resultat == "Societe Generale a Yaounde"


@pytest.mark.asyncio
async def test_gin_trgm_ops_est_resolvable(db_session):
    """
    `gin_trgm_ops` est nomme SANS qualification dans les migrations d'index.

    Sa resolution depend donc du `search_path` : hors de `public`, la creation
    d'index echouerait avec « operator class does not exist ».
    """
    existe = (
        await db_session.execute(
            text(
                "SELECT count(*) FROM pg_opclass o "
                "JOIN pg_namespace n ON n.oid = o.opcnamespace "
                "WHERE o.opcname = 'gin_trgm_ops' AND n.nspname = 'public'"
            )
        )
    ).scalar_one()

    assert existe == 1
