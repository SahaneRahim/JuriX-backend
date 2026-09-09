"""
La garde sur `DATABASE_URL` — le test qui aurait evite une panne de deploiement.

CE QU'IL PROTEGE. Le dialecte asyncpg de SQLAlchemy tient une liste FERMEE de
parametres d'URL qu'il consomme ; tout le reste part tel quel en argument nomme
vers `asyncpg.connect()`. Mesure, sur les versions installees :

    sqlalchemy 2.0.52 — create_connect_args('...?sslmode=require')
      -> {'host': ..., 'sslmode': 'require'}        # transmis brut
    asyncpg 0.31.0    — connect() n'a ni 'sslmode' ni **kwargs
      -> TypeError: connect() got an unexpected keyword argument 'sslmode'

L'erreur ne survient qu'au PREMIER acces a la base, donc bien apres un demarrage
en apparence reussi — et c'est exactement l'URL que fournissent les boutons
« copier » des hebergeurs de bases infogerees.

L'AUTRE MOITIE DU TEST COMPTE AUTANT. `prepared_statement_cache_size` doit
rester ACCEPTE : il est indispensable derriere un pooler en mode transaction,
qui ne supporte pas les instructions preparees. Une garde trop large casserait
la configuration de production tout en pretendant la proteger.

Usage:
    pytest tests/test_core/test_url_de_base.py -v
"""

import pytest
from pydantic import ValidationError

from app.core.config import Settings

BASE = "postgresql+asyncpg://user:motdepasse@hote.example:5432/base"


@pytest.mark.parametrize("parametre", ["sslmode=require", "channel_binding=require"])
def test_les_parametres_qui_fuient_sont_refuses(parametre):
    with pytest.raises(ValidationError) as erreur:
        Settings(DATABASE_URL=f"{BASE}?{parametre}")

    message = str(erreur.value)
    # Le message doit dire QUOI FAIRE. Un refus sans remede se contourne en
    # supprimant la garde, ce qui ramene exactement la panne d'origine.
    assert "PGSSLMODE" in message


def test_les_deux_parametres_ensemble_sont_nommes():
    """Le bouton « copier » de certains hebergeurs fournit les deux d'un coup."""
    with pytest.raises(ValidationError) as erreur:
        Settings(DATABASE_URL=f"{BASE}?sslmode=require&channel_binding=require")

    message = str(erreur.value)
    assert "sslmode" in message and "channel_binding" in message


def test_prepared_statement_cache_size_reste_accepte():
    """
    Ce parametre-la est CONSOMME par le dialecte, pas transmis a asyncpg.

    Il est indispensable derriere un pooler en mode transaction. Le refuser
    casserait la production sous couvert de la proteger.
    """
    url = f"{BASE}?prepared_statement_cache_size=0"

    assert Settings(DATABASE_URL=url).DATABASE_URL == url


def test_une_url_psycopg2_n_est_pas_concernee():
    """
    La garde ne vise que le pilote asyncpg. libpq, lui, comprend `sslmode` —
    refuser ici interdirait une URL parfaitement valide.
    """
    url = "postgresql://user:motdepasse@hote.example:5432/base?sslmode=require"

    assert Settings(DATABASE_URL=url).DATABASE_URL == url


def test_une_url_sans_parametre_passe():
    assert Settings(DATABASE_URL=BASE).DATABASE_URL == BASE


def test_alembic_retire_la_chaine_de_requete():
    """
    L'autre moitie du meme probleme, dans l'autre sens.

    `alembic/env.py` echange le schema asyncpg contre psycopg2. S'il conservait
    la chaine de requete, `prepared_statement_cache_size` partirait vers libpq,
    qui le refuse :

        ProgrammingError: invalid dsn: invalid URI query parameter

    Le conteneur lance `alembic upgrade head` a chaque demarrage : l'image ne
    demarrerait pas du tout.
    """
    import pathlib

    source = pathlib.Path("alembic/env.py").read_text(encoding="utf-8")

    assert '_db_url.split("?", 1)[0]' in source, (
        "alembic/env.py ne retire plus la chaine de requete : "
        "le conteneur ne demarrera pas avec une URL de pooler"
    )
