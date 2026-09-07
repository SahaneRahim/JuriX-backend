"""
Verification d'un jeton d'identite Google (Google Identity Services).

POURQUOI PAS FIREBASE. Firebase Authentication imposerait un SDK de ~200 Ko au
navigateur, une dependance `firebase-admin` au serveur, et surtout DEUX fichiers
d'utilisateurs a garder synchronises : celui de Firebase et notre table `users`.
Ici le navigateur obtient un jeton d'identite directement de Google, ce module
le verifie, et la route emet ensuite le JWT maison. Une dependance, zero paquet
npm, un seul fichier d'utilisateurs.

CE QUE `verify_oauth2_token` VERIFIE ELLE-MEME : la signature contre les
certificats publics de Google, l'`aud` quand une audience est fournie, l'`exp`
et l'`iat` avec la tolerance d'horloge, et l'`iss`.

CE QU'ELLE NE VERIFIE PAS, et qui nous incombe : `email_verified`. Un jeton
Google parfaitement valide peut porter une adresse non verifiee.

Author: JuriX Team
"""

import logging
from typing import Any, Dict

from google.auth import exceptions as google_exceptions
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token
from starlette.concurrency import run_in_threadpool

from app.core.config import settings

logger = logging.getLogger(__name__)

# Tolerance d'horloge entre notre serveur et celui de Google. Sans elle, une
# derive de quelques secondes fait rejeter des jetons parfaitement valides.
TOLERANCE_HORLOGE_S = 10


class GoogleNonConfigure(Exception):
    """`GOOGLE_CLIENT_ID` est vide. Traduit en 503."""


class JetonGoogleInvalide(Exception):
    """Jeton illisible, expire, d'une autre application, ou adresse non verifiee. 401."""


class GoogleInjoignable(Exception):
    """Les certificats publics de Google n'ont pas pu etre recuperes. 503."""


async def verifier_jeton_google(credential: str) -> Dict[str, Any]:
    """
    Verifie un jeton d'identite et rend ses claims.

    Returns:
        Les claims du jeton, dont `sub`, `email`, `email_verified`, `name`.

    Raises:
        GoogleNonConfigure: identifiant client absent
        JetonGoogleInvalide: jeton refuse, ou adresse non verifiee
        GoogleInjoignable: certificats Google inaccessibles
    """
    # GARDE CRITIQUE, ET C'EST LA PLUS IMPORTANTE DU MODULE.
    #
    # `verify_oauth2_token(..., audience=None)` DESACTIVE la verification de
    # l'`aud`. N'importe quel jeton d'identite Google — y compris emis pour une
    # application tierce quelconque — serait alors accepte, ce qui contourne
    # l'authentification entierement. Refuser AVANT d'appeler la bibliotheque
    # est donc la seule conduite sure quand l'identifiant client manque.
    if not settings.GOOGLE_CLIENT_ID:
        raise GoogleNonConfigure(
            "GOOGLE_CLIENT_ID absent : la connexion Google n'est pas configuree."
        )

    def _verifier() -> Dict[str, Any]:
        # Fonction BLOQUANTE : elle fait un GET HTTPS synchrone pour recuperer
        # les certificats de Google. Appelee directement dans la coroutine,
        # elle bloquerait la boucle d'evenements le temps de cet aller-retour.
        # `run_in_threadpool` la deporte sur un fil.
        return id_token.verify_oauth2_token(
            credential,
            google_requests.Request(),
            settings.GOOGLE_CLIENT_ID,
            clock_skew_in_seconds=TOLERANCE_HORLOGE_S,
        )

    try:
        claims = await run_in_threadpool(_verifier)
    except google_exceptions.TransportError as e:
        # NOTRE reseau a echoue, pas celui de l'utilisateur. Repondre 401 ici
        # l'accuserait d'un defaut qui n'est pas le sien.
        logger.error("Certificats Google injoignables : %s", e)
        raise GoogleInjoignable("Verification Google momentanement indisponible.") from e
    except (ValueError, google_exceptions.GoogleAuthError) as e:
        # Jeton illisible, signature fausse, expire, mauvaise audience, mauvais
        # emetteur. Un seul message : distinguer les causes aiderait surtout qui
        # cherche a en fabriquer un.
        logger.warning("Jeton Google refuse : %s", e)
        raise JetonGoogleInvalide("Jeton Google invalide.") from e

    if not claims.get("email"):
        raise JetonGoogleInvalide("Jeton Google sans adresse e-mail.")
    if not claims.get("sub"):
        raise JetonGoogleInvalide("Jeton Google sans identifiant de compte.")

    # `email_verified` n'est PAS verifie par la bibliotheque. Sans ce controle,
    # un compte Google dont l'adresse n'a jamais ete prouvee pourrait se lier a
    # un compte JuriX existant portant la meme adresse — c'est-a-dire prendre
    # le controle du compte de quelqu'un d'autre.
    if claims.get("email_verified") is not True:
        raise JetonGoogleInvalide("Adresse Google non verifiee.")

    return claims
