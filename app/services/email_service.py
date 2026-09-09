"""
Envoi de courriel transactionnel, via l'API HTTPS de Brevo.

POURQUOI PAS SMTP. `smtplib` sortirait sur les ports 587 ou 465, et les
hebergeurs gratuits les BLOQUENT en sortie — Render l'ecrit noir sur blanc pour
ses services gratuits, Railway le reserve a ses offres payantes. Un envoi par
SMTP echouerait donc en production sans echouer en developpement, ce qui est la
pire forme de panne. Le transport est une requete HTTPS ordinaire, et il le
reste : le test `test_url_est_l_api_https_pas_smtp` existe pour que le jour ou
quelqu'un « simplifie » vers la bibliotheque standard, la suite rougisse avant
que l'hebergeur ne coupe le port en silence.

CE QUE CE MODULE NE FAIT PAS. Il ne connait ni les jetons, ni les utilisateurs,
ni les gabarits metier : il envoie un message et rend son identifiant. Les
politiques — combien de messages par adresse, que faire d'un compte Google —
appartiennent aux routes.

SUR L'EXPEDITEUR. Sans nom de domaine, l'adresse verifiee est une adresse
gratuite (gmail.com) que Brevo ne peut pas authentifier : il reecrit alors
l'expediteur en `@brevosend.com`. Le message part et arrive, mais affiche un
expediteur que le destinataire ne reconnait pas. C'est pourquoi `replyTo` porte
la vraie adresse, et pourquoi l'interface previent l'utilisateur de regarder ses
indesirables. Le correctif definitif est un nom de domaine et une signature
DKIM ; en attendant, mieux vaut le dire que le laisser deviner.
"""

import logging
from datetime import datetime, timezone

import httpx

from app.core.config import settings

logger = logging.getLogger(__name__)

API_BREVO = "https://api.brevo.com/v3/smtp/email"


class CourrielNonConfigure(Exception):
    """
    `BREVO_API_KEY` est vide : l'envoi est desactive.

    Ce n'est PAS une erreur — c'est l'etat normal en developpement et en test.
    Les appelants l'attrapent et poursuivent : une inscription doit reussir sans
    cle, sinon l'absence de configuration rendrait le service inutilisable.
    """


class CourrielRefuse(Exception):
    """Brevo a repondu 4xx : cle invalide, expediteur non verifie, quota atteint. PERMANENT."""


class CourrielInjoignable(Exception):
    """5xx, delai depasse, reseau. TRANSITOIRE — un renvoi peut reussir."""


# Compteur de messages du jour, en memoire de processus. Le palier gratuit
# plafonne a 300 par jour PARTAGES entre transactionnel et campagnes ; on
# s'arrete avant, pour qu'un pic ne consomme pas la totalite du budget.
#
# En memoire et non en base, deliberement : c'est un garde-fou, pas une
# comptabilite. Il repart a zero a chaque redeploiement, et la source de verite
# reste le refus de Brevo lui-meme. L'etranglement durable, celui qui compte,
# est en base et par adresse — voir les routes.
_budget = {"jour": None, "envoyes": 0}


def _aujourdhui() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _consommer_le_budget() -> None:
    jour = _aujourdhui()
    if _budget["jour"] != jour:
        _budget["jour"] = jour
        _budget["envoyes"] = 0
    if _budget["envoyes"] >= settings.BREVO_BUDGET_QUOTIDIEN:
        raise CourrielRefuse(
            f"Budget quotidien atteint ({settings.BREVO_BUDGET_QUOTIDIEN} messages)."
        )
    _budget["envoyes"] += 1


def envoi_configure() -> bool:
    """Vrai si un message peut partir. Les routes s'en servent pour ne pas creer de jeton inutile."""
    return bool(settings.BREVO_API_KEY and settings.BREVO_SENDER_EMAIL)


def lien_configure() -> bool:
    """Vrai si un message PORTEUR D'UN LIEN peut partir."""
    return envoi_configure() and bool(settings.FRONTEND_BASE_URL)


def url_du_front(chemin: str) -> str:
    """
    Construit un lien vers le front.

    Raises:
        CourrielNonConfigure: `FRONTEND_BASE_URL` est vide. Sans elle le lien se
            batirait sur l'hote de l'API, qui ne sert aucune page : le
            destinataire recevrait un lien mort. Mieux vaut ne rien envoyer.
    """
    if not settings.FRONTEND_BASE_URL:
        raise CourrielNonConfigure(
            "FRONTEND_BASE_URL absente : un courriel porteur d'un lien serait un lien mort."
        )
    return f"{settings.FRONTEND_BASE_URL.rstrip('/')}/{chemin.lstrip('/')}"


async def envoyer(destinataire: str, sujet: str, texte: str, html: str) -> str:
    """
    Envoie un message et rend l'identifiant rendu par Brevo.

    Raises:
        CourrielNonConfigure: envoi desactive — a attraper, pas a propager
        CourrielRefuse: refus permanent (cle, expediteur, quota)
        CourrielInjoignable: panne passagere
    """
    # GARDE, la plus importante du module — meme discipline que
    # `verifier_jeton_google`. Sans elle on POSTerait vers Brevo avec un en-tete
    # `api-key` vide : un 401 a chaque inscription, en silence, et personne
    # n'aurait le moindre moyen de comprendre pourquoi rien n'arrive.
    if not envoi_configure():
        raise CourrielNonConfigure(
            "BREVO_API_KEY ou BREVO_SENDER_EMAIL absente : l'envoi n'est pas configure."
        )

    _consommer_le_budget()

    charge = {
        "sender": {"name": settings.BREVO_SENDER_NAME, "email": settings.BREVO_SENDER_EMAIL},
        # L'expediteur affiche sera peut-etre reecrit en @brevosend.com ; le
        # replyTo, lui, ne l'est pas. C'est le seul moyen qu'une reponse arrive.
        "replyTo": {"email": settings.BREVO_SENDER_EMAIL},
        "to": [{"email": destinataire}],
        "subject": sujet,
        # LES DEUX PARTIES, TOUJOURS. Certains clients n'affichent pas le HTML,
        # et un message dont le lien n'existe que dans la partie HTML devient
        # illisible chez eux.
        "textContent": texte,
        "htmlContent": html,
    }

    try:
        async with httpx.AsyncClient(timeout=settings.BREVO_TIMEOUT_S) as client:
            reponse = await client.post(
                API_BREVO,
                # `api-key`, et surtout PAS `Authorization: Bearer` : Brevo
                # repond 401 sur Bearer, et ce 401 est indiscernable de celui
                # d'une cle invalide. Des heures perdues pour un nom d'en-tete.
                headers={
                    "api-key": settings.BREVO_API_KEY,
                    "accept": "application/json",
                    "content-type": "application/json",
                },
                json=charge,
            )
    except httpx.HTTPError as exc:
        logger.error("Brevo injoignable : %s", exc)
        raise CourrielInjoignable(f"Brevo injoignable : {exc}") from exc

    if reponse.status_code >= 500:
        logger.error("Brevo a repondu %s", reponse.status_code)
        raise CourrielInjoignable(f"Brevo a repondu {reponse.status_code}")
    if reponse.status_code >= 400:
        # Journalise le corps : c'est lui qui dit « expediteur non verifie »
        # plutot que « cle invalide ». Le destinataire, lui, n'en saura rien.
        logger.error("Brevo a refuse l'envoi (%s) : %s", reponse.status_code, reponse.text[:300])
        raise CourrielRefuse(f"Brevo a refuse l'envoi ({reponse.status_code})")

    try:
        return reponse.json().get("messageId", "")
    except ValueError:
        return ""
