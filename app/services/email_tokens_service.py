"""
Politiques autour des jetons envoyes par courriel : creation, verification, etranglement.

CE QUE CE MODULE PROTEGE. Le palier gratuit de Brevo plafonne a 300 messages par
jour, PARTAGES entre transactionnel et campagnes. Sans etranglement, quiconque
martele « mot de passe oublie » epuise le quota de la journee et empeche les
vrais utilisateurs de recevoir quoi que ce soit. Le deni de service ne coute rien
a monter et ne laisse aucune trace visible : les inscriptions continuent de
reussir, seuls les courriels cessent d'arriver.

Deux couches, deliberement de natures differentes :

  1. PAR COMPTE, EN BASE. Trois demandes par heure et par usage, plus un delai de
     garde de deux minutes entre deux envois. Durable : elle survit aux
     redemarrages, et c'est la seule qui compte vraiment.

  2. PAR ADRESSE IP, EN MEMOIRE. Dix par heure. Elle attrape ce que la premiere
     ne voit pas : le balayage d'adresses inconnues, qui ne cree aucune ligne.
     Acceptable en memoire de processus UNIQUEMENT parce que l'image tourne avec
     `--workers 1`. Si quelqu'un passe a plusieurs processus, cette couche
     devient du decor et il faudra la deplacer en base.

LE PIEGE `X-Forwarded-For`, a ne surtout pas rater — voir `adresse_du_client`.
"""

import logging
import time
from collections import defaultdict, deque
from datetime import timedelta
from typing import Optional

from fastapi import Request
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.email_token import EmailToken, hachage, maintenant_naif, nouveau_jeton
from app.models.user import User

logger = logging.getLogger(__name__)

# Etranglement par compte, en base.
DEMANDES_PAR_HEURE = 3
DELAI_DE_GARDE = timedelta(minutes=2)

# Etranglement par IP, en memoire.
DEMANDES_IP_PAR_HEURE = 10
FENETRE_IP_S = 3600

_demandes_par_ip: dict[str, deque] = defaultdict(deque)


def adresse_du_client(request: Request) -> str:
    """
    Adresse IP reelle de l'appelant, derriere un proxy.

    LE PIEGE, ET IL ANNULE COMPLETEMENT L'ETRANGLEMENT SI ON LE RATE.

    Un proxy AJOUTE l'adresse qu'il observe A DROITE de `X-Forwarded-For`. La
    valeur que le client a lui-meme envoyee se retrouve donc A GAUCHE. Lire
    `xff.split(",")[0]` — la faute classique, celle qu'on lit partout — laisse
    l'attaquant declarer une IP fictive differente a chaque requete, et il n'est
    jamais etrangle.

    Prendre la DERNIERE entree est correct quand le proxy ajoute, et donne le
    meme resultat quand il ecrase (une seule valeur : premiere == derniere). Il
    faut donc prendre la derniere dans tous les cas.

    A CONFIRMER une fois sur l'instance deployee, en journalisant l'en-tete brut
    sur une vraie requete : le comportement exact depend de l'hebergeur, et il
    n'est pas affirme ici de memoire.
    """
    entete = request.headers.get("x-forwarded-for")
    if entete:
        entrees = [p.strip() for p in entete.split(",") if p.strip()]
        if entrees:
            return entrees[-1]
    return request.client.host if request.client else "inconnu"


def ip_etranglee(ip: str) -> bool:
    """Vrai si cette adresse a deja trop demande dans l'heure. Consomme un jeton sinon."""
    maintenant = time.monotonic()
    recentes = _demandes_par_ip[ip]
    while recentes and maintenant - recentes[0] > FENETRE_IP_S:
        recentes.popleft()
    if len(recentes) >= DEMANDES_IP_PAR_HEURE:
        return True
    recentes.append(maintenant)
    return False


def reinitialiser_letranglement_ip() -> None:
    """Remet le compteur a zero. Reserve aux tests."""
    _demandes_par_ip.clear()


async def compte_etrangle(db: AsyncSession, user_id: int, usage: str) -> bool:
    """
    Vrai si ce compte a deja trop demande, ou si un envoi est trop recent.

    Le delai de garde compte autant que le plafond horaire : sans lui, un
    double-clic sur « Envoyer » expedie deux courriels, et le premier lien
    devient caduc avant meme d'etre lu.
    """
    depuis_une_heure = maintenant_naif() - timedelta(hours=1)
    combien = (
        await db.execute(
            select(func.count())
            .select_from(EmailToken)
            .where(
                EmailToken.user_id == user_id,
                EmailToken.purpose == usage,
                EmailToken.created_at > depuis_une_heure,
            )
        )
    ).scalar_one()
    if combien >= DEMANDES_PAR_HEURE:
        return True

    tout_recent = (
        await db.execute(
            select(func.count())
            .select_from(EmailToken)
            .where(
                EmailToken.user_id == user_id,
                EmailToken.purpose == usage,
                EmailToken.created_at > maintenant_naif() - DELAI_DE_GARDE,
            )
        )
    ).scalar_one()
    return tout_recent > 0


async def creer_jeton(
    db: AsyncSession,
    user: User,
    usage: str,
    duree: timedelta,
    ip: Optional[str] = None,
) -> str:
    """
    Cree un jeton et rend sa valeur EN CLAIR — la seule fois ou elle existe.

    Seul le hachage part en base. L'appelant met la valeur rendue dans le lien
    du courriel, puis l'oublie.
    """
    jeton = nouveau_jeton()
    db.add(
        EmailToken(
            user_id=user.id,
            purpose=usage,
            token_hash=hachage(jeton),
            expires_at=maintenant_naif() + duree,
            created_at=maintenant_naif(),
            requested_ip=ip,
        )
    )
    await db.commit()
    return jeton


async def consommer_jeton(db: AsyncSession, jeton: str, usage: str) -> Optional[EmailToken]:
    """
    Verifie un jeton et le marque consomme. Rend None si inutilisable.

    LA CLAUSE `purpose` EST L'INVARIANT DE SECURITE DE TOUTE LA TABLE. Sans elle,
    un jeton de verification d'adresse serait accepte par la reinitialisation de
    mot de passe : quiconque recoit un courriel de verification pourrait changer
    le mot de passe du compte. C'est le prix de la table unique, et il se paie
    ici, en une ligne, a chaque lecture.
    """
    ligne = (
        await db.execute(
            select(EmailToken).where(
                EmailToken.token_hash == hachage(jeton),
                EmailToken.purpose == usage,  # <-- l'invariant
                EmailToken.consumed_at.is_(None),
                EmailToken.expires_at > maintenant_naif(),
            )
        )
    ).scalar_one_or_none()

    if ligne is None:
        return None

    ligne.consumed_at = maintenant_naif()
    return ligne


async def invalider_les_autres(db: AsyncSession, user_id: int, usage: str) -> None:
    """
    Consomme tous les autres jetons vivants de ce compte pour cet usage.

    Apres une reinitialisation reussie, un lien plus ancien encore valide serait
    une seconde porte ouverte pendant trente minutes — y compris pour qui aurait
    intercepte le premier courriel.
    """
    await db.execute(
        update(EmailToken)
        .where(
            EmailToken.user_id == user_id,
            EmailToken.purpose == usage,
            EmailToken.consumed_at.is_(None),
        )
        .values(consumed_at=maintenant_naif())
    )
