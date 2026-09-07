"""
Derivation d'un `username` a partir d'une adresse e-mail.

POURQUOI CE MODULE EXISTE. L'inscription publique ne demande que trois choses :
un nom, une adresse, un mot de passe. Or `users.username` est NOT NULL, unique,
et long d'au moins trois caracteres. Il faut donc le fabriquer.

LE PIEGE, MESURE ET NON SUPPOSE. `UserResponse` herite de `UserBase`, donc son
validateur `validate_username` s'execute AUSSI A LA SORTIE, quand la route
serialise sa reponse. Verifie en executant le schema :

    'jeandupont'   -> accepte
    'jean-dupont'  -> accepte
    'jean.dupont'  -> REFUSE  (Username must be alphanumeric…)
    'ab'           -> REFUSE  (String should have at least 3 characters)

Un username portant un point franchirait l'insert en base puis ferait echouer
la serialisation de la reponse : l'inscription repondrait 500 apres avoir cree
le compte. Le nettoyage ci-dessous garantit un resultat qui satisfait ce
validateur — alphanumerique, ou porteur d'un `-`/`_`, et long d'au moins trois
caracteres.

Author: JuriX Team
"""

import logging
import re
import secrets
import unicodedata
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User

logger = logging.getLogger(__name__)

# Longueur maximale de la colonne, moins la place du suffixe de desambiguisation.
_LONGUEUR_MAX_BASE = 90
_LONGUEUR_MIN = 3

# Tout ce qui n'est pas une lettre ASCII, un chiffre, un tiret ou un souligne.
# Le POINT en fait partie, et c'est deliberé : « jean.dupont » est refuse par le
# validateur de sortie de UserBase.
_INTERDITS = re.compile(r"[^a-z0-9_-]+")

# Nombre de variantes numerotees essayees avant de basculer sur l'aleatoire.
_MAX_ESSAIS = 50


def base_username(email: str) -> str:
    """
    Ramene une adresse e-mail a un identifiant utilisable.

    « Jean.Dupont+spam@Example.CM » rend « jeandupont+spam » deplie puis
    nettoye, soit « jeandupontspam ». Les accents sont deplies plutot que
    supprimes : « josé » donne « jose », pas « jos ».
    """
    partie_locale = (email or "").split("@")[0].strip().lower()

    # NFKD separe la lettre de son accent, l'encodage ASCII jette l'accent seul.
    deplie = (
        unicodedata.normalize("NFKD", partie_locale)
        .encode("ascii", "ignore")
        .decode("ascii")
    )

    nettoye = _INTERDITS.sub("", deplie).strip("-_")[:_LONGUEUR_MAX_BASE]

    if len(nettoye) < _LONGUEUR_MIN:
        # `min_length=3` cote schema. Completer plutot que refuser : une adresse
        # comme « ab@x.cm » est parfaitement valide, ce n'est pas a l'utilisateur
        # de payer notre contrainte de colonne.
        nettoye = (nettoye + "usr")[:_LONGUEUR_MIN]

    return nettoye


async def unique_username(db: AsyncSession, base: str) -> str:
    """
    Rend `base`, ou la premiere variante libre.

    La boucle est BORNEE, et le repli est aleatoire : sur un `base` tres
    demande, essayer les entiers un par un jusqu'a l'infini bloquerait la
    requete. Apres cinquante tentatives, un suffixe aleatoire tranche en un
    coup.

    Ce controle ne remplace PAS la contrainte unique de la base : deux
    inscriptions simultanees peuvent le franchir toutes les deux. L'appelant
    doit donc encore traiter l'`IntegrityError`.
    """
    candidat = base
    for tour in range(1, _MAX_ESSAIS + 1):
        existe = (
            await db.execute(select(User.id).where(User.username == candidat))
        ).scalar_one_or_none()
        if existe is None:
            return candidat
        candidat = f"{base}-{tour + 1}"

    aleatoire = f"{base}-{secrets.token_hex(4)}"
    logger.warning(
        "Username '%s' sature apres %d essais, repli aleatoire", base, _MAX_ESSAIS
    )
    return aleatoire


async def username_pour_email(db: AsyncSession, email: str) -> str:
    """Raccourci : de l'adresse au username libre."""
    return await unique_username(db, base_username(email))


def normaliser_email(email: Optional[str]) -> str:
    """
    Forme sous laquelle une adresse est ECRITE en base.

    `_authenticate` cherche avec `email.lower().strip()`, mais
    `admin.create_user` ecrivait l'adresse telle quelle : un compte cree avec
    « Jean@X.cm » ne pouvait donc JAMAIS se connecter. Cette fonction existe
    pour que l'ecriture et la lecture parlent enfin de la meme chose.
    """
    return (email or "").strip().lower()
