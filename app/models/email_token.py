"""
Jetons a usage unique envoyes par courriel.

DEUX USAGES, UNE TABLE. `verify_email` prouve que l'adresse existe ;
`reset_password` autorise a en changer le mot de passe. Le cycle de vie est
strictement le meme — creer, hacher, expirer, consommer, purger — et les
differences tiennent en deux lignes de politique (duree de vie, effet de la
consommation). Deux tables auraient impose deux migrations, deux modeles, deux
boucles de purge, et un `UNION` des qu'il faut compter les envois du jour.

LE RISQUE DE LA TABLE UNIQUE, et comment il est ferme. Qu'un jeton
`verify_email` soit accepte par la route de reinitialisation serait une prise de
controle complete du compte : n'importe qui recevant un courriel de verification
pourrait changer le mot de passe. Chaque lecture porte donc une clause
`purpose == <celui attendu>`, et un test dedie
(`test_jeton_de_verification_refuse_par_la_reinitialisation`) echoue si elle
disparait. Une clause et un test coutent moins cher qu'une seconde table.

POURQUOI SHA256 ET PAS BCRYPT — quelqu'un voudra « ameliorer » ceci. Le jeton
est un `secrets.token_urlsafe(32)`, soit 256 bits tires au hasard : il n'est pas
devinable, et une fonction de derivation lente ne protege donc de rien ici. Elle
couterait en revanche l'INDEXABILITE : bcrypt sale chaque ligne, si bien que
retrouver un jeton imposerait de parcourir toute la table et de verifier chaque
enregistrement un par un. sha256 est deterministe, donc indexable, donc O(1). Le
sel n'a de sens que contre les mots de passe faibles ; il n'existe pas de jeton
faible ici.

CE QUI EST STOCKE EST LE HACHAGE, jamais le jeton. Une lecture de la base — une
sauvegarde qui traine, un acces en lecture seule accorde a un tiers — ne doit
pas livrer de quoi prendre un compte.
"""

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

from sqlalchemy import (
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
)
from sqlalchemy.orm import relationship

from app.core.database import Base

# Les deux seuls usages admis. La contrainte CHECK en base porte la meme liste :
# une valeur inventee doit echouer a l'insertion, pas se retrouver dans une
# table ou personne ne la cherchera jamais.
VERIFICATION_ADRESSE = "verify_email"
REINITIALISATION_MOT_DE_PASSE = "reset_password"
USAGES = (VERIFICATION_ADRESSE, REINITIALISATION_MOT_DE_PASSE)

# Durees de vie. La reinitialisation est plus courte : elle donne acces au
# compte, la verification ne fait que confirmer une adresse.
DUREE_REINITIALISATION = timedelta(minutes=30)
DUREE_VERIFICATION = timedelta(hours=24)


def nouveau_jeton() -> str:
    """Jeton en clair, a mettre dans le lien et a ne jamais stocker."""
    return secrets.token_urlsafe(32)


def hachage(jeton: str) -> str:
    """Empreinte a stocker. Deterministe, donc indexable — voir le docstring du module."""
    return hashlib.sha256(jeton.encode("utf-8")).hexdigest()


def maintenant_naif() -> datetime:
    """UTC sans fuseau, comme toutes les colonnes DateTime du schema."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class EmailToken(Base):
    """Un jeton a usage unique, envoye par courriel."""

    __tablename__ = "email_tokens"

    id = Column(Integer, primary_key=True, index=True)
    # ON DELETE CASCADE : sans lui, supprimer un compte violerait la cle
    # etrangere et la route d'administration repondrait 500. Elle supprime deja
    # les conversations explicitement, pour une raison voisine.
    user_id = Column(
        Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    purpose = Column(String(32), nullable=False)
    token_hash = Column(String(64), nullable=False, unique=True, index=True)
    expires_at = Column(DateTime, nullable=False)
    consumed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=maintenant_naif)
    # 45 caracteres : la longueur d'une adresse IPv6 en notation textuelle.
    requested_ip = Column(String(45), nullable=True)

    user = relationship("User", lazy="raise")

    __table_args__ = (
        CheckConstraint(
            "purpose IN ('verify_email', 'reset_password')",
            name="ck_email_tokens_purpose",
        ),
        # Sert l'etranglement par adresse : « combien de demandes de ce type ce
        # compte a-t-il faites depuis une heure ? »
        Index("ix_email_tokens_user_purpose", "user_id", "purpose"),
        # Sert la purge des jetons expires.
        Index("ix_email_tokens_expires_at", "expires_at"),
    )

    def est_utilisable(self) -> bool:
        return self.consumed_at is None and self.expires_at > maintenant_naif()

    def __repr__(self) -> str:
        # Ni le hachage ni le jeton : un repr finit dans les journaux.
        return f"<EmailToken id={self.id} purpose={self.purpose!r} user_id={self.user_id}>"
