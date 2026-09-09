"""Jetons a usage unique envoyes par courriel

Une seule table pour les deux usages — verification d'adresse et
reinitialisation de mot de passe. Le cycle de vie est identique (creer, hacher,
expirer, consommer, purger) et les differences tiennent en deux lignes de
politique. Le risque que cela introduit — qu'un jeton de verification soit
accepte par la route de reinitialisation, donc une prise de controle de compte —
est ferme par la clause `purpose` de chaque lecture, par la contrainte CHECK
ci-dessous, et par un test dedie.

Trois index, aucun decoratif :
  - `token_hash` UNIQUE : c'est l'index de recherche. Le jeton presente est
    hache puis cherche ici ; sans index unique, chaque validation ferait un
    parcours complet, et rien n'empecherait deux lignes de porter la meme
    empreinte.
  - `(user_id, purpose)` : sert l'etranglement — « combien de demandes de ce
    type ce compte a-t-il faites depuis une heure ? ».
  - `expires_at` : sert la purge.

ON DELETE CASCADE sur `user_id` : sans lui, supprimer un compte violerait la cle
etrangere et `DELETE /admin/users/{id}` repondrait 500. Cette route supprime
deja les conversations explicitement, pour une raison voisine.

Revision ID: b3c4d5e6f7a8
Revises: a2b3c4d5e6f7
"""

import sqlalchemy as sa

from alembic import op

revision = "b3c4d5e6f7a8"
down_revision = "a2b3c4d5e6f7"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "email_tokens",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("user_id", sa.Integer(), nullable=False),
        sa.Column("purpose", sa.String(length=32), nullable=False),
        # 64 caracteres : la longueur d'un sha256 en hexadecimal.
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("consumed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        # 45 caracteres : une adresse IPv6 en notation textuelle.
        sa.Column("requested_ip", sa.String(length=45), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["user_id"], ["users.id"], ondelete="CASCADE"),
        sa.CheckConstraint(
            "purpose IN ('verify_email', 'reset_password')",
            name="ck_email_tokens_purpose",
        ),
    )
    op.create_index("ix_email_tokens_id", "email_tokens", ["id"])
    op.create_index("ix_email_tokens_token_hash", "email_tokens", ["token_hash"], unique=True)
    op.create_index("ix_email_tokens_user_id", "email_tokens", ["user_id"])
    op.create_index("ix_email_tokens_user_purpose", "email_tokens", ["user_id", "purpose"])
    op.create_index("ix_email_tokens_expires_at", "email_tokens", ["expires_at"])


def downgrade() -> None:
    # Aucun message d'echec particulier, contrairement a la migration 010 :
    # perdre des jetons non consommes ne prive personne de connexion. Au pire,
    # une demande de reinitialisation en cours doit etre refaite.
    op.drop_index("ix_email_tokens_expires_at", table_name="email_tokens")
    op.drop_index("ix_email_tokens_user_purpose", table_name="email_tokens")
    op.drop_index("ix_email_tokens_user_id", table_name="email_tokens")
    op.drop_index("ix_email_tokens_token_hash", table_name="email_tokens")
    op.drop_index("ix_email_tokens_id", table_name="email_tokens")
    op.drop_table("email_tokens")
