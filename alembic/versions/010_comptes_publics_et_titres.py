"""Comptes publics : identite Google, mot de passe facultatif, titres de conversation

Trois changements, tous exiges par l'inscription publique et la liste des
conversations :

1. `users.google_sub` — le claim `sub` d'un jeton d'identite Google. C'est le
   SEUL identifiant stable d'un compte Google : l'adresse e-mail, elle, peut
   changer. Unique et nullable : PostgreSQL autorise plusieurs NULL sur un index
   unique, donc les comptes par mot de passe cohabitent sans contrainte.

2. `users.hashed_password` devient NULLABLE — un compte cree par Google n'a pas
   de mot de passe. Une empreinte sentinelle aurait menti au schema : « ce
   compte a-t-il un mot de passe ? » serait devenu insoluble en SQL. La
   contrainte CHECK garantit qu'il reste toujours au moins une porte d'entree.

3. `conversations.title` et l'index (user_id, updated_at DESC) — le panneau
   lateral du chat. Le titre est ECRIT une fois, a la premiere interaction,
   plutot que derive a la lecture : la requete de liste reste ainsi sans
   jointure, et un titre stocke ne change pas tout seul le jour ou le premier
   message disparait.

Revision ID: a2b3c4d5e6f7
Revises: f1a2b3c4d5e6
"""

import sqlalchemy as sa

from alembic import op

revision = "a2b3c4d5e6f7"
down_revision = "f1a2b3c4d5e6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("users", sa.Column("google_sub", sa.String(length=255), nullable=True))
    op.create_index("ix_users_google_sub", "users", ["google_sub"], unique=True)

    op.alter_column(
        "users", "hashed_password", existing_type=sa.String(length=255), nullable=True
    )
    op.create_check_constraint(
        "ck_users_au_moins_une_identite",
        "users",
        "hashed_password IS NOT NULL OR google_sub IS NOT NULL",
    )

    op.add_column("conversations", sa.Column("title", sa.String(length=120), nullable=True))
    # Index composite, et l'ordre des colonnes compte : la liste filtre sur
    # user_id puis trie sur updated_at decroissant. C'est lui qui rend
    # GET /rag/conversations servable sans jointure ni tri en memoire.
    op.create_index(
        "idx_conversations_user_updated",
        "conversations",
        ["user_id", sa.text("updated_at DESC")],
    )


def downgrade() -> None:
    op.drop_index("idx_conversations_user_updated", table_name="conversations")
    op.drop_column("conversations", "title")

    # Remettre NOT NULL echouerait de toute facon s'il reste des comptes sans
    # mot de passe. Autant echouer AVANT, avec un message qui dit quoi faire :
    # l'erreur brute de PostgreSQL ne nomme ni la cause ni le remede.
    restants = (
        op.get_bind()
        .execute(sa.text("SELECT count(*) FROM users WHERE hashed_password IS NULL"))
        .scalar()
    )
    if restants:
        raise RuntimeError(
            f"{restants} compte(s) sans mot de passe (crees par Google). Revenir en "
            "arriere leur retirerait toute possibilite de connexion. Supprimez-les "
            "ou donnez-leur un mot de passe avant de rejouer ce downgrade."
        )

    op.drop_constraint("ck_users_au_moins_une_identite", "users", type_="check")
    op.alter_column(
        "users", "hashed_password", existing_type=sa.String(length=255), nullable=False
    )
    op.drop_index("ix_users_google_sub", table_name="users")
    op.drop_column("users", "google_sub")
