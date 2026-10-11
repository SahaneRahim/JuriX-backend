"""Le profil (persona) d'un compte.

Le profil choisi a l'entree du site suit l'utilisateur connecte d'un appareil
a l'autre : il est garde sur son compte, en plus du navigateur.

NULL veut dire « aucun choix enregistre ». Un refus de repondre n'est JAMAIS
stocke ici : il vaut pour la visite seulement, et la question revient a la
suivante. D'ou la contrainte, qui n'admet que les quatre profils proposes —
pas « general », la valeur d'une conversation sans profil.

La liste est figee dans la contrainte, comme dans toute migration : elle ne
suit pas app/core/personas.py. Le test de schema verifie que chaque valeur de
PERSONAS y est acceptee, ce qui detecte une divergence. « étudiant » est
ecrit en NFC (U+00E9) : la contrainte compare des octets, et la forme
decomposee d'un copier-coller serait refusee.

Sous PostgreSQL 16, ajouter une colonne nullable ne touche que les
metadonnees, et la contrainte se valide sur une colonne entierement NULL :
l'operation est instantanee, meme en production.

Revision ID: f7a8b9c0d1e2
Revises: e6f7a8b9c0d1
Create Date: 2026-10-11
"""

from typing import Sequence, Union

from alembic import op

revision: str = "f7a8b9c0d1e2"
down_revision: Union[str, None] = "e6f7a8b9c0d1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

PROFILS_DU_COMPTE = ("citoyen", "avocat", "entrepreneur", "\u00e9tudiant")


def upgrade() -> None:
    # Rejouable, comme 013 et 014 : test_categories_013 recule la version par
    # `stamp`, sans rien defaire, puis rejoue tout ce qui suit 013.
    op.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS persona VARCHAR(50)")
    valeurs = ", ".join(f"'{profil}'" for profil in PROFILS_DU_COMPTE)
    op.execute(
        "DO $$ BEGIN "
        "IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_users_persona_connu' "
        "AND conrelid = 'users'::regclass) THEN "
        "ALTER TABLE users ADD CONSTRAINT ck_users_persona_connu "
        f"CHECK (persona IS NULL OR persona IN ({valeurs})); "
        "END IF; END $$"
    )


def downgrade() -> None:
    # Sans garde, comme 011 : perdre une preference ne prive personne de
    # connexion, et le profil sera simplement redemande.
    op.execute("ALTER TABLE users DROP CONSTRAINT IF EXISTS ck_users_persona_connu")
    op.execute("ALTER TABLE users DROP COLUMN IF EXISTS persona")
