"""Canonical legal domains for categories.

La table `categories` melangeait des domaines juridiques (Droit Fiscal) et des
types de texte (Lois, Ordonnances, Decrets). Un decret fiscal n'avait donc
aucune ligne correcte ou aller. Cette migration ramene la table a 14 domaines,
et rien que des domaines : le type de texte vit deja dans `laws.type`.

L'ordre des etapes n'est pas negociable. `laws_category_id_fkey` n'a pas de
`ON DELETE`, donc toute suppression avant l'etape 4 leverait ForeignKeyViolation.
Chaque etape resout ses lignes par `lower(name)`, jamais par identifiant : c'est
ce qui rend la migration sure quelle que soit la base rencontree.

Revision ID: d9e0f1a2b3c4
Revises: c8d9e0f1a2b3
Create Date: 2026-09-03
"""

from typing import Sequence, Union

import sqlalchemy as sa

from alembic import op

revision: str = "d9e0f1a2b3c4"
down_revision: Union[str, None] = "c8d9e0f1a2b3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


# (nom, icone, description). L'index dans ce tuple est le display_order.
CANONICAL: tuple[tuple[str, str, str], ...] = (
    ("Droit Constitutionnel", "🏛️", "Constitution, institutions et élections"),
    ("Droit Administratif", "🏢", "Organisation et fonctionnement des services publics"),
    ("Fonction Publique", "👔", "Statut des agents publics, carrières et distinctions"),
    ("Droit International", "🌍", "Traités, conventions et accords ratifiés"),
    ("Finances Publiques et Fiscalité", "💰", "Budget de l'État, impôts, douanes et emprunts"),
    ("Droit Pénal", "⚖️", "Infractions et peines"),
    ("Procédure Pénale", "🔍", "Instruction, poursuite et jugement en matière pénale"),
    ("Droit Civil", "👥", "Personnes, biens, obligations et contrats"),
    ("Procédure Civile", "📋", "Instances civiles et voies d'exécution"),
    ("Droit de la Famille", "👪", "Mariage, filiation, successions et état civil"),
    ("Droit du Travail et Sécurité Sociale", "🛠️", "Relations de travail et prévoyance sociale"),
    ("Droit des Affaires et OHADA", "💼", "Sociétés, commerce et actes uniformes OHADA"),
    ("Droit Foncier et Domanial", "🏞️", "Domaine de l'État, titres fonciers et expropriation"),
    ("Droit de l'Environnement et des Ressources Naturelles", "🌱",
     "Environnement, mines, forêts, eau et hydrocarbures"),
)

# Anciens noms renommes SUR PLACE : l'identifiant et les cles etrangeres qui le
# referencent sont donc preserves. C'est la raison d'etre de l'etape 1.
RENAMES: tuple[tuple[str, str], ...] = (
    ("droit fiscal", "Finances Publiques et Fiscalité"),
    ("droit du travail", "Droit du Travail et Sécurité Sociale"),
    ("droit des affaires", "Droit des Affaires et OHADA"),
    ("droit foncier", "Droit Foncier et Domanial"),
    ("droit de l'environnement", "Droit de l'Environnement et des Ressources Naturelles"),
)

# Fusions : les lois qui pointaient sur la gauche pointeront sur la droite.
MERGES: tuple[tuple[str, str], ...] = (
    ("droit commercial", "Droit des Affaires et OHADA"),
    ("droit ohada", "Droit des Affaires et OHADA"),
    ("droit commercial ohada", "Droit des Affaires et OHADA"),
    ("lois internationales ratifiées", "Droit International"),
    ("lois internationales ratifiees", "Droit International"),
)


def upgrade() -> None:
    conn = op.get_bind()

    # --- 0. DEDUPE ---------------------------------------------------------
    # Aucune contrainte d'unicite sur categories.name aujourd'hui. Les doublons
    # sont fusionnes sur le plus petit identifiant AVANT tout le reste, sinon
    # l'index unique de l'etape 6 echouerait.
    conn.execute(sa.text("""
        WITH keep AS (
            SELECT lower(name) AS key, min(id) AS keep_id
            FROM categories GROUP BY lower(name) HAVING count(*) > 1
        )
        UPDATE laws l SET category_id = keep.keep_id
        FROM categories c JOIN keep ON keep.key = lower(c.name)
        WHERE l.category_id = c.id AND c.id <> keep.keep_id
    """))
    conn.execute(sa.text("""
        DELETE FROM categories c USING (
            SELECT lower(name) AS key, min(id) AS keep_id FROM categories GROUP BY lower(name)
        ) keep
        WHERE lower(c.name) = keep.key AND c.id <> keep.keep_id
    """))

    # --- 1. RENOMMER sur place --------------------------------------------
    # Saute le renommage si la cible existe deja sous son nom canonique : la
    # ligne source devient alors un doublon traite comme une fusion en etape 3.
    for old_lower, new_name in RENAMES:
        conn.execute(
            sa.text("""
                UPDATE categories SET name = :new
                WHERE lower(name) = :old
                  AND NOT EXISTS (SELECT 1 FROM categories WHERE lower(name) = lower(:new))
            """),
            {"old": old_lower, "new": new_name},
        )

    # --- 2. INSERER les domaines manquants ---------------------------------
    for order, (name, icon, description) in enumerate(CANONICAL):
        conn.execute(
            sa.text("""
                INSERT INTO categories (name, icon, description, display_order)
                SELECT :name, :icon, :description, :order
                WHERE NOT EXISTS (SELECT 1 FROM categories WHERE lower(name) = lower(:name))
            """),
            {"name": name, "icon": icon, "description": description, "order": order},
        )

    # --- 3. RE-POINTER les lois AVANT toute suppression --------------------
    for old_lower, target in MERGES:
        conn.execute(
            sa.text("""
                UPDATE laws SET category_id = (SELECT id FROM categories WHERE lower(name) = lower(:target))
                WHERE category_id IN (SELECT id FROM categories WHERE lower(name) = :old)
            """),
            {"old": old_lower, "target": target},
        )

    # Les lois rangees dans un TYPE de document (Lois, Decrets, Arretes...)
    # repassent a NULL : l'ancienne valeur ne portait aucune information de
    # domaine, en inventer une serait un second bug. Le script
    # scripts/maintenance/reclassify_domains.py les rattrape ensuite.
    canonical_lower = [name.lower() for name, _, _ in CANONICAL]
    conn.execute(
        sa.text("""
            UPDATE laws SET category_id = NULL
            WHERE category_id IN (
                SELECT id FROM categories WHERE lower(name) <> ALL(:keep)
            )
        """),
        {"keep": canonical_lower},
    )

    # --- 4. SUPPRIMER les lignes hors des 14, prouvees non referencees ------
    conn.execute(
        sa.text("DELETE FROM categories WHERE lower(name) <> ALL(:keep)"),
        {"keep": canonical_lower},
    )

    # --- 5. display_order / icon / description -----------------------------
    for order, (name, icon, description) in enumerate(CANONICAL):
        conn.execute(
            sa.text("""
                UPDATE categories SET display_order = :order, icon = :icon, description = :description
                WHERE lower(name) = lower(:name)
            """),
            {"name": name, "icon": icon, "description": description, "order": order},
        )

    # --- 6. UNICITE sur lower(name) ----------------------------------------
    # C'est ce qui rend `resolve_domain_id` totale : un nom canonique designe
    # au plus une ligne, pour toujours.
    op.execute("CREATE UNIQUE INDEX IF NOT EXISTS uq_categories_name_lower ON categories (lower(name))")

    # --- 7. ASSERT ---------------------------------------------------------
    count = conn.execute(sa.text("SELECT count(*) FROM categories")).scalar_one()
    if count != len(CANONICAL):
        raise RuntimeError(
            f"categories devrait contenir {len(CANONICAL)} lignes apres migration, en contient {count}"
        )
    orphans = conn.execute(sa.text("""
        SELECT count(*) FROM laws l LEFT JOIN categories c ON c.id = l.category_id
        WHERE l.category_id IS NOT NULL AND c.id IS NULL
    """)).scalar_one()
    if orphans:
        raise RuntimeError(f"{orphans} loi(s) pointent sur une categorie inexistante")


def downgrade() -> None:
    """
    Defait l'index et les renommages. Ne ressuscite PAS les lignes fusionnees
    ou supprimees : un downgrade qui invente des lignes est pire que celui qui
    admet la perte.
    """
    op.execute("DROP INDEX IF EXISTS uq_categories_name_lower")
    conn = op.get_bind()
    for old_lower, new_name in RENAMES:
        original = old_lower.title().replace("L'", "l'").replace(" De ", " de ").replace(" Du ", " du ")
        conn.execute(
            sa.text("UPDATE categories SET name = :old WHERE lower(name) = lower(:new)"),
            {"old": original, "new": new_name},
        )
