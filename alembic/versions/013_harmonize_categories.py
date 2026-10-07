"""Harmonize categories: merge procedures, expand domains, add health and education.

Cette migration harmonise la table categories pour le corpus juridique camerounais :
1. Fusionne Procédure Civile dans Droit Civil et Procédure Civile.
2. Fusionne Procédure Pénale dans Droit Pénal et Procédure Pénale.
3. Élargit Droit de la Famille -> Droit des Personnes, de la Famille et État Civil.
4. Élargit Droit des Affaires et OHADA -> Droit des Affaires, Banque et OHADA.
5. Crée Santé Publique et Sécurité Sanitaire.
6. Crée Éducation, Recherche, Culture et Médias.
7. Met à jour les descriptions et ordonne les 14 catégories canoniques.

Revision ID: d5e6f7a8b9c0
Revises: c4d5e6f7a8b9
Create Date: 2026-10-04
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "d5e6f7a8b9c0"
down_revision: Union[str, None] = "c4d5e6f7a8b9"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# Les 14 catégories canoniques définitives (nom, icone, description)
CANONICAL_CATEGORIES = (
    ("Droit Constitutionnel", "🏛️", "Constitution, révisions, élections, mandats et institutions publiques"),
    ("Droit Administratif", "🏢", "Organisation et fonctionnement des services publics et collectivités"),
    ("Fonction Publique", "👔", "Statut général des agents publics, carrières, nominations et distinctions"),
    ("Droit International", "🌍", "Traités, conventions et accords bilatéraux ou multilatéraux ratifiés"),
    ("Finances Publiques et Fiscalité", "💰", "Budget de l'État, fiscalité, douanes, emprunts publics et accords de prêts"),
    ("Droit Pénal et Procédure Pénale", "⚖️", "Infractions, peines, poursuites judiciaires et justice militaire"),
    ("Droit Civil et Procédure Civile", "👥", "Personnes, obligations, contrats, instances civiles et voies d'exécution"),
    ("Droit des Personnes, de la Famille et État Civil", "👪", "Mariage, filiation, successions, actes d'état civil, CNI et nationalité"),
    ("Droit du Travail et Sécurité Sociale", "🛠️", "Relations de travail, conventions collectives, syndicats et prévoyance sociale"),
    ("Droit des Affaires, Banque et OHADA", "💼", "Sociétés, commerce, secret bancaire, marchés publics et droit OHADA"),
    ("Droit Foncier et Domanial", "🏞️", "Domaine de l'État, titres fonciers, expropriations et cadastre"),
    ("Droit de l'Environnement et des Ressources Naturelles", "🌱", "Environnement, mines, forêts, eau, hydrocarbures et biodiversité"),
    ("Santé Publique et Sécurité Sanitaire", "🏥", "Santé publique, médecine, pharmacie, sécurité sanitaire et sûreté radiologique"),
    ("Éducation, Recherche, Culture et Médias", "🎓", "Enseignement, universités, recherche, culture, patrimoine, médias et langues officielles"),
)

RENAMES = (
    ("droit civil", "Droit Civil et Procédure Civile"),
    ("droit pénal", "Droit Pénal et Procédure Pénale"),
    ("droit de la famille", "Droit des Personnes, de la Famille et État Civil"),
    ("droit des affaires et ohada", "Droit des Affaires, Banque et OHADA"),
)

MERGES = (
    ("procédure civile", "Droit Civil et Procédure Civile"),
    ("procedure civile", "Droit Civil et Procédure Civile"),
    ("procédure pénale", "Droit Pénal et Procédure Pénale"),
    ("procedure penale", "Droit Pénal et Procédure Pénale"),
)


def upgrade() -> None:
    conn = op.get_bind()

    # --- 1. RENOMMER sur place ---
    for old_lower, new_name in RENAMES:
        conn.execute(
            sa.text("""
                UPDATE categories
                SET name = :new
                WHERE lower(name) = :old
                  AND NOT EXISTS (SELECT 1 FROM categories WHERE lower(name) = lower(:new))
            """),
            {"old": old_lower, "new": new_name},
        )

    # --- 2. RE-POINTER les lois des catégories fusionnées ---
    for old_lower, target_name in MERGES:
        conn.execute(
            sa.text("""
                UPDATE laws
                SET category_id = (SELECT id FROM categories WHERE lower(name) = lower(:target))
                WHERE category_id IN (SELECT id FROM categories WHERE lower(name) = :old)
            """),
            {"old": old_lower, "target": target_name},
        )

    # --- 3. SUPPRIMER les catégories fusionnées devenues obsolètes ---
    for old_lower, _ in MERGES:
        conn.execute(
            sa.text("DELETE FROM categories WHERE lower(name) = :old"),
            {"old": old_lower},
        )

    # --- 4. INSERER les nouvelles catégories ou METTRE A JOUR existantes ---
    for order, (name, icon, description) in enumerate(CANONICAL_CATEGORIES):
        conn.execute(
            sa.text("""
                INSERT INTO categories (name, icon, description, display_order)
                SELECT :name, :icon, :description, :order
                WHERE NOT EXISTS (SELECT 1 FROM categories WHERE lower(name) = lower(:name))
            """),
            {"name": name, "icon": icon, "description": description, "order": order},
        )
        conn.execute(
            sa.text("""
                UPDATE categories
                SET icon = :icon, description = :description, display_order = :order
                WHERE lower(name) = lower(:name)
            """),
            {"name": name, "icon": icon, "description": description, "order": order},
        )


def downgrade() -> None:
    pass
