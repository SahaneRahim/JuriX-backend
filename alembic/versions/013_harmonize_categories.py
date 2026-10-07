"""Harmonize categories: merge procedures, expand domains, add health and education.

Cette migration harmonise la table categories pour le corpus juridique camerounais :
1. Fusionne Procédure Civile dans Droit Civil et Procédure Civile.
2. Fusionne Procédure Pénale dans Droit Pénal et Procédure Pénale.
3. Élargit Droit de la Famille -> Droit des Personnes, de la Famille et État Civil.
4. Élargit Droit des Affaires et OHADA -> Droit des Affaires, Banque et OHADA.
5. Crée Santé Publique et Sécurité Sanitaire.
6. Crée Éducation, Recherche, Culture et Médias.
7. Met à jour les descriptions et ordonne les 14 catégories canoniques.

Comme 007, la migration est CONVERGENTE : elle part de n'importe quel etat
plausible (avant 013, 013 a moitie passee, une categorie cible deja creee a la
main) et aboutit aux memes 14 lignes, verifiees a la fin. Chaque ligne est
resolue par `lower(name)`, jamais par identifiant.

L'ordre des etapes n'est pas negociable : `laws_category_id_fkey` n'a pas de
`ON DELETE`, donc les lois sont re-pointees AVANT toute suppression, et les
categories cibles inserees AVANT le re-pointage. La premiere version faisait
l'inverse : une fusion vers une categorie absente ecrivait NULL.

`laws.suggested_categories` n'a pas de cle etrangere : c'est un tableau
d'identifiants. La premiere version supprimait les categories fusionnees sans
y toucher, et 5 lois de jurix_dev gardaient des identifiants disparus (10 et
11). Les suggestions suivent desormais les fusions, et tout identifiant
inconnu en est retire.

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

# Les 14 catégories canoniques définitives (nom, icone, description).
# L'index dans ce tuple est le display_order.
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

# Anciens noms renommes SUR PLACE : l'identifiant, donc les cles etrangeres et
# les suggestions qui le portent, sont preserves. Si la cible existe deja, le
# renommage est saute et la ligne source est traitee comme une fusion.
RENAMES = (
    ("Droit Civil", "Droit Civil et Procédure Civile"),
    ("Droit Pénal", "Droit Pénal et Procédure Pénale"),
    ("Droit de la Famille", "Droit des Personnes, de la Famille et État Civil"),
    ("Droit des Affaires et OHADA", "Droit des Affaires, Banque et OHADA"),
)

# Fusions : les lois et les suggestions qui designaient la gauche designeront
# la droite. Les graphies sans accent couvrent une saisie manuelle.
MERGES = (
    ("Procédure Civile", "Droit Civil et Procédure Civile"),
    ("Procedure Civile", "Droit Civil et Procédure Civile"),
    ("Procédure Pénale", "Droit Pénal et Procédure Pénale"),
    ("Procedure Penale", "Droit Pénal et Procédure Pénale"),
)

# Les deux domaines crees par 013 : le retour arriere les retire.
NOUVEAUX = ("Santé Publique et Sécurité Sanitaire", "Éducation, Recherche, Culture et Médias")

# Les deux domaines fusionnes par 013 : le retour arriere les recree, vides.
PROCEDURES = ("Procédure Pénale", "Procédure Civile")

# Etat laisse par 007, que le retour arriere restaure : (nom, icone, description).
CANONICAL_007 = (
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


def _renommer(conn, ancien: str, nouveau: str) -> None:
    conn.execute(
        sa.text("""
            UPDATE categories SET name = :nouveau
            WHERE lower(name) = lower(:ancien)
              AND NOT EXISTS (SELECT 1 FROM categories WHERE lower(name) = lower(:nouveau))
        """),
        {"ancien": ancien, "nouveau": nouveau},
    )


def _inserer(conn, ordre: int, nom: str, icone: str, description: str) -> None:
    conn.execute(
        sa.text("""
            INSERT INTO categories (name, icon, description, display_order)
            SELECT :nom, :icone, :description, :ordre
            WHERE NOT EXISTS (SELECT 1 FROM categories WHERE lower(name) = lower(:nom))
        """),
        {"nom": nom, "icone": icone, "description": description, "ordre": ordre},
    )


def _rediriger(conn, source: str, cible: str) -> None:
    """Les lois ET les suggestions qui designent `source` designent `cible`."""
    parametres = {"source": source, "cible": cible}
    conn.execute(
        sa.text("""
            UPDATE laws SET category_id = c.id
            FROM categories c, categories s
            WHERE lower(c.name) = lower(:cible) AND lower(s.name) = lower(:source)
              AND laws.category_id = s.id
        """),
        parametres,
    )
    conn.execute(
        sa.text("""
            UPDATE laws SET suggested_categories = array_replace(suggested_categories, s.id, c.id)
            FROM categories c, categories s
            WHERE lower(c.name) = lower(:cible) AND lower(s.name) = lower(:source)
              AND s.id = ANY(laws.suggested_categories)
        """),
        parametres,
    )


# Les noms gardes, mis en minuscules PAR POSTGRES. Un `.lower()` Python ne
# s'accorderait avec `lower(name)` que sous certaines locales : sous une locale
# C, « Éducation » garde sa majuscule cote base, et la ligne serait supprimee.
_HORS_DE = "lower(name) <> ALL(SELECT lower(n) FROM unnest(CAST(:noms AS text[])) AS n)"


_PARMI = "lower(name) = ANY(SELECT lower(n) FROM unnest(CAST(:noms AS text[])) AS n)"


def _supprimer(conn, condition: str, noms: list) -> None:
    """
    Supprime les lignes qui remplissent `condition` : les lois qui y pointent
    encore passent d'abord a NULL (en inventer la categorie serait un second
    bug), et les lignes sont supprimees, prouvees non referencees.
    """
    conn.execute(
        sa.text(f"""
            UPDATE laws SET category_id = NULL
            WHERE category_id IN (SELECT id FROM categories WHERE {condition})
        """),
        {"noms": list(noms)},
    )
    conn.execute(sa.text(f"DELETE FROM categories WHERE {condition}"), {"noms": list(noms)})


def _nettoyer_les_suggestions(conn) -> None:
    """
    Retire des suggestions les identifiants inconnus et les doublons, en
    gardant l'ordre : la premiere suggestion est le domaine retenu par le
    classement. Seules les lignes qui changent sont reecrites.
    """
    conn.execute(sa.text("""
        WITH propres AS (
            SELECT l.id, ARRAY(
                SELECT u.x
                FROM unnest(l.suggested_categories) WITH ORDINALITY AS u(x, rang)
                WHERE u.x IN (SELECT id FROM categories)
                GROUP BY u.x
                ORDER BY min(u.rang)
            ) AS suggestions
            FROM laws l
            WHERE l.suggested_categories IS NOT NULL
        )
        UPDATE laws SET suggested_categories = propres.suggestions
        FROM propres
        WHERE laws.id = propres.id
          AND laws.suggested_categories IS DISTINCT FROM propres.suggestions
    """))


def _mettre_a_jour(conn, categories) -> None:
    for ordre, (nom, icone, description) in enumerate(categories):
        conn.execute(
            sa.text("""
                UPDATE categories SET icon = :icone, description = :description, display_order = :ordre
                WHERE lower(name) = lower(:nom)
            """),
            {"nom": nom, "icone": icone, "description": description, "ordre": ordre},
        )


def _verifier_les_lignes(conn, attendus: list, *, exactement: bool) -> None:
    """Les lignes `attendus` sont la ; avec `exactement`, il n'y en a pas d'autre."""
    voulus = set(conn.execute(
        sa.text("SELECT lower(n) FROM unnest(CAST(:noms AS text[])) AS n"),
        {"noms": list(attendus)},
    ).scalars().all())
    presents = set(
        conn.execute(sa.text("SELECT lower(name) FROM categories")).scalars().all()
    )
    manquantes = sorted(voulus - presents)
    en_trop = sorted(presents - voulus) if exactement else []
    if manquantes or en_trop:
        raise RuntimeError(
            f"categories : lignes manquantes {manquantes}, lignes en trop {en_trop}"
        )


def _verifier_les_references(conn) -> None:
    """Aucun identifiant orphelin, ni en cle etrangere ni en suggestion."""
    orphelines = conn.execute(sa.text("""
        SELECT count(*) FROM laws l LEFT JOIN categories c ON c.id = l.category_id
        WHERE l.category_id IS NOT NULL AND c.id IS NULL
    """)).scalar_one()
    if orphelines:
        raise RuntimeError(f"{orphelines} loi(s) pointent sur une categorie inexistante")
    suggestions = conn.execute(sa.text("""
        SELECT count(*) FROM laws l, unnest(l.suggested_categories) AS x
        WHERE x NOT IN (SELECT id FROM categories)
    """)).scalar_one()
    if suggestions:
        raise RuntimeError(f"{suggestions} suggestion(s) designent une categorie inexistante")


def upgrade() -> None:
    conn = op.get_bind()

    # --- 1. RENOMMER sur place ---------------------------------------------
    for ancien, nouveau in RENAMES:
        _renommer(conn, ancien, nouveau)

    # --- 2. INSERER les domaines manquants, AVANT tout re-pointage ---------
    for ordre, (nom, icone, description) in enumerate(CANONICAL_CATEGORIES):
        _inserer(conn, ordre, nom, icone, description)

    # --- 3 et 4. RE-POINTER les lois et FAIRE SUIVRE les suggestions --------
    # Les renommages sont repris ici : un renommage saute (cible deja
    # presente) laisse une ligne source a fusionner.
    for source, cible in (*RENAMES, *MERGES):
        _rediriger(conn, source, cible)

    # --- 5. SUPPRIMER ce qui est hors des 14 --------------------------------
    _supprimer(conn, _HORS_DE, [nom for nom, _, _ in CANONICAL_CATEGORIES])

    # --- 6. NETTOYER les suggestions ----------------------------------------
    _nettoyer_les_suggestions(conn)

    # --- 7. icones, descriptions, ordre d'affichage -------------------------
    _mettre_a_jour(conn, CANONICAL_CATEGORIES)

    # --- 8. CONTROLES : exactement les 14, aucune reference orpheline -------
    _verifier_les_lignes(conn, [nom for nom, _, _ in CANONICAL_CATEGORIES], exactement=True)
    _verifier_les_references(conn)


def downgrade() -> None:
    """
    Defait ce que 013 a fait, et rien d'autre, avec deux pertes assumees
    plutot qu'inventees :

    - les lois rangees en Santé ou en Éducation repassent a NULL : 007 n'a
      aucune ligne pour elles, et le reclassement les rattrape ;
    - « Procédure Pénale » et « Procédure Civile » reviennent VIDES : apres la
      fusion, rien ne distingue une loi de procedure d'une loi de fond.

    Sur une base qui ne porte AUCUN nom propre a 013, rien n'est fait : le
    harnais de test fabrique un etat anterieur a 007, qui doit arriver intact
    au retour arriere de 007. Sinon, seules les lignes creees ou renommees par
    013 sont retirees ou renommees ; les autres gardent leur identifiant, et
    celles qui portent un nom de 007 retrouvent ses descriptions.
    """
    conn = op.get_bind()

    noms_de_013 = [cible for _, cible in RENAMES] + list(NOUVEAUX)
    if not conn.execute(
        sa.text(f"SELECT count(*) FROM categories WHERE {_PARMI}"), {"noms": noms_de_013}
    ).scalar_one():
        return

    # --- 1. Santé et Éducation : lois a NULL, lignes retirees ---------------
    _supprimer(conn, _PARMI, list(NOUVEAUX))

    # --- 2. Les 4 anciens noms reviennent, identifiants preserves -----------
    for ancien, nouveau in RENAMES:
        _renommer(conn, nouveau, ancien)

    # --- 3. Les deux procedures sont recreees, vides ------------------------
    for ordre, (nom, icone, description) in enumerate(CANONICAL_007):
        if nom in PROCEDURES:
            _inserer(conn, ordre, nom, icone, description)

    # --- 4. Suggestions, descriptions de 007, controles ---------------------
    _nettoyer_les_suggestions(conn)
    _mettre_a_jour(conn, CANONICAL_007)
    _verifier_les_lignes(conn, list(PROCEDURES), exactement=False)
    _verifier_les_references(conn)
