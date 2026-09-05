"""
Tests de la resolution d'un domaine vers `categories.id`.

C'est ici que vivait le bug : le pipeline ecrivait dans `laws.category_id` un
ENTIER qui etait une position dans un dictionnaire Python, pas une cle etrangere.
Les anciens tests ne pouvaient pas l'attraper — aucun n'ouvrait la base, alors
que le defaut vivait precisement dans la jointure `laws -> categories`.

Chaque test ici va jusqu'a la base et verifie le NOM obtenu par jointure.

Usage:
    pytest tests/test_services/test_category_resolution.py -v
"""

import pytest
from sqlalchemy import select, text

from app.models.law import Category, Law
from app.services.category_resolver import (
    UnknownDomainError,
    load_domain_map,
    resolve_domain_id,
    try_resolve_domain_id,
)
from app.services.legal_domain_classifier import CANONICAL_DOMAINS


@pytest.fixture
def scrambled_categories(sync_db_session):
    """
    Seme les 14 domaines avec des identifiants VOLONTAIREMENT desordonnes.

    Les identifiants demarrent a 1000 et les domaines sont inseres dans l'ordre
    INVERSE de CANONICAL_DOMAINS. Tout code qui resoudrait un domaine par sa
    position — la cause exacte du bug d'origine — designerait donc
    systematiquement le mauvais domaine, et echoue des la premiere assertion.
    """
    sync_db_session.execute(text("DELETE FROM laws"))
    sync_db_session.execute(text("DELETE FROM categories"))
    sync_db_session.execute(text("SELECT setval('categories_id_seq', 1000, true)"))
    for order, name in enumerate(reversed(CANONICAL_DOMAINS)):
        sync_db_session.add(Category(name=name, display_order=order))
    sync_db_session.commit()
    return sync_db_session


class TestResolveByName:
    def test_every_canonical_domain_resolves(self, scrambled_categories):
        for domain in CANONICAL_DOMAINS:
            assert resolve_domain_id(scrambled_categories, domain) > 0

    def test_resolution_is_by_name_not_by_position(self, scrambled_categories):
        """Le nom lu en base doit etre celui demande, malgre l'ordre inverse."""
        for domain in CANONICAL_DOMAINS:
            resolved = resolve_domain_id(scrambled_categories, domain)
            name = scrambled_categories.execute(
                select(Category.name).where(Category.id == resolved)
            ).scalar_one()
            assert name == domain

    def test_case_insensitive(self, scrambled_categories):
        assert resolve_domain_id(scrambled_categories, "droit pénal") == resolve_domain_id(
            scrambled_categories, "Droit Pénal"
        )

    def test_unknown_domain_raises_and_lists_available(self, scrambled_categories):
        with pytest.raises(UnknownDomainError) as excinfo:
            resolve_domain_id(scrambled_categories, "Droit Martien")
        assert "Droit Martien" in str(excinfo.value)
        assert "Droit Pénal" in str(excinfo.value)

    def test_try_resolve_returns_none_instead_of_raising(self, scrambled_categories):
        assert try_resolve_domain_id(scrambled_categories, "Droit Martien") is None

    def test_domain_map_covers_all_canonical_domains(self, scrambled_categories):
        mapping = load_domain_map(scrambled_categories)
        for domain in CANONICAL_DOMAINS:
            assert domain.lower() in mapping


class TestPipelineWritesTheRightCategory:
    """
    Le test decisif : faire tourner le classement puis lire le NOM par jointure.
    Jamais l'entier.
    """

    CASES = [
        ("Loi N°2015/019 portant Loi de finances pour l'exercice 2016",
         "Finances Publiques et Fiscalité"),
        ("Loi N°2023/014 portant Code Minier",
         "Droit de l'Environnement et des Ressources Naturelles"),
        ("Décret N°2018/420 portant nomination du Secrétaire Général", "Fonction Publique"),
        ("Loi portant Code de Procédure Pénale", "Procédure Pénale"),
        ("Décret ratifiant l'accord de prêt avec la BAD", "Finances Publiques et Fiscalité"),
    ]

    @pytest.mark.parametrize("title,expected", CASES)
    def test_law_joins_to_the_expected_category_name(
        self, scrambled_categories, title, expected
    ):
        from app.tasks.process_law import _classify_category

        law = Law(reference=f"REF-{abs(hash(title)) % 100000}", title=title, type="loi",
                  content="Contenu du document.", language="fr", status="published")
        scrambled_categories.add(law)
        scrambled_categories.commit()

        result = _classify_category(law.title, law.content, law.type)
        law.category_id = result["category_id"]
        scrambled_categories.commit()

        joined = scrambled_categories.execute(
            select(Category.name).join(Law, Law.category_id == Category.id).where(Law.id == law.id)
        ).scalar_one()
        assert joined == expected

    def test_pipeline_does_not_overwrite_an_existing_category(self, scrambled_categories):
        """
        L'administrateur choisit une categorie a l'upload et le pipeline
        l'ecrasait quelques secondes plus tard.
        """
        from app.tasks.process_law import _update_law_metadata

        chosen = resolve_domain_id(scrambled_categories, "Droit Civil")
        proposed = resolve_domain_id(scrambled_categories, "Fonction Publique")
        law = Law(reference="REF-ADMIN", title="Décret portant nomination", type="decret",
                  content="Contenu.", language="fr", status="draft", category_id=chosen)
        scrambled_categories.add(law)
        scrambled_categories.commit()
        law_id = law.id

        _update_law_metadata(
            law_id, language="fr", language_confidence=0.9,
            category="Fonction Publique", category_confidence=0.9,
            category_id=proposed, suggested_categories=[proposed],
        )

        scrambled_categories.expire_all()
        refreshed = scrambled_categories.get(Law, law_id)
        assert refreshed.category_id == chosen
        assert refreshed.suggested_categories == [proposed]

    def test_pipeline_fills_a_null_category(self, scrambled_categories):
        from app.tasks.process_law import _update_law_metadata

        proposed = resolve_domain_id(scrambled_categories, "Fonction Publique")
        law = Law(reference="REF-NULL", title="Décret portant nomination", type="decret",
                  content="Contenu.", language="fr", status="draft")
        scrambled_categories.add(law)
        scrambled_categories.commit()
        law_id = law.id

        _update_law_metadata(
            law_id, language="fr", language_confidence=0.9,
            category="Fonction Publique", category_confidence=0.9,
            category_id=proposed, suggested_categories=[proposed],
        )

        scrambled_categories.expire_all()
        assert scrambled_categories.get(Law, law_id).category_id == proposed


class TestMigrationInvariants:
    def test_table_holds_exactly_the_canonical_domains(self, migrated_categories):
        names = {
            row[0] for row in migrated_categories.execute(select(Category.name)).all()
        }
        assert names == set(CANONICAL_DOMAINS)

    def test_unique_index_on_lower_name_exists(self, migrated_categories):
        found = migrated_categories.execute(text(
            "SELECT 1 FROM pg_indexes WHERE tablename='categories' "
            "AND indexname='uq_categories_name_lower'"
        )).first()
        assert found is not None, "l'index unique sur lower(name) est absent"

    def test_duplicate_name_is_rejected(self, migrated_categories):
        from sqlalchemy.exc import IntegrityError

        migrated_categories.add(Category(name="droit pénal", display_order=99))
        with pytest.raises(IntegrityError):
            migrated_categories.commit()
        migrated_categories.rollback()

    def test_display_order_puts_droit_penal_before_procedure_penale(self, migrated_categories):
        """
        Le front resout par correspondance partielle : le slug `penal`
        attraperait « Procédure Pénale » si celle-ci etait affichee avant.
        """
        rows = migrated_categories.execute(
            select(Category.name).order_by(Category.display_order)
        ).scalars().all()
        assert rows.index("Droit Pénal") < rows.index("Procédure Pénale")

    def test_no_law_is_orphaned(self, migrated_categories):
        orphans = migrated_categories.execute(text(
            "SELECT count(*) FROM laws l LEFT JOIN categories c ON c.id = l.category_id "
            "WHERE l.category_id IS NOT NULL AND c.id IS NULL"
        )).scalar_one()
        assert orphans == 0
