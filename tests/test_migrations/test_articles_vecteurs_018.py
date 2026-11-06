"""
Migration 018 : articles_vecteurs, la table etroite que lit la recherche exacte.

Ce que ces tests figent, sur un vrai PostgreSQL :
- le declencheur suit chaque ecriture d'un vecteur ou de la loi, dans la
  transaction de l'article, et rien d'autre ne reecrit la ligne ;
- la cle etrangere emporte la ligne avec l'article, et avec la loi ;
- le vecteur est bien DANS la ligne : c'est tout l'objet de la table ;
- la montee remplit une base en 017, se rejoue sans rien reecrire, et repare une
  derive, le remede que nomme le rapport du controle ;
- la descente ne laisse rien, et laisse `articles` intact ;
- PostgreSQL refuse ALTER COLUMN TYPE tant que le declencheur existe : une
  future migration de type doit le retirer d'abord.

Ce que la recherche fait de cette table : tests/test_services/
test_recherche_exacte.py ; ce que le controle en dit :
tests/test_services/test_controle_du_chat.py.

Usage:
    pytest tests/test_migrations/test_articles_vecteurs_018.py -v
"""

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from app.models.law import ArticleVecteur
from app.services.embedding_service import EmbeddingService
from tests.test_migrations.test_embeddings_voyage_016 import _alembic

AVANT_018 = "017_blog_posts_table"
REVISION = "018_articles_vecteurs"
DIM = EmbeddingService.EMBEDDING_DIM
DECLENCHEUR = "articles_vecteurs_suivre"
LOI_A, LOI_B = 1, 2


def _vecteur(graine: int) -> str:
    v = np.random.default_rng(graine).normal(size=DIM)
    return "[" + ",".join(f"{x:.6g}" for x in v / np.linalg.norm(v)) + "]"


def _article(session, article_id: int, loi: int = LOI_A, graine=None) -> None:
    """Un article, vectorise si `graine` est donnee."""
    session.execute(text(
        "INSERT INTO articles (id, law_id, number, content, \"order\", embedding) "
        "VALUES (:id, :loi, :n, 'Article.', :id, CAST(:v AS halfvec))"
    ), {"id": article_id, "loi": loi, "n": str(article_id),
        "v": None if graine is None else _vecteur(graine)})


def _poser(session, article_id: int, graine=None) -> None:
    session.execute(text("UPDATE articles SET embedding = CAST(:v AS halfvec) WHERE id = :id"),
                    {"id": article_id, "v": None if graine is None else _vecteur(graine)})


def _table(session) -> dict:
    return {
        ligne.article_id: (ligne.law_id, ligne.e)
        for ligne in session.execute(text(
            "SELECT article_id, law_id, embedding::text AS e FROM articles_vecteurs"
        ))
    }


def _articles_vectorises(session) -> dict:
    return {
        ligne.id: (ligne.law_id, ligne.e)
        for ligne in session.execute(text(
            "SELECT id, law_id, embedding::text AS e FROM articles WHERE embedding IS NOT NULL"
        ))
    }


def _versions_des_lignes(session) -> dict:
    """xmin et ctid : ils changent des qu'une ligne est reecrite, meme a l'identique."""
    return {
        ligne.article_id: (ligne.xmin, ligne.ctid)
        for ligne in session.execute(text(
            "SELECT article_id, xmin::text AS xmin, ctid::text AS ctid FROM articles_vecteurs"
        ))
    }


def _etat_du_declencheur(session):
    return session.execute(text(
        "SELECT tgenabled FROM pg_trigger "
        "WHERE tgrelid = 'articles'::regclass AND tgname = :nom"
    ), {"nom": DECLENCHEUR}).scalar()


def _rejouer_la_montee(session) -> None:
    """Le remede a une derive, tel que le dit la migration."""
    session.commit()
    _alembic("stamp", AVANT_018)
    _alembic("upgrade", REVISION)
    _alembic("stamp", "head")


@pytest.fixture
def session(sync_db_session):
    """Deux lois ; base remise a `head`, declencheur en service, meme apres un echec."""
    for loi in (LOI_A, LOI_B):
        sync_db_session.execute(text(
            "INSERT INTO laws (id, reference, title, content, type, language, status) "
            "VALUES (:id, :ref, 'Loi', 'Contenu.', 'loi', 'fr', 'published')"
        ), {"id": loi, "ref": f"LOI-018-{loi}"})
    sync_db_session.commit()
    try:
        yield sync_db_session
    finally:
        sync_db_session.rollback()
        _alembic("upgrade", "head")
        sync_db_session.execute(text("ALTER TABLE articles ENABLE TRIGGER USER"))
        sync_db_session.commit()


class TestLeDeclencheur:

    def test_un_article_insere_avec_son_vecteur_y_entre(self, session):
        _article(session, 1, graine=1)

        assert _table(session) == _articles_vectorises(session)
        assert list(_table(session)) == [1]

    def test_une_insertion_en_masse_aussi(self, session):
        session.execute(text(
            "INSERT INTO articles (id, law_id, number, content, \"order\", embedding) "
            "SELECT i, 1, i::text, 'Article.', i, CAST(:v AS halfvec) FROM generate_series(1, 5) i"
        ), {"v": _vecteur(1)})

        assert sorted(_table(session)) == [1, 2, 3, 4, 5]

    def test_un_article_sans_vecteur_n_y_entre_pas(self, session):
        _article(session, 1)

        assert _table(session) == {}

    def test_un_vecteur_pose_apres_coup_y_entre(self, session):
        _article(session, 1)

        _poser(session, 1, graine=1)

        assert _table(session) == _articles_vectorises(session)
        assert list(_table(session)) == [1]

    def test_un_vecteur_change_y_change(self, session):
        _article(session, 1, graine=1)
        avant = _table(session)

        _poser(session, 1, graine=2)

        assert _table(session) == _articles_vectorises(session)
        assert _table(session) != avant

    def test_un_vecteur_retire_en_sort(self, session):
        _article(session, 1, graine=1)
        _article(session, 2, graine=2)

        _poser(session, 1, graine=None)

        assert list(_table(session)) == [2]

    def test_un_changement_de_loi_est_suivi(self, session):
        """Le filtre sur les lois joint par cette colonne, et non par `articles`."""
        _article(session, 1, loi=LOI_A, graine=1)

        session.execute(text("UPDATE articles SET law_id = :loi WHERE id = 1"), {"loi": LOI_B})

        assert _table(session)[1][0] == LOI_B
        assert _table(session) == _articles_vectorises(session)

    def test_un_changement_de_loi_sans_vecteur_n_ecrit_rien(self, session):
        _article(session, 1, loi=LOI_A)

        session.execute(text("UPDATE articles SET law_id = :loi WHERE id = 1"), {"loi": LOI_B})

        assert _table(session) == {}

    def test_le_vecteur_et_la_loi_changes_ensemble(self, session):
        _article(session, 1, loi=LOI_A, graine=1)

        session.execute(text(
            "UPDATE articles SET law_id = :loi, embedding = CAST(:v AS halfvec) WHERE id = 1"
        ), {"loi": LOI_B, "v": _vecteur(2)})

        assert _table(session) == _articles_vectorises(session)

    def test_une_autre_colonne_ne_reecrit_pas_la_ligne(self, session):
        """Corriger un texte ne doit pas recopier 2 Ko de vecteur."""
        _article(session, 1, graine=1)
        session.commit()
        avant = _versions_des_lignes(session)

        session.execute(text("UPDATE articles SET content = 'Corrige.' WHERE id = 1"))
        session.commit()

        assert _versions_des_lignes(session) == avant

    def test_il_part_avec_la_transaction_de_l_article(self, session):
        """Un article annule ne laisse rien : un meme instantane voit les deux, ou aucun."""
        _article(session, 1, graine=1)

        session.rollback()

        assert _table(session) == {}


class TestLaCascade:

    def test_la_suppression_d_un_article_emporte_sa_ligne(self, session):
        _article(session, 1, graine=1)
        _article(session, 2, graine=2)

        session.execute(text("DELETE FROM articles WHERE id = 1"))

        assert list(_table(session)) == [2]

    def test_la_suppression_d_une_loi_passe_par_ses_articles(self, session):
        _article(session, 1, loi=LOI_A, graine=1)
        _article(session, 2, loi=LOI_B, graine=2)

        session.execute(text("DELETE FROM laws WHERE id = :loi"), {"loi": LOI_A})

        assert list(_table(session)) == [2]

    def test_truncate_cascade_la_vide(self, session):
        """Comme le nettoyage de tests/conftest.py et scripts/reset_database.py."""
        _article(session, 1, graine=1)

        session.execute(text("TRUNCATE articles RESTART IDENTITY CASCADE"))

        assert _table(session) == {}


class TestLeSchema:

    def test_le_modele_et_la_table_s_accordent(self, session):
        colonnes = dict(session.execute(text(
            "SELECT attname, format_type(atttypid, atttypmod) FROM pg_attribute "
            "WHERE attrelid = 'articles_vecteurs'::regclass AND attnum > 0 AND NOT attisdropped"
        )).all())

        assert colonnes == {
            "article_id": "integer", "law_id": "integer", "embedding": f"halfvec({DIM})",
        }
        assert {c.name for c in ArticleVecteur.__table__.c} == set(colonnes)
        assert not any(c.nullable for c in ArticleVecteur.__table__.c)

    def test_le_vecteur_est_dans_la_ligne(self, session):
        """Tout l'objet de la table : rien dans la TOAST, aucune compression."""
        _article(session, 1, graine=1)
        session.commit()

        stockage = session.execute(text(
            "SELECT attstorage FROM pg_attribute "
            "WHERE attrelid = 'articles_vecteurs'::regclass AND attname = 'embedding'"
        )).scalar()
        taille, toast = session.execute(text(
            "SELECT pg_column_size(embedding), "
            "(SELECT coalesce(pg_relation_size(NULLIF(reltoastrelid, 0)), 0) FROM pg_class "
            " WHERE oid = 'articles_vecteurs'::regclass) FROM articles_vecteurs"
        )).one()

        assert stockage == "p"
        assert taille == 2 * DIM + 8
        assert toast == 0

    def test_la_cle_etrangere_emporte_la_ligne(self, session):
        cible, suppression = session.execute(text(
            "SELECT confrelid::regclass::text, confdeltype FROM pg_constraint "
            "WHERE conrelid = 'articles_vecteurs'::regclass AND contype = 'f'"
        )).one()

        assert (cible, suppression) == ("articles", "c")

    def test_le_declencheur_suit_le_vecteur_et_la_loi_apres_l_ecriture(self, session):
        definition = session.execute(text(
            "SELECT pg_get_triggerdef(oid) FROM pg_trigger "
            "WHERE tgrelid = 'articles'::regclass AND tgname = :nom"
        ), {"nom": DECLENCHEUR}).scalar()

        assert "AFTER INSERT OR UPDATE OF embedding, law_id ON public.articles" in definition
        assert "FOR EACH ROW EXECUTE FUNCTION articles_vecteurs_suivre()" in definition

    def test_aucun_index_vectoriel_sur_la_table(self, session):
        """Exacte par construction : sans index, aucun parcours approche n'est possible."""
        index = session.execute(text(
            "SELECT indexname FROM pg_indexes WHERE tablename = 'articles_vecteurs'"
        )).scalars().all()

        assert index == ["articles_vecteurs_pkey"]


class TestLaMontee:

    def test_elle_remplit_une_base_en_017(self, session):
        session.commit()
        _alembic("downgrade", AVANT_018)
        for article_id in (3, 1, 2):
            _article(session, article_id, loi=LOI_B, graine=article_id)
        _article(session, 4)
        session.commit()

        _alembic("upgrade", "head")

        assert sorted(_table(session)) == [1, 2, 3]
        assert _table(session) == _articles_vectorises(session)
        # Dans l'ordre des articles, quel que soit celui de leur insertion, et
        # avec ses statistiques.
        ordre = session.execute(text("SELECT article_id FROM articles_vecteurs")).scalars().all()
        assert ordre == [1, 2, 3]
        assert session.execute(text(
            "SELECT reltuples FROM pg_class WHERE oid = 'articles_vecteurs'::regclass"
        )).scalar() == 3

    def test_la_rejouer_ne_reecrit_rien(self, session):
        _article(session, 1, graine=1)
        _article(session, 2, loi=LOI_B, graine=2)
        _article(session, 3)
        session.commit()
        versions = _versions_des_lignes(session)
        contenu = _table(session)
        session.commit()

        _rejouer_la_montee(session)

        assert _versions_des_lignes(session) == versions
        assert _table(session) == contenu
        assert session.execute(text("SELECT version_num FROM alembic_version")).scalar() == REVISION
        assert session.execute(text(
            "SELECT count(*) FROM pg_trigger WHERE tgname = :nom AND NOT tgisinternal"
        ), {"nom": DECLENCHEUR}).scalar() == 1
        assert session.execute(text(
            "SELECT count(*) FROM pg_proc WHERE proname = :nom"
        ), {"nom": DECLENCHEUR}).scalar() == 1

    def test_la_rejouer_repare_une_derive(self, session):
        """
        Chaque sorte d'ecart qu'une ecriture sans declencheur peut laisser, et
        celui que seul le mode replica laisse, la cascade etant coupee avec lui.
        """
        for article_id in (1, 2, 3, 5):
            _article(session, article_id, graine=article_id)
        _article(session, 4)
        session.commit()
        session.execute(text(f"ALTER TABLE articles DISABLE TRIGGER {DECLENCHEUR}"))
        _poser(session, 1, graine=10)       # vecteur change
        _poser(session, 2, graine=None)     # vecteur retire
        session.execute(text("UPDATE articles SET law_id = :loi WHERE id = 3"), {"loi": LOI_B})
        _poser(session, 4, graine=40)       # vecteur pose
        _article(session, 6, graine=60)     # article ajoute
        session.commit()
        session.execute(text("SET LOCAL session_replication_role = replica"))
        session.execute(text("DELETE FROM articles WHERE id = 5"))
        session.commit()
        assert _table(session) != _articles_vectorises(session)
        assert 5 in _table(session)
        session.commit()

        _rejouer_la_montee(session)

        assert _table(session) == _articles_vectorises(session)
        assert sorted(_table(session)) == [1, 3, 4, 6]

    def test_la_rejouer_remet_le_declencheur_en_service(self, session):
        session.execute(text(f"ALTER TABLE articles DISABLE TRIGGER {DECLENCHEUR}"))
        assert _etat_du_declencheur(session) == "D"

        _rejouer_la_montee(session)

        assert _etat_du_declencheur(session) == "O"
        _article(session, 1, graine=1)
        assert list(_table(session)) == [1]


class TestLaDescente:

    def test_elle_ne_laisse_rien_et_garde_les_vecteurs_d_articles(self, session):
        _article(session, 1, graine=1)
        vecteurs = _articles_vectorises(session)
        session.commit()

        _alembic("downgrade", AVANT_018)

        assert session.execute(text("SELECT to_regclass('articles_vecteurs')")).scalar() is None
        assert _etat_du_declencheur(session) is None
        assert session.execute(text(
            "SELECT count(*) FROM pg_proc WHERE proname = :nom"
        ), {"nom": DECLENCHEUR}).scalar() == 0
        assert _articles_vectorises(session) == vecteurs
        # Les ecritures passent : rien ne reste accroche a la table retiree.
        _poser(session, 1, graine=2)
        session.commit()

    def test_la_montee_qui_suit_rend_la_meme_table(self, session):
        _article(session, 1, graine=1)
        _article(session, 2, loi=LOI_B, graine=2)
        contenu = _table(session)
        session.commit()

        _alembic("downgrade", AVANT_018)
        _alembic("upgrade", "head")

        assert _table(session) == contenu

    def test_elle_se_rejoue(self, session):
        session.commit()
        _alembic("downgrade", AVANT_018)
        _alembic("stamp", REVISION)

        _alembic("downgrade", AVANT_018)

        assert session.execute(text("SELECT version_num FROM alembic_version")).scalar() == AVANT_018
        session.commit()


class TestUneFutureMigrationDeType:

    def test_postgresql_la_refuse_tant_que_le_declencheur_existe(self, session):
        """
        Une migration comme la 016 (ALTER COLUMN TYPE) ne peut donc pas faire
        deriver la table en silence : elle echoue, et doit retirer le
        declencheur, puis changer aussi la colonne d'articles_vecteurs.
        """
        with pytest.raises(DBAPIError) as erreur:
            session.execute(text("ALTER TABLE articles ALTER COLUMN embedding TYPE halfvec(1024)"))

        assert "used in a trigger definition" in str(erreur.value)
