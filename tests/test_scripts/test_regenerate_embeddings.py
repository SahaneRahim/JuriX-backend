"""
Tests de scripts/regenerate_embeddings.py : ce qui est a refaire, et la trace
laissee.

La selection repose sur la PROVENANCE des vecteurs (articles.embedding_model),
plus seulement sur leur absence. C'est ce qui rend un changement de fournisseur
sur : chaque vecteur de l'ancien modele est reconnu et refait, au lieu de
cohabiter en silence avec ceux du nouveau — deux espaces dont les cosinus
croises ne veulent rien dire.

Usage:
    pytest tests/test_scripts/test_regenerate_embeddings.py -v
"""

import numpy as np
import pytest
from sqlalchemy import text

from app.models.law import Article, Law
from app.services.embedding_service import EmbeddingService
from scripts import regenerate_embeddings as regen

EMPREINTE = "doublure|actuelle"
DIM = EmbeddingService.EMBEDDING_DIM


def _vecteur(graine: int) -> np.ndarray:
    v = np.random.default_rng(graine).normal(size=DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture
def base(sync_db_session):
    """Un article dans chacun des etats possibles."""
    session = sync_db_session
    session.add(Law(id=1, reference="LOI-REGEN", title="Loi", content="Contenu.",
                    type="loi", language="fr", status="published"))
    session.flush()

    etats = {
        1: (None, None, True),                        # jamais vectorise
        2: (_vecteur(2), "gemini|ancien", True),      # autre fournisseur
        3: (_vecteur(3), None, True),                 # origine inconnue
        4: (_vecteur(4), EMPREINTE, True),            # a jour
        5: (None, None, False),                       # ecarte par le raffineur
    }
    for article_id, (vecteur, empreinte, embed) in etats.items():
        session.add(Article(
            id=article_id, law_id=1, number=str(article_id),
            content=f"Article {article_id}, contenu de test.", order=article_id,
            embedding=None if vecteur is None else vecteur.tolist(),
            embedding_model=empreinte, embed=embed,
        ))
    session.commit()
    return session


def _ids(lot):
    return [r["id"] for r in lot]


class TestSelection:

    def test_refait_ce_qui_manque_ou_vient_d_ailleurs(self, base):
        """
        L'article 3 est le cas qui justifie IS DISTINCT FROM : avec `<>`, son
        empreinte NULL rendrait la condition NULL, et ce vecteur d'origine
        inconnue passerait pour bon.
        """
        lot = regen._fetch_batch(base, None, False, EMPREINTE, 0, 100)

        assert _ids(lot) == [1, 2, 3]

    def test_le_compte_annonce_suit_la_meme_selection(self, base):
        assert regen._count_remaining(base, None, False, EMPREINTE) == 3

    def test_force_reprend_tout_ce_qui_est_vectorisable(self, base):
        lot = regen._fetch_batch(base, None, True, EMPREINTE, 0, 100)

        assert _ids(lot) == [1, 2, 3, 4]
        assert regen._count_remaining(base, None, True, EMPREINTE) == 4

    def test_un_autre_fournisseur_voit_tout_a_refaire(self, base):
        """Le changement de fournisseur, vu du script : rien n'est a jour."""
        assert regen._count_remaining(base, None, False, "gemma|nouveau") == 4


class TestEcriture:

    def test_le_vecteur_part_avec_son_empreinte(self, base):
        vecteurs = [_vecteur(10 + i) for i in range(3)]

        regen._write_batch(base, [1, 2, 3], vecteurs, EMPREINTE)
        base.commit()

        rows = base.execute(text(
            "SELECT id, embedding_model FROM articles WHERE embed ORDER BY id"
        )).all()
        assert [(r.id, r.embedding_model) for r in rows] == [
            (i, EMPREINTE) for i in (1, 2, 3, 4)
        ]
        relu = np.array(base.get(Article, 1).embedding, dtype=np.float32)
        assert np.allclose(relu, vecteurs[0], atol=1e-6)
        assert regen._count_remaining(base, None, False, EMPREINTE) == 0


class _Fournisseur:
    label = "doublure"
    fingerprint = EMPREINTE
    inter_batch_delay_s = 0.0


class _Service:
    """Doublure d'EmbeddingService : aucun modele, aucun reseau."""

    MAX_TEXT_LENGTH = EmbeddingService.MAX_TEXT_LENGTH
    EMBEDDING_DIM = DIM
    encodes = []

    def __init__(self, *args, **kwargs):
        self.provider = _Fournisseur()

    def generate_batch_embeddings(self, texts, batch_size=None, normalize=True):
        _Service.encodes.extend(texts)
        return [_vecteur(len(t)) for t in texts]


class TestDeBoutEnBout:

    @pytest.fixture(autouse=True)
    def _doublure(self, monkeypatch):
        _Service.encodes = []
        monkeypatch.setattr(regen, "EmbeddingService", _Service)

    def test_une_passe_suffit_et_la_suivante_ne_trouve_rien(self, base):
        assert regen.main(["--all", "--batch-size", "2"]) == 0

        assert len(_Service.encodes) == 3
        assert regen._count_remaining(base, None, False, EMPREINTE) == 0

        _Service.encodes = []
        assert regen.main(["--all"]) == 0
        assert _Service.encodes == []

    def test_le_compte_a_blanc_n_ecrit_rien(self, base):
        assert regen.main(["--all", "--dry-run"]) == 0

        assert _Service.encodes == []
        assert regen._count_remaining(base, None, False, EMPREINTE) == 3


def _index_embedding(session):
    return session.execute(text(
        "SELECT indexrelid::regclass::text, indisvalid FROM pg_index "
        "WHERE indrelid = 'articles'::regclass "
        "AND indexrelid::regclass::text LIKE 'idx_articles_embedding_hnsw%'"
    )).all()


class TestReconstructionDeLIndex:

    def test_index_reconstruit_et_valide(self, base):
        regen.reindex(base)

        assert _index_embedding(base) == [("idx_articles_embedding_hnsw_vector", True)]

    def test_echec_ne_laisse_pas_de_copie_invalide(self, base, monkeypatch):
        """
        Un REINDEX CONCURRENTLY interrompu laisse une copie _ccnew invalide,
        maintenue a chaque ecriture. On la simule, puis on fait echouer.
        """
        with regen.sync_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(
                "CREATE INDEX idx_articles_embedding_hnsw_vector_ccnew ON articles (id)"
            ))
        monkeypatch.setattr(
            regen, "text",
            lambda sql: text(sql.replace("REINDEX INDEX CONCURRENTLY", "REINDEX INDEX CONCURRENTLY inexistant_")
                             if sql.startswith("REINDEX") else sql),
        )

        with pytest.raises(Exception):
            regen.reindex(base)

        assert _index_embedding(base) == [("idx_articles_embedding_hnsw_vector", True)]
