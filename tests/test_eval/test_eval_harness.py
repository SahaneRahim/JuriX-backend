"""
Tests du harnais d'evaluation.

Ce qui est verifie ici, c'est que le harnais REFUSE de produire des chiffres
faux : jeu non relu, corpus qui a bouge, recoupement lexical trop eleve. Un
harnais qui accepte tout produit des nombres, pas des mesures.
"""

import json

import numpy as np
import pytest

from scripts.eval.run_eval import (
    _brute_force_rankings,
    _load_eval_set,
    _slice_and_renormalise,
)


def _write_set(path, items, snapshot=None):
    path.write_text(json.dumps({
        "version": 1,
        "corpus_snapshot": snapshot or {"laws": 1, "articles": 1, "alembic_head": "abc"},
        "items": items,
    }, ensure_ascii=False), encoding="utf-8")
    return path


def _item(**kwargs):
    base = {
        "question": "Quelle est la responsabilité des dirigeants ?",
        "expected_article_id": 1,
        "expected_law_id": 1,
        "expected_article_number": "161",
        "lexical_overlap": 0.1,
        "reviewed": True,
        "split": "dev",
    }
    base.update(kwargs)
    return base


class TestGuards:

    def test_missing_file_is_explicit(self, tmp_path):
        with pytest.raises(SystemExit, match="absent"):
            _load_eval_set(tmp_path / "nexiste.pas", "dev", False, True)

    def test_unreviewed_items_are_refused(self, tmp_path):
        path = _write_set(tmp_path / "set.json", [_item(reviewed=False)])

        with pytest.raises(SystemExit, match="non relus"):
            _load_eval_set(path, "dev", False, True)

    def test_unreviewed_can_be_forced_explicitly(self, tmp_path):
        path = _write_set(tmp_path / "set.json", [_item(reviewed=False)])

        payload = _load_eval_set(path, "dev", True, True)

        assert len(payload["items"]) == 1

    @pytest.mark.asyncio
    async def test_stale_corpus_snapshot_is_refused(self, tmp_path, db_session):
        """
        Les identifiants d'article ne designent quelque chose que contre un etat
        de corpus donne. Sans ce controle, les chiffres deviennent faux en
        silence apres une reingestion.
        """
        path = _write_set(
            tmp_path / "set.json", [_item()],
            snapshot={"laws": 999, "articles": 999999, "alembic_head": "perimee"},
        )

        with pytest.raises(SystemExit, match="corpus a change"):
            _load_eval_set(path, "dev", False, False)

    def test_split_selection(self, tmp_path):
        path = _write_set(tmp_path / "set.json", [
            _item(split="dev", expected_article_id=1),
            _item(split="test", expected_article_id=2),
        ])

        assert len(_load_eval_set(path, "dev", False, True)["items"]) == 1
        assert len(_load_eval_set(path, "test", False, True)["items"]) == 1
        assert len(_load_eval_set(path, "all", False, True)["items"]) == 2


class TestMatryoshkaSlicing:

    def test_slicing_renormalises(self):
        """
        Le prefixe d'un vecteur unitaire n'est PAS unitaire : sans
        renormalisation, toute la comparaison cosinus est fausse.
        """
        rng = np.random.default_rng(0)
        matrix = rng.normal(size=(5, 3072)).astype(np.float32)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

        sliced = _slice_and_renormalise(matrix, 768)

        assert sliced.shape == (5, 768)
        assert np.allclose(np.linalg.norm(sliced, axis=1), 1.0, atol=1e-5)

    def test_zero_vector_does_not_divide_by_zero(self):
        matrix = np.zeros((2, 3072), dtype=np.float32)

        sliced = _slice_and_renormalise(matrix, 768)

        assert np.isfinite(sliced).all()


class TestBruteForce:

    def test_ranks_by_cosine_similarity(self):
        article_ids = np.array([10, 20, 30])
        corpus = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7071, 0.7071],
        ], dtype=np.float32)
        queries = np.array([[1.0, 0.0]], dtype=np.float32)

        rankings = _brute_force_rankings(article_ids, corpus, queries, top_k=3)

        assert rankings == [[10, 30, 20]]

    def test_respects_top_k(self):
        article_ids = np.array([1, 2, 3, 4])
        corpus = np.eye(4, dtype=np.float32)
        queries = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)

        assert len(_brute_force_rankings(article_ids, corpus, queries, top_k=2)[0]) == 2
