"""
Tests des metriques d'evaluation.

Jeu synthetique calcule a la main : si ces valeurs derivent, tous les
resultats de mesure derivent avec elles sans que rien ne le signale.
"""

import pytest

from scripts.eval.metrics import (
    bootstrap_ci,
    hits_at_k,
    mrr,
    recall_at_k,
    reciprocal_ranks,
    summarise,
)


# 4 questions, classements connus :
#   q1 : attendu 1, rang 2
#   q2 : attendu 7, rang 3
#   q3 : attendu 3, rang 1
#   q4 : attendu 99, absent
RANKINGS = [[9, 1, 2], [5, 6, 7], [3, 8], [4]]
GOLD = [1, 7, 3, 99]


class TestRecall:

    def test_recall_at_1(self):
        # seule q3 a l'article attendu en premiere position
        assert recall_at_k(RANKINGS, GOLD, 1) == 0.25

    def test_recall_at_5(self):
        # q1, q2, q3 : trois sur quatre
        assert recall_at_k(RANKINGS, GOLD, 5) == 0.75

    def test_absent_gold_never_counts(self):
        assert hits_at_k(RANKINGS, GOLD, 10)[3] == 0.0

    def test_empty_ranking_does_not_crash(self):
        assert recall_at_k([[]], [1], 5) == 0.0

    def test_empty_dataset(self):
        assert recall_at_k([], [], 5) == 0.0


class TestMRR:

    def test_mrr_matches_hand_computation(self):
        # 1/2 + 1/3 + 1/1 + 0, divise par 4
        assert mrr(RANKINGS, GOLD) == pytest.approx((0.5 + 1 / 3 + 1.0 + 0.0) / 4)

    def test_cutoff_excludes_late_hits(self):
        # avec cutoff=2, q2 (rang 3) ne compte plus
        assert mrr(RANKINGS, GOLD, cutoff=2) == pytest.approx((0.5 + 0.0 + 1.0 + 0.0) / 4)

    def test_reciprocal_ranks_per_item(self):
        assert reciprocal_ranks(RANKINGS, GOLD) == pytest.approx([0.5, 1 / 3, 1.0, 0.0])

    def test_empty_dataset(self):
        assert mrr([], []) == 0.0


class TestBootstrap:

    def test_is_deterministic_under_seed(self):
        per_item = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]

        assert bootstrap_ci(per_item, seed=7) == bootstrap_ci(per_item, seed=7)

    def test_interval_brackets_the_mean(self):
        per_item = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0]
        low, high = bootstrap_ci(per_item, seed=0)

        assert low <= sum(per_item) / len(per_item) <= high

    def test_interval_is_wide_on_a_small_sample(self):
        """
        Le harnais doit rendre visible qu'un petit echantillon ne tranche rien :
        c'est la seule protection contre un "gagnant" a deux points d'ecart.
        """
        low, high = bootstrap_ci([1.0] * 4 + [0.0] * 2, seed=0)

        assert high - low > 0.3

    def test_empty(self):
        assert bootstrap_ci([]) == (0.0, 0.0)


class TestSummary:

    def test_contains_every_metric_with_its_interval(self):
        summary = summarise(RANKINGS, GOLD, ks=(1, 5), mrr_cutoff=10)

        assert summary["n"] == 4
        assert summary["recall@1"] == 0.25
        assert summary["recall@5"] == 0.75
        assert summary["mrr@10"] == pytest.approx((0.5 + 1 / 3 + 1.0) / 4)
        for key in ("recall@1_ci", "recall@5_ci", "mrr@10_ci"):
            low, high = summary[key]
            assert low <= high

    def test_mismatched_lengths_are_rejected(self):
        with pytest.raises(AssertionError):
            recall_at_k([[1]], [1, 2], 1)
