"""
Metriques de recuperation : recall@k, MRR, intervalle de confiance bootstrap.

Fonctions pures, sans E/S : testables sur un jeu calcule a la main.

DEFINITIONS, a lire avant de "corriger" quoi que ce soit. Le jeu d'evaluation
associe UN SEUL article pertinent a chaque question. Dans ce cadre :

  - recall@k = taux de succes = proportion de questions dont l'article attendu
    figure dans les k premiers resultats. Avec un unique document pertinent,
    recall et precision@k ne portent pas la meme information et recall@k est
    celle qui compte : le RAG envoie 8 articles au modele, la question est de
    savoir si le bon en fait partie.
  - MRR = moyenne de 1/rang de l'article attendu, 0 s'il est absent. C'est la
    metrique principale : ce qui compte n'est pas seulement que le bon article
    soit present, mais qu'il soit HAUT dans les 8 envoyes.

TAILLE D'ECHANTILLON. Avec n = 60, l'erreur type sur un taux proche de 0,7 est
d'environ 6 points. Un ecart de 2 points entre deux configurations n'est pas un
resultat. C'est pourquoi bootstrap_ci existe et pourquoi le rapport affiche
l'intervalle a cote de chaque nombre.

Author: JuriX Team
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np


def _rank_of(ranking: Sequence[int], gold: int) -> Optional[int]:
    """Rang (1-indexe) de l'article attendu, None s'il est absent."""
    for position, article_id in enumerate(ranking, start=1):
        if article_id == gold:
            return position
    return None


def hits_at_k(rankings: Sequence[Sequence[int]], gold: Sequence[int], k: int) -> List[float]:
    """Succes par question (1.0 / 0.0), pour l'intervalle de confiance."""
    assert len(rankings) == len(gold), "un classement par question attendu"
    assert k > 0, "k doit etre positif"
    return [
        1.0 if (rank := _rank_of(ranking[:k], expected)) is not None and rank <= k else 0.0
        for ranking, expected in zip(rankings, gold)
    ]


def recall_at_k(rankings: Sequence[Sequence[int]], gold: Sequence[int], k: int) -> float:
    """Taux de succes dans les k premiers. 0.0 sur un jeu vide."""
    per_item = hits_at_k(rankings, gold, k)
    return float(np.mean(per_item)) if per_item else 0.0


def reciprocal_ranks(
    rankings: Sequence[Sequence[int]],
    gold: Sequence[int],
    cutoff: Optional[int] = None,
) -> List[float]:
    """1/rang par question, 0 si l'article attendu est absent (ou au-dela de cutoff)."""
    assert len(rankings) == len(gold), "un classement par question attendu"
    values = []
    for ranking, expected in zip(rankings, gold):
        window = ranking[:cutoff] if cutoff else ranking
        rank = _rank_of(window, expected)
        values.append(1.0 / rank if rank else 0.0)
    return values


def mrr(
    rankings: Sequence[Sequence[int]],
    gold: Sequence[int],
    cutoff: Optional[int] = None,
) -> float:
    """Mean Reciprocal Rank. 0.0 sur un jeu vide."""
    values = reciprocal_ranks(rankings, gold, cutoff)
    return float(np.mean(values)) if values else 0.0


def bootstrap_ci(
    per_item: Sequence[float],
    n_resamples: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Tuple[float, float]:
    """
    Intervalle de confiance par bootstrap sur les valeurs par question.

    Graine fixee : deux executions sur les memes donnees doivent rendre le meme
    intervalle, sans quoi un ecart de bruit passerait pour un resultat.
    """
    if not per_item:
        return (0.0, 0.0)

    values = np.asarray(per_item, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(n_resamples, values.size), replace=True)
    means = draws.mean(axis=1)

    tail = (1.0 - confidence) / 2.0
    return (
        float(np.quantile(means, tail)),
        float(np.quantile(means, 1.0 - tail)),
    )


def summarise(
    rankings: Sequence[Sequence[int]],
    gold: Sequence[int],
    ks: Sequence[int] = (1, 5, 10),
    mrr_cutoff: int = 10,
    seed: int = 0,
) -> dict:
    """Toutes les metriques, chacune avec son intervalle de confiance."""
    summary = {"n": len(gold)}

    for k in ks:
        per_item = hits_at_k(rankings, gold, k)
        low, high = bootstrap_ci(per_item, seed=seed)
        summary[f"recall@{k}"] = float(np.mean(per_item)) if per_item else 0.0
        summary[f"recall@{k}_ci"] = [low, high]

    per_item = reciprocal_ranks(rankings, gold, mrr_cutoff)
    low, high = bootstrap_ci(per_item, seed=seed)
    summary[f"mrr@{mrr_cutoff}"] = float(np.mean(per_item)) if per_item else 0.0
    summary[f"mrr@{mrr_cutoff}_ci"] = [low, high]

    return summary
