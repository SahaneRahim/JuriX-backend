"""« Lire aussi l'article renvoye » (RAGService._avec_les_renvois)."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.schemas.search import ChunkResult
from app.services.rag_service import (
    RENVOIS_MAX,
    SOURCE_ARTICLE_RENVOYE,
    RAGService,
    numero_renvoye,
)


def _extrait(numero, contenu="Texte.", law_id=1, ident=None):
    return ChunkResult(
        article_id=ident or abs(hash((law_id, numero))) % 100000, law_id=law_id, number=numero,
        content=contenu, excerpt=contenu[:40], reference=f"L{law_id}", law_title="Loi",
        type="loi", status="published", relevance_score=0.5,
    )


@pytest.mark.parametrize("numero,contenu,attendu", [
    ("320", "Les peines de l'article 318 ci-dessus sont doublées si...", "318"),
    ("321", "Les peines prévues à l’article 318 sont doublées.", "318"),
    ("65.2", "Est puni des peines prévues à l'alinéa 1 tout accès non autorisé.", "65.1"),
    ("65.1", "Est puni des peines prévues à l'alinéa 1 ...", None),
    ("318", "Les peines de l'article 318 s'appliquent.", None),
    ("12", "Dans les conditions de l'article 40 ci-dessous.", None),
    ("12", "Est puni d'une peine de dix ans.", None),
])
def test_numero_renvoye(numero, contenu, attendu):
    assert numero_renvoye(_extrait(numero, contenu)) == attendu


def _rag(lignes):
    rag = RAGService.__new__(RAGService)
    rag.db = MagicMock()

    async def lire(_, params):
        ligne = lignes.get(params["numero"])
        return SimpleNamespace(first=lambda: ligne)

    rag.db.execute = AsyncMock(side_effect=lire)
    return rag


def _ligne(numero, law_id=1):
    return SimpleNamespace(
        article_id=900 + int(numero.split(".")[0]), law_id=law_id, number=numero, article_title=None,
        section=None, page_number=1, content="La peine.", reference="L1", law_title="Loi", type="loi",
        language="fr", status="published", category_id=None, publication_date=None,
    )


async def test_l_article_vise_est_lu_juste_apres_le_renvoi():
    rag = _rag({"318": _ligne("318")})
    extraits = [_extrait("320", "Les peines de l'article 318 sont doublées."), _extrait("56", law_id=2)]
    rendus = await rag._avec_les_renvois(extraits)
    assert [c.number for c in rendus] == ["320", "318", "56"]
    assert rendus[1].source == SOURCE_ARTICLE_RENVOYE


async def test_un_article_deja_lu_remonte_sans_requete():
    rag = _rag({})
    extraits = [
        _extrait("320", "Les peines de l'article 318 sont doublées."),
        _extrait("56", law_id=2), _extrait("318"),
    ]
    rendus = await rag._avec_les_renvois(extraits)
    assert [c.number for c in rendus] == ["320", "318", "56"]
    rag.db.execute.assert_not_awaited()


async def test_un_article_absent_ne_change_rien():
    extraits = [_extrait("320", "Les peines de l'article 318 sont doublées.")]
    assert await _rag({})._avec_les_renvois(extraits) == extraits


async def test_le_nombre_de_lectures_est_borne():
    extraits = [
        _extrait(str(n), f"Les peines de l'article {n + 100} sont doublées.") for n in (1, 2, 3)
    ]
    rag = _rag({str(n + 100): _ligne(str(n + 100)) for n in (1, 2, 3)})
    rendus = await rag._avec_les_renvois(extraits)
    assert len(rendus) == len(extraits) + RENVOIS_MAX
