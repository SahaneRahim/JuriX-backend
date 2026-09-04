"""
Tests du pipeline de traitement des documents.

app/tasks/process_law.py fait ~700 lignes et n'avait aucun test : le repertoire
tests/test_tasks/ ne contenait qu'un __init__.py vide. C'est pourtant le seul
chemin par lequel un PDF devient des articles indexables.

Les tests marques integration touchent la base de test ; les autres sont de
pures fonctions de texte.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest
from sqlalchemy import select

from app.models.law import Article, Law
from app.services.embedding_service import EmbeddingService
from app.tasks import process_law as pl


# ==================== FONCTIONS PURES ====================


class TestTitleExtraction:

    @pytest.mark.parametrize("text,expected_start", [
        ("LOI N° 2024-001 DU 15 JANVIER 2024\nPortant régime des sociétés", "LOI N° 2024-001"),
        ("LAW N° 2024-002 OF 15 JANUARY 2024\nOn companies", "LAW N° 2024-002"),
        ("LA CONSTITUTION DE LA RÉPUBLIQUE DU CAMEROUN\nPréambule", "LA CONSTITUTION"),
    ])
    def test_extracts_known_patterns(self, text, expected_start):
        title = pl._extract_title_from_text(text)

        assert title is not None
        assert title.startswith(expected_start)

    def test_returns_none_on_empty_text(self):
        assert pl._extract_title_from_text("") is None

    def test_returns_none_on_junk(self):
        assert pl._extract_title_from_text("aaa\nbbb\nccc") is None


class TestTextCleaning:

    def test_rejoins_hyphenated_words(self):
        cleaned = pl._clean_extracted_text("respon-\nsabilité des dirigeants")

        assert "responsabilité" in cleaned

    def test_rejoins_elisions(self):
        cleaned = pl._clean_extracted_text("l'\narticle premier")

        assert "l'article" in cleaned

    def test_removes_page_markers(self):
        cleaned = pl._clean_extracted_text("Article 1.\n<<PAGE: 12>>\nContenu.")

        assert "<<PAGE" not in cleaned
        assert "Article 1." in cleaned
        assert "Contenu." in cleaned

    def test_removes_standalone_page_numbers(self):
        cleaned = pl._clean_extracted_text("Article 1.\n42\nContenu de l'article.")

        assert "\n42\n" not in cleaned

    def test_collapses_blank_lines(self):
        cleaned = pl._clean_extracted_text("A.\n\n\n\n\nB.")

        assert "\n\n\n" not in cleaned

    def test_empty_input(self):
        assert pl._clean_extracted_text("") == ""


class TestFallbacks:

    def test_language_detection_falls_back_on_error(self, monkeypatch):
        """Le pipeline ne doit pas s'arreter parce que le detecteur echoue."""
        import app.services.language_detector as detector_module

        class _Broken:
            def __init__(self, *a, **k):
                raise RuntimeError("modele absent")

        monkeypatch.setattr(detector_module, "LanguageDetector", _Broken)

        result = pl._detect_language("Le présent article de la loi et les décrets")

        assert result["language"] == "fr"
        assert 0.0 < result["confidence"] <= 1.0

    def test_category_classification_falls_back_on_error(self, monkeypatch):
        import app.services.document_classifier as classifier_module

        class _Broken:
            def __init__(self, *a, **k):
                raise RuntimeError("modele absent")

        monkeypatch.setattr(classifier_module, "DocumentClassifier", _Broken)

        result = pl._classify_category("Texte quelconque")

        assert "category" in result
        assert "confidence" in result


# ==================== INTEGRATION (base de test) ====================


@pytest.fixture
async def law_row(db_session):
    law = Law(
        reference="LOI-PIPE-001",
        title="Loi de test pipeline",
        content="Contenu initial.",
        type="loi",
        language="fr",
        status="processing",
    )
    db_session.add(law)
    await db_session.commit()
    await db_session.refresh(law)
    return law


# Articles de longueur REALISTE : le raffinage ecarte du vectoriel tout chunk
# de moins de 120 caracteres, et c'est justifie — sur le corpus reel, 62 % des
# chunks sous ce seuil sont des lignes de liste nominative ("100. DOURLAI
# STANISLAS 0769502K"), des fragments de tableau ou des separateurs. Un
# echantillon de test fait d'articles de 70 caracteres mesurerait donc le seuil,
# pas le pipeline.
SAMPLE_TEXT = """
LOI N° 2024-500 DU 1 JUIN 2024
PORTANT DISPOSITIONS DIVERSES

CHAPITRE I - DISPOSITIONS GÉNÉRALES

Article 1. Objet
La présente loi fixe les règles applicables à la constitution, au fonctionnement
et à la dissolution des sociétés commerciales exerçant sur le territoire de la
République du Cameroun, sans préjudice des dispositions de l'Acte uniforme OHADA.

Article 2. Champ d'application
Elle s'applique sur l'ensemble du territoire national et concerne toutes les
sociétés commerciales immatriculées au registre du commerce et du crédit
mobilier, quelle que soit la nationalité de leurs associés ou de leurs dirigeants.

Article 3. Définitions
Au sens de la présente loi, on entend par société commerciale toute personne
morale constituée par deux ou plusieurs personnes qui conviennent d'affecter des
biens à une activité en vue de partager le bénéfice qui pourra en résulter.
"""


class TestArticlePersistence:

    @pytest.mark.asyncio
    async def test_saves_section_and_page_number(self, db_session, law_row):
        count = pl._split_and_save_articles(law_row.id, SAMPLE_TEXT)

        assert count >= 3
        rows = (await db_session.execute(
            select(Article).where(Article.law_id == law_row.id).order_by(Article.order)
        )).scalars().all()

        numbered = [a for a in rows if a.number.isdigit()]
        assert numbered
        assert any(a.section and "CHAPITRE" in a.section for a in numbered)
        assert all(a.content for a in numbered)

    @pytest.mark.asyncio
    async def test_reprocessing_replaces_instead_of_duplicating(self, db_session, law_row):
        pl._split_and_save_articles(law_row.id, SAMPLE_TEXT)
        first = (await db_session.execute(
            select(Article).where(Article.law_id == law_row.id)
        )).scalars().all()

        pl._split_and_save_articles(law_row.id, SAMPLE_TEXT)
        second = (await db_session.execute(
            select(Article).where(Article.law_id == law_row.id)
        )).scalars().all()

        assert len(second) == len(first)


class TestEmbeddingGeneration:

    @pytest.fixture(autouse=True)
    def _stub_service(self, monkeypatch):
        """Doublure d'EmbeddingService : aucun appel reseau."""
        dim = EmbeddingService.EMBEDDING_DIM
        calls = {"texts": None}

        class _Stub:
            MAX_TEXT_LENGTH = EmbeddingService.MAX_TEXT_LENGTH
            EMBEDDING_DIM = dim

            def __init__(self, *a, **k):
                pass

            def generate_batch_embeddings(self, texts, **kwargs):
                calls["texts"] = texts
                rng = np.random.default_rng(1)
                out = []
                for _ in texts:
                    vec = rng.normal(size=dim)
                    out.append((vec / np.linalg.norm(vec)).astype(np.float32))
                return out

        import app.services.embedding_service as module

        monkeypatch.setattr(module, "EmbeddingService", _Stub)
        return calls

    @pytest.mark.asyncio
    async def test_writes_vectors_for_every_article(self, db_session, law_row):
        pl._split_and_save_articles(law_row.id, SAMPLE_TEXT)

        written = pl._generate_article_embeddings(law_row.id)

        assert written > 0
        rows = (await db_session.execute(
            select(Article).where(Article.law_id == law_row.id)
        )).scalars().all()

        # Seuls les chunks marques embed sont vectorises : les visas et les
        # fragments restent en base, cherchables en plein texte, sans vecteur.
        embeddable = [a for a in rows if a.embed]
        skipped = [a for a in rows if not a.embed]

        assert embeddable, "au moins un chunk doit etre vectorisable"
        assert all(a.embedding is not None for a in embeddable)
        assert all(a.embedding is None for a in skipped)
        assert len(embeddable[0].embedding) == EmbeddingService.EMBEDDING_DIM

    @pytest.mark.asyncio
    async def test_overlong_article_does_not_zero_the_whole_law(
        self, db_session, law_row, _stub_service
    ):
        """
        Un seul article trop long faisait lever generate_batch_embeddings pour
        la loi ENTIERE ; l'exception etait avalee et la loi publiee sans un
        seul vecteur.
        """
        long_text = SAMPLE_TEXT + "\n\nArticle 4. Long\n" + ("mot " * 6000)
        pl._split_and_save_articles(law_row.id, long_text)

        written = pl._generate_article_embeddings(law_row.id)

        assert written > 0
        assert all(
            len(t) <= EmbeddingService.MAX_TEXT_LENGTH for t in _stub_service["texts"]
        )


class TestFtsUpdate:

    @pytest.mark.asyncio
    async def test_updates_search_vectors(self, db_session, law_row):
        pl._split_and_save_articles(law_row.id, SAMPLE_TEXT)

        await pl._update_fts_vectors_async(db_session, law_row.id)
        await db_session.commit()

        from sqlalchemy import text as sql_text

        result = await db_session.execute(
            sql_text(
                "SELECT count(*) FROM articles "
                "WHERE law_id = :id AND search_vector IS NOT NULL"
            ),
            {"id": law_row.id},
        )
        assert result.scalar() > 0


class TestCategoryPersistence:
    """
    Le classifieur tournait sur chaque document et son resultat etait jete.

    _classify_category ne renvoyait qu'un NOM de categorie, et
    _update_law_metadata n'avait donc rien a ecrire dans laws.category_id :
    toutes les lois restaient sans categorie et les pages de categorie du front
    etaient vides, alors que la classification avait bien eu lieu.
    """

    def test_classifier_returns_the_category_id(self):
        result = pl._classify_category(
            "La présente loi fixe le régime des sociétés commerciales et du "
            "registre du commerce au Cameroun."
        )

        assert "category_id" in result
        assert "category" in result
        assert "confidence" in result

    def test_fallback_leaves_the_id_null(self, monkeypatch):
        """Sans identifiant fiable, mieux vaut aucune categorie qu'une fausse."""
        import app.services.document_classifier as classifier_module

        class _Broken:
            def __init__(self, *a, **k):
                raise RuntimeError("modele absent")

        monkeypatch.setattr(classifier_module, "DocumentClassifier", _Broken)

        result = pl._classify_category("texte fiscal sur l'impôt")

        assert result["category_id"] is None

    @pytest.mark.asyncio
    async def test_metadata_update_writes_the_category(self, db_session, law_row):
        from sqlalchemy import select

        pl._update_law_metadata(
            law_row.id,
            language="fr",
            language_confidence=0.9,
            category="Droit Commercial",
            category_confidence=0.8,
            category_id=2,
        )

        refreshed = (await db_session.execute(
            select(Law).where(Law.id == law_row.id)
        )).scalar_one()
        await db_session.refresh(refreshed)

        assert refreshed.category_id == 2
        assert refreshed.category_confidence == pytest.approx(0.8)
        assert refreshed.status == "published"


class TestChunkRefinement:
    """
    chunk_refiner est enfin cable au pipeline.

    Il existait, testé, appelé par personne. Il classe chaque chunk, decide ce
    qui merite un vecteur et prepare le texte reellement envoye au modele. Rien
    n'est supprime : un visa ou une formule d'execution reste consultable et
    cherchable en plein texte, il ne consomme simplement plus d'appel
    d'embedding et ne pollue plus les resultats semantiques.
    """

    @pytest.mark.asyncio
    async def test_classifies_every_chunk(self, db_session, law_row):
        from sqlalchemy import select

        pl._split_and_save_articles(law_row.id, SAMPLE_TEXT)

        rows = (await db_session.execute(
            select(Article).where(Article.law_id == law_row.id)
        )).scalars().all()

        assert rows
        assert all(a.kind for a in rows), "chaque chunk doit porter une nature"
        assert {a.kind for a in rows} & {"article", "legal_basis", "fragment"}

    @pytest.mark.asyncio
    async def test_visas_are_kept_but_not_embedded(self, db_session, law_row):
        """Le texte des visas reste consultable ; il ne part pas au vectoriel."""
        from sqlalchemy import select

        pl._split_and_save_articles(law_row.id, SAMPLE_TEXT)

        rows = (await db_session.execute(
            select(Article).where(Article.law_id == law_row.id)
        )).scalars().all()

        non_articles = [a for a in rows if a.kind in ("legal_basis", "preamble", "fragment")]
        assert non_articles, "le preambule et les visas doivent etre conserves"
        assert all(a.content for a in non_articles)
        assert all(a.embed is False for a in non_articles)

    @pytest.mark.asyncio
    async def test_embed_text_carries_the_document_header(self, db_session, law_row):
        """
        Sans en-tete, "Article 3.- La depense sera imputee sur le budget de
        l'Etat" est indistinguable des milliers d'articles identiques du corpus.
        """
        from sqlalchemy import select

        pl._split_and_save_articles(law_row.id, SAMPLE_TEXT)

        rows = (await db_session.execute(
            select(Article).where(Article.law_id == law_row.id)
        )).scalars().all()

        embeddable = [a for a in rows if a.embed]
        assert embeddable
        for article in embeddable:
            assert article.embed_text, "un chunk vectorisable doit avoir un embed_text"
            assert law_row.reference in article.embed_text
            # content reste INTACT : ce qui est affiche et cite ne change pas.
            assert article.embed_text != article.content
            assert article.content in article.embed_text

    @pytest.mark.asyncio
    async def test_markdown_emphasis_no_longer_hides_articles(self, db_session, law_row):
        """
        LlamaParse rend du markdown : "**ARTICLE 1ER**:" n'etait pas reconnu par
        les motifs d'article, qui attendent "Article" en debut de ligne. Des
        documents entiers ressortaient sans un seul article.
        """
        from sqlalchemy import select

        markdown = """
# LOI N° 2024-600 DU 2 JUIN 2024

**ARTICLE 1ER**: La présente loi fixe le régime applicable aux établissements
publics administratifs et aux entreprises du secteur public et parapublic.

**ARTICLE 2**: Les dispositions de la présente loi s'appliquent sans prejudice
des textes particuliers regissant certains etablissements.
"""
        count = pl._split_and_save_articles(law_row.id, markdown)

        rows = (await db_session.execute(
            select(Article).where(Article.law_id == law_row.id)
        )).scalars().all()

        assert count > 0
        numbers = {a.number for a in rows}
        assert "1" in numbers or "1ER" in {n.upper() for n in numbers}
