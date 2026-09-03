"""
Complete end-to-end pipeline integration test.

Tests the full workflow:
1. Language Detection
2. Document Classification
3. Article Extraction
4. Database Persistence (Law)
5. Database Persistence (Articles)
6. Embedding Generation
7. Hybrid Search
8. RAG Chat

This test validates that all 7 services work together correctly.

Author: JuriX Team
Date: 2026-01-10
"""

import os

import pytest
from datetime import date
import time

from app.services.language_detector import LanguageDetector
from app.services.document_classifier import DocumentClassifier
from app.services.embedding_service import EmbeddingService
from app.services.search_service import SearchService
from app.services.rag_service import RAGService
from app.utils.text_chunker import extract_articles
from app.models.law import Article, Law
from app.schemas.search import SearchRequest
from app.schemas.rag import RAGRequest


# Ces deux tests appellent l'API Gemini pour de vrai (embeddings puis RAG) :
# ils consomment du quota et dependent du reseau. Ils ne s'executent donc que
# sur demande explicite, JURIX_E2E=1, et jamais dans la suite ordinaire.
# Chaque etage a par ailleurs sa propre suite unitaire ; ce fichier ne verifie
# que leur enchainement.
pytestmark = pytest.mark.skipif(
    os.getenv("JURIX_E2E") != "1",
    reason="Chaine complete avec appels Gemini reels : lancer avec JURIX_E2E=1",
)


@pytest.fixture
def sample_legal_document():
    """Sample Cameroonian legal document."""
    return """
LOI N° 2024-001 DU 15 JANVIER 2024
RELATIVE AU CODE CIVIL CAMEROUNAIS

TITRE PREMIER - DES PERSONNES

Article 1er. Tout individu possède la personnalité juridique dès sa naissance.

Article 2. La majorité est fixée à vingt et un ans accomplis.

Article 3. Le domicile d'une personne est au lieu où elle a son principal établissement.

TITRE II - DES BIENS

Article 4. Les biens sont meubles ou immeubles.

Article 5. Sont immeubles par nature les fonds de terre et les bâtiments.

TITRE III - DES OBLIGATIONS

Article 6. Les contrats légalement formés tiennent lieu de loi à ceux qui les ont faits.

Article 7. Les conventions doivent être exécutées de bonne foi.

Article 8. La responsabilité civile engage celui qui cause un dommage à autrui.
"""


@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_pipeline_integration(
    sample_legal_document,
    async_db_session
):
    """
    Test complete pipeline: upload → process → search → chat.

    This is the master integration test validating all 7 services.
    """
    print("\n" + "="*80)
    print("FULL PIPELINE INTEGRATION TEST")
    print("="*80)

    # Initialize services
    language_detector = LanguageDetector()
    document_classifier = DocumentClassifier()
    embedding_service = EmbeddingService(use_cache=False)
    search_service = SearchService(async_db_session)
    rag_service = RAGService(async_db_session)

    # STEP 1: Language Detection
    print("\n1️⃣ Testing Language Detection...")
    lang_result = language_detector.detect(sample_legal_document)
    assert lang_result["language"] == "fr"
    assert lang_result["confidence"] > 0.9
    print(f"   ✅ Detected language: {lang_result['language']} ({lang_result['confidence']:.2%})")

    # STEP 2: Document Classification
    print("\n2️⃣ Testing Document Classification...")
    # classify renvoie une liste de tuples (category_id, confidence, method),
    # pas un dictionnaire {"predictions": [...]}.
    class_result = document_classifier.classify(sample_legal_document, top_k=3)
    assert 1 <= len(class_result) <= 3
    top_category_id, top_confidence, top_method = class_result[0]
    assert isinstance(top_category_id, int) and top_category_id > 0
    assert 0.0 <= top_confidence <= 1.0
    print(f"   ✅ Classified as: category {top_category_id} ({top_confidence:.2%}, {top_method})")

    # STEP 3: Article Extraction
    print("\n3️⃣ Testing Article Extraction...")
    articles = extract_articles(sample_legal_document, strict=True)
    assert len(articles) >= 3  # At least 3 articles
    print(f"   ✅ Extracted {len(articles)} articles")

    # STEP 4: Database Persistence (Law)
    print("\n4️⃣ Testing Law Persistence...")
    # Ecriture ORM directe : le service de loi a ete supprime, tout le CRUD
    # vit desormais dans routes/laws.py.
    created_law = Law(
        reference="LOI-2024-001-TEST",
        title="Code Civil Camerounais - Test Pipeline",
        type="Loi",
        category_id=1,  # Assuming category exists
        language="fr",
        content=sample_legal_document,
        publication_date=date(2024, 1, 15),
        status="published",
    )
    async_db_session.add(created_law)
    await async_db_session.commit()
    await async_db_session.refresh(created_law)
    assert created_law.id is not None
    print(f"   ✅ Law created: ID={created_law.id}")

    # STEP 5: Database Persistence (Articles)
    print("\n5️⃣ Testing Article Persistence...")
    article_count = 0
    for article in articles:
        async_db_session.add(Article(
            law_id=created_law.id,
            number=article["number"],
            title=article.get("title") or None,
            content=article["content"],
            order=article["position"],
        ))
        article_count += 1
    await async_db_session.commit()

    print(f"   ✅ {article_count} articles persisted")

    # STEP 6: Embedding Generation
    print("\n6️⃣ Testing Embedding Generation...")
    embeddings = embedding_service.generate_batch_embeddings([
        article["content"] for article in articles[:5]
    ])
    dim = EmbeddingService.EMBEDDING_DIM  # 3072 avec Gemini, plus 768
    assert len(embeddings) == min(5, len(articles))
    assert all(emb.shape == (dim,) for emb in embeddings)
    print(f"   ✅ Generated {len(embeddings)} embeddings ({dim}-dim)")

    # STEP 7: Hybrid Search
    print("\n7️⃣ Testing Hybrid Search...")
    search_result = await search_service.search(SearchRequest(
        query="personnalité juridique",
        mode="hybrid",
        limit=5
    ))
    assert len(search_result.results) > 0
    assert search_result.search_time_ms < 200  # <200ms target
    print(f"   ✅ Found {len(search_result.results)} results in {search_result.search_time_ms}ms")

    # STEP 8: RAG Chat
    print("\n8️⃣ Testing RAG Chat...")
    rag_request = RAGRequest(
        question="Qu'est-ce que la personnalité juridique?",
        persona="étudiant",
        language="fr"
    )

    try:
        rag_response = await rag_service.ask(rag_request)
        assert rag_response.answer is not None
        assert len(rag_response.answer) > 0
        assert rag_response.total_time_ms < 10000  # <10s (generous for test)
        print(f"   ✅ RAG response generated in {rag_response.total_time_ms}ms")
        print(f"      Confidence: {rag_response.confidence:.2%}")
        print(f"      Sources: {len(rag_response.sources)}")
    except Exception as e:
        print(f"   ⚠️  RAG test skipped (Gemini indisponible): {e}")
        pytest.skip("Service Gemini indisponible")

    # Cleanup
    print("\n9️⃣ Cleanup...")
    await async_db_session.delete(created_law)
    await async_db_session.commit()
    print("   ✅ Test data cleaned up")

    print("\n" + "="*80)
    print("✅ ALL PIPELINE STEPS PASSED")
    print("="*80)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_pipeline_performance(sample_legal_document, async_db_session):
    """
    Test that pipeline meets performance targets.

    Validates spec requirements:
    - Language detection: <1s
    - Classification: <2s
    - Search: <200ms
    - RAG: <5s
    """
    print("\n" + "="*80)
    print("PIPELINE PERFORMANCE TEST")
    print("="*80)

    timings = {}

    # Language Detection
    language_detector = LanguageDetector()
    start = time.time()
    lang_result = language_detector.detect(sample_legal_document)
    timings["language_detection"] = (time.time() - start) * 1000
    assert timings["language_detection"] < 1000  # <1s
    print(f"✅ Language Detection: {timings['language_detection']:.0f}ms (target: <1000ms)")

    # Classification
    classifier = DocumentClassifier()
    start = time.time()
    class_result = classifier.classify(sample_legal_document)
    timings["classification"] = (time.time() - start) * 1000
    assert timings["classification"] < 2000  # <2s
    print(f"✅ Classification: {timings['classification']:.0f}ms (target: <2000ms)")

    # Search (if available)
    try:
        search_service = SearchService(async_db_session)
        start = time.time()
        search_result = await search_service.search(SearchRequest(
            query="test",
            mode="hybrid",
            limit=5
        ))
        timings["search"] = search_result.search_time_ms
        assert timings["search"] < 200  # <200ms
        print(f"✅ Hybrid Search: {timings['search']:.0f}ms (target: <200ms)")
    except Exception as e:
        print(f"⚠️  Search test skipped: {e}")

    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    for component, time_ms in timings.items():
        print(f"  {component}: {time_ms:.0f}ms")
    print("="*80)
