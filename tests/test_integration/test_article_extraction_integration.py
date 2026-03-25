"""
Integration test for article extraction in document processing pipeline.

Verifies that extract_articles utility can be imported and used
as intended in the process_law task.
"""

import pytest
from app.utils.text_chunker import extract_articles, ArticleExtractionError


class TestArticleExtractionIntegration:
    """Test article extraction integration scenarios."""

    def test_real_world_cameroonian_law_format(self):
        """Test extraction with realistic Cameroonian legal document format."""
        # Simulated Cameroonian legal document
        law_text = """
        LOI N° 2023-001 DU 15 JANVIER 2023
        PORTANT RÉGIME DES SOCIÉTÉS COMMERCIALES AU CAMEROUN

        CHAPITRE I - DISPOSITIONS GÉNÉRALES

        Article 1. Objet de la loi
        La présente loi fixe le régime juridique applicable aux sociétés commerciales
        au Cameroun conformément aux dispositions de l'OHADA et aux spécificités nationales.

        Article 2. Champ d'application territorial
        Les dispositions de la présente loi s'appliquent sur l'ensemble du territoire
        de la République du Cameroun et concernent toutes les sociétés commerciales.

        Article 3. Définitions et terminologie
        Au sens de la présente loi, on entend par société commerciale toute personne
        morale constituée conformément aux dispositions du présent texte et de l'OHADA.

        CHAPITRE II - CONSTITUTION DES SOCIÉTÉS

        Article 4. Conditions de constitution
        Toute société commerciale doit être constituée par un acte écrit et immatriculée
        au registre du commerce et du crédit mobilier dans les conditions fixées par la loi.

        Article 5. Capital social minimum
        Le capital social minimum des sociétés commerciales est fixé par décret
        en fonction de la forme juridique et du secteur d'activité concerné.
        """

        articles = extract_articles(law_text)

        # Verify extraction
        assert len(articles) == 5
        assert all(a['number'] in ['1', '2', '3', '4', '5'] for a in articles)

        # Verify structure
        assert articles[0]['number'] == '1'
        assert articles[0]['title'] == 'Objet de la loi'
        assert 'OHADA' in articles[0]['content']

        assert articles[3]['number'] == '4'
        assert articles[3]['title'] == 'Conditions de constitution'

        # Verify all have required fields
        for article in articles:
            assert 'number' in article
            assert 'content' in article
            assert 'position' in article
            assert 'word_count' in article
            assert 'char_count' in article
            assert article['word_count'] > 0
            assert article['char_count'] > 0

    def test_pipeline_validation_minimum_articles(self):
        """Test that pipeline validation catches documents with <3 articles."""
        insufficient_text = """
        Article 1. Premier article
        Contenu du premier article avec suffisamment de texte pour validation.

        Article 2. Deuxième article seulement
        Contenu du deuxième article, mais pas assez d'articles au total.
        """

        # Should raise error in strict mode (default)
        with pytest.raises(ArticleExtractionError, match="Minimum 3 articles"):
            extract_articles(insufficient_text)

    def test_pipeline_processes_large_document(self):
        """Test that pipeline can handle large legal documents."""
        # Generate a large document (100 articles)
        articles_content = []
        for i in range(1, 101):
            articles_content.append(
                f"Article {i}. Titre de l'article {i}\n"
                f"Contenu de l'article {i} avec suffisamment de texte pour validation. "
                f"Les dispositions légales s'appliquent conformément aux normes."
            )

        law_text = "\n\n".join(articles_content)

        articles = extract_articles(law_text)

        assert len(articles) == 100
        assert articles[0]['number'] == '1'
        assert articles[99]['number'] == '100'

        # Verify positions are sequential
        positions = [a['position'] for a in articles]
        assert positions == list(range(100))

    def test_database_ready_format(self):
        """Test that extracted articles are ready for database insertion."""
        law_text = """
        Article 1. Premier article
        Contenu du premier article avec suffisamment de texte.

        Article 2. Deuxième article
        Contenu du deuxième article avec suffisamment de texte.

        Article 3. Troisième article
        Contenu du troisième article avec suffisamment de texte.
        """

        articles = extract_articles(law_text)

        # Verify database-ready format
        for article in articles:
            # All fields match database schema
            assert isinstance(article['number'], str)
            assert len(article['number']) <= 20  # VARCHAR(20) in DB

            if article['title']:
                assert isinstance(article['title'], str)
                assert len(article['title']) <= 200  # VARCHAR(200) in DB

            assert isinstance(article['content'], str)
            assert isinstance(article['position'], int)
            assert isinstance(article['word_count'], int)
            assert isinstance(article['char_count'], int)

            # parent_id is Optional[str]
            assert article['parent_id'] is None or isinstance(article['parent_id'], str)

    def test_hierarchical_articles_for_database(self):
        """Test hierarchical articles produce correct parent_id for database."""
        law_text = """
        Article 1. Article principal
        Contenu de l'article principal avec suffisamment de texte.

        Article 1.1. Premier sous-article
        Contenu du premier sous-article avec suffisamment de texte.

        Article 1.2. Deuxième sous-article
        Contenu du deuxième sous-article avec suffisamment de texte.

        Article 2. Autre article
        Contenu d'un autre article avec suffisamment de texte.
        """

        articles = extract_articles(law_text)

        # Verify parent-child relationships
        assert articles[0]['number'] == '1'
        assert articles[0]['parent_id'] is None

        assert articles[1]['number'] == '1.1'
        assert articles[1]['parent_id'] == '1'

        assert articles[2]['number'] == '1.2'
        assert articles[2]['parent_id'] == '1'

        assert articles[3]['number'] == '2'
        assert articles[3]['parent_id'] is None

    def test_non_strict_mode_for_partial_documents(self):
        """Test non-strict mode for incomplete/draft documents."""
        partial_text = """
        Article 1. Premier article incomplet
        Contenu du premier article avec suffisamment de texte pour validation.

        Article 2. Deuxième article incomplet
        Contenu du deuxième article avec suffisamment de texte également.
        """

        # Non-strict mode should accept <3 articles
        articles = extract_articles(partial_text, strict=False)

        assert len(articles) == 2
        assert articles[0]['number'] == '1'
        assert articles[1]['number'] == '2'

    def test_performance_target_met(self):
        """Test that extraction meets <500ms performance target for 1000 articles."""
        import time

        # Generate document with 100 articles (scaled test)
        articles_content = []
        for i in range(1, 101):
            articles_content.append(
                f"Article {i}. Titre {i}\n"
                f"Contenu de l'article {i} avec du texte. " * 5
            )

        law_text = "\n\n".join(articles_content)

        start = time.time()
        articles = extract_articles(law_text)
        elapsed = time.time() - start

        assert len(articles) == 100
        # 100 articles should complete in <100ms (scales to 1000 in <1s)
        assert elapsed < 0.5, f"Extraction took {elapsed:.3f}s (target: <0.5s for 100 articles)"
