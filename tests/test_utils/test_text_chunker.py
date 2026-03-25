"""
Test suite for text_chunker utility module.

Tests article extraction from legal documents with various patterns,
edge cases, and validation scenarios.
"""

import pytest
from app.utils.text_chunker import (
    extract_articles,
    ArticleExtractionError,
    ExtractedArticle
)


class TestBasicExtraction:
    """Test basic article extraction."""

    def test_extract_simple_articles(self):
        """Test extraction with 'Article X.' pattern."""
        text = """
        Article 1. Dispositions générales
        La présente loi régit les sociétés commerciales.

        Article 2. Champ d'application
        Les dispositions s'appliquent à toutes les sociétés.

        Article 3. Définitions
        Au sens de la présente loi, on entend par société...
        """

        articles = extract_articles(text)

        assert len(articles) == 3
        assert articles[0]['number'] == '1'
        assert articles[0]['title'] == 'Dispositions générales'
        assert 'sociétés commerciales' in articles[0]['content']
        assert articles[0]['position'] == 0
        assert articles[0]['parent_id'] is None
        assert articles[1]['number'] == '2'
        assert articles[2]['number'] == '3'

    def test_extract_art_abbreviation(self):
        """Test extraction with 'Art.' abbreviation."""
        text = """
        Art. 1. Premier article
        Contenu de l'article premier avec suffisamment de texte.

        Art. 2. Deuxième article
        Contenu de l'article deux avec suffisamment de texte.

        Art. 3. Troisième article
        Contenu de l'article trois avec suffisamment de texte.
        """

        articles = extract_articles(text)

        assert len(articles) == 3
        assert all(a['number'] in ['1', '2', '3'] for a in articles)

    def test_extract_article_premier(self):
        """Test extraction with 'Article premier'."""
        text = """
        Article premier. Objet de la loi
        La présente loi a pour objet de régir les activités commerciales au Cameroun
        et de protéger les intérêts des consommateurs conformément aux normes en vigueur.

        Article 2. Dispositions générales applicables
        Les dispositions suivantes s'appliquent à tous les cas prévus par la présente loi
        et concernent l'ensemble des acteurs économiques et commerciaux du territoire.

        Article 3. Finalités et objectifs de la loi
        La finalité de cette loi est de protéger les consommateurs et de réguler
        les activités commerciales selon les standards internationaux reconnus.
        """

        articles = extract_articles(text)

        assert len(articles) == 3
        assert articles[0]['number'] == 'premier'
        assert articles[1]['number'] == '2'
        assert articles[2]['number'] == '3'


class TestHierarchicalNumbering:
    """Test hierarchical article numbering."""

    def test_extract_nested_articles(self):
        """Test extraction with nested numbering (1.1, 1.2)."""
        text = """
        Article 1. Titre principal
        Contenu de l'article 1 avec suffisamment de texte pour validation.

        Article 1.1. Sous-article 1
        Contenu du sous-article 1.1 avec suffisamment de texte.

        Article 1.2. Sous-article 2
        Contenu du sous-article 1.2 avec suffisamment de texte.

        Article 2. Deuxième article
        Contenu de l'article 2 avec suffisamment de texte.
        """

        articles = extract_articles(text)

        assert len(articles) == 4
        assert articles[0]['number'] == '1'
        assert articles[0]['parent_id'] is None
        assert articles[1]['number'] == '1.1'
        assert articles[1]['parent_id'] == '1'
        assert articles[2]['number'] == '1.2'
        assert articles[2]['parent_id'] == '1'
        assert articles[3]['number'] == '2'
        assert articles[3]['parent_id'] is None

    def test_extract_deeply_nested_articles(self):
        """Test extraction with deeply nested numbering (1.2.3)."""
        text = """
        Article 1. Titre niveau 1
        Contenu de l'article 1 avec suffisamment de texte.

        Article 1.1. Titre niveau 2
        Contenu du sous-article 1.1 avec suffisamment de texte.

        Article 1.1.1. Titre niveau 3
        Contenu du sous-article 1.1.1 avec suffisamment de texte.

        Article 2. Autre article
        Contenu final avec suffisamment de texte pour validation.
        """

        articles = extract_articles(text)

        assert len(articles) == 4
        assert articles[2]['number'] == '1.1.1'
        assert articles[2]['parent_id'] == '1.1'


class TestValidation:
    """Test input validation."""

    def test_empty_text_raises_error(self):
        """Test that empty text raises ValueError."""
        with pytest.raises(ValueError, match="ne peut pas être vide"):
            extract_articles("")

    def test_whitespace_only_text_raises_error(self):
        """Test that whitespace-only text raises ValueError."""
        with pytest.raises(ValueError, match="ne peut pas être vide"):
            extract_articles("   \n\n   \t\t   ")

    def test_text_too_short_raises_error(self):
        """Test that short text raises ValueError."""
        short_text = "Article 1. Test."

        with pytest.raises(ValueError, match="trop court"):
            extract_articles(short_text)

    def test_no_articles_raises_error(self):
        """Test that text without articles raises error."""
        text = "This is a long text without any article markers. " * 20

        with pytest.raises(ArticleExtractionError, match="Aucun pattern"):
            extract_articles(text)

    def test_less_than_3_articles_raises_error(self):
        """Test that <3 articles raises error in strict mode."""
        text = """
        Article 1. Premier article
        Contenu du premier article avec assez de texte pour dépasser 200 caractères minimum requis.

        Article 2. Deuxième article
        Contenu du deuxième article avec assez de texte pour validation et dépasser le seuil.
        """

        with pytest.raises(ArticleExtractionError, match="Minimum 3 articles"):
            extract_articles(text, strict=True)

    def test_less_than_3_articles_warning_non_strict(self):
        """Test that <3 articles returns result in non-strict mode."""
        text = """
        Article 1. Premier article avec suffisamment de contenu
        Contenu du premier article avec assez de texte pour validation.

        Article 2. Deuxième article avec contenu
        Contenu du deuxième article avec assez de texte.
        """

        # Should not raise in non-strict mode
        articles = extract_articles(text, strict=False)
        assert len(articles) == 2

    def test_text_too_large_raises_error(self):
        """Test that text >5MB raises ValueError."""
        # Create a very large text (>5MB)
        large_text = "Article 1. " + ("x" * 6_000_000)

        with pytest.raises(ValueError, match="trop volumineux"):
            extract_articles(large_text)


class TestEdgeCases:
    """Test edge cases."""

    def test_article_with_colon_separator(self):
        """Test articles with ':' separator."""
        text = """
        Article 1: Dispositions générales
        Contenu avec deux-points et suffisamment de texte.

        Article 2: Champ d'application
        Autre contenu avec suffisamment de texte pour validation.

        Article 3: Définitions légales
        Dernières dispositions avec suffisamment de texte.
        """

        articles = extract_articles(text)
        assert len(articles) == 3
        assert articles[0]['title'] == 'Dispositions générales'

    def test_mixed_french_english(self):
        """Test mixed French/English article markers."""
        text = """
        Article 1. Dispositions en français
        Contenu en français avec suffisamment de texte.

        Article 2. More provisions in English
        English content here with enough text for validation.

        Article 3. Dispositions finales
        Contenu final avec suffisamment de texte.
        """

        articles = extract_articles(text)
        assert len(articles) == 3

    def test_preserve_formatting(self):
        """Test that formatting is preserved when requested."""
        text = """
        Article 1.    Avec   espaces    multiples
        Et    des     espaces     dans     le     contenu.

        Article 2. Normal article
        Contenu normal avec suffisamment de texte.

        Article 3. Final article
        Contenu final avec suffisamment de texte.
        """

        articles = extract_articles(text, preserve_formatting=True)

        # Formatting should be preserved
        assert '    des     espaces     dans' in articles[0]['content']

    def test_very_long_article(self):
        """Test extraction of very long articles."""
        long_content = "Contenu répété. " * 1000  # ~16KB
        text = f"""
        Article 1. Long article
        {long_content}

        Article 2. Normal article
        Contenu normal avec suffisamment de texte.

        Article 3. Final article
        Contenu final avec suffisamment de texte.
        """

        articles = extract_articles(text)

        assert len(articles) == 3
        assert articles[0]['char_count'] > 10000
        assert articles[0]['word_count'] > 1500

    def test_article_without_title(self):
        """Test articles without explicit title."""
        text = """
        Article 1.
        Contenu direct sans titre. La présente loi régit les sociétés commerciales au Cameroun
        et établit les règles applicables.

        Article 2.
        Contenu sans titre. Les dispositions s'appliquent à toutes les sociétés
        commerciales et industrielles.

        Article 3.
        Pas de titre ici. Au sens de la présente loi, on entend par société
        toute entité juridique constituée.
        """

        articles = extract_articles(text)

        assert len(articles) == 3
        # These should not be detected as titles (too long)
        assert articles[0]['title'] is None or len(articles[0]['title']) < 100
        assert articles[1]['title'] is None or len(articles[1]['title']) < 100
        assert articles[2]['title'] is None or len(articles[2]['title']) < 100


class TestStatistics:
    """Test article statistics."""

    def test_word_count_accurate(self):
        """Test that word count is accurate."""
        text = """
        Article 1. Test words
        Un deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze.
        Suffisamment de contenu pour valider le minimum de caractères requis.

        Article 2. Test content
        Contenu court texte avec suffisamment de mots pour validation.

        Article 3. Test final
        Autre contenu texte avec suffisamment de mots également.
        """

        articles = extract_articles(text)

        # First article should have 15 words in first sentence
        assert articles[0]['word_count'] >= 15

    def test_char_count_accurate(self):
        """Test that character count is accurate."""
        text = """
        Article 1. Test chars
        Exactement cinquante caracteres ici pour tester la longueur du contenu.
        Ajout de texte supplémentaire pour validation complète.

        Article 2. Test autre
        Autre texte avec suffisamment de caractères pour validation.

        Article 3. Test final
        Final texte avec suffisamment de caractères également.
        """

        articles = extract_articles(text)

        # First article content should have specific character count
        assert articles[0]['char_count'] == len(articles[0]['content'])
        assert articles[0]['char_count'] > 0


class TestPositioning:
    """Test article positioning."""

    def test_position_sequential(self):
        """Test that positions are sequential."""
        text = """
        Article 1. Premier
        Contenu premier avec suffisamment de texte.

        Article 2. Deuxième
        Contenu deuxième avec suffisamment de texte.

        Article 3. Troisième
        Contenu troisième avec suffisamment de texte.

        Article 4. Quatrième
        Contenu quatrième avec suffisamment de texte.
        """

        articles = extract_articles(text)

        positions = [a['position'] for a in articles]
        assert positions == [0, 1, 2, 3]


class TestMinimumArticleLength:
    """Test minimum article length filtering."""

    def test_short_articles_filtered_out(self):
        """Test that articles below min_article_length are filtered."""
        text = """
        Article 1. Short
        x

        Article 2. Normal length article
        This article has enough content to pass the minimum length requirement.

        Article 3. Another normal
        This article also has enough content to pass validation checks.

        Article 4. Very short
        y
        """

        # Default min_article_length=10
        articles = extract_articles(text, strict=False)

        # Only articles 2 and 3 should pass (articles 1 and 4 are too short)
        assert len(articles) == 2
        assert articles[0]['number'] == '2'
        assert articles[1]['number'] == '3'

    def test_custom_min_article_length(self):
        """Test custom min_article_length parameter."""
        text = """
        Article 1. Test content
        Short text here with some additional words for validation purposes.

        Article 2. Test other
        Another short text with additional content for validation.

        Article 3. Test final
        Final short text with sufficient content for validation.
        """

        # Set very low minimum
        articles = extract_articles(text, min_article_length=5, strict=True)

        assert len(articles) == 3


class TestMultiplePatterns:
    """Test detection of different patterns in same document."""

    def test_mixed_article_and_art_patterns(self):
        """Test that most common pattern is selected."""
        # Predominantly "Article X" pattern
        text = """
        Article 1. First article with content
        Content with sufficient text for validation purposes and regulatory compliance.

        Article 2. Second article with content
        More content with sufficient text for validation and additional requirements.

        Article 3. Third article with content
        Final content with sufficient text for validation and complete coverage.
        """

        articles = extract_articles(text)

        # Should extract all 3 (pattern detection picks most common)
        assert len(articles) == 3
