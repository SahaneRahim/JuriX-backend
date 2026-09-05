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
        # normalize_article_number convertit 'premier' -> '1'. C'est voulu :
        # une reference stable ('Article 1') est citable, pas 'Article premier'.
        assert articles[0]['number'] == '1'
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

    def test_short_text_is_accepted(self):
        """
        Un texte court n'est plus rejete.

        Le seuil de 200 caracteres a ete retire volontairement du chunker pour
        accepter les documents courts — la majorite du corpus prc.cm est faite
        de decrets d'une a deux pages. Seul le texte vide reste une erreur.
        """
        articles = extract_articles("Article 1. Test.", strict=False, min_article_length=1)
        assert isinstance(articles, list)

    def test_empty_text_raises_value_error(self):
        """Le texte vide reste refuse."""
        with pytest.raises(ValueError):
            extract_articles("   ")

    def test_text_without_articles_falls_back_to_paragraphs(self):
        """
        Un texte sans marqueur d'article n'est plus une erreur.

        Le chunker retombe sur un decoupage par paragraphes (PARA_n). C'est
        indispensable pour le corpus reel : beaucoup de documents scannes
        n'exposent aucun "Article N" exploitable apres OCR.
        """
        text = "This is a long text without any article markers. " * 20
        chunks = extract_articles(text, strict=False)
        assert chunks, "le repli doit produire au moins un chunk"
        # Selon la structure du texte, le repli produit soit des paragraphes
        # (PARA_n) soit un unique bloc FULL_TEXT. Les deux sont acceptables :
        # ce qui compte est qu'aucun contenu ne soit perdu.
        assert all(
            str(c["number"]).startswith(("PARA_", "FULL_TEXT")) for c in chunks
        ), [c["number"] for c in chunks]
        assert sum(c["char_count"] for c in chunks) > 0

    def test_less_than_3_articles_is_accepted(self):
        """
        Moins de 3 articles n'est plus une erreur.

        Le minimum de 3 articles a ete retire : la majorite des decrets du corpus
        n'en comptent que 2 ou 3, et les lois de ratification un seul.
        """
        text = """
        Article 1. Premier article
        Contenu du premier article avec assez de texte pour dépasser 200 caractères minimum requis.

        Article 2. Deuxième article
        Contenu du deuxième article avec assez de texte pour validation et dépasser le seuil.
        """

        articles = extract_articles(text, strict=True)
        assert len(articles) == 2

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
        """
        Les articles sous min_article_length sont ecartes.

        Le seuil porte sur le contenu REELLEMENT conserve. Le titre n'est plus
        retire du contenu (il y est copie, pas deplace), donc un article dont la
        premiere ligne ressemble a un titre compte desormais cette ligne dans sa
        longueur. Article 4 ("Very short\\ny", 12 caracteres) passe donc le seuil
        de 10, alors qu'il tombait a 1 caractere quand le titre etait ampute.
        C'est le comportement voulu : ces 12 caracteres sont du texte reel.
        """
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

        # Article 1 ("Short\nx", 7 caracteres) reste sous le seuil.
        assert [a['number'] for a in articles] == ['2', '3', '4']

        # Le titre est bien present dans le contenu, pas seulement dans `title`.
        article_4 = articles[-1]
        assert article_4['title'] == 'Very short'
        assert 'Very short' in article_4['content']

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


class TestConservationDuTexte:
    """
    Rien de ce qui est extrait ne doit disparaître entre le texte et les chunks.

    Deux fuites silencieuses existaient, toutes deux reproduites ici :

      1. Le texte situé entre un en-tête TITRE/CHAPITRE et l'article suivant
         n'était rattaché à personne — ni à l'article précédent, borné par cet
         en-tête, ni au suivant, dont le contenu ne commence qu'à son propre
         motif. Sur les codes, où chaque titre s'ouvre par un paragraphe de
         portée, et sur les annexes introduites par un CHAPITRE, la fin du
         document disparaissait entièrement.

      2. La première phrase d'un article partait dans `title`, champ qu'aucun
         endpoint n'expose et que `search_service` ignore pour ses extraits.
    """

    def _tous_les_champs(self, chunks):
        return " ".join(
            str(c.get(champ) or "")
            for c in chunks
            for champ in ("title", "section", "content")
        )

    def test_chapeau_de_titre_conserve(self):
        text = """Article 1.- Les dispositions generales s'appliquent a tous.
TITRE II
DES DISPOSITIONS PARTICULIERES
Le present titre s'applique aux collectivites territoriales decentralisees.
Article 2.- Autre disposition."""

        chunks = extract_articles(text, strict=False, min_article_length=1)
        contenus = " ".join(c["content"] for c in chunks)

        assert "collectivites territoriales decentralisees" in contenus
        # Émis comme chunk distinct : il n'appartient à aucun des deux articles.
        assert any(c["number"].startswith("SECTION_") for c in chunks)

    def test_annexe_apres_dernier_article_conservee(self):
        text = """Article 2.- Le present decret sera enregistre.
ANNEXE
CHAPITRE I - LISTE DES BENEFICIAIRES
Monsieur X, matricule 765 609-Y
Texte tres important de l'annexe."""

        chunks = extract_articles(text, strict=False, min_article_length=1)

        assert "Monsieur X" in self._tous_les_champs(chunks)
        assert "tres important" in " ".join(c["content"] for c in chunks)

    def test_premiere_phrase_reste_dans_le_contenu(self):
        text = """Article 1.
La presente loi fixe le regime des marches publics.
Elle s'applique a tous les contrats."""

        article = extract_articles(text, strict=False, min_article_length=1)[0]

        # Copiée dans `title`, PAS retirée du contenu.
        assert article["title"] == "La presente loi fixe le regime des marches publics"
        assert "La presente loi fixe le regime" in article["content"]
        assert "Elle s'applique a tous les contrats" in article["content"]

    def test_en_tete_de_section_sur_deux_lignes_conserve(self):
        # SECTION_PATTERNS capture au-delà du saut de ligne : la seconde ligne
        # de l'en-tête doit se retrouver dans `section`, jamais nulle part.
        text = """Article 1.- Dispositions generales.
TITRE II
DES DISPOSITIONS PARTICULIERES
Contenu du titre.
Article 2.- Suite."""

        chunks = extract_articles(text, strict=False, min_article_length=1)
        assert "DES DISPOSITIONS PARTICULIERES" in self._tous_les_champs(chunks)

class TestMarkdownArticleHeadings:
    """
    Les formes reellement produites par LlamaParse.

    Le motif attendait `Article <numero>` en debut de ligne. Le corpus ecrit
    `**ARTICLE 1er.**-` et parfois `**Article1er.-**` : emphase markdown avant
    le mot, et parfois aucune espace avant le numero. Mesure sur les 27 lois de
    la base : 8 avaient au moins un numero reconnu, contre 27 apres correction ;
    et sur le seul Code Minier, 41 articles indexes contre 193 presents.

    Ces tests assertent les NUMEROS, pas un decompte : un test qui compte les
    chunks passe encore quand les numeros sont faux.
    """

    def _numbers(self, text):
        return [c["number"] for c in extract_articles(text, strict=False, min_article_length=1)]

    def test_bold_with_a_space(self):
        text = ("**ARTICLE 1er.**- L'Abbé BELL Mathias est nommé Secrétaire Général "
                "du Ministère des Finances pour un mandat de trois ans.\n\n"
                "**ARTICLE 2.**- Le présent décret sera enregistré et publié au "
                "Journal Officiel en français et en anglais.")
        assert self._numbers(text) == ["1", "2"]

    def test_bold_with_a_colon(self):
        text = ("**Article 3** : Les Commissaires Divisionnaires de police sont "
                "nommés par décret du Président de la République.\n\n"
                "**Article 4** : La dépense résultant des présentes dispositions "
                "est imputée sur le budget de l'Etat.")
        assert self._numbers(text) == ["3", "4"]

    def test_no_space_before_the_number(self):
        text = ("Article1er.-  Monsieur NGOUNBE Zacharie est nommé Directeur des "
                "Affaires Générales au Ministère de la Justice.")
        assert self._numbers(text) == ["1"]

    def test_an_inline_cross_reference_is_not_a_marker(self):
        """
        Le motif utilise [ \t]* et jamais \s* devant le mot : \s avale les sauts
        de ligne et couperait un article a chaque renvoi interne.
        """
        text = ("Article 1 : Les dispositions du présent décret sont applicables "
                "sur toute l'étendue du territoire national.\n"
                "La présente mesure complète l'article 5 du décret antérieur, "
                "lequel demeure applicable pour ses autres dispositions.")
        assert self._numbers(text) == ["1"]

    def test_classic_forms_are_untouched(self):
        text = ("Article 1. Première disposition du texte, suffisamment longue "
                "pour etre retenue par le decoupeur.\n\n"
                "Article 2. Seconde disposition du texte, elle aussi de longueur "
                "raisonnable pour le decoupage.")
        assert self._numbers(text) == ["1", "2"]
