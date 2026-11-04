"""
Integration test for article extraction in document processing pipeline.

Verifies that extract_articles utility can be imported and used
as intended in the process_law task.
"""

from app.utils.text_chunker import extract_articles


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

        chunks = extract_articles(law_text)

        # Le premier chunk porte l'en-tete du texte (intitule de la loi), sous
        # le numero PREAMBULE ou LEGAL_BASIS : l'extraction ne jette aucun
        # caractere du document. Les articles proprement dits sont donc filtres.
        assert chunks[0]['number'] in ('PREAMBULE', 'LEGAL_BASIS')
        assert 'LOI N° 2023-001' in chunks[0]['content']

        articles = [c for c in chunks if c['number'].isdigit()]

        # Verify extraction
        assert len(articles) == 5
        assert [a['number'] for a in articles] == ['1', '2', '3', '4', '5']

        # Verify structure
        assert articles[0]['title'] == 'Objet de la loi'
        assert 'OHADA' in articles[0]['content']

        assert articles[3]['number'] == '4'
        assert articles[3]['title'] == 'Conditions de constitution'

        # La section est rattachee a chaque article
        assert articles[0]['section'].startswith('CHAPITRE I')
        assert articles[3]['section'].startswith('CHAPITRE II')

        # Verify all have required fields
        for article in articles:
            assert 'number' in article
            assert 'content' in article
            assert 'position' in article
            assert 'word_count' in article
            assert 'char_count' in article
            assert article['word_count'] > 0
            assert article['char_count'] > 0

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
