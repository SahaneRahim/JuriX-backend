"""
Nettoyage du markdown d'extraction.

Ces six tests ont suivi leur code : ils vivaient dans
`test_llama_parse_service.py` et auraient disparu avec lui. Les regles qu'ils
figent ne dependent d'aucun fournisseur — elles decrivent ce que le corpus
prc.cm contient et qu'il ne faut pas indexer — et servent autant a l'extraction
par Gemini qu'a celle qu'elles protegeaient avant.

Usage:
    pytest tests/test_utils/test_markdown_cleanup.py -v
"""

from app.utils.markdown_cleanup import strip_stamp_blocks


class TestNettoyageDesCachets:
    """
    Retrait du cachet officiel apposé sur chaque page du corpus prc.cm.

    L'extracteur restitue ce cachet en texte ; sans nettoyage il pollue le FTS et
    les embeddings de tous les documents.
    """

    def test_bloc_multiligne_retire(self):
        md = (
            "Article 1er. Contenu juridique réel.\n\n"
            "PRESIDENCE DE LA REPUBLIQUE\n"
            "SERVICE DU FICHIER LEGISLATIF ET REGLEMENTAIRE\n"
            "COPIE CERTIFIEE CONFORME\n"
            "CERTIFIED TRUE COPY\n\n"
            "Article 2. Autre contenu."
        )
        out = strip_stamp_blocks(md)
        assert "COPIE CERTIFIEE CONFORME" not in out
        assert "Article 1er. Contenu juridique réel." in out
        assert "Article 2. Autre contenu." in out

    def test_cachet_sur_une_seule_ligne_retire(self):
        """LlamaParse condense parfois tout le cachet sur une ligne."""
        md = (
            "Article 1er. Contenu.\n"
            "[signature: PRESIDENCE DE LA REPUBLIQUE SECRETARIAT GENERAL "
            "COPIE CERTIFIEE CONFORME CERTIFIED TRUE COPY]\n"
        )
        out = strip_stamp_blocks(md)
        assert "CERTIFIED TRUE COPY" not in out
        assert "Article 1er. Contenu." in out

    def test_en_tete_legitime_conserve(self):
        """
        Beaucoup de décrets portent « PRESIDENCE DE LA REPUBLIQUE » comme
        autorité émettrice : c'est du contenu, pas un tampon. Une seule phrase
        du vocabulaire ne doit donc rien déclencher.
        """
        md = "PRESIDENCE DE LA REPUBLIQUE\n\nDECRET N° 2024/191 du 4 juin 2024"
        out = strip_stamp_blocks(md)
        assert "PRESIDENCE DE LA REPUBLIQUE" in out
        assert "DECRET N° 2024/191" in out

    def test_filigrane_retire(self):
        md = "Article 1er.\nw\nw\nw\n.prc\n.cm\nContenu."
        out = strip_stamp_blocks(md)
        assert "Article 1er." in out and "Contenu." in out
        assert "\nw\n" not in f"\n{out}\n"

    def test_tableaux_conserves(self):
        """Les <table> portent du sens (annexes budgétaires) — jamais retirés."""
        md = "<table><tr><td>Recettes</td><td>66 900 000</td></tr></table>"
        assert "<table>" in strip_stamp_blocks(md)

    def test_balises_inline_retirees(self):
        """<sup> casse la détection d'article : ARTICLE 1<sup>ER</sup>."""
        assert strip_stamp_blocks("**ARTICLE 1<sup>ER</sup>**") == "**ARTICLE 1ER**"
