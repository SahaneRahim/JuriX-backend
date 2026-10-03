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

import pytest

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


class TestCachetFonduDansLeTexte:
    """
    L'OCR en pleine page mele parfois le tampon a une ligne de texte.

    Toute ligne portant deux expressions du vocabulaire etait supprimee entiere :
    l'article qui la partageait avec le cachet disparaissait avec lui.
    """

    def test_article_garde_son_texte(self):
        md = (
            "Article 2.- Le présent décret sera enregistré et publié au Journal "
            "Officiel en français et en anglais. COPIE CERTIFIEE CONFORME "
            "CERTIFIED TRUE COPY"
        )
        out = strip_stamp_blocks(md)
        assert "Article 2.- Le présent décret sera enregistré" in out
        assert "CERTIFIED TRUE COPY" not in out
        assert "COPIE CERTIFIEE" not in out

    def test_formule_d_execution_intacte(self):
        """Deux autorités légitimes sur une ligne ne font pas un cachet."""
        md = (
            "Le Secrétariat Général de la Présidence de la République est chargé "
            "de l'exécution du présent décret."
        )
        assert strip_stamp_blocks(md) == md

    def test_en_tete_d_autorite_sur_deux_lignes_conserve(self):
        md = "PRESIDENCE DE LA REPUBLIQUE\nSECRETARIAT GENERAL\n\nDECRET N° 2024/191"
        out = strip_stamp_blocks(md)
        assert "PRESIDENCE DE LA REPUBLIQUE" in out
        assert "SECRETARIAT GENERAL" in out

    def test_cachet_au_milieu_d_une_ligne(self):
        md = (
            "Article 3.- Le Ministre des Finances PRESIDENCE DE LA REPUBLIQUE "
            "COPIE CERTIFIEE CONFORME est chargé de l'application du présent texte."
        )
        out = strip_stamp_blocks(md)
        assert "COPIE CERTIFIEE" not in out
        assert "Article 3.- Le Ministre des Finances" in out
        assert "est chargé de l'application du présent texte." in out


class TestCachetEcorcheParLOcr:
    """
    Variantes relevees sur les sorties Docling du corpus. Avec des espaces
    obligatoires et sans confusion de lettres admise, le cachet restait dans un
    tiers des pages.
    """

    @pytest.mark.parametrize("cachet", [
        "COPIECERTIFIEECONFORME",
        "COPIECERTIFIBECONFORME",
        "COPIE CERTIFIEE CONFONME",
        "CERTIFIEDTRUECOPY",
        "CERTIFIED TRUESOPY",
        "SERVICE DUFICHIERLEGISLATIFETREGLEMENTAIRE",
        "SERTICE DU FICHIER LECICLATIF ET REGLENENTAIRE",
        "BERVICE DU PICHIER LEGISLATIF ET REGCEMENTAIRE",
        "LENSLATIVE ASD STATUTORY APFAIRS CARD IBDEX SERVICE",
        "LEOISLATIVE ABD STATUTORT APFAIRS CARO INDEX SERVICE",
        "LEGISLATIVEANDSTATUTORYAFFAIBS",
        "PREOIDENCEDE LAREPUBLIQUE PRBSIDENCYOF THEREPUBLIC ORCRETARIATGINERAL "
        "SERVICE DU PIGHIER LEGIBLATIF ET REGLENE",
        "CERTIFIED",
    ])
    def test_variante_retiree(self, cachet):
        assert strip_stamp_blocks("Article 2.- Texte de l'article.\n\n" + cachet) == (
            "Article 2.- Texte de l'article."
        )

    def test_cachet_dans_une_cellule_de_liste(self):
        """Le cachet est retire, les personnes de la meme ligne restent."""
        ligne = "| | CERTIFIED TRUE COPY 1. ENEMBI EGONG AMOUR P. 2. HAMI ABANI | 583 171-G 583 198-M |"
        out = strip_stamp_blocks(ligne)
        assert "CERTIFIED" not in out
        assert "ENEMBI EGONG AMOUR" in out and "583 198-M" in out

    def test_piece_a_fournir_en_minuscules_conservee(self):
        """
        Le cachet est toujours en capitales. « copie certifiee conforme » en
        minuscules est une piece a fournir, pas un cachet.
        """
        md = (
            "Article 5.- Le dossier comprend :\n"
            "- une copie certifiée conforme de l'acte de naissance ;\n"
            "- une copie certifiée conforme du diplôme requis."
        )
        assert strip_stamp_blocks(md) == md

    def test_titre_en_capitales_proche_du_cachet_conserve(self):
        md = "CHAPITRE III : DU CERTIFICAT DE CONFORMITE\n\nArticle 9.- Le certificat est délivré."
        assert strip_stamp_blocks(md) == md


class TestFiligrane:
    def test_filigrane_rendu_en_tableau(self):
        """Docling rend parfois le filigrane en tableau d'une cellule."""
        md = "Article 1.- Texte.\n\n| www.prc.cm   |\n|--------------|\n\nArticle 2.- Suite."
        assert strip_stamp_blocks(md) == "Article 1.- Texte.\n\n\nArticle 2.- Suite."

    def test_vrai_tableau_conserve(self):
        md = "| Poste | Montant |\n|---|---|\n| Fonctionnement | 1 200 000 |"
        assert strip_stamp_blocks(md) == md


class TestMarqueursDocling:
    """
    Le sommaire de la page de lecture cherche « Article » et « CHAPITRE » en
    debut de ligne, dans le texte affiche brut.
    """

    @pytest.mark.parametrize("brut, attendu", [
        ("## ARTICLE 2.- L'EGCIM a pour missions :", "ARTICLE 2.- L'EGCIM a pour missions :"),
        ("- ARTICLE 12.- (1) Les titres miniers", "ARTICLE 12.- (1) Les titres miniers"),
        ("- ARTiCLE 7.- (1) Le Conseil", "ARTiCLE 7.- (1) Le Conseil"),
        ("2. ARTICLE 41.- (1) Le Conseil", "ARTICLE 41.- (1) Le Conseil"),
        (". ARTICLE 31.- (1) Le Conseil", "ARTICLE 31.- (1) Le Conseil"),
        ("- ARTICLE 14-Les opérations", "ARTICLE 14-Les opérations"),
        ("ARTICLE_1er.- Le présent décret", "ARTICLE 1er.- Le présent décret"),
        ("- Article premier : Objet", "Article premier : Objet"),
        ("## CHAPITRE II DES MINES", "CHAPITRE II DES MINES"),
    ])
    def test_marqueur_mis_a_plat(self, brut, attendu):
        from app.utils.markdown_cleanup import nettoyer_markdown

        assert nettoyer_markdown(brut) == attendu

    @pytest.mark.parametrize("ligne", [
        "- article 12 de la loi n° 2001/015 du 23 juillet 2001 ;",
        "- les articles 3 et 4 du décret susvisé ;",
        "1. Le Conseil d'administration ;",
    ])
    def test_renvoi_en_liste_conserve(self, ligne):
        from app.utils.markdown_cleanup import nettoyer_markdown

        assert nettoyer_markdown(ligne) == ligne

    def test_cachet_retire_aussi(self):
        from app.utils.markdown_cleanup import nettoyer_markdown

        assert nettoyer_markdown("- ARTICLE 3.- Texte.\n\nCOPIECERTIFIEECONFORME") == "ARTICLE 3.- Texte."


@pytest.mark.parametrize("brut, attendu", [
    ("ARTIiCLE 81.- Le permis de reconnaissance", "ARTICLE 81.- Le permis de reconnaissance"),
    ("ARTICLÈ 114.- (1) La détention", "ARTICLE 114.- (1) La détention"),
    ("4. ARTICLES 170.- (1) Les titres miniers", "ARTICLE 170.- (1) Les titres miniers"),
    ("Articles 12 à 15 de la loi sont abrogés.", "Articles 12 à 15 de la loi sont abrogés."),
])
def test_mot_article_ecorche_par_l_ocr(brut, attendu):
    from app.utils.markdown_cleanup import nettoyer_markdown

    assert nettoyer_markdown(brut) == attendu


@pytest.mark.parametrize("brut, attendu", [
    ("ARTI CLE1er.- Est autorisée", "ARTICLE 1er.- Est autorisée"),
    ("Article ler.- Sont nommés", "Article 1er.- Sont nommés"),
    ("ARTTICLE 1ºr.- Sont approuvés", "ARTICLE 1ºr.- Sont approuvés"),
    ("ARTICIE 18,- L'attribution", "ARTICLE 18,- L'attribution"),
    ("Artice 1 : En application", "Article 1 : En application"),
    ("A ARTiCLE 1er.- Le Ministre", "ARTiCLE 1er.- Le Ministre"),
    ("1 Article 1er : Sont nommés", "Article 1er : Sont nommés"),
    ("_ / ■ ARTiCLE 1ºr.- Est ratifié", "ARTiCLE 1ºr.- Est ratifié"),
    ("ww ARTICLE 1ºr.- Le Président", "ARTICLE 1ºr.- Le Président"),
])
def test_marqueur_de_l_article_premier_ecorche(brut, attendu):
    """
    Le cachet et le sceau recouvrent souvent le marqueur de l'article 1er :
    illisible, l'article partait dans les visas, hors de l'index vectoriel.
    """
    from app.utils.markdown_cleanup import nettoyer_markdown

    assert nettoyer_markdown(brut) == attendu


@pytest.mark.parametrize("ligne", [
    "Vu l'Article 12 : de la loi",
    "A la demande du ministre, article 3 :",
    "2. Le Conseil d'administration ;",
])
def test_ligne_sans_marqueur_intacte(ligne):
    from app.utils.markdown_cleanup import nettoyer_markdown

    assert nettoyer_markdown(ligne) == ligne


@pytest.mark.parametrize("cachet", [
    "CERTIFITDTRUECOPY",
    "CARS INDEX SERVICE COPIECTRTIFIEE CONFORME CERIIFIEGTRUECOPY",
])
def test_cachet_ecorche_retire(cachet):
    assert strip_stamp_blocks("Article 2.- Texte de l'article.\n\n" + cachet) == "Article 2.- Texte de l'article."
