"""
Lecture des citations d'une reponse : positions, rattachement, introuvables.

Le front rend chaque citation cliquable DANS la phrase, et signale celles que
rien ne porte. Ces tests epinglent ce qu'il recoit : la place exacte de la
citation dans le texte affiche, l'extrait qui la porte, ou la raison pour
laquelle aucun ne la porte.

Author: JuriX Team
"""

from datetime import date

import pytest

from app.schemas.search import ChunkResult
from app.services.lecture_des_citations import (
    etendue_du_nom,
    indices_utf16,
    lire_les_citations,
    normaliser_le_nom,
    porte_le_numero,
)

CODE_DU_TRAVAIL = "Loi n° 92/007 du 14 août 1992 portant Code du travail"


def _extrait(article_id, law_id, numero, titre, reference, contenu="Contenu de l'article."):
    return ChunkResult(
        article_id=article_id, law_id=law_id, number=numero, content=contenu,
        reference=reference, law_title=titre, publication_date=date(1992, 8, 14),
        relevance_score=0.8,
    )


@pytest.fixture
def extraits():
    """Le decret est classe premier, et porte lui aussi un article 37."""
    return [
        _extrait(901, 9, "37", "Décret N°2019/143 fixant les modalités", "PRC-9"),
        _extrait(371, 3, "37", CODE_DU_TRAVAIL, "PRC-3"),
        _extrait(401, 3, "40", CODE_DU_TRAVAIL, "PRC-3"),
    ]


def _cite(texte, occurrence):
    return texte[occurrence.debut:occurrence.fin]


class TestPositions:

    def test_la_citation_est_reperee_a_sa_place_exacte(self, extraits):
        texte = "Vous avez droit à une indemnité (l'article 37 du Code du travail)."

        [occurrence] = lire_les_citations(texte, extraits)

        assert _cite(texte, occurrence) == "article 37 du Code du travail"
        assert occurrence.extrait.article_id == 371
        assert occurrence.article == "37"

    def test_la_suite_de_la_phrase_n_est_pas_dans_la_citation(self, extraits):
        """
        CITATION_REGEX court jusqu'a la ponctuation : « Code du travail prévoit
        une indemnité… ». Exige en entier dans le titre, ce nom ne designait
        aucune loi, et l'article 37 du DECRET, classe premier, l'emportait.
        """
        texte = "L'article 37 du Code du travail prévoit une indemnité distincte du préavis."

        [occurrence] = lire_les_citations(texte, extraits)

        assert _cite(texte, occurrence) == "article 37 du Code du travail"
        assert occurrence.extrait.article_id == 371

    def test_le_gras_markdown_reste_hors_de_la_citation(self, extraits):
        texte = "Selon **l'article 37 du Code du travail** le salarié est indemnisé."

        [occurrence] = lire_les_citations(texte, extraits)

        assert _cite(texte, occurrence) == "article 37 du Code du travail"

    def test_un_retour_a_la_ligne_arrete_le_nom(self, extraits):
        texte = "Voir l'article 40 du Code du travail\n\n**Ce que vous pouvez faire**"

        [occurrence] = lire_les_citations(texte, extraits)

        assert _cite(texte, occurrence) == "article 40 du Code du travail"
        assert occurrence.extrait.article_id == 401

    def test_chaque_occurrence_est_rendue_dans_l_ordre_du_texte(self, extraits):
        texte = (
            "L'article 40 du Code du travail impose une réunion, et l'article 37 du "
            "Code du travail fixe l'indemnité. Rappel : l'article 40 du Code du travail."
        )

        occurrences = lire_les_citations(texte, extraits)

        assert [o.extrait.article_id for o in occurrences] == [401, 371, 401]
        assert [o.debut for o in occurrences] == sorted(o.debut for o in occurrences)


class TestIntrouvables:

    def test_un_texte_non_consulte_est_signale_et_non_jete(self, extraits):
        texte = "L'article 312 du Code pénal punit l'escroquerie."

        [occurrence] = lire_les_citations(texte, extraits)

        assert occurrence.raison == "texte_non_consulte"
        assert _cite(texte, occurrence) == "article 312 du Code pénal"

    def test_un_article_absent_d_un_texte_consulte(self, extraits):
        """Le Code du travail est la, pas son article 99 : jamais l'article 99 d'un autre texte."""
        texte = "L'article 99 du Code du travail le prévoit."

        [occurrence] = lire_les_citations(texte, extraits)

        assert occurrence.raison == "article_non_consulte"
        assert occurrence.extrait is None

    def test_un_texte_designe_par_son_numero_et_non_consulte(self, extraits):
        """Le cas de la maquette : un arrete nomme sans article, que rien ne porte."""
        texte = "Le barème est fixé par l'arrêté n° 016/MTPS/SG/CJ du 26 mai 1993."

        [occurrence] = lire_les_citations(texte, extraits)

        assert occurrence.forme == "texte"
        assert occurrence.raison == "texte_non_consulte"
        assert _cite(texte, occurrence) == "arrêté n° 016/MTPS/SG/CJ du 26 mai 1993"

    def test_un_article_d_un_texte_numerote_non_consulte(self, extraits):
        """
        « Décret N°2019/144 » n'est pas le decret consulte (2019/143) : son
        article 37 n'est pas celui du decret present, meme numero ou pas.
        """
        texte = "Selon l'article 37 du Décret N°2019/144 fixant les tarifs, rien."

        [occurrence] = lire_les_citations(texte, extraits)

        assert occurrence.raison == "texte_non_consulte"
        assert _cite(texte, occurrence) == "article 37 du Décret N°2019/144"


class TestTextesNommes:

    def test_un_texte_consulte_designe_par_son_numero_est_une_citation(self, extraits):
        texte = "La loi n° 92/007 du 14 août 1992 s'applique au contrat."

        [occurrence] = lire_les_citations(texte, extraits)

        assert occurrence.forme == "texte"
        assert occurrence.extrait.law_id == 3
        assert occurrence.article is None
        assert _cite(texte, occurrence) == "loi n° 92/007 du 14 août 1992"

    def test_le_nom_d_une_citation_d_article_n_est_pas_compte_deux_fois(self, extraits):
        texte = "Voir l'article 37 du Décret N°2019/143 fixant les modalités."

        occurrences = lire_les_citations(texte, extraits)

        assert len(occurrences) == 1
        assert occurrences[0].extrait.article_id == 901

    def test_un_texte_nomme_par_une_ligne_de_statut_n_est_pas_signale(self):
        """
        Le prompt demande de nommer le texte qui change une regle ; la ligne de
        statut de l'article 14 de la Constitution nomme la loi 2008/001 (lien
        prouve, liens_entre_textes). Le signaler punirait le modele d'avoir obei.
        """
        constitution = _extrait(141, 6283, "14", "Constitution", "PRC-6283")
        texte = "L'article 14 de la Constitution a été modifié par la Loi N°2008/001 du 14 avril 2008."

        occurrences = lire_les_citations(texte, [constitution])

        assert [o.raison for o in occurrences] == [None, None]
        nomme = occurrences[1].texte_nomme
        assert nomme.reference == "PRC-6284"
        assert nomme.date == date(2008, 4, 14)


class TestArticlesDecoupes:

    def test_l_article_entier_retrouve_son_morceau(self):
        """
        5 961 morceaux « N.M » du corpus n'ont pas de « N » a cote d'eux :
        sans cela, « l'article 55 » serait signale comme introuvable.
        """
        electoral = _extrait(551, 5, "55.1", "Loi N°2012/001 portant Code électoral", "PRC-5")
        texte = "Voir l'article 55 du Code électoral."

        [occurrence] = lire_les_citations(texte, [electoral])

        assert occurrence.extrait.article_id == 551
        # Le numero de l'article, que la page du texte connait ; pas « 55.1 ».
        assert occurrence.article == "55"


CODE_PENAL = "Loi N° 2016/007 du 12 juillet 2016 portant Code Pénal"


class TestMesureSurDeVraiesReponses:
    """
    Cas releves le 2026-10-28 sur 323 reponses d'evaluation
    (data/eval_runs/mistral_*.json, avec les extraits que le modele avait lus) :
    des citations justes etaient signalees introuvables, et des citations
    debordaient sur la phrase. Signaler a tort, c'est dire au lecteur de douter
    d'un article reellement lu.
    """

    def test_le_nom_en_gras_apres_la_liaison(self):
        """« l'article 337 de la **Loi … Code Pénal** » : le texte cite devenait « de la »."""
        extraits = [
            _extrait(3371, 4722, "337", CODE_PENAL, "PRC-4722"),
            _extrait(91, 9313, "50", "Loi N°2022/014 relative à la procréation de la personne", "PRC-9313"),
        ]
        texte = "L'article 337 de la **Loi N° 2016/007 du 12 juillet 2016 portant Code Pénal** punit."

        [occurrence] = lire_les_citations(texte, extraits)

        assert occurrence.extrait.article_id == 3371
        # Jamais de marque d'emphase dans une citation : le front n'a pas a
        # couper un gras en deux.
        assert _cite(texte, occurrence) == "article 337"

    def test_le_nom_en_italique(self, extraits):
        texte = "Voir l'article 37 du _Code du travail_ prévoit une indemnité."

        [occurrence] = lire_les_citations(texte, extraits)

        assert occurrence.extrait.article_id == 371
        assert _cite(texte, occurrence) == "article 37"

    def test_la_phrase_qui_continue_n_est_pas_dans_la_citation(self, extraits):
        texte = "L'article 37 précise que l'indemnité est due."

        [occurrence] = lire_les_citations(texte, extraits)

        assert _cite(texte, occurrence) == "article 37"

    def test_un_verbe_ne_designe_aucun_texte(self):
        """« est » designait la loi dont le titre contient « de l'Est »."""
        extraits = [
            _extrait(51, 51, "5", "Décret portant création de la Région de l'Est", "PRC-51"),
            _extrait(52, 52, "5", CODE_DU_TRAVAIL, "PRC-52"),
        ]
        texte = "L'article 5 est abrogé depuis."

        [occurrence] = lire_les_citations(texte, extraits)

        # Aucun nom : le numero seul, comme avant, et la citation s'arrete au numero.
        assert occurrence.raison is None
        assert _cite(texte, occurrence) == "article 5"

    def test_l_article_range_sous_un_pseudo_numero(self):
        """La loi 2026/003 porte son article 142 dans le morceau « SECTION_1 »."""
        loi = "Loi N°2026/003 du 14 avril 2026 modifiant et complétant certaines dispositions"
        extraits = [_extrait(
            1, 10680, "SECTION_1", loi, "PRC-10680",
            "«ARTiCLE 142 (nouveau).- (1) En cas de vacance de la Présidence de la République…",
        )]
        texte = "Le Vice-Président achève le mandat (l'article 142 de la Loi N°2026/003)."

        [occurrence] = lire_les_citations(texte, extraits)

        assert occurrence.raison is None and occurrence.extrait.law_id == 10680
        # Le numero de l'article, que la page du texte connait ; pas « SECTION_1 ».
        assert occurrence.article == "142"

    def test_l_article_colle_a_la_fin_du_precedent(self):
        """Le morceau « 120.2 » du Code de procédure pénale contient l'article 121."""
        cpp = "Loi N°2005/007 du 27 juillet 2005 portant Code de procédure pénale"
        extraits = [_extrait(
            1202, 2005, "120.2", cpp, "LOI-2005-007",
            "(3) … au procès-verbal d'arrestation. Article 121 — Le délai de la garde à vue court…",
        )]
        texte = "Le délai court dès l'arrestation (l'article 121 de la Loi N°2005/007)."

        [occurrence] = lire_les_citations(texte, extraits)

        assert occurrence.extrait.article_id == 1202 and occurrence.article == "121"

    def test_un_renvoi_n_est_pas_un_en_tete(self):
        """« prévus à l'article 4 ci-dessus » ne fait pas lire l'article 4."""
        extraits = [_extrait(
            1, 1990, "SECTION_1", "Loi N°90/053 du 19 décembre 1990 sur la liberté d'association",
            "LOI-1990-053", "Art. 6: sous réserve des cas de nullité prévus à l'article 4 ci-dessus…",
        )]

        [quatre] = lire_les_citations("Voir l'article 4 de la Loi N°90/053.", extraits)
        [six] = lire_les_citations("Voir l'article 6 de la Loi N°90/053.", extraits)

        assert quatre.raison == "article_non_consulte"
        assert six.raison is None and six.article == "6"

    def test_la_meme_loi_est_celle_de_la_citation_precedente(self, extraits):
        """Le decret, classe premier, porte aussi un article 37 : la même loi n'est pas lui."""
        texte = "Voir l'article 40 du Code du travail et l'article 37 du même code."

        premiere, seconde = lire_les_citations(texte, extraits)

        assert premiere.extrait.article_id == 401
        assert seconde.extrait.article_id == 371
        assert _cite(texte, seconde) == "article 37 du même code"

    def test_la_meme_loi_qu_un_texte_non_consulte(self, extraits):
        """Le Code pénal n'est pas consulte : son article 37 n'est pas celui du decret."""
        texte = "L'article 312 du Code pénal, puis l'article 37 du même code."

        premiere, seconde = lire_les_citations(texte, extraits)

        assert premiere.raison == seconde.raison == "texte_non_consulte"

    def test_la_meme_loi_qu_un_texte_designe_par_son_numero(self, extraits):
        texte = "La loi n° 92/007 du 14 août 1992 s'applique ; l'article 37 de la même loi le dit."

        loi, article = lire_les_citations(texte, extraits)

        assert loi.forme == "texte" and loi.extrait.law_id == 3
        assert article.extrait.article_id == 371


class TestIndicesUtf16:

    def test_un_emoji_decale_les_indices_du_front(self, extraits):
        texte = "📌 Voir l'article 37 du Code du travail."
        [occurrence] = lire_les_citations(texte, extraits)
        vers_js = indices_utf16(texte)

        # Ce que `texte.slice(debut, fin)` rend en JavaScript.
        unites = texte.encode("utf-16-le")
        js = unites[2 * vers_js(occurrence.debut):2 * vers_js(occurrence.fin)].decode("utf-16-le")

        assert js == "article 37 du Code du travail"
        assert vers_js(occurrence.debut) == occurrence.debut + 1

    def test_sans_caractere_hors_plan_de_base_rien_ne_change(self):
        vers_js = indices_utf16("Pas d'emoji ici, é et — compris.")
        assert vers_js(12) == 12


class TestNoms:

    @pytest.mark.parametrize("brut, attendu", [
        ("Loi N° 92/007", "loi n°92/007"),
        ("LOI Nº92/007", "loi n°92/007"),
        ("Law No. 2016/007", "law n°2016/007"),
        ("Code  Pénal", "code penal"),
        ("l’Acte uniforme", "l'acte uniforme"),
    ])
    def test_forme_commune(self, brut, attendu):
        assert normaliser_le_nom(brut) == attendu

    def test_la_nature_du_texte_compte(self):
        assert porte_le_numero("Décret N°2019/143 du 4 avril 2019", "décret", "2019/143")
        assert not porte_le_numero("Loi N°2019/143 du 4 avril 2019", "décret", "2019/143")
        assert porte_le_numero("LAW No. 2016/007 OF 12 JULY 2016", "loi", "2016/007")
        assert not porte_le_numero("Décret N°2019/1430", "décret", "2019/143")

    @pytest.mark.parametrize("mention, nom", [
        ("Code minier dispose que le titulaire", "Code minier"),
        ("Code pénal punit", "Code pénal"),
        ("Acte uniforme relatif au droit commercial général qu'il faut lire", (
            "Acte uniforme relatif au droit commercial général"
        )),
        ("Code de la famille de", "Code de la famille"),
        ("Code pénal ne s'applique pas", "Code pénal"),
        ("Labour Code does not apply", "Labour Code"),
    ])
    def test_etendue_d_un_nom_inconnu(self, mention, nom):
        assert mention[:etendue_du_nom(mention)] == nom
