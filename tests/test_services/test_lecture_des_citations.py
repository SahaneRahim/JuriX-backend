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
    porte_l_entete,
    porte_le_numero,
)

CODE_DU_TRAVAIL = "Loi n° 92/007 du 14 août 1992 portant Code du travail"


def _extrait(article_id, law_id, numero, titre, reference, contenu="Contenu de l'article.",
             jour=date(1992, 8, 14)):
    return ChunkResult(
        article_id=article_id, law_id=law_id, number=numero, content=contenu,
        reference=reference, law_title=titre, publication_date=jour,
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
            # La date de l'acte, que la citation ecrit et qui est comparee.
            _extrait(3371, 4722, "337", CODE_PENAL, "PRC-4722", jour=date(2016, 7, 12)),
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


LIBERTE_D_ASSOCIATION = "Loi N°90/053 du 19 décembre 1990 portant sur la liberté d'association"
LOI_2021_022 = (
    "Loi N°2021/022 du 16 décembre 2021 modifiant certaines dispositions de la loi n° 90/053 du "
    "19 décembre 1990 relative à la liberté d'association"
)
LOI_2020_009 = (
    "Loi n° 2020/009 du 20 juillet 2020 modifiant et complétant certaines dispositions de la loi "
    "n° 90/053 du 19 décembre 1990 relative à la liberté d'association"
)


class TestNumeroPropre:
    """
    Revue du 2026-10-28, sur les 323 vraies reponses d'evaluation : 18 citations
    etaient liees a un texte dont l'intitule ne faisait que CITER le numero
    ecrit (« modifiant … la loi n° 90/053 »), et une citation exacte d'un
    intitule abime par l'OCR (« N°202l/345 ») etait signalee. Une citation par
    numero vise le texte dont c'est le numero PROPRE, et lui seul.
    """

    @pytest.fixture
    def association(self):
        """Les extraits que le modele avait lus, dans l'ordre du contexte (titres reels)."""
        return [
            _extrait(5, 1316, "5", LOI_2021_022, "PRC-9115",
                     "(4) (nouveau) Les partis politiques, les syndicats…",
                     jour=date(2021, 12, 16)),
            _extrait(1, 1230, "1", LOI_2020_009, "PRC-8413",
                     "Les dispositions de l'article 5 alinéa 4 de la loi n° 90/053…",
                     jour=date(2020, 7, 20)),
            _extrait(2, 2246, "2", LIBERTE_D_ASSOCIATION, "LOI-1990-053",
                     "L'association est la convention par laquelle des personnes mettent en commun "
                     "leurs connaissances. Art..5- (1) Les associations obéissent à deux régimes :",
                     jour=date(1990, 12, 19)),
        ]

    def test_la_vraie_reponse_sur_la_liberte_d_association(self, association):
        """
        mistral_liens_apres_20261014_1214.json : la loi 90/053 etait liee a la
        loi 2021/022 (law_id 1316), et la vraie loi 90/053, lue, rangee dans
        « Textes consultés, non cités ».
        """
        texte = (
            "Cela signifie que leur création, leur fonctionnement ou leur dissolution ne suivent "
            "pas les règles de déclaration ou d’autorisation prévues par la loi n°90/053 du 19 "
            "décembre 1990. Ils doivent se conformer à d’autres lois ou décrets spécifiques à leur "
            "domaine.\n\n**Source :** l’article 5 de la Loi n°90/053 du 19 décembre 1990 (modifié "
            "par la Loi n°2021/022 du 16 décembre 2021)."
        )

        loi, article, modificative = lire_les_citations(texte, association)

        assert _cite(texte, loi) == "loi n°90/053 du 19 décembre 1990"
        assert loi.forme == "texte" and loi.extrait.law_id == 2246
        assert _cite(texte, article) == "article 5 de la Loi n°90/053 du 19 décembre 1990"
        # L'article 5 est colle a la fin du morceau « 2 » : « Art..5- » (OCR).
        assert article.extrait.law_id == 2246 and article.article == "5"
        assert _cite(texte, modificative) == "Loi n°2021/022 du 16 décembre 2021"
        assert modificative.extrait.law_id == 1316

    def test_la_vraie_reponse_sur_le_gardiennage(self):
        """
        mistral_3b_14b_20261007_2300.json : « la loi n°97/021 », absente du
        corpus, etait liee au decret 2015/407 « portant application de la loi
        n°97/021 ». Elle est signalee ; l'article 13 du decret reste lie.
        """
        decret = (
            "Décret N°2015/407 du 16 septembre 2015 portant application de la loi n°97/021 du "
            "10 septembre 1997 relative aux activités privées de gardiennage, modifiée et "
            "complétée par la loi n° 2014/027 du 23 décembre 2014"
        )
        extraits = [_extrait(13, 1916, "13", decret, "PRC-4009", jour=date(2015, 9, 16))]
        texte = (
            "Ces sommes sont précisées à l’**article 13 du Décret N°2015/407 du 16 septembre "
            "2015** portant application de la loi n°97/021 du 10 septembre 1997 relative aux "
            "activités privées de gardiennage."
        )

        article, loi = lire_les_citations(texte, extraits)

        assert article.extrait.law_id == 1916 and article.article == "13"
        assert _cite(texte, loi) == "loi n°97/021 du 10 septembre 1997"
        assert loi.extrait is None and loi.raison == "texte_non_consulte"

    def test_un_intitule_abime_par_l_ocr_garde_son_numero(self):
        """Six intitules du corpus portent « 202l/… » : le l est un 1."""
        decret = _extrait(
            1, 634, "1", "Décret N°202l/345 du 17 juin 2021 portant révocation d'un Magistrat",
            "PRC-8955", jour=date(2021, 6, 17),
        )
        texte = (
            "Selon l'article 1 du décret n° 2021/345, le magistrat est révoqué ; "
            "voir le décret n° 2021/345."
        )

        article, texte_cite = lire_les_citations(texte, [decret])

        assert article.raison is None and article.extrait.law_id == 634
        assert texte_cite.raison is None and texte_cite.extrait.law_id == 634
        assert porte_le_numero(decret.law_title, "décret", "2021/345")

    def test_un_article_du_texte_modifie_et_non_du_texte_modificatif(self):
        """
        « l'article 8 de la Loi N°2006/016 » etait lie a la loi 2017/014
        « modifiant … la loi N° 2006/016 », et « l'article 6 de la Loi
        N°2008/001 » a la loi 2026/002 « modifiée … par la loi N°2008/001 ».
        """
        cour_supreme = [
            _extrait(8, 1537, "8", "Loi N° 2017/014 du 12 juillet 2017 modifiant et complétant "
                     "certaines dispositions de la loi N° 2006/016 du 29 décembre 2006", "PRC-5698",
                     jour=date(2017, 7, 12)),
            _extrait(7, 2268, "7", "Loi N°2006/016 du 29 décembre 2006 fixant l'organisation et le "
                     "fonctionnement de la Cour suprême", "LOI-2006-016",
                     "(2) … ARTICLE 8.- La Chambre judiciaire comprend des sections.",
                     jour=date(2006, 12, 29)),
        ]
        constitution = [
            _extrait(6, 1761, "6", "Loi N°2026/002 du 14 avril 2026 modifiant et complétant "
                     "certaines dispositions de la Constitution du 02 juin 1972, modifiée et "
                     "complétée par la loi N°96/06 du 18 janvier 1996 et la loi N°2008/001 du "
                     "14 avril 2008", "PRC-10679", jour=date(2026, 4, 14)),
            _extrait(1, 1541, "DISPOSITIF", "Loi N°2008/001 du 14 avril 2008 modifiant et "
                     "complétant certaines dispositions de la Loi N°96/06 du 18 janvier 1996",
                     "PRC-6284",
                     "… sont modifiées et complétées ainsi qu'il suit : «ARTICLE 6. (2) "
                     "(nouveau) : Le Président de la République est rééligible.",
                     jour=date(2008, 4, 14)),
        ]

        [huit] = lire_les_citations("Voir l'article 8 de la Loi N°2006/016.", cour_supreme)
        [six] = lire_les_citations("Voir l'article 6 de la Loi N°2008/001.", constitution)

        assert huit.extrait.law_id == 2268 and huit.article == "8"
        assert six.extrait.law_id == 1541 and six.article == "6"

    def test_un_intitule_qui_ne_fait_que_citer_le_numero_n_est_jamais_la_cible(self):
        """Le decret 94/199 est « modifié et complété par le décret n°2000/287 »."""
        statut_general = _extrait(
            120, 2279, "120",
            "Décret N°94/199 du 07 octobre 1994 portant Statut Général de la Fonction Publique de "
            "l'État, modifié et complété par le décret n°2000/287 du 12 octobre 2000",
            "DEC-1994-199", jour=date(1994, 10, 7),
        )
        texte = "Le décret n°2000/287 l'a complété, et l'article 120 du décret n°2000/287 aussi."

        texte_cite, article = lire_les_citations(texte, [statut_general])

        assert texte_cite.raison == "texte_non_consulte" and texte_cite.extrait is None
        assert article.raison == "texte_non_consulte" and article.extrait is None

    @pytest.mark.parametrize("ecrit, lie", [
        ("le décret n° 94/199 du 7 octobre 1994", True),
        ("le décret n° 94/199 du 07 oct. 1994", True),
        ("le décret n° 94/199", True),
        # La date du Code du travail, pas celle du decret : un autre texte.
        ("le décret n° 94/199 du 14 août 1992", False),
    ])
    def test_la_date_ecrite_doit_etre_celle_du_texte(self, ecrit, lie):
        statut_general = _extrait(
            120, 2279, "120", "Décret N°94/199 du 07 octobre 1994 portant Statut Général",
            "DEC-1994-199", jour=date(1994, 10, 7),
        )
        texte = f"Voir {ecrit}."

        [occurrence] = lire_les_citations(texte, [statut_general])

        assert (occurrence.extrait is not None) is lie
        assert occurrence.raison == (None if lie else "texte_non_consulte")

    def test_une_annee_seule_sure_ne_contredit_que_l_annee(self):
        """Un 1er janvier de remplissage : seule l'annee du texte est connue."""
        decret = _extrait(9, 9, "9", "Décret N°2015/405 fixant les modalités de rémunération",
                          "PRC-9", jour=date(2015, 1, 1))

        [juste] = lire_les_citations("Voir le décret n° 2015/405 du 16 septembre 2015.", [decret])
        [faux] = lire_les_citations("Voir le décret n° 2015/405 du 16 septembre 2016.", [decret])

        assert juste.extrait is not None and faux.raison == "texte_non_consulte"


class TestEmphase:
    """
    Un passage cite ne contient jamais « * » ni « _ » : le front pose ses liens
    sur le markdown brut et ne peut pas couper une emphase en deux (revue du
    2026-10-28 : « _ » final avale dans un numero, « _ » ouvrant dans un nom).
    """

    @pytest.fixture
    def lus(self):
        return [
            _extrait(351, 3, "35", CODE_DU_TRAVAIL, "PRC-3"),
            _extrait(120, 2279, "120", "Décret N°94/199 du 07 octobre 1994 portant Statut Général",
                     "DEC-1994-199", jour=date(1994, 10, 7)),
        ]

    def test_un_texte_numerote_en_italique(self, lus):
        texte = "Voir _le décret n° 94/199_ pour le détail."

        [occurrence] = lire_les_citations(texte, lus)

        assert _cite(texte, occurrence) == "décret n° 94/199"
        assert occurrence.raison is None and occurrence.extrait.law_id == 2279

    def test_un_italique_ouvert_dans_le_nom(self, lus):
        texte = "Selon l'article 35 de la loi _n° 92/007_, oui."

        [occurrence] = lire_les_citations(texte, lus)

        assert _cite(texte, occurrence) == "article 35 de la loi"
        assert occurrence.extrait.article_id == 351

    @pytest.mark.parametrize("texte", [
        "Voir _le décret n° 94/199_ pour le détail.",
        "Voir **le décret n° 94/199** pour le détail.",
        "Selon l'article 35 de la loi _n° 92/007_, oui.",
        "Selon l'article 35 de la loi *n° 92/007*, oui.",
        "Selon l'article 35 du **Code du travail**, oui.",
        "Selon l'article 35 du _Code du travail_, oui.",
        "Selon *l'article 35 du Code du travail*, oui.",
        "- **Article 35 du Code du travail** : la règle.\n"
        "- *Article 120 du décret n° 94/199* : la suite.",
        "Selon l'article 35 de __la loi n° 92/007__, oui.",
        "Voir le décret n° 94/199 du 07 _octobre_ 1994.",
    ])
    def test_aucun_passage_ne_porte_une_marque_d_emphase(self, lus, texte):
        occurrences = lire_les_citations(texte, lus)

        assert occurrences
        for occurrence in occurrences:
            assert not set(_cite(texte, occurrence)) & {"*", "_"}, _cite(texte, occurrence)
            assert occurrence.debut < occurrence.fin


class TestCitationSansNom:

    def test_un_article_en_morceaux_se_rattache_sans_nom(self):
        """« L'article 35 précise… », les morceaux 35.1 et 35.2 lus : comme sa forme nommee."""
        extraits = [
            _extrait(97420, 2274, "35.1", CODE_DU_TRAVAIL, "LOI-1992-007"),
            _extrait(97421, 2274, "35.2", CODE_DU_TRAVAIL, "LOI-1992-007"),
        ]
        texte = "L'article 35 précise que la règle s'applique."

        [occurrence] = lire_les_citations(texte, extraits)

        assert occurrence.raison is None
        assert occurrence.extrait.article_id == 97420 and occurrence.article == "35"

    def test_un_article_sans_nom_ni_extrait_n_accuse_aucun_texte(self, extraits):
        """Aucun texte n'est nomme : c'est l'article qui manque, pas un texte."""
        [occurrence] = lire_les_citations("L'article 999 précise la règle.", extraits)

        assert occurrence.raison == "article_non_consulte"


class TestEnTeteDArticle:

    @pytest.mark.parametrize("contenu, numero, porte", [
        ("«ARTiCLE 142 (nouveau).- (1) En cas de vacance", "142", True),
        ("Art. 6: sous réserve des cas de nullité", "6", True),
        ("Article premier.- La présente loi", "1", True),
        # Point double de l'OCR (loi 90/053, morceau « 2 »).
        ("leurs connaissances. Art..5- (1) Les associations obéissent", "5", True),
        # Alinea reecrit d'une loi modificative (loi 2008/001).
        ("«ARTICLE 6. (2) (nouveau) : Le Président de la République", "6", True),
        # Un renvoi n'est pas un en-tete, ni un numero sans separateur.
        ("prévus à l'article 4 ci-dessus", "4", False),
        ("Article 5 (1) Les associations", "5", False),
        ("Article 35.1 - Le contrat", "35", False),
        ("(1) Le contrat de travail peut être rompu.", "999", False),
    ])
    def test_porte_l_entete(self, contenu, numero, porte):
        assert porte_l_entete(contenu, numero) is porte


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
