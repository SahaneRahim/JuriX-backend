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
    nature_nommee,
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


def _commence(numero, suite="Le texte de l'article."):
    """
    Le contenu d'un morceau qui porte l'en-tete de son article (« Article 37.- »).
    Une citation que rien ne designe (ni nom, ni citation avant elle) n'est liee
    qu'a cette preuve (_sans_antecedent) : les morceaux de jurix_dev rangent
    l'en-tete dans leur numero, et leur contenu ne la porte pas.
    """
    return f"Article {numero}.- {suite}"


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
            _extrait(51, 51, "6", "Décret portant création de la Région de l'Est", "PRC-51"),
            _extrait(52, 52, "5", CODE_DU_TRAVAIL, "PRC-52", _commence(5)),
        ]
        texte = "L'article 5 est abrogé depuis."

        [occurrence] = lire_les_citations(texte, extraits)

        # Aucun nom : le seul texte lu qui porte l'article 5, son morceau en
        # portant l'en-tete (lue « de l'Est », la citation serait signalee : ce
        # decret n'a pas d'article 5), et la citation s'arrete au numero.
        assert occurrence.raison is None and occurrence.extrait.article_id == 52
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
        "Voir **le décret n° 94/199** pour le détail.",
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

    def test_un_article_en_morceaux_sans_nom_ni_antecedent(self):
        """
        « L'article 35 précise… », seuls les morceaux 35.1 et 35.2 lus : rien ne
        dit que le modele cite ce texte, et aucun morceau ne porte l'en-tete de
        l'article 35 (revue du 2026-10-29, tour final). Signalee ; liee au
        morceau qui porte l'en-tete, s'il y en a un. Nommee, ou apres une
        citation du Code du travail, elle reste liee (TestCheminsRestantsDeLIncident).
        """
        texte = "L'article 35 précise que la règle s'applique."
        suite = _extrait(97421, 2274, "35.2", CODE_DU_TRAVAIL, "LOI-1992-007")

        [signalee] = lire_les_citations(texte, [
            _extrait(97420, 2274, "35.1", CODE_DU_TRAVAIL, "LOI-1992-007"), suite,
        ])
        [liee] = lire_les_citations(texte, [
            _extrait(97420, 2274, "35.1", CODE_DU_TRAVAIL, "LOI-1992-007",
                     _commence(35, "(1) Pendant la durée du préavis…")),
            suite,
        ])

        assert signalee.extrait is None and signalee.raison == "article_non_consulte"
        assert liee.extrait.article_id == 97420 and liee.article == "35"

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
        # Lu sans motif par numero (tour final) : les memes reponses.
        ("ARTICLE 12 BIS.- Les dispositions", "12 BIS", True),
        ("Article 1er.- La présente loi", "1", True),
        ("Art 1ère: La présente loi", "1", True),
        ("Article 94-2 - Le titulaire", "94-2", True),
        ("Article 55.- Le titulaire", "5", False),
        ("Article 5.5- Le titulaire", "5", False),
        ("Partie I. Article 9 : Les associations", "9", True),
        ("Articles 9.- Les associations", "9", False),
        ("’article 8.- renvoi", "8", False),
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


# ==================== UN TEXTE NOMME N'EST QUE CE TEXTE ====================

LF_2023 = (
    "Loi N°2022/020 du 27 décembre 2022 portant loi de finances de la république du Cameroun "
    "pour l'exercice 2023"
)
LF_2026 = (
    "Loi N°2025/012 du 17 décembre 2025 portant loi des Finances de la République du Cameroun "
    "pour l’exercice 2026"
)
ORDONNANCE_2025_001 = (
    "Ordonnance N°2025/001 du 11 juillet 2025 modifiant et complétant certaines dispositions de "
    "la loi n°2024/013 du 23 décembre 2024 portant loi de finances de la République du Cameroun "
    "pour l’exercice 2025"
)
TABLEAU = (
    "Les montants des autorisations d'engagement et des crédits de paiement du budget général "
    "ouverts sur les programmes concourant à la réalisation des objectifs assortis "
    "d'indicateurs sont fixés comme suit :\n| N° | CODE | Programme LIBELLE | OBJECTF | AE | CP |"
)


@pytest.fixture
def tableaux_budgetaires():
    """
    Les extraits du chat de l'incident du 2026-10-29, lignes reelles de
    jurix_dev (identifiants, references, numeros, intitules) : le volet
    semantique de la recherche n'avait rendu que des tableaux de lois de
    finances, et aucun article du Code du travail.
    """
    return [
        _extrait(105698, 2237, "95.95", LF_2026, "PRC-10557", TABLEAU, jour=date(2025, 12, 17)),
        _extrait(85057, 2232, "76.37", LF_2023, "PRC-9449", TABLEAU, jour=date(2022, 12, 27)),
        _extrait(74786, 2137, "78.47", ORDONNANCE_2025_001, "PRC-10434", "(nouveau).- " + TABLEAU,
                 jour=date(2025, 7, 11)),
    ]


# La reponse de l'incident, reconstituee d'apres les captures : le modele dit
# que les documents ne contiennent rien, puis cite trois articles de memoire.
REPONSE_DE_L_INCIDENT = (
    "Les documents fournis ne contiennent pas de disposition sur les indemnités en cas de "
    "licenciement abusif.\n\nEn droit camerounais du travail :\n"
    "- l'indemnité de préavis est prévue par l'article 74 du Code du travail ;\n"
    "- l'indemnité de licenciement, par l'article 75 du Code du travail ;\n"
    "- les dommages-intérêts pour licenciement abusif, par l'article 76 du Code du travail."
)


class TestIncidentDu29Octobre:
    """
    Production, 2026-10-29, « Quelles indemnités en cas de licenciement abusif
    ? » : « l'article 76 du Code du travail » etait devenu un lien vers le
    morceau « 76.37 » de la loi de finances 2023 (PRC-9449), affichee sous
    « Articles cités ». Le Code du travail n'etait pas consulte : un nom de
    texte que rien ne porte rattachait la citation au premier article de meme
    numero, morceau « N.M » compris, de n'importe quel texte.
    """

    def test_aucun_article_du_code_du_travail_n_est_rattache_a_une_loi_de_finances(
        self, tableaux_budgetaires
    ):
        occurrences = lire_les_citations(REPONSE_DE_L_INCIDENT, tableaux_budgetaires)

        assert [_cite(REPONSE_DE_L_INCIDENT, o) for o in occurrences] == [
            "article 74 du Code du travail",
            "article 75 du Code du travail",
            "article 76 du Code du travail",
        ]
        assert [o.raison for o in occurrences] == ["texte_non_consulte"] * 3
        assert all(o.extrait is None for o in occurrences)

    def test_le_morceau_76_37_ne_porte_pas_l_article_76_du_code_du_travail(
        self, tableaux_budgetaires
    ):
        """Le cas exact : le seul morceau « 76.x » lu est celui de la loi de finances."""
        texte = "Voir l'article 76 du Code du travail."

        [occurrence] = lire_les_citations(texte, tableaux_budgetaires)

        assert occurrence.extrait is None and occurrence.raison == "texte_non_consulte"
        assert _cite(texte, occurrence) == "article 76 du Code du travail"

    def test_le_code_du_travail_lu_sans_son_article_76(self, tableaux_budgetaires):
        """
        Le contexte que la recherche doit desormais donner : le Code du travail
        ET les tableaux. Son article 76 n'est pas lu : c'est l'article qui
        manque, jamais le morceau 76.37 d'un autre texte.
        """
        extraits = [
            *tableaux_budgetaires,
            _extrait(97420, 2274, "35.1", CODE_DU_TRAVAIL, "LOI-1992-007"),
            _extrait(97421, 2274, "35.2", CODE_DU_TRAVAIL, "LOI-1992-007"),
        ]
        texte = "Le préavis (l'article 35 du Code du travail) et l'indemnité (l'article 76 du Code du travail)."

        trente_cinq, soixante_seize = lire_les_citations(texte, extraits)

        assert trente_cinq.extrait.article_id == 97420 and trente_cinq.article == "35"
        assert soixante_seize.extrait is None
        assert soixante_seize.raison == "article_non_consulte"

    def test_le_morceau_reste_rattache_a_sa_propre_loi(self, tableaux_budgetaires):
        """« 76.37 » est un morceau de l'article 76 de la loi de finances : nommee, elle le garde."""
        texte = "Les crédits sont fixés par l'article 76 de la loi de finances."

        [occurrence] = lire_les_citations(texte, tableaux_budgetaires)

        assert occurrence.extrait.article_id == 85057 and occurrence.article == "76"

    @pytest.mark.parametrize("texte", [
        "L'article 76 fixe les montants des crédits.",
        "L'article 76 prévoit des dommages-intérêts.",
        "En cas de licenciement abusif, l'article 76 de cette loi prévoit des dommages-intérêts.",
        "La loi camerounaise protège le salarié : l'article 76 de cette loi prévoit des "
        "dommages-intérêts.",
        "l'article 76 de la présente loi prévoit des dommages-intérêts.",
        "Under Cameroonian labour law, Article 76 of this Law provides for damages.",
        "l'article 76 du même texte prévoit des dommages-intérêts.",
        "l'article 76 du texte applicable prévoit des dommages-intérêts.",
        "Article 76 provides for damages.",
    ])
    def test_sans_nom_ni_antecedent_jamais_le_tableau(self, tableaux_budgetaires, texte):
        """
        Le lien de l'incident, sans le nom : rien ne designe le texte, et le seul
        texte lu qui porte un article 76 est la loi de finances 2023 (morceau
        76.37). La production et le tour precedent l'y liaient ; rejoue sur les
        lignes reelles de jurix_dev, c'etait l'affichage de l'incident (revue du
        2026-10-29, tour final). Le morceau n'est ni le debut de l'article 76
        (aucun en-tete), ni du droit (un tableau de credits) : signalee.
        """
        [occurrence] = lire_les_citations(texte, tableaux_budgetaires)

        assert occurrence.extrait is None and occurrence.raison == "article_non_consulte"

    def test_un_tableau_meme_avec_son_en_tete(self):
        """
        Le morceau porte l'en-tete « Article 76.- », mais c'est un tableau de
        credits AE/CP (est_un_tableau_de_programmes, le detecteur de la
        recherche) : jamais l'article d'une citation sans nom. Nommee, la loi
        de finances garde son lien.
        """
        tableau = _extrait(85057, 2232, "76", LF_2023, "PRC-9449", "Article 76.- " + TABLEAU,
                           jour=date(2022, 12, 27))

        [sans_nom] = lire_les_citations("L'article 76 prévoit des dommages-intérêts.", [tableau])
        [nommee] = lire_les_citations("Voir l'article 76 de la loi de finances 2023.", [tableau])

        assert sans_nom.extrait is None and sans_nom.raison == "article_non_consulte"
        assert nommee.extrait.article_id == 85057

    def test_apres_la_loi_de_finances_nommee_jamais_le_tableau(self, tableaux_budgetaires):
        """
        La citation precedente nomme la loi de finances, et la phrase dit qu'elle
        ne traite que du budget : « L'article 76 » qui suit allait au morceau
        76.37 (sondes de la revue du tour final). Une citation sans nom n'est
        jamais liee a un tableau de credits, antecedent ou non.
        """
        texte = (
            "L'article 76 de la loi de finances 2023 ne concerne que le budget. "
            "L'article 76 prévoit des dommages-intérêts en cas de licenciement abusif."
        )

        nommee, sans_nom = lire_les_citations(texte, tableaux_budgetaires)

        assert nommee.extrait.article_id == 85057
        assert sans_nom.extrait is None and sans_nom.raison == "article_non_consulte"

    @pytest.mark.parametrize("texte", [
        "La loi n° 2022/020 ne traite pas du licenciement. L'article 76 prévoit des "
        "dommages-intérêts.",
        "Les documents fournis (Loi N°2022/020 du 27 décembre 2022 portant loi de finances) ne "
        "traitent pas du licenciement. En droit du travail, l'article 76 prévoit des "
        "dommages-intérêts.",
        "Les documents fournis, notamment la Loi N°2022/020 portant loi de finances pour "
        "l'exercice 2023, ne contiennent aucune disposition sur le licenciement. L'article 76 "
        "prévoit toutefois des dommages-intérêts.",
    ])
    def test_un_numero_ecrit_dans_une_phrase_precedente_ne_choisit_pas(
        self, tableaux_budgetaires, texte
    ):
        """
        Revue du tour final : le numero de la loi de finances, ecrit en passant
        (pour dire qu'elle ne traite pas du sujet), devenait l'antecedent de la
        citation sans nom, liee au morceau 76.37.
        """
        occurrences = lire_les_citations(texte, tableaux_budgetaires)

        [article] = [o for o in occurrences if o.forme == "article"]
        assert article.extrait is None and article.raison == "article_non_consulte"
        # Le texte numerote, lui, est lu : c'est une citation a part.
        assert [o.extrait.article_id for o in occurrences if o.forme == "texte"] == [85057]

    def test_ce_code_sans_antecedent_n_est_pas_une_loi_de_finances(self, tableaux_budgetaires):
        """« de ce code », sans citation avant : un des codes lus, et aucun ne l'est."""
        texte = "Le Code du travail le prévoit : voir l'article 76 de ce code."

        [occurrence] = lire_les_citations(texte, tableaux_budgetaires)

        assert occurrence.extrait is None and occurrence.raison == "texte_non_consulte"

    @pytest.mark.parametrize("texte, cite", [
        ("Voir l'article 76 du Code du travail.", "article 76 du Code du travail"),
        ("Voir l'article 76 du Code pénal.", "article 76 du Code pénal"),
        ("Voir l'article 76 du Code civil.", "article 76 du Code civil"),
        ("Voir l'article 76 de l'Acte uniforme relatif au droit commercial général.",
         "article 76 de l'Acte uniforme relatif au droit commercial général"),
        ("Voir l'article 76 de la loi n° 92/007.", "article 76 de la loi n° 92/007"),
        ("Voir l'article 76 de la loi 92/007.", "article 76 de la loi 92/007"),
        ("Voir l'article 76 du décret 2021/745.", "article 76 du décret 2021/745"),
        ("Voir l'article 76 de la Constitution.", "article 76 de la Constitution"),
        ("See Article 76 of the Labour Code.", "Article 76 of the Labour Code"),
    ])
    def test_un_texte_nomme_non_consulte_n_emprunte_jamais_un_autre_article(
        self, tableaux_budgetaires, texte, cite
    ):
        [occurrence] = lire_les_citations(texte, tableaux_budgetaires)

        assert occurrence.extrait is None and occurrence.raison == "texte_non_consulte"
        assert _cite(texte, occurrence) == cite


class TestNomsQueLeNumeroSeulRattachait:
    """
    Les formes de nom que la lecture ne reconnaissait pas, et que le numero
    seul rattachait : sur les 323 vraies reponses d'evaluation, 20 citations
    passaient par ce repli. 18 y trouvaient leur texte par chance, et doivent
    le garder maintenant que le repli est ferme ; « l'article 31 du Traité de
    1993 » y prenait l'article 31 du traite de 2008 ; « l'article 31 du Code
    des Finances Publiques » nomme un texte qu'aucun intitule lu ne porte, et
    est desormais signale.
    """

    @pytest.fixture
    def decrets(self):
        """Un autre texte porte aussi les articles 43 et 5, et vient en premier."""
        return [
            _extrait(9043, 9, "43", CODE_DU_TRAVAIL, "LOI-1992-007"),
            _extrait(9005, 9, "5", CODE_DU_TRAVAIL, "LOI-1992-007"),
            _extrait(7043, 7278, "43", "Décret N°2019/143 du 19 mars 2019 portant réorganisation",
                     "PRC-7278", jour=date(2019, 3, 19)),
            _extrait(4505, 4505, "5", "Décret N°2021/745 du 28 décembre 2021 fixant les modalités",
                     "PRC-4505", jour=date(2021, 12, 28)),
        ]

    @pytest.mark.parametrize("texte, cite", [
        ("Voir l'article 43 et l'article 45 du Décret N°2019/143.",
         "article 43 et l'article 45 du Décret N°2019/143"),
        ("Voir l’article 43 et l’article 45 du Décret N°2019/143.",
         "article 43 et l’article 45 du Décret N°2019/143"),
        ("Voir l'article 43 et 45 du Décret N°2019/143.", "article 43 et 45 du Décret N°2019/143"),
        ("Voir l'article 43 point 8 du Décret N°2019/143.",
         "article 43 point 8 du Décret N°2019/143"),
        ("Voir l'article 43 et suivants du Décret N°2019/143.",
         "article 43 et suivants du Décret N°2019/143"),
        # Revue du 2026-10-29 : le passage s'arretait a « article 43 ».
        ("Voir l'article 43 ainsi que l'article 45 du Décret N°2019/143.",
         "article 43 ainsi que l'article 45 du Décret N°2019/143"),
        ("Voir l'article 43 combiné avec l'article 45 du Décret N°2019/143.",
         "article 43 combiné avec l'article 45 du Décret N°2019/143"),
        ("See Article 43 as well as Article 45 of Decree No. 2019/143.",
         "Article 43 as well as Article 45 of Decree No. 2019/143"),
    ])
    def test_un_nom_partage_par_une_enumeration(self, decrets, texte, cite):
        [occurrence] = lire_les_citations(texte, decrets)

        assert occurrence.extrait.article_id == 7043
        assert _cite(texte, occurrence) == cite

    def test_une_enumeration_vers_un_texte_non_consulte(self, decrets):
        texte = "Voir l'article 43 et l'article 44 du Code pénal."

        [occurrence] = lire_les_citations(texte, decrets)

        assert occurrence.extrait is None and occurrence.raison == "texte_non_consulte"

    @pytest.mark.parametrize("texte", [
        "Voir l'article 5 du décret 2021/745.",
        "Voir l'article 5 du Décret 2021/745 du 28 décembre 2021.",
        "Voir l'article 5 du décret n° 2021/745.",
    ])
    def test_un_numero_sans_n_degre(self, decrets, texte):
        [occurrence] = lire_les_citations(texte, decrets)

        assert occurrence.extrait.article_id == 4505

    def test_un_numero_sans_n_degre_a_une_autre_date(self, decrets):
        texte = "Voir l'article 5 du Décret 2021/745 du 3 mars 2020."

        [occurrence] = lire_les_citations(texte, decrets)

        assert occurrence.extrait is None and occurrence.raison == "texte_non_consulte"

    @pytest.mark.parametrize("texte", [
        "Le Décret N°2021/745 le fixe (l'article 5 de ce décret).",
        "Le Décret N°2021/745 le fixe (l'article 5 de ce même décret).",
        "Le Décret N°2021/745 le fixe (l'article 5 du présent décret).",
        "Le Décret N°2021/745 le fixe (l'article 5 du décret précité).",
    ])
    def test_le_demonstratif_renvoie_au_texte_nomme_avant(self, decrets, texte):
        """« de cette loi » apres la Loi N°2014/028 : vraie reponse d'evaluation."""
        texte_cite, article = lire_les_citations(texte, decrets)

        assert texte_cite.extrait.law_id == 4505
        assert article.extrait.article_id == 4505

    def test_le_demonstratif_sans_antecedent_vise_un_texte_de_sa_nature(self, decrets):
        """
        Vraie reponse : « l'article 57 de ce décret », sans texte nomme avant.
        Rien ne designe le decret : lie au seul decret lu qui porte l'article,
        son morceau en portant l'en-tete (jamais l'article 5 du Code du
        travail, lu en premier) ; sans en-tete, signale. La vraie reponse l'est
        desormais (revue du 2026-10-29, tour final : une fausse alerte, voulue).
        """
        texte = "Comme le précise l'article 5 de ce décret."
        avec_en_tete = [*decrets[:3], _extrait(
            4505, 4505, "5", "Décret N°2021/745 du 28 décembre 2021 fixant les modalités",
            "PRC-4505", _commence(5), jour=date(2021, 12, 28),
        )]

        [signalee] = lire_les_citations(texte, decrets)
        [liee] = lire_les_citations(texte, avec_en_tete)

        assert signalee.extrait is None and signalee.raison == "article_non_consulte"
        assert liee.extrait.article_id == 4505

    def test_la_reference_recopiee_de_l_en_tete(self):
        """Vraie reponse : « l'article 1.15 de la Loi PRC-8842 » (un arrete)."""
        extraits = [
            _extrait(1, 1, "1.15", CODE_DU_TRAVAIL, "LOI-1992-007"),
            _extrait(2, 2, "1.15", "Arrêté N°0218/CAB/PR du 18 mars 2021 portant nomination",
                     "PRC-8842", jour=date(2021, 3, 18)),
        ]
        texte = "Source : l'article 1.15 de la Loi PRC-8842."

        [occurrence] = lire_les_citations(texte, extraits)

        assert occurrence.extrait.article_id == 2
        assert _cite(texte, occurrence) == "article 1.15 de la Loi PRC-8842"

    def test_la_reference_recopiee_avec_des_espaces(self):
        """
        Vrai chat du 08/11/2026 : « l'article 309 de la Loi OHADA AUSCGIE
        2014 », pour la reference OHADA-AUSCGIE-2014. Les tirets devenus des
        espaces ne doivent pas faire signaler une citation juste ; une autre
        annee, si.
        """
        titre = ("Acte uniforme révisé relatif au droit des sociétés commerciales "
                 "et du groupement d'intérêt économique")
        extraits = [_extrait(1, 1, "309", titre, "OHADA-AUSCGIE-2014", jour=date(2014, 1, 30))]
        texte = "L'article 309 de la Loi OHADA AUSCGIE 2014 précise."

        [occurrence] = lire_les_citations(texte, extraits)
        [autre_annee] = lire_les_citations(texte.replace("2014", "2015"), extraits)

        assert occurrence.extrait.article_id == 1
        assert _cite(texte, occurrence) == "article 309 de la Loi OHADA AUSCGIE 2014"
        assert autre_annee.extrait is None

    def test_un_texte_nomme_par_sa_nature_et_son_annee(self):
        """
        Vraie reponse sur la CCJA : « l'article 31 du Traité de 1993 », les deux
        traites lus. Le numero seul donnait l'article 31 du traite revise de 2008.
        """
        extraits = [
            _extrait(2008, 2251, "31", "Traité portant révision du Traité relatif à "
                     "l'harmonisation du droit des affaires en Afrique du 17 octobre 2008",
                     "OHADA-TRAITE-2008", jour=date(2008, 10, 17)),
            _extrait(1993, 2258, "31", "Traité relatif à l'harmonisation du droit des affaires en "
                     "Afrique du 17 octobre 1993", "OHADA-TRAITE-1993", jour=date(1993, 10, 17)),
        ]
        texte = "La Cour comptait sept juges (l'article 31 du Traité de 1993)."

        [occurrence] = lire_les_citations(texte, extraits)

        assert occurrence.extrait.article_id == 1993

    @pytest.fixture
    def actes_uniformes(self):
        """Intitules reels : trois Actes uniformes du corpus portent « révisé »."""
        return [
            _extrait(1, 1, "9", CODE_DU_TRAVAIL, "LOI-1992-007"),
            _extrait(2, 2, "9", "Acte uniforme portant organisation des procédures collectives "
                     "d'apurement du passif du 10 septembre 2015", "OHADA-AUPCAP-2015",
                     jour=date(2015, 9, 10)),
            _extrait(3, 3, "9", "Acte uniforme révisé portant organisation des sûretés du 15 "
                     "décembre 2010", "OHADA-AUS-2010", jour=date(2010, 12, 15)),
            _extrait(4, 4, "9", "Acte uniforme révisé portant sur le droit commercial général du "
                     "15 décembre 2010", "OHADA-AUDCG-2010", jour=date(2010, 12, 15)),
        ]

    @pytest.mark.parametrize("texte, article_id", [
        ("Voir l'article 9 de l'Acte uniforme portant organisation des sûretés.", 3),
        ("Voir l'article 9 de l'Acte uniforme relatif au droit commercial général.", 4),
        ("Voir l'article 9 de l'Acte uniforme OHADA portant organisation des sûretés de 2010.", 3),
    ])
    def test_un_acte_uniforme_sous_son_nom_d_usage(self, actes_uniformes, texte, article_id):
        """Le nom d'usage omet « révisé » : chacun de ses mots est dans l'intitule lu."""
        [occurrence] = lire_les_citations(texte, actes_uniformes)

        assert occurrence.extrait.article_id == article_id

    def test_un_acte_uniforme_non_lu_n_emprunte_pas_un_autre_acte(self, actes_uniformes):
        """
        Les procédures collectives sont lues, pas les sûretés. Le plus long
        debut reconnu, « Acte uniforme portant organisation », est dans
        l'intitule des procédures collectives : il y rattachait la citation.
        """
        lus = actes_uniformes[:2]
        texte = "Voir l'article 9 de l'Acte uniforme portant organisation des sûretés."

        [occurrence] = lire_les_citations(texte, lus)

        assert occurrence.extrait is None and occurrence.raison == "texte_non_consulte"

    @pytest.mark.parametrize("lus, raison", [
        ("traite_avec_en_tete", None),
        ("traite", "article_non_consulte"),
        ("code_du_travail_seul", "texte_non_consulte"),
    ])
    def test_un_nom_de_famille_vise_les_textes_de_sa_nature(self, lus, raison):
        """
        « le Traité OHADA » : aucun mot ne distingue un traite d'un autre, et
        l'intitule ne porte pas « OHADA ». Rien ne designe le traite : le seul
        traite lu qui porte l'article, son morceau en portant l'en-tete
        (_sans_antecedent) ; jamais l'article 31 du Code du travail, lu en
        premier. Aucun traite lu : c'est le texte qui manque.
        """
        extraits = [_extrait(31, 2274, "31", CODE_DU_TRAVAIL, "LOI-1992-007")]
        if lus != "code_du_travail_seul":
            extraits.append(_extrait(
                1993, 2258, "31", "Traité relatif à l'harmonisation du droit des affaires en "
                "Afrique du 17 octobre 1993", "OHADA-TRAITE-1993",
                _commence(31) if lus == "traite_avec_en_tete" else "Contenu de l'article.",
                jour=date(1993, 10, 17),
            ))
        texte = "La Cour compte sept juges (l'article 31 du Traité OHADA)."

        [occurrence] = lire_les_citations(texte, extraits)

        assert (occurrence.extrait.article_id if occurrence.extrait else None) == (
            1993 if raison is None else None
        )
        assert occurrence.raison == raison

    @pytest.mark.parametrize("texte, lie", [
        # Meme debut, autre code : « procédure civile » n'est pas « procédure pénale ».
        ("Voir l'article 9 du Code de procédure civile.", False),
        # La phrase continue apres le nom : le nom reste reconnu.
        ("Voir l'article 9 du Code de procédure pénale sur la garde à vue.", True),
        ("Voir l'article 9 du Code de procédure pénale camerounais.", True),
    ])
    def test_un_debut_commun_ne_designe_pas_un_autre_code(self, texte, lie):
        cpp = "Loi N°2005/007 du 27 juillet 2005 portant Code de procédure pénale"
        [occurrence] = lire_les_citations(texte, [_extrait(9, 2005, "9", cpp, "LOI-2005-007")])

        assert (occurrence.extrait is not None) is lie
        assert occurrence.raison == (None if lie else "texte_non_consulte")

    def test_un_texte_nomme_par_sa_date(self):
        extraits = [
            _extrait(1, 1, "35", LF_2023, "PRC-9449", jour=date(2022, 12, 27)),
            _extrait(2, 2, "35", CODE_DU_TRAVAIL, "LOI-1992-007", jour=date(1992, 8, 14)),
        ]

        [juste] = lire_les_citations("Voir l'article 35 de la loi du 14 août 1992.", extraits)
        [faux] = lire_les_citations("Voir l'article 35 de la loi du 3 mars 1992.", extraits)

        assert juste.extrait.article_id == 2
        assert faux.extrait is None and faux.raison == "texte_non_consulte"

    @pytest.mark.parametrize("texte", [
        "Article 35 of the Labour Code requires notice.",
        "Under Article 35 of the Labor Code, notice is due.",
    ])
    def test_un_code_nomme_en_anglais(self, texte):
        """Le prompt anglais demande « Article X of the Y Code » ; l'intitule est francais."""
        extraits = [
            _extrait(1, 1, "35", LF_2023, "PRC-9449"),
            _extrait(97420, 2274, "35.1", CODE_DU_TRAVAIL, "LOI-1992-007"),
        ]

        [occurrence] = lire_les_citations(texte, extraits)

        assert occurrence.extrait.article_id == 97420

    @pytest.mark.parametrize("texte", [
        "Voir l'article 5 et suivants pour le détail.",
        "Voir l'article 5 point 8 pour le détail.",
        "Article 5 Les associations obéissent à deux régimes.",
    ])
    def test_une_suite_qui_ne_nomme_aucun_texte_garde_le_numero(self, decrets, texte):
        """
        Ni nature ni nom : une citation sans nom (_sans_antecedent). L'article 5
        du seul texte lu qui le porte, son morceau en portant l'en-tete (le Code
        du travail, lu seul). Le Code du travail et le Décret N°2021/745 portent
        chacun un article 5, et rien ne dit lequel : signalee, et non plus
        rattachee au premier venu (tour 3). Sans en-tete : signalee (tour final).
        """
        code_5 = _extrait(9005, 9, "5", CODE_DU_TRAVAIL, "LOI-1992-007", _commence(5))

        [seul] = lire_les_citations(texte, [decrets[0], code_5])
        [deux] = lire_les_citations(texte, [decrets[0], code_5, *decrets[2:]])
        [sans_en_tete] = lire_les_citations(texte, decrets[:2])

        assert seul.extrait.article_id == 9005
        assert deux.extrait is None and deux.raison == "article_non_consulte"
        assert sans_en_tete.extrait is None and sans_en_tete.raison == "article_non_consulte"


@pytest.mark.parametrize("nom, nature", [
    ("Code du travail", "code"),
    ("Labour Code", "code"),
    ("OHADA Uniform Act on Arbitration", "act"),
    ("Acte uniforme relatif au droit commercial général", "acte"),
    ("Constitution", "constitution"),
    ("Traité de 1993", "traite"),
    ("Statut général de la fonction publique", "statut"),
    ("Loi PRC-8842", "loi"),
    ("point 8", None),
    ("et suivants", None),
    ("en vigueur", None),
    ("contrat de travail", None),
    ("titre II", None),
    ("Les associations obéissent à deux régimes", None),
    ("in order to", None),
])
def test_nature_nommee(nom, nature):
    assert nature_nommee(nom) == nature


# ==================== REVUE DU 2026-10-29 : CE QUE LE PREMIER CORRECTIF MANQUAIT ====================

# Intitules reels de jurix_dev (export du 2026-10-29), tels que le modele les lit.
STATUT_GENERAL = (
    "Décret N°94/199 du 07 octobre 1994 portant Statut Général de la Fonction Publique de "
    "l'État, modifié et complété par le décret n°2000/287 du 12 octobre 2000"
)
ETABLISSEMENTS_PUBLICS = (
    "Loi N° 2017/010 du 12 juillet 2017 portant statut général des établissements publics."
)
AU_COOPERATIVES = "Acte uniforme relatif au droit des sociétés coopératives du 15 décembre 2010"
AU_ARBITRAGE = "Acte uniforme relatif au droit de l'arbitrage du 23 novembre 2017"
AU_SURETES = "Acte uniforme révisé portant organisation des sûretés du 15 décembre 2010"
AU_DROIT_COMMERCIAL = (
    "Acte uniforme révisé portant sur le droit commercial général du 15 décembre 2010"
)
AU_SOCIETES = (
    "Acte uniforme révisé relatif au droit des sociétés commerciales et du groupement "
    "d'intérêt économique du 30 janvier 2014"
)
CODE_ELECTORAL = (
    "Loi N°2012/001 du 19 avril 2012 portant Code électoral, modifiée et complétée par la loi "
    "n°2012/017 du 21 décembre 2012"
)
CODE_ELECTORAL_MODIFIE = (
    "Loi N°2026/003 du 14 avril 2026 modifiant et complétant certaines dispositions de la loi "
    "N°2012/001 du 19 avril 2012 portant code électoral"
)
CODE_CIVIL = "Code civil applicable au Cameroun, annoté de la jurisprudence de la Cour suprême"
RETRAITE_DES_AGENTS = (
    "Décret N°2020/7951/PM du 30 décembre 2020 portant harmonisation de l'âge de départ à la "
    "retraite des agents de l'Etat relevant du Code du Travail."
)
LF_2024 = (
    "Loi N°2023/019 du 19 décembre 2023 portant loi de finances de la République du Cameroun "
    "pour l'exercice 2024"
)
ASSOCIATIONS = "Loi N°90/053 du 19 décembre 1990 portant sur la liberté d'association"
CONSOMMATEUR = (
    "Loi-cadre N°2011/012 du 06 mai 2011 portant protection du consommateur au Cameroun"
)
TRAITE_1993 = (
    "Traité relatif à l'harmonisation du droit des affaires en Afrique du 17 octobre 1993"
)
DECRET_2021_745 = "Décret N°2021/745 du 28 décembre 2021 fixant les modalités"
REMUNERATION_DES_MAIRES = (
    "Décret N°2015/405 fixant les modalités de rémunération des Délégués du Gouvernement, des "
    "Maires et de leurs Adjoints"
)
INDEMNITES_DES_MAIRES = (
    "Décret N°2015/406 du 16 septembre 2015 fixant les indemnités et autres avantages alloués "
    "aux Délégués du Gouvernement, aux Maires"
)


def _lie(occurrence):
    """La reference du texte lie, ou la raison du signalement."""
    return occurrence.extrait.reference if occurrence.extrait else occurrence.raison


class TestDivergenceSurLesIntitulesReels:
    """
    Revue du 2026-10-29. La premiere version de _diverge comptait tout ecart
    entre la suite de la phrase et la suite de l'intitule : « tel que modifié »
    contre « de l'État », « détermine » contre « du 15 décembre 2010 ». Le bon
    texte etait ecarte, puis un debut plus court donnait un AUTRE texte lu —
    la famille de l'incident — ou la citation etait signalee a tort.
    """

    @pytest.mark.parametrize("texte, lus, attendu", [
        ("L'article 120 du Statut général de la fonction publique tel que modifié s'applique.",
         [(ETABLISSEMENTS_PUBLICS, "PRC-5693"), (STATUT_GENERAL, "DEC-1994-199")], "DEC-1994-199"),
        ("L'article 120 du Statut général de la fonction publique détermine les sanctions.",
         [(ETABLISSEMENTS_PUBLICS, "PRC-5693"), (STATUT_GENERAL, "DEC-1994-199")], "DEC-1994-199"),
        ("L'article 120 du Statut général de la fonction publique relatif aux sanctions "
         "disciplinaires s'applique.",
         [(ETABLISSEMENTS_PUBLICS, "PRC-5693"), (STATUT_GENERAL, "DEC-1994-199")], "DEC-1994-199"),
        # Un participe continue la phrase, meme quand l'intitule continue le nom.
        ("L'article 120 du Statut général de la fonction publique concernant les sanctions "
         "s'applique.",
         [(ETABLISSEMENTS_PUBLICS, "PRC-5693"), (STATUT_GENERAL, "DEC-1994-199")], "DEC-1994-199"),
        ("L'article 120 de l'Acte uniforme relatif au droit des sociétés coopératives détermine "
         "les conditions.",
         [(AU_ARBITRAGE, "OHADA-AUA-2017"), (AU_COOPERATIVES, "OHADA-AUSCOOP-2010")],
         "OHADA-AUSCOOP-2010"),
        ("L'article 120 de l'Acte uniforme révisé portant organisation des sûretés énumère les "
         "sûretés.",
         [(AU_DROIT_COMMERCIAL, "OHADA-AUDCG-2010"), (AU_SURETES, "OHADA-AUS-2010")],
         "OHADA-AUS-2010"),
        ("Voir l'article 120 du Code électoral tel que modifié en 2012.",
         [(CODE_ELECTORAL, "LOI-2012-001")], "LOI-2012-001"),
        ("L'article 120 du Code civil concernant la force obligatoire s'impose.",
         [(CODE_CIVIL, "CODE-CIVIL")], "CODE-CIVIL"),
        ("L'article 120 du Code civil pose le principe de la force obligatoire.",
         [(CODE_CIVIL, "CODE-CIVIL")], "CODE-CIVIL"),
        ("Voir l'article 120 de la loi de finances fixant les autorisations d'engagement.",
         [(ETABLISSEMENTS_PUBLICS, "PRC-5693"), (LF_2023, "PRC-9449")], "PRC-9449"),
    ])
    def test_la_suite_de_la_phrase_n_ecarte_pas_le_bon_texte(self, texte, lus, attendu):
        extraits = [_extrait(i, i, "120", titre, reference) for i, (titre, reference) in enumerate(lus)]

        [occurrence] = lire_les_citations(texte, extraits)

        assert _lie(occurrence) == attendu

    def test_le_passage_ne_porte_pas_la_prose(self):
        """« article 55 du Code électoral tel » : la prose etait soulignee avec le nom."""
        texte = "Voir l'article 55 du Code électoral tel que modifié en 2012."

        [occurrence] = lire_les_citations(texte, [_extrait(1, 1, "55", CODE_ELECTORAL, "LOI-2012-001")])

        assert _cite(texte, occurrence) == "article 55 du Code électoral"

    @pytest.mark.parametrize("texte, lus, attendu", [
        # Le plus long debut present (les coopératives) est ecarte : jamais le debut
        # plus court « Acte uniforme relatif au droit », que l'arbitrage porte aussi.
        ("Voir l'article 5 de l'Acte uniforme relatif au droit des sociétés commerciales.",
         [(AU_COOPERATIVES, "OHADA-AUSCOOP-2010"), (AU_ARBITRAGE, "OHADA-AUA-2017")],
         "texte_non_consulte"),
        # « droit | commercial général » contre « droit | de l'arbitrage » : deux noms.
        ("Voir l'article 5 de l'Acte uniforme relatif au droit commercial général.",
         [(AU_ARBITRAGE, "OHADA-AUA-2017")], "texte_non_consulte"),
        ("Voir l'article 5 de l'Acte uniforme relatif au droit commercial général.",
         [(AU_ARBITRAGE, "OHADA-AUA-2017"), (AU_DROIT_COMMERCIAL, "OHADA-AUDCG-2010")],
         "OHADA-AUDCG-2010"),
        # Un sigle du nom est porte par les initiales de l'intitule.
        ("Voir l'article 5 de l'Acte uniforme relatif au droit des sociétés commerciales et du GIE.",
         [(AU_ARBITRAGE, "OHADA-AUA-2017"), (AU_SOCIETES, "OHADA-AUSCGIE-2014")],
         "OHADA-AUSCGIE-2014"),
    ])
    def test_un_debut_ecarte_ne_cede_pas_la_place_a_un_autre_texte(self, texte, lus, attendu):
        extraits = [_extrait(i, i, "5", titre, reference) for i, (titre, reference) in enumerate(lus)]

        [occurrence] = lire_les_citations(texte, extraits)

        assert _lie(occurrence) == attendu

    def test_un_sigle_n_ecarte_pas_le_nom_qu_il_abrege(self):
        """Vraie reponse : « l’article 57 de la réorganisation de l’APZE »."""
        extraits = [
            _extrait(1, 7303, "57", "Décret N°2019/195 du 17 avril 2019 fixant les modalités de "
                     "création et de gestion des zones économiques au Cameroun", "PRC-7303"),
            _extrait(2, 7449, "57", "Décret N°2019/246 du 24 mai 2019 portant réorganisation de "
                     "l'Agence de Promotion des Zones Economiques", "PRC-7449"),
        ]
        texte = "Le Conseil l'autorise (l’article 57 de la réorganisation de l’APZE)."

        [occurrence] = lire_les_citations(texte, extraits)

        assert occurrence.extrait.article_id == 2
        assert _cite(texte, occurrence) == "article 57 de la réorganisation de l’APZE"


class TestReponsesAnglaises:
    """
    Revue du 2026-10-29 : le corpus n'a aucun intitule anglais, et le prompt
    anglais demande « Article X of the Y Law », « (Article 15 of Decree No.
    2019/143) ». La premiere version signalait des citations justes, texte et
    article lus, que la production rattachait.
    """

    @pytest.fixture
    def lus(self):
        """Chaque texte porte l'article cite ; une loi de finances le porte aussi, en premier."""
        return [
            _extrait(1, 9449, "35", LF_2023, "PRC-9449", jour=date(2022, 12, 27)),
            _extrait(2, 2274, "35.1", CODE_DU_TRAVAIL, "LOI-1992-007"),
            _extrait(3, 7278, "35", "Décret N°2019/143 du 19 mars 2019 portant réorganisation de "
                     "l'Agence des Normes et de la Qualité", "PRC-7278", jour=date(2019, 3, 19)),
            _extrait(4, 2259, "35", AU_ARBITRAGE, "OHADA-AUA-2017", jour=date(2017, 11, 23)),
            _extrait(5, 2258, "35", TRAITE_1993, "OHADA-TRAITE-1993", jour=date(1993, 10, 17)),
            _extrait(6, 2279, "35", STATUT_GENERAL, "DEC-1994-199", jour=date(1994, 10, 7)),
            _extrait(7, 2261, "35", CONSOMMATEUR, "LOI-2011-012", jour=date(2011, 5, 6)),
            _extrait(8, 2246, "35", ASSOCIATIONS, "LOI-1990-053", jour=date(1990, 12, 19)),
            _extrait(9, 2284, "35", AU_SOCIETES, "OHADA-AUSCGIE-2014", jour=date(2014, 1, 30)),
        ]

    @pytest.mark.parametrize("texte, attendu, cite", [
        ("Article 35 of Law No. 92/007 of 14 August 1992 provides a notice period.", "LOI-1992-007",
         "Article 35 of Law No. 92/007 of 14 August 1992"),
        ("See Article 35 of Decree No. 2019/143.", "PRC-7278", "Article 35 of Decree No. 2019/143"),
        ("Voir l'article 35 de la loi No. 92/007.", "LOI-1992-007", "article 35 de la loi No. 92/007"),
        ("Article 35 of the Cameroon Labour Code provides that.", "LOI-1992-007",
         "Article 35 of the Cameroon Labour Code"),
        ("Article 35 of the Uniform Act on Arbitration provides that.", "OHADA-AUA-2017",
         "Article 35 of the Uniform Act on Arbitration"),
        ("Article 35 of the General Civil Service Statute provides that.", "DEC-1994-199",
         "Article 35 of the General Civil Service Statute"),
        ("Article 35 of the Consumer Protection Law provides that.", "LOI-2011-012",
         "Article 35 of the Consumer Protection Law"),
        ("Article 35 of the Law on Freedom of Association provides that.", "LOI-1990-053",
         "Article 35 of the Law on Freedom of Association"),
        ("Article 35 of the Uniform Act on Commercial Companies provides that.",
         "OHADA-AUSCGIE-2014", "Article 35 of the Uniform Act on Commercial Companies"),
        ("Article 35 of the 2023 Finance Law provides that.", "PRC-9449",
         "Article 35 of the 2023 Finance Law"),
    ])
    def test_un_texte_lu_nomme_en_anglais(self, lus, texte, attendu, cite):
        [occurrence] = lire_les_citations(texte, lus)

        assert _lie(occurrence) == attendu
        assert _cite(texte, occurrence) == cite

    def test_le_traite_ohada_sans_antecedent(self, lus):
        """
        « Article 35 of the OHADA Treaty » : une nature, « OHADA » ne distinguant
        aucun traite. Rien ne la designe : liee au traite lu seulement si son
        morceau porte l'en-tete (_sans_antecedent), signalee sinon (tour final).
        """
        texte = "Article 35 of the OHADA Treaty provides that."
        avec_en_tete = [
            _extrait(5, 2258, "35", TRAITE_1993, "OHADA-TRAITE-1993", _commence(35),
                     jour=date(1993, 10, 17))
            if e.reference == "OHADA-TRAITE-1993" else e
            for e in lus
        ]

        [signalee] = lire_les_citations(texte, lus)
        [liee] = lire_les_citations(texte, avec_en_tete)

        assert signalee.extrait is None and signalee.raison == "article_non_consulte"
        assert _lie(liee) == "OHADA-TRAITE-1993"
        assert _cite(texte, liee) == "Article 35 of the OHADA Treaty"

    def test_le_numero_n_est_pas_une_seconde_citation(self, lus):
        """« Law No. 92/007 », lie seul quand il est seul, est le nom de la citation."""
        texte = "Article 35 of Law No. 92/007 provides it."

        occurrences = lire_les_citations(texte, lus)

        assert [o.forme for o in occurrences] == ["article"]

    @pytest.mark.parametrize("texte", [
        "Article 76 of Law No. 92/007 provides damages.",
        "Article 76 of the Labour Code provides damages.",
        "Article 76 of the Employment Act provides damages.",
    ])
    def test_un_texte_non_lu_nomme_en_anglais_reste_signale(self, tableaux_budgetaires, texte):
        [occurrence] = lire_les_citations(texte, tableaux_budgetaires)

        assert occurrence.extrait is None and occurrence.raison == "texte_non_consulte"


class TestVerbesQuiRessemblentAUneNature:
    """
    « L'article 37 traite du licenciement » : « traite » n'est pas un Traité,
    ni « arrête » un Arrêté, ni « orders », « acts » des textes. Sans mot de
    liaison entre le numero et ce qui suit, la citation ne nomme rien ; la
    premiere version la signalait, la production la rattachait. Lue comme un
    nom, elle serait signalee (aucun traite lu) ; sans nom, elle est liee au
    seul texte lu qui porte l'article, son morceau en portant l'en-tete.
    """

    @pytest.mark.parametrize("texte, cite", [
        ("L'article 37 traite du licenciement.", "article 37"),
        ("L'article 37 arrête la liste des pièces à fournir.", "article 37"),
        ("Article 37 orders the reinstatement of the worker.", "Article 37"),
        ("Article 37 acts as a safeguard.", "Article 37"),
    ])
    def test_un_verbe_n_est_pas_un_texte(self, texte, cite):
        [occurrence] = lire_les_citations(
            texte, [_extrait(37, 2274, "37", CODE_DU_TRAVAIL, "PRC-3", _commence(37))]
        )

        assert occurrence.extrait.article_id == 37 and occurrence.raison is None
        assert _cite(texte, occurrence) == cite

    def test_un_traite_nomme_reste_un_texte(self):
        texte = "Voir l'article 37 du Traité de Marrakech."

        [occurrence] = lire_les_citations(texte, [_extrait(37, 2274, "37", CODE_DU_TRAVAIL, "PRC-3")])

        assert occurrence.extrait is None and occurrence.raison == "texte_non_consulte"

    def test_un_nom_sans_mot_de_liaison_dans_une_liste(self, tableaux_budgetaires):
        """« - Article 76 Code du travail : … » : une majuscule et une nature, un nom."""
        extraits = [*tableaux_budgetaires, _extrait(35, 2274, "35.1", CODE_DU_TRAVAIL, "LOI-1992-007")]
        texte = "- Article 35 Code du travail : le préavis.\n- Article 76 Code du travail : les dommages."

        trente_cinq, soixante_seize = lire_les_citations(texte, extraits)

        assert trente_cinq.extrait.article_id == 35
        assert soixante_seize.extrait is None and soixante_seize.raison == "article_non_consulte"


class TestCheminsRestantsDeLIncident:
    """
    Revue du 2026-10-29 : avec les extraits de l'incident (tableaux de lois de
    finances, Code du travail non lu), ces formes allaient encore au morceau
    76.37 de la loi de finances 2023.
    """

    @pytest.mark.parametrize("texte", [
        "Voir l'article 76 ainsi que l'article 77 du Code du travail.",
        "Voir l'article 76 et à l'article 77 du Code du travail.",
        "Voir l'article 76 ou de l'article 77 du Code du travail.",
        "Voir l'article 76 alinéas 1 et 2 du Code du travail.",
        "Voir l'article 76 premier alinéa du Code du travail.",
        "Voir l'article 76 précité du Code du travail.",
        "Voir l'article 76 combiné avec l'article 77 du Code du travail.",
        "Voir l'article 76 de la législation du travail.",
        "Voir l'article 76 du droit du travail.",
        "Voir l'article 76 du CT.",
        "See Article 76 as well as Article 77 of the Labour Code.",
        "Article 76 subparagraph 2 of the Labour Code provides damages.",
        "Le Code du travail prévoit des dommages-intérêts (article 76 de cette loi).",
    ])
    def test_jamais_le_morceau_76_37(self, tableaux_budgetaires, texte):
        occurrences = lire_les_citations(texte, tableaux_budgetaires)

        assert occurrences and all(o.extrait is None for o in occurrences)
        assert {o.raison for o in occurrences} == {"texte_non_consulte"}

    def test_un_nom_ecrit_avant_une_citation_sans_nom(self, tableaux_budgetaires):
        """
        Rien ne designe le texte de la citation : signalee par _sans_antecedent,
        sans lire le nom ecrit avant (il ne changerait que la raison, que le
        front n'affiche pas ; revue du tour final).
        """
        texte = "Le Code du travail prévoit des dommages-intérêts. L'article 76 précise le montant."

        [occurrence] = lire_les_citations(texte, tableaux_budgetaires)

        assert occurrence.extrait is None and occurrence.raison == "article_non_consulte"

    def test_le_sigle_d_un_code_lu(self, tableaux_budgetaires):
        """« du CT » : le Code du travail lu le porte, les tableaux non."""
        extraits = [*tableaux_budgetaires, _extrait(97420, 2274, "76.1", CODE_DU_TRAVAIL, "LOI-1992-007")]

        [occurrence] = lire_les_citations("Voir l'article 76 du CT.", extraits)

        assert occurrence.extrait.article_id == 97420

    def test_une_citation_sans_nom_apres_un_texte_non_consulte(self, tableaux_budgetaires):
        """« L'article 76 prévoit aussi… » parle du Code du travail que la phrase precedente cite."""
        texte = "Voir l'article 76 du Code du travail. L'article 76 prévoit aussi des dommages."

        premiere, seconde = lire_les_citations(texte, tableaux_budgetaires)

        assert premiere.raison == seconde.raison == "texte_non_consulte"
        assert _cite(texte, seconde) == "article 76"

    def test_une_citation_sans_nom_est_lue_dans_le_texte_precedent(self):
        """
        Revue de la recuperation, panne simulee : la loi de finances 2024 venait
        en tete du contexte, et « L'article 35 dispose… » allait a son morceau
        35.19 alors que la phrase precedente cite le Code du travail.
        """
        extraits = [
            _extrait(1, 9875, "35.19", LF_2024, "PRC-9875", jour=date(2023, 12, 19)),
            _extrait(2, 2274, "35.1", CODE_DU_TRAVAIL, "LOI-1992-007"),
            _extrait(3, 2274, "37", CODE_DU_TRAVAIL, "LOI-1992-007"),
            _extrait(4, 9875, "35.8", LF_2024, "PRC-9875", jour=date(2023, 12, 19)),
        ]
        texte = "Selon l'article 37 du Code du travail, le préavis est dû. L'article 35 dispose qu'il court."

        trente_sept, trente_cinq = lire_les_citations(texte, extraits)

        assert trente_sept.extrait.article_id == 3
        assert trente_cinq.extrait.article_id == 2 and trente_cinq.article == "35"

    def test_une_citation_sans_nom_absente_du_texte_precedent(self, tableaux_budgetaires):
        """Le Code du travail est lu sans son article 76 : l'article manque, il n'est pas pris ailleurs."""
        extraits = [
            *tableaux_budgetaires,
            _extrait(97420, 2274, "35.1", CODE_DU_TRAVAIL, "LOI-1992-007"),
        ]
        texte = "Le préavis (l'article 35 du Code du travail). L'article 76 précise les dommages."

        preavis, dommages = lire_les_citations(texte, extraits)

        assert preavis.extrait.article_id == 97420
        assert dommages.extrait is None and dommages.raison == "article_non_consulte"

    def test_le_nom_apres_une_virgule(self):
        """
        Vraie reponse : « article 88, alinéa 1, point 8, du Décret N°2018/449 ». Le
        nom, apres la virgule, n'etait pas lu : l'article ne tenait qu'a son numero.
        """
        extraits = [
            _extrait(1, 8036, "88", "Loi 2019/024 du 24 décembre 2019 portant code général des "
                     "collectivités territoriales décentralisées", "PRC-8036"),
            _extrait(2, 6622, "88", "Décret N°2018/449 du 01 aout 2018 portant organisation du "
                     "Ministère de la Décentralisation et du Développement Local.", "PRC-6622",
                     jour=date(2018, 8, 1)),
        ]
        texte = "Le contrôle de légalité (**article 88, alinéa 1, point 8, du Décret N°2018/449**)."

        [occurrence] = lire_les_citations(texte, extraits)

        assert occurrence.extrait.article_id == 2
        assert _cite(texte, occurrence) == "article 88, alinéa 1, point 8, du Décret N°2018/449"

    def test_et_suivants_apres_un_texte_nomme(self):
        """Vraie reponse : « Consultez le **Décret N°2015/406** (article 6 et suivants) »."""
        extraits = [
            _extrait(1, 4007, "6", REMUNERATION_DES_MAIRES, "PRC-4007", jour=date(2015, 1, 1)),
            _extrait(2, 4008, "6", INDEMNITES_DES_MAIRES, "PRC-4008", jour=date(2015, 9, 16)),
        ]
        texte = "Consultez le **Décret N°2015/406** (article 6 et suivants) pour le détail."

        decret, article = lire_les_citations(texte, extraits)

        assert decret.forme == "texte" and decret.extrait.law_id == 4008
        assert article.extrait.article_id == 2
        assert _cite(texte, article) == "article 6 et suivants"


class TestRenvoisDeNature:
    """
    « de ce code », « de cette loi » : le dernier texte nomme de cette nature,
    citation ou phrase. La premiere version prenait la citation precedente
    quelle que soit sa nature : « l'article 37 de ce code », apres le Décret
    N°2021/745, etait cherche dans le decret.
    """

    @pytest.fixture
    def lus(self):
        return [
            _extrait(5, 4505, "5", DECRET_2021_745, "PRC-4505", jour=date(2021, 12, 28)),
            _extrait(537, 4505, "37", DECRET_2021_745, "PRC-4505", jour=date(2021, 12, 28)),
            _extrait(37, 2274, "37", CODE_DU_TRAVAIL, "LOI-1992-007"),
        ]

    def test_ce_code_n_est_pas_le_decret_cite_avant(self, lus):
        texte = (
            "Le préavis est fixé par l'article 5 du Décret N°2021/745. Le Code du travail encadre "
            "la rupture : l'article 37 de ce code précise la durée."
        )

        decret, code = lire_les_citations(texte, lus)

        assert decret.extrait.article_id == 5
        assert code.extrait.article_id == 37

    def test_ce_code_sans_code_nomme_avant(self, lus):
        """
        Aucun code nomme : « de ce code » ne designe ni le decret cite avant, ni,
        sans l'en-tete de son article 37, le seul code lu (_sans_antecedent).
        """
        texte = (
            "Le préavis est fixé par l'article 5 du Décret N°2021/745 ; l'article 37 de ce code "
            "précise la durée."
        )
        avec_en_tete = [*lus[:2], _extrait(37, 2274, "37", CODE_DU_TRAVAIL, "LOI-1992-007",
                                           _commence(37))]

        decret, signale = lire_les_citations(texte, lus)
        _, lie = lire_les_citations(texte, avec_en_tete)

        assert decret.extrait.article_id == 5
        assert signale.extrait is None and signale.raison == "article_non_consulte"
        assert lie.extrait.article_id == 37

    def test_cette_loi_apres_un_code_non_consulte(self, tableaux_budgetaires):
        """Le Code du travail est une loi : « de cette loi » ne vise pas la loi de finances lue."""
        texte = "Voir l'article 74 du Code du travail ; l'article 76 de cette loi prévoit les dommages."

        occurrences = lire_les_citations(texte, tableaux_budgetaires)

        assert [o.raison for o in occurrences] == ["texte_non_consulte"] * 2

    def test_un_nom_commun_n_est_pas_un_texte(self):
        """
        Vraies reponses : « un décret spécifique », « le décret cité », « le statut
        domanial » ne nomment aucun texte ; « du même décret » qui suit reste le
        decret cite avant.
        """
        extraits = [
            _extrait(1, 4008, "2", INDEMNITES_DES_MAIRES, "PRC-4008", jour=date(2015, 9, 16)),
            _extrait(2, 4007, "3", REMUNERATION_DES_MAIRES, "PRC-4007", jour=date(2015, 1, 1)),
            _extrait(3, 4007, "2", REMUNERATION_DES_MAIRES, "PRC-4007", jour=date(2015, 1, 1)),
        ]
        texte = (
            "1. Un traitement de base (article 3 du décret n°2015/405).\n"
            "2. Une indemnité de fonction (montant non précisé dans le décret cité, soumise à un "
            "décret spécifique, selon le statut domanial — voir l’article 2 du même décret)."
        )

        trois, deux = lire_les_citations(texte, extraits)

        assert trois.extrait.article_id == 2
        assert deux.extrait.article_id == 3

    @pytest.mark.parametrize("nom_commun", ["un Décret spécifique", "an Act of Parliament"])
    def test_un_indefini_ne_nomme_aucun_texte(self, nom_commun):
        """Meme avec une majuscule : « un Décret spécifique » n'est pas un texte nomme."""
        extraits = [
            _extrait(2, 4007, "3", REMUNERATION_DES_MAIRES, "PRC-4007", jour=date(2015, 1, 1)),
            _extrait(3, 4007, "2", REMUNERATION_DES_MAIRES, "PRC-4007", jour=date(2015, 1, 1)),
        ]
        texte = (
            f"Le traitement (article 3 du décret n°2015/405) ; l'indemnité relève de {nom_commun} "
            "(voir l’article 2 du même décret)."
        )

        trois, deux = lire_les_citations(texte, extraits)

        assert deux.extrait.article_id == 3


class TestTextesQuiNeFontQueCiterLeNom:
    """
    Un intitule qui ne fait que CITER le nom (« Décret … relevant du Code du
    Travail », « Loi … modifiant … la loi … portant code électoral ») ne porte
    l'article de ce texte que dans un morceau qui le reecrit (« (nouveau) »).
    Son propre article de meme numero etait pris pour celui du texte nomme.
    """

    def test_un_decret_qui_cite_le_code_du_travail(self):
        retraite = _extrait(1, 8690, "2", RETRAITE_DES_AGENTS, "PRC-8690", jour=date(2020, 12, 30))
        texte = "Voir l'article 2 du Code du travail."

        [seul] = lire_les_citations(texte, [retraite])
        [avec_le_code] = lire_les_citations(
            texte, [retraite, _extrait(2, 2274, "2", CODE_DU_TRAVAIL, "LOI-1992-007")]
        )

        assert seul.extrait is None and seul.raison == "texte_non_consulte"
        assert avec_le_code.extrait.article_id == 2
        [par_le_sigle] = lire_les_citations("Voir l'article 2 du CT.", [retraite])
        assert par_le_sigle.extrait is None and par_le_sigle.raison == "texte_non_consulte"

    def test_la_loi_modificative_porte_les_articles_qu_elle_reecrit(self):
        """
        Vraie reponse sur la vacance de la présidence : le morceau « 145 » de la
        loi 2026/003 est l'article 145 du Code électoral modifie, son morceau
        « SECTION_1 » porte « ARTICLE 142 (nouveau) ».
        """
        modificative = [
            _extrait(145, 10680, "145", CODE_ELECTORAL_MODIFIE, "PRC-10680",
                     "(nouveau).- (1) L'empêchement définitif du Président de la République…",
                     jour=date(2026, 4, 14)),
            _extrait(142, 10680, "SECTION_1", CODE_ELECTORAL_MODIFIE, "PRC-10680",
                     "«ARTiCLE 142 (nouveau).- (1) En cas de vacance de la Présidence…",
                     jour=date(2026, 4, 14)),
            _extrait(2, 10680, "2", CODE_ELECTORAL_MODIFIE, "PRC-10680",
                     "La présente loi sera enregistrée et publiée au Journal officiel.",
                     jour=date(2026, 4, 14)),
        ]
        texte = (
            "Voir l'article 145 du Code électoral, l'article 142 du Code électoral et l'article 2 "
            "du Code électoral."
        )

        cent_quarante_cinq, cent_quarante_deux, deux = lire_les_citations(texte, modificative)

        assert cent_quarante_cinq.extrait.article_id == 145
        assert cent_quarante_deux.extrait.article_id == 142 and cent_quarante_deux.article == "142"
        assert deux.extrait is None and deux.raison == "texte_non_consulte"

    def test_un_nom_lu_mot_a_mot_dans_le_renvoi_d_un_autre_texte(self):
        """
        « Acte uniforme portant organisation des sûretés » n'est d'un seul tenant
        dans aucun intitule (« … RÉVISÉ portant … ») : il est lu mot a mot. Le
        decret qui le publie (intitule ecrit pour le test) porte ces mots, mais
        dans son renvoi : son propre article 9 n'est pas celui de l'Acte uniforme.
        """
        publication = _extrait(
            1, 7001, "9", "Décret N°2011/001 du 3 janvier 2011 portant publication de l'Acte "
            "uniforme révisé portant organisation des sûretés", "PRC-7001", jour=date(2011, 1, 3),
        )
        texte = "Voir l'article 9 de l'Acte uniforme portant organisation des sûretés."

        [seul] = lire_les_citations(texte, [publication])
        [avec_l_acte] = lire_les_citations(
            texte, [publication, _extrait(2, 2275, "9", AU_SURETES, "OHADA-AUS-2010")]
        )

        assert seul.extrait is None and seul.raison == "texte_non_consulte"
        assert avec_l_acte.extrait.article_id == 2

    def test_la_loi_qui_autorise_la_ratification_n_est_pas_la_convention(self):
        """
        Intitule reel (PRC-9111) : « … autorisant le président de la République
        à ratifier la convention des nations unies … ». Son article 5 passait
        pour celui de la convention ; la loi elle-meme reste citable par son numero.
        """
        ratification = _extrait(
            1, 9111, "5", "Loi N°2021/018 du 16 décembre 2021 autorisant le président de la "
            "République à ratifier la convention des nations unies sur les droits des personnes "
            "handicapées, adopté le 13 décembre 2006", "PRC-9111", jour=date(2021, 12, 16),
        )

        [convention] = lire_les_citations(
            "Voir l'article 5 de la Convention des Nations Unies sur les droits des personnes "
            "handicapées.", [ratification],
        )
        [loi] = lire_les_citations("Voir l'article 5 de la loi n° 2021/018.", [ratification])

        assert convention.extrait is None and convention.raison == "texte_non_consulte"
        assert loi.extrait.article_id == 1

    def test_le_code_plutot_que_l_article_de_la_loi_qui_le_modifie(self):
        extraits = [
            _extrait(1, 10680, "5", CODE_ELECTORAL_MODIFIE, "PRC-10680", jour=date(2026, 4, 14)),
            _extrait(2, 2282, "5", CODE_ELECTORAL, "LOI-2012-001", jour=date(2012, 4, 19)),
        ]

        [occurrence] = lire_les_citations("L'article 5 du Code électoral le prévoit.", extraits)

        assert occurrence.extrait.article_id == 2


class TestVersionsDUnTexte:
    """Une annee qui suit le nom departage deux versions d'un texte lues ensemble."""

    @pytest.fixture
    def codes_miniers(self):
        return [
            _extrait(2023, 9866, "5", "Loi N°2023/014 du 19 décembre 2023 portant Code Minier",
                     "PRC-9866", jour=date(2023, 12, 19)),
            _extrait(2016, 5621, "5", "Loi N°2016/017 du 14 décembre 2016 portant code minier du "
                     "Cameroun", "PRC-5621", jour=date(2016, 12, 14)),
        ]

    def test_le_code_minier_de_2016(self, codes_miniers):
        texte = "Voir l'article 5 du Code minier de 2016."

        [deux_versions] = lire_les_citations(texte, codes_miniers)
        [seul_2023] = lire_les_citations(texte, codes_miniers[:1])

        assert deux_versions.extrait.article_id == 2016
        assert _cite(texte, deux_versions) == "article 5 du Code minier de 2016"
        assert seul_2023.extrait is None and seul_2023.raison == "texte_non_consulte"

    @pytest.mark.parametrize("texte", [
        "Voir l'article 5 de la loi de finances 2023.",
        "Voir l'article 5 de la loi de finances pour l'exercice 2023.",
        "See Article 5 of the Finance Law for 2023.",
    ])
    def test_l_exercice_d_une_loi_de_finances(self, texte):
        """La loi de finances « pour l'exercice 2023 » est votee en 2022."""
        extraits = [
            _extrait(2026, 10557, "5", LF_2026, "PRC-10557", jour=date(2025, 12, 17)),
            _extrait(2023, 9449, "5", LF_2023, "PRC-9449", jour=date(2022, 12, 27)),
        ]

        [occurrence] = lire_les_citations(texte, extraits)

        assert occurrence.extrait.article_id == 2023


@pytest.mark.parametrize("mention, nom", [
    ("Statut général de la fonction publique tel que modifié", "Statut général de la fonction publique"),
    ("Law on Freedom of Association provides that", "Law on Freedom of Association"),
    ("Uniform Act on Arbitration provides that", "Uniform Act on Arbitration"),
])
def test_etendue_d_un_nom_apres_la_revue(mention, nom):
    assert mention[:etendue_du_nom(mention)] == nom


# ==================== TOUR 3 : LES FORMES QUE LES DEUX PREMIERS MANQUAIENT ====================

ELECTRICITE = (
    "Loi N°2011/022 du 14 décembre 2011 régissant le secteur de l'électricité au Cameroun"
)
INVESTISSEMENT = (
    "Loi N°2013/004 du 18 avril 2013 fixant les incitations à l'investissement privé en "
    "République du Cameroun"
)
ACTIVITE_COMMERCIALE = "Loi N°2015/018 du 21 décembre 2015 régissant l'activité commerciale au Cameroun"
REGIME_FINANCIER = (
    "Loi N° 2018/012 du 11 juillet 2018 portant Régime Financier de l’Etat et des autres entités "
    "publiques"
)
CODE_PETROLIER = "Loi N° 2019/008 du 25 avril 2019 portant code pétrolier"
APPLICATION_DU_CODE_PETROLIER = (
    "Décret N°2023/232 du 04 mai 2023 fixant les modalités d'application de la loi n°2019/008 du "
    "25 avril 2019 portant Code Pétrolier"
)


class TestNatureSeule:
    """
    « l'article 76 de la loi », « du décret », « of the Law » : une nature
    seule, juste apres un texte nomme, est ce texte, comme « de cette loi ».
    Elle allait a toute loi lue, donc au morceau 76.37 de la loi de finances :
    l'incident, sans le nom (sondes du 2026-10-29, tour 3).
    """

    @pytest.mark.parametrize("texte", [
        "Voir l'article 74 du Code du travail. Les dommages sont prévus à l'article 76 de la loi.",
        "Article 74 of the Labour Code sets notice; Article 76 of the Law provides damages.",
        "Le Code du travail prévoit des dommages-intérêts (article 76 de la loi).",
    ])
    def test_apres_un_texte_non_consulte(self, tableaux_budgetaires, texte):
        occurrences = lire_les_citations(texte, tableaux_budgetaires)

        assert occurrences and all(o.extrait is None for o in occurrences)
        assert {o.raison for o in occurrences} == {"texte_non_consulte"}

    def test_le_code_du_travail_est_une_loi(self, tableaux_budgetaires):
        """La loi de finances porte aussi un article 37, et vient en premier."""
        extraits = [
            _extrait(1, 2232, "37", LF_2023, "PRC-9449", jour=date(2022, 12, 27)),
            _extrait(35, 2274, "35.1", CODE_DU_TRAVAIL, "LOI-1992-007"),
            _extrait(37, 2274, "37", CODE_DU_TRAVAIL, "LOI-1992-007"),
        ]
        texte = "Le préavis (l'article 35 du Code du travail) ; l'article 37 de la loi précise la durée."

        preavis, duree = lire_les_citations(texte, extraits)

        assert preavis.extrait.article_id == 35
        assert duree.extrait.article_id == 37
        assert _cite(texte, duree) == "article 37 de la loi"

    def test_le_decret_cite_avant(self):
        extraits = [
            _extrait(406, 4008, "6", INDEMNITES_DES_MAIRES, "PRC-4008", jour=date(2015, 9, 16)),
            _extrait(5, 4505, "5", DECRET_2021_745, "PRC-4505", jour=date(2021, 12, 28)),
            _extrait(6, 4505, "6", DECRET_2021_745, "PRC-4505", jour=date(2021, 12, 28)),
        ]
        texte = "Selon l'article 5 du Décret N°2021/745, oui ; l'article 6 du décret le précise."

        cinq, six = lire_les_citations(texte, extraits)

        assert cinq.extrait.article_id == 5 and six.extrait.article_id == 6

    @pytest.mark.parametrize("renvoi, attendu", [
        ("de la loi", 2), ("de cette loi", 2), ("du code", 2), ("du même décret", 1),
    ])
    def test_un_texte_qui_n_est_que_mentionne_par_l_intitule_n_est_pas_de_cette_nature(
        self, renvoi, attendu
    ):
        """
        Le decret d'application cite « la loi n°2019/008 … portant Code Pétrolier »
        dans son intitule : ce n'est ni une loi ni un code. « de la loi » allait
        a son propre article 12. Aucune loi, aucun code nomme avant : le seul
        texte de cette nature qui porte l'article et son en-tete
        (_sans_antecedent) ; le decret, s'il passait pour une loi, en ferait
        un second, et la citation serait signalee.
        """
        extraits = [
            _extrait(1, 9653, "12", APPLICATION_DU_CODE_PETROLIER, "PRC-9653", jour=date(2023, 5, 4)),
            _extrait(2, 7487, "12", CODE_PETROLIER, "PRC-7487", _commence(12),
                     jour=date(2019, 4, 25)),
            _extrait(3, 9653, "5", APPLICATION_DU_CODE_PETROLIER, "PRC-9653", jour=date(2023, 5, 4)),
        ]
        texte = f"Selon l'article 5 du Décret N°2023/232, oui ; l'article 12 {renvoi} précise le régime."

        cinq, douze = lire_les_citations(texte, extraits)

        assert cinq.extrait.article_id == 3
        assert douze.extrait.article_id == attendu

    @pytest.mark.parametrize("texte", [
        "Selon l'article 76 de la loi, les crédits sont ouverts.",
        "En cas de licenciement abusif, l'article 76 de la loi camerounaise prévoit des "
        "dommages-intérêts.",
        "Article 76 of the Law provides for damages.",
    ])
    def test_sans_antecedent_une_loi_n_est_pas_designee(self, tableaux_budgetaires, texte):
        """
        « de la loi », aucune citation avant : la production prenait la premiere
        loi lue qui porte l'article, et rejouee sur les lignes reelles de
        l'incident, « de la loi camerounaise » allait au morceau 76.37 de la loi
        de finances 2023. Rien ne designe une loi : signalee (tour final).
        """
        [occurrence] = lire_les_citations(texte, tableaux_budgetaires)

        assert occurrence.extrait is None and occurrence.raison == "texte_non_consulte"

    def test_sans_antecedent_un_code_vise_les_codes_lus(self):
        """
        Un code, lui, designe les codes lus : le seul code lu qui porte
        l'article, son morceau en portant l'en-tete (_sans_antecedent) ; jamais
        l'article 37 de la loi de finances, lue en premier. Sans en-tete :
        signalee (tour final).
        """
        finances = _extrait(1, 2232, "37", LF_2023, "PRC-9449", jour=date(2022, 12, 27))
        texte = "Selon l'article 37 du code, le préavis est dû."

        [liee] = lire_les_citations(texte, [finances, _extrait(
            37, 2274, "37", CODE_DU_TRAVAIL, "LOI-1992-007", _commence(37),
        )])
        [signalee] = lire_les_citations(texte, [finances, _extrait(
            37, 2274, "37", CODE_DU_TRAVAIL, "LOI-1992-007",
        )])

        assert liee.extrait.article_id == 37
        assert signalee.extrait is None and signalee.raison == "article_non_consulte"

    @pytest.mark.parametrize("nom", ["Code OHADA", "notre Code OHADA"])
    def test_un_nom_qui_n_est_pas_une_nature_reste_un_nom(self, nom):
        """
        « Code OHADA » : « OHADA » n'est pas une nature, c'est l'intitule entier
        d'un texte lu. Le possessif n'y change rien.
        """
        extraits = [
            _extrait(161, 1, "161", "Code OHADA", "LOI-2024-001"),
            _extrait(5, 2, "5", "Code des sociétés", "LOI-2023-015"),
        ]
        texte = f"Voir l'article 74 du Code du travail ; selon l'article 5 du {nom}, ils sont loyaux."

        travail, ohada = lire_les_citations(texte, extraits)

        assert travail.raison == "texte_non_consulte"
        assert ohada.extrait is None and ohada.raison == "article_non_consulte"


class TestQualificatifsDeLArticle:
    """« l'article 76 in fine du … », « modifié », « as amended » : le nom suit le qualificatif."""

    @pytest.mark.parametrize("texte, cite", [
        ("Voir l'article 76 in fine du Code du travail.", "article 76 in fine du Code du travail"),
        ("Voir l'article 76 modifié du Code du travail.", "article 76 modifié du Code du travail"),
        ("See Article 76 in fine of the Labour Code.", "Article 76 in fine of the Labour Code"),
        ("See Article 76 as amended of the Labour Code.",
         "Article 76 as amended of the Labour Code"),
    ])
    def test_jamais_le_morceau_76_37(self, tableaux_budgetaires, texte, cite):
        [occurrence] = lire_les_citations(texte, tableaux_budgetaires)

        assert occurrence.extrait is None and occurrence.raison == "texte_non_consulte"
        assert _cite(texte, occurrence) == cite

    def test_le_texte_lu_garde_son_article(self):
        extraits = [
            _extrait(1, 2232, "37", LF_2023, "PRC-9449", jour=date(2022, 12, 27)),
            _extrait(37, 2274, "37", CODE_DU_TRAVAIL, "LOI-1992-007"),
        ]

        [occurrence] = lire_les_citations("Voir l'article 37 in fine du Code du travail.", extraits)

        assert occurrence.extrait.article_id == 37


class TestDeterminantsDuNom:
    """Un possessif ou un demonstratif devant le nom d'un texte."""

    @pytest.fixture
    def lus(self):
        return [
            _extrait(1, 2232, "37", LF_2023, "PRC-9449", jour=date(2022, 12, 27)),
            _extrait(37, 2274, "37", CODE_DU_TRAVAIL, "LOI-1992-007"),
        ]

    @pytest.mark.parametrize("texte", [
        "Voir l'article 37 de notre Code du travail.",
        "See Article 37 of our Labour Code.",
    ])
    def test_le_possessif_n_est_pas_dans_l_intitule(self, lus, texte):
        """« notre » etait cherche dans l'intitule : la citation etait signalee, le Code lu."""
        [occurrence] = lire_les_citations(texte, lus)

        assert occurrence.extrait.article_id == 37

    def test_un_demonstratif_devant_un_nom_designe_ce_nom(self, tableaux_budgetaires):
        """« de cette loi de finances » n'est pas le Code du travail cite juste avant."""
        texte = "Voir l'article 76 du Code du travail, puis l'article 76 de cette loi de finances."

        travail, finances = lire_les_citations(texte, tableaux_budgetaires)

        assert travail.raison == "texte_non_consulte"
        assert finances.extrait.article_id == 85057

    @pytest.mark.parametrize("renvoi", ["de cette loi de finances", "de la loi de finances"])
    def test_parmi_les_textes_du_nom_celui_que_la_reponse_vient_de_citer(self, renvoi):
        """
        La loi de finances 2026 s'intitule « … portant loi DES Finances … » :
        « la loi de finances » ne nommait que celle de 2023, lue en premier.
        """
        extraits = [
            _extrait(2023, 2232, "5", LF_2023, "PRC-9449", jour=date(2022, 12, 27)),
            _extrait(2026, 2237, "5", LF_2026, "PRC-10557", jour=date(2025, 12, 17)),
            _extrait(2027, 2237, "6", LF_2026, "PRC-10557", jour=date(2025, 12, 17)),
        ]
        texte = f"Voir l'article 6 de la loi de finances 2026, et l'article 5 {renvoi}."

        six, cinq = lire_les_citations(texte, extraits)

        assert six.extrait.article_id == 2027
        assert cinq.extrait.article_id == 2026

    @pytest.mark.parametrize("intitule_2026", [
        LF_2026,
        # Variante ecrite pour le test : le nom n'y est pas d'un seul tenant.
        "Loi N°2025/012 du 17 décembre 2025 portant loi relative aux Finances de la République "
        "du Cameroun pour l’exercice 2026",
    ])
    def test_la_loi_de_finances_de_l_annee_sous_un_autre_intitule(self, intitule_2026):
        """
        « loi de finances 2026 » : seule la loi de 2023 porte ce nom d'un seul
        tenant, et l'annee l'ecarte. Le nom est relu mot a mot, annee comprise.
        """
        extraits = [
            _extrait(2023, 2232, "5", LF_2023, "PRC-9449", jour=date(2022, 12, 27)),
            _extrait(2026, 2237, "5", intitule_2026, "PRC-10557", jour=date(2025, 12, 17)),
        ]

        [occurrence] = lire_les_citations("Voir l'article 5 de la loi de finances 2026.", extraits)

        assert occurrence.extrait.article_id == 2026

    def test_ce_code_du_travail(self, lus, tableaux_budgetaires):
        texte = "Voir l'article 5 du Décret N°2021/745 ; l'article 37 de ce Code du travail."

        lu = [o for o in lire_les_citations(texte, lus) if o.forme == "article"]
        non_lu = [o for o in lire_les_citations(texte, tableaux_budgetaires) if o.forme == "article"]

        assert lu[1].extrait.article_id == 37
        assert [o.raison for o in non_lu] == ["texte_non_consulte"] * 2


class TestTexteNommeApresLaCitation:
    """
    Une citation sans nom, rien de nomme avant : le texte que la fin de sa
    proposition nomme confirme l'article, ou la citation est signalee. « En
    droit du travail, l'article 76 couvre les dommages (Code du travail). »
    allait au morceau 76.37 (jeu de sondes tenu a l'ecart, tour 3). Le nom ne
    choisit jamais un autre texte que celui de l'article retenu (tour final,
    TestUnNomEcritEnPassantNeChoisitPas).

    Les tableaux de l'incident sont signales sans lire ce nom (_sans_antecedent) :
    ces tests lisent l'article 76 du Code pénal, en-tete compris, auquel la
    citation irait sans le nom.
    """

    @pytest.fixture
    def penal_76(self):
        return [_extrait(76, 4722, "76", CODE_PENAL, "PRC-4722",
                         _commence(76, "Exécution de la loi."), jour=date(2016, 7, 12))]

    @pytest.mark.parametrize("texte", [
        "En droit du travail, l'article 76 couvre les dommages (Code du travail).",
        "L'article 76 prévoit des dommages-intérêts dans le Code du travail.",
        "L'article 76 prévoit des dommages-intérêts (Loi N°92/007).",
        "Article 76 provides for damages under the Labour Code.",
    ])
    def test_le_texte_non_consulte_nomme_dans_la_phrase(self, penal_76, texte):
        occurrences = [o for o in lire_les_citations(texte, penal_76) if o.forme == "article"]

        assert [o.raison for o in occurrences] == ["texte_non_consulte"]
        assert occurrences[0].extrait is None

    def test_le_texte_lu_nomme_dans_la_phrase(self):
        """
        Le Code du travail, seul texte lu a porter un article 76, son morceau en
        portant l'en-tete : le nom ne la signale pas. Sans en-tete, rien ne
        prouve l'article, et le nom ne choisit pas : signalee (tour final).
        """
        finances = _extrait(1, 2237, "95.95", LF_2026, "PRC-10557", jour=date(2025, 12, 17))
        texte = "En droit du travail, l'article 76 couvre les dommages (Code du travail)."

        [occurrence] = lire_les_citations(texte, [finances, _extrait(
            76, 2274, "76", CODE_DU_TRAVAIL, "LOI-1992-007", _commence(76),
        )])
        [sans_en_tete] = lire_les_citations(texte, [finances, _extrait(
            76, 2274, "76", CODE_DU_TRAVAIL, "LOI-1992-007",
        )])

        assert occurrence.extrait.article_id == 76
        assert _cite(texte, occurrence) == "article 76"
        assert sans_en_tete.extrait is None and sans_en_tete.raison == "article_non_consulte"

    def test_le_nom_dans_la_phrase_ne_leve_pas_le_doute(self):
        """
        La loi de finances (morceau 76.37) et le Code du travail portent chacun
        un article 76 : le nom ecrit dans la phrase ne choisit pas, et ne
        confirme pas le premier extrait retenu. Signalee (tour final).
        """
        extraits = [
            _extrait(1, 2232, "76.37", LF_2023, "PRC-9449", jour=date(2022, 12, 27)),
            _extrait(76, 2274, "76", CODE_DU_TRAVAIL, "LOI-1992-007"),
        ]
        texte = "En droit du travail, l'article 76 couvre les dommages (Code du travail)."

        [occurrence] = lire_les_citations(texte, extraits)

        assert occurrence.extrait is None and occurrence.raison == "article_non_consulte"
        assert _cite(texte, occurrence) == "article 76"

    @pytest.mark.parametrize("texte", [
        # La phrase suivante, ou la proposition apres un point-virgule ou une
        # virgule, n'est pas la sienne.
        "L'article 76 fixe les crédits. Le Code du travail ne s'applique pas.",
        "L'article 76 fixe les crédits ; le Code du travail ne s'applique pas.",
        "L'article 76 fixe les crédits, et le Code du travail ne s'applique pas.",
        # Le nom d'une autre citation, plus loin, est le sien.
        "L'article 76 fixe les crédits, comme l'article 77 du Code du travail.",
    ])
    def test_seulement_la_fin_de_sa_phrase(self, penal_76, texte):
        premiere = lire_les_citations(texte, penal_76)[0]

        assert premiere.extrait.article_id == 76


class TestNomsAnglaisDuCorpus:
    """
    Les mots des intitules du corpus sous leur nom anglais. Un mot que la
    table ignore est cherche tel quel, ne se trouve pas, et la citation est
    signalee : jamais rattachee a un autre texte.
    """

    @pytest.mark.parametrize("texte, titre, reference", [
        ("Article 12 of the Law governing the Electricity Sector applies.", ELECTRICITE,
         "LOI-2011-022"),
        ("Article 12 of the Law on Private Investment Incentives applies.", INVESTISSEMENT,
         "LOI-2013-004"),
        ("Article 12 of the Law governing Commercial Activity applies.", ACTIVITE_COMMERCIALE,
         "LOI-2015-018"),
        ("Article 12 of the Law on the Financial Regime of the State applies.", REGIME_FINANCIER,
         "PRC-6580"),
        ("Article 12 of the Petroleum Code applies.", CODE_PETROLIER, "PRC-7487"),
    ])
    def test_le_texte_lu_et_un_leurre_du_meme_numero(self, texte, titre, reference):
        extraits = [
            _extrait(1, 2232, "12", LF_2023, "PRC-9449", jour=date(2022, 12, 27)),
            _extrait(2, 9653, "12", APPLICATION_DU_CODE_PETROLIER, "PRC-9653", jour=date(2023, 5, 4)),
            _extrait(3, 3, "12", titre, reference),
        ]

        [occurrence] = lire_les_citations(texte, extraits)

        assert _lie(occurrence) == reference

    def test_un_mot_inconnu_signale_sans_rattacher(self, tableaux_budgetaires):
        [occurrence] = lire_les_citations(
            "Article 76 of the Law on Quantum Widgets applies.", tableaux_budgetaires
        )

        assert occurrence.extrait is None and occurrence.raison == "texte_non_consulte"


# ==================== TOUR FINAL : DANS LE DOUTE, SIGNALER ====================

CHR = (
    "Décret N°2022/013 du 11 janvier 2022 portant création, organisation et fonctionnement "
    "des Centres Hospitaliers Régionaux"
)
INFRASTRUCTURES_SPORTIVES = (
    "Décret N°2022/384 du 17 août 2022 portant création, organisation et fonctionnement de "
    "l'Office National des Infrastructures et Equipements Sportifs"
)
FACULTE_DE_GAROUA = (
    "Décret N°2018/004 du 08 janvier 2018 portant création de la Faculté de Médecine et des "
    "Sciences Biomédicales de Garoua."
)
TRAITE_DES_PERSONNES = (
    "Loi N°2011/024 du 14 décembre 2011 relative à la lutte contre le trafic et la traite des "
    "personnes"
)
ORIENTATION_DE_L_EDUCATION = "Loi N°98/004 du 14 avril 1998 d'orientation de l'éducation au Cameroun"
LF_2022 = (
    "Loi N°2021/026 du 16 décembre 2021 portant loi de finances de la République du Cameroun "
    "pour l'exercice 2022"
)
LOI_DE_REGLEMENT_2021 = (
    "Loi N°2022/016 du 27 décembre 2022 portant loi de règlement de la République du Cameroun "
    "pour l'exercice 2021"
)
AU_PROCEDURES_COLLECTIVES = (
    "Acte uniforme portant organisation des procédures collectives d'apurement du passif du 10 "
    "septembre 2015"
)


def _code_du_travail_37(contenu="Contenu de l'article."):
    return _extrait(37, 2274, "37", CODE_DU_TRAVAIL, "LOI-1992-007", contenu)


def _finances_37():
    return _extrait(9449, 2232, "37", LF_2023, "PRC-9449", jour=date(2022, 12, 27))


def _penal_37(contenu="Contenu de l'article."):
    return _extrait(4722, 4722, "37", CODE_PENAL, "PRC-4722", contenu, jour=date(2016, 7, 12))


class TestUnNomEcritEnPassantNeChoisitPas:
    """
    Revue du 2026-10-29, tour 3 : un nom ecrit dans la phrase, avant une
    citation sans nom ou apres elle, etait lu comme son texte, et la
    rattachait a un AUTRE texte lu du meme numero que celui que la production
    retenait. Il ne fait plus que confirmer l'article retenu, ou signaler.
    """

    @pytest.mark.parametrize("texte, lus", [
        # La virgule clot sa proposition : le texte nomme ensuite n'est pas le sien.
        ("L'article 37 fixe le préavis, que la Loi de finances 2023 ne modifie pas.",
         [_code_du_travail_37(), _finances_37()]),
        ("L'article 37 précise la durée du préavis, et le Code pénal punit par ailleurs la fraude.",
         [_code_du_travail_37(), _penal_37()]),
        ("L'article 37 fixe le préavis, contrairement au Code pénal qui ne dit rien.",
         [_code_du_travail_37(), _penal_37()]),
    ])
    def test_le_nom_apres_la_virgule_n_est_pas_le_sien(self, texte, lus):
        """
        Le Code du travail, lu en premier, et l'autre texte portent chacun un
        article 37 : le tour 3 la rattachait au texte nomme apres la virgule.
        Rien ne designe le sien : signalee (tour final). Le Code du travail lu
        seul, son morceau portant l'en-tete, l'autre nom, non consulte, ne la
        signale pas : la virgule clot sa proposition.
        """
        [occurrence] = lire_les_citations(texte, lus)
        [seul] = lire_les_citations(texte, [_code_du_travail_37(_commence(37))])

        assert occurrence.extrait is None and occurrence.raison == "article_non_consulte"
        assert _lie(seul) == "LOI-1992-007"

    @pytest.mark.parametrize("texte, lus", [
        ("La Loi de finances 2023 ne traite pas du licenciement. L'article 37 fixe la durée du "
         "préavis.", [_code_du_travail_37(), _finances_37()]),
        ("Le Code pénal réprime le harcèlement. En matière de préavis, l'article 37 fixe la durée.",
         [_code_du_travail_37(), _penal_37()]),
    ])
    def test_le_nom_de_la_phrase_precedente_ne_prend_pas_l_article(self, texte, lus):
        """
        Les deux textes lus portent un article 37 : la production prenait le
        premier, le tour 3 le texte nomme. Le nom contredit le premier, et
        rien ne dit lequel : signalee.
        """
        [occurrence] = lire_les_citations(texte, lus)

        assert occurrence.extrait is None and occurrence.raison == "article_non_consulte"

    def test_le_nom_qui_confirme_l_article_retenu(self):
        """
        Le Code pénal, seul texte lu a porter un article 37, son morceau en
        portant l'en-tete : le nom ecrit avant ne la signale pas. Il ne choisit
        rien pour autant : sans l'en-tete, elle est signalee (tour final).
        """
        texte = "Le Code pénal réprime le harcèlement. L'article 37 punit le harceleur."
        travail_38 = _extrait(38, 2274, "38", CODE_DU_TRAVAIL, "LOI-1992-007")

        [occurrence] = lire_les_citations(texte, [_penal_37(_commence(37)), travail_38])
        [sans_en_tete] = lire_les_citations(texte, [_penal_37(), travail_38])

        assert _lie(occurrence) == "PRC-4722"
        assert sans_en_tete.extrait is None and sans_en_tete.raison == "article_non_consulte"

    @pytest.mark.parametrize("texte, lus", [
        ("Le Code pénal réprime le harcèlement. L'article 37 punit le harceleur.",
         [_penal_37(), _code_du_travail_37()]),
        # La phrase dit que ce texte n'est PAS en cause ; lu en premier, il
        # etait « confirme » par son nom.
        ("La Loi de finances 2023 ne traite pas du licenciement. L'article 37 fixe la durée du "
         "préavis.", [_finances_37(), _code_du_travail_37()]),
    ])
    def test_le_nom_ne_leve_pas_le_doute(self, texte, lus):
        """
        Deux textes lus portent un article 37, et aucune citation ne designe le
        sien : le nom ecrit dans la phrase qui precede ne confirme pas le
        premier extrait. Signalee (tour final).
        """
        [occurrence] = lire_les_citations(texte, lus)

        assert occurrence.extrait is None and occurrence.raison == "article_non_consulte"

    @pytest.mark.parametrize("texte", [
        "Dans le code du travail, l'article 76 prévoit des dommages-intérêts.",
        "Le licenciement ouvre droit à réparation selon le code du travail. L'article 76 en fixe "
        "le montant.",
        "Le licenciement ouvre droit à réparation selon la loi de 1992. L'article 76 en fixe le "
        "montant.",
    ])
    def test_un_nom_en_minuscules_signale_aussi(self, tableaux_budgetaires, texte):
        """
        Sondes de la revue du tour 3 : ces formes allaient au morceau 76.37,
        desormais signale sans lire le nom (_sans_antecedent). Avec l'article 76
        du Code pénal, en-tete compris, auquel la citation irait sans le nom, le
        nom en minuscules la signale : texte non consulte.
        """
        penal_76 = _extrait(76, 4722, "76", CODE_PENAL, "PRC-4722", _commence(76),
                            jour=date(2016, 7, 12))

        [tableau] = lire_les_citations(texte, tableaux_budgetaires)
        [occurrence] = lire_les_citations(texte, [penal_76])

        assert tableau.extrait is None and tableau.raison == "article_non_consulte"
        assert occurrence.extrait is None and occurrence.raison == "texte_non_consulte"


class TestLeMemeArticleNommeAilleurs:
    """
    Revue du 2026-10-29, tour 3 : « **En bref** : l'article 76 prévoit… **En
    détail** : l'article 76 du Code du travail… » mettait la loi de finances
    sous « Articles cités », a cote d'une alerte sur le meme article.
    """

    def test_le_nom_ecrit_plus_loin_signale_la_citation_sans_nom(self, tableaux_budgetaires):
        """
        Avec les tableaux, « l'article 76 » est signale sans lien (_sans_antecedent).
        Avec l'article 76 du Code pénal, en-tete compris, auquel elle irait
        sinon, c'est le meme article nomme plus loin pour un autre texte (le
        Code du travail, non consulte) qui la signale.
        """
        texte = (
            "**En bref** : l'article 76 prévoit des dommages-intérêts.\n\n**En détail** : "
            "l'article 76 du Code du travail les fixe."
        )
        penal_76 = _extrait(76, 4722, "76", CODE_PENAL, "PRC-4722", _commence(76),
                            jour=date(2016, 7, 12))

        bref, detail = lire_les_citations(texte, tableaux_budgetaires)
        bref_penal, detail_penal = lire_les_citations(texte, [penal_76])

        assert bref.extrait is None and bref.raison == "article_non_consulte"
        assert _cite(texte, bref) == "article 76"
        assert detail.raison == "texte_non_consulte"
        assert bref_penal.extrait is None and bref_penal.raison == "texte_non_consulte"
        assert detail_penal.raison == "texte_non_consulte"

    def test_le_nom_ecrit_plus_loin_ne_confirme_pas(self):
        """
        Deux textes lus portent un article 37, et rien avant la citation ne
        designe le sien : signalee, meme si le meme article est nomme plus
        loin pour l'un d'eux. Une fausse alerte, voulue : la confirmation liait
        « l'article 76 » au morceau 76.37 des qu'une autre phrase nommait la
        loi de finances pour son article 76 (revue du 2026-10-29, tour final).
        """
        texte = "L'article 37 fixe le préavis. Voir l'article 37 du Code du travail."

        sans_nom, nomme = lire_les_citations(texte, [_code_du_travail_37(), _finances_37()])

        assert sans_nom.extrait is None and sans_nom.raison == "article_non_consulte"
        assert _lie(nomme) == "LOI-1992-007"

    def test_le_nom_ecrit_plus_loin_contredit_le_premier_extrait(self):
        """Le numero seul donnait la loi de finances, lue en premier : signalee."""
        texte = "L'article 37 fixe le préavis. Voir l'article 37 du Code du travail."

        sans_nom, nomme = lire_les_citations(texte, [_finances_37(), _code_du_travail_37()])

        assert sans_nom.extrait is None and sans_nom.raison == "article_non_consulte"
        assert _lie(nomme) == "LOI-1992-007"


class TestNomCoupeParUneVirguleOuUnePreposition:
    """
    Revue du 2026-10-29, tour 3 : le nom s'arretait a la virgule ou a une
    preposition de _SUITES (« contre », « pour », « for »…), et ce qui en
    restait designait un AUTRE texte lu. La suite du nom est exigee de
    l'intitule qui continue le nom ; sinon, signalee.
    """

    TEXTE_CHR = (
        "Selon l'article 10 du décret portant création, organisation et fonctionnement des "
        "Centres Hospitaliers Régionaux, le directeur est nommé par décret."
    )

    def test_le_decret_des_centres_hospitaliers_et_non_celui_des_sports(self):
        extraits = [
            _extrait(1, 2035, "10", INFRASTRUCTURES_SPORTIVES, "PRC-9354", jour=date(2022, 8, 17)),
            _extrait(2, 1905, "10", CHR, "PRC-9185", jour=date(2022, 1, 11)),
        ]

        [occurrence] = lire_les_citations(self.TEXTE_CHR, extraits)

        assert _lie(occurrence) == "PRC-9185"

    @pytest.mark.parametrize("titre, reference", [
        (INFRASTRUCTURES_SPORTIVES, "PRC-9354"),
        (FACULTE_DE_GAROUA, "PRC-6256"),
    ])
    def test_un_autre_decret_portant_creation_n_est_pas_lu(self, titre, reference):
        [occurrence] = lire_les_citations(self.TEXTE_CHR, [_extrait(1, 7, "10", titre, reference)])

        assert occurrence.extrait is None and occurrence.raison == "texte_non_consulte"

    @pytest.mark.parametrize("objet, lie", [
        ("contre le blanchiment des capitaux", "texte_non_consulte"),
        ("contre la corruption", "texte_non_consulte"),
        ("contre la traite des personnes", "LOI-2011-024"),
    ])
    def test_la_loi_relative_a_la_lutte(self, objet, lie):
        texte = f"Voir l'article 5 de la loi relative à la lutte {objet}."

        [occurrence] = lire_les_citations(
            texte, [_extrait(5, 2240, "5", TRAITE_DES_PERSONNES, "LOI-2011-024")]
        )

        assert _lie(occurrence) == lie
        assert _cite(texte, occurrence) == f"article 5 de la loi relative à la lutte {objet}"

    @pytest.mark.parametrize("texte", [
        "Voir l'article 37 de la loi contre la corruption.",
        "Voir l'article 37 de la loi pour la protection du consommateur.",
    ])
    def test_une_nature_coupee_n_est_pas_une_nature_seule(self, texte):
        """« de la loi | contre la corruption » allait au Code du travail, une loi lue."""
        [occurrence] = lire_les_citations(texte, [_code_du_travail_37()])

        assert occurrence.extrait is None and occurrence.raison == "texte_non_consulte"

    def test_en_anglais_avec_les_tableaux(self, tableaux_budgetaires):
        texte = "Article 76 of the Law for the Protection of Consumers provides damages."
        consommateur = _extrait(76, 2261, "76", CONSOMMATEUR, "LOI-2011-012", jour=date(2011, 5, 6))

        [non_lu] = lire_les_citations(texte, tableaux_budgetaires)
        [lu] = lire_les_citations(texte, [*tableaux_budgetaires, consommateur])

        assert non_lu.extrait is None and non_lu.raison == "texte_non_consulte"
        assert _lie(lu) == "LOI-2011-012"

    def test_la_loi_d_orientation_d_une_autre_matiere(self):
        [occurrence] = lire_les_citations(
            "Voir l'article 3 de la loi d'orientation pour l'enseignement supérieur.",
            [_extrait(3, 2249, "3", ORIENTATION_DE_L_EDUCATION, "LOI-1998-004",
                      jour=date(1998, 4, 14))],
        )

        assert occurrence.extrait is None and occurrence.raison == "texte_non_consulte"

    @pytest.mark.parametrize("texte, lus, lie", [
        # L'intitule s'arrete au nom : « pour les salariés » est la phrase.
        ("L'article 37 du Code du travail pour les salariés du privé fixe le préavis.",
         [_finances_37(), _code_du_travail_37()], "LOI-1992-007"),
        # Apres la virgule, un mot seul ou un determinant ouvre la phrase.
        ("Selon l'article 5 du décret portant Statut général de la fonction publique, oui.",
         [_extrait(5, 2279, "5", STATUT_GENERAL, "DEC-1994-199", jour=date(1994, 10, 7))],
         "DEC-1994-199"),
        ("Selon l'article 5 du décret portant Statut général de la fonction publique, le "
         "fonctionnaire est noté.",
         [_extrait(5, 2279, "5", STATUT_GENERAL, "DEC-1994-199", jour=date(1994, 10, 7))],
         "DEC-1994-199"),
        ("Voir l'article 75 de la Loi N°92/007. Selon l'article 76 de la loi, l'employeur doit "
         "des dommages-intérêts.",
         [_extrait(75, 2274, "75", CODE_DU_TRAVAIL, "LOI-1992-007"),
          _extrait(76, 2274, "76", CODE_DU_TRAVAIL, "LOI-1992-007")], "LOI-1992-007"),
    ])
    def test_la_suite_de_la_phrase_n_est_pas_un_complement(self, texte, lus, lie):
        occurrence = lire_les_citations(texte, lus)[-1]

        assert _lie(occurrence) == lie

    # Intitules reels (titres_dev.tsv), balayage de la revue, tour final : le
    # nom coupe designait un AUTRE texte lu.
    EMPRUNT_IDA = (
        "Décret N°2018/087 du 02 février 2018 habilitant le Ministre de l'Economie, de la "
        "Planification et de l'Aménagement du Territoire à signer un Accord de prêt avec "
        "l'Association Internationale de Développement (IDA)."
    )
    SERVICES_DU_PREMIER_MINISTRE = (
        "Décret portant nomination des responsables au Secrétariat Général des Services du "
        "Premier Ministre"
    )
    EIFORCES = (
        "Arrêté N° 0827/CAB/PR du 9 octobre 2020 portant nomination de responsables a l'Ecole "
        "Internationale des Forces de Sécurité (EIFORCES)"
    )
    MINISTERE_DE_LA_DEFENSE = (
        "Arrêté N° 0126/CAB/PR du 25 février 2021 portant nomination de Responsables au "
        "Ministère de la Défense"
    )

    @pytest.mark.parametrize("texte, titre, reference", [
        # Une virgule, puis une preposition et un nom : la suite du nom.
        ("Voir l'article 76 du Décret habilitant à signer un Accord de Prêt, pour le financement "
         "du Programme Batir les Ecoles.", EMPRUNT_IDA, "PRC-6273"),
        # « à » et « au » ne distinguent rien : ce sont les noms qui suivent.
        ("Voir l'article 76 du Décret portant nomination de responsables à l'Ecole "
         "Internationale des Forces de Sécurité.", SERVICES_DU_PREMIER_MINISTRE, "PRC-9416"),
        # Le « à » sans accent d'un intitule, que _SUITES lisait comme un verbe.
        ("Voir l'article 76 de l'Arrêté portant nomination de responsables a l'Ecole "
         "Internationale des Forces de Sécurité.", MINISTERE_DE_LA_DEFENSE, "PRC-8805"),
    ])
    def test_le_nom_coupe_ne_designe_pas_un_autre_texte_lu(self, texte, titre, reference):
        [occurrence] = lire_les_citations(texte, [_extrait(76, 7, "76", titre, reference)])

        assert occurrence.extrait is None and occurrence.raison == "texte_non_consulte"

    def test_le_nom_coupe_designe_son_texte_lu(self):
        texte = (
            "Voir l'article 2 de l'Arrêté portant nomination de responsables a l'Ecole "
            "Internationale des Forces de Sécurité."
        )

        [occurrence] = lire_les_citations(texte, [
            _extrait(1, 7, "2", self.MINISTERE_DE_LA_DEFENSE, "PRC-8805"),
            _extrait(2, 8, "2", self.EIFORCES, "PRC-8589"),
        ])

        assert _lie(occurrence) == "PRC-8589"


class TestPlusieursTextesDuMemeNom:
    """
    Plusieurs textes lus portent le nom ET l'article, rien ne dit lequel : le
    premier dans le contexte etait pris. Signalee ; un antecedent ou une
    annee departage, comme avant.
    """

    @pytest.fixture
    def codes_miniers(self):
        return [
            _extrait(2023, 9866, "5", "Loi N°2023/014 du 19 décembre 2023 portant Code Minier",
                     "PRC-9866", jour=date(2023, 12, 19)),
            _extrait(2016, 5621, "5", "Loi N°2016/017 du 14 décembre 2016 portant code minier du "
                     "Cameroun", "PRC-5621", jour=date(2016, 12, 14)),
            _extrait(2004, 5621, "4", "Loi N°2016/017 du 14 décembre 2016 portant code minier du "
                     "Cameroun", "PRC-5621", jour=date(2016, 12, 14)),
        ]

    def test_rien_ne_dit_lequel(self, codes_miniers):
        [occurrence] = lire_les_citations("Voir l'article 5 du Code minier.", codes_miniers)

        assert occurrence.extrait is None and occurrence.raison == "article_non_consulte"

    def test_la_citation_precedente_departage(self, codes_miniers):
        texte = "Voir l'article 4 du Code minier de 2016 ; l'article 5 du Code minier fixe la redevance."

        quatre, cinq = lire_les_citations(texte, codes_miniers)

        assert quatre.extrait.article_id == 2004 and cinq.extrait.article_id == 2016

    def test_un_seul_porte_l_article(self, codes_miniers):
        [occurrence] = lire_les_citations("Voir l'article 4 du Code minier.", codes_miniers)

        assert occurrence.extrait.article_id == 2004


class TestAnneeDExercice:
    """
    Revue du 2026-10-29, tour 3 : « la loi de finances … pour l'exercice
    2022 » allait a la loi de finances pour l'exercice 2023, votee en 2022.
    """

    @pytest.mark.parametrize("texte", [
        "Voir l'article 5 de la loi de finances 2022.",
        "Voir l'article 5 de la loi de finances pour l'exercice 2022.",
    ])
    def test_l_exercice_et_non_l_annee_du_vote(self, texte):
        exercice_2023 = _extrait(2023, 2232, "5", LF_2023, "PRC-9449", jour=date(2022, 12, 27))
        exercice_2022 = _extrait(2022, 2212, "5", LF_2022, "PRC-9119", jour=date(2021, 12, 16))

        [seule_2023] = lire_les_citations(texte, [exercice_2023])
        [les_deux] = lire_les_citations(texte, [exercice_2023, exercice_2022])

        assert seule_2023.extrait is None and seule_2023.raison == "texte_non_consulte"
        assert les_deux.extrait.article_id == 2022

    def test_la_date_ecrite_reste_celle_de_l_acte(self):
        [occurrence] = lire_les_citations(
            "Voir l'article 5 de la loi du 27 décembre 2022.",
            [_extrait(5, 2232, "5", LF_2023, "PRC-9449", jour=date(2022, 12, 27))],
        )

        assert occurrence.extrait.article_id == 5


class TestLoiDeReglement:
    """« règlement » n'est pas une nature : c'est le mot qui distingue une loi de reglement."""

    TEXTE = "Voir l'article 76 de la loi portant loi de règlement de la République du Cameroun."

    def test_pas_la_loi_de_finances(self, tableaux_budgetaires):
        [occurrence] = lire_les_citations(self.TEXTE, tableaux_budgetaires)

        assert occurrence.extrait is None and occurrence.raison == "texte_non_consulte"

    def test_la_loi_de_reglement_lue(self, tableaux_budgetaires):
        reglement = _extrait(76, 2155, "76", LOI_DE_REGLEMENT_2021, "PRC-9445",
                             jour=date(2022, 12, 27))

        [occurrence] = lire_les_citations(self.TEXTE, [*tableaux_budgetaires, reglement])

        assert _lie(occurrence) == "PRC-9445"


@pytest.mark.parametrize("texte, titre, reference", [
    ("Article 3 of the Law on the Fight against Trafficking in Persons provides penalties.",
     TRAITE_DES_PERSONNES, "LOI-2011-024"),
    ("Article 3 of the Anti-Trafficking Law provides penalties.", TRAITE_DES_PERSONNES,
     "LOI-2011-024"),
    ("Article 3 of the Uniform Act Organising Collective Proceedings for the Discharge of Debts "
     "provides that.", AU_PROCEDURES_COLLECTIVES, "OHADA-AUPCAP-2015"),
    ("Article 3 of the Treaty on the Harmonisation of Business Law in Africa provides that.",
     TRAITE_1993, "OHADA-TRAITE-1993"),
])
def test_noms_anglais_du_tour_final(texte, titre, reference):
    """Revue du tour 3 : signales, le texte lu (sonde_anglais, 8 fausses alertes sur 31)."""
    extraits = [
        _extrait(1, 2232, "3", LF_2023, "PRC-9449", jour=date(2022, 12, 27)),
        _extrait(3, 3, "3", titre, reference),
    ]

    [occurrence] = lire_les_citations(texte, extraits)

    assert _lie(occurrence) == reference


def test_une_nature_seule_dont_aucun_texte_n_est_lu(tableaux_budgetaires):
    """« du code », aucun code lu : le texte manque, jamais le morceau 76.37."""
    [occurrence] = lire_les_citations(
        "En cas de licenciement abusif, l'article 76 du code prévoit des dommages-intérêts.",
        tableaux_budgetaires,
    )

    assert occurrence.extrait is None and occurrence.raison == "texte_non_consulte"


_MOTS_SANS_PORTEE = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi".split()


# Lignes reelles de jurix_dev (revue du 2026-10-29, tour final) : le Code du
# travail lu sans son article 76, le Code pénal avec ses articles 74 a 76. Le
# contenu des morceaux ne porte pas leur en-tete (le decoupage le range dans le
# numero).
def _hors_tableaux():
    return [
        _extrait(97420, 2274, "35.1", CODE_DU_TRAVAIL, "LOI-1992-007",
                 "(1) Pendant la durée du préavis, l'employeur et le travailleur sont tenus…"),
        _extrait(97421, 2274, "35.2", CODE_DU_TRAVAIL, "LOI-1992-007",
                 "(6) Le salaire à prendre en considération… Article 40.- (1) Le contrat…"),
        _extrait(104968, 4722, "74", CODE_PENAL, "PRC-4722",
                 "Peine et responsabilité (1) Aucune peine ne peut être prononcée…",
                 jour=date(2016, 7, 12)),
        _extrait(104970, 4722, "75", CODE_PENAL, "PRC-4722",
                 "Ignorance et mobile L'ignorance de la loi et le mobile n'influent pas…",
                 jour=date(2016, 7, 12)),
        _extrait(104971, 4722, "76", CODE_PENAL, "PRC-4722",
                 "Exécution de la loi Ne constitue aucune infraction, le fait ordonné ou "
                 "autorisé par la loi…", jour=date(2016, 7, 12)),
    ]


class TestHorsDesTableaux:
    """
    Revue du 2026-10-29, tour final : une question de droit du travail, le Code
    du travail lu sans son article 76, le Code pénal lu avec le sien. Ces
    citations, qui ne designent pas leur texte, allaient a l'article 76 du
    Code pénal (« Exécution de la loi ») : le seul texte lu a porter ce numero.
    Le modele cite de memoire celui du Code du travail.
    """

    @pytest.mark.parametrize("texte", [
        "En cas de licenciement abusif, l'article 76 de cette loi prévoit des dommages-intérêts.",
        "La loi camerounaise protège le salarié : l'article 76 de cette loi prévoit des "
        "dommages-intérêts.",
        "Under Cameroonian labour law, Article 76 of this Law provides for damages.",
        "L'article 76 prévoit des dommages-intérêts.",
        "Le Code pénal ne traite pas du licenciement. L'article 76 prévoit des dommages-intérêts.",
        "La Loi N° 2016/007 portant Code pénal ne traite pas du licenciement. L'article 76 "
        "prévoit des dommages-intérêts.",
        "En cas de licenciement abusif, l'article 76 du code prévoit des dommages-intérêts.",
        "En cas de licenciement abusif, l'article 76 du Code prévoit des dommages-intérêts.",
        "En cas de licenciement abusif, l'article 76 du présent code prévoit des dommages-intérêts.",
        "En cas de licenciement abusif, l'article 76 dudit code prévoit des dommages-intérêts.",
        "Under the Code, Article 76 provides for damages in case of unfair dismissal.",
        "In case of unfair dismissal, Article 76 of the Code provides for damages.",
        "En droit du travail, en cas de licenciement abusif, l'article 76 du code prévoit des "
        "dommages-intérêts.",
        "Le Code du travail protège le salarié. En cas de licenciement abusif, l'article 76 du "
        "code prévoit des dommages-intérêts.",
    ])
    def test_jamais_l_article_76_du_code_penal(self, texte):
        occurrences = [o for o in lire_les_citations(texte, _hors_tableaux()) if o.forme == "article"]

        assert occurrences and all(o.extrait is None for o in occurrences)
        assert {o.raison for o in occurrences} <= {"article_non_consulte", "texte_non_consulte"}

    def test_le_renvoi_signale_n_est_pas_l_antecedent_de_tous_les_textes(self):
        """
        « of this Law », sans antecedent, designait toutes les lois lues : la
        citation sans nom qui suit les reprenait, et allait a l'article 77 du
        Code pénal par le numero seul. Signale, le renvoi designe un texte
        inconnu, et la suivante est signalee aussi (tour final).
        """
        extraits = [*_hors_tableaux(), _extrait(
            104972, 4722, "77", CODE_PENAL, "PRC-4722", "Légitime défense…", jour=date(2016, 7, 12),
        )]
        texte = (
            "Under Cameroonian labour law, Article 76 of this Law provides for damages. "
            "Article 77 sets the amount."
        )

        renvoi, suivante = lire_les_citations(texte, extraits)

        assert renvoi.extrait is None and renvoi.raison == "article_non_consulte"
        assert suivante.extrait is None

    @pytest.mark.parametrize("texte, attendu", [
        # Nommes, ils gardent leur lien : le Code pénal lu, le Code du travail
        # lu sans son article 76.
        ("Selon l'article 76 du Code pénal, l'exécution de la loi n'est pas une infraction.",
         104971),
        ("Selon l'article 75 du Code pénal, l'ignorance de la loi n'excuse pas ; l'article 76 du "
         "même code ajoute que l'exécution de la loi n'est pas une infraction.", 104971),
        ("Selon l'article 75 du Code pénal, l'ignorance de la loi n'excuse pas. L'article 76 "
         "ajoute que l'exécution de la loi n'est pas une infraction.", 104971),
        ("Le préavis est régi par l'article 35 du Code du travail ; l'article 76 dudit Code du "
         "travail fixe les dommages-intérêts.", None),
        ("Le préavis est régi par l'article 35 du Code du travail ; l'article 76 de cette loi fixe "
         "les dommages-intérêts.", None),
    ])
    def test_une_citation_qui_designe_son_texte(self, texte, attendu):
        derniere = [o for o in lire_les_citations(texte, _hors_tableaux()) if o.forme == "article"][-1]

        assert (derniere.extrait.article_id if derniere.extrait else None) == attendu
        if attendu is None:
            assert derniere.raison == "article_non_consulte"


class TestAntecedentNumeroteDansLaPhrase:
    """
    Un texte designe par son numero introduit la citation sans nom de la MEME
    phrase (vraies reponses d'evaluation) ; d'une phrase precedente, il ne la
    choisit plus (revue du 2026-10-29, tour final).
    """

    def test_dans_la_meme_phrase(self):
        """« D'après la **Loi N° 2016/007 … portant Code pénal**, l'article 337 précise »."""
        extraits = [
            _extrait(3371, 4722, "337", CODE_PENAL, "PRC-4722", jour=date(2016, 7, 12)),
            _extrait(91, 9313, "50", "Loi N°2022/014 relative à la procréation de la personne",
                     "PRC-9313"),
        ]
        texte = (
            "D'après la **Loi N° 2016/007 du 12 juillet 2016 portant Code pénal**, l'article 337 "
            "précise la peine."
        )

        loi, article = lire_les_citations(texte, extraits)

        assert loi.forme == "texte" and loi.extrait.law_id == 4722
        assert article.extrait.article_id == 3371

    def test_entre_parentheses_apres_le_numero(self):
        """« Consultez le **Décret N°2015/406** (article 6 et suivants) »."""
        extraits = [
            _extrait(1, 4007, "6", REMUNERATION_DES_MAIRES, "PRC-4007", jour=date(2015, 1, 1)),
            _extrait(2, 4008, "6", INDEMNITES_DES_MAIRES, "PRC-4008", jour=date(2015, 9, 16)),
        ]
        texte = "Consultez le **Décret N°2015/406** (article 6 et suivants) pour le détail."

        decret, article = lire_les_citations(texte, extraits)

        assert article.extrait.article_id == 2

    def test_dans_une_phrase_precedente(self):
        """
        La meme forme que l'incident (« La loi n° 2022/020 ne traite pas du
        licenciement. L'article 76… ») : rien ne dit que la phrase suivante
        cite ce texte, et le numero seul ne le prouve pas. Une fausse alerte
        possible, voulue.
        """
        extraits = [_extrait(3371, 4722, "337", CODE_PENAL, "PRC-4722", jour=date(2016, 7, 12))]
        texte = (
            "La Loi N° 2016/007 du 12 juillet 2016 portant Code pénal s'applique. L'article 337 "
            "précise la peine."
        )

        loi, article = lire_les_citations(texte, extraits)

        assert loi.extrait.law_id == 4722
        assert article.extrait is None and article.raison == "article_non_consulte"


class TestRenvoiDevantUnNom:
    """
    « dudit Code du travail », « du même Code du travail », « of the said Labour
    Code » nomment leur texte. Lus comme un renvoi de nature (« code ») sans
    antecedent, ils allaient au seul code lu qui porte le numero, Code du
    travail lu ou non (sondes du tour final).
    """

    @pytest.mark.parametrize("texte, cite", [
        ("Selon l'article 37 dudit Code du travail, le préavis est dû.",
         "article 37 dudit Code du travail"),
        ("Selon l'article 37 du même Code du travail, le préavis est dû.",
         "article 37 du même Code du travail"),
        ("Under Article 37 of the said Labour Code, notice is due.",
         "Article 37 of the said Labour Code"),
    ])
    def test_le_nom_apres_le_renvoi(self, texte, cite):
        penal = _extrait(4722, 4722, "37", CODE_PENAL, "PRC-4722", _commence(37),
                         jour=date(2016, 7, 12))

        [lu] = lire_les_citations(texte, [penal, _code_du_travail_37()])
        [non_lu] = lire_les_citations(texte, [penal])

        assert lu.extrait.article_id == 37 and _cite(texte, lu) == cite
        assert non_lu.extrait is None and non_lu.raison == "texte_non_consulte"

    @pytest.mark.parametrize("texte", [
        "Article 74 of the Penal Code applies, and Article 75 of the same Code too.",
        "Selon l'article 74 du Code pénal, oui ; l'article 75 du même texte non plus.",
    ])
    def test_un_renvoi_que_la_phrase_continue(self, texte):
        """
        « too », « non plus » continuent la phrase : le renvoi vise le code cite
        avant, et non un « Code too » que rien ne porte (sondes tenues a
        l'ecart et de la revue, tour final).
        """
        extraits = [
            _extrait(74, 4722, "74", CODE_PENAL, "PRC-4722", jour=date(2016, 7, 12)),
            _extrait(75, 4722, "75", CODE_PENAL, "PRC-4722", jour=date(2016, 7, 12)),
        ]

        premiere, seconde = lire_les_citations(texte, extraits)

        assert premiere.extrait.article_id == 74
        assert seconde.extrait.article_id == 75


class TestBudgetsDeLecture:
    """
    Une reponse degeneree du modele (8 192 jetons, 32 000 caracteres) peut
    ecrire des centaines de citations et de noms. La lecture tourne dans la
    boucle d'evenements de l'API : au-dela de budgets qu'aucune des 323 vraies
    reponses n'approche (10 citations, 5 noms lus au plus), une citation est
    signalee sans etre lue, jamais liee (revue du 2026-10-29, tour final).
    """

    def test_au_dela_des_citations_lues(self):
        from app.services.lecture_des_citations import _CITATIONS_LUES_AU_PLUS

        extraits = [_code_du_travail_37()]
        texte = " ".join(
            "Voir l'article 37 du Code du travail." for _ in range(_CITATIONS_LUES_AU_PLUS + 3)
        )

        occurrences = lire_les_citations(texte, extraits)

        assert len(occurrences) == _CITATIONS_LUES_AU_PLUS + 3
        assert all(o.extrait.article_id == 37 for o in occurrences[:_CITATIONS_LUES_AU_PLUS])
        assert all(
            o.extrait is None and o.raison == "article_non_consulte"
            for o in occurrences[_CITATIONS_LUES_AU_PLUS:]
        )
        assert _cite(texte, occurrences[-1]) == "article 37"

    def test_au_dela_des_noms_lus(self):
        """
        Chaque nom different se lit a part ; au-dela du budget, il est celui
        d'un texte non consulte. Le meme nom repete ne coute qu'une lecture.
        """
        from app.services.lecture_des_citations import _NOMS_LUS_AU_PLUS

        extraits = [_code_du_travail_37()]
        inconnus = " ".join(
            f"Voir l'article 37 du Code numéro {chr(65 + i // 26)}{chr(65 + i % 26)}x."
            for i in range(_NOMS_LUS_AU_PLUS)
        )
        repete = " ".join("Voir l'article 37 du Code du travail." for _ in range(50))

        occurrences = lire_les_citations(inconnus + " " + repete, extraits)

        assert [o.raison for o in occurrences[:_NOMS_LUS_AU_PLUS]] == (
            ["texte_non_consulte"] * _NOMS_LUS_AU_PLUS
        )
        # Le budget est epuise : le Code du travail, lu, n'est plus verifie.
        assert all(
            o.extrait is None and o.raison == "texte_non_consulte"
            for o in occurrences[_NOMS_LUS_AU_PLUS:]
        )
        [lu] = lire_les_citations("Voir l'article 37 du Code du travail.", extraits)
        assert lu.extrait.article_id == 37


    def test_les_noms_en_prose_relus_par_citation(self, monkeypatch):
        """
        Apres une citation nommee, la citation sans nom relit a reculons les noms
        ecrits depuis ; « la loi », « le code » ne designent rien et ne
        l'arretaient pas : 3 000 noms relus pour une boucle de 32 000
        caracteres (8 s, revue du tour final). Elle en relit au plus
        _NOMS_RELUS_AU_PLUS ; les 323 vraies reponses en ecrivent au plus 15.
        """
        from app.services import lecture_des_citations as lecture

        lus = []
        designation = lecture._NomsEnProse._designation
        monkeypatch.setattr(
            lecture._NomsEnProse, "_designation",
            lambda self, candidat: lus.append(candidat.start()) or designation(self, candidat),
        )
        texte = (
            "Voir l'article 37 du Code du travail. "
            + " ".join(f"la loi {i} et le code {i}" for i in range(200)) + ". L'article 38 dit."
        )

        lire_les_citations(texte, [_code_du_travail_37()])

        assert 0 < len(lus) <= lecture._NOMS_RELUS_AU_PLUS


class TestCoutBorne:
    """
    Revue du 2026-10-29, tour 3 : chaque citation sans nom relisait tous les
    noms ecrits avant elle, et chaque nom coutait O(W²) ; 189 s pour 4,7 k
    caracteres, dans la boucle d'evenements de l'API. Mesure du tour final :
    moins de 0,2 s pour chacun de ces textes ; la borne laisse de la marge a
    une machine chargee.
    """

    @pytest.mark.parametrize("texte", [
        " ".join("la Loi camerounaise applicable" for _ in range(80)) + ". "
        + " ".join(f"L'article {i} précise." for i in range(1, 101)),
        ("Selon l'article 5 du Code " + " ".join(_MOTS_SANS_PORTEE * 600))[:8000] + ".",
        " ".join(
            " ".join("le Code " + " ".join(_MOTS_SANS_PORTEE[:6]) for _ in range(30))
            + f" et l'article {i} précise que rien ne change." for i in range(1, 101)
        )[:32000],
        "".join(
            "Le Code prévoit que l'employeur qui rompt le contrat sans motif légitime doit des "
            f"dommages-intérêts que le juge apprécie. L'article {i} précise ce point. "
            for i in range(1, 121)
        ),
        # Revue du tour final : une boucle sans ponctuation APRES une citation
        # nommee (la citation sans nom qui suit lit les noms ecrits depuis),
        # 8 a 12 s a 32 000 caracteres ; des noms uniques ; des milliers de
        # citations. Moins de 50 ms chacun, machine chargee (tour final).
        "Voir l'article 5 du Code du travail. "
        + ("le code et la loi et le décret " * 1100)[:31800] + ". L'article 76 dit.",
        "Voir l'article 5 du Code du travail. "
        + ("- la loi\n- le décret\n- l'arrêté\n" * 1100)[:31800] + "\nL'article 76 de ce code dit.",
        "Voir l'article 5 du Code du travail. "
        + ("the Law and the Code and the Decree and " * 900)[:31800] + ". Article 76 says.",
        " ".join(
            f"Voir l'article {i} de la loi relative au {' '.join(_MOTS_SANS_PORTEE[i % 7:])} "
            f"numéro {i}." for i in range(1, 500)
        )[:32000],
        " ".join(f"L'article {i} dit." for i in range(1, 1501)),
        " ".join("art. 5 du CT. Art. 5 dit." for _ in range(1000)),
    ], ids=["noms-imprecis-puis-citations", "nom-de-8000-caracteres", "32k-sans-ponctuation",
            "reponse-naturelle", "boucle-apres-une-citation", "liste-markdown-puis-renvoi",
            "boucle-anglaise", "noms-uniques", "1500-citations-sans-nom", "2000-citations"])
    def test_une_reponse_adverse_se_lit_vite(self, tableaux_budgetaires, texte):
        import time

        # 22 extraits, dont 17 longs : le contexte le plus lourd que le chat lit.
        extraits = [*tableaux_budgetaires, *(
            _extrait(1000 + i, 3000 + i, str(5 + i), f"Décret N°2020/{100 + i} portant objet {i}",
                     f"PRC-{3000 + i}", "Tableau des crédits. " * 200)
            for i in range(19)
        )]

        debut = time.perf_counter()
        lire_les_citations(texte, extraits)

        # La borne laisse de la marge a une machine chargee : la production
        # (d99e23c) prenait jusqu'a 11 s sur ces textes.
        assert time.perf_counter() - debut < 1
