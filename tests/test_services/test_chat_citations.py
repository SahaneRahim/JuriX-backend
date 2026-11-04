"""
Le chat rend ses citations cliquables, signale les introuvables, separe les
articles cites des textes seulement consultes, et dit ou il en est avant le
premier mot.

Ce que le front recoit, et sur quoi il s'appuie :
- les evenements de progression (recherche, textes trouves, redaction), avant
  tout morceau de texte, et ignores par le front deja deploye ;
- dans le dernier evenement (et dans la reponse de /ask) : `citations`,
  `citations_introuvables`, `articles_cites`, `textes_consultes_non_cites`,
  avec des indices EXACTS dans le texte qu'il a affiche.

Aucun appel reel : le modele, la recherche et la base sont doubles.

Author: JuriX Team
"""

import json
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.schemas.rag import RAGRequest
from app.schemas.search import ChunkResult
from app.services.intent_classifier import IntentResult
from app.services.rag_service import RAGService

CODE_DU_TRAVAIL = "Loi n° 92/007 du 14 août 1992 portant Code du travail"


def _extrait(article_id, law_id, numero, titre, reference, contenu="Contenu de l'article.",
             pertinence=0.8):
    return ChunkResult(
        article_id=article_id, law_id=law_id, number=numero, content=contenu,
        reference=reference, law_title=titre, publication_date=date(1992, 8, 14),
        relevance_score=pertinence,
    )


@pytest.fixture
def extraits():
    return [
        _extrait(371, 3, "37", CODE_DU_TRAVAIL, "PRC-3",
                 "(1) En cas de rupture du contrat, le travailleur a droit à une indemnité."),
        _extrait(401, 3, "40", CODE_DU_TRAVAIL, "PRC-3", "Les délégués du personnel sont réunis."),
        _extrait(81, 8, "8", "Décret N°2018/9387/CAB/PM fixant les modalités", "PRC-8"),
    ]


@pytest.fixture
def base():
    session = AsyncMock()
    session.execute = AsyncMock(return_value=MagicMock(
        scalar_one_or_none=MagicMock(return_value=None)
    ))
    session.commit = AsyncMock()
    session.flush = AsyncMock()
    session.rollback = AsyncMock()
    session.add = MagicMock()
    # Un point de sauvegarde, comme AsyncSession.begin_nested : appel
    # synchrone, rendu utilisable par `async with`.
    session.begin_nested = MagicMock(return_value=MagicMock())
    return session


@pytest.fixture(autouse=True)
def routage_juridique():
    with patch(
        "app.services.rag_service.classify_intent",
        new=AsyncMock(return_value=IntentResult("juridique", 1.0, "test")),
    ) as double:
        yield double


def _service(base, extraits, morceaux=(), fin="STOP"):
    service = RAGService(base)
    service.llm = MagicMock()
    service.search_service = MagicMock()
    service.search_service.search = AsyncMock(
        return_value=MagicMock(results=[], chunks=list(extraits), search_time_ms=1)
    )

    def _flux(**kwargs):
        async def _generateur():
            for morceau in morceaux:
                yield morceau
            if kwargs.get("fin"):
                kwargs["fin"](fin)
        return _generateur()

    service.llm.generate_stream = MagicMock(side_effect=_flux)
    return service


def _requete(**champs):
    return RAGRequest(**{
        "question": "Quelles indemnités en cas de licenciement ?", "persona": "citoyen",
        "language": "fr", **champs,
    })


async def _evenements(service, requete):
    return [json.loads(e) async for e in service.ask_stream(requete)]


def _slice_js(texte, debut, fin):
    """`texte.slice(debut, fin)` en JavaScript : des unites UTF-16."""
    return texte.encode("utf-16-le")[2 * debut:2 * fin].decode("utf-16-le")


# ==================== PROGRESSION ====================


class TestProgression:

    @pytest.mark.asyncio
    async def test_trois_etapes_avant_le_premier_mot(self, base, extraits):
        service = _service(base, extraits, ["Selon l'article 37 ", "du Code du travail, oui."])

        evenements = await _evenements(service, _requete())

        etapes = [e["progression"]["etape"] for e in evenements if "progression" in e]
        assert etapes == ["recherche", "textes_trouves", "redaction"]
        premier_texte = next(i for i, e in enumerate(evenements) if e["chunk"])
        assert all("progression" in e for e in evenements[:premier_texte])

    @pytest.mark.asyncio
    async def test_le_front_deploye_n_y_voit_ni_texte_ni_fin_ni_erreur(self, base, extraits):
        """Il teste `chunk`, `done` et `error` : rien de cela dans une progression."""
        service = _service(base, extraits, ["Réponse."])

        evenements = await _evenements(service, _requete())

        for evenement in (e for e in evenements if "progression" in e):
            assert evenement["chunk"] == "" and evenement["done"] is False
            assert "error" not in evenement and "sources" not in evenement

    @pytest.mark.asyncio
    async def test_les_textes_trouves_sont_comptes_et_nommes(self, base, extraits):
        service = _service(base, extraits, ["Réponse."])

        evenements = await _evenements(service, _requete())

        trouves = next(e["progression"] for e in evenements
                       if e.get("progression", {}).get("etape") == "textes_trouves")
        assert trouves["nb_articles"] == 3 and trouves["nb_textes"] == 2
        assert trouves["libelle"] == "3 articles trouvés dans 2 textes"
        assert [(t["law_id"], t["nb_articles"]) for t in trouves["textes"]] == [(3, 2), (8, 1)]
        assert trouves["textes"][0]["law_title"] == CODE_DU_TRAVAIL

    @pytest.mark.asyncio
    async def test_la_progression_donne_l_identifiant_de_la_conversation(self, base, extraits):
        """Une reponse ARRETEE n'a pas de dernier evenement : sans cela, la
        question suivante ouvrait une autre conversation."""
        service = _service(base, extraits, ["Réponse."])

        evenements = await _evenements(service, _requete())

        progressions = [e for e in evenements if "progression" in e]
        assert evenements[-1]["session_id"]
        assert {e["session_id"] for e in progressions} == {evenements[-1]["session_id"]}
        assert {e["persona"] for e in progressions} == {"citoyen"}

    @pytest.mark.asyncio
    async def test_libelles_en_anglais(self, base, extraits):
        service = _service(base, extraits, ["Answer."])

        evenements = await _evenements(service, _requete(language="en"))

        assert [e["progression"]["libelle"] for e in evenements if "progression" in e] == [
            "Searching the texts…", "3 articles found in 2 texts", "Writing the answer…",
        ]

    @pytest.mark.asyncio
    async def test_des_articles_et_non_des_extraits(self, base):
        """
        Les morceaux d'un article long (« 55.1 », « 55.2 ») font un article, et
        la ligne niveau loi d'un texte n'en est pas un.
        """
        electoral = "Loi N°2012/001 portant Code électoral"
        extraits = [
            _extrait(551, 5, "55.1", electoral, "PRC-5"),
            _extrait(552, 5, "55.2", electoral, "PRC-5"),
            _extrait(None, 8, None, "Décret N°2018/9387/CAB/PM fixant les modalités", "PRC-8"),
        ]
        service = _service(base, extraits, ["Réponse."])

        evenements = await _evenements(service, _requete())

        trouves = next(e["progression"] for e in evenements
                       if e.get("progression", {}).get("etape") == "textes_trouves")
        assert trouves["libelle"] == "1 article trouvé dans 2 textes"
        assert [(t["law_id"], t["nb_articles"]) for t in trouves["textes"]] == [(5, 1), (8, 0)]

    @pytest.mark.asyncio
    async def test_des_textes_sans_article(self, base):
        extraits = [_extrait(None, 8, None, "Décret N°2018/9387/CAB/PM fixant les modalités", "PRC-8")]
        service = _service(base, extraits, ["Réponse."])

        evenements = await _evenements(service, _requete())

        trouves = next(e["progression"] for e in evenements
                       if e.get("progression", {}).get("etape") == "textes_trouves")
        assert trouves["libelle"] == "1 texte trouvé" and trouves["nb_articles"] == 0

    @pytest.mark.asyncio
    async def test_sans_resultat_ni_redaction_ni_appel_au_modele(self, base):
        service = _service(base, [])

        evenements = await _evenements(service, _requete())

        etapes = [e["progression"] for e in evenements if "progression" in e]
        assert [p["etape"] for p in etapes] == ["recherche", "textes_trouves"]
        assert etapes[1]["nb_articles"] == 0 and etapes[1]["libelle"] == "Aucun texte trouvé"
        service.llm.generate_stream.assert_not_called()
        assert evenements[-1]["done"] is True and evenements[-1]["citations"] == []

    @pytest.mark.asyncio
    async def test_une_reponse_conversationnelle_n_a_pas_de_progression(
        self, base, extraits, routage_juridique
    ):
        routage_juridique.return_value = IntentResult("smalltalk", 0.9, "test")
        service = _service(base, extraits)
        service.llm.generate = AsyncMock(return_value={"response": "Bonjour !"})

        evenements = await _evenements(service, _requete())

        assert len(evenements) == 1 and "progression" not in evenements[0]
        assert evenements[0]["citations"] == [] and evenements[0]["citations_introuvables"] == []


# ==================== CITATIONS DU DERNIER EVENEMENT ====================


class TestCitationsDuFlux:

    @pytest.mark.asyncio
    async def test_les_indices_designent_la_citation_dans_le_texte_affiche(self, base, extraits):
        """
        Le texte affiche est la concatenation des `chunk` recus, mention de fin
        comprise ; une citation a cheval sur deux morceaux reste entiere.
        """
        service = _service(
            base, extraits,
            ["📌 Vous avez droit à une indemnité (l'article 37 du Code ",
             "du travail) ; l'employeur réunit les délégués (l'article 40 du Code du travail)."],
            fin="MAX_TOKENS",
        )

        evenements = await _evenements(service, _requete())

        texte = "".join(e["chunk"] for e in evenements)
        fin = evenements[-1]
        assert [c["texte_cite"] for c in fin["citations"]] == [
            "article 37 du Code du travail", "article 40 du Code du travail",
        ]
        for citation in fin["citations"]:
            assert _slice_js(texte, citation["debut"], citation["fin"]) == citation["texte_cite"]
        assert fin["citations"][0]["article"] == "37"
        assert fin["citations"][0]["article_id"] == 371
        assert fin["citations"][0]["titre"] == CODE_DU_TRAVAIL
        assert fin["citations"][0]["date_du_texte"] == "1992-08-14"
        # Aucun lien prouve : un statut vide, ENTIER (etat null, pas absent).
        assert fin["citations"][0]["statut"] == {"etat": None, "liens": [], "lignes": []}

    @pytest.mark.asyncio
    async def test_une_citation_introuvable_est_signalee(self, base, extraits):
        service = _service(base, extraits, [
            "L'article 37 du Code du travail fixe l'indemnité. L'article 312 du Code pénal "
            "punit, et le barème est fixé par l'arrêté n° 016/MTPS/SG/CJ du 26 mai 1993.",
        ])

        evenements = await _evenements(service, _requete())

        texte = "".join(e["chunk"] for e in evenements)
        introuvables = evenements[-1]["citations_introuvables"]
        assert [(c["texte_cite"], c["raison"]) for c in introuvables] == [
            ("article 312 du Code pénal", "texte_non_consulte"),
            ("arrêté n° 016/MTPS/SG/CJ du 26 mai 1993", "texte_non_consulte"),
        ]
        for citation in introuvables:
            assert _slice_js(texte, citation["debut"], citation["fin"]) == citation["texte_cite"]
        # `sources` garde son contenu d'avant : la seule citation rattachee.
        assert [s["article_number"] for s in evenements[-1]["sources"]] == ["37"]

    @pytest.mark.asyncio
    async def test_articles_cites_et_textes_consultes_non_cites(self, base, extraits):
        service = _service(base, extraits, [
            "L'article 40 du Code du travail impose une réunion ; l'article 37 du Code du "
            "travail fixe l'indemnité, et l'article 40 du Code du travail le rappelle.",
        ])

        evenements = await _evenements(service, _requete())

        fin = evenements[-1]
        assert [(c["law_id"], c["article_number"]) for c in fin["articles_cites"]] == [
            (3, "40"), (3, "37"),
        ]
        assert fin["articles_cites"][0]["law_title"] == CODE_DU_TRAVAIL
        assert [t["law_id"] for t in fin["textes_consultes_non_cites"]] == [8]
        assert fin["textes_consultes_non_cites"][0]["articles"] == ["8"]
        assert len(fin["citations"]) == 3

    @pytest.mark.asyncio
    async def test_l_incident_du_29_octobre(self, base):
        """
        Production, « Quelles indemnités en cas de licenciement abusif ? » : huit
        tableaux de lois de finances lus, et « l'article 76 du Code du travail »
        cite de memoire. Le chat le liait au morceau « 76.37 » de la loi de
        finances 2023, rangee sous « Articles cités (1) ». Les trois citations
        du Code du travail sont signalees ; les lois de finances restent
        « consultées, non citées ».
        """
        tableau = "Les montants des autorisations d'engagement et des crédits de paiement…"
        lf_2023 = ("Loi N°2022/020 du 27 décembre 2022 portant loi de finances de la république "
                   "du Cameroun pour l'exercice 2023")
        extraits = [
            _extrait(105698, 2237, "95.95", "Loi N°2025/012 du 17 décembre 2025 portant loi des "
                     "Finances de la République du Cameroun pour l’exercice 2026", "PRC-10557",
                     tableau),
            _extrait(85057, 2232, "76.37", lf_2023, "PRC-9449", tableau),
            _extrait(74786, 2137, "78.47", "Ordonnance N°2025/001 du 11 juillet 2025 modifiant et "
                     "complétant certaines dispositions de la loi n°2024/013 du 23 décembre 2024 "
                     "portant loi de finances", "PRC-10434", tableau),
        ]
        service = _service(base, extraits, [
            "Les documents fournis ne contiennent pas de disposition sur ce point.\n\n"
            "- l'indemnité de préavis est prévue par l'article 74 du Code du travail ;\n"
            "- l'indemnité de licenciement, par l'article 75 du Code du travail ;\n"
            "- les dommages-intérêts, par l'article 76 du Code du travail.",
        ])

        evenements = await _evenements(service, _requete(
            question="Quelles indemnités en cas de licenciement abusif ?",
        ))

        fin = evenements[-1]
        assert fin["articles_cites"] == [] and fin["citations"] == []
        assert [(c["texte_cite"], c["raison"]) for c in fin["citations_introuvables"]] == [
            ("article 74 du Code du travail", "texte_non_consulte"),
            ("article 75 du Code du travail", "texte_non_consulte"),
            ("article 76 du Code du travail", "texte_non_consulte"),
        ]
        assert [t["law_id"] for t in fin["textes_consultes_non_cites"]] == [2237, 2232, 2137]
        # `sources`, que lisait l'ancien front : ni la loi de finances, ni une
        # source de repli sous une reponse qui dit n'avoir rien trouve.
        assert fin["sources"] == []

    @pytest.mark.asyncio
    async def test_une_citation_sans_nom_reste_dans_le_texte_cite_avant(self, base):
        """
        Revue du 2026-10-29, panne de l'index simulee : la loi de finances 2024
        venait en tete du contexte, et « L'article 35 dispose… », qui suit une
        citation du Code du travail, allait a son morceau 35.19. Elle est lue
        dans le Code du travail ; la loi de finances reste consultee, non citee.
        """
        lf_2024 = ("Loi N°2023/019 du 19 décembre 2023 portant loi de finances de la République "
                   "du Cameroun pour l'exercice 2024")
        extraits = [
            _extrait(1, 9875, "35.19", lf_2024, "PRC-9875", "Tableau des crédits…"),
            _extrait(2, 2274, "35.1", CODE_DU_TRAVAIL, "LOI-1992-007", "(1) Le préavis court…"),
            _extrait(3, 2274, "37", CODE_DU_TRAVAIL, "LOI-1992-007", "(1) L'indemnité est due…"),
        ]
        service = _service(base, extraits, [
            "Selon l'article 37 du Code du travail, l'indemnité est due. L'article 35 dispose "
            "que le préavis court dès la notification. L'article 76 prévoit les dommages-intérêts.",
        ])

        evenements = await _evenements(service, _requete())

        fin = evenements[-1]
        assert [(c["law_id"], c["article_number"]) for c in fin["articles_cites"]] == [
            (2274, "37"), (2274, "35"),
        ]
        assert [(c["texte_cite"], c["raison"]) for c in fin["citations_introuvables"]] == [
            ("article 76", "article_non_consulte"),
        ]
        assert [t["law_id"] for t in fin["textes_consultes_non_cites"]] == [9875]

    @pytest.mark.asyncio
    async def test_une_reponse_anglaise_nomme_la_loi_par_son_numero(self, base, extraits):
        """
        Le prompt anglais demande « Article X of Law No. … » : le nom etait coupe
        au point de « No. », et la citation, juste, signalee.
        """
        extraits[0] = _extrait(371, 3, "37", CODE_DU_TRAVAIL, "PRC-3",
                               "(1) En cas de rupture du contrat, le travailleur a droit à une indemnité.")
        service = _service(base, extraits, [
            "Article 37 of Law No. 92/007 of 14 August 1992 entitles the worker to compensation.",
        ])

        evenements = await _evenements(service, _requete(
            question="What compensation is due on dismissal?", language="en",
        ))

        fin = evenements[-1]
        assert [c["texte_cite"] for c in fin["citations"]] == [
            "Article 37 of Law No. 92/007 of 14 August 1992",
        ]
        assert fin["citations"][0]["article_id"] == 371
        assert fin["citations_introuvables"] == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("reponse, cite, raison", [
        # Une nature seule, apres le Code du travail : ce Code, non consulte.
        ("Voir l'article 74 du Code du travail. Les dommages relèvent de l'article 76 de la loi.",
         "article 76 de la loi", "texte_non_consulte"),
        # Le texte nomme apres coup, dans la meme phrase. Rien ne designe le
        # texte de la citation : signalee sans lire ce nom (_sans_antecedent),
        # la raison seule change, et le front ne l'affiche pas.
        ("En droit du travail, l'article 76 couvre les dommages (Code du travail).", "article 76",
         "article_non_consulte"),
        # Un qualificatif entre le numero et le nom.
        ("Voir l'article 76 in fine du Code du travail.", "article 76 in fine du Code du travail",
         "texte_non_consulte"),
        # Revue du tour 3 : sans formule d'aveu, l'extrait pertinent, la
        # source de repli sortait le tableau 76.37 sous la citation signalee.
        ("En cas de licenciement abusif, l'article 76 du Code du travail prévoit des "
         "dommages-intérêts.", "article 76 du Code du travail", "texte_non_consulte"),
        # Tour 3 : le nom en minuscules, le meme article nomme plus loin,
        # un nom coupe par une preposition.
        ("Dans le code du travail, l'article 76 prévoit des dommages-intérêts.", "article 76",
         "article_non_consulte"),
        ("**En bref** : l'article 76 prévoit des dommages-intérêts.\n\n**En détail** : "
         "l'article 76 du Code du travail les fixe.", "article 76", "article_non_consulte"),
        ("Voir l'article 76 de la loi pour la protection des travailleurs.",
         "article 76 de la loi pour la protection des travailleurs", "texte_non_consulte"),
        # Revue du tour final, rejouee sur les lignes reelles : rien ne designe
        # le texte (ni nom, ni citation avant), et le seul texte lu qui porte
        # un article 76 est ce tableau. La production et le tour precedent y
        # liaient ces citations : l'affichage de l'incident.
        ("L'article 76 prévoit des dommages-intérêts.", "article 76", "article_non_consulte"),
        ("La loi camerounaise protège le salarié : l'article 76 de cette loi prévoit des "
         "dommages-intérêts.", "article 76 de cette loi", "article_non_consulte"),
        ("En cas de licenciement abusif, l'article 76 de la présente loi prévoit des "
         "dommages-intérêts.", "article 76 de la présente loi", "article_non_consulte"),
        ("Under Cameroonian labour law, Article 76 of this Law provides for damages.",
         "Article 76 of this Law", "article_non_consulte"),
        ("Le salarié a droit à des dommages-intérêts en vertu de l'article 76 du même texte.",
         "article 76 du même texte", "article_non_consulte"),
        # Une citation qui nomme la loi de finances, puis une sans nom : jamais
        # un tableau de credits pour celle-ci (sondes de la revue du tour final).
        ("L'article 76 de la loi de finances 2023 ne concerne que le budget. L'article 76 "
         "prévoit des dommages-intérêts.", "article 76", "article_non_consulte"),
    ])
    async def test_l_incident_sous_d_autres_formes(self, base, reponse, cite, raison):
        """
        Sondes du 2026-10-29, tours 3 et final : avec les extraits de
        l'incident, ces formes allaient encore au morceau 76.37 de la loi de
        finances 2023, dans `citations`, dans `sources` (repli), ou dans le
        message enregistre, que le front affiche a la reouverture.
        """
        tableau = "Les montants des autorisations d'engagement et des crédits de paiement…"
        extraits = [
            _extrait(85057, 2232, "76.37", "Loi N°2022/020 du 27 décembre 2022 portant loi de "
                     "finances de la république du Cameroun pour l'exercice 2023", "PRC-9449",
                     tableau, pertinence=0.9),
        ]
        service = _service(base, extraits, [reponse])

        evenements = await _evenements(service, _requete(
            question="Quelles indemnités en cas de licenciement abusif ?",
        ))

        fin = evenements[-1]
        # La citation qui nomme la loi de finances garde son lien ; aucune autre.
        nommee = "L'article 76 de la loi de finances 2023" in reponse
        assert [c["texte_cite"] for c in fin["citations"]] == (
            ["article 76 de la loi de finances 2023"] if nommee else []
        )
        assert (cite, raison) in [
            (c["texte_cite"], c["raison"]) for c in fin["citations_introuvables"]
        ]
        if nommee:
            return
        assert fin["articles_cites"] == []
        assert [t["law_id"] for t in fin["textes_consultes_non_cites"]] == [2232]
        assert fin["sources"] == []
        enregistres = [appel.args[0] for appel in base.add.call_args_list if appel.args]
        reponses = [m for m in enregistres if getattr(m, "role", None) == "assistant"]
        assert reponses and all(m.sources == [] for m in reponses)

    @pytest.mark.asyncio
    async def test_une_date_de_remplissage_n_est_pas_montree(self, base, extraits):
        """
        `date_du_texte` suit la regle de la page du texte (date_signature) : le
        1er janvier que l'ingestion ecrit faute de jour et de mois n'est pas
        une date de l'acte. Le chat montrait « 2024-01-01 » partout ou la page
        dit seulement « 2024 ».
        """
        avancement = "Décret N°2024/193 juin 10 portant avancement d'échelon"
        extraits = [
            *extraits[:2],
            ChunkResult(
                article_id=91, law_id=9, number="2", content="Le présent décret prend effet.",
                reference="PRC-10017", law_title=avancement, publication_date=date(2024, 1, 1),
                relevance_score=0.7,
            ),
        ]
        service = _service(base, extraits, [
            "Selon l'article 37 du Code du travail, l'indemnité est due.",
        ])

        evenements = await _evenements(service, _requete())

        trouves = next(e["progression"] for e in evenements
                       if e.get("progression", {}).get("etape") == "textes_trouves")
        dates = {t["law_id"]: t["date_du_texte"] for t in trouves["textes"]}
        assert dates == {3: "1992-08-14", 9: None}
        fin = evenements[-1]
        assert fin["citations"][0]["date_du_texte"] == "1992-08-14"
        assert fin["articles_cites"][0]["date_du_texte"] == "1992-08-14"
        consultes = {t["law_id"]: t["date_du_texte"] for t in fin["textes_consultes_non_cites"]}
        assert consultes == {9: None}

    @pytest.mark.asyncio
    async def test_sans_citation_aucune_source_n_est_fabriquee(self, base, extraits):
        """
        Une reponse qui ne cite rien n'a pas de source : rag_service ne
        fabrique plus de source de repli avec le premier extrait (tour final,
        « PAS DE SOURCE FABRIQUEE » : un tableau de loi de finances sortait sous
        une reponse sur le Code du travail, et restait dans la conversation
        enregistree). Les textes lus restent parmi les textes consultes, non
        cites.
        """
        service = _service(base, extraits, ["Le licenciement ouvre droit à une indemnité."])

        evenements = await _evenements(service, _requete())

        fin = evenements[-1]
        assert fin["sources"] == []
        assert fin["articles_cites"] == [] and fin["citations"] == []
        assert {t["law_id"] for t in fin["textes_consultes_non_cites"]} == {3, 8}
        enregistres = [appel.args[0] for appel in base.add.call_args_list if appel.args]
        reponses = [m for m in enregistres if getattr(m, "role", None) == "assistant"]
        assert reponses and all(m.sources == [] for m in reponses)

    @pytest.mark.asyncio
    async def test_un_extrait_ecarte_du_contexte_n_a_pas_ete_consulte(self, base, monkeypatch):
        """
        Le budget du contexte ecarte les derniers extraits : le modele ne les a
        pas lus. Une citation qui les vise n'est portee par rien qu'il ait lu.
        """
        from app.services import prompts

        monkeypatch.setattr(prompts, "CONTEXT_MAX_CHARS", 1_000)
        long = "Texte. " * 200
        extraits = [
            _extrait(371, 3, "37", CODE_DU_TRAVAIL, "PRC-3", long),
            _extrait(81, 8, "8", "Décret N°2018/9387/CAB/PM fixant les modalités", "PRC-8", long),
        ]
        service = _service(base, extraits, ["Voir l'article 8 du Décret N°2018/9387/CAB/PM."])

        evenements = await _evenements(service, _requete())

        fin = evenements[-1]
        assert [c["texte_cite"] for c in fin["citations_introuvables"]] == [
            "article 8 du Décret N°2018/9387/CAB/PM",
        ]
        assert [t["law_id"] for t in fin["textes_consultes_non_cites"]] == [3]

    @pytest.mark.asyncio
    async def test_un_article_colle_au_dela_de_la_coupe_n_a_pas_ete_lu(self, base, monkeypatch):
        """
        L'article 125 du Code de procédure pénale est colle a la fin du morceau
        « 124.1 ». Si la coupe du contexte passe avant lui, le modele ne l'a
        pas lu : sa citation n'est portee par rien, meme si le morceau l'est.
        """
        from app.services import prompts

        monkeypatch.setattr(prompts, "CONTEXT_MAX_CHARS_PER_CHUNK", 1_000)
        cpp = "Loi N°2005/007 du 27 juillet 2005 portant Code de procédure pénale"
        contenu = (
            "(1) La garde à vue est une mesure de police. " * 25
            + "\n\nArticle 125 — (1) Lorsque l'officier de police judiciaire…"
        )
        extraits = [_extrait(1241, 2005, "124.1", cpp, "LOI-2005-007", contenu)]
        service = _service(base, extraits, [
            "Voir l'article 124 de la Loi N°2005/007 et l'article 125 de la Loi N°2005/007.",
        ])

        evenements = await _evenements(service, _requete())

        fin = evenements[-1]
        assert [c["article"] for c in fin["citations"]] == ["124"]
        assert [(c["texte_cite"], c["raison"]) for c in fin["citations_introuvables"]] == [
            ("article 125 de la Loi N°2005/007", "article_non_consulte"),
        ]

    @pytest.mark.asyncio
    async def test_le_statut_prouve_accompagne_la_citation(self, base):
        """
        Constitution, article 14 : reecrit par la loi 2008/001 (PRC-6284). Le
        lien vers ce texte porte son identifiant, lu en une requete.
        """
        constitution = _extrait(141, 6283, "14", "Constitution du 18 janvier 1996", "PRC-6283")
        service = _service(base, [constitution], [
            "Selon l'article 14 de la Constitution, modifié par la Loi N°2008/001 du 14 avril 2008.",
        ])
        base.execute = AsyncMock(side_effect=lambda *a, **k: MagicMock(
            scalar_one_or_none=MagicMock(return_value=None),
            all=MagicMock(return_value=[("PRC-6284", 6284)]),
        ))

        evenements = await _evenements(service, _requete())

        texte = "".join(e["chunk"] for e in evenements)
        article, loi = evenements[-1]["citations"]
        assert article["statut"]["etat"] == "modifie"
        modification = next(
            lien for lien in article["statut"]["liens"] if lien["par_reference"] == "PRC-6284"
        )
        assert modification["par_law_id"] == 6284
        assert modification["par_date"] == "2008-04-14"
        # La loi nommee par la ligne de statut : liee, pas signalee.
        assert loi["law_id"] == 6284 and loi["article"] is None
        assert loi["titre"] == "Loi N°2008/001 du 14 avril 2008"
        assert _slice_js(texte, loi["debut"], loi["fin"]) == "Loi N°2008/001 du 14 avril 2008"
        assert evenements[-1]["citations_introuvables"] == []


@pytest.mark.parametrize("intitule, titre", [
    ("la Loi N°2008/001 du 14 avril 2008", "Loi N°2008/001 du 14 avril 2008"),
    ("l'Ordonnance N°2025/002", "Ordonnance N°2025/002"),
    ("le Décret N°2019/143", "Décret N°2019/143"),
    # « le » ne mord pas sur « les », ni sur le debut d'un mot.
    ("les dispositions transitoires", "dispositions transitoires"),
    ("lesdits textes", "lesdits textes"),
])
def test_le_titre_d_un_texte_nomme_perd_son_article(intitule, titre):
    from app.services.rag_service import _SANS_ARTICLE_DEFINI

    assert _SANS_ARTICLE_DEFINI.sub("", intitule) == titre


# ==================== IDENTIFIANTS, SUR LA VRAIE BASE ====================


class TestIdentifiantsSurLaVraieBase:
    """
    Le lien vers le texte qui modifie porte l'identifiant lu dans `laws`. Et
    un echec de cette lecture, une vraie erreur SQL, ne fait pas sortir de
    l'historique une reponse deja affichee : l'echec reste dans son point de
    sauvegarde. Un rollback complet expirait la conversation, et son
    enregistrement echouait ensuite (MissingGreenlet).
    """

    REPONSE = "Selon l'article 14 de la Constitution, modifié par la Loi N°2008/001 du 14 avril 2008."

    def _service_sur(self, db_session):
        constitution = _extrait(141, 6283, "14", "Constitution du 18 janvier 1996", "PRC-6283")
        return _service(db_session, [constitution], [self.REPONSE])

    async def _messages(self, db_session, session_id):
        from sqlalchemy import select

        from app.models.conversation import Conversation, Message

        return (await db_session.execute(
            select(Message).join(Conversation, Message.conversation_id == Conversation.id)
            .where(Conversation.session_id == session_id).order_by(Message.id)
        )).scalars().all()

    @pytest.mark.asyncio
    async def test_l_identifiant_vient_de_la_table_des_textes(self, db_session):
        from app.models.law import Law

        loi = Law(reference="PRC-6284", title="Loi N°2008/001 du 14 avril 2008", type="loi",
                  content="Article 1er. La Constitution est modifiée.", status="published")
        db_session.add(loi)
        await db_session.commit()

        evenements = await _evenements(self._service_sur(db_session), _requete())

        article, nommee = evenements[-1]["citations"]
        [lien] = [lien for lien in article["statut"]["liens"] if lien["par_reference"] == "PRC-6284"]
        assert lien["par_law_id"] == loi.id
        assert nommee["law_id"] == loi.id and nommee["article"] is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("statut", ["archived", "draft"])
    async def test_un_texte_non_publie_n_a_pas_d_adresse(self, db_session, statut):
        """
        Comme sur la page d'un texte (TexteLie) : un texte archive ou en
        brouillon n'est pas un lien. Il reste nomme par le statut, sans
        adresse, et sa mention n'est ni liee ni signalee.
        """
        from app.models.law import Law

        db_session.add(Law(
            reference="PRC-6284", title="Loi N°2008/001 du 14 avril 2008", type="loi",
            content="Article 1er. La Constitution est modifiée.", status=statut,
        ))
        await db_session.commit()

        fin = (await _evenements(self._service_sur(db_session), _requete()))[-1]

        [article] = fin["citations"]
        [lien] = [lien for lien in article["statut"]["liens"]
                  if lien["par_reference"] == "PRC-6284"]
        assert lien["par_law_id"] is None
        assert fin["citations_introuvables"] == []

    @pytest.mark.asyncio
    async def test_un_echec_de_cette_lecture_ne_perd_pas_l_echange(self, db_session, monkeypatch):
        from sqlalchemy import text

        from app.services import rag_service as module

        monkeypatch.setattr(
            module, "_requete_des_identifiants",
            lambda references: text("SELECT reference, id FROM table_qui_n_existe_pas"),
        )

        evenements = await _evenements(self._service_sur(db_session), _requete())

        fin = evenements[-1]
        assert fin["done"] is True and "error" not in fin
        # Sans identifiant : le lien reste sans adresse, la loi nommee n'est ni
        # liee ni signalee.
        [article] = fin["citations"]
        assert all(lien["par_law_id"] is None for lien in article["statut"]["liens"])
        assert fin["citations_introuvables"] == []
        messages = await self._messages(db_session, fin["session_id"])
        assert [m.role for m in messages] == ["user", "assistant"]
        assert messages[1].content == self.REPONSE


# ==================== REPONSE SANS FLUX ====================


class TestReponseSansFlux:

    @pytest.mark.asyncio
    async def test_ask_rend_les_memes_champs(self, base, extraits):
        service = _service(base, extraits)
        service.llm.generate = AsyncMock(return_value={
            "response": "L'article 37 du Code du travail fixe l'indemnité ; voir l'article 312 du Code pénal.",
        })

        reponse = await service.ask(_requete())

        assert [c.texte_cite for c in reponse.citations] == ["article 37 du Code du travail"]
        assert reponse.answer[reponse.citations[0].debut:reponse.citations[0].fin] == (
            "article 37 du Code du travail"
        )
        assert [c.raison for c in reponse.citations_introuvables] == ["texte_non_consulte"]
        assert [s.article_number for s in reponse.articles_cites] == ["37"]
        assert [t.law_id for t in reponse.textes_consultes_non_cites] == [8]
        assert [s.article_number for s in reponse.sources] == ["37"]

    @pytest.mark.asyncio
    async def test_la_mention_de_fin_ne_decale_rien(self, base, extraits):
        """La mention d'une reponse tronquee suit le texte : les indices restent justes."""
        service = _service(base, extraits)
        service.llm.generate = AsyncMock(return_value={
            "response": "Voir l'article 40 du Code du travail", "tronquee": True,
        })

        reponse = await service.ask(_requete())

        [citation] = reponse.citations
        assert reponse.answer[citation.debut:citation.fin] == "article 40 du Code du travail"
        assert reponse.answer.endswith(")*")
