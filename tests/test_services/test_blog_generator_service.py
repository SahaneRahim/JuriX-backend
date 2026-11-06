"""
Le générateur d'articles du blog.

Ce que ces tests protègent, dans l'ordre d'importance :

  1. AUCUNE SOURCE PROUVÉE, AUCUN ARTICLE : sans extrait publié et
     clairement pertinent (preuve plus stricte que celle du chat, tour 2), le
     modèle n'est pas appelé. L'ancien service écrivait alors de mémoire, et le
     tour 1 appelait encore le modèle sur « Recette de cuisine du ndolé ».
  2. LES CITATIONS VIENNENT DE LA BASE, ET SONT CE QU'ELLES DISENT ÊTRE. Le
     modèle ne rend que des numéros d'extraits ; chaque extrait est UN article
     (ou la suite d'un article, citée comme telle), relu en base. Jamais la
     suite d'un article sous le numéro et le début d'un autre (revue du tour
     1 : « 35.2 » cité « article 35 » avec le début de l'article 35), jamais une
     annotation de jurisprudence comme texte de loi, jamais un extrait amputé
     de son début ni débordant sur l'article suivant.
  3. CE QUI N'EST PAS PORTÉ SE VOIT : toute mention d'article (pluriels,
     intervalles, « premier », sans nom de texte, anglais), un texte nommé ou
     le droit d'un autre pays qu'aucun extrait ne porte devient un
     avertissement.
  4. UNE RÉPONSE COUPÉE OU ILLISIBLE EST UNE ERREUR, jamais un faux article.
  5. La recherche est celle du chat (exacte, textes publiés, sans tableaux de
     programmes), sans journal de recherche.

Le modèle est une doublure (tests/blog_donnees.py) ; le vrai n'est jamais appelé.

Usage:
    pytest tests/test_services/test_blog_generator_service.py -v
"""

import json
from unittest.mock import AsyncMock

import numpy as np
import pytest
from sqlalchemy import text, update

from app.models.law import Article
from app.schemas.search import ChunkResult
from app.services.blog_generator_service import (
    DELAI_GENERATION_S,
    EXTRAITS_AU_PLUS,
    SCHEMA_DE_L_ARTICLE,
    SEUILS_ANCRES_DU_BLOG,
    SEUILS_DU_BLOG,
    SLUG_VALIDE,
    SYSTEM_PROMPT,
    BlogGeneratorService,
    CitationInvalide,
    DemandeDeCitation,
    QuotaError,
    RechercheDuBlog,
    RechercheIndisponibleError,
    ReponseInvalideError,
    SansSourceError,
    SaturationError,
    citations_verifiees,
    debut_du_corps,
    decouper,
    duree_de_lecture,
    entetes_colles,
    est_modificatif,
    est_un_article_cite,
    extrait_verbatim,
    numero_affiche,
    prose_publique,
    rubrique_connue,
    sans_renvois_aux_extraits,
    slugify,
    sujet_de_l_article,
    texte_brut,
)
from app.services.embedding_service import EmbeddingService
from app.services.gemini_service import (
    GeminiBudgetEpuiseError,
    GeminiOverloadedError,
    GeminiQuotaError,
)
from app.services.mentions_d_articles import (
    droit_etranger,
    mentions_d_articles,
    numeros_non_portes,
    textes_nommes_non_portes,
)
from app.services.mistral_service import (
    MistralOverloadedError,
    MistralQuotaError,
    MistralServiceError,
)
from app.services.search_service import (
    CHEMIN_ECHEC,
    CHEMIN_REPLI_HNSW,
    SearchService,
    ecritures_en_attente,
)
from tests.blog_donnees import (
    CC_1736,
    CC_1751_1,
    CP_299,
    CP_322,
    CPP_8,
    CPP_679_2,
    CT_35_1,
    CT_35_2,
    CT_67_1,
    CT_67_2,
    SUJET,
    TITRE_CODE_CIVIL,
    TITRE_CODE_DU_TRAVAIL,
    TITRE_CODE_PENAL,
    RechercheDoublee,
    article_du_modele,
    creer_les_lignes_reelles,
    creer_les_textes,
    extrait,
    extraits_du_sujet,
    lignes_reelles,
    modele_double,
    similarites,
)


@pytest.fixture
async def textes(db_session):
    return await creer_les_textes(db_session)


def _service(db, recherche, llm=None):
    return BlogGeneratorService(db, llm=llm or modele_double(), recherche=recherche)


def _prompt(llm) -> str:
    return llm.generate.await_args.kwargs["prompt"]


def _du_travail(t, *noms_et_sims):
    """Des extraits du Code du travail et leurs similarités, dans l'ordre donné."""
    lignes = {"ct_35_1": ("35.1", CT_35_1), "ct_35_2": ("35.2", CT_35_2),
              "ct_67_1": ("67.1", CT_67_1), "ct_67_2": ("67.2", CT_67_2)}
    trouves = [
        extrait(t.code_du_travail, getattr(t, nom), lignes[nom][0], lignes[nom][1],
                reference="LOI-1992-007")
        for nom, _ in noms_et_sims
    ]
    return RechercheDoublee(trouves, similarites(t, dict(noms_et_sims)))


# ==================== 1. RECHERCHE ET PREUVE DE PERTINENCE ====================


class TestLaRechercheDuChat:

    async def test_mots_cles_au_plein_texte_sujet_entier_au_sens_textes_publies(
        self, db_session, textes
    ):
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        await _service(db_session, recherche).rediger(SUJET)

        requete = recherche.requetes[0]
        assert requete.query == "expulsion locataire décision justice"
        assert requete.semantic_query == SUJET
        assert requete.mode == "hybrid"
        assert requete.filters.status == "published"
        assert requete.limit == 8

    async def test_sans_les_tableaux_de_programmes_sauf_sujet_budgetaire(self, db_session, textes):
        """Le sujet du chat de l'incident du 29/10/2026 : des tableaux de lois de finances lus."""
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))
        await _service(db_session, recherche).rediger(SUJET)
        assert recherche.tableaux_exclus_a_la_recherche is True

        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))
        await _service(db_session, recherche).rediger("Loi de finances 2025 : crédits de la santé")
        assert recherche.tableaux_exclus_a_la_recherche is False

    def test_la_vraie_recherche_est_exacte_sans_cache(self, db_session):
        recherche = RechercheDuBlog(db_session)

        assert recherche.semantique_exacte is True
        assert recherche.use_cache is False


class TestSansSource:

    async def test_aucun_resultat_le_modele_n_est_pas_appele(self, db_session, textes):
        llm = modele_double()

        with pytest.raises(SansSourceError) as erreur:
            await _service(db_session, RechercheDoublee([]), llm).rediger(SUJET)

        assert erreur.value.statut == 422
        assert erreur.value.code == "sans_source"
        assert "Aucun texte publié" in str(erreur.value)
        llm.generate.assert_not_awaited()

    async def test_des_resultats_tous_loin_du_sujet_ne_suffisent_pas(self, db_session, textes):
        """Similarité sous tous les seuils, et pas tous les mots-clés : rien n'est pertinent."""
        llm = modele_double()
        loin = [extrait(textes.code_penal, textes.cp_322, "322", CP_322)]
        recherche = RechercheDoublee(loin, similarites(textes, {"cp_322": 0.12}))

        with pytest.raises(SansSourceError):
            await _service(db_session, recherche, llm).rediger(SUJET)

        llm.generate.assert_not_awaited()

    async def test_ni_brouillon_ni_preambule_ni_ligne_de_loi(self, db_session, textes):
        """Proches du sujet, mais pas citables : aucun ne fait une source."""
        llm = modele_double()
        pas_citables = [
            extrait(textes.projet, textes.projet_5, "5", "x", reference="LOI-2099-001",
                    status="draft"),
            extrait(textes.code_civil, textes.cc_preambule, "PREAMBULE", "Vu la Constitution ;",
                    reference="CODE-CIVIL"),
            extrait(textes.code_civil, None, None, "Code civil.", reference="CODE-CIVIL"),
        ]
        recherche = RechercheDoublee(
            pas_citables, similarites(textes, {"projet_5": 0.9, "cc_preambule": 0.9})
        )

        with pytest.raises(SansSourceError):
            await _service(db_session, recherche, llm).rediger(SUJET)

        llm.generate.assert_not_awaited()

    async def test_seule_une_annotation_de_jurisprudence_n_est_pas_une_source(self, db_session, textes):
        """« 1751.2 » du Code civil annoté : des arrêts, pas la loi. Rien à donner au modèle."""
        llm = modele_double()
        trouves = [extrait(textes.code_civil, textes.cc_1751_2, "1751.2", "x", reference="CODE-CIVIL")]
        recherche = RechercheDoublee(trouves, similarites(textes, {"cc_1751_2": 0.9}))

        with pytest.raises(SansSourceError):
            await _service(db_session, recherche, llm).rediger(SUJET)

        llm.generate.assert_not_awaited()


class TestLaPreuveDePertinence:
    """
    Revue du tour 1 : le seul filtre était le seuil BAS du chat (0,30). « Recette
    de cuisine du ndolé » gardait 5 extraits (0,30-0,33) ; « Le SMIG au Cameroun
    c'est combien ? » 5 extraits de lois de finances (0,44-0,46) qui ne disent
    rien du SMIG. Le modèle était appelé. Seuils calibrés : SEUILS_DU_BLOG.
    """

    def test_les_seuils_calibres(self):
        assert SEUILS_DU_BLOG == {"fr": 0.48, "en": 0.42}
        assert SEUILS_ANCRES_DU_BLOG == {"fr": 0.455, "en": 0.42}

    @pytest.mark.parametrize("sujet, similarite", [
        ("Recette de cuisine du ndolé", 0.33),
        # Le SMIG : proche par le sens d'une loi de finances, sans en porter le mot.
        ("Le SMIG au Cameroun c'est combien ?", 0.46),
        # Le chat aurait gardé cet extrait (0,47 >= 0,30) : pas le blog.
        ("Expulsion du locataire sans décision de justice", 0.47),
    ])
    async def test_proche_par_le_sens_mais_sans_mot_du_sujet_refuse(
        self, db_session, textes, sujet, similarite
    ):
        """L'article 322 (filouterie de loyers) ne porte aucun mot distinctif de ces sujets."""
        llm = modele_double()
        trouves = [extrait(textes.code_penal, textes.cp_322, "322", CP_322)]
        recherche = RechercheDoublee(trouves, similarites(textes, {"cp_322": similarite}))

        with pytest.raises(SansSourceError):
            await _service(db_session, recherche, llm).rediger(sujet)

        llm.generate.assert_not_awaited()

    async def test_tres_proche_par_le_sens_suffit(self, db_session, textes):
        llm = modele_double(article_du_modele(citations=[{"extrait": 1, "objet": "Le loyer."}]))
        trouves = [extrait(textes.code_penal, textes.cp_322, "322", CP_322)]
        recherche = RechercheDoublee(trouves, similarites(textes, {"cp_322": 0.49}))

        await _service(db_session, recherche, llm).rediger(SUJET)

        assert "Article 322" in _prompt(llm)

    @pytest.mark.parametrize("similarite, admis", [(0.46, True), (0.45, False)])
    async def test_proche_et_porteur_d_un_mot_du_sujet(self, db_session, textes, similarite, admis):
        """L'article 1751 porte « expulsion », « locataire » : le seuil ancré (0,455) suffit."""
        llm = modele_double(article_du_modele(citations=[{"extrait": 1, "objet": "Le rachat."}]))
        trouves = [extrait(textes.code_civil, textes.cc_1751_1, "1751.1", CC_1751_1,
                           reference="CODE-CIVIL")]
        recherche = RechercheDoublee(trouves, similarites(textes, {"cc_1751_1": similarite}))

        if admis:
            await _service(db_session, recherche, llm).rediger(SUJET)
            assert "Article 1751" in _prompt(llm)
        else:
            with pytest.raises(SansSourceError):
                await _service(db_session, recherche, llm).rediger(SUJET)
            llm.generate.assert_not_awaited()

    @pytest.mark.parametrize("similarite, admis", [(0.43, True), (0.41, False)])
    async def test_un_sujet_anglais_a_son_seuil(self, db_session, textes, similarite, admis):
        llm = modele_double(article_du_modele(citations=[{"extrait": 1, "objet": "Trespass."}]))
        trouves = [extrait(textes.code_penal, textes.cp_299, "299", CP_299)]
        recherche = RechercheDoublee(trouves, similarites(textes, {"cp_299": similarite}))

        if admis:
            await _service(db_session, recherche, llm).rediger("Landlord breaking into the tenant's home")
            assert "Article 299" in _prompt(llm)
        else:
            with pytest.raises(SansSourceError):
                await _service(db_session, recherche, llm).rediger(
                    "Landlord breaking into the tenant's home"
                )
            llm.generate.assert_not_awaited()

    async def test_le_sujet_prouve_les_extraits_ancres_suivent(self, db_session, textes):
        """
        Prouvé par l'article 299 (0,52), le sujet garde l'article 1751, qui porte
        tous ses mots-clés, même sans similarité ; jamais l'article 322, loin et
        sans mot du sujet.
        """
        llm = modele_double()
        trouves = [
            extrait(textes.code_penal, textes.cp_299, "299", CP_299),
            extrait(textes.code_penal, textes.cp_322, "322", CP_322),
            extrait(textes.code_civil, textes.cc_1751_1, "1751.1", CC_1751_1, reference="CODE-CIVIL"),
        ]
        recherche = RechercheDoublee(trouves, similarites(textes, {"cp_299": 0.52, "cp_322": 0.1}))

        await _service(db_session, recherche, llm).rediger(SUJET)

        prompt = _prompt(llm)
        assert "Article 299" in prompt
        assert "Article 1751" in prompt
        assert "Filouteries" not in prompt
        assert "EXTRAITS (2, numérotés de 1 à 2" in prompt

    async def test_le_modele_juge_les_extraits_insuffisants(self, db_session, textes):
        """Dernier filet : le modèle dit que les extraits ne traitent pas le sujet."""
        llm = modele_double(article_du_modele(
            extraits_suffisants=False, title="", context="", summary="", key_points=[],
            citations=[], citizen_impact="", suggested_question="", tags=[],
        ))
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        with pytest.raises(SansSourceError) as erreur:
            await _service(db_session, recherche, llm).rediger(SUJET)

        assert erreur.value.code == "sans_source"


class TestRechercheIndisponible:

    async def test_la_recherche_en_panne(self, db_session, textes):
        llm = modele_double()
        recherche = RechercheDoublee([], erreur=RuntimeError("base injoignable"))

        with pytest.raises(RechercheIndisponibleError) as erreur:
            await _service(db_session, recherche, llm).rediger(SUJET)

        assert (erreur.value.statut, erreur.value.code, erreur.value.retry_after) == (
            503, "recherche_indisponible", 10
        )
        # Le message ne recopie pas l'erreur interne.
        assert "injoignable" not in str(erreur.value)
        llm.generate.assert_not_awaited()

    @pytest.mark.parametrize("chemin", [CHEMIN_REPLI_HNSW, CHEMIN_ECHEC])
    async def test_un_volet_semantique_non_verifiable(self, db_session, textes, chemin):
        """Le repli sur l'index, ou un volet en panne : le blog ne rédige pas, il fait réessayer."""
        llm = modele_double()
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes), chemin=chemin)

        with pytest.raises(RechercheIndisponibleError):
            await _service(db_session, recherche, llm).rediger(SUJET)

        llm.generate.assert_not_awaited()


# ==================== 2. LE PROMPT : UN EXTRAIT, UN ARTICLE ====================


class TestLePrompt:

    async def test_extraits_numerotes_et_nommes_par_leur_intitule(self, db_session, textes):
        """Jamais « PRC-4722 — Article ? » : l'intitulé du texte et le numéro de l'article."""
        llm = modele_double()
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        await _service(db_session, recherche, llm).rediger(SUJET)

        prompt = _prompt(llm)
        assert f"[1] {TITRE_CODE_PENAL} — Article 299" in prompt
        assert f"[2] {TITRE_CODE_CIVIL} — Article 1736" in prompt
        assert f"[3] {TITRE_CODE_CIVIL} — Article 1751\n" in prompt
        # Le morceau suivant du Code civil annoté (des arrêts) n'est pas donné.
        assert "[4]" not in prompt and "Non paiement des loyers" not in prompt
        # Ni les annotations, ni l'intitulé de la section suivante.
        assert "Jugements et arrêts" not in prompt and "SECT. II" not in prompt
        assert "1751.1" not in prompt
        assert "PRC-4722" not in prompt and "CODE-CIVIL" not in prompt
        assert "celui qui s'introduit ou se maintient dans le domicile d'autrui" in prompt

    async def test_les_articles_colles_sont_des_extraits_a_part(self, db_session, textes):
        """« 35.1 » porte les articles 35, 38 et 39 : trois extraits, chacun sous son numéro."""
        llm = modele_double(article_du_modele(citations=[{"extrait": 3, "objet": "Abus."}]))

        await _service(db_session, _du_travail(textes, ("ct_35_1", 0.6)), llm).rediger(
            "Indemnités en cas de licenciement abusif"
        )

        prompt = _prompt(llm)
        assert f"[1] {TITRE_CODE_DU_TRAVAIL} — Article 35\n" in prompt
        assert f"[2] {TITRE_CODE_DU_TRAVAIL} — Article 38\n" in prompt
        assert f"[3] {TITRE_CODE_DU_TRAVAIL} — Article 39\n" in prompt
        bloc_35 = prompt.split("[1] ")[1].split("---")[0]
        assert "Article 38.-" not in bloc_35

    async def test_la_suite_d_un_article_est_annoncee_comme_telle(self, db_session, textes):
        """« 35.2 » : la fin de l'article 39 (suite), puis l'article 40 — jamais « Article 35 »."""
        llm = modele_double(article_du_modele(citations=[{"extrait": 1, "objet": "Le salaire."}]))

        await _service(db_session, _du_travail(textes, ("ct_35_2", 0.6)), llm).rediger(
            "Indemnités en cas de licenciement abusif"
        )

        prompt = _prompt(llm)
        assert (
            f"[1] {TITRE_CODE_DU_TRAVAIL} — Article 39 (suite : ce passage commence au milieu "
            "de l'article 39, son début n'est pas fourni)"
        ) in prompt
        assert f"[2] {TITRE_CODE_DU_TRAVAIL} — Article 40\n" in prompt
        assert "Article 35" not in prompt

    async def test_les_consignes(self, db_session, textes):
        llm = modele_double()
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        await _service(db_session, recherche, llm).rediger(SUJET)

        kwargs = llm.generate.await_args.kwargs
        assert kwargs["system"] == SYSTEM_PROMPT
        assert "N'écris aucun numéro d'article ni aucun nom de texte" in SYSTEM_PROMPT
        assert "ni Markdown" in SYSTEM_PROMPT
        assert "droit d'un autre pays" in SYSTEM_PROMPT
        assert '"extraits_suffisants" vaut false' in SYSTEM_PROMPT
        assert "Chaque extrait est UN article" in SYSTEM_PROMPT
        assert kwargs["response_mime_type"] == "application/json"
        assert kwargs["response_schema"] == SCHEMA_DE_L_ARTICLE
        assert "extraits_suffisants" in SCHEMA_DE_L_ARTICLE["required"]
        # Le modèle ne rend, pour une citation, qu'un numéro d'extrait et un objet.
        citation = SCHEMA_DE_L_ARTICLE["properties"]["citations"]["items"]["properties"]
        assert set(citation) == {"extrait", "objet"}
        # Plus de phrase de repli, ni de gabarit « law_id 1, article 12 ».
        assert "dispositions officielles applicables" not in _prompt(llm)
        assert '"law_id"' not in _prompt(llm)

    async def test_le_contexte_de_l_equipe_va_au_modele_pas_a_la_recherche(
        self, db_session, textes
    ):
        llm = modele_double()
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        await _service(db_session, recherche, llm).rediger(
            SUJET, additional_context="Une famille expulsée à Douala le 3 octobre."
        )

        assert "Une famille expulsée à Douala le 3 octobre." in _prompt(llm)
        assert "Douala" not in recherche.requetes[0].query
        assert "Douala" not in recherche.requetes[0].semantic_query


# ==================== 3. LES CITATIONS ====================


class TestLesCitationsViennentDeLaBase:

    async def test_construites_par_le_serveur(self, db_session, textes):
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        article = await _service(db_session, recherche).rediger(SUJET)

        premiere, seconde = article.citations
        assert premiere == {
            "law_id": textes.code_penal,
            "law_reference": "PRC-4722",
            "article_id": textes.cp_299,
            "article_number": "299",
            "reference": TITRE_CODE_PENAL,
            "title": "La violation de domicile est punie.",
            # Le début MOT POUR MOT de l'article, intitulé compris : rien n'est
            # retiré en silence.
            "excerpt": (
                "Violation de domicile (1) Est puni d'un emprisonnement de dix (10) jours à un "
                "(01) an et d'une amende de cinq mille (5 000) à cinquante mille (50 000) "
                "francs ou de l'une de ces deux peines seulement, celui qui s'introduit ou se "
                "maintient dans le domicile d'autrui contre son gré."
            ),
            "url": f"/laws/{textes.code_penal}?article=299",
            "verifie": True,
            "suite": False,
        }
        # Le Code civil annoté : l'article, sans les « Jugements et arrêts » qui le suivent.
        assert seconde["excerpt"] == (
            "Si le bail a été fait sans écrit, l'une des parties ne pourra donner congé à "
            "l'autre qu'en observant les délais fixés par l'usage des lieux."
        )
        assert seconde["reference"] == TITRE_CODE_CIVIL

    async def test_ce_que_le_modele_ecrit_de_plus_est_ignore(self, db_session, textes):
        """Le modèle de l'essai réel écrivait « 1351 (et jurisprudence) » et un nom inventé."""
        modele = article_du_modele(citations=[{
            "extrait": 2, "objet": "Le congé.", "article_number": "1351 (et jurisprudence)",
            "reference": "Acte uniforme relatif au droit de la copropriété", "law_id": 1,
        }])
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        article = await _service(db_session, recherche, modele_double(modele)).rediger(SUJET)

        (citation,) = article.citations
        assert citation["law_id"] == textes.code_civil
        assert citation["article_number"] == "1736"
        assert citation["reference"] == TITRE_CODE_CIVIL

    async def test_un_numero_d_extrait_inconnu_est_ecarte_et_signale(self, db_session, textes):
        modele = article_du_modele(citations=[
            {"extrait": 1, "objet": "La violation de domicile."},
            {"extrait": 9, "objet": "Un extrait qui n'existe pas."},
            {"extrait": 0, "objet": "Zéro non plus."},
        ])
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        article = await _service(db_session, recherche, modele_double(modele)).rediger(SUJET)

        assert [c["article_id"] for c in article.citations] == [textes.cp_299]
        messages = [a.message for a in article.avertissements]
        assert any("l'extrait 9" in m and "écartée" in m for m in messages)
        assert any("l'extrait 0" in m for m in messages)

    async def test_aucune_citation_valable_aucun_article(self, db_session, textes):
        modele = article_du_modele(citations=[{"extrait": 7, "objet": "Hors liste."}])
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        with pytest.raises(ReponseInvalideError) as erreur:
            await _service(db_session, recherche, modele_double(modele)).rediger(SUJET)

        assert erreur.value.statut == 502

    async def test_un_texte_depublie_pendant_la_generation_est_ecarte(self, db_session, textes):
        """Les citations sont relues APRÈS la génération : le statut du moment compte."""
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        async def depublier(**_):
            await db_session.execute(
                text("UPDATE laws SET status = 'archived' WHERE id = :id"),
                {"id": textes.code_civil},
            )
            await db_session.commit()
            return {"response": json.dumps(article_du_modele()), "fin": "STOP"}

        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=depublier)
        article = await _service(db_session, recherche, llm).rediger(SUJET)

        assert [c["law_id"] for c in article.citations] == [textes.code_penal]
        assert any("n'est pas publié" in a.message for a in article.avertissements)

    async def test_l_explication_n_est_pas_coupee_a_300(self, db_session, textes):
        """Une explication coupée en silence peut perdre sa négation : 500 caractères."""
        objet = "La violation de domicile est punie " + "très " * 60 + "sévèrement, sauf exception."
        modele = article_du_modele(citations=[{"extrait": 1, "objet": objet}])
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        article = await _service(db_session, recherche, modele_double(modele)).rediger(SUJET)

        assert article.citations[0]["title"].endswith("sauf exception.")


class TestArticlesColles:
    """
    Au Code du travail de la base, le morceau « 35.1 » porte les articles 36 à
    39 : l'article 39 (dommages-intérêts pour licenciement abusif) n'a pas de
    ligne à lui. Chaque article est un extrait ; la carte est celle de
    l'article cité, et son extrait s'arrête avant l'article suivant.
    """

    SUJET = "Indemnités en cas de licenciement abusif"

    async def test_la_carte_est_celle_de_l_extrait_cite(self, db_session, textes):
        modele = article_du_modele(
            summary=(
                "Toute rupture abusive du contrat peut donner lieu à des dommages-intérêts, "
                "selon l'article 39 du Code du travail."
            ),
            citations=[{"extrait": 3, "objet": "Les dommages-intérêts pour rupture abusive."}],
        )

        article = await _service(
            db_session, _du_travail(textes, ("ct_35_1", 0.6)), modele_double(modele)
        ).rediger(self.SUJET)

        (citation,) = article.citations
        assert (citation["article_number"], citation["article_id"]) == ("39", textes.ct_35_1)
        assert citation["url"] == f"/laws/{textes.code_du_travail}?article=39"
        assert citation["excerpt"] == (
            "Article 39.- (1) Toute rupture abusive du contrat peut donner lieu à des "
            "dommages-intérêts."
        )
        assert citation["suite"] is False
        assert not any("39" in a.message for a in article.avertissements)

    async def test_l_article_de_la_ligne_s_arrete_avant_l_article_colle(self, db_session, textes):
        """L'essai réel du 30/10 : une carte « article 35 » qui montrait le texte des articles 36 à 39."""
        modele = article_du_modele(citations=[{"extrait": 1, "objet": "Le préavis."}])

        article = await _service(
            db_session, _du_travail(textes, ("ct_35_1", 0.6)), modele_double(modele)
        ).rediger(self.SUJET)

        (citation,) = article.citations
        assert citation["article_number"] == "35"
        assert citation["excerpt"] == (
            "(1) Pendant la durée du préavis, l'employeur et le travailleur sont tenus au "
            "respect de toutes les obligations réciproques qui leur incombent."
        )

    async def test_par_identifiant_et_numero(self, db_session, textes):
        (citation,), _ = await citations_verifiees(db_session, [DemandeDeCitation(
            textes.code_du_travail, article_id=textes.ct_35_1, article_number="39"
        )])

        assert citation["article_number"] == "39"
        assert citation["excerpt"].startswith("Article 39.- (1) Toute rupture abusive")
        # L'intitulé de la ligne est celui de l'article 35 : il n'est pas repris.
        assert citation["title"] == ""

    async def test_par_son_seul_numero(self, db_session, textes):
        (citation,), _ = await citations_verifiees(
            db_session, [DemandeDeCitation(textes.code_du_travail, article_number="39")]
        )

        assert (citation["article_id"], citation["article_number"], citation["suite"]) == (
            textes.ct_35_1, "39", False
        )

    async def test_un_numero_que_le_morceau_ne_porte_pas(self, db_session, textes):
        with pytest.raises(CitationInvalide) as erreur:
            await citations_verifiees(db_session, [DemandeDeCitation(
                textes.code_du_travail, article_id=textes.ct_35_1, article_number="40"
            )])

        assert "ne correspond à aucun article" in str(erreur.value)

    async def test_un_extrait_ne_deborde_pas_sur_l_article_suivant(self, db_session, textes):
        """« 679.2 » du Code de procédure pénale : la carte de l'article 683 montrait le 684."""
        (citation,), _ = await citations_verifiees(db_session, [DemandeDeCitation(
            textes.code_de_procedure_penale, article_id=textes.cpp_679_2, article_number="683"
        )])

        assert citation["excerpt"] == (
            "Article 683 — La Cour d'Appel du lieu de résidence du condamné est compétente en "
            "matière de réhabilitation."
        )
        assert "684" not in citation["excerpt"]


class TestMorceauxSuivants:
    """
    Revue du tour 1 : un morceau suivant (« N.2 ») était annoncé « Article N
    (partie d'un article long) » et cité « article N » avec le début de
    l'article N. Sur jurix_dev, c'était faux dans 43 cas (la fin d'un article
    collé) et pour 29 annotations du Code civil annoté.
    """

    async def test_la_suite_d_un_article_colle_est_citee_sous_son_numero(self, db_session, textes):
        """« 35.2 » : les alinéas (6) et (7) de l'article 39, avec leur propre début."""
        (citation,), _ = await citations_verifiees(db_session, [DemandeDeCitation(
            textes.code_du_travail, article_id=textes.ct_35_2, title="Le salaire de référence."
        )])

        assert citation["article_number"] == "39"
        assert citation["suite"] is True
        assert citation["url"] == f"/laws/{textes.code_du_travail}?article=39"
        assert citation["excerpt"] == (
            "[…] (6) Le salaire à prendre en considération aux alinéas précédents est le "
            "salaire moyen mensuel brut des douze (12) derniers mois d'activité du travailleur. "
            "(7) Ces dommages-intérêts ne se confondent ni avec l'indemnité pour non-observation "
            "du préavis, ni avec l'indemnité de licenciement."
        )
        assert "Pendant la durée du préavis" not in citation["excerpt"]

    async def test_la_suite_n_est_jamais_citee_sous_le_numero_de_sa_ligne(self, db_session, textes):
        """La forme exacte du tour 1 : (35.2, « 35 ») est refusée."""
        with pytest.raises(CitationInvalide) as erreur:
            await citations_verifiees(db_session, [DemandeDeCitation(
                textes.code_du_travail, article_id=textes.ct_35_2, article_number="35"
            )])

        assert "ne correspond à aucun article du passage 35.2" in str(erreur.value)

    async def test_citer_la_suite_depuis_la_generation(self, db_session, textes):
        modele = article_du_modele(
            summary="Le salaire de référence est le salaire moyen des douze derniers mois.",
            citations=[{"extrait": 1, "objet": "Le salaire de référence."}],
        )

        article = await _service(
            db_session, _du_travail(textes, ("ct_35_2", 0.6)), modele_double(modele)
        ).rediger("Indemnités en cas de licenciement abusif")

        (citation,) = article.citations
        assert (citation["article_number"], citation["suite"]) == ("39", True)
        assert citation["excerpt"].startswith("[…] (6) Le salaire")
        assert article.sources[0]["suite"] is True
        assert article.sources[0]["article_number"] == "39"

    async def test_le_debut_repete_d_un_article_n_est_pas_sa_suite(self, db_session, textes):
        """
        « 67.2 » répète les 200 premiers caractères de l'article 67 (et l'en-tête
        de l'article 68) avant la suite de l'article 68 : seule cette suite est
        un passage.
        """
        (citation,), _ = await citations_verifiees(db_session, [DemandeDeCitation(
            textes.code_du_travail, article_id=textes.ct_67_2
        )])

        assert (citation["article_number"], citation["suite"]) == ("68", True)
        assert citation["excerpt"] == (
            "[…] (3) En cas de résiliation ou de rupture de contrat, le salaire et les "
            "indemnités doivent être payés dès la cessation de service."
        )

    async def test_une_annotation_n_est_jamais_citee_comme_l_article(self, db_session, textes):
        """« 1751.2 » du Code civil annoté : des arrêts, ni carte « article 1751 », ni autre."""
        with pytest.raises(CitationInvalide) as erreur:
            await citations_verifiees(
                db_session, [DemandeDeCitation(textes.code_civil, article_id=textes.cc_1751_2)]
            )

        assert "ne porte aucun texte d'article citable" in str(erreur.value)

    async def test_par_numero_le_debut_et_jamais_la_suite(self, db_session, textes):
        (citation,), _ = await citations_verifiees(
            db_session, [DemandeDeCitation(textes.code_civil, article_number="1751")]
        )

        assert (citation["article_id"], citation["suite"]) == (textes.cc_1751_1, False)
        assert citation["excerpt"].startswith("L'acquéreur à pacte de rachat")
        assert "SECT. II" not in citation["excerpt"]


# ==================== 4. LA PROSE ====================


class TestLaProse:

    async def test_le_markdown_est_retire(self, db_session, textes):
        modele = article_du_modele(
            title="**Expulsion** : ce que dit la loi",
            summary="## Résumé\n**Gras** et `code`.",
            key_points=["- **Premier** point", "* Second point", "   "],
            citizen_impact="__Important__ : portez plainte.",
        )
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        article = await _service(db_session, recherche, modele_double(modele)).rediger(SUJET)

        assert article.title == "Expulsion : ce que dit la loi"
        assert article.summary == "Résumé\nGras et code."
        assert article.key_points == ["Premier point", "Second point"]
        assert article.citizen_impact == "Important : portez plainte."

    async def test_une_cote_interne_est_retiree_et_signalee(self, db_session, textes):
        modele = article_du_modele(key_points=["Le Code pénal (PRC-4722) punit l'intrusion."])
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        article = await _service(db_session, recherche, modele_double(modele)).rediger(SUJET)

        assert article.key_points == ["Le Code pénal punit l'intrusion."]
        assert any("PRC-4722" in a.message for a in article.avertissements)

    @pytest.mark.parametrize("phrase, numeros", [
        # Les six formes de la revue du tour 1 (test_r1) : aucune n'était signalée.
        ("Les articles 286 et 312 du Code pénal punissent aussi l'intrusion.", "286 et 312"),
        ("L'intrusion est aussi punie par l'article 312.", "l'article 312"),
        ("Selon l'article 312, le bailleur ne peut pas entrer.", "l'article 312"),
        ("L'article premier du Code pénal impose la loi à tous.", "l'article 1"),
        ("Les articles 312 à 315 du Code pénal punissent l'intrusion.", "312, 313, 314 et 315"),
        ("Le Code pénal, en son article 312, punit l'intrusion.", "l'article 312"),
        # Et celles de la consigne du tour 2, sans extrait qui les porte.
        ("L'art. 37 prévoit une indemnité.", "l'article 37"),
        ("Voir l'article 39 alinéa 2 du Code pénal.", "l'article 39"),
        ("Section 312 of the Penal Code punishes trespass.", "l'article 312"),
    ])
    async def test_toute_mention_qu_aucun_extrait_ne_porte_est_signalee(
        self, db_session, textes, phrase, numeros
    ):
        """L'essai réel du 29/10 : « les articles 286 et 312 du Code pénal », inventés."""
        modele = article_du_modele(
            summary=f"S'introduire chez autrui est puni par l'article 299 du Code Pénal. {phrase}"
        )
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        article = await _service(db_session, recherche, modele_double(modele)).rediger(SUJET)

        messages = [a.message for a in article.avertissements]
        assert any(numeros in m and "Résumé" in m and "extraits consultés" in m for m in messages), messages
        assert not any("299" in m for m in messages)

    async def test_un_article_que_les_extraits_portent_n_est_pas_signale(self, db_session, textes):
        """Les articles 38 et 39 sont des extraits (collés dans 35.1) : rien à dire."""
        modele = article_du_modele(
            summary=(
                "Les articles 38 et 39 du Code du travail encadrent la rupture ; selon "
                "l'article 39, alinéa 1, l'abus ouvre droit à réparation."
            ),
            citations=[{"extrait": 3, "objet": "Abus."}],
        )

        article = await _service(
            db_session, _du_travail(textes, ("ct_35_1", 0.6)), modele_double(modele)
        ).rediger("Indemnités en cas de licenciement abusif")

        assert not any("38" in a.message or "39" in a.message for a in article.avertissements)

    async def test_un_texte_nomme_par_son_numero_et_non_lu_est_signale(self, db_session, textes):
        modele = article_du_modele(
            citizen_impact="Le décret n° 2018/9387 du 30 novembre 2018 précise la procédure."
        )
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        article = await _service(db_session, recherche, modele_double(modele)).rediger(SUJET)

        assert any(
            "Impact pour le citoyen" in a.message and "2018/9387" in a.message
            for a in article.avertissements
        )

    async def test_un_texte_nomme_qu_aucun_extrait_ne_porte(self, db_session, textes):
        modele = article_du_modele(
            key_points=["Le Code de la famille protège aussi le locataire."]
        )
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        article = await _service(db_session, recherche, modele_double(modele)).rediger(SUJET)

        assert any(
            "Point clé 1" in a.message and "Code de la famille" in a.message
            for a in article.avertissements
        )

    async def test_le_droit_d_un_autre_pays(self, db_session, textes):
        """L'essai réel du 29/10 reprenait un préavis du droit français."""
        modele = article_du_modele(
            citizen_impact="Comme en droit français, le préavis est de trois mois."
        )
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        article = await _service(db_session, recherche, modele_double(modele)).rediger(SUJET)

        assert any("droit français" in a.message and "autre pays" in a.message
                   for a in article.avertissements)

    async def test_rubrique_duree_mots_cles(self, db_session, textes):
        modele = article_du_modele(category="Travail & Emploi", tags=["#bail", "Bail", " ", 3])
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        article = await _service(db_session, recherche, modele_double(modele)).rediger(SUJET)

        assert article.category == "Travail"
        assert article.tags == ["bail"]
        assert article.reading_time == "1 min"

    async def test_la_rubrique_de_l_administrateur_prime(self, db_session, textes):
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        article = await _service(db_session, recherche).rediger(SUJET, category="Logement")

        assert article.category == "Logement"

    async def test_les_sources_sont_les_extraits_lus(self, db_session, textes):
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        article = await _service(db_session, recherche).rediger(SUJET)

        assert [s["n"] for s in article.sources] == [1, 2, 3]
        assert article.sources[0] == {
            "n": 1, "law_id": textes.code_penal, "law_title": TITRE_CODE_PENAL,
            "article_number": "299", "article_id": textes.cp_299, "suite": False,
            "extrait": CP_299,
        }
        assert article.sources[2]["article_number"] == "1751"
        assert article.sources[2]["article_id"] == textes.cc_1751_1


class TestLeDetecteurDeMentions:
    """Le filet du tour 2, seul : mentions_d_articles, numeros_non_portes."""

    @pytest.mark.parametrize("phrase, numeros, nom", [
        ("Les articles 286 et 312 du Code pénal punissent.", ("286", "312"), "Code pénal"),
        ("L'intrusion est punie par l'article 312.", ("312",), None),
        ("Selon l'article 312, le bailleur ne peut pas entrer.", ("312",), None),
        ("L'article premier du Code pénal impose la loi.", ("1",), "Code pénal"),
        ("Les articles 312 à 315 du Code pénal punissent.", ("312", "313", "314", "315"), "Code pénal"),
        ("Le Code pénal, en son article 312, punit.", ("312",), "Code pénal"),
        ("l'art. 37 du Code du travail fixe la durée.", ("37",), "Code du travail"),
        ("Voir l'article 39 alinéa 2 du Code du travail.", ("39",), "Code du travail"),
        ("l'article 39, alinéa 2, du Code du travail", ("39",), "Code du travail"),
        ("l'alinéa 2 de l'article 39 du Code du travail", ("39",), "Code du travail"),
        ("article 39(2) du Code du travail", ("39",), "Code du travail"),
        ("Section 312 of the Penal Code punishes trespass.", ("312",), "Penal Code"),
        ("Articles 5 and 6 of the Labour Code apply.", ("5", "6"), "Labour Code"),
        ("L'article 1737 et l'article 1736 du Code civil.", ("1737", "1736"), None),
        ("les articles 36, 37 et 39 du Code du travail", ("36", "37", "39"), "Code du travail"),
        ("aux articles 36 et suivants du Code du travail", ("36",), "Code du travail"),
        ("l'article 12 bis de la loi n° 2016/007", ("12 BIS",), "loi n° 2016/007"),
        # Revue du tour 2 : un nom en minuscule, AVANT le numéro, n'était pas lu.
        ("Le décret d'application, en son article 40, fixe les modalités.", ("40",),
         "décret d'application"),
    ])
    def test_les_formes(self, phrase, numeros, nom):
        (mention,) = mentions_d_articles(phrase)

        assert mention.numeros == numeros
        if nom is not None:
            assert mention.nom == nom

    @pytest.mark.parametrize("phrase", [
        "la section 2 du chapitre 3 du Code",
        "Le Code du travail protège le salarié.",
        "Ces articles de presse parlent du bail.",
    ])
    def test_ce_qui_n_est_pas_une_mention_d_article(self, phrase):
        assert mentions_d_articles(phrase) == []

    def test_le_nom_s_arrete_avant_la_citation_suivante(self):
        mentions = mentions_d_articles(
            "l'article 5 de la loi n° 2016/007 et l'article 6 du décret n° 2018/1 fixe le délai"
        )

        assert [(m.numeros, m.nom) for m in mentions] == [
            (("5",), "loi n° 2016/007"), (("6",), "décret n° 2018/1"),
        ]

    def test_une_quantite_n_est_pas_un_article(self):
        """« l'article 312, 3 ans de prison » ne cite pas l'article 3 ; « 2° » non plus."""
        assert [m.numeros for m in mentions_d_articles("Selon l'article 312, 3 ans de prison.")] == [("312",)]
        assert [m.numeros for m in mentions_d_articles("l'article 2, 3° du décret")] == [("2",)]

    def _extraits(self):
        return [
            ChunkResult(article_id=1, law_id=1, number="299", content=CP_299, reference="PRC-4722",
                        law_title=TITRE_CODE_PENAL, status="published"),
            ChunkResult(article_id=2, law_id=2, number="1736", content=CC_1736, reference="CODE-CIVIL",
                        law_title=TITRE_CODE_CIVIL, status="published"),
        ]

    @pytest.mark.parametrize("phrase, manquants", [
        ("L'article 299 du Code pénal punit l'intrusion.", []),
        ("Le Code pénal, en son article 299, punit.", []),
        ("Les articles 299 et 1736 encadrent la question.", []),
        ("L'article 1736 du Code civil règlent le congé.", []),
        ("Articles 299 and 312 of the Penal Code apply.", ["312"]),
        ("L'article 299 du Code civil punit l'intrusion.", ["299"]),
        ("L'article 1736 du Code civil fixe l'usage. L'article 1737 du même code.", ["1737"]),
        ("à l'article 35.1 du Code du travail", ["35.1"]),
        # Sans son nom, le numéro passait : l'article 299 du Code pénal le portait.
        ("Le décret d'application, en son article 299, fixe les peines.", ["299"]),
    ])
    def test_porte_ou_non(self, phrase, manquants):
        signales = [n for _, ns in numeros_non_portes(phrase, self._extraits()) for n in ns]

        assert signales == manquants

    def test_textes_nommes_et_droit_etranger(self):
        extraits = self._extraits()

        assert textes_nommes_non_portes("Le Code de la famille et la Constitution.", extraits) == [
            "Code de la famille", "Constitution",
        ]
        assert textes_nommes_non_portes("Le Code pénal et le Code civil.", extraits) == []
        assert droit_etranger("Le code civil français, la loi française, French law.") == [
            "code civil français", "loi française", "French law",
        ]


# ==================== 5. LA RÉPONSE DU MODÈLE ====================


class TestLaReponseDuModele:

    @pytest.mark.parametrize("reponse", [
        {"response": json.dumps(article_du_modele()), "fin": "MAX_TOKENS", "tronquee": True},
        {"response": '{"title": "Expulsion", "summary": "Selon l\'article', "fin": "STOP"},
        {"response": "Voici l'article demandé.", "fin": "STOP"},
        {"response": json.dumps([article_du_modele()]), "fin": "STOP"},
        {"response": json.dumps({k: v for k, v in article_du_modele().items() if k != "summary"})},
        {"response": json.dumps(article_du_modele(title="**", key_points=[]))},
        {"response": json.dumps(article_du_modele(key_points=[{"point": "Un objet"}]))},
        {"response": ""},
    ], ids=[
        "coupee", "json-coupe", "prose", "liste", "champ-absent", "vide-apres-nettoyage",
        "point-objet", "vide",
    ])
    async def test_jamais_un_faux_article(self, db_session, textes, reponse):
        llm = AsyncMock()
        llm.generate = AsyncMock(return_value=reponse)
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        with pytest.raises(ReponseInvalideError) as erreur:
            await _service(db_session, recherche, llm).rediger(SUJET)

        assert (erreur.value.statut, erreur.value.code) == (502, "reponse_invalide")

    async def test_les_balises_de_code_autour_du_json_sont_tolerees(self, db_session, textes):
        llm = AsyncMock()
        llm.generate = AsyncMock(return_value={
            "response": "```json\n" + json.dumps(article_du_modele()) + "\n```", "fin": "STOP",
        })
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        article = await _service(db_session, recherche, llm).rediger(SUJET)

        assert article.title == "Expulsion sans juge : ce que dit la loi"


class TestLesErreursDuModele:

    @pytest.mark.parametrize("erreur, attendue, statut, delai", [
        (MistralQuotaError("Quota mensuel"), QuotaError, 429, 60),
        (GeminiQuotaError("429"), QuotaError, 429, 60),
        (MistralOverloadedError("503"), SaturationError, 503, 10),
        (GeminiOverloadedError("503"), SaturationError, 503, 10),
        (GeminiBudgetEpuiseError("réflexion"), ReponseInvalideError, 502, None),
        (MistralServiceError("HTTP 401 Invalid API Key"), ReponseInvalideError, 502, None),
    ])
    async def test_traduites(self, db_session, textes, erreur, attendue, statut, delai):
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        with pytest.raises(attendue) as leve:
            await _service(db_session, recherche, modele_double(side_effect=erreur)).rediger(SUJET)

        assert (leve.value.statut, leve.value.retry_after) == (statut, delai)
        # Jamais le message du fournisseur : il peut porter une clé ou un détail interne.
        assert "Invalid API Key" not in str(leve.value)


# ==================== 6. TRANSACTION ET JOURNAL ====================


class TestTransactionEtJournal:

    async def test_la_transaction_est_close_avant_le_modele(self, db_session, textes):
        """La génération peut prendre deux minutes : sans connexion tenue pendant ce temps."""
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))
        vues = []

        async def generer(**_):
            vues.append(db_session.in_transaction())
            return {"response": json.dumps(article_du_modele()), "fin": "STOP"}

        llm = AsyncMock()
        llm.generate = AsyncMock(side_effect=generer)
        await db_session.execute(text("SELECT 1"))

        await _service(db_session, recherche, llm).rediger(SUJET)

        assert vues == [False]

    def test_la_recherche_du_blog_remplace_bien_l_ecriture_du_journal(self):
        """Si SearchService renomme cette méthode, la surcharge ne sert plus à rien."""
        assert callable(getattr(SearchService, "_ecrire_apres_la_reponse", None))
        assert RechercheDuBlog._ecrire_apres_la_reponse is not SearchService._ecrire_apres_la_reponse

    async def test_de_bout_en_bout_sur_la_vraie_recherche_sans_journal(self, db_session, textes):
        """
        La VRAIE recherche du blog (exacte, sans cache), sur la base de test :
        l'article dont le vecteur est celui du sujet est retenu, celui qui porte
        tous les mots-clés aussi ; le texte en brouillon jamais ; et aucune ligne
        n'est écrite dans search_events.
        """
        dim = EmbeddingService.EMBEDDING_DIM
        sujet = np.zeros(dim, dtype=np.float32)
        sujet[0] = 1.0
        loin = np.zeros(dim, dtype=np.float32)
        loin[1] = 1.0
        for identifiant, vecteur in (
            (textes.cp_299, sujet), (textes.projet_5, sujet), (textes.cp_322, loin),
            (textes.cc_1736, loin),
        ):
            await db_session.execute(
                update(Article).where(Article.id == identifiant).values(embedding=vecteur.tolist())
            )
        await db_session.commit()

        class _Sujet:
            TASK_QUERY = EmbeddingService.TASK_QUERY
            TASK_DOCUMENT = EmbeddingService.TASK_DOCUMENT

            async def generate_embedding_async(self, texte, task_type=None):
                return sujet

        recherche = RechercheDuBlog(db_session)
        recherche.embedding_service = _Sujet()
        llm = modele_double(article_du_modele(citations=[{"extrait": 1, "objet": "Punie."}]))

        article = await _service(db_session, recherche, llm).rediger(SUJET)

        lus = {s["article_id"] for s in article.sources}
        assert textes.cp_299 in lus
        assert textes.cc_1751_1 in lus
        assert textes.projet_5 not in lus
        assert textes.cp_322 not in lus and textes.cc_1736 not in lus
        assert await ecritures_en_attente(10) == 0
        evenements = (await db_session.execute(text("SELECT count(*) FROM search_events"))).scalar()
        assert evenements == 0


# ==================== 7. CITATIONS VÉRIFIÉES EN BASE ====================


class TestCitationsVerifiees:

    async def test_par_numero_un_morceau_donne_son_article(self, db_session, textes):
        citations, refus = await citations_verifiees(db_session, [
            DemandeDeCitation(textes.code_civil, article_number="1751", title="Rachat."),
            DemandeDeCitation(textes.code_penal, article_number="Article 299"),
        ])

        assert refus == []
        assert [c["article_id"] for c in citations] == [textes.cc_1751_1, textes.cp_299]
        assert citations[0]["article_number"] == "1751"
        assert citations[0]["excerpt"].startswith("L'acquéreur à pacte de rachat")
        assert "SECT. II" not in citations[0]["excerpt"]

    async def test_un_numero_porte_par_deux_lignes_suit_l_ordre_du_texte(self, db_session, textes):
        """
        LIMITE CONNUE, écrite ici pour qu'elle se voie. Le Code pénal du corpus a
        deux lignes « 1 » : sa table des matières, puis l'article premier. Par
        son seul numéro, la citation prend la première, comme l'aperçu d'un
        article (routes/laws._choisir). L'aperçu de l'administration montre
        l'extrait ; `article_id` désigne la bonne ligne.
        """
        (citation,), _ = await citations_verifiees(
            db_session, [DemandeDeCitation(textes.code_penal, article_number="1er")]
        )

        assert citation["article_id"] == textes.cp_1_contenu

    async def test_l_extrait_garde_le_vrai_debut_de_l_article(self, db_session, textes):
        """Code de procédure pénale, art. 8 : la première ligne coupée par l'OCR est gardée."""
        (citation,), _ = await citations_verifiees(db_session, [DemandeDeCitation(
            textes.code_de_procedure_penale, article_id=textes.cpp_8_1
        )])

        assert citation["excerpt"].startswith(
            "(1) Toute personne suspectée d'avoir commis une infraction est présumée innocente"
        )
        assert citation["excerpt"].endswith(" […]")

    @pytest.mark.parametrize("demande, raison", [
        (lambda t: DemandeDeCitation(999_999, article_number="1"), "n'existe pas"),
        (lambda t: DemandeDeCitation(t.projet, article_number="5"), "n'est pas publié"),
        (lambda t: DemandeDeCitation(t.code_penal, article_number="9999"), "n'a pas d'article 9999"),
        (lambda t: DemandeDeCitation(t.code_penal, article_id=t.cc_1736), "n'appartient pas"),
        (lambda t: DemandeDeCitation(t.code_penal, article_id=t.cp_299, article_number="286"),
         "ne correspond à aucun article"),
        (lambda t: DemandeDeCitation(t.code_civil, article_id=t.cc_preambule), "n'est pas un article"),
        (lambda t: DemandeDeCitation(t.code_civil, article_number="PREAMBULE"), "n'a pas d'article"),
        (lambda t: DemandeDeCitation(t.code_civil), "indiquez l'article"),
    ])
    async def test_refusees(self, db_session, textes, demande, raison):
        with pytest.raises(CitationInvalide) as erreur:
            await citations_verifiees(db_session, [demande(textes)])

        assert raison in str(erreur.value)
        assert str(erreur.value).startswith("Citation 1 : ")

    async def test_sans_rigueur_les_refus_sont_rendus(self, db_session, textes):
        citations, refus = await citations_verifiees(db_session, [
            DemandeDeCitation(textes.projet, article_number="5"),
            DemandeDeCitation(textes.code_penal, article_id=textes.cp_299),
        ], strict=False)

        assert [c["article_id"] for c in citations] == [textes.cp_299]
        assert refus[0][0] == 1 and "n'est pas publié" in refus[0][1]


# ==================== 8. OUTILS ====================


class TestOutils:

    @pytest.mark.parametrize("titre, slug", [
        ("Licenciement abusif : ce que dit l'article 34 ?", "licenciement-abusif-ce-que-dit-l-article-34"),
        ("Bail/Expulsion Été", "bail-expulsion-ete"),
        ("Coupure d'eau : bailleur/locataire ? #droits", "coupure-d-eau-bailleur-locataire-droits"),
        ("Main-d'œuvre et cœur du métier", "main-d-oeuvre-et-coeur-du-metier"),
        ("  --Déjà--un--slug--  ", "deja-un-slug"),
    ])
    def test_slugify(self, titre, slug):
        assert slugify(titre) == slug
        assert SLUG_VALIDE.fullmatch(slug)

    def test_slugify_coupe_entre_deux_mots_a_80(self):
        slug = slugify("expulsion " * 20)

        assert len(slug) <= 80
        assert slug.endswith("expulsion")
        assert SLUG_VALIDE.fullmatch(slug)

    def test_slugify_sans_lettre(self):
        assert slugify("???").startswith("article-")
        assert SLUG_VALIDE.fullmatch(slugify("???"))
        assert slugify("???", repli=False) == ""

    def test_texte_brut(self):
        assert texte_brut("**Gras** et __souligné__\n# Titre\n- puce\n`code`") == (
            "Gras et souligné\nTitre\npuce\ncode"
        )

    @pytest.mark.parametrize("numero, affiche", [
        ("1751.1", "1751"), ("35.12", "35"), ("1er", "1er"), ("L 94 bis", "L 94 bis"),
        ("299", "299"), (None, ""),
    ])
    def test_numero_affiche(self, numero, affiche):
        assert numero_affiche(numero) == affiche

    def test_extrait_verbatim_ne_retire_rien_en_silence(self):
        # L'intitulé en tête est gardé : c'est le début du texte en base.
        assert extrait_verbatim(CP_322).startswith("1.-Filouteries de loyers (1) Est puni")
        assert extrait_verbatim(CPP_8).startswith("(1) Toute personne suspectée")
        assert extrait_verbatim("La loi pénale s'impose à tous.") == "La loi pénale s'impose à tous."
        long = extrait_verbatim("mot " * 200)
        assert long.endswith(" […]") and len(long) <= 305
        # La suite d'un article : la coupe en tête est dite.
        assert extrait_verbatim("(6) Le salaire.", suite=True) == "[…] (6) Le salaire."

    def test_decouper_les_articles_colles_et_la_suite(self):
        assert [(n, s, p) for n, _, _, s, p in decouper(CT_35_1, "35.1", None, False)] == [
            ("35", False, True), ("38", False, False), ("39", False, False),
        ]
        assert [(n, s) for n, _, _, s, _ in decouper(CT_35_2, "35.2", "39", False)] == [
            ("39", True), ("40", False),
        ]
        # Suite d'article sans morceau précédent connu, ou dans un code annoté : écartée.
        assert [n for n, *_ in decouper(CT_35_2, "35.2", None, False)] == ["40"]
        assert [n for n, *_ in decouper(CT_35_2, "35.2", "39", True)] == ["40"]

    def test_le_debut_repete_par_le_decoupage(self):
        corps = debut_du_corps(CT_67_2, CT_67_1)

        assert CT_67_2[corps:].startswith("(3) En cas de résiliation")
        assert [(n, s) for n, _, _, s, _ in decouper(CT_67_2, "67.2", "68", False, corps)] == [
            ("68", True),
        ]
        assert debut_du_corps(CT_35_2, CT_35_1) == 0

    @pytest.mark.parametrize("contenu", [
        # Une loi modificative CITE l'article qu'elle réécrit.
        "Les dispositions de l'article 5 sont modifiées : Article 5 (nouveau).- Le texte nouveau.",
        # Une ligne budgétaire d'un décret d'avancement.
        "La dépense est imputée au budget de l'État. Article 390000 : traitements.",
        # Un sommaire du Code CIMA : « 434-8 » est un numéro, pas un en-tête.
        "| 2. Article 309 | 21. Article 434-8 | 40. Article 548 |",
    ])
    def test_ce_qui_n_est_pas_un_article_colle(self, contenu):
        assert entetes_colles(contenu, "1") == []

    def test_les_annotations_et_la_section_suivante(self):
        assert decouper(CC_1736, "1736", None, True)[0][2] == CC_1736.index("\n\nJugements")
        assert CC_1751_1[:decouper(CC_1751_1, "1751.1", None, True)[0][2]].endswith("incommutable.")

    def test_duree_de_lecture(self):
        assert duree_de_lecture(["mot " * 950]) == "5 min"
        assert duree_de_lecture([""]) == "1 min"

    @pytest.mark.parametrize("nom, rubrique", [
        ("Immobilier", "Immobilier"), ("Immobilier & Logement", "Immobilier"),
        ("droit pénal", "Droit pénal"), ("Économie & Fiscalité", "Fiscalité"),
        ("Astrologie", "Actualité juridique"), ("", "Actualité juridique"),
    ])
    def test_rubrique_connue(self, nom, rubrique):
        assert rubrique_connue(nom) == rubrique


def test_les_reglages_du_contrat():
    """150 s avant la 504 « delai », huit extraits comme le chat."""
    assert DELAI_GENERATION_S == 150
    assert EXTRAITS_AU_PLUS == 8


def test_les_formes_reelles_de_la_base_sont_bien_celles_des_tests():
    """Garde-fou : les données de test portent les formes que les tests disent porter."""
    assert "Article 40.-" in CT_35_2 and CT_35_2.startswith("(6) Le salaire")
    assert CT_67_2.startswith(CT_67_1[:150]) and "Article 68.-" in CT_67_2.split("\n")[0]
    assert "présumée\ninnocente" in CPP_8
    assert "Article 683 —" in CPP_679_2 and "Article 684 —" in CPP_679_2


# ==================== 9. TOUR 3 : CE QUE LA REVUE DU TOUR 2 A TROUVÉ ====================


def _ligne_reelle(identifiant: int) -> dict:
    return next(x for x in lignes_reelles()["lignes"] if x["id"] == identifiant)


def _extrait_reel(ids: dict, identifiant: int, similarite: float = 0.6) -> RechercheDoublee:
    """La recherche doublée, qui rend la ligne réelle `identifiant` (créée par creer_les_lignes_reelles)."""
    ligne = _ligne_reelle(identifiant)
    trouve = ChunkResult(
        article_id=ids[identifiant], law_id=ids[-ligne["law_id"]], number=ligne["number"],
        content=ligne["content"], reference=f"REEL-{ligne['law_id']}",
        law_title=lignes_reelles()["textes"][str(ligne["law_id"])], status="published", language="fr",
    )
    return RechercheDoublee([trouve], {ids[identifiant]: similarite})


class TestTextesModificatifs:
    """
    Revue du tour 2 (r2) : un texte modificatif CITE la nouvelle rédaction d'un
    article d'un autre texte (« ainsi qu'il suit : - «ARTICLE 2.- (nouveau) … »).
    Cet en-tête devenait un article du texte modificatif, sous une carte
    « vérifiée » : 7 en-têtes dans 7 des 65 textes modificatifs de jurix_dev.
    """

    @pytest.mark.parametrize("identifiant", [56289, 58734])
    def test_les_lignes_reelles(self, identifiant):
        """Décret 2021/597 (« (nouveau) » après le séparateur) ; loi 2020/009 (« (4) (nouveau) »)."""
        ligne = _ligne_reelle(identifiant)

        assert entetes_colles(ligne["content"], ligne["number"]) == []

    @pytest.mark.parametrize("contenu", [
        "Sont modifiées ainsi qu'il suit :\n\n- «ARTiCLE 2.-(1) Le classement vaut expropriation. »",
        "Modifiées comme suit : ARTICLE 49.- (1) Il existe des établissements.",
        "LE PRESIDENT DE LA REPUBLIQUE, ARTICLE 6.- (nouveau) (1) Sont assujetties les entreprises.",
        "L'article 3 est ainsi rédigé : Article 3.- Le texte nouveau.",
        'Il se lit désormais comme suit : "Article 4.- Le texte nouveau."',
    ], ids=["guillemet", "comme-suit", "nouveau-apres", "ainsi-redige", "guillemet-droit"])
    def test_les_formes_d_une_citation(self, contenu):
        assert entetes_colles(contenu, "1") == []

    def test_un_vrai_article_colle_reste_lu(self):
        """Ni un guillemet fermant, ni des guillemets dans la phrase d'avant n'annoncent une citation."""
        assert [n for _, n in entetes_colles(CT_35_1, "35.1")] == ["38", "39"]
        contenu = "La mention « pour solde de tout compte » est nulle. Article 5.- Le suivant."
        assert [n for _, n in entetes_colles(contenu, "4")] == ["5"]
        assert not est_un_article_cite(contenu, contenu.index("Article 5"), contenu.index(" Le suivant"))

    @pytest.mark.parametrize("titre, modificatif", [
        ("Décret N°2021/597 du 13 octobre 2021 modifiant et complétant certaines dispositions", True),
        ("Loi n° 2020/009 du 20 juillet 2020 modifiant et complétant certaines dispositions", True),
        ("Décret portant modification du décret n° 2004/073", True),
        ("Loi N°2012/001 du 19 avril 2012 portant Code électoral, modifiée et complétée par la loi "
         "n°2012/017 du 21 décembre 2012", False),
        ("Loi N°92/007 du 14 août 1992 portant Code du travail", False),
        (None, False),
    ])
    def test_un_texte_modificatif_se_lit_a_son_titre(self, titre, modificatif):
        assert est_modificatif(titre) is modificatif

    def test_dans_un_texte_consolide_nouveau_marque_son_propre_article(self):
        """
        Sonde du tour 3 sur jurix_dev : au Code électoral, texte CONSOLIDÉ, « Art.
        166 - (nouveau) » est son article 166, collé dans la ligne 165. Écarté,
        il versait son texte dans la carte de l'article 165.
        """
        ligne = _ligne_reelle(91549)

        assert [n for _, n in entetes_colles(ligne["content"], "165", modificatif=False)] == ["166"]
        # Dans un texte modificatif, la même marque dit une citation.
        assert entetes_colles(ligne["content"], "165", modificatif=True) == []
        # Une citation annoncée en est une partout.
        annonce = "Modifiées ainsi qu'il suit : « Article 5.- Le texte nouveau. »"
        assert entetes_colles(annonce, "1", modificatif=False) == []

    async def test_l_article_du_code_electoral_garde_sa_carte(self, db_session):
        ids = await creer_les_lignes_reelles(db_session, [91549])
        texte = ids[-2282]

        (article_166,), _ = await citations_verifiees(db_session, [DemandeDeCitation(texte, article_number="166")])
        (article_165,), _ = await citations_verifiees(db_session, [DemandeDeCitation(texte, article_id=ids[91549])])

        assert article_166["article_id"] == ids[91549]
        assert article_166["excerpt"].startswith("Art. 166 - (nouveau) (1) Le candidat titulaire")
        assert article_165["article_number"] == "165"
        assert "Art. 166" not in article_165["excerpt"]

    async def test_la_citation_garde_le_numero_du_texte_modificatif(self, db_session):
        ids = await creer_les_lignes_reelles(db_session, [56288, 56289, 56290])

        (citation,), _ = await citations_verifiees(
            db_session, [DemandeDeCitation(ids[-663], article_id=ids[56289])]
        )
        (par_le_numero,), _ = await citations_verifiees(
            db_session, [DemandeDeCitation(ids[-663], article_number="2")]
        )
        with pytest.raises(CitationInvalide):
            await citations_verifiees(
                db_session, [DemandeDeCitation(ids[-663], article_id=ids[56289], article_number="2")]
            )

        assert (citation["article_number"], citation["suite"]) == ("1", False)
        assert "«ARTICLE 2.- (nouveau)" in citation["excerpt"]
        # L'article 2 du décret 2021/597 est le sien, et rien d'autre.
        assert par_le_numero["excerpt"].startswith("Le présent décret sera enregistré")

    async def test_la_generation_le_donne_comme_l_article_1(self, db_session):
        ids = await creer_les_lignes_reelles(db_session, [56288, 56289, 56290])
        llm = modele_double(article_du_modele(
            summary=("Selon l'article 2 du Décret N°2021/597 du 13 octobre 2021, la juridiction du "
                     "consulat couvre la Sangha Mbaéré."),
            citations=[{"extrait": 1, "objet": "La juridiction du consulat."}],
        ))

        article = await _service(db_session, _extrait_reel(ids, 56289), llm).rediger(
            "Le consulat du Cameroun à Bouar"
        )

        prompt = _prompt(llm)
        assert "— Article 1\n" in prompt and "[2]" not in prompt
        assert [c["article_number"] for c in article.citations] == ["1"]
        assert any("article 2" in a.message and "Résumé" in a.message for a in article.avertissements)


class TestSuitesDouteuses:
    """
    Revue du tour 2 (r3) : l'en-tête d'un article abîmé par l'OCR à la fin d'un
    morceau, et la tête du morceau suivant rattachée au dernier article LU. Code
    du travail : la tête de « 67.3 », l'alinéa (2) de l'article 73, citée
    « article 72 (suite) », vérifiée ; loi sur la cybersécurité : la tête de
    « 52.3 » (article 57 (2)) citée « article 56 (suite) ». Dans le doute,
    écarter.
    """

    @pytest.mark.parametrize("lignes, tete, law_id, premier, faux", [
        ([97434, 97435, 97436], 97436, 2274, "74", "72"),
        ([97293, 97294, 97295], 97295, 2271, "58", "56"),
        # Même forme, rattachement juste (article 162), perdu aussi : c'est le prix du doute.
        ([97475, 97476], 97476, 2274, "164", "162"),
    ], ids=["code-du-travail-67.3", "cybersecurite-52.3", "code-du-travail-161.2"])
    async def test_la_tete_est_ecartee(self, db_session, lignes, tete, law_id, premier, faux):
        ids = await creer_les_lignes_reelles(db_session, lignes)

        (citation,), _ = await citations_verifiees(
            db_session, [DemandeDeCitation(ids[-law_id], article_id=ids[tete])]
        )
        with pytest.raises(CitationInvalide):
            await citations_verifiees(
                db_session, [DemandeDeCitation(ids[-law_id], article_id=ids[tete], article_number=faux)]
            )

        assert (citation["article_number"], citation["suite"]) == (premier, False)

    async def test_la_generation_ne_la_donne_pas(self, db_session):
        ids = await creer_les_lignes_reelles(db_session, [97434, 97435, 97436])
        llm = modele_double(article_du_modele(
            summary="L'action en paiement du salaire se prescrit par trois ans.",
            citations=[{"extrait": 1, "objet": "La prescription du salaire."}],
        ))

        article = await _service(db_session, _extrait_reel(ids, 97436), llm).rediger(
            "La prescription de l'action en paiement du salaire"
        )

        prompt = _prompt(llm)
        assert "Article 72 (suite" not in prompt and "L'assistance judiciaire" not in prompt
        assert "— Article 74\n" in prompt
        assert [c["article_number"] for c in article.citations] == ["74"]

    async def test_une_suite_qui_enchaine_reste_citee(self, db_session, textes):
        """« 679.2 » enchaîne sur l'article 683 : sa tête est bien la suite de l'article 682."""
        (citation,), _ = await citations_verifiees(db_session, [DemandeDeCitation(
            textes.code_de_procedure_penale, article_id=textes.cpp_679_2
        )])

        assert (citation["article_number"], citation["suite"]) == ("682", True)
        assert citation["excerpt"].startswith("[…] (4) Si le condamné prétend")


class TestRenvoisAuxExtraits:
    """Essai réel du 30/10 : « (extrait 2) », « (extraits 5 et 9) » dans les points clés publiés."""

    @pytest.mark.parametrize("texte, propre", [
        ("Le préavis non respecté est dû (extrait 2).", "Le préavis non respecté est dû."),
        ("Le juge fixe le montant (extraits 5 et 9), selon l'ancienneté.",
         "Le juge fixe le montant, selon l'ancienneté."),
        ("Voir (cf. extrait 3) et (Extraits n° 1, 2 et 4) ici.", "Voir et ici."),
        ("Le bail se termine selon le Code civil [2].", "Le bail se termine selon le Code civil."),
        ("Le délai est fixé (voir les extraits 2-3) par l'usage.", "Le délai est fixé par l'usage."),
        # Rien d'autre ne change : l'espace avant « : » reste, une parenthèse ordinaire aussi.
        ("Attention : le délai (un mois) court ; voilà.", "Attention : le délai (un mois) court ; voilà."),
    ])
    def test_sans_renvois_aux_extraits(self, texte, propre):
        assert sans_renvois_aux_extraits(texte) == propre

    async def test_retires_a_la_generation_et_un_point_par_ligne(self, db_session, textes):
        modele = article_du_modele(
            key_points=["Entrer chez autrui est puni (extrait 1).", "Le congé suit\nl'usage (extraits 2 et 3)."],
            citations=[{"extrait": 1, "objet": "La violation de domicile est punie (extrait 1)."}],
        )
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        article = await _service(db_session, recherche, modele_double(modele)).rediger(SUJET)

        assert article.key_points == ["Entrer chez autrui est puni.", "Le congé suit l'usage."]
        assert article.citations[0]["title"] == "La violation de domicile est punie."

    async def test_un_renvoi_dans_la_phrase_est_signale(self, db_session, textes):
        modele = article_du_modele(citizen_impact="Comme le dit l'extrait 2, le congé suit l'usage.")
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        article = await _service(db_session, recherche, modele_double(modele)).rediger(SUJET)

        assert any(
            "Impact pour le citoyen" in a.message and "« l'extrait 2 »" in a.message
            for a in article.avertissements
        )


class TestLesConsignesDuTour3:

    def test_le_prompt(self):
        assert "N'écris jamais de renvoi aux numéros des extraits" in SYSTEM_PROMPT
        assert "Une règle ne vaut que pour le cas que vise son extrait" in SYSTEM_PROMPT
        assert "de qui doit prouver, de qui décide (un juge ou non)" in SYSTEM_PROMPT
        assert "n'affirme jamais qu'une chose est possible sans juge" in SYSTEM_PROMPT

    async def test_le_pluriel_des_mentions(self, db_session, textes):
        modele = article_du_modele(summary="Les articles 286 et 312 du Code pénal punissent aussi l'intrusion.")
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        article = await _service(db_session, recherche, modele_double(modele)).rediger(SUJET)

        assert any("les articles 286 et 312 ne figurent dans aucun" in a.message for a in article.avertissements)


# ==================== TOUR 4 : LES SOURCES ET LA PROSE ====================


class TestLesSourcesDUnArticleEcrit:
    """
    POST /{id}/sources (tour 4) : les sources d'un article qui n'en a pas
    (article d'avant, ou écrit à la main), par la MÊME recherche que la
    génération, sans le modèle.
    """

    async def test_la_meme_recherche_que_la_generation_sans_le_modele(self, db_session, textes):
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))
        llm = modele_double()
        genere = await _service(db_session, recherche).rediger(SUJET)

        sources = await _service(db_session, recherche, llm).sources_du_sujet(SUJET)

        assert sources == genere.sources
        assert [s["article_number"] for s in sources] == ["299", "1736", "1751"]
        llm.generate.assert_not_awaited()
        # La recherche de la génération : mots-clés pour le plein texte, sujet entier pour le sens.
        assert recherche.requetes[-1].semantic_query == SUJET
        assert recherche.requetes[-1].filters.status == "published"

    async def test_sans_texte_pertinent_sans_source(self, db_session, textes):
        recherche = RechercheDoublee(extraits_du_sujet(textes), {textes.cp_299: 0.2})

        with pytest.raises(SansSourceError) as erreur:
            await _service(db_session, recherche).sources_du_sujet("Recette de cuisine du ndolé")

        assert erreur.value.statut == 422
        assert "les sources n'ont pas été reconstruites, rien n'a changé" in str(erreur.value)

    async def test_recherche_en_panne(self, db_session, textes):
        recherche = RechercheDoublee([], erreur=RuntimeError("base injoignable"))

        with pytest.raises(RechercheIndisponibleError):
            await _service(db_session, recherche).sources_du_sujet(SUJET)

    @pytest.mark.parametrize("titre, contexte, sujet", [
        ("Le congé d'un bail verbal", "Un bailleur veut reprendre son logement.",
         "Le congé d'un bail verbal. Un bailleur veut reprendre son logement."),
        ("Expulsion sans juge : quels cas ?", "Un locataire trouve ses affaires dehors.",
         "Expulsion sans juge : quels cas ? Un locataire trouve ses affaires dehors."),
        ("**Le bail**", "Le congé\n\nsuit l'usage (extrait 2).", "Le bail. Le congé suit l'usage."),
        ("", "Seulement un contexte.", "Seulement un contexte."),
        ("Seulement un titre", "", "Seulement un titre"),
    ])
    def test_le_sujet_est_le_titre_puis_le_contexte(self, titre, contexte, sujet):
        assert sujet_de_l_article(titre, contexte) == sujet

    def test_le_sujet_garde_la_longueur_d_un_sujet_de_generation(self):
        sujet = sujet_de_l_article("Un titre", "mot " * 500)

        assert len(sujet) <= 300 and sujet.endswith("mot")


class TestLaProseEstStable:
    """
    Revue du tour 3 : un renvoi « (extrait N) » retiré seul sur sa ligne, ou une
    cote en tête de ligne, laissait trois sauts de ligne ou une espace de tête ;
    le formulaire renvoyé les ramenait à la normale, et chaque enregistrement
    « modifiait » la partie, périmant le contrôle en boucle.
    """

    @pytest.mark.parametrize("texte", [
        "Premier paragraphe.\n\n(extrait 1)\n\nSecond paragraphe.",
        "(extrait 2) Le préavis est dû.\nLa suite (extraits 5 et 9).",
        "  **Gras** en tête.\n\n\n\n- une puce\n# un titre",
        "[3] Le bail.\n[4]\nLa fin.",
    ])
    def test_appliquee_deux_fois_elle_ne_change_plus(self, texte):
        une = prose_publique(texte)

        assert prose_publique(une) == une
        assert "\n\n\n" not in une
        assert all(ligne == ligne.strip() for ligne in une.splitlines())

    async def test_le_texte_genere_est_deja_stable(self, db_session, textes):
        modele = article_du_modele(
            context="Un locataire rentre.\n\n(extrait 1)\n\nSes affaires sont dehors.",
            summary="PRC-4722 La violation de domicile est punie.",
        )
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

        article = await _service(db_session, recherche, modele_double(modele)).rediger(SUJET)

        assert article.context == "Un locataire rentre.\n\nSes affaires sont dehors."
        assert article.summary == "La violation de domicile est punie."
        for texte in (article.context, article.summary):
            assert prose_publique(texte) == texte
