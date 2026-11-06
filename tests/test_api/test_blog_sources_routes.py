"""
L'avenant du tour 4 du blog, par les routes : le titre relu, les sources
gardées, leur reconstruction, la relecture liée au contrôle.

POURQUOI. Revue du tour 3 (data/blog/tour3-resultat.json, dépôt du backend) :
  - le TITRE partait en ligne sans contrôle ni relecture, alors que c'est la
    phrase la plus lue et, dans l'essai réel du 30/10, celle qui portait la
    règle fausse (« Expulsion locative sans jugement : quels cas au
    Cameroun ? ») ;
  - les SOURCES n'étaient rendues par aucune réponse d'administration : la
    relecture « contre les sources », obligatoire, devenait impossible dès
    qu'on rouvrait un brouillon ;
  - un ARTICLE D'AVANT ne pouvait jamais revenir en ligne (aucune source,
    citations jamais vérifiables) : seule issue, le supprimer, ce que la
    décision du propriétaire exclut (« gardés en base, corrigeables, remis en
    ligne par le nouveau parcours ») ;
  - on pouvait cocher la relecture, PUIS lancer le contrôle, et publier sans
    avoir vu ses verdicts.

Ce que ces tests protègent :
  1. Le titre : jugé, relu, et sa modification périme le contrôle ;
     chaque partie aussi, une par une.
  2. Les sources : enregistrées à la génération, rendues par chaque lecture
     d'administration, jamais au public.
  3. POST /{id}/sources : reconstruites par la recherche de la génération ;
     citations gardées si portées, retirées sinon (avec un avertissement) ;
     contrôle effacé si elles changent ; l'article repasse en brouillon ;
     ses erreurs ne changent rien.
  4. Un article d'avant n'est jamais public avant sources, citations,
     contrôle, relecture et publication — même avec des citations marquées
     « vérifiées » à la main (plus strict que le correctif 5bdcb4f de main).
  5. La relecture tient au contrôle : cochée avant, elle ne compte pas ; un
     nouveau verdict fait relire sa partie.
  6. Le passage copié par le contrôle est vérifié : un appui inventé ne
     soutient pas, une contradiction sans texte ne bloque pas.
  7. Chaque état non publiable refuse la publication (422, code = l'état) ;
     les codes d'erreur du contrôle ; le retrait exact ; le PUT d'un bloc.

Le modèle et la recherche sont doublés (tests/blog_donnees.py) : le vrai modèle
n'est jamais appelé.

Usage:
    pytest tests/test_api/test_blog_sources_routes.py -v
"""

import asyncio
import json

import pytest
from sqlalchemy import text

from app.api.routes import blog as routes_du_blog
from app.api.routes.blog import get_blog_generator, get_controleur
from app.main import app
from app.models.blog import BlogPost
from app.services.blog_generator_service import BlogGeneratorService, sujet_de_l_article
from app.services.controle_du_billet import PARTIES, PHRASES_AU_PLUS, ControleDuBillet
from app.services.mistral_service import (
    MistralOverloadedError,
    MistralQuotaError,
    MistralServiceError,
)
from app.services.search_service import CHEMIN_REPLI_HNSW
from tests.blog_donnees import (
    CITATIONS_D_AVANT,
    CP_299,
    SUJET,
    TITRE_CODE_PENAL,
    RechercheDoublee,
    appels_du_controle,
    article_d_avant,
    article_du_modele,
    creer_les_textes,
    extrait,
    extraits_du_prompt,
    extraits_du_sujet,
    jugement,
    modele_double,
    passage_de,
    phrases_du_prompt,
    relecture_cochee,
    similarites,
)

BLOG = "/api/v1/blog"
ORIGINE = {"Origin": "http://localhost:5173"}
TOUT_RELU = {"parties": {partie: True for partie in PARTIES}}
CLES_DES_SOURCES = {"n", "law_id", "law_title", "article_number", "article_id", "suite", "extrait"}


@pytest.fixture
async def textes(db_session):
    return await creer_les_textes(db_session)


@pytest.fixture
def generateur(db_session, textes):
    """La génération et la reconstruction des sources, doublées (même service, même recherche)."""
    class Reglage:
        llm = modele_double()
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

    app.dependency_overrides[get_blog_generator] = lambda: BlogGeneratorService(
        db_session, llm=Reglage.llm, recherche=Reglage.recherche
    )
    try:
        yield Reglage
    finally:
        app.dependency_overrides.pop(get_blog_generator, None)


@pytest.fixture(autouse=True)
def controleur(db_session):
    """Le contrôle de POST /{id}/controle, doublé : tout « soutenue » sauf réglage."""
    class Reglage:
        llm = modele_double()

    app.dependency_overrides[get_controleur] = lambda: ControleDuBillet(db_session, llm=Reglage.llm)
    try:
        yield Reglage
    finally:
        app.dependency_overrides.pop(get_controleur, None)


def _oublier_les_articles(db_session) -> None:
    """Un article modifié en SQL brut resterait périmé dans la session partagée."""
    for objet in list(db_session.identity_map.values()):
        if isinstance(objet, BlogPost):
            db_session.expire(objet)


async def _genere(admin_client) -> dict:
    reponse = await admin_client.post(f"{BLOG}/generate", json={"topic": SUJET})
    assert reponse.status_code == 201, reponse.text
    return reponse.json()


async def _admin(admin_client, post_id) -> dict:
    return (await admin_client.get(f"{BLOG}/admin/{post_id}")).json()


async def _publiable(admin_client) -> dict:
    post = (await _genere(admin_client))["post"]
    reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={"relecture": relecture_cochee(post)})
    assert reponse.json()["etat_publication"] == "publiable", reponse.json()
    return reponse.json()


async def _en_ligne(admin_client) -> dict:
    post = await _publiable(admin_client)
    reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={"is_published": True})
    assert reponse.status_code == 200, reponse.text
    return reponse.json()


async def _dossier(db_session, post_id) -> dict:
    return (await db_session.execute(
        text("SELECT citations FROM blog_posts WHERE id = :id"), {"id": post_id}
    )).scalar_one()


# ==================== 1. LE TITRE ====================


class TestLeTitre:

    async def test_le_titre_est_juge_le_premier_et_se_relit(self, admin_client, generateur):
        post = (await _genere(admin_client))["post"]

        (titre, *_) = post["controle_ia"]["phrases"]
        coche = await admin_client.put(
            f"{BLOG}/{post['id']}", json={"relecture": relecture_cochee(post, {"titre": True})}
        )

        assert (titre["id"], titre["partie"], titre["index"], titre["texte"]) == ("P1", "titre", 0, post["title"])
        assert list(post["relecture"]["parties"]) == list(PARTIES)
        assert post["relecture"]["parties"]["titre"] is False
        assert coche.json()["relecture"]["parties"]["titre"] is True

    async def test_un_titre_contredit_bloque_la_publication(self, admin_client, generateur):
        """L'essai réel du 30/10 : le titre présupposait une expulsion sans juge."""
        titre = "Expulsion locative sans jugement : quels cas au Cameroun ?"
        generateur.llm = modele_double(article_du_modele(title=titre), controle={titre: "contredite"})
        post = (await _genere(admin_client))["post"]

        reponse = await admin_client.put(
            f"{BLOG}/{post['id']}", json={"relecture": relecture_cochee(post), "is_published": True}
        )

        assert post["controle_ia"]["phrases"][0]["verdict"] == "contredite"
        assert reponse.status_code == 422
        assert reponse.json()["code"] == "phrases_contredites"
        assert f"titre : « {titre} »" in reponse.json()["detail"]
        assert (await _admin(admin_client, post["id"]))["isPublished"] is False


# ==================== 2. CHAQUE PARTIE PÉRIME LE CONTRÔLE ====================


def _modification(post, partie):
    analyse = post["botAnalysis"]
    if partie == "explications":
        return {"bot_analysis": {"citations": [
            {**analyse["citations"][0], "title": "Une explication réécrite."}, *analyse["citations"][1:],
        ]}}
    return {
        "titre": {"title": "Un autre titre"},
        "contexte": {"context": "Un autre contexte."},
        "resume": {"bot_analysis": {"summary": "Un autre résumé."}},
        "points_cles": {"bot_analysis": {"keyPoints": ["Un seul point, réécrit."]}},
        "impact": {"bot_analysis": {"citizenImpact": "Un autre impact."}},
        "question": {"suggested_question": "Une autre question ?"},
    }[partie]


_NOMS = {
    "titre": "titre", "contexte": "contexte", "resume": "résumé", "points_cles": "points clés",
    "impact": "impact pour le citoyen", "explications": "explications des citations",
    "question": "question suggérée",
}


class TestChaquePartiePerimeLeControle:

    @pytest.mark.parametrize("partie", PARTIES)
    async def test_un_brouillon(self, admin_client, generateur, partie):
        post = await _publiable(admin_client)

        modifie = await admin_client.put(f"{BLOG}/{post['id']}", json=_modification(post, partie))
        publie = await admin_client.put(f"{BLOG}/{post['id']}", json={"is_published": True})

        assert modifie.status_code == 200, modifie.text
        assert modifie.json()["etat_publication"] == "controle_a_relancer"
        assert modifie.json()["relecture"]["parties"] == {p: p != partie for p in PARTIES}
        # Le contrôle d'avant reste lisible, périmé.
        assert modifie.json()["controle_ia"] == post["controle_ia"]
        assert publie.status_code == 422 and publie.json()["code"] == "controle_a_relancer"
        assert f"parties modifiées : {_NOMS[partie]})" in publie.json()["detail"]

    @pytest.mark.parametrize("partie", PARTIES)
    async def test_un_article_en_ligne_ne_change_pas_sans_etre_retire(
        self, admin_client, client, generateur, partie
    ):
        post = await _en_ligne(admin_client)

        refuse = await admin_client.put(f"{BLOG}/{post['id']}", json=_modification(post, partie))

        assert refuse.status_code == 422 and refuse.json()["code"] == "controle_a_relancer"
        assert await _admin(admin_client, post["id"]) == post
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 200


# ==================== 3. LES SOURCES GARDÉES ====================


class TestLesSourcesGardees:

    async def test_rendues_par_chaque_lecture_d_administration(
        self, admin_client, client, generateur, db_session
    ):
        corps = await _genere(admin_client)
        post, sources = corps["post"], corps["sources"]

        lectures = {
            "generate": post,
            "admin/{id}": await _admin(admin_client, post["id"]),
            "admin/all": (await admin_client.get(f"{BLOG}/admin/all")).json()[0],
            "PUT": (await admin_client.put(
                f"{BLOG}/{post['id']}", json={"relecture": relecture_cochee(post)}
            )).json(),
            "controle": (await admin_client.post(f"{BLOG}/{post['id']}/controle")).json(),
            "publier": (await admin_client.put(f"{BLOG}/{post['id']}", json={"is_published": True})).json(),
        }

        assert [s["n"] for s in sources] == [1, 2, 3]
        assert all(set(s) == CLES_DES_SOURCES for s in sources)
        assert sources[0]["law_title"] == TITRE_CODE_PENAL and sources[0]["article_number"] == "299"
        assert sources[0]["extrait"].startswith("Violation de domicile")
        for nom, lecture in lectures.items():
            assert lecture["sources"] == sources, nom
        # Dans la colonne JSON de 017, sans colonne nouvelle.
        assert (await _dossier(db_session, post["id"]))["sources"] == sources
        # Jamais au public.
        page = (await client.get(f"{BLOG}/{post['slug']}")).json()
        liste = (await client.get(BLOG)).json()
        assert "sources" not in page and "sources" not in liste[0]

    async def test_un_brouillon_rouvert_plus_tard_les_a_encore(self, admin_client, generateur, db_session):
        """Revue du tour 3 : rouvert lors d'une autre visite, le brouillon n'avait plus ses sources."""
        corps = await _genere(admin_client)
        _oublier_les_articles(db_session)

        rouvert = await _admin(admin_client, corps["post"]["id"])

        assert rouvert["sources"] == corps["sources"]
        assert rouvert["controle_ia"]["phrases"][0]["extraits"] == [1]

    async def test_l_extrait_d_une_citation_ajoutee_apres_le_controle(
        self, admin_client, generateur, controleur, textes
    ):
        post = (await _genere(admin_client))["post"]
        citations = post["botAnalysis"]["citations"] + [
            {"law_id": textes.code_penal, "article_number": "322", "title": "La filouterie de loyers."}
        ]
        await admin_client.put(f"{BLOG}/{post['id']}", json={"bot_analysis": {"citations": citations}})

        controle = (await admin_client.post(f"{BLOG}/{post['id']}/controle")).json()

        prompt = appels_du_controle(controleur.llm)[0]["prompt"]
        assert f"[4] {TITRE_CODE_PENAL} — Article 322" in prompt
        assert [s["n"] for s in controle["sources"]] == [1, 2, 3, 4]
        assert (controle["sources"][3]["article_number"], controle["sources"][3]["law_title"]) == (
            "322", TITRE_CODE_PENAL
        )
        # Les sources de l'article, elles, ne changent pas.
        assert [s["n"] for s in (await _admin(admin_client, post["id"]))["sources"]] == [1, 2, 3, 4]


# ==================== 4. LA RECONSTRUCTION DES SOURCES ====================


class TestLaReconstruction:

    async def test_les_memes_sources_gardent_citations_et_controle(self, admin_client, generateur):
        post = await _publiable(admin_client)

        reponse = await admin_client.post(f"{BLOG}/{post['id']}/sources")

        assert reponse.status_code == 200, reponse.text
        corps = reponse.json()
        assert set(corps) == {"post", "sources", "avertissements"}
        assert corps["sources"] == post["sources"] == corps["post"]["sources"]
        assert corps["post"]["botAnalysis"]["citations"] == post["botAnalysis"]["citations"]
        assert corps["post"]["controle_ia"] == post["controle_ia"]
        assert corps["post"]["etat_publication"] == "publiable"
        assert not any("Retirée" in a for a in corps["avertissements"])

    async def test_le_sujet_vient_du_titre_et_du_contexte(self, admin_client, generateur):
        post = (await _genere(admin_client))["post"]
        generateur.recherche.requetes.clear()

        await admin_client.post(f"{BLOG}/{post['id']}/sources")

        (requete,) = generateur.recherche.requetes
        assert requete.semantic_query == sujet_de_l_article(post["title"], post["context"])
        assert requete.semantic_query.startswith("Expulsion sans juge : ce que dit la loi. Un locataire")
        # Deux appels : la génération et son contrôle ; la reconstruction n'appelle pas le modèle.
        assert len(generateur.llm.generate.await_args_list) == 2

    async def test_une_citation_que_les_sources_ne_portent_plus_est_retiree(
        self, admin_client, generateur, textes
    ):
        post = await _publiable(admin_client)
        generateur.recherche = RechercheDoublee(
            [extrait(textes.code_penal, textes.cp_299, "299", CP_299)], {textes.cp_299: 0.6},
        )

        corps = (await admin_client.post(f"{BLOG}/{post['id']}/sources")).json()

        apres = corps["post"]
        assert [(s["n"], s["article_number"]) for s in corps["sources"]] == [(1, "299")]
        assert [c["article_number"] for c in apres["botAnalysis"]["citations"]] == ["299"]
        (retiree,) = [a for a in corps["avertissements"] if "Retirée" in a]
        assert retiree.startswith("Citation 2 (Article 1736 · Code civil applicable au Cameroun")
        assert "aucune des sources reconstruites ne porte cet article" in retiree
        # Ses numéros d'extraits ne valaient plus : le contrôle d'avant est effacé, tout est à relire.
        assert apres["controle_ia"] is None
        assert any("contrôle IA d'avant" in a for a in corps["avertissements"])
        assert apres["etat_publication"] == "controle_a_relancer"
        assert not any(apres["relecture"]["parties"].values())

    async def test_un_article_en_ligne_repasse_en_brouillon(self, admin_client, client, generateur):
        post = await _en_ligne(admin_client)

        corps = (await admin_client.post(f"{BLOG}/{post['id']}/sources")).json()

        assert corps["post"]["isPublished"] is False
        assert any("repasse en brouillon" in a for a in corps["avertissements"])
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 404

    async def test_cocher_avant_la_reconstruction_ne_compte_plus(
        self, admin_client, generateur, db_session, textes
    ):
        """
        Un article d'avant, sans contrôle, ne se coche pas (avenant du tour 5 : une case nomme le
        contrôle relu) ; reconstruit, tout est à relire contre les sources.
        """
        post_id = await article_d_avant(db_session, publie=False)
        coche = await admin_client.put(f"{BLOG}/{post_id}", json={"relecture": TOUT_RELU})

        corps = (await admin_client.post(f"{BLOG}/{post_id}/sources")).json()

        assert (coche.status_code, coche.json()["code"]) == (422, "relecture_perimee")
        # Ses citations d'avant retirées, les explications sont vides : rien à y relire.
        assert corps["post"]["relecture"]["parties"] == {p: p == "explications" for p in PARTIES}

    @pytest.mark.parametrize("recherche, statut, code, delai", [
        ("vide", 422, "sans_source", None),
        ("faible", 422, "sans_source", None),
        ("en_panne", 503, "recherche_indisponible", "10"),
        ("par_l_index", 503, "recherche_indisponible", "10"),
    ])
    async def test_ses_erreurs_ne_changent_rien(
        self, admin_client, generateur, textes, db_session, recherche, statut, code, delai
    ):
        post = await _en_ligne(admin_client)
        avant = await _dossier(db_session, post["id"])
        generateur.recherche = {
            "vide": RechercheDoublee([]),
            "faible": RechercheDoublee(extraits_du_sujet(textes), {textes.cp_299: 0.2}),
            "en_panne": RechercheDoublee([], erreur=RuntimeError("base injoignable")),
            "par_l_index": RechercheDoublee(
                extraits_du_sujet(textes), similarites(textes), chemin=CHEMIN_REPLI_HNSW
            ),
        }[recherche]

        reponse = await admin_client.post(f"{BLOG}/{post['id']}/sources", headers=ORIGINE)

        assert reponse.status_code == statut, reponse.text
        assert reponse.json()["code"] == code and isinstance(reponse.json()["detail"], str)
        assert reponse.headers.get("retry-after") == delai
        assert reponse.headers["access-control-allow-origin"] == ORIGINE["Origin"]
        if code == "sans_source":
            assert "rien n'a changé" in reponse.json()["detail"]
        _oublier_les_articles(db_session)
        assert await _dossier(db_session, post["id"]) == avant
        assert (await _admin(admin_client, post["id"]))["isPublished"] is True

    async def test_au_dela_du_delai_504(self, admin_client, generateur, db_session, monkeypatch):
        post = await _en_ligne(admin_client)
        avant = await _dossier(db_session, post["id"])

        class Lente(RechercheDoublee):
            async def search(self, requete):
                await asyncio.sleep(5)

        generateur.recherche = Lente([])
        monkeypatch.setattr(routes_du_blog, "DELAI_SOURCES_S", 0.2)

        reponse = await admin_client.post(f"{BLOG}/{post['id']}/sources")

        assert reponse.status_code == 504 and reponse.json()["code"] == "delai"
        _oublier_les_articles(db_session)
        assert await _dossier(db_session, post["id"]) == avant

    async def test_introuvable(self, admin_client, generateur):
        inconnu = await admin_client.post(f"{BLOG}/999999/sources")

        assert inconnu.status_code == 404 and inconnu.json()["code"] == "introuvable"

    async def test_reserve_aux_administrateurs(self, client, auth_headers):
        anonyme = await client.post(f"{BLOG}/1/sources")
        ordinaire = await client.post(f"{BLOG}/1/sources", headers=auth_headers)

        assert (anonyme.status_code, ordinaire.status_code) == (401, 403)


# ==================== 5. LES ARTICLES D'AVANT ====================


class TestLesArticlesDAvant:

    async def test_jamais_publics_meme_avec_des_citations_marquees_verifiees(
        self, admin_client, client, db_session
    ):
        """
        Plus strict que le correctif de main (5bdcb4f : publié, et toutes les
        citations `verifie: true`) : sans sources, contrôle ni relecture, un
        article n'est pas servi, même si ses citations portent « verifie: true ».
        """
        forgees = [{**c, "verifie": True, "article_id": 1} for c in CITATIONS_D_AVANT]
        await article_d_avant(db_session, citations=forgees)

        assert (await client.get(BLOG)).json() == []
        assert (await client.get(f"{BLOG}/expulsion-locative-loi-cameroun")).status_code == 404

    async def test_le_parcours_complet_avec_la_recherche_de_la_generation(
        self, admin_client, client, generateur, controleur, db_session, textes
    ):
        """Comme les lignes du 26/10 : publié en base, citations jamais vérifiées, ni sources ni contrôle."""
        post_id = await article_d_avant(db_session)
        slug = "expulsion-locative-loi-cameroun"
        etapes = []

        async def etat():
            etapes.append((await _admin(admin_client, post_id))["etat_publication"])
            assert (await client.get(f"{BLOG}/{slug}")).status_code == 404, etapes

        await etat()
        sources = (await admin_client.post(f"{BLOG}/{post_id}/sources")).json()
        await etat()
        source = next(s for s in sources["sources"] if s["article_number"] == "299")
        await admin_client.put(f"{BLOG}/{post_id}", json={"bot_analysis": {"citations": [
            {"law_id": source["law_id"], "article_id": source["article_id"],
             "article_number": "299", "title": "La violation de domicile est punie."},
        ]}})
        await etat()
        controle = (await admin_client.post(f"{BLOG}/{post_id}/controle")).json()
        await etat()
        await admin_client.put(f"{BLOG}/{post_id}", json={"relecture": relecture_cochee(controle)})
        await etat()
        publie = await admin_client.put(f"{BLOG}/{post_id}", json={"is_published": True})

        assert etapes == [
            "sources_absentes", "citations_non_verifiees", "controle_a_relancer",
            "relecture_incomplete", "publiable",
        ]
        assert publie.status_code == 200, publie.text
        page = await client.get(f"{BLOG}/{slug}")
        assert page.status_code == 200
        (citation,) = page.json()["botAnalysis"]["citations"]
        assert citation["verifie"] is True and citation["excerpt"].startswith("Violation de domicile")
        assert "trois mois à trois ans" not in page.text
        # Le contrôle a jugé l'article contre les sources reconstruites.
        prompt = appels_du_controle(controleur.llm)[0]["prompt"]
        assert f"[1] {TITRE_CODE_PENAL} — Article 299" in prompt


# ==================== 6. LA RELECTURE TIENT AU CONTRÔLE ====================


class TestLaRelectureTientAuControle:

    async def test_cochee_avant_le_controle_elle_ne_compte_pas(self, admin_client, generateur):
        """Avenant du tour 5 : sans contrôle enregistré, une case n'a rien à attester, et se refuse."""
        generateur.llm = modele_double(controle=MistralOverloadedError("503"))
        post = (await _genere(admin_client))["post"]
        coche = await admin_client.put(f"{BLOG}/{post['id']}", json={"relecture": relecture_cochee(post)})

        controle = (await admin_client.post(f"{BLOG}/{post['id']}/controle")).json()

        assert (coche.status_code, coche.json()["code"]) == (422, "relecture_perimee")
        assert controle["etat_publication"] == "relecture_incomplete"
        assert not any(controle["relecture"]["parties"].values())

    async def test_un_nouveau_verdict_fait_relire_sa_partie_seule(self, admin_client, generateur, controleur):
        post = await _publiable(admin_client)
        controleur.llm = modele_double(controle={"Entrer chez quelqu'un": "non_soutenue"})

        controle = (await admin_client.post(f"{BLOG}/{post['id']}/controle")).json()

        # Une phrase de règle non trouvée : à confirmer d'abord (tour 7), et sa partie à relire.
        assert controle["etat_publication"] == "phrases_a_confirmer"
        assert controle["relecture"]["parties"] == {p: p != "points_cles" for p in PARTIES}

    async def test_les_memes_verdicts_gardent_la_relecture(self, admin_client, generateur):
        post = await _publiable(admin_client)

        controle = (await admin_client.post(f"{BLOG}/{post['id']}/controle")).json()

        assert controle["controle_ia"]["fait_le"]
        assert controle["etat_publication"] == "publiable"


# ==================== 7. LE PASSAGE COPIÉ PAR LE CONTRÔLE ====================


def _controle_avec(**par_phrase):
    """Un contrôle doublé où certaines phrases (par un morceau de texte) ont un jugement imposé."""
    def repondre(prompt):
        passage = passage_de(extraits_du_prompt(prompt)[1])
        jugees = []
        for identifiant, phrase in phrases_du_prompt(prompt):
            impose = next((f for morceau, f in par_phrase.items() if morceau in phrase), None)
            jugees.append(impose(identifiant) if impose else jugement(identifiant, "soutenue", passage, [1]))
        return {"response": json.dumps({"phrases": jugees}), "fin": "STOP"}
    return repondre


class TestLePassageCopie:

    async def test_un_appui_invente_n_est_pas_soutenu(self, admin_client, generateur, controleur):
        post = (await _genere(admin_client))["post"]
        controleur.llm = modele_double(controle=_controle_avec(**{
            "S'introduire dans le domicile": lambda i: jugement(
                i, "soutenue", "le bailleur peut expulser sans jugement en cas de reprise", [1],
                raison="L'extrait 1 permet l'expulsion sans jugement.",
            ),
        }))

        controle = (await admin_client.post(f"{BLOG}/{post['id']}/controle")).json()

        (resume,) = [p for p in controle["controle_ia"]["phrases"] if p["partie"] == "resume"]
        assert resume["verdict"] == "non_soutenue"
        assert resume["raison"].startswith("À vérifier : le contrôle la disait soutenue")
        # Une phrase de règle : elle bloque jusqu'à sa confirmation (tour 7), la relecture ne suffit plus.
        cases = await admin_client.put(
            f"{BLOG}/{post['id']}", json={"relecture": relecture_cochee(controle, confirmer=False)}
        )
        assert cases.json()["etat_publication"] == "phrases_a_confirmer"
        relu = await admin_client.put(f"{BLOG}/{post['id']}", json={"relecture": relecture_cochee(controle)})
        assert relu.json()["etat_publication"] == "publiable"

    async def test_une_contradiction_bloque_seulement_avec_son_passage(
        self, admin_client, generateur, controleur
    ):
        post = (await _genere(admin_client))["post"]
        controleur.llm = modele_double(controle=_controle_avec(**{
            "S'introduire dans le domicile": lambda i: jugement(
                i, "contredite", "dix (10) jours à un (01) an", [1],
                raison="L'extrait 1 fixe une autre peine.",
            ),
            "Entrer chez quelqu'un": lambda i: jugement(i, "contredite", "", [], raison="Faux."),
        }))

        controle = (await admin_client.post(f"{BLOG}/{post['id']}/controle")).json()

        phrases = {p["partie"]: p for p in controle["controle_ia"]["phrases"] if p["index"] == 0}
        assert phrases["resume"]["verdict"] == "contredite"
        assert phrases["resume"]["raison"].endswith("— Extrait 1 : « dix (10) jours à un (01) an »")
        assert phrases["point_cle"]["verdict"] == "non_soutenue"
        assert controle["etat_publication"] == "phrases_contredites"


# ==================== 8. CHAQUE ÉTAT REFUSE LA PUBLICATION ====================


async def _dans_l_etat(etat, admin_client, generateur, db_session, textes) -> dict:
    """Un brouillon dans l'état demandé, entièrement relu quand l'état le permet."""
    if etat == "sources_absentes":
        cree = await admin_client.post(BLOG, json={
            "title": "Le congé d'un bail verbal", "category": "Immobilier",
            "context": "Un bailleur veut reprendre son logement loué sans contrat écrit.",
            "bot_analysis": {
                "summary": "Le congé suit les délais fixés par l'usage des lieux.",
                "keyPoints": ["Sans écrit, l'usage des lieux fixe le délai."],
                "citations": [{"law_id": textes.code_civil, "article_number": "1736"}],
                "citizenImpact": "Renseignez-vous sur l'usage local.",
            },
        })
        # Sans contrôle, rien à cocher (avenant du tour 5 : relecture_perimee).
        return await _admin(admin_client, cree.json()["id"])
    if etat == "phrases_contredites":
        generateur.llm = modele_double(controle={"Entrer chez quelqu'un": "contredite"})
    if etat == "phrases_a_confirmer":
        generateur.llm = modele_double(controle={"Entrer chez quelqu'un": "non_soutenue"})
    post = (await _genere(admin_client))["post"]
    if etat != "relecture_incomplete":
        await admin_client.put(f"{BLOG}/{post['id']}", json={
            "relecture": relecture_cochee(post, confirmer=etat != "phrases_a_confirmer"),
        })
    if etat == "citations_non_verifiees":
        await db_session.execute(
            text("UPDATE laws SET status = 'archived' WHERE id = :id"), {"id": textes.code_civil}
        )
        await db_session.commit()
    if etat == "controle_a_relancer":
        await db_session.execute(
            text("UPDATE blog_posts SET title = 'Réécrit en base' WHERE id = :id"), {"id": post["id"]}
        )
        await db_session.commit()
        _oublier_les_articles(db_session)
    return await _admin(admin_client, post["id"])


class TestChaqueEtatRefuseLaPublication:

    @pytest.mark.parametrize("etat", [
        "sources_absentes", "citations_non_verifiees", "controle_a_relancer",
        "phrases_contredites", "phrases_a_confirmer", "relecture_incomplete",
    ])
    async def test_refus_au_code_de_l_etat_et_rien_n_est_applique(
        self, admin_client, client, generateur, db_session, textes, etat
    ):
        post = await _dans_l_etat(etat, admin_client, generateur, db_session, textes)
        assert post["etat_publication"] == etat

        seul = await admin_client.put(f"{BLOG}/{post['id']}", json={"is_published": True})
        avec_un_autre_champ = await admin_client.put(
            f"{BLOG}/{post['id']}", json={"is_published": True, "category": "Famille"}
        )

        for refus in (seul, avec_un_autre_champ):
            assert refus.status_code == 422, refus.text
            assert refus.json()["code"] == etat
            assert refus.json()["detail"].startswith("Publication refusée : ")
        relu = await _admin(admin_client, post["id"])
        assert relu["isPublished"] is False and relu["category"] == post["category"]
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 404

    async def test_publiable_se_publie(self, admin_client, client, generateur):
        post = await _publiable(admin_client)

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={"is_published": True})

        assert reponse.status_code == 200 and reponse.json()["isPublished"] is True
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 200


# ==================== 9. LES CODES D'ERREUR DU CONTRÔLE ====================


class TestLesCodesDuControle:

    @pytest.mark.parametrize("panne, statut, code, delai", [
        (MistralQuotaError("Quota mensuel épuisé"), 429, "quota", "60"),
        (MistralOverloadedError("503"), 503, "saturation", "10"),
        (MistralServiceError("HTTP 401 Invalid API Key"), 502, "reponse_invalide", None),
        ({"response": "pas du JSON", "fin": "STOP"}, 502, "reponse_invalide", None),
        ({"response": '{"phrases": [', "fin": "MAX_TOKENS", "tronquee": True}, 502, "reponse_invalide", None),
        ({"response": json.dumps({"phrases": []}), "fin": "STOP"}, 502, "reponse_invalide", None),
        (RuntimeError("panne imprévue"), 500, "erreur_interne", None),
    ], ids=["quota", "saturation", "fournisseur", "illisible", "coupee", "vide", "imprevue"])
    async def test_les_pannes_du_modele(
        self, admin_client, generateur, controleur, panne, statut, code, delai
    ):
        post = (await _genere(admin_client))["post"]
        controleur.llm = modele_double(controle=panne)

        reponse = await admin_client.post(f"{BLOG}/{post['id']}/controle", headers=ORIGINE)

        assert reponse.status_code == statut, reponse.text
        assert reponse.json()["code"] == code and isinstance(reponse.json()["detail"], str)
        assert reponse.headers.get("retry-after") == delai
        assert reponse.headers["access-control-allow-origin"] == ORIGINE["Origin"]
        assert "Invalid API Key" not in reponse.text and "panne imprévue" not in reponse.text
        # Rien n'est perdu : le contrôle d'avant reste.
        assert (await _admin(admin_client, post["id"]))["controle_ia"] == post["controle_ia"]

    async def test_au_dela_du_delai_504(self, admin_client, generateur, controleur, monkeypatch):
        post = (await _genere(admin_client))["post"]

        async def lent(**_):
            await asyncio.sleep(5)

        controleur.llm = modele_double(side_effect=lent)
        monkeypatch.setattr(routes_du_blog, "DELAI_CONTROLE_S", 0.2)

        reponse = await admin_client.post(f"{BLOG}/{post['id']}/controle")

        assert (reponse.status_code, reponse.json()["code"]) == (504, "delai")

    async def test_sans_sources_422(self, admin_client, controleur, db_session):
        post_id = await article_d_avant(db_session)

        reponse = await admin_client.post(f"{BLOG}/{post_id}/controle")

        assert (reponse.status_code, reponse.json()["code"]) == (422, "sources_absentes")
        controleur.llm.generate.assert_not_awaited()

    async def test_trop_de_phrases_422(self, admin_client, generateur, controleur):
        post = (await _genere(admin_client))["post"]
        long = " ".join(f"Phrase {i} du contexte." for i in range(PHRASES_AU_PLUS + 1))
        await admin_client.put(f"{BLOG}/{post['id']}", json={"context": long})

        reponse = await admin_client.post(f"{BLOG}/{post['id']}/controle")

        assert (reponse.status_code, reponse.json()["code"]) == (422, "entree_invalide")
        assert f"{PHRASES_AU_PLUS} au plus" in reponse.json()["detail"]
        controleur.llm.generate.assert_not_awaited()

    async def test_introuvable_404(self, admin_client):
        reponse = await admin_client.post(f"{BLOG}/999999/controle")

        assert (reponse.status_code, reponse.json()["code"]) == (404, "introuvable")


# ==================== 10. LE PUT : RETIRER EXACTEMENT, OU TOUT D'UN BLOC ====================


class TestLeCorpsDuPut:

    async def test_retirer_exactement_meme_un_article_non_publiable(self, admin_client, db_session):
        post_id = await article_d_avant(db_session)
        avant = await _dossier(db_session, post_id)

        reponse = await admin_client.put(f"{BLOG}/{post_id}", json={"is_published": False})

        assert reponse.status_code == 200 and reponse.json()["isPublished"] is False
        assert await _dossier(db_session, post_id) == avant

    @pytest.mark.parametrize("corps", [
        {"is_published": False, "title": ""},
        {"is_published": False, "slug": "!!!"},
        {"is_published": False, "relecture": {"parties": {"inconnue": True}}},
        {"is_published": False, "bot_analysis": {"citations": [{"law_id": 999999, "article_number": "1"}]}},
    ], ids=["titre-vide", "adresse-vide", "case-inconnue", "citation-inconnue"])
    async def test_un_corps_plus_large_est_valide_en_entier(self, admin_client, client, generateur, corps):
        post = await _en_ligne(admin_client)

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json=corps)

        assert reponse.status_code == 422, reponse.text
        assert reponse.json()["code"] == "entree_invalide"
        assert await _admin(admin_client, post["id"]) == post
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 200

    async def test_d_un_bloc_rien_n_est_applique_d_un_refus(self, admin_client, generateur, textes):
        """Titre, rubrique et cases valides, adresse prise : rien n'est enregistré."""
        autre = (await _genere(admin_client))["post"]
        post = (await _genere(admin_client))["post"]

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={
            "title": "Un titre neuf", "category": "Famille", "relecture": relecture_cochee(post),
            "slug": autre["slug"],
        })

        assert reponse.status_code == 409 and reponse.json()["code"] == "slug_pris"
        assert await _admin(admin_client, post["id"]) == post

    async def test_d_un_bloc_tout_est_applique(self, admin_client, generateur):
        post = (await _genere(admin_client))["post"]

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={
            "title": "Un titre neuf", "category": "Famille", "relecture": relecture_cochee(post),
        })

        apres = reponse.json()
        assert (apres["title"], apres["category"]) == ("Un titre neuf", "Famille")
        # Le titre a changé dans la même requête : sa case ne vaut pas pour lui.
        assert apres["relecture"]["parties"] == {p: p != "titre" for p in PARTIES}
        assert apres["etat_publication"] == "controle_a_relancer"


# ==================== 11. LA GÉNÉRATION ET LE CONTRÔLE, ENSEMBLE ====================


class TestLeDelaiDeLaRedaction:

    async def test_le_controle_garde_le_temps_qui_reste(self, admin_client, generateur, monkeypatch):
        """Revue du tour 3 : 150 s de génération puis 120 s de contrôle, quand le front abandonne à 160 s."""
        vus = []
        vrai = routes_du_blog._controler_apres_la_generation

        async def espion(db, post_id, llm, delai):
            vus.append(delai)
            return await vrai(db, post_id, llm, delai)

        monkeypatch.setattr(routes_du_blog, "_controler_apres_la_generation", espion)
        monkeypatch.setattr(routes_du_blog, "DELAI_DE_LA_REDACTION_S", 40)

        await _genere(admin_client)

        (delai,) = vus
        assert 30 <= delai <= 40

    async def test_au_moins_de_quoi_juger_un_article_court(self, admin_client, generateur, monkeypatch):
        vus = []
        vrai = routes_du_blog._controler_apres_la_generation

        async def espion(db, post_id, llm, delai):
            vus.append(delai)
            return await vrai(db, post_id, llm, delai)

        monkeypatch.setattr(routes_du_blog, "_controler_apres_la_generation", espion)
        monkeypatch.setattr(routes_du_blog, "DELAI_DE_LA_REDACTION_S", 0)

        await _genere(admin_client)

        assert vus == [routes_du_blog.DELAI_CONTROLE_APRES_GENERATION_AU_MOINS_S]

