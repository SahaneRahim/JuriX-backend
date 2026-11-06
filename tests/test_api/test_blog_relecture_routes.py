"""
Le contrôle IA et la relecture humaine du blog, par les routes (tour 3).

POURQUOI. Essai réel du 30/10/2026 : deux articles publiés sous des cartes
« vérifiées », sans un avertissement, disaient le faux — l'expulsion d'un
locataire possible sans juge, la charge de la preuve du licenciement abusif
inversée, et une explication contraire à l'article 1752 du Code civil. Les
citations vérifiées ne rendent pas la prose juste. Décision du propriétaire
(29-30/10) : un contrôle IA de chaque phrase contre les sources, ET une
relecture humaine de chaque partie, avant toute publication ; les articles
d'avant, cachés du public.

Ce que ces tests protègent :
  1. Le contrôle : un appel, des verdicts par phrase, rangés dans controle_ia ;
     ses erreurs traduites (429, 503, 502, 504) sans rien perdre.
  2. L'empreinte : tout changement du texte périme le contrôle et décoche la
     relecture des parties changées, par quelque chemin qu'il passe.
  3. La publication : refusée (422, code = l'état) dans chaque état non
     publiable, et rien n'est alors appliqué.
  4. PUT : {"is_published": false} exactement retire toujours ; tout autre
     corps est validé en entier, puis appliqué d'un bloc.
  5. Le public ne voit que les articles publiés ET publiables, et rien de leur
     contrôle ni de leur relecture (avenant du tour 7).
  6. Les phrases à confirmer (avenant du tour 7, décision du 04/11) : une phrase
     de règle non trouvée dans les extraits se réécrit ou se confirme, sur chaque
     chemin qui écrit la relecture.

Le modèle et la recherche sont doublés (tests/blog_donnees.py) : le vrai modèle
n'est jamais appelé.

Usage:
    pytest tests/test_api/test_blog_relecture_routes.py -v
"""

import asyncio
import json
import re
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from sqlalchemy import func, select, text

from app.api.routes import blog as routes_du_blog
from app.api.routes.blog import get_blog_generator, get_controleur
from app.core.config import settings
from app.core.limiteur import Limiteur
from app.main import app
from app.models.blog import BlogPost
from app.services import controle_du_billet, mistral_service
from app.services.blog_generator_service import BlogGeneratorService
from app.services.controle_du_billet import (
    DUREE_D_UN_APPEL_AU_PIRE_S,
    ETATS,
    MARGE_DU_PLAN_S,
    PARTIES,
    PHRASES_AU_PLUS,
    PHRASES_PAR_APPEL,
    RAISON_NON_JUGEE,
    SCHEMA_DU_CONTROLE,
    SYSTEM_DU_CONTROLE,
    ControleDuBillet,
)
from app.services.mistral_service import (
    MistralOverloadedError,
    MistralQuotaError,
    MistralServiceError,
)
from tests.blog_donnees import (
    CC_1736,
    CP_299,
    SUJET,
    TITRE_CODE_PENAL,
    GrandModeleAuRythme,
    RechercheDoublee,
    a_confirmer,
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
    reponse_du_controle,
    similarites,
)

BLOG = "/api/v1/blog"
ORIGINE = {"Origin": "http://localhost:5173"}
TOUT_RELU = {"parties": {partie: True for partie in PARTIES}}
_EMPREINTE = re.compile(r"^[0-9a-f]{64}$")
# Ce que seule l'administration lit (avenant du tour 7).
CHAMPS_DE_L_ADMINISTRATION = {
    "controle_ia", "controleIa", "relecture", "etat_publication", "etatPublication", "sources",
    "avertissements",
}


@pytest.fixture
async def textes(db_session):
    return await creer_les_textes(db_session)


@pytest.fixture
def generateur(db_session, textes):
    """
    La génération doublée ; son modèle sert aussi au contrôle qui la suit, sauf si le test
    en donne un autre (`llm_du_controle`, tour 6 : en production, celui du contrôle).
    """
    class Reglage:
        llm = modele_double()
        llm_du_controle = None
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

    app.dependency_overrides[get_blog_generator] = lambda: BlogGeneratorService(
        db_session, llm=Reglage.llm, recherche=Reglage.recherche, llm_du_controle=Reglage.llm_du_controle,
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


async def _nombre_d_articles(db_session) -> int:
    return (await db_session.execute(select(func.count()).select_from(BlogPost))).scalar_one()


async def _genere(admin_client) -> dict:
    reponse = await admin_client.post(f"{BLOG}/generate", json={"topic": SUJET})
    assert reponse.status_code == 201, reponse.text
    return reponse.json()


async def _controle(admin_client, post_id, **options):
    return await admin_client.post(f"{BLOG}/{post_id}/controle", **options)


async def _admin(admin_client, post_id) -> dict:
    return (await admin_client.get(f"{BLOG}/admin/{post_id}")).json()


async def _publiable(admin_client, generateur) -> dict:
    """Un brouillon généré, contrôlé et entièrement relu : publiable, pas encore publié."""
    post = (await _genere(admin_client))["post"]
    reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={"relecture": relecture_cochee(post)})
    assert reponse.status_code == 200, reponse.text
    assert reponse.json()["etat_publication"] == "publiable", reponse.json()
    return reponse.json()


async def _en_ligne(admin_client, generateur) -> dict:
    post = await _publiable(admin_client, generateur)
    reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={"is_published": True})
    assert reponse.status_code == 200, reponse.text
    return reponse.json()


def _corps_du_front(post, **modifications):
    """Le formulaire entier, tel que le front l'envoie (relecture.ts)."""
    analyse = post["botAnalysis"]
    corps = {
        "title": post["title"], "category": post["category"], "context": post["context"],
        "bot_analysis": {
            "summary": analyse["summary"], "keyPoints": analyse["keyPoints"],
            "citations": analyse["citations"], "citizenImpact": analyse["citizenImpact"],
        },
        "suggested_question": post["suggestedQuestion"], "tags": post["tags"],
    }
    for cle, valeur in modifications.items():
        if cle in ("summary", "citizenImpact", "citations", "keyPoints"):
            corps["bot_analysis"][cle] = valeur
        else:
            corps[cle] = valeur
    return corps


# ==================== 1. LE CONTRÔLE À LA GÉNÉRATION ====================


class TestLeControleALaGeneration:

    async def test_chaque_phrase_est_jugee_et_rangee(self, admin_client, generateur, db_session):
        corps = await _genere(admin_client)
        post = corps["post"]

        controle = post["controle_ia"]
        assert set(controle) == {"fait_le", "modele", "empreinte", "phrases"}
        assert controle["modele"] == "modele-double"
        assert _EMPREINTE.match(controle["empreinte"])
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$", controle["fait_le"])
        phrases = controle["phrases"]
        assert len({p["id"] for p in phrases}) == len(phrases)
        for phrase in phrases:
            assert set(phrase) == {"id", "partie", "index", "texte", "verdict", "extraits", "raison", "nature"}
            assert phrase["verdict"] == "soutenue" and phrase["nature"] == "regle"
        assert {p["partie"] for p in phrases} == {
            "titre", "contexte", "resume", "point_cle", "impact", "explication", "question",
        }
        # Tour 4 : le titre est jugé, en premier.
        assert (phrases[0]["partie"], phrases[0]["texte"]) == ("titre", post["title"])
        # `index` : le rang du point clé ou de la citation, à partir de 0.
        points = [p for p in phrases if p["partie"] == "point_cle"]
        assert [p["index"] for p in points] == [0, 1, 2]
        assert points[1]["texte"] == post["botAnalysis"]["keyPoints"][1]
        explications = [p for p in phrases if p["partie"] == "explication"]
        assert [p["index"] for p in explications] == [0, 1]
        assert explications[0]["texte"] == post["botAnalysis"]["citations"][0]["title"]
        assert post["etat_publication"] == "relecture_incomplete"
        assert corps["avertissements"] == []
        # Enregistré en base, sous la colonne JSON `citations` de 017.
        dossier = (await db_session.execute(
            text("SELECT citations FROM blog_posts WHERE id = :id"), {"id": post["id"]}
        )).scalar_one()
        assert set(dossier) == {"citations", "sources", "controle_ia"}
        assert [s["n"] for s in dossier["sources"]] == [1, 2, 3]

    async def test_le_controle_juge_avec_son_modele_et_non_celui_de_la_redaction(
        self, admin_client, generateur
    ):
        """
        Tour 6 : la génération rédige avec le sien (MISTRAL_MODEL), le contrôle qui la suit juge
        avec le sien (BLOG_CONTROLE_MODELE) ; controle_ia nomme le modèle qui a répondu.
        """
        juge = modele_double(controle=lambda prompt: {**reponse_du_controle(prompt), "modele": "mistral-large-4"})
        juge.model_name = "mistral-large-4"
        generateur.llm_du_controle = juge

        post = (await _genere(admin_client))["post"]

        assert appels_du_controle(generateur.llm) == []
        (appel,) = appels_du_controle(juge)
        assert appel["reflexion"] == controle_du_billet.settings.BLOG_CONTROLE_REFLEXION
        assert appel["rendre_le_modele"] is True
        assert post["controle_ia"]["modele"] == "mistral-large-4"

    async def test_un_appel_a_temperature_zero_au_schema_strict(self, admin_client, generateur):
        await _genere(admin_client)

        (appel,) = appels_du_controle(generateur.llm)
        assert appel["temperature"] == 0.0
        assert appel["response_schema"] == SCHEMA_DU_CONTROLE
        assert appel["response_mime_type"] == "application/json"
        assert appel["system"] == SYSTEM_DU_CONTROLE
        prompt = appel["prompt"]
        # Les extraits de la génération, sous leurs numéros ; puis les phrases.
        assert f"[1] {TITRE_CODE_PENAL} — Article 299" in prompt
        assert "PHRASES À JUGER" in prompt
        assert phrases_du_prompt(prompt)[0][0] == "P1"
        assert "Explication de la citation 1, qui accompagne l'extrait 1 (Article 299)" in prompt
        assert "PHRASES À JUGER" in prompt and "\nTitre de l'article\nP1 : Expulsion sans juge" in prompt

    async def test_une_phrase_contredite(self, admin_client, generateur):
        """La forme de l'essai réel : une règle que l'extrait cité contredit."""
        generateur.llm = modele_double(article_du_modele(
            summary="Le salarié doit prouver le caractère abusif du licenciement.",
        ), controle={"Le salarié doit prouver": "contredite"})

        post = (await _genere(admin_client))["post"]
        relu = await admin_client.put(f"{BLOG}/{post['id']}", json={"relecture": relecture_cochee(post)})
        publie = await admin_client.put(f"{BLOG}/{post['id']}", json={"is_published": True})

        (contredite,) = [p for p in post["controle_ia"]["phrases"] if p["verdict"] == "contredite"]
        assert (contredite["partie"], contredite["index"]) == ("resume", 0)
        assert relu.json()["etat_publication"] == "phrases_contredites"
        assert publie.status_code == 422
        assert publie.json()["code"] == "phrases_contredites"
        assert "Le salarié doit prouver" in publie.json()["detail"]

    @pytest.mark.parametrize("panne, mot", [
        (MistralQuotaError("Quota mensuel épuisé"), "quota"),
        (MistralOverloadedError("503"), "saturé"),
        (MistralServiceError("HTTP 401 Invalid API Key"), "erreur"),
        ({"response": "Je ne peux pas juger.", "fin": "STOP"}, "lisible"),
        ({"response": json.dumps({"phrases": []}), "fin": "STOP"}, "aucune phrase"),
        ({"response": '{"phrases": [', "fin": "MAX_TOKENS", "tronquee": True}, "coupée"),
    ], ids=["quota", "saturation", "erreur-du-fournisseur", "illisible", "vide", "coupee"])
    async def test_un_controle_en_echec_garde_le_brouillon(
        self, admin_client, generateur, db_session, panne, mot
    ):
        generateur.llm = modele_double(controle=panne)

        reponse = await admin_client.post(f"{BLOG}/generate", json={"topic": SUJET}, headers=ORIGINE)

        assert reponse.status_code == 201, reponse.text
        corps = reponse.json()
        assert corps["post"]["controle_ia"] is None
        assert corps["post"]["etat_publication"] == "controle_a_relancer"
        (avertissement,) = [a for a in corps["avertissements"] if "contrôle IA" in a]
        assert "relancez le contrôle IA avant de publier" in avertissement
        assert mot in avertissement
        assert "Invalid API Key" not in reponse.text
        assert await _nombre_d_articles(db_session) == 1
        assert reponse.headers["access-control-allow-origin"] == ORIGINE["Origin"]

    async def test_un_controle_trop_long_garde_le_brouillon(
        self, admin_client, generateur, db_session, monkeypatch
    ):
        async def controle_lent(**kwargs):
            if kwargs.get("system") == SYSTEM_DU_CONTROLE:
                await asyncio.sleep(5)
            return {"response": json.dumps(article_du_modele()), "fin": "STOP"}

        generateur.llm = modele_double(side_effect=controle_lent)
        monkeypatch.setattr(routes_du_blog, "DELAI_CONTROLE_S", 0.2)

        corps = (await admin_client.post(f"{BLOG}/generate", json={"topic": SUJET})).json()

        assert corps["post"]["controle_ia"] is None
        assert any("dépassé 0.2 secondes" in a for a in corps["avertissements"])
        assert await _nombre_d_articles(db_session) == 1


# ==================== 2. LE CONTRÔLE À LA DEMANDE ====================


class TestLeControleALaDemande:

    async def test_relancer_sur_le_texte_actuel(self, admin_client, generateur, controleur):
        post = (await _genere(admin_client))["post"]
        modifie = await admin_client.put(f"{BLOG}/{post['id']}", json={
            "bot_analysis": {"summary": "Un résumé corrigé par l'administrateur."},
        })
        assert modifie.json()["etat_publication"] == "controle_a_relancer"

        reponse = await _controle(admin_client, post["id"])

        assert reponse.status_code == 200, reponse.text
        controle = reponse.json()["controle_ia"]
        assert controle["empreinte"] != post["controle_ia"]["empreinte"]
        resume = [p for p in controle["phrases"] if p["partie"] == "resume"]
        assert [p["texte"] for p in resume] == ["Un résumé corrigé par l'administrateur."]
        assert reponse.json()["etat_publication"] == "relecture_incomplete"
        assert len(appels_du_controle(controleur.llm)) == 1

    @pytest.mark.parametrize("panne, statut, code, delai", [
        (MistralQuotaError("Quota mensuel épuisé"), 429, "quota", "60"),
        (MistralOverloadedError("503"), 503, "saturation", "10"),
        # Tour 6 : un 429 du modèle du contrôle, ou son budget de jetons épuisé, avec le délai
        # connu (arrondi au-dessus) ; jamais un autre modèle.
        (MistralOverloadedError("429", retry_after=42.3), 503, "saturation", "43"),
        (MistralServiceError("HTTP 500"), 502, "reponse_invalide", None),
        ({"response": "pas du JSON", "fin": "STOP"}, 502, "reponse_invalide", None),
    ], ids=["quota", "saturation", "saturation-avec-delai", "erreur", "illisible"])
    async def test_les_erreurs_du_controle_ne_perdent_rien(
        self, admin_client, generateur, controleur, panne, statut, code, delai
    ):
        post = (await _genere(admin_client))["post"]
        await admin_client.put(f"{BLOG}/{post['id']}", json={
            "bot_analysis": {"summary": "Un résumé corrigé."},
        })
        controleur.llm = modele_double(controle=panne)

        reponse = await admin_client.post(f"{BLOG}/{post['id']}/controle", headers=ORIGINE)

        assert reponse.status_code == statut
        assert reponse.json()["code"] == code
        assert isinstance(reponse.json()["detail"], str)
        assert reponse.headers.get("retry-after") == delai
        assert reponse.headers["access-control-allow-origin"] == ORIGINE["Origin"]
        # Tour 5 : le délai est lisible par le front (fetch), pas seulement par le navigateur.
        exposes = [e.strip().lower() for e in reponse.headers["access-control-expose-headers"].split(",")]
        assert "retry-after" in exposes
        # Le contrôle d'avant reste, périmé : rien n'est perdu.
        relu = await _admin(admin_client, post["id"])
        assert relu["controle_ia"] == post["controle_ia"]
        assert relu["etat_publication"] == "controle_a_relancer"

    async def test_au_dela_du_delai_504(self, admin_client, generateur, controleur, monkeypatch):
        post = (await _genere(admin_client))["post"]

        async def lent(**_):
            await asyncio.sleep(5)

        controleur.llm = modele_double(side_effect=lent)
        monkeypatch.setattr(routes_du_blog, "DELAI_CONTROLE_S", 0.2)

        reponse = await _controle(admin_client, post["id"])

        assert reponse.status_code == 504
        assert reponse.json()["code"] == "delai"
        assert (await _admin(admin_client, post["id"]))["controle_ia"] == post["controle_ia"]

    async def test_un_article_d_avant_n_a_pas_de_sources(self, admin_client, controleur, db_session):
        """Tour 4 : sans sources, rien contre quoi juger ; le modèle n'est pas appelé."""
        post_id = await article_d_avant(db_session)

        reponse = await _controle(admin_client, post_id, headers=ORIGINE)

        assert reponse.status_code == 422
        assert reponse.json()["code"] == "sources_absentes"
        assert "Reconstruisez d'abord ses sources" in reponse.json()["detail"]
        assert reponse.headers["access-control-allow-origin"] == ORIGINE["Origin"]
        controleur.llm.generate.assert_not_awaited()

    async def test_un_article_ecrit_a_la_main_se_controle_apres_ses_sources(
        self, admin_client, controleur, generateur, textes
    ):
        """Tour 4 : ses citations ne suffisent plus ; ses sources se reconstruisent d'abord."""
        generateur.recherche = RechercheDoublee(
            [extrait(textes.code_civil, textes.cc_1736, "1736", CC_1736, reference="CODE-CIVIL")],
            {textes.cc_1736: 0.6},
        )
        cree = await admin_client.post(BLOG, json={
            "title": "Le congé d'un bail verbal", "category": "Immobilier",
            "context": "Un bailleur veut reprendre son logement loué sans contrat écrit.",
            "bot_analysis": {
                "summary": "Le congé suit les délais fixés par l'usage des lieux.",
                "keyPoints": ["Sans écrit, l'usage des lieux fixe le délai."],
                "citations": [{"law_id": textes.code_civil, "article_number": "1736",
                               "title": "Le congé d'un bail verbal suit l'usage."}],
                "citizenImpact": "Renseignez-vous sur l'usage local.",
            },
        })
        assert cree.json()["controle_ia"] is None
        assert cree.json()["etat_publication"] == "sources_absentes"
        assert (await _controle(admin_client, cree.json()["id"])).json()["code"] == "sources_absentes"

        sources = await admin_client.post(f"{BLOG}/{cree.json()['id']}/sources")
        reponse = await _controle(admin_client, cree.json()["id"])

        assert sources.status_code == 200, sources.text
        assert reponse.status_code == 200, reponse.text
        (appel,) = appels_du_controle(controleur.llm)
        assert "[1] Code civil applicable au Cameroun" in appel["prompt"]
        assert "Article 1736" in appel["prompt"]
        assert "Explication de la citation 1, qui accompagne l'extrait 1 (Article 1736)" in appel["prompt"]

    async def test_une_citation_ajoutee_apres_la_generation_devient_un_extrait(
        self, admin_client, generateur, controleur, textes
    ):
        post = (await _genere(admin_client))["post"]
        citations = post["botAnalysis"]["citations"] + [
            {"law_id": textes.code_penal, "article_number": "322", "title": "La filouterie."}
        ]
        await admin_client.put(f"{BLOG}/{post['id']}", json={"bot_analysis": {"citations": citations}})

        await _controle(admin_client, post["id"])

        (appel,) = appels_du_controle(controleur.llm)
        # Les trois sources de la génération, puis l'article ajouté.
        assert f"[4] {TITRE_CODE_PENAL} — Article 322" in appel["prompt"]
        assert "Explication de la citation 3, qui accompagne l'extrait 4 (Article 322)" in appel["prompt"]

    async def test_une_phrase_oubliee_par_le_modele_est_a_relire(
        self, admin_client, generateur, controleur
    ):
        post = (await _genere(admin_client))["post"]

        def sans_la_premiere(prompt):
            identifiants = [i for i, _ in phrases_du_prompt(prompt)]
            passage = passage_de(extraits_du_prompt(prompt)[1])
            return {"response": json.dumps({"phrases": [
                jugement(i, "Soutenue", passage, [1, 99]) for i in identifiants[1:]
            ] + [jugement("P999", "soutenue", passage, [1])]}), "fin": "STOP"}

        controleur.llm = modele_double(controle=sans_la_premiere)
        controle = (await _controle(admin_client, post["id"])).json()["controle_ia"]

        premiere, seconde = controle["phrases"][:2]
        assert (premiere["verdict"], premiere["raison"]) == ("non_soutenue", RAISON_NON_JUGEE)
        # Un numéro d'extrait qui n'existe pas est retiré ; un verdict mal écrit est relu.
        assert (seconde["verdict"], seconde["extraits"]) == ("soutenue", [1])
        assert "P999" not in {p["id"] for p in controle["phrases"]}

    @pytest.mark.parametrize("methode", ["sans_jeton", "compte_ordinaire"])
    async def test_reserve_aux_administrateurs(self, client, auth_headers, methode):
        options = {"headers": auth_headers} if methode == "compte_ordinaire" else {}

        reponse = await client.post(f"{BLOG}/1/controle", **options)

        assert reponse.status_code == (403 if methode == "compte_ordinaire" else 401)


# ==================== 3. LA RELECTURE ET L'EMPREINTE ====================


class TestLaRelecture:

    async def test_cocher_des_parties(self, admin_client, generateur, test_admin_user):
        post = (await _genere(admin_client))["post"]

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={
            "relecture": {"parties": {"resume": True, "contexte": True},
                          "controle": post["controle_ia"]["fait_le"],
                          "par": "quelqu'un d'autre", "le": "2000-01-01"},
        })

        relecture = reponse.json()["relecture"]
        assert relecture["parties"] == {
            "titre": False, "contexte": True, "resume": True, "points_cles": False, "impact": False,
            "explications": False, "question": False,
        }
        # Noté par le serveur : le corps n'y peut rien.
        assert relecture["par"] == (test_admin_user.full_name or test_admin_user.username)
        assert relecture["le"] != "2000-01-01"
        assert re.match(r"^\d{4}-\d{2}-\d{2}T", relecture["le"])
        assert reponse.json()["etat_publication"] == "relecture_incomplete"

    @pytest.mark.parametrize("relecture, raison", [
        ({"parties": {"slug": True}}, "partie(s) inconnue(s) : slug"),
        ({"parties": {"resume": "oui"}}, "vrai ou faux attendu"),
        ({"parties": ["resume"]}, "objet attendu"),
    ])
    async def test_des_cases_illisibles_422(self, admin_client, generateur, relecture, raison):
        post = (await _genere(admin_client))["post"]

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={"relecture": relecture})

        assert reponse.status_code == 422
        assert reponse.json()["code"] == "entree_invalide"
        assert raison in reponse.json()["detail"]

    async def test_une_modification_perime_le_controle_et_decoche_sa_partie(
        self, admin_client, generateur
    ):
        """Même cochée dans la même requête : la case valait pour l'ancien texte."""
        post = await _publiable(admin_client, generateur)

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={
            "bot_analysis": {"summary": "Un nouveau résumé."}, "relecture": relecture_cochee(post),
        })

        modifie = reponse.json()
        assert modifie["etat_publication"] == "controle_a_relancer"
        assert modifie["relecture"]["parties"] == {
            "titre": True, "contexte": True, "resume": False, "points_cles": True, "impact": True,
            "explications": True, "question": True,
        }
        # Le contrôle d'avant reste lisible, périmé.
        assert modifie["controle_ia"] == post["controle_ia"]

    @pytest.mark.parametrize("colonne, valeur, partie", [
        ("key_points", json.dumps(["Un point réécrit en base."]), "points_cles"),
        ("context", "Un contexte réécrit en base.", "contexte"),
        ("suggested_question", "Une autre question ?", "question"),
    ])
    async def test_une_modification_hors_de_l_api_aussi(
        self, admin_client, generateur, db_session, colonne, valeur, partie
    ):
        """L'empreinte se juge à la lecture : une écriture directe en base ne passe pas inaperçue."""
        post = await _publiable(admin_client, generateur)
        cast = "CAST(:v AS jsonb)" if colonne == "key_points" else ":v"
        await db_session.execute(
            text(f"UPDATE blog_posts SET {colonne} = {cast} WHERE id = :id"),
            {"v": valeur, "id": post["id"]},
        )
        await db_session.commit()
        _oublier_les_articles(db_session)

        relu = await _admin(admin_client, post["id"])

        assert relu["etat_publication"] == "controle_a_relancer"
        assert relu["relecture"]["parties"][partie] is False
        assert sum(relu["relecture"]["parties"].values()) == len(PARTIES) - 1

    async def test_retirer_une_citation_decoche_les_explications(self, admin_client, generateur):
        post = await _publiable(admin_client, generateur)

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={
            "bot_analysis": {"citations": post["botAnalysis"]["citations"][:1]},
        })

        assert reponse.json()["relecture"]["parties"]["explications"] is False
        assert reponse.json()["etat_publication"] == "controle_a_relancer"

    async def test_le_formulaire_renvoye_tel_quel_ne_perime_rien(self, admin_client, generateur):
        """Le front renvoie tout le formulaire, citations comprises : rien n'a changé, rien ne se périme."""
        post = await _publiable(admin_client, generateur)

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json=_corps_du_front(post))

        assert reponse.status_code == 200, reponse.text
        assert reponse.json()["etat_publication"] == "publiable"
        assert reponse.json()["controle_ia"]["empreinte"] == post["controle_ia"]["empreinte"]

    async def test_une_partie_vide_n_a_rien_a_relire(self, admin_client, generateur):
        generateur.llm = modele_double(article_du_modele(suggested_question=""))
        post = (await _genere(admin_client))["post"]

        assert post["relecture"]["parties"]["question"] is True
        assert not any(p["partie"] == "question" for p in post["controle_ia"]["phrases"])

    async def test_decocher_une_partie(self, admin_client, generateur):
        post = await _publiable(admin_client, generateur)

        reponse = await admin_client.put(
            f"{BLOG}/{post['id']}", json={"relecture": {"parties": {"impact": False}}}
        )

        assert reponse.json()["relecture"]["parties"]["impact"] is False
        assert reponse.json()["etat_publication"] == "relecture_incomplete"


# ==================== 4. LA PUBLICATION ====================


class TestLaPublication:

    async def test_un_article_publiable_se_publie_et_se_lit(self, admin_client, client, generateur):
        post = await _en_ligne(admin_client, generateur)

        page = await client.get(f"{BLOG}/{post['slug']}")
        liste = (await client.get(BLOG)).json()

        assert post["isPublished"] is True
        assert page.status_code == 200
        assert [p["id"] for p in liste] == [post["id"]]
        assert post["etat_publication"] == "publiable" and post["relecture"]["par"]
        # Avenant du tour 7 : rien de la relecture dans une réponse publique (TestLePublic).
        assert not set(page.json()) & CHAMPS_DE_L_ADMINISTRATION

    async def test_refuse_sans_controle(self, admin_client, generateur):
        generateur.llm = modele_double(controle=MistralOverloadedError("503"))
        post = (await _genere(admin_client))["post"]

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={"is_published": True})
        # Avenant du tour 5 : sans contrôle enregistré, aucune case n'a rien à attester.
        cochees = await admin_client.put(
            f"{BLOG}/{post['id']}", json={"relecture": relecture_cochee(post), "is_published": True}
        )

        assert reponse.status_code == 422
        assert reponse.json()["code"] == "controle_a_relancer"
        assert "n'a pas été fait" in reponse.json()["detail"]
        assert cochees.status_code == 422
        assert cochees.json()["code"] == "relecture_perimee"
        assert "aucun contrôle IA n'est enregistré" in cochees.json()["detail"]
        # Refusée d'un bloc : même les cases n'ont pas été enregistrées.
        relu = await _admin(admin_client, post["id"])
        assert relu["isPublished"] is False
        assert not any(relu["relecture"]["parties"][p] for p in ("contexte", "resume", "impact"))

    async def test_refuse_avec_un_controle_perime(self, admin_client, generateur, db_session):
        post = await _publiable(admin_client, generateur)
        await db_session.execute(
            text("UPDATE blog_posts SET summary = 'Réécrit en base.' WHERE id = :id"),
            {"id": post["id"]},
        )
        await db_session.commit()
        _oublier_les_articles(db_session)

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={"is_published": True})

        assert reponse.status_code == 422
        assert reponse.json()["code"] == "controle_a_relancer"
        assert "parties modifiées : résumé" in reponse.json()["detail"]

    async def test_refuse_avec_une_phrase_contredite(self, admin_client, generateur):
        generateur.llm = modele_double(controle={"Entrer chez quelqu'un": "contredite"})
        post = (await _genere(admin_client))["post"]
        await admin_client.put(f"{BLOG}/{post['id']}", json={"relecture": relecture_cochee(post)})

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={"is_published": True})

        assert reponse.status_code == 422
        assert reponse.json()["code"] == "phrases_contredites"
        assert "une phrase que les extraits contredisent" in reponse.json()["detail"]
        assert "points clés : « Entrer chez quelqu'un" in reponse.json()["detail"]

    async def test_refuse_avec_une_relecture_incomplete(self, admin_client, generateur):
        post = (await _genere(admin_client))["post"]
        parties = {p: True for p in PARTIES if p != "explications"}

        reponse = await admin_client.put(
            f"{BLOG}/{post['id']}", json={"relecture": relecture_cochee(post, parties), "is_published": True}
        )

        assert reponse.status_code == 422
        assert reponse.json()["code"] == "relecture_incomplete"
        assert reponse.json()["detail"].endswith("puis à cocher : explications des citations.")

    async def test_refuse_avec_une_citation_qui_n_est_plus_verifiee(
        self, admin_client, generateur, db_session, textes
    ):
        post = await _publiable(admin_client, generateur)
        await db_session.execute(
            text("UPDATE laws SET status = 'archived' WHERE id = :id"), {"id": textes.code_civil}
        )
        await db_session.commit()

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={"is_published": True})

        assert reponse.status_code == 422
        assert reponse.json()["code"] == "citations_non_verifiees"
        assert "la citation 2 n'est pas vérifiée" in reponse.json()["detail"]

    async def test_refuse_sans_aucune_citation(self, admin_client, generateur):
        """Revue du tour 2 : on retirait les citations d'un article d'avant, et sa prose repartait en ligne."""
        post = await _publiable(admin_client, generateur)

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={
            "bot_analysis": {"citations": []}, "is_published": True,
        })

        assert reponse.status_code == 422
        assert reponse.json()["code"] == "citations_non_verifiees"
        assert "ne cite aucun article" in reponse.json()["detail"]
        assert len((await _admin(admin_client, post["id"]))["botAnalysis"]["citations"]) == 2

    async def test_le_premier_blocage_dans_l_ordre(self, admin_client, db_session):
        """Un article d'avant cumule tout : ses sources d'abord, que le reste suppose."""
        post_id = await article_d_avant(db_session, publie=False)

        reponse = await admin_client.put(f"{BLOG}/{post_id}", json={"is_published": True})
        # Ses cases, elles, n'ont aucun contrôle à attester (avenant du tour 5).
        cochees = await admin_client.put(
            f"{BLOG}/{post_id}", json={"relecture": TOUT_RELU, "is_published": True}
        )

        assert reponse.json()["code"] == "sources_absentes"
        assert "Reconstruisez-les" in reponse.json()["detail"]
        assert cochees.json()["code"] == "relecture_perimee"

    async def test_modifier_le_texte_d_un_article_en_ligne_sans_le_retirer(
        self, admin_client, client, generateur
    ):
        post = await _en_ligne(admin_client, generateur)

        reponse = await admin_client.put(
            f"{BLOG}/{post['id']}", json={"bot_analysis": {"summary": "Un résumé à relire."}}
        )

        assert reponse.status_code == 422
        assert reponse.json()["code"] == "controle_a_relancer"
        relu = await _admin(admin_client, post["id"])
        assert relu["isPublished"] is True
        assert relu["botAnalysis"]["summary"] == post["botAnalysis"]["summary"]
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 200

    async def test_modifier_et_retirer_d_un_bloc(self, admin_client, client, generateur):
        """« Publier les modifications » : le texte part en brouillon, hors ligne, d'un bloc."""
        post = await _en_ligne(admin_client, generateur)

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json=_corps_du_front(
            post, summary="Un résumé à relire.", is_published=False,
        ))

        assert reponse.status_code == 200, reponse.text
        assert reponse.json()["isPublished"] is False
        assert reponse.json()["botAnalysis"]["summary"] == "Un résumé à relire."
        assert reponse.json()["relecture"]["parties"]["resume"] is False
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 404

    async def test_rester_en_ligne_apres_une_modification_hors_relecture(
        self, admin_client, client, generateur
    ):
        """Rubrique, mots-clés, durée : ni contrôlés ni relus, l'article reste en ligne."""
        post = await _en_ligne(admin_client, generateur)

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={
            "category": "Droit pénal", "tags": ["domicile"], "reading_time": "3 min",
        })

        assert reponse.status_code == 200, reponse.text
        assert reponse.json()["isPublished"] is True
        assert reponse.json()["etat_publication"] == "publiable"
        assert (await client.get(f"{BLOG}/{post['slug']}")).json()["category"] == "Droit pénal"

    async def test_le_titre_d_un_article_en_ligne_se_relit(self, admin_client, client, generateur):
        """
        Revue du tour 3 : un titre changé en « Licenciement abusif : le salarié doit
        prouver l'abus… » partait en ligne d'un seul PUT, sans contrôle ni relecture.
        """
        post = await _en_ligne(admin_client, generateur)
        faux = "Licenciement abusif : le salarié doit prouver l'abus pour être indemnisé"

        refuse = await admin_client.put(f"{BLOG}/{post['id']}", json={"title": faux})
        toujours_en_ligne = await client.get(f"{BLOG}/{post['slug']}")
        retire = await admin_client.put(f"{BLOG}/{post['id']}", json={"title": faux, "is_published": False})

        assert refuse.status_code == 422 and refuse.json()["code"] == "controle_a_relancer"
        assert "parties modifiées : titre" in refuse.json()["detail"]
        assert toujours_en_ligne.json()["title"] == post["title"]
        assert retire.status_code == 200, retire.text
        assert retire.json()["relecture"]["parties"]["titre"] is False
        assert retire.json()["etat_publication"] == "controle_a_relancer"
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 404


# ==================== 5. PUT : RETIRER, OU TOUT D'UN BLOC ====================


class TestLeCorpsDuPut:

    async def test_retirer_exactement_retire_toujours(self, admin_client, client, db_session):
        """Un article d'avant, en ligne en base, qu'aucune modification ne peut valider."""
        post_id = await article_d_avant(db_session)

        reponse = await admin_client.put(f"{BLOG}/{post_id}", json={"is_published": False})

        assert reponse.status_code == 200, reponse.text
        assert reponse.json()["isPublished"] is False
        # Rien d'autre n'a changé : citations d'avant, dossier, texte.
        dossier = (await db_session.execute(
            text("SELECT citations FROM blog_posts WHERE id = :id"), {"id": post_id}
        )).scalar_one()
        assert isinstance(dossier, list) and len(dossier) == 2

    async def test_retirer_un_brouillon_ne_change_rien(self, admin_client, generateur):
        post = (await _genere(admin_client))["post"]

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={"is_published": False})

        assert reponse.status_code == 200
        assert reponse.json()["isPublished"] is False

    @pytest.mark.parametrize("corps, statut, code", [
        ({"is_published": False, "slug": "le-meme-que-l-autre"}, 409, "slug_pris"),
        ({"is_published": False, "bot_analysis": {"keyPoints": ["x"] * 13}}, 422, "entree_invalide"),
        ({"is_published": False, "bot_analysis": {"citations": [{"law_id": 999999,
                                                                 "article_number": "1"}]}},
         422, "entree_invalide"),
        ({"is_published": False, "relecture": {"parties": {"slug": True}}}, 422, "entree_invalide"),
        ({"is_published": "non"}, 422, "entree_invalide"),
    ], ids=["adresse-prise", "trop-de-points", "citation-refusee", "case-inconnue", "booleen"])
    async def test_un_corps_refuse_ne_retire_rien(
        self, admin_client, client, generateur, corps, statut, code
    ):
        """Revue du tour 2 : un refus laissait l'article hors ligne, et le front le croyait en ligne."""
        autre = (await _genere(admin_client))["post"]
        await admin_client.put(f"{BLOG}/{autre['id']}", json={"slug": "le-meme-que-l-autre"})
        post = await _en_ligne(admin_client, generateur)

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json=corps)

        assert reponse.status_code == statut, reponse.text
        assert reponse.json()["code"] == code
        assert "retiré" not in reponse.json()["detail"]
        relu = await _admin(admin_client, post["id"])
        assert relu["isPublished"] is True and relu["slug"] == post["slug"]
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 200

    async def test_un_corps_plus_large_que_le_retrait_est_valide_en_entier(
        self, admin_client, generateur
    ):
        """{"is_published": false, "title": …} n'est PAS le retrait seul : tout est appliqué d'un bloc."""
        post = await _en_ligne(admin_client, generateur)

        reponse = await admin_client.put(
            f"{BLOG}/{post['id']}", json={"is_published": False, "title": "Retiré et renommé"}
        )

        assert reponse.status_code == 200
        assert (reponse.json()["isPublished"], reponse.json()["title"]) == (False, "Retiré et renommé")


# ==================== 6. LE PUBLIC ====================


class TestLePublic:

    async def test_seuls_les_articles_publies_et_publiables(
        self, admin_client, client, generateur, db_session
    ):
        """Les articles d'avant (prose fausse du 26/10) et les articles devenus non publiables : jamais."""
        await article_d_avant(db_session, slug="rupture-contrat-travail-preavis-indemnites")
        await db_session.execute(text(
            "UPDATE blog_posts SET key_points = CAST(:kp AS jsonb) WHERE slug = :slug"
        ), {"kp": json.dumps(["Une indemnité distincte plafonnée entre 1 et 12 mois de salaire."]),
            "slug": "rupture-contrat-travail-preavis-indemnites"})
        await db_session.commit()
        en_ligne = await _en_ligne(admin_client, generateur)
        perime = await _en_ligne(admin_client, generateur)
        await db_session.execute(
            text("UPDATE blog_posts SET summary = 'Réécrit hors contrôle.' WHERE id = :id"),
            {"id": perime["id"]},
        )
        await db_session.commit()
        _oublier_les_articles(db_session)

        liste = await client.get(BLOG)
        pages = [
            await client.get(f"{BLOG}/{slug}")
            for slug in ("rupture-contrat-travail-preavis-indemnites", perime["slug"], en_ligne["slug"])
        ]

        assert [p["id"] for p in liste.json()] == [en_ligne["id"]]
        assert [p.status_code for p in pages] == [404, 404, 200]
        assert "plafonnée entre 1 et 12 mois" not in liste.text
        assert "Réécrit hors contrôle" not in liste.text

    async def test_la_page_compte_les_articles_servis(self, admin_client, client, generateur, db_session):
        """`limit` et `offset` comptent les articles servis, pas ceux que la requête écarte ensuite."""
        await article_d_avant(db_session, slug="ancien-1")
        servis = [await _en_ligne(admin_client, generateur) for _ in range(3)]
        await article_d_avant(db_session, slug="ancien-2")

        premiere = (await client.get(BLOG, params={"limit": 2})).json()
        suite = (await client.get(BLOG, params={"limit": 2, "offset": 2})).json()

        assert [p["id"] for p in premiere] == [servis[2]["id"], servis[1]["id"]]
        assert [p["id"] for p in suite] == [servis[0]["id"]]

    async def test_ni_le_controle_ni_la_relecture_dans_une_reponse_publique(
        self, admin_client, client, generateur
    ):
        """
        Avenant du tour 7 (essai réel du 04/11) : GET /blog et GET /blog/{slug} rendaient
        `controle_ia`, chaque verdict, « À vérifier » compris, et ses raisons ; des notes internes,
        lisibles par quiconque appelle l'API. Administration seulement.
        """
        generateur.llm = modele_double(controle={"Un locataire trouve": "non_soutenue"})
        brouillon = (await _genere(admin_client))["post"]
        reponse = await admin_client.put(
            f"{BLOG}/{brouillon['id']}", json={"relecture": relecture_cochee(brouillon), "is_published": True},
        )
        assert reponse.status_code == 200, reponse.text
        admin = reponse.json()

        page = await client.get(f"{BLOG}/{admin['slug']}")
        liste = await client.get(BLOG)

        assert page.status_code == liste.status_code == 200
        for public in (page.json(), liste.json()[0]):
            assert set(public) == {
                "id", "slug", "title", "category", "publishedAt", "readingTime", "context", "botAnalysis",
                "suggestedQuestion", "tags", "isPublished",
            }
            assert set(public["botAnalysis"]) == {"summary", "keyPoints", "citations", "citizenImpact"}
        for brut in (page.text, liste.text):
            for note in (
                *CHAMPS_DE_L_ADMINISTRATION, "phrases_confirmees", "deja_publie", '"nature"', '"verdict"',
                "non_soutenue", "Verdict de la doublure", admin["relecture"]["par"],
                admin["controle_ia"]["fait_le"],
            ):
                assert note not in brut, note
        # L'administration, elle, a tout.
        assert CHAMPS_DE_L_ADMINISTRATION - {"controleIa", "etatPublication"} <= set(admin)
        for lu in (
            await _admin(admin_client, admin["id"]),
            (await admin_client.get(f"{BLOG}/admin/all")).json()[0],
        ):
            assert lu["controle_ia"] == admin["controle_ia"] and lu["relecture"] == admin["relecture"]
            assert lu["etat_publication"] == "publiable"


# ==================== 7. LES RENVOIS AUX EXTRAITS ====================


class TestLesRenvoisAuxExtraits:
    """Essai réel du 30/10 : « (extrait 2) », « (extraits 5 et 9) » dans les points clés publiés."""

    async def test_retires_de_la_generation(self, admin_client, generateur):
        generateur.llm = modele_double(article_du_modele(
            key_points=[
                "Entrer chez quelqu'un contre son gré est une infraction (extrait 1).",
                "Sans bail écrit, le congé suit l'usage des lieux (extraits 2 et 3).",
                "Le bail se termine selon les règles du Code civil [2].",
            ],
            citations=[{"extrait": 1, "objet": "La violation de domicile est punie (extrait 1)."}],
        ))

        post = (await _genere(admin_client))["post"]

        assert post["botAnalysis"]["keyPoints"] == [
            "Entrer chez quelqu'un contre son gré est une infraction.",
            "Sans bail écrit, le congé suit l'usage des lieux.",
            "Le bail se termine selon les règles du Code civil.",
        ]
        assert post["botAnalysis"]["citations"][0]["title"] == "La violation de domicile est punie."

    async def test_un_renvoi_dans_la_phrase_est_signale(self, admin_client, generateur):
        generateur.llm = modele_double(article_du_modele(
            citizen_impact="Comme le précise l'extrait 2, le congé suit l'usage des lieux.",
        ))

        corps = await _genere(admin_client)

        assert any("« l'extrait 2 »" in a and "Reformulez" in a for a in corps["avertissements"])

    async def test_retires_d_une_modification(self, admin_client, generateur):
        post = (await _genere(admin_client))["post"]

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={
            "bot_analysis": {"summary": "Le congé suit l'usage (extrait 2).",
                             "keyPoints": ["Un point\nsur deux lignes (extrait 1)."]},
        })

        assert reponse.json()["botAnalysis"]["summary"] == "Le congé suit l'usage."
        assert reponse.json()["botAnalysis"]["keyPoints"] == ["Un point sur deux lignes."]


# ==================== 8. L'AVENANT DU TOUR 5 : LA CASE NOMME LE CONTRÔLE RELU ====================


class TestLaCaseNommeLeControleRelu:
    """
    Revue du tour 4 (défauts D3 et D4) : l'API attestait le contrôle enregistré au moment du
    PUT ; relancé ailleurs (autre onglet, autre administrateur, réponse perdue), un contrôle dont
    personne n'avait lu les verdicts passait pour relu, et l'article partait en ligne en un PUT.
    """

    async def test_deux_onglets(self, admin_client, client, generateur, controleur):
        vu_par_a = (await _genere(admin_client))["post"]
        controleur.llm = modele_double(controle={"Un locataire trouve": "non_soutenue"})
        relance_par_b = (await _controle(admin_client, vu_par_a["id"])).json()

        refus = await admin_client.put(
            f"{BLOG}/{vu_par_a['id']}", json={"relecture": relecture_cochee(vu_par_a), "is_published": True},
            headers=ORIGINE,
        )

        assert refus.status_code == 422
        assert refus.json()["code"] == "relecture_perimee"
        assert f"fait le {relance_par_b['controle_ia']['fait_le']}" in refus.json()["detail"]
        assert refus.headers["access-control-allow-origin"] == ORIGINE["Origin"]
        relu = await _admin(admin_client, vu_par_a["id"])
        # Le contrôle de B a une phrase de règle à confirmer : personne ne l'a lue.
        assert relu["isPublished"] is False and relu["etat_publication"] == "phrases_a_confirmer"
        assert not any(relu["relecture"]["parties"][p] for p in PARTIES if p != "question")
        assert (await client.get(f"{BLOG}/{vu_par_a['slug']}")).status_code == 404
        # Relu contre le contrôle enregistré : accepté.
        publie = await admin_client.put(
            f"{BLOG}/{vu_par_a['id']}", json={"relecture": relecture_cochee(relance_par_b), "is_published": True},
        )
        assert publie.status_code == 200, publie.text
        assert publie.json()["etat_publication"] == "publiable"
        assert publie.json()["relecture"]["controle"] == relance_par_b["controle_ia"]["fait_le"]

    async def test_deux_controles_finis_dans_la_meme_seconde(
        self, admin_client, generateur, controleur, monkeypatch
    ):
        """`fait_le` est à la seconde : le second contrôle enregistré a quand même le sien."""
        monkeypatch.setattr(controle_du_billet, "maintenant_iso", lambda: "2026-11-04T15:00:00+00:00")
        vu_par_a = (await _genere(admin_client))["post"]
        relance = (await _controle(admin_client, vu_par_a["id"])).json()

        refus = await admin_client.put(f"{BLOG}/{vu_par_a['id']}", json={"relecture": relecture_cochee(vu_par_a)})

        assert vu_par_a["controle_ia"]["fait_le"] == "2026-11-04T15:00:00+00:00"
        assert relance["controle_ia"]["fait_le"] == "2026-11-04T15:00:01+00:00"
        assert refus.json()["code"] == "relecture_perimee"

    @pytest.mark.parametrize("cas", ["relecture", "publier", "modifier", "retirer-et-modifier", "formulaire"])
    @pytest.mark.parametrize("controle", [None, "", "2026-01-01T00:00:00+00:00"], ids=["absent", "vide", "un-autre"])
    async def test_chaque_chemin_du_put_qui_coche(self, admin_client, client, generateur, cas, controle):
        """Toute requête qui coche une partie nomme le contrôle relu ; refusée, elle n'applique RIEN."""
        post = await _en_ligne(admin_client, generateur)
        coche = {"parties": {"contexte": True, "impact": False}}
        if controle is not None:
            coche["controle"] = controle
        corps = {
            "relecture": {"relecture": coche},
            "publier": {"relecture": coche, "is_published": True},
            "modifier": {"relecture": coche, "category": "Droit pénal", "bot_analysis": {"summary": "Corrigé."}},
            "retirer-et-modifier": {
                "relecture": coche, "is_published": False, "title": "Retitré", "slug": "une-autre-adresse",
            },
            "formulaire": {**_corps_du_front(post, summary="Corrigé."), "relecture": coche},
        }[cas]

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json=corps)

        assert reponse.status_code == 422, reponse.text
        assert reponse.json()["code"] == "relecture_perimee"
        relu = await _admin(admin_client, post["id"])
        for champ in ("title", "slug", "category", "isPublished", "botAnalysis", "relecture", "controle_ia"):
            assert relu[champ] == post[champ], champ
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 200

    async def test_le_bon_controle_sur_chaque_chemin(self, admin_client, client, generateur):
        post = (await _genere(admin_client))["post"]
        fait_le = post["controle_ia"]["fait_le"]

        modifie = await admin_client.put(f"{BLOG}/{post['id']}", json={
            "bot_analysis": {"summary": "Un résumé corrigé."}, "relecture": relecture_cochee(post),
        })
        formulaire = await admin_client.put(
            f"{BLOG}/{post['id']}", json={**_corps_du_front(post), "relecture": relecture_cochee(post)},
        )

        # La partie modifiée tombe, les autres sont cochées contre le contrôle nommé.
        assert modifie.status_code == 200, modifie.text
        assert modifie.json()["relecture"]["parties"] == {p: p != "resume" for p in PARTIES}
        assert modifie.json()["relecture"]["controle"] == fait_le
        assert modifie.json()["etat_publication"] == "controle_a_relancer"
        assert formulaire.status_code == 200, formulaire.text

    async def test_chaque_reponse_rend_le_controle_relu(self, admin_client, client, generateur, controleur):
        corps = await _genere(admin_client)
        post = corps["post"]
        fait_le = post["controle_ia"]["fait_le"]

        coche = (await admin_client.put(f"{BLOG}/{post['id']}", json={"relecture": relecture_cochee(post)})).json()
        publie = (await admin_client.put(f"{BLOG}/{post['id']}", json={"is_published": True})).json()
        lectures = {
            "generate": post["relecture"]["controle"],
            "PUT": coche["relecture"]["controle"],
            "admin/{id}": (await _admin(admin_client, post["id"]))["relecture"]["controle"],
            "admin/all": (await admin_client.get(f"{BLOG}/admin/all")).json()[0]["relecture"]["controle"],
        }
        relance = (await admin_client.post(f"{BLOG}/{post['id']}/controle", json={"confirmer": True})).json()

        assert lectures == {"generate": None, "PUT": fait_le, "admin/{id}": fait_le, "admin/all": fait_le}
        assert publie["relecture"]["controle"] == fait_le
        # Un nouveau contrôle ne change pas ce que les cases ont attesté, ni contre quoi.
        assert relance["relecture"]["controle"] == fait_le
        assert relance["controle_ia"]["fait_le"] != fait_le

    async def test_decocher_ne_nomme_aucun_controle(self, admin_client, generateur):
        post = await _publiable(admin_client, generateur)

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={"relecture": {"parties": {"impact": False}}})

        assert reponse.status_code == 200, reponse.text
        assert reponse.json()["relecture"]["parties"]["impact"] is False

    @pytest.mark.parametrize("controle, raison", [
        (12, "relecture › contrôle relu : texte attendu"),
        ("x" * 65, "relecture › contrôle relu : 64 caractères au maximum"),
    ])
    async def test_un_controle_illisible_422(self, admin_client, generateur, controle, raison):
        post = (await _genere(admin_client))["post"]

        reponse = await admin_client.put(
            f"{BLOG}/{post['id']}", json={"relecture": {"parties": {"titre": True}, "controle": controle}},
        )

        assert (reponse.status_code, reponse.json()["code"]) == (422, "entree_invalide")
        assert raison in reponse.json()["detail"]


# ==================== 9. L'AVENANT DU TOUR 5 : UN ARTICLE EN LIGNE ====================


class TestRelancerLeControleDUnArticleEnLigne:
    """
    Essai réel du tour 4 : à la température 0, des verdicts changent d'un contrôle à l'autre.
    Relancé sur un article en ligne, un contrôle qui jugeait autrement une partie la décochait,
    et l'article quittait le public sans que l'administrateur l'ait voulu ni su.
    """

    @pytest.mark.parametrize("corps", [None, {}, {"confirmer": False}, {"confirmer": "oui"}, {"confirmer": 1}, []],
                             ids=["sans-corps", "vide", "faux", "chaine", "un", "liste"])
    async def test_sans_confirmer_409_et_rien_n_est_fait(self, admin_client, client, generateur, controleur, corps):
        post = await _en_ligne(admin_client, generateur)
        options = {} if corps is None else {"json": corps}

        reponse = await admin_client.post(f"{BLOG}/{post['id']}/controle", headers=ORIGINE, **options)

        assert reponse.status_code == 409
        assert reponse.json()["code"] == "article_en_ligne"
        assert "confirmez la relance" in reponse.json()["detail"]
        assert reponse.headers["access-control-allow-origin"] == ORIGINE["Origin"]
        controleur.llm.generate.assert_not_awaited()
        relu = await _admin(admin_client, post["id"])
        assert relu["controle_ia"] == post["controle_ia"] and relu["etat_publication"] == "publiable"
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 200

    async def test_confirme_le_nouveau_controle_peut_retirer_l_article_du_public(
        self, admin_client, client, generateur, controleur
    ):
        post = await _en_ligne(admin_client, generateur)
        controleur.llm = modele_double(controle={"Entrer chez quelqu'un": "non_soutenue"})

        reponse = await admin_client.post(f"{BLOG}/{post['id']}/controle", json={"confirmer": True})

        assert reponse.status_code == 200, reponse.text
        corps = reponse.json()
        assert corps["controle_ia"]["fait_le"] != post["controle_ia"]["fait_le"]
        assert corps["isPublished"] is True
        # Une phrase de règle non trouvée : à confirmer, avant même la relecture de sa partie.
        assert corps["etat_publication"] == "phrases_a_confirmer"
        assert corps["relecture"]["parties"] == {p: p != "points_cles" for p in PARTIES}
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 404
        # Relu et confirmé contre le nouveau contrôle : servi de nouveau.
        relu = await admin_client.put(f"{BLOG}/{post['id']}", json={"relecture": relecture_cochee(corps)})
        assert relu.json()["etat_publication"] == "publiable"
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 200

    @pytest.mark.parametrize("verdict_de_y", ["contredite", "non_soutenue"])
    async def test_un_article_cache_par_un_controle_ne_revient_pas_au_suivant(
        self, admin_client, client, generateur, controleur, verdict_de_y
    ):
        """
        Revue du front du tour 5 (data/blog/tour5/revue-front, retour-en-ligne*.test.ts, par la
        vraie API) : le contrôle Y juge le contexte autrement, l'article quitte le public ; le
        contrôle Z redonne le verdict de X (les verdicts changent d'un contrôle à l'autre à la
        température 0). La case du contexte, cochée contre X, revivait : l'article revenait en
        ligne sans que personne ait lu ce que Y avait trouvé, ni relu, ni publié.
        """
        post = await _en_ligne(admin_client, generateur)
        controleur.llm = modele_double(controle={"Un locataire trouve ses affaires": verdict_de_y})
        cache = (await _controle(admin_client, post["id"], json={"confirmer": True})).json()
        assert cache["relecture"]["parties"]["contexte"] is False
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 404

        controleur.llm = modele_double()
        relance = await _controle(admin_client, post["id"], json={"confirmer": True})

        assert relance.status_code == 200, relance.text
        corps = relance.json()
        assert corps["relecture"]["parties"] == {p: p != "contexte" for p in PARTIES}
        assert corps["etat_publication"] == "relecture_incomplete"
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 404
        # Relu contre le contrôle enregistré, par une personne : servi de nouveau.
        relu = await admin_client.put(
            f"{BLOG}/{post['id']}", json={"relecture": relecture_cochee(corps, {"contexte": True})}
        )
        assert relu.json()["etat_publication"] == "publiable"
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 200

    async def test_publie_pendant_le_controle_rien_n_est_enregistre(
        self, admin_client, client, generateur, controleur, db_session
    ):
        """Un autre onglet publie pendant le contrôle : ses verdicts ne s'enregistrent pas en silence."""
        post = await _publiable(admin_client, generateur)
        repondre = controleur.llm.generate.side_effect

        async def publier_pendant(**kwargs):
            await db_session.execute(
                text("UPDATE blog_posts SET is_published = true WHERE id = :id"), {"id": post["id"]}
            )
            await db_session.commit()
            return await repondre(**kwargs)

        controleur.llm.generate = AsyncMock(side_effect=publier_pendant)

        reponse = await _controle(admin_client, post["id"])

        assert (reponse.status_code, reponse.json()["code"]) == (409, "article_en_ligne")
        assert "publié pendant le contrôle" in reponse.json()["detail"]
        relu = await _admin(admin_client, post["id"])
        assert relu["controle_ia"] == post["controle_ia"] and relu["etat_publication"] == "publiable"
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 200

    async def test_un_brouillon_n_a_rien_a_confirmer(self, admin_client, generateur, controleur):
        post = (await _genere(admin_client))["post"]

        reponse = await _controle(admin_client, post["id"])

        assert reponse.status_code == 200, reponse.text
        assert len(appels_du_controle(controleur.llm)) == 1

    async def test_un_article_d_avant_en_ligne_dit_d_abord_ses_sources(self, admin_client, controleur, db_session):
        """Sans sources, rien à relancer : reconstruire les sources (qui dépublie) vient d'abord."""
        post_id = await article_d_avant(db_session)

        reponse = await _controle(admin_client, post_id)

        assert (reponse.status_code, reponse.json()["code"]) == (422, "sources_absentes")
        controleur.llm.generate.assert_not_awaited()


# ==================== 10. LE DÉLAI DU CONTRÔLE (TOUR 5, D2 ; TOUR 6) ====================


def _controle_au_rythme(
    par_phrase_s: float, phrases_par_appel: list, article=None, moments: list = None
) -> AsyncMock:
    """
    Le modèle doublé, qui juge à `par_phrase_s` secondes par phrase de son prompt, comme
    mistral-large-4 à 3,4 s par phrase au pire mesuré (04/11), en accéléré ; le délai de la
    route l'est aussi (monkeypatch de DELAI_CONTROLE_S). `moments` reçoit le (début, fin) de
    chaque appel de contrôle.
    """
    double = modele_double(article)
    repondre = double.generate.side_effect

    async def generer(**kwargs):
        if kwargs.get("system") == SYSTEM_DU_CONTROLE:
            phrases_par_appel.append(len(phrases_du_prompt(kwargs["prompt"])))
            debut = time.perf_counter()
            await asyncio.sleep(par_phrase_s * phrases_par_appel[-1])
            if moments is not None:
                moments.append((debut, time.perf_counter()))
        return await repondre(**kwargs)

    double.generate = AsyncMock(side_effect=generer)
    return double


def _article_de(phrases_par_partie: int, question: int = 1):
    """Un article généré long : contexte, résumé et impact de N phrases, 8 points clés de 2."""
    def partie(nom: str) -> str:
        return " ".join(f"{nom}, phrase numéro {i}." for i in range(phrases_par_partie))

    return article_du_modele(
        context=partie("Le contexte"), summary=partie("Le résumé"), citizen_impact=partie("L'impact"),
        key_points=[f"Le point {i}, première phrase. Le point {i}, seconde phrase." for i in range(8)],
        suggested_question=" ".join(f"Question numéro {i} ?" for i in range(question)),
    )


class TestLeDelaiDuControle:
    """
    Revue du tour 4 (sondes TestLaDureeDuControle, à l'échelle de 1/100) : 35 et 69 phrases,
    sous le plafond de 80, répondaient 504 « delai » à chaque essai, et l'article ne pouvait
    jamais être publié. Tour 6 : les appels se suivent, l'article se juge dans la SOMME de ses
    appels, et le plafond tombe à 25 phrases. Tour 7 : ici un modèle SANS budget de jetons (la
    doublure), qui garde ses lots ; mistral-large-4 juge en UN appel (TestLePlanDuControleParLaRoute).
    """

    async def test_vingt_cinq_phrases_en_appels_l_un_apres_l_autre_dans_le_delai(
        self, admin_client, generateur, controleur, monkeypatch
    ):
        # 25 × 0,02 s = 0,5 s de modèle, en deux appels qui se suivent, sous un délai de 1,2 s.
        monkeypatch.setattr(routes_du_blog, "DELAI_CONTROLE_S", 1.2)
        a_la_generation, a_la_demande, moments = [], [], []
        generateur.llm = _controle_au_rythme(0.02, a_la_generation, _article_de(1, question=3), moments)
        controleur.llm = _controle_au_rythme(0.02, a_la_demande, moments=moments)

        post = (await _genere(admin_client))["post"]
        relance = await _controle(admin_client, post["id"])

        assert len(post["controle_ia"]["phrases"]) == PHRASES_AU_PLUS == 25
        assert relance.status_code == 200, relance.text
        for appels in (a_la_generation, a_la_demande):
            assert appels == [13, 12] and max(appels) <= PHRASES_PAR_APPEL
        # L'un après l'autre : chaque appel part après la fin du précédent.
        assert len(moments) == 4
        assert all(suivant[0] >= precedent[1] for precedent, suivant in zip(moments, moments[1:]))
        assert [p["id"] for p in relance.json()["controle_ia"]["phrases"]] == [f"P{i}" for i in range(1, 26)]

    async def test_plus_de_vingt_cinq_phrases_dit_de_raccourcir(self, admin_client, generateur, controleur):
        appels = []
        generateur.llm = _controle_au_rythme(0, appels, _article_de(1, question=4))

        corps = await _genere(admin_client)
        reponse = await _controle(admin_client, corps["post"]["id"])

        # Généré, enregistré sans contrôle : l'avertissement dit de raccourcir.
        assert corps["post"]["controle_ia"] is None
        assert any("26 phrases, 25 au plus" in a and "raccourcissez-le" in a for a in corps["avertissements"])
        assert (reponse.status_code, reponse.json()["code"]) == (422, "entree_invalide")
        assert "raccourcissez-le" in reponse.json()["detail"]
        assert appels == []
        controleur.llm.generate.assert_not_awaited()


# ==================== 10 bis. LE PLAN DU CONTRÔLE, PAR LA ROUTE (REVUE DU TOUR 6) ====================

GRAND = "mistral-large-4"


class _Horloge:
    """L'horloge des limiteurs : leurs sommeils avancent le temps, sans vraie attente."""

    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def dormir(self, secondes):
        self.t += secondes

    async def dormir_async(self, secondes):
        self.t += secondes


@pytest.fixture
def grand_modele(monkeypatch):
    """
    Le câblage de production du contrôle : son vrai client (MistralService, mistral-large-4,
    sans secours), les limiteurs que limites_du_modele donne (20 000 jetons par minute) sur
    horloge factice, et mistral-large-4 doublé au niveau HTTP, au coût et à la durée mesurés le
    04/11 (GrandModeleAuRythme, « pire » : le pire appel mesuré). Aucun appel réseau.
    """
    for nom, valeur in {
        "LLM_PROVIDER": "mistral", "BLOG_CONTROLE_MODELE": GRAND, "BLOG_CONTROLE_REFLEXION": "low",
        "BLOG_CONTROLE_JETONS_PAR_MINUTE": 20_000, "MISTRAL_MODEL": "ministral-14b-2512",
        "MISTRAL_MODEL_SECOURS": "ministral-8b-2512", "MISTRAL_ATTENTE_MAX_S": 20.0,
    }.items():
        monkeypatch.setattr(settings, nom, valeur)
    horloge = _Horloge()
    limiteurs = {
        modele: Limiteur(
            modele, horloge=horloge, dormir=horloge.dormir, dormir_async=horloge.dormir_async,
            **mistral_service.limites_du_modele(modele),
        )
        for modele in (GRAND, "ministral-14b-2512", "ministral-8b-2512")
    }
    monkeypatch.setattr(mistral_service, "_LIMITEURS", limiteurs)
    double = GrandModeleAuRythme(horloge, "pire")
    monkeypatch.setattr(mistral_service, "transport_http", httpx.MockTransport(double))
    return SimpleNamespace(horloge=horloge, limiteur=limiteurs[GRAND], double=double)


class TestLePlanDuControleParLaRoute:
    """
    Revue du tour 6 : le second lot d'un contrôle n'avait que les 20 s d'attente du chat, et
    rien ne comptait ensemble les attentes et la durée des lots ; d'où des 503 APRÈS un lot payé,
    à chaque relance, et des 504. Le contrôle se planifie : un plan qui ne tient pas dans le
    délai de la route est refusé avant tout envoi, avec le Retry-After où il tiendra.
    Tour 7 (essai réel du 04/11) : un lot de 11 ou 12 phrases prend 40 à 76 s quelle que soit sa
    taille ; deux lots ont fini en 504 après avoir été payés. UN appel par article, compté au
    pire d'un appel (125 s, et 10 s de marge).
    """

    async def test_un_plan_qui_ne_tient_pas_est_refuse_avant_tout_envoi_puis_aboutit(
        self, admin_client, generateur, controleur, grand_modele
    ):
        """
        Le budget de jetons plein (un appel d'ailleurs sur la même clé : le fournisseur dit 0
        jeton restant), un article de 24 phrases ne finirait pas à temps : 503 `saturation`, un
        message sur le budget de jetons de l'article, le Retry-After où son plan tient (60 s
        d'attente, 125 s d'appel et 10 s de marge pour 165 : 30 s), lisible par le front ; rien
        d'envoyé, rien de perdu. Relancé alors : 200, en UN appel.
        """
        generateur.llm = modele_double(article_du_modele(
            context=" ".join(f"Phrase {i} du contexte." for i in range(8)),
            summary=" ".join(f"Phrase {i} du résumé." for i in range(8)),
        ))
        post = (await _genere(admin_client))["post"]
        controleur.llm = mistral_service.mistral_du_controle()
        grand_modele.limiteur.recaler(jetons_restants_minute=0)

        reponse = await _controle(admin_client, post["id"], headers=ORIGINE)

        assert reponse.status_code == 503, reponse.text
        assert reponse.json()["code"] == "saturation"
        detail = reponse.json()["detail"]
        assert "20 000 jetons par minute" in detail and "Rien n'a été envoyé au modèle" in detail
        reprise = int(reponse.headers["retry-after"])
        assert reprise == 30 and detail.endswith(f"Relancez le contrôle dans {reprise} s.")
        exposes = [e.strip().lower() for e in reponse.headers["access-control-expose-headers"].split(",")]
        assert "retry-after" in exposes
        assert grand_modele.double.requetes == []
        assert (await _admin(admin_client, post["id"]))["controle_ia"] == post["controle_ia"]

        grand_modele.horloge.t += reprise
        relance = await _controle(admin_client, post["id"])

        assert relance.status_code == 200, relance.text
        assert relance.json()["controle_ia"]["modele"] == GRAND
        assert [r["phrases"] for r in grand_modele.double.requetes] == [24]
        assert len(relance.json()["controle_ia"]["phrases"]) == 24

    @pytest.mark.parametrize("cas", ["generation", "demande"])
    async def test_un_article_de_vingt_cinq_phrases_part_en_un_seul_appel(
        self, admin_client, generateur, controleur, grand_modele, cas
    ):
        """
        Le cas de l'essai réel du 04/11 (23 phrases, lots de 12 et 11 : 504 après deux lots payés),
        par les deux routes, au plafond : le prompt porte toutes les phrases, et rien n'y dit
        qu'il en reste à juger à part ; au pire mesuré d'un appel, sous les 165 s de la route.
        """
        generateur.llm = _controle_au_rythme(0, [], _article_de(1, question=3))
        generateur.llm_du_controle = mistral_service.mistral_du_controle()
        controleur.llm = mistral_service.mistral_du_controle()

        post = (await _genere(admin_client))["post"]
        reponse = await _controle(admin_client, post["id"]) if cas == "demande" else None

        requete = grand_modele.double.requetes[-1]
        assert (requete["phrases"], requete["un_seul_appel"]) == (PHRASES_AU_PLUS, True)
        assert (requete["fin"] - requete["depart"]) + MARGE_DU_PLAN_S <= routes_du_blog.DELAI_CONTROLE_S
        corps = reponse.json() if reponse is not None else post
        assert corps["controle_ia"]["modele"] == GRAND and len(corps["controle_ia"]["phrases"]) == PHRASES_AU_PLUS

    async def test_au_dela_du_plafond_rien_ne_part(self, admin_client, generateur, controleur, grand_modele):
        """26 phrases : 422 « raccourcissez-le » par la route, et rien n'est envoyé au modèle."""
        generateur.llm = _controle_au_rythme(0, [], _article_de(1, question=4))
        generateur.llm_du_controle = mistral_service.mistral_du_controle()
        controleur.llm = mistral_service.mistral_du_controle()

        corps = await _genere(admin_client)
        reponse = await _controle(admin_client, corps["post"]["id"])

        assert corps["post"]["controle_ia"] is None
        assert (reponse.status_code, reponse.json()["code"]) == (422, "entree_invalide")
        assert "raccourcissez-le" in reponse.json()["detail"]
        assert grand_modele.double.requetes == []

    async def test_le_modele_ne_change_jamais_quand_le_budget_manque(
        self, admin_client, generateur, controleur, grand_modele
    ):
        """Pas de secours pour le contrôle : refusé, il ne passe ni à ministral ni à un autre."""
        generateur.llm = modele_double()
        post = (await _genere(admin_client))["post"]
        controleur.llm = mistral_service.mistral_du_controle()
        grand_modele.limiteur.recaler(jetons_restants_minute=0)
        assert (await _controle(admin_client, post["id"])).status_code == 503

        grand_modele.horloge.t += 30
        assert (await _controle(admin_client, post["id"])).status_code == 200

        assert {r["modele"] for r in grand_modele.double.requetes} == {GRAND}

    async def test_ce_que_generate_laisse_ne_suffit_pas_le_brouillon_reste_a_controler(
        self, admin_client, generateur, controleur, grand_modele, monkeypatch
    ):
        """
        /generate ne laisse au contrôle que ce qui reste de ses 190 s (30 s au moins) : un plan
        qui n'y tient pas n'envoie rien (avant, ses lots partaient, payés, et l'échéance le
        coupait). Le brouillon est enregistré sans contrôle, l'avertissement dit pourquoi et de
        relancer ; à la demande (165 s), le contrôle aboutit, en un appel.
        """
        monkeypatch.setattr(routes_du_blog, "DELAI_DE_LA_REDACTION_S", 40)
        generateur.llm_du_controle = mistral_service.mistral_du_controle()

        corps = await _genere(admin_client)

        assert corps["post"]["controle_ia"] is None and grand_modele.double.requetes == []
        (avertissement,) = [a for a in corps["avertissements"] if "contrôle IA" in a]
        assert "jetons par minute" in avertissement and "Rien n'a été envoyé au modèle" in avertissement
        assert "relancez le contrôle IA avant de publier" in avertissement
        controleur.llm = mistral_service.mistral_du_controle()
        relance = await _controle(admin_client, corps["post"]["id"])
        assert relance.status_code == 200, relance.text
        assert len(grand_modele.double.requetes) == 1


class TestLesDelaisDesRoutes:
    """
    Tour 7 : les délais des routes pour UN appel de mistral-large-4 (les deux réglages que le
    front attend aussi : DELAI_DE_GENERATION_MS et DELAI_DU_CONTROLE_MS, 10 s au-delà).
    """

    def test_ils_tiennent_une_generation_et_un_appel_avec_de_la_marge(self):
        """Mesuré : génération 15 à 20 s (essai réel du 04/11) ; un appel 125 s au pire (06/11), et 10 s de marge."""
        appel = DUREE_D_UN_APPEL_AU_PIRE_S + MARGE_DU_PLAN_S

        assert (routes_du_blog.DELAI_CONTROLE_S, routes_du_blog.DELAI_DE_LA_REDACTION_S) == (165, 190)
        assert appel + 30 <= routes_du_blog.DELAI_CONTROLE_S
        assert 20 + appel + 30 <= routes_du_blog.DELAI_DE_LA_REDACTION_S

    @pytest.mark.parametrize("ecoule, delai", [
        (0, 165), (15, 165), (25, 165), (30, 160), (45, 145), (55, 135), (56, 134), (160, 30), (189, 30),
    ])
    def test_la_part_du_controle_dans_generate(self, ecoule, delai):
        """
        Après 15 à 20 s de génération, le contrôle a tout son délai ; jusqu'à 60 s de génération,
        de quoi faire son appel et sa marge ; au-delà, son plan le refuse avant tout envoi. Le
        minimum de 30 s ne sert qu'aux modèles sans plan.
        """
        assert routes_du_blog.delai_du_controle_apres_la_generation(ecoule) == delai

    async def test_la_route_et_le_plan_ont_la_meme_echeance(
        self, admin_client, generateur, controleur, monkeypatch
    ):
        """L'échéance asyncio de la route et le délai du plan sont le même nombre, sur les deux routes."""
        delais = []
        juger = ControleDuBillet.controler

        async def controler(self, post, *, delai=None):
            delais.append(delai)
            return await juger(self, post, delai=delai)

        monkeypatch.setattr(ControleDuBillet, "controler", controler)
        monkeypatch.setattr(routes_du_blog, "delai_du_controle_apres_la_generation", lambda ecoule: 77.0)

        post = (await _genere(admin_client))["post"]
        assert (await _controle(admin_client, post["id"])).status_code == 200

        assert delais == [77.0, routes_du_blog.DELAI_CONTROLE_S]


# ==================== 11. LE PASSAGE CITÉ, PAR LA ROUTE (TOUR 5, D1) ====================


class TestLePassageCiteParLaRoute:
    """
    Revue du tour 4 : un dernier mot changé passait, un tronçon court après « … » n'était
    jamais cherché, et la raison montrait les mots du modèle sous « Extrait N : ».
    """

    async def test_seuls_les_mots_de_l_extrait_appuient_un_verdict(self, admin_client, generateur, controleur):
        post = (await _genere(admin_client))["post"]
        passages = {
            # Le dernier mot change : « contre son gré » devient « avec son accord ».
            "S'introduire": '"celui qui s\'introduit ou se maintient dans le domicile d\'autrui avec son '
                            'accord" [1]',
            # Un tronçon court inventé après « … » : l'art. 299 ne dit pas « six mois ».
            "Entrer chez": '"Est puni d\'un emprisonnement… six mois" [1]',
            # « (10) » et « (01) » sautés : une omission, citée dans les mots de l'extrait.
            "Sans bail écrit": '"Est puni d\'un emprisonnement de dix jours à un an" [1]',
        }

        def repondre(prompt):
            jugees = [
                jugement(identifiant, "soutenue", next(
                    (p for debut, p in passages.items() if phrase.startswith(debut)),
                    f'"{passage_de(extraits_du_prompt(prompt)[1])}" [1]',
                ), [1])
                for identifiant, phrase in phrases_du_prompt(prompt)
            ]
            return {"response": json.dumps({"phrases": jugees}), "fin": "STOP"}

        controleur.llm = modele_double(controle=repondre)

        controle = (await _controle(admin_client, post["id"])).json()["controle_ia"]

        par_debut = {p["texte"].split(" ")[0]: p for p in controle["phrases"] if p["partie"] in ("resume", "point_cle")}
        resume, entrer, sans = par_debut["S'introduire"], par_debut["Entrer"], par_debut["Sans"]
        for rabattue in (resume, entrer):
            assert rabattue["verdict"] == "non_soutenue"
            assert rabattue["raison"].startswith("À vérifier : le contrôle la disait soutenue")
            assert "Extrait 1 :" not in rabattue["raison"]
        assert "« celui qui s'introduit ou se maintient dans le domicile d'autrui avec son accord »" in resume["raison"]
        assert "(introuvable tel quel dans les extraits : « six mois »)" in entrer["raison"]
        assert sans["verdict"] == "soutenue"
        assert sans["raison"].endswith("— Extrait 1 : « Est puni d'un emprisonnement de dix (10) jours à un (01) an »")

    async def test_une_coupure_qui_saute_la_negation_n_appuie_rien(self, admin_client, generateur, controleur):
        """
        Revue de l'API du tour 5 (data/blog/tour5/revue-api, serveur_double.py, controle-route-1.json
        et -2.json) : par la vraie route, des mots vrais recollés autour de « … » en la règle inverse
        gardaient « soutenue », et la raison citait le collage comme texte de l'extrait.
        """
        post = (await _genere(admin_client))["post"]

        def repondre(prompt):
            extraits = extraits_du_prompt(prompt)
            rachat = next(n for n, texte in extraits.items() if "pacte de rachat" in texte)
            conge = next(n for n, texte in extraits.items() if "sans écrit" in texte)
            colles = {
                # « ne peut user de la faculté d'expulser » : la coupure saute « ne ».
                "S'introduire": (f'"L\'acquéreur à pacte de rachat… peut user de la faculté d\'expulser le '
                                 f'preneur" [{rachat}]', rachat),
                # « ne pourra donner congé à l'autre qu'en observant » : « ne » et « qu' ».
                "Sans bail écrit": (f'"l\'une des parties… pourra donner congé à l\'autre… en observant les '
                                    f'délais fixés par l\'usage des lieux" [{conge}]', conge),
            }
            jugees = []
            for identifiant, phrase in phrases_du_prompt(prompt):
                passage, n = next(
                    (p for debut, p in colles.items() if phrase.startswith(debut)),
                    (f'"{passage_de(extraits[1])}" [1]', 1),
                )
                jugees.append(jugement(identifiant, "soutenue", passage, [n], raison=f"L'extrait {n} le dit."))
            return {"response": json.dumps({"phrases": jugees}), "fin": "STOP"}

        controleur.llm = modele_double(controle=repondre)

        controle = (await _controle(admin_client, post["id"])).json()["controle_ia"]

        par_debut = {p["texte"].split(" ")[0]: p for p in controle["phrases"] if p["partie"] in ("resume", "point_cle")}
        for debut, saute in (("S'introduire", "« ne »"), ("Sans", "« ne », « qu' »")):
            rabattue = par_debut[debut]
            n = rabattue["extraits"][0]
            assert rabattue["verdict"] == "non_soutenue"
            assert rabattue["raison"].startswith("À vérifier : le contrôle la disait soutenue")
            assert f"(sa coupure « … » saute {saute} dans l'extrait {n}, entre les mots copiés « " in rabattue["raison"]
            assert f"Extrait {n} :" not in rabattue["raison"] and "[…]" not in rabattue["raison"]
        assert par_debut["Entrer"]["verdict"] == "soutenue"


# ==================== 12. LES PHRASES À CONFIRMER (AVENANT DU TOUR 7) ====================

# Deux phrases de l'article doublé : le contexte, et un point clé.
LOCATAIRE = "Un locataire trouve"
ENTRER = "Entrer chez quelqu'un"
DEUX_A_CONFIRMER = {LOCATAIRE: "non_soutenue", ENTRER: "hors_sources"}
_CE_QU_UN_REFUS_NE_TOUCHE_PAS = (
    "title", "slug", "category", "isPublished", "botAnalysis", "relecture", "controle_ia", "etat_publication",
)


def _natures(verdicts: dict, natures: dict):
    """Un contrôle doublé : `verdicts`, et la nature des phrases qui contiennent un morceau de `natures`."""
    def repondre(prompt: str) -> dict:
        jugees = json.loads(reponse_du_controle(prompt, verdicts)["response"])["phrases"]
        textes = dict(phrases_du_prompt(prompt))
        for jugee in jugees:
            for morceau, nature in natures.items():
                if morceau in textes[jugee["id"]]:
                    jugee["nature"] = nature
        return {"response": json.dumps({"phrases": jugees}), "fin": "STOP"}

    return repondre


async def _a_confirmer(admin_client, generateur, controle=None) -> dict:
    """Un brouillon généré dont deux phrases de règle sont à confirmer."""
    generateur.llm = modele_double(controle=controle or DEUX_A_CONFIRMER)
    post = (await _genere(admin_client))["post"]
    assert post["etat_publication"] == "phrases_a_confirmer", post["etat_publication"]
    return post


def _confirmation(post: dict, ids=None) -> dict:
    """Le corps `relecture` qui confirme `ids` (toutes les phrases à confirmer par défaut)."""
    return {
        "controle": post["controle_ia"]["fait_le"],
        "phrases_confirmees": a_confirmer(post) if ids is None else ids,
    }


async def _confirme_et_relu(admin_client, generateur, controle=None) -> dict:
    """Un brouillon aux phrases confirmées et aux parties relues : publiable."""
    post = await _a_confirmer(admin_client, generateur, controle)
    reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={"relecture": relecture_cochee(post)})
    assert reponse.status_code == 200, reponse.text
    assert reponse.json()["etat_publication"] == "publiable", reponse.json()
    return reponse.json()


async def _confirme_en_ligne(admin_client, generateur, controle=None) -> dict:
    post = await _confirme_et_relu(admin_client, generateur, controle)
    reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={"is_published": True})
    assert reponse.status_code == 200, reponse.text
    return reponse.json()


class TestLesPhrasesAConfirmer:
    """
    Décision du propriétaire (04/11) : « Confirmer phrase par phrase ». À l'essai réel, les 10
    phrases du bail qui disaient ou présupposaient l'expulsion sans juge étaient « À vérifier »,
    du même gris que les fausses alertes, et 7 cases cochées suffisaient à publier « peut être
    expulsé sans décision de justice ». Une phrase de RÈGLE que le contrôle n'a pas trouvée dans
    les extraits se réécrit ou se confirme une à une.
    """

    def test_l_ordre_des_etats_du_contrat(self):
        assert ETATS == (
            "publiable", "sources_absentes", "citations_non_verifiees", "controle_a_relancer",
            "phrases_contredites", "phrases_a_confirmer", "relecture_incomplete",
        )

    @pytest.mark.parametrize("verdict", ["non_soutenue", "hors_sources"])
    async def test_toutes_les_cases_ne_suffisent_plus(self, admin_client, client, generateur, verdict):
        post = await _a_confirmer(admin_client, generateur, {LOCATAIRE: verdict})
        cases = relecture_cochee(post, confirmer=False)

        coche = await admin_client.put(f"{BLOG}/{post['id']}", json={"relecture": cases})
        publie = await admin_client.put(f"{BLOG}/{post['id']}", json={"is_published": True})

        assert coche.status_code == 200, coche.text
        assert all(coche.json()["relecture"]["parties"].values())
        assert coche.json()["etat_publication"] == "phrases_a_confirmer"
        assert (publie.status_code, publie.json()["code"]) == (422, "phrases_a_confirmer")
        assert "une phrase de règle" in publie.json()["detail"] and LOCATAIRE in publie.json()["detail"]
        assert "J'ai vérifié cette phrase dans le texte" in publie.json()["detail"]
        assert (await _admin(admin_client, post["id"]))["isPublished"] is False
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 404

    async def test_elle_vient_apres_les_contredites_et_avant_la_relecture(self, admin_client, generateur):
        generateur.llm = modele_double(controle={LOCATAIRE: "non_soutenue", ENTRER: "contredite"})
        contredite = (await _genere(admin_client))["post"]
        confirmee = await admin_client.put(
            f"{BLOG}/{contredite['id']}", json={"relecture": _confirmation(contredite)},
        )

        assert contredite["etat_publication"] == "phrases_contredites"
        assert confirmee.json()["etat_publication"] == "phrases_contredites"

        post = await _a_confirmer(admin_client, generateur)
        sans_les_cases = await admin_client.put(f"{BLOG}/{post['id']}", json={"relecture": _confirmation(post)})
        assert sans_les_cases.json()["etat_publication"] == "relecture_incomplete"

    async def test_une_phrase_qui_n_enonce_pas_de_regle_ne_se_confirme_pas(self, admin_client, generateur):
        """Nature « autre » (un fait, un conseil) : elle se relit avec sa partie, sans plus."""
        generateur.llm = modele_double(controle=_natures(
            {LOCATAIRE: "non_soutenue", ENTRER: "hors_sources"}, {LOCATAIRE: "autre", ENTRER: "autre"},
        ))

        post = (await _genere(admin_client))["post"]
        relu = await admin_client.put(f"{BLOG}/{post['id']}", json={"relecture": relecture_cochee(post)})

        assert post["etat_publication"] == "relecture_incomplete" and a_confirmer(post) == []
        assert {p["nature"] for p in post["controle_ia"]["phrases"]} == {"regle", "autre"}
        assert relu.json()["etat_publication"] == "publiable"

    async def test_un_a_verifier_compte_comme_une_phrase_non_soutenue(self, admin_client, generateur):
        """Le modèle la dit soutenue, mais son passage n'est dans aucun extrait : rabattue, à confirmer."""
        def repondre(prompt: str) -> dict:
            jugees = json.loads(reponse_du_controle(prompt)["response"])["phrases"]
            for jugee in jugees:
                if LOCATAIRE in dict(phrases_du_prompt(prompt))[jugee["id"]]:
                    jugee["passage"] = "Un passage qu'aucun extrait ne porte, mot pour mot."
            return {"response": json.dumps({"phrases": jugees}), "fin": "STOP"}

        post = await _a_confirmer(admin_client, generateur, repondre)
        (phrase,) = [p for p in post["controle_ia"]["phrases"] if p["id"] in a_confirmer(post)]

        assert phrase["verdict"] == "non_soutenue" and phrase["raison"].startswith("À vérifier")
        assert LOCATAIRE in phrase["texte"]

    async def test_une_a_une_puis_publier(self, admin_client, client, generateur):
        post = await _a_confirmer(admin_client, generateur)
        premiere, seconde = a_confirmer(post)
        assert post["relecture"]["phrases_confirmees"] == []

        une = await admin_client.put(f"{BLOG}/{post['id']}", json={
            "relecture": {**relecture_cochee(post, confirmer=False), **_confirmation(post, [premiere])},
            "is_published": True,
        })
        sans_publier = await admin_client.put(f"{BLOG}/{post['id']}", json={
            "relecture": {**relecture_cochee(post, confirmer=False), **_confirmation(post, [premiere])},
        })

        # Refusée d'un bloc, puis appliquée sans publier.
        assert (une.status_code, une.json()["code"]) == (422, "phrases_a_confirmer")
        assert "ni réécrite ni confirmée" in une.json()["detail"] and seconde in une.json()["detail"]
        assert premiere not in une.json()["detail"]
        assert sans_publier.status_code == 200, sans_publier.text
        apres = sans_publier.json()
        assert apres["relecture"]["phrases_confirmees"] == [premiere]
        assert apres["relecture"]["controle"] == post["controle_ia"]["fait_le"]
        assert apres["relecture"]["par"] and apres["relecture"]["le"]
        assert apres["etat_publication"] == "phrases_a_confirmer"

        fini = await admin_client.put(f"{BLOG}/{post['id']}", json={
            "relecture": _confirmation(post, [seconde, premiere]), "is_published": True,
        })

        assert fini.status_code == 200, fini.text
        assert fini.json()["relecture"]["phrases_confirmees"] == [premiere, seconde]  # l'ordre de l'article
        assert fini.json()["etat_publication"] == "publiable" and fini.json()["isPublished"] is True
        for lu in (
            await _admin(admin_client, post["id"]),
            (await admin_client.get(f"{BLOG}/admin/all")).json()[0],
        ):
            assert lu["relecture"]["phrases_confirmees"] == [premiere, seconde]
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 200

    async def test_les_deux_noms_du_champ(self, admin_client, generateur):
        post = await _a_confirmer(admin_client, generateur)

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={"relecture": {
            "controle": post["controle_ia"]["fait_le"], "phrasesConfirmees": a_confirmer(post)[:1],
        }})

        assert reponse.status_code == 200, reponse.text
        assert reponse.json()["relecture"]["phrases_confirmees"] == a_confirmer(post)[:1]

    async def test_une_phrase_connue_qui_n_est_pas_a_confirmer_n_est_pas_gardee(self, admin_client, generateur):
        post = await _a_confirmer(admin_client, generateur)
        soutenue = next(p["id"] for p in post["controle_ia"]["phrases"] if p["verdict"] == "soutenue")

        reponse = await admin_client.put(
            f"{BLOG}/{post['id']}", json={"relecture": _confirmation(post, [soutenue, a_confirmer(post)[0]])},
        )

        assert reponse.status_code == 200, reponse.text
        assert reponse.json()["relecture"]["phrases_confirmees"] == a_confirmer(post)[:1]

    @pytest.mark.parametrize("cas", ["relecture", "publier", "modifier", "retirer-et-modifier", "formulaire"])
    @pytest.mark.parametrize("en_ligne", [False, True], ids=["brouillon", "en-ligne"])
    @pytest.mark.parametrize("refus", ["id-inconnu", "sans-controle", "controle-vide", "un-autre-controle"])
    async def test_chaque_chemin_du_put_refuse_et_n_applique_rien(
        self, admin_client, client, generateur, cas, en_ligne, refus
    ):
        """
        AVENANT DU TOUR 7 : la liste part TOUJOURS avec le contrôle relu, et chaque identifiant est
        une phrase du contrôle ENREGISTRÉ ; sinon 422 relecture_perimee, et RIEN n'est appliqué.
        """
        post = await (_confirme_en_ligne if en_ligne else _confirme_et_relu)(admin_client, generateur)
        fait_le = post["controle_ia"]["fait_le"]
        relecture = {
            "id-inconnu": {"controle": fait_le, "phrases_confirmees": [a_confirmer(post)[0], "P999"]},
            "sans-controle": {"phrases_confirmees": a_confirmer(post)},
            "controle-vide": {"controle": "", "phrases_confirmees": []},
            "un-autre-controle": {"controle": "2026-01-01T00:00:00+00:00", "phrases_confirmees": a_confirmer(post)},
        }[refus]
        corps = {
            "relecture": {"relecture": relecture},
            "publier": {"relecture": relecture, "is_published": True},
            "modifier": {"relecture": relecture, "category": "Droit pénal", "bot_analysis": {"summary": "Corrigé."}},
            "retirer-et-modifier": {
                "relecture": relecture, "is_published": False, "title": "Retitré", "slug": "une-autre-adresse",
            },
            "formulaire": {**_corps_du_front(post, summary="Corrigé."), "relecture": relecture},
        }[cas]

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json=corps, headers=ORIGINE)

        assert (reponse.status_code, reponse.json()["code"]) == (422, "relecture_perimee"), reponse.text
        assert "Rien n'a été enregistré" in reponse.json()["detail"]
        assert reponse.headers["access-control-allow-origin"] == ORIGINE["Origin"]
        relu = await _admin(admin_client, post["id"])
        for champ in _CE_QU_UN_REFUS_NE_TOUCHE_PAS:
            assert relu[champ] == post[champ], champ
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == (200 if en_ligne else 404)

    @pytest.mark.parametrize("ids", [[], ["P1"]], ids=["vide", "un-id"])
    async def test_sans_controle_enregistre_rien_a_confirmer(self, admin_client, generateur, ids):
        generateur.llm = modele_double(controle=MistralOverloadedError("503"))
        post = (await _genere(admin_client))["post"]

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={"relecture": {
            "controle": "2026-11-05T10:00:00+00:00", "phrases_confirmees": ids,
        }})

        assert (reponse.status_code, reponse.json()["code"]) == (422, "relecture_perimee")
        assert "aucun contrôle IA n'est enregistré" in reponse.json()["detail"]

    async def test_deux_onglets(self, admin_client, generateur, controleur):
        """Relancé dans un autre onglet : l'onglet d'avant confirme une phrase d'un contrôle remplacé."""
        vu_par_a = await _a_confirmer(admin_client, generateur)
        controleur.llm = modele_double(controle=DEUX_A_CONFIRMER)
        relance_par_b = (await _controle(admin_client, vu_par_a["id"])).json()

        refus = await admin_client.put(f"{BLOG}/{vu_par_a['id']}", json={"relecture": _confirmation(vu_par_a)})
        accepte = await admin_client.put(
            f"{BLOG}/{vu_par_a['id']}", json={"relecture": _confirmation(relance_par_b)},
        )

        assert (refus.status_code, refus.json()["code"]) == (422, "relecture_perimee")
        assert f"fait le {relance_par_b['controle_ia']['fait_le']}" in refus.json()["detail"]
        assert accepte.status_code == 200, accepte.text
        assert accepte.json()["relecture"]["controle"] == relance_par_b["controle_ia"]["fait_le"]
        assert accepte.json()["relecture"]["phrases_confirmees"] == a_confirmer(relance_par_b)

    @pytest.mark.parametrize("valeur, raison", [
        ("P4", "relecture › phrases confirmées : liste attendue"),
        ([4], "relecture › phrases confirmées › n° 1 : texte attendu"),
        ([f"P{i}" for i in range(101)], "relecture › phrases confirmées : 100 éléments au maximum"),
    ], ids=["pas-une-liste", "un-nombre", "trop"])
    async def test_une_liste_illisible_422(self, admin_client, generateur, valeur, raison):
        post = await _a_confirmer(admin_client, generateur)

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={"relecture": {
            "controle": post["controle_ia"]["fait_le"], "phrases_confirmees": valeur,
        }})

        assert (reponse.status_code, reponse.json()["code"]) == (422, "entree_invalide")
        assert raison in reponse.json()["detail"]

    async def test_la_liste_est_entiere(self, admin_client, client, generateur):
        """Une phrase absente de la liste n'est plus confirmée ; en ligne, c'est refusé d'un bloc."""
        brouillon = await _confirme_et_relu(admin_client, generateur)
        en_ligne = await _confirme_en_ligne(admin_client, generateur)
        reste = a_confirmer(brouillon)[:1]

        retiree = await admin_client.put(f"{BLOG}/{brouillon['id']}", json={"relecture": _confirmation(brouillon, reste)})
        rien_d_envoye = await admin_client.put(f"{BLOG}/{brouillon['id']}", json={"tags": ["bail"]})
        refuse = await admin_client.put(f"{BLOG}/{en_ligne['id']}", json={"relecture": _confirmation(en_ligne, reste)})

        assert retiree.json()["relecture"]["phrases_confirmees"] == reste
        assert retiree.json()["etat_publication"] == "phrases_a_confirmer"
        # Une modification qui n'en dit rien n'y touche pas.
        assert rien_d_envoye.json()["relecture"]["phrases_confirmees"] == reste
        assert (refuse.status_code, refuse.json()["code"]) == (422, "phrases_a_confirmer")
        relu = await _admin(admin_client, en_ligne["id"])
        assert relu["relecture"] == en_ligne["relecture"] and relu["isPublished"] is True
        assert (await client.get(f"{BLOG}/{en_ligne['slug']}")).status_code == 200

    async def test_les_memes_verdicts_gardent_les_confirmations(self, admin_client, client, generateur, controleur):
        post = await _confirme_en_ligne(admin_client, generateur)
        controleur.llm = modele_double(controle=DEUX_A_CONFIRMER)

        relance = (await _controle(admin_client, post["id"], json={"confirmer": True})).json()

        assert relance["controle_ia"]["fait_le"] != post["controle_ia"]["fait_le"]
        assert relance["relecture"]["phrases_confirmees"] == a_confirmer(post)
        assert relance["etat_publication"] == "publiable"
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 200

    @pytest.mark.parametrize("verdict_de_y", ["non_soutenue", "contredite", "soutenue"])
    async def test_un_nouveau_controle_la_fait_tomber_comme_une_case(
        self, admin_client, client, generateur, controleur, verdict_de_y
    ):
        """
        Un autre verdict : la confirmation de CETTE phrase tombe, l'autre reste. Effacée, elle ne
        revit pas au contrôle suivant qui redonne le verdict d'avant (revue du front du tour 5,
        pour les cases) : l'article caché n'y revient pas sans qu'une personne ait confirmé.
        """
        post = await _confirme_en_ligne(admin_client, generateur)
        phrases = {p["texte"]: p["id"] for p in post["controle_ia"]["phrases"]}
        locataire = next(i for t, i in phrases.items() if LOCATAIRE in t)
        entrer = next(i for t, i in phrases.items() if ENTRER in t)

        controleur.llm = modele_double(controle={LOCATAIRE: "non_soutenue", ENTRER: verdict_de_y})
        y = (await _controle(admin_client, post["id"], json={"confirmer": True})).json()

        # hors_sources → autre chose : l'attestation portait sur ce verdict-là.
        assert y["relecture"]["phrases_confirmees"] == [locataire]
        assert y["etat_publication"] != "publiable"
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 404

        controleur.llm = modele_double(controle=DEUX_A_CONFIRMER)
        z = (await _controle(admin_client, post["id"], json={"confirmer": True})).json()

        assert z["relecture"]["phrases_confirmees"] == [locataire]
        assert a_confirmer(z) == [locataire, entrer]
        assert z["etat_publication"] == "phrases_a_confirmer"
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 404
        # Confirmée par une personne, contre le contrôle enregistré : servi de nouveau.
        confirme = await admin_client.put(
            f"{BLOG}/{post['id']}", json={"relecture": relecture_cochee(z, {"points_cles": True})},
        )
        assert confirme.json()["etat_publication"] == "publiable", confirme.json()
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 200

    async def test_une_phrase_reecrite_perd_sa_confirmation_et_un_nouveau_controle_la_juge(
        self, admin_client, generateur, controleur
    ):
        post = await _confirme_et_relu(admin_client, generateur)
        locataire, entrer = a_confirmer(post)

        reecrite = await admin_client.put(f"{BLOG}/{post['id']}", json={
            "context": "Un voisin voit des affaires sur le trottoir.", "relecture": _confirmation(post),
        })

        assert reecrite.status_code == 200, reecrite.text
        # Même envoyée, sa confirmation ne compte pas : elle portait sur l'ancienne phrase.
        assert reecrite.json()["relecture"]["phrases_confirmees"] == [entrer]
        assert reecrite.json()["etat_publication"] == "controle_a_relancer"

        controleur.llm = modele_double(controle={ENTRER: "hors_sources"})
        juge = (await _controle(admin_client, post["id"])).json()

        assert a_confirmer(juge) == [entrer]
        assert juge["relecture"]["phrases_confirmees"] == [entrer]
        assert juge["relecture"]["parties"] == {p: p != "contexte" for p in PARTIES}
        assert juge["etat_publication"] == "relecture_incomplete"

    async def test_des_sources_reconstruites_autrement_la_font_tomber(
        self, admin_client, generateur, controleur, textes
    ):
        post = await _confirme_et_relu(admin_client, generateur)

        memes = (await admin_client.post(f"{BLOG}/{post['id']}/sources")).json()["post"]
        generateur.recherche = RechercheDoublee(
            [extrait(textes.code_penal, textes.cp_299, "299", CP_299)], {textes.cp_299: 0.6},
        )
        autres = (await admin_client.post(f"{BLOG}/{post['id']}/sources")).json()["post"]
        controleur.llm = modele_double(controle=DEUX_A_CONFIRMER)
        refait = (await _controle(admin_client, post["id"])).json()

        assert memes["relecture"]["phrases_confirmees"] == a_confirmer(post)
        assert memes["etat_publication"] == "publiable"
        assert autres["controle_ia"] is None and autres["relecture"]["phrases_confirmees"] == []
        assert refait["relecture"]["phrases_confirmees"] == []
        assert refait["etat_publication"] == "phrases_a_confirmer"

    async def test_retirer_du_site_les_garde(self, admin_client, client, generateur):
        post = await _confirme_en_ligne(admin_client, generateur)

        retire = await admin_client.put(f"{BLOG}/{post['id']}", json={"is_published": False})
        remis = await admin_client.put(f"{BLOG}/{post['id']}", json={"is_published": True})

        assert retire.json()["relecture"] == post["relecture"]
        assert retire.json()["etat_publication"] == "publiable"
        assert remis.status_code == 200, remis.text
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 200

    async def test_un_article_en_ligne_non_confirme_n_est_pas_servi(
        self, admin_client, client, generateur, db_session
    ):
        """En ligne en base avec une phrase à confirmer (publié avant l'avenant) : caché du public."""
        post = await _a_confirmer(admin_client, generateur)
        await admin_client.put(f"{BLOG}/{post['id']}", json={"relecture": relecture_cochee(post, confirmer=False)})
        await db_session.execute(
            text("UPDATE blog_posts SET is_published = true WHERE id = :id"), {"id": post["id"]},
        )
        await db_session.commit()
        _oublier_les_articles(db_session)

        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 404
        assert (await client.get(BLOG)).json() == []
