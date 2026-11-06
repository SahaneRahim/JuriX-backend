"""
Les routes du blog : /api/v1/blog.

Ce que ces tests protègent, dans l'ordre d'importance :

  1. UN BROUILLON N'EST JAMAIS PUBLIC. GET /blog/{slug} servait tout article
     par son adresse (essai réel du 29/10/2026 : brouillon lu en 200), et un
     article dépublié restait lisible par les liens déjà partagés.
  2. UN ARTICLE GÉNÉRÉ NAÎT BROUILLON, avec des citations construites par le
     serveur ; une génération sans source, coupée ou illisible n'enregistre
     RIEN.
  3. UNE CITATION N'EST GARDÉE OU AJOUTÉE QUE SI LA BASE LA PORTE (texte publié,
     article existant), et toute publication revérifie les citations.
  4. LES ERREURS SONT LISIBLES ET PASSENT PAR CORS : `{"detail": "<texte>",
     "code": "<code>"}`. Une 500 hors CORS s'affichait « Failed to fetch »,
     un `detail` en tableau « [object Object] ».
  5. Seul un administrateur écrit, ou voit un brouillon.

Et, depuis le tour 2 :

  6. LE PUBLIC NE LIT JAMAIS UNE CITATION NON VÉRIFIÉE COMME UNE VRAIE. Toute
     citation porte `verifie` (booléen) ; dans les réponses publiques, une
     citation non vérifiée — d'avant le contrat, ou dont le texte a été
     dépublié ou l'article modifié depuis — ne garde que sa référence et son
     numéro, sans extrait ni lien.
  7. RETIRER DU SITE MARCHE TOUJOURS : exactement {"is_published": false}
     retire l'article, quoi qu'il porte (tour 3 ; au tour 2, tout corps qui
     dépubliait dépubliait d'abord).

Et, depuis le tour 3 (le contrôle IA et la relecture ont leurs propres tests,
test_blog_relecture_routes.py) : publier exige un article PUBLIABLE, et le
public ne voit que les articles publiés et publiables ; un article d'avant
n'est jamais servi.

Le modèle et la recherche sont doublés (tests/blog_donnees.py).

Usage:
    pytest tests/test_api/test_blog_routes.py -v
"""

import asyncio
import json
from typing import List
from unittest.mock import AsyncMock

import pytest
from httpx import AsyncClient
from sqlalchemy import func, select, text

from app.api.routes import blog as routes_du_blog
from app.api.routes.blog import get_blog_generator, get_controleur
from app.main import app
from app.models.blog import BlogPost
from app.schemas.search import ChunkResult
from app.services import blog_generator_service as service_du_blog
from app.services.blog_generator_service import BlogGeneratorService
from app.services.controle_du_billet import PARTIES, ControleDuBillet
from app.services.mistral_service import MistralOverloadedError, MistralQuotaError
from tests.blog_donnees import (
    CITATIONS_D_AVANT,
    CT_35_2,
    SUJET,
    TITRE_CODE_CIVIL,
    TITRE_CODE_PENAL,
    RechercheDoublee,
    article_d_avant,
    article_du_modele,
    creer_les_textes,
    extrait,
    extraits_du_sujet,
    modele_double,
    relecture_cochee,
    similarites,
)

BLOG = "/api/v1/blog"
ORIGINE = {"Origin": "http://localhost:5173"}


@pytest.fixture
async def textes(db_session):
    return await creer_les_textes(db_session)


@pytest.fixture
def generateur(db_session, textes):
    """
    Pose la doublure du générateur ; rend un objet dont on règle le modèle et
    la recherche avant l'appel.
    """
    class Reglage:
        llm = modele_double()
        recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))

    Reglage.textes = textes

    app.dependency_overrides[get_blog_generator] = lambda: BlogGeneratorService(
        db_session, llm=Reglage.llm, recherche=Reglage.recherche
    )
    try:
        yield Reglage
    finally:
        app.dependency_overrides.pop(get_blog_generator, None)


@pytest.fixture(autouse=True)
def controleur(db_session):
    """Le contrôle IA de POST /{id}/controle, doublé : tout « soutenue » sauf réglage."""
    class Reglage:
        llm = modele_double()

    app.dependency_overrides[get_controleur] = lambda: ControleDuBillet(db_session, llm=Reglage.llm)
    try:
        yield Reglage
    finally:
        app.dependency_overrides.pop(get_controleur, None)


class _Reconstruction:
    """
    La recherche de POST /{id}/sources, doublée : les lignes que l'article cite
    (posées par _relu_et_publie), jugées très proches du sujet. Un test qui
    demande `generateur` la remplace par la sienne.
    """

    extraits: List[ChunkResult] = []


@pytest.fixture(autouse=True)
def reconstruction(db_session):
    _Reconstruction.extraits = []
    app.dependency_overrides[get_blog_generator] = lambda: BlogGeneratorService(
        db_session, llm=modele_double(), recherche=RechercheDoublee(
            list(_Reconstruction.extraits), {e.article_id: 0.6 for e in _Reconstruction.extraits},
        ),
    )
    try:
        yield _Reconstruction
    finally:
        app.dependency_overrides.pop(get_blog_generator, None)


TOUT_RELU = {"parties": {partie: True for partie in PARTIES}}


async def _controle(admin_client, post_id) -> dict:
    reponse = await admin_client.post(f"{BLOG}/{post_id}/controle")
    assert reponse.status_code == 200, reponse.text
    return reponse.json()


async def _sources(admin_client, post_id) -> dict:
    """Reconstruit les sources d'un article écrit à la main, à partir des lignes qu'il cite."""
    post = (await admin_client.get(f"{BLOG}/admin/{post_id}")).json()
    _Reconstruction.extraits = [
        ChunkResult(
            article_id=c["article_id"], law_id=c["law_id"], number=c["article_number"],
            content=c["excerpt"], reference=c.get("law_reference") or "", law_title=c["reference"],
            status="published", language="fr",
        )
        for c in post["botAnalysis"]["citations"] if c["verifie"]
    ]
    reponse = await admin_client.post(f"{BLOG}/{post_id}/sources")
    assert reponse.status_code == 200, reponse.text
    return reponse.json()


async def _relu_et_publie(admin_client, post_id) -> dict:
    """Le parcours des tours 3 et 4 : sources, contrôle, relecture de chaque partie, publication."""
    if not (await admin_client.get(f"{BLOG}/admin/{post_id}")).json()["sources"]:
        await _sources(admin_client, post_id)
    controle = await _controle(admin_client, post_id)
    reponse = await admin_client.put(
        f"{BLOG}/{post_id}", json={"relecture": relecture_cochee(controle), "is_published": True}
    )
    assert reponse.status_code == 200, reponse.text
    return reponse.json()


def _oublier_les_articles(db_session) -> None:
    """
    L'application partage la session du test (conftest.client) : un article
    modifié en SQL brut y resterait périmé. Les articles seulement :
    l'administrateur doublé (as_admin) vit dans la même session.
    """
    for objet in list(db_session.identity_map.values()):
        if isinstance(objet, BlogPost):
            db_session.expire(objet)


async def _nombre_d_articles(db_session) -> int:
    return (await db_session.execute(select(func.count()).select_from(BlogPost))).scalar_one()


def _citation(law_id, article_number=None, article_id=None, title="Pourquoi cet article."):
    citation = {"law_id": law_id, "title": title}
    if article_number is not None:
        citation["article_number"] = article_number
    if article_id is not None:
        citation["article_id"] = article_id
    return citation


def _corps(textes, **champs):
    """Un article écrit à la main, au format de l'administration."""
    corps = {
        "title": "Le congé d'un bail verbal",
        "category": "Immobilier",
        "context": "Un bailleur veut reprendre son logement loué sans contrat écrit.",
        "bot_analysis": {
            "summary": "Le congé suit les délais fixés par l'usage des lieux.",
            "keyPoints": ["Sans écrit, l'usage des lieux fixe le délai."],
            "citations": [_citation(textes.code_civil, "1736")],
            "citizenImpact": "Renseignez-vous sur l'usage local avant de donner congé.",
        },
        "suggested_question": "Quel délai pour quitter un logement loué sans contrat ?",
        "tags": ["bail", "congé"],
    }
    corps.update(champs)
    return corps


async def _brouillon(admin_client, textes, **champs) -> dict:
    reponse = await admin_client.post(BLOG, json=_corps(textes, **champs))
    assert reponse.status_code == 201, reponse.text
    return reponse.json()


async def _publie(admin_client, textes, **champs) -> dict:
    post = await _brouillon(admin_client, textes, **champs)
    return await _relu_et_publie(admin_client, post["id"])


# ==================== AUTHENTIFICATION ====================

_ROUTES_ADMIN = [
    ("post", f"{BLOG}/generate", {"topic": SUJET}),
    ("get", f"{BLOG}/admin/all", None),
    ("get", f"{BLOG}/admin/1", None),
    ("post", BLOG, {"title": "x"}),
    ("put", f"{BLOG}/1", {"title": "x"}),
    ("delete", f"{BLOG}/1", None),
    ("post", f"{BLOG}/1/controle", None),
    ("post", f"{BLOG}/1/sources", None),
]


class TestAuthentification:

    @pytest.mark.parametrize("methode, chemin, corps", _ROUTES_ADMIN)
    async def test_sans_jeton_401(self, client: AsyncClient, methode, chemin, corps):
        options = {"json": corps} if corps is not None else {}
        reponse = await client.request(methode.upper(), chemin, **options)

        assert reponse.status_code == 401

    @pytest.mark.parametrize("methode, chemin, corps", _ROUTES_ADMIN)
    async def test_un_compte_ordinaire_403(
        self, client: AsyncClient, auth_headers, methode, chemin, corps
    ):
        options = {"json": corps} if corps is not None else {}
        reponse = await client.request(methode.upper(), chemin, headers=auth_headers, **options)

        assert reponse.status_code == 403

    async def test_un_compte_admin_par_son_jeton(self, client: AsyncClient, admin_headers):
        reponse = await client.get(f"{BLOG}/admin/all", headers=admin_headers)

        assert reponse.status_code == 200
        assert reponse.json() == []


# ==================== LE BROUILLON N'EST JAMAIS PUBLIC ====================


class TestBrouillons:

    async def test_un_brouillon_n_est_ni_liste_ni_servi(self, admin_client, textes):
        post = await _brouillon(admin_client, textes)

        assert post["isPublished"] is False
        assert (await admin_client.get(BLOG)).json() == []
        reponse = await admin_client.get(f"{BLOG}/{post['slug']}")
        assert reponse.status_code == 404
        assert reponse.json() == {"detail": "Article introuvable.", "code": "introuvable"}

    async def test_l_administration_voit_le_brouillon(self, admin_client, textes):
        post = await _brouillon(admin_client, textes)

        liste = (await admin_client.get(f"{BLOG}/admin/all")).json()
        un = (await admin_client.get(f"{BLOG}/admin/{post['id']}")).json()

        assert [p["id"] for p in liste] == [post["id"]]
        assert un["id"] == post["id"] and un["isPublished"] is False
        assert "avertissements" in un
        # Écrit à la main : ni sources ni contrôle (tour 4).
        assert un["sources"] == [] and un["etat_publication"] == "sources_absentes"

    async def test_publier_puis_depublier(self, admin_client, textes):
        post = await _publie(admin_client, textes)
        assert post["isPublished"] is True
        assert (await admin_client.get(f"{BLOG}/{post['slug']}")).status_code == 200

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={"is_published": False})

        assert reponse.status_code == 200
        assert (await admin_client.get(f"{BLOG}/{post['slug']}")).status_code == 404
        assert (await admin_client.get(BLOG)).json() == []

    async def test_la_publication_date_l_article_du_jour(self, admin_client, textes, db_session):
        post = await _brouillon(admin_client, textes)
        await db_session.execute(
            text("UPDATE blog_posts SET published_at = DATE '2026-01-02' WHERE id = :id"),
            {"id": post["id"]},
        )
        await db_session.commit()
        _oublier_les_articles(db_session)

        publie = await _relu_et_publie(admin_client, post["id"])

        jour = (await db_session.execute(text("SELECT CURRENT_DATE"))).scalar_one()
        assert publie["publishedAt"] == jour.isoformat()

    async def test_un_identifiant_inconnu_404(self, admin_client, textes):
        for methode, chemin, options in (
            ("GET", f"{BLOG}/admin/999999", {}),
            ("PUT", f"{BLOG}/999999", {"json": {"title": "x"}}),
            ("PUT", f"{BLOG}/999999", {"json": {"is_published": False}}),
            ("DELETE", f"{BLOG}/999999", {}),
            ("POST", f"{BLOG}/999999/controle", {}),
            ("POST", f"{BLOG}/999999/sources", {}),
        ):
            reponse = await admin_client.request(methode, chemin, **options)
            assert reponse.status_code == 404, (methode, chemin)
            assert reponse.json()["code"] == "introuvable"


# ==================== LISTE PUBLIQUE ====================


class TestListePublique:

    async def test_publies_seulement_du_plus_recent_au_plus_ancien(
        self, admin_client, textes, db_session
    ):
        ancien = await _publie(admin_client, textes, title="Ancien")
        recent = await _publie(admin_client, textes, title="Récent")
        await _brouillon(admin_client, textes, title="Brouillon")
        await db_session.execute(
            text("UPDATE blog_posts SET published_at = DATE '2026-01-01' WHERE id = :id"),
            {"id": ancien["id"]},
        )
        await db_session.commit()
        _oublier_les_articles(db_session)

        liste = (await admin_client.get(BLOG)).json()

        assert [p["title"] for p in liste] == ["Récent", "Ancien"]
        assert "avertissements" not in liste[0]
        assert liste[0]["id"] == recent["id"]

    async def test_meme_jour_le_dernier_publie_d_abord(self, admin_client, textes):
        await _publie(admin_client, textes, title="Premier")
        await _publie(admin_client, textes, title="Second")

        assert [p["title"] for p in (await admin_client.get(BLOG)).json()] == ["Second", "Premier"]

    async def test_filtre_de_rubrique_par_egalite(self, admin_client, textes):
        await _publie(admin_client, textes, title="Bail", category="Immobilier")
        await _publie(admin_client, textes, title="Travail", category="Travail & Emploi")

        def titres(reponse):
            return [p["title"] for p in reponse.json()]

        assert titres(await admin_client.get(BLOG, params={"category": "immobilier"})) == ["Bail"]
        assert titres(await admin_client.get(BLOG, params={"category": "Travail"})) == []
        assert titres(await admin_client.get(BLOG, params={"category": "%"})) == []
        assert len((await admin_client.get(BLOG, params={"category": "Tous"})).json()) == 2

    async def test_la_forme_que_lit_le_front(self, admin_client, textes):
        post = await _publie(admin_client, textes)

        public = (await admin_client.get(f"{BLOG}/{post['slug']}")).json()

        assert set(public) == {
            "id", "slug", "title", "category", "publishedAt", "readingTime", "context",
            "botAnalysis", "suggestedQuestion", "tags", "isPublished",
        }
        assert set(public["botAnalysis"]) == {"summary", "keyPoints", "citations", "citizenImpact"}
        (citation,) = public["botAnalysis"]["citations"]
        assert citation["verifie"] is True
        assert citation["law_id"] == textes.code_civil
        assert citation["article_number"] == "1736"
        assert citation["reference"] == TITRE_CODE_CIVIL
        # La cote interne ne sert à rien au lecteur (revue du tour 2).
        assert "law_reference" not in citation


# ==================== GÉNÉRATION ====================


class TestGeneration:

    async def test_201_un_brouillon_enregistre(self, admin_client, generateur, textes, db_session):
        reponse = await admin_client.post(f"{BLOG}/generate", json={"topic": SUJET})

        assert reponse.status_code == 201, reponse.text
        corps = reponse.json()
        assert set(corps) == {"post", "sources", "avertissements"}
        post = corps["post"]
        assert post["isPublished"] is False
        assert post["slug"] == "expulsion-sans-juge-ce-que-dit-la-loi"
        assert post["category"] == "Immobilier"
        assert post["readingTime"] == "1 min"
        assert "avertissements" in post
        # Les citations : celles du serveur, relues en base.
        premiere = post["botAnalysis"]["citations"][0]
        assert premiere["reference"] == TITRE_CODE_PENAL
        assert premiere["article_number"] == "299"
        assert premiere["verifie"] is True
        assert premiere["url"] == f"/laws/{textes.code_penal}?article=299"
        # Les sources : ce que le modèle a lu, un article par extrait. Le
        # morceau suivant du Code civil annoté (des arrêts) n'en est pas un.
        assert [s["n"] for s in corps["sources"]] == [1, 2, 3]
        assert set(corps["sources"][0]) == {
            "n", "law_id", "law_title", "article_number", "article_id", "suite", "extrait"
        }
        # Contrôlé aussitôt, rien de relu : pas publiable.
        assert post["controle_ia"] is not None
        assert post["etat_publication"] == "relecture_incomplete"
        # `controle` (avenant du tour 5) : aucune case cochée, contre aucun contrôle.
        assert post["relecture"] == {
            "parties": {partie: False for partie in PARTIES}, "par": None, "le": None, "controle": None,
            "phrases_confirmees": [],
        }
        # Enregistré, et invisible du public.
        assert await _nombre_d_articles(db_session) == 1
        assert (await admin_client.get(f"{BLOG}/{post['slug']}")).status_code == 404
        assert (await admin_client.get(BLOG)).json() == []

    async def test_les_avertissements_de_la_generation(self, admin_client, generateur):
        generateur.llm = modele_double(article_du_modele(
            summary="L'intrusion est punie par l'article 312 du Code Pénal.",
            citations=[{"extrait": 1, "objet": "Punie."}, {"extrait": 12, "objet": "Inventé."}],
        ))

        corps = (await admin_client.post(f"{BLOG}/generate", json={"topic": SUJET})).json()

        avertissements = corps["avertissements"]
        assert any("article 312 du Code Pénal" in a and "extraits consultés" in a for a in avertissements)
        assert any("l'extrait 12" in a for a in avertissements)
        # La même mention n'est pas signalée deux fois (génération, puis article).
        assert sum("article 312" in a for a in avertissements) == 1
        assert any("article 312" in a for a in corps["post"]["avertissements"])

    async def test_le_titre_donne_une_adresse_propre_et_libre(self, admin_client, generateur):
        generateur.llm = modele_double(article_du_modele(title="Bail/Expulsion Été ? #droits"))
        premier = (await admin_client.post(f"{BLOG}/generate", json={"topic": SUJET})).json()

        generateur.llm = modele_double(article_du_modele(title="Bail/Expulsion Été ? #droits"))
        second = (await admin_client.post(f"{BLOG}/generate", json={"topic": SUJET})).json()

        assert premier["post"]["slug"] == "bail-expulsion-ete-droits"
        assert second["post"]["slug"] == "bail-expulsion-ete-droits-2"

    async def test_sans_source_422_le_modele_n_est_pas_appele(
        self, admin_client, generateur, db_session
    ):
        generateur.recherche = RechercheDoublee([])

        reponse = await admin_client.post(f"{BLOG}/generate", json={"topic": SUJET}, headers=ORIGINE)

        assert reponse.status_code == 422
        assert reponse.json()["code"] == "sans_source"
        assert isinstance(reponse.json()["detail"], str)
        assert reponse.headers["access-control-allow-origin"] == ORIGINE["Origin"]
        generateur.llm.generate.assert_not_awaited()
        assert await _nombre_d_articles(db_session) == 0

    @pytest.mark.parametrize("reponse_du_modele", [
        {"response": json.dumps(article_du_modele()), "fin": "MAX_TOKENS", "tronquee": True},
        {"response": '{"title": "Expulsion", "summary": "Selon', "fin": "STOP"},
        {"response": json.dumps(["pas", "un", "objet"]), "fin": "STOP"},
    ], ids=["coupee", "json-illisible", "pas-un-objet"])
    async def test_502_et_rien_d_enregistre(
        self, admin_client, generateur, db_session, reponse_du_modele
    ):
        generateur.llm = AsyncMock()
        generateur.llm.generate = AsyncMock(return_value=reponse_du_modele)

        reponse = await admin_client.post(f"{BLOG}/generate", json={"topic": SUJET})

        assert reponse.status_code == 502
        assert reponse.json()["code"] == "reponse_invalide"
        assert await _nombre_d_articles(db_session) == 0

    @pytest.mark.parametrize("erreur, statut, code, delai", [
        (MistralQuotaError("Quota mensuel épuisé"), 429, "quota", "60"),
        (MistralOverloadedError("503"), 503, "saturation", "10"),
    ])
    async def test_quota_et_saturation(
        self, admin_client, generateur, erreur, statut, code, delai
    ):
        generateur.llm = modele_double(side_effect=erreur)

        reponse = await admin_client.post(f"{BLOG}/generate", json={"topic": SUJET}, headers=ORIGINE)

        assert reponse.status_code == statut
        assert reponse.json()["code"] == code
        assert reponse.headers["retry-after"] == delai
        assert reponse.headers["access-control-allow-origin"] == ORIGINE["Origin"]

    async def test_recherche_indisponible_503(self, admin_client, generateur):
        generateur.recherche = RechercheDoublee([], erreur=RuntimeError("panne"))

        reponse = await admin_client.post(f"{BLOG}/generate", json={"topic": SUJET})

        assert reponse.status_code == 503
        assert reponse.json()["code"] == "recherche_indisponible"
        assert reponse.headers["retry-after"] == "10"

    async def test_au_dela_du_delai_504(self, admin_client, generateur, db_session, monkeypatch):
        async def lent(**_):
            await asyncio.sleep(5)

        generateur.llm = modele_double(side_effect=lent)
        monkeypatch.setattr(routes_du_blog, "DELAI_GENERATION_S", 0.2)

        reponse = await admin_client.post(f"{BLOG}/generate", json={"topic": SUJET})

        assert reponse.status_code == 504
        assert reponse.json()["code"] == "delai"
        assert await _nombre_d_articles(db_session) == 0

    async def test_une_erreur_inattendue_500_lisible_et_avec_cors(self, admin_client, generateur):
        """Le message interne ne sort pas ; la réponse passe par CORS."""
        generateur.llm = modele_double(side_effect=KeyError("cle-secrete-xyz"))

        reponse = await admin_client.post(f"{BLOG}/generate", json={"topic": SUJET}, headers=ORIGINE)

        assert reponse.status_code == 500
        assert reponse.json()["code"] == "erreur_interne"
        assert "cle-secrete" not in reponse.text
        assert reponse.headers["access-control-allow-origin"] == ORIGINE["Origin"]

    @pytest.mark.parametrize("corps, champ", [
        ({"topic": "ab"}, "sujet"),
        ({"topic": "   ab   "}, "sujet"),
        ({"topic": "x" * 301}, "sujet"),
        ({"topic": SUJET, "category": "x" * 101}, "rubrique"),
        ({"topic": SUJET, "additional_context": "x" * 4001}, "contexte fourni"),
        ({}, "sujet"),
    ])
    async def test_entree_invalide_422_lisible(self, admin_client, generateur, corps, champ):
        reponse = await admin_client.post(f"{BLOG}/generate", json=corps, headers=ORIGINE)

        assert reponse.status_code == 422
        assert reponse.json()["code"] == "entree_invalide"
        detail = reponse.json()["detail"]
        assert isinstance(detail, str) and champ in detail
        assert reponse.headers["access-control-allow-origin"] == ORIGINE["Origin"]
        generateur.llm.generate.assert_not_awaited()

    async def test_une_generation_n_est_pas_une_recherche_d_un_lecteur(
        self, admin_client, generateur, db_session
    ):
        await admin_client.post(f"{BLOG}/generate", json={"topic": SUJET})

        evenements = (await db_session.execute(text("SELECT count(*) FROM search_events"))).scalar()
        assert evenements == 0


# ==================== CRÉATION À LA MAIN ====================


class TestCreation:

    async def test_toujours_un_brouillon(self, admin_client, textes):
        post = await _brouillon(admin_client, textes, is_published=True)

        assert post["isPublished"] is False

    async def test_les_citations_sont_relues_en_base(self, admin_client, textes):
        citation = {
            **_citation(textes.code_penal, "299"),
            "reference": "Un nom inventé", "excerpt": "Un extrait réécrit.", "verifie": True,
        }
        corps = _corps(textes)
        corps["bot_analysis"]["citations"] = [citation]

        post = (await admin_client.post(BLOG, json=corps)).json()

        (relue,) = post["botAnalysis"]["citations"]
        assert relue["reference"] == TITRE_CODE_PENAL
        assert relue["excerpt"].startswith("Violation de domicile (1) Est puni d'un emprisonnement de dix")
        assert relue["title"] == "Pourquoi cet article."
        assert relue["verifie"] is True

    @pytest.mark.parametrize("citation, raison", [
        (lambda t: _citation(t.code_penal, "9999"), "n'a pas d'article 9999"),
        (lambda t: _citation(t.projet, "5"), "n'est pas publié"),
        (lambda t: _citation(999_999, "1"), "n'existe pas"),
        (lambda t: _citation(t.code_penal), "indiquez l'article"),
        # La forme d'avant : « article 12 » et un nom inventé, sur un vrai texte.
        (lambda t: {"law_id": t.code_penal, "article_number": "12", "reference": "Nom inventé",
                    "excerpt": "x"}, "n'a pas d'article 12"),
    ])
    async def test_une_citation_que_la_base_ne_porte_pas_422(
        self, admin_client, textes, db_session, citation, raison
    ):
        corps = _corps(textes)
        corps["bot_analysis"]["citations"] = [citation(textes)]

        reponse = await admin_client.post(BLOG, json=corps)

        assert reponse.status_code == 422
        assert reponse.json()["code"] == "entree_invalide"
        assert raison in reponse.json()["detail"]
        assert await _nombre_d_articles(db_session) == 0

    async def test_adresse_normalisee_et_conflit_409(self, admin_client, textes):
        premier = await _brouillon(admin_client, textes, slug="Mon Article : Été / 2026 ?")
        assert premier["slug"] == "mon-article-ete-2026"

        reponse = await admin_client.post(BLOG, json=_corps(textes, slug="mon-article-ete-2026"))

        assert reponse.status_code == 409
        assert reponse.json()["code"] == "slug_pris"

    async def test_sans_adresse_le_titre_la_donne_et_la_suivante_est_libre(self, admin_client, textes):
        premier = await _brouillon(admin_client, textes, title="Le congé d'un bail")
        second = await _brouillon(admin_client, textes, title="Le congé d'un bail")

        assert premier["slug"] == "le-conge-d-un-bail"
        assert second["slug"] == "le-conge-d-un-bail-2"

    @pytest.mark.parametrize("champs, champ", [
        ({"title": "   "}, "titre"),
        ({"title": "**"}, "titre"),
        ({"title": "x" * 501}, "titre"),
        ({"category": "x" * 101}, "rubrique"),
        ({"slug": "???"}, "adresse"),
        ({"suggested_question": "x" * 501}, "question suggérée"),
        ({"bot_analysis": {"summary": "x", "citizenImpact": "y", "keyPoints": "pas une liste"}},
         "points clés"),
        ({"bot_analysis": {"summary": "x", "citizenImpact": "y", "keyPoints": [{"point": "x"}]}},
         "points clés"),
    ])
    async def test_entree_invalide(self, admin_client, textes, db_session, champs, champ):
        reponse = await admin_client.post(BLOG, json=_corps(textes, **champs))

        assert reponse.status_code == 422, reponse.text
        assert reponse.json()["code"] == "entree_invalide"
        assert champ in reponse.json()["detail"]
        assert await _nombre_d_articles(db_session) == 0

    async def test_le_markdown_est_retire_des_textes(self, admin_client, textes):
        corps = _corps(textes, title="**Le congé** d'un bail")
        corps["bot_analysis"]["keyPoints"] = ["- **Premier**", "", "  "]

        post = (await admin_client.post(BLOG, json=corps)).json()

        assert post["title"] == "Le congé d'un bail"
        assert post["botAnalysis"]["keyPoints"] == ["Premier"]


# ==================== MODIFICATION ====================


class TestModification:

    async def test_chaque_champ_de_texte(self, admin_client, textes):
        post = await _brouillon(admin_client, textes)

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={
            "title": "Nouveau titre", "category": "Famille", "context": "Nouveau contexte.",
            "reading_time": "7 min", "suggested_question": "Nouvelle question ?",
            "tags": ["a", "b"],
            "bot_analysis": {
                "summary": "Nouveau résumé.", "keyPoints": ["Un", "Deux"],
                "citizenImpact": "Nouvel impact.",
            },
        })

        modifie = reponse.json()
        assert reponse.status_code == 200
        assert (modifie["title"], modifie["category"], modifie["context"]) == (
            "Nouveau titre", "Famille", "Nouveau contexte."
        )
        assert modifie["readingTime"] == "7 min"
        assert modifie["suggestedQuestion"] == "Nouvelle question ?"
        assert modifie["tags"] == ["a", "b"]
        assert modifie["botAnalysis"]["summary"] == "Nouveau résumé."
        assert modifie["botAnalysis"]["keyPoints"] == ["Un", "Deux"]
        assert modifie["botAnalysis"]["citizenImpact"] == "Nouvel impact."
        # Un brouillon jamais publié : son adresse suit son titre (TestLAdresseDUnBrouillon).
        assert modifie["slug"] == "nouveau-titre" != post["slug"]
        # Sans `citations`, les citations ne changent pas (front actuel).
        assert modifie["botAnalysis"]["citations"] == post["botAnalysis"]["citations"]

    async def test_adresse_normalisee_et_409(self, admin_client, textes):
        un = await _brouillon(admin_client, textes, title="Un")
        deux = await _brouillon(admin_client, textes, title="Deux")

        renomme = await admin_client.put(f"{BLOG}/{un['id']}", json={"slug": "Été 2026 / Bail ?"})
        conflit = await admin_client.put(f"{BLOG}/{deux['id']}", json={"slug": "ete-2026-bail"})
        meme = await admin_client.put(f"{BLOG}/{un['id']}", json={"slug": "ete-2026-bail"})

        assert renomme.json()["slug"] == "ete-2026-bail"
        assert conflit.status_code == 409 and conflit.json()["code"] == "slug_pris"
        assert meme.status_code == 200

    async def test_retirer_une_citation_et_corriger_une_explication(self, admin_client, textes):
        corps = _corps(textes)
        corps["bot_analysis"]["citations"] = [
            _citation(textes.code_civil, "1736"), _citation(textes.code_penal, "299"),
        ]
        post = (await admin_client.post(BLOG, json=corps)).json()
        gardee = {
            **post["botAnalysis"]["citations"][1],
            "title": "Explication corrigée.", "excerpt": "Un extrait réécrit à la main.",
        }

        reponse = await admin_client.put(
            f"{BLOG}/{post['id']}", json={"bot_analysis": {"citations": [gardee]}}
        )

        (citation,) = reponse.json()["botAnalysis"]["citations"]
        assert citation["article_number"] == "299"
        assert citation["title"] == "Explication corrigée."
        # L'extrait reste celui de la base.
        assert citation["excerpt"].startswith("Violation de domicile (1) Est puni d'un emprisonnement")

    async def test_ajouter_une_citation_par_son_numero(self, admin_client, textes):
        post = await _brouillon(admin_client, textes)
        citations = post["botAnalysis"]["citations"] + [_citation(textes.code_civil, "1751")]

        reponse = await admin_client.put(
            f"{BLOG}/{post['id']}", json={"bot_analysis": {"citations": citations}}
        )

        assert [c["article_number"] for c in reponse.json()["botAnalysis"]["citations"]] == [
            "1736", "1751"
        ]

    async def test_un_article_colle_dans_un_autre_se_cite_par_son_numero(
        self, admin_client, textes
    ):
        """L'article 39 du Code du travail, collé dans le morceau « 35.1 » de la base."""
        post = await _brouillon(admin_client, textes)
        citations = post["botAnalysis"]["citations"] + [_citation(textes.code_du_travail, "39")]

        cite = await admin_client.put(
            f"{BLOG}/{post['id']}", json={"bot_analysis": {"citations": citations}}
        )
        publie = await _relu_et_publie(admin_client, post["id"])
        # La rubrique n'est pas une partie relue : l'article reste en ligne.
        modifie = await admin_client.put(f"{BLOG}/{post['id']}", json={"category": "Travail"})

        assert cite.status_code == 200, cite.text
        ajoutee = publie["botAnalysis"]["citations"][-1]
        assert (ajoutee["article_number"], ajoutee["article_id"]) == ("39", textes.ct_35_1)
        assert ajoutee["excerpt"].startswith("Article 39.-")
        assert modifie.status_code == 200, modifie.text
        assert modifie.json()["isPublished"] is True
        assert modifie.json()["etat_publication"] == "publiable"
        assert modifie.json()["botAnalysis"]["citations"][-1]["article_number"] == "39"

    async def test_une_citation_invalide_ne_change_rien(self, admin_client, textes):
        post = await _brouillon(admin_client, textes)

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={
            "title": "Titre qui ne doit pas passer",
            "bot_analysis": {"citations": [_citation(textes.code_penal, "286")]},
        })
        relu = (await admin_client.get(f"{BLOG}/admin/{post['id']}")).json()

        assert reponse.status_code == 422
        assert "n'a pas d'article 286" in reponse.json()["detail"]
        assert relu["title"] == post["title"]
        assert relu["botAnalysis"]["citations"] == post["botAnalysis"]["citations"]

    async def test_publier_reverifie_les_citations_enregistrees(
        self, admin_client, textes, db_session
    ):
        """Une citation d'avant la vérification (law_id 1, « article 12 ») ne passe pas en ligne."""
        post = await _brouillon(admin_client, textes)
        await db_session.execute(
            text("UPDATE blog_posts SET citations = CAST(:c AS jsonb) WHERE id = :id"),
            {"c": json.dumps([{"law_id": 1, "article_number": "12", "reference": "Loi",
                               "title": "x", "excerpt": "Extrait écrit par le modèle."}]),
             "id": post["id"]},
        )
        await db_session.commit()
        _oublier_les_articles(db_session)

        refuse = await admin_client.put(f"{BLOG}/{post['id']}", json={"is_published": True})
        brouillon = (await admin_client.get(f"{BLOG}/admin/{post['id']}")).json()
        corrige = await admin_client.put(f"{BLOG}/{post['id']}", json={
            "bot_analysis": {"citations": [_citation(textes.code_civil, "1736")]},
        })
        publie = await _relu_et_publie(admin_client, post["id"])

        # Écrit à la main : ses sources d'abord (tour 4), puis ses citations.
        assert refuse.status_code == 422 and refuse.json()["code"] == "sources_absentes"
        assert brouillon["isPublished"] is False
        assert brouillon["etat_publication"] == "sources_absentes"
        assert brouillon["botAnalysis"]["citations"][0]["verifie"] is False
        assert any("non vérifiée" in a for a in brouillon["avertissements"])
        assert corrige.status_code == 200, corrige.text
        assert publie["isPublished"] is True

    async def test_un_texte_depublie_bloque_la_modification_d_un_article_publie(
        self, admin_client, client, textes, db_session
    ):
        post = await _publie(admin_client, textes)
        await db_session.execute(
            text("UPDATE laws SET status = 'archived' WHERE id = :id"), {"id": textes.code_civil}
        )
        await db_session.commit()
        _oublier_les_articles(db_session)

        titre = await admin_client.put(f"{BLOG}/{post['id']}", json={"title": "Autre titre"})
        retrait = await admin_client.put(f"{BLOG}/{post['id']}", json={"is_published": False})

        # Rester en ligne exige un article publiable : refusé, rien n'est changé.
        assert titre.status_code == 422
        assert titre.json()["code"] == "citations_non_verifiees"
        # Dépublier reste toujours possible.
        assert retrait.status_code == 200 and retrait.json()["isPublished"] is False
        assert retrait.json()["title"] == post["title"]

    async def test_la_duree_de_lecture_suit_le_texte(self, admin_client, textes):
        post = await _brouillon(admin_client, textes)

        reponse = await admin_client.put(
            f"{BLOG}/{post['id']}", json={"context": "mot " * 1000}
        )

        assert reponse.json()["readingTime"] == "5 min"

    @pytest.mark.parametrize("champs, champ", [
        ({"title": ""}, "titre"),
        ({"bot_analysis": {"summary": "  "}}, "résumé"),
        ({"bot_analysis": {"citations": [{"law_id": 0, "article_number": "1"}]}}, "texte"),
        ({"is_published": "peut-être"}, "publication"),
    ])
    async def test_entree_invalide(self, admin_client, textes, champs, champ):
        post = await _brouillon(admin_client, textes)

        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json=champs)

        assert reponse.status_code == 422
        assert reponse.json()["code"] == "entree_invalide"
        assert champ in reponse.json()["detail"]


# ==================== L'ADRESSE D'UN BROUILLON (AVENANT DU TOUR 7) ====================


class TestLAdresseDUnBrouillon:
    """
    Essai réel du 04/11 : le titre du bail, corrigé (il portait la fausse prémisse de
    l'expulsion sans juge), laissait l'adresse la dire encore, jusqu'à ce que l'administrateur la
    réécrive à la main. Personne ne connaît l'adresse d'un brouillon jamais publié : elle suit son
    titre, tant que le serveur l'a tirée de lui. Celle d'un article publié une fois a pu
    circuler, et une adresse écrite à la main est un choix : gardées.
    """

    async def _titre(self, admin_client, post, titre, **corps) -> dict:
        reponse = await admin_client.put(f"{BLOG}/{post['id']}", json={"title": titre, **corps})
        assert reponse.status_code == 200, reponse.text
        return reponse.json()

    async def test_elle_suit_le_titre_a_chaque_changement(self, admin_client, textes):
        post = await _brouillon(admin_client, textes)
        assert post["slug"] == "le-conge-d-un-bail-verbal"

        apres = await self._titre(admin_client, post, "Le préavis d'un bail écrit")
        encore = await self._titre(admin_client, post, "Été 2026 : quel préavis ?")

        assert apres["slug"] == "le-preavis-d-un-bail-ecrit"
        assert encore["slug"] == "ete-2026-quel-preavis"
        assert (await admin_client.get(f"{BLOG}/admin/{post['id']}")).json()["slug"] == encore["slug"]

    async def test_un_brouillon_genere_comme_a_l_essai_reel(self, admin_client, generateur):
        post = (await admin_client.post(f"{BLOG}/generate", json={"topic": SUJET})).json()["post"]
        assert post["slug"] == "expulsion-sans-juge-ce-que-dit-la-loi"

        apres = await self._titre(admin_client, post, "Bail d'habitation : ce que dit la loi")

        assert apres["slug"] == "bail-d-habitation-ce-que-dit-la-loi"

    async def test_le_formulaire_renvoie_l_adresse_telle_quelle(self, admin_client, textes):
        """Le front envoie tout le formulaire : l'adresse inchangée n'est pas un choix."""
        post = await _brouillon(admin_client, textes)

        apres = await self._titre(admin_client, post, "Un autre titre", slug=post["slug"])

        assert apres["slug"] == "un-autre-titre"

    async def test_unique_comme_a_la_generation(self, admin_client, textes):
        prise = await _brouillon(admin_client, textes, title="Le préavis")
        deuxieme = await _brouillon(admin_client, textes, title="Le préavis, encore")
        post = await _brouillon(admin_client, textes)

        premier = await self._titre(admin_client, deuxieme, "Le préavis")
        second = await self._titre(admin_client, post, "Le préavis")

        assert prise["slug"] == "le-preavis"
        assert (premier["slug"], second["slug"]) == ("le-preavis-2", "le-preavis-3")

    async def test_une_adresse_de_la_redaction_n_est_jamais_prise(self, admin_client, textes):
        post = await _brouillon(admin_client, textes)

        apres = await self._titre(admin_client, post, "Litiges locatifs : résiliation, bail, expulsion abusive")

        assert apres["slug"] == "litiges-locatifs-resiliation-bail-expulsion-abusive-2"

    async def test_une_adresse_suffixee_suit_aussi(self, admin_client, textes):
        """« -2 » : le serveur l'a tirée du titre, l'autre article ayant la première."""
        await _brouillon(admin_client, textes)
        post = await _brouillon(admin_client, textes)
        assert post["slug"] == "le-conge-d-un-bail-verbal-2"

        apres = await self._titre(admin_client, post, "Le préavis d'un bail")

        assert apres["slug"] == "le-preavis-d-un-bail"

    async def test_un_long_titre_coupe_suit_aussi(self, admin_client, textes):
        long = " ".join(["préavis"] * 30)
        await _brouillon(admin_client, textes, title=long)
        post = await _brouillon(admin_client, textes, title=long)
        assert len(post["slug"]) <= 80 and post["slug"].endswith("-2")

        apres = await self._titre(admin_client, post, "Un titre court")

        assert apres["slug"] == "un-titre-court"

    async def test_le_meme_titre_ne_change_pas_l_adresse(self, admin_client, textes):
        await _brouillon(admin_client, textes)
        post = await _brouillon(admin_client, textes)

        apres = await self._titre(admin_client, post, "Le congé d'un bail verbal !")

        assert apres["slug"] == post["slug"] == "le-conge-d-un-bail-verbal-2"
        assert apres["title"] == "Le congé d'un bail verbal !"

    async def test_un_titre_sans_lettre_ne_change_pas_l_adresse(self, admin_client, textes):
        post = await _brouillon(admin_client, textes)

        apres = await self._titre(admin_client, post, "???")

        assert apres["slug"] == post["slug"]

    @pytest.mark.parametrize("cas", ["a-la-creation", "par-un-put", "avec-le-titre"])
    async def test_une_adresse_ecrite_a_la_main_est_gardee(self, admin_client, textes, cas):
        post = await _brouillon(admin_client, textes, **({"slug": "mon-adresse"} if cas == "a-la-creation" else {}))
        if cas == "par-un-put":
            await admin_client.put(f"{BLOG}/{post['id']}", json={"slug": "mon-adresse"})

        apres = await self._titre(
            admin_client, post, "Un autre titre", **({"slug": "mon-adresse"} if cas == "avec-le-titre" else {}),
        )
        encore = await self._titre(admin_client, post, "Encore un autre titre")

        assert apres["slug"] == encore["slug"] == "mon-adresse"

    @pytest.mark.parametrize("retrait", ["retrait-seul", "avec-modifications", "sources"])
    async def test_un_article_publie_une_fois_garde_son_adresse(self, admin_client, textes, retrait):
        """En ligne, ou retiré par n'importe quel chemin : son adresse a pu circuler."""
        post = await _publie(admin_client, textes)
        if retrait == "retrait-seul":
            await admin_client.put(f"{BLOG}/{post['id']}", json={"is_published": False})
        elif retrait == "avec-modifications":
            retire = await self._titre(admin_client, post, "Un titre retiré", is_published=False)
            assert retire["slug"] == post["slug"]
        else:
            await _sources(admin_client, post["id"])

        apres = await self._titre(admin_client, post, "Un autre titre encore")

        assert apres["isPublished"] is False and apres["slug"] == post["slug"] == "le-conge-d-un-bail-verbal"

    @pytest.mark.parametrize("retrait", ["retrait-seul", "avec-modifications", "sources"])
    async def test_publie_avant_cet_avenant_il_le_note_en_quittant_le_site(
        self, admin_client, db_session, textes, retrait
    ):
        """
        En ligne sans la trace `deja_publie` (publié avant cet avenant) : chaque chemin qui le
        retire la note, sinon son adresse, qui a pu circuler, suivrait le titre une fois retiré.
        """
        post = await _publie(admin_client, textes)
        await db_session.execute(
            text("UPDATE blog_posts SET citations = citations - 'deja_publie' WHERE id = :id"), {"id": post["id"]},
        )
        await db_session.commit()
        _oublier_les_articles(db_session)
        if retrait == "retrait-seul":
            await admin_client.put(f"{BLOG}/{post['id']}", json={"is_published": False})
        elif retrait == "avec-modifications":
            await admin_client.put(f"{BLOG}/{post['id']}", json={"tags": ["bail"], "is_published": False})
        else:
            await _sources(admin_client, post["id"])

        apres = await self._titre(admin_client, post, "Un autre titre encore")

        assert apres["isPublished"] is False and apres["slug"] == "le-conge-d-un-bail-verbal"

    async def test_publier_fige_l_adresse_que_le_titre_a_donnee(self, admin_client, textes):
        post = await _brouillon(admin_client, textes)

        retitre = await self._titre(admin_client, post, "Le titre définitif")
        publie = await _relu_et_publie(admin_client, post["id"])
        apres = await self._titre(admin_client, post, "Un autre titre", is_published=False)

        assert retitre["slug"] == publie["slug"] == apres["slug"] == "le-titre-definitif"

    @pytest.mark.parametrize("forme", ["sa-liste", "devenue-dossier"])
    async def test_un_article_d_avant_garde_son_adresse(self, admin_client, db_session, forme):
        """
        Une LISTE dans le dossier (017 publiait par défaut) : il a pu circuler, même hors ligne
        aujourd'hui. Sa liste devient un dossier à la première écriture : la trace y passe avec elle.
        """
        tiree_du_titre = "litige-bailleur-locataire-votre-proprietaire-a-t-il-le-droit-de-couper-l-eau"
        post_id = await article_d_avant(db_session, slug=tiree_du_titre, publie=False)
        lu = (await admin_client.get(f"{BLOG}/admin/{post_id}")).json()
        assert routes_du_blog.adresse_tiree_du_titre(lu["slug"], lu["title"])
        if forme == "devenue-dossier":
            ecrit = await admin_client.put(f"{BLOG}/{post_id}", json={"bot_analysis": {"citations": []}})
            assert ecrit.status_code == 200, ecrit.text

        reponse = await admin_client.put(f"{BLOG}/{post_id}", json={"title": "Un autre titre"})

        assert reponse.status_code == 200, reponse.text
        assert reponse.json()["slug"] == tiree_du_titre

    async def test_le_deja_publie_ne_sort_pas_de_l_administration(self, admin_client, client, textes):
        post = await _publie(admin_client, textes)

        lu = await admin_client.get(f"{BLOG}/admin/{post['id']}")
        public = await client.get(f"{BLOG}/{post['slug']}")

        assert "deja_publie" not in lu.text and "deja_publie" not in public.text


# ==================== SUPPRESSION ====================


class TestSuppression:

    async def test_204_puis_introuvable(self, admin_client, textes, db_session):
        post = await _publie(admin_client, textes)

        reponse = await admin_client.delete(f"{BLOG}/{post['id']}")

        assert reponse.status_code == 204
        assert (await admin_client.get(f"{BLOG}/{post['slug']}")).status_code == 404
        assert await _nombre_d_articles(db_session) == 0


# ==================== AVERTISSEMENTS DE L'ARTICLE ====================


class TestAvertissementsDeLArticle:

    async def test_une_mention_sans_citation_est_signalee(self, admin_client, textes):
        corps = _corps(textes)
        corps["bot_analysis"]["summary"] = (
            "Le congé suit l'usage des lieux selon l'article 1736 du Code civil, et "
            "l'article 299 du Code pénal punit l'intrusion."
        )

        post = (await admin_client.post(BLOG, json=corps)).json()

        (mention,) = [a for a in post["avertissements"] if not a.startswith("Sources absentes")]
        assert "article 299 du Code pénal" in mention
        assert "Résumé" in mention

    async def test_aucune_citation(self, admin_client, textes):
        corps = _corps(textes)
        corps["bot_analysis"]["citations"] = []

        post = (await admin_client.post(BLOG, json=corps)).json()

        assert post["avertissements"] == [
            "Aucune citation : l'article ne s'appuie sur aucun texte de la base, et ne peut pas "
            "être publié.",
            routes_du_blog.SANS_SOURCES,
        ]
        assert post["etat_publication"] == "sources_absentes"

    async def test_si_la_relecture_echoue_la_liste_reste_lisible(
        self, admin_client, textes, monkeypatch
    ):
        """Une exception de la lecture des citations ne rend pas l'administration inutilisable."""
        await _brouillon(admin_client, textes)

        def en_panne(*_):
            raise RuntimeError("lecture impossible")

        monkeypatch.setattr(service_du_blog, "lire_les_citations", en_panne)
        reponse = await admin_client.get(f"{BLOG}/admin/all")

        assert reponse.status_code == 200
        assert any("relecture automatique" in a for a in reponse.json()[0]["avertissements"])


class TestConcurrence:

    async def test_une_adresse_prise_entre_la_verification_et_l_ecriture(
        self, admin_client, textes, monkeypatch
    ):
        """
        Deux enregistrements simultanés sous la même adresse : la contrainte
        UNIQUE refuse le second, qui prend l'adresse suivante au lieu d'une 500.
        """
        premier = await _brouillon(admin_client, textes, title="Le congé")
        vraie = routes_du_blog._slug_libre
        appels = []

        async def en_retard(db, base):
            appels.append(base)
            # Le premier essai croit l'adresse libre : elle vient d'être prise.
            return premier["slug"] if len(appels) == 1 else await vraie(db, base)

        monkeypatch.setattr(routes_du_blog, "_slug_libre", en_retard)
        reponse = await admin_client.post(BLOG, json=_corps(textes, title="Le congé"))

        assert reponse.status_code == 201, reponse.text
        assert reponse.json()["slug"] == "le-conge-2"
        assert len(appels) == 2


# ==================== TOUR 2 : LES ARTICLES D'AVANT LA VÉRIFICATION ====================


def _corps_du_front(post, **modifications):
    """
    Le corps que le front envoie pour une modification (relecture.ts) : tout le
    formulaire, chaque citation TELLE QU'IL L'A REÇUE, l'explication seule
    modifiable.
    """
    analyse = post["botAnalysis"]
    corps = {
        "title": post["title"], "category": post["category"], "reading_time": post["readingTime"],
        "context": post["context"],
        "bot_analysis": {
            "summary": analyse["summary"], "keyPoints": analyse["keyPoints"],
            "citations": analyse["citations"], "citizenImpact": analyse["citizenImpact"],
        },
        "suggested_question": post["suggestedQuestion"], "tags": post["tags"],
        "is_published": post["isPublished"],
    }
    for cle, valeur in modifications.items():
        if cle in ("summary", "citizenImpact", "citations", "keyPoints"):
            corps["bot_analysis"][cle] = valeur
        else:
            corps[cle] = valeur
    return corps


def _citations_de(reponse_json):
    if isinstance(reponse_json, list):
        return [c for p in reponse_json for c in p["botAnalysis"]["citations"]]
    if "post" in reponse_json:
        return reponse_json["post"]["botAnalysis"]["citations"]
    return reponse_json["botAnalysis"]["citations"]


class TestArticlesDAvantLaVerification:
    """
    Les trois lignes du 26/10 de jurix_dev : publiées avant le contrat, law_id 1
    du gabarit, extraits écrits par le modèle. Revue du tour 1 : le public lisait
    « Est puni d'un emprisonnement de trois mois à trois ans » pour l'article 299
    du Code pénal (en base : « dix (10) jours à un (01) an »), comme une vraie
    citation. Revue du tour 2 : leur PROSE, fausse, restait publique. Décision
    du propriétaire (30/10) : cachées du public, gardées en base, visibles et
    corrigeables dans l'administration, remises en ligne seulement par le
    contrôle et la relecture.
    """

    async def test_jamais_servi_au_public(self, admin_client, client, textes, db_session):
        await article_d_avant(db_session)

        liste = await client.get(BLOG)
        page = await client.get(f"{BLOG}/expulsion-locative-loi-cameroun")

        assert liste.status_code == 200 and liste.json() == []
        assert page.status_code == 404
        assert page.json() == {"detail": "Article introuvable.", "code": "introuvable"}
        for reponse in (liste, page):
            assert "trois mois à trois ans" not in reponse.text
            assert "La justice privée est interdite" not in reponse.text

    async def test_l_administration_voit_tout_marque_non_verifie(self, admin_client, textes, db_session):
        post_id = await article_d_avant(db_session)

        post = (await admin_client.get(f"{BLOG}/admin/{post_id}")).json()

        assert post["isPublished"] is True
        assert post["etat_publication"] == "sources_absentes"
        assert post["controle_ia"] is None and post["sources"] == []
        assert [c["verifie"] for c in post["botAnalysis"]["citations"]] == [False, False]
        assert post["botAnalysis"]["citations"][1]["excerpt"] == CITATIONS_D_AVANT[1]["excerpt"]
        assert sum("non vérifiée" in a for a in post["avertissements"]) == 2

    async def test_retirer_du_site_avec_des_modifications_en_cours(
        self, admin_client, client, textes, db_session
    ):
        """
        Le résumé réécrit, et le formulaire entier envoyé avec is_published
        false, citations d'avant comprises : validé en entier, appliqué d'un bloc.
        """
        post_id = await article_d_avant(db_session)
        post = (await admin_client.get(f"{BLOG}/admin/{post_id}")).json()

        reponse = await admin_client.put(f"{BLOG}/{post_id}", json=_corps_du_front(
            post, summary="Résumé réécrit par l'administrateur.", is_published=False,
        ))

        assert reponse.status_code == 200, reponse.text
        retire = reponse.json()
        assert retire["isPublished"] is False
        assert retire["botAnalysis"]["summary"] == "Résumé réécrit par l'administrateur."
        # Les citations d'avant restent ce qu'elles sont : non vérifiées.
        assert [c["verifie"] for c in retire["botAnalysis"]["citations"]] == [False, False]
        assert (await client.get(f"{BLOG}/expulsion-locative-loi-cameroun")).status_code == 404

    @pytest.mark.parametrize("modification, raison", [
        ({"citizenImpact": ""}, "impact pour le citoyen"),
        ({"summary": "   "}, "résumé"),
        ({"citations": [{"law_id": 0, "article_number": "1"}]}, "texte"),
    ])
    async def test_un_corps_refuse_ne_retire_rien(
        self, admin_client, textes, db_session, modification, raison
    ):
        """
        Tour 3 : un corps autre que {"is_published": false} est validé EN ENTIER
        avant d'être appliqué ; refusé, il ne retire rien (au tour 2, il retirait
        l'article d'abord, et le front le croyait encore en ligne).
        """
        post_id = await article_d_avant(db_session)
        post = (await admin_client.get(f"{BLOG}/admin/{post_id}")).json()

        reponse = await admin_client.put(
            f"{BLOG}/{post_id}", json=_corps_du_front(post, is_published=False, **modification)
        )

        assert reponse.status_code == 422
        assert reponse.json()["code"] == "entree_invalide"
        assert raison in reponse.json()["detail"]
        assert "retiré" not in reponse.json()["detail"]
        relu = (await admin_client.get(f"{BLOG}/admin/{post_id}")).json()
        assert relu["isPublished"] is True
        assert relu["botAnalysis"]["summary"] == post["botAnalysis"]["summary"]

    async def test_une_citation_que_la_base_refuse_ne_retire_rien_non_plus(
        self, admin_client, textes, db_session
    ):
        post_id = await article_d_avant(db_session)
        post = (await admin_client.get(f"{BLOG}/admin/{post_id}")).json()
        citations = post["botAnalysis"]["citations"] + [_citation(textes.projet, "5")]

        reponse = await admin_client.put(
            f"{BLOG}/{post_id}", json=_corps_du_front(post, is_published=False, citations=citations)
        )
        exact = await admin_client.put(f"{BLOG}/{post_id}", json={"is_published": False})

        assert reponse.status_code == 422
        assert "Citation 3 : le texte" in reponse.json()["detail"]
        # Retirer du site, lui, passe toujours, quoi que l'article porte.
        assert exact.status_code == 200 and exact.json()["isPublished"] is False

    async def test_une_citation_d_avant_n_est_jamais_relue_sous_son_law_id(
        self, admin_client, textes, db_session
    ):
        """
        Son law_id vient du gabarit : relue en base, elle deviendrait une carte
        « vérifiée » d'un article sans rapport, sous l'explication du modèle.
        Ici, le law_id est celui du Code pénal, qui a un article 299.
        """
        d_avant = [{**CITATIONS_D_AVANT[1], "law_id": textes.code_penal}]
        post_id = await article_d_avant(db_session, citations=d_avant, publie=False)
        post = (await admin_client.get(f"{BLOG}/admin/{post_id}")).json()

        reponse = await admin_client.put(
            f"{BLOG}/{post_id}", json=_corps_du_front(post, title="Titre corrigé")
        )

        (citation,) = reponse.json()["botAnalysis"]["citations"]
        assert reponse.status_code == 200
        assert citation["verifie"] is False
        assert citation["excerpt"] == CITATIONS_D_AVANT[1]["excerpt"]

    async def test_republier_exige_le_nouveau_parcours(
        self, admin_client, client, textes, db_session, reconstruction
    ):
        """
        Décision du propriétaire (30/10) : un article d'avant, PUBLIÉ en base comme
        les lignes du 26/10, ne revient en ligne que par le nouveau parcours —
        sources reconstruites, citations de la base, contrôle IA, relecture de
        chaque partie APRÈS le contrôle, puis « Publier ».
        """
        post_id = await article_d_avant(db_session)
        post = (await admin_client.get(f"{BLOG}/admin/{post_id}")).json()

        seul = await admin_client.put(f"{BLOG}/{post_id}", json={"is_published": True})
        renvoyees = await admin_client.put(
            f"{BLOG}/{post_id}", json=_corps_du_front(post, is_published=True)
        )
        sans_sources = await admin_client.post(f"{BLOG}/{post_id}/controle")
        for refus in (seul, renvoyees, sans_sources):
            assert refus.status_code == 422
            assert refus.json()["code"] == "sources_absentes"

        # 1. Les sources, par la recherche de la génération (doublée ici).
        reconstruction.extraits = [
            extrait(textes.code_civil, textes.cc_1736, "1736", "x", reference="CODE-CIVIL"),
        ]
        reconstruit = await admin_client.post(f"{BLOG}/{post_id}/sources")
        assert reconstruit.status_code == 200, reconstruit.text
        corps = reconstruit.json()
        assert set(corps) == {"post", "sources", "avertissements"}
        assert [(s["n"], s["article_number"]) for s in corps["sources"]] == [(1, "1736")]
        assert corps["post"]["sources"] == corps["sources"]
        # Les citations d'avant ne renvoient à aucun article lu : retirées, et dit.
        assert corps["post"]["botAnalysis"]["citations"] == []
        assert sum("écrite avant la vérification" in a for a in corps["avertissements"]) == 2
        assert any("repasse en brouillon" in a for a in corps["avertissements"])
        assert corps["post"]["isPublished"] is False
        assert corps["post"]["etat_publication"] == "citations_non_verifiees"
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 404

        # 2. Une citation parmi les sources ; 3. cocher AVANT le contrôle est refusé (avenant
        # du tour 5 : une case nomme le contrôle relu, et il n'y en a pas).
        cite = await admin_client.put(f"{BLOG}/{post_id}", json={
            "bot_analysis": {"citations": [_citation(textes.code_civil, "1736")]},
        })
        trop_tot = await admin_client.put(f"{BLOG}/{post_id}", json={"is_published": True})
        coche_trop_tot = await admin_client.put(f"{BLOG}/{post_id}", json={"relecture": TOUT_RELU})
        controle = await _controle(admin_client, post_id)
        assert cite.status_code == 200, cite.text
        assert cite.json()["etat_publication"] == "controle_a_relancer"
        assert trop_tot.status_code == 422 and trop_tot.json()["code"] == "controle_a_relancer"
        assert coche_trop_tot.status_code == 422 and coche_trop_tot.json()["code"] == "relecture_perimee"
        assert controle["etat_publication"] == "relecture_incomplete"
        assert not any(controle["relecture"]["parties"].values())

        # 4. La relecture, contre les verdicts ; 5. « Publier ».
        relu = await admin_client.put(f"{BLOG}/{post_id}", json={"relecture": relecture_cochee(controle)})
        assert relu.json()["etat_publication"] == "publiable"
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 404
        publie = await admin_client.put(f"{BLOG}/{post_id}", json={"is_published": True})

        assert publie.status_code == 200, publie.text
        assert [c["verifie"] for c in publie.json()["botAnalysis"]["citations"]] == [True]
        page = await client.get(f"{BLOG}/{post['slug']}")
        assert page.status_code == 200
        assert "trois mois à trois ans" not in page.text

    async def test_chaque_citation_de_chaque_reponse_porte_verifie(
        self, admin_client, client, generateur, textes, db_session
    ):
        """Le contrat du tour 2, des deux côtés : `verifie` (booléen) partout."""
        await article_d_avant(db_session)
        cree = await admin_client.post(BLOG, json=_corps(textes))
        publie = await _relu_et_publie(admin_client, cree.json()["id"])
        genere = await admin_client.post(f"{BLOG}/generate", json={"topic": SUJET})
        reponses = [
            cree.json(), publie, genere.json(), (await client.get(BLOG)).json(),
            (await client.get(f"{BLOG}/{publie['slug']}")).json(),
            (await admin_client.get(f"{BLOG}/admin/all")).json(),
            (await admin_client.get(f"{BLOG}/admin/{cree.json()['id']}")).json(),
        ]

        for reponse in reponses:
            citations = _citations_de(reponse)
            assert citations, reponse
            assert all(isinstance(c.get("verifie"), bool) for c in citations), reponse


# ==================== TOUR 2 ET 3 : LA BASE D'AUJOURD'HUI ====================


class TestCitationsPubliquesJugeesAujourdHui:
    """
    Revue du tour 1 : une citation dont le texte était dépublié APRÈS la
    publication de l'article restait servie comme vérifiée. Tour 3 : l'article
    n'est plus publiable, et n'est plus servi du tout.
    """

    async def test_un_texte_depublie_apres_la_publication(
        self, admin_client, client, textes, db_session
    ):
        post = await _publie(admin_client, textes)
        await db_session.execute(
            text("UPDATE laws SET status = 'archived' WHERE id = :id"), {"id": textes.code_civil}
        )
        await db_session.commit()
        _oublier_les_articles(db_session)

        page = await client.get(f"{BLOG}/{post['slug']}")
        liste = (await client.get(BLOG)).json()
        admin = (await admin_client.get(f"{BLOG}/admin/{post['id']}")).json()

        assert page.status_code == 404 and liste == []
        assert admin["isPublished"] is True
        assert admin["etat_publication"] == "citations_non_verifiees"
        assert admin["botAnalysis"]["citations"][0]["verifie"] is False
        assert any("n'est plus publié" in a for a in admin["avertissements"])

    async def test_un_article_modifie_dans_la_base_depuis(
        self, admin_client, client, textes, db_session
    ):
        post = await _publie(admin_client, textes)
        await db_session.execute(
            text("UPDATE articles SET content = 'Texte remplacé.' WHERE id = :id"),
            {"id": textes.cc_1736},
        )
        await db_session.commit()

        assert (await client.get(BLOG)).json() == []
        assert (await client.get(f"{BLOG}/{post['slug']}")).status_code == 404

    async def test_une_citation_toujours_portee_reste_entiere(self, admin_client, client, textes):
        post = await _publie(admin_client, textes)

        (citation,) = (await client.get(f"{BLOG}/{post['slug']}")).json()["botAnalysis"]["citations"]

        assert citation["verifie"] is True
        assert citation["excerpt"].startswith("Si le bail a été fait sans écrit")
        assert citation["url"] == f"/laws/{textes.code_civil}?article=1736"


# ==================== TOUR 2 : MENTIONS ET PASSAGES, PAR LES ROUTES ====================


class TestCeQueLaGenerationSignale:

    @pytest.mark.parametrize("phrase, numero", [
        ("Les articles 286 et 312 du Code pénal punissent aussi l'intrusion.", "286"),
        ("L'intrusion est aussi punie par l'article 312.", "312"),
        ("Selon l'article 312, le bailleur ne peut pas entrer.", "312"),
        ("L'article premier du Code pénal impose la loi à tous.", "l'article 1 "),
        ("Les articles 312 à 315 du Code pénal punissent l'intrusion.", "315"),
        ("Le Code pénal, en son article 312, punit l'intrusion.", "312"),
    ])
    async def test_les_formes_de_la_revue_sont_signalees_jusqu_a_la_publication(
        self, admin_client, generateur, phrase, numero
    ):
        """test_r1 de la revue : 201, AUCUN avertissement, puis publié sans un mot."""
        generateur.llm = modele_double(article_du_modele(summary=(
            "S'introduire dans le domicile d'autrui contre son gré est puni par "
            f"l'article 299 du Code Pénal. {phrase}"
        )))

        reponse = await admin_client.post(f"{BLOG}/generate", json={"topic": SUJET})
        assert reponse.status_code == 201, reponse.text
        corps = reponse.json()
        # Contrôlé à la génération : il ne reste que la relecture.
        publie = await admin_client.put(
            f"{BLOG}/{corps['post']['id']}", json={"relecture": relecture_cochee(corps["post"]), "is_published": True}
        )

        assert any(numero in a for a in corps["avertissements"]), corps["avertissements"]
        assert any(numero in a for a in corps["post"]["avertissements"])
        assert publie.status_code == 200, publie.text
        assert any(numero in a for a in publie.json()["avertissements"])

    async def test_une_annotation_seule_ne_fait_pas_un_article(self, admin_client, generateur, db_session):
        """test_r2 de la revue : l'annotation « 1751.2 » ne devient plus « article 1751 »."""
        textes_ = generateur.textes
        generateur.recherche = RechercheDoublee(
            [extrait(textes_.code_civil, textes_.cc_1751_2, "1751.2", "x", reference="CODE-CIVIL")],
            similarites(textes_, {"cc_1751_2": 0.6}),
        )

        reponse = await admin_client.post(f"{BLOG}/generate", json={"topic": SUJET})

        assert reponse.status_code == 422
        assert reponse.json()["code"] == "sans_source"
        generateur.llm.generate.assert_not_awaited()
        assert await _nombre_d_articles(db_session) == 0

    async def test_la_suite_d_un_article_colle_garde_son_numero(self, admin_client, generateur):
        """test_r4 de la revue : « 35.2 » cité « article 35 » avec le début de l'article 35."""
        textes_ = generateur.textes
        generateur.recherche = RechercheDoublee(
            [extrait(textes_.code_du_travail, textes_.ct_35_2, "35.2", CT_35_2,
                     reference="LOI-1992-007")],
            similarites(textes_, {"ct_35_2": 0.6}),
        )
        generateur.llm = modele_double(article_du_modele(
            summary=("Selon l'article 35 du Code du travail, le salaire à prendre en considération "
                     "est le salaire moyen mensuel brut des douze derniers mois."),
            citations=[{"extrait": 1, "objet": "Le salaire de référence des dommages-intérêts."}],
        ))

        corps = (await admin_client.post(
            f"{BLOG}/generate", json={"topic": "Indemnités en cas de licenciement abusif"}
        )).json()

        (carte,) = corps["post"]["botAnalysis"]["citations"]
        assert (carte["article_number"], carte["suite"], carte["verifie"]) == ("39", True, True)
        assert carte["url"].endswith("?article=39")
        assert carte["excerpt"].startswith("[…] (6) Le salaire à prendre en considération")
        # La prose dit « article 35 » d'une règle qu'aucun extrait ne porte sous ce numéro.
        assert any("article 35" in a for a in corps["avertissements"])


class TestAdressesDeLaRedaction:
    """
    Un article de l'API à l'adresse d'un article de la rédaction le masquait,
    dans la liste et à l'adresse partagée (revue du tour 1).
    """

    SLUG = "litiges-locatifs-resiliation-bail-expulsion-abusive"

    async def test_une_adresse_de_la_redaction_n_est_jamais_prise(self, admin_client, textes):
        cree = await admin_client.post(BLOG, json=_corps(textes, slug=self.SLUG))
        autre = await _brouillon(admin_client, textes)
        renomme = await admin_client.put(f"{BLOG}/{autre['id']}", json={"slug": self.SLUG})
        tire_du_titre = await _brouillon(
            admin_client, textes, title="Litiges locatifs : résiliation, bail, expulsion abusive"
        )

        assert cree.status_code == 409 and cree.json()["code"] == "slug_pris"
        assert renomme.status_code == 409 and renomme.json()["code"] == "slug_pris"
        assert tire_du_titre["slug"] == f"{self.SLUG}-2"
