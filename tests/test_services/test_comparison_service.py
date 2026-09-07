"""
Tests du mode COMPARAISON.

Ce que ces tests protegent, dans l'ordre d'importance :

  1. DEUX recherches, une par sujet. C'est la raison d'etre du mode ; une seule
     requete melangee ratait le bloc d'articles definissant le second regime.
  2. Une citation que le modele invente ne devient JAMAIS une source affichee.
     Elle part dans `unmatched_citations`, ou l'utilisateur la voit.
  3. Une cellule sans source reste vide plutot que d'etre comblee.

Le LLM est une doublure injectee, qui rend un dict `{"response": "<json>"}` —
le vrai contrat de GeminiService.generate.

Usage:
    pytest tests/test_services/test_comparison_service.py -v
"""

import json
from unittest.mock import AsyncMock

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.law import Article, Law
from app.schemas.comparison import (
    CRITERES_PAR_DEFAUT,
    MENTION_ABSENCE,
    MENTIONS_ABSENCE,
)
from app.services.comparison_service import (
    COMPARISON_MAX_TOKENS,
    COMPARISON_TEMPERATURE,
    ComparisonError,
    ComparisonOverloadedError,
    ComparisonQuotaError,
    ComparisonService,
    NoContextError,
)
from app.services.gemini_service import (
    GeminiOverloadedError,
    GeminiQuotaError,
    GeminiServiceError,
)

DOMAINE = "Droit de l'Environnement et des Ressources Naturelles"

A_32 = "Le permis de recherche est delivre a une societe miniere en vue de mener des investigations."
A_33 = "Le permis de recherche est delivre pour une duree initiale maximale de trois ans."
A_49 = "Le permis d'exploitation de la petite mine est delivre par l'Administration en charge des mines."
A_53 = "Le permis d'exploitation de la mine industrielle est attribue pour une duree de vingt ans au plus."


def _grille(criteres, sources_a=("32",), sources_b=("53",)):
    """Sortie type du modele, deja au format du schema impose."""
    return json.dumps({
        "lignes": [
            {
                "index": i,
                "critere": c,
                "valeur_a": f"Regime A sur {c}",
                "sources_a": list(sources_a),
                "valeur_b": f"Regime B sur {c}",
                "sources_b": list(sources_b),
            }
            for i, c in enumerate(criteres)
        ],
        "differences_majeures": ["La duree differe."],
        "angles_morts": ["Les sanctions ne sont pas couvertes."],
    })


@pytest.fixture
def llm() -> AsyncMock:
    double = AsyncMock()
    double.generate = AsyncMock(
        return_value={"response": _grille(CRITERES_PAR_DEFAUT)}
    )
    return double


@pytest.fixture
async def code_minier(db_session: AsyncSession, category_ids: dict) -> Law:
    """Un texte portant les deux regimes, comme le vrai Code Minier."""
    law = Law(
        reference="LOI-2016-017",
        title="Loi portant Code Minier",
        type="loi",
        content="\n\n".join([A_32, A_33, A_49, A_53]),
        language="fr",
        status="published",
        category_id=category_ids[DOMAINE],
    )
    db_session.add(law)
    await db_session.flush()
    for ordre, (numero, contenu) in enumerate(
        [("32", A_32), ("33", A_33), ("49", A_49), ("53", A_53)], start=1
    ):
        db_session.add(
            Article(law_id=law.id, number=numero, content=contenu,
                    order=ordre, page_number=ordre, kind="article")
        )
    await db_session.commit()
    await db_session.refresh(law)
    return law


class TestDeuxRecherches:
    """Le coeur du mode : un sujet, une recherche."""

    async def test_une_recherche_par_sujet(self, db_session, code_minier, llm, monkeypatch):
        service = ComparisonService(db_session, llm=llm)
        requetes = []
        vraie = service.search_service.search

        async def espion(requete):
            requetes.append(requete.query)
            return await vraie(requete)

        monkeypatch.setattr(service.search_service, "search", espion)

        await service.compare("permis de recherche", "permis d'exploitation")

        assert requetes == ["permis de recherche", "permis d'exploitation"]

    async def test_les_recherches_ne_se_chevauchent_pas(
        self, db_session, code_minier, llm, monkeypatch
    ):
        """
        Les deux recherches partagent une AsyncSession, dont l'usage concurrent
        n'est pas supporte par SQLAlchemy.

        Mesure sur le corpus avec `asyncio.gather` : le sujet A rendait 8
        articles et le sujet B ZERO, sans lever la moindre exception. La
        comparaison rendait alors un 404 « aucun texte trouve » sur un sujet
        qui en regorge — et un cote seulement TRONQUE serait passe inapercu.

        Ce test verifie le remede a la source : la seconde recherche ne commence
        pas avant que la premiere soit finie.
        """
        service = ComparisonService(db_session, llm=llm)
        journal = []
        vraie = service.search_service.search

        async def espion(requete):
            journal.append(("debut", requete.query))
            resultat = await vraie(requete)
            journal.append(("fin", requete.query))
            return resultat

        monkeypatch.setattr(service.search_service, "search", espion)

        await service.compare("permis de recherche", "permis d'exploitation")

        # Sequentiel : debut A, fin A, debut B, fin B. Un chevauchement
        # donnerait debut A, debut B, ...
        assert [e for e, _ in journal] == ["debut", "fin", "debut", "fin"]

    async def test_les_deux_contextes_atteignent_le_modele(
        self, db_session, code_minier, llm
    ):
        service = ComparisonService(db_session, llm=llm)

        resultat = await service.compare("permis de recherche", "permis d'exploitation")

        prompt = llm.generate.await_args.kwargs["prompt"]
        assert "Sujet A" in prompt and "Sujet B" in prompt
        assert resultat.articles_a and resultat.articles_b

    async def test_sujet_sans_matiere(self, db_session, code_minier, llm):
        """Un sujet absent du corpus arrete la comparaison au lieu de l'inventer."""
        service = ComparisonService(db_session, llm=llm)

        with pytest.raises(NoContextError):
            await service.compare("permis de recherche", "zzzzqqqxyw inexistant")

        llm.generate.assert_not_awaited()

    async def test_restriction_a_un_document(self, db_session, code_minier, llm):
        service = ComparisonService(db_session, llm=llm)

        resultat = await service.compare(
            "permis de recherche", "permis d'exploitation", law_id=code_minier.id
        )

        assert {a.law_id for a in resultat.articles_a} == {code_minier.id}
        assert {a.law_id for a in resultat.articles_b} == {code_minier.id}


class TestCitations:
    """Une source affichee est une source reelle."""

    async def test_citation_resolue_vers_le_texte(self, db_session, code_minier, llm):
        service = ComparisonService(db_session, llm=llm)

        resultat = await service.compare("permis de recherche", "permis d'exploitation")

        cellule = resultat.rows[0].a
        assert cellule.sources, "la cellule doit porter sa source"
        source = cellule.sources[0]
        assert source.number == "32"
        assert source.content == A_32, "le texte integral est deplie sous la cellule"
        assert source.law_id == code_minier.id
        assert resultat.unmatched_citations == []

    async def test_citation_inventee_ecartee_et_signalee(
        self, db_session, code_minier, llm
    ):
        """
        Le point le plus important du fichier.

        Un numero que le modele invente ne doit pas devenir un lien vers un
        article : il n'en existe pas. Il ne doit pas non plus disparaitre en
        silence — l'utilisateur doit voir qu'une citation n'a pas ete retrouvee.
        """
        llm.generate.return_value = {
            "response": _grille(CRITERES_PAR_DEFAUT, sources_a=("32", "9999"))
        }
        service = ComparisonService(db_session, llm=llm)

        resultat = await service.compare("permis de recherche", "permis d'exploitation")

        numeros = [s.number for s in resultat.rows[0].a.sources]
        assert numeros == ["32"], "l'article fantome ne devient pas une source"
        assert "9999" in resultat.unmatched_citations

    async def test_numero_normalise_des_deux_cotes(self, db_session, category_ids, llm):
        """« Article 1er » cite, « 1er » en base : la meme ligne."""
        law = Law(reference="LOI-1ER", title="Texte", type="loi",
                  content="Article premier du texte.", language="fr",
                  status="published", category_id=category_ids[DOMAINE])
        db_session.add(law)
        await db_session.flush()
        db_session.add(Article(law_id=law.id, number="1er", order=1, kind="article",
                               content="Le present texte fixe le regime des licences."))
        await db_session.commit()

        llm.generate.return_value = {
            "response": _grille(CRITERES_PAR_DEFAUT, sources_a=("Article 1er",),
                                sources_b=("1",))
        }
        service = ComparisonService(db_session, llm=llm)

        resultat = await service.compare("licences", "licences")

        assert resultat.rows[0].a.sources[0].number == "1er"
        assert resultat.unmatched_citations == []

    async def test_meme_numero_dans_deux_lois(self, db_session, category_ids, llm):
        """
        Un numero d'article n'identifie rien a lui seul.

        Mesure sur le corpus reel : une seule recuperation de huit articles a
        rendu trois « article 1 » appartenant a trois lois differentes. Avec un
        index a valeur unique et partage entre les deux sujets, une cellule
        citant « article 1 » affichait le texte de la premiere loi rencontree —
        une source FAUSSE presentee comme verifiee, donc pire que l'invention
        que ce mode combat.

        Chaque cote resout desormais contre SES propres articles.
        """
        textes = {
            "alpha": "Le present decret porte creation de l'organisme alpha zephyr.",
            "beta": "Le present decret porte organisation du service beta quinoa.",
        }
        for nom, contenu in textes.items():
            law = Law(reference=f"DECRET-{nom.upper()}", title=f"Decret {nom}",
                      type="decret", content=contenu, language="fr",
                      status="published", category_id=category_ids[DOMAINE])
            db_session.add(law)
            await db_session.flush()
            # Les deux lois ont un « article 1 » : c'est tout le probleme.
            db_session.add(Article(law_id=law.id, number="1", order=1,
                                   kind="article", content=contenu))
        await db_session.commit()

        llm.generate.return_value = {"response": _grille(
            CRITERES_PAR_DEFAUT, sources_a=("1",), sources_b=("1",)
        )}
        service = ComparisonService(db_session, llm=llm)

        resultat = await service.compare("alpha zephyr", "beta quinoa")

        loi_a = {a.law_id for a in resultat.articles_a}
        loi_b = {a.law_id for a in resultat.articles_b}
        assert loi_a and loi_b and loi_a != loi_b, "les deux sujets viennent de lois differentes"
        # Chaque cote cite SA loi, pas celle de l'autre.
        assert {s.law_id for s in resultat.rows[0].a.sources} <= loi_a
        assert {s.law_id for s in resultat.rows[0].b.sources} <= loi_b
        assert resultat.rows[0].a.sources[0].content == textes["alpha"]
        assert resultat.rows[0].b.sources[0].content == textes["beta"]

    async def test_cellule_sans_source_reste_vide(self, db_session, code_minier, llm):
        llm.generate.return_value = {"response": json.dumps({
            "lignes": [{
                "index": i, "critere": c, "valeur_a": MENTION_ABSENCE, "sources_a": [],
                "valeur_b": "Vingt ans", "sources_b": ["53"],
            } for i, c in enumerate(CRITERES_PAR_DEFAUT)],
            "differences_majeures": [], "angles_morts": [],
        })}
        service = ComparisonService(db_session, llm=llm)

        resultat = await service.compare("permis de recherche", "permis d'exploitation")

        assert resultat.rows[0].a.sources == []
        assert resultat.rows[0].a.value == MENTION_ABSENCE
        assert resultat.rows[0].b.sources


class TestGrille:
    """Les axes sont imposes, pas negocies."""

    async def test_criteres_par_defaut(self, db_session, code_minier, llm):
        service = ComparisonService(db_session, llm=llm)

        resultat = await service.compare("permis de recherche", "permis d'exploitation")

        assert [r.criterion for r in resultat.rows] == CRITERES_PAR_DEFAUT

    async def test_criteres_sur_mesure(self, db_session, code_minier, llm):
        axes = ["Duree", "Autorite"]
        llm.generate.return_value = {"response": _grille(axes)}
        service = ComparisonService(db_session, llm=llm)

        resultat = await service.compare(
            "permis de recherche", "permis d'exploitation", criteria=axes
        )

        assert [r.criterion for r in resultat.rows] == axes
        assert "0. Duree" in llm.generate.await_args.kwargs["prompt"]

    async def test_ligne_manquante_devient_une_absence(
        self, db_session, code_minier, llm
    ):
        """Le modele saute un critere : la ligne existe quand meme, vide."""
        llm.generate.return_value = {"response": json.dumps({
            "lignes": [{"index": 0, "critere": CRITERES_PAR_DEFAUT[0], "valeur_a": "A",
                        "sources_a": ["32"], "valeur_b": "B", "sources_b": ["53"]}],
            "differences_majeures": [], "angles_morts": [],
        })}
        service = ComparisonService(db_session, llm=llm)

        resultat = await service.compare("permis de recherche", "permis d'exploitation")

        assert len(resultat.rows) == len(CRITERES_PAR_DEFAUT)
        assert resultat.rows[-1].a.value == MENTION_ABSENCE

    async def test_reponses_desordonnees_ne_glissent_pas(
        self, db_session, code_minier, llm
    ):
        """
        Le défaut le plus dangereux du mode, parce qu'il est INVISIBLE.

        L'ancien code retombait sur la position des que le nombre de lignes
        correspondait, sans verifier l'ordre. Un modele qui reformule ses
        libelles et repond en ordre inverse plaçait six reponses sur sept sous
        le mauvais critere — avec des articles cites qui se resolvaient
        normalement et `unmatched_citations` vide. Rien ne le signalait.

        Ici les libelles sont tous reformules ET l'ordre est inverse : seul le
        champ `index` peut rattacher chaque reponse a son critere.
        """
        lignes = [
            {"index": i, "critere": f"Libelle reformule {i}",
             "valeur_a": f"REPONSE-{i}", "sources_a": ["32"],
             "valeur_b": f"REPONSE-{i}", "sources_b": ["53"]}
            for i in reversed(range(len(CRITERES_PAR_DEFAUT)))
        ]
        llm.generate.return_value = {"response": json.dumps({
            "lignes": lignes, "differences_majeures": [], "angles_morts": [],
        })}
        service = ComparisonService(db_session, llm=llm)

        resultat = await service.compare("permis de recherche", "permis d'exploitation")

        for position, ligne in enumerate(resultat.rows):
            assert ligne.criterion == CRITERES_PAR_DEFAUT[position]
            assert ligne.a.value == f"REPONSE-{position}", (
                "la reponse d'index n a glisse sous un autre critere"
            )

    async def test_index_absent_repli_sur_le_libelle(
        self, db_session, code_minier, llm
    ):
        """Sans `index`, le libelle exact rattache encore ; rien d'autre."""
        llm.generate.return_value = {"response": json.dumps({
            "lignes": [
                {"critere": CRITERES_PAR_DEFAUT[3], "valeur_a": "TROUVE",
                 "sources_a": ["33"], "valeur_b": "x", "sources_b": []},
                {"critere": "Libelle inconnu", "valeur_a": "IGNORE",
                 "sources_a": ["32"], "valeur_b": "y", "sources_b": []},
            ],
            "differences_majeures": [], "angles_morts": [],
        })}
        service = ComparisonService(db_session, llm=llm)

        resultat = await service.compare("permis de recherche", "permis d'exploitation")

        assert resultat.rows[3].a.value == "TROUVE"
        # « Libelle inconnu » ne se rattache a rien : les autres restent vides,
        # au lieu d'heriter d'une reponse qui ne les concerne pas.
        assert resultat.rows[0].a.value == MENTION_ABSENCE

    async def test_schema_json_impose_au_modele(self, db_session, code_minier, llm):
        service = ComparisonService(db_session, llm=llm)

        await service.compare("permis de recherche", "permis d'exploitation")

        kwargs = llm.generate.await_args.kwargs
        assert kwargs["response_mime_type"] == "application/json"
        schema = kwargs["response_schema"]
        ligne = schema["properties"]["lignes"]["items"]
        # Les sources sont REQUISES : c'est ce qui interdit de combler un trou.
        assert "sources_a" in ligne["required"]
        assert "sources_b" in ligne["required"]

    async def test_parametres_de_generation(self, db_session, code_minier, llm):
        service = ComparisonService(db_session, llm=llm)

        await service.compare("permis de recherche", "permis d'exploitation")

        kwargs = llm.generate.await_args.kwargs
        assert kwargs["temperature"] == COMPARISON_TEMPERATURE == 0.2
        assert kwargs["max_tokens"] == COMPARISON_MAX_TOKENS == 8192


class TestRobustesse:
    """Ce que le modele rend de travers ne doit pas casser la reponse."""

    async def test_document_sans_le_sujet_ne_ment_pas_sur_la_cause(
        self, db_session, code_minier, category_ids, llm
    ):
        """
        Le filtre `law_id` s'applique APRES la recherche.

        Sans sur-echantillonnage, un texte mieux classe occupait toutes les
        places, le filtre les jetait, et l'utilisateur recevait « aucun texte
        dans le corpus » alors que l'article existait dans le document demande.
        Le message doit au moins dire que la restriction est en cause.
        """
        leurre = Law(reference="DECRET-LEURRE", title="Decret d'application minier",
                     type="decret", language="fr", status="published",
                     category_id=category_ids[DOMAINE],
                     content="Permis d'exploitation de la mine industrielle.")
        db_session.add(leurre)
        await db_session.flush()
        for i in range(1, 12):
            db_session.add(Article(
                law_id=leurre.id, number=str(100 + i), order=i, kind="article",
                content="Le permis d'exploitation de la mine industrielle est "
                        f"regi par la presente disposition {i}."))
        await db_session.commit()

        service = ComparisonService(db_session, llm=llm)
        resultat = await service.compare(
            "permis de recherche", "permis d'exploitation",
            law_id=code_minier.id, top_k=3,
        )

        # Le sur-echantillonnage a laisse passer les articles du bon document.
        assert {a.law_id for a in resultat.articles_b} == {code_minier.id}

    async def test_message_404_designe_le_document(self, db_session, code_minier, llm):
        service = ComparisonService(db_session, llm=llm)

        with pytest.raises(NoContextError, match="document"):
            await service.compare("permis de recherche", "zzzqqq inexistant",
                                  law_id=code_minier.id)

    async def test_mention_absence_suit_la_langue(self, db_session, code_minier, llm):
        """Une comparaison anglaise ne rend pas ses cases vides en francais."""
        llm.generate.return_value = {"response": json.dumps({
            "lignes": [{"index": i, "critere": c, "valeur_a": "", "sources_a": [],
                        "valeur_b": "", "sources_b": []}
                       for i, c in enumerate(CRITERES_PAR_DEFAUT)],
            "differences_majeures": [], "angles_morts": [],
        })}
        service = ComparisonService(db_session, llm=llm)

        fr = await service.compare("permis de recherche", "permis d'exploitation", "fr")
        en = await service.compare("permis de recherche", "permis d'exploitation", "en")

        assert fr.rows[0].a.value == MENTIONS_ABSENCE["fr"] == MENTION_ABSENCE
        assert en.rows[0].a.value == MENTIONS_ABSENCE["en"]
        assert "é" in MENTIONS_ABSENCE["fr"], "la mention francaise porte ses accents"

    @pytest.mark.parametrize("brut,attendu", [
        ("Les durees different.", ["Les durees different."]),   # chaine au lieu de liste
        ([{"point": "x"}], []),                                  # objets au lieu de chaines
        (None, []),
        (["  ", "vrai"], ["vrai"]),
    ])
    async def test_differences_mal_formees(
        self, db_session, code_minier, llm, brut, attendu
    ):
        """
        `list("une chaine")` rendait une liste de CARACTERES, affichee en autant
        de puces ; une liste d'objets faisait echouer Pydantic en fin de course,
        donc un 500 apres avoir paye la generation.
        """
        llm.generate.return_value = {"response": json.dumps({
            "lignes": [{"index": i, "critere": c, "valeur_a": "a", "sources_a": ["32"],
                        "valeur_b": "b", "sources_b": ["53"]}
                       for i, c in enumerate(CRITERES_PAR_DEFAUT)],
            "differences_majeures": brut, "angles_morts": brut,
        })}
        service = ComparisonService(db_session, llm=llm)

        resultat = await service.compare("permis de recherche", "permis d'exploitation")

        assert resultat.key_differences == attendu
        assert resultat.blind_spots == attendu

    async def test_orphelines_dedupliquees_sur_la_forme_normalisee(
        self, db_session, code_minier, llm
    ):
        """« Article 40 » et « art. 40 » sont le meme fantome."""
        llm.generate.return_value = {
            "response": _grille(CRITERES_PAR_DEFAUT,
                                sources_a=("Article 40", "art. 40", "40"))
        }
        service = ComparisonService(db_session, llm=llm)

        resultat = await service.compare("permis de recherche", "permis d'exploitation")

        assert len(resultat.unmatched_citations) == 1

    async def test_articles_consultes_sans_texte_integral(
        self, db_session, code_minier, llm
    ):
        """Ces listes ne servent qu'a compter : le texte y etait du poids mort."""
        service = ComparisonService(db_session, llm=llm)

        resultat = await service.compare("permis de recherche", "permis d'exploitation")

        assert all(a.content == "" for a in resultat.articles_a)
        # Le texte reste la ou il sert : sous la cellule.
        assert resultat.rows[0].a.sources[0].content


class TestEchecs:
    async def test_json_tronque(self, db_session, code_minier, llm):
        """Sortie coupee au plafond de jetons : message qui dit la vraie cause."""
        llm.generate.return_value = {"response": '{"lignes": [{"critere": "Duree"'}
        service = ComparisonService(db_session, llm=llm)

        with pytest.raises(ComparisonError, match="incomplete"):
            await service.compare("permis de recherche", "permis d'exploitation")

    @pytest.mark.parametrize(
        "erreur,attendue",
        [
            (GeminiQuotaError("quota"), ComparisonQuotaError),
            (GeminiOverloadedError("sature"), ComparisonOverloadedError),
            (GeminiServiceError("vide"), ComparisonError),
        ],
    )
    async def test_traduction_des_erreurs(
        self, db_session, code_minier, llm, erreur, attendue
    ):
        llm.generate.side_effect = erreur
        service = ComparisonService(db_session, llm=llm)

        with pytest.raises(attendue):
            await service.compare("permis de recherche", "permis d'exploitation")
