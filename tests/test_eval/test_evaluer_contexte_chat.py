"""
scripts/eval/evaluer_contexte_chat.py mesure ce que le chat donne au modele,
et ce qu'il refuse.

Ce qui doit tenir pour que ses chiffres veuillent dire quelque chose : le rang
porte sur l'article attendu (loi, numero de base), « lu » sur les extraits que
le budget du contexte garde, un texte ajoute par le chat n'est hors sujet que
s'il n'est pas attendu, et le volet semantique simule est rendu tel quel apres
la mesure — sinon les tests suivants chercheraient dans des tableaux.
"""

import asyncio
from pathlib import Path

import pytest

from app.schemas.search import ChunkResult
from app.services import search_service as module_recherche
from app.services.rag_service import RAGService
from scripts.eval import evaluer_contexte_chat as module


def _extrait(article_id, law_id, numero, contenu="Contenu.", source="semantic", score=0.5):
    return ChunkResult(
        article_id=article_id, law_id=law_id, number=numero, content=contenu,
        reference=f"R-{law_id}", law_title=f"Loi {law_id}", relevance_score=score,
        source=source,
    )


class TestRang:

    def test_un_morceau_compte_pour_son_article(self):
        extraits = [_extrait(1, 9, "12"), _extrait(2, 2274, "35.2"), _extrait(3, 2274, "35.1")]

        assert module.rang(extraits, 2274, "35") == 2

    def test_le_numero_seul_ne_suffit_pas(self):
        assert module.rang([_extrait(1, 9, "35")], 2274, "35") is None

    def test_absent(self):
        assert module.rang([], 2274, "35") is None


class TestAjoutsHorsSujet:

    def test_un_texte_ajoute_et_non_attendu(self):
        avant = [_extrait(1, 9, "12")]
        apres = [_extrait(2, 2274, "35.1"), _extrait(3, 77, "3"), _extrait(1, 9, "12")]

        assert module.ajouts_hors_sujet(avant, apres, ["R-2274"]) == ["R-77"]

    def test_un_texte_deja_la_n_est_pas_un_ajout(self):
        avant = [_extrait(1, 77, "1")]

        assert module.ajouts_hors_sujet(avant, [_extrait(3, 77, "3")], []) == []


class TestSynthese:

    def test_taux_mrr_et_compteurs(self):
        lignes = [
            {"rang_avant": 1, "rang": 1, "rang_lu": 1, "loi_lue": True, "refus": False,
             "mode": "sain", "ajouts": [], "ajouts_hors_sujet": [], "similarite": 0.5,
             "premier_voisin": 0.5, "tableaux_lus": 0, "duree_ms": 10, "sql_ms": 100,
             "chemin": "exact"},
            {"rang_avant": None, "rang": 4, "rang_lu": None, "loi_lue": True, "refus": False,
             "mode": "faible", "ajouts": ["R-1 art. 2", "R-3 art. 4"], "ajouts_hors_sujet": ["R-3"],
             "similarite": 0.3, "premier_voisin": 0.31, "tableaux_lus": 2, "duree_ms": 30,
             "sql_ms": 120, "chemin": "exact"},
            {"rang_avant": 2, "rang": None, "rang_lu": None, "loi_lue": False, "refus": True,
             "mode": "faible", "ajouts": [], "ajouts_hors_sujet": None, "similarite": None,
             "premier_voisin": None, "tableaux_lus": 0, "duree_ms": 20, "sql_ms": None,
             "chemin": "echec"},
        ]

        s = module.synthese(lignes)

        assert (s["n"], s["refus"], s["faibles"]) == (3, 1, 2)
        assert (s["ajouts"], s["ajouts_hors_sujet"], s["questions_avec_ajout_hors_sujet"]) == (2, 1, 1)
        assert (s["tableaux_lus"], s["questions_avec_tableaux_lus"]) == (2, 1)
        assert s["avant"]["hit@8"] == round(2 / 3, 3)
        assert s["apres"]["hit@1"] == round(1 / 3, 3)
        assert s["apres"]["mrr@8"] == round((1 + 1 / 4) / 3, 3)
        assert s["apres"]["lu"] == round(1 / 3, 3)
        assert s["loi_lue"] == round(2 / 3, 3)
        assert (s["similarite_min"], s["similarite_max"], s["premier_voisin_min"]) == (0.3, 0.5, 0.31)
        assert (s["duree_mediane_ms"], s["sql_semantique_median_ms"]) == (20, 110)
        assert s["chemins"] == {"echec": 1, "exact": 2}


def test_tableaux_lus_compte_les_lois_de_finances():
    finances = _extrait(1, 2232, "76.37").model_copy(update={
        "law_title": "Loi N°2022/020 portant loi de finances pour l'exercice 2023"})
    assert module.tableaux_lus([finances, _extrait(2, 2274, "35.1")]) == 1
    # « loi des Finances » (2026) et les lois de reglement en sont aussi.
    des_finances = finances.model_copy(update={
        "article_id": 3, "law_title": "Loi N°2025/012 portant loi des Finances de la République"})
    reglement = finances.model_copy(update={
        "article_id": 4, "law_title": "Loi N°2020/017 portant Loi de Règlement"})
    assert module.tableaux_lus([des_finances, reglement]) == 2


def test_les_tableaux_de_finances_sont_les_lignes_de_tableau():
    """Le rejeu de l'incident : des baremes d'amendes, pas seulement des tableaux AE/CP."""
    titre = "Loi N°2023/019 portant loi de finances pour l'exercice 2024"
    bareme = _extrait(1, 2300, "35.8", contenu="| N° | Libellé | Montant |\n|---|---|---|\n| 1 | x | 5 |")
    bareme = bareme.model_copy(update={"law_title": titre})
    regle = bareme.model_copy(update={"article_id": 2, "content": "Sont exonérées les indemnités."})
    code = _extrait(3, 2274, "35.1", contenu="| a | b |\n|---|---|")

    assert module.tableaux_de_finances_lus([bareme, regle, code]) == 1
    assert module.tableaux_lus([bareme, regle, code]) == 2


class TestCalibration:
    """La grille dit ce que la barriere refuserait, seuil par seuil."""

    @staticmethod
    def _l(question, langue, similarite, confirmee=False):
        return {"question": question, "langue": langue, "similarite": similarite,
                "confirmee": confirmee}

    def test_refuserait(self):
        assert module.refuserait(self._l("q", "fr", 0.30), 0.37)
        assert not module.refuserait(self._l("q", "fr", 0.40), 0.37)
        # Sous le seuil, une question confirmee (article nomme, tous les
        # mots-cles) passe ; sans similarite (volet en panne), aussi.
        assert not module.refuserait(self._l("q", "fr", 0.30, confirmee=True), 0.37)
        assert module.refuserait(self._l("q", "fr", None), 0.37)
        assert not module.refuserait(self._l("q", "fr", None, confirmee=True), 0.37)

    def test_grille_marge_et_langue_de_la_question(self):
        resultats = {
            "normal": {
                "jeu_fr": {"items": [
                    self._l("a", "fr", 0.40), self._l("b", "fr", 0.50),
                    # Langue incertaine : seuil anglais, meme dans le jeu francais.
                    self._l("c", None, 0.29),
                    self._l("d", "fr", 0.33, confirmee=True),
                ]},
                "suggerees_en": {"items": [self._l("e", "en", 0.36)]},
                # Les questions de tous les jours sont legitimes.
                "reelles_fr": {"items": [self._l("Ma femme est partie", "fr", 0.351)]},
                "hors_sujet_fr": {"items": [self._l("f", "fr", 0.35), self._l("g", "fr", 0.41)]},
            },
            "leurre": {"jeu_fr": {"items": [self._l("a", "fr", 0.34), self._l("b", "fr", 0.38)]}},
            "leurre2": {"jeu_fr": {"items": [self._l("a", "fr", 0.31)]}},
        }

        c = module.calibration(resultats, {"fr": 0.37, "en": 0.26})

        fr, en = c["fr"], c["en"]
        assert fr["effectifs"] == {"legitimes": 4, "hors_sujet": 2, "leurre": 2, "leurre2": 1}
        assert (fr["legitimes_similarite_min"], fr["marge"]) == (0.33, -0.04)
        # La question confirmee ne compte pas dans la marge « sans autre preuve ».
        assert (fr["legitimes_sans_autre_preuve_similarite_min"], fr["marge_sans_autre_preuve"]) == (
            0.351, -0.019,
        )
        assert fr["hors_sujet_similarite_max"] == 0.41
        retenu = next(g for g in fr["grille"] if g["seuil"] == 0.37)
        assert retenu == {"seuil": 0.37, "refus_legitimes": 1, "refus_hors_sujet": 1,
                          "refus_leurre": 1, "refus_leurre2": 1}
        assert next(g for g in fr["grille"] if g["seuil"] == 0.42)["refus_legitimes"] == 2
        assert (en["legitimes_similarite_min"], en["marge"]) == (0.29, 0.03)
        assert fr["grille"][0]["seuil"] == 0.20 and en["grille"][0]["seuil"] == 0.10

    def test_une_relance_n_est_refusee_que_seule_et_jointe(self):
        """
        Une relance est d'abord jugee seule, puis jointe a la question
        precedente (RAGService._chercher_la_relance) : elle n'est refusee que
        si les deux le sont.
        """
        relance = {"question": "Peux-tu détailler ?", "langue": "fr",
                   "similarite_seule": 0.29, "confirmee_seule": False,
                   "similarite_jointe": 0.47, "confirmee_jointe": False}

        assert not module.refuserait(relance, 0.30)
        assert module.refuserait(relance, 0.50)
        resultats = {"normal": {"relances_fr": {"items": [relance]}}}
        c = module.calibration(resultats, {"fr": 0.30, "en": 0.22})
        # La similarite qui decide est la meilleure des deux.
        assert c["fr"]["legitimes_similarite_min"] == 0.47


def test_une_relance_reconnue_n_est_jugee_que_jointe():
    """
    Reconnue (est_une_relance), une relance n'est cherchee que jointe a son
    sujet : c'est cette recherche qui decide, a tout seuil.
    """
    relance = {"question": "Peux-tu détailler ?", "langue": "fr", "relance": True,
               "similarite_jointe": 0.47, "confirmee_jointe": False}

    assert not module.refuserait(relance, 0.30)
    assert module.refuserait(relance, 0.50)
    assert module.refuserait({**relance, "confirmee_jointe": None, "similarite_jointe": None}, 0.30)


def test_une_relance_cherchee_seule_et_jointe_n_est_refusee_que_deux_fois():
    """
    Une relance qui peut changer de sujet (« Et la nationalité ? ») est
    cherchee seule ET jointe : refusee seulement si les deux le sont, et la
    meilleure similarite decide.
    """
    ligne = {"question": "Et la nationalité ?", "langue": "fr", "relance": True,
             "seule_et_jointe": True,
             "similarite_seule": 0.445, "confirmee_seule": False,
             "similarite_jointe": 0.395, "confirmee_jointe": False}

    assert not module.refuserait(ligne, 0.40)
    assert module.refuserait(ligne, 0.45)
    resultats = {"normal": {"changements_de_sujet_fr": {"items": [ligne]}}}
    c = module.calibration(resultats, {"fr": 0.30, "en": 0.22})
    # Un changement de sujet est legitime, et sa meilleure similarite compte.
    assert c["fr"]["effectifs"] == {"legitimes": 1}
    assert c["fr"]["legitimes_similarite_min"] == 0.445


def test_porte_ses_mots():
    """
    La mesure verifie du dehors qu'un extrait lu porte un mot propre de la
    question : « la dot » lue dans le Code du travail est la forme de la revue.
    """
    dot = _extrait(1, 1804, "1382.2", contenu="Refus de restituer la dot.")
    code = _extrait(2, 2274, "35.1", contenu="Indemnité de licenciement.")

    assert module.porte_ses_mots("la dot", [code, dot])
    assert not module.porte_ses_mots("la dot", [code])
    # Au pluriel pres ; et sans mot propre, rien a porter.
    assert module.porte_ses_mots("Et les indemnités ?", [code])
    assert not module.porte_ses_mots("Peux-tu détailler ?", [code, dot])


def test_servie_par_l_index_seule_la_confirmation_compte():
    """Sur le repli, l'index n'est pas cru seul : refusee sauf texte confirme, a tout seuil."""
    ligne = {"question": "q", "langue": "fr", "similarite": 0.60, "confirmee": False,
             "chemin": module_recherche.CHEMIN_REPLI_HNSW}

    assert module.refuserait(ligne, 0.10)
    assert not module.refuserait({**ligne, "confirmee": True}, 0.90)


def test_l_historique_d_une_mesure():
    messages = module.historique_de(["Q1", "Q2"], reponse="R")

    assert [(m.role, m.content) for m in messages] == [
        ("user", "Q1"), ("assistant", "R"), ("user", "Q2"), ("assistant", "R"),
    ]
    assert messages[1].sources is None


def test_l_etat_repli_force_le_repli_sur_l_index():
    rag = module._rag(object(), "repli")

    assert rag.search_service.BUDGET_RECHERCHE_EXACTE_MS == 1
    assert rag.search_service.semantique_exacte is True and rag.llm is None


def test_programmes_lus_ne_compte_que_les_tableaux_ae_cp():
    tableau = _extrait(1, 2232, "76.37", contenu="| N° | Programme | AE | CP |\n|---|---|---|---|")
    tableau = tableau.model_copy(update={
        "law_title": "Loi N°2022/020 portant loi de finances pour l'exercice 2023"})
    taux = tableau.model_copy(update={"article_id": 2, "content": "Le taux de la TVA est de 17,5 %."})

    assert module.programmes_lus([tableau, taux, _extrait(3, 2274, "35.1")]) == 1
    assert module.tableaux_lus([tableau, taux]) == 2


def test_lus_par_le_modele_suit_le_budget_du_contexte():
    """Huit extraits de 5 500 caracteres : le budget de 24 000 en ecarte."""
    rag = RAGService.__new__(RAGService)
    extraits = [_extrait(i, 1, str(i), contenu="x " * 2750) for i in range(1, 9)]

    lus = module.lus_par_le_modele(rag, "Que prevoit la loi ?", extraits)

    assert 0 < len(lus) < len(extraits)
    assert [c.number for c in lus] == [str(i) for i in range(1, len(lus) + 1)]


@pytest.mark.asyncio
async def test_le_leurre_rend_des_copies_et_restaure_la_recherche():
    """
    La fusion RRF ecrase relevance_score en place : sans copies, la premiere
    question modifierait la liste servie a toutes les suivantes.
    """
    classe = module_recherche.SearchService
    avant = (classe._lancer_vectorisation, classe._chunks_semantiques)
    leurre = [_extrait(1, 2232, "76.37", score=0.34)]
    cle = {"cle": "question"}

    with module.volet_semantique("leurre", {"question": leurre}, cle):
        service = classe.__new__(classe)
        tache = service._lancer_vectorisation("question")
        rendu = await service._chunks_semantiques(tache, None, 32, 0)
        rendu[0].relevance_score = 1.0
        assert tache.done()

    assert [c.number for c in rendu] == ["76.37"]
    assert leurre[0].relevance_score == 0.34
    # Le chat ne lit l'index que par le repli de sa recherche exacte.
    assert service.chemin_semantique == module_recherche.CHEMIN_REPLI_HNSW
    assert (classe._lancer_vectorisation, classe._chunks_semantiques) == avant


@pytest.mark.asyncio
async def test_le_second_leurre_passe_par_le_meme_chemin():
    """Des voisins quelconques, servis comme les tableaux : seule la barriere les ecarte."""
    classe = module_recherche.SearchService
    avant = (classe._lancer_vectorisation, classe._chunks_semantiques)
    quelconques = [_extrait(9, 77, "3", score=0.28)]

    with module.volet_semantique("leurre2", {"q": quelconques}, {"cle": "q"}):
        service = classe.__new__(classe)
        rendu = await service._chunks_semantiques(service._lancer_vectorisation("q"), None, 32, 0)

    assert [c.article_id for c in rendu] == [9]
    assert service.chemin_semantique == module_recherche.CHEMIN_REPLI_HNSW
    assert (classe._lancer_vectorisation, classe._chunks_semantiques) == avant


@pytest.mark.asyncio
async def test_la_panne_leve_l_erreur_du_fournisseur():
    classe = module_recherche.SearchService

    with module.volet_semantique("panne", {}, {"cle": None}):
        service = classe.__new__(classe)
        with pytest.raises(module_recherche.VectorSearchError):
            await service._chunks_semantiques(service._lancer_vectorisation("q"), None, 32, 0)


def test_les_questions_suggerees_sont_celles_du_front():
    """Seize par langue, chacune avec au moins un texte vise."""
    for langue in ("fr", "en"):
        questions = module.SUGGEREES[langue]
        assert len(questions) == 16 == len({q for q, _ in questions})
        assert all(references for _, references in questions)
    assert module.SUGGEREES["fr"][0][0] == module.QUESTION_DE_L_INCIDENT


def test_aucun_appel_au_modele_et_journal_neutralise(monkeypatch):
    """
    La mesure ne declenche aucune generation : re-ranking par le modele coupe,
    et rag.llm retire question par question (voir `contexte`).
    """
    vus = {}

    async def _evaluer(etats, questions_en, limite):
        from app.core.config import settings

        vus["rerank_llm"] = settings.RERANK_LLM_ENABLED
        vus["journal"] = module_recherche.record_search_event
        vus["etats"] = etats
        return {"resultats": {}}

    monkeypatch.setattr(module, "evaluer", _evaluer)
    monkeypatch.setattr(module_recherche, "precharger_les_embeddings", lambda: True)
    monkeypatch.setattr(module_recherche, "record_search_event", module_recherche.record_search_event)
    from app.core.config import settings

    monkeypatch.setattr(settings, "RERANK_LLM_ENABLED", True)
    monkeypatch.setattr(module.moteur, "echo", module.moteur.echo)

    assert module.main(["--etats", "leurre", "normal", "leurre"]) == 0

    assert vus["rerank_llm"] is False
    assert asyncio.iscoroutinefunction(vus["journal"])
    assert vus["journal"].__name__ == "_rien_journaliser"
    assert vus["etats"] == ["leurre", "normal"]


def test_les_jeux_de_tous_les_jours_et_les_relances(monkeypatch):
    """
    Les jeux ajoutes au tour 3 : des questions de tous les jours en francais
    et en anglais, des relances avec leur question precedente, et des
    questions nommees dont le texte attendu est connu, ou rien.
    """
    from scripts.eval import questions_courantes as q

    assert len(q.REELLES["fr"]) >= 100 and len(q.REELLES["en"]) >= 20
    assert all(p and r for paires in q.RELANCES_AVEC_HISTORIQUE.values() for p, r in paires)
    attendus = {ref for qs in q.NOMMEES_REVUE.values() for _, ref in qs}
    assert None in attendus and "LOI-1992-007" in attendus
    assert all(module_recherche.question_budgetaire(x) for xs in q.BUDGETAIRES.values() for x in xs)
    assert not any(module_recherche.question_budgetaire(x) for xs in q.REELLES.values() for x in xs)


def test_les_jeux_du_tour_4():
    """
    Les montants « prévus » ne sont pas budgetaires ; les relances en chaine
    sont toutes reconnues, et aucune question de fond des jeux n'est prise
    pour une relance (sinon elle serait cherchee avec la precedente).
    """
    from app.services.rag_service import est_une_relance
    from scripts.eval import questions_courantes as q

    assert not any(
        module_recherche.question_budgetaire(x) for xs in q.MONTANTS_PREVUS.values() for x in xs
    )
    assert all(est_une_relance(r) for xs in q.RELANCES_EN_CHAINE.values() for _, r, _ in xs)
    assert all(attendus for xs in q.RELANCES_EN_CHAINE.values() for _, _, attendus in xs)
    fond = [x for jeu in (q.REELLES, q.MONTANTS, q.MONTANTS_PREVUS, q.BUDGETAIRES, module.COURTES)
            for xs in jeu.values() for x in xs]
    fond += [x for xs in module.SUGGEREES.values() for x, _ in xs]
    fond += [x for xs in module.HORS_SUJET.values() for x in xs]
    fond += [x for xs in q.NOMMEES_REVUE.values() for x, _ in xs]
    assert not [x for x in fond if est_une_relance(x)]


def test_les_jeux_du_tour_5():
    """
    Les cas de la revue du tour 4 : les changements de sujet sont des
    relances cherchees seules et jointes, avec le texte du nouveau sujet
    attendu ; les secondes chances ne sont pas des relances ; les articles
    nommes avec un renvoi au texte sont des relances sans changement de sujet ;
    les budgets prives avec une annee ne sont pas budgetaires.
    """
    from app.services.rag_service import est_une_relance, peut_changer_de_sujet, renvoi_au_texte
    from scripts.eval import questions_courantes as q

    changements = [x for xs in q.CHANGEMENTS_DE_SUJET.values() for x in xs]
    # Les 13 de sonde_nouveau_sujet et les 3 autres de sonde_detecteurs ; la
    # derniere, quatre mots propres, n'est pas une relance (cherchee seule).
    assert len(changements) == 16
    assert all(attendus for _, _, attendus in changements)
    assert all(
        peut_changer_de_sujet(question) or not est_une_relance(question)
        for _, question, _ in changements
    )
    assert sum(1 for _, question, _ in changements if peut_changer_de_sujet(question)) == 15
    secondes = [question for xs in q.SECONDES_CHANCES.values() for _, question, _ in xs]
    assert secondes and not any(est_une_relance(x) for x in secondes)
    renvois = [r for xs in q.RELANCES_EN_CHAINE.values() for _, r, _ in xs if renvoi_au_texte(r)]
    assert len(renvois) == 10
    assert not any(peut_changer_de_sujet(r) for r in renvois)
    assert "Quel budget pour un mariage en 2024 ?" in q.MONTANTS_PREVUS["fr"]
    assert all(nom.startswith(module.LEGITIMES) for nom in ("changements_de_sujet_fr",))
    assert not "secondes_chances_fr".startswith(module.LEGITIMES)


def test_la_sonde_du_fournisseur_est_reessayee(monkeypatch):
    """
    Reseau instable (nuit du 30/10/2026, ECONNRESET) : la sonde est
    reessayee ; un echec passager laisse partir une mesure qui a son fichier
    de vecteurs, jamais une mesure dont le service est ecarte.
    """
    essais = []
    monkeypatch.setattr(module.time, "sleep", lambda secondes: None)
    monkeypatch.setattr(module_recherche, "_embedding_service_instance", object())
    monkeypatch.setattr(module_recherche, "precharger_les_embeddings",
                        lambda: essais.append(1) or len(essais) == 3)
    assert module._sonder_le_fournisseur(None) is True
    assert len(essais) == 3

    monkeypatch.setattr(module_recherche, "precharger_les_embeddings", lambda: False)
    fichier = Path(module.__file__)
    assert module._sonder_le_fournisseur(None) is False
    assert module._sonder_le_fournisseur(fichier.with_name("absent.json")) is False
    assert module._sonder_le_fournisseur(fichier) is True
    monkeypatch.setattr(module_recherche, "_embedding_service_instance", None)
    assert module._sonder_le_fournisseur(fichier) is False


def test_un_vecteur_est_reessaye(monkeypatch):
    """Une coupure de quelques secondes ne fait pas echouer la mesure."""
    appels = []

    class Fournisseur:
        TASK_QUERY = "requete"

        async def generate_embedding_async(self, texte, task_type=None):
            appels.append(task_type)
            if len(appels) < 3:
                raise ConnectionError("ECONNRESET")
            return [1.0]

    async def _sans_attente(secondes):
        return None

    monkeypatch.setattr(module.asyncio, "sleep", _sans_attente)
    assert asyncio.run(module._vecteur_reessaye(Fournisseur(), "q")) == [1.0]
    assert appels == ["requete"] * 3
    appels.clear()
    with pytest.raises(ConnectionError):
        asyncio.run(module._vecteur_reessaye(Fournisseur(), "q", essais=2))


def test_le_releve_ne_mesure_rien(monkeypatch):
    """--releve relit la table des noms et compte les tableaux, sans vectoriser."""
    vus = {}

    async def _releve():
        vus["releve"] = True
        return {}

    monkeypatch.setattr(module, "releve", _releve)
    monkeypatch.setattr(module_recherche, "precharger_les_embeddings",
                        lambda: (_ for _ in ()).throw(AssertionError("fournisseur appele")))
    monkeypatch.setattr(module_recherche, "record_search_event", module_recherche.record_search_event)
    monkeypatch.setattr(module.moteur, "echo", module.moteur.echo)

    assert module.main(["--releve"]) == 0
    assert vus == {"releve": True}

