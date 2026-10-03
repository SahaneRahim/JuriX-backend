"""
Classement local de l'intention (app/services/intent_local.py).

Deux familles :

- Sur DOUBLURE d'embeddings : les regles de decision — la categorie la plus
  proche, la marge exigee d'une categorie non juridique, les suites de
  question, le garde-fou du corpus, et l'absence d'exception.
- Sur le VRAI EmbeddingGemma, marque `gemma` : le jeu d'evaluation
  tests/fixtures/intent_eval.json, distinct de la banque d'exemples. Aucune
  question de droit ne doit partir en conversation.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.services import intent_local
from app.services.intent_local import ClasseurLocal, question_precedente

# Quatre directions, une par categorie : un texte est un melange de ces axes.
AXES = {"juridique": 0, "smalltalk": 1, "meta": 2, "hors_sujet": 3}


class _Embeddings:
    """Doublure : le vecteur d'un texte est donne par une table, sinon nul."""

    TASK_QUERY = "RETRIEVAL_QUERY"

    def __init__(self, table):
        self.table = table
        self.appels = 0

    def generate_embedding(self, texte, normalize=True, task_type=None):
        self.appels += 1
        return np.array(self.table.get(texte, [0.25, 0.25, 0.25, 0.25]), dtype=np.float32)


def _vecteur(**poids):
    v = [0.0] * 4
    for intention, valeur in poids.items():
        v[AXES[intention]] = valeur
    return v


EXEMPLES = {
    "juridique": ("j1", "j2", "j3"),
    "smalltalk": ("s1", "s2", "s3"),
    "meta": ("m1", "m2", "m3"),
    "hors_sujet": ("h1", "h2", "h3"),
}


def _classeur(table, **kw):
    banque = {t: _vecteur(**{i: 1.0}) for i, textes in EXEMPLES.items() for t in textes}
    return ClasseurLocal(_Embeddings({**banque, **table}), exemples=EXEMPLES, **kw)


class TestRegles:

    def test_la_categorie_la_plus_proche_l_emporte(self):
        c = _classeur({"bonjour toi": _vecteur(smalltalk=1.0, juridique=0.2)})
        assert c.classer("bonjour toi")[0] == "smalltalk"

    def test_question_de_droit(self):
        c = _classeur({"mon patron ne me paie plus": _vecteur(juridique=1.0, smalltalk=0.3)})
        intention, _, regle = c.classer("mon patron ne me paie plus")
        assert (intention, regle) == ("juridique", "local")

    def test_dans_le_doute_c_est_une_question_de_droit(self):
        """Une categorie non juridique doit devancer « juridique » d'une marge."""
        c = _classeur({"message ambigu": _vecteur(hors_sujet=0.71, juridique=0.70)})
        assert c.classer("message ambigu")[:3:2] == ("juridique", "local-doute")

    def test_une_suite_de_question_de_droit(self):
        c = _classeur({
            "et pour un mineur ?": _vecteur(smalltalk=0.9, juridique=0.3),
            "quelle peine pour un vol ? et pour un mineur ?": _vecteur(juridique=1.0),
        })
        intention, _, regle = c.classer("et pour un mineur ?", precedente="quelle peine pour un vol ?")
        assert (intention, regle) == ("juridique", "local-suite")

    def test_un_long_message_n_est_pas_traite_en_suite(self):
        long = "raconte-moi une longue histoire de pirates sur une ile tropicale au soleil"
        c = _classeur({long: _vecteur(hors_sujet=1.0)})
        assert c.classer(long, precedente="quelle peine pour un vol ?")[0] == "hors_sujet"

    def test_le_garde_fou_du_corpus_rattrape_une_question_de_droit(self):
        c = _classeur(
            {"la chasse est-elle autorisée ?": _vecteur(hors_sujet=1.0, juridique=0.4)},
            similarite_corpus=lambda v: 0.49,
        )
        intention, confiance, regle = c.classer("la chasse est-elle autorisée ?")
        assert (intention, regle) == ("juridique", "local-corpus")
        assert confiance == 0.49

    def test_le_garde_fou_laisse_passer_ce_qui_est_loin_du_corpus(self):
        c = _classeur(
            {"va-t-il pleuvoir ?": _vecteur(hors_sujet=1.0)},
            similarite_corpus=lambda v: 0.38,
        )
        assert c.classer("va-t-il pleuvoir ?")[0] == "hors_sujet"

    def test_garde_fou_indisponible(self):
        c = _classeur({"va-t-il pleuvoir ?": _vecteur(hors_sujet=1.0)}, similarite_corpus=lambda v: None)
        assert c.classer("va-t-il pleuvoir ?")[0] == "hors_sujet"

    def test_le_garde_fou_n_est_pas_consulte_pour_une_question_de_droit(self):
        appels = []
        c = _classeur(
            {"comment créer une SARL ?": _vecteur(juridique=1.0)},
            similarite_corpus=lambda v: appels.append(1) or 0.9,
        )
        c.classer("comment créer une SARL ?")
        assert appels == []

    def test_la_banque_est_vectorisee_une_seule_fois(self):
        c = _classeur({})
        c.classer("a")
        apres_premier = c.service.appels
        c.classer("b")
        assert c.service.appels == apres_premier + 1, "seul le message est vectorise"


class TestQuestionPrecedente:

    def test_derniere_question_de_l_utilisateur(self):
        historique = [
            SimpleNamespace(role="user", content="quelle peine pour un vol ?"),
            SimpleNamespace(role="assistant", content="L'article 318..."),
        ]
        assert question_precedente(historique) == "quelle peine pour un vol ?"

    def test_sans_historique(self):
        assert question_precedente([]) is None
        assert question_precedente(None) is None


class TestBranchement:

    @pytest.mark.asyncio
    async def test_classify_intent_passe_par_le_local_sans_appeler_le_modele(self, monkeypatch):
        from unittest.mock import AsyncMock

        from app.core.config import settings
        from app.services import intent_classifier

        monkeypatch.setattr(settings, "INTENT_CLASSIFIER", "local")
        classeur = _classeur({"raconte une blague": _vecteur(hors_sujet=1.0)})
        monkeypatch.setattr(intent_classifier, "classeur_local", lambda: classeur)
        llm = AsyncMock()

        resultat = await intent_classifier.classify_intent("raconte une blague", llm=llm)

        assert resultat.intent == "hors_sujet"
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_panne_du_local_retombe_sur_juridique(self, monkeypatch):
        from unittest.mock import AsyncMock

        from app.core.config import settings
        from app.services import intent_classifier

        monkeypatch.setattr(settings, "INTENT_CLASSIFIER", "local")

        class _EnPanne:
            def classer(self, *a):
                raise RuntimeError("modele absent")

        monkeypatch.setattr(intent_classifier, "classeur_local", lambda: _EnPanne())
        llm = AsyncMock()

        resultat = await intent_classifier.classify_intent("raconte une blague", llm=llm)

        assert (resultat.intent, resultat.rule) == ("juridique", "defaut-local-erreur")
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_les_court_circuits_passent_avant(self, monkeypatch):
        from unittest.mock import AsyncMock

        from app.core.config import settings
        from app.services import intent_classifier

        monkeypatch.setattr(settings, "INTENT_CLASSIFIER", "local")
        monkeypatch.setattr(intent_classifier, "classeur_local", lambda: None)

        salut = await intent_classifier.classify_intent("Bonjour !", llm=AsyncMock())
        article = await intent_classifier.classify_intent("que dit l'article 33 ?", llm=AsyncMock())

        assert salut.rule == "court-circuit-salutation"
        assert article.rule == "court-circuit-article"


# ==================== VRAI MODELE ====================

JEU = Path(__file__).resolve().parents[1] / "fixtures" / "intent_eval.json"


@pytest.mark.gemma
def test_jeu_d_evaluation_avec_le_vrai_modele():
    """
    Mesure sans garde-fou du corpus (le jeu de test n'a pas de corpus) : c'est
    la banque seule qui est evaluee. Aucune question de droit ne doit partir
    en conversation ; les confusions entre bavardage, hors sujet et questions
    sur l'assistant menent toutes au chemin conversationnel.
    """
    pytest.importorskip("onnxruntime")
    from app.core.config import settings
    from app.services.embedding_service import EmbeddingService

    if not Path(settings.GEMMA_MODEL_DIR).is_dir():
        pytest.skip("modele EmbeddingGemma absent")
    settings.EMBEDDING_PROVIDER = "gemma"
    service = EmbeddingService(use_cache=False)
    classeur = ClasseurLocal(service)
    messages = json.loads(JEU.read_text(encoding="utf-8"))["messages"]

    verdicts = [
        (m, classeur.classer(m["message"], m.get("precedente"))[0]) for m in messages
    ]

    perdues = [m["message"] for m, v in verdicts if m["intention"] == "juridique" and v != "juridique"]
    justes = sum(1 for m, v in verdicts if m["intention"] == v)
    assert perdues == [], f"questions de droit classees en conversation : {perdues}"
    assert justes / len(messages) >= 0.9, f"{justes}/{len(messages)}"
    assert intent_local.K_VOISINS == 3
