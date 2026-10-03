"""
Classement LOCAL de l'intention : par proximite avec des exemples, sans Gemini.

POURQUOI. Le classement par Gemini coutait un appel par question : 3,6 s
mesurees en reflexion « minimal », 8,5 s par defaut, et une requete sur deux
du quota gratuit. Le modele d'embeddings, EmbeddingGemma, tourne deja en
local pour la recherche : rapprocher le message d'exemples etiquetes
(intent_examples.py) coute quelques dizaines de millisecondes et aucun appel.

LA REGLE. Chaque categorie recoit la moyenne de ses K_VOISINS exemples les plus
proches. Une categorie autre que « juridique » ne l'emporte qu'avec une MARGE
sur « juridique » : dans le doute, c'est une question de droit. L'erreur n'est
pas symetrique (voir intent_classifier.classify_intent) : une salutation
traitee en question de droit coute une reponse trop serieuse ; une question de
droit traitee en conversation coute une reponse sans source, que l'utilisateur
ne peut pas reperer.

LES SUITES. « et pour un fonctionnaire ? » n'est juridique que par la question
precedente. Un message COURT qui n'est pas clairement juridique est donc
reclasse accole a la question precedente de l'utilisateur.

LE GARDE-FOU DU CORPUS. Un verdict non juridique est encore contredit si un
article du corpus est tres proche du message : la banque ne peut pas couvrir
tout le droit. Mesure sur le jeu d'evaluation : le message non juridique le
plus proche d'un article est a 0,38 (« va-t-il pleuvoir a Yaounde ce
week-end ? »), et « la chasse est-elle autorisee en saison des pluies ? »,
que la banque classait hors sujet, est a 0,49 d'un article sur la chasse.

Les seuils sont regles sur tests/fixtures/intent_eval.json, jeu distinct de la
banque d'exemples ; le test marque `gemma` les verifie avec le vrai modele.
"""

import logging
import threading
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from app.services.intent_examples import EXEMPLES

logger = logging.getLogger(__name__)

INTENT_JURIDIQUE = "juridique"

# Voisins moyennes par categorie : un seul voisin rend le verdict sensible a
# un exemple isole ; trop, et les petites categories sont diluees.
K_VOISINS = 3

# Avance minimale d'une categorie non juridique sur « juridique » pour
# l'emporter. Reglee sur le jeu d'evaluation (voir intent_local et le test
# `gemma`).
MARGE_NON_JURIDIQUE = 0.04

# Un message de ce nombre de mots ou moins, apres une question de
# l'utilisateur, peut en etre la suite.
MOTS_D_UNE_SUITE = 8

# Similarite (cosinus) avec l'article le plus proche au-dela de laquelle un
# message est une question de droit, quoi qu'en dise la banque. Entre le
# message non juridique le plus proche (0,38) et la question rattrapee (0,49).
SEUIL_CORPUS = 0.42


class ClasseurLocal:
    """
    Classe un message par proximite avec la banque d'exemples.

    `service` est un EmbeddingService (ou une doublure) : seul compte
    `generate_embedding(texte, normalize, task_type)`. La banque est vectorisee
    une fois, a la premiere utilisation, sous verrou.
    """

    def __init__(
        self,
        service,
        exemples: Dict[str, Tuple[str, ...]] = EXEMPLES,
        similarite_corpus: Optional[Callable[[np.ndarray], Optional[float]]] = None,
    ):
        self.service = service
        self.exemples = exemples
        # Similarite du message avec l'article le plus proche du corpus ;
        # None = pas de garde-fou (tests, harnais sans base).
        self.similarite_corpus = similarite_corpus
        self._banque: Optional[Dict[str, np.ndarray]] = None
        self._verrou = threading.Lock()

    def _vecteur(self, texte: str) -> np.ndarray:
        v = np.asarray(
            self.service.generate_embedding(texte, True, self.service.TASK_QUERY),
            dtype=np.float32,
        )
        norme = float(np.linalg.norm(v))
        return v / norme if norme > 0 else v

    def _vecteurs(self, textes: List[str]) -> np.ndarray:
        """Plusieurs textes d'un coup : le fournisseur les encode par lots."""
        fournisseur = getattr(self.service, "provider", None)
        if fournisseur is None:
            return np.stack([self._vecteur(t) for t in textes])
        matrice = np.stack([
            np.asarray(v, dtype=np.float32)
            for v in fournisseur.embed(list(textes), self.service.TASK_QUERY,
                                       self.service.EMBEDDING_DIM)
        ])
        normes = np.linalg.norm(matrice, axis=1, keepdims=True)
        return matrice / np.where(normes > 0, normes, 1.0)

    def banque(self) -> Dict[str, np.ndarray]:
        """Vecteurs des exemples, par categorie, calcules une seule fois."""
        if self._banque is None:
            with self._verrou:
                if self._banque is None:
                    self._banque = {
                        intention: self._vecteurs(list(textes))
                        for intention, textes in self.exemples.items()
                    }
                    logger.info(
                        "🧭 Banque d'intentions vectorisee : %s exemples",
                        sum(len(t) for t in self.exemples.values()),
                    )
        return self._banque

    def scores(self, texte: str) -> Dict[str, float]:
        """Moyenne des K_VOISINS similarites les plus fortes, par categorie."""
        return self._scores_de(self._vecteur(texte))

    def _scores_de(self, v: np.ndarray) -> Dict[str, float]:
        resultat = {}
        for intention, matrice in self.banque().items():
            similarites = np.sort(matrice @ v)[::-1]
            resultat[intention] = float(similarites[:K_VOISINS].mean())
        return resultat

    def classer(self, question: str, precedente: Optional[str] = None) -> Tuple[str, float, str]:
        """
        Returns:
            (intention, confiance, regle). La confiance est l'avance de la
            categorie retenue ; la regle nomme ce qui a decide.
        """
        vecteur = self._vecteur(question)
        scores = self._scores_de(vecteur)
        meilleure = max(scores, key=scores.get)
        avance = scores[meilleure] - scores[INTENT_JURIDIQUE]

        if meilleure == INTENT_JURIDIQUE:
            seconde = max(v for k, v in scores.items() if k != INTENT_JURIDIQUE)
            return INTENT_JURIDIQUE, round(scores[INTENT_JURIDIQUE] - seconde, 3), "local"

        if precedente and len(question.split()) <= MOTS_D_UNE_SUITE:
            suite = self.scores(f"{precedente} {question}")
            if max(suite, key=suite.get) == INTENT_JURIDIQUE:
                return INTENT_JURIDIQUE, 0.0, "local-suite"

        if avance < MARGE_NON_JURIDIQUE:
            return INTENT_JURIDIQUE, 0.0, "local-doute"

        if self.similarite_corpus is not None:
            proche = self.similarite_corpus(vecteur)
            if proche is not None and proche >= SEUIL_CORPUS:
                return INTENT_JURIDIQUE, round(proche, 3), "local-corpus"
        return meilleure, round(avance, 3), "local"


def question_precedente(history: Optional[List]) -> Optional[str]:
    """Derniere question de l'utilisateur dans l'historique, s'il y en a une."""
    for message in reversed(history or []):
        if getattr(message, "role", None) == "user" and getattr(message, "content", None):
            return message.content
    return None
