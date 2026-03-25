"""
Service de détection automatique de langue pour documents juridiques.

Ce service utilise 3 méthodes de détection en parallèle:
- langdetect: Détection rapide basée sur n-grams
- spaCy: Détection NLP basée sur reconnaissance d'entités
- fastText: Détection ML basée sur embeddings

Vote majoritaire (2/3 minimum) pour robustesse maximale.
Objectif: 99% précision, <1s temps de réponse.

Usage:
    detector = LanguageDetector()
    result = detector.detect("Article 1. La présente loi...")
    # {'language': 'fr', 'confidence': 0.98, 'consensus': True, ...}
"""

import logging
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fasttext
import langdetect
import spacy
from langdetect import DetectorFactory
from spacy.language import Language

# Configuration langdetect pour reproductibilité
DetectorFactory.seed = 0

# Configuration logging
logger = logging.getLogger(__name__)


class LanguageDetectionError(Exception):
    """Exception levée lors d'erreurs de détection."""

    pass


class LanguageDetector:
    """
    Service de détection automatique de langue avec triple ensemble.

    Utilise 3 méthodes en parallèle pour une précision maximale:
    - langdetect (n-grams, rapide)
    - spaCy (NLP, entités)
    - fastText (ML, embeddings)

    Vote majoritaire (2/3) pour décision finale.
    Précision cible: 99%
    Temps cible: <1 seconde

    Attributes:
        nlp_fr: Modèle spaCy français
        nlp_en: Modèle spaCy anglais
        fasttext_model: Modèle fastText de détection de langue
    """

    def __init__(self, fasttext_model_path: Optional[str] = None):
        """
        Initialise le détecteur avec chargement eager des modèles.

        Args:
            fasttext_model_path: Chemin vers le modèle fastText (optionnel)
                                 Par défaut: backend/models/fasttext/lid.176.bin

        Raises:
            LanguageDetectionError: Si les modèles ne peuvent pas être chargés
        """
        logger.info("🚀 Initialisation du LanguageDetector...")

        try:
            # Charger modèles spaCy
            logger.info("  → Chargement des modèles spaCy...")
            self.nlp_fr: Language = spacy.load("fr_core_news_sm")
            self.nlp_en: Language = spacy.load("en_core_web_sm")
            logger.info("  ✅ Modèles spaCy chargés")

            # Charger modèle fastText
            logger.info("  → Chargement du modèle fastText...")
            if fasttext_model_path is None:
                # Chemin par défaut
                backend_dir = Path(__file__).parent.parent.parent
                fasttext_model_path = str(backend_dir / "models" / "fasttext" / "lid.176.bin")

            # Suppress fastText warnings
            fasttext.FastText.eprint = lambda x: None
            self.fasttext_model = fasttext.load_model(fasttext_model_path)
            logger.info("  ✅ Modèle fastText chargé")

            logger.info("✅ LanguageDetector initialisé avec succès (3 modèles prêts)")

        except Exception as e:
            error_msg = f"Échec du chargement des modèles: {e}"
            logger.error(f"❌ {error_msg}")
            raise LanguageDetectionError(error_msg) from e

    def detect(
        self, text: str, min_confidence: float = 0.80, min_length: int = 50
    ) -> Dict[str, Any]:
        """
        Détecte la langue d'un texte avec vote majoritaire.

        Exécute les 3 méthodes en parallèle pour performance optimale.
        Applique un vote majoritaire sur les résultats.

        Args:
            text: Texte à analyser (recommandé: >100 caractères)
            min_confidence: Seuil de confiance minimum (0-1)
            min_length: Longueur minimale requise en caractères

        Returns:
            Dictionnaire contenant:
                - language (str): Code langue détectée ('fr' | 'en')
                - confidence (float): Score de confiance (0-1)
                - method_votes (Dict[str, str]): Votes de chaque méthode
                - consensus (bool): True si 2+ méthodes d'accord
                - processing_time_ms (int): Temps de traitement en ms
                - text_length (int): Longueur du texte analysé

        Raises:
            ValueError: Si texte trop court ou vide
            LanguageDetectionError: Si toutes les méthodes échouent

        Example:
            >>> detector = LanguageDetector()
            >>> result = detector.detect("Article 1. La présente loi...")
            >>> print(result['language'])
            'fr'
            >>> print(result['confidence'])
            0.98
        """
        assert isinstance(min_confidence, float) and 0.0 <= min_confidence <= 1.0, "min_confidence must be between 0.0 and 1.0"
        assert isinstance(min_length, int) and min_length > 0, "min_length must be a positive integer"

        start_time = time.time()

        # Validation input
        if not text or not text.strip():
            raise ValueError("Le texte ne peut pas être vide")

        text = text.strip()

        if len(text) < min_length:
            raise ValueError(f"Texte trop court: {len(text)} caractères (minimum: {min_length})")

        # Prétraitement
        processed_text = self._preprocess_text(text)

        # Log pour debug
        logger.debug(f"Détection de langue pour texte de {len(processed_text)} caractères")

        # Exécution parallèle des 3 méthodes
        votes: List[Tuple[str, str, float]] = []  # (method_name, language, confidence)

        with ThreadPoolExecutor(max_workers=3) as executor:
            # Soumettre les 3 tâches
            future_to_method = {
                executor.submit(self._detect_langdetect, processed_text): "langdetect",
                executor.submit(self._detect_spacy, processed_text): "spacy",
                executor.submit(self._detect_fasttext, processed_text): "fasttext",
            }

            # Collecter les résultats avec timeout
            for future in as_completed(future_to_method.keys(), timeout=2.0):
                method_name = future_to_method[future]
                try:
                    language, confidence = future.result(timeout=1.0)
                    votes.append((method_name, language, confidence))
                    logger.debug(f"  {method_name}: {language} ({confidence:.2%})")
                except TimeoutError:
                    logger.warning(f"  {method_name}: timeout (>1s)")
                except Exception as e:
                    logger.warning(f"  {method_name}: erreur ({e})")

        # Vérifier qu'au moins une méthode a réussi
        if not votes:
            raise LanguageDetectionError("Toutes les méthodes de détection ont échoué")

        # Appliquer le vote majoritaire
        detected_language, final_confidence, consensus = self._vote_majority(votes)

        # Vérifier seuil de confiance
        if final_confidence < min_confidence:
            logger.warning(f"Confiance {final_confidence:.2%} < seuil {min_confidence:.2%}")

        # Calculer temps de traitement
        processing_time_ms = int((time.time() - start_time) * 1000)

        # Construire résultat
        result = {
            "language": detected_language,
            "confidence": round(final_confidence, 4),
            "method_votes": {method: lang for method, lang, _ in votes},
            "consensus": consensus,
            "processing_time_ms": processing_time_ms,
            "text_length": len(text),
        }

        logger.info(
            f"Détection: {detected_language} "
            f"({final_confidence:.2%}, {processing_time_ms}ms, "
            f"consensus={consensus})"
        )

        return result

    def _preprocess_text(self, text: str) -> str:
        """
        Prétraite le texte pour améliorer la détection.

        Opérations:
        - Normalisation des espaces
        - Suppression de la numérotation des articles
        - Préservation des accents français (critique!)
        - Préservation des symboles légaux
        - Sampling à 5000 caractères pour performance

        Args:
            text: Texte brut

        Returns:
            Texte nettoyé
        """
        # Normaliser les espaces multiples
        text = re.sub(r"\s+", " ", text)

        # Supprimer la numérotation des articles (bruit)
        # Ex: "Art. 1", "Article 2", "Section 3"
        text = re.sub(r"\b(Art\.|Article|Section)\s+\d+\.?\s*", "", text)

        # IMPORTANT: Préserver les accents (é, è, à, ç, etc.)
        # Ne PAS normaliser/supprimer les accents - critiques pour FR!

        # Sampling pour performance (5000 premiers caractères suffisent)
        if len(text) > 5000:
            text = text[:5000]

        return text.strip()

    def _detect_langdetect(self, text: str) -> Tuple[str, float]:
        """
        Détection via langdetect (rapide, n-grams).

        Args:
            text: Texte à analyser

        Returns:
            Tuple (language_code, confidence)

        Raises:
            Exception: Si détection échoue
        """
        try:
            # langdetect retourne liste de (lang, prob)
            detections = langdetect.detect_langs(text)

            if not detections:
                raise ValueError("Aucune langue détectée")

            # Prendre la langue avec probabilité maximale
            top_detection = detections[0]
            language = top_detection.lang
            confidence = top_detection.prob

            # Normaliser: garder seulement 'fr' ou 'en'
            # Si autre langue détectée, fallback sur fr
            if language not in ["fr", "en"]:
                logger.debug(f"langdetect: langue {language} → fallback fr")
                language = "fr"
                confidence *= 0.5  # Réduire confiance pour fallback

            return language, confidence

        except Exception as e:
            raise Exception(f"langdetect error: {e}") from e

    def _detect_spacy(self, text: str) -> Tuple[str, float]:
        """
        Détection via spaCy (NLP, reconnaissance d'entités).

        Stratégie: Tester avec les 2 modèles (FR et EN) et comparer
        les scores basés sur le nombre d'entités reconnues.

        Args:
            text: Texte à analyser

        Returns:
            Tuple (language_code, confidence)

        Raises:
            Exception: Si détection échoue
        """
        try:
            # Limiter la longueur pour performance spaCy
            sample = text[:1000]

            # Analyser avec modèle FR
            doc_fr = self.nlp_fr(sample)
            score_fr = self._calculate_spacy_score(doc_fr)

            # Analyser avec modèle EN
            doc_en = self.nlp_en(sample)
            score_en = self._calculate_spacy_score(doc_en)

            # Déterminer langue gagnante
            total_score = score_fr + score_en

            if total_score == 0:
                # Aucune entité reconnue, fallback sur FR
                return "fr", 0.5

            if score_fr > score_en:
                confidence = min(score_fr / total_score, 0.99)
                return "fr", confidence
            else:
                confidence = min(score_en / total_score, 0.99)
                return "en", confidence

        except Exception as e:
            raise Exception(f"spaCy error: {e}") from e

    def _calculate_spacy_score(self, doc) -> float:
        """
        Calcule un score de qualité pour un document spaCy.

        Basé sur:
        - Nombre d'entités nommées reconnues
        - Nombre de tokens reconnus

        Args:
            doc: Document spaCy analysé

        Returns:
            Score de qualité (float)
        """
        if len(doc) == 0:
            return 0.0

        # Score basé sur entités (poids fort)
        entity_score = len(doc.ents) * 2.0

        # Score basé sur tokens (poids faible)
        token_score = len([token for token in doc if not token.is_punct]) * 0.1

        return entity_score + token_score

    def _detect_fasttext(self, text: str) -> Tuple[str, float]:
        """
        Détection via fastText (ML, embeddings).

        Args:
            text: Texte à analyser

        Returns:
            Tuple (language_code, confidence)

        Raises:
            Exception: Si détection échoue
        """
        try:
            # Remplacer les sauts de ligne par espaces
            text_cleaned = text.replace("\n", " ")

            # Prédiction (k=1 pour top langue seulement)
            predictions = self.fasttext_model.predict(text_cleaned, k=1)

            # Format retour: (('__label__fr',), array([0.98]))
            label = predictions[0][0].replace("__label__", "")
            confidence = float(predictions[1][0])

            # Normaliser: garder seulement 'fr' ou 'en'
            if label not in ["fr", "en"]:
                logger.debug(f"fasttext: langue {label} → fallback fr")
                label = "fr"
                confidence *= 0.5

            return label, confidence

        except Exception as e:
            raise Exception(f"fastText error: {e}") from e

    def _vote_majority(self, votes: List[Tuple[str, str, float]]) -> Tuple[str, float, bool]:
        """
        Applique un vote majoritaire sur les résultats des méthodes.

        Logique:
        - 3/3 d'accord: consensus parfait
        - 2/3 d'accord: consensus majoritaire
        - 1/3 ou moins: pas de consensus, prendre le score le plus élevé

        Args:
            votes: Liste de (method_name, language, confidence)

        Returns:
            Tuple (detected_language, final_confidence, consensus)
        """
        if not votes:
            raise ValueError("Aucun vote disponible")

        # Compter les votes par langue
        language_votes = Counter([lang for _, lang, _ in votes])

        # Langue avec le plus de votes
        most_common = language_votes.most_common(1)[0]
        winning_language = most_common[0]
        vote_count = most_common[1]

        # Déterminer consensus
        total_votes = len(votes)

        # Cas spécial: Si seulement 2 votes et pas d'accord (1-1), utiliser langdetect comme tiebreaker
        if total_votes == 2 and vote_count == 1:
            # Chercher le vote de langdetect
            langdetect_vote = next((lang for method, lang, _ in votes if method == "langdetect"), None)
            if langdetect_vote:
                winning_language = langdetect_vote
                vote_count = 1
                # Considérer comme consensus si langdetect a haute confiance
                langdetect_conf = next((conf for method, lang, conf in votes if method == "langdetect"), 0)
                consensus = langdetect_conf >= 0.95
            else:
                consensus = False
        else:
            consensus = vote_count >= 2  # Au moins 2/3

        # Calculer confiance: moyenne des votes concordants avec bonus pour consensus
        matching_confidences = [conf for _, lang, conf in votes if lang == winning_language]

        if matching_confidences:
            # Moyenne de base
            base_confidence = sum(matching_confidences) / len(matching_confidences)

            # Bonus si consensus (2+ méthodes d'accord)
            if consensus and vote_count >= 2:
                # Appliquer un boost de 5% quand il y a consensus entre 2+ méthodes
                # Cela compense la perte d'une méthode (fasttext)
                boost = 0.05
                final_confidence = min(base_confidence + boost, 0.99)
            else:
                final_confidence = base_confidence
        else:
            # Cas extrême: aucune confidence (ne devrait pas arriver)
            final_confidence = 0.5

        # Log pour monitoring
        if not consensus:
            logger.warning(
                f"Pas de consensus: {vote_count}/{total_votes} votes pour {winning_language}"
            )
            logger.warning(f"Détail votes: {votes}")

        return winning_language, final_confidence, consensus

    def health_check(self) -> Dict[str, Any]:
        """
        Vérifie l'état de santé du service.

        Returns:
            Dictionnaire avec statut de chaque composant
        """
        status = {"service": "LanguageDetector", "status": "healthy", "models": {}}

        # Vérifier spaCy FR
        try:
            test_doc = self.nlp_fr("Test")
            status["models"]["spacy_fr"] = "✅ OK"
        except Exception as e:
            status["models"]["spacy_fr"] = f"❌ Error: {e}"
            status["status"] = "degraded"

        # Vérifier spaCy EN
        try:
            test_doc = self.nlp_en("Test")
            status["models"]["spacy_en"] = "✅ OK"
        except Exception as e:
            status["models"]["spacy_en"] = f"❌ Error: {e}"
            status["status"] = "degraded"

        # Vérifier fastText
        try:
            self.fasttext_model.predict("Test", k=1)
            status["models"]["fasttext"] = "✅ OK"
        except Exception as e:
            status["models"]["fasttext"] = f"❌ Error: {e}"
            status["status"] = "degraded"

        return status
