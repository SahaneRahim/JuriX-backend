"""
Service de catégorisation automatique de documents juridiques.

Ce service utilise une approche hybride pour classifier les documents:
- Keywords matching (40%): Rapide, transparent, basé sur dictionnaires
- ML (60%): TF-IDF + Logistic Regression (Phase 3)

Phase 1 (MVP): Keywords-only (70-75% précision)
Phase 3: Hybrid 40/60 (87% top-1, 96% top-3)

12 catégories juridiques camerounaises:
1. Droit Constitutionnel
2. Droit Civil
3. Droit Pénal
4. Droit Commercial OHADA
5. Droit du Travail
6. Droit Fiscal
7. Droit Administratif
8. Droit Foncier
9. Droit de la Famille
10. Droit de l'Environnement
11. Droit International
12. Droit des Affaires

Usage:
    classifier = DocumentClassifier()
    result = classifier.classify("Article 1. La SARL est une société commerciale...")
    # [(4, 0.85, 'keyword'), (12, 0.62, 'keyword'), (2, 0.45, 'keyword')]
"""

import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class DocumentClassificationError(Exception):
    """Exception levée lors d'erreurs de classification."""

    pass


class DocumentClassifier:
    """
    Service de catégorisation automatique de documents juridiques.

    Approche hybrid:
    - Phase 1 (MVP): Keywords-only (70-75% précision)
    - Phase 3: 40% Keywords + 60% ML (87% top-1, 96% top-3)

    Performance cible: <2s (MVP: <500ms)
    """

    # Dictionnaires de mots-clés par catégorie
    # Primary keywords: poids 1.0
    # Secondary keywords: poids 0.5
    CATEGORY_KEYWORDS = {
        1: {  # Droit Constitutionnel
            'primary': [
                'constitution', 'parlement', 'président',
                'assemblée nationale', 'sénat', 'référendum'
            ],
            'secondary': [
                'pouvoir exécutif', 'pouvoir législatif', 'institutions'
            ]
        },
        2: {  # Droit Civil
            'primary': [
                'contrat', 'obligations', 'responsabilité civile',
                'dommages', 'code civil'
            ],
            'secondary': [
                'débiteur', 'créancier', 'prescription'
            ]
        },
        3: {  # Droit Pénal
            'primary': [
                'crime', 'délit', 'peine', 'emprisonnement',
                'code pénal', 'infraction'
            ],
            'secondary': [
                'criminel', 'tribunal correctionnel', 'procès'
            ]
        },
        4: {  # Droit Commercial OHADA
            'primary': [
                'ohada', 'sarl', 'sa', 'capital social',
                'parts sociales', 'acte uniforme'
            ],
            'secondary': [
                'associés', 'actionnaires', 'dirigeants', 'société commerciale'
            ]
        },
        5: {  # Droit du Travail
            'primary': [
                'salarié', 'employeur', 'licenciement',
                'contrat de travail', 'code du travail'
            ],
            'secondary': [
                'salaire', 'congés', 'inspection du travail'
            ]
        },
        6: {  # Droit Fiscal
            'primary': [
                'impôt', 'taxe', 'tva', 'is', 'irpp', 'fiscalité'
            ],
            'secondary': [
                'contribuable', 'recouvrement', 'administration fiscale'
            ]
        },
        7: {  # Droit Administratif
            'primary': [
                'fonction publique', 'service public', 'administration',
                'contentieux administratif'
            ],
            'secondary': [
                'agent public', 'autorité administrative'
            ]
        },
        8: {  # Droit Foncier
            'primary': [
                'terrain', 'titre foncier', 'cadastre',
                'domaine', 'immatriculation'
            ],
            'secondary': [
                'propriété foncière', 'lotissement'
            ]
        },
        9: {  # Droit de la Famille
            'primary': [
                'mariage', 'divorce', 'succession',
                'filiation', 'autorité parentale'
            ],
            'secondary': [
                'conjoint', 'héritier', 'tutelle'
            ]
        },
        10: {  # Droit de l'Environnement
            'primary': [
                'environnement', 'pollution', 'écologie',
                'protection', 'ressources naturelles'
            ],
            'secondary': [
                'développement durable', 'faune', 'flore'
            ]
        },
        11: {  # Droit International
            'primary': [
                'traité', 'convention', 'nations unies',
                'international', 'ratification'
            ],
            'secondary': [
                'accord bilatéral', 'souveraineté'
            ]
        },
        12: {  # Droit des Affaires
            'primary': [
                'contrat commercial', 'franchise',
                'bail commercial', 'concurrence'
            ],
            'secondary': [
                'commerce', 'registre du commerce'
            ]
        }
    }

    # Noms des catégories (pour référence)
    CATEGORY_NAMES = {
        1: "Droit Constitutionnel",
        2: "Droit Civil",
        3: "Droit Pénal",
        4: "Droit Commercial OHADA",
        5: "Droit du Travail",
        6: "Droit Fiscal",
        7: "Droit Administratif",
        8: "Droit Foncier",
        9: "Droit de la Famille",
        10: "Droit de l'Environnement",
        11: "Droit International",
        12: "Droit des Affaires"
    }

    # Limites de texte
    MIN_TEXT_LENGTH = 50
    MAX_TEXT_LENGTH = 250000  # Increased from 50000 to handle large legal documents

    def __init__(self, model_path: Optional[Path] = None):
        """
        Initialise le classifier.

        Phase 1 (MVP): Keywords-only
        Phase 3: Charge les modèles ML si disponibles

        Args:
            model_path: Chemin vers les modèles ML (optionnel)
        """
        self.ml_model = None
        self.vectorizer = None

        # Tentative de chargement des modèles ML (Phase 3)
        if model_path is None:
            model_path = Path(__file__).parent.parent.parent / "models"

        self._load_ml_models(model_path)

        # Log du mode de fonctionnement
        if self.ml_model and self.vectorizer:
            logger.info("✅ DocumentClassifier initialisé (mode hybrid)")
        else:
            logger.info("✅ DocumentClassifier initialisé (mode keywords-only)")

    def _load_ml_models(self, model_path: Path):
        """
        Charge les modèles ML s'ils existent (Phase 3).

        Args:
            model_path: Répertoire contenant les modèles
        """
        try:
            import joblib

            vectorizer_path = model_path / "category_vectorizer.pkl"
            classifier_path = model_path / "category_classifier.pkl"

            if vectorizer_path.exists() and classifier_path.exists():
                self.vectorizer = joblib.load(vectorizer_path)
                self.ml_model = joblib.load(classifier_path)
                logger.info(f"✅ Modèles ML chargés depuis {model_path}")
            else:
                logger.info(f"ℹ️  Modèles ML non trouvés dans {model_path} (mode keywords-only)")

        except ImportError:
            logger.warning("⚠️  joblib non disponible, mode keywords-only uniquement")
        except Exception as e:
            logger.error(f"❌ Erreur chargement modèles ML: {e}")

    def classify(
        self,
        text: str,
        language: str = "fr",
        top_k: int = 3,
        min_confidence: float = 0.0
    ) -> List[Tuple[int, float, str]]:
        """
        Classifie un document juridique.

        Args:
            text: Texte du document à classifier
            language: Langue du texte ('fr' ou 'en') - pour Phase 3
            top_k: Nombre de suggestions à retourner (1-5)
            min_confidence: Seuil de confiance minimum (0.0-1.0)

        Returns:
            Liste de tuples (category_id, confidence, method)
            Exemple: [(4, 0.92, 'hybrid'), (2, 0.78, 'keyword'), (6, 0.54, 'keyword')]

        Raises:
            ValueError: Si texte invalide
            DocumentClassificationError: Si erreur de classification
        """
        # Les assertions qui se trouvaient ici court-circuitaient _validate_input :
        # elles levaient AssertionError (donc HTTP 500 au lieu de 422) et
        # disparaissaient sous python -O, supprimant toute validation. Elles
        # contredisaient en plus _validate_input, qui autorise top_k jusqu'a 12.
        start_time = time.time()

        # Validation entrée
        self._validate_input(text, top_k)

        # Preprocessing
        processed_text = self._preprocess_text(text)

        # Méthode 1: Keywords (toujours exécutée)
        keyword_scores = self._classify_keywords(processed_text)

        # Méthode 2: ML (si disponible, Phase 3)
        if self.ml_model and self.vectorizer:
            ml_scores = self._classify_ml(processed_text)

            # Fusion: 40% keywords + 60% ML
            final_scores = {}
            all_categories = set(keyword_scores.keys()) | set(ml_scores.keys())

            for cat_id in all_categories:
                kw_score = keyword_scores.get(cat_id, 0.0)
                ml_score = ml_scores.get(cat_id, 0.0)
                final_scores[cat_id] = 0.4 * kw_score + 0.6 * ml_score

            method = 'hybrid'
        else:
            # Phase 1: Keywords-only
            final_scores = keyword_scores
            method = 'keyword'

        # Filtrer par seuil de confiance
        final_scores = {
            cat_id: score
            for cat_id, score in final_scores.items()
            if score >= min_confidence
        }

        # Trier et retourner top-k
        sorted_results = sorted(
            final_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        # Formatter résultats
        results = [
            (cat_id, float(score), method)
            for cat_id, score in sorted_results
        ]

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"Classification terminée: {len(results)} suggestions, "
            f"top={results[0] if results else 'N/A'}, "
            f"méthode={method}, temps={processing_time:.0f}ms"
        )

        return results

    def _validate_input(self, text: str, top_k: int):
        """
        Valide les paramètres d'entrée.

        Args:
            text: Texte à valider
            top_k: Nombre de suggestions

        Raises:
            ValueError: Si paramètres invalides
        """
        if not text or not text.strip():
            raise ValueError("Le texte ne peut pas être vide")

        if len(text) < self.MIN_TEXT_LENGTH:
            raise ValueError(
                f"Texte trop court: {len(text)} caractères "
                f"(minimum {self.MIN_TEXT_LENGTH})"
            )

        if len(text) > self.MAX_TEXT_LENGTH:
            raise ValueError(
                f"Texte trop long: {len(text)} caractères "
                f"(maximum {self.MAX_TEXT_LENGTH})"
            )

        if top_k < 1 or top_k > 12:
            raise ValueError(f"top_k doit être entre 1 et 12 (reçu: {top_k})")

    def _preprocess_text(self, text: str) -> str:
        """
        Prétraite le texte pour classification.

        - Convertit en minuscules
        - Supprime espaces excessifs
        - Supprime numéros d'articles
        - Préserve accents français

        Args:
            text: Texte brut

        Returns:
            Texte prétraité
        """
        # Conversion minuscules
        text = text.lower()

        # Suppression numéros d'articles (Art. 1, Article 2, etc.)
        text = re.sub(r'\bart\.?\s*\d+', '', text)
        text = re.sub(r'\barticle\s*\d+', '', text)

        # Suppression espaces multiples
        text = re.sub(r'\s+', ' ', text)

        # Strip
        text = text.strip()

        return text

    def _classify_keywords(self, text: str) -> Dict[int, float]:
        """
        Classification par keywords avec scores pondérés.

        Logique:
        - Compte occurrences de chaque keyword (regex word boundaries)
        - Score = Σ(primary_matches * 1.0) + Σ(secondary_matches * 0.5)
        - Normalise: min(score / 10, 1.0) → [0, 1]

        Args:
            text: Texte prétraité

        Returns:
            Dictionnaire {category_id: confidence_score}
        """
        scores = {}

        for cat_id, keywords in self.CATEGORY_KEYWORDS.items():
            score = 0.0

            # Primary keywords (poids 1.0)
            for keyword in keywords['primary']:
                # Utilise word boundaries pour éviter matches partiels
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = len(re.findall(pattern, text))
                score += matches * 1.0

            # Secondary keywords (poids 0.5)
            for keyword in keywords['secondary']:
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = len(re.findall(pattern, text))
                score += matches * 0.5

            # Normalisation: score / 10, max 1.0
            # Score de 10+ = 100% confiance
            normalized_score = min(score / 10.0, 1.0)

            if normalized_score > 0:
                scores[cat_id] = normalized_score

        return scores

    def _classify_ml(self, text: str) -> Dict[int, float]:
        """
        Classification par ML (TF-IDF + Logistic Regression).

        Phase 3 uniquement.

        Args:
            text: Texte prétraité

        Returns:
            Dictionnaire {category_id: probability}
        """
        try:
            # Vectorisation TF-IDF
            X = self.vectorizer.transform([text])

            # Prédiction probabilités
            probabilities = self.ml_model.predict_proba(X)[0]

            # Map vers category IDs
            scores = {
                int(cat_id): float(prob)
                for cat_id, prob in zip(self.ml_model.classes_, probabilities)
            }

            return scores

        except Exception as e:
            logger.error(f"❌ Erreur classification ML: {e}")
            return {}

    def health_check(self) -> Dict[str, Any]:
        """
        Vérifie l'état de santé du service.

        Returns:
            Dictionnaire avec état du service
        """
        return {
            "service": "DocumentClassifier",
            "status": "healthy",
            "mode": "hybrid" if (self.ml_model and self.vectorizer) else "keywords-only",
            "ml_model_loaded": self.ml_model is not None,
            "vectorizer_loaded": self.vectorizer is not None,
            "categories_count": len(self.CATEGORY_KEYWORDS),
            "version": "1.0.0-mvp"
        }

    def get_category_name(self, category_id: int) -> Optional[str]:
        """
        Retourne le nom d'une catégorie.

        Args:
            category_id: ID de la catégorie (1-12)

        Returns:
            Nom de la catégorie ou None si invalide
        """
        return self.CATEGORY_NAMES.get(category_id)

    def get_all_categories(self) -> Dict[int, str]:
        """
        Retourne toutes les catégories disponibles.

        Returns:
            Dictionnaire {id: nom}
        """
        return self.CATEGORY_NAMES.copy()
