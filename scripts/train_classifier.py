"""
Script d'entraînement du modèle ML pour DocumentClassifier.

Entraîne un modèle TF-IDF + Logistic Regression pour classification
des documents juridiques en 12 catégories.

Architecture:
- TF-IDF Vectorizer (max_features=5000, ngram_range=(1,2))
- Logistic Regression (max_iter=1000, class_weight='balanced')

Pipeline:
1. Charger données (data/training/legal_categories.json)
2. Split train/test 80/20 (stratifié)
3. Vectorisation TF-IDF
4. Entraînement Logistic Regression
5. Évaluation (accuracy, precision, recall, F1)
6. Sauvegarde modèles (.pkl)
7. Génération rapport

Usage:
    python scripts/train_classifier.py

Options:
    --input: Chemin du dataset (défaut: data/training/legal_categories.json)
    --output-dir: Répertoire de sortie (défaut: models/)
    --test-size: Proportion test set (défaut: 0.2)
    --max-features: Features TF-IDF max (défaut: 5000)
    --seed: Seed random (défaut: 42)
"""

import json
import argparse
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
import sys

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
import numpy as np

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))


class ClassifierTrainer:
    """Entraîneur de modèle pour DocumentClassifier."""

    def __init__(
        self,
        max_features: int = 5000,
        ngram_range: Tuple[int, int] = (1, 2),
        max_iter: int = 1000,
        random_state: int = 42
    ):
        """
        Initialise le trainer.

        Args:
            max_features: Nombre max de features TF-IDF
            ngram_range: Range des n-grams (1,2) = unigrams + bigrams
            max_iter: Itérations max pour LogReg
            random_state: Seed pour reproductibilité
        """
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_iter = max_iter
        self.random_state = random_state

        self.vectorizer = None
        self.model = None
        self.training_report = {}

    def load_data(self, data_path: Path) -> Tuple[List[str], List[int]]:
        """
        Charge le dataset d'entraînement.

        Args:
            data_path: Chemin vers legal_categories.json

        Returns:
            (texts, labels)
        """
        print(f"📂 Chargement des données depuis {data_path}...")

        with open(data_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        texts = [sample['text'] for sample in dataset]
        labels = [sample['category_id'] for sample in dataset]

        print(f"✅ {len(texts)} échantillons chargés")
        print(f"   - Catégories: {len(set(labels))}")
        print(f"   - Longueur moyenne: {np.mean([len(t) for t in texts]):.0f} caractères")

        return texts, labels

    def prepare_data(
        self,
        texts: List[str],
        labels: List[int],
        test_size: float = 0.2
    ) -> Tuple[List[str], List[str], List[int], List[int]]:
        """
        Prépare les données (split train/test).

        Args:
            texts: Textes
            labels: Labels
            test_size: Proportion du test set

        Returns:
            (X_train, X_test, y_train, y_test)
        """
        print(f"\n📊 Split train/test ({int((1-test_size)*100)}/{int(test_size*100)})...")

        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=test_size,
            random_state=self.random_state,
            stratify=labels  # Maintenir distribution des classes
        )

        print(f"✅ Train: {len(X_train)} échantillons")
        print(f"✅ Test:  {len(X_test)} échantillons")

        # Vérifier distribution
        unique_train, counts_train = np.unique(y_train, return_counts=True)
        print(f"\n   Distribution train: {dict(zip(unique_train, counts_train))}")

        return X_train, X_test, y_train, y_test

    def train_vectorizer(self, X_train: List[str]) -> np.ndarray:
        """
        Entraîne le vectorizer TF-IDF.

        Args:
            X_train: Textes d'entraînement

        Returns:
            Matrice TF-IDF train
        """
        print(f"\n🔤 Vectorisation TF-IDF...")
        print(f"   - max_features: {self.max_features}")
        print(f"   - ngram_range: {self.ngram_range}")

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=2,  # Ignorer termes très rares
            max_df=0.95,  # Ignorer termes trop fréquents
            strip_accents=None,  # Préserver accents français
            lowercase=True,
            analyzer='word',
            token_pattern=r'\b\w+\b'
        )

        start_time = time.time()
        X_train_vec = self.vectorizer.fit_transform(X_train)
        elapsed = time.time() - start_time

        print(f"✅ Vectorisation terminée en {elapsed:.2f}s")
        print(f"   - Vocabulaire: {len(self.vectorizer.vocabulary_)} termes")
        print(f"   - Matrice shape: {X_train_vec.shape}")
        print(f"   - Sparsity: {(1 - X_train_vec.nnz / (X_train_vec.shape[0] * X_train_vec.shape[1]))*100:.1f}%")

        return X_train_vec

    def train_model(
        self,
        X_train_vec: np.ndarray,
        y_train: List[int]
    ):
        """
        Entraîne le modèle Logistic Regression.

        Args:
            X_train_vec: Matrice TF-IDF train
            y_train: Labels train
        """
        print(f"\n🤖 Entraînement Logistic Regression...")
        print(f"   - max_iter: {self.max_iter}")
        print(f"   - class_weight: balanced")

        self.model = LogisticRegression(
            max_iter=self.max_iter,
            class_weight='balanced',  # Gérer déséquilibre classes
            random_state=self.random_state,
            solver='lbfgs',  # Efficace pour multi-class
            verbose=0
        )

        start_time = time.time()
        self.model.fit(X_train_vec, y_train)
        elapsed = time.time() - start_time

        print(f"✅ Entraînement terminé en {elapsed:.2f}s")
        print(f"   - Classes: {len(self.model.classes_)}")
        print(f"   - Convergence: {'Oui' if self.model.n_iter_[0] < self.max_iter else 'Non (max iter atteint)'}")

        # Sauvegarde temps entraînement
        self.training_report['training_time_sec'] = elapsed

    def evaluate(
        self,
        X_test: List[str],
        y_test: List[int]
    ) -> Dict[str, Any]:
        """
        Évalue le modèle sur le test set.

        Args:
            X_test: Textes test
            y_test: Labels test

        Returns:
            Dictionnaire de métriques
        """
        print(f"\n📈 Évaluation sur test set...")

        # Vectorisation test
        X_test_vec = self.vectorizer.transform(X_test)

        # Prédictions
        y_pred = self.model.predict(X_test_vec)
        y_pred_proba = self.model.predict_proba(X_test_vec)

        # Métriques principales
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        print(f"\n✅ Résultats:")
        print(f"   - Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"   - Precision: {precision:.4f}")
        print(f"   - Recall:    {recall:.4f}")
        print(f"   - F1-Score:  {f1:.4f}")

        # Top-3 accuracy
        top3_correct = 0
        for i, true_label in enumerate(y_test):
            top3_pred = np.argsort(y_pred_proba[i])[-3:][::-1]
            top3_labels = [self.model.classes_[idx] for idx in top3_pred]
            if true_label in top3_labels:
                top3_correct += 1

        top3_accuracy = top3_correct / len(y_test)
        print(f"   - Top-3 Acc: {top3_accuracy:.4f} ({top3_accuracy*100:.2f}%)")

        # Rapport détaillé
        print(f"\n📊 Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        # Matrice de confusion
        conf_matrix = confusion_matrix(y_test, y_pred)
        print(f"\n📋 Confusion Matrix:")
        print(conf_matrix)

        # Sauvegarder métriques
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'top3_accuracy': float(top3_accuracy),
            'classification_report': classification_report(y_test, y_pred, output_dict=True, zero_division=0),
            'confusion_matrix': conf_matrix.tolist()
        }

        return metrics

    def cross_validate(
        self,
        X_train: List[str],
        y_train: List[int],
        cv: int = 5
    ):
        """
        Cross-validation 5-fold.

        Args:
            X_train: Textes train
            y_train: Labels train
            cv: Nombre de folds
        """
        print(f"\n🔄 Cross-Validation ({cv}-fold)...")

        X_train_vec = self.vectorizer.transform(X_train)

        scores = cross_val_score(
            self.model,
            X_train_vec,
            y_train,
            cv=cv,
            scoring='accuracy',
            n_jobs=-1
        )

        print(f"✅ CV Scores: {scores}")
        print(f"   - Mean: {scores.mean():.4f} ({scores.mean()*100:.2f}%)")
        print(f"   - Std:  {scores.std():.4f}")

        self.training_report['cv_scores'] = scores.tolist()
        self.training_report['cv_mean'] = float(scores.mean())
        self.training_report['cv_std'] = float(scores.std())

    def save_models(self, output_dir: Path):
        """
        Sauvegarde les modèles entraînés.

        Args:
            output_dir: Répertoire de sortie
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n💾 Sauvegarde des modèles dans {output_dir}...")

        # Sauvegarder vectorizer
        vectorizer_path = output_dir / "category_vectorizer.pkl"
        joblib.dump(self.vectorizer, vectorizer_path)
        vec_size = vectorizer_path.stat().st_size / 1024 / 1024  # MB
        print(f"✅ Vectorizer: {vectorizer_path.name} ({vec_size:.2f} MB)")

        # Sauvegarder modèle
        model_path = output_dir / "category_classifier.pkl"
        joblib.dump(self.model, model_path)
        model_size = model_path.stat().st_size / 1024 / 1024  # MB
        print(f"✅ Classifier: {model_path.name} ({model_size:.2f} MB)")

        # Sauvegarder rapport
        report_path = output_dir / "training_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_report, f, ensure_ascii=False, indent=2)
        print(f"✅ Report: {report_path.name}")

        print(f"\n📦 Modèles totaux: {vec_size + model_size:.2f} MB")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Entraîne le modèle ML pour DocumentClassifier"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/training/legal_categories.json',
        help="Chemin du dataset (défaut: data/training/legal_categories.json)"
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models',
        help="Répertoire de sortie (défaut: models/)"
    )
    parser.add_argument(
        '--test-size',
        type=float,
        default=0.2,
        help="Proportion test set (défaut: 0.2)"
    )
    parser.add_argument(
        '--max-features',
        type=int,
        default=5000,
        help="Features TF-IDF max (défaut: 5000)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Seed random (défaut: 42)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("🤖 ENTRAÎNEMENT DOCUMENTCLASSIFIER ML")
    print("=" * 70)

    # Initialiser trainer
    trainer = ClassifierTrainer(
        max_features=args.max_features,
        ngram_range=(1, 2),
        max_iter=1000,
        random_state=args.seed
    )

    # 1. Charger données
    data_path = Path(args.input)
    texts, labels = trainer.load_data(data_path)

    # 2. Préparer données
    X_train, X_test, y_train, y_test = trainer.prepare_data(
        texts,
        labels,
        test_size=args.test_size
    )

    # 3. Vectorisation
    X_train_vec = trainer.train_vectorizer(X_train)

    # 4. Entraînement
    trainer.train_model(X_train_vec, y_train)

    # 5. Évaluation
    metrics = trainer.evaluate(X_test, y_test)
    trainer.training_report['test_metrics'] = metrics

    # 6. Cross-validation
    trainer.cross_validate(X_train, y_train, cv=5)

    # 7. Sauvegarder
    output_dir = Path(args.output_dir)
    trainer.save_models(output_dir)

    # Résumé final
    print("\n" + "=" * 70)
    print("🎯 ENTRAÎNEMENT TERMINÉ")
    print("=" * 70)
    print(f"✅ Accuracy test: {metrics['accuracy']*100:.2f}%")
    print(f"✅ Top-3 accuracy: {metrics['top3_accuracy']*100:.2f}%")
    print(f"✅ CV mean: {trainer.training_report['cv_mean']*100:.2f}%")
    print(f"✅ Modèles sauvegardés dans: {output_dir}")

    # Vérifier objectifs
    if metrics['accuracy'] >= 0.80:
        print("\n🎉 Objectif atteint: Accuracy ≥80%")
    else:
        print(f"\n⚠️  Objectif non atteint: {metrics['accuracy']*100:.2f}% < 80%")

    if metrics['top3_accuracy'] >= 0.95:
        print("🎉 Objectif atteint: Top-3 Accuracy ≥95%")
    else:
        print(f"⚠️  Top-3: {metrics['top3_accuracy']*100:.2f}% < 95%")


if __name__ == "__main__":
    main()
