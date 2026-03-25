"""
Script de téléchargement et vérification des modèles NLP.

Ce script télécharge:
- spaCy models: fr_core_news_sm, en_core_web_sm
- fastText model: lid.176.bin

Usage:
    python scripts/download_models.py
"""

import os
import subprocess
import sys
import urllib.request
from pathlib import Path

# Chemins des modèles
BACKEND_DIR = Path(__file__).parent.parent
MODELS_DIR = BACKEND_DIR / "models"
SPACY_DIR = MODELS_DIR / "spacy"
FASTTEXT_DIR = MODELS_DIR / "fasttext"

# URL fastText
FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
FASTTEXT_PATH = FASTTEXT_DIR / "lid.176.bin"

# Modèles spaCy
SPACY_MODELS = ["fr_core_news_sm", "en_core_web_sm"]


def create_directories():
    """Crée les répertoires nécessaires."""
    print("📁 Création des répertoires...")
    SPACY_DIR.mkdir(parents=True, exist_ok=True)
    FASTTEXT_DIR.mkdir(parents=True, exist_ok=True)
    print("✅ Répertoires créés")


def download_spacy_models():
    """Télécharge les modèles spaCy."""
    print("\n🔽 Téléchargement des modèles spaCy...")

    for model_name in SPACY_MODELS:
        print(f"\n  → Téléchargement de {model_name}...")
        try:
            # Download model
            result = subprocess.run(
                [sys.executable, "-m", "spacy", "download", model_name],
                capture_output=True,
                text=True,
                check=True,
            )
            print(f"  ✅ {model_name} téléchargé avec succès")

        except subprocess.CalledProcessError as e:
            print(f"  ❌ Erreur lors du téléchargement de {model_name}")
            print(f"     {e.stderr}")
            return False

    return True


def verify_spacy_models():
    """Vérifie que les modèles spaCy sont utilisables."""
    print("\n🔍 Vérification des modèles spaCy...")

    try:
        import spacy

        for model_name in SPACY_MODELS:
            print(f"  → Chargement de {model_name}...")
            try:
                nlp = spacy.load(model_name)
                # Test simple
                doc = nlp("Test")
                print(f"  ✅ {model_name} fonctionne correctement")
            except Exception as e:
                print(f"  ❌ Erreur avec {model_name}: {e}")
                return False

        return True

    except ImportError:
        print("  ❌ spaCy n'est pas installé")
        print("     Exécutez: poetry install")
        return False


def download_fasttext_model():
    """Télécharge le modèle fastText."""
    print("\n🔽 Téléchargement du modèle fastText...")

    if FASTTEXT_PATH.exists():
        print(f"  ℹ️  Modèle déjà présent: {FASTTEXT_PATH}")
        return True

    try:
        print(f"  → Téléchargement depuis {FASTTEXT_URL}")
        print(f"     (Fichier ~130MB, cela peut prendre quelques minutes...)")

        def progress_hook(count, block_size, total_size):
            """Affiche la progression du téléchargement."""
            percent = int(count * block_size * 100 / total_size)
            sys.stdout.write(f"\r     Progression: {percent}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(FASTTEXT_URL, FASTTEXT_PATH, reporthook=progress_hook)

        print("\n  ✅ Modèle fastText téléchargé avec succès")
        return True

    except Exception as e:
        print(f"\n  ❌ Erreur lors du téléchargement: {e}")
        print("     Essayez manuellement:")
        print(f"     wget {FASTTEXT_URL} -O {FASTTEXT_PATH}")
        return False


def verify_fasttext_model():
    """Vérifie que le modèle fastText est utilisable."""
    print("\n🔍 Vérification du modèle fastText...")

    if not FASTTEXT_PATH.exists():
        print(f"  ❌ Fichier non trouvé: {FASTTEXT_PATH}")
        return False

    try:
        import fasttext

        print(f"  → Chargement de {FASTTEXT_PATH}...")
        model = fasttext.load_model(str(FASTTEXT_PATH))

        # Test simple
        prediction = model.predict("This is a test")
        print(f"  ✅ Modèle fastText fonctionne correctement")
        print(f"     Test: {prediction}")

        return True

    except ImportError:
        print("  ❌ fasttext n'est pas installé")
        print("     Exécutez: poetry install")
        return False
    except Exception as e:
        print(f"  ❌ Erreur lors du chargement: {e}")
        return False


def display_summary():
    """Affiche un résumé des modèles installés."""
    print("\n" + "=" * 60)
    print("📊 RÉSUMÉ DES MODÈLES INSTALLÉS")
    print("=" * 60)

    # spaCy
    print("\n🔤 Modèles spaCy:")
    try:
        import spacy

        for model_name in SPACY_MODELS:
            try:
                nlp = spacy.load(model_name)
                print(f"  ✅ {model_name}")
            except:
                print(f"  ❌ {model_name}")
    except:
        print("  ❌ spaCy non disponible")

    # fastText
    print("\n🚀 Modèle fastText:")
    if FASTTEXT_PATH.exists():
        size_mb = FASTTEXT_PATH.stat().st_size / (1024 * 1024)
        print(f"  ✅ lid.176.bin ({size_mb:.1f} MB)")
    else:
        print(f"  ❌ lid.176.bin")

    # Espace disque
    total_size = 0
    if MODELS_DIR.exists():
        for file in MODELS_DIR.rglob("*"):
            if file.is_file():
                total_size += file.stat().st_size

    print(f"\n💾 Espace total utilisé: {total_size / (1024 * 1024):.1f} MB")
    print("=" * 60)


def main():
    """Fonction principale."""
    print("🚀 JuriX - Téléchargement des modèles NLP")
    print("=" * 60)

    # Étape 1: Créer les répertoires
    create_directories()

    # Étape 2: Télécharger spaCy
    if not download_spacy_models():
        print("\n❌ Échec du téléchargement des modèles spaCy")
        sys.exit(1)

    # Étape 3: Vérifier spaCy
    if not verify_spacy_models():
        print("\n❌ Les modèles spaCy ne fonctionnent pas correctement")
        sys.exit(1)

    # Étape 4: Télécharger fastText
    if not download_fasttext_model():
        print("\n❌ Échec du téléchargement du modèle fastText")
        sys.exit(1)

    # Étape 5: Vérifier fastText
    if not verify_fasttext_model():
        print("\n❌ Le modèle fastText ne fonctionne pas correctement")
        sys.exit(1)

    # Résumé
    display_summary()

    print("\n✅ Tous les modèles sont prêts à l'emploi!")
    print("\n💡 Prochaine étape:")
    print("   python -m pytest backend/tests/test_services/test_language_detector.py")


if __name__ == "__main__":
    main()
