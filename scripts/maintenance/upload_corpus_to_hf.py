"""
Upload du corpus de documents (PDF) vers le dataset Hugging Face.

Ce script :
1. Lit le token HF et les paramètres depuis .env
2. Compare les fichiers locaux de data/uploads avec ceux déjà sur Hugging Face
3. Envoie les fichiers manquants par lots (commits atomiques de 100 fichiers)
4. Affiche une barre de progression claire et reprend automatiquement en cas d'interruption
"""

import os
import sys
import time
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd
from dotenv import load_dotenv

REPO_ID = "rahim237cm/jurix-documents"
REPO_TYPE = "dataset"
BATCH_SIZE = 120  # ~250-300 Mo par lot, taille idéale pour un commit fiable

def main():
    backend_dir = Path(__file__).resolve().parent.parent.parent
    load_dotenv(backend_dir / ".env")

    token = os.getenv("huggingface_token") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HF_TOKEN")
    if not token:
        print("❌ Erreur : aucun token Hugging Face trouvé dans .env (variable huggingface_token)")
        sys.exit(1)

    uploads_dir = backend_dir / "data" / "uploads"
    if not uploads_dir.is_dir():
        print(f"❌ Erreur : le dossier {uploads_dir} n'existe pas.")
        sys.exit(1)

    local_files = sorted(list(uploads_dir.glob("*.pdf")))
    total_local = len(local_files)
    total_size_mb = sum(f.stat().st_size for f in local_files) / (1024 * 1024)
    print(f"📁 Fichiers locaux trouvés : {total_local} PDF ({total_size_mb:.1f} Mo)")

    api = HfApi(token=token)

    # Vérification identité
    try:
        user = api.whoami()
        print(f"👤 Connecté à Hugging Face sous le compte : {user.get('name')}")
    except Exception as e:
        print(f"❌ Erreur d'authentification HF avec le token : {e}")
        sys.exit(1)

    # Récupération des fichiers déjà sur le repo pour reprise automatique
    print(f"🔍 Vérification des fichiers déjà présents sur {REPO_ID}...")
    try:
        remote_files = set(api.list_repo_files(repo_id=REPO_ID, repo_type=REPO_TYPE))
    except Exception as e:
        print(f"⚠️ Impossible de lister les fichiers distants (le repo est peut-être vide) : {e}")
        remote_files = set()

    to_upload = [f for f in local_files if f.name not in remote_files]
    already_done = total_local - len(to_upload)

    print(f"ℹ️ {already_done} fichiers déjà en ligne.")
    print(f"🚀 Fichiers restant à envoyer : {len(to_upload)} ({sum(f.stat().st_size for f in to_upload)/(1024*1024):.1f} Mo)")

    if not to_upload:
        print("🎉 Tous les fichiers sont déjà synchronisés sur Hugging Face !")
        return

    # Découpage par lots
    batches = [to_upload[i:i + BATCH_SIZE] for i in range(0, len(to_upload), BATCH_SIZE)]
    total_batches = len(batches)

    print(f"\n📦 Début de l'envoi en {total_batches} lots de ~{BATCH_SIZE} fichiers...\n")

    start_time = time.time()
    uploaded_count = already_done

    for idx, batch in enumerate(batches, 1):
        batch_size_mb = sum(f.stat().st_size for f in batch) / (1024 * 1024)
        print(f"➡️ [Lot {idx}/{total_batches}] Préparation de {len(batch)} fichiers ({batch_size_mb:.1f} Mo)...")

        operations = [
            CommitOperationAdd(path_in_repo=f.name, path_or_fileobj=str(f))
            for f in batch
        ]

        # Retry logic pour robustesse réseau
        max_retries = 3
        for attempt in range(1, max_retries + 1):
            try:
                commit_info = api.create_commit(
                    repo_id=REPO_ID,
                    repo_type=REPO_TYPE,
                    operations=operations,
                    commit_message=f"Ajout corpus lot {idx}/{total_batches} ({len(batch)} fichiers)",
                )
                uploaded_count += len(batch)
                elapsed = time.time() - start_time
                percent = (uploaded_count / total_local) * 100
                print(f"   ✅ Lot {idx}/{total_batches} validé ! Progression globale : {uploaded_count}/{total_local} ({percent:.1f}%) [écoulé: {elapsed/60:.1f} min]")
                break
            except Exception as exc:
                print(f"   ⚠️ Échec de l'envoi du lot {idx} (tentative {attempt}/{max_retries}) : {exc}")
                if attempt < max_retries:
                    time.sleep(5 * attempt)
                else:
                    print(f"❌ Erreur critique sur le lot {idx}. Vous pouvez relancer le script pour reprendre.")
                    raise

    total_elapsed = time.time() - start_time
    print(f"\n🎉 Terminé ! {total_local} fichiers envoyés vers {REPO_ID} en {total_elapsed/60:.1f} minutes.")

if __name__ == "__main__":
    main()
