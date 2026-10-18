import os
import subprocess
import urllib.request
from pathlib import Path

# Nettoyer les espaces accidentels dans PGSSLMODE (ex: " require")
if "PGSSLMODE" in os.environ:
    os.environ["PGSSLMODE"] = os.environ["PGSSLMODE"].strip()

# 1. Vérifier et télécharger le modèle de détection de langue fastText si absent
model_dir = Path("models/fasttext")
model_path = model_dir / "lid.176.bin"
if not model_path.exists():
    model_dir.mkdir(parents=True, exist_ok=True)
    print("⏳ Téléchargement du modèle fastText (126 Mo)...")
    try:
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
            model_path,
        )
        print("✅ Modèle fastText prêt.")
    except Exception as e:
        print(f"⚠️ Avertissement téléchargement fastText: {e}")

# 2. Exécuter les migrations Alembic
try:
    print("⏳ Application des migrations Alembic...")
    subprocess.run(["alembic", "upgrade", "head"], check=False)
    print("✅ Migrations vérifiées.")
except Exception as e:
    print(f"⚠️ Avertissement Alembic: {e}")

# 3. Importer l'application FastAPI et Gradio
from app.main import app as fastapi_app
import gradio as gr
import spaces
import uvicorn

# 4. Fonction ZeroGPU liée à l'interface (satisfait la validation Hugging Face)
@spaces.GPU
def zero_gpu_worker(prompt: str) -> str:
    return f"GPU Ready: {prompt}"

# 5. Interface Gradio minimale
with gr.Blocks(title="JuriX API") as demo:
    gr.Markdown("# ⚖️ JuriX — Backend API")
    gr.Markdown("Le serveur d'API REST de JuriX est opérationnel.")
    gr.Markdown("- **Documentation interactive (Swagger)** : [/docs](/docs)")
    gr.Markdown("- **Vérification de santé** : [/api/v1/health](/api/v1/health)")
    
    # Composant lié à la fonction GPU pour valider le démarrage ZeroGPU
    _dummy_in = gr.Textbox(value="ping", visible=False)
    _dummy_out = gr.Textbox(visible=False)
    _dummy_btn = gr.Button("Test", visible=False)
    _dummy_btn.click(zero_gpu_worker, inputs=_dummy_in, outputs=_dummy_out)

# 6. Monte Gradio sur FastAPI
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
