"""
scripts/eval/mesurer_latence_recherche.py mesure la recherche exacte du chat,
seule et sous charge. Ce qui doit tenir : des statistiques justes, et une
mesure qui ne lit que les vecteurs du fichier (aucun appel au fournisseur) et
n'ecrit rien (journal des recherches neutralise).
"""

import json

from app.services import search_service as module_recherche
from scripts.eval import mesurer_latence_recherche as module


def test_statistiques():
    assert module._stats([]) == {"n": 0, "mediane": None, "p90": None, "max": None}
    assert module._stats([30.0, 10.0, 20.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]) == {
        "n": 10, "mediane": 55.0, "p90": 100.0, "max": 100.0,
    }


def test_les_questions_viennent_du_fichier_des_vecteurs(tmp_path, monkeypatch):
    jeu = json.loads(module.JEU.read_text(encoding="utf-8"))["items"]
    connues = [it["question"] for it in jeu[:3]]
    fichier = tmp_path / "vecteurs.json"
    fichier.write_text(json.dumps({q: [0.1, 0.2] for q in connues}), encoding="utf-8")
    vus = {}

    async def _mesurer(questions, vecteurs):
        vus["questions"], vus["journal"] = questions, module_recherche.record_search_event
        return {}

    monkeypatch.setattr(module, "mesurer", _mesurer)
    monkeypatch.setattr(module_recherche, "record_search_event", module_recherche.record_search_event)
    monkeypatch.setattr(module_recherche, "_init_global_singletons", lambda: None)
    monkeypatch.setattr(module.moteur, "echo", module.moteur.echo)

    assert module.main(["--vecteurs", str(fichier), "--questions", "2"]) == 0

    assert vus["questions"] == connues[:2]
    assert vus["journal"].__name__ == "_rien_journaliser"
