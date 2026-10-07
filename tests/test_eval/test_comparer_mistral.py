"""
Les mesures de scripts/eval/comparer_mistral.py, sans appel.

La regle de decision (garder le 14b sauf si le 8b egale la justesse des
citations) repose sur l'extraction des numeros cites : une extraction qui en
rate la moitie fausserait le choix du modele.

Usage:
    pytest tests/test_eval/test_comparer_mistral.py -v
"""

from scripts.eval.comparer_mistral import citations, numeros_cites, synthese


def test_numeros_cites_sous_toutes_leurs_formes():
    texte = (
        "Selon l'article 32 et les articles 40.1 et 41, ainsi que l'art. 1er "
        "et l'article premier ; voir l'Article 5 bis. Le décret 2018/420 n'est pas un article."
    )

    assert numeros_cites(texte) == ["32", "40.1", "41", "1", "5 BIS"]


def test_aucune_citation():
    assert numeros_cites("Je ne trouve pas cette information.") == []


def test_citations_presentes_et_absentes():
    resultat = citations("L'article 33 et l'article 99 le prévoient.", {"32", "33"})

    assert resultat == {"cites": ["33", "99"], "presents": ["33"], "absents": ["99"]}


def test_synthese_justesse_des_citations():
    def reponse(modele, cites, presents, premier=1.0, duree=5.0, fin="STOP"):
        return {
            "modele": modele, "fin": fin, "erreur": None, "premier_morceau_s": premier,
            "duree_s": duree, "jetons_sortie": 100,
            "citations": {"cites": cites, "presents": presents, "absents": []},
        }

    resultats = {
        "questions": [
            {"reponses": [reponse("ministral-14b-latest", ["1", "2"], ["1", "2"]),
                          reponse("ministral-8b-latest", ["1", "2"], ["1"])]},
        ],
        "comparaisons": [
            {"reponses": [{"modele": "ministral-14b-latest", "json": True, "lignes_remplies": 7},
                          {"modele": "ministral-8b-latest", "json": False}]},
        ],
    }

    bilan = synthese(resultats)

    assert bilan["ministral-14b-latest"]["justesse_citations"] == 1.0
    assert bilan["ministral-8b-latest"]["justesse_citations"] == 0.5
    assert bilan["ministral-14b-latest"]["comparaisons_7_sur_7"] == 1
    assert bilan["ministral-8b-latest"]["comparaisons_json"] == 0
