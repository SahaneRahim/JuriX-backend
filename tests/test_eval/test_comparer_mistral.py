"""
Les mesures de scripts/eval/comparer_mistral.py, sans appel.

La regle de decision (garder le 14b sauf si le 8b egale la justesse des
citations) repose sur l'extraction des numeros cites : une extraction qui en
rate la moitie fausserait le choix du modele.

Usage:
    pytest tests/test_eval/test_comparer_mistral.py -v
"""

import pytest

from app.services.prompts import get_system_prompt
from scripts.eval.comparer_mistral import (
    REGLES_DE_FOND,
    TEMOIN,
    _parseur,
    appels_prevus,
    citations,
    distinction,
    forme_relue,
    main,
    numeros_cites,
    similarite,
    synthese,
    systeme_pour,
    traits,
    variantes_de,
)


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


# ---------------------------------------------------------------- profils


def test_le_prompt_mesure_est_celui_du_profil():
    assert systeme_pour("actuelles", "avocat") == get_system_prompt("avocat", "fr")
    assert systeme_pour("renforcees", "avocat").startswith(REGLES_DE_FOND)


def test_les_alias_de_profil_sont_acceptes():
    args = _parseur().parse_args(["--personas", "etudiant", "Avocat"])

    assert args.personas == ["étudiant", "avocat"]


def test_un_profil_inconnu_est_refuse_par_le_parseur():
    with pytest.raises(SystemExit):
        _parseur().parse_args(["--personas", "astronaute"])


def test_le_temoin_rejoue_la_premiere_variante():
    args = _parseur().parse_args(["--modeles", "m", "--personas", "citoyen", "avocat", "--temoin"])

    assert variantes_de(args) == [
        ("m", "actuelles", "citoyen"), ("m", "actuelles", "avocat"),
        ("m", "actuelles", "citoyen" + TEMOIN),
    ]


def test_le_budget_refuse_avant_tout_appel(capsys):
    """8 questions x 2 modeles x 5 profils = 80 appels, au-dela des 48 : rien ne part."""
    code = main([
        "--questions", "8", "--comparaisons", "0", "--modeles", "a", "b",
        "--personas", "citoyen", "avocat", "entrepreneur", "étudiant", "citoyen",
    ])

    assert code == 2
    assert "Budget depasse" in capsys.readouterr().out
    assert appels_prevus(8, 10, 0, 2) == 80


def test_un_profil_sans_prompt_n_est_pas_mesure(monkeypatch, capsys):
    """Sinon get_system_prompt se replierait en silence sur un autre profil."""
    from scripts.eval import comparer_mistral

    monkeypatch.setattr(comparer_mistral, "SYSTEM_PROMPTS", {"fr": {"citoyen": "x"}})

    assert main(["--personas", "avocat", "--dry-run"]) == 2
    assert "sans prompt" in capsys.readouterr().out


# ---------------------------------------------------------------- mesures pures


def test_forme_relue_distingue_ce_que_le_chat_sait_relire():
    assert forme_relue("Selon l'article 33 du Code Minier, alinéa 2.") == {"vus": 1, "relus": 1}
    assert forme_relue("Voir art. 33.") == {"vus": 1, "relus": 0}
    assert forme_relue("Aucune citation.") == {"vus": 0, "relus": 0}


def test_traits_d_une_reponse():
    t = traits("1. Premier point\n2. Second point\nConsultez un avocat. La jurisprudence manque.")

    assert t["liste_numerotee"] and not t["puces"]
    assert t["renvoi_avocat"] and t["jurisprudence"]
    assert t["mots"] > 5


def test_similarite_bornes():
    assert similarite("un deux trois quatre", "un deux trois quatre") == 1.0
    assert similarite("un deux trois", "quatre cinq six") == 0.0


def test_distinction_profils_divergents_temoin_proche():
    base = "la loi prevoit une sanction penale pour ce fait precis selon le code"
    autre = "en pratique vous devez deposer un dossier aupres de la mairie competente"
    resultats = {"questions": [{"reponses": [
        {"variante": "m|actuelles|citoyen", "texte": base, "erreur": None},
        {"variante": "m|actuelles|avocat", "texte": autre, "erreur": None},
        {"variante": "m|actuelles|citoyen" + TEMOIN, "texte": base, "erreur": None},
    ]}]}

    bilan = distinction(resultats)

    assert bilan["temoin"] == 1.0
    assert bilan["entre_profils"] < bilan["temoin"]
