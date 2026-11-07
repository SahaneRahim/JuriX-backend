"""
En-tetes d'article colles dans un paragraphe (app/utils/text_chunker.py).

Cas reel : la loi 92/007 portant Code du travail. Son texte extrait met
« Article 34.- » et « Article 36.- » au milieu d'une ligne : ces articles
restaient fondus dans les morceaux 33, 35.1 et 35.2.
"""
from app.utils.text_chunker import extract_articles


def _numeros(texte: str):
    return [c["number"] for c in extract_articles(texte, min_article_length=1, language="fr")]


CODE_DU_TRAVAIL = (
    "Article 33.- (1) L'employeur verse une indemnite egale au salaire.\n"
    "(3) Les conditions sont fixees par arrete du ministre du travail. "
    "Article 34.- (1) Le contrat a duree indeterminee peut toujours etre resilie.\n\n"
    "Article 35.- (1) Pendant le preavis, les parties restent tenues. "
    "Article 36.- (1) Toute rupture sans preavis oblige a verser une indemnite. "
    "Article 37.- (1) Le travailleur a droit a une indemnite de licenciement.\n\n"
    "Article 38.- La suite du texte.\n"
)


def test_un_en_tete_colle_dans_un_paragraphe_ouvre_son_article():
    assert _numeros(CODE_DU_TRAVAIL) == ["33", "34", "35", "36", "37", "38"]


def test_le_texte_d_un_article_ne_contient_plus_le_suivant():
    morceaux = {c["number"]: c["content"] for c in extract_articles(CODE_DU_TRAVAIL, language="fr")}
    assert "Article 36" not in morceaux["35"]
    assert morceaux["36"].lstrip(" .-").startswith("(1) Toute rupture")


def test_autres_separateurs_du_corpus():
    texte = (
        "Article 683 — La Cour est competente. Article 684 — (1) Le condamne adresse la demande. "
        "Article 685 — En vue de l'instruction.\nArticle 686 — Le Procureur saisit la Cour.\n"
    )
    assert _numeros(texte) == ["683", "684", "685", "686"]
    assert _numeros("Art. 30. Declaration.\nTexte. Art. 31. Lorsque l'enfant est ne.\nArt. 32. Fin.\n") == [
        "30", "31", "32"]


def test_un_renvoi_ne_coupe_pas_l_article():
    texte = (
        "Article 2.- Les peines de l'article 3.- ci-dessus sont doublees, "
        "selon Article 3 : la loi s'applique.\n"
        "Article 3.- Fin du texte.\n"
    )
    assert _numeros(texte) == ["2", "3"]


def test_une_citation_hors_de_la_suite_reste_dans_son_article():
    # Loi modificative : l'article 2 reproduit l'article 5 d'une autre loi.
    texte = (
        "Article 1er.- Objet de la loi.\n"
        "Article 2.- Les dispositions sont modifiees. Article 5.- (nouveau) Le texte cite.\n"
        "Article 3.- Fin.\n"
    )
    assert _numeros(texte) == ["1", "2", "3"]


def test_un_numero_du_code_cima_n_est_pas_un_en_tete():
    texte = "Article 333.- Regles.\nVoir aussi. Article 333-8 Sanctions.\nArticle 334.- Fin.\n"
    assert _numeros(texte) == ["333", "334"]


def test_une_ligne_de_tableau_n_est_pas_coupee():
    texte = "Article 1.- Sommaire.\n| Texte. Article 2.- Objet | 12 |\nArticle 3.- Fin.\n"
    assert _numeros(texte) == ["1", "3"]
