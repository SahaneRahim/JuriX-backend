"""
Les reperes d'un texte : numero d'acte, nature, dates de l'acte.

Le front compose « modifié par la loi n° 2026/002 du 14 avril 2026 » dans la
langue de l'interface, a partir de ces pieces. Un numero mal lu y ecrirait le
numero d'un AUTRE acte : les intitules ci-dessous sont ceux du corpus.

Usage:
    pytest tests/test_services/test_reperes_textes.py -v
"""

from datetime import date
from types import SimpleNamespace

import pytest

from app.services.prompts import _date_du_texte
from app.services.reperes_textes import (
    DatesDeLActe,
    dates_de_l_acte,
    intitule_sans_article,
    nature_depuis_intitule,
    numero_de_l_acte,
)


@pytest.mark.parametrize("titre,numero", [
    ("Décret N°2021/745 du 28 décembre 2021 fixant les modalités d'exercice", "2021/745"),
    ("Loi N° 2017/015 du 12 juillet 2017 modifiant et complétant certaines dispositions de la loi "
     "N° 2013/004 du 18 avril 2013", "2017/015"),
    ("DECRET N° 2013/297 DU 0 9 SEP. 2013 modifiant et complétant", "2013/297"),
    ("Arrêté N°0827/CAB/PR du 19 novembre 2019 modifiant", "0827/CAB/PR"),
    ("Loi N°69/LF/18 du 10 novembre 1969 instituant un régime d'assurance", "69/LF/18"),
    ("Loi N°67-LF-7 du 12 juin 1967 instituant un Code des prestations familiales", "67-LF-7"),
    ("Décret No. 2020/556 du 14 septembre 2020 ratifiant l'Accord de Prêt N° 2000200004657", "2020/556"),
    ("Arrêté N°0967 du 11 novembre 2024 portant nomination", "0967"),
    ("Loi 2019/023 du 24 décembre 2019 portant loi de finances", "2019/023"),
    ("Décret N°2017/591 04 décembre 2017 modifiant et complétant", "2017/591"),
    ("la Loi N°83-19 du 26 novembre 1983", "83-19"),
    ("l'Ordonnance N°77/2 du 10 janvier 1977", "77/2"),
    ("Loi-cadre N°2011/012 du 06 mai 2011 portant protection du consommateur", "2011/012"),
])
def test_le_numero_est_celui_de_l_acte(titre, numero):
    assert numero_de_l_acte(titre) == numero


def test_le_numero_d_un_acte_cite_dans_l_objet_n_est_pas_retenu():
    """« Décret N°82/241 … modifiant … le décret n°77/245 » : le sien seulement."""
    titre = (
        "Décret N°82/241 du 24 juin 1982 modifiant et complétant certaines dispositions "
        "du décret n°77/245 du 15 juillet 1977 portant organisation des chefferies traditionnelles."
    )
    assert numero_de_l_acte(titre) == "82/241"
    assert numero_de_l_acte("Décret portant nomination du décret n° 2019/001") is None


@pytest.mark.parametrize("titre", [
    "Acte uniforme révisé portant organisation des sûretés du 15 décembre 2010",
    "Code civil applicable au Cameroun, annoté de la jurisprudence de la Cour suprême",
    "LOI N° DU MODIFIANT ET COMPLETANT",
    "",
    None,
])
def test_un_intitule_sans_numero_n_en_donne_aucun(titre):
    assert numero_de_l_acte(titre) is None


@pytest.mark.parametrize("titre,nature", [
    ("la Loi n° 2012/015 du 21 décembre 2012", "loi"),
    ("l'Ordonnance N°77/2 du 10 janvier 1977", "ordonnance"),
    ("le Décret N°2005/481 du 16 décembre 2005", "décret"),
    ("DECRET N° 2013/297", "décret"),
    ("l'Acte uniforme révisé portant organisation des sûretés", "acte uniforme"),
    ("Loi-cadre N°2011/012", "loi"),
    ("le Traité portant révision du Traité relatif à l'harmonisation", "autre"),
    ("Lettre circulaire", "autre"),
])
def test_la_nature_se_lit_sur_le_premier_mot(titre, nature):
    """Les valeurs sont celles de laws.type : un texte absent se decrit comme un texte du corpus."""
    assert nature_depuis_intitule(titre) == nature


def test_l_intitule_du_releve_perd_son_article_defini():
    assert intitule_sans_article("la Loi n° 2012/015 du 21 décembre 2012") == (
        "Loi n° 2012/015 du 21 décembre 2012"
    )
    assert intitule_sans_article("l'Ordonnance N°77/2") == "Ordonnance N°77/2"
    assert intitule_sans_article("Lettre circulaire") == "Lettre circulaire"


@pytest.mark.parametrize("jour,titre,attendu", [
    (date(2021, 12, 28), "Décret N°2021/745 du 28 décembre 2021",
     DatesDeLActe(date(2021, 12, 28), 2021, None)),
    # 1er janvier de remplissage : l'intitule ne donne ni jour ni mois.
    (date(2024, 1, 1), "Décret N°2024/193 juin 10", DatesDeLActe(None, 2024, None)),
    (date(2024, 1, 1), "Décret N°2024/001 du 1er janvier 2024",
     DatesDeLActe(date(2024, 1, 1), 2024, None)),
    (date(2024, 1, 1), "DECRET N° 2024/002 DU 1ER JANVIER 2024",
     DatesDeLActe(date(2024, 1, 1), 2024, None)),
    (None, "Code civil applicable au Cameroun", DatesDeLActe(None, None, None)),
])
def test_les_dates_de_l_acte(jour, titre, attendu):
    assert dates_de_l_acte(jour, titre) == attendu


def test_une_date_de_journal_officiel_n_est_pas_une_signature():
    """Seule date de l'intitule de deux Actes uniformes OHADA du corpus : « Signé le … » serait faux."""
    titre = (
        "Acte uniforme portant organisation des procédures simplifiées de recouvrement et des voies "
        "d'exécution, révisé, publié au Journal officiel de l'OHADA du 15 novembre 2023"
    )

    assert dates_de_l_acte(date(2023, 11, 15), titre) == DatesDeLActe(None, 2023, date(2023, 11, 15))


@pytest.mark.parametrize("jour,titre", [
    (date(2021, 12, 28), "Décret N°2021/745 du 28 décembre 2021"),
    (date(2024, 1, 1), "Décret N°2024/193 juin 10"),
    (date(2024, 1, 1), "Décret N°2024/001 du 1er janvier 2024"),
])
def test_la_page_et_le_chat_disent_la_meme_precision(jour, titre):
    """Le chat ne donne que l'annee au modele exactement quand la page n'a pas de jour."""
    extrait = SimpleNamespace(publication_date=jour, law_title=titre)
    chat_annee_seule = "jour non indiqué" in _date_du_texte(extrait)

    assert chat_annee_seule == (dates_de_l_acte(jour, titre).date_signature is None)
