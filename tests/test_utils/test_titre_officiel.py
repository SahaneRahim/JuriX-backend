"""
Objet, numero et date lus dans l'intitule officiel d'un texte.

Les intitules viennent tous de la base de developpement (jurix_dev, releve du
28/10/2026) : chaque regle y est prise sur un cas reel, fautes de saisie
comprises. tests/fixtures/titres_objets.json fige en plus 118 intitules tires
au hasard et relus a la main ; sur les 2 287 intitules de la base, 2 225
recoivent un objet (97,3 %), les autres sont des cas de doute assumes.
"""

import json
from datetime import date
from pathlib import Path

import pytest

from app.utils.titre_officiel import date_du_texte, numero_du_titre, objet_du_titre

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "titre, objet",
    [
        # portant : l'objet est ce qui suit, premiere lettre en capitale.
        (
            "Décret N°2018/763 du 10 décembre 2018 portant révocation d'un magistrat",
            "Révocation d'un magistrat",
        ),
        ("Loi N° 2016/007 du 12 juillet 2016 portant Code Pénal", "Code Pénal"),
        (
            "decret n° 2013/034 du 14 février 2013 portant nomination du pca de l'Ecole "
            "Internationale des Forces de Sécurité (eiforces)",
            "Nomination du pca de l'Ecole Internationale des Forces de Sécurité (eiforces)",
        ),
        # fixant : article retire, y compris dans l'enumeration de tete.
        (
            "Loi N° 2017/009 du 12 juillet 2017 fixant les attributions, l’organisation et le "
            "fonctionnement du Conseil Economique et Social du Cameroun.",
            "Attributions, organisation et fonctionnement du Conseil Economique et Social du Cameroun",
        ),
        (
            "Décret N°2024/555 du 14 novembre 2024 fixant les règles applicables au recrutement "
            "et à la discipline du Personnel de la Justice Militaire",
            "Règles applicables au recrutement et à la discipline du Personnel de la Justice Militaire",
        ),
        # modifiant : nominalise, « de » contracte ; la date de l'acte cite reste.
        (
            "Décret N°2013/332 du 13 septembre 2013 modifiant et complétant certaines dispositions "
            "du décret n°77/245 du 15 juillet 1977 portant organisation des chefferies traditionnelles",
            "Modification de certaines dispositions du décret n°77/245 du 15 juillet 1977 portant "
            "organisation des chefferies traditionnelles",
        ),
        (
            "Décret N°81/477 du 19 novembre 1981 modifiant le décret n° 74/759 du 26 août 1974 "
            "portant organisation du régime des pensions civiles",
            "Modification du décret n° 74/759 du 26 août 1974 portant organisation du régime des "
            "pensions civiles",
        ),
        # complétant, et annee collee au mois (« 07septembre »).
        (
            "Décret N°82/408 du 07septembre 1982 complétant certaines dispositions du décret "
            "n° 74/759 du 26 août 1974 portant organisation du régime des Pensions Civiles",
            "Complément à certaines dispositions du décret n° 74/759 du 26 août 1974 portant "
            "organisation du régime des Pensions Civiles",
        ),
        (
            "Décret n°2014/418 du 24 octobre 2014 ratifiant l'accord de prêt n°2200160001189 "
            "d'un montant de 5 millions d'unités de compte",
            "Ratification de l'accord de prêt n°2200160001189 d'un montant de 5 millions d'unités de compte",
        ),
        # « I' Accord » : un l lu comme un I par l'OCR.
        (
            "Décret N°2021/439 du 20 août 2021 portant ratification de I' Accord de coopération "
            "relatif à la suppression réciproque de l'obligation de visa avec le Conseil Fédéral Suisse",
            "Ratification de l'Accord de coopération relatif à la suppression réciproque de "
            "l'obligation de visa avec le Conseil Fédéral Suisse",
        ),
        # autorisant : ce qui est autorise, pas qui l'est.
        (
            "Loi n°2014/008 du 18 juillet 2014 autorisant le Président de la République à "
            "ratifier l'accord entre le gouvernement de la République du Cameroun et le "
            "gouvernement du royaume du Maroc",
            "Autorisation de ratifier l'accord entre le gouvernement de la République du Cameroun "
            "et le gouvernement du royaume du Maroc",
        ),
        (
            "Décret N°2026/013 du 21 janvier 2026 autorisant des emprunts intérieurs et extérieurs "
            "destinés au financement des projets de développement",
            "Autorisation d'emprunts intérieurs et extérieurs destinés au financement des projets "
            "de développement",
        ),
        # habilitant : le signataire (le ministre et tout son titre) est omis.
        (
            "Décret N°2015/395 du 15 septembre 2015 habilitant le Ministre de l'Economie, de la "
            "Planification et de l'Aménagement du Territoire, à signer avec la Banque de droit "
            "autrichien Raffeisen Bank International, un accord de prêt",
            "Habilitation à signer avec la Banque de droit autrichien Raffeisen Bank International, "
            "un accord de prêt",
        ),
        (
            "Décret N°2022/452 du 28 septembre 2022 habilitant un Accord de Prêt pour le "
            "financement du Programme d'Appui au Renforcement de la Production Agricole au Cameroun",
            "Accord de Prêt pour le financement du Programme d'Appui au Renforcement de la "
            "Production Agricole au Cameroun",
        ),
        # relatif / relative : la preposition repetee dans l'enumeration tombe.
        (
            "Loi N°2025/009 du 15 juillet 2025 relative au don, au prélèvement et à la greffe "
            "de matériel biologique humain au Cameroun",
            "Don, prélèvement et greffe de matériel biologique humain au Cameroun",
        ),
        (
            "Circulaire N°001/CAB/PRC du 10 juillet 2020 relative à la préparation du budget "
            "de l'Etat pour l'exercice 2021",
            "Préparation du budget de l'Etat pour l'exercice 2021",
        ),
        # Actes uniformes et traites : la date finale est celle de l'acte.
        ("Acte uniforme relatif à la médiation du 23 novembre 2017", "Médiation"),
        (
            "Acte uniforme révisé portant sur le droit commercial général du 15 décembre 2010",
            "Droit commercial général",
        ),
        (
            "Acte uniforme portant organisation des procédures simplifiées de recouvrement et des "
            "voies d'exécution, révisé, publié au Journal officiel de l'OHADA du 15 novembre 2023",
            "Organisation des procédures simplifiées de recouvrement et des voies d'exécution",
        ),
        (
            "Traité portant révision du Traité relatif à l'harmonisation du droit des affaires en "
            "Afrique du 17 octobre 2008",
            "Révision du Traité relatif à l'harmonisation du droit des affaires en Afrique",
        ),
        ("Loi N°98/004 du 14 avril 1998 d'orientation de l'éducation au Cameroun",
         "Orientation de l'éducation au Cameroun"),
        (
            "Décret N° 2015/124 du 18 mars 2015 instituant des obsèques officielles en l'honneur "
            "de Madame Françoise FONING",
            "Obsèques officielles en l'honneur de Madame Françoise FONING",
        ),
        (
            "Décret N° 2014/424 du 28 octobre 2014 ordonnant la publication au Journal Officiel "
            "du texte de l'accord de crédit N° 5422-CM",
            "Publication au Journal Officiel du texte de l'accord de crédit N° 5422-CM",
        ),
        # Verbe absent, nom d'objet frequent en tete de l'objet.
        (
            "Décret N°2019/233 du 08 mai 2019 nomination d'un Inspecteur Général au Ministère "
            "de la Santé Publique",
            "Nomination d'un Inspecteur Général au Ministère de la Santé Publique",
        ),
        # Fautes de saisie du corpus.
        (
            "Décret N°2021/069 du 04 février 2021 port an t agrément de la société « AGENCE "
            "DOBERMANN » à l'exercice des activités privées de gardiennage.",
            "Agrément de la société « AGENCE DOBERMANN » à l'exercice des activités privées de gardiennage",
        ),
        (
            "Décret N°2025/014 du 14 janvier 2025portant nomination d’un Inspecteur Général au "
            "Ministère de la Santé Publique",
            "Nomination d’un Inspecteur Général au Ministère de la Santé Publique",
        ),
        (
            "Arrêté N°0590/cab/pr du 17 aout 2026 portant portant titularisation de "
            "quarante-neuf (49) Inspecteurs de Police Stagiaires",
            "Titularisation de quarante-neuf (49) Inspecteurs de Police Stagiaires",
        ),
        ("Décret N°202l/345 du 17 juin 2021 portant révocation d'un Magistrat",
         "Révocation d'un Magistrat"),
        (
            "Décret N°2023/311 du 26 juilet 2023 instituant des obsèques officielles en "
            "l'honneur de Monsieur Ni John Fru Ndi",
            "Obsèques officielles en l'honneur de Monsieur Ni John Fru Ndi",
        ),
        (
            "DECRET N° 2013/297 DU 0 9 SEP. 2013 modifiant et complétant certaines dispositions "
            "du décret n° 2013/092 du 03 avril 2013 portant organisation et fonctionnement de "
            "l'Agence de Promotion des Petites et Moyennes Entreprises.",
            "Modification de certaines dispositions du décret n° 2013/092 du 03 avril 2013 portant "
            "organisation et fonctionnement de l'Agence de Promotion des Petites et Moyennes Entreprises",
        ),
        (
            "Décret N°2024/512 du 30 octobre 2024 fixant les règles applicables à la solde "
            "consulaire, aux frais de relève et à l'allocation d'installation_ du personnel de "
            "l'Etat en service hors du Cameroun",
            "Règles applicables à la solde consulaire, aux frais de relève et à l'allocation "
            "d'installation du personnel de l'Etat en service hors du Cameroun",
        ),
    ],
)
def test_objet_du_titre(titre, objet):
    assert objet_du_titre(titre) == objet


@pytest.mark.parametrize(
    "titre",
    [
        None,
        "",
        # Nom de fichier pris pour intitule.
        "decret_n_2014_543_du_10.12.2014_nomination_dg_apme",
        "arrête_n_0 453_cab_pr_du_05.06.2024",
        # Verbe que la decoupe ne sait pas nominaliser.
        "Arrêté N° 8448/CAB/PR du 24 juin 2017 déclarant fériée et chômée la journée du lundi 26 juin 2017",
        "Arrêté N° 0225/CAB/PR du 11 avril 2019 relevant un responsable de la Délégation Générale "
        "à la Sûreté Nationale de ses fonctions",
        # Un nom, pas un acte suivi de son objet.
        "Code civil applicable au Cameroun, annoté de la jurisprudence de la Cour suprême",
        "Accord-cadre entre la République du Cameroun et le Saint-Siège, relatif au statut "
        "juridique de l’Eglise Catholique au Cameroun",
        # « instituant » appartient au traite cite, pas au code.
        "Code des assurances des États membres de la CIMA, édition 2019, et Traité instituant une "
        "organisation intégrée de l'industrie des assurances dans les États africains du 10 juillet 1992",
        # En-tete qui porte autre chose qu'une nature, un numero et une date.
        "Décret N° 2016/341 du 22 juillet 2016 d'Etat portant nomination de responsables dans les "
        "services du Contrôle Supérieur de l'Etat.",
        # Nom d'objet perdu dans l'intitule : « portant de la médaille ».
        "Décret N° 2015/193 portant de la médaille de la Vaillance à titre collectif aux étendards "
        "des Formations et Unités de la Gendarmerie Nationale",
        # Second verbe coordonne : la moitie de l'objet resterait conjuguee.
        "Décret N°2023/475 du 7 novembre 2023 précisant certaines attributions et fixant les "
        "avantages des comptables publics placés auprès des Collectivités Territoriales Décentralisées",
        # Signataire sans infinitif reconnu : l'intitule est tronque.
        "Décret N°2019/079 du 22 février 2019 habilitant le Ministre de l'Economie, de la "
        "Planification et de l'Aménagement du Territoire",
        # Verbe remplace par des points de suspension.
        "Loi N°2025/001 du 25 avril 2025 ... pour l'élimination de la double imposition en "
        "matière d'impôts sur le revenu",
    ],
)
def test_objet_du_titre_rend_none_quand_il_doute(titre):
    assert objet_du_titre(titre) is None


def test_echantillon_relu_a_la_main():
    """118 intitules tires au hasard dans jurix_dev, relus un par un."""
    cas = json.loads(
        (Path(__file__).parents[1] / "fixtures" / "titres_objets.json").read_text(encoding="utf-8")
    )["cas"]
    assert len(cas) == 118
    ecarts = [
        (c["titre"], c["objet"], objet_du_titre(c["titre"]))
        for c in cas
        if objet_du_titre(c["titre"]) != c["objet"]
    ]
    assert ecarts == []
    ecarts = [c["titre"] for c in cas if numero_du_titre(c["titre"]) != c["numero"]]
    assert ecarts == []


@pytest.mark.parametrize(
    "titre, numero",
    [
        ("Décret N°2018/763 du 10 décembre 2018 portant révocation d'un magistrat", "2018/763"),
        ("Arrêté N°0342/CAB/PR du 24 mai 2019 portant nomination", "0342/CAB/PR"),
        ("Arrêté N1078/CAB/PR du 31 décembre 2022 portant avancement", "1078/CAB/PR"),
        ("Arrêté N°0095/CAB/PR/du 16 février 2026 portant nomination", "0095/CAB/PR"),
        ("Décret N° 2015/ 437 du 02 octobre 2015 Portant nomination", "2015/437"),
        ("Arrêté N°0292 /CAB/PR du 27 avril 2018 déclarant fériée", "0292/CAB/PR"),
        ("Décret 2017/209 du 12 mai 2017 portant reconnaissance", "2017/209"),
        ("Loi N°67-LF-7 du 12 juin 1967 instituant un Code des prestations familiales", "67-LF-7"),
        ("Décret No. 2020/038 du 21 Janvier 2020 habilitant MINEPAT", "2020/038"),
        ("Pleins Pouvoirs N°2020/137 du 25 mars 2020 habilitant le Ministre", "2020/137"),
        ("Loi-cadre N°2011/012 du 06 mai 2011 portant protection du consommateur", "2011/012"),
        ("Décret N°2024/193 juin 10 portant avancement d'échelon", "2024/193"),
        ("Décret N°202l/345 du 17 juin 2021 portant révocation", "2021/345"),
        ("Acte uniforme relatif à la médiation du 23 novembre 2017", None),
        ("Code civil applicable au Cameroun", None),
        ("Décret portant convocation du Corps électoral", None),
        ("decret_n_2014_543_du_10.12.2014_nomination_dg_apme", None),
        (None, None),
    ],
)
def test_numero_du_titre(titre, numero):
    assert numero_du_titre(titre) == numero


def test_date_du_texte_ecarte_le_1er_janvier_de_remplissage():
    """
    L'ingestion pose le 1er janvier de l'annee du numero quand l'intitule ne
    donne ni jour ni mois : ce n'est pas une date juridique.
    """
    assert date_du_texte(date(2024, 1, 1), "Décret N°2024/193 juin 10 portant avancement") is None
    assert date_du_texte(date(2015, 1, 1), "Décret N°2015/405 fixant les modalités") is None


def test_date_du_texte_garde_les_vraies_dates():
    assert date_du_texte(date(2018, 12, 10), "Décret N°2018/763 du 10 décembre 2018") == date(
        2018, 12, 10
    )
    # Un vrai 1er janvier, que l'intitule nomme.
    assert date_du_texte(date(2020, 1, 1), "Décret N°2020/001 du 1er janvier 2020") == date(2020, 1, 1)
    assert date_du_texte(date(2020, 1, 1), "Décret N°2020/001 du 01 JANV. 2020") == date(2020, 1, 1)
    assert date_du_texte(None, "Décret sans date") is None
