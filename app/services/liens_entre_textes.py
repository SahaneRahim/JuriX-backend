"""
Les liens entre textes successifs du corpus : quel texte plus recent a
modifie, complete, abroge ou remplace quel texte plus ancien, et quels
articles. Ce module ne contient que les donnees et leurs preuves ;
app/services/textes_successifs.py en tire les lignes de statut.

D'OU VIENNENT CES LIENS. Releves en base le 2026-10-14 dans les textes
eux-memes (clauses d'abrogation, intitules « modifiant et completant »,
articles reecrits marques « (nouveau) »), chacun confirme par une seconde
lecture independante, saisi par un redacteur qui a controle les morceaux
marques en base, puis relu par un contradicteur. La preuve de chaque lien est
la disposition citee en commentaire.

AJOUTER UN LIEN. Ne saisir que ce qu'un texte dit expressement : une ligne de
statut fausse egare le modele plus surement qu'une ligne absente. Puis :
    python scripts/maintenance/verifier_liens_textes.py --detail
qui rejoue les lignes sur les vrais morceaux. A relancer aussi apres toute
ingestion ou re-decoupage d'un texte lie.
"""

from dataclasses import dataclass
from datetime import date
from typing import Optional, Tuple


@dataclass(frozen=True)
class Lien:
    """Un texte plus recent qui touche un texte plus ancien."""

    texte: str
    """Reference en base du texte touche (« LOI-2012-001 »)."""
    texte_nom: str
    """Son nom, article compris : « la Loi N°2012/001 portant Code électoral »."""
    nature: str
    """Une cle de NATURES."""
    par_titre: str
    """Intitule du texte qui le touche, article et date compris : « la Loi N°2026/003 du … »."""
    par_date: date
    par_reference: Optional[str] = None
    """Sa reference en base ; None quand il manque au corpus."""
    articles: Tuple[str, ...] = ()
    """Articles touches (« 55 », « 34 à 101 ») ; vide : le texte entier."""
    par_articles: Tuple[str, ...] = ()
    """
    Morceaux du texte recent qui portent le changement ; vide : tous. Une loi
    de finances modifie un texte dans un seul de ses articles : annoncer le
    changement sous chacun des autres egarerait le modele. Un pseudo-numero
    (« SECTION_1 ») s'y ecrit tel quel.
    """
    section: Optional[str] = None
    """
    Debut de l'intitule de section des morceaux vises (« TITRE 2 Des actes de
    l'état civil. »). La compilation du Code civil reprend les memes numeros
    d'articles dans plusieurs titres : sans section, « 53 » marquerait aussi
    l'article 53 de l'Acte uniforme sur les suretes.
    """
    morceaux: Tuple[Tuple[str, str, str], ...] = ()
    """
    (numero du morceau, article qu'il contient, repere) : un article touche
    range sous un autre numero (la fin de l'article 6 de la Constitution est le
    morceau « SECTION_2 »). Le repere, un passage du texte de l'article, doit
    figurer dans le morceau : si un nouveau decoupage renumerote, la ligne
    disparait au lieu de se poser sur un texte sans rapport.
    """
    sauf: Tuple[Tuple[str, str], ...] = ()
    """
    (numero du morceau, repere) : morceau designe par la liste mais a ne pas
    marquer, parce qu'il porte un autre texte sous le meme numero.
    """
    portee: str = ""
    """Ce qui change, en une phrase, quand l'intitule ne le dit pas."""




_CONSTITUTION = (
    "la Constitution (Loi N°96/06 du 18 janvier 1996 portant révision "
    "de la Constitution du 02 juin 1972)"
)
_LOI_DE_FINANCES_2025 = (
    "la Loi n° 2024/013 du 23 décembre 2024 portant loi de finances pour l'exercice 2025"
)

# Releves verifies le 2026-10-14 : voir la docstring du module. La preuve de
# chaque lien est la disposition citee en commentaire, relue en base.
#
# Ecartes, et pourquoi :
# - Code electoral et loi 2012/017 : le document en base est la version
#   consolidee par ELECAM, qui integre deja ses quatre articles.
# - Code electoral et loi 2019/005 (PRC-7480, articles 249, 250 et 257) : le
#   document en base recopie la loi de 2019 en fin de texte, sous les memes
#   numeros d'articles. Le statut « modifié » marquerait aussi la copie, qui
#   est la nouvelle redaction. A reprendre quand la copie sera retiree.
# - Loi 2004/004 et loi 2008/001, article 7 : la duree du mandat contredit la
#   Constitution revisee, mais la loi de 2008 ne vise pas la loi de 2004.
#   Y voir une abrogation est une appreciation, pas un lien lu dans un texte.
# - Loi 2015/018, article 91, et loi de finances 2025 : celle-ci retablit le
#   bareme de 2015. L'article 91 dit donc la regle ; c'est la loi de finances
#   2024 qui est perimee, d'ou le lien sur son article 24.
# - Loi de finances 2024, articles 21 et 35, et loi de finances 2025 : les
#   memes numeros designent en base des articles du Code general des impots
#   recopies dans la loi de 2024 (morceaux « 21 » et « 35 »), et ses annexes
#   (35.2 a 35.20), que la loi de 2025 ne touche pas. Ses articles 16, 26 et
#   30, que la loi de 2025 modifie aussi, ne sont pas saisis faute d'une portee
#   relue.
LIENS: Tuple[Lien, ...] = (
    # ========================================================================
    # Constitution, Code electoral, Conseil constitutionnel, Cour supreme,
    # traite OHADA, loi de finances 2024, Code penal
    # ========================================================================

    # Constitution. Loi 2008/001, art. 1er : « ARTICLE 6 … ARTICLE 14 …
    # ARTICLE 15 … ARTICLE 51 … ARTICLE 53 … ARTICLE 67 » (nouveau). Les
    # articles 6 et 53 ont ete reecrits depuis par la loi 2026/002 : ils sont
    # dans les liens suivants.
    Lien(
        texte="PRC-6283",
        texte_nom=_CONSTITUTION,
        nature="modifie",
        par_titre="la Loi N°2008/001 du 14 avril 2008",
        par_date=date(2008, 4, 14),
        par_reference="PRC-6284",
        articles=("14", "15", "51"),
        par_articles=("14", "15"),
        portee=(
            "seuls sont réécrits l'alinéa 3 a) de l'article 14 (sessions ordinaires en mars, juin "
            "et novembre), l'alinéa 4 de l'article 15 (élection de la nouvelle Assemblée quarante "
            "jours au moins et cent vingt jours au plus après la fin du délai de prorogation ou "
            "d'abrègement) et l'alinéa 1 de l'article 51 (membres du Conseil constitutionnel "
            "désignés pour six ans, mandat éventuellement renouvelable)"
        ),
    ),
    Lien(
        texte="PRC-6283",
        texte_nom=_CONSTITUTION,
        nature="complete",
        par_titre="la Loi N°2008/001 du 14 avril 2008",
        par_date=date(2008, 4, 14),
        par_reference="PRC-6284",
        articles=("67",),
        par_articles=("67",),
        portee=(
            "alinéa 6 ajouté : si le Sénat est mis en place avant les Régions, le collège électoral "
            "des sénateurs est composé exclusivement des conseillers municipaux"
        ),
    ),
    # Loi 2026/002, art. 1er : « Les dispositions des articles 5, 6, 7, 10, 53
    # et 66 de la Constitution […] sont modifiées et complétées ».
    Lien(
        texte="PRC-6283",
        texte_nom=_CONSTITUTION,
        nature="modifie",
        par_titre="la Loi N°2026/002 du 14 avril 2026",
        par_date=date(2026, 4, 14),
        par_reference="PRC-10679",
        articles=("6",),
        par_articles=("6",),
        morceaux=(("SECTION_2", "6", "par intérim"),),
        portee=(
            "réécrit en entier ; en cas de vacance de la présidence pour cause de décès, de démission "
            "ou d'empêchement définitif constaté par le Conseil constitutionnel, le Vice-Président "
            "achève le mandat (alinéa 5) ; l'intérim et l'élection d'un nouveau Président ne jouent "
            "que si le Vice-Président est à son tour empêché"
        ),
    ),
    Lien(
        texte="PRC-6283",
        texte_nom=_CONSTITUTION,
        nature="modifie",
        par_titre="la Loi N°2026/002 du 14 avril 2026",
        par_date=date(2026, 4, 14),
        par_reference="PRC-10679",
        articles=("5", "7", "10", "53", "66"),
        par_articles=("5", "7", "10", "53", "66"),
        portee=(
            "le Vice-Président y est ajouté : le Président de la République peut être assisté d'un "
            "Vice-Président (article 5), qui prête serment (article 7), est nommé par lui et peut "
            "recevoir délégation (article 10), relève de la Haute Cour de justice (article 53) et "
            "déclare ses biens (article 66)"
        ),
    ),
    # La loi 2026/002 vise la Constitution « modifiée et complétée par […] la
    # loi n° 2008/001 » : la redaction que la loi de 2008 donnait aux articles
    # 6 (son morceau DISPOSITIF) et 53 est remplacee.
    Lien(
        texte="PRC-6284",
        texte_nom="la Loi N°2008/001 du 14 avril 2008 modifiant la Constitution",
        nature="modifie",
        par_titre="la Loi N°2026/002 du 14 avril 2026",
        par_date=date(2026, 4, 14),
        par_reference="PRC-10679",
        # Pas de morceau « 6 » dans la loi de 2008 : son article 6 est dans le
        # morceau DISPOSITIF.
        articles=("6", "53"),
        par_articles=("6", "53"),
        morceaux=(("DISPOSITIF", "6", "ARTICLE 6"),),
        portee="la rédaction des articles 6 et 53 de la Constitution issue de la loi de 2008 est remplacée",
    ),
    # Code electoral. Loi 2026/003, art. 1er : « Les dispositions des articles
    # 140, 142, 143, 144, 145, 146 et 170 de la loi n° 2012/001 du 19 avril
    # 2012 portant Code Electoral sont modifiées et complétées ».
    Lien(
        texte="LOI-2012-001",
        texte_nom="la Loi N°2012/001 du 19 avril 2012 portant Code électoral",
        nature="modifie",
        par_titre="la Loi N°2026/003 du 14 avril 2026",
        par_date=date(2026, 4, 14),
        par_reference="PRC-10680",
        articles=("140", "142", "143", "144", "145", "146"),
        # L'article 142 nouveau est en base le morceau SECTION_1 ; le 146 est
        # fondu dans le 145.
        par_articles=("1", "140", "SECTION_1", "143", "144", "145"),
        portee=(
            "en cas de vacance de la présidence, le Vice-Président prête serment et achève le mandat ; "
            "le Conseil constitutionnel, saisi par le Vice-Président, constate l'empêchement définitif ; "
            "l'intérim du Président du Sénat et l'élection anticipée ne jouent que si le Vice-Président "
            "est à son tour empêché"
        ),
    ),
    Lien(
        texte="LOI-2012-001",
        texte_nom="la Loi N°2012/001 du 19 avril 2012 portant Code électoral",
        nature="modifie",
        par_titre="la Loi N°2026/003 du 14 avril 2026",
        par_date=date(2026, 4, 14),
        par_reference="PRC-10680",
        articles=("170",),
        par_articles=("170",),
        portee=(
            "alinéa 2 réécrit : le mandat des conseillers municipaux peut être prorogé ou abrégé par "
            "décret sans le plafond de dix-huit mois"
        ),
    ),
    # Conseil constitutionnel. Loi 2026/004, art. 1er : « Les dispositions des
    # articles 38 et 39 de la loi n° 2004/004 du 21 avril 2004 […] sont
    # modifiées et complétées ».
    Lien(
        texte="LOI-2004-004",
        texte_nom=(
            "la Loi N°2004/004 du 21 avril 2004 portant organisation et fonctionnement "
            "du Conseil constitutionnel"
        ),
        nature="modifie",
        par_titre="la Loi N°2026/004 du 14 avril 2026",
        par_date=date(2026, 4, 14),
        par_reference="PRC-10681",
        articles=("38", "39"),
        par_articles=("1", "38", "39"),
        portee=(
            "réécrits en entier ; saisi par le Vice-Président, le Conseil constitutionnel constate "
            "l'empêchement définitif du Président de la République à la majorité des deux tiers de "
            "ses membres ; la lettre de démission est adressée au Président du Conseil "
            "constitutionnel et au Vice-Président"
        ),
    ),
    # Titre de PRC-10681 et visas des decrets de nomination (PRC-6279,
    # PRC-9936) : la loi 2004/004 est « modifiée et complétée par la loi n°
    # 2012/015 du 21 décembre 2012 ». Ce texte manque au corpus, et le document
    # en base est la version d'origine : on ne sait pas quels articles ont change.
    Lien(
        texte="LOI-2004-004",
        texte_nom=(
            "la Loi N°2004/004 du 21 avril 2004 portant organisation et fonctionnement "
            "du Conseil constitutionnel"
        ),
        nature="modifie",
        par_titre="la Loi n° 2012/015 du 21 décembre 2012",
        par_date=date(2012, 12, 21),
        portee="les articles modifiés ne sont pas connus",
    ),
    # Cour supreme. Loi 2017/014, art. 1er : les articles 8, 11 et 37 de la loi
    # n° 2006/016 sont modifies et completes ; un article 37-1 est ajoute.
    # L'ancien article 8 est en base a la fin du morceau « 7 ».
    Lien(
        texte="LOI-2006-016",
        texte_nom=(
            "la Loi N°2006/016 du 29 décembre 2006 fixant l'organisation et le fonctionnement "
            "de la Cour suprême"
        ),
        nature="modifie",
        par_titre="la Loi N° 2017/014 du 12 juillet 2017",
        par_date=date(2017, 7, 12),
        par_reference="PRC-5698",
        articles=("8", "37"),
        par_articles=("1", "8", "37"),
        morceaux=(("7", "8", "ARTICLE 8"),),
        portee=(
            "la chambre judiciaire comprend aussi une section de Common Law (article 8), compétente "
            "pour les pourvois en matière de Common Law (article 37, alinéa a, et article 37-1 ajouté)"
        ),
    ),
    Lien(
        texte="LOI-2006-016",
        texte_nom=(
            "la Loi N°2006/016 du 29 décembre 2006 fixant l'organisation et le fonctionnement "
            "de la Cour suprême"
        ),
        nature="complete",
        par_titre="la Loi N° 2017/014 du 12 juillet 2017",
        par_date=date(2017, 7, 12),
        par_reference="PRC-5698",
        articles=("11",),
        par_articles=("11",),
        portee="alinéa 3 ajouté : les magistrats de la section de Common Law sont de culture juridique anglo-saxonne",
    ),
    # Traite OHADA. Traite de Quebec du 17 octobre 2008, art. 1er : vingt
    # articles du traite de 1993 sont remplaces par une nouvelle redaction.
    # L'ancien article 42 est en base dans le morceau « 40 ».
    Lien(
        texte="OHADA-TRAITE-1993",
        texte_nom="le Traité relatif à l'harmonisation du droit des affaires en Afrique du 17 octobre 1993",
        nature="modifie",
        par_titre=(
            "le Traité portant révision du Traité relatif à l'harmonisation du droit des affaires "
            "en Afrique du 17 octobre 2008"
        ),
        par_date=date(2008, 10, 17),
        par_reference="OHADA-TRAITE-2008",
        articles=(
            "3", "4", "7", "9", "12", "14", "17", "27", "31", "39",
            "40", "41", "42", "43", "45", "49", "57", "59", "61", "63",
        ),
        par_articles=(
            "1", "3", "4", "7", "9", "12", "14", "17", "27", "31", "39",
            "40", "41", "42", "43", "45", "49", "57", "59", "61", "63",
        ),
        morceaux=(("40", "42", "langue de travail"),),
        portee=(
            "articles réécrits, notamment : la Cour commune de justice et d'arbitrage compte neuf "
            "juges (article 31), les Actes uniformes entrent en vigueur quatre-vingt-dix jours après "
            "leur publication au Journal officiel de l'OHADA (article 9), quatre langues de travail "
            "(article 42)"
        ),
    ),
    # Loi de finances 2025, art. 24 : « Les dispositions de l'article 24 de la
    # loi n° 2023/019 […] sont modifiées et complétées ». Son alinea 7
    # retablit, pour les infractions a la loi n° 2015/018 regissant l'activite
    # commerciale, le bareme de l'article 91 de cette loi.
    Lien(
        texte="PRC-9875",
        texte_nom="la Loi N°2023/019 du 19 décembre 2023 portant loi de finances pour l'exercice 2024",
        nature="modifie",
        par_titre=_LOI_DE_FINANCES_2025,
        par_date=date(2024, 12, 23),
        par_reference="PRC-10254",
        articles=("24",),
        par_articles=("24",),
        portee=(
            "réécrit en entier (recettes non fiscales du commerce) ; son alinéa 7 rétablit, pour les "
            "infractions à la loi n° 2015/018 régissant l'activité commerciale, la sanction de "
            "l'article 91 de cette loi : 5 % du chiffre d'affaires annuel, 10 % pour les infractions "
            "de l'alinéa 2, avec un minimum"
        ),
    ),
    # Code penal. Loi 2019/020, art. 1er : « La loi n° 2016/007 du 12 juillet
    # 2016 portant Code Pénal est modifiée et complétée ainsi qu'il suit :
    # ARTICLE 241.- (nouveau) […] ARTICLE 241-1.- (nouveau) ». L'article
    # 241-1 est AJOUTE : il n'existe pas dans le texte de 2016.
    Lien(
        texte="PRC-4722",
        texte_nom="la Loi N° 2016/007 du 12 juillet 2016 portant Code Pénal",
        nature="modifie",
        par_titre="la Loi N° 2019/020 du 24 décembre 2019",
        par_date=date(2019, 12, 24),
        par_reference="PRC-8028",
        articles=("241",),
        # Le nouvel article 241 est dans le morceau « 1 », le 241-1 dans le « 241 ».
        par_articles=("1", "241"),
        portee=(
            "l'article 241 (outrage aux races et aux religions) est réécrit, et un article 241-1 "
            "(outrage à la tribu ou à l'ethnie) est ajouté"
        ),
    ),

    # ========================================================================
    # Code civil : etat civil et etat des personnes (loi 2024/016, ordonnance 81/02)
    # ========================================================================
    # Liens candidats du groupe cc_etat_civil : la compilation du Code civil
    # (CODE-CIVIL) face a la loi n° 2024/016 sur l'etat civil (PRC-10257) et a
    # l'ordonnance n° 81/02 (ORD-1981-002).
    #
    # Version relue en base : portee du titre 2 completee pour les formalites du
    # mariage, repere de l'art. 228 pris dans son texte, morceau « 335 » limite a
    # l'art. 335 (il porte aussi les art. 336 et 337).
    #
    # Les titres 2 et 5 du livre I melangent des articles du Code civil et des
    # articles de l'ordonnance inseres par la compilation, sous des numeros qui se
    # recoupent : chaque lien porte donc sa section, et les plages sont decoupees
    # pour laisser hors marque ce qui reste en vigueur (conditions de fond du
    # mariage, nom et prenoms, consentement, nullites, effets du mariage).

    # ------------------------------------------------------------------
    # Loi n° 2024/016 (PRC-10257).
    # Art. 129 : « Sont abrogées, toutes dispositions antérieures contraires,
    # notamment celles de l'ordonnance n°81-02 du 29 juin 1981 portant
    # organisation de l'état civil et diverses dispositions relatives à l'état
    # des personnes physiques, modifiée et complétée par la loi n° 2011/011 du
    # 06 mai 2011, en ses aspects relatifs à l'état civil. »
    # Art. 1er : « (1) La présente loi porte organisation du système
    # d'enregistrement des faits d'état civil au Cameroun. » ; « (2) Des lois
    # particulières fixentles règlesrelatives à l'état des personnes. »
    # ------------------------------------------------------------------
    #
    # Titre 2 du livre I : articles 34 à 101 du Code civil et, inseres par la
    # compilation, articles 1 à 11, 22 à 30, 34 à 40, 48 à 57 et 78 à 80 de
    # l'ordonnance. Regles contraires, par exemple :
    # - Code civil, art. 55 : « Les déclarations de naissance seront faites
    #   dans les douze jours de l'accouchement » ; ordonnance, art. 30 : « dans
    #   les 30 jours suivant l'accouchement » ; loi 2024/016, art. 10 (1) :
    #   « La naissance est déclarée à l'officier d'état civil du lieu de
    #   naissance dans un délai de quatre-vingt-dix (90) jours suivant
    #   l'accouchement. »
    # - Ordonnance, art. 78 (1) : déclaration de décès « dans le mois » ; loi
    #   2024/016, art. 21 (1) : « dans les quatre-vingt-dix (90) jours ».
    # - Code civil, art. 40 : « registres tenus doubles » ; loi 2024/016,
    #   art. 71 : « tenu en trois (03) exemplaires à souches ».
    # - Code civil, art. 64 : l'affiche de publication du mariage « restera
    #   apposée à la porte de la maison commune pendant dix jours » ; loi
    #   2024/016, art. 16 (1) : « Au moins trente (30) joursavant la
    #   célébration du mariage, l'officier d'état civil est saisi d'une
    #   déclaration d'intention de mariage des futurs époux. »
    # Non marques : le morceau 52 du titre (ordonnance, art. 52 : « Aucun
    # mariage ne peut être célébré », conditions de fond ; l'art. 52 du Code
    # civil est a la fin du morceau 51) et les articles 35, 36, 37 et 39 de
    # l'ordonnance sur le nom et les prénoms, qui relevent de l'etat des
    # personnes (loi 2024/016, art. 40 : l'officier « attribue des nom(s)
    # etprénom(s) à l'enfant conformément à la législation en vigueur ») ;
    # ils portent les memes numeros que les articles 35, 36, 37 et 39 du Code
    # civil, d'ou `sauf`.
    # Le chapitre III du titre (« Des actes de mariage ») est marque lui aussi :
    # publication, oppositions et celebration (Code civil, art. 63 a 76 ;
    # ordonnance, art. 48 a 50 et 53 a 57, dont le morceau 53 du titre 2 qui
    # porte aussi les art. 54 et 55). La portee nomme donc ces formalites et le
    # delai de trente jours (loi 2024/016, art. 16 (1), art. 17 (1), art. 42 a
    # 47), comme celle du lien du titre 5 : sans eux, la ligne ne disait que
    # « l'enregistrement des [...] mariages », et la copie de l'art. 53 de
    # l'ordonnance (« Un mois au moins avant la célébration du mariage ») ne
    # recevait pas, au titre 2, le delai que sa copie du titre 5 recoit.
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun, titre « Des actes de l'état civil »",
        nature="abroge_partiellement",
        par_titre="la Loi n° 2024/016 du 23 décembre 2024",
        par_date=date(2024, 12, 23),
        par_reference="PRC-10257",
        articles=("1 à 11", "22 à 30", "34 à 51", "53 à 101"),
        par_articles=("129",),
        section="TITRE 2 Des actes de l'état civil.",
        sauf=(
            ("35", "librement choisis par ses parents"),
            ("36", "choisis comme prénoms"),
            ("37", "réunion de plusieurs autres noms"),
            ("39", "mêmes noms et prénoms à plusieurs enfants"),
        ),
        portee=(
            "pour la déclaration et l'enregistrement des naissances, des mariages et des décès, y "
            "compris la déclaration d'intention de mariage, sa publication, la forme et l'instruction "
            "des oppositions et la célébration du mariage, pour l'établissement, la tenue, la "
            "rectification et la reconstitution des actes et registres d'état civil et pour "
            "l'organisation des officiers et centres d'état civil, cette loi s'applique ; elle fixe à "
            "quatre-vingt-dix (90) jours le délai de déclaration de la naissance et du décès, prévoit "
            "que l'officier d'état civil est saisi de la déclaration d'intention de mariage au moins "
            "trente (30) jours avant la célébration et que chaque registre est tenu en trois (03) "
            "exemplaires, et abroge notamment l'ordonnance n° 81/02 du 29 juin 1981, dont ce titre "
            "reproduit des articles, en ses aspects relatifs à l'état civil"
        ),
    ),
    # Titre 5 du livre I : formalites du mariage. Loi 2024/016, art. 16 (1) :
    # « Au moins trente (30) joursavant la célébration du mariage, l'officier
    # d'état civil est saisi d'une déclaration d'intention de mariage des
    # futurs époux. » ; art. 17 (1) : publication « par voie d'affichage au
    # centre d'état civil du lieu de célébration du mariage » ; art. 18 à 20 :
    # dispense de publication ; art. 42 a 45 : forme, transmission et
    # instruction des oppositions, art. 44 : « Le Président du Tribunal statue
    # sur l'oppositionà la célébration d'un mariage dans le délai de dix (10)
    # jours à compter de sa saisine. » ; art. 46 a 48 : celebration, et
    # mariage d'un national celebre a l'etranger (« est tenu de déclarer et
    # de faire transcrire »).
    # Contre : Code civil, art. 165 a 170 (celebration devant l'officier du
    # domicile ou de la residence, publications, dispense) et 176 a 179
    # (forme de l'acte d'opposition, mainlevee par le tribunal de premiere
    # instance « dans les dix jours », appel) ; ordonnance, art. 53 (« Un mois
    # au moins avant la célébration du mariage »), 54, 55, 57, 59, 60 et 62.
    # Non marques : l'art. 171 du Code civil, deja « Abrogé par L. 10 mars
    # 1938 » ; les art. 173 et 174 (qui peut former opposition : la loi de 2024
    # ne le dit pas, voir le lien de l'ordonnance plus bas) ; les morceaux de
    # l'ordonnance qui portent des regles de fond encore en vigueur : 52
    # (conditions du mariage), 58 (« toute personne justifiant d'un intérêt
    # légitime peut faire opposition »), 61 (« est irrecevable, d'ordre public,
    # toute opposition tenant à l'existence, au paiement ou modalités de
    # paiement de la dot coutumière »), 63 (nullite), 64 a 67 (consentement),
    # 74, 75 et 77.
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun, titre « Du mariage »",
        nature="abroge_partiellement",
        par_titre="la Loi n° 2024/016 du 23 décembre 2024",
        par_date=date(2024, 12, 23),
        par_reference="PRC-10257",
        articles=("53 à 57", "59", "60", "62", "165 à 170", "176 à 179"),
        par_articles=("129",),
        section="TITRE 5 Du mariage.",
        portee=(
            "pour la déclaration d'intention de mariage, dont l'officier d'état civil est saisi au "
            "moins trente (30) jours avant la célébration et qu'il publie par voie d'affichage, la "
            "dispense de publication, la forme et l'instruction des oppositions, sur lesquelles le "
            "Président du Tribunal statue dans le délai de dix (10) jours, la célébration du mariage "
            "et la transcription de celui d'un national marié à l'étranger, cette loi s'applique ; "
            "elle abroge notamment l'ordonnance n° 81/02 du 29 juin 1981, dont ce titre reproduit "
            "des articles, en ses aspects relatifs à l'état civil"
        ),
    ),
    # ------------------------------------------------------------------
    # Ordonnance n° 81/02 (ORD-1981-002), toujours en vigueur pour l'etat des
    # personnes (la loi 2024/016 ne l'abroge qu'« en ses aspects relatifs à
    # l'état civil »). Art. 84 : « Sont abrogées toutes dispositions
    # antérieures contraires notamment celles des lois nº s 66/2/COR du 7
    # juillet 1966 et 68/LF/2 du 11 juin 1968. » Seules les regles du Code
    # civil que l'ordonnance regle autrement, sur la meme question, sont
    # retenues.
    # ------------------------------------------------------------------
    #
    # Consentement des parents au mariage du mineur. Ordonnance, art. 64 :
    # « 2. Le consentement d'un futur époux mineur n'est valable que s'il est
    # appuyé de celui de ses père et mère. 3. Le consentement d'un seul des
    # parents est suffisant : [a) pour les enfants naturels, lorsque leur
    # filiation est légalement établie à l'égard d'un seul de leurs auteurs
    # seulement ;] b) en cas de décès ou d'absence judiciairement constatée de
    # l'un des auteurs ou si l'un d'eux se trouve dans l'incapacité ou
    # l'impossibilité d'exprimer son consentement ; c) en cas de dissentiment
    # entre le père et la mère, si L'auteur consentant est celui qui exerce la
    # puissance paternelle ou assume la garde de l'enfant, sauf décision
    # contraire du juge [...]. 4. Le consentement du tuteur ou du responsable
    # coutumier remplace valablement a) celui des père et mère de l'enfant né
    # de parents demeurés inconnus ; b) celui des père et mère de l'enfant
    # orphelin ; c) celui des père et mère de l'enfant dont les parents sont
    # dans l'impossibilité ou l'incapacité d'exprimer leur consentement. »
    # (le a) est illisible dans ORD-1981-002 et lu dans la reproduction du
    # titre 5). Contre : Code civil, art. 148 et 158 (« en cas de dissentiment
    # entre le père et la mère, ce partage emporte consentement »), 150 (« les
    # aïeuls et aïeules les remplacent »), 159 et 160 (consentement ou
    # autorisation du conseil de famille). L'art. 158 est dans le morceau 157.
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun, titre « Du mariage »",
        nature="abroge_partiellement",
        par_titre="l'Ordonnance N°81/002 du 29 juin 1981",
        par_date=date(1981, 6, 29),
        par_reference="ORD-1981-002",
        articles=("148", "150", "158", "159", "160"),
        par_articles=("64",),
        section="TITRE 5 Du mariage.",
        morceaux=(("157", "158", "applicables à l'enfant naturel mineur"),),
        portee=(
            "le consentement du futur époux mineur n'est valable que s'il est appuyé de celui de ses "
            "père et mère ; celui d'un seul suffit lorsque la filiation de l'enfant né hors mariage "
            "n'est établie qu'à l'égard de l'un d'eux, en cas de décès ou d'absence judiciairement "
            "constatée de l'autre ou s'il ne peut exprimer son consentement, et, en cas de "
            "dissentiment, si le parent consentant exerce la puissance paternelle ou assume la garde "
            "de l'enfant, sauf décision contraire du juge ; le tuteur ou le responsable coutumier "
            "consent valablement à la place des père et mère de l'enfant né de parents demeurés "
            "inconnus, de l'enfant orphelin ou de celui dont les parents ne peuvent exprimer leur "
            "consentement (article 64 de l'ordonnance)"
        ),
    ),
    # Qui peut former opposition. Ordonnance, art. 58 : « Dans le délai prévu
    # à l'article 53 ci-dessus, toute personne justifiant d'un intérêt
    # légitime peut faire opposition à la célébration du mariage, notamment: -
    # le père, la mère, le tuteur pour les futurs époux mineurs - le
    # responsable coutumier, notamment en cas d'inceste coutumier ; - l'époux
    # d'une femme engagée dans les liens d'un précédent mariage non dissous ;
    # - l'épouse d'un homme engagé dans les liens d'un précédent mariage à
    # régime monogamique non dissous. » Contre : Code civil, art. 173 (les
    # aïeuls seulement « à défaut de père et de mère ») et 174 (les
    # collatéraux « ne peuvent former aucune opposition que dans les deux cas
    # suivants » ; le morceau porte aussi l'art. 175, le tuteur ou curateur
    # autorisé par le conseil de famille). La loi 2024/016 ne dit pas qui peut
    # former opposition ; son art. 42 (4) exige seulement d'énoncer « la
    # qualité qui lui confère le droit de formuler l'opposition ».
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun, titre « Du mariage »",
        nature="abroge_partiellement",
        par_titre="l'Ordonnance N°81/002 du 29 juin 1981",
        par_date=date(1981, 6, 29),
        par_reference="ORD-1981-002",
        articles=("173", "174"),
        par_articles=("58",),
        section="TITRE 5 Du mariage.",
        portee=(
            "toute personne justifiant d'un intérêt légitime peut faire opposition à la célébration "
            "du mariage, notamment le père, la mère, le tuteur pour les futurs époux mineurs, le "
            "responsable coutumier, notamment en cas d'inceste coutumier, l'époux d'une femme engagée "
            "dans les liens d'un précédent mariage non dissous et l'épouse d'un homme engagé dans les "
            "liens d'un précédent mariage à régime monogamique non dissous (article 58 de "
            "l'ordonnance)"
        ),
    ),
    # Profession separee de la femme mariee. Ordonnance, art. 74 : « 1. La
    # femme mariée peut exercer une profession séparée de celle de son mari.
    # 2. Le mari peut s'opposer à l'exercice d'une telle profession dans
    # l'intérêt du mariage et des enfants. 3. Il est statué sur l'opposition du
    # mari par ordonnance du Président tribunal compétent rendue sans frais
    # dans les dix jours de la saisine, après audition obligatoire des
    # parties. » Contre : Code civil, art. 223, dans le morceau 222 : « Si
    # l'opposition du mari n'est pas justifiée par l'intérêt de la famille, la
    # femme peut être autorisée par justice à passer outre ». L'art. 222
    # lui-meme (compte courant) n'est pas contraire a l'art. 75 de
    # l'ordonnance : seul l'art. 223 est vise.
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun, titre « Du mariage »",
        nature="abroge_partiellement",
        par_titre="l'Ordonnance N°81/002 du 29 juin 1981",
        par_date=date(1981, 6, 29),
        par_reference="ORD-1981-002",
        articles=("223",),
        par_articles=("74",),
        section="TITRE 5 Du mariage.",
        morceaux=(("222", "223", "Si l'opposition du mari n'est pas justifiée"),),
        portee=(
            "la femme mariée peut exercer une profession séparée de celle de son mari ; le mari peut "
            "s'y opposer dans l'intérêt du mariage et des enfants, et il est statué sur son "
            "opposition par ordonnance du président du tribunal compétent, rendue sans frais dans les "
            "dix jours de la saisine, après audition obligatoire des parties (article 74 de "
            "l'ordonnance)"
        ),
    ),
    # Delai de viduite. Ordonnance, art. 77 : la veuve, « sous la réserve du
    # délai de viduité de 180 jours à compter du décès de son mari, peut se
    # remarier librement ». Contre : Code civil, art. 228 (« CHAP. VIII Des
    # seconds mariages », a la fin du morceau 77 du titre 5, qui reproduit
    # d'abord l'art. 77 de l'ordonnance) : delai qui court depuis la
    # dissolution du mariage precedent, abregeable quand « depuis trois cents
    # jours, le précédent mari n'a pas cohabité avec sa femme ». L'ordonnance
    # ne vise que la veuve : pour la femme divorcee, l'article garde son champ.
    # Repere pris dans le texte de l'art. 228 (« Ce délai prend fin en cas
    # d'accouchement survenu depuis le décès du mari »), pas dans l'intitule du
    # chapitre : un redecoupage qui separerait l'intitule de l'article ne doit
    # pas laisser la ligne sur un morceau sans l'art. 228.
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun, titre « Du mariage »",
        nature="abroge_partiellement",
        par_titre="l'Ordonnance N°81/002 du 29 juin 1981",
        par_date=date(1981, 6, 29),
        par_reference="ORD-1981-002",
        articles=("228",),
        par_articles=("77",),
        section="TITRE 5 Du mariage.",
        morceaux=(("77", "228", "Ce délai prend fin en cas d'accouchement"),),
        portee=(
            "la veuve peut se remarier librement sous la réserve du délai de viduité de 180 jours à "
            "compter du décès de son mari (article 77 de l'ordonnance)"
        ),
    ),
    # Reconnaissance et legitimation des enfants nes hors mariage. Ordonnance,
    # art. 41 : « Toutefois, l'accouchement vaut reconnaissance à l'égard de
    # la mère et le mariage célébré après la reconnaissance emporte
    # légitimation des enfants reconnus nés des époux. La reconnaissance et la
    # légitimation, à l'exception de la légitimation adoptive, sont fondées
    # sur le lien de sang. Quand celui-ci est établi, nul ne peut faire
    # obstacle à la reconnaissance. » ; art. 43 : « 2. Toutefois, l'enfant né
    # du commerce adultérin de sa mère ne peut être reconnu par le père
    # naturel qu'après désaveu du mari en justice. 3. Est irrecevable toute
    # action en reconnaissance d'un enfant issu d'un viol. » Contre : Code
    # civil, art. 331 (sont légitimés les enfants nés hors mariage « autres
    # que ceux nés d'un commerce adultérin », sauf trois cas) et 335 (« Cette
    # reconnaissance ne pourra avoir lieu au profit des enfants nés d'un
    # commerce incestueux ou adultérin »). La compilation renvoie elle-meme a
    # l'ordonnance en tete de la section sur la reconnaissance. Pas de
    # `par_articles` « 43 » : le morceau « 43 » de l'ordre 53 d'ORD-1981-002
    # est l'art. 48 (celebration du mariage).
    # Le morceau « 335 » porte aussi les art. 336 (reconnaissance du père
    # « sans l'indication et l'aveu de la mère ») et 337 (reconnaissance faite
    # pendant le mariage, qui « ne pourra nuire ni à celui-ci, ni aux enfants
    # nés de ce mariage »), qui ne sont pas contraires à l'ordonnance : `sauf`
    # lui retire la ligne generale et `morceaux` lui donne celle qui precise que
    # le reste de l'extrait n'est pas concerne (meme procede que le groupe
    # foncier pour le morceau « 5 » d'ORD-1974-002).
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun, titre « De la paternité et de la filiation »",
        nature="abroge_partiellement",
        par_titre="l'Ordonnance N°81/002 du 29 juin 1981",
        par_date=date(1981, 6, 29),
        par_reference="ORD-1981-002",
        articles=("331", "335"),
        par_articles=("41",),
        section="TITRE 7 De la paternité et de la filiation.",
        morceaux=(("335", "335", "commerce incestueux ou adultérin"),),
        sauf=(("335", "commerce incestueux ou adultérin"),),
        portee=(
            "la reconnaissance et la légitimation d'un enfant né hors mariage, à l'exception de la "
            "légitimation adoptive, sont fondées sur le lien de sang, et quand celui-ci est établi "
            "nul ne peut faire obstacle à la reconnaissance ; le mariage célébré après la "
            "reconnaissance emporte légitimation des enfants reconnus nés des époux ; l'enfant né du "
            "commerce adultérin de sa mère ne peut être reconnu par le père naturel qu'après désaveu "
            "du mari en justice, et toute action en reconnaissance d'un enfant issu d'un viol est "
            "irrecevable (articles 41 et 43 de l'ordonnance)"
        ),
    ),
    # Recherche de paternite. Ordonnance, art. 46 : « 1. La mère pour l'enfant
    # mineur ou l'enfant majeur peut, par une requête à la juridiction
    # compétente, m'enter une action en recherche de paternité. 2. Toutefois,
    # est irrecevable toute action en recherche de paternité lorsque pendant
    # la période légale de conception, la mère a été d'une inconduite notoire
    # ou si elle a eu un commerce avec un autre homme ou si le père prétendu
    # était dans l'impossibilité physique d'être le père. 3. A peine la
    # forclusion, l'action en recherche de paternité doit être intentée a) Par
    # la mère dans le délai de deux (2) ans à compter de l'accouchement ou du
    # jour où le père a cessé de pourvoir à l'entretien de l'enfant ; b) Par
    # l'enfant majeur dans le délai d'un (1) an à compter de sa majorité. »
    # Contre : Code civil, art. 340 (paternite declaree dans cinq cas
    # seulement ; « L'action n'appartient qu'à l'enfant ») et 342 (pas de
    # recherche quand la reconnaissance n'est pas admise ; action « pendant
    # toute la minorité de l'enfant »). Le second morceau « 342 » est l'art.
    # 342 bis (filiation deja etablie), qui n'est pas contraire : `sauf`.
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun, titre « De la paternité et de la filiation »",
        nature="abroge_partiellement",
        par_titre="l'Ordonnance N°81/002 du 29 juin 1981",
        par_date=date(1981, 6, 29),
        par_reference="ORD-1981-002",
        articles=("340", "342"),
        par_articles=("46",),
        section="TITRE 7 De la paternité et de la filiation.",
        sauf=(("342", "Lorsqu'une filiation est établie par un acte ou par un jugement"),),
        portee=(
            "l'action en recherche de paternité peut être intentée par la mère pour l'enfant mineur "
            "ou par l'enfant majeur ; elle est irrecevable si, pendant la période légale de "
            "conception, la mère a été d'une inconduite notoire ou a eu un commerce avec un autre "
            "homme, ou si le père prétendu était dans l'impossibilité physique d'être le père ; à "
            "peine de forclusion, elle doit être intentée par la mère dans le délai de deux (2) ans à "
            "compter de l'accouchement ou du jour où le père a cessé de pourvoir à l'entretien de "
            "l'enfant, et par l'enfant majeur dans le délai d'un (1) an à compter de sa majorité "
            "(article 46 de l'ordonnance)"
        ),
    ),
    # Puissance paternelle sur l'enfant ne hors mariage. Ordonnance, art. 47 :
    # « La puissance paternelle sur les enfants nés hors mariage est
    # conjointement exercée par la mère et par le père à l'égard duquel la
    # filiation a été légalement établie. En cas de désaccord, elle est exercée
    # par le parent qui a la garde effective de l'enfant sauf décision
    # contraire du juge. » Contre : Code civil, art. 383, dans le morceau 382 :
    # elle appartient a celui des pere et mere « qui les aura reconnus le
    # premier; en cas de reconnaissance simultanée par le père et la mère, le
    # père seul exerce l'autorité attachée à la puissance paternelle ».
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun, titre « De la puissance paternelle »",
        nature="abroge_partiellement",
        par_titre="l'Ordonnance N°81/002 du 29 juin 1981",
        par_date=date(1981, 6, 29),
        par_reference="ORD-1981-002",
        articles=("383",),
        par_articles=("47",),
        section="TITRE 9 De la puissance paternelle",
        morceaux=(("382", "383", "reconnaissance simultanée"),),
        portee=(
            "la puissance paternelle sur les enfants nés hors mariage est conjointement exercée par la "
            "mère et par le père à l'égard duquel la filiation a été légalement établie ; en cas de "
            "désaccord, elle est exercée par le parent qui a la garde effective de l'enfant, sauf "
            "décision contraire du juge (article 47 de l'ordonnance)"
        ),
    ),

    # ========================================================================
    # Code civil : suretes, voies d'execution, procedures collectives (Actes uniformes OHADA)
    # ========================================================================
    # Groupe cc_suretes : Code civil (compilation) et Actes uniformes OHADA des
    # suretes, des voies d'execution et des procedures collectives. Releves de
    # cc_suretes.json relus en base le 2026-10-14.
    #
    # LA COMPILATION REPRODUIT DES ACTES UNIFORMES ABROGES. Sous les titres 14 a 19
    # du livre III, le Code civil en base ne garde des articles 2011 a 2039, 2071 a
    # 2091, 2095 a 2113 et 2180 qu'une note (« abrogés implicitement et remplacés
    # par les art. ... de l'Acte uniforme ... »), puis recopie, sous leurs propres
    # numeros, le texte des Actes uniformes de 1997 et 1998 :
    # - « TITRE I Sûretés personnelles » : articles 3 a 27 de l'Acte uniforme
    #   portant organisation des suretes du 17 avril 1997 (cautionnement) ;
    # - « TITRE 17 Du nantissement. » : ses articles 63 a 105 puis 44 a 62 ;
    # - « TITRE 18 Des privilèges et hypothèques » : ses articles 106 a 116 et 117
    #   a 146 (124 et 125 deux fois), et les articles 72 a 111 de l'Acte uniforme
    #   sur les procedures collectives du 10 avril 1998 ;
    # - « TITRE IV Distribution et classement des sûretés » : ses articles 147, 148 ;
    # - « TITRE VIII : La saisie immobilière » et « TITRE IX : DISTRIBUTION DU
    #   PRIX » : articles 246 a 334 de l'Acte uniforme sur les voies d'execution du
    #   10 avril 1998, puis l'article 166 de celui sur les procedures collectives.
    # Les trois Actes uniformes revises les remplacent (l'AUS 2010 et l'AUPSRVE
    # 2023 disent « abroge » ; le debut de l'article 257 de l'AUPCAP 2015 est
    # illisible en base) ; chacun ne s'applique qu'aux suretes constituees, voies
    # d'execution engagees ou procedures ouvertes apres son entree en vigueur. D'ou
    # « abroge_partiellement » : pour les situations anterieures, l'ancien article
    # s'applique encore.
    #
    # Les numeros se recoupent dans le titre 18 : « 107 » y est l'article 107 de
    # l'Acte uniforme sur les suretes ET celui de l'Acte uniforme sur les
    # procedures collectives ; « 106 » a « 110 » sont des articles du premier,
    # « 111 » du second. D'ou les exceptions `sauf`.
    #
    # REVUE ADVERSARIALE DU 2026-10-14 (base relue morceau par morceau). Trois
    # corrections, le reste confirme :
    # - distribution du prix : l'AUPSRVE 2023 la traite aux articles 324 a 334, non
    #   335. Son article 335 est penal (« Encourt une sanction pénale le débiteur
    #   saisi ou le tiers détenteur qui ne se conforme pas aux obligations
    #   attachées à sa qualité de gardien par l'article 36 du présent acte
    #   uniforme. ») et releve du « TITRE X LES DISPOSITIONS PÉNALES, DIVERSES ET
    #   FINALES » (table des matieres, morceau LEGAL_BASIS) ; la section
    #   « TITRE IX » que la base lui donne est un retard du decoupage ;
    # - caution legale et judiciaire : l'article 14 de l'AUS 2010 regit la preuve
    #   du cautionnement (« Il se prouve par un acte comportant la signature de la
    #   caution et du créancier »), non sa forme ;
    # - expropriation forcee (2204 a 2217) : la ligne disait seulement « Hors de
    #   ce que vise ce texte, l'article s'applique encore », ce qui laissait les
    #   articles du Code civil regir sans reserve les saisies engagees avant 2024.
    #   Or l'Acte uniforme de 1998 les regissait aussi, et l'emportait sur le Code
    #   civil quand il le contredit (Traite OHADA, art. 10) : morceau 2208 « se
    #   poursuit contre le mari débiteur seul » contre morceau 250 « est
    #   poursuivie contre les deux époux » ; morceau 2212 « pendant une année »
    #   contre morceau 265 « pendant deux années ». La portee le dit desormais.
    # Ecartes relus : aucun a reintegrer (voir la revue rendue avec ce fichier).

    # Cautionnement. AUS 2010, art. 227 : « Le présent Acte uniforme, qui abroge
    # l'Acte uniforme portant organisation des sûretés du 17 avril 1997, n'est
    # applicable qu'aux sûretés consenties ou constituées après son entrée en
    # vigueur. » Code civil, morceau 2011 : « 2039: abrogés implicitement et
    # remplacés par les art. 3 à 27 de l'Acte uniforme OHADA portant [...] » ;
    # morceau 4 de ce titre : « Les dispositions du présent art. s'appliquent
    # également au cautionnement exigé par la loi de chaque Etat partie [...] ».
    # L'AUS 2010 traite du cautionnement aux articles 13 a 38 (« Chapitre 1 -
    # Cautionnement » apres l'art. 12, « Chapitre 2 - Garantie et
    # contre-garantie autonomes » apres l'art. 38).
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun",
        nature="abroge_partiellement",
        par_titre="l'Acte uniforme révisé portant organisation des sûretés du 15 décembre 2010",
        par_date=date(2010, 12, 15),
        par_reference="OHADA-AUS-2010",
        articles=("3 à 27",),
        par_articles=("227",),
        section="TITRE I Sûretés personnelles",
        portee=(
            "texte de l'Acte uniforme portant organisation des sûretés du 17 avril 1997, que reproduit "
            "le Code civil ; l'Acte uniforme révisé, qui l'abroge et traite du cautionnement aux "
            "articles 13 à 38, n'est applicable qu'aux sûretés consenties ou constituées après son "
            "entrée en vigueur (article 227)"
        ),
    ),
    # Gage et nantissements. AUS 2010, art. 227 (cite ci-dessus). Code civil,
    # morceau 2071 : « et 2072. – Ils ont été abrogées implicitement et
    # remplacées par les art. 63 à 105 de l'Acte uniforme sur le droit des
    # sûretés. » ; morceau 2073 : « à 2091 : Ces art. ont été abrogées et
    # remplacées implicitement par les art. 44 à 62 de l'Acte uniforme sur le
    # droit des sûretés. » Les seuls morceaux numerotes 44 a 105 du titre 17 sont
    # ce texte de 1997 ; 2071 et 2073 (les notes) restent hors de la plage.
    # AUS 2010 : « Chapitre 4 - Gage de meubles corporels » (art. 92 a 124),
    # « Chapitre 5 - Nantissement de meubles incorporels » (art. 125 a 178).
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun",
        nature="abroge_partiellement",
        par_titre="l'Acte uniforme révisé portant organisation des sûretés du 15 décembre 2010",
        par_date=date(2010, 12, 15),
        par_reference="OHADA-AUS-2010",
        articles=("44 à 105",),
        par_articles=("227",),
        section="TITRE 17 Du nantissement.",
        portee=(
            "texte de l'Acte uniforme portant organisation des sûretés du 17 avril 1997, que reproduit "
            "le Code civil ; l'Acte uniforme révisé, qui l'abroge et traite du gage aux articles 92 à "
            "124 et du nantissement aux articles 125 à 178, n'est applicable qu'aux sûretés consenties "
            "ou constituées après son entrée en vigueur (article 227)"
        ),
    ),
    # Privileges et hypotheques. AUS 2010, art. 227 (cite ci-dessus). Code
    # civil, morceau 106 : « à 116 de l'Acte Uniforme Ohada portant organisation
    # des Sûretés : CHAP. IV Privilèges » ; fin du morceau 2131.1 : « Les
    # hypothèques dans I'Acte Uniforme OHADA portant organisation des sûretés » ;
    # morceau 132 : « Les hypothèques forcées autres que celles prévues par le
    # présent Acte uniforme [...] ». Exceptes : les morceaux 107 et 111 de la
    # meme plage portent les articles 107 et 111 de l'Acte uniforme sur les
    # procedures collectives (lien AUPCAP 2015 de ce titre, plus bas). AUS 2010 : « Chapitre 6
    # - Privilèges » (art. 179 a 189), « Titre 3 - Hypothèques » (art. 190 a 223).
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun",
        nature="abroge_partiellement",
        par_titre="l'Acte uniforme révisé portant organisation des sûretés du 15 décembre 2010",
        par_date=date(2010, 12, 15),
        par_reference="OHADA-AUS-2010",
        articles=("106 à 146",),
        par_articles=("227",),
        section="TITRE 18 Des privilèges et hypothèques",
        sauf=(
            ("107", "cessation des paiements déclarée par décision de justice"),
            ("111", "ordre des licenciements établi par le syndic"),
        ),
        portee=(
            "texte de l'Acte uniforme portant organisation des sûretés du 17 avril 1997, que reproduit "
            "le Code civil ; l'Acte uniforme révisé, qui l'abroge et traite des privilèges aux articles "
            "179 à 189 et des hypothèques aux articles 190 à 223, n'est applicable qu'aux sûretés "
            "consenties ou constituées après son entrée en vigueur (article 227)"
        ),
    ),
    # Distribution des deniers. AUS 2010, art. 227 (cite ci-dessus). Code civil,
    # morceau 2204 : « Arts. 147 et 148 de l'AU OHADA portant organisation des
    # sûretés ». AUS 2010 : « Titre 4 - Distribution des deniers et classement
    # des sûretés » (art. 224 a 226).
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun",
        nature="abroge_partiellement",
        par_titre="l'Acte uniforme révisé portant organisation des sûretés du 15 décembre 2010",
        par_date=date(2010, 12, 15),
        par_reference="OHADA-AUS-2010",
        articles=("147", "148"),
        par_articles=("227",),
        section="TITRE IV Distribution et classement des sûretés",
        portee=(
            "texte de l'Acte uniforme portant organisation des sûretés du 17 avril 1997, que reproduit "
            "le Code civil ; l'Acte uniforme révisé, qui l'abroge et traite de la distribution des "
            "deniers aux articles 224 à 226, n'est applicable qu'aux sûretés consenties ou constituées "
            "après son entrée en vigueur (article 227)"
        ),
    ),
    # Caution legale et judiciaire (Code civil, art. 2040 a 2043, tous dans le
    # morceau 2040, range en base sous « TITRE I Sûretés personnelles » ; aucune
    # note d'abrogation). AUS 2010, art. 14, dernier alinea : « Les dispositions
    # du présent article s'appliquent également au cautionnement exigé par la
    # loi de chaque Etat Partie ou par une décision de justice. » ; art. 15 :
    # « Lorsque le débiteur est tenu, par la convention, la loi de chaque Etat
    # Partie ou la décision de justice, de fournir une caution, celle-ci doit
    # être domiciliée ou faire élection de domicile dans le ressort territorial
    # de la juridiction où elle doit être fournie, sauf dispense du créancier ou
    # de la juridiction compétente. La caution doit présenter des garanties de
    # solvabilité appréciées en tenant compte de tous les éléments de son
    # patrimoine. Le débiteur qui ne peut trouver une caution pourra la remplacer
    # par toute sûreté réelle donnant les mêmes garanties au créancier. » ;
    # art. 27 : « La caution judiciaire et la caution solidaire ne disposent pas
    # du bénéfice de discussion. » L'Acte uniforme de 1997 reproduit dans ce
    # titre disait deja de meme (morceaux 4, 5, 6 et, dans le 15, « Art. 16.- La
    # caution judiciaire et la caution solidaire ne disposent pas du bénéfice de
    # discussion. »). Traite OHADA, art. 10 : « Les actes uniformes sont
    # directement applicables et obligatoires dans les Etats Parties, nonobstant
    # toute disposition contraire de droit interne, antérieure ou postérieure. »
    # Revue : l'article 14 de l'AUS 2010 dit « Le cautionnement ne se présume
    # pas, quelle que soit la nature de l'obligation garantie. » puis « Il se
    # prouve par un acte comportant la signature de la caution et du créancier
    # [...] » : il regit la preuve de l'engagement, d'ou « preuve » au lieu de
    # « forme ».
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun",
        nature="abroge_partiellement",
        par_titre="l'Acte uniforme révisé portant organisation des sûretés du 15 décembre 2010",
        par_date=date(2010, 12, 15),
        par_reference="OHADA-AUS-2010",
        articles=("2040 à 2043",),
        par_articles=("14", "15", "27"),
        section="TITRE I Sûretés personnelles",
        portee=(
            "pour le cautionnement exigé par la loi ou par une décision de justice, l'Acte uniforme "
            "révisé régit la preuve de l'engagement (article 14) et la caution à fournir (article 15 : "
            "domicile ou élection de domicile dans le ressort de la juridiction où elle doit être "
            "fournie, sauf dispense, garanties de solvabilité, remplacement possible par toute sûreté "
            "réelle donnant les mêmes garanties) et refuse à la caution judiciaire le bénéfice de "
            "discussion (article 27), comme avant lui l'Acte uniforme du 17 avril 1997 (articles 4 à "
            "6 et 16)"
        ),
    ),
    # Saisie immobiliere. AUPSRVE 2023, art. 337 : « Le présent acte uniforme,
    # qui abroge et remplace l'Acte uniforme portant organisation des procédures
    # simplifiées de recouvrement et des voies d'exécution, adopté le 10 avril
    # 1998, n'est applicable qu'aux procédures simplifiées de recouvrement et
    # aux voies d'exécution engagées après son entrée en vigueur. Les procédures
    # simplifiées de recouvrement et les voies d'exécution engagées avant son
    # entrée en vigueur demeurent soumises à la législation alors en vigueur. »
    # Code civil, morceau 148 : « Arts. 246 à 334 de l'Acte Uniforme OHADA
    # portant organisation des voies d'exécution » ; tout le titre VIII est ce
    # texte de 1998. AUPSRVE 2023 : « TITRE VIII LA SAISIE IMMOBILIÈRE »
    # (art. 246 a 323).
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun",
        nature="abroge_partiellement",
        par_titre=(
            "l'Acte uniforme portant organisation des procédures simplifiées de recouvrement et des "
            "voies d'exécution, révisé, publié au Journal officiel de l'OHADA du 15 novembre 2023"
        ),
        par_date=date(2023, 11, 15),
        par_reference="OHADA-AUPSRVE-2023",
        articles=("246 à 323",),
        par_articles=("337",),
        section="TITRE VIII : La saisie immobilière",
        portee=(
            "texte de l'Acte uniforme portant organisation des procédures simplifiées de recouvrement "
            "et des voies d'exécution adopté le 10 avril 1998, que reproduit le Code civil ; l'Acte "
            "uniforme révisé, qui l'abroge et le remplace et traite de la saisie immobilière aux "
            "articles 246 à 323, n'est applicable qu'aux voies d'exécution engagées après son entrée "
            "en vigueur, celles engagées avant demeurant soumises à la législation alors en vigueur "
            "(article 337)"
        ),
    ),
    # Distribution du prix. AUPSRVE 2023, art. 337 (cite ci-dessus). Code civil,
    # titre IX : morceaux 324 a 334 du texte de 1998 ; les morceaux 2218 (Code
    # civil) et 166 (procedures collectives) restent hors de la plage.
    # AUPSRVE 2023 : « TITRE IX LA DISTRIBUTION DU PRIX » (art. 324 a 334 ;
    # l'art. 335, « Encourt une sanction pénale le débiteur saisi ou le tiers
    # détenteur [...] », ouvre le « TITRE X LES DISPOSITIONS PÉNALES, DIVERSES
    # ET FINALES » malgre la section que lui donne la base).
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun",
        nature="abroge_partiellement",
        par_titre=(
            "l'Acte uniforme portant organisation des procédures simplifiées de recouvrement et des "
            "voies d'exécution, révisé, publié au Journal officiel de l'OHADA du 15 novembre 2023"
        ),
        par_date=date(2023, 11, 15),
        par_reference="OHADA-AUPSRVE-2023",
        articles=("324 à 334",),
        par_articles=("337",),
        section="TITRE IX : DISTRIBUTION DU PRIX",
        portee=(
            "texte de l'Acte uniforme portant organisation des procédures simplifiées de recouvrement "
            "et des voies d'exécution adopté le 10 avril 1998, que reproduit le Code civil ; l'Acte "
            "uniforme révisé, qui l'abroge et le remplace et traite de la distribution du prix aux "
            "articles 324 à 334, n'est applicable qu'aux voies d'exécution engagées après son entrée "
            "en vigueur, celles engagées avant demeurant soumises à la législation alors en vigueur "
            "(article 337)"
        ),
    ),
    # Expropriation forcee (Code civil, art. 2204 a 2217). AUPSRVE 2023,
    # art. 336 : « Sauf dans les cas où il est renvoyé aux stipulations des
    # conventions internationales ou aux règles applicables dans les États
    # parties, seules les dispositions du présent acte uniforme sont applicables
    # aux procédures et mesures conservatoires ou d'exécution qu'il régit. » ;
    # art. 337 (cite ci-dessus) ; art. 246 : « Le créancier ne peut faire vendre
    # les immeubles appartenant à son débiteur qu'en respectant les formalités
    # prescrites par les dispositions qui suivent. » La note du Code civil
    # (morceau 2204 : « à 2217 : - Ils sont complétés implicitement par les art.
    # 147 et 148 AUDS, et les arts 246 à 334 AUVE ») ne vise que l'Acte uniforme
    # de 1998. Le debut de l'article 2204 (« [...] 2º de l'usufruit appartenant
    # au débiteur sur les biens de même nature ») est dans le morceau SECTION_26 ;
    # le 2210 est dans le 2209, le 2217 dans le 2216.
    # Revue : « Hors de ce que vise ce texte, l'article s'applique encore » vise
    # les saisies engagees avant l'entree en vigueur, que l'art. 337 laisse a
    # « la législation alors en vigueur ». Celle-ci comprenait l'Acte uniforme
    # de 1998 (articles 246 a 334 recopies au titre VIII et au titre IX), que le
    # Traite OHADA, art. 10, rend applicable « nonobstant toute disposition
    # contraire de droit interne, antérieure ou postérieure » ; plusieurs
    # articles du titre 19 le contredisent (2208 « se poursuit contre le mari
    # débiteur seul » / 250 « est poursuivie contre les deux époux » ; 2212
    # « pendant une année » / 265 « pendant deux années »). La portee le dit,
    # pour que la consequence affichee ne les fasse pas appliquer sans reserve.
    # « 2204 » reste seul dans `articles` : le test des liens saisis exige que
    # l'article d'un `morceaux` y figure tel quel.
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun",
        nature="abroge_partiellement",
        par_titre=(
            "l'Acte uniforme portant organisation des procédures simplifiées de recouvrement et des "
            "voies d'exécution, révisé, publié au Journal officiel de l'OHADA du 15 novembre 2023"
        ),
        par_date=date(2023, 11, 15),
        par_reference="OHADA-AUPSRVE-2023",
        articles=("2204", "2205 à 2217"),
        par_articles=("336", "337"),
        section="TITRE 19 De l'expropriation forcée - et des ordres entre les créanciers.",
        morceaux=(("SECTION_26", "2204", "usufruit appartenant au débiteur"),),
        portee=(
            "seules les dispositions de l'Acte uniforme révisé, qui traite de la saisie immobilière aux "
            "articles 246 à 323, sont applicables aux saisies immobilières engagées après son entrée en "
            "vigueur, sauf renvoi aux conventions internationales ou aux règles applicables dans les "
            "États parties (article 336) ; celles engagées avant demeurent soumises à la législation "
            "alors en vigueur (article 337), dont les articles 246 à 334 de l'Acte uniforme sur les "
            "voies d'exécution du 10 avril 1998, que reproduit le Code civil, applicables nonobstant "
            "toute disposition contraire de droit interne (article 10 du Traité OHADA)"
        ),
    ),
    # Procedures collectives, titre 18. AUPCAP 2015, art. 257, dont le debut est
    # illisible en base : « [...] du passif du 10 avril 1998, n'est applicable
    # qu'aux procédures collectives ouvertes après son entrée en vigueur. »
    # Code civil, morceau 72 : « à 111 de l'acte uniforme portant organisation
    # des procédures collectives d'apurement du passif : CHAP. IV : Effets de la
    # décision d'ouverture à l'égard des créanciers » ; morceau 1864 (titre 9) :
    # « Voir A.U. OHADA du 10 avril 1998 portant organisation des procédures
    # collectives d'appurement du passif ». L'AUPCAP 2015 garde le meme plan aux
    # articles 72 a 111 (sections « Constitution de la masse et effets
    # suspensifs », « Production et vérification des créances », « Cautions et
    # coobligés », « Super privilège des salariés », « Droit de résiliation et
    # privilège du bailleur d'immeuble », « Droits du conjoint », « Droits du
    # vendeur de meubles et revendications », « Contrats en cours et
    # licenciements »). Exceptes : les morceaux 106 a 110 de la plage portent
    # l'Acte uniforme de 1997 sur les suretes (lien de ce titre plus haut).
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun",
        nature="abroge_partiellement",
        par_titre=(
            "l'Acte uniforme portant organisation des procédures collectives d'apurement du passif "
            "du 10 septembre 2015"
        ),
        par_date=date(2015, 9, 10),
        par_reference="OHADA-AUPCAP-2015",
        articles=("72 à 111",),
        par_articles=("257",),
        section="TITRE 18 Des privilèges et hypothèques",
        sauf=(
            ("106", "privilèges généraux"),
            ("107", "Sont privilégiés, sans publicité"),
            ("108", "privilégiées au delà du montant"),
            ("109", "titulaires de privilèges spéciaux"),
            ("110", "sur le meuble vendu"),
        ),
        portee=(
            "texte de l'Acte uniforme portant organisation des procédures collectives d'apurement du "
            "passif du 10 avril 1998, que reproduit le Code civil ; l'Acte uniforme du 10 septembre "
            "2015, qui traite des mêmes questions aux articles 72 à 111, n'est applicable qu'aux "
            "procédures collectives ouvertes après son entrée en vigueur (article 257)"
        ),
    ),
    # Procedures collectives, titre IX. AUPCAP 2015, art. 257 (cite ci-dessus) ;
    # art. 166 : « Les deniers provenant de la réalisation des immeubles sont
    # distribués dans l'ordre suivant : [...] ». Code civil, morceau 2218 :
    # « Il est complété par l'art. 166 de l'AU OHADA sur les procédures
    # collectives », puis morceau 166 : « de l'acte uniforme sur les procédures
    # collectives Les deniers provenant de la réalisation des immeubles sont
    # distribués ainsi : [...] » (redaction de 1998).
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun",
        nature="abroge_partiellement",
        par_titre=(
            "l'Acte uniforme portant organisation des procédures collectives d'apurement du passif "
            "du 10 septembre 2015"
        ),
        par_date=date(2015, 9, 10),
        par_reference="OHADA-AUPCAP-2015",
        articles=("166",),
        par_articles=("257",),
        section="TITRE IX : DISTRIBUTION DU PRIX",
        portee=(
            "texte de l'Acte uniforme portant organisation des procédures collectives d'apurement du "
            "passif du 10 avril 1998, que reproduit le Code civil ; l'Acte uniforme du 10 septembre "
            "2015, qui fixe à son article 166 l'ordre de distribution des deniers provenant de la "
            "réalisation des immeubles, n'est applicable qu'aux procédures collectives ouvertes après "
            "son entrée en vigueur (article 257)"
        ),
    ),

    # ========================================================================
    # Code civil : autres textes speciaux (vente et bail commerciaux, prescription, societes, transport, assurances, travail, commerce electronique, OPCVM)
    # ========================================================================

    # Code civil, article 815, et loi 2016/010 sur les OPCVM. Art. 19 (1) :
    # « LeFCPest une copropriété de valeurs mobilières et de dépôts […]. Ce
    # fonds n'a pas de personnalité juridique. » ; (2) : « Les dispositions du
    # Code Civil relatives à l'indivision ne s'appliquent pas au fonds. » ;
    # (3) : « Les dispositions de l'alinéa 2 ci-dessus s'appliquent aux
    # héritiers, à leurs ayants droits et à leurs créanciers. » ; art. 28 (1) :
    # « Les porteurs de parts, leurs héritiers,leurs ayants droit et leurs
    # créanciers ne peuvent provoquer le partage du Fonds. » (l'article 815
    # dit au contraire que « le partage peut être toujours provoqué »). Seul
    # le fonds commun de placement échappe à l'article 815 : toute autre
    # indivision y reste soumise. Le morceau 815 porte aussi, a sa fin, le
    # debut illisible de l'article 816 (partage), que la derogation vise
    # egalement.
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun",
        nature="abroge_partiellement",
        par_titre=(
            "la Loi N°2016/010 du 12 juillet 2016 régissant les organismes de placement "
            "collectif en valeurs mobilières"
        ),
        par_date=date(2016, 7, 12),
        par_reference="PRC-4724",
        articles=("815",),
        section="TITRE 1 Des successions.",
        par_articles=("19", "28"),
        portee=(
            "les dispositions du Code civil relatives à l'indivision ne s'appliquent pas au fonds "
            "commun de placement, copropriété de valeurs mobilières et de dépôts sans personnalité "
            "juridique (article 19), dont les porteurs de parts, leurs héritiers, leurs ayants droit "
            "et leurs créanciers ne peuvent provoquer le partage (article 28)"
        ),
    ),
    # Code civil, preuve litterale (articles 1317 a 1340), et loi 2010/021 sur
    # le commerce electronique. Art. 13 (1) : « Lorsqu'un écrit est exigé pour
    # la validité d'un acte juridique, il peut être établi et conservé sous
    # forme électronique dans les conditions prévues aux articles 1317 et
    # suivants du Code civil, relatifs à la preuve littérale. » ; art. 13 (2),
    # fin : « exception faite aux dispositions de l'alinéa 1 ci-dessus pour :
    # - les actes sous seing privé relatifs au droit de la famille et des
    # successions ; - les actes sous seing privé relatifs aux sûretés
    # personnelles ou réelles, de nature civile ou commerciale, sauf s'ils
    # sont passés par une personne pour les besoins de sa profession. » La loi
    # ne reecrit aucun article : elle ajoute l'ecrit electronique et renvoie
    # aux conditions du Code, qui restent en vigueur ; d'ou « complete » (la
    # verification retenait « modifie » « faute d'autre catégorie »). Sans
    # les exceptions de l'alinea 2, la ligne laisserait croire qu'un acte
    # sous seing prive de famille, de succession ou de surete civile peut
    # etre electronique. La fin de l'article 1317 est en base dans le morceau
    # 1316.3.
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun",
        nature="complete",
        par_titre="la Loi N°2010/021 du 21 décembre 2010 régissant le commerce électronique",
        par_date=date(2010, 12, 21),
        par_reference="LOI-2010-021",
        articles=("1317 à 1340",),
        section="TITRE 3 Des contrats ou obligations conventionnelles en général.",
        par_articles=("13",),
        morceaux=(("1316.3", "1317", "avec les solennités requises"),),
        portee=(
            "lorsqu'un écrit est exigé pour la validité d'un acte juridique, il peut être établi et "
            "conservé sous forme électronique dans les conditions prévues aux articles 1317 et "
            "suivants du Code civil, relatifs à la preuve littérale (article 13, alinéa 1), exception "
            "faite des actes sous seing privé relatifs au droit de la famille et des successions et "
            "des actes sous seing privé relatifs aux sûretés personnelles ou réelles, de nature "
            "civile ou commerciale, sauf s'ils sont passés par une personne pour les besoins de sa "
            "profession (article 13, alinéa 2)"
        ),
    ),
    # Code civil, articles 1382 a 1384, et Code CIMA. Art. 225 : « Les
    # dispositions du présent Code s'appliquent, même lorsqu'elles sont
    # transportées en vertu d'un contrat, aux victimes d'un accident causé par
    # un véhicule terrestre à moteur ainsi que ses remorques ou
    # semi-remorques. » ; art. 226 : « Les victimes, y compris les conducteurs
    # ne peuvent se voir opposer la force majeure ou le fait d'un tiers par le
    # conducteur ou le gardien d'un véhicule mentionné à l'article 225. » ;
    # art. 257 : « Les seuls préjudices susceptibles d'être indemnisés sont
    # ceux mentionnés aux articles 258 à 266. » ; Traite CIMA, art. 47 (meme
    # document, morceau 39.1) : « Les juridictions nationales appliquent les
    # dispositions du présent Traité et les actes établis par les organes de
    # la Conférence nonobstant toute disposition nationale contraire
    # antérieure ou postérieure à ces textes. » Pas de clause d'abrogation ni
    # de note de la compilation : regime special des accidents de vehicules
    # terrestres a moteur, le droit commun restant la regle pour tout autre
    # dommage.
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun",
        nature="abroge_partiellement",
        par_titre=(
            "le Code des assurances des États membres de la CIMA, annexé au Traité du "
            "10 juillet 1992"
        ),
        par_date=date(1992, 7, 10),
        par_reference="CIMA-CODE-2019",
        articles=("1382 à 1384",),
        section="TITRE 4 Des engagements qui se forment sans convention.",
        par_articles=("225", "226", "227", "228", "257"),
        portee=(
            "pour les victimes d'un accident causé par un véhicule terrestre à moteur ou ses "
            "remorques ou semi-remorques, le Code CIMA s'applique (article 225) ; la force majeure "
            "et le fait d'un tiers ne peuvent leur être opposés par le conducteur ou le gardien du "
            "véhicule (article 226), la faute du conducteur ou de la victime ne joue que dans les "
            "conditions des articles 227 et 228, et seuls sont indemnisés les préjudices mentionnés "
            "aux articles 258 à 266 (article 257)"
        ),
    ),
    # Code civil, titre De la vente, et Acte uniforme droit commercial general
    # 2010. Art. 234 : « Les dispositions du présent Livre s'appliquent aux
    # contrats de vente de marchandises entre commerçants, personnes physiques
    # ou personnes morales, y compris les contrats de fourniture de
    # marchandises destinées à des activités de fabrication ou de
    # production. » ; art. 237 : « La vente commerciale est soumise aux règles
    # du droit commun des contrats et de la vente qui ne sont pas contraires
    # aux dispositions du présent Livre. » Le Livre 8 ne regit que la vente de
    # marchandises : ne sont pas vises les articles qui ne regissent que la
    # vente d'immeubles (1617 a 1623, contenance : « Si la vente d'un immeuble
    # a été faite avec indication de la contenance » ; 1634, 1635 et 1637,
    # « fonds » ; 1638, « l'héritage vendu » ; 1655 et 1656, « la vente
    # d'immeubles »), ni l'article 1607 (« La tradition des droits
    # incorporels »), dont le morceau porte la fin de l'article 1608 (frais de
    # delivrance), d'ou le morceau range. Les articles 1594 à 1601 (capacite,
    # chose d'autrui, chose perie) et 1658 et suivants (remere, lesion,
    # licitation, cession de creances) ne sont pas traites par le Livre 8.
    # L'article 1582 est dans l'en-tete du titre (SECTION_16) ; le morceau
    # « 1657 » du titre 8 (en fait l'article 1757) est hors de la section.
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun",
        nature="abroge_partiellement",
        par_titre="l'Acte uniforme révisé portant sur le droit commercial général du 15 décembre 2010",
        par_date=date(2010, 12, 15),
        par_reference="OHADA-AUDCG-2010",
        articles=(
            "1582 à 1593", "1602 à 1606", "1608", "1609 à 1616", "1624 à 1633", "1636",
            "1639 à 1654", "1657",
        ),
        section="TITRE 6 De la vente.",
        par_articles=("234", "237"),
        morceaux=(
            ("SECTION_16", "1582", "Elle peut être faite par acte authentique ou sous seing privé"),
            ("1607", "1608", "de l'acheteur, s'il n'y a eu stipulation contraire"),
        ),
        portee=(
            "pour les contrats de vente de marchandises entre commerçants, personnes physiques ou "
            "personnes morales, y compris les contrats de fourniture de marchandises destinées à "
            "des activités de fabrication ou de production (article 234), hors ceux qu'excluent les "
            "articles 235 et 236, le Livre 8 de l'Acte uniforme (vente commerciale) s'applique, et "
            "les règles du droit commun des contrats et de la vente n'y valent qu'en ce qu'elles ne "
            "lui sont pas contraires (article 237)"
        ),
    ),
    # Code civil, regles communes du louage des choses et baux a loyer
    # (articles 1714 a 1762), et Acte uniforme droit commercial general 2010.
    # Art. 103 : « Est réputé bail à usage professionnel toute convention,
    # écrite ou non, entre une personne investie par la loi ou une convention
    # du droit de donner en location tout ou partie d'un immeuble compris dans
    # le champ d'application du présent Titre, et une autre personne physique
    # ou morale, permettant à celle-ci, le preneur, d'exercer dans les lieux
    # avec l'accord de celle-là, le bailleur, une activité commerciale,
    # industrielle, artisanale ou toute autre activité professionnelle. » ;
    # art. 134 : « Sont d'ordre public les dispositions des articles 101, 102,
    # 103, 107, 110, 111, 117, 123, 124, 125, 126, 127, 130 et 133 du présent
    # Acte uniforme. » ; Traite OHADA, art. 10 : « Les actes uniformes sont
    # directement applicables et obligatoires dans les Etats Parties,
    # nonobstant toute disposition contraire de droit interne, antérieure ou
    # postérieure. » L'article 1757 (bail des meubles, morceau « 1657 ») n'est
    # pas vise : l'Acte uniforme ne regit que la location d'immeubles.
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun",
        nature="abroge_partiellement",
        par_titre="l'Acte uniforme révisé portant sur le droit commercial général du 15 décembre 2010",
        par_date=date(2010, 12, 15),
        par_reference="OHADA-AUDCG-2010",
        articles=("1714 à 1762",),
        section="TITRE 8 Du contrat de louage.",
        par_articles=("101", "103", "134"),
        portee=(
            "pour un bail à usage professionnel, c'est-à-dire la location de tout ou partie d'un "
            "immeuble où le preneur exerce une activité commerciale, industrielle, artisanale ou "
            "toute autre activité professionnelle (article 103), les articles 101 à 134 de l'Acte "
            "uniforme s'appliquent nonobstant toute disposition contraire de droit interne "
            "(article 10 du Traité OHADA), et ceux qu'énumère l'article 134 sont d'ordre public"
        ),
    ),
    # Code civil, article 1780 (louage des domestiques et ouvriers), et Code du
    # travail. Art. 176 (1) (morceau 174) : « Sont abrogées, toutes les
    # dispositions antérieures contraires à celles de la présente loi » ;
    # art. 24 (1), numero illisible en base, dans le contenu du morceau
    # SECTION_2 (et dans l'intitule de section des morceaux suivants) :
    # « tout contrat de travail conclu pour être exécuté au Cameroun, est
    # soumis aux dispositions de la présente loi » ; art. 25 (morceau 25) :
    # « Le contrat de travail peut être conclu pour une durée déterminée ou
    # indéterminée » ; « Section III: De la suspension et de la résiliation
    # du contrat de travail ». L'article 1780 n'a pas de morceau propre : son
    # debut, illisible, suit dans le morceau 1779 l'intitule de sa section.
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun",
        nature="abroge_partiellement",
        par_titre="la Loi N°92/007 du 14 août 1992 portant Code du travail",
        par_date=date(1992, 8, 14),
        par_reference="LOI-1992-007",
        articles=("1780",),
        section="TITRE 8 Du contrat de louage.",
        # SECTION_2 porte l'article 24, le morceau 25 l'article 25, le
        # morceau 174 l'article 176.
        par_articles=("SECTION_2", "25", "174"),
        morceaux=(("1779", "1780", "Du louage des domestiques et ouvriers"),),
        portee=(
            "tout contrat de travail conclu pour être exécuté au Cameroun est soumis au Code du "
            "travail, qui en règle la durée, la suspension et la résiliation, et dont l'article 176 "
            "abroge les dispositions antérieures contraires"
        ),
    ),
    # Code civil, voituriers par terre et par eau (articles 1782 a 1786), et
    # Acte uniforme transport routier 2003. Art. 1er, 1°) : « Le présent Acte
    # uniforme s'applique à tout contrat de transport de marchandises par
    # route lorsque le lieu de prise en charge de la marchandise et le lieu
    # prévu pour la livraison, tels qu'ils sont indiqués au contrat, sont
    # situés soit sur le territoire d'un État membre de l'OHADA, soit sur le
    # territoire de deux États différents dont l'un au moins est membre de
    # l'OHADA. » ; 2°) : « L'Acte uniforme ne s'applique pas aux transports de
    # marchandises dangereuses, aux transports funéraires, […] » ; art. 16,
    # 1°) : « Le transporteur est tenu de livrer la marchandise à destination.
    # Il est responsable de l'avarie, de la perte totale ou partielle qui se
    # produit pendant la période de transport, ainsi que du retard à la
    # livraison. » Les articles 1785 et 1786 sont dans le morceau 1784.
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun",
        nature="abroge_partiellement",
        par_titre=(
            "l'Acte uniforme relatif aux contrats de transport de marchandises par route du "
            "22 mars 2003"
        ),
        par_date=date(2003, 3, 22),
        par_reference="OHADA-AUCTMR-2003",
        articles=("1782 à 1786",),
        section="TITRE 8 Du contrat de louage.",
        par_articles=("1", "16 à 21"),
        portee=(
            "pour un contrat de transport de marchandises par route dont le lieu de prise en charge "
            "et le lieu prévu pour la livraison sont situés dans un État membre de l'OHADA, ou dans "
            "deux États dont l'un au moins en est membre, hors les exclusions de l'article 1er "
            "(notamment marchandises dangereuses et transports funéraires), la responsabilité du "
            "transporteur pour l'avarie, la perte ou le retard suit les articles 16 à 21 de l'Acte "
            "uniforme ; le transport de personnes et le transport par eau n'en relèvent pas"
        ),
    ),
    # Code civil, titre Du contrat de societe, et Acte uniforme societes
    # commerciales 2014. Art. 1er : « Toute société commerciale, y compris
    # celle dans laquelle un État ou une personne morale de droit public est
    # associé, dont le siège social est situé sur le territoire de l'un des
    # États parties […] est soumise aux dispositions du présent Acte uniforme.
    # Tout groupement d'intérêt économique est également soumis aux
    # dispositions du présent Acte uniforme. » ; art. 919 : « Est abrogé, […]
    # l'Acte uniforme du 17 avril 1997 relatif au droit des sociétés
    # commerciales et du groupement d'intérêt économique. » ; Code civil, art.
    # 1873 : « Les dispositions du présent titre ne s'appliquent aux sociétés
    # de commerce que dans les points qui n'ont rien de contraire aux lois et
    # usages du commerce. », suivi en base d'un renvoi a l'Acte uniforme de
    # 1997. L'article 1862 n'est pas vise : il ne regit que « les sociétés
    # autres que celles de commerce ».
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun",
        nature="abroge_partiellement",
        par_titre=(
            "l'Acte uniforme révisé relatif au droit des sociétés commerciales et du groupement "
            "d'intérêt économique du 30 janvier 2014"
        ),
        par_date=date(2014, 1, 30),
        par_reference="OHADA-AUSCGIE-2014",
        articles=("1832 à 1861", "1863 à 1873"),
        section="TITRE 9 Du contrat de société.",
        par_articles=("1", "919"),
        portee=(
            "toute société commerciale ayant son siège social dans un État partie et tout "
            "groupement d'intérêt économique sont soumis à l'Acte uniforme (article 1er), qui abroge "
            "celui du 17 avril 1997 auquel renvoie la compilation (article 919) ; le titre 9 du Code "
            "civil ne s'applique aux sociétés de commerce que dans les points qui n'ont rien de "
            "contraire aux lois et usages du commerce (article 1873)"
        ),
    ),
    # Code civil, titre De la prescription, et Acte uniforme droit commercial
    # general 2010. Art. 16 : « Les obligations nées à l'occasion de leur
    # commerce entre commerçants, ou entre commerçants et non-commerçants, se
    # prescrivent par cinq ans si elles ne sont pas soumises à des
    # prescriptions plus courtes. » ; art. 33 : « Les obligations nées à
    # l'occasion de leurs activités entre entreprenants, ou entre entreprenants
    # et non entreprenants, se prescrivent par cinq ans […]. Le régime de la
    # prescription prévu aux articles 17 à 29 du présent Acte uniforme
    # s'applique à l'entreprenant. » Ne sont vises que les articles dont
    # l'Acte uniforme regle lui-meme l'objet, d'apres son texte en base :
    # renonciation et moyen souleve d'office (2220 a 2225 ; articles 26 et
    # 28), interruption et regle generale de suspension (2242 a 2251 ;
    # articles 21 a 25), creance conditionnelle ou a terme (2257 ; article
    # 19), prescription trentenaire (2262 ; article 16).
    # Le morceau 2243 commence par l'article 2243 (interruption naturelle,
    # propre a la prescription acquisitive), mais porte ensuite les articles
    # 2244 a 2246 (citation en justice, conciliation, juge incompetent), que
    # les articles 21 et 23 reglent : il reste marque.
    # Restent hors du lien : definition (2219), domaine public et Etat (2226,
    # 2227), mineurs, epoux et successions (2252 a 2256, 2258, 2259), calcul
    # du delai (2260, 2261 : l'article 18, qui en traiterait, est illisible en
    # base), rentes (2263, 2264), prescriptions plus courtes (2271 et
    # suivants, que l'article 16 reserve) et prescription acquisitive.
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun",
        nature="abroge_partiellement",
        par_titre="l'Acte uniforme révisé portant sur le droit commercial général du 15 décembre 2010",
        par_date=date(2010, 12, 15),
        par_reference="OHADA-AUDCG-2010",
        articles=("2220 à 2225", "2242 à 2251", "2257", "2262"),
        section="TITRE 20 De la prescription.",
        par_articles=("16", "33"),
        portee=(
            "les obligations nées à l'occasion de leur commerce entre commerçants, ou entre "
            "commerçants et non-commerçants, se prescrivent par cinq ans si elles ne sont pas "
            "soumises à des prescriptions plus courtes (article 16), selon les règles des articles "
            "17 à 29 sur le point de départ, la suspension, l'interruption et la renonciation ; il "
            "en va de même des obligations nées des activités des entreprenants (article 33)"
        ),
    ),
    # Code civil, articles 2271 et 2272, et Code du travail. Art. 74 (1) :
    # « L'action en paiement du salaire se prescrit par trois (3) ans. A
    # l'égard de la prescription, les indemnités liées à la rupture de contrat
    # de travail sont assimilées au salaire. » (morceau 67.3) ; art. 176 (1) :
    # « Sont abrogées, toutes les dispositions antérieures contraires à celles
    # de la présente loi » (morceau 174). Seules les actions des ouvriers et
    # gens de travail (2271) et des domestiques (2272) en paiement de leur
    # salaire sont visees.
    Lien(
        texte="CODE-CIVIL",
        texte_nom="le Code civil applicable au Cameroun",
        nature="abroge_partiellement",
        par_titre="la Loi N°92/007 du 14 août 1992 portant Code du travail",
        par_date=date(1992, 8, 14),
        par_reference="LOI-1992-007",
        articles=("2271", "2272"),
        section="TITRE 20 De la prescription.",
        # « 67.3 » designe aussi les morceaux 67.1 et 67.2 (meme numero de
        # base) : la ligne y reste exacte.
        par_articles=("67.3", "174"),
        portee=(
            "pour l'action des ouvriers et gens de travail (article 2271) et des domestiques "
            "(article 2272) en paiement de leur salaire, la prescription est de trois ans "
            "(article 74 du Code du travail), les indemnités liées à la rupture du contrat de "
            "travail étant assimilées au salaire"
        ),
    ),

    # ========================================================================
    # Liberte d'association et cybersecurite
    # ========================================================================
    # Groupe association_cyber : liberte d'association (LOI-1990-053) et
    # cybersecurite (LOI-2010-012). Releves et contre-verification :
    # association_cyber.json ; chaque preuve ci-dessous a ete relue en base.
    # Version revue (association_cyber_corrige.py) : portee de l'article 81
    # completee (articles 37 et 43 de la loi 2023/009), alinea 2 de l'article 74
    # ajoute au lien de la loi 2024/017, passage illisible de la loi 2021/022
    # situe, intitule de la loi 2020/009 retabli.
    #
    # Ecartes, et pourquoi :
    # - Loi 90/053, article 5, et loi n° 99/011 du 20 juillet 1999 (absente du
    #   corpus ; recopiee dans le morceau SIGNATURE de LOI-1990-053), puis loi
    #   2020/009 (PRC-8413, art. 1er) : leur redaction de l'alinea 4 n'est plus
    #   en vigueur, la loi 2021/022 l'a reecrit en dernier lieu. « Pour ce qui a
    #   changé, ce texte fait foi » serait faux sous leur nom. Seul le lien de
    #   2021 est saisi ; il marque aussi la copie de la loi de 1999 (comme la loi
    #   2008/001 ne garde pas les articles 6 et 53 de la Constitution reecrits en
    #   2026).
    # - Loi 90/053 et loi 2020/009, art. 2 (« abroge toutes les dispositions
    #   antérieures contraires, notamment celles de la loi n° 90/053 ») : aucun
    #   article designe, et la seule disposition contraire est l'ancien alinea 4
    #   de l'article 5, reecrit par l'art. 1er puis par la loi 2021/022.
    # - Loi 90/053, article 17, et loi de finances 2026 (PRC-10557, art. 32) :
    #   elle modifie l'article VINGT-UNIEME de la loi de finances 2023 et renvoie a
    #   la loi 90/053 (« sont encadrées par la loi N°90/053 ») sans la modifier ;
    #   la validite de cinq ans qu'elle fixe ne contredit pas l'article 17 (1)
    #   (« à titre temporaire ou soumise à un renouvellement périodique »).
    # - Loi 2010/012, articles 76, 80 et 82, et loi 2023/009 (art. 37, 38, 39 et
    #   41) : memes faits, memes peines ; rien de « contraire » au sens de son
    #   article 47.
    # - Loi 2010/012, article 9, et lois de finances 2020 (PRC-8034, art. 24 et
    #   34) et 2022 (PRC-9119, art. 22) : elles ne citent pas la loi 2010/012 ;
    #   elles ouvrent et modifient le compte d'affectation speciale du Fonds, sans
    #   reecrire ni contredire l'article 9.
    # - Loi 2010/012, article 74, alineas 1, 3, 7 et 8, et loi 2024/017 : voir
    #   le dernier lien.

    # Loi 2021/022 (PRC-9115), art. 1er (morceau « 1 ») : « Les dispositions
    # de l'article 5 alinéa 4 de la loi n° 90/050 du 19 décembre 1990 relative
    # à la liberté d'association sont modifiées ainsi qu'il suit: » ; morceau
    # « 5 » : « (4) (nouveau) Les partis politiques, les syndicats, les asoins
    # s aotirns sgoits agootines associations artistiques et culturelles,
    # ainsi que les organisations interprofessionnelles sont régis par des
    # textes particuliers. » « 90/050 » est une coquille : titre et en-tete
    # portent 90/053. Aucun texte posterieur du corpus ne modifie la loi
    # 90/053 : c'est la redaction en vigueur. En base, l'article 5 est dans le
    # morceau « 2 » (articles 2 a 5, « (4) les parties politiques et les
    # syndicats sont régis par des textes particuliers ») ; le morceau
    # SIGNATURE (hors vectoriel, mais cherchable en plein texte) recopie la loi
    # n° 99/011, dont l'article 1er porte « Article 5 (4) (nouveau)- « Les
    # partis politiques, les syndicats, les associations sportives et les
    # organisations non gouvernementales sont régies sont régis par des textes
    # particuliers ». ». La loi 2020/009 (PRC-8413, art. 1er) porte au meme
    # endroit « les syndicats, les associations sportives, les organisations
    # non gouvernementales, ainsi qus es a es a e ae aues es particuliers ».
    # D'ou la derniere proposition de la portee : sans elle, le passage
    # illisible de 2021 laisse croire que ces deux categories ont disparu de
    # la liste.
    Lien(
        texte="LOI-1990-053",
        texte_nom="la Loi N°90/053 du 19 décembre 1990 portant sur la liberté d'association",
        nature="modifie",
        par_titre="la Loi N°2021/022 du 16 décembre 2021",
        par_date=date(2021, 12, 16),
        par_reference="PRC-9115",
        articles=("5",),
        # Le morceau « 5 » de la loi 2021/022 est le nouvel alinea, pas un
        # article de cette loi ; son article 2 est la formule de publication.
        par_articles=("1", "5"),
        morceaux=(
            ("2", "5", "les parties politiques et les syndicats"),
            ("SIGNATURE", "5", "Article 5 (4) (nouveau)"),
        ),
        portee=(
            "seul l'alinéa 4 est réécrit ; cette rédaction, qui remplace celles des lois n° 99/011 "
            "du 20 juillet 1999 et n° 2020/009 du 20 juillet 2020, se lit « Les partis politiques, "
            "les syndicats, [passage illisible] associations artistiques et culturelles, ainsi que "
            "les organisations interprofessionnelles sont régis par des textes particuliers » ; là "
            "où le passage est illisible en base, ces deux lois nommaient les associations "
            "sportives et les organisations non gouvernementales"
        ),
    ),
    # Meme preuve. La loi 2020/009 (PRC-8413), art. 1er (morceau « 1 ») :
    # « Les dispositions de l'article 5 alinéa 4 de la loi n° 90/053 […] sont
    # modifiées et complétées ainsi qu'il suit : «ARTICLE 5.- (4) (nouveau) Les
    # partis politiques, les syndicats, les associations sportives, les
    # organisations non gouvernementales, ainsi qus es a es a e ae aues es
    # particuliers. » ». La loi 2021/022 reecrit le meme alinea : la redaction
    # de 2020 ne s'applique plus. La loi 2021/022 ne cite pas la loi 2020/009
    # (pas de visa) : le lien tient a ce que les deux reecrivent le meme
    # alinea, la plus recente l'emportant. Ce morceau nomme en clair les ONG et
    # les associations sportives, alors que le passage correspondant de la loi
    # 2021/022 est illisible en base : une recherche sur ces mots remonte la
    # redaction de 2020, pas celle de 2021. Meme demarche que le lien de la
    # loi 2026/002 sur la loi 2008/001 dans le module.
    Lien(
        texte="PRC-8413",
        texte_nom=(
            "la Loi n° 2020/009 du 20 juillet 2020 modifiant et complétant certaines dispositions "
            "de la loi n° 90/053 du 19 décembre 1990 relative à la liberté d'association"
        ),
        nature="remplace",
        par_titre="la Loi N°2021/022 du 16 décembre 2021",
        par_date=date(2021, 12, 16),
        par_reference="PRC-9115",
        articles=("1",),
        par_articles=("1", "5"),
        portee=(
            "l'alinéa 4 de l'article 5 de la loi n° 90/053 se lit désormais « Les partis politiques, "
            "les syndicats, [passage illisible] associations artistiques et culturelles, ainsi que "
            "les organisations interprofessionnelles sont régis par des textes particuliers », à la "
            "place de la rédaction que donnait cet article ; là où le passage est illisible en base, "
            "celle-ci nommait les associations sportives et les organisations non gouvernementales"
        ),
    ),
    # Loi 2023/009 (PRC-9701), art. 47 : « Sont abrogées toutes les
    # dispositions antérieures contraires. » ; art. 40 : « Est puni d'un
    # emprisonnement de deux (02) à cinq (05) ans et d'une amende de cinq
    # millions (5 000 000) à dix millions (10 000 000) de francs CFA, ou de
    # l'une de ces deux peines seulement, l'auteur des faits suivants,
    # lorsqu'ils sont commis en utilisant un réseau de communications
    # électroniques ou un système d'information : - l'offre, la production, la
    # mise à disposition de pornographie enfantine en vue de sa diffusion; - le
    # fait de se procurer ou de procurer à autrui de la pornographie enfantine
    # par le biais d'un système d'information ; - le fait pour les personnes
    # majeures de faire des propositions sexuelles à des enfants ou une
    # personne se présentant comme telle ; - la diffusion ou la transmission
    # de pornographie enfantine par le biais d'un système d'information. » ;
    # art. 42 (1) : « Est puni d'un emprisonnement de deux (02) à cinq (05) ans
    # et d'une amende d'un million (1000000) à cinq millions (5 000 000) de
    # francsCFA, ou de l'une de ces deux peines seulement, toute personne qui,
    # par voie de communications électroniques, fait des propositions
    # sexuelles à un enfant ou une personne se présentant comme telle. » ;
    # art. 3 : « Enfant : toute personne âgée de moins de dix-huit (18) ans ».
    # Art. 37 : « Est puni d'un emprisonnement de cinq (05) à dix (10) ans et
    # d'une amende de cinq millions (5 000 000) à dix millions (10 000 000) de
    # francs CFA, ou de l'une de ces deux peines seulement, celui qui
    # confectionne, transporte, diffuse, par voie de communications
    # électroniques ou d'un système d'information, un message à caractère
    # pédopornographique […] » ; art. 43 (1) : memes peines pour « celui qui
    # fixe, enregistre, diffuse, transmet à titre onéreux ou gratuit par voie
    # de communications électroniques ou d'un système d'information, un contenu
    # à caractère sexuel impliquant un enfant ». Sans ces deux articles, la
    # portee donnait deux a cinq ans comme LA peine de la diffusion.
    # Loi 2010/012, art. 81 (1) : les memes quatre faits (propositions « à des
    # mineurs de moins de quinze (15) ans ») sont « punis des peines prévues à
    # l'article 82 », soit le double de l'article 79 : dix a vingt ans et dix a
    # vingt millions. L'alinea 2 (definition de la pornographie enfantine)
    # n'est contredit par rien : la loi 2023/009 emploie le terme sans le
    # definir.
    Lien(
        texte="LOI-2010-012",
        texte_nom=(
            "la Loi N°2010/012 du 21 décembre 2010 relative à la cybersécurité et la "
            "cybercriminalité au Cameroun"
        ),
        nature="abroge_partiellement",
        par_titre="la Loi N°2023/009 du 25 juillet 2023",
        par_date=date(2023, 7, 25),
        par_reference="PRC-9701",
        articles=("81",),
        par_articles=("40", "42", "47"),
        portee=(
            "les faits de l'alinéa 1, commis en utilisant un réseau de communications électroniques "
            "ou un système d'information (pornographie enfantine, propositions sexuelles d'un majeur, "
            "désormais à tout enfant de moins de dix-huit (18) ans), sont punis par l'article 40 de "
            "la loi n° 2023/009 d'un emprisonnement de deux (02) à cinq (05) ans et d'une amende de "
            "cinq millions (5 000 000) à dix millions (10 000 000) de francs CFA, ou de l'une de ces "
            "deux peines seulement, et non plus des peines de l'article 82 ; les propositions "
            "sexuelles à un enfant relèvent aussi de son article 42, et ses articles 37 et 43 "
            "punissent d'un emprisonnement de cinq (05) à dix (10) ans et d'une amende de cinq "
            "millions (5 000 000) à dix millions (10 000 000) de francs CFA, ou de l'une de ces deux "
            "peines seulement, la diffusion d'un message à caractère pédopornographique ou d'un "
            "contenu à caractère sexuel impliquant un enfant ; l'alinéa 2, qui définit la "
            "pornographie enfantine, n'est pas touché"
        ),
    ),
    # Loi 2024/017 (PRC-10258), art. 75 : « La présente loi, qui abroge toutes
    # les dispositions antérieures contraires, sera enregistrée […] » ;
    # art. 63 (1) : « Est puni d'un emprisonnement de deux (02) à cinq (05) ans
    # et d'une amende de deux cent mille (200 000) à cinq millions (5 000 000)
    # de francs CFA, ou de l'une de ces deux peines seulement, celui qui
    # collecte les données à caractère personnel ou y accède par un moyen
    # frauduleux, déloyal ou illicite. » ; art. 66 (1) : « Est puni d'un
    # emprisonnement de six (06) mois àdeux (02) ans et d'une amende de cinq
    # cent mille (500 000) à cinq millions (5 000 000) de francs CFA, ou de
    # l'une de ces deux peines seulement, leresponsable de traitement ou le
    # sous-traitant, qui met ou conserve en mémoire, informatisée ou non, sans
    # le consentement exprès de la personne concernée, des données sensibles
    # au sens de présente la loi. » ; art. 67 : memes peines pour « le
    # responsable de traitement ou le sous-traitant qui, à l'occasion de leur
    # enregistrement, de leur classement, de leur transmission ou de toute
    # autre forme de traitement, détourne ces informations de leur finalité 1
    # initiale » (« 1 » : bruit d'OCR).
    # Loi 2010/012, art. 74 (morceaux « 69.1 » pour les al. 1 et 2, « 69.2 »
    # pour les al. 3 a 8) : al. 2, « les personnes qui, sans droit,
    # interceptent des données personnelles lors de leur transmission d'un
    # système d'information à un autre », peines de l'al. 1 (un a deux ans) ;
    # son art. 4, 49) definit l'« Interception illégale : accès sans en avoir
    # le droit ou l'autorisation, aux données d'un réseau de communications
    # électroniques, d'un système d'information ou d'un équipement terminal » :
    # c'est l'acces illicite aux donnees que vise l'art. 63 (1) ; al. 4,
    # collecte illicite de donnees nominatives, « six (06) mois à deux (02)
    # ans » ; al. 5, donnees sensibles mises en « mémoire informatisée, sans
    # l'accord exprès de l'intéressé », peines de l'al. 4 doublees ; al. 6, ces
    # peines « s'appliquent aux personnes qui détournent les informations,
    # notamment, à l'occasion de leur enregistrement, de leur classement, de
    # leur transmission ». Memes faits, peines contraires. Pour les al. 2 et 4,
    # meme si l'on y voyait un concours de qualifications plutot qu'une
    # abrogation, la peine la plus forte, celle de l'art. 63, s'appliquerait :
    # la ligne reste juste. La loi 2024/017 ne s'applique pas a tout
    # traitement (art. 2, 4 et 72) : d'ou « dans le champ de la loi
    # n° 2024/017 ».
    # Non retenus : al. 1 (sans equivalent) ; al. 3 et 7 (la loi 2024/017 n'y
    # oppose que des sanctions administratives, art. 55 et 57) ; al. 8
    # (l'art. 68 (1) punit en base « celui qui recueille […] des données […]
    # dont la divulgation aurait pour effet », et seulement a l'occasion d'un
    # traitement : y lire la divulgation serait une interpretation).
    Lien(
        texte="LOI-2010-012",
        texte_nom=(
            "la Loi N°2010/012 du 21 décembre 2010 relative à la cybersécurité et la "
            "cybercriminalité au Cameroun"
        ),
        nature="abroge_partiellement",
        par_titre="la Loi n° 2024/017 du 23 décembre 2024",
        par_date=date(2024, 12, 23),
        par_reference="PRC-10258",
        articles=("74",),
        # L'art. 75, qui porte la clause d'abrogation, est l'article final de
        # publication.
        par_articles=("63", "66", "67"),
        morceaux=(
            ("69.1", "74", "interceptent des données personnelles"),
            ("69.2", "74", "le fait de collecter par des moyens illicites"),
        ),
        portee=(
            "dans le champ de la loi n° 2024/017, l'interception sans droit de données personnelles "
            "lors de leur transmission (alinéa 2) et leur collecte par des moyens illicites "
            "(alinéa 4) relèvent de son article 63, qui punit celui qui collecte les données à "
            "caractère personnel ou y accède par un moyen frauduleux, déloyal ou illicite d'un "
            "emprisonnement de deux (02) à cinq (05) ans et d'une amende de deux cent mille "
            "(200 000) à cinq millions (5 000 000) de francs CFA, ou de l'une de ces deux peines "
            "seulement ; la mise ou la conservation en mémoire de données sensibles sans "
            "consentement exprès (alinéa 5) et le détournement des informations de leur finalité "
            "initiale (alinéa 6), par le responsable de traitement ou le sous-traitant, sont punis "
            "par ses articles 66 et 67 d'un emprisonnement de six (06) mois à deux (02) ans et d'une "
            "amende de cinq cent mille (500 000) à cinq millions (5 000 000) de francs CFA, ou de "
            "l'une de ces deux peines seulement"
        ),
    ),

    # ========================================================================
    # Regime foncier et domanial, titre foncier
    # ========================================================================
    # Liens candidats, groupe foncier, apres revue adversariale : ORD-1974-001
    # (regime foncier), ORD-1974-002 (regime domanial), DEC-1976-165 (titre foncier).
    # Releves d'origine dans foncier.json. Controle en base :
    # scripts/maintenance/verifier_liens_textes.py --candidats foncier_corrige.py --detail
    #
    # Changements par rapport a foncier.py :
    # - ajout du lien ORD-1974-001 <- loi n° 83-19 du 26 novembre 1983 (article 5,
    # alinea 3), ecarte a tort : son texte est retrouve et la base ne l'integre pas ;
    # - ORD-1974-002 : la portee nomme ce qui change (convention : « si seule une
    # partie change, la portee nomme les alineas ») ; sans elle, la ligne faisait
    # douter des alineas 3 et 4 de l'article 5, encore en vigueur ;
    # - DEC-1976-165 : lien sur tout le texte, la portee donnant la liste exacte des
    # articles. Les designations « 11 à 20, 23, 29 et 31 » et « 34, 35, 37, 39 et
    # 41 » echouent a tests/test_services/test_textes_successifs.py
    # (test_chaque_lien_saisi_est_bien_forme, _UN_ARTICLE), et aucune designation
    # d'un seul article ou d'une seule plage ne donne une ligne exacte sur les
    # morceaux SECTION_2.2 et « 32 » (voir le commentaire du lien).
    # Ecartes, et pourquoi :
    # - ORD-1974-001 et ordonnance 77/1 (articles 4, 5 et 18), loi 80-21 (article
    #   10) : le document en base est une version consolidee. Elle donne deja la
    #   redaction nouvelle et la signale en tete de l'article : « (Ordonnance 77-1
    #   du 10 janvier 1977). Les titulaires de jugements définitifs constitutifs »
    #   (article 5), « (Loi N° 80-21 du 14 juillet 1980) » (article 10). La copie
    #   consolidee d'a-mla.org porte les memes mentions, et aucune autre.
    # - ORD-1974-001, article 19, et lois de finances 90/001, 91/003, 2023, 2024,
    #   2025 et 2026 : elles fixent ou modifient le bareme des droits, comme
    #   l'article 19 le prevoit (« Les tarifs de ces droits sont fixés, le cas
    #   échéant, par la loi de finances. »). Elles reecrivent l'article 14 de la loi
    #   de finances 90/001, puis les articles des lois de finances suivantes ;
    #   jamais l'ordonnance.
    # - ORD-1974-002, repli sans liste d'articles (foncier_repli_74_2.py) : non
    #   retenu. La liste 4, 5, 11 de FAOLEX est confirmee en base pour l'article 5
    #   (voir le lien), et un lien sur tout le texte marquerait aussi les articles
    #   que la version consolidee de FAOLEX montre inchanges (1 a 3, 6 a 10, 12 a
    #   15).

    # Regime foncier, article 5, alinea 3. Texte de la loi (a-mla.org,
    # a-mla.org/en/country/pdf/243, hors base : la loi manque au corpus) :
    # « Loi n°19 du 26 novembre 1983 Modifiant les dispositions de l'article 5
    # de l'ordonnance n° 74-1 du 6 juillet 1974 fixant le régime foncier » ;
    # « Article 5.- alinéa 3 (nouveau)- Les compétences des juridictions
    # judiciaires et celles des commissions consultatives visées à l'article 16
    # ci-dessous en matière de règlement des litiges fonciers définies comme
    # suit : a) Relève de la compétence des commissions consultatives, le
    # règlement des litiges fonciers ci-après : […] b) est de la compétence des
    # juridictions judiciaires le règlement de tous les autres litiges fonciers
    # à l'exclusion de ceux relatifs aux conflits frontaliers. » Meme texte cite
    # par un memoire independant (memoireonline.com, « La procédure
    # d'immatriculation des terrains en droit camerounais selon le décret
    # n°2005/481 », paragraphes 180 et 181, notes 99 et 102). Numero
    # « 83-19 » : « Loi n° 83-19 du 26 novembre 1983 modifiant les dispositions
    # de l'article 5 de l'ordonnance n° 74-1du 6 juillet 1974 fixant le régime
    # foncier. » (memoireonline.com, « Foncier et stratégies d'accès et de
    # contrôle dans les anciennes plantations coloniales au Cameroun »,
    # bibliographie).
    # La base n'integre pas ce changement : le morceau « 5 » n'est annote que
    # « (Ordonnance 77-1 du 10 janvier 1977) » et garde l'ancien alinea 3 :
    # « Tous les litiges fonciers pendants devant les juridictions et introduits
    # en dehors de la procédure de l'immatriculation sont de la compétence des
    # commissions prévues à l'article 16 ci-dessous. » Aucun texte du corpus ne
    # cite la loi (« 83/19 », « 83-19 », « 26 novembre 1983 », « conflits
    # frontaliers » : rien). Les alineas 1 et 2 (delais de transformation des
    # jugements, droit de preference) ne sont pas touches.
    Lien(
        texte="ORD-1974-001",
        texte_nom="l'Ordonnance N°74/1 du 6 juillet 1974 fixant le régime foncier",
        nature="modifie",
        par_titre="la Loi N°83-19 du 26 novembre 1983",
        par_date=date(1983, 11, 26),
        articles=("5",),
        portee=(
            "seul l'alinéa 3 (litiges fonciers pendants devant les juridictions, confiés aux "
            "commissions de l'article 16) est réécrit"
        ),
    ),
    # Regime domanial. Visa du decret n° 2025/037 (PRC-10406), repris par
    # quatorze autres decrets de 2025 et 2026 : « Vu l'ordonnance n° 74/2 du 06
    # juillet 1974 fixant le régime domanial, modifiée et complétée par
    # l'ordonnance n° 77/2 du 10 janvier 1977 ; ». L'ordonnance 77/2 manque au
    # corpus. La version consolidee de FAOLEX (faolex.fao.org/docs/pdf/cmr1216.pdf,
    # hors base) annote trois articles, et eux seuls, dans ce qu'elle reproduit :
    # « Art.4. (Ordonnance n° 77-2 du 10 janvier 1977). Font partie du domaine
    # public artificiel : a) Les autoroutes et une emprise de cent (100) mètres
    # […] b) Les routes nationales et provinciales et une emprise de quarante
    # (40) mètres […] d) Les pistes carrossables d'intérêt local et une emprise
    # de dix (10) mètres […] i) Les lignes télégraphiques, téléphoniques, leurs
    # dépendances […] » ; « Art. 5. (Ordonnance 77-2 du 10 janvier 1977). Les
    # immeubles destinés à faire partie du domaine public artificiel de l'Etat
    # sont classés par décret. Le décret de classement vaut acte
    # d'expropriation, opère le transfert de propriété […] » ; « Art.11.
    # (Ordonnance n 77-2 du 10 janvier 1977) A partir du 5 août 1974, date
    # d'entrée en vigueur de l'ordonnance n° 2 du 6 juillet 1974, peuvent […] ».
    # Le document en base est la redaction de 1974 : article 4 sans les
    # autoroutes, « une emprise de vingt-cinq mètres » pour les pistes
    # carrossables, « Les lignes télégraphiques, leurs dépendances » ; article
    # 5 « Les immeubles destinés à faire partie du domaine artificiel sont
    # classés par décret. Le décret de classement vaut déclaration d'utilité
    # publique » ; article 11 « pendant une période transitoire de deux ans ».
    # Les autres points de l'article 4 (pistes non carrossables, chemins de fer,
    # ports, alluvions, monuments, chefferies), les alineas 3 et 4 de l'article
    # 5 (declassement, contestation des limites) et les articles 1 a 3, 6 a 10
    # et 12 a 15 sont identiques dans les deux versions ; la copie de FAOLEX
    # s'arrete a la premiere phrase de l'article 16.
    # Confirmation en base pour l'article 5 : les decrets de classement de 2025
    # appliquent la redaction de 1977, par exemple PRC-10416, article 2 : « Le
    # classement opéré à l'article 1er ci-dessus vaut expropriation totale du
    # Titre Foncier n° 12401 du Département de la Lékié. » (de meme PRC-10350,
    # 10415, 10417, 10418, 10419, 10420, 10519).
    # Le morceau « 5 » porte aussi les articles 6 a 9 : `sauf` lui retire la
    # ligne generale, et `morceaux` lui donne celle qui precise que le reste de
    # l'extrait n'est pas concerne. L'article 11 est dans le morceau SECTION_1,
    # entre la fin de l'article 10 et l'article 12.
    Lien(
        texte="ORD-1974-002",
        texte_nom="l'Ordonnance N°74/2 du 6 juillet 1974 fixant le régime domanial",
        nature="modifie",
        par_titre="l'Ordonnance N°77/2 du 10 janvier 1977",
        par_date=date(1977, 1, 10),
        articles=("4", "5", "11"),
        morceaux=(
            ("5", "5", "classés par décret"),
            ("SECTION_1", "11", "période transitoire de deux ans"),
        ),
        sauf=(("5", "classés par décret"),),
        portee=(
            "sont réécrits l'article 4 pour les routes, les pistes carrossables et les lignes "
            "télégraphiques (un point y est ajouté pour les autoroutes), les alinéas 1 et 2 de "
            "l'article 5 (classement par décret et effet du décret de classement) et l'article 11 "
            "en entier (incorporation sans indemnité des propriétés rurales non régénérées)"
        ),
    ),
    # Titre foncier. Decret n° 2005/481 du 16 decembre 2005, article premier
    # (faolex.fao.org/docs/pdf/cmr62163.pdf, hors base : le decret manque au
    # corpus) : « Les dispositions des articles 2, 11, 12,
    # 13,14,15,16,17,18,19,20,23,29,31,34,35,37,39, 41 et 43 du décret N° 76/165
    # du 27 avril 1976 fixant les conditions d'obtention du titre foncier, sont
    # modifiées et complétées ainsi qu'il suit ». Son article 2 ne regle que la
    # publication. Le document en base est la redaction de 1976 : article 20,
    # oppositions « soumises au ministre chargé des domaines pour règlement »
    # (2005 : « au gouverneur territorialement compétent ») ; articles 17 a 19,
    # « chef de service provincial des domaines » (2005 : « conservateur
    # foncier »).
    # Pourquoi un lien sur tout le texte, la portee donnant la liste. Aucun de
    # ces numeros n'est un morceau en base : l'article 2 est dans le morceau
    # « 1 », les articles 11 a 31 dans SECTION_2.2, 32 a 41 dans « 32 », 42 a
    # 46 dans « 42 ». La ligne d'un morceau designe par `morceaux` ne nomme
    # qu'un article ou une plage, puis dit « Le reste de l'extrait n'est pas
    # concerné » : faux pour SECTION_2.2 (11 a 20, mais aussi 23, 29 et 31) et
    # pour « 32 » (34, 35, 37, 39 et 41). De plus, la section en base des
    # morceaux « 32 » et « 42 » recopie le texte entier des articles 11 a 31 :
    # l'en-tete du bloc l'affiche (« Section : PARAGRAPHE II - Procédure
    # Article 11.- … ») et l'embedding le contient. La liste de la portee
    # couvre aussi ces articles-la.
    Lien(
        texte="DEC-1976-165",
        texte_nom=(
            "le Décret N°76/165 du 27 avril 1976 fixant les conditions d'obtention "
            "du titre foncier"
        ),
        nature="modifie",
        par_titre="le Décret N°2005/481 du 16 décembre 2005",
        par_date=date(2005, 12, 16),
        portee=(
            "seuls les articles 2, 11 à 20, 23, 29, 31, 34, 35, 37, 39, 41 et 43 du décret de 1976 "
            "sont modifiés et complétés"
        ),
    ),

    # ========================================================================
    # Etat civil et incitations a l'investissement (textes archives), organisation judiciaire, communications electroniques, pensions, lois de finances
    # ========================================================================
    # Groupe « divers » : ORD-1981-002 (etat civil), LOI-2013-004 (incitations a
    # l'investissement), LOI-2006-015 (organisation judiciaire), LOI-2010-013
    # (communications electroniques), LOI-1969-LF-18 (pensions). Chaque preuve
    # ci-dessous est citee telle qu'elle figure en base (graphie OCR comprise).
    #
    # Version relue par la revue adversariale (2026-10-14) : portees precisees
    # (ORD-1981-002 / loi 2024/016 ; LOI-2013-004 / loi 2017/015), par_articles de
    # l'ordonnance 2025/002 limites au regime des incitations, et deux liens
    # ajoutes en fin de fichier : l'article 22 de la loi de finances 2020, et sa
    # reecriture de 2023, abroges par l'article 38 (1) de la loi de finances 2024.

    # ------------------------------------------------------------------
    # ORD-1981-002 (etat civil). Le document en base est la version signee
    # le 29 juin 1981 (« Yaoundé, le 29 juin 1981 Le Président de la
    # République AHMADOU AHIDJO »), sans trace de la loi de 2011.
    #
    # Loi 2011/011, absente du corpus. PRC-4012 (decret n° 2015/431), visa :
    # « Vu la loi n° 2011/011 du 06 mai 2011 modifiant et complétant
    # certaines dispositions de l'ordonnance nº 81/02 du 29 juin 1981 portant
    # organisation de l'état civil et diverses dispositions relatives à l'état
    # des personnes physiques » ; PRC-10257, art. 129 : « l'ordonnance
    # n°81-02 […] modifiée et complétée par la loi n° 2011/011 du 06 mai
    # 2011 ». Aucun texte du corpus ne dit quels articles elle touche.
    Lien(
        texte="ORD-1981-002",
        texte_nom=(
            "l'Ordonnance N°81/002 du 29 juin 1981 portant organisation de l'état civil et diverses "
            "dispositions relatives à l'état des personnes physiques"
        ),
        nature="modifie",
        par_titre="la Loi n° 2011/011 du 06 mai 2011",
        par_date=date(2011, 5, 6),
        portee="les articles modifiés ou complétés ne sont pas connus",
    ),
    # Loi 2024/016 (PRC-10257), art. 129 : « Sont abrogées, toutes
    # dispositions antérieures contraires, notamment celles de l'ordonnance
    # n°81-02 du 29 juin 1981 portant organisation de l'état civil et
    # diverses dispositions relatives à l'état des personnes physiques,
    # modifiée et complétée par la loi n° 2011/011 du 06 mai 2011, en ses
    # aspects relatifs à l'état civil. » Art. 1er (2) : « Des lois
    # particulières fixentles règlesrelatives à l'état des personnes. »
    # Art. 127 : « Les affaires pendantes et les dossiers en instance à la date
    # de promulgation de la présente loi, en matière de déclaration des faits
    # d'état civil, d'établissement, de délivrance et de rectification des
    # actes d'état civil, de reconstitution des registres et actes d'état
    # civil, demeurent régis par les procédures en vigueur au moment de leur
    # introduction si celles-ci sont plus favorables. »
    # L'art. 129 ne cite aucun numero : la liste est celle de la
    # contre-verification, article par article (2->7, 4-5->6, 6->14 et 27,
    # 7-11->84-99, 12-21->5, 31-36 et 69-74, 22-29->25 et 55-66, 30-33->10-13,
    # 34->37, 38->15 et 40, 40->38, 48-51->46, 41 et 66, 53-57->16-20,
    # 59-60->42-43, 62->45, 68-69->46-47, 78-80->21-23 et 52-54, 82->124,
    # 83->115). Hors liste : le nom (35-37, 39), la filiation, la puissance
    # paternelle, les conditions, effets et dissolution du mariage, la dot
    # (42, 43, 45, 47, 52, 63-67, 70-77), et les articles mixtes (1, 41, 44,
    # 46, 58, 61, 81). Nature « abroge_partiellement » : seuls les aspects
    # d'etat civil tombent, et la consequence (« Hors de ce que vise ce texte,
    # l'article s'applique encore ») ne fait ecarter aucune regle d'etat des
    # personnes restee en vigueur.
    # Decoupage en base : l'art. 3 est dans le morceau « 2 », l'art. 31 dans
    # le « 30 », l'art. 60 dans le « 59 » ; l'art. 48 est le morceau « 43 »
    # du TITRE VI (ordre 53), le vrai art. 43 (reconnaissance, ordre 45) n'est
    # pas touche et ne contient pas le repere.
    # Revue : l'art. 127 ne garde les anciennes procedures que pour les
    # affaires et dossiers « en matière de déclaration des faits d'état civil,
    # d'établissement, de délivrance et de rectification des actes d'état
    # civil, de reconstitution des registres et actes d'état civil » ; la
    # portee ne l'etend plus a la publication, aux oppositions ni a la
    # celebration du mariage. Les sanctions (chapitre IX, art. 110 a 123 :
    # art. 4, 50 (2) et 83 de l'ordonnance) sont ajoutees a la liste.
    Lien(
        texte="ORD-1981-002",
        texte_nom=(
            "l'Ordonnance N°81/002 du 29 juin 1981 portant organisation de l'état civil et diverses "
            "dispositions relatives à l'état des personnes physiques"
        ),
        nature="abroge_partiellement",
        par_titre="la Loi n° 2024/016 du 23 décembre 2024",
        par_date=date(2024, 12, 23),
        par_reference="PRC-10257",
        articles=(
            "2 à 34", "38", "40", "48 à 51", "53 à 57", "59", "60", "62", "68", "69",
            "78 à 80", "82", "83",
        ),
        par_articles=("129",),
        morceaux=(
            ("43", "48", "Le mariage est célébré par l'officier d'état civil"),
            ("59", "60", "Il notifie l'opposition aux futurs époux"),
        ),
        portee=(
            "l'article 129 abroge les dispositions contraires, « notamment celles de l'ordonnance "
            "n°81-02 […] en ses aspects relatifs à l'état civil » ; pour les déclarations, les "
            "officiers et centres d'état civil, les registres, les actes de naissance, de mariage et "
            "de décès, la publication, les oppositions et la célébration du mariage, la rectification "
            "et la reconstitution des actes, et les sanctions, cette loi s'applique ; les affaires "
            "pendantes et les dossiers en instance à sa promulgation, en matière de déclaration, "
            "d'établissement, de délivrance et de rectification des actes et de reconstitution des "
            "registres et actes, demeurent régis par les procédures en vigueur au moment de leur "
            "introduction si celles-ci sont plus favorables (article 127)"
        ),
    ),
    # ------------------------------------------------------------------
    # LOI-2013-004 (incitations a l'investissement). Le document en base est
    # la version d'origine (art. 18 (3) « deux jours », art. 19 (1) « par le
    # Ministre chargé des inveștissements privés, après avis conforme du
    # Ministre des Finances », art. 35 « Les modalités d'application de la
    # présente loi sont déterminées par voie réglementaire »).
    #
    # Loi 2017/015 (PRC-5697), art. 1er : « Les dispositions des articles 3,
    # 8, 11, 18,19,21, 25, 35 et 36 de la loi n° 2013/004 du 18 avril 2013
    # fixant les incitations à l'investissement privé en République du
    # Cameroun sont modifiées et complétées ainsi qu'il suit: » ; chaque
    # article « (nouveau) » est un morceau de meme numero ; l'art. 2 est la
    # formule de publication. Art. 3 (nouveau) : début de la phase
    # d'exploitation « tel que constaté par l'organisme encharge de la
    # promotion des investissements ou des petites et moyennes entreprises »
    # (2013 : « par un arrêté conjoint des ministres en charge des
    # investissements privés, des finances et du commerce ») ; art. 18 (3) :
    # « Il dispose d'un délai de cinq (05) jours pour examiner le dossier et le
    # transmettre au Ministre èn charge des finances » ; art. 19 : « (1)
    # L'agrément est accordé à l'investisseur par l'organisme en charge de la
    # promotion des investissements ou des petites et moyennes entreprises,
    # après avis du Ministre en charge des finances […] (2) […] quinze (15)
    # jours ouvrables pour délivrer son avis. (3) Passé ce délai, l'avis
    # favorable du Ministre en charge des finances est réputé acquis » ; art.
    # 21 (3) : « l'organisme ayant octroyé l'agrément dispose d'un délai de 15
    # jours ouvrables pour statuer » ; art. 25 : « Pendant les phases
    # d'installation et d'exploitation, toutes les demandes d'importation et
    # d'achat locaux doivent préalablement être revêtues du visa de
    # l'organisme ayant octroyé l'agrément » ; art. 35 : « 15 % des ressources
    # collectées au titre de la contribution au Crédit Foncier […] 15% des
    # ressources collectées au bénéfice du Fonds Spécial des
    # Télécommunications » ; art. 36 : clause d'abrogation qui nomme les lois
    # n° 77/10, n° 98/014 et n° 2010/013. Art. 8 : un alinea 2 s'ajoute ; le
    # premier critere de l'alinea 1 est illisible en base dans la version de
    # 2013 (« emb (is s s ent supérieur par an ») : on ne peut pas affirmer
    # qu'il est inchange, d'ou « modifie » plutot que « complete ».
    # Deux liens « modifie » de meme nature, pour que chaque article porte une
    # portee courte qui le concerne : la procedure d'agrement (3, 18, 19, 21,
    # 25), puis le reste (8, 35, 36).
    # Revue : chaque article « (nouveau) » de 2017 est donne en entier ; la
    # portee le dit (« réécrits en entier », comme les liens LOI-2004-004 du
    # module), sinon une regle de 2013 absente de la redaction nouvelle passe
    # pour intacte. Art. 19 de 2013 : « après avis conforme du Ministre des
    # Finances » et « (4) Passé ce délai et sauf rejet motivé, l'agrément est
    # réputé accordé » ; art. 19 (nouveau) : « après avis du Ministre en charge
    # des finances » et « (4) En cas d'avis favorable [...] dispose d'un délai
    # de trois(03) jours ouvrables pour délivrer l'agrément », sans agrement
    # tacite.
    Lien(
        texte="LOI-2013-004",
        texte_nom=(
            "la Loi N°2013/004 du 18 avril 2013 fixant les incitations à l'investissement privé "
            "en République du Cameroun"
        ),
        nature="modifie",
        par_titre="la Loi N° 2017/015 du 12 juillet 2017",
        par_date=date(2017, 7, 12),
        par_reference="PRC-5697",
        articles=("3", "18", "19", "21", "25"),
        par_articles=("1", "3", "18", "19", "21", "25"),
        portee=(
            "réécrits en entier ; l'organisme en charge de la promotion des investissements ou des "
            "petites et moyennes entreprises constate le début de la phase d'exploitation (article 3) "
            "et accorde l'agrément après avis, et non plus avis conforme, du Ministre en charge des "
            "finances, réputé favorable passé quinze (15) jours ouvrables, l'agrément n'étant plus "
            "réputé accordé à l'expiration du délai de délivrance (article 19) ; il statue en 15 jours "
            "ouvrables sur les délais supplémentaires (article 21) et vise les demandes d'importation "
            "et d'achat locaux dès la phase d'installation (article 25) ; le Guichet Unique dispose "
            "de cinq (05) jours pour examiner le dossier et le transmettre au Ministre en charge des "
            "finances (article 18)"
        ),
    ),
    Lien(
        texte="LOI-2013-004",
        texte_nom=(
            "la Loi N°2013/004 du 18 avril 2013 fixant les incitations à l'investissement privé "
            "en République du Cameroun"
        ),
        nature="modifie",
        par_titre="la Loi N° 2017/015 du 12 juillet 2017",
        par_date=date(2017, 7, 12),
        par_reference="PRC-5697",
        articles=("8", "35", "36"),
        par_articles=("1", "8", "35", "36"),
        portee=(
            "réécrits en entier ; un alinéa 2 renvoie les modalités du crédit d'impôt à la voie "
            "réglementaire (article 8) ; "
            "l'organisme en charge de la promotion des investissements reçoit 15 % des ressources "
            "collectées au titre de la contribution au Crédit Foncier et 15 % de celles collectées au "
            "bénéfice du Fonds Spécial des Télécommunications (article 35) ; la clause d'abrogation "
            "est réécrite et nomme notamment les lois n° 77/10, n° 98/014 et n° 2010/013 (article 36)"
        ),
    ),
    # Meme loi, art. 11 (nouveau) : l'alinea 1 reprend mot pour mot l'article
    # de 2013 (« En raison de l'importance du projet dûment évaluée, l'Etat
    # peut exceptionnellement étendre le bénéfice de quelques exonérations
    # fiscales et douanières aux actionnaires, aux promoteurs et aux
    # contractants locaux de l'investisseur par voie contractuelle ») et
    # ajoute « (2) Les modalités d'application de l'alinéa 1 ci-dessus sont
    # fixéespar voie règlementaire. »
    Lien(
        texte="LOI-2013-004",
        texte_nom=(
            "la Loi N°2013/004 du 18 avril 2013 fixant les incitations à l'investissement privé "
            "en République du Cameroun"
        ),
        nature="complete",
        par_titre="la Loi N° 2017/015 du 12 juillet 2017",
        par_date=date(2017, 7, 12),
        par_reference="PRC-5697",
        articles=("11",),
        par_articles=("11",),
        portee="alinéa 2 ajouté : les modalités d'application de l'alinéa 1 sont fixées par voie réglementaire",
    ),
    # Ordonnance 2025/002 (PRC-10455), art. 51 : « Sont abrogées toutes
    # dispositions antérieures contraires à la présente ordonnance, notamment
    # celles de la loi n° 2008/009 du 16 juillet 2008 […], et de la loi ns n n
    # on t en République du Cameroun, modifiée et complétée par la loi n°
    # 2017/015du 12 juillet 2017. » Le numero illisible est celui de la loi
    # 2013/004 : visa de l'ordonnance, « Vu la loi n° 2013/004 du 18 avril 2013
    # fixant les incitations à l'investissement privé en République du
    # Cameroun, modifiée et complétée par la loi n°2017/015 du _ 12 juillet
    # 2017 », et la loi 2017/015 ne modifie qu'elle. Meme objet (art. 1er (1)
    # : « La présente ordonnance fixe les incitations à l'investissement en
    # République du Cameroun »), toute la matiere reprise : « remplace ».
    # Transition, morceau « 46 » (debut illisible) : « leurs avantages
    # jusqu'à l'expiration dudit régime,sans possibilité d'amendement […]ice
    # des incitations prévues par la présente ordonnance pour la période
    # restant à courir de leur régime initial, sous réserve de satisfaire aux
    # conditions requises, sans cumul d'avantages. Dans ce cas, l'admission au
    # régime prévu par la présente ordonnance ne peut leur être accordée
    # qu'après audit préalable de l'utilisation des précédentes facilités » ;
    # art. 50 : « Dans l'attente de la mise en place du Guichet Unique et du
    # Comité d'audit et de recours, les instances en vigueur restent
    # compétentes pour instruire les demandes d'agrément et les recours. »
    # Ratifiee par la loi n° 2025/015 (PRC-10560, art. 1er).
    # Revue : par_articles n'est plus vide. Les articles 16 a 18 et 21 a 28
    # (incitations et regime comptable des contrats de partenariat public-prive,
    # « conformément aux dispositions de la loi n° 2023/008 ») et 44 (« les
    # dispositions applicables aux contrats de partenariat public-privé, sont
    # celles prévues par la loi n° 2023/008 ») ne reprennent rien de la loi
    # 2013/004, qui excluait « le régime général des contrats de partenariat »
    # (art. 2 (3)) ; l'article 52 est la formule de publication. Les autres
    # morceaux (objet, criteres, incitations, Guichet Unique, convention,
    # controle, sanctions, transition, abrogation) portent le regime nouveau.
    Lien(
        texte="LOI-2013-004",
        texte_nom=(
            "la Loi N°2013/004 du 18 avril 2013 fixant les incitations à l'investissement privé "
            "en République du Cameroun"
        ),
        nature="remplace",
        par_titre="l'Ordonnance N°2025/002 du 18 juillet 2025",
        par_date=date(2025, 7, 18),
        par_reference="PRC-10455",
        par_articles=("1 à 15", "19", "20", "29 à 43", "45 à 51"),
        portee=(
            "l'ordonnance régit désormais les incitations à l'investissement et abroge les dispositions "
            "contraires de la loi n° 2013/004 modifiée par la loi n° 2017/015 (article 51) ; les "
            "bénéficiaires du régime antérieur conservent « leurs avantages jusqu'à l'expiration dudit "
            "régime, sans possibilité d'amendement », ou peuvent demander le nouveau régime pour la "
            "période restant à courir, sans cumul d'avantages et après audit préalable ; en attendant le "
            "Guichet Unique et le Comité d'audit et de recours, « les instances en vigueur restent "
            "compétentes pour instruire les demandes d'agrément et les recours » (article 50)"
        ),
    ),
    # La loi 2017/015 elle-meme (PRC-5697), que l'art. 51 de l'ordonnance vise
    # avec la loi qu'elle modifie (« modifiée et complétée par la loi n°
    # 2017/015du 12 juillet 2017 ») : ses articles « (nouveau) » tombent avec
    # la loi 2013/004. Meme principe que le lien PRC-6284 du module (la
    # redaction issue d'une loi modificative, remplacee a son tour).
    Lien(
        texte="PRC-5697",
        texte_nom=(
            "la Loi N° 2017/015 du 12 juillet 2017 modifiant et complétant certaines dispositions de la "
            "loi N° 2013/004 du 18 avril 2013 fixant les incitations à l'investissement privé en "
            "République du Cameroun"
        ),
        nature="remplace",
        par_titre="l'Ordonnance N°2025/002 du 18 juillet 2025",
        par_date=date(2025, 7, 18),
        par_reference="PRC-10455",
        par_articles=("51",),
        portee=(
            "l'article 51 abroge les dispositions contraires de la loi n° 2013/004 « modifiée et "
            "complétée par la loi n° 2017/015 », et l'ordonnance régit désormais les incitations à "
            "l'investissement ; les bénéficiaires du régime antérieur conservent « leurs avantages "
            "jusqu'à l'expiration dudit régime, sans possibilité d'amendement »"
        ),
    ),
    # ------------------------------------------------------------------
    # LOI-2006-015 (organisation judiciaire). Version d'origine (signee
    # « Yaoundé, le 29 Décembre 2006 »).
    #
    # Loi 2011/027, absente du corpus. PRC-534 (decret n° 2013/288), visa :
    # « Vu la loi 2006/015 du 29 décembre 2006 portant organisation
    # judiciaire, modifiée et complétée par la loi 2011/027 du 14 décembre
    # 2011 » ; PRC-10544 (decret n° 2025/527), visa : « Vu la loi n° 2006/015
    # du 29 décembre 2006 portant organisation judiciaire, modifiée et
    # complétée par la loi nº2011/027 du 14 décembre 2011 » (23 textes en
    # base la citent). Aucun ne cite d'article modifie.
    Lien(
        texte="LOI-2006-015",
        texte_nom="la Loi N°2006/015 du 29 décembre 2006 portant organisation judiciaire",
        nature="modifie",
        par_titre="la Loi n° 2011/027 du 14 décembre 2011",
        par_date=date(2011, 12, 14),
        portee="les articles modifiés ou complétés ne sont pas connus",
    ),
    # Code de justice militaire, loi 2017/012 (PRC-5695), art. 25 : « (3) [les
    # appels] obéissent aux règles prévues par la loi portant organisation
    # judiciaire au Cameroun et le Code de Procédure Pénale. (4) Pour
    # l'instruction desdits appels, le Commissaire du Gouvernement chargé de
    # l'action publique à la Cour d'Appel exerce toutes les attributions
    # prévues par le Code de Procédure Pénale. (5) En casde pourvoi, il en
    # assure lesdiligences. » Art. 70 : « Sont abrogées, toutes dispositions
    # antérieures contraires à la présente loi […] ». Contre LOI-2006-015,
    # art. 20 (3) : « Lorsque la Cour statue sur les appels interjetés contre
    # les jugements d'un Tribunal militaire […], un magistrat militaire nommé à
    # cet effet occupe le banc du Ministère Public. Les activités de ce
    # magistrat militaire se limitent à la représentation du Ministère Public
    # à l'audience. » Le Code vise la loi d'organisation judiciaire dans
    # l'article meme qui pose la regle contraire ; seule la derniere phrase
    # de l'alinea 3 est touchee (contre-verification : « confiance moyenne »).
    # Revue : maintenu. Le debut de l'alinea 3 de l'art. 25 est illisible en
    # base (« oits its p s es iests tes tt εe obéissent aux règles prévues
    # par la loi portant organisation judiciaire au Cameroun et le Code de
    # Procédure Pénale »), d'ou « [les appels] » ci-dessus. Le magistrat
    # militaire de l'art. 20 (3) et le Commissaire du Gouvernement de l'art.
    # 25 (4) sont la meme fonction : PRC-10592 (2025) nomme un « Capitaine
    # [...], Commissaire du Gouvernement près la Cour d'Appel du Centre ».
    # A la difference de l'ecart LOI-2004-004 / loi 2008/001 du module, le
    # Code a une clause d'abrogation (art. 70) et nomme la loi d'organisation
    # judiciaire dans l'article qui la contredit.
    Lien(
        texte="LOI-2006-015",
        texte_nom="la Loi N°2006/015 du 29 décembre 2006 portant organisation judiciaire",
        nature="abroge_partiellement",
        par_titre="la Loi N° 2017/012 du 12 juillet 2017",
        par_date=date(2017, 7, 12),
        par_reference="PRC-5695",
        articles=("20",),
        par_articles=("25",),
        portee=(
            "dernière phrase de l'alinéa 3 : pour l'instruction des appels des jugements du Tribunal "
            "Militaire, le Code de justice militaire donne au Commissaire du Gouvernement chargé de "
            "l'action publique à la Cour d'Appel toutes les attributions prévues par le Code de "
            "Procédure Pénale et, en cas de pourvoi, les diligences (article 25) ; il n'est plus "
            "limité à la représentation du Ministère Public à l'audience"
        ),
    ),
    # ------------------------------------------------------------------
    # LOI-2010-013 (communications electroniques). Version d'origine de 2010.
    #
    # Loi 2015/006, absente du corpus. PRC-8643 (decret n° 2020/727), visa :
    # « Vu la loi n° 2010/013 du 21 décembre 2010 régissant les
    # communications électroniques au Cameroun, modifiée et complétée par la
    # loi n° 2015/006 du 20 avril 2015 » ; meme visa dans PRC-7468 et
    # PRC-9324 ; PRC-10261 et PRC-10557, art. C 131 : « loi n° 2010/013 […]
    # modifiée et complétée par la loi n° 2015/006 du 20 avril 2015 ».
    # Aucun texte ne cite d'article modifie.
    Lien(
        texte="LOI-2010-013",
        texte_nom=(
            "la Loi N°2010/013 du 21 décembre 2010 régissant les communications électroniques au Cameroun"
        ),
        nature="modifie",
        par_titre="la Loi n° 2015/006 du 20 avril 2015",
        par_date=date(2015, 4, 20),
        portee="les articles modifiés ou complétés ne sont pas connus",
    ),
    # Fonds Special des Telecommunications, institue par l'art. 34 (« Il est
    # institué par la présente loi, un Fonds Spécial des Télécommunications »,
    # ressources « déposées dans un compte ouvert à la Banque Centrale »).
    # Loi de finances 2024 (PRC-9875), art. 38 : « Reversement des opérations
    # financières du Fonds Spécial des Télécommunications dans le Budget
    # Général de l'Etat. (1) Les dispositions de l'article VING-DEUXIEME de la
    # Loi n°2019/023 […] instituant le Compte d'Affectation Spéciale […] sont
    # abrogées. (2) Les recettes et les dépenses initialement affectées audit
    # Fonds sont entièrement reversées dans le Budget Général de l'Etat. (3)
    # […] a) En recettes : […] 50% de l'excèdent budgétaire constaté à la fin
    # de l'exercice sur les opérations de I' Agence de Régulation des
    # Télécommunications ; […] b) En Dépenses : […] toutes autres opérations à
    # caractère stratégique décidées par le Président de la République. »
    # Lien propose par la contre-verification ; les lois de finances 2020
    # (PRC-8034, art. 22) et 2023 (PRC-9449, art. 22 nouveau), qu'il abroge,
    # ne sont pas saisies contre la loi 2010/013 (regime supprime) ; leur
    # abrogation fait l'objet des deux liens ajoutes en fin de fichier. Aucune
    # loi de finances posterieure (2025, 2026) ne touche a cet article 38.
    Lien(
        texte="LOI-2010-013",
        texte_nom=(
            "la Loi N°2010/013 du 21 décembre 2010 régissant les communications électroniques au Cameroun"
        ),
        nature="modifie",
        par_titre="la Loi N°2023/019 du 19 décembre 2023 portant loi de finances pour l'exercice 2024",
        par_date=date(2023, 12, 19),
        par_reference="PRC-9875",
        articles=("34",),
        par_articles=("38",),
        portee=(
            "les recettes et les dépenses du Fonds Spécial des Télécommunications sont entièrement "
            "reversées dans le Budget Général de l'Etat (article 38) ; parmi les recettes figurent 50 % "
            "de l'excédent budgétaire de l'Agence de Régulation des Télécommunications, et parmi les "
            "dépenses « toutes autres opérations à caractère stratégique décidées par le Président de "
            "la République »"
        ),
    ),
    # Redevance d'utilisation des frequences, art. 39 (2) : « Les modalités de
    # paiement et de répartition de cette redevance, entre le Trésor Public,
    # le Comité visé à l'alinéa 3 de l'article 37 et l'Agence, sont fixées par
    # un texte particulier du Ministre chargé des Télécommunications. » (Le
    # morceau « 39 » contient aussi les art. 40 et 41.)
    # Loi 2024/020 portant fiscalite locale (PRC-10261), art. C 131 : « (1)
    # Par dérogation aux dispositions de la loin2010/013 du 21 décembre 2010
    # régissant les communications électroniques modifiée etcomplétée par la
    # loi n°2015/006 du 20 avril 2015, une quote-part de soixante pour cent
    # (60%) du produit de la redevance d'utilisation des fréquences
    # radioélectriques est affectée aux régions. (2) La redevance prévue à
    # l'alinéa 1 ci-dessus est recouvrée par les services fiscaux de l'Etat. »
    # La LF 2026 reprend l'alinea 1 ; son alinea 2 se reduit en base a
    # « (2) .. » : la ligne ne dit rien de plus que ces deux alineas.
    Lien(
        texte="LOI-2010-013",
        texte_nom=(
            "la Loi N°2010/013 du 21 décembre 2010 régissant les communications électroniques au Cameroun"
        ),
        nature="modifie",
        par_titre="la Loi n° 2024/020 du 23 décembre 2024 portant fiscalité locale",
        par_date=date(2024, 12, 23),
        par_reference="PRC-10261",
        articles=("39",),
        par_articles=("C 131",),
        portee=(
            "alinéa 2 : par dérogation à la loi n° 2010/013, une quote-part de soixante pour cent "
            "(60%) du produit de la redevance d'utilisation des fréquences radioélectriques est "
            "affectée aux régions, et la redevance est recouvrée par les services fiscaux de l'Etat "
            "(article C 131)"
        ),
    ),
    # Loi de finances 2026 (PRC-10557), art. 23 (morceau « VINGT-TROISIÈME ») :
    # « Les dispositions des articles […] C 131, et C132 du Code Général des
    # Impôts sont modifiées et/ou complétées ainsi qu'il suit » ; morceau
    # « C131 » : « Par dérogation aux dispositions de la loi n° 2010/013 du 21
    # décembre 2010 régissant les communications électroniques, modifiée et
    # complétée par la loi n° 2015/006 du 20 avril 2015, une quote-part de
    # soixante pour cent (60 %) du produit de la redevance des fréquences
    # radioélectriques est affectée aux Régions. (2) .. (3) La redevance
    # d'utilisation des fréquences radioélectriques est déclarée et acquittée
    # par les assujettis, via la plateforme informatique de l'administration
    # fiscale, en quatre tranches d'égal montant, suivant l'échéancier
    # ci-après : - au plus tard le 15 mars […] - au plus tard le 15juin […] -
    # au plus tard le 15 septembre […] - au plus tard le 15 décembre […] (6)Les
    # quatre tranches trimestrielles d'égal montant devant être acquittées par
    # les assujettis sont déterminées sur la base des facturations émises par
    # le régulateur, au titre de l'exercice précédent (n-1). »
    Lien(
        texte="LOI-2010-013",
        texte_nom=(
            "la Loi N°2010/013 du 21 décembre 2010 régissant les communications électroniques au Cameroun"
        ),
        nature="modifie",
        par_titre="la Loi N°2025/012 du 17 décembre 2025 portant loi de finances pour l'exercice 2026",
        par_date=date(2025, 12, 17),
        par_reference="PRC-10557",
        articles=("39",),
        par_articles=("C131",),
        portee=(
            "alinéa 2 : la quote-part de soixante pour cent (60 %) affectée aux Régions est maintenue, "
            "et la redevance d'utilisation des fréquences radioélectriques est déclarée et acquittée "
            "via la plateforme informatique de l'administration fiscale, en quatre tranches d'égal "
            "montant, au plus tard les 15 mars, 15 juin, 15 septembre et 15 décembre, déterminées sur "
            "la base des facturations émises par le régulateur au titre de l'exercice précédent "
            "(article C131)"
        ),
    ),
    # ------------------------------------------------------------------
    # LOI-1969-LF-18 (pensions). Art. 19 : « En cas de cumul de deux ou
    # plusieurs pensions ou rentes allouées en vertu des dispositions
    # législatives régissant la prévoyance sociale, le titulaire a droit à la
    # totalité de la pension ou de la rente dont le montant est le plus élevé
    # et à la moitié de l'autre ou des autres pensions ou rentes. »
    #
    # Loi 77/11 (LOI-1977-011), art. 33 : « Le cumul de pensions ou de rentes
    # allouées en application de la présente loi et de la loi nº 69-LF-18 du
    # 10 Novembre 1979 est admis sans restriction d'aucune sorte. » (« 1979 »
    # sic, aussi dans laws.content) ; art. 55 : « 1. Sont abrogées toutes
    # dispositions contraires à la présente loi ». La loi de 1977 nomme celle
    # de 1969 et regle le point meme de l'art. 19 ; les autres cumuls restent
    # soumis a l'art. 19.
    Lien(
        texte="LOI-1969-LF-18",
        texte_nom=(
            "la Loi N°69/LF/18 du 10 novembre 1969 instituant un régime d'assurance de pensions de "
            "vieillesse, d'invalidité et de décès"
        ),
        nature="abroge_partiellement",
        par_titre="la Loi N°77/11 du 13 juillet 1977",
        par_date=date(1977, 7, 13),
        par_reference="LOI-1977-011",
        articles=("19",),
        par_articles=("33",),
        portee=(
            "le cumul de pensions ou de rentes allouées en application de la loi n° 77/11 (accidents "
            "du travail et maladies professionnelles) et de la loi de 1969 est « admis sans "
            "restriction d'aucune sorte » (article 33)"
        ),
    ),
    # Loi 90/063, absente du corpus. PRC-8690 (decret n° 2020/7951/PM), visa :
    # « Vu la loi n° 90/063 du 19 décembre 1990 modifiant certaines
    # dispositions de la loi n° 69/LF/18 du 10 novembre 1969 instituant un
    # régime d'assurance de pensions vieillesse, d'invalidité et de décès ».
    # Seule mention en base ; articles touches inconnus.
    Lien(
        texte="LOI-1969-LF-18",
        texte_nom=(
            "la Loi N°69/LF/18 du 10 novembre 1969 instituant un régime d'assurance de pensions de "
            "vieillesse, d'invalidité et de décès"
        ),
        nature="modifie",
        par_titre="la Loi n° 90/063 du 19 décembre 1990",
        par_date=date(1990, 12, 19),
        portee="les articles modifiés ne sont pas connus",
    ),
    # ------------------------------------------------------------------
    # Ajouts de la revue. Le compte d'affectation speciale du Fonds Special des
    # Telecommunications, que la LF 2024 supprime (voir le lien LOI-2010-013 /
    # PRC-9875 ci-dessus). LF 2024 (PRC-9875), art. 38 : « (1) Les dispositions
    # de l'article VING-DEUXIEME de la Loi n°2019/023 du 24 décembre 2019
    # portant Loi de finances de la République du Cameroun pour l'exercice
    # 2020, instituant le Compte d'Affectation Spéciale intitulé « Fonds
    # Spécial pour le Développement des Télécommunications », sont abrogées.
    # (2) Les recettes et les dépenses initialement affectées audit Fonds sont
    # entièrement reversées dans le Budget Général de l'Etat. »
    #
    # LF 2020 (PRC-8034), morceau « 22 » (seul morceau de numero 22) : « (1)
    # Il est ouvert un compte d'affectation spéciale intitulé « Fonds Spécial
    # des Télécommunications ». » Aucune loi de finances posterieure (2025,
    # 2026) ne le retablit : elles ne fixent que le plafond de la quote-part
    # affectee a l'Agence de Promotion des Investissements.
    Lien(
        texte="PRC-8034",
        texte_nom="la Loi N°2019/023 du 24 décembre 2019 portant loi de finances pour l'exercice 2020",
        nature="abroge",
        par_titre="la Loi N°2023/019 du 19 décembre 2023 portant loi de finances pour l'exercice 2024",
        par_date=date(2023, 12, 19),
        par_reference="PRC-9875",
        articles=("22",),
        par_articles=("38",),
        portee=(
            "cet article ouvre le compte d'affectation spéciale du Fonds Spécial des "
            "Télécommunications ; les recettes et les dépenses initialement affectées audit Fonds "
            "sont entièrement reversées dans le Budget Général de l'Etat (article 38)"
        ),
    ),
    # LF 2023 (PRC-9449), art. 34 (morceau « 34 ») : « Les dispositions des
    # articles dix-neuvième, vingtdeuxième et vingt-cinquième de la loi
    # n°2019/023 du 24 décembre 2019 [...] sont modifiées et complétées ainsi
    # qu'il suit » ; l'article 22 (nouveau) est le morceau « 22 » d'ordre 121
    # (« (nouveau): (1) il est ouvert un compte d'affectation spéciale intitulé
    # « Fonds Spécial des Télécommunications » »). C'est la redaction que la
    # LF 2024 abroge. Le morceau « 22 » d'ordre 102 est l'article 22 propre a la
    # LF 2023 (« prélèvement au titre des frais de visa de travail »), non
    # touche : exclu par `sauf`.
    Lien(
        texte="PRC-9449",
        texte_nom="la Loi N°2022/020 du 27 décembre 2022 portant loi de finances pour l'exercice 2023",
        nature="abroge",
        par_titre="la Loi N°2023/019 du 19 décembre 2023 portant loi de finances pour l'exercice 2024",
        par_date=date(2023, 12, 19),
        par_reference="PRC-9875",
        articles=("22",),
        par_articles=("38",),
        sauf=(("22", "visa de travail"),),
        portee=(
            "il s'agit de l'article vingt-deuxième de la loi de finances pour l'exercice 2020 "
            "(compte d'affectation spéciale du Fonds Spécial des Télécommunications), dans la "
            "rédaction que lui donne l'article 34 de la loi de finances pour l'exercice 2023 ; les "
            "recettes et les dépenses initialement affectées audit Fonds sont entièrement reversées "
            "dans le Budget Général de l'Etat (article 38)"
        ),
    ),
)
