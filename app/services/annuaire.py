"""
Annuaire des institutions : les contacts que le chat montre sous une reponse
qui renvoie vers elles, et que la page /annuaire liste.

POURQUOI. Le chat conseillait de « saisir l'inspecteur du travail » sans dire
comment le joindre (retour d'un utilisateur, 08/11/2026). Les coordonnees ne
viennent JAMAIS du modele : seulement de cette liste, chaque fiche avec sa
source et la date de cette source, affichees a l'utilisateur.

Choix de l'utilisateur (08/11/2026) : institutions nationales et delegations
regionales ; publication directe, avec la source affichee.

LES NUMEROS PEUVENT AVOIR CHANGE. La source des delegations du travail est
l'annuaire statistique du MINTSS (edition de septembre 2020). Le front affiche
« source, date » sur chaque fiche.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Dict, List, Optional

SOURCE_MINTSS = {
    "titre": "Annuaire statistique du MINTSS, annexe 8 (édition de septembre 2020)",
    "url": "https://ins-cameroun.cm/wp-content/uploads/2025/06/Annuaire-francais-MINTSS.pdf",
    "date": "2020-09",
}
SOURCE_BARREAU = {
    "titre": "Communiqués de l'Ordre des avocats (2026)",
    "url": "https://barreaucameroun.org/fr/",
    "date": "2026",
}
SOURCE_CONAC = {
    "titre": "CONAC, page « Hotline 1517 »",
    "url": "https://conac.cm/en/?p=74",
    "date": "2025",
}
SOURCE_CDHC = {
    "titre": "Fiche CDHC de l'annuaire Osidimbea",
    "url": "https://www.osidimbea.cm/institutions/cssion-droits-homme/",
    "date": "2024",
}
SOURCE_MINJUSTICE = {
    "titre": "Fiche du ministère de la Justice, annuaire Osidimbea",
    "url": "https://www.osidimbea.cm/ministere-de-la-justice/",
    "date": "2024",
}
SOURCE_URGENCES = {
    "titre": "Conseils aux voyageurs du gouvernement du Canada (Cameroun)",
    "url": "https://voyage.gc.ca/fr/assistance/ambassades-consulats/cameroun",
    "date": "2026",
}
SOURCE_MINCOMMERCE = {
    "titre": "Annonce du numéro 1502 par le ministre du Commerce",
    "url": "https://ecomatin.net/commerce-deja-un-numero-vert-pour-denoncer-des-pratiques-illicites",
    "date": "2019-09",
}

# Les delegations regionales du travail (DRTSS) : chacune a une brigade
# d'inspection du travail et de la securite sociale.
_DRTSS = [
    ("adamaoua", "Adamaoua", "Ngaoundéré", "222 25 12 90"),
    ("centre", "Centre", "Yaoundé", "222 20 44 61"),
    ("est", "Est", "Bertoua", "222 24 10 90"),
    ("extreme-nord", "Extrême-Nord", "Maroua", "222 29 16 15"),
    ("littoral", "Littoral", "Douala", "233 42 16 26"),
    ("nord", "Nord", "Garoua", "222 27 12 70"),
    ("nord-ouest", "Nord-Ouest", "Bamenda", "233 36 39 59"),
    ("ouest", "Ouest", "Bafoussam", "233 44 13 08"),
    ("sud", "Sud", "Ebolowa", "222 74 70 08"),
    ("sud-ouest", "Sud-Ouest", "Buéa", "233 32 24 17"),
]

REGIONS: Dict[str, str] = {code: nom for code, nom, _, _ in _DRTSS}


def _fiche(**champs) -> Dict:
    fiche = {
        "region": None, "ville": None, "telephones": [], "email": None, "site": None,
        "adresse": None, "note": None,
    }
    fiche.update(champs)
    return fiche


ANNUAIRE: List[Dict] = [
    # ---- Travail et securite sociale ----
    _fiche(
        id="mintss", theme="travail",
        nom="Ministère du Travail et de la Sécurité sociale (MINTSS)",
        ville="Yaoundé", telephones=["222 23 00 04"], site="https://www.mintss.gov.cm",
        adresse="BP MINTSS, Yaoundé",
        note="Cabinet du ministre. Direction des relations professionnelles : 222 23 89 97.",
        source=SOURCE_MINTSS,
    ),
    *[
        _fiche(
            id=f"drtss-{code}", theme="travail",
            nom=f"Délégation régionale du Travail et de la Sécurité sociale — {nom}",
            region=code, ville=ville, telephones=[tel],
            note="Inspection du travail : conflits avec l'employeur, licenciement, salaires, "
                 "conciliation. Chaque département a aussi sa délégation.",
            source=SOURCE_MINTSS,
        )
        for code, nom, ville, tel in _DRTSS
    ],
    _fiche(
        id="cnps", theme="social",
        nom="Caisse nationale de prévoyance sociale (CNPS)",
        ville="Yaoundé", telephones=["222 23 08 31", "222 22 08 31"],
        email="cnps.cameroun@cnps.cm", site="https://www.cnps.cm", adresse="BP 441, Yaoundé",
        note="Immatriculation, cotisations, pensions, accidents du travail, allocations familiales.",
        source=SOURCE_MINTSS,
    ),
    # ---- Justice et avocats ----
    _fiche(
        id="barreau", theme="avocats",
        nom="Ordre des avocats au Barreau du Cameroun",
        ville="Yaoundé", telephones=["222 22 03 85", "652 70 91 22", "655 72 36 98"],
        site="https://barreaucameroun.org",
        adresse="Avenue Charles Atangana, derrière le Mess des officiers d'Olézoa, BP 13488, Yaoundé",
        note="Pour trouver un avocat ou signaler un manquement d'un avocat.",
        source=SOURCE_BARREAU,
    ),
    _fiche(
        id="minjustice", theme="justice",
        nom="Ministère de la Justice (MINJUSTICE)",
        ville="Yaoundé", telephones=["222 23 26 23"], site="https://minjustice.cm",
        adresse="Quartier des ministères, BP 1000, Yaoundé",
        source=SOURCE_MINJUSTICE,
    ),
    # ---- Droits et corruption ----
    _fiche(
        id="cdhc", theme="droits",
        nom="Commission des droits de l'homme du Cameroun (CDHC)",
        ville="Yaoundé", telephones=["1523", "242 22 61 17"], site="https://www.cdhc.cm",
        adresse="Immeuble SGBC, 2e étage, face à la Chambre d'agriculture, BP 20317, Yaoundé",
        note="1523 : numéro vert. Violations des droits de l'homme, abus d'une autorité.",
        source=SOURCE_CDHC,
    ),
    _fiche(
        id="conac", theme="corruption",
        nom="Commission nationale anti-corruption (CONAC)",
        ville="Yaoundé", telephones=["1517", "658 26 26 82"], email="info@conac.cm",
        site="https://conac.cm", adresse="BP 33200, Yaoundé",
        note="1517 : numéro vert, gratuit, signalement anonyme possible. 658 26 26 82 : WhatsApp.",
        source=SOURCE_CONAC,
    ),
    # ---- Consommation ----
    _fiche(
        id="mincommerce", theme="consommation",
        nom="Ministère du Commerce — signalement des pratiques illicites",
        ville="Yaoundé", telephones=["1502"], site="https://www.mincommerce.cm",
        note="Arnaques, contrefaçon, prix abusifs. Numéro annoncé en 2019 : à vérifier.",
        source=SOURCE_MINCOMMERCE,
    ),
    # ---- Urgences ----
    _fiche(
        id="police", theme="urgence", nom="Police secours",
        telephones=["117"], note="Urgence, agression, vol en cours.", source=SOURCE_URGENCES,
    ),
    _fiche(
        id="pompiers", theme="urgence", nom="Sapeurs-pompiers",
        telephones=["118"], source=SOURCE_URGENCES,
    ),
    _fiche(
        id="urgences-medicales", theme="urgence", nom="Urgences médicales",
        telephones=["119"], source=SOURCE_URGENCES,
    ),
]

THEMES: Dict[str, str] = {
    "travail": "Travail", "social": "Sécurité sociale", "avocats": "Avocats",
    "justice": "Justice", "droits": "Droits de l'homme", "corruption": "Corruption",
    "consommation": "Consommation", "urgence": "Urgences",
}

# Ce qui, dans une reponse, renvoie vers une institution : (motif, fiches).
# Les motifs sont lus sur le texte plie (minuscules, sans accents).
_RENVOIS = [
    (r"inspect(ion|eur|rice)s? du travail|delegation (regionale |departementale )?du travail"
     r"|labou?r inspect", ["drtss-*", "mintss"]),
    (r"\bcnps\b|prevoyance sociale|social insurance fund", ["cnps"]),
    (r"\bavocats?\b|\bbarreau\b|\blawyers?\b|\bbar association\b", ["barreau"]),
    (r"\bconac\b|anti-?corruption|denonc\w* (la |un acte de )?corruption", ["conac"]),
    (r"droits de l'homme|\bcdhc\b|human rights commission", ["cdhc"]),
    (r"\bpolice\b|commissariat|\bgendarmerie\b", ["police"]),
    (r"ministere du commerce|\bmincommerce\b|protection du consommateur", ["mincommerce"]),
]


def _plier(texte: str) -> str:
    sans_accents = unicodedata.normalize("NFKD", texte)
    return "".join(c for c in sans_accents if not unicodedata.combining(c)).casefold()


def contacts_pour(reponse: str) -> List[Dict]:
    """Les fiches des institutions vers lesquelles la reponse renvoie, sans doublon."""
    plie = _plier(reponse or "")
    ids: List[str] = []
    for motif, cibles in _RENVOIS:
        if re.search(motif, plie):
            for cible in cibles:
                if cible not in ids:
                    ids.append(cible)
    fiches: List[Dict] = []
    for cible in ids:
        if cible.endswith("*"):
            fiches.extend(f for f in ANNUAIRE if f["id"].startswith(cible[:-1]))
        else:
            fiches.extend(f for f in ANNUAIRE if f["id"] == cible)
    return fiches


def lister(theme: Optional[str] = None, region: Optional[str] = None) -> List[Dict]:
    """L'annuaire, filtre par theme et par region (les fiches nationales restent)."""
    return [
        f for f in ANNUAIRE
        if (theme is None or f["theme"] == theme)
        and (region is None or f["region"] in (None, region))
    ]
