"""
Tests du classement en domaine juridique, par lots, via Groq.

Le service Groq est double : chaque test programme ce que « le modele »
repond, et verifie ce que le classifieur en fait — noms complets, lots
incomplets redemandes, documents « a revoir », et surtout AUCUN domaine
invente quand Groq ne repond pas. Un seul test appelle le vrai Groq
(`groq_live`) : un lot de 5 lois, donc une requete.

Usage:
    pytest tests/test_services/test_legal_domain_classifier.py -v
    pytest tests/test_services/test_legal_domain_classifier.py -m groq_live
"""

import importlib.util
from pathlib import Path

import pytest

from app.services.groq_service import (
    GroqIndisponibleError,
    GroqQuotaError,
    GroqReponseInvalideError,
    ReponseGroq,
)
from app.services.legal_domain_classifier import (
    ADMINISTRATIF,
    AFFAIRES,
    CANONICAL_DOMAINS,
    CODES_DOMAINES,
    DESCRIPTIONS_DOMAINES,
    EDUCATION,
    ENVIRONNEMENT,
    FINANCES,
    FONCTION_PUBLIQUE,
    PENAL,
    SANTE,
    ClassementIndisponible,
    DocumentAClasser,
    DocumentVide,
    LegalDomainClassifier,
    extrait_pour_classement,
    get_legal_domain_classifier,
)

MODELE = "openai/gpt-oss-120b"


class FauxGroq:
    """
    Doublure de GroqService : chaque appel rend la reponse suivante (un dict
    de verdicts, ou une exception a lever), et garde le message envoye.
    """

    def __init__(self, *reponses):
        self.reponses = list(reponses)
        self.messages = []
        self.appels = []

    def _repondre(self, reglages):
        self.appels.append(reglages)
        self.messages.append(reglages["message"])
        reponse = self.reponses.pop(0)
        if isinstance(reponse, Exception):
            raise reponse
        return ReponseGroq(reponse, 100, reglages["modele"])

    def completer_json_sync(self, **reglages):
        return self._repondre(reglages)

    async def completer_json(self, **reglages):
        return self._repondre(reglages)


def verdicts(*lignes):
    """verdicts((1, "FIN"), (2, "FP", ["ADMIN"], 0.7)) : la reponse du modele."""
    sortie = []
    for n, code, *reste in lignes:
        secondaires = reste[0] if reste else []
        confiance = reste[1] if len(reste) > 1 else 0.9
        sortie.append({"n": n, "domaine": code, "secondaires": secondaires, "confiance": confiance})
    return {"verdicts": sortie}


def documents(n):
    return [DocumentAClasser(titre=f"Décret n°{i}") for i in range(1, n + 1)]


def classeur(*reponses):
    faux = FauxGroq(*reponses)
    return LegalDomainClassifier(groq=faux, modele=MODELE), faux


class TestSurface:
    def test_quatorze_domaines(self):
        assert len(CANONICAL_DOMAINS) == 14
        assert len(set(CANONICAL_DOMAINS)) == 14

    def test_aucun_type_de_document_parmi_les_domaines(self):
        for interdit in ("Lois", "Décrets", "Arrêtés", "Ordonnances", "Autres"):
            assert interdit not in CANONICAL_DOMAINS

    def test_un_code_par_domaine(self):
        assert sorted(CODES_DOMAINES.values()) == sorted(CANONICAL_DOMAINS)

    def test_descriptions_identiques_a_la_migration(self):
        """Le modele classe sur la definition que l'interface affiche."""
        chemin = (
            Path(__file__).resolve().parents[2]
            / "alembic" / "versions" / "013_harmonize_categories.py"
        )
        spec = importlib.util.spec_from_file_location("migration_013", chemin)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert DESCRIPTIONS_DOMAINES == {
            nom: description for nom, _, description in module.CANONICAL_CATEGORIES
        }

    def test_singleton(self):
        assert get_legal_domain_classifier() is get_legal_domain_classifier()

    def test_sante_declarative(self):
        rapport = LegalDomainClassifier(modele=MODELE).health_check()

        assert rapport["domains"] == 14
        assert rapport["model"] == MODELE


class TestConsignes:
    def test_chaque_code_et_chaque_arbitrage(self):
        consignes = LegalDomainClassifier.consignes()

        for code, nom in CODES_DOMAINES.items():
            assert f"{code} — {nom} : {DESCRIPTIONS_DOMAINES[nom]}" in consignes
        assert "Accord de prêt" in consignes
        assert "visas" in consignes

    def test_documents_numerotes(self):
        message = LegalDomainClassifier.message([
            DocumentAClasser(titre="Loi  portant\nCode minier", type_acte="loi"),
            DocumentAClasser(titre="Décret n°2", extrait="Article premier : est nommé..."),
        ])

        assert message == (
            "1. [loi] Loi portant Code minier\n"
            "2. Décret n°2\n"
            "   Extrait : Article premier : est nommé..."
        )

    def test_schema_strict_a_chaque_niveau(self):
        schema = LegalDomainClassifier.schema()
        verdict = schema["properties"]["verdicts"]["items"]

        assert schema["additionalProperties"] is False
        assert schema["required"] == ["verdicts"]
        assert verdict["additionalProperties"] is False
        assert set(verdict["required"]) == set(verdict["properties"])
        assert verdict["properties"]["domaine"]["enum"] == list(CODES_DOMAINES)


class TestExtrait:
    def test_part_de_l_article_premier(self):
        texte = (
            "DECRET N°2018/420\n"
            "Vu la Constitution ;\n"
            "Vu la loi n°2001/001, notamment son article 1er ;\n"
            "DECRETE :\n"
            "Article 1er.- M. X est nommé Secrétaire Général."
        )

        assert extrait_pour_classement(texte) == "Article 1er.- M. X est nommé Secrétaire Général."

    def test_sans_article_premier_les_visas_sont_retires(self):
        texte = "Vu la Constitution ;\nVu le décret n°1 ;\nLe Premier Ministre arrête la liste."

        assert extrait_pour_classement(texte) == "Le Premier Ministre arrête la liste."

    def test_longueur_et_espaces(self):
        extrait = extrait_pour_classement("Article premier\n\n" + "mot   " * 500)

        assert len(extrait) == 500
        assert "  " not in extrait

    def test_texte_vide(self):
        assert extrait_pour_classement(None) == ""


class TestLot:
    def test_noms_complets_secondaires_et_confiance(self):
        classifieur, faux = classeur(verdicts(
            (1, "FIN", ["INTL", "FIN", "INCONNU"], 0.93),
            (2, "FP", [], 1.7),
        ))

        resultat = classifieur.classer_lot(documents(2))

        premier, second = resultat.verdicts
        assert premier.domain == FINANCES
        # Le domaine retenu et les codes inconnus sont retires des secondaires.
        assert [nom for nom, _ in premier.runners_up] == ["Droit International"]
        assert premier.confidence == 0.93
        assert premier.rule == f"groq:{MODELE}"
        assert second.domain == FONCTION_PUBLIQUE
        assert second.confidence == 1.0  # bornee
        assert (resultat.requetes, resultat.jetons, resultat.modele) == (1, 100, MODELE)
        assert faux.appels[0]["modele"] == MODELE

    def test_les_manquants_sont_redemandes_seuls(self):
        classifieur, faux = classeur(
            verdicts((1, "FIN"), (3, "ENV")),
            verdicts((1, "SANTE")),
        )

        resultat = classifieur.classer_lot(documents(3))

        assert [v.domain for v in resultat.verdicts] == [FINANCES, SANTE, ENVIRONNEMENT]
        assert faux.messages[1] == "1. Décret n°2"

    def test_un_document_toujours_absent_est_a_revoir(self):
        classifieur, faux = classeur(verdicts((1, "FIN")), verdicts())

        resultat = classifieur.classer_lot(documents(2))

        assert resultat.verdicts[0].domain == FINANCES
        assert resultat.verdicts[1] is None
        assert resultat.requetes == 2

    def test_rien_d_exploitable_coupe_le_lot_en_deux(self):
        classifieur, faux = classeur(
            GroqReponseInvalideError("coupee"),
            verdicts((1, "FIN"), (2, "FP")),
            verdicts((1, "PENAL"), (2, "EDUC")),
        )

        resultat = classifieur.classer_lot(documents(4))

        assert [v.domain for v in resultat.verdicts] == [FINANCES, FONCTION_PUBLIQUE, PENAL, EDUCATION]
        assert faux.messages[1] == "1. Décret n°1\n2. Décret n°2"
        assert faux.messages[2] == "1. Décret n°3\n2. Décret n°4"

    def test_numeros_hors_lot_doublons_et_codes_inconnus_ignores(self):
        classifieur, faux = classeur(
            {"verdicts": [
                {"n": 1, "domaine": "FIN", "secondaires": [], "confiance": 0.9},
                {"n": 1, "domaine": "FP", "secondaires": [], "confiance": 0.9},
                {"n": 7, "domaine": "FP", "secondaires": [], "confiance": 0.9},
                {"n": 2, "domaine": "DROIT_MARTIEN", "secondaires": [], "confiance": 0.9},
            ]},
            verdicts((1, "AFFAIRES")),
        )

        resultat = classifieur.classer_lot(documents(2))

        assert [v.domain for v in resultat.verdicts] == [FINANCES, AFFAIRES]

    def test_quota_epuise_rien_d_invente(self):
        """L'ancien classifieur rendait ici « Droit Administratif », confiance 0,20."""
        classifieur, faux = classeur(
            GroqQuotaError("TPD", retry_after=3600, raison="quota Groq")
        )

        with pytest.raises(ClassementIndisponible) as erreur:
            classifieur.classer_lot(documents(3))

        assert erreur.value.quota is True
        assert erreur.value.retry_after == 3600

    def test_panne(self):
        classifieur, faux = classeur(GroqIndisponibleError("503"))

        with pytest.raises(ClassementIndisponible) as erreur:
            classifieur.classer_lot(documents(1))

        assert erreur.value.quota is False

    def test_lot_vide(self):
        classifieur, faux = classeur()

        assert classifieur.classer_lot([]).verdicts == []
        assert faux.appels == []

    async def test_version_asynchrone(self):
        classifieur, faux = classeur(verdicts((2, "SANTE")), verdicts((1, "EDUC")))

        resultat = await classifieur.classer_lot_async(documents(2))

        assert [v.domain for v in resultat.verdicts] == [EDUCATION, SANTE]


class TestUnDocument:
    def test_classify_envoie_titre_type_et_extrait(self):
        classifieur, faux = classeur(verdicts((1, "FP")))

        resultat = classifieur.classify(
            "Décret portant nomination", "Vu la Constitution ;\nArticle 1er.- Est nommé...", "decret"
        )

        assert resultat.domain == FONCTION_PUBLIQUE
        assert faux.messages[0] == (
            "1. [decret] Décret portant nomination\n   Extrait : Article 1er.- Est nommé..."
        )

    def test_document_vide(self):
        classifieur, faux = classeur()

        with pytest.raises(DocumentVide):
            classifieur.classify("", "   ")
        assert faux.appels == []

    def test_texte_reduit_aux_visas_est_vide(self):
        """Les visas sont retires de l'extrait : il ne reste rien a classer."""
        classifieur, faux = classeur()

        with pytest.raises(DocumentVide):
            classifieur.classify("", "Vu la Constitution ;\nVu le décret n°2011/408 ;")
        assert faux.appels == []

    def test_aucun_verdict(self):
        classifieur, faux = classeur(verdicts(), verdicts())

        with pytest.raises(ClassementIndisponible):
            classifieur.classify("Décret")

    async def test_classify_async(self):
        classifieur, faux = classeur(verdicts((1, "ADMIN")))

        assert (await classifieur.classify_async("Arrêté")).domain == ADMINISTRATIF


@pytest.mark.groq_live
def test_vrai_groq_un_lot_de_cinq():
    """
    Le vrai modele, sur cinq cas qui ont chacun pose probleme. UNE requete.

    Les verdicts attendus sont ceux des regles d'arbitrage : un accord de pret
    va en Finances meme s'il est ratifie, un code minier en Environnement.
    """
    cas = [
        ("Loi N°2015/019 portant Loi de finances pour l'exercice 2016", FINANCES),
        ("Loi N°2023/014 portant Code Minier", ENVIRONNEMENT),
        ("Décret N°2018/420 portant nomination du Secrétaire Général", FONCTION_PUBLIQUE),
        ("Loi portant Code de Procédure Pénale", PENAL),
        ("Décret ratifiant l'accord de prêt avec la BAD", FINANCES),
    ]

    resultat = LegalDomainClassifier().classer_lot(
        [DocumentAClasser(titre=titre) for titre, _ in cas]
    )

    assert resultat.requetes == 1
    assert [v.domain if v else None for v in resultat.verdicts] == [attendu for _, attendu in cas]
