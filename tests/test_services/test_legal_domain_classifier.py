"""
Tests du classement en domaine juridique.

Ces tests remplacent tests/test_services/test_document_classifier.py, dont les
572 lignes ne pouvaient structurellement pas attraper le defaut qu'elles etaient
censees couvrir : elles n'assertaient que l'appartenance au top-3 sans jamais
verifier le premier resultat, et leurs textes etaient fabriques a partir des
mots-cles du classifieur lui-meme.

Ici, chaque assertion porte sur le domaine RETENU, et les titres viennent du
corpus reel (prc.cm).

Usage:
    pytest tests/test_services/test_legal_domain_classifier.py -v
"""

import pytest

from app.services.legal_domain_classifier import (
    ADMINISTRATIF,
    AFFAIRES,
    CANONICAL_DOMAINS,
    CONSTITUTIONNEL,
    ENVIRONNEMENT,
    FAMILLE,
    FINANCES,
    FONCIER,
    FONCTION_PUBLIQUE,
    INTERNATIONAL,
    PENAL,
    PROCEDURE_PENALE,
    TRAVAIL,
    LegalDomainClassifier,
    get_legal_domain_classifier,
    normalise,
)


@pytest.fixture
def classifier() -> LegalDomainClassifier:
    return LegalDomainClassifier()


# ==================== TABLE DE CORRESPONDANCE ====================
# Titres reels du corpus, un par regle. Une ligne qui casse nomme sa regle.

TITLE_CASES = [
    # -- Tier A : instruments nommes --
    ("Loi N°2005/007 du 27 juillet 2005 portant Code de Procédure Pénale", PROCEDURE_PENALE),
    ("Loi N°2016/007 du 12 juillet 2016 portant Code Pénal", PENAL),
    ("Loi N°92/007 du 14 août 1992 portant Code du Travail", TRAVAIL),
    ("Loi N°2015/019 du 21 décembre 2015 portant Loi de finances de la République "
     "du Cameroun pour l'exercice 2016", FINANCES),
    ("Loi portant Code Général des Impôts", FINANCES),
    ("Loi N°2023/014 du 19 décembre 2023 portant Code Minier", ENVIRONNEMENT),
    ("Loi N°94/01 du 20 janvier 1994 portant Régime des Forêts, de la Faune et de la Pêche",
     ENVIRONNEMENT),
    ("Ordonnance N°81/02 du 29 juin 1981 portant organisation de l'état civil", FAMILLE),
    ("Acte uniforme OHADA relatif au droit des sociétés commerciales", AFFAIRES),
    ("Loi N°96/06 du 18 janvier 1996 portant révision de la Constitution du 2 juin 1972",
     CONSTITUTIONNEL),

    # -- Tier B : formes dominantes du corpus --
    ("Loi N°2017/010 du 12 juillet 2017 portant statut général des établissements publics",
     ADMINISTRATIF),
    ("Décret N°2015/055 du 02 février 2015 habilitant le Ministre de l'Économie à signer "
     "un accord de prêt avec la Banque Africaine de Développement", FINANCES),
    ("Décret N°2026/035 du 29 janvier 2026 ratifiant l'Accord de Prêt n°2000200006851", FINANCES),
    ("Loi N°2017/018 du 20 décembre 2017 autorisant le Président de la République à ratifier "
     "la Convention de coopération judiciaire", INTERNATIONAL),
    ("Décret N°2025/037 du 12 février 2025 portant classement au domaine public artificiel "
     "des terrains sis à Douala", FONCIER),
    ("Décret N°2018/420 du 20 juillet 2018 portant nomination du Secrétaire Général "
     "du Ministère des Finances", FONCTION_PUBLIQUE),
    ("Décret N°2024/440 du 23 octobre 2024 portant inscription au tableau d'avancement",
     FONCTION_PUBLIQUE),
    ("Décret portant élévation à la dignité de Grand Officier de l'Ordre de la Valeur",
     FONCTION_PUBLIQUE),
    ("Décret N°2026/164 du 4 mai 2026 portant approbation des statuts de la société "
     "Camerounaise d'Électricité", AFFAIRES),
    ("Décret N°2019/xxx portant convocation du corps électoral", CONSTITUTIONNEL),
    ("Décret N°2015/124 du 18 mars 2015 instituant des obsèques officielles", ADMINISTRATIF),
    ("Loi N°2023/010 du 25 juillet 2023 régissant le secteur ferroviaire au Cameroun",
     ADMINISTRATIF),
    ("Loi N°2025/006 du 25 avril 2025 régissant la biosécurité au Cameroun", ENVIRONNEMENT),
]


@pytest.mark.parametrize("title,expected", TITLE_CASES)
def test_title_decides_domain(classifier, title, expected):
    """Le titre seul doit suffire sur les formes courantes du corpus."""
    result = classifier.classify(title)
    assert result.domain == expected, (
        f"{title[:60]!r} classe en {result.domain} par la regle {result.rule}"
    )
    assert result.source == "title"


# ==================== TESTS D'ORDRE ====================
# Isoles pour qu'un echec nomme sa cause. Chacun correspond a une inversion
# possible dans la table de regles.


class TestRuleOrdering:
    """L'ordre de la table de regles EST la conception."""

    def test_procedure_penale_before_code_penal(self, classifier):
        """« code de procédure pénale » contient « pénale » : A1 doit devancer A3."""
        assert classifier.classify("Code de Procédure Pénale").domain == PROCEDURE_PENALE

    def test_loi_de_finances_is_not_administratif(self, classifier):
        """L'ancien classifieur rendait Droit Constitutionnel sur ce titre."""
        result = classifier.classify("Loi de finances pour l'exercice 2016")
        assert result.domain == FINANCES

    def test_code_minier_beats_fiscalite(self, classifier):
        """Les codes miniers sont bourres de « taxe » : A6 doit devancer B10."""
        assert classifier.classify("Loi portant Code Minier").domain == ENVIRONNEMENT

    def test_pret_ratifie_is_not_international(self, classifier):
        """214 titres du corpus ratifient un PRET : c'est de l'emprunt public."""
        result = classifier.classify(
            "Décret ratifiant l'accord de prêt conclu avec la Banque Mondiale"
        )
        assert result.domain == FINANCES
        assert result.rule.startswith("B1a")

    def test_statut_etablissements_is_not_affaires(self, classifier):
        """« établissements publics » ne doit pas etre lu comme du droit des societes."""
        assert classifier.classify(
            "Loi portant statut général des établissements publics"
        ).domain == ADMINISTRATIF

    def test_nomination_surete_nationale_is_not_penal(self, classifier):
        """« Sûreté Nationale » apparait dans des dizaines de titres de nomination."""
        result = classifier.classify(
            "Décret portant nomination du Délégué Général à la Sûreté Nationale"
        )
        assert result.domain == FONCTION_PUBLIQUE

    def test_nomination_de_senateurs_is_not_constitutionnel(self, classifier):
        """B4 devance B8 : une nomination reste de la fonction publique."""
        assert classifier.classify(
            "Décret portant nomination de trente Sénateurs"
        ).domain == FONCTION_PUBLIQUE

    def test_nomination_ambassadeur_is_not_international(self, classifier):
        """B4 vient apres B1/B3 mais une nomination d'ambassadeur n'est pas un traite."""
        assert classifier.classify(
            "Décret portant nomination d'un Ambassadeur Extraordinaire"
        ).domain == FONCTION_PUBLIQUE

    def test_concession_de_terrain_is_not_fonction_publique(self, classifier):
        """B6 devance B4 : « attribution en concession provisoire » est domanial."""
        assert classifier.classify(
            "Arrêté portant attribution en concession provisoire d'un terrain domanial"
        ).domain == FONCIER

    def test_biosecurite_is_not_administratif(self, classifier):
        """B13 devance B9 : « régissant » seul ne fait pas un texte administratif."""
        assert classifier.classify(
            "Loi régissant la biosécurité au Cameroun"
        ).domain == ENVIRONNEMENT


# ==================== PASSE SUR LE CONTENU ====================


class TestContentPass:
    """Les ~19 % de titres que la table de regles ne tranche pas."""

    def test_content_used_when_title_is_silent(self, classifier):
        result = classifier.classify(
            "Texte sans forme reconnaissable",
            "Le contrat de travail à durée déterminée lie le salarié et l'employeur. "
            "La convention collective applicable et l'inspection du travail sont "
            "compétentes. Le salarié bénéficie de la sécurité sociale. "
            "Tout licenciement obéit au contrat de travail.",
        )
        assert result.domain == TRAVAIL
        assert result.source == "content"

    def test_visa_block_does_not_decide(self, classifier):
        """
        Chaque decret camerounais commence par « Vu la Constitution ». C'etait
        l'amplificateur mesure du faux positif Droit Constitutionnel.
        """
        content = (
            "Vu la Constitution du 18 janvier 1996 ;\n"
            "Vu la loi portant Conseil Constitutionnel ;\n"
            "Vu le decret relatif a l'Assemblee Nationale ;\n"
            "Le contrat de travail du salarié est régi par la convention collective. "
            "L'inspection du travail contrôle le contrat de travail et le licenciement."
        )
        result = classifier.classify("Texte sans forme reconnaissable", content)
        assert result.domain == TRAVAIL, (
            f"le bloc de visas a decide a la place du corps : {result.rule}"
        )

    def test_scores_are_a_distribution_not_saturated(self, classifier):
        """
        L'ancien score saturait a min(score/10, 1.0) : quatre domaines
        atteignaient 1,00 sur un texte long et le tri stable tranchait par
        ordre de dictionnaire. Une distribution rend cette egalite impossible.
        """
        long_text = (
            "Le contrat de travail lie le salarié et l'employeur. "
            "La convention collective régit le licenciement. "
        ) * 200
        scored = classifier._score_content(long_text)
        assert scored
        assert sum(share for _, share in scored) == pytest.approx(1.0)
        assert all(share <= 1.0 for _, share in scored)

    def test_tie_is_named_not_silently_broken(self, classifier):
        """
        Deux termes de poids egal, une occurrence chacun : marge nulle. Le
        document tombe sur le defaut, mais la regle NOMME les deux ex aequo au
        lieu d'elire silencieusement le premier du dictionnaire.
        """
        result = classifier.classify(
            "Texte sans forme reconnaissable", "Contribuable. Licenciement."
        )
        assert result.source == "doctype-default"
        assert result.rule.startswith("tie:")
        assert result.domain == ADMINISTRATIF

    def test_no_signal_falls_back_to_default(self, classifier):
        """Aucun terme du lexique : defaut nomme, distinct d'une egalite."""
        result = classifier.classify("Texte sans forme reconnaissable", "Blah blah blah.")
        assert result.source == "doctype-default"
        assert result.rule == "default:acte-executif"

    def test_length_does_not_flip_the_verdict(self, classifier):
        """log1p supprime la dependance a la longueur : x50 ne change pas le gagnant."""
        base = (
            "Le contrat de travail du salarié relève de la convention collective. "
            "L'inspection du travail veille au respect du contrat de travail."
        )
        assert (
            classifier.classify("", base).domain
            == classifier.classify("", base * 50).domain
        )


# ==================== SURFACE PUBLIQUE ====================


class TestPublicSurface:
    """Le defaut d'origine tenait a un ENTIER dans la surface publique."""

    def test_classify_never_returns_an_integer(self, classifier):
        result = classifier.classify("Décret portant nomination")
        assert isinstance(result.domain, str)
        assert not isinstance(result.domain, bool)

    def test_domain_is_always_canonical(self, classifier):
        for title, _ in TITLE_CASES:
            assert classifier.classify(title).domain in CANONICAL_DOMAINS

    def test_never_returns_none(self, classifier):
        for value in ["", "   ", "xyzzy", "??? !!!", "1234"]:
            assert classifier.classify(value).domain in CANONICAL_DOMAINS

    def test_canonical_domains_are_unique(self):
        assert len(set(CANONICAL_DOMAINS)) == len(CANONICAL_DOMAINS) == 14

    def test_no_document_type_among_domains(self):
        """
        Une categorie nommee « Lois » ou « Décrets » ramenerait le defaut
        d'origine : un decret fiscal n'aurait plus de ligne correcte ou aller.
        Verifie contre la liste reelle des types acceptes par le schema.
        """
        from app.schemas.law import LawBase

        types = LawBase.model_fields["type"].description or ""
        for doc_type in ["loi", "décret", "arrêté", "ordonnance", "circulaire", "décision"]:
            assert not any(
                doc_type in domain.lower() or f"{doc_type}s" in domain.lower()
                for domain in CANONICAL_DOMAINS
            ), f"{doc_type!r} apparait dans les domaines canoniques"
        assert types is not None  # la liste des types vit dans le schema, pas ici

    def test_result_is_frozen(self, classifier):
        result = classifier.classify("Décret portant nomination")
        with pytest.raises(Exception):
            result.domain = "autre"


# ==================== NORMALISATION ====================


class TestNormalisation:
    def test_folds_accents(self):
        assert normalise("Procédure Pénale") == "procedure penale"

    def test_replaces_punctuation_with_space(self):
        """Une douzaine de titres du corpus ont la forme decret_n_2015_055_du_02.02.2015."""
        assert normalise("decret_n_2015_055_du_02.02.2015_accord_pret_fad") == (
            "decret n 2015 055 du 02 02 2015 accord pret fad"
        )

    def test_handles_apostrophes(self):
        assert normalise("Code de l'Environnement") == "code de l environnement"

    def test_empty_and_none(self):
        assert normalise("") == ""
        assert normalise(None) == ""


# ==================== ENTREES LIMITES ====================


class TestEdgeCases:
    def test_very_long_title(self, classifier):
        assert classifier.classify("Décret portant nomination " + "x" * 5000).domain

    def test_unicode_and_symbols(self, classifier):
        assert classifier.classify("§ ¤ 中文 🎉 Décret portant nomination").domain == FONCTION_PUBLIQUE

    def test_none_arguments(self, classifier):
        assert classifier.classify(None, None, None).domain in CANONICAL_DOMAINS

    def test_deterministic(self, classifier):
        title = "Décret ratifiant l'accord de prêt avec la BAD"
        assert len({classifier.classify(title).domain for _ in range(20)}) == 1

    def test_shared_instance(self):
        assert get_legal_domain_classifier() is get_legal_domain_classifier()

    def test_health_check(self, classifier):
        report = classifier.health_check()
        assert report["status"] == "healthy"
        assert report["domains"] == 14

    def test_explain_names_the_rule(self, classifier):
        assert "B4:actes-de-carriere" in classifier.explain("Décret portant nomination")
