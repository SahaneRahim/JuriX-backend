"""
Le contrôle IA d'un article du blog, sa relecture et son état de publication
(app/services/controle_du_billet.py, tours 3 et 4).

Ce que ces tests protègent, dans l'ordre d'importance :

  1. UN ARTICLE N'EST PUBLIABLE QUE SI TOUT TIENT : ses sources (tour 4),
     des citations vérifiées (au moins une), un contrôle à jour, aucune phrase
     contredite, chaque partie relue, le TITRE compris (tour 4). Sinon le
     PREMIER blocage, dans un ordre fixe.
  2. LES EMPREINTES : tout changement d'une partie, ou des sources, périme le
     contrôle. Une case atteste le texte de sa partie, les sources ET les
     verdicts que le contrôle lui donne : cochée avant le contrôle, ou jugée
     autrement par un nouveau contrôle, elle tombe (revue du tour 3).
  3. LA RÉPONSE DU MODÈLE est lue sans jamais inventer un accord : phrase
     oubliée ou verdict illisible = « non_soutenue » ; une phrase dite
     soutenue ou contredite sans passage réel des extraits = « non_soutenue »
     (tour 4 : au tour 3, le modèle inventait l'appui) ; réponse coupée,
     illisible ou vide = erreur 502.
  4. LE CONTRÔLE : un appel, température 0, schéma strict, sources sous leurs
     numéros puis passages des citations ajoutées (gardés pour
     l'administration), transaction close avant le modèle, erreurs
     traduites ; sans sources, aucun appel.
  5. LE MODÈLE DU CONTRÔLE (tour 6, mistral-large-4) : la requête porte ce
     que ce modèle exige (reasoning_effort, top_p 1 à la température 0), et
     celles du chat et de la génération ne changent pas ; une réponse en
     morceaux se lit sans sa réflexion ; un refus 400 ou un 429 devient une
     erreur claire, jamais un autre modèle ; controle_ia.modele est celui qui
     a répondu ; un contrôle à la fois, ses appels l'un après l'autre, dans le
     budget de jetons ; 25 phrases au plus, preuve par la mesure.
  6. LE PLAN DU CONTRÔLE (revue du tour 6, TestLePlanDuControle) : avant son
     premier lot, le contrôle compte ensemble l'attente d'un autre contrôle,
     celles du budget de jetons et la durée de ses lots, dans l'échéance de sa
     route. Un plan qui ne tient pas est refusé AVANT tout envoi (503, le
     Retry-After où il tiendra, et relancé alors, il aboutit) ; sinon chaque
     lot attend ce que le plan lui laisse, et non les 20 s du chat. Sur
     l'article RÉEL, au coût et au rythme mesurés le 04/11. Tour 7 : UN appel
     par article pour le modèle à budget de jetons, compté au pire mesuré
     d'un appel (140 s), quelle que soit sa taille.
  7. LES PHRASES À CONFIRMER (tour 7, décision du 04/11) : une phrase de règle
     non trouvée dans les extraits se réécrit ou se confirme une à une avant
     publication ; une confirmation tombe comme une case.

Le modèle est une doublure (tests/blog_donnees.py), ou le vrai client sur un
httpx.MockTransport (sections 7 et 7 bis) ; le vrai modèle n'est jamais appelé.

Usage:
    pytest tests/test_services/test_controle_du_billet.py -v
"""

import asyncio
import json
import math
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from pydantic import ValidationError
from sqlalchemy import text

from app.core.config import Settings, settings
from app.core.limiteur import Limiteur
from app.services import controle_du_billet, mistral_service
from app.services.blog_generator_service import (
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    SCHEMA_DE_L_ARTICLE,
    SYSTEM_PROMPT,
    BlogGeneratorService,
    DemandeDeCitation,
    QuotaError,
    ReponseInvalideError,
    SaturationError,
    citations_verifiees,
)
from app.services.controle_du_billet import (
    CONTROLE_MAX_TOKENS,
    CONTROLE_TEMPERATURE,
    DELAI_CONTROLE_S,
    DUREE_D_UN_APPEL_AU_PIRE_S,
    ETATS,
    JETONS_DE_SORTIE_D_UN_APPEL_AU_PIRE,
    JETONS_PAR_SECONDE_AU_PIRE,
    MARGE_DU_PLAN_S,
    PARTIES,
    PHRASES_AU_PLUS,
    PHRASES_PAR_APPEL,
    PORTEE_D_UNE_COUPURE,
    RAISON_NON_JUGEE,
    SCHEMA_DU_CONTROLE,
    SYSTEM_DU_CONTROLE,
    BudgetDuControle,
    BudgetDuControleError,
    ContenuDuBillet,
    ControleDuBillet,
    ExtraitDuControle,
    PhraseAJuger,
    SourcesAbsentesError,
    TexteTropLongError,
    appuis_du_passage,
    confirmations_perimees,
    controle_affiche,
    decouper_en_phrases,
    empreinte_du_controle,
    empreintes_a_confirmer,
    empreintes_de_relecture,
    empreintes_des_parties,
    etat_de_publication,
    extrait_du_passage,
    extraits_du_controle,
    fait_le_apres,
    lire_la_nature,
    lire_le_passage,
    lire_le_passage_en_detail,
    lire_le_verdict,
    lire_les_verdicts,
    lots_du_controle,
    memes_sources,
    motif_du_refus,
    nouvelle_relecture,
    parties_relues,
    passage_present,
    phrases_a_confirmer,
    phrases_confirmees,
    phrases_du_billet,
    planifier,
    prompt_du_controle,
    relecture_affichee,
    relecture_apres_un_controle,
    relecture_perimee,
    renvois_aux_extraits,
    sources_affichees,
    sources_valables,
)
from app.services.gemini_service import GeminiBudgetEpuiseError, GeminiQuotaError
from app.services.mistral_service import (
    MistralOverloadedError,
    MistralQuotaError,
    MistralService,
    MistralServiceError,
    schema_strict,
    texte_des_morceaux,
)
from tests.blog_donnees import (
    APPELS_MESURES,
    SUJET,
    TITRE_CODE_PENAL,
    GrandModeleAuRythme,
    RechercheDoublee,
    appels_du_controle,
    article_du_modele,
    article_reel,
    creer_les_textes,
    extraits_du_sujet,
    jugement,
    modele_double,
    phrases_du_prompt,
    reponse_du_controle,
    similarites,
)


def _citation(n: int = 1, verifie: bool = True, **champs) -> dict:
    return {
        "law_id": 10 + n, "article_id": 100 + n, "article_number": str(n), "suite": False,
        "verifie": verifie, "title": f"Explication {n}.", "excerpt": "x", **champs,
    }


_SOURCES = [
    {"n": 1, "law_id": 11, "law_title": "Code X", "article_number": "1", "article_id": 101,
     "suite": False, "extrait": "Nul ne peut entrer dans le domicile d'autrui contre son gré."},
    {"n": 2, "law_id": 12, "law_title": "Code Y", "article_number": "2", "article_id": 102,
     "suite": False, "extrait": "La juridiction compétente prononce l'expulsion du preneur."},
]


def _contenu(**champs) -> ContenuDuBillet:
    """P1 titre ; P2, P3 contexte ; P4, P5 résumé ; P6, P7 points clés ; P8 impact ; P9, P10 explications ; P11 question."""
    valeurs = dict(
        title="Expulsion sans juge : ce que dit la loi",
        context="Un locataire rentre chez lui. Ses affaires sont dehors.",
        summary="La loi punit l'intrusion. Le juge décide.",
        key_points=["Entrer chez autrui est puni.", "Le congé suit l'usage."],
        citizen_impact="Portez plainte.",
        citations=[_citation(1), _citation(2)],
        suggested_question="Que faire si mon bailleur change la serrure ?",
        sources=[dict(s) for s in _SOURCES],
    )
    valeurs.update(champs)
    return ContenuDuBillet(**valeurs)


def _controle_sur(contenu: ContenuDuBillet, *contredites: str) -> dict:
    """Un contrôle à jour de `contenu`, avec les phrases (P…) contredites données."""
    phrases = phrases_du_billet(contenu)
    return {
        "fait_le": "2026-10-30T10:00:00+00:00", "modele": "modele-double",
        "empreinte": empreinte_du_controle(contenu),
        "empreintes_des_parties": empreintes_des_parties(contenu),
        "extraits_ajoutes": [],
        "phrases": [
            {"id": p.id, "partie": p.partie, "index": p.index, "texte": p.texte,
             "verdict": "contredite" if p.id in contredites else "soutenue", "extraits": [1],
             "raison": "r"}
            for p in phrases
        ],
    }


def _tout_relu(contenu: ContenuDuBillet, controle) -> dict:
    return {
        "parties": empreintes_de_relecture(contenu, controle), "par": "admin",
        "le": "2026-10-30T10:00:00+00:00",
    }


def _tout(valeur: bool) -> dict:
    return {p: valeur for p in PARTIES}


# ==================== 1. L'ÉTAT DE PUBLICATION ====================


class TestEtatDePublication:

    def test_publiable_quand_tout_tient(self):
        contenu = _contenu()
        controle = _controle_sur(contenu)

        assert etat_de_publication(contenu, controle, _tout_relu(contenu, controle), [True, True]) == (
            "publiable"
        )

    @pytest.mark.parametrize("sources", [
        [], None, [{"n": 1, "extrait": "   "}], [{"extrait": "Un texte sans numéro."}],
        ["pas un objet"], {"n": 1, "extrait": "pas une liste"},
    ], ids=["vide", "absente", "extrait-vide", "sans-numero", "pas-un-objet", "pas-une-liste"])
    def test_sources_absentes(self, sources):
        """Un article d'avant (le 26/10) ou écrit à la main : rien contre quoi juger ni relire."""
        contenu = _contenu(sources=sources)
        controle = _controle_sur(contenu)

        assert etat_de_publication(contenu, controle, _tout_relu(contenu, controle), [True, True]) == (
            "sources_absentes"
        )

    @pytest.mark.parametrize("valides, citations", [
        ([True, False], None),
        ([False, False], None),
        ([], []),
        ([True], None),  # une validité manquante ne vaut jamais « valide »
    ], ids=["une-invalide", "toutes", "aucune-citation", "longueurs-differentes"])
    def test_citations_non_verifiees(self, valides, citations):
        contenu = _contenu() if citations is None else _contenu(citations=citations)
        controle = _controle_sur(contenu)

        etat = etat_de_publication(contenu, controle, _tout_relu(contenu, controle), valides)

        assert etat == "citations_non_verifiees"

    def test_controle_absent_ou_perime(self):
        contenu = _contenu()
        controle = _controle_sur(contenu)
        modifie = replace(contenu, summary="Un autre résumé.")
        retitre = replace(contenu, title="Un autre titre")
        autres_sources = replace(contenu, sources=[_SOURCES[0]])

        assert etat_de_publication(contenu, None, _tout_relu(contenu, None), [True, True]) == (
            "controle_a_relancer"
        )
        for apres in (modifie, retitre, autres_sources):
            assert etat_de_publication(apres, controle, _tout_relu(apres, controle), [True, True]) == (
                "controle_a_relancer"
            )
        assert etat_de_publication(
            contenu, {"empreinte": controle["empreinte"]}, _tout_relu(contenu, controle), [True, True]
        ) == "controle_a_relancer"

    def test_phrases_contredites(self):
        contenu = _contenu()
        controle = _controle_sur(contenu, "P4")

        assert etat_de_publication(contenu, controle, _tout_relu(contenu, controle), [True, True]) == (
            "phrases_contredites"
        )

    def test_un_titre_contredit_bloque_aussi(self):
        contenu = _contenu()
        controle = _controle_sur(contenu, "P1")

        assert controle["phrases"][0]["partie"] == "titre"
        assert etat_de_publication(contenu, controle, _tout_relu(contenu, controle), [True, True]) == (
            "phrases_contredites"
        )

    def test_relecture_incomplete(self):
        contenu = _contenu()
        controle = _controle_sur(contenu)
        relecture = _tout_relu(contenu, controle)
        relecture["parties"]["titre"] = None

        assert etat_de_publication(contenu, controle, relecture, [True, True]) == "relecture_incomplete"
        assert etat_de_publication(contenu, controle, {}, [True, True]) == "relecture_incomplete"

    def test_l_ordre_des_blocages(self):
        """Le premier blocage, dans l'ordre où l'administrateur peut le lever."""
        contenu = _contenu()
        contredit = _controle_sur(contenu, "P2")

        assert etat_de_publication(replace(contenu, sources=[]), contredit, {}, [False, True]) == (
            "sources_absentes"
        )
        assert etat_de_publication(contenu, contredit, {}, [False, True]) == "citations_non_verifiees"
        assert etat_de_publication(replace(contenu, summary="x"), contredit, {}, [True, True]) == (
            "controle_a_relancer"
        )
        assert etat_de_publication(contenu, contredit, {}, [True, True]) == "phrases_contredites"

    @pytest.mark.parametrize("etat, attendu", [
        ("sources_absentes", "Reconstruisez-les, citez des articles parmi elles"),
        ("citations_non_verifiees", "la citation 2 n'est pas vérifiée"),
        ("controle_a_relancer", "parties modifiées : résumé"),
        ("phrases_contredites", "une phrase que les extraits contredisent (résumé : « La loi punit"),
        ("relecture_incomplete", "puis à cocher : explications des citations."),
    ])
    def test_le_motif_dit_quoi_faire(self, etat, attendu):
        contenu = _contenu()
        controle = _controle_sur(contenu, "P4")
        relecture = _tout_relu(contenu, controle)
        relecture["parties"]["explications"] = None
        if etat == "controle_a_relancer":
            contenu = replace(contenu, summary="Un autre résumé.")

        motif = motif_du_refus(etat, contenu, controle, relecture, [True, False])

        assert motif.startswith("Publication refusée : ")
        assert attendu in motif

    def test_le_motif_sans_citation_et_sans_controle(self):
        contenu = _contenu(citations=[])

        assert "ne cite aucun article" in motif_du_refus("citations_non_verifiees", contenu, None, {}, [])
        assert "n'a pas été fait" in motif_du_refus("controle_a_relancer", _contenu(), None, {}, [True, True])
        assert "les citations 1 et 2 ne sont pas vérifiées" in motif_du_refus(
            "citations_non_verifiees", _contenu(), None, {}, [False, False]
        )
        assert "parties modifiées : titre" in motif_du_refus(
            "controle_a_relancer", _contenu(title="Autre"), _controle_sur(_contenu()), {}, [True, True]
        )


# ==================== 2. EMPREINTES ET RELECTURE ====================


class TestEmpreintes:

    @pytest.mark.parametrize("changement, partie", [
        ({"title": "Un autre titre"}, "titre"),
        ({"context": "Autre contexte."}, "contexte"),
        ({"summary": "Autre résumé."}, "resume"),
        ({"key_points": ["Un seul point."]}, "points_cles"),
        ({"citizen_impact": "Autre impact."}, "impact"),
        ({"suggested_question": "Autre question ?"}, "question"),
        ({"citations": [_citation(1, title="Explication réécrite."), _citation(2)]}, "explications"),
        # La même explication, sous un autre article : l'explication change de sens.
        ({"citations": [_citation(1, article_number="2 bis"), _citation(2)]}, "explications"),
        ({"citations": [_citation(1, suite=True), _citation(2)]}, "explications"),
        ({"citations": [_citation(1)]}, "explications"),
        # Revue du tour 3 : l'article a changé en base, la carte montre un autre extrait.
        ({"citations": [_citation(1, excerpt="Un autre extrait de carte."), _citation(2)]}, "explications"),
    ])
    def test_chaque_partie_a_la_sienne(self, changement, partie):
        avant = _contenu()
        apres = replace(avant, **changement)

        textes_avant, textes_apres = empreintes_des_parties(avant), empreintes_des_parties(apres)

        assert empreinte_du_controle(avant) != empreinte_du_controle(apres)
        assert [p for p in PARTIES if textes_avant[p] != textes_apres[p]] == [partie]

    def test_les_blancs_ne_comptent_pas(self):
        """Revue du tour 3 : trois sauts de ligne ramenés à deux « modifiaient » la partie, en boucle."""
        avant = _contenu()

        apres = replace(
            avant, title=f"  {avant.title} ", summary=avant.summary.replace(" ", "  ") + "\n\n\n",
            key_points=[f" {p}\n" for p in avant.key_points],
            citations=[_citation(1, excerpt="x \n "), _citation(2)],
        )

        assert empreinte_du_controle(avant) == empreinte_du_controle(apres)
        assert empreintes_des_parties(avant) == empreintes_des_parties(apres)

    def test_les_sources_perime_le_controle_pas_les_parties(self):
        avant = _contenu()
        apres = replace(avant, sources=[{**_SOURCES[0], "extrait": "Un autre texte."}, _SOURCES[1]])
        rangees = replace(avant, sources=[
            _SOURCES[1], {**_SOURCES[0], "extrait": f"  {_SOURCES[0]['extrait']}\n"},
        ])

        assert empreinte_du_controle(avant) != empreinte_du_controle(apres)
        assert empreintes_des_parties(avant) == empreintes_des_parties(apres)
        # Les mêmes sources, rangées autrement, aux blancs près : rien ne change.
        assert empreinte_du_controle(rangees) == empreinte_du_controle(avant)

    def test_une_partie_vide_n_a_pas_d_empreinte(self):
        empreintes = empreintes_des_parties(_contenu(title="", suggested_question="", key_points=[]))

        assert empreintes["titre"] is None and empreintes["question"] is None
        assert empreintes["points_cles"] is None

    def test_relue_seulement_sur_le_texte_d_aujourd_hui(self):
        contenu = _contenu()
        controle = _controle_sur(contenu)
        relecture = _tout_relu(contenu, controle)

        assert all(parties_relues(contenu, controle, relecture).values())
        assert parties_relues(replace(contenu, citizen_impact="Autre."), controle, relecture) == {
            **_tout(True), "impact": False,
        }
        # Rien d'enregistré : seules les parties vides sont « relues ».
        assert parties_relues(_contenu(suggested_question=""), None, {}) == {
            **_tout(False), "question": True,
        }

    def test_une_case_atteste_les_verdicts_de_sa_partie(self):
        """Un nouveau contrôle qui juge autrement une partie la fait relire, elle seule."""
        contenu = _contenu()
        controle = _controle_sur(contenu)
        relecture = _tout_relu(contenu, controle)

        meme = _controle_sur(contenu)
        autre = _controle_sur(contenu, "P6")

        assert all(parties_relues(contenu, meme, relecture).values())
        assert parties_relues(contenu, autre, relecture) == {**_tout(True), "points_cles": False}

    def test_cochee_avant_le_controle_ne_compte_plus_apres(self):
        """Revue du tour 3 : cocher, PUIS contrôler, et publier sans avoir vu les verdicts."""
        contenu = _contenu()
        sans_controle = _tout_relu(contenu, None)

        assert all(parties_relues(contenu, None, sans_controle).values())
        assert not any(parties_relues(contenu, _controle_sur(contenu), sans_controle).values())

    def test_de_nouvelles_sources_font_tout_relire(self):
        contenu = _contenu()
        controle = _controle_sur(contenu)
        relecture = _tout_relu(contenu, controle)

        reconstruit = replace(contenu, sources=[_SOURCES[1]])

        assert not any(parties_relues(reconstruit, controle, relecture).values())

    def test_la_forme_du_contrat(self):
        """Réponses d'administration seulement depuis l'avenant du tour 7 : pas de forme publique."""
        contenu = _contenu()
        controle = _controle_sur(contenu)
        relecture = _tout_relu(contenu, controle)

        admin = relecture_affichee(contenu, controle, relecture)
        nommee = relecture_affichee(contenu, controle, {**relecture, "controle": controle["fait_le"]})

        assert admin == {
            "parties": _tout(True), "par": "admin", "le": "2026-10-30T10:00:00+00:00", "controle": None,
            "phrases_confirmees": [],
        }
        assert list(admin["parties"]) == list(PARTIES)
        # Avenant du tour 5 : le contrôle contre lequel les cases ont été cochées.
        assert nommee["controle"] == "2026-10-30T10:00:00+00:00"
        assert relecture_affichee(contenu, None, None) == {
            "parties": _tout(False), "par": None, "le": None, "controle": None, "phrases_confirmees": [],
        }


class TestLesSources:

    def test_sources_valables(self):
        brutes = [
            {"n": 2, "law_id": "12", "article_id": 102, "article_number": 2, "extrait": " B ",
             "law_title": "Code Y", "suite": "oui"},
            {"n": 1, "law_id": 11, "article_id": 101, "article_number": "1", "extrait": "A",
             "law_title": "Code X", "suite": True},
            {"n": 2, "extrait": "doublon"}, {"n": 0, "extrait": "zéro"}, {"n": "x", "extrait": "?"},
            {"n": 3, "extrait": ""}, "pas un objet",
        ]

        assert sources_valables(brutes) == [
            {"n": 1, "law_id": 11, "law_title": "Code X", "article_number": "1", "article_id": 101,
             "suite": True, "extrait": "A"},
            # « suite » n'est vraie que pour le booléen JSON.
            {"n": 2, "law_id": 12, "law_title": "Code Y", "article_number": "2", "article_id": 102,
             "suite": False, "extrait": "B"},
        ]
        assert sources_valables(None) == [] and sources_valables("x") == []

    def test_les_sources_affichees_et_les_extraits_ajoutes_par_le_controle(self):
        ajoute = {"n": 3, "law_id": 13, "law_title": "Code Z", "article_number": "322",
                  "article_id": 103, "suite": False, "extrait": "Le texte ajouté."}
        controle = {"phrases": [], "extraits_ajoutes": [ajoute, {**ajoute, "n": 2}]}

        affichees = sources_affichees(_SOURCES, controle)

        # Les numéros des verdicts ; un numéro déjà pris par une source n'est pas doublé.
        assert [s["n"] for s in affichees] == [1, 2, 3]
        assert affichees[1]["law_title"] == "Code Y" and affichees[2] == ajoute
        assert sources_affichees(_SOURCES, None) == sources_valables(_SOURCES)
        assert sources_affichees([], None) == []

    def test_memes_sources(self):
        assert memes_sources(_SOURCES, list(reversed(_SOURCES)))
        assert not memes_sources(_SOURCES, _SOURCES[:1])
        assert not memes_sources(_SOURCES, [{**_SOURCES[0], "suite": True}, _SOURCES[1]])


class TestNouvelleRelecture:

    def test_les_cases_de_l_administrateur(self):
        contenu = _contenu()
        controle = _controle_sur(contenu)

        relecture = nouvelle_relecture(
            contenu, contenu, controle, {}, {"titre": True, "impact": True}, "Awa", 7,
            "2026-10-30T11:00:00+00:00",
        )

        assert relecture["par"] == "Awa" and relecture["par_id"] == 7
        assert relecture["le"] == "2026-10-30T11:00:00+00:00"
        assert [p for p, relue in parties_relues(contenu, controle, relecture).items() if relue] == [
            "titre", "impact",
        ]

    def test_une_partie_changee_est_decochee_meme_cochee_dans_la_requete(self):
        avant = _contenu()
        controle = _controle_sur(avant)
        apres = replace(avant, summary="Un nouveau résumé.")

        relecture = nouvelle_relecture(
            avant, apres, controle, _tout_relu(avant, controle), _tout(True), "Awa", 7,
            "2026-10-30T11:00:00+00:00",
        )

        assert parties_relues(apres, controle, relecture) == {**_tout(True), "resume": False}
        # Revenir à l'ancien texte ne recoche rien : la case a été retirée.
        assert parties_relues(avant, controle, relecture)["resume"] is False

    def test_le_titre_se_relit_comme_les_autres(self):
        avant = _contenu()
        controle = _controle_sur(avant)
        relu = _tout_relu(avant, controle)

        retitre = nouvelle_relecture(
            avant, replace(avant, title="Un titre plus juste"), controle, relu, None, "Awa", 7, "t",
        )

        assert retitre["parties"]["titre"] is None
        assert sum(v is not None for v in retitre["parties"].values()) == len(PARTIES) - 1

    def test_par_et_le_ne_changent_que_par_l_administrateur(self):
        contenu = _contenu()
        controle = _controle_sur(contenu)
        enregistree = _tout_relu(contenu, controle)

        meme = nouvelle_relecture(contenu, contenu, controle, enregistree, {"resume": True}, "Autre", 9, "plus tard")
        texte = nouvelle_relecture(
            contenu, replace(contenu, summary="x"), controle, enregistree, None, "Autre", 9, "plus tard"
        )
        decoche = nouvelle_relecture(contenu, contenu, controle, enregistree, {"impact": False}, "Autre", 9, "plus tard")

        assert (meme["par"], meme["le"]) == ("admin", "2026-10-30T10:00:00+00:00")
        assert (texte["par"], texte["le"]) == ("admin", "2026-10-30T10:00:00+00:00")
        assert (decoche["par"], decoche["le"]) == ("Autre", "plus tard")

    def test_recocher_apres_un_nouveau_controle(self):
        """La case tombée avec un nouveau verdict se recoche contre ce verdict."""
        contenu = _contenu()
        premier, second = _controle_sur(contenu), _controle_sur(contenu, "P6")
        relue_avant = _tout_relu(contenu, premier)

        recochee = nouvelle_relecture(
            contenu, contenu, second, relue_avant, {"points_cles": True}, "Awa", 7, "t",
        )

        assert all(parties_relues(contenu, second, recochee).values())

    def test_rien_a_noter(self):
        contenu = _contenu()

        assert nouvelle_relecture(contenu, contenu, None, {}, None, "Awa", 7, "t") == {}
        assert nouvelle_relecture(contenu, replace(contenu, summary="x"), None, {}, None, "Awa", 7, "t") == {}

    def test_la_relecture_garde_le_controle_relu(self):
        """Avenant du tour 5 : des cases cochées gardent le contrôle contre lequel elles l'ont été."""
        contenu = _contenu()
        controle = _controle_sur(contenu)
        fait_le = controle["fait_le"]

        cochee = nouvelle_relecture(contenu, contenu, controle, {}, {"titre": True}, "Awa", 7, "t", fait_le)
        decochee = nouvelle_relecture(contenu, contenu, controle, cochee, {"titre": False}, "Awa", 7, "t2")
        modifiee = nouvelle_relecture(
            contenu, replace(contenu, title="Autre"), controle, {}, {"titre": True}, "Awa", 7, "t", fait_le,
        )
        ancienne = nouvelle_relecture(contenu, contenu, controle, _tout_relu(contenu, controle), None, "A", 1, "t")

        assert cochee["controle"] == fait_le
        # Décocher n'atteste rien : le contrôle nommé reste celui des cases d'avant.
        assert decochee["controle"] == fait_le
        # Une case cochée sur une partie qu'on modifie tombe : rien n'est attesté.
        assert "controle" not in modifiee
        # Une relecture d'avant le tour 5 garde sa forme tant que rien n'est coché.
        assert "controle" not in ancienne


class TestLaRelecturePerimee:
    """Avenant du tour 5 (revue du tour 4) : une case cochée nomme le contrôle relu, celui enregistré."""

    def test_le_controle_enregistre_est_accepte(self):
        controle = _controle_sur(_contenu())

        assert relecture_perimee({"titre": True}, controle["fait_le"], controle) is None
        assert relecture_perimee({"titre": True}, f" {controle['fait_le']} ", controle) is None

    @pytest.mark.parametrize("relu, attendu", [
        (None, "ne disent pas contre quel contrôle IA"),
        ("", "ne disent pas contre quel contrôle IA"),
        ("2026-10-30T09:59:59+00:00", "Relecture périmée : les cases cochées portent sur un autre contrôle IA"),
    ], ids=["absent", "vide", "un-autre"])
    def test_absent_ou_un_autre_est_refuse(self, relu, attendu):
        controle = _controle_sur(_contenu())

        refus = relecture_perimee({"resume": True, "titre": False}, relu, controle)

        assert attendu in refus
        assert "Rien n'a été enregistré" in refus

    @pytest.mark.parametrize("controle", [None, {"fait_le": "2026-10-30T10:00:00+00:00"}, {"phrases": []}])
    def test_sans_controle_enregistre_rien_a_attester(self, controle):
        refus = relecture_perimee({"titre": True}, "2026-10-30T10:00:00+00:00", controle)

        assert refus.startswith("Relecture refusée : aucun contrôle IA n'est enregistré")

    def test_decocher_n_atteste_rien(self):
        assert relecture_perimee({"titre": False, "resume": False}, None, None) is None
        assert relecture_perimee({}, None, _controle_sur(_contenu())) is None


class TestLeFaitLeDUnControle:
    """Avenant du tour 5 : chaque contrôle a son `fait_le`, qui le désigne."""

    def test_apres_le_precedent(self):
        assert fait_le_apres("2026-11-04T15:00:57+00:00", "2026-11-04T15:00:00+00:00") == (
            "2026-11-04T15:00:57+00:00"
        )
        # Même seconde (deux onglets), ou horloge en retard : une seconde après le plus récent.
        assert fait_le_apres("2026-11-04T15:00:57+00:00", "2026-11-04T15:00:57+00:00") == (
            "2026-11-04T15:00:58+00:00"
        )
        assert fait_le_apres(
            "2026-11-04T15:00:57+00:00", "2026-11-04T15:00:59+00:00", "2026-11-04T15:01:10+00:00",
        ) == "2026-11-04T15:01:11+00:00"

    def test_sans_precedent_lisible(self):
        assert fait_le_apres("2026-11-04T15:00:57+00:00") == "2026-11-04T15:00:57+00:00"
        assert fait_le_apres("2026-11-04T15:00:57+00:00", None, "", "pas une date") == (
            "2026-11-04T15:00:57+00:00"
        )


# ==================== 3. LES PHRASES ====================


class TestLesPhrases:

    @pytest.mark.parametrize("texte, phrases", [
        ("Une phrase. Une autre ! Et une question ? Fin.",
         ["Une phrase.", "Une autre !", "Et une question ?", "Fin."]),
        ("Selon l'art. 39 al. 3, la preuve incombe à l'employeur. M. Ndong le sait.",
         ["Selon l'art. 39 al. 3, la preuve incombe à l'employeur.", "M. Ndong le sait."]),
        ("Le délai est de 3 mois. 12 mois au plus.", ["Le délai est de 3 mois.", "12 mois au plus."]),
        ("Il dit « Sortez. » Puis il part.", ["Il dit « Sortez. »", "Puis il part."]),
        ("Première ligne sans point\nSeconde ligne.", ["Première ligne sans point", "Seconde ligne."]),
        ("La loi n° 2016/007. Le Code pénal.", ["La loi n° 2016/007.", "Le Code pénal."]),
        ("Expulsion locative sans jugement : quels cas au Cameroun ?",
         ["Expulsion locative sans jugement : quels cas au Cameroun ?"]),
        ("", []), ("...", []),
    ])
    def test_decouper_en_phrases(self, texte, phrases):
        assert decouper_en_phrases(texte) == phrases

    def test_les_phrases_de_l_article_dans_l_ordre_le_titre_d_abord(self):
        phrases = phrases_du_billet(_contenu())

        assert [p.id for p in phrases] == [f"P{i}" for i in range(1, len(phrases) + 1)]
        assert [(p.partie, p.index) for p in phrases] == [
            ("titre", 0), ("contexte", 0), ("contexte", 0), ("resume", 0), ("resume", 0),
            ("point_cle", 0), ("point_cle", 1), ("impact", 0),
            ("explication", 0), ("explication", 1), ("question", 0),
        ]
        assert phrases[0] == PhraseAJuger("P1", "titre", 0, "Expulsion sans juge : ce que dit la loi")
        assert phrases[6].texte == "Le congé suit l'usage."


# ==================== 4. LA RÉPONSE DU MODÈLE ====================

_ART_39_3 = (
    "Article 39.- (3) Dans tous cas de licenciement, il appartient à l'employeur d'apporter la "
    "preuve du caractère légitime du motif qu'il allègue."
)
_ART_39_7 = (
    "(7) Ces dommages-intérêts ne se confondent ni avec l'indemnité pour non-observation du "
    "préavis, ni avec l'indemnité de licenciement."
)
_EXTRAITS = [
    ExtraitDuControle(5, "Code du travail", "39", False, _ART_39_3),
    ExtraitDuControle(9, "Code du travail", "39", True, _ART_39_7),
]
_PREUVE = "il appartient à l'employeur d'apporter la preuve du caractère légitime du motif"
# La forme de la mesure réelle du 04/11 : deux extraits, chacun entre guillemets, suivi de sa source.
_DEUX_EXTRAITS = (
    '"il appartient à l\'employeur d\'apporter la preuve" (Extrait 5) et "Ces dommages-intérêts '
    'ne se confondent ni avec l\'indemnité pour non-observation du préavis" (Extrait 9)'
)
_PHRASES = [
    PhraseAJuger("P1", "resume", 0, "Le salarié doit prouver l'abus."),
    PhraseAJuger("P2", "point_cle", 0, "Les dommages-intérêts ne se confondent pas avec le préavis."),
    PhraseAJuger("P3", "impact", 0, "Consultez un avocat."),
]


def _reponse(*phrases, **autres) -> dict:
    return {"response": json.dumps({"phrases": list(phrases)}), "fin": "STOP", **autres}


class TestLaReponseDuModele:

    @pytest.mark.parametrize("brut, verdict", [
        ("soutenue", "soutenue"), ("Non soutenue", "non_soutenue"), ("non-soutenue", "non_soutenue"),
        ("NON_SOUTENUE", "non_soutenue"), ("contredite", "contredite"), ("Contredit", "contredite"),
        ("hors sources", "hors_sources"), ("hors-source", "hors_sources"),
        ("peut-être", None), ("", None), (None, None), (3, None),
        # Mesure réelle du 04/11 : « partiellement_soutenue », jamais un accord.
        ("partiellement_soutenue", "non_soutenue"), ("Partiellement soutenue", "non_soutenue"),
        ("en partie soutenue", "non_soutenue"), ("partially supported", "non_soutenue"),
        ("partiellement contredite", "contredite"), ("partiellement", None),
    ])
    def test_lire_le_verdict(self, brut, verdict):
        assert lire_le_verdict(brut) == verdict

    def test_partiellement_soutenue_se_relit(self):
        (jugee, *_) = lire_les_verdicts(_reponse(
            jugement("P1", "partiellement_soutenue", _PREUVE, [5], raison="Une partie manque."),
        ), _PHRASES, _EXTRAITS)

        assert (jugee["verdict"], jugee["raison"]) == ("non_soutenue", "Une partie manque.")

    @pytest.mark.parametrize("brut, nature", [
        ("autre", "autre"), ("Autre", "autre"), ("conseil", "autre"), ("question", "autre"),
        ("regle", "regle"), ("règle", "regle"), ("", "regle"), (None, "regle"), (3, "regle"),
    ])
    def test_lire_la_nature_dans_le_doute_une_regle(self, brut, nature):
        assert lire_la_nature(brut) == nature

    def test_chaque_phrase_garde_son_texte_son_rang_et_son_passage(self):
        jugees = lire_les_verdicts(_reponse(
            jugement("P1", "contredite", _PREUVE, [5, 9, 5],
                     raison="L'article 39 met la preuve à la charge de l'employeur."),
            {**jugement("p2", "soutenue", "Ces dommages-intérêts ne se confondent ni avec "
                        "l'indemnité pour non-observation du préavis", [5],
                        raison="**L'article 39** le dit."), "verdict": "Soutenue"},
            jugement("P3", "hors sources", raison="Aucun extrait n'en parle."),
        ), _PHRASES, _EXTRAITS)

        premiere = jugees[0]
        assert {cle: premiere[cle] for cle in ("id", "partie", "index", "texte", "verdict", "extraits")} == {
            "id": "P1", "partie": "resume", "index": 0, "texte": "Le salarié doit prouver l'abus.",
            "verdict": "contredite", "extraits": [5, 9],
        }
        # Le passage trouvé est cité à la suite de la raison, pour la relecture.
        assert premiere["raison"] == (
            f"L'article 39 met la preuve à la charge de l'employeur. — Extrait 5 : « {_PREUVE} »"
        )
        assert premiere["nature"] == "regle" and premiere["passage"] == _PREUVE
        # Le passage est dans l'extrait 9, pas dans le 5 que citait le modèle.
        assert (jugees[1]["verdict"], jugees[1]["extraits"]) == ("soutenue", [9, 5])
        assert jugees[1]["raison"].startswith("L'article 39 le dit. — Extrait 9 : « Ces dommages")
        assert (jugees[2]["verdict"], jugees[2]["extraits"]) == ("hors_sources", [])

    def test_un_appui_invente_ne_soutient_pas_une_contradiction_sans_texte_ne_bloque_pas(self):
        """Tour 3, essai réel : « l'employeur doit prouver…, tandis que le salarié peut prouver le caractère abusif »."""
        jugees = lire_les_verdicts(_reponse(
            jugement("P1", "soutenue", "le salarié peut prouver le caractère abusif du licenciement",
                     [5], raison="L'article 39 dit que le salarié peut prouver l'abus."),
            jugement("P2", "contredite", "", [9], raison="Le préavis est dû."),
            jugement("P3", "soutenue", nature="autre", raison="Un conseil pratique."),
        ), _PHRASES, _EXTRAITS)

        assert [p["verdict"] for p in jugees] == ["non_soutenue", "non_soutenue", "soutenue"]
        assert jugees[0]["raison"].startswith("À vérifier : le contrôle la disait soutenue")
        assert "« L'article 39 dit que le salarié peut prouver l'abus. »" in jugees[0]["raison"]
        assert jugees[1]["raison"].startswith("À vérifier : le contrôle la disait contredite")
        assert jugees[2]["raison"] == "Un conseil pratique."

    def test_un_mot_cle_change_au_milieu_du_passage_ne_compte_pas(self):
        (jugee, *_) = lire_les_verdicts(_reponse(jugement(
            "P1", "soutenue",
            "il appartient au salarié d'apporter la preuve du caractère légitime du motif qu'il allègue",
            [5],
        )), _PHRASES, _EXTRAITS)

        assert jugee["verdict"] == "non_soutenue"

    @pytest.mark.parametrize("passage, present", [
        (_PREUVE, True),
        ("IL APPARTIENT A L'EMPLOYEUR, d’apporter la preuve", True),
        ("Dans tous cas de licenciement […] apporter la preuve du caractère légitime", True),
        ("Dans tous cas de licenciement... apporter la preuve du caractère légitime", True),
        # Un mot de bord ajouté par le modèle : 16 mots sur 17 d'un seul tenant. Le tour 4
        # l'acceptait ; c'est le trou de la revue (un dernier mot change, et il décide).
        ("il appartient à l'employeur d'apporter la preuve du caractère légitime du motif qu'il "
         "allègue toujours", False),
        ("la preuve", False),
        ("", False),
        ("il appartient au salarié d'apporter la preuve du caractère légitime du motif qu'il allègue",
         False),
        ("Dans tous cas de licenciement […] le salarié prouve l'abus du motif", False),
    ])
    def test_passage_present(self, passage, present):
        assert passage_present(passage, _ART_39_3) is present

    def test_extrait_du_passage(self):
        textes = {e.n: e.texte for e in _EXTRAITS}

        assert extrait_du_passage(_PREUVE, [9], textes) == 5
        assert extrait_du_passage(_PREUVE, [5], textes) == 5
        assert extrait_du_passage("une phrase qui n'est nulle part dans les textes", [5, 9], textes) is None
        assert passage_present(_PREUVE, "") is False

    @pytest.mark.parametrize("passage, present", [
        # Mesure réelle du 04/11 : des morceaux entre guillemets, suivis de leur source.
        (f'"{_PREUVE}" (Extrait 5)', True),
        (f'"{_PREUVE}" [5]', True),
        ('"Dans tous cas de licenciement..." (Art. 39(3)) ; "il appartient à l\'employeur '
         'd\'apporter la preuve" (Art. 39(3))', True),
        (f"“{_PREUVE}”", True),
        (f"« {_PREUVE} » [5]", True),
        # Un morceau vrai, un morceau inventé : le passage tombe.
        (f'"{_PREUVE}" (Extrait 5) et "le salarié doit prouver le caractère abusif" (Extrait 5)', False),
        # Un guillemet jamais fermé : son morceau est vérifié aussi.
        (f'"{_PREUVE}" et "le salarié doit prouver le caractère abusif', False),
        # Des renvois seuls, ou des morceaux trop courts pour rien prouver.
        ('"la preuve" (Extrait 5) "du motif" [5]', False),
        ("(Extrait 5) [5]", False),
    ])
    def test_les_morceaux_entre_guillemets(self, passage, present):
        assert passage_present(passage, _ART_39_3) is present

    def test_des_morceaux_de_deux_extraits(self):
        textes = {e.n: e.texte for e in _EXTRAITS}

        assert appuis_du_passage(_DEUX_EXTRAITS, [9, 5], textes) == [
            (5, "il appartient à l'employeur d'apporter la preuve"),
            (9, "Ces dommages-intérêts ne se confondent ni avec l'indemnité pour non-observation du préavis"),
        ]
        assert extrait_du_passage(_DEUX_EXTRAITS, [9], textes) == 5
        # Un morceau ne passe pas d'un extrait à l'autre, même par « … ».
        a_cheval = '"il appartient à l\'employeur d\'apporter la preuve… Ces dommages-intérêts ne se confondent ni"'
        assert appuis_du_passage(a_cheval, [5, 9], textes) is None
        # Chacun des morceaux doit être dans un extrait : un seul n'en porte pas deux.
        assert passage_present(_DEUX_EXTRAITS, _ART_39_3) is False

    def test_un_passage_de_deux_extraits_soutient_et_se_cite(self):
        """Mesure réelle du 04/11 : lus d'un bloc, 17 des 18 phrases justes passaient « À vérifier »."""
        (jugee, *_) = lire_les_verdicts(_reponse(
            jugement("P1", "soutenue", _DEUX_EXTRAITS, [9], raison="Les extraits 5 et 9 le disent."),
        ), _PHRASES, _EXTRAITS)

        assert (jugee["verdict"], jugee["extraits"]) == ("soutenue", [5, 9])
        assert jugee["raison"] == (
            "Les extraits 5 et 9 le disent. — Extrait 5 : « il appartient à l'employeur d'apporter la "
            "preuve » ; extrait 9 : « Ces dommages-intérêts ne se confondent ni avec l'indemnité pour "
            "non-observation du préavis »"
        )

    def test_un_appui_invente_est_nomme_et_jamais_cite_comme_l_extrait(self):
        """Tour 5 : la raison dit quel morceau manque, sans le donner pour le texte d'un extrait."""
        (jugee, *_) = lire_les_verdicts(_reponse(jugement(
            "P1", "soutenue", '"il appartient au salarié d\'apporter la preuve" [5]', [5],
            raison="L'extrait 5 le dit.",
        )), _PHRASES, _EXTRAITS)

        assert jugee["verdict"] == "non_soutenue"
        assert jugee["raison"] == (
            "À vérifier : le contrôle la disait soutenue, sans pouvoir copier un passage des extraits "
            "qui le dise (introuvable tel quel dans les extraits : « il appartient au salarié "
            "d'apporter la preuve »). Sa raison : « L'extrait 5 le dit. »"
        )
        assert "Extrait 5 :" not in jugee["raison"]

    def test_jamais_un_accord_par_defaut(self):
        """Phrase oubliée, verdict illisible : « non_soutenue », à relire."""
        jugees = lire_les_verdicts(_reponse(
            jugement("P1", "soutenue", _PREUVE, [5]),
            jugement("P2", "plutôt oui", _PREUVE, [5]),
            jugement("P7", "soutenue", _PREUVE, [5]),
        ), _PHRASES, _EXTRAITS)

        assert [(p["id"], p["verdict"]) for p in jugees] == [
            ("P1", "soutenue"), ("P2", "non_soutenue"), ("P3", "non_soutenue"),
        ]
        assert jugees[1]["raison"] == jugees[2]["raison"] == RAISON_NON_JUGEE
        assert jugees[2]["extraits"] == []

    def test_le_premier_verdict_d_une_phrase_compte(self):
        jugees = lire_les_verdicts(_reponse(
            jugement("P1", "contredite", _PREUVE, [5]),
            jugement("P1", "soutenue", _PREUVE, [5]),
        ), _PHRASES, _EXTRAITS)

        assert jugees[0]["verdict"] == "contredite"

    def test_des_numeros_d_extraits_inconnus_ou_mal_ecrits(self):
        extraits = [ExtraitDuControle(n, "Code", str(n), False, f"Texte {n}.") for n in (1, 2, 3)]

        (jugee, *_) = lire_les_verdicts(_reponse(
            {**jugement("P1", "non_soutenue"), "extraits": [0, 2, "3", 99, "x", True]},
        ), _PHRASES, extraits)

        assert jugee["extraits"] == [2, 3]

    def test_les_balises_de_code_sont_tolerees(self):
        brut = "```json\n" + json.dumps({"phrases": [jugement("P1", "soutenue", _PREUVE, [5])]}) + "\n```"

        assert lire_les_verdicts({"response": brut}, _PHRASES, _EXTRAITS)[0]["verdict"] == "soutenue"

    @pytest.mark.parametrize("reponse", [
        {"response": json.dumps({"phrases": [{"id": "P1", "verdict": "soutenue"}]}), "tronquee": True},
        {"response": "Toutes les phrases sont fidèles."},
        {"response": json.dumps(["P1", "soutenue"])},
        {"response": json.dumps({"verdicts": []})},
        {"response": json.dumps({"phrases": []})},
        {"response": json.dumps({"phrases": [{"id": "P9", "verdict": "soutenue"}]})},
        {"response": json.dumps({"phrases": [{"id": "P1", "verdict": "bof"}]})},
        {"response": ""},
        {},
    ], ids=[
        "coupee", "prose", "liste", "autre-cle", "vide", "aucun-id-connu", "aucun-verdict-lisible",
        "chaine-vide", "rien",
    ])
    def test_une_reponse_inutilisable_est_une_erreur(self, reponse):
        with pytest.raises(ReponseInvalideError) as erreur:
            lire_les_verdicts(reponse, _PHRASES, _EXTRAITS)

        assert (erreur.value.statut, erreur.value.code) == (502, "reponse_invalide")
        assert "l'article est inchangé" in str(erreur.value)

    def test_controle_affiche_suit_le_contrat(self):
        contenu = _contenu()
        controle = _controle_sur(contenu)
        controle["phrases"][0].update(verdict="inventé", affirmation="a", nature="regle", passage="p")
        controle["phrases"].append("pas un objet")

        affiche = controle_affiche(controle)

        assert set(affiche) == {"fait_le", "modele", "empreinte", "phrases"}
        assert affiche["phrases"][0]["verdict"] == "non_soutenue"
        assert set(affiche["phrases"][0]) == {
            "id", "partie", "index", "texte", "verdict", "extraits", "raison", "nature",
        }
        assert affiche["phrases"][0]["nature"] == "regle"
        assert len(affiche["phrases"]) == len(phrases_du_billet(contenu))
        assert controle_affiche(None) is None and controle_affiche({"phrases": "x"}) is None


# ==================== 4 bis. LE PASSAGE COPIÉ (TOUR 5, D1) ====================

# Des extraits réels de jurix_dev (lus en lecture seule), OCR compris : Code du
# travail art. 36, 37, 39 (4) et 40 (8) ; AUDCG art. 127 et 133 (data/blog/tour4).
_ART_36 = (
    "Article 36.- (1) Toute rupture de contrat à durée indéterminée, sans préavis ou sans que le "
    "délai de préavis ait été intégralement observé, emporte obligation pour la partie responsable "
    "de verser à lu se e e o n u ne out,l nature dont aurait bénéficié le travailleur durant le "
    "délai de préavis qui n'aura pas été effectivement respecté."
)
_ART_37 = (
    "Article 37.- (1) En cas de rupture de contrat à durée indéterminée du fait de l'employeur, "
    "hormis le cas de faute lourde, le travailleur ayant accompli dans l'entreprise une durée de "
    "service continue au moins égale à deux (2) ans, a droit à une indemnité de licenciement "
    "distincte de celle du préavis dont la détermination tient compte de l'ancienneté."
)
_ART_39_4 = (
    "Toutefois, le montant des dommages-intérêts, sans excéder un (1) mois de salaire par année "
    "d'ancienneté dans l'entreprise, ne peut être inférieur à trois (3) mois de salaire."
)
_ART_40_8 = (
    "(8) En cas de contestation sur le motif ou l'ordre des licenciements, la charge de la preuve "
    "incombe à l'employeur. De e i d, se , e e dt b ee (2) ans dans la même entreprise."
)
_ART_127 = (
    "Le bailleur peut s'opposer au droit au renouvellement du bail à durée déterminée ou "
    "indéterminée, sans avoir à régler d'indemnité d'éviction, dans les cas suivants :\n\n"
    "- 1°) s'il justifie d'un motif grave et légitime à l'encontre du preneur sortant.\n\nCe motif "
    "ne peut être invoqué que si les faits se sont poursuivis ou rnon m s n e s e ss d s oe "
    "notification par tout moyen permettant d'établir la réception effective par le destinataire, "
    "d'avoir à les faire cesser."
)
_ART_133 = (
    "La demande en justice aux fins de résiliation du bail doit être précédée d'une mise en demeure "
    "d'avoir à respecter la ou les clauses ou conditions violées.\n\nLe contrat de bail peut prévoir "
    "une clause résolutoire de plein droit. La juridiction compétente statuant à bref délai constate "
    "la résiliation du bail et prononce, le cas échéant, l'expulsion du preneur."
)


def _cite(passage: str, texte: str):
    """Les mots de l'extrait que le passage cite, ou None."""
    appuis = appuis_du_passage(passage, [1], {1: texte})
    return [mots for _, mots in appuis] if appuis else None


class TestLePassageCopie:
    """
    Revue du tour 4 (data/blog/revue-api-tour4b) : un dernier mot changé passait à 85 % d'un
    seul tenant, un tronçon de moins de 3 mots après « … » n'était jamais cherché, et la raison
    citait les mots DU MODÈLE comme ceux de l'extrait.
    """

    @pytest.mark.parametrize("passage", [
        "En cas de contestation sur le motif ou l'ordre des licenciements, la charge de la preuve "
        "incombe au travailleur",
        "la charge de la preuve incombe au salarié",
        "la charge de la preuve incombe […] au travailleur",
        "la charge de la preuve incombe… au salarié",
        "la charge de la preuve incombe [...] à l'employé",
    ])
    def test_la_charge_de_la_preuve_inversee_ne_passe_plus(self, passage):
        assert not passage_present(passage, _ART_40_8)

    def test_la_preuve_inversee_ne_garde_pas_soutenue(self):
        """Le scénario de la revue : la phrase fausse restait soutenue, sous un faux « Extrait 4 »."""
        extraits = [ExtraitDuControle(4, "Code du travail", "40", True, _ART_40_8)]
        phrases = [PhraseAJuger("P1", "resume", 0, "C'est au travailleur de prouver que son licenciement est abusif.")]

        (jugee,) = lire_les_verdicts(_reponse(jugement(
            "P1", "soutenue", "…la charge de la preuve incombe au travailleur", [4],
            raison="L'extrait 4 met la preuve à la charge du travailleur.",
        )), phrases, extraits)

        assert jugee["verdict"] == "non_soutenue"
        assert "Extrait 4 :" not in jugee["raison"]
        assert "(introuvable tel quel dans les extraits : « la charge de la preuve incombe au travailleur »)" in (
            jugee["raison"]
        )

    @pytest.mark.parametrize("passage, present", [
        # Le premier ou le dernier mot change : jamais toléré.
        ("au salarié d'apporter la preuve du caractère légitime du motif", False),
        ("il appartient à l'employeur d'apporter la preuve du caractère légitime du motif qu'il invoque", False),
        # Un mot qui compte change au milieu.
        ("il appartient à l'employeur d'apporter la preuve du caractère sérieux du motif", False),
    ])
    def test_chaque_mot_qui_compte_premier_et_dernier_compris(self, passage, present):
        assert passage_present(passage, _ART_39_3) is present

    @pytest.mark.parametrize("passage, cite", [
        # « trois (3) mois » copié « trois mois » : une omission, pas une invention.
        ("ne peut être inférieur à trois mois de salaire",
         "ne peut être inférieur à trois (3) mois de salaire"),
        ("le travailleur ayant accompli dans l'entreprise une durée de service continue au moins "
         "égale à deux ans",
         "le travailleur ayant accompli dans l'entreprise une durée de service continue au moins "
         "égale à deux (2) ans"),
        # Mesure réelle du 04/11 (bail P8) : « droit au » sauté.
        ("Le bailleur peut s'opposer au renouvellement du bail",
         "Le bailleur peut s'opposer au droit au renouvellement du bail"),
        # Un mot outil pour un autre, ou en trop.
        ("La demande en justice aux fins de la résiliation du bail",
         "La demande en justice aux fins de résiliation du bail"),
    ])
    def test_une_omission_est_toleree_et_citee_dans_les_mots_de_l_extrait(self, passage, cite):
        texte = f"{_ART_37} {_ART_39_4} {_ART_127} {_ART_133}"

        assert _cite(f'"{passage}"', texte) == [cite]

    @pytest.mark.parametrize("passage", [
        # « ne … que » sauté : la condition devient la règle.
        "Ce motif peut être invoqué si les faits se sont poursuivis",
        "Ce motif ne peut être invoqué si les faits se sont poursuivis",
        # « sans versement… » sauté.
        "Le bailleur peut, refuser le renouvellement du bail",
        # Trop de mots sautés (5 pour un tronçon de 7).
        "Le bailleur peut s'opposer au droit au renouvellement à régler d'indemnité",
        # Un mot qui compte en plus.
        "La demande en justice aux fins de résiliation du bail commercial doit être précédée",
        # Le dernier mot, un mot outil, n'est pas là.
        "La demande en justice aux fins de résiliation du bail doit être précédée de",
        # « contre son gré » devenu « avec son gré » : « contre » décide.
        "celui qui s'introduit ou se maintient dans le domicile d'autrui avec son gré",
    ])
    def test_jamais_un_mot_qui_decide_ni_trop_de_mots(self, passage):
        texte = (
            f"{_ART_127} {_ART_133} Le bailleur peut, sans versement d'indemnité d'éviction, refuser "
            "le renouvellement du bail portant sur les locaux d'habitation accessoires. Est puni "
            "celui qui s'introduit ou se maintient dans le domicile d'autrui contre son gré."
        )

        assert _cite(f'"{passage}"', texte) is None

    def test_un_troncon_court_apres_une_coupure_est_cherche_juste_apres(self):
        # Mesure réelle du 04/11 (bail P12) : vrai, dans l'ordre, et proche. Deux phrases de
        # l'extrait : deux citations, jamais recollées en une (revue de l'API du tour 5).
        assert _cite(
            '"La demande en justice... doit être précédée d\'une mise en demeure... La juridiction '
            'compétente... prononce... l\'expulsion" (Extrait 6)', _ART_133,
        ) == [
            "La demande en justice […] doit être précédée d'une mise en demeure",
            "La juridiction compétente […] prononce […] l'expulsion",
        ]
        # Mesure réelle du 04/11 (travail P20) : « une indemnité », écrit par le modèle à la place
        # d'un passage illisible, n'était jamais cherché.
        assert _cite(
            '"Toute rupture de contrat... sans préavis... emporte obligation... de verser... une '
            'indemnité" (Art. 36(1))', _ART_36,
        ) is None
        # Avant le tronçon précédent : ce n'est pas lui.
        assert _cite('"la juridiction compétente statuant à bref délai… La demande"', _ART_133) is None

    def test_un_troncon_court_trop_loin_du_precedent(self):
        def texte(mots_entre: int, fin: str = ",") -> str:
            remplissage = " ".join(f"mot{i}" for i in range(mots_entre))
            return (
                f"La juridiction compétente statue{fin} {remplissage} elle prononce{fin} {remplissage} "
                f"alors l'expulsion du preneur a lieu."
            )

        proche, loin = texte(10), texte(PORTEE_D_UNE_COUPURE)

        assert _cite('"La juridiction compétente statue… prononce"', proche) is not None
        assert _cite('"La juridiction compétente statue… prononce"', loin) is None
        # Un tronçon court qui ouvre le morceau : le suivant doit le suivre d'aussi près.
        assert _cite('"prononce… l\'expulsion du preneur"', proche) is not None
        assert _cite('"prononce… l\'expulsion du preneur"', loin) is None

    @pytest.mark.parametrize("entre, meme_phrase", [
        (", puis elle", True),
        (". Elle", False),
        (" ; elle", False),
        ("\nelle", False),
        (" (2) Elle", False),
        # « (3) » d'un nombre écrit en lettres n'ouvre pas d'alinéa.
        (" trois (3) fois, elle", True),
        # Ni le point d'une abréviation.
        (" selon l'art. 39, elle", True),
    ], ids=["virgule", "point", "point-virgule", "ligne", "alinea", "nombre", "abreviation"])
    def test_un_troncon_court_reste_dans_la_phrase_du_precedent(self, entre, meme_phrase):
        """
        Revue de l'API du tour 5 : à 80 mots, un tronçon court passait dans une autre phrase, un
        autre alinéa (« il appartient… au travailleur », art. 39 al. 3, puis al. 4). Proche ne
        suffit plus : dans la même phrase.
        """
        texte = f"La juridiction compétente statue{entre} prononce l'expulsion du preneur."

        assert (_cite('"La juridiction compétente statue… prononce"', texte) is not None) is meme_phrase
        # Le tronçon long qui suit un tronçon court qui ouvre le morceau, de même.
        assert (_cite('"statue… prononce l\'expulsion du preneur"', texte) is not None) is meme_phrase

    def test_deux_troncons_longs_cites_dans_l_autre_ordre(self):
        """
        Mesure réelle du 04/11 (travail P14) : le plancher, puis le plafond. Cités dans l'autre
        ordre que l'extrait, ils ne se recollent pas autour d'un « […] » (revue de l'API du
        tour 5) : la citation donne les mots du trou.
        """
        assert _cite(
            '"Le montant des dommages-intérêts... ne peut être inférieur à trois (3) mois de salaire... '
            'sans excéder un (1) mois de salaire par année d\'ancienneté" (Art. 39(4))', _ART_39_4,
        ) == [
            "le montant des dommages-intérêts, sans excéder un (1) mois de salaire par année "
            "d'ancienneté dans l'entreprise, ne peut être inférieur à trois (3) mois de salaire"
        ]

    def test_les_mots_de_l_extrait_leurs_parentheses_et_leurs_guillemets(self):
        # La casse, les accents et la ponctuation de l'extrait, pas ceux du modèle.
        assert _cite("IL APPARTIENT A L'EMPLOYEUR, d’apporter la preuve", _ART_39_3) == [
            "il appartient à l'employeur d'apporter la preuve"
        ]
        # « 2 ans » dans « deux (2) ans » : la parenthèse entière.
        assert _cite(
            '"le travailleur ayant accompli... 2 ans... a droit à une indemnité de licenciement"', _ART_37,
        ) == ["le travailleur ayant accompli […] (2) ans, a droit à une indemnité de licenciement"]
        # Les « » d'un texte de loi deviennent “ ” : le front ferme une citation au » qui
        # équilibre son «.
        assert _cite(
            '"le terme « bail » désigne la convention"',
            "Au sens de la présente loi, le terme « bail » désigne la convention de location.",
        ) == ["le terme “bail” désigne la convention"]

    def test_le_morceau_introuvable_est_nomme(self):
        """Mesure réelle du 04/11 : bail P11, travail P7 (passages illisibles par l'OCR)."""
        assert lire_le_passage(
            '"Le bailleur peut s\'opposer au renouvellement... pour motif grave... après notification '
            'par tout moyen permettant d\'établir la réception effective" (Extrait 5)', [5], {5: _ART_127},
        ) == (None, "pour motif grave … après notification par tout moyen permettant d'établir la réception effective")
        assert lire_le_passage(
            '"Toute rupture de contrat... de verser... une indemnité dont aurait bénéficié le travailleur '
            'durant le délai de préavis" (Art. 36(1))', [2], {2: _ART_36},
        ) == (None, "une indemnité dont aurait bénéficié le travailleur durant le délai de préavis")
        # Chaque tronçon est quelque part, mais pas ensemble : le morceau entier.
        assert lire_le_passage(
            '"la juridiction compétente statuant à bref délai… La demande"', [6], {6: _ART_133},
        ) == (None, "la juridiction compétente statuant à bref délai… La demande")
        assert lire_le_passage("", [6], {6: _ART_133}) == (None, "")


# ==================== 4 ter. UNE COUPURE NE CACHE PAS CE QUI DÉCIDE (TOUR 5) ====================

# Les extraits RÉELS du jeu fixe du tour 4, OCR compris (fixtures/blog/jeu_controle_tour4.json,
# recopiés de data/blog/tour4/jeu-controle.json) : Code civil art. 1743 et 1752, AUDCG art. 127 et
# 133, Code du travail art. 39 ; et les passages qu'y a copiés ministral-14b (bail P12, travail P14).
_JEU = json.loads(
    (Path(__file__).parents[1] / "fixtures" / "blog" / "jeu_controle_tour4.json").read_text(encoding="utf-8")
)
_BAIL = {int(n): e["texte"] for n, e in _JEU["extraits"]["bail"].items()}
_TRAVAIL = {int(n): e["texte"] for n, e in _JEU["extraits"]["travail"].items()}
_ART_1743, _ART_127_REEL, _ART_133_REEL, _ART_1752 = _BAIL[3], _BAIL[5], _BAIL[6], _BAIL[7]
_ART_39_REEL = _TRAVAIL[5]


def _extraits_du_jeu(textes: dict, numeros: dict) -> list:
    return [ExtraitDuControle(n, "Texte", numeros[n], False, textes[n]) for n in sorted(textes)]


class TestUneCoupureNeCachePasCeQuiDecide:
    """
    Revue de l'API du tour 5 (data/blog/tour5/revue-api, r4_coupures_reelles.py) : tenu mot à mot,
    le passage ne l'était pas pour le sens. Le modèle recollait autour de « … » des mots VRAIS de
    l'extrait en la règle inverse ; le serveur gardait « soutenue » et citait le collage comme texte
    de l'extrait. Ce sont l'expulsion sans juge et la preuve inversée de l'essai réel.
    """

    @pytest.mark.parametrize("passage, texte", [
        # Art. 39 : « au travailleur » vient de l'alinéa 4 a), une autre phrase, 50 mots plus loin.
        ('"Dans tous cas de licenciement, il appartient… au travailleur" (Art. 39(3))', _ART_39_REEL),
        ('"Dans tous cas de licenciement, il appartient […] au travailleur" [5]', _ART_39_REEL),
        ('"il appartient… au travailleur, de son niveau de qualification"', _ART_39_REEL),
        # Art. 1743 : la coupure cache « ne » (« l'acquéreur ne peut expulser »).
        ('"l\'acquéreur… peut expulser le fermier ou le locataire"', _ART_1743),
        # Art. 1752 : « qui ne garnit pas la maison de meubles suffisants, peut être expulsé ».
        ('"Le locataire qui… peut être expulsé"', _ART_1752),
        # Art. 127 : « ne … que » caché, en entier ou à moitié.
        ('"Ce motif… peut être invoqué… si les faits se sont poursuivis"', _ART_127_REEL),
        ('"Ce motif ne peut être invoqué… si les faits se sont poursuivis"', _ART_127_REEL),
    ], ids=["art39-au-travailleur", "art39-crochets", "art39-qualification", "art1743", "art1752",
            "art127-ne-que", "art127-que"])
    def test_le_collage_n_appuie_rien(self, passage, texte):
        assert lire_le_passage(passage, [1], {1: texte})[0] is None

    @pytest.mark.parametrize("passage, texte", [
        ('"l\'acquéreur peut expulser le fermier ou le locataire"', _ART_1743),
        ('"il appartient au travailleur d\'apporter la preuve"', _ART_39_REEL),
    ], ids=["art1743", "art39"])
    def test_temoins_sans_coupure_refuses_aussi(self, passage, texte):
        assert lire_le_passage(passage, [1], {1: texte})[0] is None

    def test_une_coupure_qui_ne_cache_rien_qui_decide_passe(self):
        """Mesure réelle du 04/11 (bail P5, P9) : la règle ET son exception, dans la même phrase."""
        assert lire_le_passage(
            '"L\'acquéreur ne peut expulser le fermier ou le locataire... à moins qu\'il ne se soit '
            'réservé ce droit par le contrat de bail" (Extrait 3)', [3], {3: _ART_1743},
        ) == ([(3, "l'acquéreur ne peut expulser le fermier ou le locataire […] à moins qu'il ne se soit "
                   "réservé ce droit par le contrat de bail")], "")

    def test_deux_phrases_ou_l_autre_ordre_cites_a_part(self):
        """Jamais « il appartient […] incombe au travailleur » : chaque phrase sa citation, dans l'ordre du modèle."""
        assert lire_le_passage(
            '"incombe au travailleur… Dans tous cas de licenciement, il appartient"', [5], {5: _ART_39_REEL},
        ) == ([(5, "incombe au travailleur"), (5, "Dans tous cas de licenciement, il appartient")], "")

    @pytest.mark.parametrize("cle, cite", [
        # Bail P12, art. 133 réel : deux phrases, deux citations ; la « juridiction compétente »
        # qui « prononce » est celle de l'alinéa 4, pas celle de l'alinéa 3.
        ("bail P12", ["La demande en justice […] doit être précédée d'une mise en demeure",
                      "La juridiction compétente […] prononce […] l'expulsion"]),
        # Travail P14, art. 39 réel : le plancher, puis le plafond, de la même phrase.
        ("travail P14", ["le montant des dommages-intérêts, sans excéder un (1) mois de salaire par année "
                         "d'ancienneté dans l'entreprise, ne peut être inférieur à trois (3) mois de salaire"]),
    ])
    def test_les_cas_reels_legitimes_restent_soutenus(self, cle, cite):
        reel = _JEU["passages_reels"][cle]
        textes = _BAIL if cle.startswith("bail") else _TRAVAIL
        numeros = {n: e["numero"] for n, e in (
            (int(n), e) for n, e in _JEU["extraits"]["bail" if cle.startswith("bail") else "travail"].items()
        )}
        phrases = [PhraseAJuger("P1", "resume", 0, reel["phrase"])]

        (jugee,) = lire_les_verdicts(_reponse(jugement(
            "P1", "soutenue", reel["passage"], reel["extraits"], raison="L'extrait le dit.",
        )), phrases, _extraits_du_jeu(textes, numeros))

        assert jugee["verdict"] == "soutenue"
        n = reel["extraits"][0]
        assert jugee["raison"] == f"L'extrait le dit. — Extrait {n} : " + " ".join(f"« {c} »" for c in cite)

    @pytest.mark.parametrize("phrase, passage, n, textes, saute", [
        # Le scénario de la revue, par la route (controle-route-1.json) : la preuve inversée.
        ("Il appartient au travailleur de prouver que son licenciement est abusif.",
         '"Dans tous cas de licenciement, il appartient… au travailleur" [5]', 5, _TRAVAIL, None),
        # L'expulsion sans juge : l'acquéreur, le locataire qui ne garnit pas la maison.
        ("L'acheteur du logement peut expulser le locataire.",
         '"l\'acquéreur… peut expulser le fermier ou le locataire" [3]', 3, _BAIL, "« ne »"),
        ("Un locataire peut être expulsé.", '"Le locataire qui… peut être expulsé" [7]', 7, _BAIL,
         "« ne », « pas »"),
        ("Le bailleur peut invoquer un motif grave si les faits se sont poursuivis.",
         '"Ce motif… peut être invoqué… si les faits se sont poursuivis" [5]', 5, _BAIL, "« ne », « que »"),
    ], ids=["preuve-inversee", "art1743", "art1752", "art127"])
    def test_la_phrase_fausse_ne_garde_pas_soutenue(self, phrase, passage, n, textes, saute):
        numeros = {k: str(k) for k in textes}

        (jugee,) = lire_les_verdicts(_reponse(jugement(
            "P1", "soutenue", passage, [n], raison=f"L'extrait {n} le dit.",
        )), [PhraseAJuger("P1", "resume", 0, phrase)], _extraits_du_jeu(textes, numeros))

        assert jugee["verdict"] == "non_soutenue"
        assert jugee["raison"].startswith("À vérifier : le contrôle la disait soutenue")
        # Jamais sous la forme des passages vérifiés, que le front montre en citation.
        assert f"Extrait {n} :" not in jugee["raison"] and "[…]" not in jugee["raison"]
        if saute:
            assert f"(sa coupure « … » saute {saute} dans l'extrait {n}, entre les mots copiés « " in jugee["raison"]

    def test_une_contradiction_recollee_ne_bloque_pas_non_plus(self):
        """« contredite » sur un collage : la contradiction n'est pas dans le texte, elle se relit."""
        (jugee,) = lire_les_verdicts(_reponse(jugement(
            "P1", "contredite", '"l\'acquéreur… peut expulser le fermier ou le locataire" [3]', [3],
        )), [PhraseAJuger("P1", "resume", 0, "L'acheteur ne peut pas expulser le locataire.")],
            _extraits_du_jeu(_BAIL, {k: str(k) for k in _BAIL}))

        assert jugee["verdict"] == "non_soutenue"
        assert jugee["raison"].startswith("À vérifier : le contrôle la disait contredite")

    @pytest.mark.parametrize("texte, passage, saute", [
        ("Le juge ne statue que si le preneur est entendu.",
         '"Le juge ne statue… si le preneur est entendu"', ("que",)),
        ("Le juge saisi n'est compétent qu'après la mise en demeure.",
         '"Le juge saisi… est compétent… après la mise en demeure"', ("n'", "qu'")),
        # Le « ne » d'après « que » est explétif : seul « à moins » décide.
        ("Le bail est résilié, à moins que le preneur ne paie dans le mois.",
         '"Le bail est résilié… le preneur ne paie dans le mois"', ("à moins",)),
        # « que » sans « ne » dans sa phrase, « n » de « n° » : rien ne décide.
        ("Le juge statue après que le preneur est entendu.", '"Le juge statue… le preneur est entendu"', None),
        ("La loi applicable, n° 2016/007, fixe les peines du juge.",
         '"La loi applicable… fixe les peines du juge"', None),
    ], ids=["ne-que", "n-qu", "a-moins", "que-seul", "numero"])
    def test_ce_que_cache_une_coupure(self, texte, passage, saute):
        """« que » ne décide qu'après un « ne » de sa phrase ; « n » qu'élidé (pas le « n » de « n° »)."""
        lecture = lire_le_passage_en_detail(passage, [1], {1: texte})

        assert (lecture.appuis is None) is (saute is not None)
        assert lecture.coupure == ((1, saute) if saute else None)


class TestUneCaseTombeeEstEffacee:
    """
    Revue du front du tour 5, par la vraie API (data/blog/tour5/revue-front) : une case gardait
    l'empreinte de ce qu'elle atteste ; un contrôle qui jugeait autrement sa partie la faisait
    seulement ne plus compter, et le contrôle suivant, qui redonnait les verdicts d'avant, la
    faisait revivre. L'article caché revenait en ligne sans relecture ni « Publier ».
    """

    def test_un_controle_qui_juge_autrement_efface_la_case_et_le_suivant_ne_la_fait_pas_revivre(self):
        contenu = _contenu()
        x = _controle_sur(contenu)
        y = _controle_sur(contenu, "P2")  # le contexte, contredit
        z = _controle_sur(contenu)  # les verdicts de X, de nouveau
        relu = _tout_relu(contenu, x)

        apres_y = relecture_apres_un_controle(contenu, x, y, relu)
        apres_z = relecture_apres_un_controle(contenu, y, z, apres_y)

        assert apres_y["parties"]["contexte"] is None
        assert parties_relues(contenu, y, apres_y) == {p: p != "contexte" for p in PARTIES}
        assert parties_relues(contenu, z, apres_z) == {p: p != "contexte" for p in PARTIES}
        # Ce qui ne bouge pas reste : qui a relu, quand, contre quel contrôle.
        assert {k: apres_z[k] for k in ("par", "le")} == {"par": "admin", "le": "2026-10-30T10:00:00+00:00"}

    def test_les_memes_verdicts_gardent_la_relecture_telle_quelle(self):
        contenu = _contenu()
        x = _controle_sur(contenu)
        relu = _tout_relu(contenu, x)

        assert relecture_apres_un_controle(contenu, x, {**x, "fait_le": "2026-10-30T11:00:00+00:00"}, relu) is relu
        assert relecture_apres_un_controle(contenu, x, x, {}) == {}
        assert relecture_apres_un_controle(contenu, None, x, None) is None

    def test_une_case_qui_ne_comptait_plus_ne_revit_pas(self):
        """Tombée avant ce correctif (enregistrée, mais périmée) : effacée au contrôle suivant."""
        contenu = _contenu()
        x, y = _controle_sur(contenu), _controle_sur(contenu, "P2")
        relu = _tout_relu(contenu, x)  # cochée contre X, alors que Y est enregistré

        apres = relecture_apres_un_controle(contenu, y, _controle_sur(contenu), relu)

        assert apres["parties"]["contexte"] is None
        assert parties_relues(contenu, _controle_sur(contenu), apres)["contexte"] is False


def _controle_avec(contenu: ContenuDuBillet, verdicts: dict, fait_le: str = "2026-11-05T10:00:00+00:00") -> dict:
    """Un contrôle à jour de `contenu` : tout « soutenue », sauf `verdicts` ({id: (verdict, nature)})."""
    controle = _controle_sur(contenu)
    controle["fait_le"] = fait_le
    for phrase in controle["phrases"]:
        phrase["nature"] = "regle"
        if phrase["id"] in verdicts:
            phrase["verdict"], phrase["nature"] = verdicts[phrase["id"]]
    return controle


def _relu_et_confirme(contenu: ContenuDuBillet, controle, *ids) -> dict:
    """Chaque partie relue, et les phrases `ids` confirmées, contre `controle`."""
    return {
        **_tout_relu(contenu, controle), "controle": controle["fait_le"],
        "phrases_confirmees": {i: e for i, e in empreintes_a_confirmer(contenu, controle).items() if i in ids},
    }


class TestLesPhrasesAConfirmer:
    """
    Décision du propriétaire (04/11) : « Confirmer phrase par phrase ». Essai réel du tour 6 :
    les 10 phrases du bail qui disaient ou présupposaient l'expulsion sans juge étaient toutes
    « À vérifier » (non_soutenue), du même gris que les fausses alertes ; 7 cases cochées
    suffisaient à publier « peut être expulsé sans décision de justice ». Une phrase de RÈGLE
    que le contrôle n'a pas trouvée dans les extraits se réécrit, ou se confirme une à une.
    """

    def test_quelles_phrases_sont_a_confirmer(self):
        contenu = _contenu()
        controle = _controle_avec(contenu, {
            "P2": ("non_soutenue", "regle"), "P3": ("hors_sources", "regle"), "P4": ("non_soutenue", "autre"),
            "P5": ("contredite", "regle"), "P6": ("soutenue", "regle"), "P7": ("illisible", "regle"),
        })
        del controle["phrases"][7]["nature"]  # P8 : une phrase que le modèle n'a pas jugée
        controle["phrases"][7]["verdict"] = "non_soutenue"

        assert [p["id"] for p in phrases_a_confirmer(controle)] == ["P2", "P3", "P7", "P8"]
        assert [p["nature"] for p in controle_affiche(controle)["phrases"]][:8] == [
            "regle", "regle", "regle", "autre", "regle", "regle", "regle", "regle",
        ]
        assert phrases_a_confirmer(None) == [] and phrases_a_confirmer({"phrases": "x"}) == []

    def test_l_etat_et_son_ordre(self):
        """« phrases_a_confirmer » vient après les phrases contredites, avant la relecture."""
        contenu = _contenu()
        controle = _controle_avec(contenu, {"P4": ("non_soutenue", "regle"), "P8": ("hors_sources", "regle")})
        contredit = _controle_avec(contenu, {"P4": ("non_soutenue", "regle"), "P5": ("contredite", "regle")})

        assert ETATS.index("phrases_contredites") < ETATS.index("phrases_a_confirmer") < ETATS.index(
            "relecture_incomplete"
        )
        assert etat_de_publication(contenu, controle, _tout_relu(contenu, controle), [True, True]) == (
            "phrases_a_confirmer"
        )
        assert etat_de_publication(contenu, controle, {}, [True, True]) == "phrases_a_confirmer"
        assert etat_de_publication(contenu, controle, _relu_et_confirme(contenu, controle, "P4"), [True, True]) == (
            "phrases_a_confirmer"
        )
        assert etat_de_publication(
            contenu, controle, _relu_et_confirme(contenu, controle, "P4", "P8"), [True, True]
        ) == "publiable"
        confirme_sans_cases = {**_relu_et_confirme(contenu, controle, "P4", "P8"), "parties": {}}
        assert etat_de_publication(contenu, controle, confirme_sans_cases, [True, True]) == "relecture_incomplete"
        assert etat_de_publication(contenu, contredit, _tout_relu(contenu, contredit), [True, True]) == (
            "phrases_contredites"
        )
        # Une phrase d'une autre nature (un fait, un conseil) ne se confirme pas.
        fait = _controle_avec(contenu, {"P2": ("non_soutenue", "autre")})
        assert etat_de_publication(contenu, fait, _tout_relu(contenu, fait), [True, True]) == "publiable"

    def test_le_motif_nomme_les_phrases_et_dit_quoi_faire(self):
        contenu = _contenu()
        un = _controle_avec(contenu, {"P4": ("non_soutenue", "regle")})
        deux = _controle_avec(contenu, {"P4": ("non_soutenue", "regle"), "P8": ("hors_sources", "regle")})

        motif = motif_du_refus("phrases_a_confirmer", contenu, un, _tout_relu(contenu, un), [True, True])
        assert motif.startswith("Publication refusée : une phrase de règle que le contrôle IA n'a pas trouvée")
        assert "P4, résumé : « La loi punit l'intrusion. »" in motif
        assert "J'ai vérifié cette phrase dans le texte" in motif and "Réécrivez-les" in motif
        motif = motif_du_refus(
            "phrases_a_confirmer", contenu, deux, _relu_et_confirme(contenu, deux, "P4"), [True, True]
        )
        assert "une phrase de règle" in motif and "P8, impact pour le citoyen" in motif and "P4" not in motif
        motif = motif_du_refus("phrases_a_confirmer", contenu, deux, {}, [True, True])
        assert "2 phrases de règle que le contrôle IA n'a pas trouvées dans les extraits ne sont ni" in motif

    def test_confirmer_puis_retirer_une_confirmation(self):
        contenu = _contenu()
        controle = _controle_avec(contenu, {"P4": ("non_soutenue", "regle"), "P8": ("hors_sources", "regle")})
        relue = _tout_relu(contenu, controle)

        confirmee = nouvelle_relecture(
            contenu, contenu, controle, relue, None, "Awa", 7, "2026-11-05T11:00:00+00:00",
            controle_relu=controle["fait_le"], confirmees=["P8", "P4", "P8"],
        )
        inchangee = nouvelle_relecture(
            contenu, contenu, controle, confirmee, None, "Paul", 8, "2026-11-05T12:00:00+00:00",
        )
        retiree = nouvelle_relecture(
            contenu, contenu, controle, confirmee, None, "Paul", 8, "2026-11-05T12:00:00+00:00",
            controle_relu=controle["fait_le"], confirmees=["P4"],
        )

        assert phrases_confirmees(contenu, controle, confirmee) == ["P4", "P8"]
        assert set(confirmee["phrases_confirmees"]) == {"P4", "P8"}
        assert (confirmee["par"], confirmee["par_id"], confirmee["le"]) == ("Awa", 7, "2026-11-05T11:00:00+00:00")
        assert confirmee["controle"] == controle["fait_le"]
        assert etat_de_publication(contenu, controle, confirmee, [True, True]) == "publiable"
        # Rien d'envoyé : rien ne change, ni qui ni quand.
        assert inchangee == confirmee
        # La liste ENTIÈRE : une phrase absente n'est plus confirmée.
        assert phrases_confirmees(contenu, controle, retiree) == ["P4"]
        assert (retiree["par"], retiree["le"]) == ("Paul", "2026-11-05T12:00:00+00:00")
        assert etat_de_publication(contenu, controle, retiree, [True, True]) == "phrases_a_confirmer"
        assert relecture_affichee(contenu, controle, retiree)["phrases_confirmees"] == ["P4"]

    def test_seule_une_phrase_a_confirmer_se_garde(self):
        """Une phrase connue mais soutenue, contredite, ou d'une autre nature : pas gardée."""
        contenu = _contenu()
        controle = _controle_avec(contenu, {
            "P4": ("non_soutenue", "regle"), "P5": ("contredite", "regle"), "P6": ("non_soutenue", "autre"),
        })

        relecture = nouvelle_relecture(
            contenu, contenu, controle, {}, None, "Awa", 7, "2026-11-05T11:00:00+00:00",
            controle_relu=controle["fait_le"], confirmees=["P2", "P4", "P5", "P6"],
        )

        assert list(relecture["phrases_confirmees"]) == ["P4"]
        assert nouvelle_relecture(
            contenu, contenu, controle, {}, None, "Awa", 7, "2026-11-05T11:00:00+00:00",
            controle_relu=controle["fait_le"], confirmees=["P2"],
        ) == {}

    def test_une_phrase_d_une_partie_modifiee_perd_sa_confirmation(self):
        """Comme la case de sa partie : sa réécriture se fait juger par un nouveau contrôle."""
        contenu = _contenu()
        controle = _controle_avec(contenu, {"P4": ("non_soutenue", "regle"), "P8": ("hors_sources", "regle")})
        confirmee = _relu_et_confirme(contenu, controle, "P4", "P8")
        apres = replace(contenu, summary="La loi punit l'intrusion. Le juge seul décide.")

        relecture = nouvelle_relecture(
            contenu, apres, controle, confirmee, None, "Awa", 7, "2026-11-05T11:00:00+00:00",
            controle_relu=controle["fait_le"], confirmees=["P4", "P8"],
        )
        sans_liste = nouvelle_relecture(contenu, apres, controle, confirmee, None, "Awa", 7, "2026-11-05T11:00:00+00:00")

        assert list(relecture["phrases_confirmees"]) == ["P8"]
        assert list(sans_liste["phrases_confirmees"]) == ["P8"]

    @pytest.mark.parametrize("ids, relu, controle, attendu", [
        (None, None, "un", None),
        ([], "2026-11-05T10:00:00+00:00", "un", None),
        (["P2", "P4"], "2026-11-05T10:00:00+00:00", "un", None),
        (["P4"], None, "un", "ne disent pas contre quel contrôle IA"),
        ([], " ", "un", "ne disent pas contre quel contrôle IA"),
        (["P4"], "2026-11-01T10:00:00+00:00", "un", "fait le 2026-11-05T10:00:00+00:00, relancé depuis"),
        (["P4", "P31"], "2026-11-05T10:00:00+00:00", "un", "n'a pas la phrase P31"),
        (["P31", "P32", "x"], "2026-11-05T10:00:00+00:00", "un", "n'a pas les phrases P31, P32 et x"),
        (["P4"], "2026-11-05T10:00:00+00:00", None, "aucun contrôle IA n'est enregistré"),
        ([], None, None, "aucun contrôle IA n'est enregistré"),
    ], ids=[
        "absentes", "vides", "connues", "sans-controle-nomme", "controle-blanc", "un-autre-controle",
        "une-inconnue", "des-inconnues", "aucun-controle", "vides-sans-aucun-controle",
    ])
    def test_le_refus_des_confirmations_perimees(self, ids, relu, controle, attendu):
        """
        AVENANT DU TOUR 7 : toute liste part avec le contrôle relu (comme une case cochée), et
        chaque identifiant est une phrase du contrôle ENREGISTRÉ ; sinon 422, rien n'est appliqué.
        """
        contenu = _contenu()
        enregistre = _controle_avec(contenu, {"P4": ("non_soutenue", "regle")}) if controle else None

        refus = confirmations_perimees(ids, relu, enregistre)

        if attendu is None:
            assert refus is None
        else:
            assert attendu in refus and "Rien n'a été enregistré" in refus

    def test_un_nouveau_controle_fait_tomber_une_confirmation_comme_une_case(self):
        """
        Elle ne reste que si la phrase, son verdict et les sources sont les mêmes avec le nouveau
        contrôle ; tombée, elle est effacée, et un contrôle suivant qui redonne le verdict
        d'avant ne la fait pas revivre (revue du front du tour 5, pour les cases).
        """
        contenu = _contenu()
        x = _controle_avec(contenu, {"P4": ("non_soutenue", "regle"), "P8": ("hors_sources", "regle")})
        relue = _relu_et_confirme(contenu, x, "P4", "P8")
        y = _controle_avec(contenu, {"P4": ("non_soutenue", "regle"), "P8": ("non_soutenue", "regle")},
                           fait_le="2026-11-05T11:00:00+00:00")
        z = _controle_avec(contenu, {"P4": ("non_soutenue", "regle"), "P8": ("hors_sources", "regle")},
                           fait_le="2026-11-05T12:00:00+00:00")

        apres_y = relecture_apres_un_controle(contenu, x, y, relue)
        apres_z = relecture_apres_un_controle(contenu, y, z, apres_y)
        meme = relecture_apres_un_controle(contenu, x, z, relue)

        assert phrases_confirmees(contenu, y, apres_y) == ["P4"] and set(apres_y["phrases_confirmees"]) == {"P4"}
        assert phrases_confirmees(contenu, z, apres_z) == ["P4"]
        assert etat_de_publication(contenu, z, apres_z, [True, True]) == "phrases_a_confirmer"
        # Les mêmes verdicts : la relecture telle quelle, cases et confirmations.
        assert meme is relue
        # Des sources reconstruites autrement : plus rien ne compte, et le contrôle suivant efface.
        autres_sources = replace(contenu, sources=[_SOURCES[0]])
        assert phrases_confirmees(autres_sources, x, relue) == []
        refait = _controle_avec(autres_sources, {"P4": ("non_soutenue", "regle")})
        assert relecture_apres_un_controle(autres_sources, None, refait, relue)["phrases_confirmees"] == {}

    def test_ce_que_l_api_montre(self):
        contenu = _contenu()
        controle = _controle_avec(contenu, {"P4": ("non_soutenue", "regle")})

        affichee = relecture_affichee(contenu, controle, _relu_et_confirme(contenu, controle, "P4"))
        vide = relecture_affichee(contenu, controle, {})

        assert affichee["phrases_confirmees"] == ["P4"] and affichee["controle"] == controle["fait_le"]
        assert vide == {
            "parties": {p: False for p in PARTIES}, "par": None, "le": None, "controle": None,
            "phrases_confirmees": [],
        }
        assert set(controle_affiche(controle)["phrases"][0]) == {
            "id", "partie", "index", "texte", "verdict", "extraits", "raison", "nature",
        }


class TestLesRenvoisAuxExtraits:
    """Revue du tour 4 (contrôle de la copie) : « L'extrait 36 confirme… » pour l'ARTICLE 36."""

    # Les extraits du contrôle de la copie du 04/11 (data/blog/e2e-tour4/controle-ancien.json).
    _EXTRAITS = [
        ExtraitDuControle(n, "Code du travail", numero, False, f"Texte {n}.")
        for n, numero in {1: "39", 2: "40", 3: "35", 4: "36", 5: "37", 6: "38", 7: "39", 11: "34"}.items()
    ]

    @pytest.mark.parametrize("raison, attendue", [
        ("Les extraits 7 et 37 parlent de dommages-intérêts.",
         "Les extraits 7 et 5 (article 37) parlent de dommages-intérêts."),
        ("L'extrait 40 évoque les licenciements pour motif économique.",
         "L'extrait 2 (article 40) évoque les licenciements pour motif économique."),
        ("Les extraits 34, 36, 37 et 40 encadrent la rupture.",
         "Les extraits 11 (article 34), 4 (article 36), 5 (article 37) et 2 (article 40) encadrent la rupture."),
        ("L'extrait 39 confirme ces montants.", "L'extrait 1 ou 7 (article 39) confirme ces montants."),
        ("Passages [36] et [4].", "Passages [4] (article 36) et [4]."),
        ("L'extrait n° 99 le dit.", "L'extrait n° 99 (aucun extrait n'a ce numéro) le dit."),
        ("L'extrait 5 et l'article 36 le disent.", "L'extrait 5 et l'article 36 le disent."),
        ("Les extraits 1, 2 et 7 le disent.", "Les extraits 1, 2 et 7 le disent."),
    ])
    def test_un_renvoi_nomme_un_extrait_qui_existe(self, raison, attendue):
        assert renvois_aux_extraits(raison, self._EXTRAITS) == attendue

    def test_dans_les_verdicts(self):
        (jugee, *_) = lire_les_verdicts(_reponse(
            jugement("P1", "non_soutenue", raison="L'extrait 36 ne le dit pas."),
        ), _PHRASES, self._EXTRAITS)

        assert jugee["raison"] == "L'extrait 4 (article 36) ne le dit pas."

    @pytest.mark.parametrize("raison, attendue", [
        ("Extrait 5 : « la preuve incombe au travailleur ».", "Extrait 5, « la preuve incombe au travailleur »."),
        ("Non. — Extrait 5 : « la preuve incombe au travailleur »", "Non. — Extrait 5, « la preuve incombe au travailleur »"),
        ("Les extraits 5 : « a » et 7 : « b ».", "Les extraits 5, « a » et 7 : « b »."),
        ("L'extrait 5 : il le dit.", "L'extrait 5 : il le dit."),
    ])
    def test_le_modele_n_ecrit_pas_la_forme_d_un_passage_verifie(self, raison, attendue):
        """
        Le front montre « Extrait 5 : « … » » comme un texte vérifié par le serveur (controle.ts) :
        écrit par le modèle, il n'en a plus la forme.
        """
        assert renvois_aux_extraits(raison, self._EXTRAITS) == attendue

    def test_seul_le_serveur_cite_un_extrait(self):
        (jugee, *_) = lire_les_verdicts(_reponse(
            jugement("P1", "non_soutenue", raison="Non soutenue — Extrait 5 : « il appartient au salarié »"),
        ), _PHRASES, _EXTRAITS)

        assert "Extrait 5 :" not in jugee["raison"]


# ==================== 5. LE PROMPT ====================


class TestLePrompt:

    def test_extraits_puis_phrases_groupees_le_titre_d_abord(self):
        contenu = _contenu(citations=[_citation(1), _citation(2, suite=True, article_number="39"),
                                      _citation(3, verifie=False)])
        extraits = [
            ExtraitDuControle(1, "Code du travail", "36", False, "Article 36.- (1) Toute rupture…"),
            ExtraitDuControle(2, "Code du travail", "39", True, "(6) Le salaire…"),
        ]

        prompt = prompt_du_controle(extraits, phrases_du_billet(contenu), contenu, {0: 1, 1: 2})

        assert prompt.startswith("EXTRAITS (2, numérotés ; les seuls textes contre lesquels juger) :")
        assert "[1] Code du travail — Article 36\n\nArticle 36.- (1) Toute rupture…" in prompt
        assert ("[2] Code du travail — Article 39 (suite : ce passage commence au milieu de "
                "l'article 39, son début n'est pas fourni)") in prompt
        assert "PHRASES À JUGER (12, dans l'ordre de l'article) :\n\nTitre de l'article\nP1 : Expulsion" in prompt
        assert "\nContexte\nP2 : Un locataire rentre chez lui.\nP3 : Ses affaires sont dehors.\n" in prompt
        assert "\nPoint clé 2\nP7 : Le congé suit l'usage.\n" in prompt
        # Une explication se juge contre l'extrait qu'elle accompagne ; une suite se dit « (suite) ».
        assert "Explication de la citation 2, qui accompagne l'extrait 2 (Article 39 (suite))" in prompt
        assert "Explication de la citation 3 (Article 3, citation non vérifiée" in prompt
        assert "de P1 à P12, sans en oublier ni en ajouter : affirmation" in prompt
        assert prompt.endswith("passage copié mot pour mot, raison, verdict.")

    def test_un_lot_dit_que_le_reste_de_l_article_est_juge_a_part(self):
        contenu = _contenu()
        phrases = phrases_du_billet(contenu)
        extraits = [ExtraitDuControle(1, "Code du travail", "36", False, "Article 36.- (1) Toute rupture…")]

        tout = prompt_du_controle(extraits, phrases, contenu, {0: 1})
        lot = prompt_du_controle(extraits, phrases[3:8], contenu, {0: 1}, autres=len(phrases) - 5)

        # Tout l'article en un appel : le prompt du tour 4, mot pour mot.
        assert "PHRASES À JUGER (11, dans l'ordre de l'article) :\n" in tout
        assert (
            "PHRASES À JUGER (5, dans l'ordre de l'article ; les 6 autres phrases de l'article sont "
            "jugées à part) :\n\nRésumé\nP4 : La loi punit l'intrusion.\n"
        ) in lot
        assert "P1 :" not in lot and "P9 :" not in lot
        assert "de P4 à P8, sans en oublier ni en ajouter" in lot

    def test_les_consignes_et_le_schema(self):
        assert "QUE d'après les extraits" in SYSTEM_DU_CONTROLE
        # Tour 5 : un extrait abîmé par la numérisation ne se répare pas (mesure du 04/11 :
        # « une indemnité » écrit à la place d'un passage illisible de l'art. 36).
        assert "ne répare jamais ces mots et n'en ajoute aucun à leur place" in SYSTEM_DU_CONTROLE
        assert "Une phrase affirme aussi ce qu'elle présuppose ; un titre ou une question aussi" in (
            SYSTEM_DU_CONTROLE
        )
        assert "COPIÉS MOT POUR MOT" in SYSTEM_DU_CONTROLE
        assert '"contredite" : le passage dit le contraire' in SYSTEM_DU_CONTROLE
        assert "qui doit prouver ; qui décide (un juge, une juridiction, ou une partie seule)" in (
            SYSTEM_DU_CONTROLE
        )
        assert "un bail professionnel étendu au bail d'habitation" in SYSTEM_DU_CONTROLE
        assert "Une exception est une affirmation à part entière" in SYSTEM_DU_CONTROLE
        assert "Si tu ne peux pas copier un passage qui dit ce que dit la phrase" in SYSTEM_DU_CONTROLE
        # Mesure réelle du 04/11 : la forme du passage que le programme vérifie, quatre verdicts
        # et pas un de plus, les phrases composées, la charge de la preuve.
        assert '"Il appartient à l\'administration de prouver l\'infraction" [1]' in SYSTEM_DU_CONTROLE
        assert "l'un de ces quatre mots, et aucun autre" in SYSTEM_DU_CONTROLE
        assert "Une phrase qui joint plusieurs affirmations" in SYSTEM_DU_CONTROLE
        assert "La preuve : si un passage la met à la charge de quelqu'un" in SYSTEM_DU_CONTROLE
        item = SCHEMA_DU_CONTROLE["properties"]["phrases"]["items"]
        # Ce que la phrase affirme, le passage copié, la comparaison, PUIS le verdict.
        ordre = ["id", "affirmation", "nature", "extraits", "passage", "raison", "verdict"]
        assert list(item["properties"]) == ordre and item["required"] == ordre
        # Gemini range les champs par ordre alphabétique : le verdict y reste le dernier.
        assert sorted(ordre)[-1] == "verdict"
        assert CONTROLE_TEMPERATURE == 0.0
        assert DELAI_CONTROLE_S == 180


# ==================== 5 bis. LES APPELS DU CONTRÔLE (TOUR 5, D2) ====================


def _phrases_en_groupes(*groupes) -> list:
    """Des phrases P1, P2… par groupe : (partie, rang, nombre de phrases)."""
    phrases = []
    for partie, index, nombre in groupes:
        for _ in range(nombre):
            phrases.append(PhraseAJuger(f"P{len(phrases) + 1}", partie, index, f"Phrase {len(phrases) + 1}."))
    return phrases


def _groupes(lot) -> list:
    return list(dict.fromkeys((p.partie, p.index) for p in lot))


async def _appels_du_plan(post, un_seul_appel: bool = True) -> tuple:
    """
    Les (phrases, jetons d'entrée) des lots de `post`, tels que le contrôle les planifie, sans
    base : ses citations n'y sont pas (comme dans la base de test), donc aucun extrait ajouté.
    `un_seul_appel` : comme pour le modèle à budget de jetons (tour 7).
    """
    contenu = ContenuDuBillet.de(post)
    phrases = phrases_du_billet(contenu)
    extraits, rattachement, _ = await extraits_du_controle(None, replace(contenu, citations=[]))
    client = MistralService(api_key="cle-test", model_name=GRAND)
    return tuple(
        (len(lot), client.jetons_d_entree(
            prompt_du_controle(extraits, lot, contenu, rattachement, autres=len(phrases) - len(lot)),
            SYSTEM_DU_CONTROLE,
        ))
        for lot in lots_du_controle(phrases, un_seul_appel=un_seul_appel)
    )


class TestLesLots:
    """
    Revue du tour 4 (défaut D2) : un article de ~30 phrases passait le plafond de 80, mais son
    contrôle, d'un seul appel (~4 s par phrase mesurées), ne finissait jamais en 120 s.
    Tour 6 : les appels se suivent (mistral-large-4, 20 000 jetons par minute), l'article se
    juge dans la SOMME de ses appels, et le plafond tombe à 25 phrases.
    """

    def test_jusqu_a_vingt_phrases_un_seul_appel(self):
        phrases = _phrases_en_groupes(("titre", 0, 1), ("contexte", 0, 4), ("resume", 0, 15))

        assert lots_du_controle(phrases) == [phrases]
        assert PHRASES_PAR_APPEL == 20 and PHRASES_AU_PLUS == 25

    @pytest.mark.parametrize("caracteres", [None, 17_000, 24_000, 30_000, 36_000])
    async def test_le_plafond_tient_dans_le_delai_au_pire_mesure(self, grand_modele, caracteres):
        """
        LA PREUVE DU PLAFOND (tour 7). Au tour 6, l'article au plafond se jugeait en deux lots
        (13 + 12), comptés à 3,42 s par phrase : ~101 s prévus. À l'essai réel, un lot de 12
        phrases a pris 76 s (prévu 41), et le contrôle a fini en 504 après deux lots payés : la
        réflexion du modèle domine chaque appel, quelle que soit sa taille.
        Ici, l'article RÉEL du travail porté au plafond (25 phrases), avec ses extraits ou
        jusqu'aux 36 000 caractères du contrôle, au plan de production :
        - UN appel ; seul, il dure DUREE_D_UN_APPEL_AU_PIRE_S (au plafond, à la vitesse la plus
          lente mesurée) : marge comprise, 150 s, sous les 180 s du délai, et 30 s restent pour attendre ;
        - un appel rend au plus CONTROLE_MAX_TOKENS : à la vitesse la plus lente
          mesurée, il dure au plus ~135,6 s, sous les 140 s du plan ;
        - lancé à chaque seconde des 20 dernières d'un autre pareil (prévu au pire), il tient
          APRÈS lui : les jetons de l'autre sont sortis de la minute quand il part.
        """
        appels = await _appels_du_plan(article_reel(caracteres, au_plafond=True))
        limiteur = grand_modele.limiteurs[GRAND]
        seul = planifier(limiteur, 0.0, appels)

        assert [n for n, _ in appels] == [PHRASES_AU_PLUS]
        assert [lot.depart for lot in seul.lots] == [0.0]
        assert seul.duree == DUREE_D_UN_APPEL_AU_PIRE_S >= 105
        assert seul.duree + MARGE_DU_PLAN_S + 30 <= DELAI_CONTROLE_S
        assert CONTROLE_MAX_TOKENS / JETONS_PAR_SECONDE_AU_PIRE + 1 <= DUREE_D_UN_APPEL_AU_PIRE_S
        devant = controle_du_billet._Prevu(seul.fin, seul.limiteur)
        for avant_la_fin in range(21):
            arrivee = seul.fin - avant_la_fin
            budget = BudgetDuControle(
                GRAND, limiteur.copie(lambda t=arrivee: t), appels, arrivee + DELAI_CONTROLE_S,
                caracteres or 10_052, PHRASES_AU_PLUS,
            )

            plan, attente = budget.dans_la_file(devant)

            assert plan is not None, avant_la_fin
            assert plan.lots[0].depart == pytest.approx(seul.fin)
            assert plan.fin + MARGE_DU_PLAN_S <= arrivee + DELAI_CONTROLE_S + 1e-6
            assert attente >= avant_la_fin
        # Le plafond dépasse ceux qui ont coupé les essais réels du 06/11 (12 000) et du 08/11
        # (14 000), et tient, avec l'entrée d'un article généré (~4 000 jetons, marge comprise),
        # dans la minute ; le plan le compte entier pour le suivant.
        assert JETONS_DE_SORTIE_D_UN_APPEL_AU_PIRE == CONTROLE_MAX_TOKENS
        assert 14_000 < CONTROLE_MAX_TOKENS <= 20_000 - 4_400

    @pytest.mark.parametrize("groupes", [
        (("titre", 0, 1), ("contexte", 0, 3), ("resume", 0, 5), *(("point_cle", i, 2) for i in range(5)),
         ("impact", 0, 4), ("question", 0, 2)),
        (("titre", 0, 1), ("contexte", 0, 21), ("resume", 0, 1), ("impact", 0, 2)),
        (("titre", 0, 1), ("contexte", 0, 4), ("resume", 0, 20)),
    ], ids=["article", "une-partie-de-21", "lots-5-20-du-tour-6"])
    def test_en_un_seul_appel_tout_l_article_jusqu_au_plafond(self, groupes):
        """
        Tour 7 (le modèle à budget de jetons) : tout l'article en UN lot, une partie de plus
        de 20 phrases comprise ; au-delà du plafond, toujours 422.
        """
        phrases = _phrases_en_groupes(*groupes)

        assert len(phrases) == PHRASES_AU_PLUS
        assert lots_du_controle(phrases, un_seul_appel=True) == [phrases]
        with pytest.raises(TexteTropLongError) as erreur:
            lots_du_controle([*phrases, PhraseAJuger("P26", "question", 0, "Une de plus ?")], un_seul_appel=True)
        assert "26 phrases, 25 au plus" in str(erreur.value)
        with pytest.raises(TexteTropLongError):
            lots_du_controle([], un_seul_appel=True)

    @pytest.mark.parametrize("groupes, tailles", [
        # La forme d'un article généré au plafond.
        ((("titre", 0, 1), ("contexte", 0, 3), ("resume", 0, 5), *(("point_cle", i, 2) for i in range(5)),
          ("impact", 0, 4), ("question", 0, 2)), [13, 12]),
        # Des parties qui tombent mal : trois appels, jamais quatre.
        ((("titre", 0, 1), ("contexte", 0, 20), ("resume", 0, 1), ("impact", 0, 3)), [1, 20, 4]),
        ((("titre", 0, 1), ("contexte", 0, 1), ("resume", 0, 19), ("impact", 0, 4)), [2, 19, 4]),
    ], ids=["article", "parties-1-20-1-3", "parties-1-1-19-4"])
    def test_vingt_cinq_phrases_en_trois_appels_au_plus(self, groupes, tailles):
        phrases = _phrases_en_groupes(*groupes)

        lots = lots_du_controle(phrases)

        assert len(phrases) == PHRASES_AU_PLUS
        assert [len(lot) for lot in lots] == tailles
        assert [p for lot in lots for p in lot] == phrases

    def test_un_article_genere_en_deux_lots_egaux_sans_couper_une_partie(self):
        """La forme des articles de l'essai réel du 04/11 : 23 phrases."""
        phrases = _phrases_en_groupes(
            ("titre", 0, 1), ("contexte", 0, 3), ("resume", 0, 4), *(("point_cle", i, 1) for i in range(5)),
            ("impact", 0, 4), *(("explication", i, 1) for i in range(4)), ("question", 0, 2),
        )

        lots = lots_du_controle(phrases)

        assert [len(lot) for lot in lots] == [12, 11]
        assert [p for lot in lots for p in lot] == phrases
        # Aucun groupe n'est coupé entre deux lots.
        assert not set(_groupes(lots[0])) & set(_groupes(lots[1]))

    @pytest.mark.parametrize("phrases, message", [
        ([], "aucune phrase à contrôler"),
        (_phrases_en_groupes(*(("point_cle", i, 2) for i in range(13))), "26 phrases, 25 au plus"),
        # L'ancien plafond : 80 phrases passaient, et ne tenaient plus dans le délai, en série.
        (_phrases_en_groupes(*(("point_cle", i, 8) for i in range(10))), "80 phrases, 25 au plus"),
        (_phrases_en_groupes(("titre", 0, 1), ("contexte", 0, 21)),
         "Le contexte compte 21 phrases, 20 au plus par partie"),
        (_phrases_en_groupes(("titre", 0, 1), *(("explication", i, 1) for i in range(2)), ("explication", 2, 21)),
         "L'explication de la citation 3 compte 21 phrases"),
    ], ids=["vide", "trop", "ancien-plafond", "contexte", "explication"])
    def test_trop_long_dit_de_raccourcir(self, phrases, message):
        with pytest.raises(TexteTropLongError) as erreur:
            lots_du_controle(phrases)

        assert (erreur.value.statut, erreur.value.code) == (422, "entree_invalide")
        assert message in str(erreur.value)


# ==================== 6. LE CONTRÔLE, SUR LA BASE DE TEST ====================


@pytest.fixture
async def textes(db_session):
    return await creer_les_textes(db_session)


async def _article_genere(db_session, textes, llm=None):
    """Un article rédigé par le vrai générateur (modèle et recherche doublés), comme la route l'enregistre."""
    recherche = RechercheDoublee(extraits_du_sujet(textes), similarites(textes))
    article = await BlogGeneratorService(db_session, llm=llm or modele_double(), recherche=recherche).rediger(SUJET)
    return SimpleNamespace(
        title=article.title, context=article.context, summary=article.summary,
        key_points=article.key_points, citizen_impact=article.citizen_impact,
        citations=article.citations, suggested_question=article.suggested_question,
        sources=article.sources,
    )


class TestLeControle:

    async def test_un_appel_sur_les_sources_de_la_generation(self, db_session, textes):
        post = await _article_genere(db_session, textes)
        llm = modele_double()

        controle = await ControleDuBillet(db_session, llm=llm).controler(post)

        (appel,) = appels_du_controle(llm)
        assert appel["temperature"] == 0.0 and appel["response_schema"] == SCHEMA_DU_CONTROLE
        assert appel["system"] == SYSTEM_DU_CONTROLE and appel["response_mime_type"] == "application/json"
        assert f"[1] {TITRE_CODE_PENAL} — Article 299" in appel["prompt"]
        assert "[3]" in appel["prompt"] and "[4]" not in appel["prompt"]
        assert f"Titre de l'article\nP1 : {post.title}" in appel["prompt"]
        assert controle["modele"] == "modele-double"
        assert controle["empreinte"] == empreinte_du_controle(ContenuDuBillet.de(post))
        assert [p["id"] for p in controle["phrases"]] == [i for i, _ in phrases_du_prompt(appel["prompt"])]
        assert controle["phrases"][0]["partie"] == "titre"
        assert {p["verdict"] for p in controle["phrases"]} == {"soutenue"}
        assert set(controle["empreintes_des_parties"]) == set(PARTIES)
        assert controle["extraits_ajoutes"] == []

    async def test_la_transaction_est_close_avant_le_modele(self, db_session, textes):
        post = await _article_genere(db_session, textes)
        vues = []
        double = modele_double()
        repondre = double.generate.side_effect

        async def generer(**kwargs):
            vues.append(db_session.in_transaction())
            return await repondre(**kwargs)

        double.generate = AsyncMock(side_effect=generer)
        await ControleDuBillet(db_session, llm=double).controler(post)

        assert vues == [False]

    @pytest.mark.parametrize("sources", [[], None, [{"n": 1, "extrait": ""}]])
    async def test_sans_sources_aucun_appel(self, db_session, textes, sources):
        """Un article d'avant, ou écrit à la main : ses citations ne suffisent plus (tour 4)."""
        post = await _article_genere(db_session, textes)
        post.sources = sources
        llm = modele_double()

        with pytest.raises(SourcesAbsentesError) as erreur:
            await ControleDuBillet(db_session, llm=llm).controler(post)

        assert (erreur.value.statut, erreur.value.code) == (422, "sources_absentes")
        assert "Reconstruisez d'abord ses sources" in str(erreur.value)
        llm.generate.assert_not_awaited()

    async def test_une_citation_ajoutee_devient_un_extrait_garde(self, db_session, textes):
        post = await _article_genere(db_session, textes)
        (ajoutee,), _ = await citations_verifiees(db_session, [DemandeDeCitation(
            textes.code_penal, article_number="322", title="La filouterie de loyers est punie."
        )])
        post.citations = [*post.citations, ajoutee]
        llm = modele_double()

        controle = await ControleDuBillet(db_session, llm=llm).controler(post)

        prompt = appels_du_controle(llm)[0]["prompt"]
        assert f"[4] {TITRE_CODE_PENAL} — Article 322" in prompt
        assert "Explication de la citation 3, qui accompagne l'extrait 4 (Article 322)" in prompt
        (garde,) = controle["extraits_ajoutes"]
        assert {cle: garde[cle] for cle in ("n", "law_id", "law_title", "article_number", "article_id", "suite")} == {
            "n": 4, "law_id": textes.code_penal, "law_title": TITRE_CODE_PENAL,
            "article_number": "322", "article_id": textes.cp_322, "suite": False,
        }
        assert garde["extrait"].startswith("1.-Filouteries de loyers")
        assert [s["n"] for s in sources_affichees(post.sources, controle)] == [1, 2, 3, 4]

    async def test_un_article_change_en_base_se_juge_sur_le_texte_d_aujourd_hui(self, db_session, textes):
        """L'explication se juge sur ce que la carte montre aujourd'hui, pas sur la source d'hier."""
        post = await _article_genere(db_session, textes)
        await db_session.execute(
            text("UPDATE articles SET content = :c WHERE id = :id"),
            {"c": "Violation de domicile\n\n(1) Est puni d'un emprisonnement de deux (02) ans.",
             "id": textes.cp_299},
        )
        await db_session.commit()
        llm = modele_double()

        controle = await ControleDuBillet(db_session, llm=llm).controler(post)

        prompt = appels_du_controle(llm)[0]["prompt"]
        assert "Explication de la citation 1, qui accompagne l'extrait 4 (Article 299)" in prompt
        assert "deux (02) ans" in prompt
        assert [e["article_number"] for e in controle["extraits_ajoutes"]] == ["299"]

    async def test_un_texte_trop_long(self, db_session, textes):
        post = await _article_genere(db_session, textes)
        post.context = " ".join(f"Phrase {i} du contexte." for i in range(PHRASES_AU_PLUS + 1))
        llm = modele_double()

        with pytest.raises(TexteTropLongError) as erreur:
            await ControleDuBillet(db_session, llm=llm).controler(post)

        assert (erreur.value.statut, erreur.value.code) == (422, "entree_invalide")
        llm.generate.assert_not_awaited()

    async def test_un_article_long_se_juge_en_appels_l_un_apres_l_autre(self, db_session, textes):
        """
        Un modèle SANS budget de jetons (ici la doublure ; ministral-14b, Gemini) : 24 phrases (un
        article généré), deux appels, le second parti après la FIN du premier ; les verdicts
        réunis, dans l'ordre. Le modèle à budget de jetons, lui, juge l'article en UN appel (tour
        7, TestLePlanDuControle).
        """
        post = await _article_genere(db_session, textes, llm=modele_double(article_du_modele(
            context=" ".join(f"Phrase {i} du contexte." for i in range(8)),
            summary=" ".join(f"Phrase {i} du résumé." for i in range(8)),
        )))
        debuts, fins = [], []
        double = modele_double()
        repondre = double.generate.side_effect

        async def generer(**kwargs):
            debuts.append(time.perf_counter())
            await asyncio.sleep(0.2)
            fins.append(time.perf_counter())
            return await repondre(**kwargs)

        double.generate = AsyncMock(side_effect=generer)
        controle = await ControleDuBillet(db_session, llm=double).controler(post)

        appels = appels_du_controle(double)
        total = len(phrases_du_billet(ContenuDuBillet.de(post)))
        assert total == 24 and len(appels) == 2
        # L'un après l'autre : le second ne part qu'une fois le premier fini.
        assert debuts[1] >= fins[0]
        assert [p["id"] for p in controle["phrases"]] == [f"P{i}" for i in range(1, total + 1)]
        assert {p["verdict"] for p in controle["phrases"]} == {"soutenue"}
        par_appel = [len(phrases_du_prompt(appel["prompt"])) for appel in appels]
        assert sum(par_appel) == total and max(par_appel) <= PHRASES_PAR_APPEL
        assert all(
            f"les {total - n} autres phrases de l'article sont jugées à part" in appel["prompt"]
            for appel, n in zip(appels, par_appel)
        )
        # Gardé pour mesurer le contrôle sur les vrais articles ; pas dans l'API.
        assert [a["phrases"] for a in controle["appels"]] == par_appel
        assert all(a["duree_s"] >= 0.2 for a in controle["appels"])
        assert "appels" not in controle_affiche(controle)

    async def test_un_appel_qui_echoue_arrete_le_controle_et_les_suivants_ne_partent_pas(
        self, db_session, textes
    ):
        """Un contrôle à moitié fait n'en est pas un : l'erreur du premier appel, et rien d'autre ne part."""
        post = await _article_genere(db_session, textes, llm=modele_double(article_du_modele(
            context=" ".join(f"Phrase {i} du contexte." for i in range(8)),
            summary=" ".join(f"Phrase {i} du résumé." for i in range(8)),
        )))

        async def generer(**kwargs):
            if "\nP1 : " in kwargs["prompt"]:
                raise MistralOverloadedError("503")
            raise AssertionError("le second appel ne devait pas partir")

        double = modele_double(side_effect=generer)

        with pytest.raises(SaturationError):
            await ControleDuBillet(db_session, llm=double).controler(post)

        assert len(appels_du_controle(double)) == 1

    async def test_un_article_sans_phrase(self, db_session, textes):
        post = await _article_genere(db_session, textes)
        post.title = post.context = post.summary = post.citizen_impact = post.suggested_question = ""
        post.key_points, post.citations = [], []
        llm = modele_double()

        with pytest.raises(TexteTropLongError) as erreur:
            await ControleDuBillet(db_session, llm=llm).controler(post)

        assert "aucune phrase" in str(erreur.value)
        llm.generate.assert_not_awaited()

    @pytest.mark.parametrize("panne, attendue, statut, delai", [
        (MistralQuotaError("Quota mensuel"), QuotaError, 429, 60),
        (GeminiQuotaError("429"), QuotaError, 429, 60),
        (MistralOverloadedError("503"), SaturationError, 503, 10),
        # Tour 6 : le délai que le client connaît (Retry-After, budget de jetons), arrondi au-dessus.
        (MistralOverloadedError("429", retry_after=37.2), SaturationError, 503, 38),
        (GeminiBudgetEpuiseError("réflexion"), ReponseInvalideError, 502, None),
        (MistralServiceError("HTTP 401 Invalid API Key"), ReponseInvalideError, 502, None),
    ])
    async def test_les_erreurs_du_modele_traduites(self, db_session, textes, panne, attendue, statut, delai):
        post = await _article_genere(db_session, textes)

        with pytest.raises(attendue) as leve:
            await ControleDuBillet(db_session, llm=modele_double(controle=panne)).controler(post)

        assert (leve.value.statut, leve.value.retry_after) == (statut, delai)
        assert "contrôle" in str(leve.value) and "Invalid API Key" not in str(leve.value)

    def test_le_nom_du_modele(self, db_session):
        double = modele_double()
        sans_nom = AsyncMock()
        sans_nom.model_name = None

        assert ControleDuBillet(db_session, llm=double).modele == "modele-double"
        assert ControleDuBillet(db_session, llm=sans_nom).modele


# ==================== 7. LE MODÈLE DU CONTRÔLE (TOUR 6) ====================
#
# Décision du propriétaire du 04/11 : mistral-large-4 pour le contrôle seulement. Ces tests
# passent par le VRAI client (MistralService), et par le câblage de production
# (ControleDuBillet(db) sans modèle injecté), sur un httpx.MockTransport : le grand modèle y
# est doublé au niveau HTTP avec ce que le vrai a fait le 04/11 (data/blog/tour5-modeles).
# Aucun appel réseau, aucune vraie attente : les limiteurs sont sur horloge factice.

GRAND = "mistral-large-4"


class _Horloge:
    """L'horloge des limiteurs : leurs sommeils avancent le temps, sans vraie attente."""

    def __init__(self):
        self.t = 0.0
        self.sommeils = []

    def __call__(self):
        return self.t

    def dormir(self, secondes):
        self.sommeils.append(round(secondes, 3))
        self.t += secondes

    async def dormir_async(self, secondes):
        self.dormir(secondes)


def _refus(statut, code, message, entetes=None):
    """Un refus du fournisseur, à la forme des vrais (data/blog/tour5-modeles/large4*/brut-bail.json)."""
    return httpx.Response(statut, headers=entetes or {}, json={
        "object": "error", "message": message, "type": "invalid_request_error", "param": None,
        "code": code,
    })


class GrandModeleDouble:
    """
    mistral-large-4 au niveau HTTP. Il refuse ce que le vrai a refusé le 04/11, avec les
    mêmes codes, avant toute génération : un effort hors de low, medium, high (code 3051),
    une température 0 sans top_p 1 (code 3054). Sinon il répond comme lui : le contenu EN
    MORCEAUX, la réflexion d'abord (avec ses propres morceaux « text »), puis le JSON des
    verdicts (tout « soutenue » : reponse_du_controle) ; l'usage, et l'en-tête des jetons
    restants s'il est donné. `duree_s` avance l'horloge factice (la durée d'un appel) ;
    `refus` : des réponses imposées, dans l'ordre, avant les vraies (un 429…).
    """

    REFLEXION = "Réflexion du modèle sur l'extrait 1, qui n'est pas la réponse."

    def __init__(self, horloge=None, *, servi=GRAND, jetons=14_766, restants=None, duree_s=0.0, refus=()):
        self.horloge = horloge
        self.servi = servi
        self.jetons = jetons
        self.restants = restants
        self.duree_s = duree_s
        self.refus = list(refus)
        self.requetes = []

    def __call__(self, requete: httpx.Request) -> httpx.Response:
        corps = json.loads(requete.content)
        self.requetes.append({"instant": self.horloge() if self.horloge else None, "corps": corps})
        if self.refus:
            return self.refus.pop(0)
        # Les exigences de mistral-large-4 (ses deux identifiants) ; ministral-8b, le secours
        # du chat, accepte la température 0 sans top_p (le contrôle de production l'a fait).
        if corps.get("model") in (GRAND, "mistral-large-4-0"):
            effort = corps.get("reasoning_effort")
            if effort is not None and effort not in ("low", "medium", "high"):
                return _refus(400, "3051", f"reasoning_effort {effort} is not supported for this model, "
                                           "supported values: high, low, medium")
            if corps.get("temperature") == 0 and corps.get("top_p") != 1:
                return _refus(400, "3054", "top_p must be 1 when using greedy sampling.")
        if self.horloge is not None:
            self.horloge.t += self.duree_s
        contenu = [
            {"type": "thinking", "thinking": [{"type": "text", "text": self.REFLEXION}], "closed": True},
            {"type": "text", "text": reponse_du_controle(corps["messages"][-1]["content"])["response"]},
        ]
        entetes = {"x-ratelimit-limit-tokens-minute": "20000", "x-ratelimit-tokens-query-cost": str(self.jetons)}
        if self.restants is not None:
            entetes["x-ratelimit-remaining-tokens-minute"] = str(self.restants)
        entree = self.jetons // 3
        return httpx.Response(200, headers=entetes, json={
            "id": "double", "object": "chat.completion", "model": self.servi,
            "choices": [{
                "index": 0, "finish_reason": "stop",
                "message": {"role": "assistant", "content": contenu, "tool_calls": None},
            }],
            "usage": {"prompt_tokens": entree, "completion_tokens": self.jetons - entree, "total_tokens": self.jetons},
        })

    def modeles(self):
        return [r["corps"]["model"] for r in self.requetes]


@pytest.fixture
def grand_modele(monkeypatch):
    """
    Les réglages de production du contrôle, les limiteurs du processus sur horloge factice
    (ceux que limites_du_modele donne à chaque modèle : 20 000 jetons par minute et une
    requête à la fois pour mistral-large-4), et un modèle de secours posé, pour voir qu'il
    n'est jamais pris. `grand_modele(**reglages)` pose le double HTTP et le rend.
    """
    monkeypatch.setattr(settings, "LLM_PROVIDER", "mistral")
    monkeypatch.setattr(settings, "BLOG_CONTROLE_MODELE", GRAND)
    monkeypatch.setattr(settings, "BLOG_CONTROLE_REFLEXION", "low")
    monkeypatch.setattr(settings, "BLOG_CONTROLE_JETONS_PAR_MINUTE", 20_000)
    monkeypatch.setattr(settings, "MISTRAL_MODEL", "ministral-14b-2512")
    monkeypatch.setattr(settings, "MISTRAL_MODEL_SECOURS", "ministral-8b-2512")
    monkeypatch.setattr(settings, "MISTRAL_ATTENTE_MAX_S", 20.0)
    horloge = _Horloge()
    limiteurs = {
        modele: Limiteur(
            modele, horloge=horloge, dormir=horloge.dormir, dormir_async=horloge.dormir_async,
            **mistral_service.limites_du_modele(modele),
        )
        for modele in (GRAND, "ministral-14b-2512", "ministral-8b-2512")
    }
    monkeypatch.setattr(mistral_service, "_LIMITEURS", limiteurs)

    def poser(**reglages):
        double = GrandModeleDouble(horloge, **reglages)
        monkeypatch.setattr(mistral_service, "transport_http", httpx.MockTransport(double))
        return double

    def au_rythme(rythme="travail", **reglages):
        """Le grand modèle au coût et au rythme mesurés le 04/11 (revue du tour 6)."""
        double = GrandModeleAuRythme(horloge, rythme, **reglages)
        monkeypatch.setattr(mistral_service, "transport_http", httpx.MockTransport(double))
        return double

    poser.horloge = horloge
    poser.limiteurs = limiteurs
    poser.au_rythme = au_rythme
    return poser


async def _article_de_vingt_quatre_phrases(db_session, textes):
    """
    Un article généré de 24 phrases (la taille de ceux de l'essai réel) : deux appels pour un
    modèle sans budget de jetons, un seul pour le modèle du contrôle (tour 7).
    """
    return await _article_genere(db_session, textes, llm=modele_double(article_du_modele(
        context=" ".join(f"Phrase {i} du contexte." for i in range(8)),
        summary=" ".join(f"Phrase {i} du résumé." for i in range(8)),
    )))


class TestLeModeleDuControle:

    def test_les_reglages(self):
        """Le contrôle seul change de modèle ; un effort que le modèle refuse ne démarre pas."""
        defauts = {nom: Settings.model_fields[nom].default for nom in Settings.model_fields}

        assert defauts["BLOG_CONTROLE_MODELE"] == "mistral-large-4"
        assert defauts["BLOG_CONTROLE_REFLEXION"] == "low"
        assert defauts["BLOG_CONTROLE_JETONS_PAR_MINUTE"] == 20_000
        assert defauts["MISTRAL_MODEL"] == "ministral-14b-2512"
        assert Settings(_env_file=None, BLOG_CONTROLE_REFLEXION="high").BLOG_CONTROLE_REFLEXION == "high"
        for refuse in ({"BLOG_CONTROLE_REFLEXION": "none"}, {"BLOG_CONTROLE_JETONS_PAR_MINUTE": -1}):
            with pytest.raises(ValidationError):
                Settings(_env_file=None, **refuse)

    def test_le_controle_prend_son_modele_sans_secours_et_la_generation_le_sien(self, grand_modele, monkeypatch):
        du_chat = SimpleNamespace(model_name="ministral-14b-2512")
        monkeypatch.setattr("app.services.blog_generator_service.get_llm_service", lambda: du_chat)
        controleur = ControleDuBillet(None)
        generateur = BlogGeneratorService(None, recherche=object())
        double = modele_double()

        assert isinstance(controleur.llm, MistralService)
        assert (controleur.llm.model_name, controleur.llm.avec_secours) == (GRAND, False)
        assert controleur.modele == GRAND
        # La génération garde le modèle du chat ; le contrôle qui la suit prend le sien (None :
        # celui de ControleDuBillet), sauf la doublure injectée des tests, qui sert aux deux.
        assert generateur.llm is du_chat and generateur.llm_du_controle is None
        assert BlogGeneratorService(None, llm=double, recherche=object()).llm_du_controle is double
        autre = modele_double()
        assert BlogGeneratorService(None, llm=double, recherche=object(), llm_du_controle=autre).llm_du_controle is autre
        # Le limiteur du modèle du contrôle : son budget de jetons, une requête à la fois.
        assert mistral_service.limites_du_modele(GRAND) == {
            "requetes_par_minute": 30.0, "concurrence": 1, "jetons_par_minute": 20_000,
        }
        assert "jetons_par_minute" not in mistral_service.limites_du_modele("ministral-14b-2512")

    def test_vide_le_modele_du_chat(self, grand_modele, monkeypatch):
        monkeypatch.setattr(settings, "BLOG_CONTROLE_MODELE", " ")

        assert ControleDuBillet(None).llm.model_name == "ministral-14b-2512"
        assert "jetons_par_minute" not in mistral_service.limites_du_modele("ministral-14b-2512")

    @pytest.mark.parametrize("effort", ["low", "medium", "high"])
    async def test_la_requete_porte_ce_que_le_modele_exige(self, grand_modele, db_session, textes, monkeypatch, effort):
        """
        Les deux refus du 04/11 (codes 3051 et 3054) ne se reproduisent plus : reasoning_effort
        accepté, top_p 1 à la température 0, et le schéma strict que la mesure a fait accepter.
        """
        monkeypatch.setattr(settings, "BLOG_CONTROLE_REFLEXION", effort)
        double = grand_modele()
        post = await _article_genere(db_session, textes)

        controle = await ControleDuBillet(db_session).controler(post)

        (requete,) = double.requetes
        corps = requete["corps"]
        assert {cle: corps[cle] for cle in ("model", "reasoning_effort", "top_p", "temperature", "max_tokens")} == {
            "model": GRAND, "reasoning_effort": effort, "top_p": 1.0, "temperature": 0.0,
            "max_tokens": CONTROLE_MAX_TOKENS,
        }
        assert corps["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "reponse", "schema": schema_strict(SCHEMA_DU_CONTROLE), "strict": True},
        }
        assert corps["messages"][0] == {"role": "system", "content": SYSTEM_DU_CONTROLE}
        assert "stream" not in corps
        assert {p["verdict"] for p in controle["phrases"]} == {"soutenue"}

    async def test_un_effort_que_le_modele_refuse_part_en_low(self, grand_modele):
        """« none » (refusé, code 3051) ou le « minimal » de Gemini : le plus bas qu'il accepte."""
        double = grand_modele(jetons=100)
        client = mistral_service.mistral_du_controle()

        for demande in ("none", "minimal", None):
            await client.generate("P1 : Une phrase.", system="s", temperature=0, max_tokens=50, reflexion=demande)

        assert [r["corps"]["reasoning_effort"] for r in double.requetes] == ["low", "low", "low"]

    async def test_une_reponse_en_morceaux_se_lit_sans_sa_reflexion(self, grand_modele, db_session, textes):
        """Avant le tour 6, le client passait la liste telle quelle et le contrôle n'y lisait rien (502)."""
        grand_modele()
        post = await _article_genere(db_session, textes)

        controle = await ControleDuBillet(db_session).controler(post)

        assert controle["phrases"] and {p["verdict"] for p in controle["phrases"]} == {"soutenue"}
        assert GrandModeleDouble.REFLEXION not in json.dumps(controle, ensure_ascii=False)

    async def test_le_client_rend_le_texte_et_le_modele_qui_a_repondu(self, grand_modele):
        grand_modele(servi="mistral-large-4-0", jetons=100)
        client = mistral_service.mistral_du_controle()

        reponse = await client.generate(
            "P1 : Une phrase.", system="s", temperature=0, max_tokens=50,
            response_schema=SCHEMA_DU_CONTROLE, rendre_le_modele=True,
        )
        sans_modele = await client.generate("P1 : Une phrase.", system="s", temperature=0, max_tokens=50)

        assert reponse == {
            "response": reponse_du_controle("P1 : Une phrase.")["response"], "fin": "STOP",
            "modele": "mistral-large-4-0",
        }
        assert "modele" not in sans_modele

    def test_le_texte_des_morceaux(self):
        reflexion = {"type": "thinking", "thinking": [{"type": "text", "text": "NON"}], "closed": True}

        assert texte_des_morceaux("Un texte.") == "Un texte."
        assert texte_des_morceaux(None) == ""
        assert texte_des_morceaux([reflexion, {"type": "text", "text": '{"a": 1}'}]) == '{"a": 1}'
        assert texte_des_morceaux([{"type": "text", "text": "Un "}, reflexion, {"type": "text", "text": "texte."}]) == "Un texte."
        assert texte_des_morceaux([{"type": "image_url", "image_url": "x"}, "brut", {"type": "text", "text": None}]) == "brut"

    @pytest.mark.parametrize("cas", ["effort-refuse-3051", "top-p-absent-3054"])
    async def test_un_refus_400_est_une_erreur_claire_sans_nouvel_essai_ni_autre_modele(
        self, grand_modele, db_session, textes, monkeypatch, cas
    ):
        """
        Les deux requêtes refusées le 04/11, reproduites : reasoning_effort « none »
        (data/blog/tour5-modeles/large4/), puis la requête de production d'avant le tour 6,
        sans top_p (large4-low-bail/). Une seule requête, aucun secours, rien d'enregistrable.
        """
        if cas == "effort-refuse-3051":
            monkeypatch.setattr(settings, "BLOG_CONTROLE_REFLEXION", "none")
            monkeypatch.setattr(mistral_service, "EFFORTS_DE_RAISONNEMENT", ("none", "low", "medium", "high"))
        else:
            monkeypatch.setattr(mistral_service, "MODELES_A_RAISONNEMENT", frozenset())
        double = grand_modele()
        post = await _article_genere(db_session, textes)

        with pytest.raises(ReponseInvalideError) as erreur:
            await ControleDuBillet(db_session).controler(post)

        assert double.modeles() == [GRAND]
        assert (erreur.value.statut, erreur.value.code) == (502, "reponse_invalide")
        assert "contrôle IA n'a pas été fait" in str(erreur.value) and GRAND in str(erreur.value)
        assert "not supported" not in str(erreur.value) and "top_p" not in str(erreur.value)

    @pytest.mark.parametrize("retry_after, delai", [("25", 25), (None, None)])
    async def test_un_429_ne_passe_jamais_a_un_autre_modele(
        self, grand_modele, db_session, textes, retry_after, delai
    ):
        """
        Revue du tour 5 : sur un 429, le client passait au secours (ministral-8b), et
        controle_ia aurait dit le nom du modèle demandé. Le client du contrôle n'a pas de
        secours : 503 « saturation », avec le délai du fournisseur s'il le donne.
        Revue du tour 6 : il rejoue le 429 tant que le plan du contrôle finit dans le délai (la
        marge du plan, et non plus les 20 s du chat), ESSAIS envois au plus. Tour 7 : son appel
        part au plus tard DUREE_D_UN_APPEL_AU_PIRE_S et MARGE_DU_PLAN_S avant l'échéance (30 s
        d'attente permise sur 120) : au Retry-After de 25 s, deux envois.
        """
        entetes = {"retry-after": retry_after} if retry_after else {}
        double = grand_modele(refus=[_refus(429, "1300", "Rate limit exceeded", entetes)] * 4)
        post = await _article_genere(db_session, textes)
        # Un seul appel : l'attente permise est ce qui reste du délai après sa durée au pire
        # mesuré et sa marge.
        marge = DELAI_CONTROLE_S - DUREE_D_UN_APPEL_AU_PIRE_S - MARGE_DU_PLAN_S
        essais = min(mistral_service.ESSAIS, math.floor(marge / 25) + 1) if retry_after else mistral_service.ESSAIS

        with pytest.raises(SaturationError) as erreur:
            await ControleDuBillet(db_session).controler(post)

        assert double.modeles() == [GRAND] * essais
        if retry_after:
            assert essais == 2 and [r["instant"] for r in double.requetes] == [0.0, 25.0]
        assert (erreur.value.statut, erreur.value.code) == (503, "saturation")
        assert GRAND in str(erreur.value)
        if delai:
            assert erreur.value.retry_after == delai and f"dans {delai} s" in str(erreur.value)
        else:
            assert erreur.value.retry_after >= 1
        # Le même 429, pour un client AVEC secours (celui du chat), une fois la pénalité passée :
        # il y passerait. C'est le secours retiré qui garde le contrôle au grand modèle.
        grand_modele.horloge.t += 61
        avec_secours = MistralService(api_key="cle-test", model_name=GRAND, limiteur_pour=grand_modele.limiteurs.get)
        double.refus = [_refus(429, "1300", "Rate limit exceeded", {"retry-after": "30"})]
        double.requetes.clear()
        await avec_secours.generate("P1 : Une phrase.", system="s", temperature=0, max_tokens=50)
        assert double.modeles() == [GRAND, "ministral-8b-2512"]

    async def test_le_modele_qui_a_repondu_est_note(self, grand_modele, db_session, textes):
        grand_modele(servi="mistral-large-4-0")
        post = await _article_genere(db_session, textes)

        controle = await ControleDuBillet(db_session).controler(post)

        assert controle["modele"] == "mistral-large-4-0"
        assert controle_affiche(controle)["modele"] == "mistral-large-4-0"

    async def test_des_lots_rendus_par_deux_modeles_les_nomment_tous_les_deux(self, db_session, textes):
        post = await _article_de_vingt_quatre_phrases(db_session, textes)
        servis = iter(["mistral-large-4", "mistral-large-4-0"])

        def repondre(prompt):
            return {**reponse_du_controle(prompt), "modele": next(servis)}

        controle = await ControleDuBillet(db_session, llm=modele_double(controle=repondre)).controler(post)

        assert len(controle["appels"]) == 2
        assert controle["modele"] == "mistral-large-4 + mistral-large-4-0"

    async def test_un_article_de_vingt_quatre_phrases_en_un_appel_dans_le_budget_de_jetons(
        self, grand_modele, db_session, textes
    ):
        """
        Tour 7 : 24 phrases (la taille des articles de l'essai réel), UN appel. Il coûte 19 000
        jetons et dure 45 s : le contrôle suivant, dont l'entrée ne tient plus dans la minute,
        attend qu'il en sorte (60 s après son départ), dans la marge de son plan, puis part.
        """
        double = grand_modele(jetons=19_000, duree_s=45.0)
        post = await _article_de_vingt_quatre_phrases(db_session, textes)

        premier = await ControleDuBillet(db_session).controler(post)
        second = await ControleDuBillet(db_session).controler(post)

        assert [r["instant"] for r in double.requetes] == [0.0, 60.0]
        assert 15.0 in grand_modele.horloge.sommeils
        for controle in (premier, second):
            assert [a["phrases"] for a in controle["appels"]] == [24] and controle["modele"] == GRAND
            assert len(controle["phrases"]) == 24
        assert "autres phrases de l'article sont jugées à part" not in double.requetes[0]["corps"]["messages"][-1]["content"]

    async def test_le_budget_epuise_chez_le_fournisseur_fait_attendre_le_controle_suivant(
        self, grand_modele, db_session, textes
    ):
        """
        Le fournisseur compte peut-être plus longtemps qu'une minute (04/11 : 0 jeton restant
        après le contrôle du travail) : l'en-tête recale le limiteur, et le contrôle suivant,
        lancé à la fin du premier (5 s), doit attendre que la minute se vide : 55 s. Il ne
        tiendrait pas (55 s, puis son appel au pire et sa marge : 145 s pour 120) : refusé avant
        tout envoi, Retry-After 25. Relancé alors, il attend les 30 s qui restent, dans la marge
        de son plan, et aboutit.
        """
        double = grand_modele(jetons=12_000, duree_s=5.0, restants=0)
        post = await _article_de_vingt_quatre_phrases(db_session, textes)
        await ControleDuBillet(db_session).controler(post)

        with pytest.raises(BudgetDuControleError) as refus:
            await ControleDuBillet(db_session).controler(post)
        assert refus.value.retry_after == 25 and len(double.requetes) == 1
        grand_modele.horloge.t += 25
        controle = await ControleDuBillet(db_session).controler(post)

        assert [r["instant"] for r in double.requetes] == [0.0, 60.0]
        assert 30.0 in grand_modele.horloge.sommeils
        assert len(controle["phrases"]) == 24 and controle["modele"] == GRAND

    async def test_deux_controles_en_meme_temps_passent_l_un_apres_l_autre(self, db_session, textes):
        post = await _article_genere(db_session, textes)
        appels, premier_parti, laisser_finir = [], asyncio.Event(), asyncio.Event()

        def double_qui_note(nom, attendre=None):
            double = modele_double()
            repondre = double.generate.side_effect

            async def generer(**kwargs):
                appels.append((nom, "début"))
                premier_parti.set()
                if attendre is not None:
                    await attendre.wait()
                appels.append((nom, "fin"))
                return await repondre(**kwargs)

            double.generate = AsyncMock(side_effect=generer)
            return double

        premier = asyncio.create_task(
            ControleDuBillet(db_session, llm=double_qui_note("A", laisser_finir)).controler(post)
        )
        await premier_parti.wait()
        # Le second, sur le même modèle (« modele-double »), part pendant que le premier juge.
        second = asyncio.create_task(ControleDuBillet(db_session, llm=double_qui_note("B")).controler(post))
        await asyncio.sleep(0.05)
        assert appels == [("A", "début")]
        laisser_finir.set()

        resultats = await asyncio.gather(premier, second)

        assert appels == [("A", "début"), ("A", "fin"), ("B", "début"), ("B", "fin")]
        assert all(r["phrases"] for r in resultats)

    async def test_un_controle_qui_attend_trop_un_autre_dit_de_relancer(self, db_session, textes, monkeypatch):
        monkeypatch.setattr(controle_du_billet, "ATTENTE_D_UN_AUTRE_CONTROLE_S", 0.05)
        post = await _article_genere(db_session, textes)
        parti, laisser_finir = asyncio.Event(), asyncio.Event()
        occupe = modele_double()
        repondre = occupe.generate.side_effect

        async def lent(**kwargs):
            parti.set()
            await laisser_finir.wait()
            return await repondre(**kwargs)

        occupe.generate = AsyncMock(side_effect=lent)
        premier = asyncio.create_task(ControleDuBillet(db_session, llm=occupe).controler(post))
        await parti.wait()
        second = modele_double()

        with pytest.raises(SaturationError) as erreur:
            await ControleDuBillet(db_session, llm=second).controler(post)

        assert (erreur.value.statut, erreur.value.retry_after) == (503, 60)
        assert "Un autre contrôle IA est en cours" in str(erreur.value)
        second.generate.assert_not_awaited()
        laisser_finir.set()
        assert (await premier)["phrases"]

    async def test_le_verrou_est_rendu_quand_le_controle_est_coupe(self, db_session, textes, monkeypatch):
        """L'échéance de la route annule le contrôle pendant l'appel : le suivant n'attend pas."""
        monkeypatch.setattr(controle_du_billet, "ATTENTE_D_UN_AUTRE_CONTROLE_S", 0.05)
        post = await _article_genere(db_session, textes)
        parti = asyncio.Event()

        async def interminable(**_):
            parti.set()
            await asyncio.sleep(5)

        coupe = asyncio.create_task(
            ControleDuBillet(db_session, llm=modele_double(side_effect=interminable)).controler(post)
        )
        await parti.wait()
        coupe.cancel()
        with pytest.raises(asyncio.CancelledError):
            await coupe

        assert (await ControleDuBillet(db_session, llm=modele_double()).controler(post))["phrases"]


# ==================== 7 bis. LE PLAN DU CONTRÔLE (REVUE DU TOUR 6) ====================


class TestLePlanDuControle:
    """
    Revue du tour 6 (data/blog/tour6/revue-large4), puis son essai réel : le vrai client
    (MistralService), les limiteurs de production sur horloge factice, mistral-large-4 doublé au
    coût et à la durée MESURÉS d'un appel (GrandModeleAuRythme, APPELS_MESURES : « pire », les
    23 phrases du bail en un appel, 78,7 s et 10 720 jetons de sortie ; « vite », 40,5 s),
    l'article RÉEL du travail (article_reel).
    TOUR 7 : l'article se juge en UN appel, compté au pire mesuré d'un appel. En deux lots,
    chacun payait la réflexion du modèle : 100,9 et 106,4 s à l'essai réel, puis un 504 à
    120 s après deux lots payés (un lot de 12 phrases, prévu à 41 s, en a pris 76).
    """

    @pytest.mark.parametrize("caracteres", [None, 18_000, 24_000, 36_000])
    @pytest.mark.parametrize("appel", ["pire", "lot-de-12", "vite"])
    async def test_l_article_reel_en_un_seul_appel_dans_le_delai(
        self, grand_modele, db_session, caracteres, appel
    ):
        """Le cas du 504 de l'essai réel : un appel, quelle que soit la taille des extraits."""
        double = grand_modele.au_rythme(appel)

        controle = await ControleDuBillet(db_session).controler(article_reel(caracteres))

        (requete,) = double.requetes
        assert (requete["phrases"], requete["depart"], requete["un_seul_appel"]) == (22, 0.0, True)
        assert grand_modele.horloge.t == pytest.approx(APPELS_MESURES[appel][1])
        assert grand_modele.horloge.t <= DELAI_CONTROLE_S - MARGE_DU_PLAN_S
        assert len(controle["phrases"]) == 22 and controle["modele"] == GRAND
        assert [a["phrases"] for a in controle["appels"]] == [22]

    async def test_les_lots_qui_ne_tenaient_pas_au_tour_6_se_jugent_en_un_appel(self, grand_modele, db_session):
        """
        Au tour 6, cet article (un contexte de 4 phrases, un résumé de 20, 24 000 caractères
        d'extraits) se jugeait en lots 5 + 20, qui ne tenaient pas même seuls : 422
        « raccourcissez-le ». En un appel, au plafond, il se juge dans le délai. Au-delà du
        plafond, toujours 422, et rien ne part.
        """
        double = grand_modele.au_rythme("pire")
        post = article_reel(24_000)
        post.context = " ".join(f"Phrase {i} du contexte." for i in range(4))
        post.summary = " ".join(f"Phrase {i} du résumé." for i in range(20))
        post.key_points, post.citations = [], []
        post.citizen_impact = post.suggested_question = ""
        assert len(phrases_du_billet(ContenuDuBillet.de(post))) == PHRASES_AU_PLUS
        assert [len(lot) for lot in lots_du_controle(phrases_du_billet(ContenuDuBillet.de(post)))] == [5, 20]

        controle = await ControleDuBillet(db_session).controler(post)

        assert [r["phrases"] for r in double.requetes] == [PHRASES_AU_PLUS]
        assert len(controle["phrases"]) == PHRASES_AU_PLUS
        post.summary += " Une phrase de trop."
        with pytest.raises(TexteTropLongError) as erreur:
            await ControleDuBillet(db_session).controler(post)
        assert (erreur.value.statut, erreur.value.code) == (422, "entree_invalide")
        assert "raccourcissez-le" in str(erreur.value) and len(double.requetes) == 1

    @pytest.mark.parametrize("cas", ["fournisseur-a-zero", "juste-apres-un-autre"])
    async def test_un_plan_qui_ne_tient_pas_est_refuse_avant_tout_envoi_et_la_relance_aboutit(
        self, grand_modele, db_session, cas
    ):
        """
        La minute de jetons pleine : le contrôle devrait attendre 60 s, puis son appel au pire
        mesuré et sa marge, soit 150 s pour 120. Refusé AVANT tout envoi : 503, un message sur
        le budget de jetons de l'article (pas « saturé »), et le Retry-After où il tiendra
        (30 s). Relancé alors, il aboutit dans le délai, marge comprise.
        - fournisseur-a-zero : un appel d'ailleurs sur la même clé, le fournisseur dit 0 jeton ;
        - juste-apres-un-autre : le fournisseur dit 0 jeton à la fin d'un contrôle (le 04/11,
          après celui du travail), et le suivant est lancé aussitôt.
        """
        horloge = grand_modele.horloge
        double = grand_modele.au_rythme("pire", restants=0 if cas == "juste-apres-un-autre" else None)
        post = article_reel(18_000)
        if cas == "fournisseur-a-zero":
            grand_modele.limiteurs[GRAND].recaler(jetons_restants_minute=0)
        else:
            assert (await ControleDuBillet(db_session).controler(post))["phrases"]
        avant, debut = len(double.requetes), horloge.t

        with pytest.raises(BudgetDuControleError) as refus:
            await ControleDuBillet(db_session).controler(post)

        message = str(refus.value)
        assert len(double.requetes) == avant
        assert (refus.value.statut, refus.value.code) == (503, "saturation")
        assert "20 000 jetons par minute" in message and "saturé" not in message
        assert "un appel au modèle du contrôle" in message
        assert "Rien n'a été envoyé au modèle" in message and "l'article est inchangé" in message
        assert refus.value.retry_after == 30 and message.endswith("Relancez le contrôle dans 30 s.")

        horloge.t = debut + refus.value.retry_after
        controle = await ControleDuBillet(db_session).controler(post)

        assert controle["phrases"] and len(double.requetes) == avant + 1
        assert double.requetes[-1]["depart"] == pytest.approx(debut + 60.0, abs=0.1)
        assert horloge.t - (debut + refus.value.retry_after) <= DELAI_CONTROLE_S - MARGE_DU_PLAN_S + 1e-6

    @pytest.mark.parametrize("delai, tient", [(40, False), (149, False), (150, True), (180, True)])
    async def test_ce_que_generate_laisse_au_controle(self, grand_modele, db_session, delai, tient):
        """
        Après la génération, le contrôle n'a que ce qui reste (routes/blog,
        delai_du_controle_apres_la_generation). Son appel au pire et sa marge (150 s) y
        tiennent, ou il est refusé avant tout envoi, au lieu d'un 504 après un appel payé ;
        relancé à la demande (180 s), il tient tout de suite : pas de Retry-After.
        """
        double = grand_modele.au_rythme("pire")
        post = article_reel(18_000)

        if tient:
            assert (await ControleDuBillet(db_session).controler(post, delai=delai))["phrases"]
            assert len(double.requetes) == 1 and grand_modele.horloge.t <= delai
        else:
            with pytest.raises(BudgetDuControleError) as refus:
                await ControleDuBillet(db_session).controler(post, delai=delai)
            assert double.requetes == [] and refus.value.retry_after is None
            assert f"il en reste {delai}" in str(refus.value)
            assert str(refus.value).endswith("Relancez le contrôle.")
            assert (await ControleDuBillet(db_session).controler(post))["phrases"]
            assert len(double.requetes) == 1

    async def test_l_appel_ne_part_jamais_sans_sa_marge(self, grand_modele, db_session, textes):
        """
        Un 429 du fournisseur pendant le contrôle : le nouvel essai attend son Retry-After tant
        que l'appel peut encore finir MARGE_DU_PLAN_S avant l'échéance, au pire mesuré (30 s
        d'attente sur 120). Au-delà (35 s), il ne part pas : 503 avec ce délai, un seul envoi
        payé (refusé, donc rien). Avant le tour 7, l'attente permise allait jusqu'à l'échéance
        elle-même, et un appel parti à 40 s pouvait finir après les 120 s.
        """
        refus_429 = _refus(429, "1300", "Rate limit exceeded", {"retry-after": "35"})
        double = grand_modele(refus=[refus_429])
        post = await _article_genere(db_session, textes)

        with pytest.raises(SaturationError) as sature:
            await ControleDuBillet(db_session).controler(post)

        assert [r["instant"] for r in double.requetes] == [0.0]
        assert (sature.value.statut, sature.value.retry_after) == (503, 35)

    @pytest.mark.parametrize("restants", [None, 0], ids=["minute-videe", "fournisseur-a-zero"])
    @pytest.mark.parametrize("avant_la_fin_s", [20.0, 10.0, 1.0])
    async def test_un_controle_lance_pendant_la_fin_d_un_autre(
        self, grand_modele, db_session, restants, avant_la_fin_s
    ):
        """
        Le cas de la revue du tour 6 (repro-3) : lancé pendant la fin d'un autre, un contrôle de
        deux lots dépassait le délai (504, deux lots payés), puis la file le refusait. En UN
        appel, il tient APRÈS l'autre : la file le garde, il part quand l'autre finit, les jetons
        de l'autre sortis de la minute, et finit dans son délai. Si le fournisseur dit 0 jeton à
        la fin de l'autre, le plan refait sur l'état réel ne tient plus : refusé, sans rien
        envoyer, et relancé au Retry-After, il aboutit.
        """
        horloge = grand_modele.horloge
        fin_du_premier = APPELS_MESURES["plafond"][1]
        double = grand_modele.au_rythme("plafond", restants=restants, pas_s=0.5, pause_a=fin_du_premier - avant_la_fin_s)
        post = article_reel(24_000, au_plafond=True)
        premier = asyncio.create_task(ControleDuBillet(db_session).controler(post))
        await double.en_pause.wait()
        lance_a = horloge.t

        second = asyncio.create_task(ControleDuBillet(db_session).controler(post))
        await asyncio.wait({second}, timeout=0.5)
        garde_dans_la_file = not second.done()
        double.reprendre.set()
        resultat_du_premier, resultat = await asyncio.gather(premier, second, return_exceptions=True)

        assert garde_dans_la_file and resultat_du_premier["phrases"]
        if restants is None:
            assert resultat["phrases"]
            assert [r["phrases"] for r in double.requetes] == [PHRASES_AU_PLUS, PHRASES_AU_PLUS]
            assert double.requetes[1]["depart"] >= double.requetes[0]["fin"]
            assert horloge.t - lance_a <= DELAI_CONTROLE_S - MARGE_DU_PLAN_S + 1e-6
        else:
            assert isinstance(resultat, BudgetDuControleError) and len(double.requetes) == 1
            assert "Rien n'a été envoyé au modèle" in str(resultat)
            refuse_a = horloge.t
            horloge.t = refuse_a + resultat.retry_after
            assert (await ControleDuBillet(db_session).controler(post))["phrases"]
            assert len(double.requetes) == 2
            assert horloge.t - (refuse_a + resultat.retry_after) <= DELAI_CONTROLE_S - MARGE_DU_PLAN_S + 1e-6

    @pytest.mark.parametrize("en_retard", [False, True], ids=["dans-son-plan", "au-dela-de-son-plan"])
    async def test_lasse_d_attendre_l_autre_il_dit_quand_relancer_d_apres_son_plan(
        self, grand_modele, db_session, monkeypatch, en_retard
    ):
        """
        Un contrôle que la file garde, mais que l'autre fait attendre plus que son plan ne le
        permet : refusé sans rien envoyer, avec le Retry-After que donne le plan de l'autre (sa
        fin prévue, un appel au pire mesuré après son départ). Si l'autre a dépassé son propre
        plan (plus lent que le pire mesuré), sa fin n'est plus connue : la minute d'avant, et le
        message le dit.
        """
        monkeypatch.setattr(controle_du_billet, "ATTENTE_D_UN_AUTRE_CONTROLE_S", 0.05)
        double = grand_modele.au_rythme("pire", pas_s=0.5, pause_a=60.0)
        post = article_reel()
        premier = asyncio.create_task(ControleDuBillet(db_session).controler(post))
        await double.en_pause.wait()
        if en_retard:
            # Le premier devait finir à 140 s au pire ; il juge encore à 145 s.
            grand_modele.horloge.t = 145.0

        with pytest.raises(BudgetDuControleError) as refus:
            await ControleDuBillet(db_session).controler(post)

        double.reprendre.set()
        assert (await premier)["phrases"] and len(double.requetes) == 1
        message = str(refus.value)
        assert "Un autre contrôle IA est en cours" in message and "rien n'a été envoyé" in message
        assert message.endswith(f"Relancez le contrôle dans {refus.value.retry_after} s.")
        if en_retard:
            assert refus.value.retry_after == 60 and "il dure plus que prévu" in message
        else:
            assert refus.value.retry_after == 80 and "sa fin est prévue dans ~80 s" in message

    async def test_le_client_attend_ce_que_le_controle_lui_permet(self, grand_modele):
        """
        Le budget de jetons plein (le fournisseur dit 0 jeton restant) : sans attente_max, le
        client n'attend que les 20 s du chat, et rien ne part ; avec celle du plan, il attend
        que la minute se vide (60 s), puis envoie.
        """
        double = grand_modele(jetons=100)
        client = mistral_service.mistral_du_controle()
        grand_modele.limiteurs[GRAND].recaler(jetons_restants_minute=0)

        with pytest.raises(MistralOverloadedError) as sature:
            await client.generate("P1 : Une phrase.", system="s", temperature=0, max_tokens=50)
        assert double.requetes == [] and sature.value.retry_after == pytest.approx(60.0)

        await client.generate("P1 : Une phrase.", system="s", temperature=0, max_tokens=50, attente_max=61)

        assert [r["instant"] for r in double.requetes] == [60.0]

    def test_un_plan_de_plusieurs_appels_qui_ne_tient_jamais_est_a_raccourcir(self, grand_modele):
        """
        Le plan garde sa forme générale (des appels qui se suivent) : deux appels au pire mesuré,
        dont le second attend que le premier sorte de la minute, ne tiendraient jamais dans le
        délai, même seuls : 422. Le modèle à budget de jetons ne fait plus qu'un appel (tour 7),
        qui tient toujours seul (90 s pour 120).
        """
        limiteur = grand_modele.limiteurs[GRAND]
        deux = BudgetDuControle(GRAND, limiteur, ((12, 9_000), (13, 9_000)), DELAI_CONTROLE_S, 24_000, 25)
        un = BudgetDuControle(GRAND, limiteur, ((25, 9_000),), DELAI_CONTROLE_S, 24_000, 25)

        with pytest.raises(TexteTropLongError) as erreur:
            deux.verifier_seul()
        un.verifier_seul()

        assert "Raccourcissez-le" in str(erreur.value) and "20 000 jetons par minute" in str(erreur.value)


class TestLeChatEtLaGenerationNeChangentPas:
    """
    Leurs requêtes restent celles d'avant le tour 6 (ministral-14b) : aucun reasoning_effort,
    aucun top_p, les mêmes champs ; leur réponse, la même forme ; leur secours, le même corps.
    """

    @pytest.fixture
    def ministral(self, monkeypatch):
        monkeypatch.setattr(settings, "MISTRAL_MODEL", "ministral-14b-2512")
        monkeypatch.setattr(settings, "MISTRAL_MODEL_SECOURS", "ministral-8b-2512")
        monkeypatch.setattr(settings, "MISTRAL_ATTENTE_MAX_S", 20.0)
        monkeypatch.setattr(settings, "BLOG_CONTROLE_MODELE", GRAND)
        horloge = _Horloge()
        limiteurs = {}

        def limiteur_pour(modele):
            if modele not in limiteurs:
                limiteurs[modele] = Limiteur(
                    modele, horloge=horloge, dormir=horloge.dormir, dormir_async=horloge.dormir_async,
                    **mistral_service.limites_du_modele(modele),
                )
            return limiteurs[modele]

        requetes, imposees = [], []

        def repondre(requete):
            corps = json.loads(requete.content)
            requetes.append(corps)
            if imposees:
                return imposees.pop(0)
            if corps.get("stream"):
                fin = {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 9, "completion_tokens": 2}}
                debut = {"choices": [{"delta": {"content": "Bonjour."}, "finish_reason": None}]}
                return httpx.Response(200, content=(
                    f"data: {json.dumps(debut)}\n\ndata: {json.dumps(fin)}\n\ndata: [DONE]\n\n"
                ).encode())
            return httpx.Response(200, json={
                "model": corps["model"], "choices": [{"message": {"content": '{"ok": true}'}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 2, "total_tokens": 11},
                # Les en-têtes du vrai ministral-14b : sans effet sur son limiteur, sans budget.
            }, headers={"x-ratelimit-remaining-tokens-minute": "937494"})

        monkeypatch.setattr(mistral_service, "transport_http", httpx.MockTransport(repondre))
        service = MistralService(api_key="cle-test", limiteur_pour=limiteur_pour)
        return SimpleNamespace(service=service, requetes=requetes, imposees=imposees, horloge=horloge)

    @staticmethod
    def _corps_d_avant(system, prompt, temperature, max_tokens, modele="ministral-14b-2512", **suite):
        """Le corps que _corps construisait avant le tour 6."""
        return {
            "model": modele,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "temperature": temperature, "max_tokens": max_tokens, **suite,
        }

    async def test_la_generation_du_blog(self, ministral):
        generateur = BlogGeneratorService(None, llm=ministral.service, recherche=object())

        reponse = await generateur._generer("Le prompt de l'article.")

        assert ministral.requetes == [self._corps_d_avant(
            SYSTEM_PROMPT, "Le prompt de l'article.", GENERATION_TEMPERATURE, GENERATION_MAX_TOKENS,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "reponse", "schema": schema_strict(SCHEMA_DE_L_ARTICLE), "strict": True},
            },
        )]
        assert reponse == {"response": '{"ok": true}', "fin": "STOP"}

    async def test_la_generation_passee_au_secours_garde_son_corps(self, ministral):
        ministral.imposees.append(_refus(429, "1300", "Rate limit exceeded", {"retry-after": "30"}))
        generateur = BlogGeneratorService(None, llm=ministral.service, recherche=object())

        await generateur._generer("Le prompt de l'article.")

        premier, secours = ministral.requetes
        assert secours == {**premier, "model": "ministral-8b-2512"}
        assert list(secours) == list(premier)

    @pytest.mark.parametrize("reflexion", [None, "low", "minimal"])
    async def test_le_chat_hors_flux_et_en_flux(self, ministral, reflexion):
        reponse = await ministral.service.generate(
            prompt="Ma question.", system="Le système du chat.", temperature=0.7, max_tokens=2048,
            reflexion=reflexion,
        )
        morceaux = [m async for m in ministral.service.generate_stream(
            prompt="Ma question.", system="Le système du chat.", temperature=0.7, max_tokens=8192,
            reflexion=reflexion,
        )]

        assert ministral.requetes == [
            self._corps_d_avant("Le système du chat.", "Ma question.", 0.7, 2048),
            self._corps_d_avant("Le système du chat.", "Ma question.", 0.7, 8192, stream=True),
        ]
        assert reponse == {"response": '{"ok": true}', "fin": "STOP"}
        assert morceaux == ["Bonjour."]
