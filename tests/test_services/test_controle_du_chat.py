"""
Le controle du chat (app/services/controle_du_chat.py) et ses questions de
reference (app/services/questions_de_reference.py).

Sans base : la forme des questions et leur accord avec le front, la lecture
d'un article dans un morceau, le rappel, le verdict, la sonde du fournisseur
(deux essais), la regle du repli sur l'index, les questions garanties et les
faiblesses connues (seules les premieres font le 503), et le rapport garde
(un calcul par periode, attente bornee, erreurs nommees par leur seule
classe).

Sur la base de test : la recuperation du chat SANS TRACE, sur un petit corpus
ou le volet semantique sait, puis se perd comme le 29/10/2026 : le controle
doit le voir, meme quand le plein texte sauve le contexte ; et un volet en
panne une fois n'est pas un echec.
"""

import asyncio
import os
import re
from datetime import date
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.schemas.search import ChunkResult
from app.services import controle_du_chat as controle
from app.services import search_service
from app.services.controle_du_chat import (
    ContexteDuChat,
    ControleEnCache,
    Verdict,
    contexte_du_chat,
    controler_le_chat,
    juger,
    porte_l_article,
    rappel_a_k,
    sonder_le_fournisseur,
    verifier_une_question,
)
from app.services.embedding_service import EmbeddingService
from app.services.questions_de_reference import (
    FAIBLESSES_CONNUES,
    FOND,
    QUESTIONS_DE_REFERENCE,
    SUGGESTION,
    QuestionDeReference,
)

RACINE = Path(__file__).resolve().parents[2]
QUESTION_DE_L_INCIDENT = "Quelles indemnités en cas de licenciement abusif ?"
INDEMNITES = (("LOI-1992-007", "36"), ("LOI-1992-007", "37"), ("LOI-1992-007", "39"))
# Le morceau 35.1 du Code du travail, tel que la base le porte : la fin de
# l'article 35, puis les en-tetes des articles 36 a 39.
MORCEAU_35_1 = (
    "(1) Pendant la durée du préavis, l'employeur et le travailleur sont tenus au respect "
    "de toutes les obligations réciproques. Article 36.- (1) Toute rupture de contrat à "
    "durée indéterminée, sans préavis... Article 37.- (1) En cas de rupture de contrat... "
    "Article 38.- Le contrat de travail à durée déterminée... Article 39.- (1) Toute "
    "rupture abusive du contrat peut donner lieu à des dommages-intérêts."
)
# Le morceau suivant : la fin de l'article 39, puis l'article 40.
SUITE_35_2 = (
    "(6) Le salaire à prendre en considération aux alinéas précédents est le salaire moyen "
    "mensuel brut... Article 40.- (1) Les dispositions de l'article 34 alinéa (1)"
)


def _question(textes=(("LOI-1992-007",),), rappel=False, question=QUESTION_DE_L_INCIDENT,
              articles_attendus=(), article_nomme=None, toleres_avant=(), regles=(),
              renvoi=(), faiblesse=None):
    return QuestionDeReference(
        question=question, langue="fr", textes=textes, origine=FOND,
        articles="art. 37 et 39", articles_attendus=articles_attendus, rappel=rappel,
        article_nomme=article_nomme, toleres_avant=toleres_avant, regles=regles,
        renvoi=renvoi, faiblesse=faiblesse,
    )


def _de_reference(question: str) -> QuestionDeReference:
    return next(q for q in QUESTIONS_DE_REFERENCE if q.question == question)


# Des morceaux tels que la base les porte (jurix_dev, relus le 30/10/2026),
# abreges : les mots de la regle y sont, au mot pres.
ARTICLE_318 = (
    "Vol, abus de confiance, escroquerie\n\n(1) Est puni d'un emprisonnement de cinq (5) à dix "
    "(10) ans et d'une amende de cent mille (100 000) à un million (1 000 000) de francs, celui "
    "qui porte atteinte à la fortune d'autrui :\n- a) par vol, c'est-à-dire en soustrayant la "
    "chose d'autrui ;\n- b) par abus de confiance, c'est-à-dire en détournant ou détruisant..."
)
ARTICLE_320 = (
    "Vol aggravé\n\n(1) Les peines de l'article 318 ci-dessus sont doublées si le vol a été "
    "commis soit :\n- a) à l'aide de violences ;\n- b) avec port d'arme ;"
)
ARTICLE_321 = (
    "Abus de confiance et escroquerie aggravés\n\nLes peines de l'article 318 ci-dessus sont "
    "doublées si l'abus de confiance ou l'escroquerie ont été commis soit :"
)
# Le morceau 33 du Code du travail : la fin de l'article 33, puis l'article 34.
MORCEAU_33 = (
    "(1) Dans chacun des cas a, b et c visés à l'article 32 ci-dessus, l'employeur est tenu... "
    "Article 34.- (1) Le contrat de travail à durée indéterminée peut toujours être résilié par "
    "la volonté de l'une des parties. Cette résiliation est subordonnée à un préavis... (3) Un "
    "arrêté du ministre chargé du Travail, pris après avis de la Commission nationale "
    "consultative du\n\ntravail, détermine les conditions et la durée du préavis compte tenu de "
    "l'ancienneté du travailleur et de sa classification professionnelle."
)
# 35.1 en entier pour ce qui compte ici : les obligations PENDANT le preavis,
# puis l'article 39 et ses montants.
MORCEAU_35_1_REEL = (
    "(1) Pendant la durée du préavis, l'employeur et le travailleur sont tenus au respect de "
    "toutes les obligations réciproques qui leur incombent. (2) ... un jour de liberté par "
    "semaine... Article 39.- (1) Toute rupture abusive du contrat peut donner lieu à des "
    "dommages-intérêts. ... Toutefois, le montant des dommages-intérêts, sans excéder un (1) "
    "mois de salaire par année d'ancienneté dans l'entreprise, ne peut être inférieur à trois "
    "(3) mois de salaire."
)
AUDCG_228 = (
    "Lorsque le contrat est à durée indéterminée, chacune des parties peut y mettre fin "
    "moyennant un préavis. La durée du préavis est d'un mois pour la première année du contrat, "
    "de deux mois pour la deuxième année commencée, de trois mois pour la troisième..."
)
# L'acces non autorise : l'alinea 2 renvoie aux peines de l'alinea 1.
ALINEA_65_2 = (
    "(2) Est puni des peines prévues à l'alinéa 1 ci-dessus, tout accès non autorisé, à "
    "l'ensemble ou à une partie d'un réseau de communications électroniques ou d'un système "
    "d'information ou d'un équipement terminal. ... Article 68.- (1) Est puni d'un "
    "emprisonnement de cinq (05) à dix (10) ans et d'une amende de 10.000.000 (dix millions) à "
    "50.000.000 (cinquante millions) F CFA..."
)
ALINEA_65_1 = (
    "(1) Est puni d'un emprisonnement de cinq (05) à dix (10) ans et d'une amende de 5.000.000 "
    "(cinq millions) à 10.000.000 (dix millions) F CFA ou de l'une de ces deux peines "
    "seulement, celui qui effectue, sans droit ni autorisation, l'interception..."
)
ARTICLE_69 = (
    "Est puni d'un emprisonnement de cinq (05) à dix (10) ans et d'une amende de 10.000.000 "
    "(dix millions) à 100.000.000 (cent millions) F CFA ou de l'une de ces deux peines "
    "seulement, celui qui accède sans droit, et en violation des mesures de sécurité..."
)


def _extrait(reference: str, numero: str = "1", article_id: int = 1,
             contenu: str = "Texte.", source: str = "semantic") -> ChunkResult:
    return ChunkResult(
        article_id=article_id, law_id=1, number=numero, content=contenu, excerpt=contenu[:400],
        reference=reference, law_title=f"Texte {reference}", type="loi", language="fr",
        status="published", relevance_score=0.5, source=source,
    )


Lu = Union[str, Tuple[str, str], Tuple[str, str, str], Tuple[str, str, str, str]]


def _contexte(lus: List[Lu], voisins: Optional[List[str]] = None, degradee=False,
              refus=False, chemin="exact") -> ContexteDuChat:
    """`lus` : des references, ou (reference, numero[, contenu[, origine]])."""
    extraits = []
    for i, lu in enumerate(lus, start=1):
        reference, numero, *reste = (lu, "1") if isinstance(lu, str) else lu
        extraits.append(_extrait(
            reference, numero, article_id=i, contenu=reste[0] if reste else "Texte.",
            source=reste[1] if len(reste) > 1 else "semantic",
        ))
    return ContexteDuChat(
        extraits=extraits, lus=extraits,
        voisins=None if voisins is None else [(100 + i, ref) for i, ref in enumerate(voisins)],
        semantique_degradee=degradee, chemin=chemin, refus=refus,
    )


# ==================== LES QUESTIONS DE REFERENCE ====================


_ARTICLE_NOMME = re.compile(r"\barticle\s+\d+", re.IGNORECASE)


class TestLesQuestionsDeReference:

    def test_chaque_question_une_fois_dans_les_deux_langues(self):
        questions = [q.question for q in QUESTIONS_DE_REFERENCE]
        assert len(questions) == len(set(questions))
        assert {q.langue for q in QUESTIONS_DE_REFERENCE} == {"fr", "en"}
        assert {q.origine for q in QUESTIONS_DE_REFERENCE} == {SUGGESTION, FOND}

    def test_toutes_les_suggestions_des_questions_de_fond_et_des_articles_nommes(self):
        suggestions = [q for q in QUESTIONS_DE_REFERENCE if q.origine == SUGGESTION]
        fond = [q for q in QUESTIONS_DE_REFERENCE if q.origine == FOND]
        nommes = [q for q in QUESTIONS_DE_REFERENCE if q.article_nomme is not None]
        # 4 profils x 4 questions, dans chaque langue ; le profil general
        # reprend trois questions du citoyen.
        assert len(suggestions) == 32
        assert sum(q.langue == "fr" for q in suggestions) == 16
        assert 10 <= len(fond) - len(nommes) <= 14
        # La forme de l'incident (« article 76 du Code du travail »), dans
        # les deux langues : elle passe par la lecture de l'article nomme.
        assert len(nommes) >= 4 and {q.langue for q in nommes} == {"fr", "en"}
        assert "Que dit l'article 76 du Code du travail ?" in {q.question for q in nommes}

    def test_le_champ_article_nomme_marque_chaque_question_qui_nomme_un_article(self):
        """
        Le champ, et non le texte, dit au controle d'exiger l'article en tete :
        il doit marquer TOUTES les questions qui en nomment un, et le bon.
        """
        for q in QUESTIONS_DE_REFERENCE:
            nomme_dans_le_texte = _ARTICLE_NOMME.search(q.question)
            assert (q.article_nomme is not None) == bool(nomme_dans_le_texte), q.question
            if q.article_nomme is None:
                continue
            reference, numero = q.article_nomme
            assert re.search(rf"\b[Aa]rticle\s+{numero}\b", q.question), q.question
            assert q.article_nomme in q.articles_attendus, q.question
            assert reference in q.attendues(), q.question

    def test_les_tolerances_sont_d_autres_textes_relues_question_par_question(self):
        """
        Un extrait tolere avant la reponse est lu et juge : la liste est fermee,
        et l'ajout d'une tolerance doit passer par ici. Jamais pour la question
        de l'incident ni pour sa jumelle : le Statut de la fonction publique
        avant le Code du travail est le cas d'ecole.
        """
        toleres = {q.question: q.toleres_avant for q in QUESTIONS_DE_REFERENCE if q.toleres_avant}

        # La question francaise de l'election n'a jamais lu la Constitution de
        # 1996 avant sa reponse (revue du tour 4) : pas de tolerance pour elle.
        assert toleres == {
            "How is the President of the Republic elected?": (("PRC-6283", "6"),),
            "What distinguishes a fixed-term employment contract from an open-ended one?":
                (("PRC-6506", "84"),),
            "Quels sont les éléments constitutifs de l’abus de confiance ?":
                (("CODE-CIVIL", "1384.2"),),
        }
        for q in QUESTIONS_DE_REFERENCE:
            for reference, _ in q.toleres_avant:
                # Un texte attendu n'a pas besoin d'etre tolere.
                assert reference not in q.attendues(), (q.question, reference)
        assert QUESTION_DE_L_INCIDENT not in toleres
        assert "What damages does the Labour Code provide for unfair dismissal?" not in toleres

    def test_la_constitution_de_1996_n_est_toleree_qu_avec_sa_ligne_de_statut(self):
        """
        Sa tolerance tient a une condition : le prompt donne son article 6
        avec « Statut : article modifié par la Loi N°2026/002 [...] réécrit
        en entier [...] ce texte fait foi ». Sans ce lien, le modele lirait en
        tete « renouvelable une fois » sans avertissement, et la tolerance
        devrait tomber. Revue du tour 4 : verifier le lien ne suffit pas, le
        bloc que le modele lit est juge ici, tel que le prompt le construit.
        """
        from app.services.prompts import format_chunk_block
        from app.services.statut_des_sources import liens_qui_touchent

        liens = liens_qui_touchent("PRC-6283", "6")
        bloc = format_chunk_block(_extrait("PRC-6283", "6", contenu=(
            "(2) Le Président de la République est élu pour un mandat de sept (7) ans "
            "renouvelable une fois.")), 1)

        assert any(lien.par_reference == "PRC-10679" and lien.nature != "complete"
                   for lien in liens), liens
        statut = next(ligne for ligne in bloc.splitlines() if ligne.startswith("Statut :"))
        assert "Loi N°2026/002" in statut and "(PRC-10679)" in statut, statut
        # Tout l'article, mandat compris, et non la seule vacance.
        assert "réécrit en entier" in statut and "ce texte fait foi" in statut, statut

    def test_chaque_question_dit_ses_textes_et_ses_articles_par_leur_reference(self):
        for q in QUESTIONS_DE_REFERENCE:
            assert q.textes and all(q.textes), q.question
            for reference in q.attendues():
                assert re.fullmatch(r"[A-Z0-9]+(?:-[A-Z0-9]+)+", reference), reference
            assert q.articles.strip(), q.question
            # L'article seul dit si le modele peut repondre : chaque question
            # en a, et chacun est d'un texte attendu. « 65.2 » : un morceau
            # precis, quand la reponse est dans la suite d'un article.
            assert q.articles_attendus, q.question
            for reference, numero in q.articles_attendus + q.renvoi:
                assert reference in q.attendues(), (q.question, reference)
                assert re.fullmatch(r"\d+(?:-\d+)?(?:\.(?!1$)\d+)?", numero), (q.question, numero)

    def test_chaque_article_qui_repond_dit_les_mots_de_sa_regle(self):
        """
        Porter l'article ne suffit pas : la regle demandee (la peine, la duree)
        doit etre dans l'extrait. Chaque article attendu d'une question qui
        n'en nomme pas, et chaque article vise par un renvoi, dit les mots de
        sa regle ; le test d'integration verifie qu'ils sont dans la base.
        Une question qui nomme un article demande l'article entier : sans mots.
        """
        for q in QUESTIONS_DE_REFERENCE:
            declarees = {(ref, num) for ref, num, _ in q.regles}
            assert all(mots.strip() for _, _, mots in q.regles), q.question
            assert declarees <= set(q.articles_attendus + q.renvoi), q.question
            if q.article_nomme is None:
                assert declarees == set(q.articles_attendus + q.renvoi), q.question
            else:
                assert not q.regles and not q.renvoi, q.question

    # Des articles qui ne portent pas la regle que leur question demande
    # (relus le 30/10/2026) : les attendre gardait des questions vertes la ou
    # le modele ne lisait pas la reponse (revue du tour 4).
    QUI_NE_REPONDENT_PAS = {
        # « Les peines de l'article 318 ci-dessus sont doublées » : sans peine.
        ("PRC-4722", "320"): "renvoie aux peines de l'art. 318",
        ("PRC-4722", "321"): "renvoie aux peines de l'art. 318",
        # Les obligations PENDANT le preavis, puis les art. 36 a 39.
        ("LOI-1992-007", "35"): "ne dit pas la duree du preavis",
        # La prorogation par la distance, « nonobstant l'article 119 ».
        ("LOI-2005-007", "120"): "ne dit pas la duree qu'il proroge",
        # L'obligation de s'immatriculer, pas la personnalite juridique.
        ("OHADA-AUSCGIE-2014", "97"): "ne dit pas quand nait la personnalite",
        # Les mentions de la demande, pas les pieces justificatives.
        ("OHADA-AUDCG-2010", "46"): "ne dit pas les pieces",
    }

    def test_aucun_article_qui_ne_porte_pas_la_regle_n_est_attendu(self):
        for q in QUESTIONS_DE_REFERENCE:
            if q.article_nomme is not None:
                continue
            for article in q.articles_attendus:
                assert article not in self.QUI_NE_REPONDENT_PAS, (
                    q.question, article, self.QUI_NE_REPONDENT_PAS.get(article))

    def test_la_peine_du_vol_et_de_l_escroquerie_est_celle_de_l_article_318(self):
        """La revue du tour 4 : la peine est a l'art. 318, et nulle part ailleurs."""
        peines = [q for q in QUESTIONS_DE_REFERENCE if q.article_nomme is None and any(
            mot in q.question for mot in ("vol ?", "theft", "escroquerie", "deceiving"))]

        assert len(peines) == 6, [q.question for q in peines]
        for q in peines:
            assert q.articles_attendus == (("PRC-4722", "318"),), q.question
            assert re.search(r"cinq \(5\) à dix \(10\) ans|cent mille \(100 000\)",
                             q.regle_de("PRC-4722", "318")), q.question

    def test_le_delai_de_preavis_est_celui_de_l_article_34(self):
        preavis = [q for q in QUESTIONS_DE_REFERENCE
                   if "préavis" in q.question or "notice period" in q.question]

        assert {q.langue for q in preavis} == {"fr", "en"}
        for q in preavis:
            assert q.articles_attendus == (("LOI-1992-007", "34"),), q.question
            assert "durée du préavis" in q.regle_de("LOI-1992-007", "34")

    def test_l_acces_non_autorise_renvoie_aux_peines_de_l_alinea_1(self):
        acces = [q for q in QUESTIONS_DE_REFERENCE if "accès non autorisé" in q.question
                 or "unauthorised access" in q.question]

        assert {q.langue for q in acces} == {"fr", "en"}
        for q in acces:
            assert q.articles_attendus == (("LOI-2010-012", "65.2"),), q.question
            assert q.renvoi == (("LOI-2010-012", "65"),), q.question

    def test_l_aupsrve_n_attend_jamais_son_numero_1(self):
        """Il y range les articles 1 et 1-2 a 1-16 : « 1 » n'y designe rien de sur."""
        for q in QUESTIONS_DE_REFERENCE:
            assert ("OHADA-AUPSRVE-2023", "1") not in q.articles_attendus, q.question

    def test_cinq_questions_mesurent_l_index_dont_les_plus_fragiles(self):
        rappel = {q.question: q for q in QUESTIONS_DE_REFERENCE if q.rappel}
        assert len(rappel) == 5
        assert QUESTION_DE_L_INCIDENT in rappel
        # Les deux qui s'effondraient les premieres sur l'index local (ef 40).
        assert "Quelle est la peine pour la corruption d’un fonctionnaire ?" in rappel
        assert any("investment incentives ordinance" in q for q in rappel)
        assert {q.langue for q in rappel.values()} == {"fr", "en"}

    # Les faiblesses connues du 30/10/2026 (revue du tour 5), mesurees sur la
    # recuperation finale : la liste est FERMEE, toute entree ou sortie passe
    # par ici, avec sa mesure (questions_de_reference, FAIBLESSES_CONNUES).
    FAIBLESSES_DU_30_10_2026 = {
        "Quelle peine encourt l’auteur d’un vol ?",
        "What penalty does the Penal Code provide for theft?",
        "Quel est le montant de l'amende prévue pour le vol ?",
        "Quel est le délai de préavis en cas de licenciement ?",
        "How does the law punish unauthorised access to a computer system?",
        "How much compensation is allocated for unfair dismissal?",
    }

    def test_les_faiblesses_connues_sont_la_liste_fermee_mesuree(self):
        faibles = {q.question: q for q in QUESTIONS_DE_REFERENCE if not q.garantie}

        assert set(FAIBLESSES_CONNUES) == self.FAIBLESSES_DU_30_10_2026
        # Chaque cle est une question de reference, au mot pres : une cle mal
        # recopiee laisserait sa question garantie, et la route rouge.
        assert set(faibles) == set(FAIBLESSES_CONNUES)
        for question, item in faibles.items():
            assert item.faiblesse == FAIBLESSES_CONNUES[question] and item.faiblesse.strip()
        assert sum(q.garantie for q in QUESTIONS_DE_REFERENCE) == len(QUESTIONS_DE_REFERENCE) - 6

    def test_les_questions_qui_voient_les_pannes_rejouees_sont_garanties(self):
        """
        Une faiblesse ne fait jamais rendre 503 : une question qui sert a voir
        une panne n'est donc jamais une faiblesse, sinon la panne passerait.
        - les questions `rappel` : seules a mesurer l'index (l'incident) ;
        - celles qui nomment un article : seules a voir la lecture directe
          coupee, ou l'art. 318 ecarte par _sans_doute ;
        - celles que les pannes rejouees font echouer (mesure du 30/10/2026,
          data/fiabilite/tmp/controle6/) : le licenciement abusif dans les
          deux langues (morceau 35.1 perdu, poids de fusion), l'escroquerie
          (morceau 318 perdu), le preavis en anglais (morceau 33 perdu), la
          duree de la garde a vue (RRF_K a 500, poids 0,1 / 0,9), l'acces non
          autorise en francais (poids 0,9 / 0,1).
        """
        sentinelles = [q for q in QUESTIONS_DE_REFERENCE
                       if q.rappel or q.article_nomme is not None] + [
            _de_reference(question) for question in (
                QUESTION_DE_L_INCIDENT,
                "What damages does the Labour Code provide for unfair dismissal?",
                "Quelle est la peine prévue pour l’escroquerie ?",
                "What is the notice period for dismissing an employee?",
                "Quelle est la durée maximale de la garde à vue ?",
                "What is the maximum length of police custody?",
                "Quelles peines pour l’accès non autorisé à un système d’information ?",
            )]

        assert len({q.question for q in sentinelles}) == 16
        assert [q.question for q in sentinelles if not q.garantie] == []

    def test_l_escroquerie_exige_le_code_penal_seul(self):
        """
        La loi sur la cybercriminalite n'a aucun article sur l'escroquerie :
        l'admettre masquait le contexte de l'ancienne suggestion anglaise, qui
        ne portait aucun extrait du Code penal (29/10/2026).
        """
        escroquerie = [
            q for q in QUESTIONS_DE_REFERENCE
            if q.origine == SUGGESTION and ("escroquerie" in q.question
                                            or "deceiving someone" in q.question)
        ]

        assert {q.langue for q in escroquerie} == {"fr", "en"}
        for q in escroquerie:
            assert q.textes == (("PRC-4722",),), q.question
            # L'art. 321 ne fait que doubler les peines de l'art. 318.
            assert q.articles_attendus == (("PRC-4722", "318"),), q.question

    def test_la_constitution_de_1996_seule_ne_suffit_pas_pour_l_election(self):
        """Son art. 6 dit « renouvelable une fois » : une reponse perimee."""
        election = [q for q in QUESTIONS_DE_REFERENCE if "Président" in q.question
                    or "President" in q.question]

        assert len(election) == 2
        for q in election:
            assert q.exigences_manquantes(["PRC-6283"]) == list(q.textes)
            assert q.exigences_manquantes(["PRC-10679"]) == []
            assert q.exigences_manquantes(["LOI-2012-001"]) == []

    def test_une_exigence_se_remplit_par_l_un_de_ses_textes(self):
        q = _question(textes=(("PRC-4722", "LOI-2010-012"), ("LOI-2005-007",)))

        assert q.exigences_manquantes(["LOI-2010-012", "LOI-2005-007"]) == []
        assert q.exigences_manquantes(["LOI-2010-012"]) == [("LOI-2005-007",)]
        assert q.exigences_manquantes([]) == list(q.textes)
        assert q.attendues() == ("PRC-4722", "LOI-2010-012", "LOI-2005-007")


# ==================== ACCORD AVEC LE FRONT ====================


_LITTERAL = re.compile(r"'((?:[^'\\]|\\.)*)'|\"((?:[^\"\\]|\\.)*)\"")

# Les suggestions anglaises reformulees le 29/10/2026 (mesures dans
# questions_de_reference) : seulement pour que l'echec dise quoi publier.
REFORMULEES = {
    "What compensation is due for unfair dismissal?":
        "What damages does the Labour Code provide for unfair dismissal?",
    "What does the Penal Code say about obtaining money by false pretences on the internet?":
        "How does the Penal Code punish deceiving someone to obtain their money?",
    "How do I register a company in the trade and personal property credit register?":
        "From when does a company have legal personality under OHADA law?",
    "How can I make a customer pay an unpaid invoice?":
        "How can a business recover an unpaid invoice from a customer under OHADA law?",
}


def suggestions_du_front(source: str) -> Set[Tuple[str, str]]:
    """(langue, question) de SUGGESTIONS, dans le source de suggestions.ts."""
    corps = source[source.index("export const SUGGESTIONS"):]
    debut_fr = corps.index("fr: {")
    debut_en = corps.index("en: {")
    fin = debut_en + re.search(r"\n\s*\};", corps[debut_en:]).start()
    paires = set()
    for langue, bloc in (("fr", corps[debut_fr:debut_en]), ("en", corps[debut_en:fin])):
        # Les questions sont dans les tableaux ; les cles ('étudiant') hors.
        for tableau in re.findall(r"\[(.*?)\]", bloc, flags=re.S):
            for simple, double in _LITTERAL.findall(tableau):
                question = (simple or double).replace("\\'", "'").replace('\\"', '"')
                paires.add((langue, question))
    return paires


def _fichier_du_front() -> Optional[Path]:
    """
    src/lib/suggestions.ts du depot du front : JURIX_FRONT, sinon un
    JuriX-frontend-main a cote de ce depot ou d'un de ses parents (un
    worktree vit sous data/).
    """
    if os.environ.get("JURIX_FRONT"):
        return Path(os.environ["JURIX_FRONT"]) / "src" / "lib" / "suggestions.ts"
    for dossier in (RACINE, *RACINE.parents):
        candidat = dossier / "JuriX-frontend-main" / "src" / "lib" / "suggestions.ts"
        if candidat.exists():
            return candidat
    return None


class TestAccordAvecLeFront:

    def test_le_lecteur_prend_les_questions_et_pas_les_cles(self):
        source = """
        export const SUGGESTIONS: Record<Language, Record<Cle, readonly string[]>> = {
          fr: {
            citoyen: ['Quels sont mes droits ?', 'Que risque l’auteur ?'],
            'étudiant': [
              'Qu’est-ce qu’un GIE ?',
            ],
          },
          en: {
            citoyen: ["What are the employer's obligations?", 'It\\'s here?'],
          },
        };
        """

        assert suggestions_du_front(source) == {
            ("fr", "Quels sont mes droits ?"), ("fr", "Que risque l’auteur ?"),
            ("fr", "Qu’est-ce qu’un GIE ?"),
            ("en", "What are the employer's obligations?"), ("en", "It's here?"),
        }

    def test_chaque_suggestion_du_front_est_une_question_de_reference(self):
        """
        Un clic ENVOIE la question : une suggestion ajoutee ou reformulee au
        front, sans ses textes ici, ne serait plus controlee ; et une question
        reformulee ici, mais pas au front, laisserait les utilisateurs cliquer
        une question que le chat ne sait pas servir.
        """
        fichier = _fichier_du_front()
        if fichier is None or not fichier.exists():
            pytest.skip("depot du front introuvable (JURIX_FRONT=chemin du depot)")
        front = suggestions_du_front(fichier.read_text(encoding="utf-8"))
        ici = {(q.langue, q.question) for q in QUESTIONS_DE_REFERENCE if q.origine == SUGGESTION}
        a_publier = [
            f"remplacer « {ancienne} » par « {REFORMULEES[ancienne]} »"
            for langue, ancienne in sorted(front - ici) if ancienne in REFORMULEES
        ]

        assert front, "aucune suggestion lue : le format de suggestions.ts a change ?"
        assert not a_publier, (
            f"{fichier} propose encore des suggestions que le chat sert mal (mesures dans "
            "app/services/questions_de_reference.py) ; au front : " + " ; ".join(a_publier)
        )
        assert front - ici == set(), "suggestions du front sans textes attendus ici"
        assert ici - front == set(), (
            "suggestions retirees du front, a passer en « fond » ou a retirer"
        )


# ==================== UN EXTRAIT PORTE-T-IL UN ARTICLE ? ====================


class TestLaLectureDUnArticle:

    @pytest.mark.parametrize("numero, contenu, attendu", [
        ("318", "Vol, abus de confiance, escroquerie (1) Est puni...", True),
        ("35.1", "(1) Pendant la durée du préavis...", False),
    ], ids=["meme-numero", "autre-numero"])
    def test_par_le_numero_du_morceau(self, numero, contenu, attendu):
        assert porte_l_article(_extrait("PRC-4722", numero, contenu=contenu),
                               "PRC-4722", "318") is attendu

    def test_un_morceau_n_x_est_une_part_de_l_article_n(self):
        assert porte_l_article(_extrait("LOI-1992-007", "35.1"), "LOI-1992-007", "35")
        assert not porte_l_article(_extrait("LOI-2005-007", "120.2"), "LOI-2005-007", "12")

    def test_un_morceau_n_2_ne_compte_pas_pour_n_a_son_numero(self):
        """
        « 35.2 » du Code du travail : l'alinea 6 de l'article 39 (son en-tete
        est dans « 35.1 »), puis l'article 40, et rien de l'article 35. Revue
        du tour 3 : compte pour 35, il gardait vertes les questions du preavis.
        Compte pour 39, il garderait verte celle du licenciement abusif avec
        le seul salaire de reference.
        """
        suite = _extrait("LOI-1992-007", "35.2", contenu=SUITE_35_2)

        for numero in ("35", "36", "37", "39"):
            assert not porte_l_article(suite, "LOI-1992-007", numero), numero
        # L'article dont il porte l'en-tete, lui, compte.
        assert porte_l_article(suite, "LOI-1992-007", "40")

    def test_un_morceau_precis_s_attend_par_son_numero_exact(self):
        """« 65.2 » : l'alinea 2 de l'article 65 de la loi sur la cybercriminalite, la reponse."""
        alinea_2 = _extrait("LOI-2010-012", "65.2", contenu=(
            "(2) Est puni des peines prévues à l'alinéa 1 ci-dessus, tout accès non autorisé... "
            "Article 66.- (1) Est puni d'un emprisonnement de deux (02) à cinq (05) ans"))

        assert porte_l_article(alinea_2, "LOI-2010-012", "65.2")
        assert not porte_l_article(alinea_2, "LOI-2010-012", "65")
        assert not porte_l_article(_extrait("LOI-2010-012", "65.1"), "LOI-2010-012", "65.2")

    @pytest.mark.parametrize("numero", ["36", "37", "38", "39"])
    def test_par_l_en_tete_dans_un_morceau_qui_en_porte_plusieurs(self, numero):
        assert porte_l_article(_extrait("LOI-1992-007", "35.1", contenu=MORCEAU_35_1),
                               "LOI-1992-007", numero)

    @pytest.mark.parametrize("contenu", [
        "Article 122 — (1) a) Le suspect doit être immédiatement informé",
        "Art. 122.- La personne gardée à vue",
        "ARTICLE 122 : Le suspect",
    ], ids=["tiret-long", "abrege", "majuscules"])
    def test_les_formes_d_en_tete(self, contenu):
        assert porte_l_article(_extrait("LOI-2005-007", "120.2", contenu=contenu),
                               "LOI-2005-007", "122")

    @pytest.mark.parametrize("contenu", [
        "dans les cas visés à l'article 37 ci-dessus, l'employeur",
        "Article 370.- Les dispositions",
        "Article 37-1 : Le travailleur",
        "Les articles 37 et 39 ne s'appliquent pas",
    ], ids=["renvoi", "autre-numero", "article-insere", "pluriel"])
    def test_un_renvoi_ou_un_autre_article_n_est_pas_un_en_tete(self, contenu):
        assert not porte_l_article(_extrait("LOI-1992-007", "33", contenu=contenu),
                                   "LOI-1992-007", "37")

    def test_l_article_premier(self):
        for contenu in ("Article premier.- La présente loi", "Art. 1er : La présente loi"):
            assert porte_l_article(_extrait("PRC-1", "SECTION_1", contenu=contenu), "PRC-1", "1")

    def test_le_meme_numero_d_un_autre_texte_ne_compte_pas(self):
        """« article 76 du Code du travail » n'est pas l'article 76.37 d'une loi de finances."""
        assert not porte_l_article(_extrait("PRC-9449", "76.37"), "LOI-1992-007", "76")


# ==================== RAPPEL ET VERDICT ====================


def test_le_rappel_compte_les_vrais_voisins_retrouves():
    exacts = list(range(10))

    assert rappel_a_k(exacts, exacts) == 1.0
    assert rappel_a_k(list(range(2, 12)), exacts) == 0.8
    assert rappel_a_k(list(range(100, 110)), exacts) == 0.0
    # Au-dela des k premiers de l'index, un vrai voisin ne compte pas.
    assert rappel_a_k(list(range(100, 110)) + exacts, exacts) == 0.0
    assert rappel_a_k([1, 2], []) == 0.0


class TestLeVerdict:

    def test_textes_lus_et_premier_voisin_la_question_tient(self):
        verdict = juger(_question(), _contexte(["LOI-1992-007", "PRC-9449"], ["LOI-1992-007"]))

        assert verdict.ok and verdict.raisons == []
        assert verdict.rang == 1 and verdict.rang_semantique == 1 and verdict.intrus == 0
        assert verdict.lus == ["LOI-1992-007 art. 1", "PRC-9449 art. 1"]
        # Rien a dire de plus pour une question qui ne nomme pas d'article.
        assert verdict.article_nomme_en_tete is None
        assert "article_nomme_en_tete" not in verdict.resume()

    def test_la_forme_de_l_incident_deux_raisons(self):
        tableaux = ["PRC-9449"] * 8
        verdict = juger(_question(), _contexte(tableaux, tableaux * 4))

        assert not verdict.ok
        assert verdict.raisons[0].startswith("textes attendus absents du contexte lu")
        assert "LOI-1992-007" in verdict.raisons[0]
        assert "10 premiers voisins" in verdict.raisons[1]

    def test_le_plein_texte_sauve_le_contexte_mais_le_volet_semantique_echoue(self):
        """Ce que le seul controle du contexte laisserait passer."""
        verdict = juger(
            _question(), _contexte(["LOI-1992-007", "PRC-9449"], ["PRC-9449"] * 32)
        )

        assert not verdict.ok and verdict.rang == 1
        assert verdict.raisons == [
            "aucun texte attendu parmi les 10 premiers voisins du volet semantique"
        ]

    def test_un_texte_attendu_au_onzieme_voisin_ne_suffit_pas(self):
        voisins = ["PRC-9449"] * 10 + ["LOI-1992-007"]

        verdict = juger(_question(), _contexte(["LOI-1992-007"], voisins))

        assert not verdict.ok and verdict.rang_semantique == 11

    def test_un_contexte_ecarte_par_la_barriere_est_un_echec_qui_le_dit(self):
        verdict = juger(_question(), _contexte([], ["LOI-1992-007"], refus=True))

        assert not verdict.ok
        assert "barriere du chat" in verdict.raisons[0]

    @pytest.mark.parametrize("contexte", [
        _contexte(["LOI-1992-007"], ["LOI-1992-007"], degradee=True),
        _contexte(["LOI-1992-007"], None),
        _contexte(["LOI-1992-007"], ["LOI-1992-007"], chemin="echec"),
    ], ids=["degradee", "sans-volet", "chemin-echec"])
    def test_un_volet_semantique_en_panne_est_un_echec(self, contexte):
        verdict = juger(_question(), contexte)

        assert contexte.volet_en_panne
        assert verdict.raisons == [
            "volet semantique indisponible : le chat repondrait du plein texte seul"
        ]

    def test_un_rappel_sous_le_seuil_a_l_un_des_parcours_est_un_echec(self):
        contexte = _contexte(["LOI-1992-007"], ["LOI-1992-007"])

        assert juger(_question(rappel=True), contexte, rappels={200: 0.8, 400: 1.0}).ok
        marge = juger(_question(rappel=True), contexte, rappels={200: 0.0, 400: 1.0})
        recherche = juger(_question(rappel=True), contexte, rappels={200: 1.0, 400: 0.5})

        assert marge.raisons == [
            "rappel@10 de l'index HNSW a ef_search 200 : 0.0, sous le seuil de 0.8"
        ]
        assert recherche.raisons == [
            "rappel@10 de l'index HNSW a ef_search 400 : 0.5, sous le seuil de 0.8"
        ]

    def test_sans_articles_declares_un_autre_texte_avant_le_texte_attendu_est_un_echec(self):
        """La meme regle, le premier extrait d'un texte attendu tenant lieu d'article."""
        verdict = juger(_question(), _contexte(["PRC-9449", "LOI-1992-007"], ["LOI-1992-007"]))

        assert verdict.rang == 2 and verdict.intrus == 1
        assert verdict.raisons == [
            "un autre texte est lu avant l'extrait qui repond (rang 2) : PRC-9449 art. 1"
        ]

    def test_le_detail_d_un_echec_dit_ce_qui_manque_et_ce_qui_a_ete_lu(self):
        verdict = juger(_question(articles_attendus=INDEMNITES),
                        _contexte(["PRC-9449"], ["PRC-9449"]))

        detail = verdict.detail()
        assert detail["attendus"] == [["LOI-1992-007"]]
        assert detail["articles_attendus"] == "LOI-1992-007 art. 36, 37, 39"
        assert detail["lus"] == ["PRC-9449 art. 1"]
        assert detail["ok"] is False and len(detail["raisons"]) == 2

    def test_le_detail_nomme_les_autres_textes_lus_avant_la_reponse(self):
        lus = [("DEC-1994-199", "120"), ("LOI-1992-007", "35.1", MORCEAU_35_1)]

        detail = juger(_question(articles_attendus=INDEMNITES),
                       _contexte(lus, ["LOI-1992-007"])).detail()

        assert detail["autres_textes_avant_la_reponse"] == ["DEC-1994-199 art. 120"]
        assert detail["intrus_avant_la_reponse"] == 1


class TestLArticleQuiRepond:
    """La revue du 29/10/2026 : le bon texte, par un article hors sujet, passait."""

    def test_le_morceau_qui_porte_les_articles_36_a_39_lu_en_tete(self):
        verdict = juger(
            _question(articles_attendus=INDEMNITES),
            _contexte([("LOI-1992-007", "35.1", MORCEAU_35_1), "DEC-1994-199"],
                      ["LOI-1992-007"]),
        )

        assert verdict.ok, verdict.raisons
        assert verdict.rang_article == 1 and verdict.intrus == 0

    def test_le_bon_texte_sans_l_article_qui_repond_est_un_echec(self):
        """Le morceau 35.1 perdu : 35.2 (fin du 39, art. 40) et 33 ne repondent pas."""
        sans_35_1 = [("DEC-1994-199", "120"), ("CODE-CIVIL", "107"),
                     ("LOI-1992-007", "35.2", "(6) Le salaire... Article 40.- (1) Les "
                                              "dispositions de l'article 34 alinéa (1)"),
                     ("LOI-1992-007", "33", "(1) Dans chacun des cas... Article 34.- (1)")]

        verdict = juger(_question(articles_attendus=INDEMNITES),
                        _contexte(sans_35_1, ["LOI-1992-007"]))

        assert verdict.rang == 3 and verdict.rang_article is None
        assert verdict.raisons == [
            "aucun article qui repond n'est lu par le modele : LOI-1992-007 art. 36, 37, 39"
        ]

    def test_l_article_derriere_trois_extraits_d_autres_regimes_est_un_echec(self):
        """« What compensation is due for unfair dismissal? », mesure du 29/10/2026."""
        lus = [("LOI-1992-007", "153"), ("OHADA-AUDCG-2010", "231"), ("DEC-1994-199", "120"),
               ("LOI-1992-007", "30"), ("PRC-8798", "20"),
               ("LOI-1992-007", "35.1", MORCEAU_35_1)]

        verdict = juger(_question(articles_attendus=INDEMNITES), _contexte(lus, ["LOI-1992-007"]))

        # Le texte est en tete : seul l'article dit que le modele lirait
        # d'abord l'indemnite d'un agent commercial et celle d'un fonctionnaire.
        assert verdict.rang == 1 and verdict.rang_article == 6 and verdict.intrus == 3
        assert verdict.raisons == [
            "un autre texte est lu avant l'extrait qui repond (rang 6) : "
            "OHADA-AUDCG-2010 art. 231, DEC-1994-199 art. 120, PRC-8798 art. 20",
            "extrait qui repond au rang 6, au-dela des 4 premiers extraits lus",
        ]

    def test_derriere_des_articles_du_meme_texte_l_article_tient_jusqu_au_rang_4(self):
        """« Quels sont mes droits en garde à vue ? » : 120.2 au rang 4, derriere le CPP."""
        droits = (("LOI-2005-007", "122"), ("LOI-2005-007", "123"))
        lus = [("LOI-2005-007", "92.1"), ("LOI-2005-007", "86"), ("LOI-2005-007", "135"),
               ("LOI-2005-007", "120.2", "(3) Mention... Article 122 — (1) a) Le suspect")]

        verdict = juger(_question(textes=(("LOI-2005-007",),), articles_attendus=droits),
                        _contexte(lus, ["LOI-2005-007"]))

        assert verdict.ok and verdict.rang_article == 4 and verdict.intrus == 0

    def test_au_dela_du_rang_4_l_article_est_noye_meme_derriere_son_texte(self):
        """
        RRF_K a 500 (un reglage d'exploitation), mesure du 29/10/2026 : pour
        « Quelle est la durée maximale de la garde à vue ? », le modele lit la
        detention provisoire (art. 218) et la garde du MINEUR (art. 702) avant
        l'article 120. Tout vient du Code de procedure penale : seul le rang
        le voit.
        """
        duree = (("LOI-2005-007", "119"), ("LOI-2005-007", "120"))
        lus = [("LOI-2005-007", "124.1"), ("LOI-2005-007", "118"), ("LOI-2005-007", "92.1"),
               ("LOI-2005-007", "218"), ("LOI-2005-007", "702"), ("LOI-2005-007", "120.1"),
               ("LOI-2005-007", "119")]

        verdict = juger(_question(textes=(("LOI-2005-007",),), articles_attendus=duree),
                        _contexte(lus, ["LOI-2005-007"]))

        assert verdict.rang_article == 6 and verdict.intrus == 0
        assert verdict.raisons == [
            "extrait qui repond au rang 6, au-dela des 4 premiers extraits lus"
        ]

    @pytest.mark.parametrize("lus, autres", [
        # Revue du tour 3 : le plein texte prive du morceau 35.1.
        ([("DEC-1994-199", "120"), ("CODE-CIVIL", "107"), ("LOI-1992-007", "35.1", MORCEAU_35_1)],
         "DEC-1994-199 art. 120, CODE-CIVIL art. 107"),
        # Revue du tour 3 : les poids de fusion 0,1 / 0,9.
        ([("DEC-1994-199", "120"), ("LOI-1992-007", "35.1", MORCEAU_35_1)],
         "DEC-1994-199 art. 120"),
        # Ce que le test du tour 3 jugeait correct.
        (["PRC-9449", "DEC-1994-199", ("LOI-1992-007", "35.1", MORCEAU_35_1)],
         "PRC-9449 art. 1, DEC-1994-199 art. 1"),
    ], ids=["plein-texte-sans-35-1", "poids-0-1-0-9", "loi-de-finances-puis-statut"])
    def test_un_seul_autre_regime_avant_l_article_est_un_echec(self, lus, autres):
        """L'indemnite du fonctionnaire licencie, lue avant le Code du travail."""
        verdict = juger(_question(articles_attendus=INDEMNITES), _contexte(lus, ["LOI-1992-007"]))

        assert not verdict.ok
        assert verdict.raisons == [
            f"un autre texte est lu avant l'extrait qui repond (rang {len(lus)}) : {autres}"
        ]

    def test_un_morceau_qui_continue_un_autre_article_ne_repond_pas(self):
        """
        Le plein texte sans 35.1, « Quel est le délai de préavis en cas de
        licenciement ? » (mesure du 29/10/2026) : 35.2, puis le preavis d'un
        AGENT COMMERCIAL, puis le morceau 33 qui porte l'article 34.
        """
        preavis = (("LOI-1992-007", "34"), ("LOI-1992-007", "35"))
        lus = [("LOI-1992-007", "35.2", SUITE_35_2), ("OHADA-AUDCG-2010", "228"),
               ("LOI-1992-007", "33", "(1) Dans chacun des cas... Article 34.- (1) Le contrat")]

        verdict = juger(_question(articles_attendus=preavis), _contexte(lus, ["LOI-1992-007"]))

        assert verdict.rang_article == 3
        assert verdict.raisons == [
            "un autre texte est lu avant l'extrait qui repond (rang 3) : OHADA-AUDCG-2010 art. 228"
        ]

    def test_un_autre_texte_apres_l_article_ne_compte_pas(self):
        """Ce qui suit la reponse n'est pas juge : les lignes de date et de statut l'eclairent."""
        lus = [("LOI-1992-007", "35.1", MORCEAU_35_1), "DEC-1994-199", "CODE-CIVIL", "PRC-9449"]

        verdict = juger(_question(articles_attendus=INDEMNITES), _contexte(lus, ["LOI-1992-007"]))

        assert verdict.ok and verdict.intrus == 0


class TestLaRegleDemandee:
    """
    Revue du tour 4 : des « articles qui repondent » ne portaient pas la regle
    demandee (l'art. 320 pour la peine du vol, le morceau 35.1 pour la duree
    du preavis), et le controle restait vert la ou le modele lisait d'abord
    un autre regime. L'extrait qui repond porte les mots de la regle.
    """

    def test_les_mots_se_lisent_aux_espaces_et_apostrophes_pres(self):
        extrait = _extrait("LOI-1992-007", "33", contenu=MORCEAU_33.replace("'", "’"))

        assert controle.porte_la_regle(extrait, "détermine les conditions et la  durée du préavis")
        assert controle.porte_la_regle(extrait, "consultative du travail, détermine")
        assert controle.porte_la_regle(extrait, "L'ANCIENNETÉ DU TRAVAILLEUR")
        assert controle.porte_la_regle(extrait, None) and controle.porte_la_regle(extrait, "")
        assert not controle.porte_la_regle(extrait, "durée du préavis est d'un mois")

    def test_le_meme_numero_sans_la_regle_ne_repond_pas(self):
        """
        « 134 » du Code penal est aussi l'art. 134-1, qui renvoie aux peines
        de l'art. 134 : lu seul, le modele n'a pas la peine.
        """
        item = _de_reference("Quelle est la peine pour la corruption d’un fonctionnaire ?")
        article_134_1 = ("PRC-4722", "134", "1.-Corruption passive (1) Quiconque [...] est puni "
                                            "des peines prévues à l'article 134 alinéa 1")
        article_134 = ("PRC-4722", "134", "Corruption active (1) Est puni d'un emprisonnement de "
                                          "cinq (05) à dix (10) ans et d'une amende de deux cent "
                                          "mille (200 000) à deux millions (2 000 000) de francs")

        seul = juger(item, _contexte([article_134_1, ("PRC-4722", "140")], ["PRC-4722"]))
        les_deux = juger(item, _contexte([article_134_1, article_134], ["PRC-4722"]))

        assert seul.rang_article is None
        assert seul.raisons == [
            "aucun article qui repond n'est lu par le modele : PRC-4722 art. 134 "
            "(lu sans la regle demandee : PRC-4722 art. 134)"
        ]
        assert les_deux.ok and les_deux.rang_article == 2

    def test_le_capital_de_la_sarl_n_est_pas_l_article_311_1(self):
        item = _de_reference("Quel est le capital social minimum d’une SARL ?")
        article_311_1 = ("OHADA-AUSCGIE-2014", "311", "1\n\nLes parts sociales doivent être "
                                                     "souscrites en totalité par les associés.")

        verdict = juger(item, _contexte([article_311_1], ["OHADA-AUSCGIE-2014"]))

        assert not verdict.ok and verdict.rang_article is None

    @pytest.mark.parametrize("question, renvoi", [
        ("Quelle peine encourt l’auteur d’un vol ?", ("PRC-4722", "320", ARTICLE_320)),
        ("What penalty does the Penal Code provide for theft?", ("PRC-4722", "320", ARTICLE_320)),
        ("Quelle est la peine prévue pour l’escroquerie ?", ("PRC-4722", "321", ARTICLE_321)),
        ("How does the Penal Code punish deceiving someone to obtain their money?",
         ("PRC-4722", "321", ARTICLE_321)),
    ], ids=["vol-320", "theft-320", "escroquerie-321", "deceiving-321"])
    def test_un_article_qui_double_les_peines_de_l_article_318_ne_repond_pas(
        self, question, renvoi
    ):
        """« Les peines de l'article 318 ci-dessus sont doublées » : aucune peine lisible."""
        verdict = juger(_de_reference(question), _contexte([renvoi], ["PRC-4722"]))

        assert verdict.raisons == [
            "aucun article qui repond n'est lu par le modele : PRC-4722 art. 318"
        ]

    def test_le_vol_en_francais_tel_que_la_revue_l_a_mesure(self):
        """
        Le contexte de « Quelle peine encourt l’auteur d’un vol ? » (revue du
        tour 4, jurix_dev, recuperation de 22 h 54) : la seule peine du vol, a
        l'art. 318, n'est lue qu'au rang 7, apres le vol et le recel d'effets
        MILITAIRES et le vol de matieres nucleaires. Vert par l'art. 320.
        """
        lus = [("PRC-4722", "320", ARTICLE_320), ("PRC-4722", "91"), ("PRC-5695", "56"),
               ("PRC-4722", "104"), ("PRC-5695", "54"), ("PRC-7769", "108"),
               ("PRC-4722", "318", ARTICLE_318), ("PRC-7769", "103")]

        verdict = juger(_de_reference("Quelle peine encourt l’auteur d’un vol ?"),
                        _contexte(lus, ["PRC-4722"]))

        assert verdict.rang_article == 7 and verdict.intrus == 3
        assert verdict.raisons == [
            "un autre texte est lu avant l'extrait qui repond (rang 7) : PRC-5695 art. 56, "
            "PRC-5695 art. 54, PRC-7769 art. 108",
            "extrait qui repond au rang 7, au-dela des 4 premiers extraits lus",
        ]

    def test_le_vol_en_anglais_tel_que_la_revue_l_a_mesure(self):
        """L'art. 318 jamais lu : la seule peine de vol chiffree est celle du Code de justice militaire."""
        lus = [("PRC-4722", "139"), ("PRC-4722", "188"), ("PRC-4722", "320", ARTICLE_320),
               ("PRC-5695", "54"), ("PRC-4722", "193"), ("PRC-5695", "56"), ("PRC-4722", "213"),
               ("PRC-4722", "322")]

        verdict = juger(_de_reference("What penalty does the Penal Code provide for theft?"),
                        _contexte(lus, ["PRC-4722"]))

        assert verdict.rang == 1 and verdict.rang_article is None
        assert verdict.raisons == [
            "aucun article qui repond n'est lu par le modele : PRC-4722 art. 318"
        ]

    def test_le_preavis_tel_que_la_revue_l_a_mesure(self):
        """
        « Quel est le délai de préavis en cas de licenciement ? » : 35.1 (les
        obligations PENDANT le preavis) en tete le gardait vert ; le modele
        lit le preavis d'un AGENT COMMERCIAL avant l'article 34.
        """
        lus = [("LOI-1992-007", "35.1", MORCEAU_35_1_REEL), ("LOI-1992-007", "35.2", SUITE_35_2),
               ("OHADA-AUPCAP-2015", "110"), ("OHADA-AUDCG-2010", "228", AUDCG_228),
               ("CODE-CIVIL", "107"), ("LOI-1992-007", "33", MORCEAU_33)]

        verdict = juger(_de_reference("Quel est le délai de préavis en cas de licenciement ?"),
                        _contexte(lus, ["LOI-1992-007"]))

        assert verdict.rang_article == 6
        assert verdict.raisons == [
            "un autre texte est lu avant l'extrait qui repond (rang 6) : OHADA-AUPCAP-2015 art. "
            "110, OHADA-AUDCG-2010 art. 228, CODE-CIVIL art. 107",
            "extrait qui repond au rang 6, au-dela des 4 premiers extraits lus",
        ]

    def test_le_morceau_33_en_tete_le_preavis_tient(self):
        """La question anglaise, mesuree le 30/10/2026 : le morceau 33 au rang 1."""
        lus = [("LOI-1992-007", "33", MORCEAU_33), ("LOI-1992-007", "35.1", MORCEAU_35_1_REEL),
               ("CODE-CIVIL", "111")]

        verdict = juger(_de_reference("What is the notice period for dismissing an employee?"),
                        _contexte(lus, ["LOI-1992-007"]))

        assert verdict.ok and verdict.rang_article == 1

    def test_l_article_39_repond_au_licenciement_abusif_par_ses_montants(self):
        """Le morceau 35.1 repond par l'article 39 ; 35.2 (sa fin, l'art. 40) non."""
        item = _de_reference("Quelles indemnités en cas de licenciement abusif ?")

        tient = juger(item, _contexte([("LOI-1992-007", "35.1", MORCEAU_35_1_REEL)],
                                      ["LOI-1992-007"]))
        sans_35_1 = juger(item, _contexte([("LOI-1992-007", "35.2", SUITE_35_2)],
                                          ["LOI-1992-007"]))

        assert tient.ok and tient.rang_article == 1
        assert sans_35_1.raisons == [
            "aucun article qui repond n'est lu par le modele : LOI-1992-007 art. 39"
        ]


class TestLeRenvoi:
    """
    « Est puni des peines prévues à l'alinéa 1 ci-dessus, tout accès non
    autorisé » (art. 65 al. 2, morceau 65.2) : la peine est a l'alinea 1
    (morceau 65.1). La reponse n'est lue qu'avec les deux.
    """

    QUESTION = "Quelles peines pour l’accès non autorisé à un système d’information ?"

    def _juger(self, lus):
        return juger(_de_reference(self.QUESTION), _contexte(lus, ["LOI-2010-012"]))

    def test_l_alinea_qui_renvoie_lu_seul_ne_repond_pas(self):
        verdict = self._juger([("LOI-2010-012", "65.2", ALINEA_65_2), ("LOI-2010-012", "86")])

        assert verdict.rang_article is None
        assert verdict.raisons == [
            "l'extrait qui repond (LOI-2010-012 art. 65.2, rang 1) renvoie a LOI-2010-012 art. "
            "65, que le modele ne lit pas"
        ]
        assert verdict.detail()["renvoi_de_l_article_qui_repond"] == "LOI-2010-012 art. 65"

    def test_la_reponse_est_lue_au_rang_du_plus_tardif_des_deux(self):
        """Le contexte francais du 30/10/2026 : 65.2, 86, 69.1, puis 65.1 au rang 4."""
        verdict = self._juger([("LOI-2010-012", "65.2", ALINEA_65_2), ("LOI-2010-012", "86"),
                               ("LOI-2010-012", "69.1", ARTICLE_69),
                               ("LOI-2010-012", "65.1", ALINEA_65_1)])

        assert verdict.ok and verdict.rang_article == 4

    def test_l_alinea_vise_au_dela_du_rang_4_est_un_echec(self):
        """
        Le contexte anglais du 30/10/2026 : la peine de l'alinea 1 au rang 5,
        apres l'art. 69 (10 a 100 millions, l'acces EN VIOLATION des mesures
        de securite) et l'art. 68 que porte 65.2 (10 a 50 millions).
        """
        verdict = self._juger([("LOI-2010-012", "65.2", ALINEA_65_2),
                               ("LOI-2010-012", "69.1", ARTICLE_69), ("LOI-2010-012", "86"),
                               ("LOI-2010-012", "84"), ("LOI-2010-012", "65.1", ALINEA_65_1)])

        assert verdict.rang_article == 5 and verdict.intrus == 0
        assert verdict.raisons == [
            "extrait qui repond au rang 5, au-dela des 4 premiers extraits lus"
        ]

    def test_un_autre_texte_entre_les_deux_est_lu_avant_la_reponse(self):
        verdict = self._juger([("LOI-2010-012", "65.2", ALINEA_65_2), ("PRC-9701", "40"),
                               ("LOI-2010-012", "65.1", ALINEA_65_1)])

        assert verdict.raisons == [
            "un autre texte est lu avant l'extrait qui repond (rang 3) : PRC-9701 art. 40"
        ]

    def test_l_alinea_vise_lu_sans_sa_peine_le_dit(self):
        """Un redecoupage qui laisse a 65.1 l'interception sans son amende."""
        sans_amende = "(1) Est puni [...] celui qui effectue, sans droit ni autorisation, l'interception"

        verdict = self._juger([("LOI-2010-012", "65.2", ALINEA_65_2),
                               ("LOI-2010-012", "65.1", sans_amende)])

        assert verdict.raisons == [
            "l'extrait qui repond (LOI-2010-012 art. 65.2, rang 1) renvoie a LOI-2010-012 art. "
            "65, que le modele ne lit pas (lu sans la regle demandee : LOI-2010-012 art. 65.1)"
        ]

    def test_l_article_69_seul_ne_repond_pas(self):
        """Son amende (10 a 100 millions) n'est pas celle de tout acces non autorise."""
        verdict = self._juger([("LOI-2010-012", "69.1", ARTICLE_69)])

        assert verdict.raisons == [
            "aucun article qui repond n'est lu par le modele : LOI-2010-012 art. 65.2"
        ]


class TestLesExtraitsToleres:
    """Un extrait d'un autre texte, lu et declare inoffensif avant la reponse, par question."""

    ELECTION = (("PRC-10679", "6"), ("LOI-2012-001", "116"))

    def _election(self, toleres=()):
        return _question(textes=(("PRC-10679", "LOI-2012-001"),),
                         articles_attendus=self.ELECTION, toleres_avant=toleres)

    def test_l_un_des_textes_admis_et_son_article_apres_un_extrait_tolere(self):
        """L'election en anglais : la Constitution de 1996 d'abord, avec sa ligne de statut."""
        lus = [("PRC-6283", "6"), ("LOI-2012-001", "116")]

        verdict = juger(self._election(toleres=(("PRC-6283", "6"),)),
                        _contexte(lus, ["LOI-2012-001"]))

        assert verdict.ok and verdict.rang_article == 2 and verdict.intrus == 0
        assert verdict.toleres_lus_avant == ["PRC-6283 art. 6"]
        assert verdict.resume()["toleres_avant_la_reponse"] == ["PRC-6283 art. 6"]

    def test_sans_la_tolerance_le_meme_contexte_echoue(self):
        lus = [("PRC-6283", "6"), ("LOI-2012-001", "116")]

        verdict = juger(self._election(), _contexte(lus, ["LOI-2012-001"]))

        assert verdict.raisons == [
            "un autre texte est lu avant l'extrait qui repond (rang 2) : PRC-6283 art. 6"
        ]

    def test_la_tolerance_vaut_pour_un_morceau_pas_pour_son_texte(self):
        """L'article 5 de la meme Constitution n'a ete ni lu ni juge : il echoue."""
        lus = [("PRC-6283", "5"), ("PRC-6283", "6"), ("LOI-2012-001", "116")]

        verdict = juger(self._election(toleres=(("PRC-6283", "6"),)),
                        _contexte(lus, ["LOI-2012-001"]))

        assert verdict.intrus == 1 and verdict.toleres_lus_avant == ["PRC-6283 art. 6"]
        assert verdict.raisons == [
            "un autre texte est lu avant l'extrait qui repond (rang 3) : PRC-6283 art. 5"
        ]

    def test_un_extrait_tolere_compte_dans_le_rang(self):
        lus = [("PRC-6283", "6"), ("PRC-10679", "5"), ("PRC-10679", "7"), ("LOI-2012-001", "38"),
               ("LOI-2012-001", "116")]

        verdict = juger(self._election(toleres=(("PRC-6283", "6"),)),
                        _contexte(lus, ["LOI-2012-001"]))

        assert verdict.raisons == [
            "extrait qui repond au rang 5, au-dela des 4 premiers extraits lus"
        ]


class TestLArticleNomme:
    """
    Revue du tour 3 : les questions qui nomment un article restaient vertes la
    lecture directe coupee (article 76 au rang 5, 122 au rang 7), et quand
    l'art. 318 du Code de procedure penale passait avant celui du Code penal.
    """

    ARTICLE_318 = ("PRC-4722", "318")

    def _article_318(self):
        return _question(textes=(("PRC-4722",),), question="What does Article 318 of the "
                         "Penal Code say?", articles_attendus=(self.ARTICLE_318,),
                         article_nomme=self.ARTICLE_318)

    def test_lu_en_tete_par_la_lecture_directe_la_question_tient(self):
        lus = [("PRC-4722", "318", "Vol, abus de confiance, escroquerie", "article_nomme"),
               ("LOI-2005-007", "318")]

        verdict = juger(self._article_318(), _contexte(lus, ["PRC-4722"]))

        assert verdict.ok, verdict.raisons
        assert verdict.article_nomme_en_tete is True
        assert verdict.resume()["article_nomme_en_tete"] is True

    def test_le_meme_numero_d_un_autre_texte_en_tete_est_un_echec(self):
        """L'art. 318 ecarte par _sans_doute : le Code de procedure penale lu d'abord."""
        lus = [("LOI-2005-007", "318"), ("PRC-4722", "318")]

        verdict = juger(self._article_318(), _contexte(lus, ["PRC-4722"]))

        assert verdict.article_nomme_en_tete is False
        assert verdict.raisons == [
            "un autre texte est lu avant l'extrait qui repond (rang 2) : LOI-2005-007 art. 318",
            "l'article nomme par la question (PRC-4722 art. 318) n'est pas le premier extrait "
            "lu : lu au rang 2, derriere LOI-2005-007 art. 318",
        ]
        assert verdict.detail()["article_nomme"] == "PRC-4722 art. 318"

    def test_au_rang_2_derriere_son_propre_texte_est_un_echec(self):
        """« Que dit l'article 39 » la lecture directe coupee : 35.1 au rang 4, derriere le Code."""
        article_39 = ("LOI-1992-007", "39")
        lus = [("LOI-1992-007", "126.2"), ("LOI-1992-007", "35.1", MORCEAU_35_1)]

        verdict = juger(_question(articles_attendus=(article_39,), article_nomme=article_39),
                        _contexte(lus, ["LOI-1992-007"]))

        assert verdict.raisons == [
            "l'article nomme par la question (LOI-1992-007 art. 39) n'est pas le premier extrait "
            "lu : lu au rang 2, derriere LOI-1992-007 art. 126.2"
        ]

    def test_lu_en_tete_par_la_recherche_seule_est_un_echec(self):
        """La lecture directe en panne ne doit pas se cacher derriere un bon classement."""
        lus = [("PRC-4722", "318", "Vol, abus de confiance, escroquerie", "semantic")]

        verdict = juger(self._article_318(), _contexte(lus, ["PRC-4722"]))

        assert verdict.raisons == [
            "l'article nomme par la question (PRC-4722 art. 318) est lu en tete par la recherche "
            "(origine semantic), pas par la lecture directe de l'article nomme"
        ]

    def test_non_lu_est_un_echec(self):
        verdict = juger(self._article_318(), _contexte([("PRC-4722", "319")], ["PRC-4722"]))

        assert verdict.raisons == [
            "aucun article qui repond n'est lu par le modele : PRC-4722 art. 318",
            "l'article nomme par la question (PRC-4722 art. 318) n'est pas lu "
            "(en tete : PRC-4722 art. 319)",
        ]

    def test_un_contexte_vide_le_dit(self):
        verdict = juger(self._article_318(), _contexte([], ["PRC-4722"], refus=True))

        assert verdict.article_nomme_en_tete is False
        assert "l'article nomme par la question (PRC-4722 art. 318) n'est pas lu " \
               "(contexte vide)" in verdict.raisons

    def test_l_origine_attendue_est_celle_du_chat(self):
        """Le meme nom que RAGService : un renommage d'un cote seul ferait tout echouer."""
        from app.services import rag_service

        assert controle._SOURCE_ARTICLE_NOMME == rag_service.SOURCE_ARTICLE_NOMME


# ==================== LE RAPPORT GARDE ====================


class _Horloge:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


def _rapport(statut="ok"):
    return {"statut": statut, "questions": 1, "en_echec": 0 if statut == "ok" else 1}


class _Fournisseur:
    """Un fournisseur d'embeddings double, au contrat de VoyageProvider."""

    label = "fournisseur double"
    fingerprint = "double|test"
    native_dim = 2048
    retryable = False

    def __init__(self, erreurs=()):
        # Les erreurs des premiers appels, dans l'ordre ; ensuite, un vecteur.
        self.erreurs = list(erreurs)
        self.appels = []

    def embed(self, texts, task_type, dim):
        self.appels.append((list(texts), task_type))
        if self.erreurs:
            raise self.erreurs.pop(0)
        return [np.ones(dim, dtype=np.float32) for _ in texts]


class TestLaSondeDuFournisseur:

    async def test_un_appel_au_fournisseur_meme_hors_de_la_memoire_des_questions(
        self, monkeypatch
    ):
        fournisseur = _Fournisseur()
        service = EmbeddingService(use_cache=True, provider=fournisseur)
        monkeypatch.setattr(search_service, "_embedding_service_instance", service)
        # La question est deja en memoire : la recherche n'appellerait plus Voyage.
        await service.generate_embedding_async("Question ?", task_type=service.TASK_QUERY)

        assert await sonder_le_fournisseur("Question ?") is None
        assert fournisseur.appels == [(["Question ?"], service.TASK_QUERY)] * 2

    async def test_un_appel_lent_isole_ne_fait_pas_echouer_la_sonde(self, monkeypatch):
        fournisseur = _Fournisseur(erreurs=[TimeoutError("lecture")])
        monkeypatch.setattr(search_service, "_embedding_service_instance",
                            EmbeddingService(use_cache=True, provider=fournisseur))

        assert await sonder_le_fournisseur("Question ?", pause_s=0) is None
        assert len(fournisseur.appels) == 2

    async def test_deux_echecs_de_suite_sont_une_panne_nommee_par_sa_classe(self, monkeypatch):
        fournisseur = _Fournisseur(erreurs=[ConnectionError("cle sk-secrete refusee")] * 2)
        monkeypatch.setattr(search_service, "_embedding_service_instance",
                            EmbeddingService(use_cache=True, provider=fournisseur))

        raison = await sonder_le_fournisseur("Question ?", pause_s=0)

        assert raison.startswith("fournisseur d'embeddings en echec, 2 essais (")
        assert "secrete" not in raison
        assert len(fournisseur.appels) == 2

    async def test_sans_service_d_embeddings_le_chat_est_en_plein_texte_seul(
        self, monkeypatch
    ):
        monkeypatch.setattr(search_service, "_embedding_service_instance", None)

        assert await sonder_le_fournisseur("Question ?") == (
            "service d'embeddings absent : le chat repondrait du plein texte seul"
        )


class TestLeRepliSurLIndex:
    """La recherche exacte du chat coupee pour la plupart des questions : la protection saute."""

    @staticmethod
    def _verifier_avec(chemins):
        suite = iter(chemins)

        async def verifier(fabrique, item):
            return Verdict(question=item, chemin=next(suite))

        return verifier

    @pytest.mark.parametrize("chemins, alerte", [
        (["repli_hnsw", "repli_hnsw", "exact"], True),
        (["repli_hnsw", "exact"], False),
        (["repli_hnsw", "exact", "exact", "exact"], False),
    ], ids=["la-plupart", "la-moitie", "une-sur-quatre"])
    async def test_au_dela_de_la_moitie(self, chemins, alerte, monkeypatch):
        monkeypatch.setattr(controle, "verifier_une_question", self._verifier_avec(chemins))
        questions = [_question(question=f"Q{i} ?") for i in range(len(chemins))]

        rapport = await controler_le_chat(None, questions)

        assert (rapport["statut"] == "echec") is alerte
        assert rapport["en_echec"] == 0
        assert rapport["chemins_semantiques"]["repli_hnsw"] == chemins.count("repli_hnsw")
        if alerte:
            assert rapport["raisons"][0].startswith(
                "recherche exacte du chat en repli sur l'index HNSW pour 2 questions sur 3"
            )


class TestLesFaiblessesConnues:
    """
    Revue du tour 5 : comptees comme les autres, les faiblesses connues
    faisaient rendre 503 a chaque calcul, et l'index perdu ou Voyage en panne
    rendaient le meme code ; cron-job.org ne voyait plus rien de neuf. Seules
    les questions GARANTIES font le 503 ; une faiblesse est rapportee a part,
    qu'elle echoue ou qu'elle tienne de nouveau (« faiblesse résolue »).
    """

    VOL = "Quelle peine encourt l’auteur d’un vol ?"
    # Le contexte mesure du vol en francais (revue du tour 4, tour 5) : la peine
    # de l'art. 318 au rang 7, derriere le Code de justice militaire.
    LUS_DU_VOL = [("PRC-4722", "320", ARTICLE_320), ("PRC-4722", "91"), ("PRC-5695", "56"),
                  ("PRC-4722", "104"), ("PRC-5695", "54"), ("PRC-7769", "108"),
                  ("PRC-4722", "318", ARTICLE_318), ("PRC-7769", "103")]

    @staticmethod
    def _verifier_avec(contextes):
        """verifier_une_question rejoue : le verdict de chaque question sur son contexte."""

        async def verifier(fabrique, item):
            return juger(item, contextes[item.question])

        return verifier

    def _vol(self, lus=None):
        return _de_reference(self.VOL), _contexte(lus or self.LUS_DU_VOL, ["PRC-4722"])

    @staticmethod
    def _garantie(lus=(("LOI-1992-007", "35.1", MORCEAU_35_1_REEL),)):
        """La question de l'incident, garantie, et un contexte (par defaut celui ou elle tient)."""
        return _de_reference(QUESTION_DE_L_INCIDENT), _contexte(list(lus), ["LOI-1992-007"])

    async def _controler(self, monkeypatch, *paires):
        monkeypatch.setattr(controle, "verifier_une_question",
                            self._verifier_avec({q.question: c for q, c in paires}))

        async def sonde_ok():
            return None

        cache = ControleEnCache(lambda: controler_le_chat(None, [q for q, _ in paires],
                                                          sonde=sonde_ok))
        return await cache.lire()

    def test_les_questions_de_l_essai_sont_ce_qu_elles_disent(self):
        vol, contexte = self._vol()
        assert not vol.garantie and not juger(vol, contexte).ok
        incident, contexte = self._garantie()
        assert incident.garantie and juger(incident, contexte).ok

    async def test_une_faiblesse_en_echec_est_rapportee_sans_503(self, monkeypatch):
        code, rapport = await self._controler(monkeypatch, self._garantie(), self._vol())

        assert code == 200 and rapport["statut"] == "ok" and rapport["raisons"] == []
        assert (rapport["verifiees"], rapport["garanties"], rapport["en_echec"]) == (2, 1, 0)
        assert (rapport["faiblesses_connues"], rapport["faiblesses_en_echec"]) == (1, 1)
        assert rapport["echecs"] == [] and rapport["faiblesses_resolues"] == []
        faiblesse = rapport["faiblesses"][0]
        # Son rang, et ce que le modele lit avant la reponse : de quoi agir.
        assert faiblesse["question"] == self.VOL and faiblesse["etat"] == "faiblesse connue"
        assert faiblesse["rang_de_l_article"] == 7 and faiblesse["ok"] is False
        assert faiblesse["lus_avant_la_reponse"] == [
            "PRC-4722 art. 320", "PRC-4722 art. 91", "PRC-5695 art. 56", "PRC-4722 art. 104",
            "PRC-5695 art. 54", "PRC-7769 art. 108"]
        assert faiblesse["autres_textes_avant_la_reponse"] == [
            "PRC-5695 art. 56", "PRC-5695 art. 54", "PRC-7769 art. 108"]
        assert faiblesse["diagnostic_connu"] == FAIBLESSES_CONNUES[self.VOL]
        assert faiblesse["raisons"][1] == (
            "extrait qui repond au rang 7, au-dela des 4 premiers extraits lus")
        # Dans la liste de toutes les questions, elle est marquee.
        resumes = {v["question"]: v for v in rapport["questions_verifiees"]}
        assert resumes[self.VOL]["faiblesse_connue"] is True
        assert "faiblesse_connue" not in resumes[QUESTION_DE_L_INCIDENT]

    async def test_une_faiblesse_jamais_lue_donne_tout_ce_que_le_modele_lit(self, monkeypatch):
        """« What penalty does the Penal Code provide for theft? » : l'art. 318 jamais lu."""
        theft = _de_reference("What penalty does the Penal Code provide for theft?")
        lus = [("PRC-4722", "139"), ("PRC-4722", "320", ARTICLE_320), ("PRC-5695", "54")]

        code, rapport = await self._controler(
            monkeypatch, self._garantie(), (theft, _contexte(lus, ["PRC-4722"])))

        assert code == 200
        faiblesse = rapport["faiblesses"][0]
        assert faiblesse["rang_de_l_article"] is None
        assert faiblesse["lus_avant_la_reponse"] == faiblesse["lus"] == [
            "PRC-4722 art. 139", "PRC-4722 art. 320", "PRC-5695 art. 54"]

    async def test_une_faiblesse_qui_tient_est_dite_resolue_sans_503(self, monkeypatch, caplog):
        lus = [("PRC-4722", "318", ARTICLE_318), ("PRC-4722", "320", ARTICLE_320)]

        with caplog.at_level("WARNING", logger=controle.logger.name):
            code, rapport = await self._controler(monkeypatch, self._garantie(), self._vol(lus))

        assert code == 200 and rapport["statut"] == "ok" and rapport["raisons"] == []
        assert rapport["faiblesses_resolues"] == [self.VOL]
        assert (rapport["faiblesses_connues"], rapport["faiblesses_en_echec"]) == (1, 0)
        faiblesse = rapport["faiblesses"][0]
        assert faiblesse["etat"] == "faiblesse résolue" and faiblesse["ok"] is True
        assert faiblesse["rang_de_l_article"] == 1 and faiblesse["lus_avant_la_reponse"] == []
        # Une action pour le proprietaire : la sortir de la liste.
        assert any("faiblesse résolue" in r.getMessage() and "FAIBLESSES_CONNUES" in r.getMessage()
                   for r in caplog.records), caplog.records

    async def test_une_question_garantie_en_echec_fait_rendre_503_a_cote_des_faiblesses(
        self, monkeypatch
    ):
        """Le Statut de la fonction publique avant le Code du travail : une panne nouvelle."""
        perdue = self._garantie(lus=[("DEC-1994-199", "120"),
                                     ("LOI-1992-007", "35.1", MORCEAU_35_1_REEL)])

        code, rapport = await self._controler(monkeypatch, perdue, self._vol())

        assert code == 503 and rapport["statut"] == "echec"
        assert rapport["raisons"] == ["1 question(s) garantie(s) en echec sur 1"]
        assert [e["question"] for e in rapport["echecs"]] == [QUESTION_DE_L_INCIDENT]
        assert rapport["echecs"][0]["autres_textes_avant_la_reponse"] == ["DEC-1994-199 art. 120"]
        # La faiblesse reste a part : elle n'est ni dans les echecs ni dans le compte.
        assert [f["question"] for f in rapport["faiblesses"]] == [self.VOL]
        assert rapport["en_echec"] == 1 and rapport["faiblesses_en_echec"] == 1

    async def test_une_faiblesse_ne_fait_jamais_rendre_503_quelle_que_soit_sa_raison(
        self, monkeypatch
    ):
        """
        La regle : jamais. Une panne qui toucherait une faiblesse (volet
        semantique, delai, index) touche aussi les questions garanties, ou la
        sonde : c'est par elles que la route rend 503. La raison reste lue.
        """
        vol, _ = self._vol()
        en_panne = _contexte(self.LUS_DU_VOL, ["PRC-4722"], degradee=True)

        code, rapport = await self._controler(monkeypatch, self._garantie(), (vol, en_panne))

        assert code == 200
        assert "volet semantique indisponible : le chat repondrait du plein texte seul" in (
            rapport["faiblesses"][0]["raisons"])

    async def test_des_faiblesses_seules_ne_garantissent_rien(self, monkeypatch):
        """Rien de verifie n'est pas un succes : une faiblesse qui tient ne garantit rien."""
        code, rapport = await self._controler(
            monkeypatch, self._vol([("PRC-4722", "318", ARTICLE_318)]))

        assert code == 503 and rapport["raisons"] == ["aucune question garantie verifiee"]
        assert rapport["faiblesses_resolues"] == [self.VOL]

    async def test_le_repli_se_compte_sur_toutes_les_questions(self, monkeypatch):
        """Un etat de la base, pas d'une question : les faiblesses en repli comptent."""
        incident, contexte = self._garantie()
        vol, _ = self._vol()
        replis = {
            incident.question: contexte,
            vol.question: _contexte(self.LUS_DU_VOL, ["PRC-4722"], chemin="repli_hnsw"),
            "Q ?": _contexte(["LOI-1992-007"], ["LOI-1992-007"], chemin="repli_hnsw"),
        }
        monkeypatch.setattr(controle, "verifier_une_question", self._verifier_avec(replis))

        rapport = await controler_le_chat(None, [incident, vol, _question(question="Q ?")])

        assert rapport["statut"] == "echec" and rapport["en_echec"] == 0
        assert rapport["raisons"][0].startswith(
            "recherche exacte du chat en repli sur l'index HNSW pour 2 questions sur 3")


class TestLeRapportGarde:

    async def test_un_calcul_par_periode_quel_que_soit_le_nombre_d_appels(self):
        horloge = _Horloge()
        appels = []

        async def calculer():
            appels.append(horloge.t)
            return _rapport()

        cache = ControleEnCache(calculer, horloge=horloge)
        assert (await cache.lire())[0] == 200
        for _ in range(5):
            horloge.t += 60
            code, rapport = await cache.lire()
            assert code == 200 and rapport["statut"] == "ok"
        assert len(appels) == 1 and cache.calculs == 1
        assert rapport["age_s"] == 300 and rapport["prochain_calcul_dans_s"] == 1500

        horloge.t = 1000.0 + 30 * 60
        await cache.lire()
        assert len(appels) == 2

    async def test_un_echec_rend_503_et_reste_garde_toute_la_periode(self):
        horloge = _Horloge()
        appels = []

        async def calculer():
            appels.append(1)
            return _rapport("echec")

        cache = ControleEnCache(calculer, horloge=horloge)
        assert (await cache.lire())[0] == 503
        horloge.t += 29 * 60
        code, rapport = await cache.lire()
        assert code == 503 and rapport["statut"] == "echec"
        assert len(appels) == 1

    async def test_des_appels_simultanes_attendent_le_meme_calcul(self):
        appels = []

        async def calculer():
            appels.append(1)
            await asyncio.sleep(0.05)
            return _rapport()

        cache = ControleEnCache(calculer)
        resultats = await asyncio.gather(*(cache.lire() for _ in range(10)))

        assert len(appels) == 1
        assert {code for code, _ in resultats} == {200}

    async def test_un_calcul_trop_long_rend_en_cours_puis_sert_l_appel_suivant(self):
        appels = []

        async def calculer():
            appels.append(1)
            await asyncio.sleep(0.2)
            return _rapport()

        cache = ControleEnCache(calculer, attente_max_s=0.01)
        code, rapport = await cache.lire()
        # Ni succes ni echec : le premier appel apres un redemarrage ne doit
        # pas faire sonner cron-job.org pour rien.
        assert code == 202 and rapport["statut"] == "en_cours"

        await asyncio.sleep(0.3)
        code, rapport = await cache.lire()
        assert code == 200 and rapport["statut"] == "ok"
        assert len(appels) == 1

    async def test_un_calcul_trop_long_rend_le_precedent_marque_perime(self):
        horloge = _Horloge()
        lents = []

        async def calculer():
            if lents:
                await asyncio.sleep(0.2)
                return _rapport("echec")
            lents.append(1)
            return _rapport()

        cache = ControleEnCache(calculer, attente_max_s=0.01, horloge=horloge)
        assert (await cache.lire())[0] == 200
        horloge.t += 31 * 60

        code, rapport = await cache.lire()
        assert code == 200 and rapport["perime"] is True and rapport["statut"] == "ok"
        assert rapport["age_s"] == 31 * 60

        await asyncio.sleep(0.3)
        code, rapport = await cache.lire()
        assert code == 503 and rapport["statut"] == "echec" and "perime" not in rapport

    async def test_un_precedent_trop_vieux_n_est_pas_rendu(self):
        horloge = _Horloge()
        lents = []

        async def calculer():
            if lents:
                await asyncio.sleep(0.2)
            lents.append(1)
            return _rapport()

        cache = ControleEnCache(calculer, attente_max_s=0.01, horloge=horloge)
        await cache.lire()
        horloge.t += 4 * 3600

        code, rapport = await cache.lire()
        assert code == 202 and rapport["statut"] == "en_cours"
        await asyncio.sleep(0.3)

    async def test_une_erreur_n_est_nommee_que_par_sa_classe(self):
        async def calculer():
            raise RuntimeError("postgresql://jurix:motdepasse@hote-secret/jurix sk-cle-secrete")

        code, rapport = await ControleEnCache(calculer).lire()

        assert code == 503 and rapport["statut"] == "echec"
        assert rapport["raison"] == "controle en erreur (RuntimeError)"
        assert "secret" not in repr(rapport) and "motdepasse" not in repr(rapport)

    @pytest.mark.parametrize("issue", ["erreur", "delai"])
    async def test_une_erreur_ou_un_delai_ne_relance_pas_le_calcul_avant_la_periode(
        self, issue
    ):
        """Un calcul qui echoue ne doit pas laisser un inconnu en relancer un a chaque appel."""
        horloge = _Horloge()
        appels = []

        async def calculer():
            appels.append(1)
            if issue == "erreur":
                raise RuntimeError("panne")
            await asyncio.sleep(10)

        cache = ControleEnCache(calculer, delai_max_s=0.02, horloge=horloge)
        for _ in range(10):
            code, rapport = await cache.lire()
            assert code == 503 and rapport["statut"] == "echec"
            horloge.t += 60
        assert len(appels) == 1

        horloge.t = 1000.0 + 30 * 60
        await cache.lire()
        assert len(appels) == 2

    async def test_un_calcul_sans_fin_est_abandonne(self):
        async def calculer():
            await asyncio.sleep(10)

        code, rapport = await ControleEnCache(calculer, delai_max_s=0.05).lire()

        assert code == 503 and rapport["raison"] == "controle abandonne apres 0 s"

    async def test_un_appelant_qui_raccroche_n_annule_pas_le_calcul(self):
        appels = []

        async def calculer():
            appels.append(1)
            await asyncio.sleep(0.1)
            return _rapport()

        cache = ControleEnCache(calculer)
        appelant = asyncio.create_task(cache.lire())
        await asyncio.sleep(0.02)
        appelant.cancel()
        with pytest.raises(asyncio.CancelledError):
            await appelant

        code, rapport = await cache.lire()
        assert code == 200 and len(appels) == 1


# ==================== SUR LA BASE : UN PETIT CORPUS ====================


DIM = EmbeddingService.EMBEDDING_DIM
CODE_DU_TRAVAIL_ID, LOI_DE_FINANCES_ID = 1992, 2023
TABLEAU = "Chapitre 45 — Programme 512 — Dotations : 1 250 000 | 980 000 | 2 230 000. " * 20


def _unitaire(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def _aleatoire(graine):
    return _unitaire(np.random.default_rng(graine).normal(size=DIM))


VECTEUR_DE_LA_QUESTION = _aleatoire(7)


class _VecteurDeLaQuestion:
    """
    Le fournisseur d'embeddings, double : toute question a ce vecteur. Les
    `echecs` premiers appels levent, comme un Voyage lent.
    """

    TASK_QUERY = EmbeddingService.TASK_QUERY
    TASK_DOCUMENT = EmbeddingService.TASK_DOCUMENT

    def __init__(self, echecs: int = 0):
        self.appels = 0
        self.echecs = echecs

    async def generate_embedding_async(self, texte, task_type=None):
        self.appels += 1
        if self.appels <= self.echecs:
            raise TimeoutError("Pas de vecteur pour la question en 4.0 s")
        return np.array(VECTEUR_DE_LA_QUESTION, dtype=np.float32)


@pytest.fixture
def vecteur_de_la_question(monkeypatch):
    service = _VecteurDeLaQuestion()
    monkeypatch.setattr(search_service, "_embedding_service_instance", service)
    return service


async def _corpus(db_session, category_ids, *, panne: bool):
    """
    Le Code du travail (articles 37 et 39) et une loi de finances de 30
    tableaux. Sain : l'article 39 est le plus proche voisin de la question.
    En panne : les tableaux le sont, comme le parcours de l'index de
    production le 29/10/2026 ; le plein texte trouve encore le Code.
    """
    from app.models.law import Article, Law
    from app.services.legal_domain_classifier import FINANCES, TRAVAIL
    from app.services.search_vectors import REINDEX_ARTICLES_SQL, REINDEX_LAWS_SQL

    db_session.add_all([
        Law(id=CODE_DU_TRAVAIL_ID, reference="LOI-1992-007",
            title="Loi N°92/007 du 14 août 1992 portant Code du travail",
            content="Code du travail.", type="loi", language="fr", status="published",
            category_id=category_ids[TRAVAIL], publication_date=date(1992, 8, 14)),
        Law(id=LOI_DE_FINANCES_ID, reference="PRC-9449",
            title="Loi N°2022/020 du 27 décembre 2022 portant loi de finances pour 2023",
            content=TABLEAU, type="loi", language="fr", status="published",
            category_id=category_ids[FINANCES], publication_date=date(2022, 12, 27)),
    ])
    await db_session.flush()
    question = np.asarray(VECTEUR_DE_LA_QUESTION)
    bruit = np.random.default_rng(11)
    proche = lambda: _unitaire(question + 0.05 * bruit.normal(size=DIM)).tolist()  # noqa: E731
    articles = {
        "37": "En cas de licenciement abusif, le travailleur a droit à une indemnité "
              "de licenciement distincte de celle du préavis.",
        "39": "Toute rupture abusive du contrat donne lieu à des dommages-intérêts : "
              "la juridiction fixe les indemnités du licenciement abusif.",
    }
    db_session.add_all([
        Article(id=19920000 + int(n), law_id=CODE_DU_TRAVAIL_ID, number=n, content=contenu,
                order=int(n),
                embedding=(_aleatoire(int(n)).tolist() if panne or n == "37" else proche()))
        for n, contenu in articles.items()
    ] + [
        Article(id=20230000 + i, law_id=LOI_DE_FINANCES_ID, number=f"76.{i}", content=TABLEAU,
                order=i, embedding=(proche() if panne else _aleatoire(100 + i).tolist()))
        for i in range(1, 31)
    ])
    await db_session.commit()
    await db_session.execute(text(REINDEX_LAWS_SQL))
    await db_session.execute(text(REINDEX_ARTICLES_SQL))
    await db_session.commit()


@pytest.fixture
def fabrique(pg_engine):
    """Des sessions neuves sur la base de test, comme AsyncSessionLocal en production."""
    return async_sessionmaker(pg_engine, class_=AsyncSession, expire_on_commit=False)


async def _ecritures(db_session) -> Tuple[int, int, int]:
    await search_service.ecritures_en_attente(5)
    lignes = (await db_session.execute(text(
        "SELECT (SELECT count(*) FROM search_events), (SELECT count(*) FROM query_cache), "
        "(SELECT count(*) FROM conversations)"
    ))).one()
    await db_session.rollback()
    return tuple(lignes)


class TestSurLaBase:

    async def test_la_recuperation_du_chat_sans_rien_ecrire_ni_appeler_le_modele(
        self, db_session, category_ids, fabrique, vecteur_de_la_question, monkeypatch
    ):
        await _corpus(db_session, category_ids, panne=False)
        touches: List[str] = []

        class _Modele:
            """Le modele du chat : toute utilisation est relevee."""

            def __getattr__(self, nom):
                touches.append(nom)
                raise AttributeError(nom)

        from app.core.config import settings

        # Re-ranking par le modele ACTIVE : il partirait si un modele etait branche.
        monkeypatch.setattr(settings, "RERANK_LLM_ENABLED", True)
        monkeypatch.setattr("app.services.rag_service.get_llm_service", lambda: _Modele())
        async with fabrique() as db:
            contexte, recherche = await contexte_du_chat(db, QUESTION_DE_L_INCIDENT, "fr")
            await db.rollback()

        assert any(c.reference == "LOI-1992-007" for c in contexte.lus)
        assert contexte.voisins and contexte.voisins[0] == (19920039, "LOI-1992-007")
        assert not contexte.semantique_degradee
        assert recherche.use_cache is False
        assert touches == []
        assert await _ecritures(db_session) == (0, 0, 0)

    async def test_sain_la_question_tient_et_l_index_rend_les_vrais_voisins(
        self, db_session, category_ids, fabrique, vecteur_de_la_question
    ):
        await _corpus(db_session, category_ids, panne=False)
        item = _question(rappel=True, articles_attendus=(("LOI-1992-007", "37"),
                                                         ("LOI-1992-007", "39")))

        verdict = await verifier_une_question(fabrique, item)

        assert verdict.ok, verdict.raisons
        assert verdict.rang_article is not None and verdict.rang_semantique == 1
        assert verdict.rappels == {200: 1.0, 400: 1.0}
        assert verdict.essais == 1

    async def test_la_forme_de_l_incident_est_vue_meme_si_le_plein_texte_sauve_le_contexte(
        self, db_session, category_ids, fabrique, vecteur_de_la_question
    ):
        await _corpus(db_session, category_ids, panne=True)

        verdict = await verifier_une_question(fabrique, _question())

        assert not verdict.ok
        assert "aucun texte attendu parmi les 10 premiers voisins du volet semantique" in (
            verdict.raisons
        )
        # Le Code du travail ne vient qu'apres les 30 tableaux.
        assert verdict.rang_semantique > 10

    async def test_un_index_qui_se_perd_fait_echouer_le_rappel(
        self, db_session, category_ids, fabrique, vecteur_de_la_question, monkeypatch
    ):
        await _corpus(db_session, category_ids, panne=False)

        async def index_perdu(db, vecteur, k=10, ef_search=400):
            # Des articles qu'aucune recherche exacte ne rend.
            return list(range(-10, 0))

        monkeypatch.setattr(controle, "voisins_par_l_index", index_perdu)
        verdict = await verifier_une_question(fabrique, _question(rappel=True))

        assert verdict.rappels == {200: 0.0, 400: 0.0}
        assert verdict.raisons == [
            "rappel@10 de l'index HNSW a ef_search 200 : 0.0, sous le seuil de 0.8",
            "rappel@10 de l'index HNSW a ef_search 400 : 0.0, sous le seuil de 0.8",
        ]

    async def test_un_volet_semantique_en_panne_une_fois_a_droit_a_un_second_essai(
        self, db_session, category_ids, fabrique, monkeypatch
    ):
        """Un appel lent isole a Voyage, au premier calcul : pas d'alerte."""
        await _corpus(db_session, category_ids, panne=False)
        monkeypatch.setattr(controle, "PAUSE_ENTRE_DEUX_ESSAIS_S", 0)
        service = _VecteurDeLaQuestion(echecs=1)
        monkeypatch.setattr(search_service, "_embedding_service_instance", service)

        verdict = await verifier_une_question(fabrique, _question())

        assert verdict.ok, verdict.raisons
        assert verdict.essais == 2 and service.appels == 2

    async def test_deux_pannes_de_suite_sont_un_echec(
        self, db_session, category_ids, fabrique, monkeypatch
    ):
        await _corpus(db_session, category_ids, panne=False)
        monkeypatch.setattr(controle, "PAUSE_ENTRE_DEUX_ESSAIS_S", 0)
        monkeypatch.setattr(search_service, "_embedding_service_instance",
                            _VecteurDeLaQuestion(echecs=10))

        verdict = await verifier_une_question(fabrique, _question(rappel=True))

        assert not verdict.ok and verdict.essais == 2
        assert "volet semantique indisponible : le chat repondrait du plein texte seul" in (
            verdict.raisons
        )
        # Sans vecteur, pas de mesure de l'index.
        assert verdict.rappels is None

    async def test_le_rapport_nomme_les_questions_en_echec_sans_message_d_erreur(
        self, db_session, category_ids, fabrique, vecteur_de_la_question, monkeypatch
    ):
        await _corpus(db_session, category_ids, panne=False)
        tient = _question(rappel=True)
        manque = _question(
            textes=(("CODE-CIVIL",),), question="Quelles sont les causes de divorce ?"
        )
        casse = _question(question="Une question qui casse la recuperation ?")
        recuperer = controle.contexte_du_chat

        async def contexte_qui_casse(db, question, langue):
            if question == casse.question:
                raise RuntimeError("postgresql://jurix:motdepasse@hote-secret/jurix")
            return await recuperer(db, question, langue)

        monkeypatch.setattr(controle, "contexte_du_chat", contexte_qui_casse)
        rapport = await controler_le_chat(fabrique, [tient, manque, casse], concurrence=2)

        assert rapport["statut"] == "echec"
        assert rapport["raisons"] == ["2 question(s) garantie(s) en echec sur 3"]
        assert (rapport["questions"], rapport["verifiees"], rapport["en_echec"]) == (3, 3, 2)
        assert rapport["faiblesses"] == [] and rapport["faiblesses_connues"] == 0
        echecs = {e["question"]: e for e in rapport["echecs"]}
        assert set(echecs) == {manque.question, casse.question}
        assert echecs[manque.question]["attendus"] == [["CODE-CIVIL"]]
        assert echecs[casse.question]["raisons"] == ["recuperation en echec (RuntimeError)"]
        assert rapport["rappel_hnsw"] == [{
            "question": tient.question, "rappel_a_10": {"200": 1.0, "400": 1.0}, "seuil": 0.8,
        }]
        assert "motdepasse" not in repr(rapport) and "hote-secret" not in repr(rapport)
        assert await _ecritures(db_session) == (0, 0, 0)

    async def test_tout_tient_le_rapport_est_ok(
        self, db_session, category_ids, fabrique, vecteur_de_la_question
    ):
        await _corpus(db_session, category_ids, panne=False)

        rapport = await controler_le_chat(fabrique, [_question()])

        assert rapport["statut"] == "ok" and rapport["raisons"] == []
        assert rapport["en_echec"] == 0 and rapport["verifiees"] == 1
        assert rapport["questions_verifiees"][0]["ok"] is True

    async def test_au_dela_du_delai_les_questions_restantes_echouent(self, monkeypatch):
        async def lente(fabrique, item):
            await asyncio.sleep(5)
            return Verdict(question=item)

        monkeypatch.setattr(controle, "verifier_une_question", lente)
        rapport = await controler_le_chat(None, [_question()], delai_s=0.05)

        assert rapport["statut"] == "echec"
        assert rapport["echecs"][0]["raisons"] == ["non verifiee : le controle a depasse 0 s"]

    async def test_une_sonde_en_echec_fait_echouer_le_controle_sans_verifier_les_questions(
        self, db_session, category_ids, fabrique, vecteur_de_la_question
    ):
        await _corpus(db_session, category_ids, panne=False)

        async def sonde_ok():
            return None

        async def sonde_en_panne():
            return "fournisseur d'embeddings en echec, 2 essais (ConnectError)"

        sain = await controler_le_chat(fabrique, [_question()], sonde=sonde_ok)
        assert sain["statut"] == "ok" and sain["fournisseur_d_embeddings"] == "ok"

        rapport = await controler_le_chat(fabrique, [_question()], sonde=sonde_en_panne)
        assert rapport["statut"] == "echec"
        assert (rapport["questions"], rapport["verifiees"], rapport["en_echec"]) == (1, 0, 0)
        assert rapport["fournisseur_d_embeddings"] == (
            "fournisseur d'embeddings en echec, 2 essais (ConnectError)"
        )
        assert rapport["raisons"] == [
            "fournisseur d'embeddings en echec, 2 essais (ConnectError) : questions non verifiees"
        ]

    async def test_une_sonde_qui_ne_rend_pas_la_main_est_bornee(self, monkeypatch):
        async def sonde_sans_fin():
            await asyncio.sleep(10)

        rapport = await controler_le_chat(None, [_question()], delai_s=0.05,
                                          sonde=sonde_sans_fin)

        assert rapport["statut"] == "echec" and rapport["verifiees"] == 0
        assert rapport["fournisseur_d_embeddings"] == "sonde du fournisseur sans reponse en 0 s"

    async def test_sans_sonde_le_rapport_le_dit(
        self, db_session, category_ids, fabrique, vecteur_de_la_question
    ):
        await _corpus(db_session, category_ids, panne=False)

        rapport = await controler_le_chat(fabrique, [_question()])

        assert rapport["fournisseur_d_embeddings"] == "non sonde"

    async def test_aucune_question_n_est_pas_un_succes(self):
        rapport = await controler_le_chat(None, [])

        assert rapport["statut"] == "echec" and rapport["questions"] == 0
        assert rapport["raisons"] == ["aucune question verifiee"]
