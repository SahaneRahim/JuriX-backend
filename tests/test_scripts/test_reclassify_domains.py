"""
Le reclassement par lots : journal reprenable, quota, application relue.

Le classifieur est double : ces tests portent sur ce que le script fait des
verdicts — journal, reprise sans rappel, arret propre sur quota, rien
d'applique sous le seuil, rien d'ecrit en simulation.

Usage:
    pytest tests/test_scripts/test_reclassify_domains.py -v
"""

import csv
import json

import pytest
from sqlalchemy import text

from app.models.law import Category, Law
from app.services.legal_domain_classifier import (
    ADMINISTRATIF,
    CANONICAL_DOMAINS,
    FINANCES,
    FONCTION_PUBLIQUE,
    INTERNATIONAL,
    ClassementIndisponible,
    DomainResult,
    ResultatLot,
)
from scripts.maintenance import reclassify_domains as script


class _Classeur:
    """Rend, dans l'ordre, les verdicts programmes (un domaine, None ou une exception)."""

    modele = "doublure"

    def __init__(self, *reponses):
        self.reponses = list(reponses)
        self.lots = []

    def classer_lot(self, documents, *, attente_max=None):
        self.lots.append(list(documents))
        reponse = self.reponses.pop(0)
        if isinstance(reponse, Exception):
            raise reponse
        verdicts = [
            None if domaine is None else DomainResult(
                domain=domaine, confidence=confiance, rule="groq:doublure", source="groq",
                runners_up=((INTERNATIONAL, 0.5),),
            )
            for domaine, confiance in reponse
        ]
        return ResultatLot(verdicts, jetons=300, requetes=1, modele="doublure")


@pytest.fixture
def base(sync_db_session):
    """Les 14 domaines et 5 lois publiees, toutes rangees en Administratif."""
    for ordre, nom in enumerate(CANONICAL_DOMAINS):
        sync_db_session.add(Category(name=nom, description=nom, display_order=ordre))
    sync_db_session.commit()
    ids = dict(sync_db_session.execute(text("SELECT name, id FROM categories")).all())
    for i in range(1, 6):
        sync_db_session.add(Law(
            reference=f"R-{i}", title=f"Décret n°{i}", type="decret", content="Contenu.",
            language="fr", status="published", category_id=ids[ADMINISTRATIF],
        ))
    sync_db_session.commit()
    lois = [loi.id for loi in sync_db_session.query(Law).order_by(Law.id)]
    return sync_db_session, ids, lois


def _journal(chemin):
    return [json.loads(ligne) for ligne in chemin.read_text(encoding="utf-8").splitlines()]


class TestClasser:
    def test_ecrit_un_verdict_par_loi(self, base, tmp_path):
        session, ids, lois = base
        journal = tmp_path / "verdicts.jsonl"
        classeur = _Classeur(
            [(FINANCES, 0.9), (FONCTION_PUBLIQUE, 0.8), (None, None)],
            [(FINANCES, 0.4), (ADMINISTRATIF, 0.7)],
        )

        code = script.classer(session, classeur, journal=journal, taille_lot=3)

        assert code == 0
        assert [len(lot) for lot in classeur.lots] == [3, 2]
        lignes = _journal(journal)
        assert [ligne["law_id"] for ligne in lignes] == lois
        assert lignes[0]["domaine"] == FINANCES
        assert lignes[0]["secondaires"] == [INTERNATIONAL]
        assert lignes[2]["statut"] == "a_revoir"
        assert lignes[2]["domaine"] is None
        assert {ligne["consignes"] for ligne in lignes} == {script.VERSION_DES_CONSIGNES}
        # Classer n'ecrit rien en base.
        assert session.query(Law).filter(Law.category_id == ids[ADMINISTRATIF]).count() == 5

    def test_reprise_sans_rappel(self, base, tmp_path):
        session, ids, lois = base
        journal = tmp_path / "verdicts.jsonl"
        script.classer(
            session, _Classeur([(FINANCES, 0.9)] * 5), journal=journal, taille_lot=5
        )

        relance = _Classeur()
        code = script.classer(session, relance, journal=journal, taille_lot=5)

        assert code == 0
        assert relance.lots == []

    def test_quota_arret_propre_code_4(self, base, tmp_path):
        session, ids, lois = base
        journal = tmp_path / "verdicts.jsonl"
        classeur = _Classeur(
            [(FINANCES, 0.9), (FINANCES, 0.9)],
            ClassementIndisponible("TPD", retry_after=3600, quota=True),
        )

        code = script.classer(session, classeur, journal=journal, taille_lot=2)

        assert code == script.SORTIE_QUOTA
        # Le lot deja classe est garde : la relance repart de la troisieme loi.
        assert [ligne["law_id"] for ligne in _journal(journal)] == lois[:2]
        relance = _Classeur([(FINANCES, 0.9)] * 3)
        script.classer(session, relance, journal=journal, taille_lot=5)
        assert [d.titre for d in relance.lots[0]] == ["Décret n°3", "Décret n°4", "Décret n°5"]

    def test_saturation_courte_attendue_sur_place(self, base, tmp_path):
        session, ids, lois = base
        sommeils = []
        classeur = _Classeur(
            ClassementIndisponible("RPM", retry_after=20),
            [(FINANCES, 0.9)] * 5,
        )

        code = script.classer(
            session, classeur, journal=tmp_path / "v.jsonl", taille_lot=5, dormir=sommeils.append
        )

        assert code == 0
        assert sommeils == [20]

    def test_panne_code_3(self, base, tmp_path):
        session, ids, lois = base
        classeur = _Classeur(ClassementIndisponible("503"))

        code = script.classer(session, classeur, journal=tmp_path / "v.jsonl", taille_lot=5)

        assert code == script.SORTIE_PANNE

    def test_incertains_seulement_avec_extrait(self, base, tmp_path):
        session, ids, lois = base
        journal = tmp_path / "verdicts.jsonl"
        session.execute(
            text("INSERT INTO articles (law_id, number, content, \"order\", kind) "
                 "VALUES (:id, '1', 'Article 1er.- Est ratifié l''accord de prêt.', 1, 'article')"),
            {"id": lois[1]},
        )
        session.commit()
        script.classer(
            session,
            _Classeur([(FINANCES, 0.9), (INTERNATIONAL, 0.4), (None, None), (FINANCES, 0.95), (FINANCES, 0.9)]),
            journal=journal, taille_lot=5,
        )

        seconde = _Classeur([(FINANCES, 0.85), (FONCTION_PUBLIQUE, 0.8)])
        script.classer(
            session, seconde, journal=journal, taille_lot=15,
            longueur_extrait=500, incertains=True,
        )

        assert [d.titre for d in seconde.lots[0]] == ["Décret n°2", "Décret n°3"]
        assert seconde.lots[0][0].extrait == "Article 1er.- Est ratifié l'accord de prêt."
        derniers = script.charger_verdicts(journal)
        assert derniers[lois[1]]["domaine"] == FINANCES
        assert derniers[lois[1]]["extrait"] == 500


def _verdicts(lois, *domaines):
    """{law_id: verdict} : (domaine, confiance) ou None pour « a revoir »."""
    verdicts = {}
    for loi, verdict in zip(lois, domaines):
        domaine, confiance = verdict if verdict else (None, None)
        verdicts[loi] = {
            "law_id": loi, "statut": "ok" if verdict else "a_revoir", "domaine": domaine,
            "secondaires": [INTERNATIONAL] if verdict else [], "confiance": confiance,
        }
    return verdicts


def _csv(chemin):
    with chemin.open(encoding="utf-8") as fichier:
        return list(csv.DictReader(fichier))


class TestAppliquer:
    def test_dry_run_n_ecrit_rien_en_base(self, base, tmp_path):
        session, ids, lois = base
        verdicts = _verdicts(lois, (FINANCES, 0.9), (FONCTION_PUBLIQUE, 0.8))

        bilan = script.appliquer(session, verdicts, dossier=tmp_path, force=True, dry_run=True)

        session.expire_all()
        assert bilan.changees == 2
        assert session.query(Law).filter(Law.category_id == ids[ADMINISTRATIF]).count() == 5
        assert [ligne["apres"] for ligne in _csv(tmp_path / "changements.csv")] == [
            FINANCES, FONCTION_PUBLIQUE,
        ]

    def test_force_applique_categorie_confiance_et_suggestions(self, base, tmp_path):
        session, ids, lois = base
        verdicts = _verdicts(lois, (FINANCES, 0.9))

        script.appliquer(session, verdicts, dossier=tmp_path, force=True)

        session.expire_all()
        loi = session.get(Law, lois[0])
        assert loi.category_id == ids[FINANCES]
        assert loi.category_confidence == pytest.approx(0.9)
        assert loi.suggested_categories == [ids[FINANCES], ids[INTERNATIONAL]]

    def test_sans_force_une_categorie_posee_est_gardee(self, base, tmp_path):
        session, ids, lois = base
        session.execute(text("UPDATE laws SET category_id = NULL WHERE id = :id"), {"id": lois[1]})
        session.commit()
        verdicts = _verdicts(lois, (FINANCES, 0.9), (FONCTION_PUBLIQUE, 0.9))

        bilan = script.appliquer(session, verdicts, dossier=tmp_path)

        session.expire_all()
        assert bilan.protegees == 1
        assert session.get(Law, lois[0]).category_id == ids[ADMINISTRATIF]
        assert session.get(Law, lois[1]).category_id == ids[FONCTION_PUBLIQUE]

    def test_a_revoir_et_sous_le_seuil_non_appliques(self, base, tmp_path):
        """Jamais d'Administratif par defaut, ni d'autre domaine invente."""
        session, ids, lois = base
        session.execute(text("UPDATE laws SET category_id = NULL"))
        session.commit()
        verdicts = _verdicts(lois, None, (FINANCES, 0.5), (FINANCES, 0.6))

        bilan = script.appliquer(session, verdicts, dossier=tmp_path, seuil=0.6, force=True)

        session.expire_all()
        assert bilan.a_revoir == 2
        assert session.get(Law, lois[0]).category_id is None
        assert session.get(Law, lois[1]).category_id is None
        assert session.get(Law, lois[2]).category_id == ids[FINANCES]
        revoir = _csv(tmp_path / "a_revoir.csv")
        assert [int(ligne["law_id"]) for ligne in revoir] == lois[:2]
        assert revoir[0]["statut"] == "a_revoir"


class TestJournal:
    def test_le_dernier_verdict_l_emporte_et_une_ligne_tronquee_est_ignoree(self, tmp_path):
        journal = tmp_path / "verdicts.jsonl"
        journal.write_text(
            '{"law_id": 1, "statut": "ok", "domaine": "A"}\n'
            '{"law_id": 1, "statut": "ok", "domaine": "B"}\n'
            '{"law_id": 2, "sta\n',
            encoding="utf-8",
        )

        assert script.charger_verdicts(journal) == {
            1: {"law_id": 1, "statut": "ok", "domaine": "B"}
        }

    def test_journal_absent(self, tmp_path):
        assert script.charger_verdicts(tmp_path / "absent.jsonl") == {}
