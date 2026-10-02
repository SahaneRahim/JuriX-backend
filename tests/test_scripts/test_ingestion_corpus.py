"""
Tests des deux passes de l'ingestion : extraction supervisee, puis indexation.

L'ingestion du corpus dure une vingtaine d'heures, sans surveillance. Chaque test
fige une facon de perdre ce temps : un processus qui meurt et fait tout
reprendre, un lot toxique qui tue chaque nouveau processus, un echec qui laisse
une loi bloquee en 'processing', un document indexe deux fois.

Usage:
    pytest tests/test_scripts/test_ingestion_corpus.py -v
"""

import json
import os
import re
import signal
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from app.models.law import Article, Law
from app.services import pdf_extraction_service
from app.services.docling_extraction import (
    TENTATIVES_MAX,
    DoclingPdfExtractor,
    sha256_fichier,
)
from scripts import extraire_corpus, ingest_corpus
from scripts.extraire_corpus import Document, superviser

# ==================== DOUBLURES ====================


class _Statut:
    value = "success"


class _Document:
    """Document image : pages numerotees a partir de 1 dans le lot."""

    def __init__(self, debut, fin):
        self.pages = {
            n - debut + 1: f"Article {n}.- Le present article fixe une regle applicable a tous, "
                           f"enregistree a la page {n} du texte, assez longue pour etre indexee."
            for n in range(debut, fin + 1)
        }

    def export_to_markdown(self, page_no, **kwargs):
        return self.pages.get(page_no, "")


class _Resultat:
    def __init__(self, debut, fin):
        self.status = _Statut()
        self.document = _Document(debut, fin)
        self.errors = []


class _Convertisseur:
    def __init__(self):
        self.appels = []

    def convert(self, source, raises_on_error=True):
        debut, fin = map(int, re.search(r"_p(\d+)-(\d+)\.tiff$", source.name).groups())
        self.appels.append((debut, fin))
        return _Resultat(debut, fin)


@pytest.fixture(autouse=True)
def _sigint_restaure():
    """
    travailler() ignore SIGINT, comme le vrai processus enfant. Appele ici dans
    le processus des tests, il laisserait Ctrl-C sans effet sur le reste de la
    suite : on restaure le gestionnaire.
    """
    avant = signal.getsignal(signal.SIGINT)
    yield
    signal.signal(signal.SIGINT, avant)


def _extracteur(cache: Path) -> DoclingPdfExtractor:
    ext = DoclingPdfExtractor(cache_dir=str(cache), pages_par_lot=2)
    ext.doublure = _Convertisseur()
    ext._construire = lambda: ext.doublure
    return ext


def _pdf(chemin: Path, nb_pages: int) -> Path:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument.new()
    for _ in range(nb_pages):
        pdf.new_page(595, 842)
    pdf.save(str(chemin))
    pdf.close()
    return chemin


def _document(chemin: Path, doc_id="77", pages=4, titre="Décret N° 2024/191 du 4 juin 2024 portant organisation"):
    return Document(doc_id=doc_id, chemin=str(chemin), titre=titre,
                    sha256=sha256_fichier(chemin), pages=pages)


# ==================== PASSE 1 : SUPERVISION ====================


class TestSupervision:

    def test_processus_mort_compte_une_tentative_et_la_suite_reprend(self, tmp_path):
        ext = _extracteur(tmp_path / "cache")
        doc = _document(_pdf(tmp_path / "d.pdf", 4))
        suivi = tmp_path / "suivi"
        suivi.mkdir()
        lancements = []

        def lancer(groupe, dossier, rss_max):
            lancements.append(1)
            if len(lancements) == 1:
                # Le lot 1-2 est fait, puis le processus meurt pendant le lot 3-4
                ext.traiter_lot(Path(doc.chemin), doc.sha256, 1, 2)
                (dossier / "en_cours.json").write_text(
                    json.dumps({**asdict(doc), "debut": 3, "fin": 4}), encoding="utf-8"
                )
                return -9
            for debut, fin in ext.etat(Path(doc.chemin), doc.sha256, doc.pages).lots_a_faire:
                ext.traiter_lot(Path(doc.chemin), doc.sha256, debut, fin)
            return 0

        assert superviser([doc], ext, suivi, lancer=lancer) == 0

        assert ext.doublure.appels == [(1, 2), (3, 4)], "le lot 1-2 n'est pas refait"
        lot = json.loads(next(ext.dossier_cache.rglob("lot-0003-0004.json")).read_text())
        assert lot["statut"] == "ok"
        assert lot["tentatives"] == 2, "la mort du processus compte une tentative"
        progression = json.loads((suivi / "progression.json").read_text())
        assert progression["documents"]["termines"] == 1
        assert progression["processus"]["morts"] == 1

    def test_un_lot_toxique_finit_abandonne(self, tmp_path):
        """
        Un lot qui tue chaque processus ne bloque pas l'ingestion entiere : il
        est repris page a page, puis chaque page qui tue encore est abandonnee.
        """
        ext = _extracteur(tmp_path / "cache")
        doc = _document(_pdf(tmp_path / "d.pdf", 2), pages=2)
        suivi = tmp_path / "suivi"
        suivi.mkdir()
        lancements = []

        def lancer(groupe, dossier, rss_max):
            # Comme le vrai processus : il marque le premier lot restant, et meurt
            lancements.append(1)
            debut, fin = ext.etat(Path(doc.chemin), doc.sha256, 2).lots_a_faire[0]
            (dossier / "en_cours.json").write_text(
                json.dumps({**asdict(doc), "debut": debut, "fin": fin}), encoding="utf-8"
            )
            return -11

        assert superviser([doc], ext, suivi, lancer=lancer) == 0
        assert len(lancements) == 3 * TENTATIVES_MAX, "le lot, puis chacune de ses pages"
        assert ext.etat(Path(doc.chemin), doc.sha256, 2).lots_abandonnes == [(1, 1), (2, 2)]

    def test_page_toxique_n_emporte_pas_les_pages_saines(self, tmp_path):
        ext = _extracteur(tmp_path / "cache")
        doc = _document(_pdf(tmp_path / "d.pdf", 2), pages=2)
        suivi = tmp_path / "suivi"
        suivi.mkdir()

        def lancer(groupe, dossier, rss_max):
            debut, fin = ext.etat(Path(doc.chemin), doc.sha256, 2).lots_a_faire[0]
            if fin >= 2:   # la page 2 tue le processus
                (dossier / "en_cours.json").write_text(
                    json.dumps({**asdict(doc), "debut": debut, "fin": fin}), encoding="utf-8"
                )
                return -11
            ext.traiter_lot(Path(doc.chemin), doc.sha256, debut, fin)
            return 0

        assert superviser([doc], ext, suivi, lancer=lancer) == 0
        etat = ext.etat(Path(doc.chemin), doc.sha256, 2)
        assert etat.lots_faits == [(1, 1)], "la page saine du lot est extraite"
        assert etat.lots_abandonnes == [(2, 2)]

    def test_processus_qui_ecrirait_dans_un_autre_cache_arrete_tout(self, tmp_path):
        """
        .env ou paquets modifies pendant l'extraction : l'enfant ecrirait dans
        un autre cache, et le superviseur relancerait le meme groupe sans fin.
        """
        ext = _extracteur(tmp_path / "cache")
        doc = _document(_pdf(tmp_path / "d.pdf", 2), pages=2)
        suivi = tmp_path / "suivi"
        suivi.mkdir()
        lancements = []

        def lancer(groupe, dossier, rss_max):
            lancements.append(1)
            return extraire_corpus.SORTIE_AUTRE_CACHE

        assert superviser([doc], ext, suivi, lancer=lancer) == 2
        assert len(lancements) == 1

    def test_l_enfant_refuse_un_cache_different(self, tmp_path, monkeypatch):
        ext = _extracteur(tmp_path / "cache")
        monkeypatch.setattr(extraire_corpus, "DoclingPdfExtractor", lambda: ext)
        monkeypatch.setattr(extraire_corpus, "configurer_journal", lambda chemin: None)
        doc = _document(_pdf(tmp_path / "d.pdf", 2), pages=2)
        suivi = tmp_path / "suivi"
        suivi.mkdir()

        code = extraire_corpus.travailler(
            [asdict(doc)], str(suivi), 5000, cache_attendu=str(tmp_path / "autre")
        )

        assert code == extraire_corpus.SORTIE_AUTRE_CACHE
        assert ext.etat(Path(doc.chemin), doc.sha256, 2).lots_faits == [], "rien n'est converti"

    def test_morts_en_boucle_hors_conversion_arretent_tout(self, tmp_path):
        """Pilote CUDA casse, modeles introuvables : inutile d'insister."""
        ext = _extracteur(tmp_path / "cache")
        doc = _document(_pdf(tmp_path / "d.pdf", 2), pages=2)
        suivi = tmp_path / "suivi"
        suivi.mkdir()

        code = superviser([doc], ext, suivi, lancer=lambda g, d, r: 1, morts_consecutives_max=3)

        assert code == 2

    def test_arret_demande(self, tmp_path):
        ext = _extracteur(tmp_path / "cache")
        doc = _document(_pdf(tmp_path / "d.pdf", 2), pages=2)
        suivi = tmp_path / "suivi"
        suivi.mkdir()
        (suivi / "STOP").touch()

        def lancer(*args):
            raise AssertionError("aucun processus ne doit partir apres STOP")

        assert superviser([doc], ext, suivi, lancer=lancer) == 1

    def test_les_petits_documents_d_abord(self, tmp_path):
        docs = [
            Document("3", "c", "t", "s3", 40),
            Document("1", "a", "t", "s1", 1),
            Document("2", "b", "t", "s2", 5),
        ]
        assert [d.doc_id for d in extraire_corpus.selectionner(docs, None, None)] == ["1", "2", "3"]
        assert [d.doc_id for d in extraire_corpus.selectionner(docs, None, 2)] == ["1", "2"]
        assert [d.doc_id for d in extraire_corpus.selectionner(docs, ["3"], None)] == ["3"]

    def test_travailler_s_arrete_sur_stop(self, tmp_path, monkeypatch):
        """Le processus enfant termine son lot puis s'arrete si STOP est pose."""
        ext = _extracteur(tmp_path / "cache")
        doc = _document(_pdf(tmp_path / "d.pdf", 4))
        monkeypatch.setattr(extraire_corpus, "DoclingPdfExtractor", lambda: ext)
        monkeypatch.setattr(extraire_corpus, "configurer_journal", lambda chemin: None)
        suivi = tmp_path / "suivi"
        suivi.mkdir()
        original = ext.traiter_lot

        def _puis_stop(*args):
            lot = original(*args)
            (suivi / "STOP").touch()
            return lot

        ext.traiter_lot = _puis_stop
        extraire_corpus.travailler([asdict(doc)], str(suivi), rss_max_mo=10**6)

        assert ext.doublure.appels == [(1, 2)]
        assert not (suivi / "en_cours.json").exists()


# ==================== PASSE 2 : INDEXATION ====================


@pytest.fixture
def stockage(tmp_path, monkeypatch):
    from app.services import file_upload_service

    service = file_upload_service.FileUploadService(
        storage_path=str(tmp_path / "uploads"), max_size_mb=50,
        allowed_formats=("pdf", "docx"), clamav_enabled=False, cleanup_hours=24,
    )
    monkeypatch.setattr(file_upload_service, "get_upload_service", lambda: service)
    return service.storage_path


@pytest.fixture
def extraction_en_cache(tmp_path, monkeypatch):
    """
    L'extracteur du pipeline lit le MEME cache que celui de la passe 1, et ne
    doit jamais charger Docling.
    """
    from app.core.config import settings

    cache = tmp_path / "ocr_cache"
    monkeypatch.setattr(settings, "OCR_CACHE_DIR", str(cache))
    monkeypatch.setattr(settings, "PDF_EXTRACTION_ENGINE", "docling")
    pdf_extraction_service._extracteur_docling.cache_clear()
    yield cache
    pdf_extraction_service._extracteur_docling.cache_clear()


@pytest.fixture
def embeddings_doubles(monkeypatch):
    """Aucun modele ni reseau : des vecteurs unitaires deterministes."""
    import app.services.embedding_service as module

    class _Fournisseur:
        label = "doublure"
        fingerprint = "doublure|ingestion"

    class _Service:
        MAX_TEXT_LENGTH = 10000
        provider = _Fournisseur()

        def __init__(self, *args, **kwargs):
            pass

        def generate_batch_embeddings(self, texts, **kwargs):
            rng = np.random.default_rng(0)
            out = []
            for _ in texts:
                v = rng.normal(size=768)
                out.append((v / np.linalg.norm(v)).astype(np.float32))
            return out

    monkeypatch.setattr(module, "EmbeddingService", _Service)


class TestIndexation:

    def test_document_non_extrait_reste_en_attente(self, sync_db_session, tmp_path, stockage):
        ext = _extracteur(tmp_path / "cache")
        doc = _document(_pdf(tmp_path / "d.pdf", 2), pages=2)

        def traiter(*args, **kwargs):
            raise AssertionError("un document non extrait ne doit pas etre indexe")

        bilan = ingest_corpus.indexer([doc], ext, stockage, traiter=traiter)

        assert bilan["en_attente"] == 1
        assert sync_db_session.query(Law).count() == 0

    def test_de_bout_en_bout_puis_ignore_a_la_relance(
        self, sync_db_session, tmp_path, stockage, extraction_en_cache, embeddings_doubles
    ):
        ext = _extracteur(extraction_en_cache)
        doc = _document(_pdf(tmp_path / "d.pdf", 4))
        ext.extraire(Path(doc.chemin))          # passe 1

        bilan = ingest_corpus.indexer([doc], ext, stockage)

        assert bilan["publies"] == 1
        loi = sync_db_session.query(Law).filter(Law.reference == "PRC-77").one()
        assert loi.status == "published"
        assert loi.type == "décret"
        assert loi.publication_date == date(2024, 6, 4)
        articles = sync_db_session.query(Article).filter(Article.law_id == loi.id).all()
        assert {a.page_number for a in articles if a.number.isdigit()} == {1, 2, 3, 4}
        copie = stockage / f"{loi.file_id}.pdf"
        assert copie.is_file()
        assert copie.stat().st_mtime > os.stat(doc.chemin).st_mtime - 1

        relance = ingest_corpus.indexer([doc], ext, stockage)
        assert relance == {"publies": 0, "deja_publies": 1, "en_attente": 0, "doublons": 0, "echecs": 0}

    def test_un_echec_est_repris_a_la_relance(
        self, sync_db_session, tmp_path, stockage, extraction_en_cache, embeddings_doubles,
        monkeypatch,
    ):
        """
        Une loi en echec passait autrefois 'processing' pour toujours, et la
        relance la sautait. Elle passe 'refused', puis la relance la reprend.
        """
        from app.tasks import process_law

        ext = _extracteur(extraction_en_cache)
        doc = _document(_pdf(tmp_path / "d.pdf", 2), pages=2)
        ext.extraire(Path(doc.chemin))
        original = process_law._split_and_save_articles

        def _panne(*args, **kwargs):
            raise RuntimeError("panne simulee du decoupage")

        monkeypatch.setattr(process_law, "_split_and_save_articles", _panne)
        premier = ingest_corpus.indexer([doc], ext, stockage)
        sync_db_session.expire_all()
        loi = sync_db_session.query(Law).filter(Law.reference == "PRC-77").one()
        assert premier["echecs"] == 1
        assert loi.status == "refused"
        assert "panne simulee" in loi.processing_error

        monkeypatch.setattr(process_law, "_split_and_save_articles", original)
        second = ingest_corpus.indexer([doc], ext, stockage)
        sync_db_session.expire_all()
        assert second["publies"] == 1
        assert sync_db_session.query(Law).filter(Law.reference == "PRC-77").one().status == "published"

    def test_pages_abandonnees_signalees_sur_la_loi(
        self, sync_db_session, tmp_path, stockage, extraction_en_cache, embeddings_doubles
    ):
        ext = _extracteur(extraction_en_cache)
        doc = _document(_pdf(tmp_path / "d.pdf", 4))
        ext.traiter_lot(Path(doc.chemin), doc.sha256, 1, 2)
        for lot in ((3, 4), (3, 3), (4, 4)):
            for _ in range(TENTATIVES_MAX):
                ext.noter_echec(Path(doc.chemin), *lot, "page illisible")

        ingest_corpus.indexer([doc], ext, stockage)

        loi = sync_db_session.query(Law).filter(Law.reference == "PRC-77").one()
        assert loi.status == "published"
        assert "Pages non extraites : 3, 4" in (loi.processing_error or "")


# ==================== METADONNEES ====================


class TestMetadonnees:

    @pytest.mark.parametrize("titre, attendu", [
        ("Loi N° 2016/007 du 12 juillet 2016 portant ratification de l'ordonnance N° 2015/001", "loi"),
        ("Décret N° 2024/191 du 4 juin 2024", "décret"),
        ("Arrêté N° 0405/CAB/PR du 24 mai 2024 portant nomination", "arrêté"),
        ("Ordonnance N° 2020/001 du 3 juin 2020", "ordonnance"),
        ("decret_n_2014_543_du_10.12.2014_portant", "décret"),
        ("Arrête N° 0632 du 22 juillet 2016 portant nomination", "arrêté"),
        ("Accord de Paris sur les Changements Climatiques", "autre"),
        ("Mémorandum d'Entente à l'issue du Sommet des Chefs d'Etat", "autre"),
    ])
    def test_type_lu_sur_le_premier_mot(self, titre, attendu):
        assert ingest_corpus._infer_type(titre) == attendu

    @pytest.mark.parametrize("titre, attendu", [
        ("Décret N° 2024/191 du 4 juin 2024", date(2024, 6, 4)),
        ("Décret N° 2024/200 du 1er juin 2024", date(2024, 6, 1)),
        ("decret_n_2014_543_du_10.12.2014_portant", date(2014, 12, 10)),
        # Annee mal tapee : reparee par l'annee du numero
        ("Décret n°2013 / 418 du 25 novembre 1013 portant attribution", date(2013, 11, 25)),
        # La date de l'acte, pas celle de l'acte qu'il modifie
        ("Décret N°2017/591 04 décembre 2017 modifiant le décret n°2001/066 du 12 mars 2001",
         date(2017, 12, 4)),
        ("Décret N°82/408 du 07septembre 1982 complétant le décret n° 74/759 du 26 août 1974",
         date(1982, 9, 7)),
        ("DECRET N° 2013/297 DU 0 9 SEP. 2013 modifiant", date(2013, 9, 9)),
        ("Décret N°2026/023 of 26 janvier 2026 ordonnant la publication", date(2026, 1, 26)),
        ("Décret N°2017/476 du 06 septembre 2017portant avancement", date(2017, 9, 6)),
        ("arrête_n_0 453_cab_pr_du_05.06.2024", date(2024, 6, 5)),
        ("Décret N°2018/148 portant modification du Décret N°2014/308 du 14 aout 2014",
         date(2018, 1, 1)),
    ])
    def test_date_de_signature(self, titre, attendu):
        assert ingest_corpus._infer_date(titre) == attendu


# ==================== PROCESSUS DE TRAVAIL ET CHIEN DE GARDE ====================


def _cible_bloquee(groupe, dossier, rss_max):
    """Processus qui commence un lot puis ne rend jamais la main."""
    import time as _time

    Path(dossier, "en_cours.json").write_text("{}", encoding="utf-8")
    _time.sleep(120)


class TestProcessusDeTravail:

    def test_panne_d_initialisation_sans_lot_en_cours(self, tmp_path, monkeypatch):
        ext = _extracteur(tmp_path / "cache")

        def _panne():
            raise RuntimeError("poids introuvables")

        ext._construire = _panne
        monkeypatch.setattr(extraire_corpus, "DoclingPdfExtractor", lambda: ext)
        monkeypatch.setattr(extraire_corpus, "configurer_journal", lambda chemin: None)
        doc = _document(_pdf(tmp_path / "d.pdf", 2), pages=2)
        suivi = tmp_path / "suivi"
        suivi.mkdir()

        code = extraire_corpus.travailler([asdict(doc)], str(suivi), rss_max_mo=10**6)

        assert code == extraire_corpus.SORTIE_PANNE_INIT
        assert not (suivi / "en_cours.json").exists(), "aucun lot ne doit payer la panne"
        assert list(ext.racine_cache.rglob("lot-*.json")) == []

    def test_disjoncteur_sur_echecs_en_serie(self, tmp_path, monkeypatch):
        ext = _extracteur(tmp_path / "cache")

        def _toujours(source, raises_on_error=True):
            raise RuntimeError("GPU indisponible")

        ext.doublure.convert = _toujours
        monkeypatch.setattr(extraire_corpus, "DoclingPdfExtractor", lambda: ext)
        monkeypatch.setattr(extraire_corpus, "configurer_journal", lambda chemin: None)
        docs = [
            _document(_pdf(tmp_path / f"d{i}.pdf", 1), doc_id=str(i), pages=1)
            for i in range(extraire_corpus.ECHECS_EN_SERIE_MAX + 2)
        ]
        suivi = tmp_path / "suivi"
        suivi.mkdir()

        code = extraire_corpus.travailler([asdict(d) for d in docs], str(suivi), rss_max_mo=10**6)

        assert code == extraire_corpus.SORTIE_PANNE_EN_SERIE
        tentes = len(list(ext.racine_cache.rglob("lot-*.json")))
        assert tentes == extraire_corpus.ECHECS_EN_SERIE_MAX, "l'arret evite d'user tout le corpus"

    def test_chien_de_garde_tue_un_lot_bloque(self, tmp_path):
        suivi = tmp_path / "suivi"
        suivi.mkdir()
        doc = Document("1", "x.pdf", "t", "s", 1)

        code = extraire_corpus.lancer_processus(
            [doc], suivi, 10**6, delai_lot_s=0, marge_s=1, pas_s=0.5, cible=_cible_bloquee,
        )

        assert code != 0


# ==================== CORRECTIONS DE LA REVUE (INDEXATION) ====================


class TestIndexationRobuste:

    def test_un_meme_pdf_n_est_indexe_qu_une_fois(
        self, sync_db_session, tmp_path, stockage, extraction_en_cache, embeddings_doubles
    ):
        """
        12 PDF du corpus sont publies sous deux doc_id ; indexes deux fois, ils
        doublaient chaque article dans la recherche et le chat.
        """
        ext = _extracteur(extraction_en_cache)
        pdf = _pdf(tmp_path / "d.pdf", 2)
        canonique = _document(pdf, doc_id="8467", pages=2, titre="Décret N°2020/433 du 2 juin 2020 portant nomination")
        doublon = _document(pdf, doc_id="8480", pages=2, titre="Décret N°2020/443 du 2 juin 2020 portant nomination")
        ext.extraire(pdf)

        bilan = ingest_corpus.indexer([doublon, canonique], ext, stockage)

        assert bilan["publies"] == 1 and bilan["doublons"] == 1
        lois = {loi.reference: loi for loi in sync_db_session.query(Law).all()}
        assert lois["PRC-8467"].status == "published"
        assert lois["PRC-8480"].status == "refused"
        assert "PRC-8467" in lois["PRC-8480"].processing_error
        assert "titres est faux" in lois["PRC-8480"].processing_error
        assert sync_db_session.query(Article).filter(Article.law_id == lois["PRC-8480"].id).count() == 0

    def test_une_loi_archivee_n_est_pas_republiee(
        self, sync_db_session, tmp_path, stockage, extraction_en_cache, embeddings_doubles
    ):
        ext = _extracteur(extraction_en_cache)
        doc = _document(_pdf(tmp_path / "d.pdf", 2), pages=2)
        ext.extraire(Path(doc.chemin))
        ingest_corpus.indexer([doc], ext, stockage)
        loi = sync_db_session.query(Law).filter(Law.reference == "PRC-77").one()
        loi.status = "archived"
        sync_db_session.commit()

        bilan = ingest_corpus.indexer([doc], ext, stockage)

        sync_db_session.expire_all()
        assert bilan["deja_publies"] == 1
        assert sync_db_session.query(Law).filter(Law.reference == "PRC-77").one().status == "archived"

    def test_une_copie_tronquee_est_refaite(self, tmp_path):
        source = _pdf(tmp_path / "source.pdf", 3)
        destination = tmp_path / "copie.pdf"
        destination.write_bytes(source.read_bytes()[:100])   # copie interrompue

        ingest_corpus._copier(str(source), destination)

        assert destination.read_bytes() == source.read_bytes()
        assert not destination.with_name("copie.pdf.tmp").exists()

    def test_une_panne_de_base_rejoue_le_meme_document(
        self, sync_db_session, tmp_path, stockage, extraction_en_cache
    ):
        from sqlalchemy.exc import OperationalError

        ext = _extracteur(extraction_en_cache)
        doc = _document(_pdf(tmp_path / "d.pdf", 2), pages=2)
        ext.extraire(Path(doc.chemin))
        appels = []

        def traiter(law_id, file_id, extraction_cache_seulement):
            appels.append(law_id)
            if len(appels) < 3:
                raise OperationalError("SELECT 1", {}, Exception("server closed the connection"))
            return {"articles_count": 1, "embeddings_generated": 1}

        bilan = ingest_corpus.indexer([doc], ext, stockage, traiter=traiter, attentes_s=(0, 0, 0))

        assert bilan == {"publies": 1, "deja_publies": 0, "en_attente": 0, "doublons": 0, "echecs": 0}
        assert len(appels) == 3

    def test_une_base_qui_ne_revient_pas_arrete_la_passe(
        self, sync_db_session, tmp_path, stockage, extraction_en_cache
    ):
        from sqlalchemy.exc import OperationalError

        ext = _extracteur(extraction_en_cache)
        doc = _document(_pdf(tmp_path / "d.pdf", 2), pages=2)
        ext.extraire(Path(doc.chemin))

        def traiter(*args, **kwargs):
            raise OperationalError("SELECT 1", {}, Exception("connection refused"))

        with pytest.raises(ingest_corpus.PanneDeBase):
            ingest_corpus.indexer([doc], ext, stockage, traiter=traiter, attentes_s=(0, 0))
