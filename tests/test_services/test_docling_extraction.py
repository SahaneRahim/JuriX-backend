"""
Tests de l'extracteur Docling.

Deux familles :

- Sur DOUBLURE de convertisseur, pour la logique : contrat de sortie, lots,
  cache, reprise apres echec, abandon au bout de TENTATIVES_MAX, mode
  « cache seulement » de la passe d'indexation, separation des caches.
  Rapides, sans modele.
- Sur le VRAI Docling, marque `docling` : la preuve que les options passees
  sont acceptees et que l'OCR pleine page lit une page scannee. Saute si
  Docling ou ses poids manquent.

Une ingestion du corpus dure une vingtaine d'heures : chaque test de la
premiere famille correspond a une facon de perdre ce temps — un plantage qui
fait tout reprendre, un lot qui echoue en boucle, un document repris deux fois.
"""

import json
import re
from pathlib import Path

import pytest

from app.services import docling_extraction as module
from app.services.docling_extraction import (
    TENTATIVES_MAX,
    DoclingPdfExtractor,
    ExtractionEnAttente,
)

# ==================== DOUBLURES ====================


class _Statut:
    def __init__(self, value):
        self.value = value


class _Document:
    def __init__(self, pages):
        self._pages = pages

    def export_to_markdown(self, page_no, **kwargs):
        return self._pages.get(page_no, "")


class _Resultat:
    """Pages en numerotation ABSOLUE ; le convertisseur double les renumerote."""

    def __init__(self, statut, pages, erreurs=()):
        self.status = _Statut(statut)
        self.pages_absolues = pages
        self.errors = list(erreurs)
        self.document = None


class _Convertisseur:
    """
    Rend, pour chaque page demandee, un texte qui porte son numero.

    Le vrai convertisseur recoit un TIFF des pages du lot, numerotees a partir
    de 1 : la doublure lit le lot dans le nom du flux et renumerote de meme.
    """

    def __init__(self, textes=None, comportement=None):
        self.appels = []
        self.textes = textes or {}
        self.comportement = comportement

    def convert(self, source, raises_on_error=True):
        debut, fin = map(int, re.search(r"_p(\d+)-(\d+)\.tiff$", source.name).groups())
        self.appels.append((debut, fin))
        if self.comportement:
            res = self.comportement((debut, fin))
        else:
            res = _Resultat(
                "success",
                {n: self.textes.get(n, f"Article {n}.- Texte de la page {n}.") for n in range(debut, fin + 1)},
            )
        res.document = _Document({n - debut + 1: md for n, md in res.pages_absolues.items()})
        return res


def _pdf(chemin: Path, nb_pages: int) -> Path:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument.new()
    for _ in range(nb_pages):
        pdf.new_page(595, 842)
    pdf.save(str(chemin))
    pdf.close()
    return chemin


@pytest.fixture
def extracteur(tmp_path, monkeypatch):
    """Extracteur a lots de 2 pages, cache temporaire, convertisseur double."""
    ext = DoclingPdfExtractor(cache_dir=str(tmp_path / "cache"), pages_par_lot=2)
    ext.doublure = _Convertisseur()
    ext.constructions = 0

    def _construire():
        ext.constructions += 1
        return ext.doublure

    monkeypatch.setattr(ext, "_construire", _construire)
    return ext


# ==================== CONTRAT DE SORTIE ====================


class TestContratDeSortie:

    def test_marqueur_par_page_non_vide_numerotation_physique(self, extracteur, tmp_path):
        """Une page blanche consomme son numero : les suivantes restent alignees."""
        extracteur.doublure.textes = {1: "Article 1.- A.", 2: "", 3: "Article 2.- B."}
        resultat = extracteur.extraire(_pdf(tmp_path / "d.pdf", 3))

        assert resultat.texte == "<<PAGE:1>>\nArticle 1.- A.\n\n<<PAGE:3>>\nArticle 2.- B."
        assert resultat.nb_pages == 3
        assert resultat.pages_en_echec == []

    def test_cachet_retire_de_chaque_page(self, extracteur, tmp_path):
        extracteur.doublure.textes = {
            1: "Article 1.- Texte.\nPRESIDENCE DE LA REPUBLIQUE\nCOPIE CERTIFIEE CONFORME"
        }
        resultat = extracteur.extraire(_pdf(tmp_path / "d.pdf", 1))

        assert "COPIE CERTIFIEE" not in resultat.texte
        assert "Article 1.- Texte." in resultat.texte


# ==================== LOTS ET CACHE ====================


class TestLotsEtCache:

    def test_conversion_par_lots_de_pages(self, extracteur, tmp_path):
        extracteur.extraire(_pdf(tmp_path / "d.pdf", 5))

        assert extracteur.doublure.appels == [(1, 2), (3, 4), (5, 5)]

    def test_une_seconde_extraction_ne_convertit_rien(self, extracteur, tmp_path):
        pdf = _pdf(tmp_path / "d.pdf", 5)
        premier = extracteur.extraire(pdf)
        extracteur.doublure.appels.clear()

        second = extracteur.extraire(pdf)

        assert extracteur.doublure.appels == []
        assert second.texte == premier.texte

    def test_le_modele_ne_se_charge_pas_pour_relire_le_cache(self, extracteur, tmp_path):
        """
        La passe d'indexation relit un cache complet : charger Docling pour
        rien lui couterait des Go de memoire, a cote d'EmbeddingGemma.
        """
        pdf = _pdf(tmp_path / "d.pdf", 3)
        extracteur.extraire(pdf)
        neuf = DoclingPdfExtractor(cache_dir=str(tmp_path / "cache"), pages_par_lot=2)

        def _interdit():
            raise AssertionError("Docling charge alors que tout est en cache")

        neuf._construire = _interdit
        assert "Article 3" in neuf.extraire(pdf).texte

    def test_changer_la_taille_des_lots_ne_perd_pas_le_cache(self, extracteur, tmp_path):
        """
        Les lots se lisent tels qu'ils sont sur le disque : changer
        DOCLING_PAGES_PAR_LOT entre deux executions rendait sinon tout le
        cache invisible, et des heures d'extraction etaient refaites.
        """
        pdf = _pdf(tmp_path / "d.pdf", 5)
        extracteur.traiter_lot(pdf, module.sha256_fichier(pdf), 1, 2)
        autre = DoclingPdfExtractor(cache_dir=str(tmp_path / "cache"), pages_par_lot=8)
        autre.doublure = _Convertisseur()
        autre._construire = lambda: autre.doublure

        assert autre.etat(pdf).lots_a_faire == [(3, 5)], "seul le reste est a faire"
        resultat = autre.extraire(pdf)

        assert autre.doublure.appels == [(3, 5)]
        assert all(f"Article {n}" in resultat.texte for n in range(1, 6))

    def test_cache_separe_par_configuration(self, extracteur, tmp_path):
        """Un autre moteur OCR ne relit pas, et n'ecrase pas, ce cache."""
        pdf = _pdf(tmp_path / "d.pdf", 2)
        extracteur.extraire(pdf)
        autre = DoclingPdfExtractor(
            cache_dir=str(tmp_path / "cache"), pages_par_lot=2, moteur_ocr="tesseract"
        )

        assert autre.dossier_cache != extracteur.dossier_cache
        assert autre.etat(pdf).lots_a_faire == [(1, 2)]

    def test_les_extractions_gemini_ne_sont_pas_relues(self, extracteur, tmp_path):
        """Le cache Gemini (`{sha}.json` a la racine) est hors du dossier docling/."""
        assert extracteur.dossier_cache.parent.name == "docling"


# ==================== REPRISE ====================


class TestReprise:

    def test_un_lot_en_echec_est_seul_rejoue(self, extracteur, tmp_path):
        pdf = _pdf(tmp_path / "d.pdf", 5)
        normal = extracteur.doublure.convert
        echecs = {"n": 0}

        def _une_fois(page_range):
            if page_range == (3, 4) and echecs["n"] == 0:
                echecs["n"] += 1
                raise RuntimeError("CUDA out of memory")
            debut, fin = page_range
            return _Resultat("success", {n: f"Article {n}.- Page {n}." for n in range(debut, fin + 1)})

        extracteur.doublure.comportement = _une_fois
        premier = extracteur.extraire(pdf)
        assert premier.pages_en_echec == [3, 4]

        extracteur.doublure.appels.clear()
        second = extracteur.extraire(pdf)

        assert extracteur.doublure.appels == [(3, 4)], "seul le lot en echec est rejoue"
        assert second.pages_en_echec == []
        assert "Article 4" in second.texte
        assert normal  # la doublure reste utilisable

    def test_abandon_apres_tentatives_max(self, extracteur, tmp_path):
        """
        Une page qui echoue toujours ne doit pas bloquer l'ingestion : au bout
        de TENTATIVES_MAX, son lot est repris page a page, puis la page seule
        est abandonnee, et le document part sans elle.
        """
        pdf = _pdf(tmp_path / "d.pdf", 4)

        def _page_4_illisible(page_range):
            debut, fin = page_range
            if debut <= 4 <= fin:
                raise RuntimeError("page illisible")
            return _Resultat("success", {n: f"Page {n}." for n in range(debut, fin + 1)})

        extracteur.doublure.comportement = _page_4_illisible
        while not extracteur.etat(pdf).termine:
            extracteur.extraire(pdf)
        extracteur.doublure.appels.clear()

        resultat = extracteur.extraire(pdf)

        assert extracteur.doublure.appels == [], "un lot abandonne n'est plus retente"
        assert resultat.pages_en_echec == [4]
        assert "Page 3." in resultat.texte, "la page saine du lot est sauvee"
        etat = extracteur.etat(pdf)
        assert etat.lots_abandonnes == [(4, 4)]
        assert etat.termine

    def test_page_toxique_isolee_sans_perdre_son_lot(self, extracteur, tmp_path):
        """
        Une page qui TUE le processus (plantage natif) faisait abandonner son
        lot entier apres TENTATIVES_MAX morts : les pages saines avec elle.
        """
        pdf = _pdf(tmp_path / "d.pdf", 2)
        for _ in range(TENTATIVES_MAX):
            extracteur.noter_echec(pdf, 1, 2, "processus de travail mort (code -11)")

        etat = extracteur.etat(pdf)
        assert etat.lots_abandonnes == [], "le lot n'est pas abandonne : il est redecoupe"
        assert etat.lots_a_faire == [(1, 1), (2, 2)]

        # La page 1 passe seule ; la page 2 tue encore le processus
        extracteur.traiter_lot(pdf, module.sha256_fichier(pdf), 1, 1)
        for _ in range(TENTATIVES_MAX):
            extracteur.noter_echec(pdf, 2, 2, "processus de travail mort (code -11)")

        etat = extracteur.etat(pdf)
        assert etat.termine
        assert etat.lots_abandonnes == [(2, 2)]
        resultat = extracteur.extraire(pdf)
        assert "Article 1" in resultat.texte
        assert resultat.pages_en_echec == [2]

    def test_page_lue_d_un_lot_abandonne_n_est_pas_refaite(self, extracteur, tmp_path):
        """Seules les pages fautives ou vides d'un lot abandonne sont reprises."""
        pdf = _pdf(tmp_path / "d.pdf", 2)
        extracteur.doublure.comportement = lambda pr: _Resultat(
            "partial_success", {1: "Article 1.- Lu.", 2: ""}, ["timeout"]
        )
        for _ in range(TENTATIVES_MAX):
            extracteur.extraire(pdf)

        assert extracteur.etat(pdf).lots_a_faire == [(2, 2)]

    def test_succes_partiel_garde_le_texte_et_sera_rejoue(self, extracteur, tmp_path):
        pdf = _pdf(tmp_path / "d.pdf", 2)
        extracteur.doublure.comportement = lambda pr: _Resultat(
            "partial_success", {1: "Article 1.- Lu.", 2: ""}, ["timeout"]
        )
        premier = extracteur.extraire(pdf)

        assert "Article 1.- Lu." in premier.texte
        assert premier.pages_en_echec == [1, 2]
        assert extracteur.etat(pdf).lots_a_faire == [(1, 2)]

    def test_plantage_du_processus_compte_une_tentative(self, extracteur, tmp_path):
        """
        Le superviseur note l'echec d'un lot pendant lequel le processus est
        mort : sinon ce lot tuerait chaque nouveau processus, indefiniment.
        """
        pdf = _pdf(tmp_path / "d.pdf", 1)
        for _ in range(TENTATIVES_MAX):
            extracteur.noter_echec(pdf, 1, 1, "processus tue (signal 9)")

        assert extracteur.etat(pdf).lots_abandonnes == [(1, 1)]
        assert extracteur.doublure.appels == []

    def test_noter_echec_n_efface_pas_un_lot_reussi(self, extracteur, tmp_path):
        pdf = _pdf(tmp_path / "d.pdf", 2)
        extracteur.extraire(pdf)
        extracteur.noter_echec(pdf, 1, 2, "faux positif")

        assert extracteur.etat(pdf).lots_faits == [(1, 2)]

    def test_entree_de_cache_corrompue_est_refaite(self, extracteur, tmp_path):
        pdf = _pdf(tmp_path / "d.pdf", 2)
        extracteur.extraire(pdf)
        lot = next(extracteur.dossier_cache.rglob("lot-*.json"))
        lot.write_text("{tronque", encoding="utf-8")
        extracteur.doublure.appels.clear()

        extracteur.extraire(pdf)

        assert extracteur.doublure.appels == [(1, 2)]
        assert json.loads(lot.read_text(encoding="utf-8"))["statut"] == "ok"


# ==================== PASSE D'INDEXATION ====================


class TestCacheSeulement:

    def test_document_non_extrait_mis_en_attente(self, extracteur, tmp_path):
        with pytest.raises(ExtractionEnAttente):
            extracteur.extraire(_pdf(tmp_path / "d.pdf", 3), cache_seulement=True)
        assert extracteur.constructions == 0

    def test_document_extrait_relu_sans_conversion(self, extracteur, tmp_path):
        pdf = _pdf(tmp_path / "d.pdf", 3)
        extracteur.extraire(pdf)
        extracteur.doublure.appels.clear()

        resultat = extracteur.extraire(pdf, cache_seulement=True)

        assert extracteur.doublure.appels == []
        assert "Article 3" in resultat.texte


# ==================== LE VRAI DOCLING ====================


def _page_scannee(chemin: Path) -> Path:
    """Une page de texte RASTERISEE, comme un scan : sans aucune couche texte."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1240, 1754), "white")
    dessin = ImageDraw.Draw(image)
    police = ImageFont.load_default(size=44)
    lignes = [
        "DECRET N° 2024/191 DU 4 JUIN 2024",
        "",
        "Article 1er.- Le present decret fixe les regles",
        "applicables aux marches publics.",
        "",
        "Article 2.- Le present decret sera enregistre.",
    ]
    for i, ligne in enumerate(lignes):
        dessin.text((120, 200 + i * 80), ligne, fill="black", font=police)
    image.save(chemin, "PDF", resolution=150)
    return chemin


@pytest.mark.docling
def test_vrai_docling_lit_une_page_scannee(tmp_path):
    pytest.importorskip("docling")
    ext = DoclingPdfExtractor(cache_dir=str(tmp_path / "cache"), pages_par_lot=4)
    try:
        resultat = ext.extraire(_page_scannee(tmp_path / "scan.pdf"))
    except Exception as e:  # poids absents du cache Hugging Face
        if "offline" in str(e).lower() or "not found" in str(e).lower():
            pytest.skip(f"poids Docling indisponibles : {e}")
        raise

    assert resultat.pages_en_echec == []
    assert resultat.texte.startswith("<<PAGE:1>>")
    texte = resultat.texte.lower()
    assert "article 2" in texte
    assert "marches publics" in texte.replace("é", "e")
    assert module.SCHEMA_CACHE == json.loads(
        next(ext.dossier_cache.rglob("lot-*.json")).read_text(encoding="utf-8")
    )["schema"]


# ==================== CORRECTIONS DE LA REVUE ====================


class TestPannesEtFusion:

    def test_panne_d_initialisation_n_est_pas_l_echec_d_un_lot(self, extracteur, tmp_path, monkeypatch):
        """
        Comptee comme l'echec du lot en cours, une panne de chargement des
        modeles abandonnait tout le corpus, lot apres lot.
        """
        pdf = _pdf(tmp_path / "d.pdf", 2)

        def _panne():
            raise RuntimeError("CUDA out of memory au chargement")

        monkeypatch.setattr(extracteur, "_construire", _panne)

        with pytest.raises(RuntimeError, match="chargement"):
            extracteur.traiter_lot(pdf, module.sha256_fichier(pdf), 1, 2)
        assert list(extracteur.racine_cache.rglob("lot-*.json")) == []

    def test_une_tentative_ne_retire_jamais_de_texte(self, extracteur, tmp_path):
        pdf = _pdf(tmp_path / "d.pdf", 3)
        sha = module.sha256_fichier(pdf)
        reponses = [
            _Resultat("partial_success", {1: "Page un.", 2: "Page deux.", 3: ""}),
            _Resultat("partial_success", {1: "Page un.", 2: "", 3: ""}),
        ]
        extracteur.doublure.comportement = lambda pr: reponses.pop(0)
        extracteur.pages_par_lot = 3

        extracteur.traiter_lot(pdf, sha, 1, 3)
        lot = extracteur.traiter_lot(pdf, sha, 1, 3)

        assert lot["pages"]["2"] == "Page deux.", "la seconde tentative a efface la page 2"
        assert lot["tentatives"] == 2

    def test_seules_les_pages_designees_par_docling_sont_en_echec(self, extracteur, tmp_path):
        class _Erreur:
            error_message = "delai depasse"
            page_no = 2   # page 2 du LOT, donc page 4 du PDF

        pdf = _pdf(tmp_path / "d.pdf", 4)
        extracteur.doublure.comportement = lambda pr: (
            _Resultat("partial_success", {3: "Page trois.", 4: ""}, [_Erreur()])
            if pr == (3, 4) else _Resultat("success", {1: "Un.", 2: "Deux."})
        )
        for _ in range(TENTATIVES_MAX):
            resultat = extracteur.extraire(pdf)

        assert resultat.pages_en_echec == [4]
        assert "Page trois." in resultat.texte
        assert resultat.erreurs and "delai depasse" in resultat.erreurs[-1]

    def test_le_cache_garde_le_markdown_brut(self, extracteur, tmp_path):
        """Ameliorer le nettoyage ne doit jamais obliger a refaire l'OCR."""
        cachet = "PRESIDENCE DE LA REPUBLIQUE\nCOPIE CERTIFIEE CONFORME"
        extracteur.doublure.textes = {1: f"Article 1.- Texte suffisamment long pour etre lisible.\n{cachet}"}
        pdf = _pdf(tmp_path / "d.pdf", 1)

        resultat = extracteur.extraire(pdf)

        lot = json.loads(next(extracteur.dossier_cache.rglob("lot-*.json")).read_text())
        assert "COPIE CERTIFIEE CONFORME" in lot["pages"]["1"], "le cache doit rester brut"
        assert "COPIE CERTIFIEE" not in resultat.texte, "le nettoyage se fait a la lecture"

    def test_page_reduite_au_cachet_signalee(self, extracteur, tmp_path):
        extracteur.doublure.textes = {
            1: "Article 1.- Le present decret fixe les regles applicables.",
            2: "PRESIDENCE DE LA REPUBLIQUE\nCOPIE CERTIFIEE CONFORME\nCERTIFIED TRUE COPY",
        }
        resultat = extracteur.extraire(_pdf(tmp_path / "d.pdf", 2))

        assert resultat.pages_illisibles == [2]
        assert resultat.pages_en_echec == []

    def test_empreinte_porte_le_rendu_et_les_poids(self, extracteur):
        empreinte = extracteur.empreinte()
        assert empreinte["entree"].startswith("images-pdfium")
        assert set(empreinte["modeles"]) == {d for d, _ in module.MODELES_HF}
        assert "torch" in empreinte or "onnxruntime" in empreinte
