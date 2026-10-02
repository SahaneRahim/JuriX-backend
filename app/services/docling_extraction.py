"""
Extraction des PDF par Docling, en local : OCR pleine page sur TOUTES les pages.

POURQUOI L'OCR PARTOUT. Mesure sur les 13 970 pages du corpus prc.cm : 97 % sont
des scans. La couche texte que porte 55 % d'entre elles est l'OCR du scanner,
bruite (« portantnomination deresponsables », « DECRET N°_2_0_2_1 _i _ »), mesure
jadis a ~20 % de rappel ; 2,4 % seulement des pages sont du texte natif. Lire la
couche texte, c'est indexer ce bruit.

POURQUOI DES IMAGES RENDUES PAR PDFIUM, et non le lecteur PDF de Docling. Son
rendu (docling-parse) ignore les masques d'image des scans « MRC » — un calque
d'encre plus un masque de texte : la page sort NOIRE, l'OCR n'y lit que le
cachet, et le lot passe pour reussi. Mesure : 88 a 100 % de pixels sombres
contre 1,5 a 9,5 % avec pdfium, sur 59 documents du corpus dont 8842 (169 pages
de nominations). Les pages sont donc rendues par pdfium a 216 dpi — l'echelle
qu'applique l'OCR de Docling, sans remise a l'echelle — puis passees a Docling
en image. Sur l'echantillon de reference, le F1 passe de 0,75 a 0,86.

POURQUOI PAR LOTS DE PAGES. Un document se convertit par plages de quelques
pages, chacune ecrite en cache des qu'elle est faite :
- une interruption — coupure de courant, plantage, Ctrl-C — ne coute que le lot
  en cours ;
- la memoire reste bornee sur les gros documents ;
- un lot qui echoue est rejoue seul, jusqu'a TENTATIVES_MAX fois, et chaque
  tentative ne peut qu'ajouter du texte, jamais en retirer.

LE CACHE GARDE LE MARKDOWN BRUT de Docling ; le cachet est retire a la LECTURE.
Ameliorer le nettoyage ne demande donc jamais de refaire l'OCR — une quinzaine
d'heures —, et un nettoyage trop zele ne detruit rien d'irrecuperable.

L'EMPREINTE DU CACHE porte tout ce qui change le texte produit : rendu et
resolution, versions de pypdfium2, docling et du moteur OCR, revisions des
poids des modeles, langue. Une autre configuration ecrit ailleurs, jamais
par-dessus. Les extractions Gemini deja payees (`data/ocr_cache/{sha}.json`)
ne sont ni relues ni ecrasees.

CONTRAT DE SORTIE, celui de pdf_extraction_service, impose par text_chunker :
`<<PAGE:n>>\\n<markdown>` par page non vide, n = page PHYSIQUE a partir de 1,
pages jointes par `\\n\\n`.

Les modeles se chargent paresseusement, au premier lot a convertir : relire un
document deja extrait ne coute ni GPU ni memoire, et n'importe pas Docling.
"""

import asyncio
import gc
import hashlib
import io
import json
import logging
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import metadata
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from app.core.config import settings
from app.services.pdf_extraction_service import PdfExtractionError, ResultatExtraction
from app.utils.markdown_cleanup import nettoyer_markdown

logger = logging.getLogger(__name__)

# Version du format d'une entree de cache.
SCHEMA_CACHE = 2
# A incrementer quand l'appel a export_to_markdown change : l'empreinte change
# avec lui. Le NETTOYAGE, lui, se fait a la lecture et n'en fait pas partie.
VERSION_EXPORT = 1
# Au-dela, un lot en echec est abandonne : ses pages manquantes sont signalees
# dans `laws.processing_error`, et le document est publie sans elles.
TENTATIVES_MAX = 3

# ---- Configuration figee : source UNIQUE pour la conversion ET l'empreinte ----
# Resolution du rendu : l'echelle 3 (216 dpi) de l'OCR de Docling.
DPI_RENDU = 216
LANGUE_RAPIDOCR = "fr"
LANGUE_TESSERACT = "fra"
# Depots Hugging Face des poids charges par le pipeline : mise en page (Heron)
# et structure des tableaux (TableFormer). Docling les charge en revision
# « main » ou par etiquette ; c'est le commit reellement present en cache qui
# entre dans l'empreinte.
MODELES_HF = (
    ("docling-project/docling-layout-heron", "main"),
    ("docling-project/docling-models", "v2.3.0"),
)

# PDFium n'est pas sur entre fils d'execution (pypdfium2, « Known
# limitations ») : un seul appel a la fois dans le processus. Docling, lui,
# n'y touche plus : il recoit des images.
_VERROU_PDFIUM = threading.Lock()


class ExtractionEnAttente(PdfExtractionError):
    """Le document n'est pas entierement extrait, et seul le cache etait permis."""


@dataclass
class EtatDocument:
    """Avancement de l'extraction d'un document, lu dans le cache."""

    sha256: str
    nb_pages: int
    lots_faits: List[Tuple[int, int]] = field(default_factory=list)
    lots_a_faire: List[Tuple[int, int]] = field(default_factory=list)
    lots_abandonnes: List[Tuple[int, int]] = field(default_factory=list)

    @property
    def termine(self) -> bool:
        """Plus rien a tenter : chaque lot est fait ou abandonne."""
        return not self.lots_a_faire

    @property
    def pages_restantes(self) -> int:
        """Pages distinctes qu'il reste a tenter."""
        return len({n for debut, fin in self.lots_a_faire for n in range(debut, fin + 1)})


# ==================== OUTILS ====================


def sha256_fichier(chemin: Path) -> str:
    h = hashlib.sha256()
    with open(chemin, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def compter_pages(chemin: Path) -> int:
    """
    Nombre de pages PHYSIQUES, lu par pdfium.

    pypdf se trompe sur les fichiers a revisions multiples : il voit 45 pages
    dans le Code des marches publics 2018 (fichier 6521), qui en a 98.
    """
    import pypdfium2 as pdfium

    with _VERROU_PDFIUM:
        pdf = pdfium.PdfDocument(str(chemin))
        try:
            return len(pdf)
        finally:
            pdf.close()


def rasteriser(chemin: Path, debut: int, fin: int, dpi: int = DPI_RENDU) -> io.BytesIO:
    """
    Pages debut..fin rendues par pdfium, en un TIFF multipage en memoire.

    Le fichier n'est pas lu en entier : pdfium l'ouvre par son chemin et ne
    decode que les pages demandees.
    """
    import pypdfium2 as pdfium

    images = []
    with _VERROU_PDFIUM:
        pdf = pdfium.PdfDocument(str(chemin))
        try:
            for n in range(debut, fin + 1):
                page = pdf[n - 1]
                try:
                    images.append(page.render(scale=dpi / 72).to_pil().convert("RGB"))
                finally:
                    page.close()
        finally:
            pdf.close()
    flux = io.BytesIO()
    images[0].save(
        flux, format="TIFF", save_all=True, append_images=images[1:],
        compression="tiff_lzw", dpi=(dpi, dpi),
    )
    flux.seek(0)
    return flux


def nettoyer_page(markdown: str) -> str:
    """Le nettoyage applique a la lecture du cache : cachet, filigrane, balises, marqueurs."""
    return nettoyer_markdown(markdown) if markdown.strip() else ""


def joindre_pages(pages: List[str]) -> str:
    """Contrat de sortie : un marqueur par page NON vide, numerotation physique."""
    return "\n\n".join(
        f"<<PAGE:{n}>>\n{md}" for n, md in enumerate(pages, start=1) if md.strip()
    )


# Une page dont il ne reste, apres nettoyage, que quelques caracteres de mot
# n'a pas ete lue : rendu rate, ou seul le cachet a ete reconnu.
_RESTE_LISIBLE = 30


def _lisible(markdown: str) -> bool:
    return len(re.sub(r"[\W_]+", "", markdown)) >= _RESTE_LISIBLE


def _version(paquet: str) -> str:
    try:
        return metadata.version(paquet)
    except metadata.PackageNotFoundError:
        return "absent"


def _version_publique(paquet: str) -> str:
    """Sans la partie locale : 2.14.0+cu126 et 2.14.0+cpu donnent 2.14.0."""
    return _version(paquet).split("+", 1)[0]


@lru_cache(maxsize=4)
def _version_tesseract(commande: str) -> str:
    """
    Version de tesseract, ou ERREUR. Aucune valeur de repli : un « absent »
    passager (machine chargee) changerait l'empreinte, donc le dossier du cache,
    et des processus voisins ne liraient plus le meme cache.
    """
    sortie = subprocess.run([commande, "--version"], capture_output=True, text=True, timeout=60)
    premiere = (sortie.stdout or sortie.stderr).strip().splitlines()
    if sortie.returncode != 0 or not premiere:
        raise PdfExtractionError(f"tesseract illisible ({commande}) : code {sortie.returncode}")
    return premiere[0].strip()


def _revision_hf(depot: str, reference: str) -> str:
    """Commit des poids reellement presents dans le cache Hugging Face."""
    racine = Path(
        os.environ.get("HF_HUB_CACHE")
        or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    )
    chemin = racine / f"models--{depot.replace('/', '--')}" / "refs" / reference
    try:
        return chemin.read_text(encoding="utf-8").strip()
    except OSError:
        return "absent"


def _fils_par_defaut() -> int:
    try:
        import psutil

        return psutil.cpu_count(logical=False) or 4
    except ImportError:
        return max(1, (os.cpu_count() or 4) // 2)


# ==================== EXTRACTEUR ====================


class DoclingPdfExtractor:
    """
    Extracteur PDF -> markdown pagine, par Docling, en local.

    Meme surface que GeminiPdfExtractor pour le pipeline : `is_available()`,
    `raison_indisponible()`, `extraire(path) -> ResultatExtraction`, et
    `async extract_text(path) -> str`.
    """

    nom = "Docling"

    def __init__(
        self,
        cache_dir: Optional[str] = None,
        moteur_ocr: Optional[str] = None,
        device: Optional[str] = None,
        pages_par_lot: Optional[int] = None,
        fils: Optional[int] = None,
        timeout_lot_s: Optional[float] = None,
    ):
        self.racine_cache = Path(cache_dir or settings.OCR_CACHE_DIR) / "docling"
        self.moteur_ocr = moteur_ocr or settings.EXTRACTION_OCR_ENGINE
        self.device = device or settings.EXTRACTION_DEVICE
        self.pages_par_lot = pages_par_lot or settings.EXTRACTION_PAGES_PAR_LOT
        self.fils = fils or settings.EXTRACTION_FILS or _fils_par_defaut()
        self.timeout_lot_s = timeout_lot_s or settings.EXTRACTION_LOT_TIMEOUT_S
        self._convertisseur = None
        self._verrou_init = threading.Lock()
        # Une seule conversion a la fois dans le processus : l'API peut appeler
        # l'extracteur partage depuis deux fils, et deux convert() sur le meme
        # pipeline se partagent son etat et la memoire du GPU.
        self._verrou_conversion = threading.Lock()
        self._empreinte: Optional[Dict[str, Any]] = None
        if self.pages_par_lot < 1:
            raise ValueError("EXTRACTION_PAGES_PAR_LOT doit valoir au moins 1")
        if self.moteur_ocr not in ("rapidocr_torch", "rapidocr", "tesseract"):
            raise ValueError(f"Moteur OCR inconnu : {self.moteur_ocr}")

    # ==================== DISPONIBILITE ====================

    def is_available(self) -> bool:
        return find_spec("docling") is not None and find_spec("pypdfium2") is not None

    def raison_indisponible(self) -> str:
        return (
            "Docling n'est pas installe : extraction impossible. "
            "Installer requirements-ingestion.txt, ou PDF_EXTRACTION_ENGINE=gemini."
        )

    # ==================== EMPREINTE ET CACHE ====================

    def _commande_tesseract(self) -> str:
        return settings.TESSERACT_PATH or "tesseract"

    def empreinte(self) -> Dict[str, Any]:
        """
        Tout ce qui change le texte produit. Ni le peripherique ni le nettoyage
        n'en font partie.

        Calculee une fois par instance : elle sert a chaque lecture du cache.
        """
        if self._empreinte is None:
            self._empreinte = self._calculer_empreinte()
        return self._empreinte

    def _calculer_empreinte(self) -> Dict[str, Any]:
        empreinte: Dict[str, Any] = {
            "schema": SCHEMA_CACHE,
            "export": VERSION_EXPORT,
            "entree": f"images-pdfium-{DPI_RENDU}dpi",
            "pypdfium2": _version("pypdfium2"),
            "docling": _version("docling"),
            "docling_core": _version("docling-core"),
            "docling_ibm_models": _version("docling-ibm-models"),
            "transformers": _version("transformers"),
            "modeles": {depot: _revision_hf(depot, ref) for depot, ref in MODELES_HF},
            "ocr": self.moteur_ocr,
            "ocr_mode": "full_page",
            "tables": "accurate",
        }
        if self.moteur_ocr == "tesseract":
            empreinte["tesseract"] = _version_tesseract(self._commande_tesseract())
            empreinte["langue"] = LANGUE_TESSERACT
            empreinte["osd"] = "desactivee"
        else:
            empreinte["rapidocr"] = _version("rapidocr")
            empreinte["langue"] = LANGUE_RAPIDOCR
            moteur = "torch" if self.moteur_ocr == "rapidocr_torch" else "onnxruntime"
            empreinte[moteur] = _version_publique(moteur)
        return empreinte

    @property
    def dossier_cache(self) -> Path:
        cle = hashlib.sha256(
            json.dumps(self.empreinte(), sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        return self.racine_cache / cle

    def lots(self, nb_pages: int) -> List[Tuple[int, int]]:
        """Plages de pages d'un document neuf, bornes incluses, a partir de 1."""
        return self._decouper(list(range(1, nb_pages + 1)))

    def _decouper(self, pages: List[int]) -> List[Tuple[int, int]]:
        """Pages triees -> plages CONTIGUES d'au plus pages_par_lot pages."""
        plages: List[Tuple[int, int]] = []
        for n in pages:
            if plages and n == plages[-1][1] + 1 and n - plages[-1][0] < self.pages_par_lot:
                plages[-1] = (plages[-1][0], n)
            else:
                plages.append((n, n))
        return plages

    def _dossier_document(self, sha: str) -> Path:
        return self.dossier_cache / sha[:2] / sha

    def _chemin_lot(self, sha: str, debut: int, fin: int) -> Path:
        return self._dossier_document(sha) / f"lot-{debut:04d}-{fin:04d}.json"

    def _lire_lot(self, chemin: Path) -> Optional[Dict[str, Any]]:
        """Entree de cache, ou None si absente, illisible ou d'un autre schema."""
        try:
            donnees = json.loads(chemin.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if donnees.get("schema") != SCHEMA_CACHE or not isinstance(donnees.get("pages"), dict):
            return None
        return donnees

    def _ecrire_lot(self, chemin: Path, donnees: Dict[str, Any]) -> None:
        """Ecriture atomique : un lot a moitie ecrit ne doit jamais passer pour fait."""
        chemin.parent.mkdir(parents=True, exist_ok=True)
        tmp = chemin.with_name(f"{chemin.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(donnees, ensure_ascii=False), encoding="utf-8")
        tmp.replace(chemin)

    @staticmethod
    def _lot_acquis(lot: Optional[Dict[str, Any]]) -> bool:
        """Fait, ou abandonne apres TENTATIVES_MAX echecs : plus rien a tenter."""
        return bool(lot) and (
            lot.get("statut") == "ok" or lot.get("tentatives", 0) >= TENTATIVES_MAX
        )

    @staticmethod
    def _pages_en_echec_du_lot(lot: Dict[str, Any]) -> List[int]:
        """Les pages reellement fautives d'un lot en echec ; toutes, a defaut."""
        fautives = lot.get("pages_fautives")
        if fautives:
            return list(fautives)
        return list(range(lot["debut"], lot["fin"] + 1))

    def _inventaire(
        self, sha: str, nb_pages: int
    ) -> Tuple[EtatDocument, Dict[int, str], Set[int], List[str]]:
        """
        Ce que le cache contient pour un document, quelle que soit la taille
        des lots qui l'ont rempli.

        Les lots se lisent tels qu'ils sont sur le disque, et non d'apres
        EXTRACTION_PAGES_PAR_LOT : changer ce reglage entre deux executions
        rendait tout le cache invisible. Ce qui manque est redecoupe a la
        taille courante.

        Returns:
            (etat, markdown BRUT par page, pages en echec, erreurs des lots
            abandonnes)
        """
        etat = EtatDocument(sha256=sha, nb_pages=nb_pages)
        textes: Dict[int, str] = {}
        faites: Set[int] = set()
        abandonnees: Set[int] = set()
        en_echec: Set[int] = set()
        erreurs: List[str] = []
        a_rejouer: List[Tuple[int, int]] = []
        rejouees: Set[int] = set()
        a_isoler: Set[int] = set()

        lus = []
        dossier = self._dossier_document(sha)
        if dossier.is_dir():
            for chemin in sorted(dossier.glob("lot-*.json")):
                lot = self._lire_lot(chemin)
                if lot and 1 <= lot.get("debut", 0) <= lot.get("fin", 0) <= nb_pages:
                    lus.append(lot)

        # Priorite au lot reussi, puis au lot abandonne, puis au lot a rejouer
        for lot in (lt for lt in lus if lt.get("statut") == "ok"):
            etat.lots_faits.append((lot["debut"], lot["fin"]))
            for n in range(lot["debut"], lot["fin"] + 1):
                faites.add(n)
                textes[n] = lot["pages"].get(str(n), "")
        # Lots en echec, les plus petits d'abord : une page isolee apres
        # l'abandon de son lot a son propre sort, qui prime sur celui du lot.
        non_ok = sorted(
            (lt for lt in lus if lt.get("statut") != "ok"),
            key=lambda lt: (lt["fin"] - lt["debut"], lt["debut"]),
        )
        for lot in non_ok:
            plage = (
                set(range(lot["debut"], lot["fin"] + 1))
                - faites - abandonnees - rejouees - a_isoler
            )
            if not plage:
                continue
            for n in plage:
                if not textes.get(n, "").strip():
                    textes[n] = lot["pages"].get(str(n), "")
            if not self._lot_acquis(lot):
                # Rejoue a l'identique : ses tentatives se comptent sur ce lot
                a_rejouer.append((lot["debut"], lot["fin"]))
                rejouees |= plage
                continue

            fautives = set(self._pages_en_echec_du_lot(lot)) & plage
            # Un lot de plusieurs pages abandonne est REPRIS PAGE A PAGE, pour
            # ses seules pages sans texte ou fautives ; ses pages lues sont
            # acquises telles quelles. Une page toxique — image malformee qui
            # fait planter pdfium ou l'OCR, page qui epuise la memoire — tuait
            # le processus a chaque tentative : son lot entier etait abandonne,
            # et les pages saines avec elle. Le sort des pages reprises se lit
            # ensuite dans leurs propres lots.
            if lot["fin"] > lot["debut"]:
                isoler = fautives | {n for n in plage if not textes.get(n, "").strip()}
                a_isoler |= isoler
                abandonnees |= plage - isoler
                if isoler:
                    erreurs.extend((lot.get("erreurs") or [])[-1:])
                continue
            etat.lots_abandonnes.append((lot["debut"], lot["fin"]))
            abandonnees |= plage
            en_echec |= fautives
            erreurs.extend((lot.get("erreurs") or [])[-1:])

        couvertes = faites | abandonnees | rejouees | a_isoler
        manquantes = [n for n in range(1, nb_pages + 1) if n not in couvertes]
        etat.lots_a_faire = sorted(
            a_rejouer + [(n, n) for n in a_isoler] + self._decouper(manquantes)
        )
        return etat, textes, en_echec, erreurs

    def etat(
        self, chemin_pdf: Path, sha: Optional[str] = None, nb_pages: Optional[int] = None
    ) -> EtatDocument:
        """Avancement d'un document, sans rien convertir."""
        sha = sha or sha256_fichier(Path(chemin_pdf))
        nb_pages = nb_pages if nb_pages is not None else compter_pages(Path(chemin_pdf))
        return self._inventaire(sha, nb_pages)[0]

    def noter_echec(
        self, chemin_pdf: Path, debut: int, fin: int, raison: str, sha: Optional[str] = None
    ) -> None:
        """
        Compte une tentative perdue pour un lot, sans le convertir.

        Appele par le superviseur quand le processus de travail meurt PENDANT
        ce lot (memoire epuisee, plantage natif, blocage) : sans ce compte, le
        meme lot serait relance indefiniment et tuerait chaque nouveau
        processus.
        """
        sha = sha or sha256_fichier(Path(chemin_pdf))
        chemin = self._chemin_lot(sha, debut, fin)
        precedent = self._lire_lot(chemin) or {}
        if precedent.get("statut") == "ok":
            return
        self._ecrire_lot(chemin, self._lot_en_echec(debut, fin, precedent, raison))

    # ==================== EXTRACTION ====================

    def extraire(self, chemin_pdf: Path, cache_seulement: bool = False) -> ResultatExtraction:
        """
        Markdown pagine du PDF, lot par lot, chaque lot mis en cache aussitot.

        Args:
            cache_seulement: ne rien convertir. Leve ExtractionEnAttente si un
                lot reste a tenter — la passe d'indexation saute alors le
                document, que la passe d'extraction n'a pas fini.

        Raises:
            PdfExtractionError: fichier introuvable ou illisible
            ExtractionEnAttente: cache_seulement et document incomplet
        """
        chemin_pdf = Path(chemin_pdf)
        if not chemin_pdf.exists():
            raise PdfExtractionError(f"Fichier introuvable: {chemin_pdf}")
        try:
            nb_pages = compter_pages(chemin_pdf)
        except Exception as e:
            raise PdfExtractionError(f"PDF illisible ({chemin_pdf.name}) : {e}") from e
        sha = sha256_fichier(chemin_pdf)

        etat = self._inventaire(sha, nb_pages)[0]
        if etat.lots_a_faire:
            if cache_seulement:
                debut, fin = etat.lots_a_faire[0]
                raise ExtractionEnAttente(
                    f"{chemin_pdf.name} : pages {debut}-{fin} pas encore extraites"
                )
            for debut, fin in etat.lots_a_faire:
                self.traiter_lot(chemin_pdf, sha, debut, fin)

        etat, textes, en_echec, erreurs = self._inventaire(sha, nb_pages)
        en_echec |= {n for debut, fin in etat.lots_a_faire for n in range(debut, fin + 1)}

        pages: List[str] = []
        illisibles: List[int] = []
        for n in range(1, nb_pages + 1):
            brut = textes.get(n, "")
            propre = nettoyer_page(brut)
            # Une page dont l'OCR a lu quelque chose mais dont il ne reste rien
            # de lisible : rendu rate, ou cachet seul. Signalee, pas tue.
            if brut.strip() and not _lisible(propre) and n not in en_echec:
                illisibles.append(n)
            pages.append(propre)

        return ResultatExtraction(
            texte=joindre_pages(pages),
            nb_pages=nb_pages,
            pages_en_echec=sorted(en_echec),
            pages_illisibles=illisibles,
            erreurs=erreurs,
        )

    def traiter_lot(self, chemin_pdf: Path, sha: str, debut: int, fin: int) -> Dict[str, Any]:
        """
        Convertit UN lot s'il reste a tenter, et l'ecrit en cache aussitot.

        C'est l'unite de travail de la passe d'extraction : le superviseur
        note le lot en cours avant l'appel, pour pouvoir compter une tentative
        perdue si le processus meurt pendant la conversion.

        Raises:
            Exception: si le convertisseur ne peut pas etre construit. Une
                panne d'initialisation n'est PAS l'echec d'un lot : comptee
                comme telle, elle abandonnait tout le corpus, lot apres lot.
        """
        chemin_lot = self._chemin_lot(sha, debut, fin)
        precedent = self._lire_lot(chemin_lot)
        if self._lot_acquis(precedent):
            return precedent
        convertisseur = self._convertisseur_pret()
        lot = self._convertir_lot(convertisseur, Path(chemin_pdf), debut, fin, precedent=precedent)
        self._ecrire_lot(chemin_lot, lot)
        return lot

    async def extract_text(self, file_path: Path) -> str:
        resultat = await asyncio.to_thread(self.extraire, Path(file_path))
        return resultat.texte

    def precharger(self) -> None:
        """Construit le convertisseur maintenant, pour qu'une panne se voie avant tout lot."""
        self._convertisseur_pret()

    # ==================== CONVERSION ====================

    @staticmethod
    def _lot_en_echec(
        debut: int, fin: int, precedent: Dict[str, Any], raison: str
    ) -> Dict[str, Any]:
        return {
            "schema": SCHEMA_CACHE,
            "debut": debut,
            "fin": fin,
            "statut": "echec",
            "tentatives": precedent.get("tentatives", 0) + 1,
            "erreurs": (precedent.get("erreurs") or [])[-4:] + [raison[:500]],
            "pages_fautives": precedent.get("pages_fautives") or list(range(debut, fin + 1)),
            # On garde ce qu'une tentative precedente avait pu lire
            "pages": precedent.get("pages") or {str(n): "" for n in range(debut, fin + 1)},
        }

    def _convertir_lot(
        self, convertisseur, chemin_pdf: Path, debut: int, fin: int,
        precedent: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        precedent = precedent or {}
        t0 = time.perf_counter()
        try:
            from docling.datamodel.base_models import DocumentStream

            flux = rasteriser(chemin_pdf, debut, fin)
            with self._verrou_conversion:
                res = convertisseur.convert(
                    DocumentStream(name=f"{chemin_pdf.stem}_p{debut}-{fin}.tiff", stream=flux),
                    raises_on_error=False,
                )
                statut = str(getattr(res.status, "value", res.status))
                erreurs = [str(getattr(e, "error_message", e))[:300] for e in (res.errors or [])]
                # page_no est indexe a partir de 1 dans le document image
                fautives = sorted({
                    debut + e.page_no - 1 for e in (res.errors or [])
                    if getattr(e, "page_no", None) and 1 <= e.page_no <= fin - debut + 1
                })
                brut = self._exporter_pages(res.document, debut, fin)
            del res, flux
        except Exception as e:
            logger.error(
                f"❌ Docling : {chemin_pdf.name} pages {debut}-{fin} : {type(e).__name__}: {e}"
            )
            return self._lot_en_echec(debut, fin, precedent, f"{type(e).__name__}: {e}")
        finally:
            gc.collect()

        # Fusion page par page : une tentative ne peut qu'ajouter du texte. Une
        # tentative plus pauvre (machine chargee, delai atteint plus tot)
        # ecrasait le texte deja obtenu par la precedente.
        anciennes = precedent.get("pages") or {}
        pages = {n: md if md.strip() else anciennes.get(n, "") for n, md in brut.items()}

        if statut != "success":
            lot = self._lot_en_echec(
                debut, fin, precedent, f"statut {statut} : {'; '.join(erreurs)[:400]}"
            )
            lot["pages"] = pages
            # Les pages fautives que Docling designe ; a defaut, celles restees vides
            lot["pages_fautives"] = fautives or [
                int(n) for n, md in pages.items() if not md.strip()
            ] or list(range(debut, fin + 1))
            logger.warning(
                f"⚠️ Docling : {chemin_pdf.name} pages {debut}-{fin} en {statut} "
                f"(tentative {lot['tentatives']}/{TENTATIVES_MAX})"
            )
            return lot

        return {
            "schema": SCHEMA_CACHE,
            "debut": debut,
            "fin": fin,
            "statut": "ok",
            "tentatives": precedent.get("tentatives", 0) + 1,
            "erreurs": [],
            "duree_s": round(time.perf_counter() - t0, 2),
            "pages": pages,
        }

    @staticmethod
    def _exporter_pages(document, debut: int, fin: int) -> Dict[str, str]:
        """
        Markdown BRUT de chaque page — le nettoyage se fait a la lecture.

        Le document image numerote ses pages a partir de 1 ; elles sont
        renumerotees sur la page physique du PDF.
        - traverse_pictures : en OCR pleine page, le texte reconnu dans une
          image fait partie de la page ; sans ce reglage, il disparait.
        - escape_html / escape_underscores : desactives, sinon « & » devient
          « &amp; » et « _ » « \\_ » dans le contenu et les vecteurs.
        - image_placeholder vide : pas de « <!-- image --> » dans les articles.
        - en-tetes et pieds de page (couche FURNITURE) exclus, comme par
          defaut : numeros de page et mentions repetees n'ont rien a faire
          dans un article.
        """
        pages: Dict[str, str] = {}
        for i, n in enumerate(range(debut, fin + 1), start=1):
            markdown = ""
            if document is not None:
                markdown = document.export_to_markdown(
                    page_no=i,
                    traverse_pictures=True,
                    escape_html=False,
                    escape_underscores=False,
                    image_placeholder="",
                )
            pages[str(n)] = markdown
        return pages

    def _convertisseur_pret(self):
        with self._verrou_init:
            if self._convertisseur is None:
                self._convertisseur = self._construire()
            return self._convertisseur

    def _construire(self):
        """Convertisseur Docling pour des pages en image, cree une fois par processus."""
        if settings.EXTRACTION_HORS_LIGNE:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        from docling.datamodel.accelerator_options import AcceleratorOptions
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            OcrMode,
            PdfPipelineOptions,
            RapidOcrOptions,
            TesseractCliOcrOptions,
        )
        from docling.document_converter import DocumentConverter, ImageFormatOption

        if self.moteur_ocr == "rapidocr_torch":
            ocr = RapidOcrOptions(lang=[LANGUE_RAPIDOCR], backend="torch", mode=OcrMode.FULL_PAGE)
        elif self.moteur_ocr == "rapidocr":
            ocr = RapidOcrOptions(
                lang=[LANGUE_RAPIDOCR],
                backend="onnxruntime",
                mode=OcrMode.FULL_PAGE,
                # onnxruntime n'a ici que le fournisseur CPU : on coupe la
                # tentative CUDA que Docling deduirait du peripherique.
                rapidocr_params={"EngineConfig.onnxruntime.use_cuda": False},
            )
        else:
            _desactiver_osd_tesseract()
            ocr = TesseractCliOcrOptions(
                lang=[LANGUE_TESSERACT], tesseract_cmd=self._commande_tesseract(),
                mode=OcrMode.FULL_PAGE,
            )

        options = PdfPipelineOptions(
            do_ocr=True,
            ocr_options=ocr,
            do_table_structure=True,
            accelerator_options=AcceleratorOptions(device=self.device, num_threads=self.fils),
            document_timeout=self.timeout_lot_s,
        )
        t0 = time.perf_counter()
        convertisseur = DocumentConverter(
            allowed_formats=[InputFormat.IMAGE],
            format_options={InputFormat.IMAGE: ImageFormatOption(pipeline_options=options)},
        )
        convertisseur.initialize_pipeline(InputFormat.IMAGE)
        logger.info(
            f"✅ Docling pret en {time.perf_counter() - t0:.1f} s "
            f"(OCR {self.moteur_ocr}, peripherique {self.device}, {self.fils} fils)"
        )
        return convertisseur


def _desactiver_osd_tesseract() -> None:
    """
    Coupe la detection d'orientation que Docling lance avant Tesseract.

    Docling execute « tesseract --psm 0 -l osd » sur chaque page et tourne
    l'image des que l'orientation lue differe de 0, SANS seuil de confiance.
    Mesure sur le banc : la desactiver fait passer le F1 de 0,79 a 0,85 et la
    vitesse de 5,1 a 3,8 s par page. Les pages du corpus sont droites.
    """
    try:
        import pandas as pd
        from docling.models.stages.ocr.tesseract_ocr_cli_model import TesseractOcrCliModel
    except ImportError:
        return

    def _sans_osd(self, ifilename):
        return pd.DataFrame({"key": ["Orientation in degrees"], "value": [" 0"]})

    if hasattr(TesseractOcrCliModel, "_perform_osd"):
        TesseractOcrCliModel._perform_osd = _sans_osd
    else:
        logger.warning("Docling : _perform_osd introuvable, orientation non desactivee")
