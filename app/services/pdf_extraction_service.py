"""
Extraction de PDF en markdown par Gemini multimodal.

Remplace LlamaParse. La raison n'est pas le prix mais la PAGINATION.

LlamaParse rend le document en un bloc et ne signale pas ses coupures de page ;
`_split_markdown_pages` les reconstruisait a l'heuristique, en coupant sur les
lignes de trois tirets, et sa propre docstring admettait la limite : « la
pagination ne devient exacte qu'avec un moteur qui renvoie un numero de page
explicite ». Resultat mesure sur la base : les 710 articles portaient tous
`page_number = 1`, et sur les 27 lois, 271 pages de PDF ne donnaient que 27
pages extraites — couverture moyenne 0,60.

Gemini est ce moteur. Mesure sur le Code Minier, dont la sortie LlamaParse est
en base pour comparaison :

    densite            2352 caracteres/page  contre 2405 — equivalent
    marqueurs de page  1 par page            contre 1 pour 69 pages
    lot pages 21 a 32  numerote 21..32       exactement

CONTRAT DE SORTIE, impose par `text_chunker.py` et non negociable :
`<<PAGE:n>>\\n<contenu>` par page, pages jointes par `\\n\\n`, numerotation a
partir de 1 sur la page PHYSIQUE du PDF. Sans espace apres le deux-points :
`text_chunker` compile `r'<<PAGE:(\\d+)>>'`, et un `<<PAGE: 12>>` serait
invisible pour lui tout en etant retire ailleurs.

Author: JuriX Team
"""

import asyncio
import hashlib
import io
import json
import logging
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from app.core.config import settings
from app.services.gemini_service import (
    _finish_reason,
    _is_overloaded,
    _is_quota_exhausted,
    _visible_text,
    retry_after_seconds,
)
from app.utils.markdown_cleanup import strip_stamp_blocks

logger = logging.getLogger(__name__)


def _decrire(err: Exception) -> str:
    """
    Description utilisable d'une exception, meme muette.

    `httpx.ReadTimeout` a un `str()` VIDE : le message remontait « Extraction de
    X impossible : » suivi de rien, et atterrissait tel quel dans
    `laws.processing_error`. Un message qui ne dit rien coute autant a lire
    qu'un message juste, et ne sert a rien.
    """
    texte = str(err).strip()
    return texte if texte else type(err).__name__


def _est_un_depassement(err: Exception) -> bool:
    """Vrai pour un depassement de delai reseau, quelle que soit la couche."""
    return "Timeout" in type(err).__name__ or isinstance(err, TimeoutError)


class PdfExtractionError(Exception):
    """
    Extraction impossible.

    Son `str()` atterrit tel quel dans `laws.processing_error`, colonne visible
    par l'administrateur : le message doit se lire, pas se decoder.
    """


class PdfExtractionQuotaError(PdfExtractionError):
    """
    Quota de generation epuise chez le fournisseur.

    Type distinct pour que les scripts d'ingestion s'arretent proprement et
    reprennent le lendemain, au lieu de compter le document comme un echec
    definitif. Sous-classe de PdfExtractionError pour que le pipeline, lui,
    n'ait rien a savoir de cette nuance.
    """


class PdfExtractionRefusedError(PdfExtractionError):
    """
    Le modele refuse de transcrire une page, la jugeant recitee.

    Gemini rend `finish_reason=RECITATION` sur certaines pages d'actes officiels
    — vraisemblablement celles dont la forme lui est familiere. Mesure sur un
    echantillon de 7 documents : UNE page de garde refusee, soit environ 14 %.

    Ce n'est PAS transitoire : la meme page refuse a l'identique, quelle que
    soit la formulation de la consigne (trois variantes essayees, toutes
    refusees) et quelle que soit la taille du lot. En revanche c'est PAR PAGE :
    dans le document mesure, la page 1 refuse et les pages 2 et 3 s'extraient
    normalement.
    """


# Consigne d'extraction. Ecrite en francais parce que le corpus l'est, et
# formulee en regles numerotees : les consignes en prose donnaient des
# preambules du genre « Voici la transcription du document ».
_CONSIGNE = """Transcris ce document juridique en Markdown, fidelement et integralement.

{pagination}

Regles :
1. Commence CHAQUE page par une ligne `<<PAGE:n>>` seule, sans espace apres le deux-points.
2. Conserve les numeros d'article exactement tels qu'ils apparaissent, en debut de ligne.
3. Restitue les tableaux en Markdown, sans en omettre de ligne.
4. N'ajoute aucun commentaire, aucune introduction, aucune conclusion, aucune note.
5. Ignore les filigranes, les cachets et les mentions de copie certifiee conforme.
6. Si une page est vide, emets quand meme son marqueur, suivi de rien."""


class GeminiPdfExtractor:
    """
    Extracteur PDF -> markdown pagine.

    Interface volontairement identique a celle de l'ancien service : le
    pipeline (`app/tasks/process_law.py:_extract_pdf_text`) ne fait que
    `is_available()` puis `await extract_text(path)`, et convertit toute
    exception en `status='refused'` + `processing_error`.
    """

    CACHE_SCHEMA = 3

    # Nombre de tentatives sur une saturation passagere (503). Le palier gratuit
    # en rend par salves de quelques secondes ; un 429, lui, n'est JAMAIS
    # retente — cela brulerait des appels sur un quota deja epuise.
    OVERLOAD_MAX_ATTEMPTS = 3
    OVERLOAD_BASE_DELAY_S = 2.0

    # Pause entre deux appels d'un meme document. Le palier gratuit limite AUSSI
    # le nombre de requetes par minute — le 429 renvoyait « reprise dans 23
    # secondes », pas « demain ». Sans cette pause, un document de plusieurs
    # lots epuise la fenetre a son deuxieme appel et s'arrete, alors qu'il
    # suffisait d'attendre.
    INTER_CALL_DELAY_S = 6.0

    # Budget de sortie par appel. La limite du modele est de 65 536 jetons ;
    # a ~2400 caracteres par page, un lot de 20 pages produit ~48 Ko, soit
    # ~13 000 jetons. La marge couvre la reflexion interne du modele.
    MAX_OUTPUT_TOKENS = 60_000

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        pages_per_call: Optional[int] = None,
        cache_dir: Optional[Path] = None,
    ):
        # `if api_key is None` et non `api_key or ...` : une chaine vide passee
        # EXPLICITEMENT veut dire « pas de cle », et retombait sinon sur le
        # reglage global — rendant intestable le cas du service non configure.
        self.api_key = settings.GEMINI_API_KEY if api_key is None else api_key
        self.model_name = model_name or settings.GEMINI_MODEL
        self.pages_per_call = pages_per_call or settings.PDF_EXTRACTION_PAGES_PER_CALL
        self.max_batch_mb = settings.PDF_EXTRACTION_MAX_BATCH_MB

        self.cache_dir: Optional[Path] = Path(
            cache_dir or getattr(settings, "OCR_CACHE_DIR", "./data/ocr_cache")
        )
        try:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"⚠️ Cache d'extraction indisponible ({e}), desactive")
            self.cache_dir = None

        # Client construit paresseusement : `is_available()` doit pouvoir
        # repondre sans cle, et le pipeline l'interroge avant tout travail.
        self._client: Optional[genai.Client] = None

        # Pages que le modele a refuse de transcrire lors de la derniere
        # extraction. Lue par le script de re-extraction pour signaler les
        # documents incomplets plutot que de les faire passer pour entiers.
        self.pages_refusees: List[int] = []

    # ==================== SURFACE PUBLIQUE ====================

    def is_available(self) -> bool:
        return bool(self.api_key)

    async def extract_text(self, file_path: Path) -> str:
        """
        Rend le markdown pagine du PDF.

        Raises:
            PdfExtractionError: extraction impossible
            PdfExtractionQuotaError: quota du fournisseur epuise
        """
        pages = await self.extract_pages(file_path)
        if not pages:
            raise PdfExtractionError(f"Aucune page extraite de {file_path.name}")

        # Numerotation AVANT filtrage : une page blanche consomme son numero,
        # ce qui garde les suivantes alignees sur la page physique du PDF.
        parts = [f"<<PAGE:{i}>>\n{md}" for i, md in enumerate(pages, start=1) if md.strip()]
        texte = "\n\n".join(parts)
        logger.info(
            f"✅ Gemini : {len(pages)} page(s), {len(texte)} caracteres "
            f"depuis {file_path.name}"
        )
        return texte

    async def extract_pages(self, file_path: Path) -> List[str]:
        """Markdown page par page, index 0 = page 1 du PDF."""
        if not self.is_available():
            raise PdfExtractionError(
                "GEMINI_API_KEY absente : extraction impossible. "
                "Aucun repli degrade n'est utilise."
            )
        if not file_path.exists():
            raise PdfExtractionError(f"Fichier introuvable: {file_path}")

        # Remis a zero a chaque document : cette liste decrit CE document.
        self.pages_refusees: List[int] = []

        cle = await asyncio.to_thread(self._sha256, file_path)
        en_cache = self._read_cache(cle)
        if en_cache is not None:
            logger.info(f"♻️ Cache d'extraction : {file_path.name} ({len(en_cache)} page(s))")
            return en_cache

        lots = await asyncio.to_thread(self._decouper_en_lots, file_path)
        pages: List[str] = []
        for index, (premiere, octets) in enumerate(lots):
            if index:
                await asyncio.sleep(self.INTER_CALL_DELAY_S)
            try:
                pages.extend(await self._extraire_lot(octets, premiere, file_path.name))
            except PdfExtractionRefusedError:
                # Le refus est PAR PAGE, pas par document : mesure sur un
                # document reel, la page 1 refuse et les pages 2 et 3 passent.
                # On rejoue donc le lot page par page pour ne perdre que ce qui
                # est reellement refuse.
                logger.warning(
                    f"⚠️ Lot refuse a partir de la page {premiere}, reprise page par page"
                )
                pages.extend(
                    await self._extraire_lot_page_par_page(
                        file_path, premiere, len(octets), file_path.name
                    )
                )

        pages = [strip_stamp_blocks(p) for p in pages]
        self._write_cache(cle, pages)
        return pages

    async def _extraire_lot_page_par_page(
        self, file_path: Path, premiere_page: int, _taille: int, nom: str
    ) -> List[str]:
        """
        Rejoue un lot refuse, une page a la fois.

        Une page qui refuse encore est rendue VIDE plutot que de faire echouer
        tout le document : perdre une page de garde vaut mieux que perdre les
        soixante-huit autres. Le refus est journalise en ERREUR — il ne doit pas
        se noyer — et la liste des pages perdues remonte a l'appelant, qui
        decide.
        """
        pages_du_lot = await asyncio.to_thread(
            self._pages_du_lot, file_path, premiere_page
        )
        resultat: List[str] = []
        perdues: List[int] = []

        for decalage, octets in enumerate(pages_du_lot):
            numero = premiere_page + decalage
            try:
                extrait = await self._extraire_lot(octets, numero, nom)
                resultat.extend(extrait)
            except PdfExtractionRefusedError:
                logger.error(f"❌ Page {numero} de {nom} refusee : contenu perdu")
                perdues.append(numero)
                resultat.append("")

        if perdues:
            self.pages_refusees.extend(perdues)
        return resultat

    def _pages_du_lot(self, file_path: Path, premiere_page: int) -> List[bytes]:
        """Chaque page du lot, isolee, en memoire."""
        from pypdf import PdfReader, PdfWriter

        lecteur = PdfReader(str(file_path))
        debut = premiere_page - 1
        fin = min(debut + self.pages_per_call, len(lecteur.pages))
        sortie = []
        for index in range(debut, fin):
            ecrivain = PdfWriter()
            ecrivain.add_page(lecteur.pages[index])
            tampon = io.BytesIO()
            ecrivain.write(tampon)
            sortie.append(tampon.getvalue())
        return sortie

    async def health_check(self) -> Dict[str, Any]:
        if not self.is_available():
            return {"service": "GeminiPdfExtractor", "status": "unconfigured"}
        return {
            "service": "GeminiPdfExtractor",
            "status": "healthy",
            "model": self.model_name,
            "pages_per_call": self.pages_per_call,
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
        }

    def compter_appels(self, file_path: Path) -> int:
        """
        Nombre d'appels que coutera ce document, cache non compris.

        Sert au `--dry-run` du script de ré-extraction : sur un palier limite a
        20 appels par jour, savoir AVANT de commencer vaut mieux que decouvrir
        a mi-parcours.
        """
        total = self._compter_pages(file_path)
        return (total + self.pages_per_call - 1) // self.pages_per_call

    # ==================== APPEL AU MODELE ====================

    @property
    def client(self) -> genai.Client:
        if self._client is None:
            if not self.api_key:
                raise PdfExtractionError("GEMINI_API_KEY absente")
            # Meme delai que les autres services Gemini : sans http_options, un
            # appel peut bloquer indefiniment sur une lecture de socket. Un
            # incident de quarante minutes l'a etabli.
            self._client = genai.Client(
                api_key=self.api_key,
                # Delai PROPRE a l'extraction, et non celui du chat : un lot de
                # vingt pages scannees demande plusieurs minutes de traitement
                # cote modele, la ou une question de RAG se compte en secondes.
                # Mesure : un lot de 9,8 Mo depassait les 120 s du reglage
                # general et remontait un httpx.ReadTimeout muet.
                http_options=types.HttpOptions(
                    timeout=settings.PDF_EXTRACTION_TIMEOUT_S * 1000
                ),
            )
        return self._client

    async def _extraire_lot(
        self, octets: bytes, premiere_page: int, nom: str
    ) -> List[str]:
        """Envoie un lot de pages et rend son markdown, page par page."""
        if premiere_page == 1:
            pagination = "Ce document commence a sa premiere page."
        else:
            # SANS cette phrase, le modele renumerote chaque lot a partir de 1
            # et chaque lot ecrase la pagination du precedent. Verifie : lot des
            # pages 21 a 32 rendu « 21..32 » avec la phrase, « 1..12 » sans.
            pagination = (
                f"ATTENTION : ces pages sont un EXTRAIT. La premiere page fournie "
                f"est la page {premiere_page} du document complet. Numerote donc "
                f"les marqueurs a partir de {premiere_page}, en numerotation ABSOLUE."
            )

        contenu = [
            types.Part.from_bytes(data=octets, mime_type="application/pdf"),
            _CONSIGNE.format(pagination=pagination),
        ]
        config = types.GenerateContentConfig(
            max_output_tokens=self.MAX_OUTPUT_TOKENS,
            # Une transcription n'a pas a etre creative. La temperature nulle
            # rend aussi le resultat reproductible d'un appel a l'autre, ce qui
            # est indispensable pour comparer deux extractions.
            temperature=0.0,
        )

        for essai in range(1, self.OVERLOAD_MAX_ATTEMPTS + 1):
            try:
                # client.aio : surface asynchrone native du SDK. La surface
                # synchrone bloquerait la boucle pendant tout l'aller-retour,
                # soit des dizaines de secondes par lot.
                reponse = await self.client.aio.models.generate_content(
                    model=self.model_name, contents=contenu, config=config
                )
                break
            except Exception as err:
                if _is_quota_exhausted(err):
                    delai = retry_after_seconds(err)
                    raise PdfExtractionQuotaError(
                        f"Quota d'extraction epuise. Reprise possible dans {delai} secondes."
                    ) from err
                # Un depassement de delai est transitoire au meme titre qu'une
                # saturation : sur un lot de 10 Mo d'images scannees, la
                # premiere tentative depassait les 120 s du reglage general.
                transitoire = _is_overloaded(err) or _est_un_depassement(err)
                if not transitoire or essai == self.OVERLOAD_MAX_ATTEMPTS:
                    raise PdfExtractionError(
                        f"Extraction de {nom} impossible : {_decrire(err)}"
                    ) from err
                attente = self.OVERLOAD_BASE_DELAY_S * (2 ** (essai - 1))
                logger.warning(
                    f"⚠️ Modele sature (essai {essai}/{self.OVERLOAD_MAX_ATTEMPTS}), "
                    f"nouvelle tentative dans {attente:.1f}s"
                )
                await asyncio.sleep(attente)

        # `_visible_text` et non `reponse.text` : sur un modele a raisonnement,
        # `.text` concatene la reflexion interne avec la reponse.
        texte = _visible_text(reponse)
        if not texte.strip():
            raison = _finish_reason(reponse)
            if raison == "RECITATION":
                raise PdfExtractionRefusedError(
                    f"Le modele refuse de transcrire {nom}, pages {premiere_page} et "
                    f"suivantes (recitation)."
                )
            raise PdfExtractionError(
                f"Reponse vide du modele sur {nom}, pages a partir de "
                f"{premiere_page} (finish_reason={raison})"
            )
        return self._decouper_par_marqueur(texte, premiere_page)

    @staticmethod
    def _decouper_par_marqueur(texte: str, premiere_page: int) -> List[str]:
        """
        Separe le markdown sur les marqueurs `<<PAGE:n>>`.

        Le marqueur lui-meme est RETIRE : `extract_text` le repose ensuite a
        partir de la position dans la liste. C'est ce qui garantit que la
        numerotation finale est continue meme si le modele en saute un.
        """
        import re

        morceaux = re.split(r"<<PAGE:\s*(\d+)\s*>>", texte)
        if len(morceaux) < 3:
            # Aucun marqueur : le lot entier compte pour une page. Mieux vaut
            # une page unique qu'une exception — le contenu, lui, est bon.
            return [texte.strip()]

        pages: Dict[int, str] = {}
        for i in range(1, len(morceaux) - 1, 2):
            numero = int(morceaux[i])
            pages[numero] = morceaux[i + 1].strip()

        # Reconstitution dense : de la premiere page annoncee au maximum vu,
        # les numeros manquants deviennent des pages vides plutot que des trous.
        if not pages:
            return [texte.strip()]
        fin = max(pages)
        return [pages.get(n, "") for n in range(premiere_page, fin + 1)]

    # ==================== DECOUPAGE DU PDF ====================

    @staticmethod
    def _compter_pages(file_path: Path) -> int:
        from pypdf import PdfReader

        try:
            return len(PdfReader(str(file_path)).pages)
        except Exception as e:
            raise PdfExtractionError(f"PDF illisible ({file_path.name}) : {e}") from e

    def _decouper_en_lots(self, file_path: Path) -> List[tuple]:
        """
        Rend [(numero_de_premiere_page, octets_du_lot), ...].

        Decoupage EN MEMOIRE : l'ancien service ecrivait des fichiers
        temporaires a cote de l'original et les effacait dans un `finally`. Un
        `BytesIO` ne laisse rien a nettoyer et ne peut pas polluer le repertoire
        d'upload si le processus meurt.
        """
        from pypdf import PdfReader, PdfWriter

        lecteur = PdfReader(str(file_path))
        total = len(lecteur.pages)
        octets_max = int(self.max_batch_mb * 1_000_000)

        if total <= self.pages_per_call:
            entier = file_path.read_bytes()
            if len(entier) <= octets_max:
                return [(1, entier)]

        # Deux bornes, pas une. Le nombre de pages ne dit rien du poids : vingt
        # pages scannees pesent 10 Mo la ou vingt pages de texte en pesent 1, et
        # c'est le POIDS qui a fait depasser le delai sur la Loi de finances.
        # On ferme donc un lot des qu'une des deux bornes est atteinte.
        lots = []
        debut = 0
        while debut < total:
            ecrivain = PdfWriter()
            fin = debut
            while fin < total and (fin - debut) < self.pages_per_call:
                ecrivain.add_page(lecteur.pages[fin])
                fin += 1
                tampon = io.BytesIO()
                ecrivain.write(tampon)
                if tampon.tell() >= octets_max and fin - debut > 1:
                    # Le lot vient de depasser le poids : on le referme AVANT la
                    # page qui l'a fait deborder, sauf si c'est la premiere —
                    # une page seule trop lourde part quand meme, faute de mieux.
                    fin -= 1
                    ecrivain = PdfWriter()
                    for page in lecteur.pages[debut:fin]:
                        ecrivain.add_page(page)
                    break

            tampon = io.BytesIO()
            ecrivain.write(tampon)
            lots.append((debut + 1, tampon.getvalue()))
            debut = fin

        poids = [len(o) / 1e6 for _, o in lots]
        logger.info(
            f"✂️ {file_path.name} : {total} pages -> {len(lots)} lot(s), "
            f"le plus lourd {max(poids):.1f} Mo"
        )
        return lots

    # ==================== CACHE SHA256 ====================

    @staticmethod
    def _sha256(file_path: Path) -> str:
        h = hashlib.sha256()
        with file_path.open("rb") as fh:
            for block in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(block)
        return h.hexdigest()

    def _cache_path(self, key: str) -> Optional[Path]:
        return self.cache_dir / f"{key}.json" if self.cache_dir else None

    def _read_cache(self, key: str) -> Optional[List[str]]:
        path = self._cache_path(key)
        if not path or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            # `engine` remplace l'ancien `tier` : une entree produite par un
            # autre moteur — ou par un autre modele — n'est pas la meme
            # extraction, et surtout pas la meme pagination.
            if payload.get("engine") != self.model_name:
                return None
            if payload.get("schema") != self.CACHE_SCHEMA:
                return None

            pages = payload.get("pages")
            # Une liste vide comptait autrefois comme un succes : une extraction
            # ratee se gravait et se rejouait indefiniment, et le seul moyen
            # d'en sortir etait d'effacer le fichier a la main.
            if not pages:
                logger.warning("⚠️ Entree de cache vide, ignoree")
                return None
            return pages
        except Exception as e:
            logger.warning(f"⚠️ Cache d'extraction illisible ({e}), ignore")
            return None

    def _write_cache(self, key: str, pages: List[str]) -> None:
        path = self._cache_path(key)
        if not path:
            return
        # Ne jamais graver un echec.
        if not any(p.strip() for p in pages):
            logger.warning("⚠️ Extraction vide, non mise en cache")
            return
        try:
            # Ecriture atomique : un plantage en cours d'ecriture laissait un
            # JSON tronque qui occupait la place de l'entree valide.
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "schema": self.CACHE_SCHEMA,
                        "engine": self.model_name,
                        "pages": pages,
                        "cached_at": time.time(),
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            tmp.replace(path)
        except OSError as e:
            logger.warning(f"⚠️ Ecriture du cache d'extraction echouee ({e})")


@lru_cache()
def get_pdf_extractor() -> GeminiPdfExtractor:
    """Instance partagee. Le client Gemini et le cache disque sont reutilises."""
    return GeminiPdfExtractor()
