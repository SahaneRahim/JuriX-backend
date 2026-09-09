"""
Accès aux documents d'origine (PDF), sur disque local ou sur un magasin HTTPS.

POURQUOI CE MODULE EXISTE. Les cinq routes qui servent un PDF lisaient le
fichier dans `./data/uploads`. Sur un hébergeur au disque éphémère — c'est le
cas de toutes les offres gratuites — les lignes en base survivent au
redéploiement, pas les fichiers : le visualiseur et le téléchargement
répondraient 404 sur chaque loi. Ce module déplace la résolution derrière une
indirection réglée par `DOCUMENTS_BASE_URL`.

    vide      → disque local, comportement historique, développement inchangé
    non vide  → récupération HTTPS, avec cache disque

Le magasin est donc interchangeable sans toucher une ligne de code : Supabase
Storage, un dépôt public, un stockage objet — une variable d'environnement.

LA GARDE À NE JAMAIS RETIRER. `_FILE_ID_RE` protégeait un chemin de fichier ;
ici elle protège une URL. Interpoler un `file_id` non validé dans
`DOCUMENTS_BASE_URL` permettrait à une valeur contenant `../` ou
`@evil.example/` de faire sortir la requête du magasin — c'est-à-dire de
transformer le serveur en relais de requêtes (SSRF). La valeur vient de la base
et non de la requête HTTP, donc ce n'est pas exploitable aujourd'hui ; c'est
précisément pour cela que la faille apparaîtrait sans bruit le jour où une route
écrira un `file_id` d'après une entrée utilisateur. On valide d'abord, on
construit l'URL ensuite.
"""

import asyncio
import logging
import os
from pathlib import Path

import httpx

from app.core.config import settings
from app.utils.file_utils import _FILE_ID_RE, resolve_upload_path

logger = logging.getLogger(__name__)

# Extensions sondées, dans l'ordre. Miroir exact de `resolve_upload_path` : les
# deux modes doivent résoudre le même identifiant vers le même document.
EXTENSIONS = (".pdf", ".docx")

# Taille des tranches de téléchargement. Le plus gros document du corpus pèse
# 26 Mo ; en flux, il ne passe jamais en mémoire.
TAILLE_TRANCHE = 65536


class IdentifiantInvalide(ValueError):
    """`file_id` hors motif. Ni chemin ni URL n'est construit."""


class DocumentIntrouvable(Exception):
    """Le document n'existe pas — 404 en amont, ou absent du disque."""


class DocumentInaccessible(Exception):
    """
    L'amont a échoué : 5xx, délai dépassé, réseau.

    Distinct de `DocumentIntrouvable` à dessein. Répondre 404 quand c'est notre
    magasin qui est en panne accuserait l'utilisateur d'un défaut qui n'est pas
    le sien, et ferait disparaître le document de l'interface au lieu de
    signaler une panne passagère.
    """


def stockage_distant() -> bool:
    """Vrai si les documents sont servis en HTTPS plutôt que depuis le disque."""
    return bool(settings.DOCUMENTS_BASE_URL)


def url_publique(file_id: str, extension: str = ".pdf") -> str:
    """
    URL du document dans le magasin distant.

    Raises:
        IdentifiantInvalide: le motif n'est pas respecté. Voir l'avertissement
            SSRF en tête de module — cette vérification vient AVANT la
            construction de l'URL, jamais après.
    """
    if not _FILE_ID_RE.match(file_id or ""):
        raise IdentifiantInvalide(f"Identifiant de fichier invalide : {file_id!r}")
    return f"{settings.DOCUMENTS_BASE_URL.rstrip('/')}/{file_id}{extension}"


# Un verrou par identifiant : deux rendus de page simultanés sur la même loi ne
# doivent télécharger qu'une fois. Légitime en mémoire de processus parce que
# l'image tourne avec `--workers 1` (registre WebSocket en processus).
_verrous: dict[str, asyncio.Lock] = {}


def _verrou(file_id: str) -> asyncio.Lock:
    verrou = _verrous.get(file_id)
    if verrou is None:
        verrou = _verrous[file_id] = asyncio.Lock()
    return verrou


def _repertoire_du_cache() -> Path:
    chemin = Path(settings.DOCUMENTS_CACHE_DIR)
    chemin.mkdir(parents=True, exist_ok=True)
    return chemin


def _elaguer_le_cache(repertoire: Path) -> None:
    """
    Ramène le cache sous son plafond, en supprimant les moins récemment lus.

    Sans elle, `/tmp` grossit jusqu'à saturer le conteneur — et un conteneur
    dont le disque est plein ne redémarre pas proprement.
    """
    plafond = settings.DOCUMENTS_CACHE_MAX_MB * 1024 * 1024
    if plafond <= 0:
        return

    fichiers = []
    total = 0
    for f in repertoire.iterdir():
        if not f.is_file():
            continue
        stat = f.stat()
        fichiers.append((stat.st_atime, stat.st_size, f))
        total += stat.st_size

    if total <= plafond:
        return

    for _, taille, f in sorted(fichiers):
        try:
            f.unlink()
        except OSError:
            continue
        total -= taille
        if total <= plafond:
            return


async def _telecharger(file_id: str, destination: Path) -> Path:
    """
    Télécharge le document dans le cache, en flux et de façon atomique.

    L'écriture passe par un fichier `.part` renommé à la fin. Sans cela, un
    téléchargement interrompu laisserait un PDF tronqué que le cache servirait
    ensuite indéfiniment, sans jamais retenter.
    """
    partiel = destination.with_suffix(destination.suffix + ".part")
    delai = httpx.Timeout(settings.DOCUMENTS_FETCH_TIMEOUT_S)

    for extension in EXTENSIONS:
        url = url_publique(file_id, extension)
        try:
            async with httpx.AsyncClient(timeout=delai, follow_redirects=True) as client:
                async with client.stream("GET", url) as reponse:
                    if reponse.status_code == 404:
                        continue
                    if reponse.status_code >= 500:
                        raise DocumentInaccessible(
                            f"Le magasin a repondu {reponse.status_code} pour {file_id}"
                        )
                    if reponse.status_code >= 400:
                        raise DocumentIntrouvable(
                            f"Le magasin a refuse {file_id} ({reponse.status_code})"
                        )
                    with partiel.open("wb") as sortie:
                        async for tranche in reponse.aiter_bytes(TAILLE_TRANCHE):
                            sortie.write(tranche)
        except (httpx.HTTPError, OSError) as exc:
            partiel.unlink(missing_ok=True)
            raise DocumentInaccessible(f"Recuperation de {file_id} impossible : {exc}") from exc

        cible = destination.with_suffix(extension)
        os.replace(partiel, cible)
        _elaguer_le_cache(cible.parent)
        return cible

    raise DocumentIntrouvable(file_id)


async def chemin_local(file_id: str) -> Path:
    """
    Chemin d'un document sur le disque local, quel que soit le mode.

    En mode local, c'est `resolve_upload_path`. En mode distant, le document est
    téléchargé dans le cache si nécessaire, puis son chemin est rendu — les
    bibliothèques qui rendent les pages (`pdf2image`, `pypdf`) prennent un
    chemin de fichier et ne savent pas consommer une URL.

    Raises:
        IdentifiantInvalide, DocumentIntrouvable, DocumentInaccessible
    """
    if not stockage_distant():
        try:
            return resolve_upload_path(file_id)
        except ValueError as exc:
            raise IdentifiantInvalide(str(exc)) from exc
        except FileNotFoundError as exc:
            raise DocumentIntrouvable(file_id) from exc

    if not _FILE_ID_RE.match(file_id or ""):
        raise IdentifiantInvalide(f"Identifiant de fichier invalide : {file_id!r}")

    repertoire = _repertoire_du_cache()
    async with _verrou(file_id):
        for extension in EXTENSIONS:
            candidat = repertoire / f"{file_id}{extension}"
            if candidat.is_file():
                return candidat
        return await _telecharger(file_id, repertoire / file_id)
