"""
Acces aux documents d'origine, sur disque local ou sur un magasin HTTPS.

CE QUE CE FICHIER PROTEGE. Le module `document_storage` a ete ecrit pour qu'un
hebergeur au disque ephemere puisse quand meme servir les PDF : les lignes en
base survivent au redeploiement, pas les fichiers. Il introduit donc deux
choses risquees — une URL construite a partir d'une valeur de la base, et un
cache sur disque.

Le test le plus important est `test_file_id_hors_motif_ne_construit_jamais_d_url`.
En mode local, `_FILE_ID_RE` protegeait un chemin de fichier ; en mode distant
elle protege une URL. Un `file_id` contenant `../` ou `@evil.example/` ferait
sortir la requete du magasin, c'est-a-dire transformerait le serveur en relais
de requetes (SSRF). La valeur vient de la base et non de la requete HTTP, donc
ce n'est pas exploitable aujourd'hui ; c'est exactement pour cela que la faille
passerait inapercue le jour ou une route ecrira un `file_id` d'apres une entree
utilisateur.

Les autres tests couvrent le cache : ne pas retelecharger, ne jamais servir un
fichier tronque, ne pas telecharger deux fois en parallele, et ne pas grossir
sans limite jusqu'a saturer le conteneur.

AUCUN RESEAU REEL : `respx` intercepte httpx et permet d'affirmer sur le nombre
d'appels, ce qu'un simple monkeypatch ne ferait pas.

Usage:
    pytest tests/test_services/test_document_storage.py -v
"""

import asyncio

import httpx
import pytest
import respx

from app.core.config import settings
from app.services import document_storage
from app.services.document_storage import (
    DocumentInaccessible,
    DocumentIntrouvable,
    IdentifiantInvalide,
    chemin_local,
    stockage_distant,
    url_publique,
)

BASE = "https://magasin.test/documents"
FILE_ID = "abc123def456"
URL_PDF = f"{BASE}/{FILE_ID}.pdf"
URL_DOCX = f"{BASE}/{FILE_ID}.docx"
OCTETS = b"%PDF-1.7\n" + b"x" * 4096


@pytest.fixture
def distant(monkeypatch, tmp_path):
    """Mode distant, cache isole dans un repertoire temporaire."""
    monkeypatch.setattr(settings, "DOCUMENTS_BASE_URL", BASE)
    monkeypatch.setattr(settings, "DOCUMENTS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(settings, "DOCUMENTS_CACHE_MAX_MB", 96)
    # Les verrous vivent au niveau module : sans ce nettoyage, un verrou pris
    # par un test precedent survivrait au suivant.
    document_storage._verrous.clear()
    return tmp_path / "cache"


class TestModeLocal:
    def test_base_url_vide_signifie_disque_local(self):
        """
        La garantie de non-regression : sans reglage, rien ne change.

        C'est ce qui permet aux centaines de tests existants et au
        developpement local de continuer sans une seule modification.
        """
        assert settings.DOCUMENTS_BASE_URL == ""
        assert stockage_distant() is False

    @pytest.mark.asyncio
    @respx.mock
    async def test_le_mode_local_n_emet_aucune_requete(self, monkeypatch, tmp_path):
        monkeypatch.setattr(settings, "DOCUMENTS_BASE_URL", "")
        route = respx.get(url__startswith="https://").mock(return_value=httpx.Response(200))

        with pytest.raises((IdentifiantInvalide, DocumentIntrouvable)):
            await chemin_local("fichier-inexistant-mais-valide")

        assert route.call_count == 0, "le mode local est parti sur le reseau"


class TestGardeSSRF:
    @pytest.mark.parametrize(
        "file_id",
        [
            "../../etc/passwd",
            "a@evil.example",
            "http://evil.example/x",
            "avec/slash",
            "avec.point",
            "court",
            "x" * 65,
            "",
            None,
        ],
    )
    def test_file_id_hors_motif_ne_construit_jamais_d_url(self, distant, file_id):
        """LE TEST LE PLUS IMPORTANT DU FICHIER. Voir le docstring du module."""
        with pytest.raises(IdentifiantInvalide):
            url_publique(file_id)

    @pytest.mark.asyncio
    @respx.mock
    async def test_file_id_hors_motif_n_emet_aucune_requete(self, distant):
        """
        Lever ne suffit pas : il faut prouver que rien n'est PARTI.

        Une garde placee apres la construction de la requete leverait la meme
        exception tout en ayant deja contacte l'hote arbitraire.
        """
        route = respx.get(url__startswith="http").mock(return_value=httpx.Response(200))

        with pytest.raises(IdentifiantInvalide):
            await chemin_local("../../etc/passwd")

        assert route.call_count == 0, "une requete est partie vers un hote arbitraire"

    def test_url_construite_pour_un_identifiant_valide(self, distant):
        assert url_publique(FILE_ID) == URL_PDF
        assert url_publique(FILE_ID, ".docx") == URL_DOCX


class TestCache:
    @pytest.mark.asyncio
    @respx.mock
    async def test_telechargement_puis_lecture_depuis_le_cache(self, distant):
        route = respx.get(URL_PDF).mock(return_value=httpx.Response(200, content=OCTETS))

        premier = await chemin_local(FILE_ID)
        second = await chemin_local(FILE_ID)

        assert premier == second
        assert premier.read_bytes() == OCTETS
        assert route.call_count == 1, "le cache a retelecharge le document"

    @pytest.mark.asyncio
    @respx.mock
    async def test_deux_demandes_simultanees_ne_telechargent_qu_une_fois(self, distant):
        """Deux rendus de page de la meme loi ne doivent pas tirer 26 Mo deux fois."""
        route = respx.get(URL_PDF).mock(return_value=httpx.Response(200, content=OCTETS))

        resultats = await asyncio.gather(*(chemin_local(FILE_ID) for _ in range(4)))

        assert len(set(resultats)) == 1
        assert route.call_count == 1

    @pytest.mark.asyncio
    @respx.mock
    async def test_telechargement_interrompu_ne_laisse_pas_de_fichier_tronque(self, distant):
        """
        Sans ecriture atomique, un PDF tronque serait servi INDEFINIMENT : le
        cache le trouverait present et ne retenterait jamais.
        """
        def coupe(request):
            raise httpx.ReadError("connexion coupee en plein flux")

        route = respx.get(URL_PDF).mock(side_effect=coupe)

        with pytest.raises(DocumentInaccessible):
            await chemin_local(FILE_ID)

        restants = list(distant.iterdir()) if distant.exists() else []
        assert restants == [], f"le cache garde des restes : {restants}"

        # Et l'appel suivant retente bien, au lieu de servir un vide.
        route.mock(return_value=httpx.Response(200, content=OCTETS))
        assert (await chemin_local(FILE_ID)).read_bytes() == OCTETS

    @pytest.mark.asyncio
    @respx.mock
    async def test_le_cache_est_borne(self, distant, monkeypatch):
        """Sinon /tmp grossit jusqu'a saturer le conteneur."""
        monkeypatch.setattr(settings, "DOCUMENTS_CACHE_MAX_MB", 1)
        gros = b"y" * (400 * 1024)

        for i in range(6):
            ident = f"document{i:04d}"
            respx.get(f"{BASE}/{ident}.pdf").mock(
                return_value=httpx.Response(200, content=gros)
            )
            await chemin_local(ident)

        total = sum(f.stat().st_size for f in distant.iterdir() if f.is_file())
        assert total <= 1024 * 1024, f"cache non borne : {total} octets"


class TestErreursDeLAmont:
    @pytest.mark.asyncio
    @respx.mock
    async def test_404_amont_donne_document_introuvable(self, distant):
        respx.get(URL_PDF).mock(return_value=httpx.Response(404))
        respx.get(URL_DOCX).mock(return_value=httpx.Response(404))

        with pytest.raises(DocumentIntrouvable):
            await chemin_local(FILE_ID)

    @pytest.mark.asyncio
    @respx.mock
    async def test_500_amont_donne_document_inaccessible(self, distant):
        """
        Distinguer 503 de 404 : c'est NOTRE magasin qui est en panne. Un 404
        ferait disparaitre le document de l'interface au lieu de signaler une
        panne passagere.
        """
        respx.get(URL_PDF).mock(return_value=httpx.Response(503))

        with pytest.raises(DocumentInaccessible):
            await chemin_local(FILE_ID)

    @pytest.mark.asyncio
    @respx.mock
    async def test_delai_depasse_donne_document_inaccessible(self, distant):
        respx.get(URL_PDF).mock(side_effect=httpx.ConnectTimeout("delai depasse"))

        with pytest.raises(DocumentInaccessible):
            await chemin_local(FILE_ID)

    @pytest.mark.asyncio
    @respx.mock
    async def test_repli_sur_docx_quand_le_pdf_est_absent(self, distant):
        """Miroir exact de resolve_upload_path, qui sonde .pdf puis .docx."""
        respx.get(URL_PDF).mock(return_value=httpx.Response(404))
        respx.get(URL_DOCX).mock(return_value=httpx.Response(200, content=b"PK\x03\x04"))

        chemin = await chemin_local(FILE_ID)

        assert chemin.suffix == ".docx"
