"""
Extraction des PDF par Gemini.

Ces tests figent le CONTRAT que le decoupeur impose, et les trois modes d'echec
observes en production sur le palier gratuit.

Le reseau est simule par monkeypatch sur `client.aio.models.generate_content`,
et non par respx : respx intercepte httpx, alors que le SDK google-genai a sa
propre pile — un mock respx ne verrait jamais l'appel.

La fixture PDF est un VRAI PDF construit par pypdf. Le talon de 22 octets qui
suffisait a l'ancien service ne convient plus : celui-ci ouvre le fichier pour
compter et decouper ses pages.

Usage:
    pytest tests/test_services/test_pdf_extraction.py -v
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.services.pdf_extraction_service import (
    GeminiPdfExtractor,
    PdfExtractionError,
    PdfExtractionQuotaError,
    _decrire,
    _est_un_depassement,
)

SATURATION = (
    "503 UNAVAILABLE. {'error': {'code': 503, 'message': 'This model is currently "
    "experiencing high demand.', 'status': 'UNAVAILABLE'}}"
)
QUOTA = (
    "429 RESOURCE_EXHAUSTED. {'error': {'code': 429, 'message': 'You exceeded your "
    "current quota', 'details': [{'retryDelay': '42s'}]}}"
)


def _pdf(chemin: Path, pages: int = 1) -> Path:
    from pypdf import PdfWriter

    writer = PdfWriter()
    for _ in range(pages):
        writer.add_blank_page(width=200, height=200)
    with chemin.open("wb") as fh:
        writer.write(fh)
    return chemin


def _reponse(texte: str, finish: str = "STOP"):
    """Reponse a la forme de celle du SDK, reflexion comprise si demandee."""
    part = SimpleNamespace(text=texte, thought=False)
    contenu = SimpleNamespace(parts=[part] if texte else [])
    candidat = SimpleNamespace(content=contenu, finish_reason=SimpleNamespace(name=finish))
    return SimpleNamespace(candidates=[candidat], text=texte)


@pytest.fixture
def extracteur(tmp_path):
    return GeminiPdfExtractor(
        api_key="cle-de-test",
        model_name="modele-test",
        pages_per_call=2,
        cache_dir=tmp_path / "cache",
    )


def _brancher(extracteur, reponses):
    """
    Remplace l'appel au modele par une liste de reponses ou d'exceptions.

    Rend un compteur d'appels : c'est lui qui prouve qu'un quota epuise n'est
    PAS retente.
    """
    etat = {"appels": 0}
    file = list(reponses)

    async def _faux(**kwargs):
        etat["appels"] += 1
        item = file.pop(0) if file else file_defaut
        if isinstance(item, Exception):
            raise item
        return item

    file_defaut = _reponse("<<PAGE:1>>\nContenu.")
    extracteur._client = SimpleNamespace(
        aio=SimpleNamespace(models=SimpleNamespace(generate_content=_faux))
    )
    return etat


class TestContratDeSortie:
    """Ce que `text_chunker` exige, au caractere pres."""

    @pytest.mark.asyncio
    async def test_marqueur_sans_espace_apres_les_deux_points(self, extracteur, tmp_path):
        """
        `text_chunker.py` compile `r'<<PAGE:(\\d+)>>'`. Un `<<PAGE: 1>>` serait
        invisible pour lui tout en etant retire ailleurs : la page existerait
        dans le texte et pas dans la numerotation.
        """
        _brancher(extracteur, [_reponse("<<PAGE:1>>\nLe contenu.")])

        texte = await extracteur.extract_text(_pdf(tmp_path / "d.pdf"))

        assert texte.startswith("<<PAGE:1>>\n")
        assert "<<PAGE: " not in texte

    @pytest.mark.asyncio
    async def test_pages_jointes_par_deux_sauts_de_ligne(self, extracteur, tmp_path):
        _brancher(extracteur, [_reponse("<<PAGE:1>>\nUn\n<<PAGE:2>>\nDeux")])

        texte = await extracteur.extract_text(_pdf(tmp_path / "d.pdf", pages=2))

        assert texte == "<<PAGE:1>>\nUn\n\n<<PAGE:2>>\nDeux"

    @pytest.mark.asyncio
    async def test_une_page_blanche_ne_decale_pas_les_suivantes(self, extracteur, tmp_path):
        """
        La numerotation se fait AVANT le filtrage des pages vides : une page
        blanche consomme son numero, ce qui garde les suivantes alignees sur la
        page physique du PDF.
        """
        _brancher(extracteur, [_reponse("<<PAGE:1>>\nUn\n<<PAGE:2>>\n\n<<PAGE:3>>\nTrois")])

        texte = await extracteur.extract_text(_pdf(tmp_path / "d.pdf", pages=3))

        assert "<<PAGE:1>>\nUn" in texte
        assert "<<PAGE:3>>\nTrois" in texte
        assert "<<PAGE:2>>" not in texte, "la page vide est omise, pas rendue vide"

    @pytest.mark.asyncio
    async def test_texte_sans_aucun_marqueur_compte_pour_une_page(self, extracteur, tmp_path):
        """Mieux vaut une page unique qu'une exception : le contenu est bon."""
        _brancher(extracteur, [_reponse("Du texte sans le moindre marqueur.")])

        texte = await extracteur.extract_text(_pdf(tmp_path / "d.pdf"))

        assert texte == "<<PAGE:1>>\nDu texte sans le moindre marqueur."


class TestNumerotationAbsolue:
    """
    Le point qui casse en silence : sans consigne explicite, le modele
    renumerote chaque lot a partir de 1 et chaque lot ecrase le precedent.
    """

    @pytest.mark.asyncio
    async def test_la_consigne_annonce_la_premiere_page_du_lot(self, extracteur, tmp_path):
        consignes = []

        async def _capturer(**kwargs):
            consignes.append(kwargs["contents"][1])
            depart = len(consignes)  # lot 1 -> pages 1-2, lot 2 -> pages 3-4
            base = 1 if depart == 1 else 3
            return _reponse(f"<<PAGE:{base}>>\nA\n<<PAGE:{base + 1}>>\nB")

        extracteur._client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=_capturer))
        )

        await extracteur.extract_text(_pdf(tmp_path / "d.pdf", pages=4))

        assert len(consignes) == 2, "4 pages en lots de 2 font deux appels"
        assert "premiere page" in consignes[0]
        assert "page 3 du document complet" in consignes[1]
        assert "ABSOLUE" in consignes[1]

    @pytest.mark.asyncio
    async def test_les_lots_se_recollent_en_numerotation_continue(self, extracteur, tmp_path):
        import re

        appels = {"n": 0}

        async def _faux(**kwargs):
            appels["n"] += 1
            base = 1 if appels["n"] == 1 else 3
            return _reponse(f"<<PAGE:{base}>>\nA\n<<PAGE:{base + 1}>>\nB")

        extracteur._client = SimpleNamespace(
            aio=SimpleNamespace(models=SimpleNamespace(generate_content=_faux))
        )

        texte = await extracteur.extract_text(_pdf(tmp_path / "d.pdf", pages=4))

        assert [int(n) for n in re.findall(r"<<PAGE:(\d+)>>", texte)] == [1, 2, 3, 4]


class TestModesDEchec:
    @pytest.mark.asyncio
    async def test_un_quota_epuise_n_est_jamais_retente(self, extracteur, tmp_path):
        """Retenter un quota epuise brule des appels qu'on n'a deja plus."""
        etat = _brancher(extracteur, [Exception(QUOTA)])

        with pytest.raises(PdfExtractionQuotaError, match="42 secondes"):
            await extracteur.extract_text(_pdf(tmp_path / "d.pdf"))

        assert etat["appels"] == 1

    @pytest.mark.asyncio
    async def test_une_saturation_est_retentee_puis_abandonnee(self, extracteur, tmp_path):
        etat = _brancher(extracteur, [Exception(SATURATION)] * 3)
        extracteur.OVERLOAD_BASE_DELAY_S = 0

        with pytest.raises(PdfExtractionError):
            await extracteur.extract_text(_pdf(tmp_path / "d.pdf"))

        assert etat["appels"] == 3

    @pytest.mark.asyncio
    async def test_une_saturation_passagere_est_absorbee(self, extracteur, tmp_path):
        _brancher(extracteur, [Exception(SATURATION), _reponse("<<PAGE:1>>\nOK")])
        extracteur.OVERLOAD_BASE_DELAY_S = 0

        assert "OK" in await extracteur.extract_text(_pdf(tmp_path / "d.pdf"))

    @pytest.mark.asyncio
    async def test_un_refus_de_recitation_rejoue_page_par_page(self, extracteur, tmp_path):
        """
        Mesure sur un document reel : la page 1 refuse, les pages 2 et 3
        s'extraient. Refuser tout le document pour une page de garde ferait
        perdre les soixante-huit autres.
        """
        reponses = [
            _reponse("", finish="RECITATION"),   # le lot entier
            _reponse("", finish="RECITATION"),   # page 1, refusee a nouveau
            _reponse("<<PAGE:2>>\nDeux"),        # page 2, elle passe
        ]
        _brancher(extracteur, reponses)

        texte = await extracteur.extract_text(_pdf(tmp_path / "d.pdf", pages=2))

        assert "Deux" in texte
        assert extracteur.pages_refusees == [1]

    @pytest.mark.asyncio
    async def test_une_reponse_vide_sans_raison_connue_leve(self, extracteur, tmp_path):
        _brancher(extracteur, [_reponse("", finish="SAFETY")])

        with pytest.raises(PdfExtractionError, match="SAFETY"):
            await extracteur.extract_text(_pdf(tmp_path / "d.pdf"))

    @pytest.mark.asyncio
    async def test_sans_cle_le_service_refuse_de_travailler(self, tmp_path):
        muet = GeminiPdfExtractor(api_key="", cache_dir=tmp_path / "c")

        assert muet.is_available() is False
        with pytest.raises(PdfExtractionError, match="GEMINI_API_KEY"):
            await muet.extract_text(_pdf(tmp_path / "d.pdf"))

    @pytest.mark.asyncio
    async def test_fichier_absent(self, extracteur, tmp_path):
        with pytest.raises(PdfExtractionError, match="introuvable"):
            await extracteur.extract_text(tmp_path / "nexiste-pas.pdf")


class TestCache:
    @pytest.mark.asyncio
    async def test_un_second_appel_ne_touche_pas_le_modele(self, extracteur, tmp_path):
        etat = _brancher(extracteur, [_reponse("<<PAGE:1>>\nContenu.")])
        pdf = _pdf(tmp_path / "d.pdf")

        premier = await extracteur.extract_text(pdf)
        second = await extracteur.extract_text(pdf)

        assert premier == second
        assert etat["appels"] == 1

    @pytest.mark.asyncio
    async def test_une_extraction_vide_n_est_jamais_gravee(self, extracteur, tmp_path):
        """
        Une entree vide comptait autrefois comme un succes : l'echec se figeait
        et le seul moyen d'en sortir etait d'effacer le fichier a la main.
        """
        extracteur._write_cache("cle-test", ["", "   "])

        assert list((tmp_path / "cache").glob("*.json")) == []

    def test_une_entree_vide_sur_disque_est_un_defaut_de_cache(self, extracteur, tmp_path):
        chemin = tmp_path / "cache" / "cle-test.json"
        chemin.write_text(json.dumps(
            {"schema": extracteur.CACHE_SCHEMA, "engine": "modele-test", "pages": []}
        ))

        assert extracteur._read_cache("cle-test") is None

    def test_une_entree_d_un_autre_moteur_est_ignoree(self, extracteur, tmp_path):
        """
        Les 61 entrees heritees de LlamaParse portent `tier`, pas `engine` :
        elles ne doivent pas etre resservies comme si elles venaient d'ici — ce
        sont precisement les extractions non paginees qu'on remplace.
        """
        chemin = tmp_path / "cache" / "cle-test.json"
        chemin.write_text(json.dumps(
            {"schema": 2, "tier": "cost_effective", "pages": ["ancien contenu"]}
        ))

        assert extracteur._read_cache("cle-test") is None

    def test_ecriture_atomique_sans_residu(self, extracteur, tmp_path):
        extracteur._write_cache("cle-test", ["du contenu"])

        assert list((tmp_path / "cache").glob("*.tmp")) == []
        assert extracteur._read_cache("cle-test") == ["du contenu"]


class TestDecoupageEtComptage:
    def test_un_document_court_ne_fait_qu_un_lot(self, extracteur, tmp_path):
        pdf = _pdf(tmp_path / "d.pdf", pages=2)

        assert extracteur.compter_appels(pdf) == 1
        assert len(extracteur._decouper_en_lots(pdf)) == 1

    def test_un_document_long_est_decoupe(self, extracteur, tmp_path):
        pdf = _pdf(tmp_path / "d.pdf", pages=5)

        lots = extracteur._decouper_en_lots(pdf)

        assert extracteur.compter_appels(pdf) == 3
        assert [premiere for premiere, _ in lots] == [1, 3, 5]

    def test_le_decoupage_ne_laisse_aucun_fichier_temporaire(self, extracteur, tmp_path):
        """
        Decoupage EN MEMOIRE : l'ancien service ecrivait des fichiers a cote de
        l'original et les effacait dans un `finally`, ce qu'un processus tue ne
        fait jamais.
        """
        extracteur._decouper_en_lots(_pdf(tmp_path / "d.pdf", pages=5))

        assert sorted(p.name for p in tmp_path.glob("*.pdf")) == ["d.pdf"]


class TestErreursMuettes:
    """
    `httpx.ReadTimeout` a un `str()` VIDE.

    Constate en production : la Loi de finances a echoue sur un message
    « Extraction de X impossible : » suivi de rien, ecrit tel quel dans
    `laws.processing_error`. Un message qui ne dit rien coute autant a lire
    qu'un message juste.
    """

    def test_une_exception_muette_est_nommee(self):
        class ReadTimeout(Exception):
            def __str__(self):
                return ""

        assert _decrire(ReadTimeout()) == "ReadTimeout"

    def test_une_exception_bavarde_garde_son_message(self):
        assert _decrire(ValueError("quelque chose de precis")) == "quelque chose de precis"

    def test_un_message_de_blancs_compte_comme_muet(self):
        assert _decrire(ValueError("   \n  ")) == "ValueError"

    @pytest.mark.asyncio
    async def test_le_message_remonte_jusqu_a_l_appelant(self, extracteur, tmp_path):
        class ReadTimeout(Exception):
            def __str__(self):
                return ""

        _brancher(extracteur, [ReadTimeout()] * 3)
        extracteur.OVERLOAD_BASE_DELAY_S = 0

        with pytest.raises(PdfExtractionError, match="ReadTimeout"):
            await extracteur.extract_text(_pdf(tmp_path / "d.pdf"))


class TestDepassementDeDelai:
    def test_un_depassement_est_reconnu(self):
        class ReadTimeout(Exception):
            pass

        assert _est_un_depassement(ReadTimeout()) is True
        assert _est_un_depassement(TimeoutError()) is True

    def test_une_erreur_ordinaire_n_en_est_pas_un(self):
        assert _est_un_depassement(ValueError("400 INVALID_ARGUMENT")) is False

    @pytest.mark.asyncio
    async def test_un_depassement_est_retente(self, extracteur, tmp_path):
        """
        Transitoire au meme titre qu'une saturation : sur un lot de 10 Mo
        d'images scannees, la premiere tentative depassait le delai general.
        """
        class ReadTimeout(Exception):
            def __str__(self):
                return ""

        etat = _brancher(extracteur, [ReadTimeout(), _reponse("<<PAGE:1>>\nOK")])
        extracteur.OVERLOAD_BASE_DELAY_S = 0

        assert "OK" in await extracteur.extract_text(_pdf(tmp_path / "d.pdf"))
        assert etat["appels"] == 2


class TestDecoupageParPoids:
    """
    Le nombre de pages ne dit rien du poids.

    Vingt pages scannees pesent 10 Mo la ou vingt pages de texte en pesent 1,
    et c'est le POIDS qui a fait depasser le delai sur la Loi de finances :
    trois lots de 10,4 Mo au maximum, dont le premier n'a jamais abouti.
    """

    def test_un_lot_trop_lourd_est_redecoupe(self, tmp_path):
        extracteur = GeminiPdfExtractor(
            api_key="cle", pages_per_call=50, cache_dir=tmp_path / "c"
        )
        # Seuil minuscule : chaque page depasse, donc un lot par page.
        extracteur.max_batch_mb = 0.0005

        lots = extracteur._decouper_en_lots(_pdf(tmp_path / "d.pdf", pages=4))

        assert [premiere for premiere, _ in lots] == [1, 2, 3, 4]

    def test_le_poids_ne_prime_pas_sur_la_couverture(self, tmp_path):
        """Toutes les pages doivent sortir, quel que soit le decoupage."""
        extracteur = GeminiPdfExtractor(
            api_key="cle", pages_per_call=3, cache_dir=tmp_path / "c"
        )
        extracteur.max_batch_mb = 0.0005

        lots = extracteur._decouper_en_lots(_pdf(tmp_path / "d.pdf", pages=7))

        assert [premiere for premiere, _ in lots] == list(range(1, 8))

    def test_une_page_seule_trop_lourde_part_quand_meme(self, tmp_path):
        """Faute de mieux : on ne peut pas decouper une page en deux."""
        extracteur = GeminiPdfExtractor(
            api_key="cle", pages_per_call=5, cache_dir=tmp_path / "c"
        )
        extracteur.max_batch_mb = 0.0000001

        lots = extracteur._decouper_en_lots(_pdf(tmp_path / "d.pdf", pages=2))

        assert len(lots) == 2, "une page par lot, aucune page perdue"
