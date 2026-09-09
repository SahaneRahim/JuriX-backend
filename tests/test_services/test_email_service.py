"""
Le transport de courriel : API HTTPS de Brevo.

CE QUE CE FICHIER PROTEGE. Trois defauts de ce module seraient invisibles en
developpement et fatals en production :

1. PARTIR SANS CLE. Un POST vers Brevo avec un en-tete `api-key` vide rend 401.
   Sans garde, chaque inscription declencherait ce 401 en silence, et personne
   n'aurait le moindre indice sur la raison pour laquelle rien n'arrive.
   → `test_cle_vide_ne_fait_aucun_appel`

2. LE MAUVAIS NOM D'EN-TETE. Brevo attend `api-key` et repond 401 sur
   `Authorization: Bearer` — un 401 rigoureusement indiscernable de celui d'une
   cle invalide.
   → `test_entete_est_api_key_et_pas_authorization`

3. BASCULER VERS SMTP. Les hebergeurs gratuits BLOQUENT les ports sortants 25,
   465 et 587. Le jour ou quelqu'un « simplifie » vers `smtplib` parce que c'est
   la bibliotheque standard, la suite doit rougir AVANT que l'hebergeur ne coupe
   le port sans rien dire.
   → `test_url_est_l_api_https_pas_smtp`

AUCUN RESEAU REEL : `respx` intercepte httpx et permet d'affirmer sur l'URL, les
en-tetes et le corps — ce qu'un monkeypatch ne ferait pas.

Usage:
    pytest tests/test_services/test_email_service.py -v
"""

import json

import httpx
import pytest
import respx

from app.core.config import settings
from app.services import email_service
from app.services.email_service import (
    API_BREVO,
    CourrielInjoignable,
    CourrielNonConfigure,
    CourrielRefuse,
    envoyer,
    url_du_front,
)


@pytest.fixture
def configure(monkeypatch):
    monkeypatch.setattr(settings, "BREVO_API_KEY", "xkeysib-cle-de-test")
    monkeypatch.setattr(settings, "BREVO_SENDER_EMAIL", "expediteur@example.cm")
    monkeypatch.setattr(settings, "BREVO_SENDER_NAME", "JuriX")
    monkeypatch.setattr(settings, "FRONTEND_BASE_URL", "https://jurix.test")
    monkeypatch.setattr(settings, "BREVO_BUDGET_QUOTIDIEN", 200)
    email_service._budget["jour"] = None
    email_service._budget["envoyes"] = 0


async def _envoyer():
    return await envoyer("cible@example.cm", "Sujet", "Texte brut", "<p>HTML</p>")


class TestGardeDeConfiguration:
    @pytest.mark.asyncio
    @respx.mock
    async def test_cle_vide_ne_fait_aucun_appel(self, monkeypatch):
        """
        LE TEST LE PLUS IMPORTANT DU FICHIER — miroir exact de la garde qui
        refuse de verifier un jeton Google sans identifiant client.

        Lever ne suffit pas : l'assertion qui compte est `call_count == 0`.
        """
        monkeypatch.setattr(settings, "BREVO_API_KEY", "")
        route = respx.post(API_BREVO).mock(return_value=httpx.Response(201))

        with pytest.raises(CourrielNonConfigure):
            await _envoyer()

        assert route.call_count == 0, "un POST est parti avec une cle vide"

    @pytest.mark.asyncio
    @respx.mock
    async def test_expediteur_vide_ne_fait_aucun_appel(self, monkeypatch):
        monkeypatch.setattr(settings, "BREVO_API_KEY", "xkeysib-cle")
        monkeypatch.setattr(settings, "BREVO_SENDER_EMAIL", "")
        route = respx.post(API_BREVO).mock(return_value=httpx.Response(201))

        with pytest.raises(CourrielNonConfigure):
            await _envoyer()

        assert route.call_count == 0

    def test_front_url_vide_refuse_de_batir_un_lien(self, monkeypatch):
        """
        Sans `FRONTEND_BASE_URL`, le lien se batirait sur l'hote de l'API, qui
        ne sert aucune page : le destinataire recevrait un lien mort. Mieux vaut
        ne rien envoyer du tout.
        """
        monkeypatch.setattr(settings, "FRONTEND_BASE_URL", "")

        with pytest.raises(CourrielNonConfigure):
            url_du_front("/reset-password/abc")

    def test_front_url_sans_slash_final_double(self, monkeypatch):
        monkeypatch.setattr(settings, "FRONTEND_BASE_URL", "https://jurix.test/")

        assert url_du_front("/verify-email/abc") == "https://jurix.test/verify-email/abc"


class TestContratHttp:
    @pytest.mark.asyncio
    @respx.mock
    async def test_url_est_l_api_https_pas_smtp(self, configure):
        """
        Voir le point 3 du docstring : les ports SMTP sortants sont bloques chez
        les hebergeurs gratuits.
        """
        route = respx.post(API_BREVO).mock(
            return_value=httpx.Response(201, json={"messageId": "m1"})
        )

        await _envoyer()

        assert route.call_count == 1
        assert str(route.calls[-1].request.url) == "https://api.brevo.com/v3/smtp/email"

    @pytest.mark.asyncio
    @respx.mock
    async def test_entete_est_api_key_et_pas_authorization(self, configure):
        route = respx.post(API_BREVO).mock(
            return_value=httpx.Response(201, json={"messageId": "m1"})
        )

        await _envoyer()

        entetes = route.calls[-1].request.headers
        assert entetes["api-key"] == "xkeysib-cle-de-test"
        assert "authorization" not in entetes, "Brevo repond 401 sur Bearer"

    @pytest.mark.asyncio
    @respx.mock
    async def test_le_corps_porte_texte_ET_html(self, configure):
        """
        Certains clients n'affichent pas le HTML. Un message dont le lien
        n'existerait que dans la partie HTML y devient illisible.
        """
        route = respx.post(API_BREVO).mock(
            return_value=httpx.Response(201, json={"messageId": "m1"})
        )

        await _envoyer()

        corps = json.loads(route.calls[-1].request.content)
        assert corps["textContent"] == "Texte brut"
        assert corps["htmlContent"] == "<p>HTML</p>"
        assert corps["to"] == [{"email": "cible@example.cm"}]
        # replyTo porte la vraie adresse : l'expediteur affiche sera peut-etre
        # reecrit en @brevosend.com, mais une reponse doit arriver quelque part.
        assert corps["replyTo"]["email"] == "expediteur@example.cm"


class TestErreurs:
    @pytest.mark.asyncio
    @respx.mock
    async def test_4xx_est_permanent(self, configure):
        """Cle invalide, expediteur non verifie, quota : reessayer n'aiderait pas."""
        respx.post(API_BREVO).mock(return_value=httpx.Response(401, json={"message": "bad key"}))

        with pytest.raises(CourrielRefuse):
            await _envoyer()

    @pytest.mark.asyncio
    @respx.mock
    async def test_5xx_est_transitoire(self, configure):
        respx.post(API_BREVO).mock(return_value=httpx.Response(503))

        with pytest.raises(CourrielInjoignable):
            await _envoyer()

    @pytest.mark.asyncio
    @respx.mock
    async def test_reseau_coupe_est_transitoire(self, configure):
        respx.post(API_BREVO).mock(side_effect=httpx.ConnectError("injoignable"))

        with pytest.raises(CourrielInjoignable):
            await _envoyer()


class TestBudgetQuotidien:
    @pytest.mark.asyncio
    @respx.mock
    async def test_le_budget_bloque_au_plafond(self, configure, monkeypatch):
        """
        Le palier gratuit plafonne a 300 par jour, PARTAGES entre transactionnel
        et campagnes. On s'arrete avant, pour qu'un pic ne consomme pas tout.
        """
        monkeypatch.setattr(settings, "BREVO_BUDGET_QUOTIDIEN", 2)
        route = respx.post(API_BREVO).mock(
            return_value=httpx.Response(201, json={"messageId": "m"})
        )

        await _envoyer()
        await _envoyer()
        with pytest.raises(CourrielRefuse):
            await _envoyer()

        assert route.call_count == 2, "le troisieme message est parti malgre le plafond"

    @pytest.mark.asyncio
    @respx.mock
    async def test_le_budget_repart_le_lendemain(self, configure, monkeypatch):
        monkeypatch.setattr(settings, "BREVO_BUDGET_QUOTIDIEN", 1)
        respx.post(API_BREVO).mock(return_value=httpx.Response(201, json={"messageId": "m"}))

        await _envoyer()
        # Simule le passage de minuit UTC.
        email_service._budget["jour"] = "1970-01-01"

        await _envoyer()  # ne doit pas lever
