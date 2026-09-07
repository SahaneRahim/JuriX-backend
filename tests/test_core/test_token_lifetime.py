"""
Duree de vie des jetons, et mot de passe absent.

POURQUOI CE FICHIER. La session est passee de 30 minutes a 30 jours pour que le
produit ait un sens — l'interet d'un compte est de retrouver ses conversations,
et l'utilisateur etait deconnecte en pleine session. Mais le meme reglage
regissait l'administration : les JWT sont sans etat ici, `/logout` ne revoque
rien, et un jeton superadmin vole serait reste exploitable un mois. D'ou deux
durees, et ces tests pour que la distinction ne disparaisse pas au premier
refactoring.

Usage:
    pytest tests/test_core/test_token_lifetime.py -v
"""

from datetime import datetime, timedelta, timezone

import pytest
from jose import jwt

from app.core.auth import (
    ALGORITHM,
    SECRET_KEY,
    create_access_token,
    duree_de_session,
    hash_password,
    verify_password,
)
from app.core.config import settings
from app.models.user import User


def _user(role: str) -> User:
    return User(email=f"{role}@x.cm", username=role, role=role)


class TestDureeDeSession:
    def test_utilisateur_ordinaire_30_jours(self):
        assert duree_de_session(_user("user")) == timedelta(days=30)

    @pytest.mark.parametrize("role", ["admin", "superadmin"])
    def test_comptes_privilegies_12_heures(self, role):
        """
        Un jeton privilegie vit BEAUCOUP moins longtemps : il n'existe aucune
        liste de revocation, et il voyage en clair dans localStorage.
        """
        assert duree_de_session(_user(role)) == timedelta(hours=12)

    def test_admin_strictement_plus_court_que_user(self):
        assert duree_de_session(_user("admin")) < duree_de_session(_user("user"))

    def test_les_reglages_sont_lus_a_lappel(self, monkeypatch):
        """
        Et non a l'import. Les constantes de module d'`app/core/auth.py` sont
        figees au chargement : si `duree_de_session` les lisait, un test qui
        modifie le reglage ne verrait rien changer.
        """
        monkeypatch.setattr(settings, "ACCESS_TOKEN_EXPIRE_MINUTES", 5)

        assert duree_de_session(_user("user")) == timedelta(minutes=5)


class TestJetonEmis:
    def test_expiration_conforme_a_la_duree(self):
        jeton = create_access_token(
            {"sub": "a@x.cm"}, expires_delta=duree_de_session(_user("user"))
        )

        charge = jwt.decode(jeton, SECRET_KEY, algorithms=[ALGORITHM])
        reste = datetime.fromtimestamp(charge["exp"], tz=timezone.utc) - datetime.now(timezone.utc)
        assert timedelta(days=29) < reste <= timedelta(days=30)

    def test_jeton_admin_court(self):
        jeton = create_access_token(
            {"sub": "a@x.cm"}, expires_delta=duree_de_session(_user("admin"))
        )

        charge = jwt.decode(jeton, SECRET_KEY, algorithms=[ALGORITHM])
        reste = datetime.fromtimestamp(charge["exp"], tz=timezone.utc) - datetime.now(timezone.utc)
        assert reste <= timedelta(hours=12)


class TestMotDePasseAbsent:
    def test_verify_password_sur_none_rend_false(self):
        """
        Un compte cree par Google a `hashed_password` NULL. Sans la garde,
        `.encode()` levait un AttributeError — que le `except (ValueError,
        TypeError)` n'attrape pas — et la connexion repondait 500.
        """
        assert verify_password("MotDePasse1", None) is False

    def test_verify_password_sur_chaine_vide(self):
        assert verify_password("MotDePasse1", "") is False

    def test_un_vrai_mot_de_passe_fonctionne_toujours(self):
        empreinte = hash_password("MotDePasse1")

        assert verify_password("MotDePasse1", empreinte) is True
        assert verify_password("MauvaisMdp1", empreinte) is False
