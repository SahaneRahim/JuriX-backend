"""
Tests de la derivation d'un `username` a partir d'une adresse.

Le cas decisif est `test_le_username_derive_survit_a_userresponse` :
`UserResponse` herite de `UserBase`, donc `validate_username` s'execute AUSSI a
la sortie. Un username portant un point franchirait l'insert puis ferait
echouer la serialisation — l'inscription repondrait 500 apres avoir cree le
compte.

Usage:
    pytest tests/test_services/test_user_identity.py -v
"""

from datetime import datetime

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.user import User
from app.schemas.user import UserResponse
from app.services.user_identity import (
    base_username,
    normaliser_email,
    unique_username,
    username_pour_email,
)


class TestBaseUsername:
    @pytest.mark.parametrize(
        "email,attendu",
        [
            ("jeandupont@x.cm", "jeandupont"),
            ("jean.dupont@x.cm", "jeandupont"),      # le point disparait
            ("Jean-Luc@X.CM", "jean-luc"),           # le tiret survit
            ("jean+spam@x.cm", "jeanspam"),
            ("jose@x.cm", "jose"),
            ("_jean_@x.cm", "jean"),                 # bornes nettoyees
        ],
    )
    def test_nettoyage(self, email, attendu):
        assert base_username(email) == attendu

    def test_accents_deplies_et_non_supprimes(self):
        """« josé » doit donner « jose », pas « jos »."""
        assert base_username("josé.dupont@x.cm") == "josedupont"

    @pytest.mark.parametrize("email", ["ab@x.cm", "a@x.cm", "__@x.cm", "@x.cm", "...@x.cm"])
    def test_longueur_minimale_respectee(self, email):
        """`min_length=3` cote schema : completer plutot que refuser."""
        assert len(base_username(email)) >= 3

    def test_longueur_maximale(self):
        """La colonne fait 100 ; il faut laisser la place au suffixe."""
        assert len(base_username("x" * 200 + "@x.cm")) <= 90

    def test_entree_vide(self):
        assert len(base_username("")) >= 3


class TestSurvieAuValidateurDeSortie:
    @pytest.mark.parametrize(
        "email",
        [
            "jean.dupont@x.cm",
            "josé@x.cm",
            "ab@x.cm",
            "a@x.cm",
            "__@x.cm",
            "jean+spam@x.cm",
            "Jean-Luc@X.CM",
            "x" * 150 + "@x.cm",
        ],
    )
    def test_le_username_derive_survit_a_userresponse(self, email):
        """
        `UserResponse` herite de `UserBase` : son validateur s'applique A LA
        SORTIE. Un username refuse ici ferait repondre 500 a une inscription
        pourtant reussie en base.
        """
        UserResponse(
            id=1,
            email="a@b.cm",
            username=base_username(email),
            full_name=None,
            role="user",
            is_active=True,
            is_verified=False,
            created_at=datetime.now(),
        )


class TestUnicite:
    @pytest.mark.asyncio
    async def test_libre_rend_la_base(self, db_session: AsyncSession):
        assert await unique_username(db_session, "rahim") == "rahim"

    @pytest.mark.asyncio
    async def test_collision_suffixee(self, db_session: AsyncSession):
        db_session.add(
            User(
                email="a@x.cm",
                username="rahim",
                hashed_password="x",
                role="user",
                is_active=True,
                is_verified=False,
            )
        )
        await db_session.commit()

        assert await unique_username(db_session, "rahim") == "rahim-2"

    @pytest.mark.asyncio
    async def test_deux_adresses_meme_partie_locale(self, db_session: AsyncSession):
        premier = await username_pour_email(db_session, "jean@a.cm")
        db_session.add(
            User(
                email="jean@a.cm",
                username=premier,
                hashed_password="x",
                role="user",
                is_active=True,
                is_verified=False,
            )
        )
        await db_session.commit()

        second = await username_pour_email(db_session, "jean@b.cm")

        assert second != premier


class TestNormalisationEmail:
    @pytest.mark.parametrize(
        "brut,attendu",
        [
            ("  Rahim@JuriX.CM  ", "rahim@jurix.cm"),
            ("rahim@jurix.cm", "rahim@jurix.cm"),
            (None, ""),
            ("", ""),
        ],
    )
    def test_normalisation(self, brut, attendu):
        """
        `_authenticate` cherche en minuscules. Ecrire l'adresse telle quelle
        rendait le compte definitivement inconnectable.
        """
        assert normaliser_email(brut) == attendu
