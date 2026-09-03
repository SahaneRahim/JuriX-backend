"""
Routes d'authentification.

`app/core/auth.py` implémentait déjà l'intégralité de la chaîne JWT — hachage
bcrypt, création de jeton, dépendances par rôle — mais n'était importé nulle
part, et aucune route de connexion n'existait. Les endpoints « admin » passaient
par deux stubs renvoyant un dictionnaire en dur.

Deux points d'entrée de connexion, volontairement :

- `POST /login` accepte un formulaire OAuth2. C'est exactement l'URL déclarée
  par `oauth2_scheme` (`tokenUrl="/api/v1/auth/login"`), donc le bouton
  *Authorize* de /docs fonctionne — précieux pour administrer l'instance tant
  que l'interface d'administration n'est pas terminée.
- `POST /login/json` accepte du JSON, plus naturel pour le front SvelteKit.

Les deux partagent le même `_authenticate`.

Il n'y a **pas** de route d'inscription publique : la création de comptes passe
par `POST /api/v1/admin/users` ou par `scripts/create_admin.py`. Une inscription
ouverte donnerait à n'importe qui l'accès au RAG, donc à la facturation Gemini.

Author: JuriX Team
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import (
    create_access_token,
    get_current_active_user,
    verify_password,
)
from app.core.database import get_db
from app.models.user import User
from app.schemas.user import UserLogin, UserResponse, UserWithToken

logger = logging.getLogger(__name__)

router = APIRouter()

# Message unique pour « email inconnu » et « mot de passe faux » : les
# distinguer permettrait d'énumérer les comptes existants.
_INVALID = "Identifiants invalides"


async def _authenticate(db: AsyncSession, email: str, password: str) -> User:
    """
    Vérifie un couple email / mot de passe.

    Raises:
        HTTPException 401: identifiants invalides
        HTTPException 403: compte désactivé
    """
    result = await db.execute(select(User).where(User.email == email.lower().strip()))
    user = result.scalar_one_or_none()

    if user is None or not verify_password(password, user.hashed_password):
        logger.warning(f"🔒 Échec de connexion pour {email!r}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=_INVALID,
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Compte désactivé"
        )

    # Les colonnes DateTime du modele sont SANS fuseau : y ecrire un datetime
    # avec fuseau leve "can't subtract offset-naive and offset-aware datetimes"
    # au moment du flush. On stocke donc de l'UTC naif, comme le reste du schema.
    user.last_login_at = datetime.now(timezone.utc).replace(tzinfo=None)
    await db.commit()
    await db.refresh(user)

    logger.info(f"✅ Connexion de {user.email} (rôle {user.role})")
    return user


@router.post("/login", response_model=UserWithToken, status_code=status.HTTP_200_OK)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: AsyncSession = Depends(get_db),
) -> UserWithToken:
    """
    Connexion au format formulaire OAuth2 (utilisée par /docs).

    Le champ `username` porte l'adresse email.
    """
    user = await _authenticate(db, form_data.username, form_data.password)
    return UserWithToken(
        **UserResponse.model_validate(user).model_dump(),
        access_token=create_access_token({"sub": user.email}),
    )


@router.post("/login/json", response_model=UserWithToken, status_code=status.HTTP_200_OK)
async def login_json(
    credentials: UserLogin,
    db: AsyncSession = Depends(get_db),
) -> UserWithToken:
    """Connexion au format JSON (utilisée par le front)."""
    user = await _authenticate(db, credentials.email, credentials.password)
    return UserWithToken(
        **UserResponse.model_validate(user).model_dump(),
        access_token=create_access_token({"sub": user.email}),
    )


@router.get("/me", response_model=UserResponse, status_code=status.HTTP_200_OK)
async def read_me(current_user: User = Depends(get_current_active_user)) -> User:
    """Renvoie l'utilisateur associé au jeton fourni."""
    return current_user


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(current_user: User = Depends(get_current_active_user)) -> None:
    """
    Déconnexion.

    Les JWT sont sans état : rien n'est révoqué côté serveur, le client jette son
    jeton. L'endpoint existe pour donner un point d'appel explicite au front et
    pour tracer la déconnexion.
    """
    logger.info(f"👋 Déconnexion de {current_user.email}")
    return None
