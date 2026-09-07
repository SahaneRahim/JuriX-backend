"""
Authentication utilities for JuriX.

Provides JWT token generation, password hashing, and user authentication.

Author: JuriX Team
Date: 2026-01-12
"""

import base64
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.core.database import get_db
from app.models.user import User

# ==================== HACHAGE DES MOTS DE PASSE ====================
#
# bcrypt est utilise directement, sans passlib. passlib 1.7.4 (derniere version,
# 2020) lit `bcrypt.__about__.__version__` pour detecter le backend — attribut
# supprime depuis bcrypt 4.1. Avec bcrypt 5.x, le hachage echouait totalement
# avec "password cannot be longer than 72 bytes" MEME pour un mot de passe de
# 13 caracteres. L'authentification etait donc inutilisable telle que declaree.
#
# bcrypt ignore silencieusement tout ce qui depasse 72 octets : deux mots de
# passe partageant les 72 premiers octets seraient equivalents. On pre-hache donc
# en SHA-256 puis on encode en base64, ce qui donne une entree de longueur fixe
# (44 octets) et supporte les phrases de passe de n'importe quelle longueur.


def _prepare(password: str) -> bytes:
    """Normalise un mot de passe en une entree bcrypt de longueur fixe."""
    digest = hashlib.sha256(password.encode("utf-8")).digest()
    return base64.b64encode(digest)

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

# Variante tolerante : `auto_error=False` rend None au lieu de lever 401 quand
# l'en-tete Authorization est absent. Utilisee par `get_current_user_optional`.
_oauth2_optionnel = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)

# JWT settings — lus depuis la configuration.
# Auparavant ALGORITHM et ACCESS_TOKEN_EXPIRE_MINUTES etaient codes en dur ici
# (7 jours) alors que config.py declare 30 minutes : les deux valeurs se
# contredisaient et celle du fichier de configuration n'avait aucun effet.
# Le garde-fou hasattr etait vestigial : SECRET_KEY existe toujours.
SECRET_KEY = settings.SECRET_KEY
ALGORITHM = settings.ALGORITHM
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt.

    Args:
        password: Plain text password

    Returns:
        Hashed password
    """
    return bcrypt.hashpw(_prepare(password), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.

    Args:
        plain_password: Plain text password to verify
        hashed_password: Bcrypt hash to verify against

    Returns:
        True if password matches, False otherwise
    """
    # Un compte cree par Google n'a PAS de mot de passe : `hashed_password` est
    # NULL. Sans ce controle, `.encode()` leve un AttributeError — que le
    # `except (ValueError, TypeError)` ci-dessous n'attrape pas — et la simple
    # tentative de connexion par mot de passe sur un compte Google repondait 500.
    if not hashed_password:
        return False

    try:
        return bcrypt.checkpw(_prepare(plain_password), hashed_password.encode("utf-8"))
    except (ValueError, TypeError):
        # Empreinte illisible ou tronquee : on refuse plutot que de propager.
        return False


def duree_de_session(user: User) -> timedelta:
    """
    Duree de vie du jeton, selon le privilege du compte.

    Les valeurs sont lues sur `settings` A L'APPEL et non a l'import : les
    constantes de module ci-dessus sont figees au chargement, et un test qui
    modifierait le reglage ne verrait rien changer.
    """
    minutes = (
        settings.ADMIN_TOKEN_EXPIRE_MINUTES
        if user.is_admin()
        else settings.ACCESS_TOKEN_EXPIRE_MINUTES
    )
    return timedelta(minutes=minutes)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Data to encode in token (typically {"sub": user_email})
        expires_delta: Optional expiration time delta

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def get_current_user(
    token: str = Depends(oauth2_scheme), db: AsyncSession = Depends(get_db)
) -> User:
    """
    Get current authenticated user from JWT token.

    Args:
        token: JWT token from Authorization header
        db: Database session

    Returns:
        Authenticated User object

    Raises:
        HTTPException: If token is invalid or user not found
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # Get user from database
    query = select(User).where(User.email == email)
    result = await db.execute(query)
    user = result.scalar_one_or_none()

    if user is None:
        raise credentials_exception

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="User account is inactive"
        )

    return user


async def get_current_user_optional(
    token: Optional[str] = Depends(_oauth2_optionnel), db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """
    L'utilisateur courant, ou None quand la requete est anonyme.

    Existe pour les routes qui doivent servir les deux publics — le chat reste
    utilisable sans compte, mais doit rattacher la conversation quand il y en a
    un.

    LA DISTINCTION QUI COMPTE : pas d'en-tete `Authorization` rend `None` ; un
    en-tete PRESENT mais dont le jeton est invalide ou expire leve 401, jamais
    `None`. Degrader silencieusement en anonyme serait le pire comportement :
    l'utilisateur croirait sa conversation enregistree alors qu'elle partirait
    en `user_id` NULL, invisible dans sa liste. Le front sait deja traiter un
    401 (`apiFetch` : deconnexion puis redirection).
    """
    if token is None:
        return None
    return await get_current_user(token=token, db=db)


async def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Get current active user.

    Args:
        current_user: Current user from token

    Returns:
        Active user

    Raises:
        HTTPException: If user is inactive
    """
    if not current_user.is_active:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Inactive user")
    return current_user


async def get_current_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Get current user with admin role.

    Args:
        current_user: Current user from token

    Returns:
        Admin user

    Raises:
        HTTPException: If user doesn't have admin role
    """
    if not current_user.is_admin():
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return current_user


async def get_current_superadmin_user(current_user: User = Depends(get_current_user)) -> User:
    """
    Get current user with superadmin role.

    Args:
        current_user: Current user from token

    Returns:
        Superadmin user

    Raises:
        HTTPException: If user doesn't have superadmin role
    """
    if not current_user.is_superadmin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Superadmin access required"
        )
    return current_user
