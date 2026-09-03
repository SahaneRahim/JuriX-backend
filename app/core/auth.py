"""
Authentication utilities for JuriX.

Provides JWT token generation, password hashing, and user authentication.

Author: JuriX Team
Date: 2026-01-12
"""

from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
import base64
import hashlib

import bcrypt
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
    try:
        return bcrypt.checkpw(_prepare(plain_password), hashed_password.encode("utf-8"))
    except (ValueError, TypeError):
        # Empreinte illisible ou tronquee : on refuse plutot que de propager.
        return False


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
