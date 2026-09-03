"""
API routes for Admin operations.

Endpoints:
- GET /api/v1/admin/users - List users
- POST /api/v1/admin/users - Create user
- PUT /api/v1/admin/users/{id} - Update user
- DELETE /api/v1/admin/users/{id} - Delete user
- GET /api/v1/admin/system - System information

Author: JuriX Development Team
Date: 2026-01-11
"""

import logging
import platform
from datetime import datetime
from typing import Any, Dict, List, TypedDict, cast

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import (
    get_current_admin_user,
    get_current_superadmin_user,
    hash_password,
)
from app.core.database import get_db, health_check_db
from app.models.user import User
from app.schemas.user import UserCreate, UserResponse, UserUpdate

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Admin"])


# ==================== SCHEMAS ====================
#
# Les schemas UserCreate / UserUpdate / UserResponse etaient redefinis ici et
# MASQUAIENT ceux de app/schemas/user.py. Le UserCreate local n'avait aucun champ
# mot de passe : un utilisateur cree par cette API n'aurait jamais pu se
# connecter. Ils sont desormais importes depuis la source unique.
#
# La liste _mock_users (en memoire, perdue a chaque redemarrage, non partagee
# entre workers) a ete supprimee au profit du modele User et de la table users,
# qui existaient tous deux depuis la migration 0b21fb6a3651 sans etre utilises.


# ==================== USER MANAGEMENT ====================


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin_user),
) -> List[User]:
    """Liste les comptes (administrateurs)."""
    logger.info(f"📋 GET /admin/users - skip={skip}, limit={limit}")
    result = await db.execute(
        select(User).order_by(User.id).offset(skip).limit(min(limit, 500))
    )
    return list(result.scalars().all())


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    payload: UserCreate,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin_user),
) -> User:
    """
    Crée un compte (administrateurs).

    Un administrateur ne peut pas créer de compte plus privilégié que le sien :
    sans cette règle, tout admin pourrait se hisser au rang de superadmin en
    créant un second compte.
    """
    if payload.role in ("admin", "superadmin") and not current_admin.is_superadmin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Seul un superadmin peut créer un compte privilégié",
        )

    existing = await db.execute(
        select(User).where(
            or_(User.email == payload.email, User.username == payload.username)
        )
    )
    if existing.scalar_one_or_none() is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Un compte existe déjà avec cet email ou ce nom d'utilisateur",
        )

    user = User(
        email=payload.email,
        username=payload.username,
        full_name=payload.full_name,
        role=payload.role,
        is_active=payload.is_active,
        hashed_password=hash_password(payload.password),
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    logger.info(f"✅ Compte créé : {user.email} (rôle {user.role})")
    return user


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    payload: UserUpdate,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin_user),
) -> User:
    """
    Met à jour un compte (administrateurs).

    Deux garde-fous : seul un superadmin peut modifier un rôle, et personne ne
    peut se retirer ses propres droits — sinon la première erreur de manipulation
    ferme définitivement l'accès à l'instance.
    """
    user = (
        await db.execute(select(User).where(User.id == user_id))
    ).scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Compte {user_id} introuvable"
        )

    changes = payload.model_dump(exclude_unset=True)

    if "role" in changes and not current_admin.is_superadmin():
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Seul un superadmin peut modifier un rôle",
        )
    if user.id == current_admin.id and (
        changes.get("role") not in (None, current_admin.role)
        or changes.get("is_active") is False
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Impossible de réduire ses propres droits",
        )

    password = changes.pop("password", None)
    if password:
        user.hashed_password = hash_password(password)
    for field, value in changes.items():
        setattr(user, field, value)

    await db.commit()
    await db.refresh(user)
    logger.info(f"✏️ Compte mis à jour : {user.email}")
    return user


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_superadmin_user),
) -> None:
    """
    Supprime un compte (superadministrateurs).

    Refuse la suppression de soi-même et celle du dernier superadmin actif : les
    deux mènent à une instance dont plus personne n'a les clés.
    """
    user = (
        await db.execute(select(User).where(User.id == user_id))
    ).scalar_one_or_none()
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Compte {user_id} introuvable"
        )
    if user.id == current_admin.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Impossible de supprimer son propre compte",
        )
    if user.role == "superadmin":
        remaining = (
            await db.execute(
                select(User).where(
                    User.role == "superadmin",
                    User.is_active.is_(True),
                    User.id != user.id,
                )
            )
        ).scalars().first()
        if remaining is None:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Impossible de supprimer le dernier superadmin actif",
            )

    await db.delete(user)
    await db.commit()
    logger.info(f"🗑️ Compte supprimé : {user.email}")
    return None


# ==================== SYSTEM INFORMATION ====================


@router.get("/system")
async def get_system_info(
    db: AsyncSession = Depends(get_db),
    current_admin: User = Depends(get_current_admin_user),
) -> Dict:
    """
    Get system information (admin only).
    
    **Returns:**
    - Database status
    - System info
    - Service health
    - Version info
    
    **Requires:** Admin authentication
    """
    logger.info("🖥️  GET /admin/system")
    
    # Check database health
    db_healthy = await health_check_db()
    
    return {
        "system": {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": platform.python_version(),
            "hostname": platform.node(),
        },
        "database": {
            "status": "healthy" if db_healthy else "unhealthy",
            "connected": db_healthy,
        },
        "services": {
            "api": "running",
            "search": "running",
            "rag": "running",
        },
        "version": {
            "api": "2.1.0",
            "build": "production",
        },
        "timestamp": datetime.now().isoformat(),
    }


# ==================== HEALTH CHECK ====================


@router.get("/health")
async def health_check() -> Dict:
    """
    Check admin service health.
    
    **Returns:**
    - Service status
    """
    return {
        "service": "Admin",
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
    }
