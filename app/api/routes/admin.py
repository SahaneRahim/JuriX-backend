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
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db, health_check_db

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Admin"])


# ==================== SCHEMAS ====================


class UserCreate(BaseModel):
    """Schema for creating a user."""
    
    email: EmailStr = Field(..., description="User email")
    username: str = Field(..., min_length=3, max_length=50, description="Username")
    role: str = Field(default="user", description="User role (user or admin)")
    is_active: bool = Field(default=True, description="User active status")


class UserUpdate(BaseModel):
    """Schema for updating a user."""
    
    email: EmailStr | None = None
    username: str | None = Field(None, min_length=3, max_length=50)
    role: str | None = None
    is_active: bool | None = None


class UserResponse(BaseModel):
    """Schema for user response."""
    
    id: int
    email: str
    username: str
    role: str
    is_active: bool
    created_at: datetime
 
 
class UserDict(TypedDict):
    """Internal dictionary type for mock storage."""
    id: int
    email: str
    username: str
    role: str
    is_active: bool
    created_at: datetime


# ==================== MOCK USER STORAGE ====================
# TODO: Replace with actual database model


_mock_users: List[UserDict] = [
    {
        "id": 1,
        "email": "admin@jurix.cm",
        "username": "admin",
        "role": "admin",
        "is_active": True,
        "created_at": datetime.now(),
    },
    {
        "id": 2,
        "email": "user@jurix.cm",
        "username": "user1",
        "role": "user",
        "is_active": True,
        "created_at": datetime.now(),
    },
]

_next_user_id = 3


# ==================== AUTHENTICATION ====================


async def get_current_admin():
    """
    Dependency for admin authentication.
    
    TODO: Implement actual JWT authentication.
    For now, returns mock admin user.
    """
    return {"id": 1, "role": "admin", "username": "admin"}


# ==================== USER MANAGEMENT ====================


@router.get("/users", response_model=List[UserResponse])
async def list_users(
    skip: int = 0,
    limit: int = 100,
    current_admin: dict = Depends(get_current_admin),
) -> List[UserResponse]:
    """
    List all users (admin only).
    
    **Parameters:**
    - skip: Number of records to skip
    - limit: Maximum records to return
    
    **Returns:**
    - List of users
    
    **Requires:** Admin authentication
    """
    logger.info(f"📋 GET /admin/users - skip={skip}, limit={limit}")
    
    users = _mock_users[skip : skip + limit]
    return [
        UserResponse(
            id=u["id"],
            email=u["email"],
            username=u["username"],
            role=u["role"],
            is_active=u["is_active"],
            created_at=u["created_at"]
        ) 
        for u in users
    ]


@router.post("/users", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    user: UserCreate,
    current_admin: dict = Depends(get_current_admin),
) -> UserResponse:
    """
    Create a new user (admin only).
    
    **Parameters:**
    - email: User email (unique)
    - username: Username (unique)
    - role: User role (user or admin)
    - is_active: Active status
    
    **Returns:**
    - Created user
    
    **Raises:**
    - 409: Email or username already exists
    
    **Requires:** Admin authentication
    """
    global _next_user_id
    
    assert user is not None, "UserCreate must not be None"
    assert isinstance(user.email, str) and len(user.email) > 0, "Email must be a non-empty string"
    assert isinstance(user.username, str) and len(user.username) >= 3, "Username must be at least 3 characters"

    logger.info(f"➕ POST /admin/users - Creating user: {user.email}")
    
    # Check for duplicate email or username
    for existing_user in _mock_users:
        if existing_user["email"] == user.email:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Email already exists: {user.email}",
            )
        if existing_user["username"] == user.username:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"Username already exists: {user.username}",
            )
    
    # Create new user
    new_user: UserDict = {
        "id": _next_user_id,
        "email": str(user.email),
        "username": user.username,
        "role": user.role,
        "is_active": user.is_active,
        "created_at": datetime.now(),
    }
    
    _mock_users.append(new_user)
    _next_user_id += 1
    
    logger.info(f"✅ User created: ID={new_user['id']}, email={new_user['email']}")
    return UserResponse(
        id=new_user["id"],
        email=new_user["email"],
        username=new_user["username"],
        role=new_user["role"],
        is_active=new_user["is_active"],
        created_at=new_user["created_at"]
    )


@router.put("/users/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_update: UserUpdate,
    current_admin: dict = Depends(get_current_admin),
) -> UserResponse:
    """
    Update a user (admin only).
    
    **Parameters:**
    - user_id: User ID
    - Partial update fields
    
    **Returns:**
    - Updated user
    
    **Raises:**
    - 404: User not found
    
    **Requires:** Admin authentication
    """
    assert isinstance(user_id, int) and user_id > 0, "user_id must be a positive integer"
    assert user_update is not None, "UserUpdate must not be None"

    logger.info(f"✏️  PUT /admin/users/{user_id}")
    
    # Find user
    user = next((u for u in _mock_users if u["id"] == user_id), None)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User not found: {user_id}",
        )
    
    # Update fields
    update_data = user_update.model_dump(exclude_unset=True)
    user_raw = cast(Dict[str, Any], user)
    for field, value in update_data.items():
        user_raw[field] = value
    
    logger.info(f"✅ User {user_id} updated")
    return UserResponse(
        id=user["id"],
        email=user["email"],
        username=user["username"],
        role=user["role"],
        is_active=user["is_active"],
        created_at=user["created_at"]
    )


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    current_admin: dict = Depends(get_current_admin),
):
    """
    Delete a user (admin only).
    
    **Parameters:**
    - user_id: User ID
    
    **Returns:**
    - 204 No Content
    
    **Raises:**
    - 404: User not found
    
    **Requires:** Admin authentication
    """
    assert isinstance(user_id, int) and user_id > 0, "user_id must be a positive integer"

    logger.info(f"🗑️  DELETE /admin/users/{user_id}")
    
    # Find user
    user = next((u for u in _mock_users if u["id"] == user_id), None)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"User not found: {user_id}",
        )
    
    # Remove user
    _mock_users.remove(user)
    
    logger.info(f"✅ User {user_id} deleted")
    return None


# ==================== SYSTEM INFORMATION ====================


@router.get("/system")
async def get_system_info(
    db: AsyncSession = Depends(get_db),
    current_admin: dict = Depends(get_current_admin),
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
