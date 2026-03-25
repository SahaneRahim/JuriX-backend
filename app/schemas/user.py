"""
Pydantic schemas for User model.

Request/response schemas for user authentication and management.

Author: JuriX Team
Date: 2026-01-12
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, EmailStr, Field, field_validator

# ==================== Base Schemas ====================


class UserBase(BaseModel):
    """Base user schema with common fields."""

    email: EmailStr
    username: str = Field(..., min_length=3, max_length=100)
    full_name: Optional[str] = Field(None, max_length=255)
    role: str = Field(default="user", pattern="^(user|admin|superadmin)$")

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate username format."""
        if not v.isalnum() and "_" not in v and "-" not in v:
            raise ValueError("Username must be alphanumeric (with _ or - allowed)")
        return v.lower()


# ==================== Request Schemas ====================


class UserCreate(UserBase):
    """Schema for creating a new user."""

    password: str = Field(..., min_length=8, max_length=100)

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserUpdate(BaseModel):
    """Schema for updating user information."""

    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=100)
    full_name: Optional[str] = Field(None, max_length=255)
    password: Optional[str] = Field(None, min_length=8, max_length=100)
    is_active: Optional[bool] = None


class UserLogin(BaseModel):
    """Schema for user login."""

    email: EmailStr
    password: str


# ==================== Response Schemas ====================


class UserResponse(UserBase):
    """Schema for user response (public fields)."""

    id: int
    is_active: bool
    is_verified: bool
    created_at: datetime
    last_login_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class UserWithToken(UserResponse):
    """Schema for user response with authentication token."""

    access_token: str
    token_type: str = "bearer"


# ==================== Admin Schemas ====================


class UserAdminResponse(UserResponse):
    """Schema for admin view of user (includes all fields)."""

    updated_at: datetime

    class Config:
        from_attributes = True
