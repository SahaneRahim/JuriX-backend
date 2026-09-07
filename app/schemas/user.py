"""
Pydantic schemas for User model.

Request/response schemas for user authentication and management.

Author: JuriX Team
Date: 2026-01-12
"""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

# ==================== Base Schemas ====================


def valider_force_du_mot_de_passe(v: str) -> str:
    """
    Politique de mot de passe, partagee par TOUS les points d'entree.

    Elle vivait uniquement dans `UserCreate.validate_password`, donc dans la
    seule creation par un administrateur. L'inscription publique ne peut pas
    heriter de ce schema (voir `SignupRequest`), et une seconde copie aurait
    diverge. Une violation produit un 422 Pydantic, pas un 400.
    """
    if not any(c.isupper() for c in v):
        raise ValueError("Password must contain at least one uppercase letter")
    if not any(c.islower() for c in v):
        raise ValueError("Password must contain at least one lowercase letter")
    if not any(c.isdigit() for c in v):
        raise ValueError("Password must contain at least one digit")
    return v


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
    # `admin.create_user` lit `payload.is_active`, or aucun schema de la chaine
    # ne definissait ce champ : toute creation de compte levait un AttributeError
    # et repondait 500. Le defaut True reproduit l'intention du code appelant.
    is_active: bool = True

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Validate password strength."""
        return valider_force_du_mot_de_passe(v)


class UserUpdate(BaseModel):
    """Schema for updating user information."""

    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=100)
    full_name: Optional[str] = Field(None, max_length=255)
    password: Optional[str] = Field(None, min_length=8, max_length=100)
    is_active: Optional[bool] = None
    # `admin.update_user` teste `"role" in changes` pour reserver le changement
    # de role aux superadmins. Le champ n'existait pas ici : Pydantic l'ecartait
    # en silence, donc AUCUN role n'etait modifiable et le garde-fou ne se
    # declenchait jamais. Meme motif que UserBase.
    role: Optional[str] = Field(None, pattern="^(user|admin|superadmin)$")


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


# ============================================================================
# Inscription publique et connexion Google
# ============================================================================


class SignupRequest(BaseModel):
    """
    Inscription publique : nom, adresse, mot de passe. Rien d'autre.

    N'HERITE PAS DE `UserBase`, ET C'EST LE POINT CENTRAL. `UserBase` porte
    `role` (motif `^(user|admin|superadmin)$`, defaut `"user"`) et `UserCreate`
    y ajoute `is_active` — deux champs FOURNIS PAR LE CLIENT. Reutiliser
    `UserCreate` pour une route publique aurait laisse n'importe qui poster
    `{"role": "superadmin"}` et s'ouvrir l'administration. La seule protection
    existante est imperative, dans `admin.create_user`, et ne couvre pas cette
    route.

    Ici ces champs N'EXISTENT PAS, et `extra="forbid"` fait de leur envoi un
    422 plutot qu'un silence : la tentative laisse une trace dans les journaux
    d'acces au lieu d'etre ignoree sans que personne ne le sache jamais.

    `username` n'est pas demande non plus : il est derive de l'adresse par
    `app/services/user_identity.py`.
    """

    model_config = ConfigDict(extra="forbid")

    full_name: str = Field(..., min_length=2, max_length=255)
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=100)

    @field_validator("password")
    @classmethod
    def valider_mot_de_passe(cls, v: str) -> str:
        return valider_force_du_mot_de_passe(v)

    @field_validator("full_name")
    @classmethod
    def valider_nom(cls, v: str) -> str:
        nettoye = " ".join(v.split())
        if len(nettoye) < 2:
            raise ValueError("Full name must not be blank")
        return nettoye


class GoogleAuthRequest(BaseModel):
    """
    Le jeton d'identite rendu par Google Identity Services.

    Borne des deux cotes : un jeton Google fait environ un kilo-octet. Sans
    plafond, on accepterait de verifier cryptographiquement un megaoctet
    envoye par n'importe qui.
    """

    model_config = ConfigDict(extra="forbid")

    credential: str = Field(..., min_length=100, max_length=4096)
