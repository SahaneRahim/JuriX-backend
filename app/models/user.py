"""
User model for authentication and authorization.

Provides user management with role-based access control (RBAC).

Author: JuriX Team
Date: 2026-01-12
"""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Integer, String
from sqlalchemy.orm import relationship

from app.core.database import Base


class User(Base):
    """
    User model for authentication and authorization.

    Attributes:
        id: Unique user identifier
        email: User email (unique, used for login)
        username: Display username
        hashed_password: Bcrypt hashed password
        full_name: User's full name
        role: User role (user, admin, superadmin)
        is_active: Account active status
        is_verified: Email verification status
        created_at: Account creation timestamp
        updated_at: Last update timestamp
        last_login_at: Last login timestamp

    Relationships:
        conversations: User's chat conversations
    """

    __tablename__ = "users"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Authentication
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    # NULLABLE : un compte cree par Google n'a pas de mot de passe. La
    # contrainte CHECK `ck_users_au_moins_une_identite` garantit qu'un compte
    # porte toujours au moins une identite — mot de passe ou Google.
    hashed_password = Column(String(255), nullable=True)

    # Claim `sub` du jeton d'identite Google, seul identifiant stable d'un
    # compte Google (l'adresse e-mail peut changer chez eux). Unique et
    # nullable : les comptes par mot de passe le laissent a NULL, et PostgreSQL
    # tolere plusieurs NULL sur un index unique.
    google_sub = Column(String(255), nullable=True, unique=True, index=True)

    # Profile
    full_name = Column(String(255), nullable=True)

    # Authorization
    role = Column(
        String(50),
        nullable=False,
        default="user",
        index=True,
        comment="Role: user, admin, superadmin",
    )

    # Status flags
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, nullable=False)
    last_login_at = Column(DateTime, nullable=True)

    # Relationships
    #
    # `lazy="raise"` et NON `selectin`. `get_current_user` charge cet objet a
    # CHAQUE requete authentifiee : avec un chargement empresse, chacune
    # tirait en plus toutes les conversations du compte — la liste que
    # l'historique de chat fait precisement grossir, pour une relation que
    # personne ne lit.
    #
    # La cascade ORM est retiree en meme temps, et c'est indissociable : elle
    # exige de charger la collection pour supprimer un compte, ce que `raise`
    # interdit. La suppression des conversations d'un compte est desormais
    # EXPLICITE dans `admin.delete_user`.
    conversations = relationship("Conversation", back_populates="user", lazy="raise")

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email='{self.email}', role='{self.role}')>"

    def has_role(self, required_role: str) -> bool:
        """
        Check if user has required role or higher.

        Role hierarchy: superadmin > admin > user

        Args:
            required_role: Required role to check

        Returns:
            True if user has required role or higher

        Example:
            >>> user.role = "admin"
            >>> user.has_role("user")  # True
            >>> user.has_role("superadmin")  # False
        """
        role_hierarchy = {"superadmin": 3, "admin": 2, "user": 1}
        user_level = role_hierarchy.get(self.role, 0)
        required_level = role_hierarchy.get(required_role, 0)
        return user_level >= required_level

    def is_admin(self) -> bool:
        """Check if user has admin or superadmin role."""
        return self.role in ("admin", "superadmin")

    def is_superadmin(self) -> bool:
        """Check if user has superadmin role."""
        return self.role == "superadmin"
