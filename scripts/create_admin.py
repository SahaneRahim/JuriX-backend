"""
Crée ou met à jour le compte administrateur initial.

Aucun utilisateur n'était créé nulle part dans le projet : le modèle `User` et la
table `users` existaient depuis la migration 0b21fb6a3651 sans qu'aucun code ne
les alimente. Une fois l'authentification branchée, sans ce script il devient
impossible de se connecter à sa propre instance.

Idempotent : relancé, il ne duplique rien et ne modifie que ce qui doit l'être.

Usage:
    python scripts/create_admin.py
    python scripts/create_admin.py --email admin@jurix.cm --username admin
    python scripts/create_admin.py --reset-password
    python scripts/create_admin.py --role superadmin

Les valeurs par défaut viennent de .env : ADMIN_EMAIL, ADMIN_USERNAME,
ADMIN_PASSWORD. Si aucun mot de passe n'est fourni ni par option ni par
l'environnement, il est demandé de façon interactive — jamais de valeur par
défaut en dur.
"""

import argparse
import asyncio
import getpass
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv  # noqa: E402
from sqlalchemy import select  # noqa: E402

load_dotenv()

from app.core.auth import hash_password  # noqa: E402
from app.core.database import AsyncSessionLocal  # noqa: E402
from app.models.user import User  # noqa: E402
from app.schemas.user import UserCreate  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Crée ou met à jour un administrateur JuriX")
    p.add_argument("--email", default=os.getenv("ADMIN_EMAIL", "admin@jurix.cm"))
    p.add_argument("--username", default=os.getenv("ADMIN_USERNAME", "admin"))
    p.add_argument("--password", default=os.getenv("ADMIN_PASSWORD"))
    p.add_argument(
        "--role",
        default=os.getenv("ADMIN_ROLE", "superadmin"),
        choices=["admin", "superadmin"],
    )
    p.add_argument(
        "--reset-password",
        action="store_true",
        help="Réinitialise le mot de passe d'un compte existant",
    )
    return p.parse_args()


async def main() -> int:
    args = parse_args()

    password = args.password
    if not password:
        password = getpass.getpass(f"Mot de passe pour {args.email} : ")
    if not password:
        print("❌ Aucun mot de passe fourni.", file=sys.stderr)
        return 1

    # On valide via UserCreate pour que le script et l'API appliquent
    # exactement la même politique de mot de passe.
    try:
        UserCreate(
            email=args.email,
            username=args.username,
            password=password,
            role=args.role,
        )
    except Exception as e:
        print(f"❌ Données invalides : {e}", file=sys.stderr)
        return 1

    try:
        async with AsyncSessionLocal() as db:
            existing = (
                await db.execute(select(User).where(User.email == args.email.lower()))
            ).scalar_one_or_none()

            if existing is None:
                db.add(
                    User(
                        email=args.email.lower(),
                        username=args.username.lower(),
                        hashed_password=hash_password(password),
                        role=args.role,
                        is_active=True,
                        is_verified=True,
                    )
                )
                await db.commit()
                print(f"✅ Compte créé : {args.email} (rôle {args.role})")
                return 0

            changed = []
            if existing.role != args.role:
                existing.role = args.role
                changed.append("rôle")
            if not existing.is_active:
                existing.is_active = True
                changed.append("réactivation")
            if args.reset_password:
                existing.hashed_password = hash_password(password)
                changed.append("mot de passe")

            if changed:
                await db.commit()
                print(f"✅ Compte mis à jour : {args.email} ({', '.join(changed)})")
            else:
                print(
                    f"ℹ️  Compte déjà conforme : {args.email} (rôle {existing.role}). "
                    "Utilisez --reset-password pour changer le mot de passe."
                )
            return 0

    except Exception as e:
        print(f"❌ Base de données inaccessible : {e}", file=sys.stderr)
        print("   Vérifiez DATABASE_URL dans .env et que les migrations sont appliquées.",
              file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
