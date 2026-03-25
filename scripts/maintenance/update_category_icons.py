"""
Script pour mettre à jour les icônes des catégories dans le nouveau champ 'icon'
"""
import asyncio
from sqlalchemy import select
from app.core.database import AsyncSessionLocal
from app.models.law import Category

# Mapping des catégories avec leurs icônes
CATEGORY_ICONS = {
    "Droit Constitutionnel": "⚖️",
    "Droit Civil": "👥",
    "Droit Pénal": "🔒",
    "Droit Commercial": "🏪",
    "Droit du Travail": "👔",
    "Droit Fiscal": "💰",
    "Droit Administratif": "🏛️",
    "Droit OHADA": "🌍",
    "Droit des Affaires": "💼",
    "Procédure Civile": "📋",
    "Procédure Pénale": "⚖️",
}

async def update_icons():
    async with AsyncSessionLocal() as db:
        # Récupérer toutes les catégories
        result = await db.execute(select(Category))
        categories = result.scalars().all()
        
        updated_count = 0
        for category in categories:
            if category.name in CATEGORY_ICONS:
                # Mettre à jour le champ icon
                category.icon = CATEGORY_ICONS[category.name]
                
                # Nettoyer la description (enlever les icônes si présentes)
                if category.description and category.description.startswith(category.icon):
                    # Enlever l'icône du début de la description
                    category.description = category.description[len(category.icon):].strip()
                
                updated_count += 1
                print(f"✅ {category.icon} {category.name}")
        
        await db.commit()
        print(f"\n✅ {updated_count} catégories mises à jour avec des icônes")
        
        # Afficher le résultat
        result = await db.execute(select(Category).order_by(Category.id))
        categories = result.scalars().all()
        
        print(f"\n📋 Catégories finales:")
        for cat in categories:
            icon_display = cat.icon if cat.icon else "📜"
            print(f"  {icon_display} {cat.name}")

if __name__ == "__main__":
    asyncio.run(update_icons())
