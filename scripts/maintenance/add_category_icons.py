"""
Script pour ajouter des icônes à toutes les catégories et supprimer "Droit de la Famille"
"""
import asyncio
from sqlalchemy import select, delete
from app.core.database import AsyncSessionLocal
from app.models.law import Category

# Mapping des catégories avec leurs icônes appropriées
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

async def update_category_icons():
    async with AsyncSessionLocal() as db:
        # 1. Supprimer "Droit de la Famille"
        result = await db.execute(
            delete(Category).where(Category.name == "Droit de la Famille")
        )
        if result.rowcount > 0:
            print("✅ 'Droit de la Famille' supprimé")
        
        # 2. Récupérer toutes les catégories
        result = await db.execute(select(Category))
        categories = result.scalars().all()
        
        # 3. Ajouter les icônes
        updated_count = 0
        for category in categories:
            if category.name in CATEGORY_ICONS:
                icon = CATEGORY_ICONS[category.name]
                
                # Mettre à jour la description avec l'icône
                if category.description:
                    # Si la description existe déjà, ajouter l'icône au début
                    if not category.description.startswith(icon):
                        category.description = f"{icon} {category.description}"
                else:
                    # Créer une description avec l'icône
                    category.description = f"{icon} Lois relatives au {category.name.lower()}"
                
                updated_count += 1
                print(f"  {icon} {category.name}")
        
        await db.commit()
        print(f"\n✅ {updated_count} catégories mises à jour avec des icônes")
        
        # 4. Afficher le résultat final
        result = await db.execute(select(Category).order_by(Category.id))
        categories = result.scalars().all()
        
        print(f"\n📋 {len(categories)} catégories finales:")
        for cat in categories:
            desc_preview = cat.description[:50] if cat.description else "Pas de description"
            print(f"  ID={cat.id}: {cat.name}")
            print(f"       {desc_preview}")

if __name__ == "__main__":
    asyncio.run(update_category_icons())
