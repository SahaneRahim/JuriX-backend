"""
Script final pour nettoyer toutes les catégories de test
"""
import asyncio
from sqlalchemy import select, delete
from app.core.database import AsyncSessionLocal
from app.models.law import Category

async def final_cleanup():
    async with AsyncSessionLocal() as db:
        # Liste des catégories de test à supprimer
        test_categories = [
            "Category 1",
            "Category 2", 
            "Duplicate Test",
            "Get Test Category",
            "Updated Name",
            "Extra Fields Test"
        ]
        
        # Supprimer toutes les catégories de test
        result = await db.execute(
            delete(Category).where(Category.name.in_(test_categories))
        )
        deleted_count = result.rowcount
        await db.commit()
        
        print(f"✅ {deleted_count} catégories de test supprimées")
        
        # Afficher les catégories restantes
        result = await db.execute(select(Category).order_by(Category.id))
        categories = result.scalars().all()
        
        print(f"\n📋 {len(categories)} catégories restantes:")
        for cat in categories:
            icon = "⚖️" if cat.name == "Droit Constitutionnel" else "📁"
            print(f"  {icon} ID={cat.id}: {cat.name}")

if __name__ == "__main__":
    asyncio.run(final_cleanup())
