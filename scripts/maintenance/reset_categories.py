"""
Script pour recréer les catégories avec l'ordre d'affichage correct
"""
import asyncio
from sqlalchemy import delete, select
from app.core.database import AsyncSessionLocal
from app.models.law import Category, Law

# Catégories dans l'ordre souhaité par l'utilisateur
NEW_CATEGORIES = [
    {"name": "Droit Constitutionnel", "icon": "⚖️", "description": "Constitution et lois fondamentales", "display_order": 1},
    {"name": "Droit Civil", "icon": "👥", "description": "Lois relatives au droit civil", "display_order": 2},
    {"name": "Droit Pénal", "icon": "🔒", "description": "Code pénal et infractions", "display_order": 3},
    {"name": "Droit du Travail", "icon": "👔", "description": "Lois relatives au travail", "display_order": 4},
    {"name": "Droit Fiscal", "icon": "💰", "description": "Lois relatives à la fiscalité", "display_order": 5},
    {"name": "Droit des Affaires", "icon": "💼", "description": "Lois commerciales et des affaires", "display_order": 6},
    {"name": "Lois Internationales Ratifiées", "icon": "🌍", "description": "Traités et conventions internationales", "display_order": 7},
    {"name": "Lois", "icon": "📜", "description": "Lois votées par le parlement", "display_order": 8},
    {"name": "Ordonnances", "icon": "📋", "description": "Ordonnances présidentielles", "display_order": 9},
    {"name": "Décrets", "icon": "📑", "description": "Décrets gouvernementaux", "display_order": 10},
    {"name": "Arrêtés", "icon": "📄", "description": "Arrêtés ministériels et préfectoraux", "display_order": 11},
    {"name": "Circulaires", "icon": "📨", "description": "Circulaires administratives", "display_order": 12},
    {"name": "Décisions", "icon": "⚡", "description": "Décisions administratives et judiciaires", "display_order": 13},
    {"name": "Autres", "icon": "📦", "description": "Autres textes juridiques", "display_order": 14},
]

async def reset_categories():
    async with AsyncSessionLocal() as db:
        # 1. Mettre à NULL les category_id des lois
        print("🔄 Mise à NULL des category_id des lois...")
        await db.execute(Law.__table__.update().values(category_id=None))
        await db.commit()
        
        # 2. Supprimer toutes les catégories
        print("🗑️  Suppression de toutes les catégories...")
        result = await db.execute(delete(Category))
        deleted_count = result.rowcount
        await db.commit()
        print(f"✅ {deleted_count} catégories supprimées")
        
        # 3. Créer les nouvelles catégories avec l'ordre
        print("📝 Création des nouvelles catégories...")
        for cat_data in NEW_CATEGORIES:
            category = Category(
                name=cat_data["name"],
                icon=cat_data["icon"],
                description=cat_data["description"],
                display_order=cat_data["display_order"]
            )
            db.add(category)
        
        await db.commit()
        print(f"✅ {len(NEW_CATEGORIES)} nouvelles catégories créées")
        
        # 4. Afficher le résultat dans l'ordre
        result = await db.execute(select(Category).order_by(Category.display_order))
        categories = result.scalars().all()
        
        print("\n📋 Catégories (dans l'ordre d'affichage):")
        print("-" * 50)
        for cat in categories:
            print(f"  {cat.display_order:2}. {cat.icon} {cat.name}")
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(reset_categories())
