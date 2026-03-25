"""Seed database with sample Cameroonian laws for testing."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datetime import date
from app.core.database import AsyncSessionLocal, engine, Base
from app.models.law import Law, Category, Article


SAMPLE_CATEGORIES = [
    {"id": 1, "name": "Droit Civil", "description": "Droit civil camerounais"},
    {"id": 2, "name": "Droit Commercial OHADA", "description": "Droit commercial OHADA"},
    {"id": 3, "name": "Droit Pénal", "description": "Droit pénal camerounais"},
    {"id": 4, "name": "Droit Administratif", "description": "Droit administratif"},
    {"id": 5, "name": "Droit du Travail", "description": "Code du travail"},
]

SAMPLE_LAWS = [
    {
        "reference": "LOI-2024-001",
        "title": "Code Civil Camerounais - Livre 1",
        "type": "Loi",
        "category_id": 1,
        "language": "fr",
        "status": "published",
        "publication_date": date(2024, 1, 15),
        "content": """
LOI N° 2024-001 DU 15 JANVIER 2024
RELATIVE AU CODE CIVIL CAMEROUNAIS

TITRE PREMIER - DES PERSONNES

Article 1er. Tout individu possède la personnalité juridique dès sa naissance vivante et viable.

Article 2. La majorité est fixée à vingt et un ans accomplis. À cet âge, toute personne est capable d'exercer tous les droits dont elle jouit.

Article 3. Le domicile d'une personne est au lieu où elle a son principal établissement. Le domicile une fois établi subsiste tant qu'il n'en a pas été créé un autre.

TITRE II - DES BIENS

Article 4. Les biens sont meubles ou immeubles. Cette distinction s'applique à tous les biens corporels ou incorporels.

Article 5. Sont immeubles par nature les fonds de terre et les bâtiments. Sont immeubles par destination les objets mobiliers que le propriétaire a attachés au fonds à perpétuelle demeure.

Article 6. Tous les autres biens sont meubles. Sont réputés meubles par anticipation les récoltes pendantes et les fruits non encore cueillis.

TITRE III - DES OBLIGATIONS

Article 7. Les contrats légalement formés tiennent lieu de loi à ceux qui les ont faits. Ils ne peuvent être révoqués que de leur consentement mutuel ou pour les causes que la loi autorise.

Article 8. Les conventions doivent être exécutées de bonne foi. Elles obligent non seulement à ce qui y est exprimé, mais encore à toutes les suites que l'équité, l'usage ou la loi donnent à l'obligation.

Article 9. La responsabilité civile engage celui qui cause un dommage à autrui à le réparer. Chacun est responsable du dommage qu'il a causé non seulement par son fait, mais encore par sa négligence ou par son imprudence.
""",
        "articles": [
            {"number": "1", "title": "Personnalité juridique", "content": "Tout individu possède la personnalité juridique dès sa naissance vivante et viable.", "order": 1},
            {"number": "2", "title": "Majorité", "content": "La majorité est fixée à vingt et un ans accomplis. À cet âge, toute personne est capable d'exercer tous les droits dont elle jouit.", "order": 2},
            {"number": "3", "title": "Domicile", "content": "Le domicile d'une personne est au lieu où elle a son principal établissement. Le domicile une fois établi subsiste tant qu'il n'en a pas été créé un autre.", "order": 3},
            {"number": "4", "title": "Classification des biens", "content": "Les biens sont meubles ou immeubles. Cette distinction s'applique à tous les biens corporels ou incorporels.", "order": 4},
            {"number": "5", "title": "Immeubles", "content": "Sont immeubles par nature les fonds de terre et les bâtiments. Sont immeubles par destination les objets mobiliers que le propriétaire a attachés au fonds à perpétuelle demeure.", "order": 5},
            {"number": "6", "title": "Meubles", "content": "Tous les autres biens sont meubles. Sont réputés meubles par anticipation les récoltes pendantes et les fruits non encore cueillis.", "order": 6},
            {"number": "7", "title": "Force obligatoire des contrats", "content": "Les contrats légalement formés tiennent lieu de loi à ceux qui les ont faits. Ils ne peuvent être révoqués que de leur consentement mutuel ou pour les causes que la loi autorise.", "order": 7},
            {"number": "8", "title": "Bonne foi", "content": "Les conventions doivent être exécutées de bonne foi. Elles obligent non seulement à ce qui y est exprimé, mais encore à toutes les suites que l'équité, l'usage ou la loi donnent à l'obligation.", "order": 8},
            {"number": "9", "title": "Responsabilité civile", "content": "La responsabilité civile engage celui qui cause un dommage à autrui à le réparer. Chacun est responsable du dommage qu'il a causé non seulement par son fait, mais encore par sa négligence ou par son imprudence.", "order": 9},
        ]
    },
    {
        "reference": "LOI-OHADA-2023-015",
        "title": "Acte Uniforme OHADA - Droit Commercial",
        "type": "Acte Uniforme",
        "category_id": 2,
        "language": "fr",
        "status": "published",
        "publication_date": date(2023, 6, 20),
        "content": """
ACTE UNIFORME OHADA N° 2023-015 DU 20 JUIN 2023
RELATIF AU DROIT COMMERCIAL GÉNÉRAL

LIVRE PREMIER - DU COMMERÇANT

Article 1er. Est commerçant celui qui fait de l'accomplissement d'actes de commerce par nature sa profession habituelle.

Article 2. Tout commerçant doit procéder à l'immatriculation au registre du commerce et du crédit mobilier dans le mois du commencement de son activité.

Article 3. Les sociétés commerciales jouissent de la personnalité juridique à compter de leur immatriculation au registre du commerce et du crédit mobilier.

LIVRE II - DES ACTES DE COMMERCE

Article 4. Sont réputés actes de commerce par nature : l'achat de biens meubles ou immeubles en vue de leur revente, les opérations de banque, les opérations de bourse.

Article 5. Les actes de commerce par accessoire sont ceux qui, accomplis par un commerçant pour les besoins de son commerce, constituent des actes de commerce.

LIVRE III - DES SOCIÉTÉS

Article 6. La société est créée par deux ou plusieurs personnes qui conviennent par un contrat d'affecter à une entreprise commune des biens ou leur industrie en vue de partager le bénéfice ou de profiter de l'économie qui pourra en résulter.

Article 7. Les formes de sociétés commerciales reconnues sont : la société en nom collectif, la société en commandite simple, la société à responsabilité limitée, et la société anonyme.
""",
        "articles": [
            {"number": "1", "title": "Définition du commerçant", "content": "Est commerçant celui qui fait de l'accomplissement d'actes de commerce par nature sa profession habituelle.", "order": 1},
            {"number": "2", "title": "Immatriculation", "content": "Tout commerçant doit procéder à l'immatriculation au registre du commerce et du crédit mobilier dans le mois du commencement de son activité.", "order": 2},
            {"number": "3", "title": "Personnalité juridique des sociétés", "content": "Les sociétés commerciales jouissent de la personnalité juridique à compter de leur immatriculation au registre du commerce et du crédit mobilier.", "order": 3},
            {"number": "4", "title": "Actes de commerce par nature", "content": "Sont réputés actes de commerce par nature : l'achat de biens meubles ou immeubles en vue de leur revente, les opérations de banque, les opérations de bourse.", "order": 4},
            {"number": "5", "title": "Actes de commerce par accessoire", "content": "Les actes de commerce par accessoire sont ceux qui, accomplis par un commerçant pour les besoins de son commerce, constituent des actes de commerce.", "order": 5},
            {"number": "6", "title": "Définition de la société", "content": "La société est créée par deux ou plusieurs personnes qui conviennent par un contrat d'affecter à une entreprise commune des biens ou leur industrie en vue de partager le bénéfice ou de profiter de l'économie qui pourra en résulter.", "order": 6},
            {"number": "7", "title": "Formes de sociétés", "content": "Les formes de sociétés commerciales reconnues sont : la société en nom collectif, la société en commandite simple, la société à responsabilité limitée, et la société anonyme.", "order": 7},
        ]
    },
    {
        "reference": "LOI-2016-007",
        "title": "Code Pénal Camerounais",
        "type": "Loi",
        "category_id": 3,
        "language": "fr",
        "status": "published",
        "publication_date": date(2016, 7, 12),
        "content": """
LOI N° 2016-007 DU 12 JUILLET 2016
PORTANT CODE PÉNAL

LIVRE PREMIER - DISPOSITIONS GÉNÉRALES

Article 1er. Nul ne peut être puni qu'en vertu d'une loi antérieure aux faits qui lui sont reprochés.

Article 2. Les infractions sont classées en trois catégories : les contraventions, les délits et les crimes.

Article 3. La tentative d'un crime est toujours punissable. La tentative d'un délit n'est punissable que si la loi le prévoit expressément.

LIVRE II - DES INFRACTIONS

Article 4. Le vol est la soustraction frauduleuse de la chose d'autrui. Il est puni d'un emprisonnement de six mois à cinq ans et d'une amende.

Article 5. L'escroquerie est le fait d'obtenir un bien ou un avantage par des manœuvres frauduleuses. Elle est punie d'un emprisonnement d'un à cinq ans.

Article 6. Le meurtre est le fait de donner volontairement la mort à autrui. Il est puni de la réclusion criminelle à perpétuité.
""",
        "articles": [
            {"number": "1", "title": "Principe de légalité", "content": "Nul ne peut être puni qu'en vertu d'une loi antérieure aux faits qui lui sont reprochés.", "order": 1},
            {"number": "2", "title": "Classification des infractions", "content": "Les infractions sont classées en trois catégories : les contraventions, les délits et les crimes.", "order": 2},
            {"number": "3", "title": "Tentative", "content": "La tentative d'un crime est toujours punissable. La tentative d'un délit n'est punissable que si la loi le prévoit expressément.", "order": 3},
            {"number": "4", "title": "Vol", "content": "Le vol est la soustraction frauduleuse de la chose d'autrui. Il est puni d'un emprisonnement de six mois à cinq ans et d'une amende.", "order": 4},
            {"number": "5", "title": "Escroquerie", "content": "L'escroquerie est le fait d'obtenir un bien ou un avantage par des manœuvres frauduleuses. Elle est punie d'un emprisonnement d'un à cinq ans.", "order": 5},
            {"number": "6", "title": "Meurtre", "content": "Le meurtre est le fait de donner volontairement la mort à autrui. Il est puni de la réclusion criminelle à perpétuité.", "order": 6},
        ]
    }
]


async def seed_database():
    """Seed the database with sample data."""
    print("=" * 80)
    print("SEEDING DATABASE WITH SAMPLE LAWS")
    print("=" * 80)

    # Create tables
    print("\n[1/4] Creating database tables...")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("   Tables created successfully")

    async with AsyncSessionLocal() as session:
        # Create categories (skip if exist)
        print("\n[2/4] Creating categories...")
        from sqlalchemy import select
        result = await session.execute(select(Category))
        existing_categories = result.scalars().all()

        if not existing_categories:
            for cat_data in SAMPLE_CATEGORIES:
                category = Category(**cat_data)
                session.add(category)
            await session.commit()
            print(f"   Created {len(SAMPLE_CATEGORIES)} categories")
        else:
            print(f"   Skipped (already exist: {len(existing_categories)} categories)")

        # Create laws with articles
        print("\n[3/4] Creating laws and articles...")
        for law_data in SAMPLE_LAWS:
            articles_data = law_data.pop("articles")
            law = Law(**law_data)
            session.add(law)
            await session.flush()  # Get law ID

            # Create articles
            for art_data in articles_data:
                article = Article(law_id=law.id, **art_data)
                session.add(article)

            print(f"   Created law: {law.reference} with {len(articles_data)} articles")

        await session.commit()
        print(f"   Total: {len(SAMPLE_LAWS)} laws created")

        # Verify
        print("\n[4/4] Verifying data...")
        from sqlalchemy import select, func

        result = await session.execute(select(func.count(Law.id)))
        law_count = result.scalar()

        result = await session.execute(select(func.count(Article.id)))
        article_count = result.scalar()

        result = await session.execute(select(func.count(Category.id)))
        category_count = result.scalar()

        print(f"   Categories: {category_count}")
        print(f"   Laws: {law_count}")
        print(f"   Articles: {article_count}")

    print("\n" + "=" * 80)
    print("[SUCCESS] Database seeded successfully!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(seed_database())
