"""Final verification of Articles 4, 9, 10 after all fixes."""
import sys
sys.path.insert(0, '.')

from sqlalchemy import create_engine, text
from app.core.config import settings

sync_url = settings.DATABASE_URL.replace('+asyncpg', '')
engine = create_engine(sync_url)

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT number, section, content 
        FROM articles 
        WHERE law_id = 1 AND number IN ('4', '9', '10') 
        ORDER BY number::int
    """))
    
    with open('final_verification.txt', 'w', encoding='utf-8') as f:
        for row in result:
            number, section, content = row
            f.write(f"\n{'='*60}\n")
            f.write(f"ARTICLE {number}\n")
            f.write(f"Section: {section}\n")
            f.write(f"{'='*60}\n")
            f.write(f"{content}\n")
            f.write(f"\n[Length: {len(content)} chars]\n")

print("Saved to final_verification.txt")
