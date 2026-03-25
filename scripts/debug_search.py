"""
Debug search - test Meilisearch and search service directly.
"""
import sys
sys.path.insert(0, '.')

import meilisearch
from sqlalchemy import create_engine, text
from app.core.config import settings

print("🔍 DEBUG SEARCH")
print("=" * 60)

# 1. Check Meilisearch directly
print("\n1️⃣ Meilisearch Direct Test:")
client = meilisearch.Client(settings.MEILISEARCH_URL, settings.MEILISEARCH_KEY)

try:
    laws_index = client.get_index('laws')
    stats = laws_index.get_stats()
    print(f"  Documents in 'laws' index: {stats.number_of_documents}")
    
    # Test search
    results = laws_index.search('constitution')
    print(f"  Search 'constitution': {len(results['hits'])} hits")
    if results['hits']:
        print(f"    First hit: {results['hits'][0].get('title', 'N/A')}")
        
    results = laws_index.search('preambule')
    print(f"  Search 'preambule': {len(results['hits'])} hits")
    
    results = laws_index.search('La Constitution')
    print(f"  Search 'La Constitution': {len(results['hits'])} hits")
    
except Exception as e:
    print(f"  ❌ Meilisearch error: {e}")

# 2. Check PostgreSQL articles
print("\n2️⃣ PostgreSQL Articles Check:")
sync_url = settings.DATABASE_URL.replace('+asyncpg', '')
engine = create_engine(sync_url)

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT COUNT(*), 
               COUNT(embedding) as with_embedding
        FROM articles WHERE law_id = 1
    """))
    row = result.fetchone()
    print(f"  Articles: {row[0]}, With embeddings: {row[1]}")
    
    # Check for préambule content
    result = conn.execute(text("""
        SELECT number, LEFT(content, 100) as content_preview 
        FROM articles 
        WHERE law_id = 1 
        AND (LOWER(content) LIKE '%préambule%' OR LOWER(content) LIKE '%peuple camerounais%')
        LIMIT 3
    """))
    rows = result.fetchall()
    print(f"  Articles with 'préambule': {len(rows)}")
    for row in rows:
        print(f"    - Article {row[0]}: {row[1][:80]}...")
    
    # Show first 5 articles
    print("\n  First 5 articles:")
    result = conn.execute(text("""
        SELECT number, section, LEFT(content, 80) as preview 
        FROM articles 
        WHERE law_id = 1 
        ORDER BY "order" 
        LIMIT 5
    """))
    for row in result:
        print(f"    {row[0]}: [{row[1]}] {row[2]}...")

# 3. Check laws table
print("\n3️⃣ Laws Table Check:")
with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT id, title, reference, status, 
               LENGTH(content) as content_length
        FROM laws LIMIT 5
    """))
    for row in result:
        print(f"  ID={row[0]}, Title='{row[1]}', Ref='{row[2]}', Status='{row[3]}', Content={row[4]} chars")

print("\n" + "=" * 60)
