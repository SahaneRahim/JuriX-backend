"""
Index articles in Meilisearch for article-level search.
"""
import sys
sys.path.insert(0, '.')

import meilisearch
from sqlalchemy import create_engine, text
from app.core.config import settings
import time

print("📚 INDEXING ARTICLES IN MEILISEARCH")
print("=" * 60)

# Connect to Meilisearch
client = meilisearch.Client(settings.MEILISEARCH_URL, settings.MEILISEARCH_KEY)

# Get articles from PostgreSQL
print("\n📥 Loading articles from PostgreSQL...")
sync_url = settings.DATABASE_URL.replace('+asyncpg', '')
engine = create_engine(sync_url)

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT a.id, a.number, a.title, a.section, a.content, 
               l.id as law_id, l.title as law_title, l.reference as law_reference,
               c.name as category_name
        FROM articles a
        JOIN laws l ON a.law_id = l.id
        LEFT JOIN categories c ON l.category_id = c.id
        WHERE l.status = 'published'
    """))
    
    articles = []
    for row in result:
        articles.append({
            'id': row[0],
            'number': row[1],
            'title': row[2] or '',
            'section': row[3] or '',
            'content': row[4] or '',
            'law_id': row[5],
            'law_title': row[6],
            'law_reference': row[7],
            'category': row[8] or 'Non catégorisé'
        })

print(f"  Found {len(articles)} articles")

if not articles:
    print("❌ No articles to index!")
    exit(1)

# Create articles index
print("\n🔄 Creating/updating 'articles' index...")
try:
    client.create_index('articles', {'primaryKey': 'id'})
    print("  ✅ Created 'articles' index")
except Exception as e:
    print(f"  ℹ️  Index may already exist: {e}")

articles_index = client.get_index('articles')

# Configure searchable attributes
print("\n⚙️  Configuring index settings...")
try:
    articles_index.update_settings({
        'searchableAttributes': ['content', 'number', 'title', 'section', 'law_title', 'law_reference', 'category'],
        'filterableAttributes': ['law_id', 'category'],
        'sortableAttributes': ['number'],
    })
    print("  ✅ Settings updated")
except Exception as e:
    print(f"  ⚠️ Settings update: {e}")

# Add documents
print(f"\n📤 Indexing {len(articles)} articles...")
task = articles_index.add_documents(articles)
print(f"  Task UID: {task.task_uid}")

# Wait for indexing
for i in range(30):
    task_info = client.get_task(task.task_uid)
    if task_info.status in ['succeeded', 'failed']:
        break
    print(f"  ⏳ Waiting... {i+1}/30")
    time.sleep(1)

if task_info.status == 'succeeded':
    print(f"  ✅ Indexing succeeded!")
else:
    print(f"  ❌ Failed: {task_info.error}")

# Test searches
print("\n🔍 Testing article searches:")
queries = ['préambule', 'constitution', 'article 4', 'peuple camerounais', 'droits']
for q in queries:
    results = articles_index.search(q)
    hits = len(results.get('hits', []))
    print(f"  '{q}': {hits} hits")
    if hits > 0:
        first = results['hits'][0]
        print(f"    -> Article {first.get('number')}: {first.get('content', '')[:60]}...")

print("\n" + "=" * 60)
print("✅ ARTICLES INDEXING COMPLETE!")
