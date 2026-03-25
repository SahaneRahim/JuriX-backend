"""
Re-index laws in Meilisearch after database reset.
"""
import sys
sys.path.insert(0, '.')

import meilisearch
from sqlalchemy import create_engine, text
from app.core.config import settings

print("🔍 RE-INDEXING MEILISEARCH")
print("=" * 60)

# Connect to Meilisearch
client = meilisearch.Client(settings.MEILISEARCH_URL, settings.MEILISEARCH_KEY)

# Check current index status
print("\n📋 Current Meilisearch Status:")
try:
    laws_index = client.get_index('laws')
    stats = laws_index.get_stats()
    print(f"  laws index: {stats.number_of_documents} documents")
except Exception as e:
    print(f"  laws index: ❌ Does not exist or error: {e}")

# Get data from PostgreSQL
print("\n📥 Loading laws from PostgreSQL...")
sync_url = settings.DATABASE_URL.replace('+asyncpg', '')
engine = create_engine(sync_url)

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT l.id, l.reference, l.title, l.type, l.content, l.status,
               c.name as category_name
        FROM laws l
        LEFT JOIN categories c ON l.category_id = c.id
        WHERE l.status = 'published'
    """))
    
    laws = []
    for row in result:
        laws.append({
            'id': row[0],
            'reference': row[1],
            'title': row[2],
            'type': row[3],
            'content': row[4][:10000] if row[4] else '',  # Limit content size
            'status': row[5],
            'category': row[6] or 'Non catégorisé'
        })

print(f"  Found {len(laws)} published laws")

if not laws:
    print("❌ No laws to index!")
    exit(1)

# Create/update laws index
print("\n🔄 Creating/updating Meilisearch index...")
try:
    client.create_index('laws', {'primaryKey': 'id'})
    print("  ✅ Created 'laws' index")
except Exception as e:
    print(f"  ℹ️  Index already exists: {e}")

laws_index = client.get_index('laws')

# Configure searchable and filterable attributes
print("\n⚙️  Configuring index settings...")
laws_index.update_settings({
    'searchableAttributes': ['title', 'reference', 'content', 'category'],
    'filterableAttributes': ['status', 'type', 'category'],
    'sortableAttributes': ['title', 'reference'],
})

# Add documents
print(f"\n📤 Indexing {len(laws)} laws...")
task = laws_index.add_documents(laws)
print(f"  Task UID: {task.task_uid}")

# Wait for indexing to complete
import time
max_wait = 30
for i in range(max_wait):
    task_info = client.get_task(task.task_uid)
    if task_info.status in ['succeeded', 'failed']:
        break
    print(f"  ⏳ Waiting... ({i+1}/{max_wait})")
    time.sleep(1)

if task_info.status == 'succeeded':
    print(f"  ✅ Indexing succeeded!")
else:
    print(f"  ❌ Indexing failed: {task_info.error}")

# Verify
stats = laws_index.get_stats()
print(f"\n📊 Final stats: {stats.number_of_documents} documents indexed")

# Test search
print("\n🔍 Testing search for 'constitution'...")
results = laws_index.search('constitution')
print(f"  Found {len(results['hits'])} results")
if results['hits']:
    print(f"  First hit: {results['hits'][0].get('title', 'N/A')}")

print("\n" + "=" * 60)
print("✅ MEILISEARCH RE-INDEXING COMPLETE!")
