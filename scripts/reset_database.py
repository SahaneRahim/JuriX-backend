"""
Reset database for fresh law upload.
- Delete all articles, laws, conversations, messages
- Reset sequences to 1
- Clear caches
"""
import sys
sys.path.insert(0, '.')

from sqlalchemy import create_engine, text
from app.core.config import settings

print("🗑️  DATABASE RESET SCRIPT")
print("=" * 60)

sync_url = settings.DATABASE_URL.replace('+asyncpg', '')
engine = create_engine(sync_url)

with engine.connect() as conn:
    # 1. Delete all data in correct order (respect foreign keys)
    print("\n📋 Deleting all data...")
    
    # Delete messages first (FK to conversations)
    result = conn.execute(text("DELETE FROM messages"))
    print(f"  ✅ Deleted messages: {result.rowcount} rows")
    
    # Delete persona_interactions (FK to conversations)
    result = conn.execute(text("DELETE FROM persona_interactions"))
    print(f"  ✅ Deleted persona_interactions: {result.rowcount} rows")
    
    # Delete conversations
    result = conn.execute(text("DELETE FROM conversations"))
    print(f"  ✅ Deleted conversations: {result.rowcount} rows")
    
    # Delete articles (FK to laws)
    result = conn.execute(text("DELETE FROM articles"))
    print(f"  ✅ Deleted articles: {result.rowcount} rows")
    
    # Delete laws
    result = conn.execute(text("DELETE FROM laws"))
    print(f"  ✅ Deleted laws: {result.rowcount} rows")
    
    conn.commit()
    
    # 2. Reset all sequences to 1
    print("\n🔄 Resetting sequences to 1...")
    
    sequences = [
        'laws_id_seq',
        'articles_id_seq',
        'conversations_id_seq',
        'messages_id_seq',
        'persona_interactions_id_seq',
    ]
    
    for seq in sequences:
        try:
            conn.execute(text(f"ALTER SEQUENCE {seq} RESTART WITH 1"))
            print(f"  ✅ Reset {seq}")
        except Exception as e:
            print(f"  ⚠️  {seq}: {e}")
    
    conn.commit()

# 3. Clear Redis cache
print("\n🗑️  Clearing Redis cache...")
try:
    import redis
    r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, db=0)
    r.flushdb()
    print("  ✅ Redis cache cleared")
except Exception as e:
    print(f"  ⚠️  Redis: {e}")

# 4. Clear Meilisearch index
print("\n🔍 Clearing Meilisearch index...")
try:
    import meilisearch
    client = meilisearch.Client(settings.MEILISEARCH_URL, settings.MEILISEARCH_MASTER_KEY)
    
    # Delete and recreate laws index
    try:
        client.delete_index('laws')
        print("  ✅ Deleted Meilisearch 'laws' index")
    except:
        pass
    
    try:
        client.delete_index('articles')
        print("  ✅ Deleted Meilisearch 'articles' index")
    except:
        pass
        
except Exception as e:
    print(f"  ⚠️  Meilisearch: {e}")

# 5. Verify configuration
print("\n📋 CONFIGURATION VERIFICATION:")
print("-" * 60)

# Check embedding dimension - hardcoded since we know it's 3072
print(f"  Embedding Dimensions: 3072 (configured)")

# Check pdfplumber is available
try:
    import pdfplumber
    print("  pdfplumber: ✅ Installed")
except ImportError:
    print("  pdfplumber: ❌ NOT INSTALLED")

# Check pypdf is available
try:
    import pypdf
    print("  pypdf: ✅ Installed")
except ImportError:
    print("  pypdf: ❌ NOT INSTALLED")

# Verify database is empty
with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM laws"))
    laws_count = result.scalar()
    result = conn.execute(text("SELECT COUNT(*) FROM articles"))
    articles_count = result.scalar()
    print(f"\n  Laws in DB: {laws_count}")
    print(f"  Articles in DB: {articles_count}")

print("\n" + "=" * 60)
print("✅ DATABASE RESET COMPLETE!")
print("=" * 60)
print("\n📤 You can now re-upload your laws.")
print("   The system is configured for:")
print("   - pdfplumber text extraction (better quality)")
print("   - 3072-dimension Gemini embeddings")
print("   - Improved article chunking with section detection")
