"""
Test Meilisearch index and search directly.
"""
import sys
sys.path.insert(0, '.')

import meilisearch
from app.core.config import settings

print("🔍 MEILISEARCH TEST")
print("=" * 60)

client = meilisearch.Client(settings.MEILISEARCH_URL, settings.MEILISEARCH_KEY)

# Check laws index
print("\n📊 Laws Index:")
try:
    laws_index = client.get_index('laws')
    stats = laws_index.get_stats()
    print(f"  Documents: {stats.number_of_documents}")
    
    # Get documents using correct API
    docs = laws_index.get_documents({'limit': 3})
    print(f"\n📄 Documents:")
    for doc in docs.results:
        # Access as dict
        doc_dict = dict(doc)
        print(f"  ID={doc_dict.get('id')}")
        print(f"  Title='{doc_dict.get('title', 'N/A')}'")
        print(f"  Reference='{doc_dict.get('reference', 'N/A')}'")
        content = str(doc_dict.get('content', ''))[:200]
        print(f"  Content: {content}...")
        print()
    
    # Test searches
    print("🔍 Search Tests:")
    
    queries = ['constitution', 'preambule', 'article', 'cameroun', 'peuple']
    for q in queries:
        results = laws_index.search(q)
        hits = len(results.get('hits', []))
        print(f"  '{q}': {hits} hits")
    
except Exception as e:
    print(f"  ❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
