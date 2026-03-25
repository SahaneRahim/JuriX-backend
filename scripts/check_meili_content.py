"""
Check what's in Meilisearch and if keywords exist.
"""
import sys
sys.path.insert(0, '.')

import meilisearch
from app.core.config import settings

client = meilisearch.Client(settings.MEILISEARCH_URL, settings.MEILISEARCH_KEY)

# Get law content from Meilisearch
laws_index = client.get_index('laws')
docs = laws_index.get_documents({'limit': 1})

for doc in docs.results:
    doc_dict = dict(doc)
    content = str(doc_dict.get('content', ''))
    title = doc_dict.get('title', 'N/A')
    
    print(f"Title: {title}")
    print(f"Content length: {len(content)} chars")
    print()
    
    # Check for specific keywords
    keywords = ['constitution', 'préambule', 'preambule', 'cameroun', 'peuple', 'article']
    print("Keywords in content:")
    for kw in keywords:
        if kw.lower() in content.lower():
            print(f"  ✅ '{kw}' FOUND")
        else:
            print(f"  ❌ '{kw}' NOT FOUND")
    
    print()
    print("First 500 chars of content:")
    print("-" * 40)
    print(content[:500])
