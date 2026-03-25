"""
Script pour régénérer les embeddings avec respect des limites de quota.
Attend automatiquement quand le quota est atteint.
"""
import sys
import time
sys.path.insert(0, '.')

from sqlalchemy import create_engine, text
from app.core.config import settings
from app.services.embedding_service import EmbeddingService, EmbeddingServiceError

# Configuration
LAW_ID = 3  # Code Pénal
BATCH_SIZE = 10  # Petit batch pour éviter timeout
DELAY_BETWEEN_CHUNKS = 2  # Secondes entre chaque chunk
QUOTA_WAIT_TIME = 65  # Secondes à attendre si quota dépassé

print(f"🔄 REGENERATION EMBEDDINGS - Law ID {LAW_ID}")
print("=" * 60)

# Connect to database
sync_url = settings.DATABASE_URL.replace('+asyncpg', '')
engine = create_engine(sync_url)

# Initialize embedding service
embedding_service = EmbeddingService(use_cache=True)

with engine.connect() as conn:
    # Get law info
    result = conn.execute(text("""
        SELECT title, reference FROM laws WHERE id = :law_id
    """), {"law_id": LAW_ID})
    law = result.fetchone()
    if not law:
        print(f"❌ Law ID {LAW_ID} not found!")
        sys.exit(1)
    
    print(f"📚 Law: {law[0]}")
    
    # Get articles without embeddings or all articles
    result = conn.execute(text("""
        SELECT id, number, content 
        FROM articles 
        WHERE law_id = :law_id
        ORDER BY id
    """), {"law_id": LAW_ID})
    
    articles = result.fetchall()
    print(f"📄 Total articles: {len(articles)}")
    
    if not articles:
        print("✅ No articles found!")
        sys.exit(0)
    
    # Check which already have embeddings
    result = conn.execute(text("""
        SELECT COUNT(*) FROM articles 
        WHERE law_id = :law_id AND embedding IS NOT NULL
    """), {"law_id": LAW_ID})
    existing_count = result.fetchone()[0]
    print(f"📊 Already have embeddings: {existing_count}")
    
    # Get articles without embeddings
    result = conn.execute(text("""
        SELECT id, number, content 
        FROM articles 
        WHERE law_id = :law_id AND embedding IS NULL
        ORDER BY id
    """), {"law_id": LAW_ID})
    
    articles_to_process = result.fetchall()
    print(f"📄 Articles to process: {len(articles_to_process)}")
    
    if not articles_to_process:
        print("✅ All articles already have embeddings!")
        sys.exit(0)
    
    # Process in small batches with rate limiting
    total_processed = 0
    chunk_num = 0
    total_chunks = (len(articles_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"\n🔢 Processing {len(articles_to_process)} articles in {total_chunks} chunks")
    print(f"   Batch size: {BATCH_SIZE}, Delay: {DELAY_BETWEEN_CHUNKS}s between chunks")
    print("")
    
    for i in range(0, len(articles_to_process), BATCH_SIZE):
        chunk = articles_to_process[i:i + BATCH_SIZE]
        chunk_num += 1
        
        article_ids = [a[0] for a in chunk]
        article_texts = [a[2] or f"Article {a[1]}" for a in chunk]
        
        print(f"📝 Chunk {chunk_num}/{total_chunks}: {len(chunk)} articles (IDs {article_ids[0]}-{article_ids[-1]})...")
        
        success = False
        retries = 0
        max_retries = 5
        
        while not success and retries < max_retries:
            try:
                # Generate embeddings one at a time to avoid batch issues
                embeddings = []
                for idx, text_content in enumerate(article_texts):
                    emb = embedding_service.generate_embedding(text_content)
                    embeddings.append(emb)
                    if idx > 0 and idx % 5 == 0:
                        time.sleep(0.5)  # Small delay within chunk
                
                # Save to database
                for article_id, embedding in zip(article_ids, embeddings):
                    emb_list = embedding.tolist()
                    conn.execute(text("""
                        UPDATE articles 
                        SET embedding = :embedding 
                        WHERE id = :article_id
                    """), {"embedding": emb_list, "article_id": article_id})
                
                conn.commit()
                total_processed += len(chunk)
                print(f"   ✅ Saved {len(chunk)} embeddings (total: {total_processed}/{len(articles_to_process)})")
                success = True
                
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "quota" in error_str.lower():
                    retries += 1
                    wait_time = QUOTA_WAIT_TIME * retries
                    print(f"   ⏳ Quota exceeded! Waiting {wait_time}s before retry ({retries}/{max_retries})...")
                    time.sleep(wait_time)
                else:
                    print(f"   ❌ Error: {e}")
                    retries += 1
                    if retries < max_retries:
                        print(f"   🔄 Retrying in 10s ({retries}/{max_retries})...")
                        time.sleep(10)
        
        if not success:
            print(f"❌ Failed to process chunk {chunk_num} after {max_retries} attempts")
            print(f"   Processed {total_processed}/{len(articles_to_process)} articles before failure")
            sys.exit(1)
        
        # Wait between chunks to avoid rate limiting
        if i + BATCH_SIZE < len(articles_to_process):
            print(f"   ⏳ Waiting {DELAY_BETWEEN_CHUNKS}s before next chunk...")
            time.sleep(DELAY_BETWEEN_CHUNKS)

print("\n" + "=" * 60)
print(f"🎉 REGENERATION COMPLETE! Processed {total_processed} articles")
print("=" * 60)
print("\nN'oubliez pas d'indexer dans Meilisearch:")
print("python scripts/index_articles_meili.py")
