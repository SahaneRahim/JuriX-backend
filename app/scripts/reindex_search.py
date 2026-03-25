"""
Script pour réindexer tous les documents dans Meilisearch.

Usage:
    python -m app.scripts.reindex_search
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from app.core.config import settings
from app.core.database import AsyncSessionLocal
from app.models.law import Law

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def reindex_all():
    """Reindex all published laws in Meilisearch."""
    import meilisearch
    
    # Initialize Meilisearch client
    client = meilisearch.Client(settings.MEILISEARCH_URL, settings.MEILISEARCH_KEY)
    
    # Delete old index if exists
    try:
        client.delete_index("laws_index")
        logger.info("🗑️ Deleted old index")
    except Exception:
        logger.info("ℹ️ No existing index to delete")
    
    # Create new index
    client.create_index("laws_index", {"primaryKey": "id"})
    index = client.index("laws_index")
    
    # Configure index settings
    index.update_settings({
        "searchableAttributes": ["title", "reference", "content"],
        "filterableAttributes": [
            "id", "category_id", "type", "language",
            "status", "publication_year"
        ],
        "sortableAttributes": ["publication_year", "created_at_timestamp"],
        "rankingRules": [
            "words",
            "typo",
            "proximity",
            "attribute",
            "sort",
            "exactness"
        ],
        "typoTolerance": {
            "enabled": True,
            "minWordSizeForTypos": {
                "oneTypo": 4,
                "twoTypos": 8
            }
        }
    })
    
    logger.info("✅ Index created and configured")
    
    # Fetch all published laws
    async with AsyncSessionLocal() as db:
        stmt = (
            select(Law)
            .where(Law.status == "published")
            .options(joinedload(Law.category))
        )
        result = await db.execute(stmt)
        laws = result.scalars().all()
    
    logger.info(f"📚 Found {len(laws)} published laws")
    
    # Prepare documents
    documents = []
    for law in laws:
        doc = {
            "id": law.id,
            "reference": law.reference,
            "title": law.title,
            "content": law.content[:10000],  # Truncate to 10k chars
            "type": law.type,
            "language": law.language or "unknown",
            "status": law.status,
            "category_id": law.category_id,
            "category_name": law.category.name if law.category else None,
            "publication_year": (
                law.publication_date.year if law.publication_date else None
            ),
            "created_at_timestamp": int(law.created_at.timestamp())
        }
        documents.append(doc)
    
    # Index documents in batches
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        batch = documents[i:i + batch_size]
        index.add_documents(batch)
        logger.info(f"📦 Indexed batch {i//batch_size + 1}: {i + len(batch)}/{len(documents)}")
    
    logger.info(f"✅ Reindexing complete! {len(documents)} documents indexed")


if __name__ == "__main__":
    asyncio.run(reindex_all())
