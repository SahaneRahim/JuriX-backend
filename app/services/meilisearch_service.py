"""
Meilisearch indexing service for automatic article indexing.
Provides methods to add/remove articles from Meilisearch index.
"""
import meilisearch
from typing import List, Optional
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)


class MeilisearchService:
    """
    Service for managing Meilisearch article indexing.
    
    Provides automatic indexing when articles are created and
    automatic removal when laws are deleted.
    """
    
    ARTICLES_INDEX = "articles"
    LAWS_INDEX = "laws"
    
    _client: Optional[meilisearch.Client] = None
    
    @classmethod
    def get_client(cls) -> meilisearch.Client:
        """Get or create Meilisearch client singleton."""
        if cls._client is None:
            cls._client = meilisearch.Client(
                settings.MEILISEARCH_URL, 
                settings.MEILISEARCH_KEY
            )
            # Ensure indexes exist
            cls._ensure_indexes()
        return cls._client
    
    @classmethod
    def _ensure_indexes(cls):
        """Ensure required indexes exist with proper settings."""
        client = cls._client
        
        # Create articles index if not exists
        try:
            client.create_index(cls.ARTICLES_INDEX, {'primaryKey': 'id'})
            logger.info(f"✅ Created '{cls.ARTICLES_INDEX}' index")
        except Exception:
            pass  # Index already exists
        
        # Create laws index if not exists
        try:
            client.create_index(cls.LAWS_INDEX, {'primaryKey': 'id'})
            logger.info(f"✅ Created '{cls.LAWS_INDEX}' index")
        except Exception:
            pass  # Index already exists
        
        # Configure articles index
        try:
            articles_index = client.get_index(cls.ARTICLES_INDEX)
            articles_index.update_settings({
                'searchableAttributes': [
                    'content', 'number', 'title', 'section', 
                    'law_title', 'law_reference', 'category'
                ],
                'filterableAttributes': ['law_id', 'category'],
                'sortableAttributes': ['number'],
            })
        except Exception as e:
            logger.warning(f"⚠️ Could not configure articles index: {e}")
    
    @classmethod
    def index_articles(cls, articles: List[dict], law_id: int, 
                       law_title: str, law_reference: str, 
                       category_name: Optional[str] = None):
        """
        Index articles in Meilisearch.
        
        Args:
            articles: List of article dicts with id, number, title, section, content
            law_id: Parent law ID
            law_title: Parent law title
            law_reference: Parent law reference
            category_name: Optional category name
        """
        client = cls.get_client()
        
        if not articles:
            logger.warning("⚠️ No articles to index")
            return
        
        # Prepare documents for indexing
        documents = []
        for article in articles:
            documents.append({
                'id': article.get('id'),
                'number': article.get('number', ''),
                'title': article.get('title', ''),
                'section': article.get('section', ''),
                'content': article.get('content', ''),
                'law_id': law_id,
                'law_title': law_title,
                'law_reference': law_reference,
                'category': category_name or 'Non catégorisé'
            })
        
        # Index in Meilisearch
        try:
            articles_index = client.get_index(cls.ARTICLES_INDEX)
            task = articles_index.add_documents(documents)
            logger.info(
                f"📤 Indexed {len(documents)} articles for law {law_id} "
                f"(task: {task.task_uid})"
            )
            return task
        except Exception as e:
            logger.error(f"❌ Failed to index articles for law {law_id}: {e}")
            raise
    
    @classmethod
    def remove_law_articles(cls, law_id: int):
        """
        Remove all articles for a law from Meilisearch.
        
        Args:
            law_id: ID of the law whose articles should be removed
        """
        client = cls.get_client()
        
        try:
            articles_index = client.get_index(cls.ARTICLES_INDEX)
            
            # Get all articles for this law
            search_result = articles_index.search('', {
                'filter': f'law_id = {law_id}',
                'limit': 1000,
                'attributesToRetrieve': ['id']
            })
            
            article_ids = [hit['id'] for hit in search_result.get('hits', [])]
            
            if article_ids:
                task = articles_index.delete_documents(article_ids)
                logger.info(
                    f"🗑️ Removed {len(article_ids)} articles for law {law_id} "
                    f"(task: {task.task_uid})"
                )
                return task
            else:
                logger.info(f"ℹ️ No articles found for law {law_id} in Meilisearch")
                return None
                
        except Exception as e:
            logger.error(f"❌ Failed to remove articles for law {law_id}: {e}")
            raise
    
    @classmethod
    def index_law(cls, law_id: int, reference: str, title: str, 
                  content: str, law_type: str, language: Optional[str],
                  status: str, category_id: Optional[int],
                  category_name: Optional[str], publication_year: Optional[int]):
        """
        Index a law in Meilisearch.
        
        Args:
            law_id: Law ID
            reference: Law reference
            title: Law title  
            content: Law content (full text)
            law_type: Law type
            language: Language code
            status: Law status
            category_id: Category ID
            category_name: Category name
            publication_year: Publication year
        """
        client = cls.get_client()
        
        try:
            laws_index = client.get_index(cls.LAWS_INDEX)
            task = laws_index.add_documents([{
                'id': law_id,
                'reference': reference,
                'title': title,
                'content': content[:50000] if content else '',  # Meilisearch limit
                'type': law_type,
                'language': language,
                'status': status,
                'category_id': category_id,
                'category_name': category_name,
                'publication_year': publication_year
            }])
            logger.info(f"📤 Indexed law {law_id} in Meilisearch (task: {task.task_uid})")
            return task
        except Exception as e:
            logger.error(f"❌ Failed to index law {law_id}: {e}")
            raise
    
    @classmethod
    def remove_law(cls, law_id: int):
        """
        Remove a law and its articles from Meilisearch.
        
        Args:
            law_id: ID of the law to remove
        """
        client = cls.get_client()
        
        # Remove articles first
        cls.remove_law_articles(law_id)
        
        # Remove law
        try:
            laws_index = client.get_index(cls.LAWS_INDEX)
            task = laws_index.delete_document(law_id)
            logger.info(f"🗑️ Removed law {law_id} from Meilisearch (task: {task.task_uid})")
            return task
        except Exception as e:
            logger.error(f"❌ Failed to remove law {law_id}: {e}")
            raise
