"""
Celery task for processing law documents.

Pipeline workflow:
1. Extract text (OCR if needed)
2. Detect language
3. Classify category
4. Split articles
5. Generate embeddings
6. Index in Meilisearch
7. Update database

Author: JuriX Development Team
Date: 2026-01-11
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict

from celery import Task
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.celery_app import celery_app
from app.core.database import AsyncSessionLocal
from app.models.law import Article, Law
from app.utils.text_chunker import extract_articles, ArticleExtractionError

logger = logging.getLogger(__name__)


class ProcessLawTask(Task):
    """Custom task class with error handling."""

    def on_failure(self, exc, task_id, args, kwargs, einfo):
        """Handle task failure."""
        logger.error(f"❌ Task {task_id} failed: {exc}")
        logger.error(f"Error info: {einfo}")


@celery_app.task(bind=True, base=ProcessLawTask, name="process_law")
def process_law(self, law_id: int, file_id: str = None) -> Dict[str, Any]:
    """
    Process a law document. If file_id is provided, extracts text first.

    Orchestrates the pipeline: load → ingest → analyze → index → update.
    Each step is delegated to focused helper functions.
    """
    assert isinstance(law_id, int) and law_id > 0, "law_id must be a positive integer"
    start_time = time.time()
    errors = []

    logger.info(f"🚀 Starting pipeline for law ID: {law_id} (file_id={file_id})")

    try:
        # 1. Load law from database
        self.update_state(
            state="PROGRESS", meta={"step": "loading", "progress": 5, "message": "Loading document"}
        )
        law = _load_law(law_id)
        if not law:
            raise ValueError(f"Law {law_id} not found")
        logger.info(f"📄 Loaded law: {law.title}")

        # 2. Ingest file content if file_id provided
        if file_id:
            self.update_state(
                state="PROGRESS",
                meta={"step": "extracting_file", "progress": 8, "message": "Processing uploaded file"},
            )
            law, file_errors = _ingest_file_content(law_id, law, file_id)
            errors.extend(file_errors)

        # 3. Validate extracted text
        self.update_state(
            state="PROGRESS",
            meta={"step": "extracting", "progress": 10, "message": "Analyzing text content"},
        )
        text = law.content
        if not text or len(text) < 50:
            raise ValueError("Insufficient content for processing")

        # 4. Extract title if pending
        extracted_title = None
        if law.title and law.title.startswith("PENDING-"):
            extracted_title = _extract_title_from_text(text)
            if extracted_title:
                logger.info(f"📝 Extracted title: {extracted_title}")

        # 5. Run analysis pipeline (language, category, articles, embeddings, search)
        result = _run_analysis_pipeline(self, law_id, law, text, extracted_title)
        result["errors"] = errors
        result["duration"] = round(time.time() - start_time, 2)

        logger.info(f"✅ Pipeline completed for law {law_id} in {result['duration']}s")
        return result

    except Exception as e:
        logger.error(f"❌ Pipeline failed for law {law_id}: {e}", exc_info=True)
        errors.append(str(e))
        return {
            "law_id": law_id,
            "status": "failed",
            "errors": errors,
            "duration": round(time.time() - start_time, 2),
        }


# ==================== PIPELINE HELPERS ====================


def _ingest_file_content(law_id: int, law, file_id: str):
    """
    Extract text content from uploaded file (PDF or DOCX).

    Tries LlamaParse first, then OCR, then pypdf as fallback.

    Args:
        law_id: Law database ID
        law: Law ORM object (will be mutated with new content)
        file_id: UUID of uploaded file

    Returns:
        Tuple of (updated_law, errors_list)
    """
    assert file_id, "file_id must not be empty"
    assert law is not None, "law object must be provided"

    errors = []
    from app.services.file_upload_service import get_upload_service
    upload_service = get_upload_service()

    # Locate file on disk
    file_path = None
    for ext in [".pdf", ".docx"]:
        p = upload_service.storage_path / f"{file_id}{ext}"
        if p.exists():
            file_path = p
            break

    if not file_path:
        logger.error(f"❌ File {file_id} not found in storage")
        errors.append(f"File {file_id} not found")
        return law, errors

    logger.info(f"📂 Found file: {file_path}")
    extracted_text = ""

    if file_path.suffix == ".pdf":
        extracted_text, pdf_errors = _extract_pdf_text(file_path)
        errors.extend(pdf_errors)
    elif file_path.suffix == ".docx":
        extracted_text, docx_errors = _extract_docx_text(file_path)
        errors.extend(docx_errors)

    if extracted_text and len(extracted_text) > 50:
        _update_law_content(law_id, extracted_text)
        law.content = extracted_text
        logger.info(f"✅ Extracted {len(extracted_text)} chars from file")
    else:
        logger.warning("⚠️ No text extracted or text too short")

    return law, errors


def _extract_pdf_text(file_path) -> tuple:
    """Extract text from PDF using LlamaParse → OCR → pypdf fallback chain."""
    import asyncio
    from app.core.config import settings

    try:
        from app.services.llama_parse_service import LlamaParseService, LlamaParseError
        logger.info(f"📄 Processing PDF with LlamaParse: {file_path.name}")
        llama_service = LlamaParseService()

        if llama_service.is_available():
            text = asyncio.run(llama_service.extract_text(file_path))
            logger.info(f"✅ LlamaParse Complete: {len(text)} chars (clean)")
            return text, []
        raise LlamaParseError("LlamaParse not configured")

    except Exception as e:
        logger.warning(f"LlamaParse failed ({e}), trying OCR fallback...")
        return _extract_pdf_ocr_fallback(file_path, settings)


def _extract_pdf_ocr_fallback(file_path, settings) -> tuple:
    """OCR then pypdf fallback for PDF text extraction."""
    import asyncio

    try:
        from app.services.ocr_service import OCRService
        ocr_service = OCRService(tesseract_path=settings.TESSERACT_PATH)
        ocr_result = asyncio.run(ocr_service.process_pdf(file_path))
        logger.info(f"✅ OCR Fallback: {len(ocr_result.text)} chars")
        return ocr_result.text, []
    except Exception as ocr_e:
        logger.error(f"OCR failed ({ocr_e}), reverting to pypdf fallback...")
        from pypdf import PdfReader
        reader = PdfReader(file_path)
        pages_text = []
        for page_num, page in enumerate(reader.pages, start=1):
            pages_text.append(f"<<PAGE:{page_num}>>\n{page.extract_text()}")
        text = _clean_extracted_text("\n".join(pages_text))
        logger.info(f"✅ pypdf Fallback: {len(text)} chars from {len(reader.pages)} pages")
        return text, []


def _extract_docx_text(file_path) -> tuple:
    """Extract text from DOCX file."""
    try:
        import docx
        doc = docx.Document(file_path)
        text = "\n".join([p.text for p in doc.paragraphs])
        return _clean_extracted_text(text), []
    except Exception as e:
        logger.error(f"DOCX extraction failed: {e}")
        return "", [f"DOCX extraction failed: {str(e)}"]


def _run_analysis_pipeline(task, law_id: int, law, text: str, extracted_title=None):
    """
    Run the full analysis pipeline: language → category → articles → embeddings → index.

    Args:
        task: Celery task (for progress updates)
        law_id: Law database ID
        law: Law ORM object
        text: Extracted text content
        extracted_title: Optional title extracted from text

    Returns:
        Result dict with analysis outcomes
    """
    assert text and len(text) >= 50, "Text must be at least 50 chars"
    assert isinstance(law_id, int) and law_id > 0, "law_id must be positive"

    # Language detection
    task.update_state(
        state="PROGRESS", meta={"step": "detecting", "progress": 30, "message": "Detecting language"}
    )
    language_result = _detect_language(text)
    logger.info(
        f"🌍 Language: {language_result['language']} "
        f"(confidence: {language_result['confidence']:.2%})"
    )

    # Category classification
    task.update_state(
        state="PROGRESS", meta={"step": "classifying", "progress": 45, "message": "Classifying category"}
    )
    category_result = _classify_category(text)
    logger.info(
        f"📂 Category: {category_result['category']} "
        f"(confidence: {category_result['confidence']:.2%})"
    )

    # Split and save articles
    task.update_state(
        state="PROGRESS", meta={"step": "splitting", "progress": 60, "message": "Splitting articles"}
    )
    articles_count = _split_and_save_articles(law_id, text)
    logger.info(f"📑 Articles extracted: {articles_count}")

    # Generate embeddings
    task.update_state(
        state="PROGRESS", meta={"step": "embeddings", "progress": 75, "message": "Generating embeddings"}
    )
    embeddings_count = _generate_article_embeddings(law_id)
    logger.info(f"🔢 Embeddings generated: {embeddings_count}")

    # Index in Meilisearch
    task.update_state(
        state="PROGRESS", meta={"step": "indexing", "progress": 85, "message": "Indexing in search"}
    )
    search_indexed = _index_in_search(law_id, law)
    logger.info(f"🔍 Search indexed: {search_indexed}")

    # Update metadata
    task.update_state(
        state="PROGRESS", meta={"step": "updating", "progress": 95, "message": "Updating database"}
    )
    _update_law_metadata(
        law_id,
        language=language_result["language"],
        language_confidence=language_result["confidence"],
        category=category_result["category"],
        category_confidence=category_result["confidence"],
        title=extracted_title,
    )

    return {
        "law_id": law_id,
        "status": "completed",
        "language": language_result["language"],
        "language_confidence": language_result["confidence"],
        "category": category_result["category"],
        "category_confidence": category_result["confidence"],
        "articles_count": articles_count,
    }


# ==================== HELPER FUNCTIONS ====================


def _extract_text_from_file(file_id: str) -> str:
    """
    DEPRECATED: Use _extract_text_with_ocr instead.

    This stub redirects to _extract_text_with_ocr which properly prioritises
    LlamaParse → OCR → pypdf fallback.

    Args:
        file_id: UUID of uploaded file

    Returns:
        Extracted text content
    """
    logger.warning(
        "_extract_text_from_file is deprecated. "
        "Delegating to _extract_text_with_ocr (LlamaParse-first)."
    )
    return _extract_text_with_ocr(file_id)


def _load_law(law_id: int) -> Law:
    """Load law from database (synchronous)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    from app.core.config import settings
    
    # Create synchronous engine for this operation
    sync_engine = create_engine(settings.DATABASE_URL.replace('+asyncpg', ''))
    Session = sessionmaker(bind=sync_engine)
    
    with Session() as session:
        law = session.query(Law).filter(Law.id == law_id).first()
        if law:
            # Eagerly load relationships
            _ = law.articles  # Trigger lazy load
        return law


def _detect_language(text: str) -> Dict[str, Any]:
    """
    Detect language using LanguageDetector.
    """
    try:
        from app.services.language_detector import LanguageDetector

        detector = LanguageDetector()
        result = detector.detect(text)
        return {"language": result["language"], "confidence": result["confidence"]}
    except Exception as e:
        logger.warning(f"⚠️ Language detection failed, using fallback: {e}")
        # Fallback to simple heuristic
        if any(word in text.lower() for word in ["le", "la", "les", "de", "et", "article"]):
            return {"language": "fr", "confidence": 0.75}
        else:
            return {"language": "en", "confidence": 0.75}


def _classify_category(text: str) -> Dict[str, Any]:
    """
    Classify document category using DocumentClassifier.
    """
    try:
        from app.services.document_classifier import DocumentClassifier

        classifier = DocumentClassifier()
        # classifier.classify returns a list of tuples: [(category_id, confidence, method), ...]
        results = classifier.classify(text)
        
        if results:
            cat_id, confidence, _ = results[0]
            # Map ID to name if needed, though ProcessLaw expects name in result dict?
            # Wait, process_law main function logs result['category'].
            # And _update_law_metadata uses result['category'].
            # In DocumentClassifier, verify what 'category' means.
            # get_category_name(cat_id) returns the string name.
            category_name = classifier.get_category_name(cat_id)
            return {"category": category_name, "confidence": confidence}
            
        return {"category": "Autre", "confidence": 0.0}

    except Exception as e:
        logger.warning(f"⚠️ Category classification failed, using fallback: {e}")
        # Fallback to simple heuristic
        if "fiscal" in text.lower() or "impôt" in text.lower():
            return {"category": "Droit Fiscal", "confidence": 0.70}
        elif "pénal" in text.lower() or "crime" in text.lower():
            return {"category": "Droit Pénal", "confidence": 0.70}
        else:
            return {"category": "Droit Civil", "confidence": 0.70}


def _split_and_save_articles(law_id: int, text: str) -> int:
    """
    Extract articles from law text and save them to database.
    
    Args:
        law_id: Law database ID
        text: Full law text content
        
    Returns:
        Number of articles extracted and saved
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    from app.core.config import settings
    
    logger.info(f"📑 Extracting articles from law {law_id}")
    
    try:
        # Extract articles using text_chunker (strict=False allows fewer than 3 articles)
        # min_article_length=1 ensures ALL articles are extracted, even very short ones
        extracted = extract_articles(text, strict=False, min_article_length=1)
        
        if not extracted:
            logger.warning(f"⚠️ No articles extracted from law {law_id}")
            return 0
        
        logger.info(f"📋 Found {len(extracted)} articles")
        
        # Create sync database session
        sync_engine = create_engine(settings.DATABASE_URL.replace('+asyncpg', ''))
        Session = sessionmaker(bind=sync_engine)
        
        with Session() as session:
            # Delete existing articles for this law (in case of re-processing)
            session.query(Article).filter(Article.law_id == law_id).delete()
            
            # Create Article objects
            for article_data in extracted:
                article = Article(
                    law_id=law_id,
                    number=str(article_data.get('number', '')),
                    title=article_data.get('title'),
                    section=article_data.get('section'),  # TITRE/CHAPITRE section header
                    content=article_data.get('content', ''),
                    order=article_data.get('position', 0),
                    page_number=article_data.get('page_number'),  # Save page number for navigation
                    # NOTE: embedding column is pgvector type in DB, leaving NULL for now
                )
                session.add(article)
                logger.debug(f"📄 Created Article {article.number}: section={article.section or 'None'} (page {article.page_number})")
            
            session.commit()
            logger.info(f"✅ Saved {len(extracted)} articles for law {law_id}")
        
        return len(extracted)
        
    except ArticleExtractionError as e:
        logger.warning(f"⚠️ Article extraction failed: {e}")
        return 0
    except Exception as e:
        logger.error(f"❌ Error saving articles: {e}", exc_info=True)
        return 0


def _generate_article_embeddings(law_id: int) -> int:
    """
    Generate embeddings for all articles/chunks of a law using Gemini API.
    
    This function processes ALL chunks created by the text extraction:
    - Legal basis (LEGAL_BASIS)
    - Preamble (PREAMBULE)
    - Articles (1, 2, 3...)
    - Paragraphs (PARA_1, PARA_2...) for non-article documents
    - Full text (FULL_TEXT) for unstructured docs
    
    Args:
        law_id: Law database ID
        
    Returns:
        Number of chunks with embeddings generated successfully
    """
    assert isinstance(law_id, int) and law_id > 0, "law_id must be a positive integer"
    assert law_id < 10_000_000, "law_id seems unreasonably large"

    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.core.config import settings
    from app.services.embedding_service import EmbeddingService, EmbeddingServiceError
    
    logger.info(f"🔢 Generating embeddings for law {law_id} chunks...")
    
    try:
        # Initialize Gemini embedding service
        embedding_service = EmbeddingService()
        
        # Connect to database
        engine = create_engine(settings.DATABASE_URL.replace('+asyncpg', ''), pool_pre_ping=True)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        try:
            # Get all articles/chunks for this law
            articles = session.query(Article).filter_by(law_id=law_id).all()
            
            if not articles:
                logger.warning(f"⚠️  No articles found for law {law_id}")
                return 0
            
            logger.info(f"📄 Found {len(articles)} chunks to process")
            
            # Prepare texts for batch embedding generation
            texts = [article.content for article in articles]
            article_ids = [article.id for article in articles]
            
            # Generate embeddings in batch for efficiency
            logger.info(f"🚀 Generating {len(texts)} embeddings via Gemini API...")
            embeddings = embedding_service.generate_batch_embeddings(
                texts=texts,
                normalize=True  # L2 normalization for cosine similarity
            )
            
            # Update articles with embeddings
            success_count = 0
            for article, embedding in zip(articles, embeddings):
                try:
                    # Convert numpy array to list for pgvector storage
                    article.embedding = embedding.tolist()
                    success_count += 1
                    logger.debug(
                        f"✅ Generated embedding for {article.number}: "
                        f"{len(embedding)} dims"
                    )
                except Exception as e:
                    logger.error(
                        f"❌ Failed to save embedding for article {article.number}: {e}"
                    )
            
            # Commit all updates
            session.commit()
            
            logger.info(
                f"✅ Successfully generated {success_count}/{len(articles)} embeddings "
                f"for law {law_id}"
            )
            
            return success_count
            
        finally:
            session.close()
    
    except EmbeddingServiceError as e:
        logger.error(f"❌ Embedding service error: {e}")
        return 0
    except Exception as e:
        logger.error(f"❌ Error generating embeddings: {e}", exc_info=True)
        return 0


def _index_in_search(law_id: int, law: Law) -> bool:
    """
    Index law AND its articles in Meilisearch for full-text search (synchronous).
    
    Returns:
        True if indexed successfully, False otherwise
    """
    try:
        import meilisearch
        from sqlalchemy import create_engine, text
        from app.core.config import settings
        
        # Initialize Meilisearch client (synchronous)
        client = meilisearch.Client(settings.MEILISEARCH_URL, settings.MEILISEARCH_KEY)
        
        # 1. Index the law in 'laws' index
        laws_index = client.index("laws")
        law_document = {
            "id": law.id,
            "law_id": law.id,  # Explicit law_id for RAG service
            "reference": law.reference,
            "title": law.title,
            "content": law.content[:50000] if law.content else "",  # Truncate for search
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
        laws_index.add_documents([law_document])
        logger.info(f"✅ Indexed law {law_id} in Meilisearch 'laws' index")
        
        # 2. Index articles in 'articles' index for article-level search
        try:
            articles_index = client.index("articles")
            
            # Ensure index exists with proper settings
            try:
                client.create_index('articles', {'primaryKey': 'id'})
            except Exception:
                pass  # Already exists
            
            # Fetch articles from database
            sync_url = settings.DATABASE_URL.replace('+asyncpg', '')
            engine = create_engine(sync_url)
            
            with engine.connect() as conn:
                result = conn.execute(text("""
                    SELECT id, number, title, section, content
                    FROM articles WHERE law_id = :law_id
                """), {"law_id": law_id})
                
                articles_docs = []
                for row in result:
                    articles_docs.append({
                        'id': row[0],
                        'number': row[1] or '',
                        'title': row[2] or '',
                        'section': row[3] or '',
                        'content': row[4] or '',
                        'law_id': law_id,
                        'law_title': law.title,
                        'law_reference': law.reference,
                        'category': law.category.name if law.category else 'Non catégorisé'
                    })
                
                if articles_docs:
                    articles_index.add_documents(articles_docs)
                    logger.info(f"✅ Indexed {len(articles_docs)} articles for law {law_id} in Meilisearch 'articles' index")
                    
        except Exception as articles_err:
            logger.warning(f"⚠️ Articles indexing failed: {articles_err}")
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Search indexing failed: {e}")
        return False


def delete_from_meilisearch(law_id: int) -> bool:
    """
    Delete a law AND its articles from Meilisearch indexes.
    
    Call this when a law is deleted from the database.
    
    Args:
        law_id: ID of the law to delete
        
    Returns:
        True if deleted successfully, False otherwise
    """
    try:
        import meilisearch
        from app.core.config import settings
        
        client = meilisearch.Client(settings.MEILISEARCH_URL, settings.MEILISEARCH_KEY)
        
        # 1. Delete articles first
        try:
            articles_index = client.index("articles")
            
            # Find all articles for this law
            search_result = articles_index.search('', {
                'filter': f'law_id = {law_id}',
                'limit': 1000,
                'attributesToRetrieve': ['id']
            })
            
            article_ids = [hit['id'] for hit in search_result.get('hits', [])]
            
            if article_ids:
                articles_index.delete_documents(article_ids)
                logger.info(f"🗑️ Deleted {len(article_ids)} articles for law {law_id} from Meilisearch")
        except Exception as e:
            logger.warning(f"⚠️ Articles deletion failed: {e}")
        
        # 2. Delete law
        laws_index = client.index("laws")
        laws_index.delete_document(law_id)
        logger.info(f"🗑️ Deleted law {law_id} from Meilisearch 'laws' index")
        
        return True
        
    except Exception as e:
        logger.warning(f"⚠️ Meilisearch delete failed for law {law_id}: {e}")
        return False


def _update_law_metadata(
    law_id: int,
    language: str,
    language_confidence: float,
    category: str,
    category_confidence: float,
    title: str = None,
) -> None:
    """Update law metadata in database (synchronous)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    from app.core.config import settings
    
    sync_engine = create_engine(settings.DATABASE_URL.replace('+asyncpg', ''))
    Session = sessionmaker(bind=sync_engine)
    
    with Session() as session:
        law = session.query(Law).filter(Law.id == law_id).first()
        if law:
            law.language = language  # Set the main language field
            law.detected_language = language
            law.language_confidence = language_confidence
            if title:  # Update title if provided
                law.title = title
                logger.info(f"📝 Updated title to: {title}")
            session.commit()
            logger.info(f"✅ Updated metadata for law {law_id}")


def _extract_title_from_text(text: str) -> str:
    """
    Extract document title from text content.
    
    Looks for title in first few lines, handling common patterns:
    - "LAW NO. 96/06 OF 18 JANUARY 1996..."
    - "LOI N° 96/06 DU 18 JANVIER 1996..."
    - "THE CONSTITUTION OF..."
    - "LA CONSTITUTION DU..."
    
    Returns:
        Extracted title or None if not found
    """
    if not text:
        return None
    
    # Get first 500 characters (usually contains title)
    header = text[:500].strip()
    
    # Split into lines
    lines = [line.strip() for line in header.split('\n') if line.strip()]
    
    if not lines:
        return None
    
    # Common patterns for legal documents
    title_patterns = [
        r'^(LAW\s+N[OoØ°]\.?\s*\d+[/-]\d+.*?)(?:\n|$)',  # English: LAW NO. 96/06...
        r'^(LOI\s+N[OoØ°]\.?\s*\d+[/-]\d+.*?)(?:\n|$)',  # French: LOI N° 96/06...
        r'^(THE\s+CONSTITUTION.*?)(?:\n|$)',  # English: THE CONSTITUTION...
        r'^(LA\s+CONSTITUTION.*?)(?:\n|$)',  # French: LA CONSTITUTION...
        r'^(CONSTITUTION.*?)(?:\n|$)',  # Generic: CONSTITUTION...
        r'^(DECREE\s+N[OoØ°]\.?\s*\d+.*?)(?:\n|$)',  # English: DECREE NO...
        r'^(DÉCRET\s+N[OoØ°]\.?\s*\d+.*?)(?:\n|$)',  # French: DÉCRET N°...
    ]
    
    import re
    
    # Try to match patterns in first 3 lines
    for line in lines[:3]:
        for pattern in title_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                title = match.group(1).strip()
                # Clean up common OCR artifacts
                title = re.sub(r'\s+', ' ', title)  # Normalize whitespace
                title = title[:200]  # Limit length
                return title
    
    # Fallback: Use first non-empty line if it looks like a title
    first_line = lines[0]
    if len(first_line) > 10 and len(first_line) < 200:
        # Check if it's all caps or starts with capital (likely a title)
        if first_line.isupper() or first_line[0].isupper():
            return first_line[:200]
    
    return None


def _update_law_content(law_id: int, content: str) -> None:
    """Update law content in database (synchronous)."""
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    
    from app.core.config import settings
    
    sync_engine = create_engine(settings.DATABASE_URL.replace('+asyncpg', ''))
    Session = sessionmaker(bind=sync_engine)
    
    with Session() as session:
        law = session.query(Law).filter(Law.id == law_id).first()
        if law:
            law.content = content
            session.commit()
            logger.info(f"✅ Updated content for law {law_id}")


# ==================== SYNCHRONOUS VERSION (Windows Fallback) ====================



def _extract_text_with_ocr(file_id: str) -> str:
    """
    Extract text from file using OCRService (Sync wrapper).
    Handles PDF (native/scanned) and DOCX.
    """
    try:
        from app.services.file_upload_service import get_upload_service
        from app.core.config import settings
        
        upload_service = get_upload_service()
        
        # Locate file
        file_path = None
        for ext in [".pdf", ".docx"]:
            p = upload_service.storage_path / f"{file_id}{ext}"
            if p.exists():
                file_path = p
                break
        
        if not file_path:
            logger.error(f"❌ File {file_id} not found")
            return ""

        if file_path.suffix.lower() == ".pdf":
            try:
                import asyncio
                from app.services.llama_parse_service import LlamaParseService, LlamaParseError
                
                logger.info(f"📄 Processing PDF with LlamaParse: {file_path.name}")
                llama_service = LlamaParseService()
                
                if llama_service.is_available():
                    extracted_text = asyncio.run(llama_service.extract_text(file_path))
                    logger.info(f"✅ LlamaParse Complete: {len(extracted_text)} chars (clean)")
                    return extracted_text
                else:
                    raise LlamaParseError("LlamaParse not configured")

            except Exception as e:
                logger.warning(f"LlamaParse failed ({e}), trying OCR fallback...")
                try:
                    from app.services.ocr_service import OCRService
                    ocr_service = OCRService(tesseract_path=settings.TESSERACT_PATH)
                    ocr_result = asyncio.run(ocr_service.process_pdf(file_path))
                    logger.info(f"✅ OCR Fallback: {len(ocr_result.text)} chars")
                    return ocr_result.text
                except Exception as ocr_e:
                    logger.error(f"OCR failed ({ocr_e}), using pypdf fallback")
                    from pypdf import PdfReader
                    reader = PdfReader(file_path)
                    pages_text = []
                    for page_num, page in enumerate(reader.pages, start=1):
                        page_text = f"<<PAGE:{page_num}>>\n{page.extract_text()}"
                        pages_text.append(page_text)
                    text = "\n".join(pages_text)
                    return _clean_extracted_text(text)

        elif file_path.suffix.lower() == ".docx":
            import docx
            doc = docx.Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
            return _clean_extracted_text(text)
            
        return ""

    except Exception as e:
        logger.error(f"Extraction error: {e}")
        return ""

def process_law_sync(law_id: int, file_id: str) -> Dict[str, Any]:
    """
    Synchronous version of process_law for Windows compatibility.
    """
    assert isinstance(law_id, int) and law_id > 0, "law_id must be a positive integer"
    assert isinstance(file_id, str) and len(file_id) > 0, "file_id must be a non-empty string"

    import asyncio

    logger.info(f"🔄 Starting synchronous processing for law {law_id}, file {file_id}")
    start_time = time.time()
    errors = []

    try:
        # Load law from database
        law = _load_law(law_id)
        if not law:
            error_msg = f"Law {law_id} not found in database"
            logger.error(f"❌ {error_msg}")
            return {"law_id": law_id, "status": "failed", "errors": [error_msg]}

        logger.info(f"📄 Processing law: {law.title} (ID={law_id})")

        # 1. Extract text from uploaded file
        logger.info(f"📄 Extracting text from file {file_id}")
        # USE NEW OCR FUNCTION HERE
        text = _extract_text_with_ocr(file_id)

        if text and len(text) >= 50:
            _update_law_content(law_id, text)
            logger.info(f"✅ Extracted {len(text)} characters")
        else:
            text = law.content
            if not text or len(text) < 50:
                raise ValueError("Insufficient content for processing")

        # 2. Detect language
        logger.info(f"🌍 Detecting language")
        language_result = _detect_language(text)
        logger.info(
            f"🌍 Language detected: {language_result['language']} "
            f"(confidence: {language_result['confidence']:.2%})"
        )

        # 3. Classify category
        logger.info(f"📂 Classifying category")
        category_result = _classify_category(text)
        logger.info(
            f"📂 Category classified: {category_result['category']} "
            f"(confidence: {category_result['confidence']:.2%})"
        )

        # 4. Extract and save articles
        print(f"DEBUG >>> About to extract articles for law {law_id}")
        logger.info(f"📑 Extracting articles")
        articles_count = _split_and_save_articles(law_id, text)
        print(f"DEBUG >>> Articles extracted: {articles_count}")
        logger.info(f"📑 Articles extracted: {articles_count}")

        # 5. Generate embeddings
        print(f"DEBUG >>> About to generate embeddings for law {law_id}")
        logger.info(f"🔢 Generating embeddings")
        try:
            embeddings_count = _generate_article_embeddings(law_id)
            print(f"DEBUG >>> Embeddings generated: {embeddings_count}")
        except Exception as emb_error:
            print(f"DEBUG >>> Embedding error: {emb_error}")
            embeddings_count = 0
        logger.info(f"🔢 Embeddings generated: {embeddings_count}")

        # 6. Index in search
        logger.info(f"🔍 Indexing in search")
        search_indexed = _index_in_search(law_id, law)
        logger.info(f"🔍 Search indexed: {search_indexed}")

        # 7. Update database with results
        logger.info(f"💾 Updating database metadata")
        _update_law_metadata(
            law_id,
            language=language_result["language"],
            language_confidence=language_result["confidence"],
            category=category_result["category"],
            category_confidence=category_result["confidence"],
        )

        # Calculate duration
        duration = time.time() - start_time

        # Final result
        result = {
            "law_id": law_id,
            "status": "completed",
            "language": language_result["language"],
            "language_confidence": language_result["confidence"],
            "category": category_result["category"],
            "category_confidence": category_result["confidence"],
            "articles_count": articles_count,
            "embeddings_generated": embeddings_count,
            "search_indexed": search_indexed,
            "duration": round(duration, 2),
            "errors": errors,
        }

        logger.info(f"✅ Synchronous processing completed for law {law_id} in {duration:.2f}s")
        return result

    except Exception as e:
        logger.error(f"❌ Synchronous processing failed for law {law_id}: {e}", exc_info=True)
        errors.append(str(e))

        return {
            "law_id": law_id,
            "status": "failed",
            "errors": errors,
            "duration": round(time.time() - start_time, 2),
        }


def _clean_extracted_text(text: str) -> str:
    """
    Clean extracted text with MINIMAL modifications.
    
    This "gentle" approach preserves the original document structure:
    - Line breaks are kept (important for legal document formatting)
    - Only obvious PDF artifacts are removed
    - Hyphenated words are merged only when appropriate
    
    Args:
        text: Raw extracted text from PDF/DOCX
        
    Returns:
        Cleaned text with preserved formatting
    """
    if not text:
        return ""

    import re
    
    # 1. Normalize line endings (Windows to Unix)
    text = text.replace("\r\n", "\n")
    
    # 2. Handle hyphenated words at end of line CAREFULLY
    # Only merge if next line starts with lowercase (indicates word continuation)
    text = re.sub(r'-\n([a-zàâçéèêëïîôùûüœæ])', r'\1', text)
    
    # 3. Handle French elision at end of line
    text = re.sub(r"([ldnsjcmtLDNSJCMT])'\n([A-Za-zÀ-ÿ])", r"\1'\2", text)
    text = re.sub(r"([ldnsjcmtLDNSJCMT])’\n([A-Za-zÀ-ÿ])", r"\1'\2", text)
    text = re.sub(r"(qu|Qu)'\n([A-Za-zÀ-ÿ])", r"\1'\2", text)
    
    # 4. Remove isolated page numbers (1-3 digits alone on a line)
    text = re.sub(r'^\s*\d{1,3}\s*$', '', text, flags=re.MULTILINE)
    
    # 5. Remove page markers but preserve surrounding structure
    text = re.sub(r'<<PAGE:\s*\d+\s*>>\n?', '', text)
    
    # 6. Collapse excessive blank lines (3+ consecutive to 2)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 7. Remove trailing spaces on each line
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    
    # 8. Limit leading spaces to 4 max
    text = re.sub(r'^[ \t]{5,}', '    ', text, flags=re.MULTILINE)
    
    return text.strip()
