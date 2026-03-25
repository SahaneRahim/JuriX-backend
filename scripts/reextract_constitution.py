"""
Re-extract text from Constitution PDF with improved extraction.
"""
import sys
import os
sys.path.insert(0, '.')
os.environ.setdefault('PYTHONPATH', '.')

from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.models.law import Law, Article
from app.utils.text_chunker import extract_articles

# File ID for Constitution
file_id = "e824bec8-f852-4a9c-aa27-5bf03f9b857f"
file_path = Path(f"data/uploads/{file_id}.pdf")

print(f"📄 Re-extracting text from: {file_path}")

if not file_path.exists():
    print(f"❌ File not found: {file_path}")
    exit(1)

# Extract text with improved method
from pypdf import PdfReader

reader = PdfReader(file_path)
total_pages = len(reader.pages)
print(f"📄 PDF has {total_pages} pages")

pages_text = []
for i, page in enumerate(reader.pages):
    try:
        # Try layout mode for better structure
        page_text = page.extract_text(extraction_mode="layout")
    except TypeError:
        page_text = page.extract_text()
    
    # Add page marker
    pages_text.append(f"<<PAGE: {i + 1}>>\n{page_text}")

extracted_text = "\n\n".join(pages_text)
print(f"✅ Extracted {len(extracted_text)} characters")

# Show Article 4 area
import re
art4_match = re.search(r'Article\s+4\s*[.:]\s*', extracted_text, re.IGNORECASE)
if art4_match:
    start = art4_match.start()
    print("\n" + "=" * 60)
    print("ARTICLE 4 in NEW extraction:")
    print("=" * 60)
    print(extracted_text[start:start+600])

art9_match = re.search(r'Article\s+9\s*[.:]\s*', extracted_text, re.IGNORECASE)
if art9_match:
    start = art9_match.start()
    print("\n" + "=" * 60)
    print("ARTICLE 9 in NEW extraction:")
    print("=" * 60)
    print(extracted_text[start:start+800])

# Now save to database
print("\n📥 Updating law content in database...")

sync_url = settings.DATABASE_URL.replace('+asyncpg', '')
engine = create_engine(sync_url)
Session = sessionmaker(bind=engine)

with Session() as session:
    # Update law content
    session.execute(
        text("UPDATE laws SET content = :content WHERE id = 1"),
        {"content": extracted_text}
    )
    
    # Delete old articles
    session.execute(text("DELETE FROM articles WHERE law_id = 1"))
    
    session.commit()
    print("✅ Law content updated")

# Now run article extraction
print("\n📊 Extracting articles with improved parser...")
articles = extract_articles(extracted_text)

print(f"📊 Extracted {len(articles)} articles")

# Save articles to database
with Session() as session:
    for article_data in articles:
        article = Article(
            law_id=1,
            number=str(article_data.get('number', '')),
            title=article_data.get('title'),
            section=article_data.get('section'),
            content=article_data.get('content', ''),
            order=article_data.get('position', 0),
            page_number=article_data.get('page_number'),
        )
        session.add(article)
    
    session.commit()
    print(f"✅ Saved {len(articles)} articles")

# Show results for Article 4 and 9
for art in articles:
    if art['number'] in ['4', '9']:
        print(f"\n--- Article {art['number']} ---")
        print(f"Section: {art.get('section', 'None')}")
        print(f"Content ({len(art['content'])} chars):")
        print(art['content'][:300])
        print("...")
