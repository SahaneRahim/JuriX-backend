"""
Re-extract Constitution with pdfplumber and save to database.
"""
import sys
import re
sys.path.insert(0, '.')

from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.models.law import Article
from app.utils.text_chunker import extract_articles

# File ID for Constitution
file_id = "e824bec8-f852-4a9c-aa27-5bf03f9b857f"
file_path = Path(f"data/uploads/{file_id}.pdf")

print(f"📄 Re-extracting with pdfplumber from: {file_path}")

if not file_path.exists():
    print(f"❌ File not found: {file_path}")
    exit(1)

# Extract with pdfplumber
import pdfplumber

pages_text = []
with pdfplumber.open(file_path) as pdf:
    total_pages = len(pdf.pages)
    print(f"📄 PDF has {total_pages} pages")
    
    for i, page in enumerate(pdf.pages):
        page_text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
        pages_text.append(f"<<PAGE: {i + 1}>>\n{page_text}")

extracted_text = "\n\n".join(pages_text)
print(f"✅ Extracted {len(extracted_text)} characters")

# Save to database
print("\n📥 Updating law content in database...")

sync_url = settings.DATABASE_URL.replace('+asyncpg', '')
engine = create_engine(sync_url)
Session = sessionmaker(bind=engine)

with Session() as session:
    session.execute(
        text("UPDATE laws SET content = :content WHERE id = 1"),
        {"content": extracted_text}
    )
    session.execute(text("DELETE FROM articles WHERE law_id = 1"))
    session.commit()
    print("✅ Law content updated")

# Extract articles
print("\n📊 Extracting articles...")
articles = extract_articles(extracted_text)
print(f"📊 Extracted {len(articles)} articles")

# Save articles with clean content
with Session() as session:
    for article_data in articles:
        content = article_data.get('content', '')
        # Additional cleaning
        content = re.sub(r'<<PAGE:\s*\d+>>', '', content)
        content = re.sub(r'\n\s*\d{1,2}\s*\n', '\n', content)  # Orphan page numbers
        content = content.strip()
        
        article = Article(
            law_id=1,
            number=str(article_data.get('number', '')),
            title=article_data.get('title'),
            section=article_data.get('section'),
            content=content,
            order=article_data.get('position', 0),
            page_number=article_data.get('page_number'),
        )
        session.add(article)
    
    session.commit()
    print(f"✅ Saved {len(articles)} articles")

# Verify results
print("\n📋 VERIFICATION:")
print("-" * 60)

for art_num in ['4', '9', '10']:
    art = [a for a in articles if str(a['number']) == art_num]
    if art:
        a = art[0]
        content = a['content']
        content = re.sub(r'<<PAGE:\s*\d+>>', '', content)
        content = re.sub(r'\n\s*\d{1,2}\s*\n', '\n', content)
        
        print(f"\nArticle {art_num}:")
        print(f"  Section: {a.get('section', 'N/A')}")
        print(f"  Content ({len(content)} chars):")
        print(f"  {content[:250]}...")

print("\n✅ DONE")
