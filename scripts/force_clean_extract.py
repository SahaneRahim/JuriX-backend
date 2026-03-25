"""
Force clean re-extraction with verification.
"""
import sys
sys.path.insert(0, '.')

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from app.core.config import settings
from app.models.law import Article
from app.utils.text_chunker import extract_articles

sync_url = settings.DATABASE_URL.replace('+asyncpg', '')
engine = create_engine(sync_url)
Session = sessionmaker(bind=engine)

# Get law content
with engine.connect() as conn:
    row = conn.execute(text("SELECT content FROM laws WHERE id = 1")).fetchone()
    raw_text = row[0]

print(f"Raw text length: {len(raw_text)} chars")

# Extract articles
articles = extract_articles(raw_text)
print(f"Extracted {len(articles)} articles")

# Verify Article 9 content is clean
art9 = [a for a in articles if a['number'] == '9'][0]
print(f"\nArticle 9 content has <<PAGE? {'<<PAGE' in art9['content']}")
print(f"Article 9 content length: {len(art9['content'])}")

# Delete and re-insert articles
with Session() as session:
    session.execute(text("DELETE FROM articles WHERE law_id = 1"))
    session.commit()
    print("Deleted old articles")

with Session() as session:
    for article_data in articles:
        content = article_data.get('content', '')
        # Double-check cleaning
        import re
        content = re.sub(r'<<PAGE:\s*\d+>>', '', content)
        content = re.sub(r'\n\s*\d+\s*\n', '\n', content)  # Remove orphan page numbers
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
    print(f"Saved {len(articles)} articles with cleaned content")

# Verify
with engine.connect() as conn:
    row = conn.execute(text("SELECT content FROM articles WHERE law_id = 1 AND number = '9'")).fetchone()
    print(f"\nVerification - Article 9 in DB has <<PAGE? {'<<PAGE' in row[0] if row else 'N/A'}")
    print(f"Content:\n{row[0][:400] if row else 'N/A'}")
