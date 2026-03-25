"""
Debug script to analyze raw text and article splitting - output to file.
"""
import sys
import os
sys.path.insert(0, '.')
os.environ.setdefault('PYTHONPATH', '.')

from sqlalchemy import create_engine, text
from app.core.config import settings

# Get raw text from DB
sync_url = settings.DATABASE_URL.replace('+asyncpg', '')
engine = create_engine(sync_url)

with engine.connect() as conn:
    result = conn.execute(text("SELECT content FROM laws WHERE id = 1"))
    row = result.fetchone()
    raw_text = row[0] if row else ""

# Write analysis to file
with open('debug_output.txt', 'w', encoding='utf-8') as f:
    f.write(f"Total raw text length: {len(raw_text)} chars\n\n")
    
    import re
    
    # Find ALL article positions
    article_pattern = re.compile(r'Article\s+(\d+)\s*[.:]', re.IGNORECASE)
    matches = list(article_pattern.finditer(raw_text))
    
    f.write(f"Found {len(matches)} articles in raw text\n\n")
    
    for i, match in enumerate(matches[:20]):  # First 20 articles
        art_num = match.group(1)
        start = match.start()
        # End is start of next article or +2000 chars
        if i + 1 < len(matches):
            end = matches[i + 1].start()
        else:
            end = start + 2000
        
        content = raw_text[start:end]
        
        f.write("=" * 70 + "\n")
        f.write(f"ARTICLE {art_num} (pos {start} to {end}, len={end-start})\n")
        f.write("=" * 70 + "\n")
        f.write(content[:1500] + "\n\n")

print("Output written to debug_output.txt")
