"""
Quick verification of article storage - outputs to file.
"""
import sys
sys.path.insert(0, '.')

from sqlalchemy import create_engine, text
from app.core.config import settings

output = []
output.append("=" * 70)
output.append("JuriX ARTICLE STORAGE VERIFICATION REPORT")
output.append("=" * 70)

sync_url = settings.DATABASE_URL.replace('+asyncpg', '')
engine = create_engine(sync_url)

with engine.connect() as conn:
    # 1. Laws (parent documents)
    output.append("\n📚 PARENT DOCUMENTS (Laws)")
    output.append("-" * 40)
    result = conn.execute(text("""
        SELECT id, reference, title, status, LENGTH(content) as content_len
        FROM laws ORDER BY id
    """))
    for row in result:
        output.append(f"Law ID: {row[0]}")
        output.append(f"  Reference: {row[1]}")
        output.append(f"  Title: {row[2]}")
        output.append(f"  Status: {row[3]}")
        output.append(f"  Content Length: {row[4]} chars")
    
    # 2. Article count per law
    output.append("\n📄 ARTICLES PER LAW")
    output.append("-" * 40)
    result = conn.execute(text("""
        SELECT l.id, l.title, COUNT(a.id) as article_count
        FROM laws l
        LEFT JOIN articles a ON l.id = a.law_id
        GROUP BY l.id, l.title
    """))
    for row in result:
        output.append(f"Law {row[0]} ({row[1]}): {row[2]} articles")
    
    # 3. All articles with names and parent reference
    output.append("\n📑 ALL ARTICLES (Name + Parent Reference)")
    output.append("-" * 70)
    output.append(f"{'ID':>4} | {'Number':<20} | {'Section':<25} | {'Len':>6} | Parent")
    output.append("-" * 70)
    
    result = conn.execute(text("""
        SELECT a.id, a.number, a.section, LENGTH(a.content) as len, l.reference
        FROM articles a
        JOIN laws l ON a.law_id = l.id
        ORDER BY a.id
    """))
    for row in result:
        num = str(row[1])[:20]
        section = (row[2] or '')[:25]
        output.append(f"{row[0]:>4} | {num:<20} | {section:<25} | {row[3]:>6} | {row[4]}")
    
    # 4. Content length statistics
    output.append("\n📊 CONTENT LENGTH STATISTICS")
    output.append("-" * 40)
    result = conn.execute(text("""
        SELECT 
            COUNT(*) as total,
            MIN(LENGTH(content)) as min_len,
            MAX(LENGTH(content)) as max_len,
            AVG(LENGTH(content))::int as avg_len
        FROM articles
    """))
    row = result.fetchone()
    if row:
        output.append(f"Total Articles: {row[0]}")
        output.append(f"Min Content: {row[1]} chars")
        output.append(f"Max Content: {row[2]} chars")
        output.append(f"Avg Content: {row[3]} chars")
    
    # 5. Sample article contents
    output.append("\n📜 SAMPLE ARTICLE CONTENTS")
    output.append("-" * 70)
    result = conn.execute(text("""
        SELECT number, content FROM articles 
        WHERE UPPER(number) LIKE '%PREAMB%' 
           OR UPPER(number) = 'ARTICLE 1' 
           OR UPPER(number) = 'ARTICLE 5'
        LIMIT 3
    """))
    for row in result:
        output.append(f"\n--- {row[0]} ---")
        content = row[1] or ""
        output.append(f"Length: {len(content)} chars")
        output.append(f"Content (first 1000 chars):")
        output.append(content[:1000])
        output.append("...")

# Write to file
with open("verification_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(output))

print("Report saved to verification_report.txt")
print("\n".join(output[:50]))  # Print first 50 lines
