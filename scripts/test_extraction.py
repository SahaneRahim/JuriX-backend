"""
Test pdfplumber extraction and compare with pypdf.
"""
import sys
sys.path.insert(0, '.')

from pathlib import Path

# File ID for Constitution
file_id = "e824bec8-f852-4a9c-aa27-5bf03f9b857f"
file_path = Path(f"data/uploads/{file_id}.pdf")

print(f"📄 Testing extraction from: {file_path}")

if not file_path.exists():
    print(f"❌ File not found: {file_path}")
    exit(1)

# Test pdfplumber
print("\n" + "=" * 60)
print("PDFPLUMBER EXTRACTION")
print("=" * 60)

import pdfplumber
import re

with pdfplumber.open(file_path) as pdf:
    # Get page 5 (where Article 4 should be)
    page5 = pdf.pages[4]  # 0-indexed
    text_pdfplumber = page5.extract_text(x_tolerance=3, y_tolerance=3)
    
    print(f"Page 5 text length: {len(text_pdfplumber)} chars")
    
    # Find Article 4
    art4_match = re.search(r'Article\s*4\s*[.:]', text_pdfplumber, re.IGNORECASE)
    if art4_match:
        start = art4_match.start()
        print("\nARTICLE 4 area:")
        print("-" * 40)
        print(text_pdfplumber[start:start+500])

# Test pypdf for comparison
print("\n" + "=" * 60)
print("PYPDF EXTRACTION (for comparison)")
print("=" * 60)

from pypdf import PdfReader

reader = PdfReader(file_path)
page5 = reader.pages[4]

try:
    text_pypdf = page5.extract_text(extraction_mode="layout")
except TypeError:
    text_pypdf = page5.extract_text()

print(f"Page 5 text length: {len(text_pypdf)} chars")

art4_match = re.search(r'Article\s*4\s*[.:]', text_pypdf, re.IGNORECASE)
if art4_match:
    start = art4_match.start()
    print("\nARTICLE 4 area:")
    print("-" * 40)
    print(text_pypdf[start:start+500])

# Compare which one correctly has "- Le Président" and "- Le Parlement"
print("\n" + "=" * 60)
print("QUALITY CHECK")
print("=" * 60)

if "- Le Président" in text_pdfplumber or "-Le Président" in text_pdfplumber:
    print("✅ pdfplumber: Found '- Le Président'")
else:
    print("❌ pdfplumber: Missing '- Le Président'")

if "- Le Président" in text_pypdf or "-Le Président" in text_pypdf:
    print("✅ pypdf: Found '- Le Président'")
else:
    print("❌ pypdf: Missing '- Le Président'")

if "- Le Parlement" in text_pdfplumber or "-Le Parlement" in text_pdfplumber:
    print("✅ pdfplumber: Found '- Le Parlement'")
else:
    print("❌ pdfplumber: Missing '- Le Parlement'")

if "- Le Parlement" in text_pypdf or "-Le Parlement" in text_pypdf:
    print("✅ pypdf: Found '- Le Parlement'")
else:
    print("❌ pypdf: Missing '- Le Parlement'")
