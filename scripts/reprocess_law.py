"""
Script to reprocess a law with the improved article parser.
"""
import sys
sys.path.insert(0, '.')

from app.tasks.process_law import process_law_sync

# Reprocess law ID 1 (La Constitution)
law_id = 1
file_id = "e824bec8-f852-4a9c-aa27-5bf03f9b857f"

print(f"🔄 Reprocessing law {law_id} with file {file_id}...")
try:
    result = process_law_sync(law_id, file_id)
    print(f"✅ Success! Result: {result}")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
