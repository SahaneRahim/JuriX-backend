"""Flush Redis cache to clear stale search results."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import redis
from app.core.config import settings

try:
    print(f"Connecting to Redis at: {settings.REDIS_URL}")
    client = redis.from_url(settings.REDIS_URL, decode_responses=True)
    client.ping()
    print("✅ Redis connected")
    
    # Count keys before flush
    keys = client.keys("*")
    print(f"Keys before flush: {len(keys)}")
    
    # Show some search-related keys
    search_keys = [k for k in keys if 'search' in k.lower() or 'jurix' in k.lower()]
    print(f"Search-related keys: {search_keys[:10]}")
    
    # Flush all keys
    client.flushall()
    print("🗑️ Redis cache FLUSHED!")
    
    # Verify
    keys_after = client.keys("*")
    print(f"Keys after flush: {len(keys_after)}")
    
except Exception as e:
    print(f"❌ Redis error: {e}")
    print("\nIf Redis is not running, start it with: redis-server")
