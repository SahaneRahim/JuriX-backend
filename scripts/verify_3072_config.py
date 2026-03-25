"""
Script de vérification de la configuration 3072 dimensions
"""
print("=" * 60)
print("VERIFICATION CONFIGURATION 3072 DIMENSIONS")
print("=" * 60)

# 1. Check config
from app.core.config import settings
print(f"\n1. CONFIG:")
print(f"   Model: {settings.GEMINI_EMBEDDING_MODEL}")

# 2. Check embedding service
from app.services.embedding_service import EmbeddingService
print(f"\n2. EMBEDDING SERVICE:")
print(f"   EMBEDDING_DIM: {EmbeddingService.EMBEDDING_DIM}")
dim_ok = EmbeddingService.EMBEDDING_DIM == 3072
print(f"   Status: {'OK' if dim_ok else 'ERREUR'}")

# 3. Check model
from app.models.law import Article
print(f"\n3. MODEL Article.embedding:")
col = Article.__table__.columns['embedding']
print(f"   Type: {col.type}")

# 4. Check database
import psycopg2
url = settings.DATABASE_URL.replace('postgresql+asyncpg://', 'postgresql://')
conn = psycopg2.connect(url)
cur = conn.cursor()

# Column type
cur.execute("""
    SELECT a.atttypmod 
    FROM pg_attribute a 
    JOIN pg_class c ON a.attrelid = c.oid 
    WHERE c.relname = 'articles' AND a.attname = 'embedding'
""")
db_dim = cur.fetchone()[0]
print(f"\n4. DATABASE:")
print(f"   Column dimension: {db_dim}")
db_dim_ok = db_dim == 3072
print(f"   Status: {'OK' if db_dim_ok else 'ERREUR'}")

# Indexes
cur.execute("""
    SELECT indexname, indexdef 
    FROM pg_indexes 
    WHERE tablename = 'articles'
""")
indexes = cur.fetchall()
print(f"   Indexes: {[i[0] for i in indexes]}")
has_vector_index = any('embedding' in i[1] for i in indexes)
print(f"   Vector index present: {'OUI (probleme)' if has_vector_index else 'NON (correct)'}")

conn.close()

# 5. Test embedding generation
print(f"\n5. TEST EMBEDDING:")
svc = EmbeddingService()
emb = svc.generate_embedding("test juridique cameroun")
print(f"   Generated shape: {emb.shape}")
emb_ok = emb.shape[0] == 3072
print(f"   Status: {'OK - 3072 dimensions' if emb_ok else 'ERREUR'}")

# 6. Check output_dimensionality not used
import inspect as py_inspect
source = py_inspect.getsource(svc.generate_embedding)
has_output_dim = 'output_dimensionality' in source
print(f"\n6. MRL SLICING (output_dimensionality):")
print(f"   Utilise: {'OUI (probleme)' if has_output_dim else 'NON (correct - native 3072)'}")

# Summary
print("\n" + "=" * 60)
all_ok = dim_ok and db_dim_ok and not has_vector_index and emb_ok and not has_output_dim
if all_ok:
    print("RESULTAT: TOUTES LES VERIFICATIONS SONT OK!")
    print("Le systeme utilise les embeddings 3072 dimensions natifs.")
else:
    print("RESULTAT: CERTAINES VERIFICATIONS ONT ECHOUE!")
print("=" * 60)
