"""Embeddings Voyage : 1024 dimensions en halfvec, index HNSW sur la colonne

Les embeddings passent a voyage-4-large (API Voyage AI), choisi par mesure le
16/10/2026 : sur le jeu relu de 102 questions et tout le corpus, MRR@20 0,758
contre 0,537 pour EmbeddingGemma, en francais. Sa dimension par defaut est
1024, une dimension que gemini-embedding-001 produit aussi sur demande.

POURQUOI halfvec. Deux octets par composante au lieu de quatre : 2 Ko par
article au lieu de 4, autant de moins pour l'index HNSW, et pour chaque
vecteur relu. La precision fp16 (environ 3 chiffres significatifs, ecart
maximal mesure de 3e-5 sur un vecteur unitaire de 1024) ne change pas le
classement. L'index se pose sur la COLONNE elle-meme, sans expression : la
requete n'a aucun cast a reproduire (voir search_service._build_semantic_statement).

CETTE MIGRATION DETRUIT TOUS LES VECTEURS. `USING NULL` est inevitable : un
vecteur EmbeddingGemma (768) n'a pas de projection dans l'espace d'un autre
modele. Apres elle, la recherche semantique ne rend plus rien et l'hybride
retombe sur le plein texte, jusqu'a l'une des deux remises a niveau :

    # vecteurs deja calcules par la mesure (aucun appel a Voyage)
    python scripts/regenerate_embeddings.py --depuis-lots data/mesure_voyage/lots/voyage-4-large \
        --instantane data/mesure_voyage/morceaux.jsonl
    # ou tout recalculer par l'API (environ 10 M de jetons, dans le quota gratuit)
    python scripts/regenerate_embeddings.py --all
    # puis, dans les deux cas
    python scripts/regenerate_embeddings.py --reindex

Le conteneur lance `alembic upgrade head` a chaque demarrage : deployer ce code
sur une base peuplee declenche cette purge. A verifier avant tout deploiement.

L'ORDRE DES OPERATIONS N'EST PAS LIBRE :
  - l'ancien index est supprime AVANT l'ALTER, dans les deux sens. Verifie :
    l'index vector_cosine_ops laisse en place fait echouer l'ALTER (« operator
    class "vector_cosine_ops" does not accept data type halfvec »), et en
    descente l'index halfvec ferait echouer le retour en vector ;
  - le nouvel index est cree APRES, sur une colonne videe : il se construit
    sur zero ligne, d'ou ni CONCURRENTLY ni maintenance_work_mem ici. La
    construction en masse appartient a scripts/regenerate_embeddings.py
    --reindex.

`embedding_model` est remis a NULL : il nommait le fournisseur de vecteurs qui
n'existent plus.

Les deux caches sont purges : embedding_cache contient des vecteurs 768, et
query_cache des resultats classes dans l'ancien espace.

Nom d'index NEUF, idx_articles_embedding_hnsw_halfvec1024 : le nom
idx_articles_embedding_hnsw_halfvec est celui de l'index d'expression 3072 de
la migration f5a6b7c8d9e0, que la descente de 012 recree. Un nom distinct
laisse reconnaitre a l'oeil une base a moitie migree, et un test ne peut pas
passer par accident contre un ancien index survivant.

Revision ID: a8b9c0d1e2f3
Revises: f7a8b9c0d1e2
"""

from typing import Sequence, Union

from alembic import op

revision: str = "a8b9c0d1e2f3"
down_revision: Union[str, None] = "f7a8b9c0d1e2"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_articles_embedding_hnsw_vector")
    # La copie invalide qu'un REINDEX CONCURRENTLY interrompu laisse derriere
    # lui : l'ALTER la reconstruirait, et vector_cosine_ops refuse halfvec.
    op.execute("DROP INDEX IF EXISTS idx_articles_embedding_hnsw_vector_ccnew")
    op.execute("DROP INDEX IF EXISTS idx_articles_embedding_hnsw_halfvec1024")

    # embedding_model remis a NULL DANS la meme reecriture de la table : un
    # UPDATE apres coup recreerait une version de chaque ligne, et une entree
    # dans chaque index (dont les GIN plein texte), pour rien.
    op.execute(
        "ALTER TABLE articles "
        "ALTER COLUMN embedding TYPE halfvec(1024) USING NULL::halfvec(1024), "
        "ALTER COLUMN embedding_model TYPE text USING NULL::text"
    )

    op.execute("DELETE FROM embedding_cache")
    op.execute("DELETE FROM query_cache")

    op.execute(
        "CREATE INDEX idx_articles_embedding_hnsw_halfvec1024 ON articles "
        "USING hnsw (embedding halfvec_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_articles_embedding_hnsw_halfvec1024")
    op.execute("DROP INDEX IF EXISTS idx_articles_embedding_hnsw_halfvec1024_ccnew")

    op.execute(
        "ALTER TABLE articles "
        "ALTER COLUMN embedding TYPE vector(768) USING NULL::vector(768), "
        "ALTER COLUMN embedding_model TYPE text USING NULL::text"
    )

    op.execute("DELETE FROM embedding_cache")
    op.execute("DELETE FROM query_cache")

    op.execute(
        "CREATE INDEX idx_articles_embedding_hnsw_vector ON articles "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )
