"""Embeddings 768 dimensions, index HNSW pose directement sur la colonne

Les embeddings passent a 768 dimensions : la sortie native d'EmbeddingGemma,
execute en local et gratuit, et une dimension que gemini-embedding-001 produit
aussi sur demande. Les deux fournisseurs partagent donc le meme schema ; en
changer est un reglage suivi d'une regeneration, plus une migration.

CE QUE CETTE MIGRATION DEFAIT. A 3072 dimensions, le type `vector` depassait le
plafond d'indexation de 2000 : l'index etait pose sur l'expression
`embedding::halfvec(3072)` (migration f5a6b7c8d9e0), et la recherche devait
caster pour l'atteindre. A 768, l'index se pose sur la colonne elle-meme, en
fp32, et la requete n'a plus de cast a reproduire au caractere pres.

CETTE MIGRATION DETRUIT TOUS LES VECTEURS. `USING NULL` est inevitable : un
vecteur 3072 n'a pas de projection en 768 dans l'espace d'un AUTRE modele.
Apres elle, la recherche semantique ne rend plus rien et l'hybride retombe sur
le plein texte, jusqu'a :

    python scripts/regenerate_embeddings.py --all
    python scripts/regenerate_embeddings.py --reindex

Le conteneur lance `alembic upgrade head` a chaque demarrage : deployer ce code
sur une base peuplee declenche cette purge. A verifier avant tout deploiement.

L'ORDRE DES OPERATIONS N'EST PAS LIBRE :
  - les anciens index sont supprimes AVANT l'ALTER. Verifie : un index
    d'expression halfvec(3072) laisse en place survit a la conversion de la
    colonne, reconstruit sur des NULL, et ne sert plus jamais aucune requete ;
  - en descente, le nouvel index est supprime AVANT le retour a 3072 : HNSW
    refuse d'indexer un `vector` de plus de 2000 dimensions, et l'ALTER
    echouerait en reconstruisant l'index.

`embedding_model` trace la PROVENANCE de chaque vecteur : l'empreinte du
fournisseur qui l'a produit. Deux modeles produisent des espaces sans rapport ;
comparer leurs vecteurs donne des cosinus absurdes, sans la moindre erreur.
Cette colonne permet de le voir, et a scripts/regenerate_embeddings.py de ne
refaire que ce qui doit l'etre.

Les deux caches sont purges : embedding_cache contient des vecteurs 3072, et
query_cache des resultats classes dans l'ancien espace.

Le nom de l'index change (_vector) pour qu'une base a moitie migree se
reconnaisse a l'oeil, et pour qu'un test ne puisse pas passer par accident
contre un ancien index survivant.

Ni CONCURRENTLY ni maintenance_work_mem ici : la colonne vient d'etre videe,
l'index se construit sur zero ligne. La construction en masse appartient a
scripts/regenerate_embeddings.py --reindex.

Revision ID: c4d5e6f7a8b9
Revises: b3c4d5e6f7a8
"""

import sqlalchemy as sa

from alembic import op

revision = "c4d5e6f7a8b9"
down_revision = "b3c4d5e6f7a8"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_articles_embedding_hnsw_halfvec")
    op.execute("DROP INDEX IF EXISTS idx_articles_embedding_hnsw")

    op.execute(
        "ALTER TABLE articles "
        "ALTER COLUMN embedding TYPE vector(768) USING NULL::vector(768)"
    )
    op.add_column("articles", sa.Column("embedding_model", sa.Text(), nullable=True))

    op.execute("DELETE FROM embedding_cache")
    op.execute("DELETE FROM query_cache")

    op.execute(
        "CREATE INDEX idx_articles_embedding_hnsw_vector ON articles "
        "USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS idx_articles_embedding_hnsw_vector")

    op.drop_column("articles", "embedding_model")
    op.execute(
        "ALTER TABLE articles "
        "ALTER COLUMN embedding TYPE vector(3072) USING NULL::vector(3072)"
    )

    op.execute("DELETE FROM embedding_cache")
    op.execute("DELETE FROM query_cache")

    op.execute(
        "CREATE INDEX idx_articles_embedding_hnsw_halfvec ON articles "
        "USING hnsw ((embedding::halfvec(3072)) halfvec_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )
