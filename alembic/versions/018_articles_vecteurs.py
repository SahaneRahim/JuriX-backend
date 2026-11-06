"""Table etroite des vecteurs : articles_vecteurs, lue par la recherche exacte du chat

POURQUOI. La recherche exacte du chat compare la question a tous les articles
vectorises (SearchService._lire_vecteurs_exacts). Or `articles.embedding` est
en stockage EXTERNAL : chaque vecteur (2 056 octets) vit dans la table TOAST,
et se lit un par un par son index. A chaque reveil de Neon, le cache est vide :
~6,5 s de SQL estimees pour la premiere question, 17 s pour les premieres
questions du controle de nuit du 04/11/2026
(data/fiabilite/froid/controle-nuit-2026-11-04.log), d'ou le budget a froid de
30 s. Ici, le vecteur est dans la ligne (STORAGE PLAIN) : un balayage
sequentiel, que Neon lit d'avance, au lieu de ~17 700 lectures TOAST une par
une. Etude et options ecartees : data/fiabilite/froid/etude-resultat.json,
option (c).

EXACTE PAR CONSTRUCTION : memes octets, meme operateur, meme tri, memes filtres
(sur laws). Seule la sous-requete `proches` de SearchService._enonce_exact et
la verite du controle (controle_du_chat.voisins_exacts) lisent cette table.
L'index HNSW, son repli et /search restent sur articles.embedding : deux copies
des vecteurs coexistent, soit ~85 Mo de plus.

REPETEE LE 05/11/2026 sur une copie de jurix_dev, 32 303 vecteurs, machine
chargee (data/fiabilite/froid/migration-018/) : montee en 3,7 s, base +89 Mo
(table 10 768 blocs, cle primaire 0,7 Mo, rien en TOAST), lectures d'`articles`
sans attente pendant la montee. Les 32 voisins du chat sont identiques, toutes
colonnes et distances comprises, sur 301 requetes sur 301 (51 questions, 200
articles, 50 tableaux budgetaires) et 360 sur 360 sous d'autres filtres et
pages. Recherche exacte, cache de PostgreSQL vide : 11 317 pages lues et 159 en
cache, contre 21 490 et 86 465 ; fichiers hors du cache du noyau : 0,15 a 0,23 s
contre 1 a 1,6 s. L'etude, sur une copie rangee comme Neon : 11 162 contre
25 788. Aucune mesure sur Neon lui-meme : relire apres l'envoi la ligne « 🧊
Recherche semantique exacte, base presumee froide » du journal.

UN DECLENCHEUR LA TIENT A JOUR, dans la transaction de l'ecriture : insertion,
mise a jour du vecteur (retire s'il devient NULL) ou de la loi. La suppression
d'un article, ou d'une loi par `articles`, passe par la cle etrangere. Une
ecriture faite declencheurs coupes la fait deriver : ALTER TABLE ... DISABLE
TRIGGER USER, comme data/neon/outils/remplir-supabase.sh, ou le mode replica
(session_replication_role), qui coupe aussi la cascade. Le controle du chat
compare alors les comptes (controle_du_chat.compter_les_vecteurs) et rend 503.
Une egalite des comptes ne prouve pas celle des vecteurs : la verite du
controle lit cette table et l'index lit `articles`, un rappel effondre peut
donc aussi dire cela.

LE REMEDE A UNE DERIVE : rejouer cette montee. Elle remet le declencheur en
service et ne reecrit que les lignes qui different :

    DATABASE_URL=... alembic stamp 017_blog_posts_table
    DATABASE_URL=... alembic upgrade 018_articles_vecteurs
    DATABASE_URL=... alembic stamp head

Pendant ce rejeu, SET STORAGE tient un verrou ACCESS EXCLUSIVE sur la seule
articles_vecteurs : la recherche exacte attend, puis passe par l'index a
l'echeance de son budget.

CETTE MIGRATION NE S'APPLIQUE PAS A LA MAIN SUR LA PRODUCTION. Le conteneur
lance `alembic upgrade head` a son demarrage : le code qui lit la table et cette
revision partent ensemble. Une base en 018 sous un code sans ce fichier arrete
le conteneur (panne du 28/10/2026). VERIFIER AVANT L'ENVOI, en lecture seule :
la base est en 017_blog_posts_table, et pg_database_size + 85 Mo reste sous le
plafond de 1 Go de l'offre gratuite de Neon. Au-dela, le remplissage echoue, et
un echec ici arrete le conteneur sans que Railway garde l'ancien deploiement.

LE RETOUR ARRIERE, DANS CET ORDRE : deployer un code qui ne lit plus la table
mais GARDE ce fichier, puis `alembic downgrade 017_blog_posts_table` a la main.

LES VERROUS. Tout passe dans la transaction d'Alembic. La cle etrangere et le
declencheur prennent SHARE ROW EXCLUSIVE sur `articles`, jusqu'au COMMIT : les
lectures passent, les ecritures attendent, et aucune ne peut se glisser entre le
declencheur et le remplissage. Aucun lock_timeout : mieux vaut attendre une
ingestion que d'echouer, un echec arretant le conteneur. POURQUOI CREATE OR
REPLACE TRIGGER et non DROP puis CREATE : sur une base deja montee, DROP TRIGGER
prendrait ACCESS EXCLUSIVE sur `articles` (vu dans pg_locks), et couperait la
recherche pendant tout le rejeu.

UNE FUTURE MIGRATION DE TYPE DOIT RETIRER LE DECLENCHEUR D'ABORD. PostgreSQL
refuse ALTER COLUMN TYPE sur une colonne nommee dans UPDATE OF (« cannot alter
type of a column used in a trigger definition ») : articles.embedding et
articles.law_id en dependent. Sans lui, la colonne d'ici n'est plus suivie : le
faire dans la meme migration, et changer aussi le type d'articles_vecteurs.

Revision ID: 018_articles_vecteurs
Revises: 017_blog_posts_table
"""

from typing import Sequence, Union

from alembic import op

revision: str = "018_articles_vecteurs"
down_revision: Union[str, None] = "017_blog_posts_table"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# POURQUOI ORDER BY id : les lignes se rangent dans l'ordre des articles, et la
# cle primaire se remplit en ordre croissant, sans eclatement de pages. POURQUOI
# le WHERE : sur une base deja remplie, une ligne identique n'est pas reecrite.
REMPLISSAGE_SQL = """
    INSERT INTO articles_vecteurs (article_id, law_id, embedding)
    SELECT id, law_id, embedding FROM articles
    WHERE embedding IS NOT NULL
    ORDER BY id
    ON CONFLICT (article_id) DO UPDATE
        SET law_id = EXCLUDED.law_id, embedding = EXCLUDED.embedding
        WHERE (articles_vecteurs.law_id, articles_vecteurs.embedding)
            IS DISTINCT FROM (EXCLUDED.law_id, EXCLUDED.embedding)
"""

# Les lignes dont l'article a perdu son vecteur, ou n'existe plus : elles
# n'arrivent qu'apres une derive, et seule une session en mode replica peut
# laisser la seconde sorte.
RETRAIT_DES_ORPHELINS_SQL = """
    DELETE FROM articles_vecteurs v
    WHERE NOT EXISTS (
        SELECT 1 FROM articles a WHERE a.id = v.article_id AND a.embedding IS NOT NULL
    )
"""


def upgrade() -> None:
    op.execute("""
        CREATE TABLE IF NOT EXISTS articles_vecteurs (
            article_id integer PRIMARY KEY REFERENCES articles(id) ON DELETE CASCADE,
            law_id integer NOT NULL,
            embedding halfvec(1024) NOT NULL
        )
    """)
    # POURQUOI PLAIN : le type est en EXTERNAL par defaut, et rien d'autre ne
    # garderait le vecteur dans la ligne (2 056 octets, sous le seuil d'un
    # vecteur compresse et sous celui d'une page). Trois lignes par page.
    op.execute("ALTER TABLE articles_vecteurs ALTER COLUMN embedding SET STORAGE PLAIN")

    op.execute("""
        CREATE OR REPLACE FUNCTION articles_vecteurs_suivre() RETURNS trigger AS $$
        BEGIN
            IF NEW.embedding IS NULL THEN
                DELETE FROM articles_vecteurs WHERE article_id = NEW.id;
            ELSE
                INSERT INTO articles_vecteurs (article_id, law_id, embedding)
                VALUES (NEW.id, NEW.law_id, NEW.embedding)
                ON CONFLICT (article_id) DO UPDATE
                    SET law_id = EXCLUDED.law_id, embedding = EXCLUDED.embedding;
            END IF;
            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql
    """)
    op.execute("""
        CREATE OR REPLACE TRIGGER articles_vecteurs_suivre
        AFTER INSERT OR UPDATE OF embedding, law_id ON articles
        FOR EACH ROW EXECUTE FUNCTION articles_vecteurs_suivre()
    """)

    op.execute(REMPLISSAGE_SQL)
    op.execute(RETRAIT_DES_ORPHELINS_SQL)
    # Sans statistiques, le planificateur estimerait une table vide.
    op.execute("ANALYZE articles_vecteurs")


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS articles_vecteurs_suivre ON articles")
    op.execute("DROP FUNCTION IF EXISTS articles_vecteurs_suivre()")
    op.execute("DROP TABLE IF EXISTS articles_vecteurs")
