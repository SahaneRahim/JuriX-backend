"""
Definition UNIQUE des tsvector de recherche.

Ces expressions etaient recopiees a l'identique dans cinq endroits : les deux
fonctions de trigger de la migration a1b2c3d4e5f6, `update_law_search_vector`,
`SearchService.reindex_all_laws` et deux fonctions de `app/tasks/process_law.py`.
Cinq copies d'une meme regle divergent a la premiere retouche ; ici la regle est
ecrite une fois, et les cinq appelants la citent.

LA PONDERATION EST LE POINT ESSENTIEL. Sans `setweight`, un mot du titre et un
mot du corps du texte sont indiscernables pour `ts_rank_cd`. Mesure sur le
corpus, requete « nomination » : le document dont le titre ne contient PAS le
mot sortait premier (rang 0,4000) devant les cinq documents dont le titre le
porte (rang 0,2000), parce qu'il le repete dans son corps.

Poids retenus :
    A  titre de la loi          — ce que l'utilisateur cherche en premier
    B  numero et titre d'article — « article 35 » doit trouver l'article 35
    D  corps du texte           — pertinent, mais jamais devant un titre

Le titre de la loi est injecte AUSSI dans `articles.search_vector` : la
recherche interroge les articles, pas les lois, et ce vecteur ne contenait ni le
titre de la loi ni celui de l'article.

Author: JuriX Team
"""

# Poids passes a ts_rank_cd : {D, C, B, A}. L'ecart entre A et D doit etre
# assez grand pour qu'aucune accumulation d'occurrences dans le corps ne
# rattrape une seule occurrence dans le titre.
RANK_WEIGHTS = "{0.05, 0.1, 0.2, 1.0}"

# Ne retient que le poids A : vrai si et seulement si la correspondance passe
# par le TITRE. Sert a separer les deux sections de resultats. Ce n'est pas une
# heuristique — mesure sur le corpus : 5 titres sur 5, 3 corps sur 3, zero erreur.
TITLE_ONLY_WEIGHTS = "{0, 0, 0, 1}"


def law_search_vector_sql(title: str = "title", content: str = "content") -> str:
    """Expression du tsvector d'une loi. Les arguments nomment les colonnes."""
    return (
        f"setweight(to_tsvector('french',  coalesce({title}, '')),   'A') || "
        f"setweight(to_tsvector('english', coalesce({title}, '')),   'A') || "
        f"setweight(to_tsvector('french',  coalesce({content}, '')), 'D') || "
        f"setweight(to_tsvector('english', coalesce({content}, '')), 'D')"
    )


def article_search_vector_sql(
    law_title: str,
    number: str = "number",
    article_title: str = "title",
    content: str = "content",
) -> str:
    """
    Expression du tsvector d'un article.

    `law_title` n'a pas de valeur par defaut : il n'existe pas sur la table
    `articles` et doit venir d'une jointure ou d'une variable plpgsql. L'oublier
    doit etre une erreur d'appel, pas un vecteur silencieusement sans titre.
    """
    return (
        f"setweight(to_tsvector('french',  coalesce({law_title}, '')),     'A') || "
        f"setweight(to_tsvector('english', coalesce({law_title}, '')),     'A') || "
        f"setweight(to_tsvector('simple',  coalesce({number}, '')),        'B') || "
        f"setweight(to_tsvector('french',  coalesce({article_title}, '')), 'B') || "
        f"setweight(to_tsvector('french',  coalesce({content}, '')),       'D') || "
        f"setweight(to_tsvector('english', coalesce({content}, '')),       'D')"
    )


# Corps des deux fonctions de trigger. La migration les cree, et un test
# compare la definition reellement en base a ces chaines : c'est ce qui empeche
# le code et le schema de diverger a nouveau.
LAWS_TRIGGER_FUNCTION = f"""
CREATE OR REPLACE FUNCTION laws_search_vector_update() RETURNS trigger AS $$
BEGIN
    NEW.search_vector := {law_search_vector_sql('NEW.title', 'NEW.content')};
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
"""

ARTICLES_TRIGGER_FUNCTION = f"""
CREATE OR REPLACE FUNCTION articles_search_vector_update() RETURNS trigger AS $$
DECLARE
    l_title text;
BEGIN
    SELECT title INTO l_title FROM laws WHERE id = NEW.law_id;
    NEW.search_vector := {article_search_vector_sql('l_title', 'NEW.number', 'NEW.title', 'NEW.content')};
    RETURN NEW;
END;
$$ LANGUAGE plpgsql
"""

# Le titre de la loi entrant dans le vecteur de ses articles, une modification
# du titre doit reindexer les articles. Sans ce second trigger, renommer une loi
# laisserait ses articles indexes sur l'ancien titre — le genre d'incoherence
# qui ne se voit qu'a la recherche, des mois plus tard.
# `SET search_vector = ...` et non `SET content = content`. La seconde forme
# reecrivait chaque ligne d'articles pour le seul effet de bord de declencher le
# trigger : sur un code de 5000 articles, cela ferait 5000 tuples morts et
# autant de mises a jour de l'index HNSW 3072 dimensions, pour un simple
# changement de titre.
# RETURN NULL : la valeur de retour d'un trigger AFTER FOR EACH ROW est ignoree.
LAWS_TITLE_CASCADE_FUNCTION = f"""
CREATE OR REPLACE FUNCTION laws_title_reindex_articles() RETURNS trigger AS $$
BEGIN
    UPDATE articles a
    SET search_vector = {article_search_vector_sql('NEW.title', 'a.number', 'a.title', 'a.content')}
    WHERE a.law_id = NEW.id;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql
"""

# Reindexation en masse, utilisee par la migration et par reindex_all_laws.
REINDEX_LAWS_SQL = f"UPDATE laws SET search_vector = {law_search_vector_sql()}"

REINDEX_ARTICLES_SQL = f"""
UPDATE articles a
SET search_vector = {article_search_vector_sql('l.title', 'a.number', 'a.title', 'a.content')}
FROM laws l
WHERE l.id = a.law_id
"""
