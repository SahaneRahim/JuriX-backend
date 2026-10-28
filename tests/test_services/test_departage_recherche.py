"""
Une meme recherche rend le meme classement, appel apres appel.

ts_rank_cd donne le meme score a tous les articles d'une loi dont seul le
titre correspond (chacun porte ce titre en poids A), et a des lois dont les
titres ont la meme forme. Les requetes triaient sur ce seul score : les ex
aequo sortaient dans l'ordre d'arrivee des lignes, que le plan parallele
choisi sur le corpus reel change d'un appel a l'autre. Mesure sur jurix_dev,
meme base et meme vecteur, trois essais par question : la reponse publique
changeait pour 48 questions d'evaluation sur 112 en mode texte, 24 en hybride.

Un plan parallele ne se provoque pas de facon fiable sur quelques lignes. Ces
tests rangent donc les lignes dans le tas A L'ENVERS de l'ordre attendu
(identifiants decroissants, verifie par ctid dans le premier test) : sans
departage, les ex aequo sortent dans l'ordre du tas et chaque test echoue
(verifie en retirant le departage). Avec lui, l'ordre est celui de la cle,
quel que soit le plan : la loi d'identifiant le plus petit, puis l'article le
plus tot dans le texte.
"""

import pytest
from sqlalchemy import text

from app.core.config import settings
from app.models.law import Article, Law
from app.services.postgres_search_service import (
    MAX_CHUNKS_PER_LAW,
    resolve_law_by_hint,
    search_articles_pg,
    search_laws_pg,
    search_titles_trgm_pg,
)

# Trois decrets « portant nomination », de meme forme : meme score de titre.
LOIS = (301, 302, 303)
ARTICLES_PAR_LOI = 5
SANS_LE_MOT = "L'interesse exerce ses fonctions des sa prise de service."
AVEC_LE_MOT = "La nomination prend effet des sa signature."


def _article(loi: int, position: int) -> int:
    """Identifiant de l'article : il suit l'ordre du texte, comme a l'ingestion."""
    return loi * 10 + position


async def _inserer_les_lois(db_session, corps=lambda loi, position: SANS_LE_MOT):
    """Les trois lois et leurs articles, A L'ENVERS : identifiants decroissants dans le tas."""
    for loi in reversed(LOIS):
        db_session.add(Law(
            id=loi, reference=f"DEC-2024-{loi}",
            title=f"Décret N°2024/{loi} portant nomination d'un Directeur",
            content="Le présent décret sera enregistré et publié au Journal Officiel.",
            type="décret", language="fr", status="published",
        ))
    await db_session.flush()
    for loi in reversed(LOIS):
        for position in range(ARTICLES_PAR_LOI, 0, -1):
            db_session.add(Article(
                id=_article(loi, position), law_id=loi, number=str(position),
                content=corps(loi, position), order=position,
            ))
    await db_session.commit()


@pytest.fixture
async def lois_ex_aequo(db_session):
    """Aucun article ne porte « nomination » : seul le titre correspond, quinze ex aequo."""
    await _inserer_les_lois(db_session)


@pytest.fixture
async def lois_melees(db_session):
    """
    L'article 4 des lois 301 et 302 porte AUSSI le mot dans son corps : il
    passe devant les autres articles, qui restent ex aequo entre eux.
    """
    await _inserer_les_lois(
        db_session,
        lambda loi, position: AVEC_LE_MOT if position == 4 and loi != 303 else SANS_LE_MOT,
    )


class TestPleinTexteArticles:

    @pytest.mark.asyncio
    async def test_le_tas_range_les_lignes_a_l_envers(self, db_session, lois_ex_aequo):
        """Le temoin : sans departage, l'ordre d'arrivee contredirait l'ordre attendu."""
        for table in ("articles", "laws"):
            tas = (await db_session.execute(
                text(f"SELECT id FROM {table} ORDER BY ctid")
            )).scalars().all()

            assert tas == sorted(tas, reverse=True), table

    @pytest.mark.asyncio
    async def test_a_score_egal_la_loi_puis_l_ordre_du_texte(self, db_session, lois_ex_aequo):
        """
        Les trois articles retenus d'une loi sont les trois premiers du texte,
        et les lois se suivent par identifiant. C'est ce qui changeait d'un
        appel a l'autre sur le corpus : le plafond par loi retenait trois
        articles quelconques parmi les ex aequo.
        """
        chunks = await search_articles_pg(db_session, "nomination", None, 60, 0)

        assert len({c.relevance_score for c in chunks}) == 1, "les lignes doivent etre ex aequo"
        assert [c.article_id for c in chunks] == [
            _article(loi, position)
            for loi in LOIS
            for position in range(1, MAX_CHUNKS_PER_LAW + 1)
        ]

    @pytest.mark.asyncio
    async def test_le_limit_coupe_toujours_au_meme_endroit(self, db_session, lois_ex_aequo):
        """Quatre lignes sur neuf ex aequo : toujours les memes quatre."""
        chunks = await search_articles_pg(db_session, "nomination", None, 4, 0)

        assert [c.article_id for c in chunks] == [3011, 3012, 3013, 3021]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("limite", [60, 8, 5, 3])
    async def test_des_ex_aequo_derriere_des_lignes_mieux_classees(
        self, db_session, lois_melees, limite
    ):
        """
        Les deux articles 4 remontent en tete ; le tri qui choisit les lignes
        du LIMIT doit les deplacer, et il n'est pas stable : sans son propre
        departage, il permutait les ex aequo restes derriere, et le LIMIT en
        gardait d'autres (verifie en retirant ce seul departage, celui de
        ROW_NUMBER en place).
        """
        chunks = await search_articles_pg(db_session, "nomination", None, limite, 0)

        assert [c.article_id for c in chunks] == [
            3014, 3024, 3011, 3012, 3021, 3022, 3031, 3032, 3033,
        ][:limite]


class TestNiveauLoi:

    @pytest.mark.asyncio
    async def test_branche_titre(self, db_session, lois_ex_aequo):
        lignes = await search_laws_pg(db_session, "nomination", None, 20, 0)

        assert len({c.relevance_score for c in lignes}) == 1, "les lois doivent etre ex aequo"
        assert [c.law_id for c in lignes] == list(LOIS)

    @pytest.mark.asyncio
    async def test_passe_floue_sur_les_titres(self, db_session, lois_ex_aequo):
        """« nominaton » : la meme similarite (0,7) pour les trois titres."""
        lignes = await search_titles_trgm_pg(
            db_session, "nominaton", None, 20, settings.TITLE_TRIGRAM_THRESHOLD
        )

        assert len({c.relevance_score for c in lignes}) == 1, "les lois doivent etre ex aequo"
        assert [c.law_id for c in lignes] == list(LOIS)


@pytest.fixture
async def articles_identiques(db_session):
    """
    Deux lois, deux articles chacune, au texte IDENTIQUE : meme similarite
    trigramme. Inseres a l'envers, comme plus haut.
    """
    for loi in (402, 401):
        db_session.add(Law(id=loi, reference=f"LOI-{loi}", title="Code du travail",
                           type="loi", content="Contenu.", language="fr",
                           status="published"))
    await db_session.flush()
    for loi in (402, 401):
        for position in (2, 1):
            db_session.add(Article(id=_article(loi, position), law_id=loi,
                                   number=str(position), content="Le licenciement abusif.",
                                   order=position))
    await db_session.commit()


class TestPasseTrigrammeArticles:

    @pytest.mark.asyncio
    async def test_a_similarite_egale_la_loi_puis_l_ordre_du_texte(
        self, db_session, articles_identiques
    ):
        # « licensiement » : le plein texte ne trouve rien, la passe trigramme
        # (mode texte) prend le relais.
        chunks = await search_articles_pg(db_session, "licensiement", None, 10, 0)

        assert {c.source for c in chunks} == {"trigram"}
        assert len({c.relevance_score for c in chunks}) == 1
        assert [c.article_id for c in chunks] == [4011, 4012, 4021, 4022]


class TestIndiceDeTitre:

    @pytest.mark.asyncio
    async def test_deux_lois_au_meme_titre(self, db_session):
        """
        Sept paires de lois portent le meme titre dans jurix_dev. L'indice
        « code minier » doit designer toujours la meme : c'est elle que la
        navigation directe ouvre si l'article demande y existe.
        """
        for loi in (503, 502, 501):
            db_session.add(Law(id=loi, reference=f"LOI-{loi}", title="Code minier",
                               type="loi", content="Contenu.", language="fr",
                               status="published"))
        await db_session.commit()

        assert await resolve_law_by_hint(db_session, "code minier") == [501, 502, 503]
