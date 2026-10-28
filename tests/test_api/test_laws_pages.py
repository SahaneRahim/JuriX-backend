"""
GET /api/v1/laws/ : page legere (`page`, `page_size`, `sort`) et liste nue
d'origine, sur une vraie base PostgreSQL.

La liste nue ne change pas de forme : le front deploye la lit comme un
tableau (`limit + 1` pour savoir s'il existe une suite). La page legere sert
les nouvelles listes : par lots de 20, le vrai total, un lien « ?page=2 » qui
marche sans JavaScript, et les reperes d'une ligne dense.
"""

from datetime import date

import pytest

from app.models.law import Article, Law
from app.services.legal_domain_classifier import FONCTION_PUBLIQUE, TRAVAIL


@pytest.fixture
async def domaine(db_session, category_ids):
    """
    23 textes de la fonction publique, dont des dates en double (l'ordre doit
    rester total) et un texte sans date ; 2 textes d'un autre domaine.
    """
    fp, travail = category_ids[FONCTION_PUBLIQUE], category_ids[TRAVAIL]
    lois = []
    for i in range(23):
        jour = None if i == 22 else date(2010 + i % 7, 3, 1 + i % 2)
        lois.append(Law(
            id=500 + i,
            reference=f"PRC-{500 + i}",
            title=f"Décret N°{2010 + i % 7}/{500 + i} du 1 mars {2010 + i % 7} portant nomination du responsable {i}",
            type="décret",
            content=f"Contenu {i}.",
            language="fr",
            status="published",
            category_id=fp,
            publication_date=jour,
        ))
    for i in range(2):
        lois.append(Law(
            id=600 + i, reference=f"PRC-{600 + i}", title=f"Loi N°2020/{600 + i} du 2 mai 2020 portant Code {i}",
            type="loi", content="Contenu.", language="fr", status="published",
            category_id=travail, publication_date=date(2020, 5, 2),
        ))
    lois.append(Law(
        id=700, reference="PRC-700", title="Décret refusé", type="décret", content="Contenu.",
        language="fr", status="refused", category_id=fp, publication_date=date(2026, 1, 2),
    ))
    for loi in lois:
        db_session.add(loi)
    await db_session.flush()
    for numero in ("1", "2", "3"):
        db_session.add(Article(law_id=500, number=numero, content=f"Article {numero}.", order=int(numero)))
    await db_session.commit()
    return fp


@pytest.mark.asyncio
async def test_page_legere_et_vrai_total(client, domaine):
    reponse = await client.get(f"/api/v1/laws/?category_id={domaine}&page=1&page_size=10")
    assert reponse.status_code == 200
    page = reponse.json()
    assert page["total"] == 23
    assert page["pages"] == 3
    assert page["page"] == 1 and page["page_size"] == 10
    assert page["sort"] == "date_desc"
    assert len(page["items"]) == 10
    ligne = page["items"][0]
    # Leger : ni le texte, ni les champs d'administration.
    assert "content" not in ligne and "file_id" not in ligne and "suggested_categories" not in ligne
    for cle in ("objet", "numero", "date_signature", "annee", "statut", "article_count", "category_name"):
        assert cle in ligne


@pytest.mark.asyncio
async def test_toutes_les_pages_sans_doublon_ni_oubli(client, domaine):
    """Des dates en double : l'id departage, aucune ligne ne saute d'une page a l'autre."""
    vus = []
    for numero in range(1, 4):
        page = (await client.get(f"/api/v1/laws/?category_id={domaine}&page={numero}&page_size=10")).json()
        vus.extend(page["items"])
    assert len(vus) == 23
    assert len({ligne["id"] for ligne in vus}) == 23
    # Date decroissante, puis id decroissant ; le texte sans date en dernier.
    cles = [(ligne["publication_date"], ligne["id"]) for ligne in vus[:-1]]
    assert cles == sorted(cles, reverse=True)
    assert vus[-1]["publication_date"] is None


@pytest.mark.asyncio
async def test_a_date_egale_l_ordre_des_actes(client, db_session, category_ids):
    """
    Cinq decrets du meme jour, ingeres dans le desordre (ids croissants,
    numeros PRC decroissants), et une loi de reference d'une autre forme : a
    date egale, le numero PRC (ordre de mise en ligne, celui des actes) range
    les decrets, et la reference sans numero PRC passe apres eux.
    """
    domaine = category_ids[FONCTION_PUBLIQUE]
    for i, prc in enumerate((10758, 10755, 10757, 10756, 10754)):
        db_session.add(Law(
            id=800 + i, reference=f"PRC-{prc}", title=f"Décret N°2026/{prc - 10474} du 31 août 2026 portant nomination",
            type="décret", content="Contenu.", language="fr", status="published",
            category_id=domaine, publication_date=date(2026, 8, 31),
        ))
    db_session.add(Law(
        id=900, reference="LOI-2026-009", title="Loi N°2026/009 du 31 août 2026 portant statut",
        type="loi", content="Contenu.", language="fr", status="published",
        category_id=domaine, publication_date=date(2026, 8, 31),
    ))
    await db_session.commit()

    page = (await client.get(f"/api/v1/laws/?category_id={domaine}&page=1")).json()
    assert [ligne["reference"] for ligne in page["items"]] == [
        "PRC-10758", "PRC-10757", "PRC-10756", "PRC-10755", "PRC-10754", "LOI-2026-009",
    ]
    croissant = (await client.get(f"/api/v1/laws/?category_id={domaine}&page=1&sort=date_asc")).json()
    assert [ligne["reference"] for ligne in croissant["items"]] == [
        "PRC-10754", "PRC-10755", "PRC-10756", "PRC-10757", "PRC-10758", "LOI-2026-009",
    ]


@pytest.mark.asyncio
async def test_tri_croissant(client, domaine):
    page = (await client.get(f"/api/v1/laws/?category_id={domaine}&page=1&page_size=5&sort=date_asc")).json()
    dates = [ligne["publication_date"] for ligne in page["items"]]
    assert dates == sorted(dates)
    assert page["sort"] == "date_asc"


@pytest.mark.asyncio
async def test_page_au_dela_de_la_derniere(client, domaine):
    page = (await client.get(f"/api/v1/laws/?category_id={domaine}&page=9&page_size=10")).json()
    assert page["items"] == []
    assert page["total"] == 23


@pytest.mark.asyncio
async def test_reperes_et_nombre_d_articles(client, domaine):
    page = (await client.get(f"/api/v1/laws/?category_id={domaine}&page=1&page_size=100")).json()
    par_id = {ligne["id"]: ligne for ligne in page["items"]}
    ligne = par_id[500]
    assert ligne["objet"] == "Nomination du responsable 0"
    assert ligne["numero"] == "2010/500"
    assert ligne["article_count"] == 3
    assert ligne["category_name"] == FONCTION_PUBLIQUE
    assert par_id[501]["article_count"] == 0
    # Le statut refuse est exclu, comme dans la liste nue.
    assert 700 not in par_id


@pytest.mark.asyncio
async def test_la_liste_nue_garde_sa_forme(client, domaine):
    """Sans `page` : un tableau, le texte entier, comme le lit le front deploye."""
    reponse = await client.get(f"/api/v1/laws/?category_id={domaine}&limit=5")
    lignes = reponse.json()
    assert isinstance(lignes, list) and len(lignes) == 5
    assert all("content" in ligne and ligne["content"] for ligne in lignes)
    assert "objet" in lignes[0] and "statut" in lignes[0]


@pytest.mark.asyncio
async def test_la_liste_nue_accepte_un_tri(client, domaine):
    lignes = (await client.get(f"/api/v1/laws/?category_id={domaine}&limit=30&sort=date_desc")).json()
    dates = [ligne["publication_date"] for ligne in lignes if ligne["publication_date"]]
    assert dates == sorted(dates, reverse=True)
    assert lignes[-1]["publication_date"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize("requete", ["page=0", "page_size=101", "page=1&sort=au_hasard"])
async def test_parametres_invalides(client, domaine, requete):
    assert (await client.get(f"/api/v1/laws/?{requete}")).status_code == 422
