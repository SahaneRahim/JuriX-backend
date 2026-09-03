"""
Tests API pour les endpoints Laws.

Ces tests vérifient:
- GET /laws (liste avec filtres)
- GET /laws/{id} (détail)
- POST /admin/laws (création)
- PUT /admin/laws/{id} (mise à jour)
- DELETE /admin/laws/{id} (suppression)

Usage:
    pytest backend/tests/test_api/test_laws_api.py -v
"""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from app.main import app
from datetime import date

from app.models.law import Law


@pytest.fixture
async def sample_law(db_session: AsyncSession):
    """Fixture: Create a sample law for testing."""
    law = Law(
        reference="LOI-2024-001",
        title="Loi sur les sociétés commerciales",
        type="loi",
        content="Article 1. La présente loi régit les sociétés commerciales au Cameroun.",
        language="fr",
        status="published",
        publication_date=date(2024, 1, 1)
    )
    db_session.add(law)
    await db_session.commit()
    await db_session.refresh(law)
    return law


# ==================== GET /laws TESTS ====================


class TestGetLaws:
    """Tests pour GET /api/v1/laws."""

    @pytest.mark.asyncio
    async def test_get_laws_success(self, client: AsyncClient, sample_law):
        """Test récupération liste des lois."""
        response = await client.get("/api/v1/laws/")
        
        assert response.status_code == 200
        laws = response.json()
        assert isinstance(laws, list)
        assert len(laws) >= 1

    @pytest.mark.asyncio
    async def test_get_laws_filter_language(self, client: AsyncClient, sample_law):
        """Test filtre par langue."""
        response = await client.get("/api/v1/laws/?language=fr")
        
        assert response.status_code == 200
        laws = response.json()
        assert all(law["language"] == "fr" for law in laws)

    @pytest.mark.asyncio
    async def test_get_laws_filter_status(self, client: AsyncClient, sample_law):
        """Test filtre par statut."""
        response = await client.get("/api/v1/laws/?law_status=published")
        
        assert response.status_code == 200
        laws = response.json()
        assert all(law["status"] == "published" for law in laws)

    @pytest.mark.asyncio
    async def test_get_laws_pagination(self, client: AsyncClient, sample_law):
        """Test pagination."""
        response = await client.get("/api/v1/laws/?skip=0&limit=10")
        
        assert response.status_code == 200
        laws = response.json()
        assert len(laws) <= 10

    @pytest.mark.asyncio
    async def test_get_laws_empty(self, client: AsyncClient):
        """Test résultat vide."""
        response = await client.get("/api/v1/laws/?language=xx")
        
        assert response.status_code == 200
        laws = response.json()
        assert laws == []


# ==================== GET /laws/{id} TESTS ====================


class TestGetLaw:
    """Tests pour GET /api/v1/laws/{id}."""

    @pytest.mark.asyncio
    async def test_get_law_success(self, client: AsyncClient, sample_law):
        """Test récupération détail loi."""
        response = await client.get(f"/api/v1/laws/{sample_law.id}")
        
        assert response.status_code == 200
        law = response.json()
        assert law["id"] == sample_law.id
        assert law["reference"] == sample_law.reference
        assert law["title"] == sample_law.title

    @pytest.mark.asyncio
    async def test_get_law_not_found(self, client: AsyncClient):
        """Test loi non trouvée."""
        response = await client.get("/api/v1/laws/99999")
        
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_get_law_with_articles(self, client: AsyncClient, sample_law):
        """Test loi avec articles."""
        response = await client.get(f"/api/v1/laws/{sample_law.id}")
        
        assert response.status_code == 200
        law = response.json()
        # LawResponse expose article_count, pas la liste des articles :
        # la charger sur chaque loi d'une liste serait un N+1.
        assert "article_count" in law


# ==================== POST /admin/laws TESTS ====================


class TestCreateLaw:
    """Tests pour POST /api/v1/laws/admin."""

    @pytest.mark.asyncio
    async def test_create_law_success(self, admin_client: AsyncClient):
        """Test création loi."""
        new_law = {
            "reference": "LOI-2024-002",
            "title": "Loi test",
            "type": "loi",
            "content": "Contenu de la loi test.",
            "language": "fr",
            "status": "published"
        }
        
        response = await admin_client.post("/api/v1/laws/admin", json=new_law)
        
        assert response.status_code == 201
        law = response.json()
        assert law["reference"] == new_law["reference"]
        assert law["title"] == new_law["title"]
        assert "id" in law

    @pytest.mark.asyncio
    async def test_create_law_duplicate_reference(self, admin_client: AsyncClient, sample_law):
        """Test création avec référence existante."""
        duplicate_law = {
            "reference": sample_law.reference,
            "title": "Loi duplicate",
            "type": "loi",
            "content": "Contenu de la loi dupliquee.",
            "language": "fr"
        }
        
        response = await admin_client.post("/api/v1/laws/admin", json=duplicate_law)
        
        assert response.status_code == 409
        assert "already exists" in response.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_create_law_validation_error(self, admin_client: AsyncClient):
        """Test création avec données invalides."""
        invalid_law = {
            "reference": "",  # Invalid: empty
            "title": "Test"
        }
        
        response = await admin_client.post("/api/v1/laws/admin", json=invalid_law)
        
        assert response.status_code == 422


# ==================== PUT /admin/laws/{id} TESTS ====================


class TestUpdateLaw:
    """Tests pour PUT /api/v1/laws/admin/{id}."""

    @pytest.mark.asyncio
    async def test_update_law_success(self, admin_client: AsyncClient, sample_law):
        """Test mise à jour loi."""
        update_data = {
            "title": "Titre mis à jour",
            "status": "archived"
        }
        
        response = await admin_client.put(
            f"/api/v1/laws/admin/{sample_law.id}",
            json=update_data
        )
        
        assert response.status_code == 200
        law = response.json()
        assert law["title"] == update_data["title"]
        assert law["status"] == update_data["status"]

    @pytest.mark.asyncio
    async def test_update_law_not_found(self, admin_client: AsyncClient):
        """Test mise à jour loi non trouvée."""
        response = await admin_client.put(
            "/api/v1/laws/admin/99999",
            json={"title": "Test"}
        )
        
        assert response.status_code == 404


# ==================== DELETE /admin/laws/{id} TESTS ====================


class TestDeleteLaw:
    """Tests pour DELETE /api/v1/laws/admin/{id}."""

    @pytest.mark.asyncio
    async def test_delete_law_success(self, admin_client: AsyncClient, sample_law):
        """Test suppression loi."""
        response = await admin_client.delete(f"/api/v1/laws/admin/{sample_law.id}")
        
        assert response.status_code == 204
        
        # Vérifier que la loi est supprimée
        get_response = await admin_client.get(f"/api/v1/laws/{sample_law.id}")
        assert get_response.status_code == 404

    @pytest.mark.asyncio
    async def test_delete_law_not_found(self, admin_client: AsyncClient):
        """Test suppression loi non trouvée."""
        response = await admin_client.delete("/api/v1/laws/admin/99999")
        
        assert response.status_code == 404


# ==================== INTEGRATION TESTS ====================


class TestLawsIntegration:
    """Tests d'intégration."""

    @pytest.mark.asyncio
    async def test_full_crud_workflow(self, admin_client: AsyncClient):
        """Test workflow CRUD complet."""
        # 1. Create
        new_law = {
            "reference": "LOI-2024-CRUD",
            "title": "Loi CRUD Test",
            "type": "loi",
            "content": "Contenu test.",
            "language": "fr",
            "status": "published"
        }
        create_response = await admin_client.post("/api/v1/laws/admin", json=new_law)
        assert create_response.status_code == 201
        law_id = create_response.json()["id"]
        
        # 2. Read
        read_response = await admin_client.get(f"/api/v1/laws/{law_id}")
        assert read_response.status_code == 200
        assert read_response.json()["title"] == new_law["title"]
        
        # 3. Update
        update_response = await admin_client.put(
            f"/api/v1/laws/admin/{law_id}",
            json={"title": "Titre modifié"}
        )
        assert update_response.status_code == 200
        assert update_response.json()["title"] == "Titre modifié"
        
        # 4. Delete
        delete_response = await admin_client.delete(f"/api/v1/laws/admin/{law_id}")
        assert delete_response.status_code == 204
        
        # 5. Verify deletion
        verify_response = await admin_client.get(f"/api/v1/laws/{law_id}")
        assert verify_response.status_code == 404
