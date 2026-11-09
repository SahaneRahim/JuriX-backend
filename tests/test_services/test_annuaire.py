"""L'annuaire des institutions et les contacts joints a une reponse du chat."""
from fastapi.testclient import TestClient

from app.main import app
from app.services.annuaire import ANNUAIRE, REGIONS, contacts_pour, lister


def test_chaque_fiche_a_une_source_et_un_moyen_de_contact():
    for fiche in ANNUAIRE:
        assert fiche["source"]["url"].startswith("https://")
        assert fiche["telephones"] or fiche["email"] or fiche["site"]


def test_une_delegation_du_travail_par_region():
    regions = {f["region"] for f in ANNUAIRE if f["id"].startswith("drtss-")}
    assert regions == set(REGIONS)


def test_l_inspection_du_travail_donne_les_delegations():
    ids = [f["id"] for f in contacts_pour("Vous pouvez saisir l'inspecteur du travail de votre ressort.")]
    assert "drtss-littoral" in ids and "mintss" in ids


def test_avocat_et_cnps():
    ids = [f["id"] for f in contacts_pour("Consultez un avocat et vérifiez vos droits à la CNPS.")]
    assert ids == ["cnps", "barreau"]


def test_aucune_institution_aucun_contact():
    assert contacts_pour("Le contrat de travail peut être conclu oralement.") == []


def test_le_filtre_par_region_garde_les_fiches_nationales():
    fiches = lister(region="littoral")
    assert any(f["id"] == "drtss-littoral" for f in fiches)
    assert not any(f["id"] == "drtss-centre" for f in fiches)
    assert any(f["id"] == "cnps" for f in fiches)


def test_la_route_publique():
    reponse = TestClient(app).get("/api/v1/annuaire?theme=travail")
    assert reponse.status_code == 200
    corps = reponse.json()
    assert corps["fiches"] and all(f["theme"] == "travail" for f in corps["fiches"])
    assert "max-age" in reponse.headers["cache-control"]
