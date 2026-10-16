"""
Tests de scripts/regenerate_embeddings.py : ce qui est a refaire, et la trace
laissee.

La selection repose sur la PROVENANCE des vecteurs (articles.embedding_model),
plus seulement sur leur absence. C'est ce qui rend un changement de fournisseur
sur : chaque vecteur de l'ancien modele est reconnu et refait, au lieu de
cohabiter en silence avec ceux du nouveau — deux espaces dont les cosinus
croises ne veulent rien dire.

Usage:
    pytest tests/test_scripts/test_regenerate_embeddings.py -v
"""

import json

import numpy as np
import pytest
from sqlalchemy import text

from app.models.law import Article, Law
from app.services.embedding_service import EmbeddingService, VoyageProvider
from scripts import regenerate_embeddings as regen

EMPREINTE = "doublure|actuelle"
DIM = EmbeddingService.EMBEDDING_DIM


def _vecteur(graine: int) -> np.ndarray:
    v = np.random.default_rng(graine).normal(size=DIM)
    return (v / np.linalg.norm(v)).astype(np.float32)


@pytest.fixture
def base(sync_db_session):
    """Un article dans chacun des etats possibles."""
    session = sync_db_session
    session.add(Law(id=1, reference="LOI-REGEN", title="Loi", content="Contenu.",
                    type="loi", language="fr", status="published"))
    session.flush()

    etats = {
        1: (None, None, True),                        # jamais vectorise
        2: (_vecteur(2), "gemini|ancien", True),      # autre fournisseur
        3: (_vecteur(3), None, True),                 # origine inconnue
        4: (_vecteur(4), EMPREINTE, True),            # a jour
        5: (None, None, False),                       # ecarte par le raffineur
    }
    for article_id, (vecteur, empreinte, embed) in etats.items():
        session.add(Article(
            id=article_id, law_id=1, number=str(article_id),
            content=f"Article {article_id}, contenu de test.", order=article_id,
            embedding=None if vecteur is None else vecteur.tolist(),
            embedding_model=empreinte, embed=embed,
        ))
    session.commit()
    return session


def _ids(lot):
    return [r["id"] for r in lot]


class TestSelection:

    def test_refait_ce_qui_manque_ou_vient_d_ailleurs(self, base):
        """
        L'article 3 est le cas qui justifie IS DISTINCT FROM : avec `<>`, son
        empreinte NULL rendrait la condition NULL, et ce vecteur d'origine
        inconnue passerait pour bon.
        """
        lot = regen._fetch_batch(base, None, False, EMPREINTE, 0, 100)

        assert _ids(lot) == [1, 2, 3]

    def test_le_compte_annonce_suit_la_meme_selection(self, base):
        assert regen._count_remaining(base, None, False, EMPREINTE) == 3

    def test_force_reprend_tout_ce_qui_est_vectorisable(self, base):
        lot = regen._fetch_batch(base, None, True, EMPREINTE, 0, 100)

        assert _ids(lot) == [1, 2, 3, 4]
        assert regen._count_remaining(base, None, True, EMPREINTE) == 4

    def test_un_autre_fournisseur_voit_tout_a_refaire(self, base):
        """Le changement de fournisseur, vu du script : rien n'est a jour."""
        assert regen._count_remaining(base, None, False, "voyage|voyage-4-large") == 4


class TestEcriture:

    def test_le_vecteur_part_avec_son_empreinte(self, base):
        vecteurs = [_vecteur(10 + i) for i in range(3)]

        regen._write_batch(base, [1, 2, 3], vecteurs, EMPREINTE)
        base.commit()

        rows = base.execute(text(
            "SELECT id, embedding_model FROM articles WHERE embed ORDER BY id"
        )).all()
        assert [(r.id, r.embedding_model) for r in rows] == [
            (i, EMPREINTE) for i in (1, 2, 3, 4)
        ]
        relu = np.array(base.get(Article, 1).embedding, dtype=np.float32)
        # halfvec = fp16, environ 3 chiffres significatifs : ecart maximal
        # mesure de 3e-5 sur un vecteur unitaire de 1024. 1e-6 echouerait.
        assert np.allclose(relu, vecteurs[0], atol=1e-4)
        assert float(relu @ vecteurs[0]) > 0.9999
        assert regen._count_remaining(base, None, False, EMPREINTE) == 0

    def test_un_vecteur_d_une_autre_dimension_est_refuse_par_la_colonne(self, base):
        """Le typmod de la colonne garde la dimension, au caractere pres."""
        court = np.ones(768, dtype=np.float32) / np.sqrt(768)

        with pytest.raises(Exception, match="expected 1024 dimensions"):
            regen._write_batch(base, [1], [court], EMPREINTE)
            base.commit()
        base.rollback()


class _Fournisseur:
    label = "doublure"
    fingerprint = EMPREINTE
    inter_batch_delay_s = 0.0


class _Service:
    """Doublure d'EmbeddingService : aucun modele, aucun reseau."""

    MAX_TEXT_LENGTH = EmbeddingService.MAX_TEXT_LENGTH
    EMBEDDING_DIM = DIM
    encodes = []

    def __init__(self, *args, **kwargs):
        self.provider = _Fournisseur()

    def generate_batch_embeddings(self, texts, batch_size=None, normalize=True):
        _Service.encodes.extend(texts)
        return [_vecteur(len(t)) for t in texts]


class TestDeBoutEnBout:

    @pytest.fixture(autouse=True)
    def _doublure(self, monkeypatch):
        _Service.encodes = []
        monkeypatch.setattr(regen, "EmbeddingService", _Service)

    def test_une_passe_suffit_et_la_suivante_ne_trouve_rien(self, base):
        assert regen.main(["--all", "--batch-size", "2"]) == 0

        assert len(_Service.encodes) == 3
        assert regen._count_remaining(base, None, False, EMPREINTE) == 0

        _Service.encodes = []
        assert regen.main(["--all"]) == 0
        assert _Service.encodes == []

    def test_le_compte_a_blanc_n_ecrit_rien(self, base):
        assert regen.main(["--all", "--dry-run"]) == 0

        assert _Service.encodes == []
        assert regen._count_remaining(base, None, False, EMPREINTE) == 3

    def test_une_cle_refusee_arrete_tout_au_premier_lot(self, base, monkeypatch):
        """
        Sans cet arret, une cle revoquee envoyait des milliers de requetes
        vouees au meme refus, un lot apres l'autre.
        """
        from app.services.embedding_service import EmbeddingDefinitiveError

        appels = []

        def _refus(self, texts, **kwargs):
            appels.append(len(texts))
            raise EmbeddingDefinitiveError("Voyage HTTP 401 : Provided API key is invalid.", code=401)

        monkeypatch.setattr(_Service, "generate_batch_embeddings", _refus)

        assert regen.main(["--all", "--batch-size", "1"]) == 2
        assert len(appels) == 1

    def test_un_arret_apres_des_lots_commites_vide_quand_meme_le_cache(self, base, monkeypatch):
        from app.services.embedding_service import EmbeddingDefinitiveError

        appels = []

        def _refus_au_second(self, texts, **kwargs):
            appels.append(len(texts))
            if len(appels) == 2:
                raise EmbeddingDefinitiveError("Voyage HTTP 401 : revoquee", code=401)
            return [_vecteur(len(t)) for t in texts]

        monkeypatch.setattr(_Service, "generate_batch_embeddings", _refus_au_second)
        base.execute(text(
            "INSERT INTO query_cache (cache_key, response_json, expires_at) "
            "VALUES ('k', '{}', now() + interval '1 hour')"
        ))
        base.commit()

        assert regen.main(["--all", "--batch-size", "1"]) == 2

        assert base.execute(text("SELECT count(*) FROM query_cache")).scalar() == 0

    def test_par_defaut_les_lots_sont_ceux_du_fournisseur(self, base, monkeypatch):
        """Des lots de 16 enverraient environ 2 000 requetes a Voyage pour le corpus."""
        tailles = []

        def _compter(self, texts, batch_size=None, **kwargs):
            tailles.append((len(texts), batch_size))
            return [_vecteur(len(t)) for t in texts]

        monkeypatch.setattr(_Fournisseur, "TEXTES_PAR_REQUETE", 2, raising=False)
        monkeypatch.setattr(_Service, "generate_batch_embeddings", _compter)

        assert regen.main(["--all"]) == 0
        assert tailles == [(2, 2), (1, 2)]

    def test_une_colonne_d_une_autre_dimension_arrete_avant_tout_appel(self, base, monkeypatch):
        """Sur une base non migree, chaque lot serait paye a Voyage, puis refuse."""
        monkeypatch.setattr(_Service, "EMBEDDING_DIM", 768)

        assert regen.main(["--all"]) == 2
        assert _Service.encodes == []

    def test_un_lot_refuse_pour_son_texte_est_saute(self, base, monkeypatch):
        """Un 400 vise un texte : les autres lots continuent."""
        from app.services.embedding_service import EmbeddingDefinitiveError

        appels = []

        def _refus_du_premier(self, texts, **kwargs):
            appels.append(len(texts))
            if len(appels) == 1:
                raise EmbeddingDefinitiveError("Voyage HTTP 400 : bad input", code=400)
            return [_vecteur(len(t)) for t in texts]

        monkeypatch.setattr(_Service, "generate_batch_embeddings", _refus_du_premier)

        assert regen.main(["--all", "--batch-size", "1"]) == 1
        assert len(appels) == 3
        assert regen._count_remaining(base, None, False, EMPREINTE) == 1

    def test_une_passe_vide_le_cache_des_recherches(self, base):
        """Les reponses gardees pendant que les vecteurs manquaient etaient du plein texte seul."""
        base.execute(text(
            "INSERT INTO query_cache (cache_key, response_json, expires_at) "
            "VALUES ('k', '{}', now() + interval '1 hour')"
        ))
        base.commit()

        assert regen.main(["--all"]) == 0

        assert base.execute(text("SELECT count(*) FROM query_cache")).scalar() == 0


def _index_embedding(session):
    return session.execute(text(
        "SELECT indexrelid::regclass::text, indisvalid FROM pg_index "
        "WHERE indrelid = 'articles'::regclass "
        "AND indexrelid::regclass::text LIKE 'idx_articles_embedding_hnsw%'"
    )).all()


class TestReconstructionDeLIndex:

    def test_le_nom_est_celui_de_la_migration(self):
        assert regen.INDEX_EMBEDDING == "idx_articles_embedding_hnsw_halfvec1024"

    def test_index_reconstruit_et_valide(self, base):
        regen.reindex(base)

        assert _index_embedding(base) == [(regen.INDEX_EMBEDDING, True)]

    def test_echec_ne_laisse_pas_de_copie_invalide(self, base, monkeypatch):
        """
        Un REINDEX CONCURRENTLY interrompu laisse une copie _ccnew invalide,
        maintenue a chaque ecriture. On la simule, puis on fait echouer.
        """
        with regen.sync_engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(
                f"CREATE INDEX {regen.INDEX_EMBEDDING}_ccnew ON articles (id)"
            ))
        monkeypatch.setattr(
            regen, "text",
            lambda sql: text(sql.replace("REINDEX INDEX CONCURRENTLY", "REINDEX INDEX CONCURRENTLY inexistant_")
                             if sql.startswith("REINDEX") else sql),
        )

        with pytest.raises(Exception):
            regen.reindex(base)

        assert _index_embedding(base) == [(regen.INDEX_EMBEDDING, True)]


# ==================== CHARGEMENT DES LOTS DEJA CALCULES ====================


def _texte(article_id):
    """Le texte vectorise de l'article : coalesce(embed_text, content)."""
    return f"Article {article_id}, contenu de test."


def _instantane(dossier, textes):
    chemin = dossier / "morceaux.jsonl"
    with open(chemin, "w", encoding="utf-8") as f:
        for article_id, texte in textes.items():
            f.write(json.dumps({"id": article_id, "law_id": 1, "texte": texte}, ensure_ascii=False) + "\n")
    return chemin


def _lots(dossier, vecteurs, nom="lot"):
    dossier.mkdir(parents=True, exist_ok=True)
    ids = sorted(vecteurs)
    np.savez(dossier / f"{nom}.npz", ids=np.array(ids, dtype=np.int64),
             v=np.array([vecteurs[i] for i in ids], dtype=np.float32))
    return dossier


@pytest.fixture
def lots(base, tmp_path):
    """
    Un cas par motif : 1 a charger ; 2 vecteur non unitaire ; 3 texte modifie
    depuis l'instantane ; 4 deja a jour ; 5 non vectorisable ; 99 absent de la
    base ; 77 absent de l'instantane.
    """
    textes = {i: _texte(i) for i in (1, 2, 4, 5, 99)}
    textes[3] = "Ancienne version de l'article 3."
    vecteurs = {i: _vecteur(100 + i) for i in (1, 3, 4, 5, 77, 99)}
    vecteurs[2] = _vecteur(102) * 0.5
    dossier = _lots(tmp_path / "voyage-4-large", vecteurs)
    return dossier, _instantane(tmp_path, textes), vecteurs


class TestChargementDesLots:

    def test_chaque_motif_est_compte_et_seul_le_bon_est_charge(self, base, lots):
        dossier, instantane, vecteurs = lots

        rapport = regen.charger_les_lots(base, dossier, instantane, EMPREINTE, DIM)

        assert dict(rapport) == {
            "charges": 1,
            "vecteur_invalide": 1,
            "texte_modifie_depuis_l_instantane": 1,
            "deja_a_jour": 1,
            "non_vectorisable": 1,
            "absent_de_la_base": 1,
            "absent_de_l_instantane": 1,
        }
        etat = {r.id: r.embedding_model for r in base.execute(text(
            "SELECT id, embedding_model FROM articles ORDER BY id"
        ))}
        assert etat == {1: EMPREINTE, 2: "gemini|ancien", 3: None, 4: EMPREINTE, 5: None}
        relu = np.array(base.get(Article, 1).embedding, dtype=np.float32)
        assert np.allclose(relu, vecteurs[1], atol=1e-4)

    def test_le_compte_a_blanc_n_ecrit_rien(self, base, lots):
        dossier, instantane, _ = lots

        rapport = regen.charger_les_lots(base, dossier, instantane, EMPREINTE, DIM, dry_run=True)

        assert rapport["a_charger"] == 1
        assert rapport["a_charger_qui_manquaient"] == 1
        assert rapport["charges"] == 0
        assert base.execute(text("SELECT embedding IS NULL FROM articles WHERE id = 1")).scalar()

    def test_le_compte_a_blanc_avec_force_distingue_ce_qui_manquait(self, base, lots):
        """Avec --force, l'article 4, deja a jour, est recharge mais ne manquait pas."""
        dossier, instantane, _ = lots

        rapport = regen.charger_les_lots(
            base, dossier, instantane, EMPREINTE, DIM, force=True, dry_run=True,
        )

        assert rapport["a_charger"] == 2
        assert rapport["a_charger_qui_manquaient"] == 1

    def test_force_reecrit_aussi_ce_qui_est_a_jour(self, base, lots):
        dossier, instantane, vecteurs = lots

        rapport = regen.charger_les_lots(base, dossier, instantane, EMPREINTE, DIM, force=True)

        assert rapport["charges"] == 2
        relu = np.array(base.get(Article, 4).embedding, dtype=np.float32)
        assert np.allclose(relu, vecteurs[4], atol=1e-4)

    def test_une_reprise_ne_recharge_rien(self, base, lots):
        dossier, instantane, _ = lots
        regen.charger_les_lots(base, dossier, instantane, EMPREINTE, DIM)

        rapport = regen.charger_les_lots(base, dossier, instantane, EMPREINTE, DIM)

        assert rapport["charges"] == 0
        assert rapport["deja_a_jour"] == 2

    def test_un_id_present_dans_deux_lots_est_ecarte(self, base, tmp_path):
        dossier = tmp_path / "voyage-4-large"
        _lots(dossier, {1: _vecteur(1)}, nom="a")
        _lots(dossier, {1: _vecteur(2)}, nom="b")
        instantane = _instantane(tmp_path, {1: _texte(1)})

        rapport = regen.charger_les_lots(base, dossier, instantane, EMPREINTE, DIM)

        assert rapport["id_en_double_dans_les_lots"] == 1
        assert rapport["charges"] == 0

    def test_un_lot_d_une_autre_dimension_arrete_tout(self, base, tmp_path):
        dossier = tmp_path / "voyage-4-large"
        dossier.mkdir()
        np.savez(dossier / "lot.npz", ids=np.array([1], dtype=np.int64),
                 v=np.ones((1, 768), dtype=np.float32))
        instantane = _instantane(tmp_path, {1: _texte(1)})

        with pytest.raises(SystemExit, match="768"):
            regen.charger_les_lots(base, dossier, instantane, EMPREINTE, DIM)


class TestChargementParLaLigneDeCommande:

    def test_l_empreinte_ecrite_est_celle_du_fournisseur(self, base, lots):
        """
        Calculee par le code, jamais recopiee : un caractere d'ecart, et les
        vecteurs charges passeraient pour etrangers.
        """
        dossier, instantane, _ = lots

        assert regen.main(["--depuis-lots", str(dossier), "--instantane", str(instantane)]) == 0

        empreinte = base.execute(text("SELECT embedding_model FROM articles WHERE id = 1")).scalar()
        assert empreinte == VoyageProvider(api_key="factice").fingerprint == "voyage|voyage-4-large"

    def test_le_chargement_vide_le_cache_des_recherches(self, base, lots):
        dossier, instantane, _ = lots
        base.execute(text(
            "INSERT INTO query_cache (cache_key, response_json, expires_at) "
            "VALUES ('k', '{}', now() + interval '1 hour')"
        ))
        base.commit()

        assert regen.main(["--depuis-lots", str(dossier), "--instantane", str(instantane)]) == 0

        assert base.execute(text("SELECT count(*) FROM query_cache")).scalar() == 0

    def test_l_instantane_est_exige(self, base, lots):
        dossier, _, _ = lots
        assert regen.main(["--depuis-lots", str(dossier)]) == 2

    def test_refuse_si_le_fournisseur_n_est_pas_voyage(self, base, lots, monkeypatch):
        dossier, instantane, _ = lots
        monkeypatch.setattr(regen.settings, "EMBEDDING_PROVIDER", "gemini")

        assert regen.main(["--depuis-lots", str(dossier), "--instantane", str(instantane)]) == 2

    def test_refuse_un_dossier_d_un_autre_modele(self, base, lots, tmp_path):
        """Des vecteurs voyage-context-4 ne doivent pas passer pour voyage-4-large."""
        dossier, instantane, _ = lots
        autre = dossier.rename(tmp_path / "voyage-context-4")

        assert regen.main(["--depuis-lots", str(autre), "--instantane", str(instantane)]) == 2
        assert base.execute(text("SELECT embedding IS NULL FROM articles WHERE id = 1")).scalar()

    def test_instantane_sans_depuis_lots_refuse(self, base, tmp_path):
        instantane = _instantane(tmp_path, {1: _texte(1)})
        assert regen.main(["--all", "--instantane", str(instantane)]) == 2
