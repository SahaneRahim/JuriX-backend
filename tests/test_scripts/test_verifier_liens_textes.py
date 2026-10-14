"""
Le controle en base des liens entre textes successifs.

Un nouveau decoupage peut renumeroter un morceau (« SECTION_2 » est un
compteur) ou deplacer un article : le lien se tait, ou se pose ailleurs, sans
qu'aucun test unitaire ne le voie. Ces tests montrent que le controle le voit.

Usage:
    pytest tests/test_scripts/test_verifier_liens_textes.py -v
"""

from datetime import date

import pytest

from app.models.law import Article, Law
from app.services.textes_successifs import Lien
from scripts.maintenance import verifier_liens_textes as script


@pytest.fixture
def base(sync_db_session):
    """Un texte ancien dont l'article 6 finit sous « SECTION_2 », et le texte qui le modifie."""
    ancien = Law(reference="T-ANCIEN", title="Loi N°2000/001", type="loi", content="…",
                 language="fr", status="published")
    recent = Law(reference="T-RECENT", title="Loi N°2026/002", type="loi", content="…",
                 language="fr", status="published")
    sync_db_session.add_all([ancien, recent])
    sync_db_session.commit()
    morceaux = [
        (ancien, "6", "Le Président de la République est élu pour sept ans."),
        (ancien, "SECTION_2", "b) le Président de la République par intérim ne peut modifier la Constitution."),
        (ancien, "7", "Le Président nomme le Premier ministre. ARTICLE 8 : La Chambre comprend deux sections."),
        (recent, "1", "Les dispositions des articles 6 et 8 sont modifiées."),
        (recent, "6", "(nouveau) Le Vice-Président achève le mandat."),
        (recent, "2", "La présente loi sera publiée."),
    ]
    for ordre, (loi, numero, contenu) in enumerate(morceaux):
        sync_db_session.add(Article(law_id=loi.id, number=numero, content=contenu, order=ordre))
    sync_db_session.commit()
    return sync_db_session


def _lien(**kwargs) -> Lien:
    valeurs = dict(
        texte="T-ANCIEN", texte_nom="la Loi N°2000/001", nature="modifie",
        par_titre="la Loi N°2026/002 du 14 avril 2026", par_date=date(2026, 4, 14),
        par_reference="T-RECENT", articles=("6", "8"), par_articles=("1", "6"),
        morceaux=(("SECTION_2", "6", "par intérim"), ("7", "8", "ARTICLE 8")),
    )
    valeurs.update(kwargs)
    return Lien(**valeurs)


def test_un_lien_juste_passe_et_liste_ce_qu_il_marque(base, capsys):
    assert script.controler([_lien()], detail=True) == 0

    sortie = capsys.readouterr().out
    assert "ok" in sortie and "3 morceau(x) marque(s)" in sortie
    assert "texte recent : 1, 6" in sortie


def test_un_morceau_renumerote_est_signale(base, capsys):
    """Apres un nouveau decoupage, SECTION_2 porte un autre passage : le repere manque."""
    base.query(Article).filter(Article.number == "SECTION_2").update(
        {"content": "CHAPITRE II Du Gouvernement"}
    )
    base.commit()

    assert script.controler([_lien()]) == 1
    assert "morceau « SECTION_2 » sans son repere « par intérim »" in capsys.readouterr().out


def test_un_article_qu_aucun_morceau_ne_porte_est_signale(base, capsys):
    assert script.controler([_lien(articles=("6", "8", "9"))]) == 1
    assert "article « 9 » porte par aucun morceau" in capsys.readouterr().out


def test_un_texte_absent_est_un_defaut_un_texte_archive_une_note(base, capsys):
    """Les liens d'un texte archive sont saisis d'avance, pour sa remise en ligne."""
    base.query(Law).filter(Law.reference == "T-ANCIEN").update({"status": "archived"})
    base.commit()

    assert script.controler([_lien(), _lien(texte="T-INCONNU", morceaux=())]) == 1
    sortie = capsys.readouterr().out
    assert "(note) texte T-ANCIEN au statut « archived »" in sortie
    assert "texte T-INCONNU absent de la base" in sortie


def test_une_exception_sans_objet_est_signalee(base, capsys):
    assert script.controler([_lien(sauf=(("6", "nom patronymique"),))]) == 1
    assert "exception « 6 » / « nom patronymique » sans objet" in capsys.readouterr().out
