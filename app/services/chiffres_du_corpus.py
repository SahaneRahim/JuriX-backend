"""
Les chiffres du corpus : combien de textes et d'articles, de quelles sources,
le plus recent, et a quelles dates.

POURQUOI. L'accueil du chat dit son perimetre (« Je réponds à partir de 2 275
textes officiels camerounais et OHADA… ; je ne connais ni la jurisprudence ni
votre dossier »), et la page « Aucun résultat » de la recherche montre ce que
contient JuriX. Ecrits en dur dans le front, ces chiffres se perimaient a chaque
ingestion. Ils viennent desormais de la base, gardes en memoire.

COUT. Quatre instructions, une fois toutes les dix minutes au plus ; entre deux,
rien. Comme la liste des categories, un seul processus sert l'API.
"""

import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.law import Article, Law
from app.schemas.corpus import CorpusResponse, NatureDuCorpus, SourceDuCorpus
from app.services.liens_entre_textes import RELEVE_LE
from app.services.statut_juridique import InfoTexte, texte_lie

# Dix minutes : les chiffres ne bougent qu'a l'ingestion, et un compte en
# retard de dix minutes ne trompe personne.
_DUREE_CACHE_S = 600.0
_cache: Optional[Tuple[float, CorpusResponse]] = None

# Familles de textes, d'apres la cote en base. « PRC-{doc_id} » est la cote
# que l'extraction donne aux documents de prc.cm (scripts/extraire_corpus.py,
# Document.cote) ; les textes ajoutes a part portent une cote parlante
# (« OHADA-AUS-2010 », « CIMA-CODE-2019 », « LOI-1992-007 », « CODE-CIVIL »).
_SOURCES = {
    "prc": {
        "fr": "Présidence de la République du Cameroun (prc.cm)",
        "en": "Presidency of the Republic of Cameroon (prc.cm)",
    },
    "ohada": {
        "fr": "Actes uniformes et traités de l'OHADA",
        "en": "OHADA Uniform Acts and treaties",
    },
    "cima": {
        "fr": "Code des assurances de la CIMA",
        "en": "CIMA Insurance Code",
    },
    "autres": {
        "fr": "Codes et lois de base camerounais ajoutés à part",
        "en": "Cameroonian codes and core laws added separately",
    },
}

# Un article decoupe en morceaux (« 55.1 », « 55.2 ») ne compte qu'une fois :
# les morceaux au-dela du premier sont exclus. Un numero « 1.2 » de
# numerotation hierarchique serait exclu a tort ; mesure le 2026-10-28 sur la
# base locale, les morceaux « .2 » et au-dela sont 2 173 sur 29 330 morceaux de
# type article, presque tous des suites (chunk_refiner les produit ainsi).
_SUITE_DE_MORCEAU = r"\.([2-9]|[1-9][0-9]+)$"


def invalider_cache_corpus() -> None:
    """Oublie les chiffres en cache : la prochaine lecture repasse par la base."""
    global _cache
    _cache = None


def _libelle_sources(familles: List[str]) -> Dict[str, str]:
    """« textes officiels camerounais et OHADA », selon les familles presentes."""
    fr = ["textes officiels camerounais"]
    en = ["official texts of Cameroon"]
    if "ohada" in familles:
        fr.append("OHADA")
        en.append("OHADA")
    if "cima" in familles:
        fr.append("CIMA")
        en.append("CIMA")

    def _joindre(mots: List[str], et: str) -> str:
        return mots[0] if len(mots) == 1 else ", ".join(mots[:-1]) + f" {et} " + mots[-1]

    return {"fr": _joindre(fr, "et"), "en": _joindre(en, "and")}


async def _calculer(db: AsyncSession) -> CorpusResponse:
    publie = Law.status == "published"
    textes, dernier_ajout = (
        await db.execute(select(func.count(Law.id), func.max(Law.created_at)).where(publie))
    ).one()

    articles = (
        await db.execute(
            select(func.count(Article.id))
            .join(Law, Law.id == Article.law_id)
            .where(
                publie,
                # NULL : morceau jamais classe par chunk_refiner, compte comme article.
                func.coalesce(Article.kind, "article") == "article",
                Article.number.op("!~")(_SUITE_DE_MORCEAU),
            )
        )
    ).scalar_one()

    # Natures et familles en une instruction : le prefixe de la cote se lit en SQL.
    prefixe = func.upper(func.split_part(Law.reference, "-", 1))
    famille = case(
        (prefixe == "PRC", "prc"),
        (prefixe == "OHADA", "ohada"),
        (prefixe == "CIMA", "cima"),
        else_="autres",
    )
    lignes = (
        await db.execute(
            select(Law.type, famille.label("famille"), func.count(Law.id))
            .where(publie)
            .group_by(Law.type, famille)
        )
    ).all()
    par_nature: Dict[str, int] = {}
    par_famille: Dict[str, int] = {}
    for nature, code, nombre in lignes:
        par_nature[nature] = par_nature.get(nature, 0) + nombre
        par_famille[code] = par_famille.get(code, 0) + nombre

    recent = (
        await db.execute(
            select(Law.id, Law.reference, Law.type, Law.title, Law.publication_date, Law.status)
            .where(publie, Law.publication_date.is_not(None))
            .order_by(Law.publication_date.desc(), Law.id.desc())
            .limit(1)
        )
    ).one_or_none()
    texte_le_plus_recent = None
    if recent is not None:
        info = InfoTexte(*recent)
        texte_le_plus_recent = texte_lie(info.reference, info.titre, info.date, {info.reference: info})

    familles = [code for code in _SOURCES if par_famille.get(code)]
    return CorpusResponse(
        textes=textes,
        articles=articles,
        textes_par_nature=[
            NatureDuCorpus(nature=nature, textes=nombre)
            for nature, nombre in sorted(par_nature.items(), key=lambda item: (-item[1], item[0]))
        ],
        texte_le_plus_recent=texte_le_plus_recent,
        dernier_ajout_le=dernier_ajout.date() if dernier_ajout else None,
        liens_releves_le=RELEVE_LE,
        sources=[
            SourceDuCorpus(code=code, libelle=_SOURCES[code], textes=par_famille[code])
            for code in familles
        ],
        libelle_sources=_libelle_sources(familles),
        calcule_le=datetime.now(timezone.utc),
    )


async def chiffres_du_corpus(db: AsyncSession) -> CorpusResponse:
    """Les chiffres du corpus, depuis le cache s'il a moins de dix minutes."""
    global _cache
    entree = _cache
    maintenant = time.monotonic()
    if entree is not None and maintenant - entree[0] < _DUREE_CACHE_S:
        return entree[1]
    chiffres = await _calculer(db)
    # Horodate au DEBUT du calcul : la donnee a cet age-la.
    _cache = (maintenant, chiffres)
    return chiffres
