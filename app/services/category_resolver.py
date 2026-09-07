"""
Resolution d'un nom de domaine vers `categories.id`.

Ce module existe pour une seule raison : le pipeline ecrivait dans
`laws.category_id` un entier qui etait une position dans un dictionnaire Python,
pas une cle etrangere. La resolution se fait desormais par le NOM, sur
`lower(name)`, appuyee par l'index unique `uq_categories_name_lower` cree par la
migration d9e0f1a2b3c4.

Aucune creation automatique de ligne : c'est ainsi que le code et la base
divergent. Si un domaine manque, l'appelant l'apprend par une exception.

Author: JuriX Team
"""

import logging
from typing import Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.law import Category

logger = logging.getLogger(__name__)


class UnknownDomainError(LookupError):
    """Le domaine demande n'a pas de ligne dans `categories`."""

    def __init__(self, domain: str, available: List[str]):
        self.domain = domain
        self.available = available
        super().__init__(
            f"Domaine inconnu en base : {domain!r}. "
            f"Domaines presents ({len(available)}) : {', '.join(sorted(available))}"
        )


def load_domain_map(session: Session) -> Dict[str, int]:
    """Rend {nom_en_minuscules: id} pour toute la table."""
    rows = session.execute(select(Category.name, Category.id)).all()
    return {name.lower(): identifier for name, identifier in rows}


def load_domain_names(session: Session) -> List[str]:
    """Rend les noms tels qu'ils sont ecrits en base, pour les messages d'erreur."""
    return list(session.execute(select(Category.name)).scalars().all())


def resolve_domain_id(session: Session, domain: str) -> int:
    """
    Rend l'identifiant de la categorie portant ce nom.

    Raises:
        UnknownDomainError: si aucune ligne ne porte ce nom.
    """
    mapping = load_domain_map(session)
    resolved = mapping.get((domain or "").strip().lower())
    if resolved is None:
        raise UnknownDomainError(domain, load_domain_names(session))
    return resolved


def try_resolve_domain_id(session: Session, domain: str) -> Optional[int]:
    """
    Variante non levante : rend None et journalise si le domaine manque.

    Le pipeline d'ingestion ne l'appelle PAS, contrairement a ce que ce
    docstring affirmait : il charge la table entiere par `load_domain_map`
    parce qu'il a aussi besoin des identifiants des domaines suggeres, et il
    trace lui-meme l'absence sur la loi. Cette fonction reste l'entree
    naturelle pour un appelant qui ne resout qu'UN nom et ne veut pas gerer
    l'exception — un futur endpoint d'administration, par exemple.
    """
    try:
        return resolve_domain_id(session, domain)
    except UnknownDomainError as exc:
        logger.error("Resolution de categorie impossible : %s", exc)
        return None
