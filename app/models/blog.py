"""
Articles du blog juridique : des décryptages d'actualité, rédigés par le modèle
à partir des textes de la base, CONTRÔLÉS par le modèle phrase à phrase et
RELUS par un administrateur avant publication.

LE MODÈLE SUIT LA MIGRATION 017, ET NON L'INVERSE. La table de production a été
créée par alembic/versions/017_blog_posts_table.py (SQL brut). La première
version de ce modèle en différait : `published_at` en DateTime (la table a une
DATE), `suggested_question` en Text NOT NULL (la table a un VARCHAR(500)
nullable), des index `ix_blog_posts_*` (la table a `idx_blog_posts_*`). Rien ne
cassait à l'exécution, mais `alembic revision --autogenerate` aurait proposé de
changer la colonne et de supprimer les index de production, et une base créée
par `create_all` (jurix_dev) n'avait pas le schéma de production : un test
local ne prouvait rien. Les colonnes, leurs défauts et leurs index sont donc
recopiés de 017 ; tests/test_migrations/test_blog_posts_017.py les compare.

LE DOSSIER DE VÉRIFICATION, DANS LA COLONNE JSONB `citations` (tour 3). Le
contrôle IA et la relecture humaine de chaque partie (décision du propriétaire,
30/10/2026) doivent être enregistrés, et 017 n'a pas de colonne pour eux. Une
colonne de plus imposait une migration, et la production refuse de démarrer
sur une révision qu'elle ne connaît pas (panne du 28/10). La colonne JSONB
`citations` porte donc, pour un article écrit depuis le tour 3, un OBJET :

    {
      "citations":   [...],   # les citations écrites par le SERVEUR (voir
                              # blog_generator_service.citations_verifiees)
      "sources":     [...],   # les extraits lus par le modèle à la génération,
                              # ou reconstruits par POST /{id}/sources
                              # ({n, law_id, law_title, article_number,
                              # article_id, suite, extrait}) : le contrôle IA
                              # juge contre eux, sous les mêmes numéros, et
                              # l'administration les montre (tour 4)
      "controle_ia": null | {fait_le, modele, empreinte, phrases: [...],
                             empreintes_des_parties, extraits_ajoutes,
                             appels},  # appels : tour 5, la durée de chacun
      "relecture":   {"parties": {partie: empreinte attestée | null}, "par",
                      "par_id", "le", "controle",  # controle : tour 5, le
                                                   # fait_le du contrôle relu
                      "phrases_confirmees": {id: empreinte attestée}},
                                                   # tour 7 : les phrases de
                                                   # règle confirmées
      "deja_publie": true  # tour 7 : publié au moins une fois (son adresse a
                           # pu circuler) ; absent sinon
    }

Un article d'AVANT (les trois lignes du 26/10 de jurix_dev) y garde une LISTE :
ses citations, rien d'autre ; sans sources, il n'est jamais publiable
(« sources_absentes »), donc jamais servi au public. Les deux formes sont
lues ; une écriture par l'API passe à l'objet. L'attribut Python
`verification` est la colonne ; `citations`, `sources`, `controle_ia` et
`relecture` en sont les parties. Chaque écriture range un NOUVEL objet (copie
profonde) : un JSONB modifié en place n'est pas vu par SQLAlchemy, et
l'écriture serait perdue.
"""

import copy
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func

from app.core.database import Base


class BlogPost(Base):
    __tablename__ = "blog_posts"

    id = Column(Integer, primary_key=True)
    # UNIQUE dans 017 : la contrainte blog_posts_slug_key, que l'API traduit
    # en 409 « slug_pris » (routes/blog.py).
    slug = Column(String(255), nullable=False, unique=True)
    title = Column(String(500), nullable=False)
    category = Column(String(100), nullable=False)
    # Une DATE, comme en production : jour de création du brouillon, puis jour
    # de sa publication (PUT is_published=true le remet à la date du jour).
    published_at = Column(Date, nullable=False, server_default=func.current_date())
    reading_time = Column(String(50), nullable=False, server_default=text("'4 min'"))

    # L'actualité en deux minutes : le contexte.
    context = Column(Text, nullable=False)

    # Ce que dit la loi : l'analyse.
    summary = Column(Text, nullable=False)
    key_points = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    # La colonne `citations` de 017 : le dossier de vérification (voir plus haut).
    verification = Column("citations", JSONB, nullable=False, server_default=text("'[]'::jsonb"))
    citizen_impact = Column(Text, nullable=False)

    # La question à poser au chat, et les mots-clés. Nullable dans 017 : la
    # longueur (500) et la présence sont tenues par les schémas de l'API.
    suggested_question = Column(String(500))
    tags = Column(JSONB, nullable=False, server_default=text("'[]'::jsonb"))

    # TRUE par défaut dans 017. L'API, elle, écrit toujours la valeur : un
    # article généré ou créé à la main naît brouillon, et seul un PUT le publie.
    is_published = Column(Boolean, nullable=False, server_default=text("true"))
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now()
    )

    # Les index de 017, sous leurs noms. Le tri du blog public (publiés, du plus
    # récent au plus ancien) est celui de idx_blog_posts_published.
    __table_args__ = (
        Index("idx_blog_posts_slug", "slug"),
        Index("idx_blog_posts_category", "category"),
        Index("idx_blog_posts_published", "is_published", published_at.desc()),
    )

    # ------------------------------------------------ le dossier de vérification

    def _partie(self, nom: str) -> Any:
        """Une partie du dossier, en COPIE profonde ; une liste d'avant n'a que ses citations."""
        brut = self.verification
        if isinstance(brut, dict):
            return copy.deepcopy(brut.get(nom))
        if nom == "citations" and isinstance(brut, list):
            return copy.deepcopy(brut)
        return None

    def _ranger(self, **parties: Any) -> None:
        """
        Range un NOUVEL objet : l'ancien n'est jamais modifié en place. Un
        article d'avant (une LISTE) qui passe à l'objet garde la trace d'avoir
        pu être publié (deja_publie) : 017 publiait par défaut.
        """
        brut = self.verification
        if isinstance(brut, dict):
            dossier = copy.deepcopy(brut)
        elif isinstance(brut, list):
            dossier = {"citations": copy.deepcopy(brut), "deja_publie": True}
        else:
            dossier = {"citations": []}
        dossier.update(copy.deepcopy(parties))
        dossier.setdefault("citations", [])
        self.verification = dossier

    @property
    def citations(self) -> List[Dict[str, Any]]:
        liste = self._partie("citations")
        return [c for c in liste if isinstance(c, dict)] if isinstance(liste, list) else []

    @citations.setter
    def citations(self, valeur: Sequence[Dict[str, Any]]) -> None:
        self._ranger(citations=list(valeur or []))

    @property
    def sources(self) -> List[Dict[str, Any]]:
        liste = self._partie("sources")
        return [s for s in liste if isinstance(s, dict)] if isinstance(liste, list) else []

    @sources.setter
    def sources(self, valeur: Sequence[Dict[str, Any]]) -> None:
        self._ranger(sources=list(valeur or []))

    @property
    def controle_ia(self) -> Optional[Dict[str, Any]]:
        controle = self._partie("controle_ia")
        return controle if isinstance(controle, dict) else None

    @controle_ia.setter
    def controle_ia(self, valeur: Optional[Dict[str, Any]]) -> None:
        self._ranger(controle_ia=valeur)

    @property
    def relecture(self) -> Dict[str, Any]:
        """La relecture TELLE QU'ENREGISTRÉE (empreintes relues) ; voir controle_du_billet."""
        relecture = self._partie("relecture")
        return relecture if isinstance(relecture, dict) else {}

    @relecture.setter
    def relecture(self, valeur: Dict[str, Any]) -> None:
        self._ranger(relecture=valeur)

    @property
    def deja_publie(self) -> bool:
        """
        Publié au moins une fois (tour 7) : en ligne, ou noté quand il l'a été
        (noter_publie). Un article d'avant (une LISTE) passe pour l'avoir été.
        POURQUOI : l'adresse d'un brouillon jamais publié suit son titre
        (routes/blog), et celle d'un article publié une fois a pu circuler.
        """
        if self.is_published:
            return True
        brut = self.verification
        if isinstance(brut, dict):
            return brut.get("deja_publie") is True
        return isinstance(brut, list)

    def noter_publie(self) -> None:
        """
        Note dans le dossier que l'article a été publié. Un article d'avant (une
        LISTE) passe déjà pour l'avoir été : il garde sa forme (le retrait seul
        ne touche à rien d'autre).
        """
        brut = self.verification
        if isinstance(brut, dict) and brut.get("deja_publie") is not True:
            self._ranger(deja_publie=True)

    # ------------------------------------------------------------ la forme du front

    def to_dict(self, publique: bool = False, valides: Optional[Sequence[bool]] = None) -> dict:
        """
        La forme que lit le front (blog/+page.ts, blog/[slug]/+page.ts et
        l'administration), plus `verifie` (booléen) sur CHAQUE citation : vrai
        pour une citation construite par le serveur et toujours portée par la
        base, faux sinon. Le contrôle IA, la relecture et l'état de publication
        sont ajoutés par la route (routes/blog._posts_json), qui sait les juger,
        aux réponses d'administration seulement (avenant du tour 7).

        `valides` : pour chaque citation, ce que la base en dit AUJOURD'HUI
        (routes/blog._validite_des_citations : texte toujours publié, article
        toujours là, extrait toujours le sien). None : seul `verifie` compte.

        `publique` : la réponse des routes publiques. Une citation dont
        verifie est faux n'y garde QUE sa référence et son numéro, en texte
        brut, sans extrait ni lien (contrat du tour 2 avec le front). Depuis le
        tour 3, une telle citation ne rend plus l'article publiable : ce n'est
        qu'une défense de plus. La cote interne (`law_reference`) n'y est pas :
        le lecteur ne peut rien en faire (revue du tour 2).
        """
        citations = []
        for rang, citation in enumerate(self.citations):
            verifie = citation.get("verifie") is True and (
                valides is None or (rang < len(valides) and valides[rang])
            )
            if publique and not verifie:
                citations.append({
                    "reference": str(citation.get("reference") or ""),
                    "article_number": str(citation.get("article_number") or ""),
                    "verifie": False,
                })
            elif publique:
                citations.append({
                    **{cle: valeur for cle, valeur in citation.items() if cle != "law_reference"},
                    "verifie": True,
                })
            else:
                citations.append({**citation, "verifie": verifie})
        return {
            "id": self.id,
            "slug": self.slug,
            "title": self.title,
            "category": self.category,
            # strftime et non isoformat : une base créée par create_all
            # (jurix_dev) a encore un timestamptz à cette place.
            "publishedAt": self.published_at.strftime("%Y-%m-%d") if self.published_at else "",
            "readingTime": self.reading_time,
            "context": self.context,
            "botAnalysis": {
                "summary": self.summary,
                "keyPoints": list(self.key_points or []),
                "citations": citations,
                "citizenImpact": self.citizen_impact,
            },
            "suggestedQuestion": self.suggested_question or "",
            "tags": list(self.tags or []),
            "isPublished": self.is_published,
        }
