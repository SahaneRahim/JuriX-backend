"""
Modèles SQLAlchemy pour les lois, articles et catégories.

Ce module définit les modèles de base de données pour le système JuriX:
- Law: Documents juridiques (lois, décrets, ordonnances)
- Article: Articles individuels d'une loi
- Category: Catégories juridiques (12 catégories camerounaises)

Features v2.1:
- Détection automatique de langue (language, language_confidence)
- Suggestions de catégories (suggested_categories, category_confidence)
- Support pgvector pour embeddings sémantiques

Author: JuriX Team
"""

from datetime import datetime, date
from typing import List, Optional

from sqlalchemy import (
    text,
    Boolean,
    Column,
    Integer,
    String,
    Text,
    Float,
    Date,
    DateTime,
    ForeignKey,
    Index,
    JSON,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector

from app.core.database import Base


# NOTE: un adaptateur get_array_type() existait ici pour choisir ARRAY sous
# PostgreSQL et JSON ailleurs. Il n'a jamais ete appele — la colonne etait
# declaree en JSON en dur alors que la migration 001 la cree en integer[],
# donc toute ecriture de suggested_categories echouait en DatatypeMismatchError
# ("column is of type integer[] but expression is of type json").
# Le projet ne cible que PostgreSQL : le type est desormais declare directement.


class Category(Base):
    """
    Modèle pour les catégories juridiques.

    Représente les 12 catégories de droit camerounais:
    1. Droit Constitutionnel
    2. Droit Civil
    3. Droit Pénal
    4. Droit Commercial OHADA
    5. Droit du Travail
    6. Droit Fiscal
    7. Droit Administratif
    8. Droit Foncier
    9. Droit de la Famille
    10. Droit de l'Environnement
    11. Droit International
    12. Droit des Affaires

    Attributes:
        id: Identifiant unique
        name: Nom de la catégorie
        description: Description détaillée
        created_at: Date de création
        laws: Relation vers les lois de cette catégorie
    """

    __tablename__ = "categories"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Fields
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    icon = Column(String(10), nullable=True)  # Emoji icon for category
    display_order = Column(Integer, nullable=False, default=0)  # Order for display

    # Metadata
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    laws = relationship("Law", back_populates="category", lazy="selectin")

    def __repr__(self) -> str:
        return f"<Category(id={self.id}, name='{self.name}')>"


class Law(Base):
    """
    Modèle pour les documents juridiques (lois).

    Représente un document juridique complet (loi, décret, ordonnance, arrêté)
    avec ses métadonnées, sa catégorisation et ses articles.

    Features v2.1:
    - Détection automatique de langue
    - Suggestions de catégories automatiques
    - Scores de confiance pour détection/classification

    Attributes:
        id: Identifiant unique
        reference: Référence officielle (ex: "LOI-2024-001")
        title: Titre de la loi
        type: Type de document (loi, décret, ordonnance, arrêté)
        content: Contenu complet de la loi
        language: Langue détectée ou définie (fr, en)
        language_confidence: Score de confiance de détection (0.0-1.0)
        detected_language: Langue auto-détectée
        suggested_categories: IDs des catégories suggérées (top 3)
        category_confidence: Score de confiance catégorie principale
        category_id: Catégorie assignée
        status: Statut (draft, published, archived)
        publication_date: Date de publication officielle
        created_at: Date de création dans le système
        updated_at: Date de dernière mise à jour
        category: Relation vers la catégorie
        articles: Relation vers les articles
    """

    __tablename__ = "laws"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Core fields
    reference = Column(String(500), nullable=False, unique=True, index=True)
    title = Column(String(500), nullable=False)
    type = Column(String(50), nullable=False)  # loi, décret, ordonnance, arrêté
    content = Column(Text, nullable=False)

    # v2.1 Auto-detection fields
    language = Column(String(2), nullable=True, index=True)  # fr, en
    language_confidence = Column(Float, nullable=True)
    detected_language = Column(String(2), nullable=True)
    suggested_categories = Column(ARRAY(Integer), nullable=True)  # ids des categories suggerees
    category_confidence = Column(Float, nullable=True)

    # File tracking
    file_id = Column(String(100), nullable=True, index=True)  # UUID of uploaded file
    original_filename = Column(String(500), nullable=True)  # Original filename

    # Relationships
    category_id = Column(Integer, ForeignKey("categories.id"), nullable=True, index=True)
    category = relationship("Category", back_populates="laws", lazy="selectin")
    articles = relationship(
        "Article",
        back_populates="law",
        cascade="all, delete-orphan",
        lazy="selectin"
    )


    # Status & metadata
    status = Column(String(20), nullable=False, server_default="draft", index=True)  # draft, published, archived, pending, processing, refused
    processing_progress = Column(Integer, nullable=True)  # 0-100 for batch upload tracking
    processing_error = Column(Text, nullable=True)  # Error message if processing failed
    processing_started_at = Column(DateTime, nullable=True)  # When processing started
    publication_date = Column(Date, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), nullable=False)
    updated_at = Column(
        DateTime,
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False
    )

    def __repr__(self) -> str:
        return f"<Law(id={self.id}, reference='{self.reference}', title='{self.title[:50]}...')>"

    @property
    def article_count(self) -> int:
        """Nombre d'articles dans cette loi."""
        return len(self.articles) if self.articles else 0


class Article(Base):
    """
    Modèle pour les articles individuels d'une loi.

    Représente un article spécifique d'un document juridique,
    avec son contenu et son embedding vectoriel pour la recherche sémantique.

    Attributes:
        id: Identifiant unique
        law_id: ID de la loi parent
        number: Numéro de l'article (ex: "1er", "2", "3bis")
        title: Titre optionnel de l'article
        content: Contenu textuel de l'article
        embedding: Vecteur d'embedding (3072 dimensions) pour recherche sémantique
        order: Ordre dans la loi (pour tri)
        created_at: Date de création
        law: Relation vers la loi parent
    """

    __tablename__ = "articles"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Foreign key
    law_id = Column(
        Integer,
        ForeignKey("laws.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    # Fields
    # 64 et non 20 : le motif d'ordinaux composes de text_chunker produit des
    # numeros comme 'QUATRE-VINGT-DIX-SEPTIEME' (25 caracteres). A 20, le commit
    # levait StringDataRightTruncation et la loi perdait TOUS ses articles
    # (migration c8d9e0f1a2b3).
    number = Column(String(64), nullable=False)
    title = Column(String(200), nullable=True)
    # Text et non String(300) : SECTION_PATTERNS capture `[^\n]*` et son `\s*`
    # initial peut franchir un saut de ligne, donc la capture s'etend souvent
    # sur deux lignes. Des en-tetes de plus de 450 caracteres sont courants sur
    # du markdown. Aucune borne superieure n'etait defendable.
    section = Column(Text, nullable=True)  # TITRE/CHAPITRE section header
    content = Column(Text, nullable=False)

    # pgvector embedding for semantic search.
    # 3072 dimensions, la sortie native de gemini-embedding-001, stockee en
    # fp32. Le plafond de 2000 dimensions de pgvector ne concerne que
    # l'indexation du type `vector` : l'index HNSW est pose sur l'expression
    # `embedding::halfvec(3072)`, indexable jusqu'a 4000 (migration
    # f5a6b7c8d9e0). La dimension est aussi declaree dans
    # settings.EMBEDDING_DIM, les deux doivent rester d'accord.
    embedding = Column(Vector(3072), nullable=True)

    # Classification produite par app/utils/chunk_refiner.py.
    # kind : article | legal_basis | preamble | boilerplate | roster | table |
    #        fragment | continuation
    # embed : faut-il vectoriser ce chunk. Rien n'est supprime — un visa reste
    #         consultable et cherchable en plein texte — mais il ne consomme
    #         plus d'appel d'embedding et ne pollue plus les resultats
    #         semantiques.
    # embed_text : le texte REELLEMENT envoye au modele, prefixe de l'en-tete du
    #         document. `content` reste intact : ce qui est affiche et cite ne
    #         change pas.
    kind = Column(String(30), nullable=True)
    embed = Column(Boolean, nullable=False, server_default=text("true"), default=True)
    embed_text = Column(Text, nullable=True)

    # Metadata
    order = Column(Integer, nullable=False)  # Position in law
    page_number = Column(Integer, nullable=True)  # PDF page number (1-indexed) for navigation
    created_at = Column(DateTime, server_default=func.now(), nullable=False)

    # Relationships
    law = relationship("Law", back_populates="articles")

    def __repr__(self) -> str:
        return f"<Article(id={self.id}, number='{self.number}', law_id={self.law_id})>"


# Indexes for performance
Index("idx_laws_language", Law.language)
Index("idx_laws_category", Law.category_id)
Index("idx_laws_status", Law.status)
Index("idx_laws_reference", Law.reference)
Index("idx_articles_law", Article.law_id)
