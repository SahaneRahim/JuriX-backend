"""
Modèles SQLAlchemy pour les lois, articles et catégories.

Ce module définit les modèles de base de données pour le système JuriX:
- Law: Documents juridiques (lois, décrets, ordonnances)
- Article: Articles individuels d'une loi
- Category: Catégories juridiques (les 14 domaines canoniques, migration 013)

Features v2.1:
- Détection automatique de langue (language, language_confidence)
- Suggestions de catégories (suggested_categories, category_confidence)
- Support pgvector pour embeddings sémantiques

Author: JuriX Team
"""


from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    text,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import deferred, relationship
from sqlalchemy.sql import func

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

    Représente les 14 domaines canoniques du droit camerounais, fixés par
    la migration 013 et repris dans legal_domain_classifier.CANONICAL_DOMAINS
    (un test vérifie l'accord) : du Droit Constitutionnel à l'Éducation,
    Recherche, Culture et Médias. Une loi s'y rattache PAR LE NOM du domaine,
    jamais par une position.

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
    #
    # lazy="raise" et non "selectin" : charger une categorie chargeait TOUTES
    # ses lois, texte integral compris, puis — Law.articles etant lui aussi en
    # selectin — tous leurs articles. GET /categories/{id} hydratait ainsi
    # 16 lois et 1 262 articles pour rendre 206 octets (3,5 s, 469 ms de SQL ;
    # 7,4 s pour la categorie 13). Personne ne lit cette collection : les
    # comptes passent par COUNT. Un acces implicite leve desormais une erreur
    # explicite au lieu de couter en silence ; l'unite de travail, elle, la
    # charge encore quand un flush en a besoin (suppression d'une categorie :
    # elle remet a NULL le category_id des lois rattachees).
    laws = relationship("Law", back_populates="category", lazy="raise")

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
    # lazy="joined" et non "selectin" : une jointure externe sur une table de
    # 14 lignes, dans la MEME requete. En selectin, chaque select(Law) payait
    # une seconde instruction, soit deux allers-retours de plus (environ
    # 170 ms vers le pooler a 83 ms) — y compris sur /laws/{id}/page/{n},
    # appele une fois par page de PDF affichee. Les lecteurs de law.category
    # (LawResponse, pipeline synchrone, explication d'article) la trouvent
    # ainsi toujours chargee.
    category = relationship("Category", back_populates="laws", lazy="joined")
    # lazy="select" (chargement a l'acces) et non "selectin" : chaque select(Law)
    # chargeait TOUS les articles de la loi, contenu et embed_text compris —
    # 2 134 articles pour le plus gros document — y compris pour servir un
    # telechargement, une page de PDF ou une liste qui n'en voulait que le
    # nombre. Les routes qui en ont besoin les demandent (selectinload).
    #
    # Pas "raise" : app/tasks/process_law.py (_load_law) lit encore
    # law.articles en session SYNCHRONE, ou un chargement a l'acces reste
    # legitime. En asynchrone, un acces implicite leve MissingGreenlet : rien
    # ne se charge en silence.
    #
    # passive_deletes : la cle etrangere porte ON DELETE CASCADE (migration
    # 001, verifiee en base). Supprimer une loi laisse donc la base effacer ses
    # articles, au lieu de les charger tous en memoire pour emettre un DELETE
    # par article.
    articles = relationship(
        "Article",
        back_populates="law",
        cascade="all, delete-orphan",
        lazy="select",
        passive_deletes=True,
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
        """
        Nombre d'articles dans cette loi.

        Le compte pose par la requete (un COUNT en SQL, voir le setter) prime :
        la liste des lois n'a besoin que de ce nombre, et le tirer de la
        collection obligeait a charger tous les articles, contenu compris.
        Sinon, la collection chargee — GET /laws/{id} la charge de toute facon
        pour son sommaire.
        """
        compte = self.__dict__.get("_nombre_articles")
        if compte is not None:
            return compte
        return len(self.articles) if self.articles else 0

    @article_count.setter
    def article_count(self, valeur: int) -> None:
        # Attribut d'instance NON mappe : rien n'est ecrit en base.
        self._nombre_articles = valeur


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
        embedding: Vecteur d'embedding (halfvec, 1024 dimensions) pour recherche sémantique
        embedding_model: Empreinte du fournisseur qui a produit le vecteur
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
    # 1024 dimensions : la dimension par defaut de voyage-4-large, et une
    # dimension que gemini-embedding-001 produit aussi sur demande. Stockee en
    # halfvec (fp16, 2 octets par composante : environ 3 chiffres
    # significatifs, sans effet mesurable sur le classement) ; l'index HNSW
    # halfvec_cosine_ops est pose sur la colonne elle-meme (migration
    # a8b9c0d1e2f3). La dimension est aussi declaree dans
    # settings.EMBEDDING_DIM ; un test verifie l'accord.
    #
    # DIFFERE (deferred) : jamais charge avec l'article, seulement a la
    # lecture de l'attribut. Law.articles et Category.laws etaient en
    # lazy="selectin" : sans cela, ouvrir un document chargeait tous ses
    # vecteurs (2 121 pour le plus gros, environ 27 Mo de texte), et une
    # categorie ceux de toutes ses lois, sans qu'aucun de ces chemins ne les
    # lise. Ils ne sont plus charges d'office (voir Law.articles), mais la
    # colonne reste differee pour les chemins qui chargent les articles. La
    # recherche les lit en SQL, sans passer par l'objet. En contexte
    # asynchrone, lire l'attribut exige undefer(Article.embedding).
    embedding = deferred(Column(HALFVEC(1024), nullable=True))
    # Empreinte du fournisseur qui a produit `embedding` : modele, revision,
    # gabarits. Deux modeles produisent des espaces sans rapport, et un cosinus
    # entre leurs vecteurs ne veut rien dire — sans que rien ne le signale.
    # NULL avec un vecteur = origine inconnue, donc a refaire au meme titre
    # qu'un vecteur absent (scripts/regenerate_embeddings.py).
    embedding_model = Column(Text, nullable=True)

    # Classification produite par app/utils/chunk_refiner.py.
    # kind : article | legal_basis | preamble | boilerplate | roster | table |
    #        fragment | continuation | signature | annexe
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
