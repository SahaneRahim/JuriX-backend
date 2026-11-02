"""
Modèle SQLAlchemy pour les articles de blog décryptés par l'IA.

Permet à l'équipe JuriX de publier et gérer les analyses d'actualité juridique
générées de manière autonome par l'IA.
"""

from sqlalchemy import Boolean, Column, DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.sql import func
from app.core.database import Base


class BlogPost(Base):
    __tablename__ = "blog_posts"

    id = Column(Integer, primary_key=True, index=True)
    slug = Column(String(255), unique=True, index=True, nullable=False)
    title = Column(String(500), nullable=False)
    category = Column(String(100), nullable=False, index=True)
    published_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    reading_time = Column(String(50), default="4 min", nullable=False)
    
    # Bloc 1 : L'Actualité en 2 minutes (Le contexte)
    context = Column(Text, nullable=False)
    
    # Bloc 2 : Ce que dit la loi — L'Analyse du Bot
    summary = Column(Text, nullable=False)
    key_points = Column(JSONB, nullable=False, default=list)
    citations = Column(JSONB, nullable=False, default=list)
    citizen_impact = Column(Text, nullable=False)
    
    # Bloc 3 : Action Virale & Tags
    suggested_question = Column(Text, nullable=False)
    tags = Column(JSONB, nullable=False, default=list)
    
    is_published = Column(Boolean, default=True, index=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "slug": self.slug,
            "title": self.title,
            "category": self.category,
            "publishedAt": self.published_at.strftime("%Y-%m-%d") if self.published_at else "",
            "readingTime": self.reading_time,
            "context": self.context,
            "botAnalysis": {
                "summary": self.summary,
                "keyPoints": self.key_points or [],
                "citations": self.citations or [],
                "citizenImpact": self.citizen_impact,
            },
            "suggestedQuestion": self.suggested_question,
            "tags": self.tags or [],
            "isPublished": self.is_published,
        }
