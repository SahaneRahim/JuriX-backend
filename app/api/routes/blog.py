"""
Routes API pour le blog juridique d'actualité décryptée par l'IA.

Endpoints :
- GET /api/v1/blog : Liste tous les articles publiés (avec filtres thématiques)
- GET /api/v1/blog/{slug} : Détail d'un article par son slug
- POST /api/v1/blog/generate : Génération autonome d'un article complet par l'IA (admin)
- POST /api/v1/blog : Publication / Enregistrement d'un article (admin)
- PUT /api/v1/blog/{id} : Mise à jour d'un article (admin)
- DELETE /api/v1/blog/{id} : Suppression d'un article (admin)
"""

import logging
import time
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.auth import get_current_admin_user
from app.core.database import get_db
from app.models.blog import BlogPost
from app.models.user import User
from app.services.blog_generator_service import BlogGeneratorService, slugify

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Blog"])


# ==================== SCHEMAS ====================

class BlogGenerateRequest(BaseModel):
    topic: str = Field(..., min_length=5, description="Sujet d'actualité ou question juridique")
    additional_context: Optional[str] = Field(None, description="Contexte optionnel ou détails sur le fait divers")


class CitationItem(BaseModel):
    law_id: Optional[int] = 1
    reference: str
    article_number: Optional[str] = None
    title: str
    excerpt: str


class BotAnalysisSchema(BaseModel):
    summary: str
    key_points: List[str]
    citations: List[Dict[str, Any]]
    citizen_impact: str


class BlogPostCreate(BaseModel):
    title: str
    slug: Optional[str] = None
    category: str
    reading_time: str = "4 min"
    context: str
    bot_analysis: Dict[str, Any]
    suggested_question: str
    tags: List[str] = []
    is_published: bool = True


class BlogPostUpdate(BaseModel):
    title: Optional[str] = None
    slug: Optional[str] = None
    category: Optional[str] = None
    reading_time: Optional[str] = None
    context: Optional[str] = None
    bot_analysis: Optional[Dict[str, Any]] = None
    suggested_question: Optional[str] = None
    tags: Optional[List[str]] = None
    is_published: Optional[bool] = None


# ==================== ENDPOINTS PUBLICS ====================

@router.get("", response_model=List[Dict[str, Any]])
async def list_blog_posts(
    category: Optional[str] = None,
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db)
):
    """Liste tous les articles de blog publiés."""
    query = select(BlogPost).where(BlogPost.is_published == True)
    if category and category.lower() != "tous":
        query = query.where(BlogPost.category.ilike(f"%{category}%"))
    
    query = query.order_by(desc(BlogPost.published_at)).limit(limit).offset(offset)
    res = await db.execute(query)
    posts = res.scalars().all()
    return [p.to_dict() for p in posts]


@router.get("/{slug}", response_model=Dict[str, Any])
async def get_blog_post(slug: str, db: AsyncSession = Depends(get_db)):
    """Récupère un article de blog par son slug."""
    query = select(BlogPost).where(BlogPost.slug == slug)
    res = await db.execute(query)
    post = res.scalar_one_or_none()
    if not post:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article '{slug}' introuvable"
        )
    return post.to_dict()


# ==================== ENDPOINTS ADMIN ====================

@router.post("/generate", response_model=Dict[str, Any])
async def generate_blog_post(
    req: BlogGenerateRequest,
    current_admin: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Génère de façon autonome un article de blog complet prêt à être relu ou publié.
    Réservé aux administrateurs.
    """
    service = BlogGeneratorService(db)
    generated = await service.generate_post(req.topic, req.additional_context)
    return generated


@router.get("/admin/all", response_model=List[Dict[str, Any]])
async def list_all_posts_admin(
    current_admin: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """Liste tous les articles (publiés et brouillons) pour l'administration."""
    query = select(BlogPost).order_by(desc(BlogPost.created_at))
    res = await db.execute(query)
    posts = res.scalars().all()
    return [p.to_dict() for p in posts]


@router.post("", response_model=Dict[str, Any], status_code=status.HTTP_201_CREATED)
async def create_blog_post(
    post_in: BlogPostCreate,
    current_admin: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """Enregistre et publie un article de blog."""
    slug = post_in.slug or slugify(post_in.title)
    
    # Vérifier l'unicité du slug
    exist = await db.execute(select(BlogPost).where(BlogPost.slug == slug))
    if exist.scalar_one_or_none():
        slug = f"{slug}-{int(time.time())}"

    bot_analysis = post_in.bot_analysis
    new_post = BlogPost(
        slug=slug,
        title=post_in.title,
        category=post_in.category,
        reading_time=post_in.reading_time,
        context=post_in.context,
        summary=bot_analysis.get("summary", ""),
        key_points=bot_analysis.get("keyPoints", bot_analysis.get("key_points", [])),
        citations=bot_analysis.get("citations", []),
        citizen_impact=bot_analysis.get("citizenImpact", bot_analysis.get("citizen_impact", "")),
        suggested_question=post_in.suggested_question,
        tags=post_in.tags,
        is_published=post_in.is_published,
    )
    db.add(new_post)
    await db.commit()
    await db.refresh(new_post)
    return new_post.to_dict()


@router.put("/{post_id}", response_model=Dict[str, Any])
async def update_blog_post(
    post_id: int,
    post_in: BlogPostUpdate,
    current_admin: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """Modifie un article de blog existant."""
    post = await db.get(BlogPost, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Article introuvable")

    if post_in.title is not None:
        post.title = post_in.title
    if post_in.slug is not None:
        post.slug = post_in.slug
    if post_in.category is not None:
        post.category = post_in.category
    if post_in.reading_time is not None:
        post.reading_time = post_in.reading_time
    if post_in.context is not None:
        post.context = post_in.context
    if post_in.suggested_question is not None:
        post.suggested_question = post_in.suggested_question
    if post_in.tags is not None:
        post.tags = post_in.tags
    if post_in.is_published is not None:
        post.is_published = post_in.is_published

    if post_in.bot_analysis is not None:
        ba = post_in.bot_analysis
        if "summary" in ba:
            post.summary = ba["summary"]
        if "keyPoints" in ba or "key_points" in ba:
            post.key_points = ba.get("keyPoints", ba.get("key_points"))
        if "citations" in ba:
            post.citations = ba["citations"]
        if "citizenImpact" in ba or "citizen_impact" in ba:
            post.citizen_impact = ba.get("citizenImpact", ba.get("citizen_impact"))

    await db.commit()
    await db.refresh(post)
    return post.to_dict()


@router.delete("/{post_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_blog_post(
    post_id: int,
    current_admin: User = Depends(get_current_admin_user),
    db: AsyncSession = Depends(get_db)
):
    """Supprime un article de blog."""
    post = await db.get(BlogPost, post_id)
    if not post:
        raise HTTPException(status_code=404, detail="Article introuvable")
    await db.delete(post)
    await db.commit()
    return None
