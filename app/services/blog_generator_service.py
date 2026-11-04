"""
Service de génération autonome d'articles de blog par l'IA.

Combine la recherche sémantique/plein texte dans le corpus juridique camerounais
et la génération structurée par LLM pour produire un article complet d'actualité décryptée.
"""

import json
import logging
import re
from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.search import SearchRequest
from app.services.llm import get_llm_service
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)


def slugify(text: str) -> str:
    """Génère un slug URL propre."""
    text = text.lower().strip()
    text = re.sub(r"[àáâãäå]", "a", text)
    text = re.sub(r"[èéêë]", "e", text)
    text = re.sub(r"[ìíîï]", "i", text)
    text = re.sub(r"[òóôõö]", "o", text)
    text = re.sub(r"[ùúûü]", "u", text)
    text = re.sub(r"[ç]", "c", text)
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text)
    return text.strip("-")[:80]


SYSTEM_PROMPT = """Tu es l'analyste juridique expert et rédacteur en chef de JuriX, la plateforme d'intelligence juridique du Cameroun.
Ta mission est de rédiger un article de décryptage d'actualité juridique à destination du grand public, des juristes et des chefs d'entreprise.
Tu t'appuies STRICTEMENT sur le droit positif applicable au Cameroun (Code civil, Code pénal, Code du travail, Code des douanes CEMAC, lois de finances, OHADA, etc.).

Règles de style et de fond :
1. Rigueur légale absolue : cite les vrais numéros d'articles et textes applicables issus du contexte fourni.
2. Pédagogie et clarté : pas de jargon incompréhensible, explique les notions de manière vivante et pragmatique.
3. Neutralité et précision : présente ce que dit la loi et les recours effectifs.
4. Tu dois répondre UNIQUEMENT par un objet JSON valide, sans texte d'introduction ni balises superflues.
"""


class BlogGeneratorService:
    def __init__(self, db: AsyncSession):
        self.db = db
        self.search_service = SearchService(db, use_cache=True)
        self.llm = get_llm_service()

    async def generate_post(
        self, topic: str, additional_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Génère un article de blog complet à partir d'un sujet ou d'une question.
        """
        logger.info(f"Génération d'article de blog pour le sujet: {topic}")

        # 1. Recherche des textes de lois pertinents dans le corpus JuriX
        search_query = topic
        if additional_context:
            search_query = f"{topic} {additional_context[:200]}"

        legal_chunks_text = ""
        found_citations = []
        try:
            search_req = SearchRequest(query=search_query, mode="hybrid", limit=5)
            search_res = await self.search_service.search(search_req)
            results = getattr(search_res, "results", []) or []

            for r in results:
                ref = getattr(r, "reference", "") or getattr(r, "law_title", "") or "Loi camerounaise"
                art_num = getattr(r, "article_number", "") or "?"
                content = getattr(r, "content", "") or getattr(r, "snippet", "")
                law_id = getattr(r, "law_id", 1) or getattr(r, "id", 1)
                
                legal_chunks_text += f"\n--- DOCUMENT [ID: {law_id}] : {ref} — Article {art_num} ---\n{content[:1500]}\n"
                found_citations.append({
                    "law_id": law_id,
                    "reference": ref,
                    "article_number": str(art_num),
                    "title": f"Article {art_num} - {ref}",
                    "excerpt": content[:250].strip()
                })
        except Exception as e:
            logger.warning(f"Recherche documentaire pour le blog: {e}")

        # 2. Construction du prompt de génération
        user_prompt = f"""Rédige un article d'actualité juridique décryptée complet sur le sujet suivant :

SUJET D'ACTUALITÉ / QUESTION :
« {topic} »

{"CONTEXTE SUPPLÉMENTAIRE FOURNI PAR L'ADMINISTRATEUR :" if additional_context else ""}
{additional_context or ""}

EXTRAITS DE TEXTES DE LOIS DISPONIBLES DANS NOTRE BASE JURIX :
{legal_chunks_text if legal_chunks_text else "Utilise les dispositions officielles applicables au Cameroun pour ce domaine."}

Réponds avec un JSON strictement formaté selon cette structure :
{{
  "title": "Titre accrocheur, journalistique et percutant (max 100 car.)",
  "slug": "slug-minuscule-sans-accents-court",
  "category": "Nom de la thématique (ex: Immobilier & Logement, Travail & Emploi, Économie & Fiscalité, Droit Pénal, Famille, etc.)",
  "reading_time": "3 min",
  "context": "Ce qui s'est passé en 2 à 3 phrases : contexte de fait concret ou situation fréquente.",
  "summary": "Explication juridique vulgarisée et claire de ce que prévoit la loi au Cameroun.",
  "key_points": [
    "Première règle essentielle",
    "Deuxième règle essentielle",
    "Troisième règle essentielle",
    "Quatrième règle essentielle"
  ],
  "citations": [
    {{
      "law_id": 1,
      "reference": "Nom officiel du texte ou de la loi",
      "article_number": "12",
      "title": "Objet ou intitulé de l'article",
      "excerpt": "Extrait fidèle et significatif du texte de loi"
    }}
  ],
  "citizen_impact": "Ce que cela change concrètement pour le citoyen ou justiciable et le conseil pratique immédiat.",
  "suggested_question": "La question exacte pré-remplie à poser à l'assistant dans le chat pour approfondir son cas personnel.",
  "tags": ["motcle1", "motcle2", "motcle3", "motcle4"]
}}
"""

        # 3. Appel au LLM
        res = await self.llm.generate(
            prompt=user_prompt,
            system=SYSTEM_PROMPT,
            temperature=0.3,
            max_tokens=3000,
            response_mime_type="application/json",
        )

        raw_response = res.get("response", "{}")
        # Nettoyage si le LLM a entouré de ```json
        cleaned_json = raw_response.strip()
        if cleaned_json.startswith("```"):
            cleaned_json = re.sub(r"^```(?:json)?\s*", "", cleaned_json)
            cleaned_json = re.sub(r"\s*```$", "", cleaned_json)

        try:
            data = json.loads(cleaned_json)
        except Exception as e:
            logger.error(f"Erreur décodage JSON LLM blog: {e}. Contenu: {cleaned_json[:200]}")
            data = {
                "title": f"Décryptage légal : {topic[:80]}",
                "slug": slugify(topic),
                "category": "Actualité juridique",
                "reading_time": "3 min",
                "context": f"Face aux interrogations récentes sur {topic}, une clarification juridique s'impose.",
                "summary": cleaned_json[:500],
                "key_points": [
                    "Le respect des dispositions en vigueur est obligatoire.",
                    "Toute infraction est susceptible de sanctions légales.",
                    "Un recours amiable ou judiciaire est possible devant la juridiction compétente.",
                    "Prenez conseil auprès d'un professionnel du droit."
                ],
                "citations": found_citations[:2],
                "citizen_impact": "Consultez les textes officiels avant toute démarche.",
                "suggested_question": f"Quels sont les détails légaux sur {topic} au Cameroun ?",
                "tags": ["droit", "cameroun", "actualite"]
            }

        # Sécurisation du slug
        if not data.get("slug"):
            data["slug"] = slugify(data.get("title", topic))

        # Vérification / Injection de citations si le LLM a mis une liste vide
        if not data.get("citations") and found_citations:
            data["citations"] = found_citations[:2]

        return data
