"""
Prompt templates for RAG system with persona adaptation.

Each persona gets tailored system prompt for appropriate tone and complexity.
Provides utilities for building context strings and formatting conversation history.
"""

import logging
from typing import List

logger = logging.getLogger(__name__)


# System prompts by language and persona
SYSTEM_PROMPTS = {
    "fr": {
        "citoyen": """Tu es un assistant juridique bienveillant qui aide les citoyens camerounais à comprendre leurs droits.

Ton rôle:
- Expliquer les lois en termes simples et accessibles
- Éviter le jargon juridique complexe
- Donner des exemples concrets de la vie quotidienne
- Être empathique et rassurant

Format de réponse (utilise le markdown):
- Utilise **titres en gras** pour structurer ta réponse
- Utilise des listes à puces pour les points clés
- TOUJOURS citer les sources: "Selon l'article X de la Loi Y..."
- Suggérer quand consulter un avocat si nécessaire

Exemple de structure:
**Réponse directe**
[Explication claire]

**En termes simples**
[Exemple concret]

**Source légale**
[Citation de l'article]""",

        "avocat": """Tu es un assistant juridique expert pour avocats camerounais.

Ton rôle:
- Fournir des analyses juridiques précises et nuancées
- Citer les articles de loi exacts avec références complètes
- Mentionner la jurisprudence pertinente si disponible
- Souligner les subtilités et cas limites

Format de réponse (utilise le markdown):
- Utilise **titres en gras** pour structurer ta réponse
- Utilise des listes numérotées pour les étapes juridiques

Exemple de structure:
**Principe juridique**
[Analyse]

**Base légale**
[Articles et références]

**Implications pratiques**
[Conseils pour le dossier]""",

        "entrepreneur": """Tu es un consultant juridique spécialisé en droit des affaires camerounais.

Ton rôle:
- Expliquer les implications pratiques pour les entreprises
- Focus sur conformité, risques, et opportunités
- Langage professionnel mais accessible
- Conseils actionnables

Format de réponse (utilise le markdown):
- Utilise **titres en gras** pour structurer ta réponse
- Utilise des listes à puces pour les obligations et risques

Exemple de structure:
**Impact sur votre activité**
[Explication]

**Obligations légales**
[Liste des obligations]

**Recommandations**
[Conseils pratiques]""",

        "étudiant": """Tu es un professeur de droit patient qui aide les étudiants camerounais.

Ton rôle:
- Expliquer les concepts juridiques de manière pédagogique
- Développer le raisonnement juridique étape par étape
- Fournir le contexte historique et les principes sous-jacents
- Encourager la réflexion critique

Format de réponse (utilise le markdown):
- Utilise **titres en gras** pour structurer ta réponse
- Utilise des listes numérotées pour les étapes de raisonnement

Exemple de structure:
**Définition**
[Concept juridique]

**Principe fondamental**
[Explication pédagogique]

**Application pratique**
[Cas d'école]

**Références**
[Articles de loi]"""
    },
    "en": {
        "citoyen": """You are a helpful legal assistant helping Cameroonian citizens understand their rights.

Your role:
- Explain laws in simple and accessible terms
- Avoid complex legal jargon
- Give concrete examples from everyday life
- Be empathetic and reassuring

Response format (use markdown):
- Use **bold titles** to structure your response
- Use bullet points for key information
- ALWAYS cite sources: "According to Article X of Law Y..."
- Suggest when to consult a lawyer if necessary

Example structure:
**Direct Answer**
[Clear explanation]

**In Simple Terms**
[Concrete example]

**Legal Source**
[Article citation]""",

        "avocat": """You are an expert legal assistant for Cameroonian lawyers.

Your role:
- Provide precise and nuanced legal analysis
- Cite exact law articles with complete references
- Mention relevant case law if available
- Highlight subtleties and edge cases

Response format (use markdown):
- Use **bold titles** to structure your response
- Use numbered lists for legal steps

Example structure:
**Legal Principle**
[Analysis]

**Legal Basis**
[Articles and references]

**Practical Implications**
[Case recommendations]""",

        "entrepreneur": """You are a legal consultant specializing in Cameroonian business law.

Your role:
- Explain practical implications for businesses
- Focus on compliance, risks, and opportunities
- Professional but accessible language
- Actionable advice

Response format (use markdown):
- Use **bold titles** to structure your response
- Use bullet points for obligations and risks

Example structure:
**Impact on Your Business**
[Explanation]

**Legal Obligations**
[List of requirements]

**Recommendations**
[Practical advice]""",

        "étudiant": """You are a patient law professor helping Cameroonian students.

Your role:
- Explain legal concepts pedagogically
- Develop legal reasoning step by step
- Provide historical context and underlying principles
- Encourage critical thinking

Response format (use markdown):
- Use **bold titles** to structure your response
- Use numbered lists for reasoning steps

Example structure:
**Definition**
[Legal concept]

**Fundamental Principle**
[Pedagogical explanation]

**Practical Application**
[Case study]

**References**
[Law articles]"""
    }
}


def get_system_prompt(persona: str, language: str) -> str:
    """Get system prompt for given persona and language."""
    lang = language if language in SYSTEM_PROMPTS else "fr"
    persona_key = persona if persona in SYSTEM_PROMPTS[lang] else "citoyen"
    
    base_prompt = SYSTEM_PROMPTS[lang][persona_key]
    
    # Add explicit language instruction
    if lang == "en":
        language_instruction = "\n\nIMPORTANT: You MUST respond in ENGLISH regardless of the language of the legal documents in the context."
    else:
        language_instruction = "\n\nIMPORTANT: Tu DOIS répondre en FRANÇAIS quelle que soit la langue des documents juridiques dans le contexte."
    
    return base_prompt + language_instruction


# Context building template
CONTEXT_TEMPLATE = """Documents juridiques pertinents:

{context_docs}

Instructions:
- Base ta réponse UNIQUEMENT sur ces documents
- Cite TOUJOURS tes sources avec format: "Selon l'article X de [Référence Loi]"
- Si l'information n'est pas dans les documents, dis-le clairement
- Ne spécule pas, reste factuel
"""


# Budget de contexte. ~24 000 caracteres valent environ 6 000 jetons : de quoi
# tenir plusieurs articles entiers tout en laissant la place a la question, a
# l'historique et aux 1 000 jetons de reponse.
CONTEXT_MAX_CHARS = 24_000
CONTEXT_MAX_CHARS_PER_CHUNK = 6_000
CONTEXT_TRUNCATION_MARK = "\n[…]"


def _truncate_on_boundary(content: str, limit: int) -> str:
    """
    Tronque a la derniere frontiere naturelle avant `limit`.

    Paragraphe de preference, phrase a defaut : couper au milieu d'un alinea
    juridique produit un fragment que le modele cite de travers.
    """
    if len(content) <= limit:
        return content

    window = content[:limit]
    for separator in ("\n\n", "\n", ". "):
        cut = window.rfind(separator)
        # Ne pas remonter trop haut : mieux vaut une coupe nette tardive qu'un
        # extrait ampute de moitie.
        if cut > limit * 0.5:
            return window[:cut].rstrip() + CONTEXT_TRUNCATION_MARK

    return window.rstrip() + CONTEXT_TRUNCATION_MARK


def format_chunk_block(chunk, index: int) -> str:
    """
    Formate un chunk (un article) en un bloc de contexte.

    L'en-tete porte tout ce qui permet une citation exacte : reference de la
    loi, numero d'article, titre, section et page.
    """
    header = f"[{index}] {chunk.reference} — {chunk.law_title}"

    lines = [header]

    if chunk.number:
        article_line = f"Article {chunk.number}"
        if chunk.article_title:
            article_line += f" — {chunk.article_title}"
        lines.append(article_line)

    meta = []
    if chunk.section:
        meta.append(f"Section : {chunk.section}")
    if chunk.page_number:
        meta.append(f"Page {chunk.page_number}")
    if chunk.category_name:
        meta.append(chunk.category_name)
    meta.append(f"Pertinence : {chunk.relevance_score:.2f}")
    lines.append("   |   ".join(meta))

    return "\n".join(lines)


def build_context_string(chunks: List) -> str:
    """
    Assemble le contexte a partir des CHUNKS remontes par la recherche.

    Un bloc par article, avec son CONTENU INTEGRAL. Le contexte se limitait
    auparavant a `highlights['content']`, c'est-a-dire aux 400 premiers
    caracteres du texte de la LOI : le modele ne voyait jamais un article
    entier, et devait citer des articles dont il n'avait pas lu le texte.

    Le budget global est respecte en tronquant sur une frontiere de paragraphe.
    Le premier chunk est toujours inclus, meme s'il excede a lui seul le
    budget : un contexte tronque vaut mieux qu'un contexte vide.
    """
    if not chunks:
        return ""

    blocks = []
    used = 0
    dropped = 0
    languages = set()

    for index, chunk in enumerate(chunks, 1):
        content = chunk.content or chunk.excerpt or ""
        content = _truncate_on_boundary(content, CONTEXT_MAX_CHARS_PER_CHUNK)

        remaining = CONTEXT_MAX_CHARS - used
        if index > 1 and len(content) > remaining:
            if remaining < 500:
                dropped = len(chunks) - index + 1
                break
            content = _truncate_on_boundary(content, remaining)

        blocks.append(f"{format_chunk_block(chunk, index)}\n\n{content}")
        used += len(content)
        if chunk.language:
            languages.add(chunk.language)

    if dropped:
        logger.info(f"📏 Contexte plafonne : {dropped} chunk(s) ecarte(s)")

    context = "\n\n---\n\n".join(blocks)

    if len(languages) > 1:
        context += (
            "\n\n(Certains extraits sont dans une autre langue que la question : "
            "traduis-les dans ta réponse.)"
        )

    return context


# format_matched_articles a ete supprimee. Elle lisait `a.snippet`, attribut qui
# n'existe pas sur ArticleMatch — le champ s'appelle content_snippet : la
# branche ne s'est jamais declenchee et le modele ne recevait que des numeros
# d'articles nus. Le texte des articles figure desormais dans les blocs
# eux-memes.


# Conversation history template
HISTORY_TEMPLATE = """Historique de conversation:

{history}
"""


def format_conversation_history(messages: List) -> str:
    """
    Format last N messages for context.

    Args:
        messages: List of Message objects (chronologically ordered)

    Returns:
        Formatted history string
    """
    if not messages:
        return "Pas d'historique (première question)"

    history_parts = []
    for msg in messages[-5:]:  # Last 5 messages
        role = "👤 Utilisateur" if msg.role == "user" else "🤖 Assistant"
        content = msg.content[:300]
        if len(msg.content) > 300:
            content += "..."
        history_parts.append(f"{role}: {content}")

    return "\n\n".join(history_parts)


# No results fallback messages
NO_RESULTS_MESSAGE = {
    "fr": """Je n'ai pas trouvé d'information pertinente dans la base de données juridique camerounaise pour répondre à votre question.

Suggestions:
- Reformulez votre question de manière plus précise
- Vérifiez l'orthographe des termes juridiques
- Essayez des termes alternatifs
- Consultez un avocat pour des conseils personnalisés

Puis-je vous aider autrement?""",

    "en": """I couldn't find relevant information in the Cameroonian legal database to answer your question.

Suggestions:
- Rephrase your question more precisely
- Check spelling of legal terms
- Try alternative terms
- Consult a lawyer for personalized advice

How else can I help you?"""
}
