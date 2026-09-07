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
# Pseudo-numeros produits par text_chunker pour le texte hors articles.
_SPECIAL_CHUNK_LABELS = {
    "PREAMBULE": "Préambule",
    "LEGAL_BASIS": "Visas et base légale",
}

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
        # PREAMBULE et LEGAL_BASIS ne sont pas des numeros d'article : ce sont
        # les pseudo-numeros que le decoupeur donne au preambule et aux visas
        # pour ne perdre aucun caractere du document. Les annoncer comme
        # "Article LEGAL_BASIS" inviterait le modele a citer un article qui
        # n'existe pas.
        label = _SPECIAL_CHUNK_LABELS.get(chunk.number.upper())
        article_line = label if label else f"Article {chunk.number}"
        if chunk.article_title and not label:
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


# ============================================================================
# Explication d'un article isole
# ============================================================================

# Consigne de tache pour le bouton « Expliquer l'article » de la page de
# lecture. Elle complete le prompt systeme « citoyen », qui donne le TON, et
# n'y touche pas : ce prompt demande explicitement du markdown (titres en gras,
# listes), et la page sait rendre cette structure sans interpreter de HTML
# (voir src/lib/texte.ts cote frontend). Lui demander ici de ne plus en
# produire reviendrait a contredire le prompt systeme dans le meme appel.
#
# « n'explique pas leur contenu » vise les blocs voisins : sans cette phrase le
# modele resume les trois articles, et le lecteur ne sait plus lequel il lit.
EXPLAIN_TASK_TEMPLATES = {
    "fr": """Tâche : explique l'article {number} du document ci-dessus à une personne sans formation juridique.

- Ce que dit l'article, en une phrase.
- Ce que cela change concrètement pour la personne concernée.
- Les conditions, délais ou exceptions à connaître.

Le bloc [1] est l'article à expliquer. Les blocs suivants ne sont là que pour le contexte : n'explique pas leur contenu.

Structure ta réponse avec des titres courts en gras et des paragraphes brefs.
N'invente aucune obligation qui ne figure pas dans le texte fourni.""",

    "en": """Task: explain article {number} of the document above to someone with no legal training.

- What the article says, in one sentence.
- What it concretely changes for the person concerned.
- The conditions, deadlines or exceptions to know about.

Block [1] is the article to explain. The following blocks are context only: do not explain their content.

Structure your answer with short bold headings and brief paragraphs.
Do not invent any obligation that is absent from the provided text.""",
}


# ============================================================================
# Comparaison de deux regimes
# ============================================================================

# Prompt systeme du mode comparaison. Il ne reprend PAS les personas : ceux-ci
# demandent du markdown et un ton, alors qu'ici la sortie est un JSON contraint
# par un schema. Les deux consignes se contrediraient.
COMPARE_SYSTEM_PROMPTS = {
    "fr": """Tu es un juriste qui compare deux regimes juridiques camerounais a partir
d'extraits de textes officiels, et de rien d'autre.

REGLE ABSOLUE : chaque cellule que tu remplis doit etre justifiee par un extrait
fourni. Si les extraits ne disent rien sur un critere pour un sujet, ecris
exactement « {absence} » dans la cellule et laisse sa liste de sources vide.
N'utilise JAMAIS tes connaissances generales sur le droit d'un autre pays, ni
sur une version anterieure du texte.

Les sources sont des NUMEROS d'article, tels qu'ils apparaissent dans les
extraits — « 32 », « 40.1 », « 1er ». Ne cite jamais un numero absent des
extraits. Ne cite qu'un article qui soutient reellement ce que tu affirmes :
une source de trop est une erreur au meme titre qu'une source fausse.""",

    "en": """You are a lawyer comparing two Cameroonian legal regimes using only the
supplied extracts of official texts, and nothing else.

ABSOLUTE RULE: every cell you fill must be supported by a supplied extract. If
the extracts say nothing about a criterion for a subject, write exactly
« {absence} » in that cell and leave its source list empty. NEVER use general
knowledge of another country's law, or of an earlier version of the text.

Sources are article NUMBERS exactly as they appear in the extracts — "32",
"40.1", "1er". Never cite a number absent from the extracts. Only cite an
article that genuinely supports your statement: one source too many is as much
an error as a wrong source.""",
}

COMPARE_TASK_TEMPLATES = {
    "fr": """Sujet A : {a}
Extraits concernant le sujet A :
{ctx_a}

=====================================

Sujet B : {b}
Extraits concernant le sujet B :
{ctx_b}

=====================================

Compare le sujet A et le sujet B sur EXACTEMENT ces criteres, dans cet ordre :
{criteres}

Reponds une ligne par critere, en reportant dans le champ `index` le NUMERO du critere tel qu'il est ecrit ci-dessus.

Remplis la grille, puis enonce les differences majeures, puis les angles morts :
les criteres que les extraits ne permettent pas de trancher.""",

    "en": """Subject A: {a}
Extracts about subject A:
{ctx_a}

=====================================

Subject B: {b}
Extracts about subject B:
{ctx_b}

=====================================

Compare subject A and subject B on EXACTLY these criteria, in this order:
{criteres}

Answer one row per criterion, copying into the `index` field the NUMBER of the criterion exactly as written above.

Fill the grid, then state the key differences, then the blind spots: the
criteria the extracts do not allow you to settle.""",
}


def get_compare_system_prompt(language: str, absence: str) -> str:
    """Prompt systeme du mode comparaison, avec la mention d'absence exacte."""
    lang = language if language in COMPARE_SYSTEM_PROMPTS else "fr"
    return COMPARE_SYSTEM_PROMPTS[lang].format(absence=absence)
