"""
RAGService - Retrieval-Augmented Generation service for legal Q&A.

Orchestrates the complete RAG pipeline:
1. Document retrieval (SearchService)
2. Conversation history loading
3. Prompt construction (persona-adapted)
4. Answer generation (GeminiService) - TODO: Implement
5. Citation extraction
6. Persistence (Database)
"""

import json
import logging
import re
import time
import uuid
from datetime import datetime
from typing import Any, AsyncIterator, Dict, List, Optional, Tuple

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.models.conversation import Conversation, Message
from app.schemas.rag import Citation, RAGRequest, RAGResponse
from app.schemas.search import ChunkResult, SearchFilters, SearchRequest
from app.services.gemini_service import get_gemini_service, GeminiServiceError
from app.services.prompts import (
    CONTEXT_TEMPLATE,
    NO_RESULTS_MESSAGE,
    SYSTEM_PROMPTS,
    build_context_string,
    format_conversation_history,
    get_system_prompt,
)
from app.services.postgres_search_service import escape_like
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)


# Formes ordinales rencontrees dans le corpus, ramenees a leur rang.
_ORDINAL_WORDS = {
    "PREMIER": "1", "PREMIERE": "1", "PREMIÈRE": "1", "1ER": "1", "1ÈRE": "1",
    "FIRST": "1", "DEUXIEME": "2", "DEUXIÈME": "2", "SECOND": "2", "SECONDE": "2",
    "TROISIEME": "3", "TROISIÈME": "3",
}


def _normalize_article_number(number: str) -> str:
    """
    Normalise un numero d'article pour comparaison.

    Les numeros circulent sous des formes multiples — "1", "1er", "PREMIER",
    "Article 1" — et etaient compares tantot par egalite de chaines, tantot par
    inclusion. L'inclusion faisait correspondre "1" a "10", "11", "100" : la
    mauvaise citation etait alors rattachee a la reponse.
    """
    if not number:
        return ""
    cleaned = str(number).strip().upper()
    cleaned = re.sub(r"^(ARTICLE|ART\.?|SECTION)\s+", "", cleaned)
    cleaned = cleaned.strip(" .:-")
    return _ORDINAL_WORDS.get(cleaned, cleaned)


class RAGServiceError(Exception):
    """Base exception for RAG service."""
    pass


class RAGService:
    """
    RAG (Retrieval-Augmented Generation) service for legal Q&A.

    Orchestrates:
    1. Document retrieval (SearchService)
    2. Conversation history loading
    3. Prompt construction (persona-adapted)
    4. Answer generation (GeminiService) - TODO
    5. Citation extraction
    6. Persistence (Database)

    Stateful service - requires database session per request.
    """

    MAX_HISTORY_MESSAGES = 5
    TOP_K_DOCUMENTS = 5
    # Nombre de CHUNKS envoyes au modele. Un chunk est un article, pas un
    # document : huit articles entiers representent un contexte plus precis, et
    # souvent plus court, que cinq lois resumees a 400 caracteres chacune.
    TOP_K_CHUNKS = 8
    # Couvre du / de la / de l' / des / de, et la numerotation composee
    # (161, 1er, 94-2, L 94 septies). L'ancienne version ne connaissait que
    # "de la " : "article 161 du Code OHADA", la formulation la plus courante
    # en francais juridique, n'etait JAMAIS reconnue — les reponses du chatbot
    # sortaient donc sans aucune source.
    CITATION_REGEX = (
        r"[Aa]rticle\s+"
        r"([LRD]\s*)?(\d+(?:[-.]\d+)*(?:\s*(?:er|bis|ter|quater|septies))?)"
        r"\s+(?:du\s+|de\s+la\s+|de\s+l['’]\s*|des\s+|de\s+)?"
        r"([A-ZÀ-Ý][^,\.]+)"
    )
    
    # French and English stopwords to filter out for better search
    STOPWORDS = {
        # French
        "le", "la", "les", "un", "une", "des", "de", "du", "d", "l",
        "ce", "cette", "ces", "mon", "ma", "mes", "ton", "ta", "tes",
        "son", "sa", "ses", "notre", "nos", "votre", "vos", "leur", "leurs",
        "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
        "me", "te", "se", "lui", "y", "en", "qui", "que", "quoi", "dont", "où",
        "et", "ou", "mais", "donc", "or", "ni", "car", "si", "quand", "comme",
        "à", "au", "aux", "avec", "sans", "sous", "sur", "dans", "par", "pour",
        "est", "sont", "être", "avoir", "fait", "faire", "dit", "dire",
        "c", "qu", "n", "s", "m", "t",
        "moi", "toi", "soi", "eux",
        "ne", "pas", "plus", "moins", "très", "bien", "mal",
        "tout", "tous", "toute", "toutes", "autre", "autres",
        "quel", "quelle", "quels", "quelles",
        "comment", "pourquoi", "quand", "combien",
        "explique", "expliquer", "donne", "donner", "dis", "parle", "parler",
        # English
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can",
        "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
        "my", "your", "his", "its", "our", "their",
        "this", "that", "these", "those",
        "what", "which", "who", "whom", "whose", "where", "when", "why", "how",
        "and", "or", "but", "if", "then", "so", "because",
        "of", "to", "in", "on", "at", "by", "for", "with", "about", "from",
        "explain", "tell", "give", "show", "describe",
    }
    
    @staticmethod
    def extract_keywords(question: str) -> str:
        """
        Extract important keywords from a natural language question.
        Removes stopwords pour ameliorer la recherche plein texte.
        """
        import re
        # Normalize: lowercase and remove punctuation except apostrophes
        words = re.findall(r"[a-zA-ZàâäéèêëïîôùûüçÀÂÄÉÈÊËÏÎÔÙÛÜÇ]+", question.lower())
        
        # Filter stopwords and short words
        keywords = [w for w in words if w not in RAGService.STOPWORDS and len(w) > 2]
        
        # Return at least 2 keywords, or original if too few keywords found
        if len(keywords) < 2:
            return question
        
        return " ".join(keywords[:10])  # Limit to 10 keywords

    def __init__(self, db: AsyncSession):
        """
        Initialize RAG service.

        Args:
            db: Async database session
        """
        self.db = db
        self.llm = get_gemini_service()
        self.search_service = SearchService(db)

    async def ask(self, request: RAGRequest) -> RAGResponse:
        """
        Main entry point for RAG question answering.

        Pipeline: load history → retrieve context → generate answer → save.

        Args:
            request: RAGRequest with question, persona, etc.

        Returns:
            RAGResponse with answer and citations
        """
        assert request and request.question, "RAGRequest with question is required"
        start_time = time.time()

        logger.info(
            f"🔍 RAG request: persona={request.persona}, "
            f"question=\"{request.question[:50]}...\""
        )

        try:
            # 1. Load conversation history
            conversation, history = await self._load_or_create_conversation(
                request.session_id, request.persona, request.language
            )

            # 2. Retrieve and merge document context
            retrieval_start = time.time()
            search_results, early_response = await self._retrieve_and_merge_context(
                request, conversation
            )
            retrieval_time_ms = int((time.time() - retrieval_start) * 1000)

            # Handle early exit (article not found in specific document)
            if early_response:
                early_response.retrieval_time_ms = retrieval_time_ms
                early_response.total_time_ms = retrieval_time_ms
                return early_response

            # 3. Fallback if no results
            if not search_results:
                search_results = await self._fallback_search(request.question, history)
            if not search_results:
                logger.warning("⚠️ No results found")
                return await self._handle_no_results(request, retrieval_time_ms)

            # 4. Generate answer, extract citations, save
            return await self._generate_and_save_response(
                request, conversation, history, search_results,
                retrieval_time_ms, start_time
            )

        except Exception as e:
            logger.error(f"❌ RAG error: {e}", exc_info=True)
            raise RAGServiceError(f"Erreur lors du traitement: {str(e)}")

    async def _retrieve_and_merge_context(self, request: RAGRequest, conversation):
        """
        Retrieve documents: priority doc first, then general search, merged.

        Returns:
            Tuple of (search_results, early_response_or_None)
        """
        priority_chunks: List[ChunkResult] = []
        not_found = None
        if request.law_id:
            # Le marqueur "article absent" est renvoye A PART et non porte par
            # un faux resultat : sonder hasattr(results[0], 'article_not_found')
            # obligeait tout le reste du code a manipuler un objet qui n'etait
            # pas un resultat de recherche.
            priority_chunks, not_found = await self._get_priority_document_context(
                request.law_id, request.question
            )
            logger.info(f"📌 Priority doc {request.law_id}: {len(priority_chunks)} chunks")

        if not_found:
            message = (
                f"Ce document ne contient pas d'article {not_found['requested_article']}. "
                f"Le document \"{not_found['law_title']}\" contient "
                f"{not_found['total_articles']} article(s). "
            )
            if not_found["total_articles"] > 0:
                message += (
                    f"Veuillez demander un article entre 1 et {not_found['total_articles']}."
                )
            else:
                message += "Ce document ne contient aucun article extrait."

            logger.info(f"📌 Article not found response: {message}")
            return [], RAGResponse(
                answer=message, confidence=1.0, sources=[],
                session_id=conversation.session_id,
                retrieval_time_ms=0, generation_time_ms=0, total_time_ms=0,
                persona=request.persona
            )

        # General search
        search_results = await self._retrieve_chunks(request.question, request.language)

        # Merge: priority first, then general (deduplicated)
        if priority_chunks:
            # Dedoublonnage sur la cle de chunk et non sur law_id : deux
            # articles differents d'une meme loi doivent tous deux survivre.
            seen = {c.fusion_key for c in priority_chunks}
            filtered = [c for c in search_results if c.fusion_key not in seen]
            search_results = priority_chunks + filtered[:self.TOP_K_CHUNKS]
            logger.info(
                f"📚 Merged: {len(priority_chunks)} priority + {len(filtered[:self.TOP_K_CHUNKS])} general"
            )

        return search_results, None

    async def _generate_and_save_response(
        self, request, conversation, history, search_results,
        retrieval_time_ms, start_time
    ):
        """Build prompt, generate LLM answer, extract citations, save."""
        prompt = self._build_prompt(
            question=request.question, search_results=search_results,
            history=history, persona=request.persona
        )

        generation_start = time.time()
        system_prompt = get_system_prompt(request.persona, request.language)

        if self.llm is None:
            raise RAGServiceError("LLM service not configured. Please set up Gemini API.")

        llm_response = await self.llm.generate(
            prompt=prompt, system=system_prompt, temperature=0.7, max_tokens=1000
        )
        generation_time_ms = int((time.time() - generation_start) * 1000)
        answer = llm_response["response"]

        # Extract citations
        citations = self._extract_citations(answer, search_results)
        if not citations and search_results:
            citations = self._create_sources_from_results(search_results, request.question)

        confidence = self._calculate_confidence(answer, citations, search_results)

        await self._save_interaction(
            conversation=conversation, question=request.question, answer=answer,
            citations=citations, confidence=confidence,
            retrieval_time_ms=retrieval_time_ms, generation_time_ms=generation_time_ms
        )

        total_time_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"✅ RAG complete: {total_time_ms}ms "
            f"(retrieval={retrieval_time_ms}ms, generation={generation_time_ms}ms)"
        )

        return RAGResponse(
            answer=answer, confidence=confidence, sources=citations,
            session_id=conversation.session_id,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms,
            total_time_ms=total_time_ms, persona=request.persona
        )

    async def ask_stream(
        self,
        request: RAGRequest
    ) -> AsyncIterator[str]:
        """
        Streaming version of ask() for better UX.

        Yields answer chunks as they're generated.
        Final chunk includes sources and confidence.

        Yields:
            JSON strings for SSE: {"chunk": "...", "done": false}
        """
        assert request is not None, "RAGRequest must not be None"
        assert isinstance(request.question, str) and len(request.question) > 0, "Question must be non-empty"

        start_time = time.time()

        try:
            # Retrieval (same as ask)
            retrieval_start = time.time()
            search_results = await self._retrieve_chunks(
                request.question,
                request.language
            )
            retrieval_time_ms = int((time.time() - retrieval_start) * 1000)

            if not search_results:
                yield json.dumps({
                    "chunk": NO_RESULTS_MESSAGE[request.language],
                    "done": True,
                    "sources": [],
                    "confidence": 0.0
                })
                return

            # Load history & build prompt
            conversation, history = await self._load_or_create_conversation(
                request.session_id,
                request.persona,
                request.language
            )

            prompt = self._build_prompt(
                request.question,
                search_results,
                history,
                request.persona
            )

            # Stream generation
            generation_start = time.time()
            # SYSTEM_PROMPTS est indexe par LANGUE ("fr"/"en"), pas par persona :
            # SYSTEM_PROMPTS["citoyen"] levait KeyError a chaque appel, donc
            # /rag/ask/stream n'a jamais rien renvoye d'autre qu'une erreur.
            # get_system_prompt(persona, language) est le bon accesseur, deja
            # utilise par le chemin non-streaming (ligne ~240).
            system_prompt = get_system_prompt(request.persona, request.language)

            # TODO: Implement Gemini streaming
            if self.llm is None:
                yield json.dumps({"chunk": "LLM service not configured. Please set up Gemini API.", "done": True, "error": True})
                return

            answer_parts = []
            async for chunk in self.llm.generate_stream(
                prompt=prompt,
                system=system_prompt,
                temperature=0.7,
                max_tokens=1000
            ):
                answer_parts.append(chunk)
                yield json.dumps({"chunk": chunk, "done": False})

            generation_time_ms = int((time.time() - generation_start) * 1000)

            # Process complete answer
            full_answer = "".join(answer_parts)
            citations = self._extract_citations(full_answer, search_results)
            confidence = self._calculate_confidence(
                full_answer, citations, search_results
            )

            # Save to DB
            await self._save_interaction(
                conversation, request.question, full_answer,
                citations, confidence,
                retrieval_time_ms, generation_time_ms
            )

            # Final chunk with metadata
            yield json.dumps({
                "chunk": "",
                "done": True,
                "sources": [c.model_dump() for c in citations],
                "confidence": confidence,
                "session_id": conversation.session_id
            })

        except Exception as e:
            logger.error(f"❌ Streaming error: {e}")
            yield json.dumps({
                "chunk": "",
                "done": True,
                "error": str(e)
            })

    async def _retrieve_chunks(
        self,
        question: str,
        language: str
    ) -> List[ChunkResult]:
        """
        Recupere les articles pertinents via SearchService.

        Mode hybride : il etait epingle sur "text" parce que le corpus n'etait
        pas vectorise et que la recherche semantique ne renvoyait rien
        d'exploitable. Elle rend desormais des articles entiers, et l'hybride
        tolere un cote semantique vide.

        Aucun filtre de langue : un texte camerounais peut n'exister qu'en
        anglais, le modele traduit dans la reponse.
        """
        search_query = self.extract_keywords(question)
        logger.info(f"🔍 RAG Search: original='{question[:50]}...' keywords='{search_query}'")

        search_response = await self.search_service.search(
            SearchRequest(
                query=search_query,
                mode="hybrid",
                filters=SearchFilters(status="published"),
                limit=self.TOP_K_CHUNKS
            )
        )

        chunks = search_response.chunks
        logger.info(
            f"📚 Retrieved {len(chunks)} chunks "
            f"in {search_response.search_time_ms}ms"
        )

        if chunks:
            first = chunks[0]
            logger.info(
                f"📄 First chunk: {first.reference} art. {first.number} - {first.law_title}"
            )
        else:
            logger.warning(f"⚠️ No search results for query: {question[:100]}")

        return chunks

    async def _get_priority_document_context(
        self, law_id: int, question: str
    ) -> Tuple[List[ChunkResult], Optional[Dict[str, Any]]]:
        """
        Contexte issu du document actuellement consulte.

        Quand l'utilisateur lit un document et pose une question, ses articles
        priment sur la recherche generale.

        Returns:
            (chunks, marqueur_article_absent). Le second element vaut None dans
            le cas nominal ; il porte sinon de quoi composer la reponse
            "cet article n'existe pas dans ce document".
        """
        assert isinstance(law_id, int) and law_id > 0, "law_id must be a positive integer"
        assert isinstance(question, str) and len(question) > 0, "question must be a non-empty string"

        from app.models.law import Law, Article

        try:
            query = select(Law).where(Law.id == law_id)
            result = await self.db.execute(query)
            law = result.scalar_one_or_none()

            if not law:
                logger.warning(f"⚠️ Priority law_id {law_id} not found")
                return [], None

            article_query = (
                select(Article).where(Article.law_id == law_id).order_by(Article.order)
            )
            article_result = await self.db.execute(article_query)
            articles = list(article_result.scalars().all())
            total_articles = len(articles)

            article_num = self._extract_article_number(question)

            if article_num:
                logger.info(
                    f"📌 Looking for article {article_num} in doc (has {total_articles} articles)"
                )

                # Un autre document est nomme : laisser la recherche generale faire
                if self._references_other_document(question, law.title):
                    return [], None

                wanted = _normalize_article_number(article_num)
                for article in articles:
                    if article.number and _normalize_article_number(article.number) == wanted:
                        logger.info(f"✅ Found article {article.number} in priority doc")
                        return [self._chunk_from_article(law, article, "priority")], None

                logger.info(
                    f"❌ Article {article_num} NOT found in doc (has {total_articles} articles)"
                )
                return [], {
                    "requested_article": article_num,
                    "law_title": law.title,
                    "total_articles": total_articles,
                }

            # Pas d'article precis : les premiers articles du document, chacun
            # comme un chunk distinct. Ils etaient auparavant concatenes dans un
            # seul bloc de texte, ce qui rendait toute citation impossible.
            chunks = [
                self._chunk_from_article(law, article, "priority")
                for article in articles[:3]
                if article.content
            ]
            if chunks:
                return chunks, None

            # Document sans article extrait : le texte integral, sans identite
            # d'article.
            if law.content:
                return [ChunkResult(
                    article_id=None,
                    law_id=law.id,
                    content=law.content,
                    excerpt=law.content[:400],
                    reference=law.reference or "",
                    law_title=law.title,
                    type=law.type or "loi",
                    language=law.language,
                    status=law.status or "published",
                    category_id=law.category_id,
                    category_name="Document actif",
                    publication_date=law.publication_date,
                    relevance_score=1.0,
                    source="priority",
                )], None

        except Exception as e:
            logger.error(f"⚠️ Error fetching priority document: {e}")

        return [], None

    @staticmethod
    def _chunk_from_article(law, article, source: str) -> ChunkResult:
        """Construit un ChunkResult a partir des lignes ORM Law + Article."""
        content = article.content or ""
        return ChunkResult(
            article_id=article.id,
            law_id=law.id,
            number=str(article.number) if article.number else None,
            article_title=article.title,
            section=article.section,
            page_number=article.page_number,
            content=content,
            excerpt=content[:400],
            reference=law.reference or "",
            law_title=law.title,
            type=law.type or "loi",
            language=law.language,
            status=law.status or "published",
            category_id=law.category_id,
            category_name="Document actif" if source == "priority" else None,
            publication_date=law.publication_date,
            relevance_score=1.0,
            source=source,
        )

    def _references_other_document(self, question: str, current_title: str) -> bool:
        """
        Check if the question mentions a law document different from the current one.

        Args:
            question: User's question
            current_title: Title of the currently viewed law

        Returns:
            True if another document is explicitly mentioned
        """
        import re
        other_doc_patterns = [
            r'constitution', r'code\s+civil', r'code\s+p[ée]nal',
            r'loi\s+ohada', r'charte', r'décret',
        ]
        question_lower = question.lower()
        for pattern in other_doc_patterns:
            if re.search(pattern, question_lower) and pattern not in current_title.lower():
                logger.info(f"📌 Another document mentioned, skipping priority")
                return True
        return False

    async def _load_or_create_conversation(
        self,
        session_id: Optional[str],
        persona: str,
        language: str
    ) -> Tuple[Conversation, List[Message]]:
        """
        Load existing conversation or create new one.

        Returns:
            (conversation, last_N_messages)
        """
        if session_id:
            # Try to load existing
            stmt = (
                select(Conversation)
                .where(Conversation.session_id == session_id)
                .options(joinedload(Conversation.messages))
            )
            result = await self.db.execute(stmt)
            # .unique() obligatoire apres un joinedload sur une collection :
            # sans lui SQLAlchemy leve InvalidRequestError, ce qui faisait
            # echouer TOUTE requete portant un session_id deja existant.
            conversation = result.unique().scalar_one_or_none()

            if conversation:
                # Get last N messages
                msg_stmt = (
                    select(Message)
                    .where(Message.conversation_id == conversation.id)
                    .order_by(desc(Message.created_at))
                    .limit(self.MAX_HISTORY_MESSAGES)
                )
                msg_result = await self.db.execute(msg_stmt)
                messages = list(msg_result.scalars().all())

                logger.info(f"📖 Loaded conversation: {len(messages)} messages")
                return conversation, list(reversed(messages))

        # Create new conversation
        conversation = Conversation(
            session_id=session_id or str(uuid.uuid4()),
            persona=persona,
            language=language
        )
        self.db.add(conversation)
        await self.db.flush()

        logger.info(f"🆕 New conversation: {conversation.session_id}")
        return conversation, []

    async def _fallback_search(self, question: str, history: List = None) -> List[ChunkResult]:
        """
        Repli multilingue quand la recherche plein texte ne rend rien.

        1. Detecte la langue de la question (FR/EN)
        2. Extrait le contexte documentaire de l'historique
        3. Cherche d'abord dans les documents de la meme langue, puis dans l'autre
        4. La traduction est laissee au modele

        Tourne sur la session async de la requete. Cette methode ouvrait son
        propre moteur synchrone — dans une coroutine — a chaque appel, sans
        jamais le fermer.
        """
        assert isinstance(question, str) and len(question) > 0, "question must be a non-empty string"
        assert history is None or isinstance(history, list), "history must be a list or None"

        prompt_lang = self._detect_language(question)
        logger.info(f"🌐 Detected prompt language: {prompt_lang}")

        article_num = self._extract_article_number(question)
        if not article_num:
            return []

        doc_context = self._extract_document_context(question, history)
        logger.info(
            f"📌 Multilingual search: article {article_num}, context: {doc_context or 'none'}"
        )

        try:
            search_order = ["fr", "en"] if prompt_lang == "fr" else ["en", "fr"]
            # Metacaracteres echappes : un % ou un _ dans le contexte
            # elargissait le motif au lieu d'etre cherche litteralement.
            raw_pattern = doc_context if doc_context else "constitution"
            search_pattern = f"%{escape_like(raw_pattern.lower())}%"

            sql = text("""
                SELECT l.id, l.title, l.reference, l.language, l.type,
                       l.category_id, l.status, l.publication_date,
                       a.id AS article_id, a.number, a.title AS article_title,
                       a.section, a.page_number, a.content AS article_content
                FROM laws l
                JOIN articles a ON a.law_id = l.id
                WHERE (l.language = :lang OR l.language IS NULL)
                  AND (LOWER(l.title) LIKE :pattern ESCAPE '\\'
                       OR LOWER(l.reference) LIKE :pattern ESCAPE '\\')
                  AND (a.number = :num OR a.number ILIKE :num_pattern ESCAPE '\\')
                ORDER BY a.order
                LIMIT 1
            """)

            for lang in search_order:
                result = await self.db.execute(sql, {
                    "lang": lang,
                    "pattern": search_pattern,
                    "num": article_num,
                    "num_pattern": f"%{escape_like(article_num)}%",
                })
                row = result.fetchone()
                if not row:
                    continue

                logger.info(
                    f"📄 Found in {lang.upper()}: {row.title} - article {row.number}"
                )
                content = row.article_content or ""
                return [ChunkResult(
                    article_id=row.article_id,
                    law_id=row.id,
                    number=str(row.number) if row.number else None,
                    article_title=row.article_title,
                    section=row.section,
                    page_number=row.page_number,
                    content=content,
                    excerpt=content[:400],
                    reference=row.reference or "",
                    law_title=row.title,
                    type=row.type or "loi",
                    language=row.language or lang,
                    status=row.status or "published",
                    category_id=row.category_id,
                    publication_date=row.publication_date,
                    relevance_score=1.0,
                    source="fallback",
                )]

            logger.warning(f"⚠️ Article/Section {article_num} not found in any language")

        except Exception as e:
            logger.warning(f"⚠️ Multilingual search failed: {e}")

        return []

    def _detect_language(self, text: str) -> str:
        """Detect if text is primarily French or English."""
        french_words = ["de", "la", "le", "les", "du", "des", "au", "aux", "un", "une", 
                       "et", "ou", "pour", "dans", "sur", "avec", "moi", "explique"]
        english_words = ["the", "of", "and", "or", "for", "in", "on", "with", "me", 
                        "explain", "what", "how", "is", "are", "does", "section"]
        
        text_lower = text.lower()
        fr_count = sum(1 for w in french_words if f" {w} " in f" {text_lower} ")
        en_count = sum(1 for w in english_words if f" {w} " in f" {text_lower} ")
        
        return "fr" if fr_count >= en_count else "en"
    
    def _extract_article_number(self, question: str) -> str:
        """Extract article/section number from question."""
        import re
        
        patterns = [
            r'article\s+(\d+|premier|première)',
            r'section\s+(\d+|first)',
            r"l'article\s+(\d+)",
            r'art\.?\s+(\d+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                num = match.group(1)
                # Normalize
                if num.lower() in ["premier", "première", "first"]:
                    return "1"
                return num
        return None
    
    def _extract_document_context(self, question: str, history: List = None) -> str:
        """
        Extract document context from current question or conversation history.
        Used for follow-up questions like "explique l'article 8" after discussing constitution.
        """
        import re
        
        # Document patterns to look for (FR and EN)
        doc_patterns = [
            (r'constitution', 'constitution'),
            (r'code\s+civil', 'code civil'),
            (r'code\s+p[ée]nal', 'code penal'),
            (r'code\s+du\s+travail', 'code du travail'),
            (r'code\s+de\s+commerce', 'code de commerce'),
            (r'loi\s+ohada', 'loi ohada'),
            (r'civil\s+code', 'code civil'),
            (r'penal\s+code', 'code penal'),
            (r'labor\s+code', 'code du travail'),
        ]
        
        # First check current question
        question_lower = question.lower()
        for pattern, doc_name in doc_patterns:
            if re.search(pattern, question_lower, re.IGNORECASE):
                logger.info(f"📚 Document context from question: {doc_name}")
                return doc_name
        
        # If not in question, check conversation history
        if history:
            for msg in reversed(history):  # Start from most recent
                if hasattr(msg, 'content'):
                    content_lower = msg.content.lower()
                    for pattern, doc_name in doc_patterns:
                        if re.search(pattern, content_lower, re.IGNORECASE):
                            logger.info(f"📚 Document context from history: {doc_name}")
                            return doc_name
                            
                # Also check sources in messages
                if hasattr(msg, 'sources') and msg.sources:
                    try:
                        import json
                        sources = msg.sources if isinstance(msg.sources, list) else json.loads(msg.sources)
                        for source in sources:
                            if isinstance(source, dict):
                                title = source.get('law_title', '').lower()
                                for pattern, doc_name in doc_patterns:
                                    if re.search(pattern, title, re.IGNORECASE):
                                        logger.info(f"📚 Document context from sources: {doc_name}")
                                        return doc_name
                    except:
                        pass
        
        return None

    def _build_prompt(
        self,
        question: str,
        search_results: List,
        history: List[Message],
        persona: str
    ) -> str:
        """
        Build complete prompt with context and history.

        Structure:
        1. Context from search results
        2. Specific article content if question mentions an article
        3. Conversation history
        4. User question
        """
        # L'article explicitement demande est remonte en tete du contexte.
        # Une methode dediee allait auparavant le RECHERCHER EN BASE, avec son
        # propre moteur synchrone, parce que le contexte ne contenait que des
        # extraits de 400 caracteres. Le contexte porte desormais le texte
        # integral : un simple reordonnancement suffit.
        chunks = self._pin_requested_article(question, search_results)

        context_str = build_context_string(chunks)
        context_section = CONTEXT_TEMPLATE.format(context_docs=context_str)

        # Build history
        history_str = format_conversation_history(history) if history else ""

        # Combine
        parts = [context_section]

        if history_str:
            parts.append(f"\n{history_str}\n")

        parts.append(f"\nQuestion actuelle de l'utilisateur:\n{question}")

        return "\n".join(parts)
    
    def _pin_requested_article(
        self, question: str, chunks: List[ChunkResult]
    ) -> List[ChunkResult]:
        """
        Remonte en tete le chunk correspondant a l'article demande.

        Sans acces a la base : le contexte contient deja le texte integral des
        articles retenus, il ne reste qu'a mettre le bon en premier.
        """
        article_num = self._extract_article_number(question)
        if not article_num or not chunks:
            return chunks

        wanted = _normalize_article_number(article_num)
        for index, chunk in enumerate(chunks):
            if chunk.number and _normalize_article_number(chunk.number) == wanted:
                if index == 0:
                    return chunks
                logger.info(f"📌 Article {chunk.number} remonte en tete du contexte")
                return [chunk] + chunks[:index] + chunks[index + 1:]

        return chunks

    def _extract_citations(
        self,
        answer: str,
        chunks: List[ChunkResult]
    ) -> List[Citation]:
        """
        Extrait et valide les citations presentes dans la reponse.

        Cherche "Article 161 du Code OHADA", "article 5 de la Loi...", etc.

        La validation se fait sur les chunks REELLEMENT places dans le prompt :
        couple (loi, numero d'article normalise). Elle se contentait de
        verifier que le nom cite apparaissait dans le titre de la loi, sans
        jamais confronter le NUMERO d'article a ce que le contexte contenait.
        L'extrait et l'identifiant d'article proviennent du chunk : aucun
        aller-retour en base, la ou trois moteurs synchrones etaient ouverts
        pour aller rechercher un texte deja present.
        """
        citations: List[Citation] = []
        seen = set()

        matches = re.finditer(self.CITATION_REGEX, answer, re.IGNORECASE)

        for match in matches:
            # Groupes : 1 = prefixe de code (L/R/D, optionnel), 2 = numero,
            # 3 = texte cite.
            prefix = (match.group(1) or "").strip()
            article_num = f"{prefix} {match.group(2)}".strip()
            law_mention = match.group(3).strip().lower()
            wanted = _normalize_article_number(match.group(2))

            for chunk in chunks:
                if chunk.number and _normalize_article_number(chunk.number) != wanted:
                    continue

                # Le nom cite doit correspondre a la loi du chunk. Conserve
                # comme depart quand plusieurs lois exposent le meme numero.
                mentions_law = (
                    law_mention in (chunk.law_title or "").lower()
                    or law_mention in (chunk.reference or "").lower()
                )
                if not mentions_law and chunk.number is None:
                    continue

                key = (chunk.law_id, wanted)
                if key in seen:
                    break

                excerpt = (chunk.content or chunk.excerpt or "")[:300]
                citations.append(Citation(
                    law_id=chunk.law_id,
                    law_reference=chunk.reference,
                    law_title=chunk.law_title,
                    article_number=chunk.number or article_num,
                    article_id=chunk.article_id,
                    excerpt=excerpt or "Voir document complet pour détails",
                    relevance_score=chunk.relevance_score,
                ))
                seen.add(key)
                break

        logger.info(f"📎 Extracted {len(citations)} citations")
        return citations

    def _create_sources_from_results(
        self, chunks: List[ChunkResult], question: str
    ) -> List[Citation]:
        """
        Fabrique des sources quand la reponse ne cite explicitement rien.

        1. Retient les lois dont le titre recoupe les mots de la question
        2. Privilegie la langue de la question
        3. A defaut, prend le premier chunk (le plus pertinent)
        """
        if not chunks:
            return []

        question_lower = question.lower()
        q_words = {w for w in re.findall(r"\w+", question_lower) if len(w) > 3}

        matched = [
            c for c in chunks[:3]
            if {w for w in re.findall(r"\w+", (c.law_title or "").lower()) if len(w) > 3}
            & q_words
        ]

        if matched:
            q_lang = self._detect_language(question)
            same_lang = [
                c for c in matched if (c.language or "fr").lower().startswith(q_lang)
            ]
            candidates = (same_lang or matched)[:1]
            logger.info(f"🔍 Filtered sources by title match: {[c.law_title for c in candidates]}")
        else:
            candidates = chunks[:1]
            logger.info("🔍 No title match, using top chunk")

        sources: List[Citation] = []
        for chunk in candidates:
            try:
                # L'extrait vient du chunk envoye au modele, jamais d'une
                # requete supplementaire.
                excerpt = (chunk.content or chunk.excerpt or "")[:300]
                sources.append(Citation(
                    law_id=chunk.law_id,
                    law_reference=chunk.reference,
                    law_title=chunk.law_title,
                    article_number=chunk.number,
                    article_id=chunk.article_id,
                    excerpt=excerpt or "Contenu du document",
                    relevance_score=chunk.relevance_score,
                ))
            except Exception as e:
                logger.warning(f"⚠️ Error creating source from chunk: {e}")

        return sources

    def _calculate_confidence(
        self,
        answer: str,
        citations: List[Citation],
        search_results: List
    ) -> float:
        """
        Calculate confidence score based on multiple factors.

        Factors:
        1. Number of citations (more = better)
        2. Average relevance of cited docs
        3. Answer length (too short/long = lower confidence)
        4. Presence of hedge words (probably, maybe = lower)

        Returns:
            Confidence score 0.0-1.0
        """
        score = 0.0

        # Factor 1: Citations (0-0.4)
        if citations:
            citation_score = min(len(citations) / 3, 1.0) * 0.4
            score += citation_score

        # Factor 2: Relevance (0-0.3)
        if citations:
            avg_relevance = sum(c.relevance_score for c in citations) / len(citations)
            score += avg_relevance * 0.3
        elif search_results:
            # No citations but have results.
            # Le diviseur est borne : un objet qui se dit non vide mais dont
            # len() vaut 0 (une doublure, par exemple) provoquait une
            # ZeroDivisionError remontee en erreur de service.
            top = list(search_results[:3])
            if top:
                avg_relevance = sum(r.relevance_score for r in top) / len(top)
                score += avg_relevance * 0.2

        # Factor 3: Answer length (0-0.2)
        answer_len = len(answer.split())
        if 50 <= answer_len <= 500:
            score += 0.2
        elif 30 <= answer_len < 50 or 500 < answer_len <= 700:
            score += 0.1

        # Factor 4: Hedge detection (0-0.1)
        hedge_words = ["peut-être", "probablement", "possiblement", "éventuellement"]
        hedge_count = sum(1 for word in hedge_words if word in answer.lower())
        if hedge_count == 0:
            score += 0.1
        elif hedge_count <= 2:
            score += 0.05

        return round(min(score, 1.0), 2)

    async def _save_interaction(
        self,
        conversation: Conversation,
        question: str,
        answer: str,
        citations: List[Citation],
        confidence: float,
        retrieval_time_ms: int,
        generation_time_ms: int
    ):
        """Save question and answer to database."""
        # User message
        user_msg = Message(
            conversation_id=conversation.id,
            role="user",
            content=question
        )
        self.db.add(user_msg)

        # Assistant message
        sources_json = [c.model_dump() for c in citations]
        assistant_msg = Message(
            conversation_id=conversation.id,
            role="assistant",
            content=answer,
            sources=sources_json,
            confidence=confidence,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=generation_time_ms
        )
        self.db.add(assistant_msg)

        # Update conversation timestamp
        conversation.updated_at = datetime.utcnow()

        await self.db.commit()
        logger.info("💾 Saved interaction to database")

        # Update persona interaction metrics
        try:
            from app.services.persona_service import PersonaService
            persona_service = PersonaService(self.db)
            await persona_service.update_interaction_metrics(
                conversation_id=conversation.id,
                persona=conversation.persona,
                language=conversation.language
            )
            logger.debug("📊 Updated persona interaction metrics")
        except Exception as e:
            logger.warning(f"⚠️ Failed to update persona metrics: {e}")

    async def _handle_no_results(
        self,
        request: RAGRequest,
        retrieval_time_ms: int
    ) -> RAGResponse:
        """Handle case when no documents found."""
        conversation, _ = await self._load_or_create_conversation(
            request.session_id,
            request.persona,
            request.language
        )

        no_results_answer = NO_RESULTS_MESSAGE[request.language]

        await self._save_interaction(
            conversation,
            request.question,
            no_results_answer,
            [],
            0.0,
            retrieval_time_ms,
            0
        )

        return RAGResponse(
            answer=no_results_answer,
            confidence=0.0,
            sources=[],
            session_id=conversation.session_id,
            retrieval_time_ms=retrieval_time_ms,
            generation_time_ms=0,
            total_time_ms=retrieval_time_ms,
            persona=request.persona
        )
