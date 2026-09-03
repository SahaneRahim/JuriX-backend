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
from typing import AsyncIterator, List, Optional, Tuple

from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import joinedload

from app.models.conversation import Conversation, Message
from app.schemas.rag import Citation, RAGRequest, RAGResponse
from app.schemas.search import SearchFilters, SearchRequest
from app.services.gemini_service import get_gemini_service, GeminiServiceError
from app.services.prompts import (
    CONTEXT_TEMPLATE,
    NO_RESULTS_MESSAGE,
    SYSTEM_PROMPTS,
    build_context_string,
    format_conversation_history,
    get_system_prompt,
)
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)


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
    CITATION_REGEX = r"[Aa]rticle\s+(\d+[a-z]?)\s+(?:de\s+)?(?:la\s+)?([A-Z][^,\.]+)"
    
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
        priority_results = []
        if request.law_id:
            priority_results = await self._get_priority_document_context(
                request.law_id, request.question
            )
            logger.info(f"📌 Priority doc {request.law_id}: {len(priority_results)} results")

        # General search
        search_results = await self._retrieve_context(request.question, request.language)

        # Check if article was not found in priority document
        if priority_results and hasattr(priority_results[0], 'article_not_found') and priority_results[0].article_not_found:
            result = priority_results[0]
            message = (
                f"Ce document ne contient pas d'article {result.requested_article}. "
                f"Le document \"{result.title}\" contient {result.total_articles} article(s). "
            )
            if result.total_articles > 0:
                message += f"Veuillez demander un article entre 1 et {result.total_articles}."
            else:
                message += "Ce document ne contient aucun article extrait."

            logger.info(f"📌 Article not found response: {message}")
            return [], RAGResponse(
                answer=message, confidence=1.0, sources=[],
                session_id=conversation.session_id,
                retrieval_time_ms=0, generation_time_ms=0, total_time_ms=0,
                persona=request.persona
            )

        # Merge: priority first, then general (deduplicated)
        if priority_results:
            seen_ids = {r.law_id for r in priority_results}
            filtered = [r for r in search_results if r.law_id not in seen_ids]
            search_results = priority_results + filtered[:3]
            logger.info(f"📚 Merged: {len(priority_results)} priority + {len(filtered[:3])} general")

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
            search_results = await self._retrieve_context(
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

    async def _retrieve_context(
        self,
        question: str,
        language: str
    ) -> List:
        """
        Retrieve relevant documents using SearchService.

        Uses text search for reliability (hybrid requires embeddings).
        Searches all languages to maximize results.
        Extracts keywords from natural language questions for better search.
        """
        # Extract keywords from question for better la recherche plein texte results
        search_query = self.extract_keywords(question)
        logger.info(f"🔍 RAG Search: original='{question[:50]}...' keywords='{search_query}'")
        
        search_response = await self.search_service.search(
            SearchRequest(
                query=search_query,
                mode="text",  # Use text search (hybrid requires embeddings)
                filters=SearchFilters(status="published"),  # No language filter
                limit=self.TOP_K_DOCUMENTS
            )
        )

        logger.info(
            f"📚 Retrieved {len(search_response.results)} documents "
            f"in {search_response.search_time_ms}ms"
        )
        
        # Debug: Log first result if any
        if search_response.results:
            first = search_response.results[0]
            logger.info(f"📄 First result: {first.reference} - {first.title}")
        else:
            logger.warning(f"⚠️ No search results for query: {question[:100]}")

        return search_response.results

    async def _get_priority_document_context(self, law_id: int, question: str) -> List:
        """
        Fetch context from the active document being viewed.
        
        When a user is viewing a specific document and asks a question,
        prioritize content from that document to provide focused answers.
        
        If a specific article is requested but not found, returns a special
        result indicating the article doesn't exist in this document.
        
        Args:
            law_id: ID of the law being viewed
            question: User's question (used to find relevant articles)
            
        Returns:
            List of search-like results from the priority document
            May include special 'article_not_found' marker
        """
        assert isinstance(law_id, int) and law_id > 0, "law_id must be a positive integer"
        assert isinstance(question, str) and len(question) > 0, "question must be a non-empty string"

        from app.models.law import Law, Article
        
        try:
            # Get the law with its articles
            query = select(Law).where(Law.id == law_id)
            result = await self.db.execute(query)
            law = result.scalar_one_or_none()
            
            if not law:
                logger.warning(f"⚠️ Priority law_id {law_id} not found")
                return []
            
            # Get articles for this law
            article_query = select(Article).where(Article.law_id == law_id).order_by(Article.order)
            article_result = await self.db.execute(article_query)
            articles = article_result.scalars().all()
            total_articles = len(articles)
            
            # Check if question asks about a specific article
            article_num = self._extract_article_number(question)
            
            if article_num:
                logger.info(f"📌 Looking for article {article_num} in doc (has {total_articles} articles)")
                
                # If another document is mentioned, let general search handle it
                if self._references_other_document(question, law.title):
                    return []
                
                # Look for the article in the current document
                if articles:
                    for article in articles:
                        if article.number and (
                            str(article.number) == str(article_num) or
                            article_num.lower() in str(article.number).lower()
                        ):
                            logger.info(f"✅ Found article {article.number} in priority doc")
                            return [self._build_priority_result(
                                law, article.content or "", total_articles
                            )]
                
                # Article not found in this document!
                logger.info(f"❌ Article {article_num} NOT found in doc (has {total_articles} articles)")
                return [self._build_priority_result(
                    law, "", total_articles,
                    article_not_found=True, requested_article=article_num
                )]
            
            # No specific article requested - use law content and first few articles
            combined_content = law.content or ""
            if articles:
                for article in articles[:3]:
                    if article.content:
                        combined_content += f"\n\nArticle {article.number}:\n{article.content}"
            
            if combined_content:
                return [self._build_priority_result(law, combined_content, total_articles)]
                
        except Exception as e:
            logger.error(f"⚠️ Error fetching priority document: {e}")
        
        return []

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

    def _build_priority_result(
        self, law, content: str, total_articles: int,
        article_not_found: bool = False, requested_article: str = ""
    ):
        """
        Build a PriorityResult dataclass instance from law data.

        Args:
            law: Law ORM object
            content: Article or combined content text
            total_articles: Total number of articles in this law
            article_not_found: Whether the requested article was missing
            requested_article: The article number that was requested

        Returns:
            PriorityResult dataclass instance
        """
        from dataclasses import dataclass

        @dataclass
        class PriorityResult:
            law_id: int
            title: str
            reference: str
            content: str
            source_language: str
            category_name: str = "Document actif"
            relevance_score: float = 1.0
            highlights: dict = None
            matched_articles: list = None
            article_not_found: bool = False
            requested_article: str = ""
            total_articles: int = 0

            def __post_init__(self):
                self.highlights = {"content": self.content if self.content else ""}
                self.matched_articles = []

        return PriorityResult(
            law_id=law.id,
            title=law.title,
            reference=law.reference or "",
            content=content,
            source_language=law.language or "fr",
            total_articles=total_articles,
            article_not_found=article_not_found,
            requested_article=requested_article,
        )

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

    async def _fallback_search(self, question: str, history: List = None) -> List:
        """
        Multilingual fallback search when la recherche plein texte returns no results.
        
        Logic:
        1. Detect prompt language (FR/EN)
        2. Extract document context from conversation history
        3. Search in same-language documents first
        4. If not found, search in other-language documents
        5. Translation handled by Gemini in response generation
        """
        assert isinstance(question, str) and len(question) > 0, "question must be a non-empty string"
        assert history is None or isinstance(history, list), "history must be a list or None"

        import re
        
        # Detect prompt language based on keywords
        prompt_lang = self._detect_language(question)
        logger.info(f"🌐 Detected prompt language: {prompt_lang}")
        
        # Extract article/section number from question
        article_num = self._extract_article_number(question)
        if not article_num:
            return []
        
        # Extract document context from conversation history
        doc_context = self._extract_document_context(question, history)
        logger.info(f"📌 Multilingual search: article {article_num}, context: {doc_context or 'none'}")
        
        try:
            from sqlalchemy import text, create_engine
            from app.core.config import settings
            from dataclasses import dataclass
            
            engine = create_engine(settings.DATABASE_URL.replace('+asyncpg', ''))
            
            @dataclass
            class FallbackResult:
                law_id: int
                title: str
                reference: str
                content: str
                source_language: str
                category_name: str = "Loi"
                relevance_score: float = 1.0
                highlights: dict = None
                matched_articles: list = None
                
                def __post_init__(self):
                    self.highlights = {"content": self.content}
                    self.matched_articles = []
            
            with engine.connect() as conn:
                # Define search order based on prompt language
                if prompt_lang == "fr":
                    search_order = [("fr", "Article"), ("en", "Section")]
                else:
                    search_order = [("en", "Section"), ("fr", "Article")]
                
                # Use document context or default to constitution
                search_pattern = f"%{doc_context}%" if doc_context else "%constitution%"
                
                for lang, term_type in search_order:
                    # Search for laws in this language
                    sql = text("""
                        SELECT l.id, l.title, l.reference, l.language,
                               a.number, a.content as article_content
                        FROM laws l
                        LEFT JOIN articles a ON a.law_id = l.id AND a.number = :article_num
                        WHERE (l.language = :lang OR l.language IS NULL)
                          AND (LOWER(l.title) LIKE :pattern 
                               OR LOWER(l.reference) LIKE :pattern)
                        LIMIT 1
                    """)
                    
                    result = conn.execute(sql, {
                        "lang": lang, 
                        "article_num": article_num,
                        "pattern": search_pattern
                    })
                    row = result.fetchone()
                    
                    if row and row[5]:  # Found law with article content
                        law_id, title, reference, law_lang, number, article_content = row
                        logger.info(f"📄 Found in {lang.upper()}: {title} - Article/Section {number}")
                        
                        return [FallbackResult(
                            law_id=law_id,
                            title=title,
                            reference=reference,
                            content=article_content,
                            source_language=lang or prompt_lang
                        )]
                    elif row:  # Found law but no article - try direct article search
                        law_id = row[0]
                        # Try ILIKE pattern for article number
                        sql2 = text("""
                            SELECT number, content FROM articles 
                            WHERE law_id = :law_id 
                              AND (number = :num OR number ILIKE :pattern)
                            LIMIT 1
                        """)
                        result2 = conn.execute(sql2, {
                            "law_id": law_id, 
                            "num": article_num,
                            "pattern": f"%{article_num}%"
                        })
                        row2 = result2.fetchone()
                        if row2:
                            logger.info(f"📄 Found article {row2[0]} in {lang.upper()}: {row[1]}")
                            return [FallbackResult(
                                law_id=law_id,
                                title=row[1],
                                reference=row[2],
                                content=row2[1],
                                source_language=lang or prompt_lang
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
        # Build context
        context_str = build_context_string(search_results)
        context_section = CONTEXT_TEMPLATE.format(context_docs=context_str)

        # Check if question mentions a specific article and get its content
        article_section = self._get_article_content_for_prompt(question, search_results)
        
        # Build history
        history_str = format_conversation_history(history) if history else ""

        # Combine
        parts = [context_section]
        
        if article_section:
            parts.append(f"\n{article_section}\n")

        if history_str:
            parts.append(f"\n{history_str}\n")

        parts.append(f"\nQuestion actuelle de l'utilisateur:\n{question}")

        return "\n".join(parts)
    
    def _get_article_content_for_prompt(self, question: str, search_results: List) -> str:
        """
        Detect if question asks about a specific article/section and fetch its content.
        Supports both French and English questions.
        Full article content is sent to ensure complete context for LLM.
        """
        assert isinstance(question, str) and len(question) > 0, "question must be a non-empty string"
        assert isinstance(search_results, list), "search_results must be a list"

        # Use shared method for article number extraction
        article_num = self._extract_article_number(question)
        if not article_num:
            return ""
        
        prompt_lang = self._detect_language(question)
        logger.info(f"📑 Detected article/section {article_num} in {prompt_lang.upper()} question")
        
        # Get FULL article content from PostgreSQL (no truncation)
        try:
            from sqlalchemy import create_engine, text
            from app.core.config import settings
            
            # FIRST: Check if FallbackResult already has the article content
            for result in search_results[:3]:
                result_content = getattr(result, 'content', None)
                # If result has content and appears to be article content (not truncated law content)
                if result_content and len(result_content) < 5000:
                    source_lang = getattr(result, 'source_language', None)
                    translation_note = ""
                    if source_lang and source_lang != prompt_lang:
                        if prompt_lang == "fr":
                            translation_note = "\n\n⚠️ Ce contenu est extrait de la version anglaise. Merci de traduire en français dans ta réponse."
                        else:
                            translation_note = "\n\n⚠️ This content is from the French version. Please translate to English in your response."
                    
                    logger.info(f"📄 Using article content from FallbackResult ({len(result_content)} chars)")
                    return f"""
=== ARTICLE/SECTION SPÉCIFIQUE DEMANDÉ ===
Document: {result.title} ({getattr(result, 'reference', '')})
Article/Section {article_num}:

CONTENU COMPLET:
{result_content}{translation_note}
==========================================
"""
            
            # SECOND: Try to find in database
            engine = create_engine(settings.DATABASE_URL.replace('+asyncpg', ''))
            with engine.connect() as conn:
                for result in search_results[:3]:
                    sql = text("""
                        SELECT number, title, content FROM articles 
                        WHERE law_id = :law_id AND (number = :number OR number ILIKE :pattern)
                        LIMIT 1
                    """)
                    res = conn.execute(sql, {
                        "law_id": result.law_id,
                        "number": article_num,
                        "pattern": f"%{article_num}%"
                    })
                    row = res.fetchone()
                    if row:
                        number, title, content = row
                        source_lang = getattr(result, 'source_language', None)
                        translation_note = ""
                        if source_lang and source_lang != prompt_lang:
                            if prompt_lang == "fr":
                                translation_note = "\n\n⚠️ Ce contenu est extrait de la version anglaise. Merci de traduire en français dans ta réponse."
                            else:
                                translation_note = "\n\n⚠️ This content is from the French version. Please translate to English in your response."
                        
                        logger.info(f"📄 Found article/section {number} in DB ({len(content)} chars)")
                        return f"""
=== ARTICLE/SECTION SPÉCIFIQUE DEMANDÉ ===
Document: {result.title} ({result.reference})
Article/Section {number}: {title or ''}

CONTENU COMPLET:
{content}{translation_note}
==========================================
"""
        except Exception as e:
            logger.warning(f"⚠️ Failed to get article for prompt: {e}")
        
        return ""

    def _extract_citations(
        self,
        answer: str,
        search_results: List
    ) -> List[Citation]:
        """
        Extract and validate citations from answer.

        Looks for patterns like:
        - "Article 161 de la Loi OHADA"
        - "article 5 du Code civil"

        Validates against search_results to ensure cited docs exist.
        """
        citations = []
        seen = set()  # Dedup

        # Regex extraction
        matches = re.finditer(self.CITATION_REGEX, answer, re.IGNORECASE)

        for match in matches:
            article_num = match.group(1)
            law_mention = match.group(2).strip()

            # Find matching document in search results
            for result in search_results:
                # Match by law title or reference
                if (law_mention.lower() in result.title.lower() or
                    law_mention.lower() in result.reference.lower()):

                    key = (result.law_id, article_num)
                    if key not in seen:
                        # Extract relevant excerpt
                        excerpt = self._find_article_excerpt(
                            result, article_num
                        )

                        citations.append(Citation(
                            law_id=result.law_id,
                            law_reference=result.reference,
                            law_title=result.title,
                            article_number=article_num,
                            excerpt=excerpt,
                            relevance_score=result.relevance_score
                        ))
                        seen.add(key)
                        break

        logger.info(f"📎 Extracted {len(citations)} citations")
        return citations
    
    def _create_sources_from_results(self, search_results: List, question: str) -> List[Citation]:
        """
        Create source citations from search results with intelligent filtering.
        1. Identifies relevant laws by matching question words to law titles.
        2. Prioritizes laws matching the question's language.
        3. Falls back to Top 1 result if no specific law is identified.
        """
        sources = []
        article_num = self._extract_article_number(question)
        question_lower = question.lower()
        
        # 1. candidate selection
        candidates = []
        
        # Check for title matches
        matched_results = []
        for res in search_results[:3]:
            # Extract significant words from title (skip short words)
            title_words = {w for w in re.findall(r'\w+', res.title.lower()) if len(w) > 3}
            q_words = {w for w in re.findall(r'\w+', question_lower) if len(w) > 3}
            
            # Check overlap "Constitution", "Charte", etc.
            if title_words.intersection(q_words):
                matched_results.append(res)
                
        if matched_results:
            # 2. Filter by Language (Deduplication)
            q_lang = self._detect_language(question)
            same_lang_matches = [
                res for res in matched_results 
                if (getattr(res, 'language', 'fr') or 'fr').lower().startswith(q_lang)
            ]
            
            # Use same-language matches if available, otherwise all matches
            # Take Top 1 to ensure "only the law used" is cited
            candidates = same_lang_matches[:1] if same_lang_matches else matched_results[:1]
            logger.info(f"🔍 Filtered sources by title match: {[c.title for c in candidates]}")
        else:
            # 3. No match found - Use Top 1 result (most relevant)
            candidates = search_results[:1]
            logger.info("🔍 No title match, using top result")

        # Create citation
        for result in candidates:
            try:
                excerpt = ""
                if article_num:
                    excerpt = self._find_article_excerpt(result, article_num)
                
                if not excerpt and hasattr(result, 'highlights') and result.highlights:
                    excerpt = result.highlights.get('content', '')[:300]
                
                if not excerpt and hasattr(result, 'content'):
                    excerpt = result.content[:300] if result.content else ""
                
                sources.append(Citation(
                    law_id=result.law_id,
                    law_reference=getattr(result, 'reference', ''),
                    law_title=result.title,
                    article_number=article_num,
                    excerpt=excerpt[:300] if excerpt else "Contenu du document",
                    relevance_score=getattr(result, 'relevance_score', 0.8)
                ))
            except Exception as e:
                logger.warning(f"⚠️ Error creating source from result: {e}")
        
        return sources

    def _find_article_excerpt(self, result, article_num: str) -> str:
        """Find excerpt for specific article from search result."""
        logger.debug(f"DEBUG: _find_article_excerpt called for article {article_num}, law_id={result.law_id}")
        
        # PRIORITY 1: Query PostgreSQL articles table directly
        try:
            from sqlalchemy import create_engine, text
            from app.core.config import settings
            
            engine = create_engine(settings.DATABASE_URL.replace('+asyncpg', ''))
            with engine.connect() as conn:
                # Try multiple number formats
                number_variants = [article_num]
                
                # Handle "premier"/"première" for article 1
                if article_num == "1":
                    number_variants.extend(["premier", "première", "1er", "1ère"])
                elif article_num.lower() in ["premier", "première"]:
                    number_variants.extend(["1", "1er", "1ère"])
                
                logger.debug(f"DEBUG: Trying number variants: {number_variants}")
                
                # Try each variant
                for num in number_variants:
                    sql = text("""
                        SELECT content FROM articles 
                        WHERE law_id = :law_id AND (number = :number OR number ILIKE :pattern)
                        LIMIT 1
                    """)
                    res = conn.execute(sql, {
                        "law_id": result.law_id, 
                        "number": num,
                        "pattern": f"%{num}%"
                    })
                    row = res.fetchone()
                    if row and row[0]:
                        content = row[0]
                        logger.debug(f"DEBUG: FOUND article {num} content: {content[:100]}...")
                        return content[:300] if len(content) > 300 else content
                        
                logger.debug(f"DEBUG: Article {article_num} NOT FOUND in PostgreSQL")
        except Exception as e:
            logger.debug(f"DEBUG: PostgreSQL query failed: {e}")

        # FALLBACK 1: Check matched_articles from la recherche plein texte
        if hasattr(result, 'matched_articles') and result.matched_articles:
            for article in result.matched_articles:
                if hasattr(article, 'number') and article.number == article_num:
                    if hasattr(article, 'snippet'):
                        return article.snippet[:300]

        # FALLBACK 2: Use highlights from la recherche plein texte
        if hasattr(result, 'highlights'):
            content = result.highlights.get("content", "")
            if content:
                logger.debug(f"DEBUG: Extraits de recherche en repli")
                return content[:300]

        return "Voir document complet pour détails"

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
            # No citations but have results
            avg_relevance = sum(r.relevance_score for r in search_results[:3]) / min(3, len(search_results))
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
