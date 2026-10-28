"""
Pydantic schemas for RAG chatbot service.

Defines request/response models for RAG endpoints, citations, and conversation management.
"""

import json
from datetime import date, datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from app.core.personas import PROFIL_GENERAL, normaliser_profil

# Ce qu'un texte plus recent fait a un texte cite : les cles de
# textes_successifs.NATURES, la ou les liens prouves sont definis.
NatureDuLien = Literal["modifie", "complete", "abroge", "abroge_partiellement", "remplace"]


class RAGRequest(BaseModel):
    """Request schema for RAG ask endpoint."""

    question: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        description="User question",
        examples=["Quelle est la responsabilité des dirigeants de société?"]
    )
    # Absent, null ou vide : le profil general, celui d'un utilisateur qui n'a
    # pas dit qui il est. Il valait « citoyen », si bien qu'un refus de
    # repondre et un citoyen etaient indiscernables, dans la reponse comme dans
    # les statistiques.
    persona: str = Field(
        PROFIL_GENERAL,
        description="Profil : citoyen, avocat, entrepreneur, étudiant, ou general (absent)",
    )
    language: str = Field(
        "fr",
        description="Response language (fr/en)"
    )
    session_id: Optional[str] = Field(
        None,
        max_length=100,
        description="Session ID for conversation continuity",
    )
    stream: bool = Field(
        False,
        description="Enable streaming response"
    )
    law_id: Optional[int] = Field(
        None,
        description="Current law ID being viewed (prioritizes this document in search)"
    )
    top_k: Optional[int] = Field(
        5,
        ge=1,
        le=20,
        description="Number of results to retrieve"
    )

    @field_validator("persona", mode="before")
    @classmethod
    def validate_persona(cls, v: object) -> str:
        """
        Forme canonique : « Etudiant » et la forme NFD donnent « étudiant »,
        null et '' donnent le profil general. Une valeur inconnue : 422.
        """
        return normaliser_profil(v)

    @field_validator("language")
    @classmethod
    def validate_language(cls, v: str) -> str:
        if v not in {"fr", "en"}:
            raise ValueError("Language must be 'fr' or 'en'")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "question": "Quelle est la responsabilité des dirigeants de société?",
                "persona": "avocat",
                "language": "fr",
                "session_id": "abc123",
                "stream": False
            }
        }
    }


class Citation(BaseModel):
    """Source citation with structured data for frontend linking."""

    law_id: int = Field(..., description="Database ID of law")
    law_reference: str = Field(..., description="Law reference (e.g., LOI-2024-001)")
    law_title: str = Field(..., description="Law title")
    article_number: Optional[str] = Field(None, description="Article number if specific")
    # Identifiant de ligne, et non un numero extrait du texte de la reponse :
    # le front peut ouvrir directement l'article cite. Optionnel, une citation
    # peut porter sur un document sans article extrait.
    article_id: Optional[int] = Field(None, description="ID de l'article cité, si résolu")
    excerpt: str = Field(..., max_length=300, description="Relevant excerpt")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")

    model_config = {
        "json_schema_extra": {
            "example": {
                "law_id": 156,
                "law_reference": "LOI-2024-001",
                "law_title": "Code OHADA",
                "article_number": "161",
                "excerpt": "Les dirigeants sont responsables...",
                "relevance_score": 0.92
            }
        }
    }


# ==================== CITATIONS LUES DANS LA REPONSE ====================
#
# POURQUOI. Les sources n'etaient listees qu'en fin de reponse, et une citation
# que rien ne portait etait jetee sans un mot : le lecteur voyait « l'article
# 312 du Code pénal » sans source ni alerte. Ces champs disent OU chaque
# citation se trouve dans le texte affiche, ce qu'elle designe, et lesquelles
# ne correspondent a aucun texte consulte. Tous sont NOUVEAUX et facultatifs :
# `sources` garde son contenu et son sens pour le front deja deploye.

# Une seule regle de date pour tout ce que le front affiche d'un texte : la page
# (date_signature), les listes et le chat (rag_service._date_affichee).
_DATE_DU_TEXTE = (
    "Date de l'acte, meme regle que `date_signature` de GET /laws/{id} : null quand "
    "l'intitule ne donne que l'annee ou ne date que sa publication au Journal officiel"
)


class LienProuve(BaseModel):
    """
    Un texte plus recent qui touche la source citee, tel que releve dans
    app/services/liens_entre_textes.py (disposition relue en base).
    """

    nature: NatureDuLien = Field(..., description="Ce que fait le texte plus recent")
    libelle: str = Field(..., description="Le participe a afficher : « modifié », « abrogé en partie »…")
    par_titre: str = Field(
        ..., description="Le texte qui touche, article et date compris : « la Loi N°2008/001 du 14 avril 2008 »"
    )
    par_date: date = Field(..., description="Date de ce texte")
    par_reference: Optional[str] = Field(
        None, description="Sa reference en base ; null quand il manque au corpus"
    )
    par_law_id: Optional[int] = Field(
        None, description="Son identifiant en base, pour le lien ; null s'il manque au corpus"
    )
    articles: List[str] = Field(
        default_factory=list, description="Articles touches du texte cite ; vide : le texte entier"
    )
    par_articles: List[str] = Field(
        default_factory=list, description="Articles du texte plus recent qui portent le changement"
    )
    portee: Optional[str] = Field(None, description="Ce qui change, en une phrase, quand c'est releve")


class StatutProuve(BaseModel):
    """
    Le statut juridique PROUVE d'une source : rien n'y est suppose.

    `etat` null et `liens` vide ne veulent pas dire « en vigueur » : aucun
    texte plus recent n'est connu pour l'avoir touchee. L'interface ecrit alors
    une mention neutre (« version d'origine, non consolidée »).
    """

    etat: Optional[NatureDuLien] = Field(
        None,
        description=(
            "Le plus fort des liens qui visent la source (abroge > remplace > "
            "abroge_partiellement > modifie > complete) ; null : rien de prouve"
        ),
    )
    liens: List[LienProuve] = Field(default_factory=list, description="Du plus recent au plus ancien")
    lignes: List[str] = Field(
        default_factory=list,
        description=(
            "Les lignes de statut que le modele a lues sous l'extrait. Pour audit : "
            "elles citent des references internes et ne s'affichent pas telles quelles"
        ),
    )


class CitationPositionnee(BaseModel):
    """
    Une citation reperee dans le texte de la reponse, et ce qu'elle designe.

    `debut` et `fin` sont des indices JavaScript — unites de code UTF-16 —
    dans le texte final de la reponse : la concatenation des `chunk` du flux,
    ou `answer` hors flux, markdown compris. `texte.slice(debut, fin)` rend
    exactement `texte_cite`. Ils coincident avec les indices Python tant que la
    reponse ne contient aucun caractere hors du plan multilingue de base (un
    emoji, par exemple).
    """

    debut: int = Field(..., ge=0, description="Indice UTF-16 du premier caractere cite")
    fin: int = Field(..., ge=0, description="Indice UTF-16 qui suit le dernier caractere cite")
    texte_cite: str = Field(..., description="Le passage, tel qu'ecrit : « article 37 du Code du travail »")
    law_id: int = Field(..., description="Le texte designe")
    article: Optional[str] = Field(
        None, description="Numero de l'article cite ; null pour un texte cite sans article"
    )
    article_id: Optional[int] = Field(None, description="La ligne d'article en base, pour l'apercu")
    reference: str = Field(..., description="Reference du texte en base")
    titre: str = Field(..., description="Intitule du texte")
    date_du_texte: Optional[date] = Field(None, description=_DATE_DU_TEXTE)
    statut: StatutProuve = Field(default_factory=StatutProuve)


class CitationIntrouvable(BaseModel):
    """
    Une citation que rien de ce que le modele a lu ne porte : a signaler, pas
    a jeter. Memes indices que CitationPositionnee.
    """

    debut: int = Field(..., ge=0)
    fin: int = Field(..., ge=0)
    texte_cite: str
    raison: Literal["article_non_consulte", "texte_non_consulte"] = Field(
        ...,
        description=(
            "article_non_consulte : le texte cite a ete consulte, pas cet article ; "
            "texte_non_consulte : aucun texte consulte ne repond a ce nom"
        ),
    )


class SourceCitee(BaseModel):
    """
    Une source citee dans la reponse, une fois par article (ou par texte cite
    sans article). Memes champs qu'une `Citation`, pour que l'affichage des
    sources serve aux deux listes, plus la date et le statut du texte.
    """

    law_id: int
    law_reference: str
    law_title: str
    article_number: Optional[str] = None
    article_id: Optional[int] = None
    excerpt: str = Field("", max_length=300)
    relevance_score: float = Field(0.0, ge=0.0, le=1.0)
    date_du_texte: Optional[date] = Field(None, description=_DATE_DU_TEXTE)
    statut: StatutProuve = Field(default_factory=StatutProuve)


class TexteConsulte(BaseModel):
    """
    Un texte remis au modele et que la reponse ne cite pas.

    Distinct d'une source : le presenter comme une citation ferait croire que
    la reponse s'appuie dessus. C'est ici que finit la source « de repli »
    que `sources` garde pour le front deja deploye.
    """

    law_id: int
    law_reference: str
    law_title: str
    date_du_texte: Optional[date] = Field(None, description=_DATE_DU_TEXTE)
    articles: List[str] = Field(
        default_factory=list, description="Numeros des articles lus par le modele, dans l'ordre"
    )
    statut: StatutProuve = Field(default_factory=StatutProuve, description="Statut du texte entier")


class TexteTrouve(BaseModel):
    """Un texte trouve par la recherche, dans l'evenement de progression."""

    law_id: int
    law_reference: str
    law_title: str
    date_du_texte: Optional[date] = Field(None, description=_DATE_DU_TEXTE)
    nb_articles: int = Field(..., description="Articles de ce texte remontes par la recherche")


class Progression(BaseModel):
    """
    Une etape du chat en flux, avant le premier mot de la reponse.

    Trois points animes ne disaient rien pendant les secondes de recherche et
    de reflexion du modele. Ces evenements disent ou en est la reponse, avec un
    libelle lisible dans la langue de la question.
    """

    etape: Literal["recherche", "textes_trouves", "redaction"]
    libelle: str = Field(..., description="La ligne a afficher, dans la langue de la question")
    nb_articles: Optional[int] = Field(None, description="textes_trouves : articles trouves")
    nb_textes: Optional[int] = Field(None, description="textes_trouves : textes distincts")
    textes: Optional[List[TexteTrouve]] = Field(
        None, description="textes_trouves : du mieux classe au moins bien classe"
    )


class RAGResponse(BaseModel):
    """Response schema for RAG ask endpoint."""

    answer: str = Field(..., description="Generated answer")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    sources: List[Citation] = Field(default_factory=list, description="Source citations")
    session_id: str = Field(..., description="Session ID for continuation")
    retrieval_time_ms: int = Field(..., description="Time for document retrieval")
    generation_time_ms: int = Field(..., description="Time for answer generation")
    total_time_ms: int = Field(..., description="Total processing time")
    persona: str = Field(..., description="Profil effectif de la reponse, general compris")
    # Defaut OBLIGATOIRE, pas une commodite : `RAGResponse` est construite sans
    # ce champ a plusieurs endroits (sortie anticipee « article absent »,
    # `_handle_no_results`, doublures de tests). Un champ requis les casserait.
    intent: str = Field(
        "juridique",
        description="Intention retenue par le routeur : juridique | smalltalk | meta | hors_sujet",
    )
    # Les citations lues dans `answer` (voir CitationPositionnee). Listes vides
    # par defaut : les reponses construites sans elles (doublures, sorties
    # anticipees) restent valides, et le front deploye les ignore.
    citations: List[CitationPositionnee] = Field(
        default_factory=list, description="Citations reperees dans `answer`, dans l'ordre du texte"
    )
    citations_introuvables: List[CitationIntrouvable] = Field(
        default_factory=list, description="Citations qu'aucun texte consulte ne porte"
    )
    articles_cites: List[SourceCitee] = Field(
        default_factory=list, description="Sources citees, une par article, dans l'ordre du texte"
    )
    textes_consultes_non_cites: List[TexteConsulte] = Field(
        default_factory=list, description="Textes remis au modele que la reponse ne cite pas"
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "answer": "La responsabilité des dirigeants de société...",
                "confidence": 0.85,
                "sources": [
                    {
                        "law_id": 156,
                        "law_reference": "LOI-2024-001",
                        "law_title": "Code OHADA",
                        "article_number": "161",
                        "excerpt": "Les dirigeants sont responsables...",
                        "relevance_score": 0.92
                    }
                ],
                "session_id": "abc123",
                "retrieval_time_ms": 180,
                "generation_time_ms": 2400,
                "total_time_ms": 2600,
                "persona": "citoyen"
            }
        }
    }


class RAGStreamChunk(BaseModel):
    """Streaming response chunk for SSE."""

    chunk: str = Field(..., description="Response text chunk")
    done: bool = Field(False, description="Whether generation is complete")
    sources: Optional[List[Citation]] = Field(
        None,
        description="Sources (only in final chunk)"
    )
    confidence: Optional[float] = Field(
        None,
        description="Confidence (only in final chunk)"
    )
    # Aussi dans les evenements de progression : une reponse ARRETEE par le
    # lecteur n'a jamais de dernier evenement, et sans identifiant la question
    # suivante ouvrait une autre conversation.
    session_id: Optional[str] = Field(
        None,
        description="Session ID (progression events and final chunk)"
    )
    # Le profil effectif de la conversation. Il peut differer de celui demande
    # sur une conversation existante : un changement de profil en ouvre une
    # nouvelle, avec un autre session_id.
    persona: Optional[str] = Field(
        None,
        description="Profil de la conversation (progression events and final chunk)"
    )
    intent: Optional[str] = Field(
        None,
        description="Intention retenue par le routeur (only in final chunk)"
    )
    error: Optional[str] = Field(
        None,
        description="Error message (only in a terminal chunk that failed)"
    )
    # Le statut HTTP d'un flux est fige a 200 des le premier octet : la cause
    # d'une panne ne peut voyager que dans le corps. Sans ce code, l'interface
    # ne distinguait plus un quota epuise d'une panne serveur, la ou `ask`
    # repond 429 ou 503.
    error_code: Optional[Literal["quota", "overloaded", "server"]] = Field(
        None,
        description="Nature de la panne : quota, overloaded, server"
    )
    # Evenement de progression : `chunk` vide, `done` faux, et ce champ. Le
    # front deploye n'y voit ni texte, ni fin, ni erreur : il l'ignore.
    progression: Optional[Progression] = Field(
        None, description="Etape en cours, avant le premier mot (evenements de progression)"
    )
    citations: Optional[List[CitationPositionnee]] = Field(
        None, description="Citations reperees dans le texte final (only in final chunk)"
    )
    citations_introuvables: Optional[List[CitationIntrouvable]] = Field(
        None, description="Citations qu'aucun texte consulte ne porte (only in final chunk)"
    )
    articles_cites: Optional[List[SourceCitee]] = Field(
        None, description="Sources citees, une par article (only in final chunk)"
    )
    textes_consultes_non_cites: Optional[List[TexteConsulte]] = Field(
        None, description="Textes remis au modele et non cites (only in final chunk)"
    )

    def en_json(self) -> str:
        """
        L'evenement tel qu'il part dans le flux.

        Au premier niveau, un champ vide est omis, comme avant : le front
        deploye teste `evenement.chunk`, `done` et `error`, et la forme de ses
        evenements ne change pas. Les structures nouvelles, elles, partent
        ENTIERES, `null` compris : avec `exclude_none` recursif, un statut sans
        lien perdait sa cle `etat`, et la meme citation n'aurait pas eu la meme
        forme dans le flux et dans la reponse de `/ask`.
        """
        donnees = self.model_dump(mode="json", exclude_none=True, exclude=set(_CHAMPS_ENTIERS))
        for champ in _CHAMPS_ENTIERS:
            valeur = getattr(self, champ)
            if valeur is None:
                continue
            if isinstance(valeur, list):
                donnees[champ] = [element.model_dump(mode="json") for element in valeur]
            else:
                # La progression : ses compteurs n'existent qu'a une etape, et
                # sont omis aux autres. Un texte trouve, lui, garde toutes ses
                # cles : sans date connue, `date_du_texte` vaut null, comme dans
                # les listes du dernier evenement.
                donnees[champ] = valeur.model_dump(mode="json", exclude_none=True)
                if getattr(valeur, "textes", None) is not None:
                    donnees[champ]["textes"] = [t.model_dump(mode="json") for t in valeur.textes]
        return json.dumps(donnees, ensure_ascii=False)


# Champs de RAGStreamChunk serialises sans `exclude_none` (voir en_json). La
# progression garde l'omission : ses compteurs n'existent qu'a une etape.
_CHAMPS_ENTIERS = (
    "progression", "citations", "citations_introuvables", "articles_cites",
    "textes_consultes_non_cites",
)


class MessageResponse(BaseModel):
    """Single message in conversation."""

    id: int
    role: str  # "user" or "assistant"
    content: str
    sources: Optional[List[Citation]] = None
    confidence: Optional[float] = None
    created_at: datetime

    model_config = {
        "from_attributes": True
    }


class ConversationSummary(BaseModel):
    """
    Une conversation dans la liste laterale — sans ses messages.

    Le detail complet passe par `ConversationResponse`. Charger les messages
    ici ferait transiter tout l'historique du compte a chaque ouverture du
    panneau.
    """

    session_id: str
    title: Optional[str] = None
    persona: str
    language: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ConversationResponse(BaseModel):
    """Conversation history response."""

    session_id: str
    title: Optional[str] = None
    persona: str
    language: str
    created_at: datetime
    updated_at: datetime
    messages: List[MessageResponse]

    model_config = {
        "from_attributes": True
    }
