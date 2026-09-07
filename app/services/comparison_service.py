"""
Comparaison de deux regimes juridiques, sourcee cellule par cellule.

POURQUOI CE SERVICE EXISTE, ET POURQUOI IL NE PASSE PAS PAR LE RAG.

Le pipeline de chat fait UNE recherche par question. Demande-lui de comparer
deux regimes et il lance une requete unique qui melange les deux termes. Mesure
sur le corpus, permis de recherche contre permis d'exploitation miniere :

    deux recherches separees  ->  14 articles distincts, 2 en commun
    une requete melangee      ->  8 articles, dont 6 du premier regime
                                  et 3 du second ; les articles 49 a 54,
                                  qui DEFINISSENT le permis d'exploitation,
                                  ne remontent pas du tout

Le modele recevait donc six articles sur un regime et trois sur l'autre, et
produisait quand meme deux colonnes — la seconde comblee par ses connaissances
generales. C'est le chemin d'hallucination le plus court du produit.

Deux mecanismes le ferment :

  1. UNE RECHERCHE PAR SUJET. C'est l'essentiel, et c'est tout ce que le RAG ne
     faisait pas.
  2. UN SCHEMA DE SORTIE ou la source est obligatoire. Le modele ne peut plus
     ecrire une cellule sans citer, il doit declarer l'absence. Mesure sur le
     meme essai : 12 citations exactes sur 13, trois absences correctement
     declarees, et une source surnumeraire — l'article 39 cite a tort a cote de
     l'article 42, qui lui etait juste.

Ce dernier cas est la limite du procede, et la raison d'etre de
`unmatched_citations` et du texte deplie sous chaque cellule : ce qu'on ne peut
pas empecher, on le rend VISIBLE.

Author: JuriX Team
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.comparison import (
    CRITERES_PAR_DEFAUT,
    ComparisonCell,
    ComparisonResponse,
    ComparisonRow,
    SourceRef,
    mention_absence,
)
from app.schemas.search import ChunkResult, SearchFilters, SearchRequest
from app.services.article_reference import normalize_number
from app.services.gemini_service import (
    GeminiOverloadedError,
    GeminiQuotaError,
    GeminiServiceError,
    get_gemini_service,
)
from app.services.prompts import (
    COMPARE_TASK_TEMPLATES,
    build_context_string,
    get_compare_system_prompt,
)
from app.services.search_service import SearchService

logger = logging.getLogger(__name__)

# Deterministe autant que possible : une comparaison de textes n'a pas a varier
# d'une execution a l'autre, et la seule chose que l'alea peut produire ici est
# une condition d'octroi qui n'existe pas.
COMPARISON_TEMPERATURE = 0.2
# Sept criteres x deux colonnes, plus les differences et les angles morts. A
# 4096 la grille se tronquait en JSON invalide sur les textes denses.
COMPARISON_MAX_TOKENS = 8192
# Facteur de sur-echantillonnage quand la comparaison est bornee a un document.
# Le plafond de `SearchRequest.limit` est 50, d'ou le min() a l'usage.
SUR_ECHANTILLON = 5


class ComparisonError(Exception):
    """Echec de comparaison. Traduit en 500."""


class NoContextError(ComparisonError):
    """Aucun texte trouve pour l'un des deux sujets. Traduit en 404."""


class ComparisonQuotaError(Exception):
    """Quota epuise. 429. Volontairement hors de ComparisonError, comme ailleurs."""


class ComparisonOverloadedError(Exception):
    """Generation saturee. 503."""


def _schema_de_sortie(criteres: List[str]) -> Dict[str, Any]:
    """
    Schema JSON impose au modele.

    `sources_a` et `sources_b` sont REQUIS. Un tableau vide est une declaration
    d'absence ; un champ absent serait une echappatoire.
    """
    return {
        "type": "object",
        "properties": {
            "lignes": {
                "type": "array",
                "minItems": len(criteres),
                "maxItems": len(criteres),
                "items": {
                    "type": "object",
                    "properties": {
                        # L'index est la SEULE cle fiable. Le libelle est
                        # reformule par le modele — « Sanctions penales » pour
                        # « Retrait, suspension ou annulation » — et l'ordre
                        # n'est garanti par rien.
                        "index": {"type": "integer"},
                        "critere": {"type": "string"},
                        "valeur_a": {"type": "string"},
                        "sources_a": {"type": "array", "items": {"type": "string"}},
                        "valeur_b": {"type": "string"},
                        "sources_b": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": [
                        "index", "critere", "valeur_a",
                        "sources_a", "valeur_b", "sources_b",
                    ],
                },
            },
            "differences_majeures": {"type": "array", "items": {"type": "string"}},
            "angles_morts": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["lignes", "differences_majeures", "angles_morts"],
    }


def _liste_de_chaines(valeur: Any) -> List[str]:
    """
    Ne garde que ce qui est reellement une liste de chaines.

    Le schema demande un tableau de chaines, le modele n'obeit pas toujours.
    `list("Les durees different.")` rendait une liste de CARACTERES, affichee
    en autant de puces ; et `[{"point": "..."}]` faisait echouer la validation
    Pydantic en fin de course, donc un 500 apres avoir paye la generation alors
    que la grille, elle, etait bonne.
    """
    if isinstance(valeur, str):
        return [valeur] if valeur.strip() else []
    if not isinstance(valeur, list):
        return []
    return [x.strip() for x in valeur if isinstance(x, str) and x.strip()]


def _source_depuis_chunk(chunk: ChunkResult, *, avec_texte: bool = True) -> SourceRef:
    return SourceRef(
        article_id=chunk.article_id,
        law_id=chunk.law_id,
        law_title=chunk.law_title,
        reference=chunk.reference,
        number=str(chunk.number) if chunk.number else "",
        article_title=chunk.article_title,
        page_number=chunk.page_number,
        # Les listes `articles_a`/`articles_b` ne servent qu'a compter : y
        # embarquer le texte integral de vingt articles par cote alourdissait
        # la reponse sans que rien ne l'affiche.
        content=(chunk.content or "") if avec_texte else "",
    )


class ComparisonService:
    """
    Compare deux sujets a partir du corpus.

    Le LLM est un PARAMETRE, pas un appel a get_gemini_service() : les tests
    injectent une doublure. Meme regle que reranker.py et explanation_service.
    """

    def __init__(self, db: AsyncSession, *, llm: Optional[Any] = None):
        self.db = db
        self.search_service = SearchService(db)
        self.llm = llm if llm is not None else get_gemini_service()

    async def compare(
        self,
        subject_a: str,
        subject_b: str,
        language: str = "fr",
        criteria: Optional[List[str]] = None,
        law_id: Optional[int] = None,
        top_k: int = 8,
    ) -> ComparisonResponse:
        """
        Raises:
            NoContextError: aucun texte pour l'un des deux sujets
            ComparisonQuotaError / ComparisonOverloadedError / ComparisonError
        """
        langue = language if language in COMPARE_TASK_TEMPLATES else "fr"
        axes = criteria or CRITERES_PAR_DEFAUT

        debut = time.perf_counter()
        chunks_a, chunks_b = await self._recuperer(subject_a, subject_b, law_id, top_k)
        duree_recherche = int((time.perf_counter() - debut) * 1000)

        if not chunks_a or not chunks_b:
            manquant = subject_a if not chunks_a else subject_b
            ou = f"dans le document {law_id}" if law_id is not None else "dans le corpus"
            raise NoContextError(f"Aucun texte trouve pour « {manquant} » {ou}.")

        logger.info(
            f"⚖️ Comparaison « {subject_a} » / « {subject_b} » : "
            f"{len(chunks_a)} + {len(chunks_b)} articles, {duree_recherche} ms"
        )

        debut = time.perf_counter()
        brut = await self._generer(subject_a, subject_b, chunks_a, chunks_b, axes, langue)
        duree_generation = int((time.perf_counter() - debut) * 1000)

        lignes, orphelines = self._assembler(
            brut, axes, self._indexer(chunks_a), self._indexer(chunks_b), langue
        )

        if orphelines:
            # Pas une erreur fatale : la cellule reste affichee, sans source, et
            # le numero fantome est remonte au client pour qu'il le voie.
            logger.warning(f"⚠️ Citations sans article correspondant : {orphelines}")

        return ComparisonResponse(
            subject_a=subject_a,
            subject_b=subject_b,
            language=langue,
            rows=lignes,
            key_differences=_liste_de_chaines(brut.get("differences_majeures")),
            blind_spots=_liste_de_chaines(brut.get("angles_morts")),
            articles_a=[_source_depuis_chunk(c, avec_texte=False) for c in chunks_a],
            articles_b=[_source_depuis_chunk(c, avec_texte=False) for c in chunks_b],
            unmatched_citations=orphelines,
            retrieval_time_ms=duree_recherche,
            generation_time_ms=duree_generation,
        )

    async def _recuperer(
        self, sujet_a: str, sujet_b: str, law_id: Optional[int], top_k: int
    ) -> Tuple[List[ChunkResult], List[ChunkResult]]:
        """
        DEUX recherches, l'une APRES l'autre.

        C'est le coeur du mode. Une requete unique melangeant les deux sujets
        rend surtout le dominant — mesure : 6 articles contre 3, et le bloc qui
        definit le second regime absent.

        SEQUENTIEL, ET SURTOUT PAS `asyncio.gather`. Les deux recherches
        partagent la meme AsyncSession, dont l'usage concurrent n'est pas
        supporte par SQLAlchemy. Mesure sur le corpus, avec gather :

            sequentiel  ->  A=8 chunks, B=8 chunks
            gather      ->  A=8 chunks, B=0 chunks, SANS AUCUNE EXCEPTION

        Le sujet B repartait vide et la comparaison rendait un 404 « aucun texte
        trouve » sur un sujet qui en regorge. Un cote seulement TRONQUE, lui,
        serait passe inapercu et aurait produit exactement la grille bancale que
        ce mode existe pour empecher.

        Le parallelisme faisait gagner une centaine de millisecondes devant un
        appel de generation de trente a soixante secondes. Il ne valait rien.
        """
        filtres = SearchFilters(status="published")
        # `SearchFilters` ne porte pas de `law_id` : le filtre ne peut pas
        # descendre dans le SQL. Applique apres coup sur `top_k` resultats, il
        # SUPPRIME sans remplacer — mesure : un decret d'application mieux
        # classe que le code occupait les trois places, le filtre les jetait
        # toutes, et l'utilisateur recevait « aucun texte trouve » alors que
        # l'article cherche existait dans le document demande. On sur-echantillonne
        # donc avant de filtrer, puis on retaille.
        limite = min(top_k * SUR_ECHANTILLON, 50) if law_id is not None else top_k

        async def une(terme: str) -> List[ChunkResult]:
            reponse = await self.search_service.search(
                SearchRequest(
                    query=terme, mode="hybrid", filters=filtres, limit=limite
                )
            )
            chunks = reponse.chunks
            if law_id is not None:
                chunks = [c for c in chunks if c.law_id == law_id][:top_k]
            return chunks

        return await une(sujet_a), await une(sujet_b)

    async def _generer(
        self,
        sujet_a: str,
        sujet_b: str,
        chunks_a: List[ChunkResult],
        chunks_b: List[ChunkResult],
        axes: List[str],
        langue: str,
    ) -> Dict[str, Any]:
        prompt = COMPARE_TASK_TEMPLATES[langue].format(
            a=sujet_a,
            b=sujet_b,
            ctx_a=build_context_string(chunks_a),
            ctx_b=build_context_string(chunks_b),
            # Numerotes : le champ `index` du schema renvoie a ce numero, et
            # c'est par lui que la reponse est rattachee a son critere.
            criteres="\n".join(f"{i}. {c}" for i, c in enumerate(axes)),
        )
        systeme = get_compare_system_prompt(langue, mention_absence(langue))

        try:
            reponse = await self.llm.generate(
                prompt=prompt,
                system=systeme,
                temperature=COMPARISON_TEMPERATURE,
                max_tokens=COMPARISON_MAX_TOKENS,
                response_mime_type="application/json",
                response_schema=_schema_de_sortie(axes),
            )
        except GeminiQuotaError as e:
            raise ComparisonQuotaError(str(e)) from e
        except GeminiOverloadedError as e:
            raise ComparisonOverloadedError(str(e)) from e
        except GeminiServiceError as e:
            raise ComparisonError(str(e)) from e

        texte = (reponse or {}).get("response", "")
        try:
            donnees = json.loads(texte)
        except (json.JSONDecodeError, TypeError) as e:
            # Arrive quand la sortie est tronquee au plafond de jetons. Le
            # message doit le dire : « JSON invalide » enverrait chercher un bug
            # la ou il n'y a qu'un budget trop court.
            raise ComparisonError(
                "La comparaison n'a pas pu etre lue (reponse incomplete du modele)."
            ) from e
        if not isinstance(donnees, dict):
            raise ComparisonError("Reponse du modele inattendue.")
        return donnees

    @staticmethod
    def _indexer(chunks: List[ChunkResult]) -> Dict[str, List[ChunkResult]]:
        """
        {numero normalise: articles portant ce numero}, pour resoudre les citations.

        `normalize_number` des deux cotes : le modele ecrit « Article 1er » la
        ou la base stocke « 1er », et « art. 40.1 » la ou elle stocke « 40.1 ».

        LA VALEUR EST UNE LISTE, ET CE N'EST PAS UNE PRECAUTION THEORIQUE. Un
        numero d'article n'identifie rien a lui seul : mesure sur le corpus, une
        seule recuperation de huit articles a rendu trois « article 1 »
        appartenant a trois lois differentes. Avec un index a valeur unique, une
        cellule citant « article 1 » affichait le texte de la premiere loi
        rencontree — une source FAUSSE, presentee comme verifiee. C'est pire que
        l'invention que ce mode est cense empecher, parce que ca ressemble a une
        verification.

        Quand un numero reste ambigu, on rend TOUS les candidats plutot que d'en
        choisir un : le lecteur voit alors deux textes sous la cellule et
        tranche lui-meme.
        """
        index: Dict[str, List[ChunkResult]] = {}
        for chunk in chunks:
            cle = normalize_number(str(chunk.number or ""))
            if not cle:
                continue
            index.setdefault(cle, []).append(chunk)
        return index

    def _assembler(
        self,
        brut: Dict[str, Any],
        axes: List[str],
        index_a: Dict[str, List[ChunkResult]],
        index_b: Dict[str, List[ChunkResult]],
        langue: str,
    ) -> Tuple[List[ComparisonRow], List[str]]:
        """
        Transforme la sortie du modele en grille, en resolvant les citations.

        Un numero cite qui ne correspond a aucun article recupere n'est PAS
        silencieusement jete : il part dans `unmatched_citations`, que le client
        affiche. Une citation inventee doit se voir.
        """
        orphelines: List[str] = []
        vues_orphelines: set = set()
        absence = mention_absence(langue)

        def cellule(
            valeur: Any, numeros: Any, index: Dict[str, List[ChunkResult]]
        ) -> ComparisonCell:
            """
            Resout les numeros cites CONTRE LE COTE AUQUEL LA CELLULE APPARTIENT.

            Chercher dans les articles des deux sujets melanges ferait resoudre
            « article 1 » du sujet B vers un article du sujet A, qui vient
            souvent d'un autre texte.
            """
            sources: List[SourceRef] = []
            vus: set = set()
            for numero in (numeros or []):
                cle = normalize_number(str(numero))
                candidats = index.get(cle) or []
                if not candidats:
                    # Deduplique sur la cle NORMALISEE : « Article 40 » et
                    # « art. 40 » designent le meme fantome et ne doivent pas
                    # apparaitre deux fois dans la banniere d'alerte.
                    if cle not in vues_orphelines:
                        vues_orphelines.add(cle)
                        orphelines.append(str(numero))
                    continue
                for chunk in candidats:
                    if chunk.article_id in vus:
                        continue
                    vus.add(chunk.article_id)
                    sources.append(_source_depuis_chunk(chunk))
            return ComparisonCell(value=str(valeur or absence), sources=sources)

        brutes = [x for x in (brut.get("lignes") or []) if isinstance(x, dict)]
        par_index: Dict[int, Any] = {}
        par_libelle: Dict[str, Any] = {}
        for ligne in brutes:
            valeur = ligne.get("index")
            if isinstance(valeur, int) and 0 <= valeur < len(axes):
                par_index.setdefault(valeur, ligne)
            libelle = str(ligne.get("critere", "")).strip().lower()
            if libelle:
                par_libelle.setdefault(libelle, ligne)

        lignes: List[ComparisonRow] = []
        for position, axe in enumerate(axes):
            # Index d'abord, libelle ensuite, ABSENCE a defaut.
            #
            # Il n'y a PLUS de repli par position, et c'est deliberé. L'ancien
            # code prenait la n-ieme reponse des que leur nombre correspondait,
            # sans verifier l'ordre. Reproduit : un modele qui reformule ses
            # libelles et repond en ordre inverse plaçait six reponses sur sept
            # sous le mauvais critere — avec des articles cites qui se
            # resolvaient normalement, un texte qui se depliait, et
            # `unmatched_citations` vide. Rien ne signalait l'erreur. Une case
            # vide est un defaut visible ; une case juste sous le mauvais
            # libelle ne l'est pas.
            ligne = par_index.get(position) or par_libelle.get(axe.strip().lower()) or {}
            lignes.append(
                ComparisonRow(
                    criterion=axe,
                    a=cellule(ligne.get("valeur_a"), ligne.get("sources_a"), index_a),
                    b=cellule(ligne.get("valeur_b"), ligne.get("sources_b"), index_b),
                )
            )
        return lignes, orphelines
