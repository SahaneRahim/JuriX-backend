"""
Classement d'un texte juridique dans les domaines canoniques via Groq / Qwen.

Ce module classifie les documents juridiques camerounais en s'appuyant
exclusivement sur un LLM haute performance (Qwen via Groq API) avec sortie
JSON structurée stricte.

Aucune règle regex fragile : la qualification juridique est effectuée par
compréhension sémantique contextuelle complète de l'objet et du texte.

Author: JuriX Team
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

from app.services.text_features import fold_accents

logger = logging.getLogger(__name__)


# ==================== LES 14 DOMAINES CANONIQUES ====================

CANONICAL_DOMAINS: Tuple[str, ...] = (
    "Droit Constitutionnel",
    "Droit Administratif",
    "Fonction Publique",
    "Droit International",
    "Finances Publiques et Fiscalité",
    "Droit Pénal et Procédure Pénale",
    "Droit Civil et Procédure Civile",
    "Droit des Personnes, de la Famille et État Civil",
    "Droit du Travail et Sécurité Sociale",
    "Droit des Affaires, Banque et OHADA",
    "Droit Foncier et Domanial",
    "Droit de l'Environnement et des Ressources Naturelles",
    "Santé Publique et Sécurité Sanitaire",
    "Éducation, Recherche, Culture et Médias",
)

CONSTITUTIONNEL = CANONICAL_DOMAINS[0]
ADMINISTRATIF = CANONICAL_DOMAINS[1]
FONCTION_PUBLIQUE = CANONICAL_DOMAINS[2]
INTERNATIONAL = CANONICAL_DOMAINS[3]
FINANCES = CANONICAL_DOMAINS[4]
PENAL = CANONICAL_DOMAINS[5]
CIVIL = CANONICAL_DOMAINS[6]
FAMILLE = CANONICAL_DOMAINS[7]
TRAVAIL = CANONICAL_DOMAINS[8]
AFFAIRES = CANONICAL_DOMAINS[9]
FONCIER = CANONICAL_DOMAINS[10]
ENVIRONNEMENT = CANONICAL_DOMAINS[11]
SANTE = CANONICAL_DOMAINS[12]
EDUCATION = CANONICAL_DOMAINS[13]

# Alias de compatibilité pour d'anciens modules/tests
PROCEDURE_CIVILE = CIVIL
PROCEDURE_PENALE = PENAL


def _normalize_key(text: str) -> str:
    """Clé insensible aux accents et à la casse pour le mapping."""
    return fold_accents(text or "").strip().lower()


_CANONICAL_MAP: Dict[str, str] = {
    _normalize_key(domain): domain for domain in CANONICAL_DOMAINS
}

# Variantes historiques courantes
_CANONICAL_MAP.update({
    _normalize_key("Droit Civil"): CIVIL,
    _normalize_key("Procédure Civile"): CIVIL,
    _normalize_key("Droit Pénal"): PENAL,
    _normalize_key("Procédure Pénale"): PENAL,
    _normalize_key("Droit de la Famille"): FAMILLE,
    _normalize_key("Droit des Affaires et OHADA"): AFFAIRES,
    _normalize_key("Droit des Affaires"): AFFAIRES,
    _normalize_key("Droit Commercial"): AFFAIRES,
    _normalize_key("Droit Fiscal"): FINANCES,
})


@dataclass(frozen=True)
class DomainResult:
    """
    Verdict de classement.

    `domain` est TOUJOURS l'un des CANONICAL_DOMAINS.
    `rule` décrit la justification fournie par le modèle ou le mécanisme de repli.
    """

    domain: str
    confidence: float
    rule: str
    source: Literal["groq", "default"]
    runners_up: Tuple[Tuple[str, float], ...] = ()


class LegalDomainClassifier:
    """Classifieur juridique basé 100 % sur Groq (Qwen)."""

    def _resolve_canonical_name(self, raw_name: Optional[str]) -> Optional[str]:
        if not raw_name:
            return None
        return _CANONICAL_MAP.get(_normalize_key(raw_name))

    def _parse_groq_response(
        self, data: Optional[Dict[str, Any]], title: str
    ) -> DomainResult:
        if not data:
            logger.warning(
                "🧭 Réponse Groq vide pour : %r — affectation par défaut (Administratif)",
                (title or "")[:80],
            )
            return DomainResult(
                domain=ADMINISTRATIF,
                confidence=0.20,
                rule="fallback:groq_empty_response",
                source="default",
            )

        cat_candidate = self._resolve_canonical_name(data.get("categorie"))
        if not cat_candidate:
            logger.warning(
                "🧭 Catégorie inconnue retournée par Groq : %r — repli Administratif",
                data.get("categorie"),
            )
            return DomainResult(
                domain=ADMINISTRATIF,
                confidence=0.20,
                rule=f"fallback:unknown_category:{data.get('categorie')}",
                source="default",
            )

        confidence = float(data.get("confiance", 0.90))
        confidence = max(0.10, min(1.0, confidence))

        justification = data.get("justification", "")
        rule_desc = f"groq:{justification[:90]}" if justification else "groq:qwen"

        # Traitement des catégories secondaires (runners_up)
        secondaires_raw = data.get("categories_secondaires", [])
        runners_up_list = []
        if isinstance(secondaires_raw, list):
            for sec in secondaires_raw:
                resolved_sec = self._resolve_canonical_name(sec)
                if resolved_sec and resolved_sec != cat_candidate:
                    runners_up_list.append((resolved_sec, round(confidence * 0.8, 2)))

        return DomainResult(
            domain=cat_candidate,
            confidence=confidence,
            rule=rule_desc,
            source="groq",
            runners_up=tuple(runners_up_list[:2]),
        )

    def classify(
        self,
        title: str,
        content: str = "",
        doc_type: Optional[str] = None,
    ) -> DomainResult:
        """
        Classification synchrone d'un document via Groq / Qwen.
        Toujours un domaine canonique valide, jamais None.
        """
        clean_title = (title or "").strip()
        if not clean_title and not (content or "").strip():
            return DomainResult(
                domain=ADMINISTRATIF,
                confidence=0.10,
                rule="default:empty_document",
                source="default",
            )

        from app.services.groq_service import get_groq_service

        groq_svc = get_groq_service()
        data = groq_svc.classify_legal_domain_sync(
            title=clean_title,
            text=content or "",
            doc_type=doc_type,
        )
        return self._parse_groq_response(data, clean_title)

    async def classify_async(
        self,
        title: str,
        content: str = "",
        doc_type: Optional[str] = None,
    ) -> DomainResult:
        """
        Classification asynchrone d'un document via Groq / Qwen.
        """
        clean_title = (title or "").strip()
        if not clean_title and not (content or "").strip():
            return DomainResult(
                domain=ADMINISTRATIF,
                confidence=0.10,
                rule="default:empty_document",
                source="default",
            )

        from app.services.groq_service import get_groq_service

        groq_svc = get_groq_service()
        data = await groq_svc.classify_legal_domain_async(
            title=clean_title,
            text=content or "",
            doc_type=doc_type,
        )
        return self._parse_groq_response(data, clean_title)

    def explain(
        self, title: str, content: str = "", doc_type: Optional[str] = None
    ) -> str:
        """Ligne de synthèse explicative."""
        result = self.classify(title, content, doc_type)
        suffix = ""
        if result.runners_up:
            suffix = " | secondaires: " + ", ".join(
                f"{d} ({s:.2f})" for d, s in result.runners_up
            )
        return f"{result.domain} [{result.rule}] {result.confidence:.2f} ({result.source}){suffix}"

    def health_check(self) -> Dict[str, object]:
        from app.core.config import settings

        return {
            "service": "LegalDomainClassifier",
            "status": "healthy",
            "domains": len(CANONICAL_DOMAINS),
            "model": settings.GROQ_MODEL,
            "mode": "groq_qwen",
        }


_instance: Optional[LegalDomainClassifier] = None


def get_legal_domain_classifier() -> LegalDomainClassifier:
    """Instance singleton partagée."""
    global _instance
    if _instance is None:
        _instance = LegalDomainClassifier()
    return _instance
