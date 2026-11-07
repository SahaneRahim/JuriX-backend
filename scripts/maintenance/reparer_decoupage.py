"""
Repare en base ce que le decoupeur corrige et la relecture du Code civil changent.

DEUX REPARATIONS, sur la base que DATABASE_URL designe :

1. LES EN-TETES COLLES (text_chunker._separer_en_tetes_en_ligne). Pour chaque
   loi, le texte est decoupe deux fois, sans puis avec le correctif : seuls les
   morceaux qui different sont touches. Les autres gardent leur id et leur
   vecteur, et ne changent que de rang. Une loi dont les morceaux a retirer ne
   sont pas en base est laissee de cote, et le rapport le dit.

2. LES REPRODUCTIONS OHADA DU CODE CIVIL. Le Code civil annote reproduit, pages
   182 a 221, d'anciennes versions d'Actes uniformes (suretes, procedures
   collectives, voies d'execution), sous leurs numeros : « Article 107 » y est
   une ligne de l'AUPCAP, et le chat la citait comme le Code civil. A ces pages,
   le Code en est a son article 2011 : un numero plus petit ne peut pas etre le
   sien. Ces lignes restent lisibles et cherchables en plein texte (rien n'est
   supprime), mais sortent du vectoriel et deviennent « ANNEXE.107 », un
   pseudo-numero qui ne se cite pas comme un article.

Par defaut, RIEN N'EST ECRIT : le script dit ce qu'il ferait. --apply ecrit, en
UNE transaction : un echec laisse la base intacte. Les vecteurs sont demandes
AVANT la transaction, pour les seuls morceaux nouveaux. Sans --vecteurs, ces
morceaux entrent sans vecteur ; `regenerate_embeddings.py --all` les rattrape.
Rejoue, le script ne trouve plus rien a faire.

La table articles_vecteurs suit par son declencheur (insertion, vecteur remis a
NULL, cle etrangere a la suppression). Le script s'arrete si ce declencheur est
coupe, et compare les comptes a la fin, avant le COMMIT.

Une base distante exige --base-distante, en plus de --apply.

    python scripts/maintenance/reparer_decoupage.py [--law-id 2274] [--json f.json]
    python scripts/maintenance/reparer_decoupage.py --apply --vecteurs --law-id 2274
"""

import argparse
import json
import logging
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.engine import make_url

sys.path.insert(0, ".")

from app.core.database import SYNC_DATABASE_URL, SyncSessionLocal
from app.utils import text_chunker
from app.utils.chunk_refiner import DocumentContext, normalize_for_chunking, refine

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger("reparer_decoupage")
logger.setLevel(logging.INFO)

CODE_CIVIL = "CODE-CIVIL"
REPRODUCTION = "reproduction"
SECTION_REPRODUCTION = "Reproduction d'un Acte uniforme OHADA, ancienne version, hors Code civil"
PAGES_REPRODUITES = (182, 221)
# Le Code civil en est a son article 2011 a la page 182.
PLUS_PETIT_NUMERO_DU_CODE_ICI = 2000

Cle = Tuple[str, str]


@dataclass
class Plan:
    law_id: int
    reference: str
    a_retirer: List[int] = field(default_factory=list)
    a_ajouter: List[Dict[str, Any]] = field(default_factory=list)
    rangs: Dict[int, int] = field(default_factory=dict)
    ecarte: Optional[str] = None


def decouper(loi, avec_correctif: bool) -> List[Dict[str, Any]]:
    """Le decoupage de process_law._split_and_save_articles, sans ecriture."""
    separer = text_chunker._separer_en_tetes_en_ligne
    if not avec_correctif:
        text_chunker._separer_en_tetes_en_ligne = lambda texte: texte
    try:
        langue = loi.language or "fr"
        morceaux = text_chunker.extract_articles(
            normalize_for_chunking(loi.content), strict=False, min_article_length=1, language=langue,
        )
    finally:
        text_chunker._separer_en_tetes_en_ligne = separer
    contexte = DocumentContext(
        reference=loi.reference or "", title=loi.title or "", doc_type=loi.type,
        date=loi.publication_date.isoformat() if loi.publication_date else None,
        category=loi.categorie, language=langue,
    )
    return refine(morceaux, contexte).chunks


def _cle(morceau: Dict[str, Any]) -> Cle:
    return str(morceau.get("number", "")), morceau.get("content", "")


def _cle_en_base(ligne) -> Cle:
    """Une reproduction deja renommee se compare sous son numero d'origine."""
    numero = ligne.number
    if ligne.kind == REPRODUCTION:
        numero = numero.removeprefix("ANNEXE.")
    return numero, ligne.content


def planifier(loi, lignes) -> Optional[Plan]:
    """Ce qui change pour cette loi ; None si le correctif n'y change rien."""
    ancien = decouper(loi, False)
    nouveau = decouper(loi, True)
    avant, apres = Counter(map(_cle, ancien)), Counter(map(_cle, nouveau))
    retires, ajoutes = avant - apres, apres - avant
    if not retires and not ajoutes:
        return None

    plan = Plan(loi.id, loi.reference)
    en_base: Dict[Cle, List[Any]] = {}
    for ligne in lignes:
        en_base.setdefault(_cle_en_base(ligne), []).append(ligne)

    deja = sum(min(n, len(en_base.get(c, []))) for c, n in ajoutes.items())
    presents = sum(min(n, len(en_base.get(c, []))) for c, n in retires.items())
    if deja == sum(ajoutes.values()) and presents == 0:
        restants = None  # deja repare : seuls les rangs sont a verifier
    elif presents != sum(retires.values()) or deja:
        plan.ecarte = (
            f"la base differe du decoupage attendu : {presents}/{sum(retires.values())} morceaux "
            f"a retirer presents, {deja}/{sum(ajoutes.values())} a ajouter deja la"
        )
        return plan
    else:
        restants = retires

    for cle, n in (restants or {}).items():
        plan.a_retirer += [en_base[cle].pop().id for _ in range(n)]

    # Les morceaux gardes prennent leur rang dans le nouveau decoupage.
    a_ajouter = Counter() if restants is None else Counter(ajoutes)
    for morceau in nouveau:
        cle = _cle(morceau)
        if a_ajouter[cle] > 0:
            a_ajouter[cle] -= 1
            plan.a_ajouter.append(morceau)
        elif en_base.get(cle):
            ligne = en_base[cle].pop()
            if ligne.order != morceau["position"]:
                plan.rangs[ligne.id] = morceau["position"]
    return plan if plan.a_retirer or plan.a_ajouter or plan.rangs else None


def _ecrire_plan(session, plan: Plan, vecteurs: Dict[str, List[float]], empreinte: Optional[str]) -> None:
    if plan.a_retirer:
        session.execute(text("DELETE FROM articles WHERE id = ANY(:ids)"), {"ids": plan.a_retirer})
    for ident, rang in plan.rangs.items():
        session.execute(text('UPDATE articles SET "order" = :r WHERE id = :i'), {"r": rang, "i": ident})
    for m in plan.a_ajouter:
        vecteur = vecteurs.get(m.get("embed_text") or "") if m.get("embed") else None
        session.execute(
            text(
                'INSERT INTO articles (law_id, number, title, section, content, "order", page_number,'
                " kind, embed, embed_text, embedding, embedding_model) VALUES (:law, :numero, :titre,"
                " :section, :contenu, :rang, :page, :nature, :embed, :embed_text,"
                " CAST(:vecteur AS halfvec), :modele)"
            ),
            {
                "law": plan.law_id, "numero": str(m.get("number", "")),
                "titre": (m.get("title") or None) and m["title"][:200], "section": m.get("section"),
                "contenu": m["content"], "rang": m["position"], "page": m.get("page_number"),
                "nature": m.get("kind"), "embed": bool(m.get("embed")),
                "embed_text": m.get("embed_text"),
                "vecteur": None if vecteur is None else str(list(vecteur)),
                "modele": empreinte if vecteur is not None else None,
            },
        )


def reproductions(session) -> List[int]:
    """Les lignes du Code civil qui reproduisent un Acte uniforme, pas encore marquees."""
    lignes = session.execute(
        text(
            "SELECT a.id, a.number FROM articles a JOIN laws l ON l.id = a.law_id"
            " WHERE l.reference = :ref AND a.page_number BETWEEN :p1 AND :p2"
            " AND a.kind IS DISTINCT FROM :nature"
        ),
        {"ref": CODE_CIVIL, "p1": PAGES_REPRODUITES[0], "p2": PAGES_REPRODUITES[1], "nature": REPRODUCTION},
    ).fetchall()
    gardees = []
    for ligne in lignes:
        chiffres = re.match(r"\d+(?:\.\d+)*$", ligne.number)
        if chiffres and int(ligne.number.split(".")[0]) < PLUS_PETIT_NUMERO_DU_CODE_ICI:
            gardees.append(ligne.id)
    return gardees


def _marquer_reproductions(session, ids: List[int]) -> None:
    session.execute(
        text(
            "UPDATE articles SET number = 'ANNEXE.' || number, kind = :nature, embed = false,"
            " embed_text = NULL, embedding = NULL, embedding_model = NULL, section = :section"
            " WHERE id = ANY(:ids)"
        ),
        {"nature": REPRODUCTION, "section": SECTION_REPRODUCTION, "ids": ids},
    )


def _verifier_le_declencheur(session) -> None:
    etat = session.execute(
        text(
            "SELECT tgenabled FROM pg_trigger WHERE tgname = 'articles_vecteurs_suivre'"
            " AND tgrelid = 'articles'::regclass"
        )
    ).scalar()
    if etat != "O":
        raise SystemExit(f"declencheur articles_vecteurs_suivre absent ou coupe ({etat!r}) : arret")


def _comptes(session) -> Tuple[int, int]:
    return (
        session.execute(text("SELECT count(*) FROM articles WHERE embedding IS NOT NULL")).scalar_one(),
        session.execute(text("SELECT count(*) FROM articles_vecteurs")).scalar_one(),
    )


def _vectoriser(textes: List[str]) -> Tuple[Dict[str, List[float]], str]:
    from app.services.embedding_service import EmbeddingService

    service = EmbeddingService(use_cache=False)
    limite = service.MAX_TEXT_LENGTH
    vecteurs = service.generate_batch_embeddings(texts=[t[:limite] for t in textes], normalize=True)
    return {t: v.tolist() for t, v in zip(textes, vecteurs)}, service.provider.fingerprint


def _lire_les_lois(session, ids: Optional[List[int]]):
    condition = "WHERE l.id = ANY(:ids)" if ids else ""
    return session.execute(
        text(
            "SELECT l.id, l.reference, l.title, l.type, l.language, l.publication_date, l.content,"
            f" c.name AS categorie FROM laws l LEFT JOIN categories c ON c.id = l.category_id {condition}"
            " ORDER BY l.id"
        ),
        {"ids": ids} if ids else {},
    ).fetchall()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="ecrire (defaut : rien n'est ecrit)")
    parser.add_argument("--vecteurs", action="store_true", help="vectoriser les morceaux nouveaux")
    parser.add_argument("--law-id", type=int, action="append", dest="law_ids")
    parser.add_argument("--sans-reproductions", action="store_true",
                        help="ne pas toucher aux reproductions du Code civil")
    parser.add_argument("--max-textes", type=int, default=None,
                        help="refuse de vectoriser plus de textes que cela")
    parser.add_argument("--base-distante", action="store_true",
                        help="autorise --apply sur une base qui n'est pas locale")
    parser.add_argument("--json", help="ecrit le plan dans ce fichier")
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    hote = make_url(SYNC_DATABASE_URL).host
    if args.apply and hote not in ("localhost", "127.0.0.1") and not args.base_distante:
        logger.error("base distante (%s) : --base-distante est exige avec --apply", hote)
        return 2

    with SyncSessionLocal() as session:
        plans: List[Plan] = []
        for loi in _lire_les_lois(session, args.law_ids):
            lignes = session.execute(
                text('SELECT id, number, content, kind, "order" FROM articles WHERE law_id = :i'),
                {"i": loi.id},
            ).fetchall()
            try:
                plan = planifier(loi, lignes) if loi.content else None
            except Exception as erreur:  # un texte indecoupable n'arrete pas les autres
                plan = Plan(loi.id, loi.reference, ecarte=f"decoupage impossible : {erreur}")
            if plan:
                plans.append(plan)
        session.rollback()

        actifs = [p for p in plans if not p.ecarte]
        for plan in plans:
            if plan.ecarte:
                logger.warning("  %s (#%s) ECARTEE : %s", plan.reference, plan.law_id, plan.ecarte)
            else:
                logger.info(
                    "  %-20s #%-5s -%-4d +%-4d rangs %d", plan.reference, plan.law_id,
                    len(plan.a_retirer), len(plan.a_ajouter), len(plan.rangs),
                )
        textes = sorted({
            m["embed_text"] for p in actifs for m in p.a_ajouter if m.get("embed") and m.get("embed_text")
        })
        logger.info(
            "%d lois, %d morceaux retires, %d ajoutes, %d textes a vectoriser (%d caracteres)",
            len(actifs), sum(len(p.a_retirer) for p in actifs), sum(len(p.a_ajouter) for p in actifs),
            len(textes), sum(map(len, textes)),
        )
        if args.json:
            with open(args.json, "w", encoding="utf-8") as sortie:
                json.dump([{
                    "law_id": p.law_id, "reference": p.reference, "ecarte": p.ecarte,
                    "retires": len(p.a_retirer), "rangs": len(p.rangs),
                    "ajoutes": [str(m.get("number")) for m in p.a_ajouter],
                } for p in plans], sortie, ensure_ascii=False)

        if not args.apply:
            if not args.sans_reproductions:
                logger.info("Reproductions du Code civil a marquer (avant decoupage) : %d",
                            len(reproductions(session)))
            logger.info("(dry-run : rien n'a ete ecrit. Relancez avec --apply.)")
            session.rollback()
            return 0

        vecteurs: Dict[str, List[float]] = {}
        empreinte = None
        if args.vecteurs and textes:
            if args.max_textes is not None and len(textes) > args.max_textes:
                logger.error("%d textes a vectoriser, plafond %d : arret", len(textes), args.max_textes)
                return 2
            vecteurs, empreinte = _vectoriser(textes)

        _verifier_le_declencheur(session)
        for plan in actifs:
            _ecrire_plan(session, plan, vecteurs, empreinte)
        marquees: List[int] = []
        if not args.sans_reproductions:
            session.flush()
            marquees = reproductions(session)
            if marquees:
                _marquer_reproductions(session, marquees)
        vectorises, table = _comptes(session)
        if vectorises != table:
            session.rollback()
            logger.error("articles vectorises %d, articles_vecteurs %d : annule", vectorises, table)
            return 1
        session.commit()
        logger.info("Ecrit : %d lois, %d reproductions ; %d vecteurs, table a egalite",
                    len(actifs), len(marquees), table)
    return 0


if __name__ == "__main__":
    sys.exit(main())
