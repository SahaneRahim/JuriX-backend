"""
Genere un jeu d'evaluation (question -> article attendu) a partir du corpus.

Le principe est simple et le piege l'est tout autant : des questions ecrites par
Gemini a partir d'un article, notees par un systeme qui doit retrouver cet
article, forment une boucle fermee. Deux garde-fous l'ouvrent, et ils ne sont pas
optionnels :

1. FILTRE DE RECOUPEMENT LEXICAL. Sans lui, le modele produit une question qui
   recopie la phrase de l'article, la recherche plein texte marque 100 % et le
   jeu ne mesure rien. Le recoupement de Jaccard entre la question et l'article
   est calcule, et tout ce qui depasse le seuil est rejete.
2. RELECTURE HUMAINE. Le script ecrit un fichier de relecture ; les items ne
   sont utilisables qu'une fois marques `reviewed`.

Un troisieme controle SIGNALE sans supprimer : si l'article attendu est absent
du haut du classement actuel, l'item est marque a relire. Le supprimer
biaiserait le jeu vers ce que le systeme trouve deja — le mode d'echec classique
des evaluations auto-generees.

Usage:
    python -m scripts.eval.generate_eval_set --sample 120 --seed 7
    python -m scripts.eval.generate_eval_set --sample 20 --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from sqlalchemy import text

sys.path.insert(0, ".")

from app.core.database import SyncSessionLocal  # noqa: E402
from app.services.text_features import content_words, jaccard  # noqa: E402

logger = logging.getLogger("generate_eval_set")

DEFAULT_OUT = Path("tests/fixtures/eval/retrieval_eval_v1.json")

# Au-dela, la question recopie le vocabulaire de l'article et ne teste plus la
# recherche mais la correspondance de chaines.
MAX_LEXICAL_OVERLAP = 0.35

# Formules d'execution : presentes dans des milliers de decrets a l'identique,
# elles ne peuvent designer aucun document en particulier.
BOILERPLATE = "%enregistr%journal officiel%"

_SCHEMA = {
    "type": "object",
    "properties": {
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "keyword": {"type": "string"},
                    "paraphrase": {"type": "string"},
                },
                "required": ["keyword", "paraphrase"],
            },
        }
    },
    "required": ["questions"],
}

_PROMPT = """Voici un article de droit camerounais.

Loi : {law_title} ({reference})
Article {number}{title_part}
Contenu :
{content}

Écris DEUX questions que poserait une personne qui n'a PAS lu ce texte et qui cherche justement l'information qu'il contient :
1. "keyword" : formulation courte, comme une recherche.
2. "paraphrase" : question complète en langage courant.

Contraintes strictes :
- ne cite JAMAIS le numéro de l'article ni le titre exact de la loi ;
- reprends au maximum DEUX mots de contenu présents dans l'article ;
- interroge la SITUATION concrète, pas la formulation du texte ;
- français, une phrase par question."""


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=120,
                        help="articles candidats (viser ~2x la taille cible)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-overlap", type=float, default=MAX_LEXICAL_OVERLAP)
    parser.add_argument("--holdout", type=int, default=15,
                        help="items reserves au lot 'test', jamais utilises pour regler")
    parser.add_argument("--dry-run", action="store_true",
                        help="echantillonne et affiche, sans appeler l'API")
    return parser.parse_args(argv)


def _sample_articles(session, sample: int, seed: int) -> List[Dict[str, Any]]:
    """
    Echantillonnage stratifie.

    Un tirage uniforme donnerait un jeu compose a moitie de decrets de
    nomination : il mesurerait la recherche sur la partie la moins interessante
    du corpus.
    """
    session.execute(text("SELECT setseed(:seed)"), {"seed": (seed % 1000) / 1000})

    strata = [
        # 60 % : articles normatifs (loi consequente, article substantiel)
        ("normatif", int(sample * 0.6), """
            SELECT a.id, a.number, a.title, a.section, a.content,
                   l.id AS law_id, l.title AS law_title, l.reference
            FROM articles a JOIN laws l ON l.id = a.law_id
            WHERE length(a.content) > 400
              AND lower(a.content) NOT LIKE :boilerplate
              AND (SELECT count(*) FROM articles x WHERE x.law_id = l.id) >= 5
            ORDER BY random() LIMIT :n
        """),
        # 20 % : gros textes (constitution, codes, actes uniformes)
        ("gros_texte", int(sample * 0.2), """
            SELECT a.id, a.number, a.title, a.section, a.content,
                   l.id AS law_id, l.title AS law_title, l.reference
            FROM articles a JOIN laws l ON l.id = a.law_id
            WHERE length(a.content) > 200
              AND lower(a.content) NOT LIKE :boilerplate
              AND (SELECT count(*) FROM articles x WHERE x.law_id = l.id) > 100
            ORDER BY random() LIMIT :n
        """),
        # 20 % : tirage uniforme, pour garder la longue traine
        ("aleatoire", sample - int(sample * 0.6) - int(sample * 0.2), """
            SELECT a.id, a.number, a.title, a.section, a.content,
                   l.id AS law_id, l.title AS law_title, l.reference
            FROM articles a JOIN laws l ON l.id = a.law_id
            WHERE length(a.content) > 200
              AND lower(a.content) NOT LIKE :boilerplate
            ORDER BY random() LIMIT :n
        """),
    ]

    seen, rows = set(), []
    for label, count, sql in strata:
        if count <= 0:
            continue
        result = session.execute(text(sql), {"n": count, "boilerplate": BOILERPLATE})
        for row in result.mappings():
            if row["id"] in seen:
                continue
            seen.add(row["id"])
            rows.append({**dict(row), "stratum": label})
    return rows


def _corpus_snapshot(session) -> Dict[str, Any]:
    """Le jeu n'a de sens que contre un etat de corpus donne : on l'enregistre."""
    return {
        "laws": int(session.execute(text("SELECT count(*) FROM laws")).scalar() or 0),
        "articles": int(session.execute(text("SELECT count(*) FROM articles")).scalar() or 0),
        "alembic_head": str(
            session.execute(text("SELECT version_num FROM alembic_version")).scalar() or ""
        ),
    }


def _build_items(article: Dict[str, Any], questions: Dict[str, str],
                 max_overlap: float) -> List[Dict[str, Any]]:
    article_words = content_words(article["content"])
    items = []
    for difficulty, question in questions.items():
        overlap = jaccard(content_words(question), article_words)
        items.append({
            "question": question.strip(),
            "expected_article_id": int(article["id"]),
            "expected_law_id": int(article["law_id"]),
            "expected_article_number": str(article["number"] or ""),
            "stratum": article["stratum"],
            "difficulty": difficulty,
            "lexical_overlap": round(overlap, 4),
            "rejected": overlap > max_overlap,
            "reviewed": False,
            "reviewer_note": "",
        })
    return items


def _write_review_file(path: Path, items: List[Dict[str, Any]],
                       articles: Dict[int, Dict[str, Any]]) -> None:
    lines = [
        "# Relecture du jeu d'evaluation",
        "",
        "Marquez `reviewed: true` dans le JSON pour chaque question retenue,",
        "corrigez celles qui sont ambigues, supprimez les autres.",
        "Un item non relu n'est pas utilise par run_eval.py.",
        "",
        "| # | recoupement | question | article attendu |",
        "|---|---|---|---|",
    ]
    for index, item in enumerate(items):
        source = articles[item["expected_article_id"]]
        excerpt = " ".join(source["content"].split())[:140]
        flag = " ⚠️" if item["rejected"] else ""
        lines.append(
            f"| {index} | {item['lexical_overlap']:.2f}{flag} | {item['question']} | "
            f"{source['reference']} art. {source['number']} — {excerpt}… |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    with SyncSessionLocal() as session:
        snapshot = _corpus_snapshot(session)
        logger.info("Corpus : %s lois, %s articles, head %s",
                    snapshot["laws"], snapshot["articles"], snapshot["alembic_head"])
        if snapshot["articles"] == 0:
            logger.error("Corpus vide : rien a echantillonner")
            return 1

        articles = _sample_articles(session, args.sample, args.seed)

    logger.info("%s articles echantillonnes", len(articles))
    if args.dry_run:
        for article in articles[:10]:
            logger.info("  [%s] %s art. %s (%s car.)", article["stratum"],
                        article["reference"], article["number"], len(article["content"]))
        logger.info("--dry-run : aucun appel a l'API, aucune ecriture")
        return 0

    from app.services.gemini_service import get_gemini_service

    llm = get_gemini_service()
    if llm is None:
        logger.error("Service Gemini indisponible")
        return 1

    import asyncio

    by_id = {a["id"]: a for a in articles}
    items: List[Dict[str, Any]] = []

    async def _generate() -> None:
        for position, article in enumerate(articles, start=1):
            prompt = _PROMPT.format(
                law_title=article["law_title"],
                reference=article["reference"],
                number=article["number"],
                title_part=f" — {article['title']}" if article["title"] else "",
                content=article["content"][:4000],
            )
            try:
                response = await llm.generate(
                    prompt=prompt, temperature=0.7, max_tokens=512,
                    response_mime_type="application/json", response_schema=_SCHEMA,
                )
                payload = json.loads(response["response"])
                pair = payload["questions"][0]
            except Exception as exc:
                logger.warning("Article %s ignore : %s", article["id"], exc)
                continue

            items.extend(_build_items(
                article,
                {"keyword": pair.get("keyword", ""), "paraphrase": pair.get("paraphrase", "")},
                args.max_overlap,
            ))
            if position % 10 == 0:
                logger.info("%s/%s articles traites", position, len(articles))

    asyncio.run(_generate())

    kept = [item for item in items if not item["rejected"] and item["question"]]
    rejected = len(items) - len(kept)

    # Lot tenu a l'ecart, jamais utilise pour regler quoi que ce soit : si les
    # deux lots ne s'accordent pas, le reglage est du bruit.
    for index, item in enumerate(kept):
        item["split"] = "test" if index < args.holdout else "dev"
        item.pop("rejected", None)

    overlaps = [item["lexical_overlap"] for item in kept]
    logger.info(
        "%s questions retenues, %s rejetees pour recoupement > %.2f "
        "(recoupement median des retenues : %.2f)",
        len(kept), rejected, args.max_overlap,
        sorted(overlaps)[len(overlaps) // 2] if overlaps else 0.0,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "version": 1,
        "generator_model": getattr(llm, "model_name", "?"),
        "sampling_seed": args.seed,
        "max_lexical_overlap": args.max_overlap,
        "corpus_snapshot": snapshot,
        "items": kept,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    _write_review_file(args.out.with_suffix(".review.md"), kept, by_id)
    logger.info("Ecrit : %s et %s", args.out, args.out.with_suffix(".review.md"))
    logger.info("RELECTURE OBLIGATOIRE : marquez reviewed=true sur les items retenus.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
