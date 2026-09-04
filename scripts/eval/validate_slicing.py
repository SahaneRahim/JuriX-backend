"""
Verifie que tronquer un vecteur 3072 puis le renormaliser donne bien le vecteur
que l'API produit a la dimension inferieure.

C'est l'hypothese qui permet de comparer 3072, 1536 et 768 sans re-embedder le
corpus a chaque dimension : gemini-embedding-001 est entraine en Matryoshka, et
output_dimensionality est cense n'etre qu'une troncature. Si l'hypothese est
fausse, le balayage coute trois passes completes sur le corpus au lieu d'une.

40 appels d'API : 20 textes, a 3072 puis a la dimension cible.

Usage:
    python -m scripts.eval.validate_slicing --dim 768 --samples 20
"""

import argparse
import asyncio
import logging
import sys
from typing import Optional, Sequence

import numpy as np

sys.path.insert(0, ".")

from app.services.embedding_service import EmbeddingService  # noqa: E402

logger = logging.getLogger("validate_slicing")

TEXTS = [
    "Les dirigeants sociaux sont responsables des fautes de gestion.",
    "La présente loi fixe le régime des sociétés commerciales au Cameroun.",
    "Le Président de la République est le Chef de l'État.",
    "Toute personne a droit à la reconnaissance de sa personnalité juridique.",
    "Les contrats légalement formés tiennent lieu de loi à ceux qui les ont faits.",
    "L'action en responsabilité se prescrit par trois ans.",
    "Est puni d'un emprisonnement de un à cinq ans quiconque détourne des fonds publics.",
    "The Commercial Code governs the incorporation of companies.",
    "Le capital social minimum est fixé par décret.",
    "La procédure de redressement judiciaire est ouverte au débiteur en cessation de paiement.",
    "Les fonctionnaires sont soumis au statut général de la fonction publique.",
    "Le mariage est célébré publiquement devant l'officier d'état civil.",
    "Toute décision de justice doit être motivée.",
    "Le contribuable dispose d'un délai de trente jours pour contester.",
    "L'expropriation pour cause d'utilité publique donne lieu à indemnisation.",
    "Les délais de recours courent à compter de la notification.",
    "La société anonyme est constituée entre deux associés ou plus.",
    "L'employeur est tenu d'assurer la sécurité de ses salariés.",
    "Les biens sont meubles ou immeubles.",
    "La nationalité camerounaise s'acquiert par filiation ou par naturalisation.",
]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--threshold", type=float, default=0.999)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")

    texts = TEXTS[: args.samples]
    native = EmbeddingService(use_cache=False)
    reduced = EmbeddingService(use_cache=False)
    reduced.EMBEDDING_DIM = args.dim

    async def _run() -> int:
        similarities = []
        for text_sample in texts:
            full = await native.generate_embedding_async(
                text_sample, task_type=EmbeddingService.TASK_DOCUMENT
            )
            api_small = await reduced.generate_embedding_async(
                text_sample, task_type=EmbeddingService.TASK_DOCUMENT
            )
            sliced = full[: args.dim]
            sliced = sliced / np.linalg.norm(sliced)
            similarities.append(float(np.dot(sliced, api_small)))

        worst = min(similarities)
        logger.info("Cosinus min %.6f, median %.6f sur %s textes",
                    worst, float(np.median(similarities)), len(similarities))

        if worst < args.threshold:
            logger.error(
                "Le decoupage local NE reproduit PAS la sortie de l'API a %s dimensions.\n"
                "  Comparer les dimensions exige alors de re-embedder le corpus a chacune.",
                args.dim,
            )
            return 1

        logger.info(
            "Decoupage valide : les dimensions inferieures s'obtiennent localement, "
            "sans appel supplementaire."
        )
        return 0

    return asyncio.run(_run())


if __name__ == "__main__":
    sys.exit(main())
