"""
Script de génération de données synthétiques pour DocumentClassifier.

Génère 1200 exemples (100 par catégorie) pour entraîner le modèle ML.

Stratégie:
1. Templates de phrases juridiques réalistes par catégorie
2. Combinaisons aléatoires de keywords dans contextes variés
3. Variations syntaxiques (permutations, ordre)
4. Ajout de bruit (numéros articles, dates, références)

Output: data/training/legal_categories.json

Usage:
    python scripts/generate_training_data.py

Options:
    --samples-per-category: Nombre d'exemples par catégorie (défaut: 100)
    --output: Chemin du fichier de sortie (défaut: data/training/legal_categories.json)
    --seed: Seed pour reproductibilité (défaut: 42)
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any
import sys

# Ajouter le répertoire parent au path pour imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.document_classifier import DocumentClassifier


class TrainingDataGenerator:
    """Générateur de données d'entraînement synthétiques."""

    # Templates de phrases par catégorie
    # Chaque template a des placeholders {keyword1}, {keyword2}, etc.
    TEMPLATES = {
        1: [  # Droit Constitutionnel
            "La {kw1} établit les pouvoirs du {kw2}.",
            "Le {kw1} comprend {kw2} et {kw3}.",
            "Les institutions de la {kw1} sont fondées sur le {kw2}.",
            "Un {kw1} peut être organisé pour modifier la {kw2}.",
            "Le {kw1} exerce le {kw2} conformément à la {kw3}.",
            "L'{kw1} vote les lois et contrôle le {kw2}.",
            "La séparation du {kw1} et du {kw2} garantit la démocratie.",
            "Le {kw1} promulgue les lois votées par le {kw2}.",
            "Les {kw1} sont les piliers de la démocratie.",
            "Le {kw1} est organisé conformément à la {kw2}.",
        ],
        2: [  # Droit Civil
            "Le {kw1} régit les {kw2} entre parties.",
            "La {kw1} s'applique en cas de {kw2}.",
            "Les {kw1} doivent être respectées par le {kw2}.",
            "Le {kw1} peut poursuivre le {kw2} en justice.",
            "Le {kw1} définit les {kw2} contractuelles.",
            "Les délais de {kw1} sont fixés par la loi.",
            "Les {kw1} engagent la {kw2} des parties.",
            "Le {kw1} doit indemniser les {kw2} causés.",
            "Les {kw1} du {kw2} sont régies par le {kw3}.",
            "La {kw1} sanctionne les manquements aux {kw2}.",
        ],
        3: [  # Droit Pénal
            "Le {kw1} définit les {kw2} et {kw3} punissables.",
            "Toute {kw1} entraîne une {kw2}.",
            "Le {kw1} juge les {kw2} graves.",
            "L'{kw1} est puni conformément au {kw2}.",
            "Les {kw1} sont régies par le {kw2}.",
            "Les {kw1} et {kw2} sont passibles d'{kw3}.",
            "Le {kw1} sanctionne les auteurs de {kw2}.",
            "Les procédures {kw1} protègent les droits de la défense.",
            "La {kw1} est proportionnelle à la gravité du {kw2}.",
            "Le {kw1} établit la culpabilité dans les affaires {kw2}.",
        ],
        4: [  # Droit Commercial OHADA
            "L'{kw1} régit les {kw2} au Cameroun.",
            "La {kw1} doit avoir un {kw2} minimum.",
            "Les {kw1} sont réparties entre les {kw2}.",
            "Les {kw1} élisent les {kw2} de la {kw3}.",
            "L'{kw1} uniforme {kw2} s'applique aux {kw3}.",
            "Le {kw1} est divisé en {kw2}.",
            "Les {kw1} sont régies par l'{kw2}.",
            "Les {kw1} dirigent la {kw2} conformément à l'{kw3}.",
            "La {kw1} est une {kw2} régie par l'{kw3}.",
            "Les {kw1} ont des droits sur le {kw2}.",
        ],
        5: [  # Droit du Travail
            "Le {kw1} régit les relations entre {kw2} et {kw3}.",
            "Le {kw1} doit respecter la procédure légale.",
            "Le {kw1} définit les conditions d'emploi.",
            "Les {kw1} et le {kw2} sont garantis par la loi.",
            "L'{kw1} veille au respect des droits des {kw2}.",
            "Le {kw1} est fixé conformément au {kw2}.",
            "Les {kw1} doivent respecter le {kw2}.",
            "Le {kw1} protège les {kw2} contre les abus.",
            "Les conditions de {kw1} sont définies par le {kw2}.",
            "Le {kw1} garantit les droits des {kw2} et {kw3}.",
        ],
        6: [  # Droit Fiscal
            "La {kw1} comprend l'{kw2}, la {kw3} et l'{kw4}.",
            "Les {kw1} doivent déclarer leurs revenus.",
            "L'{kw1} assure le {kw2} des {kw3} et {kw4}.",
            "Le taux de {kw1} est fixé par la loi de finances.",
            "Les entreprises paient l'{kw1} sur leurs bénéfices.",
            "La {kw1} s'applique sur la valeur ajoutée.",
            "Les {kw1} sont soumis à l'{kw2}.",
            "L'{kw1} fiscale contrôle les déclarations des {kw2}.",
            "Le {kw1} des {kw2} est une obligation légale.",
            "Les {kw1} et {kw2} financent les services publics.",
        ],
        7: [  # Droit Administratif
            "La {kw1} emploie des {kw2}.",
            "Le {kw1} est assuré par l'État.",
            "Le {kw1} relève de la juridiction administrative.",
            "L'{kw1} exerce le pouvoir réglementaire.",
            "Les {kw1} sont régis par le statut de la {kw2}.",
            "L'{kw1} assure les {kw2} essentiels.",
            "Le {kw1} protège les droits des citoyens contre l'{kw2}.",
            "Les décisions de l'{kw1} peuvent faire l'objet de {kw2}.",
            "La {kw1} gère les {kw2}.",
            "Les {kw1} exercent leurs fonctions au sein de la {kw2}.",
        ],
        8: [  # Droit Foncier
            "Le {kw1} confère la {kw2} d'un {kw3}.",
            "L'{kw1} au {kw2} est obligatoire.",
            "Le {kw1} et les domaines privés sont régis par la loi {kw2}.",
            "Le {kw1} doit être autorisé par l'administration.",
            "La {kw1} nécessite un {kw2} valide.",
            "Le {kw1} enregistre les droits de {kw2}.",
            "Les {kw1} doivent être immatriculés au {kw2}.",
            "Le {kw1} protège les droits de {kw2}.",
            "L'{kw1} foncière garantit la sécurité des transactions sur les {kw2}.",
            "Les {kw1} sont soumis à l'{kw2}.",
        ],
        9: [  # Droit de la Famille
            "Le {kw1} et le {kw2} sont régis par le droit de la famille.",
            "La {kw1} est ouverte au décès.",
            "L'{kw1} appartient aux parents.",
            "La {kw1} établit le lien de parenté.",
            "Le {kw1} et les {kw2} ont des droits successoraux.",
            "La {kw1} protège les mineurs.",
            "Les règles du {kw1} sont fixées par la loi.",
            "Le {kw1} peut demander le {kw2}.",
            "Les {kw1} héritent conformément à la {kw2}.",
            "L'{kw1} règle les questions de {kw2} et {kw3}.",
        ],
        10: [  # Droit de l'Environnement
            "La {kw1} de l'{kw2} est un objectif national.",
            "La {kw1} de l'air et de l'eau est interdite.",
            "Les {kw1} doivent être préservées.",
            "Le {kw1} guide les politiques {kw2}.",
            "La {kw1} et la {kw2} sont protégées.",
            "Les activités industrielles ne doivent pas causer de {kw1}.",
            "La {kw1} des {kw2} est une priorité.",
            "Les politiques d'{kw1} visent la {kw2}.",
            "L'{kw1} doit être préservé pour les générations futures.",
            "La {kw1} sanctionne les atteintes à l'{kw2}.",
        ],
        11: [  # Droit International
            "Le {kw1} doit être {kw2} par le Parlement.",
            "Les {kw1} des {kw2} s'appliquent au Cameroun.",
            "Les {kw1} renforcent la coopération.",
            "La {kw1} nationale est respectée dans les relations {kw2}.",
            "Le {kw1} engage l'État sur le plan {kw2}.",
            "Les {kw1} régissent les relations entre États.",
            "La {kw1} d'un {kw2} nécessite l'approbation parlementaire.",
            "Les {kw1} internationaux lient les États parties.",
            "La {kw1} règle les différends entre États.",
            "Les {kw1} sont régis par le droit {kw2}.",
        ],
        12: [  # Droit des Affaires
            "Le {kw1} régit les relations d'affaires.",
            "La {kw1} et le {kw2} sont des contrats spéciaux.",
            "Le droit de la {kw1} protège le marché.",
            "Le {kw1} enregistre les commerçants et sociétés.",
            "Les {kw1} sont régis par le droit des affaires.",
            "Le {kw1} lie les parties commerciales.",
            "Les règles de {kw1} favorisent la libre entreprise.",
            "Le {kw1} protège les locaux commerciaux.",
            "Les opérations de {kw1} sont encadrées par la loi.",
            "Le {kw1} facilite l'identification des entreprises.",
        ],
    }

    def __init__(self, seed: int = 42):
        """
        Initialise le générateur.

        Args:
            seed: Seed pour reproductibilité
        """
        random.seed(seed)
        self.classifier = DocumentClassifier()
        self.keywords = self.classifier.CATEGORY_KEYWORDS

    def generate_sample(self, category_id: int) -> str:
        """
        Génère un échantillon de texte pour une catégorie.

        Args:
            category_id: ID de la catégorie (1-12)

        Returns:
            Texte juridique synthétique
        """
        # Récupérer keywords de la catégorie
        cat_keywords = self.keywords[category_id]
        all_keywords = cat_keywords['primary'] + cat_keywords['secondary']

        # Sélectionner template aléatoire
        template = random.choice(self.TEMPLATES[category_id])

        # Compter le nombre de placeholders
        num_placeholders = template.count('{kw')

        # Sélectionner keywords aléatoires
        selected_keywords = random.sample(all_keywords, min(num_placeholders, len(all_keywords)))

        # Remplir template
        text = template
        for i, keyword in enumerate(selected_keywords, 1):
            text = text.replace(f'{{kw{i}}}', keyword)

        # Ajouter préfixe article
        article_num = random.randint(1, 50)
        prefix = random.choice([
            f"Article {article_num}.",
            f"Art. {article_num}.",
            f"Section {article_num}.",
            ""
        ])

        # Combiner 2-4 phrases pour plus de contexte
        num_sentences = random.randint(2, 4)
        sentences = [text]

        for _ in range(num_sentences - 1):
            # Générer phrase supplémentaire
            template2 = random.choice(self.TEMPLATES[category_id])
            num_placeholders2 = template2.count('{kw')
            selected_keywords2 = random.sample(all_keywords, min(num_placeholders2, len(all_keywords)))

            text2 = template2
            for i, keyword in enumerate(selected_keywords2, 1):
                text2 = text2.replace(f'{{kw{i}}}', keyword)

            sentences.append(text2)

        # Joindre phrases
        full_text = f"{prefix} {' '.join(sentences)}".strip()

        return full_text

    def generate_dataset(
        self,
        samples_per_category: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Génère le dataset complet.

        Args:
            samples_per_category: Nombre d'exemples par catégorie

        Returns:
            Liste de dictionnaires {text, category_id, source}
        """
        dataset = []

        print(f"🔄 Génération de {samples_per_category} échantillons par catégorie...")

        for category_id in range(1, 13):  # 12 catégories
            category_name = self.classifier.get_category_name(category_id)
            print(f"  📝 Catégorie {category_id}: {category_name}")

            for i in range(samples_per_category):
                text = self.generate_sample(category_id)

                dataset.append({
                    "text": text,
                    "category_id": category_id,
                    "category_name": category_name,
                    "source": "synthetic",
                    "index": i
                })

            print(f"    ✅ {samples_per_category} échantillons générés")

        print(f"\n✅ Dataset généré: {len(dataset)} échantillons au total")

        return dataset

    def validate_dataset(self, dataset: List[Dict[str, Any]]):
        """
        Valide le dataset généré.

        Args:
            dataset: Dataset à valider

        Raises:
            ValueError: Si validation échoue
        """
        print("\n🔍 Validation du dataset...")

        # Vérifier distribution uniforme
        category_counts = {}
        for sample in dataset:
            cat_id = sample['category_id']
            category_counts[cat_id] = category_counts.get(cat_id, 0) + 1

        print(f"  📊 Distribution par catégorie:")
        for cat_id, count in sorted(category_counts.items()):
            print(f"    - Catégorie {cat_id}: {count} échantillons")

        # Vérifier longueurs
        lengths = [len(sample['text']) for sample in dataset]
        min_len = min(lengths)
        max_len = max(lengths)
        avg_len = sum(lengths) / len(lengths)

        print(f"\n  📏 Longueurs de texte:")
        print(f"    - Min: {min_len} caractères")
        print(f"    - Max: {max_len} caractères")
        print(f"    - Moyenne: {avg_len:.0f} caractères")

        if min_len < 50:
            raise ValueError(f"Textes trop courts détectés (min: {min_len})")

        if max_len > 1000:
            print(f"    ⚠️  Certains textes très longs (max: {max_len})")

        # Vérifier unicité (pas de doublons exacts)
        texts = [sample['text'] for sample in dataset]
        unique_texts = set(texts)
        if len(unique_texts) < len(texts):
            duplicates = len(texts) - len(unique_texts)
            print(f"    ⚠️  {duplicates} doublons détectés")

        print(f"\n✅ Validation terminée")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(
        description="Génère des données d'entraînement synthétiques pour DocumentClassifier"
    )
    parser.add_argument(
        '--samples-per-category',
        type=int,
        default=100,
        help="Nombre d'échantillons par catégorie (défaut: 100)"
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/training/legal_categories.json',
        help="Chemin du fichier de sortie (défaut: data/training/legal_categories.json)"
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help="Seed pour reproductibilité (défaut: 42)"
    )

    args = parser.parse_args()

    # Créer répertoire de sortie
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Générer dataset
    generator = TrainingDataGenerator(seed=args.seed)
    dataset = generator.generate_dataset(samples_per_category=args.samples_per_category)

    # Valider
    generator.validate_dataset(dataset)

    # Sauvegarder
    print(f"\n💾 Sauvegarde dans {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    file_size = output_path.stat().st_size / 1024  # KB
    print(f"✅ Fichier créé: {file_size:.1f} KB")

    print(f"\n🎯 Dataset prêt pour entraînement ML!")
    print(f"   - Total: {len(dataset)} échantillons")
    print(f"   - Catégories: 12")
    print(f"   - Fichier: {output_path}")


if __name__ == "__main__":
    main()
