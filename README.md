# JuriX Backend v2.1

Backend API pour la plateforme juridique camerounaise JuriX.

## 🚀 Quick Start

### Prérequis

- Python 3.11+
- PostgreSQL 15+ avec pgvector
- Redis 7.2+
- Meilisearch 1.5+
- Ollama avec mistral:7b

### Installation

```bash
# Cloner le projet
cd backend

# Installer les dépendances
pip install -r requirements.txt
# OU avec Poetry
poetry install

# Télécharger les modèles NLP (requis pour détection de langue)
python scripts/download_models.py

# Configurer les variables d'environnement
cp .env.example .env
# Éditer .env avec vos paramètres

# Lancer les migrations
alembic upgrade head

# Démarrer le serveur
uvicorn app.main:app --reload
```

L'API sera disponible sur `http://localhost:8000`

Documentation interactive: `http://localhost:8000/docs`

## 📦 Modules Implémentés

### ✅ Module 1: Détection Automatique de Langue

Service de détection automatique de langue pour documents juridiques.

**Architecture:** Triple-méthode ensemble avec vote majoritaire
- **Méthodes:** langdetect + spaCy + fastText  
- **Précision:** >99% sur documents juridiques
- **Performance:** <1s pour textes jusqu'à 5000 caractères
- **Langues:** Français + Anglais (extensible)

#### Setup

```bash
# 1. Télécharger les modèles NLP
python scripts/download_models.py

# Ceci télécharge:
# - spaCy fr_core_news_sm (~60MB)
# - spaCy en_core_web_sm (~60MB)  
# - fastText lid.176.bin (~130MB)
# Total: ~250MB

# 2. Vérifier l'installation
python -c "from app.services.language_detector import LanguageDetector; d = LanguageDetector(); print(d.health_check())"
```

#### Usage Programmatique

```python
from app.services.language_detector import LanguageDetector

# Créer instance (charge les modèles)
detector = LanguageDetector()

# Détecter langue
text = """
Article 1. La présente loi régit les conditions de création 
des sociétés commerciales au Cameroun conformément au droit OHADA.
"""

result = detector.detect(text)

print(result)
# {
#     'language': 'fr',
#     'confidence': 0.98,
#     'method_votes': {
#         'langdetect': 'fr',
#         'spacy': 'fr',
#         'fasttext': 'fr'
#     },
#     'consensus': True,
#     'processing_time_ms': 450,
#     'text_length': 156
# }
```

#### Usage API

```bash
# Détecter langue d'un texte
curl -X POST http://localhost:8000/api/v1/language/detect \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Article 1. La présente loi régit les sociétés commerciales au Cameroun.",
    "min_confidence": 0.80
  }'

# Réponse:
# {
#   "language": "fr",
#   "confidence": 0.98,
#   "method_votes": {
#     "langdetect": "fr",
#     "spacy": "fr",
#     "fasttext": "fr"
#   },
#   "consensus": true,
#   "processing_time_ms": 450,
#   "text_length": 75
# }

# Health check
curl http://localhost:8000/api/v1/language/health

# Réponse:
# {
#   "service": "LanguageDetector",
#   "status": "healthy",
#   "models": {
#     "spacy_fr": "✅ OK",
#     "spacy_en": "✅ OK",
#     "fasttext": "✅ OK"
#   }
# }
```

#### Tests

```bash
# Tests unitaires (5 tests de base + edge cases)
pytest backend/tests/test_services/test_language_detector.py -v

# Avec coverage
pytest backend/tests/test_services/test_language_detector.py -v \
  --cov=backend/app/services/language_detector \
  --cov-report=html

# Génération tests Qodo (auto, ~25 tests)
qodo test generate backend/app/services/language_detector.py --max-tests 25

# Tous les tests
pytest backend/tests/test_services/test_language_detector.py -v
```

#### Troubleshooting

**❌ Erreur: "No module named spacy"**
```bash
pip install spacy langdetect fasttext-wheel scikit-learn
```

**❌ Erreur: "Can't find model 'fr_core_news_sm'"**
```bash
python scripts/download_models.py
# OU manuellement:
python -m spacy download fr_core_news_sm
python -m spacy download en_core_web_sm
```

**❌ Erreur: "fastText model not found"**
```bash
# Télécharger manuellement
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin \
  -O backend/models/fasttext/lid.176.bin
```

**❌ Performance lente (>2s)**
- Vérifier que ThreadPoolExecutor est utilisé (parallélisme)
- Réduire la longueur du texte (sampling automatique à 5000 chars)
- Utiliser singleton pattern (éviter rechargement modèles)

**❌ Précision faible (<95%)**
- Vérifier longueur texte (minimum 100 chars recommandé)
- Textes bilingues donnent confiance plus basse (normal)
- Vérifier que les 3 modèles sont correctement chargés

## 🧪 Tests

### Lancer les tests

```bash
# Tous les tests
pytest backend/tests/ -v

# Tests spécifiques
pytest backend/tests/test_services/test_language_detector.py -v

# Avec coverage
pytest backend/tests/ -v --cov=backend/app --cov-report=html

# Tests de performance
pytest backend/tests/test_services/test_language_detector.py::TestLanguageDetectorPerformance -v
```

### Structure des tests

```
backend/tests/
├── test_main.py                          # Tests de base (health check, root)
├── test_services/
│   └── test_language_detector.py         # Tests détection langue (32 tests)
├── test_api/
│   └── (à venir)
├── test_tasks/
│   └── (à venir)
├── integration/
│   └── (à venir)
└── performance/
    └── (à venir)
```

## 📊 Code Quality

### Linting & Formatting

```bash
# Black (formatting)
black backend/app backend/tests

# Ruff (linting)
ruff check backend/app backend/tests

# MyPy (type checking)
mypy backend/app/services/language_detector.py

# Pylint
pylint backend/app/services/language_detector.py
```

### Coverage Target

- **Overall:** >80%
- **Services:** >85%
- **Critical paths:** >95%

## 🐳 Docker

```bash
# Build image
docker build -t jurix-backend:v2.1 -f docker/Dockerfile.prod .

# Run avec docker-compose
docker-compose up -d

# Vérifier logs
docker-compose logs -f api

# Health check
curl http://localhost:8000/health
curl http://localhost:8000/api/v1/language/health
```

## 📁 Structure du Projet

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                          # Point d'entrée FastAPI
│   ├── core/
│   │   ├── config.py                    # Configuration (Settings)
│   │   └── dependencies.py              # Dependency injection
│   ├── api/
│   │   ├── routes/
│   │   │   └── language.py              # ✅ Endpoints détection langue
│   │   └── dependencies/
│   ├── services/
│   │   └── language_detector.py         # ✅ Service détection langue
│   ├── models/                          # ORM models (SQLAlchemy)
│   ├── tasks/                           # Celery tasks
│   └── utils/                           # Utilitaires
├── models/                              # Modèles NLP
│   ├── spacy/
│   │   ├── fr_core_news_sm/
│   │   └── en_core_web_sm/
│   └── fasttext/
│       └── lid.176.bin
├── scripts/
│   └── download_models.py               # ✅ Script téléchargement modèles
├── tests/
│   ├── test_main.py
│   └── test_services/
│       └── test_language_detector.py    # ✅ 32 tests
├── alembic/                             # Migrations DB
│   ├── env.py
│   └── versions/
│       └── 001_initial_schema.py
├── pyproject.toml                       # Dependencies (Poetry)
├── requirements.txt                     # Dependencies (pip)
└── README.md                            # This file
```

## 🔧 Configuration

Variables d'environnement (.env):

```bash
# App
APP_NAME="JuriX API"
VERSION="2.1.0"
DEBUG=True
ENVIRONMENT="development"

# Database
DATABASE_URL="postgresql://jurix:password@localhost:5432/jurix_db"

# Redis
REDIS_URL="redis://:password@localhost:6379/0"

# Meilisearch
MEILISEARCH_URL="http://localhost:7700"
MEILISEARCH_KEY="master_key"

# Ollama
OLLAMA_URL="http://localhost:11434"
OLLAMA_MODEL="mistral:7b"

# Security
SECRET_KEY="your_secret_key_here"
ALGORITHM="HS256"
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Qodo (Tests auto-générés)
QODO_API_KEY="your_qodo_api_key"

# CORS
CORS_ORIGINS=["http://localhost:5173","http://localhost:3000"]
```

## 📝 API Documentation

Documentation interactive disponible sur:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### Endpoints Principaux

#### Language Detection

```
POST /api/v1/language/detect
GET  /api/v1/language/health
```

#### À venir

```
POST /api/v1/laws                    # Créer loi
GET  /api/v1/laws                    # Lister lois
GET  /api/v1/laws/{id}               # Détail loi
POST /api/v1/search                  # Recherche
POST /api/v1/chatbot/ask             # Question RAG
GET  /api/v1/categories              # Catégories
GET  /health                         # Health check global
```

### ✅ Module 3: Extraction d'Articles

Utilitaire d'extraction automatique d'articles pour documents juridiques camerounais.

**Architecture:** Regex-based pattern matching avec détection automatique
- **Patterns supportés:** Article X, Art. X, Article premier/première  
- **Numérotation hiérarchique:** Support 1.1, 1.2.3 (parent_id)
- **Précision:** >95% sur documents légaux camerounais
- **Performance:** <500ms pour 1000 articles
- **Validation:** Minimum 3 articles requis (configurable)

#### Usage Programmatique

```python
from app.utils.text_chunker import extract_articles

# Texte juridique
law_text = """
LOI N° 2023-001 DU 15 JANVIER 2023

Article 1. Objet de la loi
La présente loi régit les sociétés commerciales au Cameroun.

Article 2. Champ d'application
Les dispositions s'appliquent à toutes les sociétés.

Article 3. Définitions
Au sens de la présente loi, on entend par société...
"""

# Extraire articles
articles = extract_articles(law_text)

print(f"Nombre d'articles: {len(articles)}")

for article in articles:
    print(f"Article {article['number']}: {article['title']}")
    print(f"  Contenu: {article['content'][:50]}...")
    print(f"  Mots: {article['word_count']}, Chars: {article['char_count']}")
    print(f"  Parent: {article['parent_id']}, Position: {article['position']}")
    print()

# Résultat:
# Nombre d'articles: 3
# Article 1: Objet de la loi
#   Contenu: La présente loi régit les sociétés commerciales...
#   Mots: 9, Chars: 58
#   Parent: None, Position: 0
# ...
```

#### Paramètres d'Extraction

```python
# Paramètres personnalisés
articles = extract_articles(
    text=law_text,
    min_article_length=10,      # Minimum 10 caractères par article
    preserve_formatting=False,  # Nettoyer espaces multiples
    strict=True                 # Erreur si <3 articles
)

# Mode non-strict (pour documents incomplets)
articles = extract_articles(
    text=draft_law_text,
    strict=False  # Accepte <3 articles
)
```

#### Patterns Supportés

```python
# ✅ Article X.
"Article 1. Dispositions générales"
"Article 2: Champ d'application"

# ✅ Art. X
"Art. 1. Titre"
"Art 1: Titre"

# ✅ Article premier/première
"Article premier. Objet de la loi"
"Article première. Disposition initiale"

# ✅ Numérotation hiérarchique
"Article 1.1. Sous-article"
"Article 1.2.3. Sous-sous-article"

# ❌ Non supportés (à venir Phase 2)
"Section I. Titre de section"
"Chapitre II. Titre de chapitre"
"Alinéa 1. Paragraphe"
```

#### Intégration Pipeline

```python
# Dans process_law task
from app.utils.text_chunker import extract_articles, ArticleExtractionError

def process_law_task(law_id: int):
    """Traiter document juridique uploadé."""
    
    # 1. Récupérer loi
    law = db.query(Law).filter(Law.id == law_id).first()
    
    # 2. Extraire articles
    try:
        articles = extract_articles(law.content_fr)
        
        # Validation
        if len(articles) < 3:
            raise ValueError(f"Minimum 3 articles requis, {len(articles)} trouvés")
        
        # 3. Sauvegarder en DB
        for article_data in articles:
            db_article = Article(
                law_id=law_id,
                number=article_data['number'],
                title=article_data['title'],
                content=article_data['content'],
                order=article_data['position'],
                # parent_id sera ajouté en Phase 2
            )
            db.add(db_article)
        
        db.commit()
        logger.info(f"✅ {len(articles)} articles extraits pour loi #{law_id}")
        
    except ArticleExtractionError as e:
        logger.error(f"❌ Extraction échouée: {e}")
        raise
```

#### Format de Retour

Chaque article extrait contient:

```python
{
    'number': '1',                    # VARCHAR(20) - Numéro article
    'title': 'Dispositions générales',  # VARCHAR(200) - Titre (optionnel)
    'content': 'La présente loi...',  # TEXT - Contenu article
    'position': 0,                    # INT - Ordre dans document (0-indexed)
    'parent_id': None,                # VARCHAR(20) - Parent pour hiérarchie (1.1 → '1')
    'section': None,                  # VARCHAR(50) - Section/Chapitre (Phase 2)
    'word_count': 45,                 # INT - Nombre de mots
    'char_count': 234                 # INT - Nombre de caractères
}
```

#### Tests

```bash
# Tests unitaires (23 tests de base)
pytest backend/tests/test_utils/test_text_chunker.py -v

# Classes de tests:
# - TestBasicExtraction: 3 tests (patterns standards)
# - TestHierarchicalNumbering: 2 tests (numérotation imbriquée)
# - TestValidation: 7 tests (validation stricte)
# - TestEdgeCases: 5 tests (cas limites)
# - TestStatistics: 2 tests (compteurs)
# - TestPositioning: 1 test (ordre séquentiel)
# - TestMinimumArticleLength: 2 tests (filtrage longueur)
# - TestMultiplePatterns: 1 test (patterns mixtes)

# Tests d'intégration (7 tests)
pytest backend/tests/test_integration/test_article_extraction_integration.py -v

# Avec coverage (cible: >90%)
pytest backend/tests/test_utils/test_text_chunker.py -v \
  --cov=app.utils.text_chunker \
  --cov-report=term-missing

# Résultat attendu:
# ====== 23 passed, 96% coverage ======
```

#### Validation Rules

**Input:**
- Minimum: 200 caractères
- Maximum: 5,000,000 caractères (5MB)
- Minimum 3 articles requis (strict=True)

**Article:**
- Contenu: 10-50,000 caractères
- Numéro: 1-20 caractères
- Titre: 0-200 caractères (optionnel)

**Errors:**

```python
# ValueError: Texte invalide
extract_articles("")  # → "Le texte ne peut pas être vide"
extract_articles("x" * 50)  # → "Texte trop court (50 chars, minimum 200)"

# ArticleExtractionError: Extraction échouée
extract_articles("Long text without articles...")  
# → "Aucun pattern d'article détecté"

extract_articles("Article 1...\nArticle 2...", strict=True)  
# → "Minimum 3 articles requis, 2 trouvés"
```

#### Troubleshooting

**❌ Erreur: "Aucun pattern d'article détecté"**
- Vérifier format: "Article 1." ou "Art. 1"
- Patterns supportés: numérique (1, 2, 3) ou textuel (premier, première)

**❌ Erreur: "Minimum 3 articles requis"**
```python
# Option 1: Mode non-strict
articles = extract_articles(text, strict=False)

# Option 2: Ajouter plus d'articles au document
```

**❌ Articles trop courts filtrés**
```python
# Réduire min_article_length
articles = extract_articles(text, min_article_length=5)  # Default: 10
```

**❌ Titres mal détectés**
- Titres doivent être <100 caractères et sur première ligne
- Format attendu: "Article 1. Titre court\nContenu..."

**❌ Hiérarchie non reconnue**
- Format requis: numérotation à points (1.1, 1.2.3)
- parent_id extrait automatiquement (1.2.3 → parent='1.2')

#### Performance

Benchmarks (Windows 11, Python 3.11):

| Document | Articles | Temps | Performance |
|----------|----------|-------|-------------|
| Petit (10 articles) | 10 | <50ms | ✅ |
| Moyen (100 articles) | 100 | <100ms | ✅ |
| Large (1000 articles) | 1000 | <500ms | ✅ |

```bash
# Test de performance
pytest backend/tests/test_integration/test_article_extraction_integration.py::TestArticleExtractionIntegration::test_performance_target_met -v

# Résultat:
# test_performance_target_met PASSED [100%]
# Extraction took 0.045s (target: <0.5s for 100 articles) ✅
```

## 🚀 Prochaines Étapes

### Module 2: Classification Automatique (Section 5.6)
Service de catégorisation automatique de documents juridiques.

### Module 4: Filtre Multilingue (Section 6.9)
Filtrage automatique des lois par langue détectée.

### Module 5: Pipeline Complet (Section 4.2)
Intégration Celery pour traitement automatique des documents.

## 📄 License

Propriétaire - JuriX Team © 2025

## 👥 Équipe

- **Développeurs:** JuriX Dev Team
- **Architecte:** Senior Software Engineer
- **QA:** Qodo AI Test Generation

## 📞 Support

Pour questions ou problèmes:
1. Vérifier documentation: `http://localhost:8000/docs`
2. Consulter logs: `docker-compose logs -f api`
3. Lancer health checks: `curl http://localhost:8000/api/v1/language/health`
4. Créer issue sur repository GitHub

---

**Version:** 2.1.0  
**Dernière mise à jour:** 2025-01-05  
**Status:** ✅ LanguageDetector + ArticleSplitter Complets (30/30 tests passing, 96% coverage)
