# JuriX Backend

API de la plateforme juridique camerounaise JuriX.

Recherche et question-réponse sur un corpus de textes officiels : recherche
plein texte et sémantique, extraction OCR des PDF scannés, découpage par
article, réponses citées via Gemini.

## Pile technique

| Composant | Choix |
|---|---|
| API | FastAPI (Python 3.11) |
| Base | PostgreSQL 16 + `pgvector` + `pg_trgm` |
| Recherche plein texte | `tsvector` / `websearch_to_tsquery`, index GIN, triggers |
| Recherche sémantique | `pgvector`, embeddings `gemini-embedding-001` (3072 dim.) |
| Cache | tables `query_cache` et `embedding_cache` |
| LLM | Gemini (`google-genai`) |
| OCR | LlamaParse v2 |
| Tâches de fond | `BackgroundTasks` FastAPI |

Pas de Redis, pas de Meilisearch, pas de Celery : recherche, cache et files
d'attente sont assurés par PostgreSQL et par le serveur applicatif.

## Prérequis

- Python 3.11
- PostgreSQL 16 avec les extensions `vector` et `pg_trgm`
- Une clé API Gemini (Google AI Studio)
- Une clé LlamaCloud pour l'extraction OCR

Le plus simple pour la base :

```bash
docker run -d --name jurix-pg -p 5432:5432 \
  -e POSTGRES_USER=jurix -e POSTGRES_PASSWORD=jurix -e POSTGRES_DB=jurix_db \
  pgvector/pgvector:pg16
```

## Installation

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # production
pip install -r requirements-dev.txt      # + outils de test

cp .env.example .env                     # puis renseigner les valeurs

alembic upgrade head                     # schéma, extensions, index, triggers
python scripts/create_admin.py           # premier compte administrateur

uvicorn app.main:app --reload
```

API sur `http://localhost:8000`, documentation interactive sur `/docs`.

Le bouton **Authorize** de `/docs` fonctionne : il appelle
`POST /api/v1/auth/login` avec le compte créé ci-dessus.

### Modèle de détection de langue

`langdetect` + fastText. Le modèle fastText (131 Mo) n'est pas versionné :

```bash
mkdir -p models/fasttext
curl -L -o models/fasttext/lid.176.bin \
  https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

Sans lui, `POST /api/v1/language/detect` échoue ; le pipeline d'ingestion
retombe sur une heuristique par mots vides.

## Configuration

Toutes les variables de `.env.example` sont réellement lues par
`app/core/config.py`. Les principales :

| Variable | Rôle |
|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://…` — le driver asyncpg est obligatoire |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | LLM des réponses. Modèles Flash actuels : `gemini-3.8-flash`, `3.7`, `3.6`, `3.5` |
| `LLAMA_CLOUD_API_KEY` / `LLAMA_PARSE_TIER` | OCR. `cost_effective` est le tier recommandé |
| `SECRET_KEY` | Signature JWT. **Obligatoire hors développement** : l'application refuse de démarrer si la valeur du dépôt est conservée |
| `ADMIN_EMAIL` / `ADMIN_PASSWORD` | Valeurs par défaut de `scripts/create_admin.py` |

## Authentification

JWT porteur, trois rôles : `user`, `admin`, `superadmin`.

| Route | Usage |
|---|---|
| `POST /api/v1/auth/login` | Formulaire OAuth2 (utilisé par `/docs`) |
| `POST /api/v1/auth/login/json` | JSON (utilisé par le front) |
| `GET /api/v1/auth/me` | Compte associé au jeton |

Les écritures, les endpoints d'administration, l'OCR et l'upload exigent un
rôle administrateur. Il n'existe pas d'inscription publique : les comptes sont
créés par `POST /api/v1/admin/users` ou par `scripts/create_admin.py`.

## Chaîne d'ingestion

```
upload → LlamaParse → normalisation → découpage par article
       → embeddings → tsvector (trigger) → published
```

Un document est créé en `processing`. Il passe à `published` en cas de succès,
à `refused` avec `processing_error` en cas d'échec — jamais de contenu
partiellement extrait dans le corpus public.

**Aucun repli dégradé.** LlamaParse réessaie 4 fois avec un délai croissant ;
au-delà, l'ingestion échoue. La couche texte des PDF scannés du corpus a été
mesurée à environ 20 % de rappel (filigrane inséré au milieu des phrases,
cachets lus comme du charabia, un document sur cinq sans aucun texte
exploitable) : la publier serait pire que de ne rien publier.

Les extractions sont mises en cache par `sha256` : un fichier déjà traité n'est
jamais repayé.

## Tests

La suite tourne sur un **vrai PostgreSQL** — le cœur du produit est du SQL
PostgreSQL brut, et les objets dont il dépend (`search_vector`, index GIN,
triggers, extensions) n'existent que dans les migrations, jamais dans
`Base.metadata`.

```bash
docker run -d --name jurix-pg-test -p 5433:5432 \
  -e POSTGRES_USER=jurix -e POSTGRES_PASSWORD=jurix -e POSTGRES_DB=jurix_test \
  pgvector/pgvector:pg16

export TEST_DATABASE_URL=postgresql+asyncpg://jurix:jurix@localhost:5433/jurix_test
pytest                                   # tout
pytest -m "not integration"              # sans base
pytest --cov=app --cov-report=term-missing
```

Sans base joignable, les tests qui en dépendent sont **ignorés avec la commande
à lancer** — jamais silencieusement verts.

## Structure

```
app/
  api/routes/     endpoints HTTP
  core/           configuration, base de données, authentification
  models/         modèles SQLAlchemy
  schemas/        schémas Pydantic
  services/       recherche, embeddings, RAG, OCR, classification
  tasks/          pipeline d'ingestion
  utils/          découpage en articles, raffinage des chunks, fichiers
alembic/versions/ migrations
scripts/          administration et maintenance
tests/            unitaires et intégration
```

## Déploiement

`Dockerfile` fourni. Points à vérifier avant de déployer :

- `alembic upgrade head` n'est pas lancé par le conteneur — à exécuter séparément.
- `data/` est éphémère : monter un volume ou un stockage objet, sinon les PDF
  uploadés disparaissent au redéploiement alors que les lignes en base subsistent.
- `--workers 1` : plusieurs états sont en mémoire du processus (connexions
  WebSocket du suivi de lot, registre des tâches de fond).
