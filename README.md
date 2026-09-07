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
| Recherche sémantique | `pgvector` + index HNSW, embeddings `gemini-embedding-001` tronqués à 1536 dim. |
| Cache | tables `query_cache` et `embedding_cache` |
| LLM | Gemini (`google-genai`) |
| Extraction PDF | Gemini multimodal |
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
docker run -d --name jurix-pg -p 5433:5432 \
  -e POSTGRES_USER=jurix -e POSTGRES_PASSWORD=jurix -e POSTGRES_DB=jurix_dev \
  pgvector/pgvector:pg16

# La base de TEST vit sur le meme serveur, sous un autre nom :
docker exec jurix-pg psql -U jurix -d jurix_dev -c "CREATE DATABASE jurix_test"
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
| `PDF_EXTRACTION_PAGES_PER_CALL` | Pages envoyées par appel d'extraction (défaut 20) |
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
upload → extraction Gemini → normalisation → découpage par article
       → embeddings → tsvector (trigger) → published
```

Un document est créé en `processing`. Il passe à `published` en cas de succès,
à `refused` avec `processing_error` en cas d'échec — jamais de contenu
partiellement extrait dans le corpus public.

**Aucun repli dégradé.** L'extraction réessaie trois fois sur une saturation du
fournisseur (503), jamais sur un quota épuisé (429) ; au-delà, l'ingestion
échoue. La couche texte des PDF scannés du corpus a été mesurée à environ 20 %
de rappel (filigrane inséré au milieu des phrases, cachets lus comme du
charabia, un document sur cinq sans aucun texte exploitable) : la publier serait
pire que de ne rien publier.

**Pagination.** Gemini rend un marqueur `<<PAGE:n>>` par page, ce que LlamaParse
ne faisait pas — d'où des articles portant tous `page_number = 1`. C'est la
raison du changement de moteur, pas le prix.

**Refus de transcription.** Le modèle rend parfois `finish_reason=RECITATION`
sur une page d'acte officiel dont la forme lui est familière : mesuré à une page
de garde sur sept documents. Le refus est **par page** — le lot est alors rejoué
page par page, et seule la page refusée est perdue, signalée dans
`processing_error`. Reformuler la consigne n'y change rien (trois variantes
essayées), réduire le lot non plus.

**Quota.** Le palier gratuit plafonne à 20 appels de génération par jour et par
modèle. Un document d'une page coûte un appel ; les 72 PDF locaux (481 pages) en
coûtent 83. Le script de ré-extraction est reprenable : le cache `sha256` fait
qu'un fichier déjà traité n'est jamais repayé.

## Découpage

`app/utils/text_chunker.py` découpe le document en articles, puis
`app/utils/chunk_refiner.py` classe chaque chunk et décide de son sort :

| `kind` | vectorisé | pourquoi |
|---|---|---|
| `article` | oui | le texte normatif |
| `legal_basis`, `preamble` | non | les visas ne répondent à aucune question |
| `boilerplate` | non | « sera enregistré, publié au Journal Officiel », identique dans des milliers de décrets |
| `roster` | non | listes nominatives, effondrées en un seul chunk |
| `fragment` | non | moins de 120 caractères : 62 % du corpus sous ce seuil sont des lignes de tableau ou de liste |
| `table`, `continuation` | selon la taille | découpés aux alinéas au-delà de 3 000 caractères |

**Rien n'est supprimé.** Un chunk non vectorisé reste en base, affichable et
trouvable en recherche plein texte — il ne consomme simplement pas d'appel
d'embedding et n'encombre pas les résultats sémantiques.

`embed_text` est le texte réellement envoyé au modèle : le contenu **préfixé de
l'en-tête du document** (référence, titre, date, catégorie, section, page). Sans
lui, « Article 3.- La dépense résultant des présentes dispositions sera imputée
sur le budget de l'État » est indistinguable des milliers d'articles identiques
du corpus. `content` reste intact : ce qui est affiché et cité ne change pas.

## Recherche et RAG

La recherche renvoie des **chunks** — un article, pas un document. `SearchResult`
(niveau loi) reste exposé pour le front et est dérivé de ces chunks ;
`SearchResponse.chunks` porte les articles eux-mêmes, avec leur numéro, leur
section, leur page et leur contenu intégral. C'est ce que consomme le RAG, et
c'est ce qui permet à une citation de pointer un `article_id`.

Les embeddings font **1536 dimensions** (`output_dimensionality`, troncature
Matryoshka renormalisée) et non les 3072 natifs : au-dessus de 2000, pgvector
refuse tout index HNSW ou IVFFlat, et chaque recherche sémantique balayait la
table.

Après un changement de dimension ou une restauration, les vecteurs doivent être
régénérés :

```bash
alembic upgrade head
python scripts/regenerate_embeddings.py --all --batch-size 16   # reprenable
python scripts/regenerate_embeddings.py --reindex               # index en masse
```

Tant que le backfill n'est pas terminé, la recherche sémantique ne renvoie rien
et le mode hybride dégrade en recherche plein texte.

### Re-ranking

Les chunks remontés passent par `app/services/reranker.py` avant d'être tronqués
et mis en cache. L'étage 1 est lexical (numéro d'article demandé, densité des
termes, expression exacte, titre de loi, pénalité des formules d'exécution) :
sans dépendance, sans réseau, actif par défaut. L'étage 2 fait noter les 20
meilleurs chunks par Gemini ; il ajoute un appel facturé sur le chemin critique,
reste désactivé (`RERANK_LLM_ENABLED`) et dégrade toujours vers l'étage 1.

### Explication et comparaison

Deux usages du modèle qui ne passent pas par le pipeline de chat, parce qu'il
n'y a rien à converser :

- `POST /laws/{id}/explain-article` explique un article en langage courant à
  partir de son texte, de ses deux voisins et des métadonnées du document
  (`app/services/explanation_service.py`). L'article est résolu en base par son
  numéro ; à défaut, l'extrait envoyé par la page sert de repli, mais seulement
  après vérification qu'il provient bien du document — sans ce contrôle, la
  route serait un proxy de prompt gratuit.
- `POST /compare` compare deux régimes sur des critères imposés
  (`app/services/comparison_service.py`). **Une recherche par sujet**, l'une
  après l'autre : une requête unique mélangeant les deux termes rendait six
  articles d'un régime contre trois de l'autre, en ratant le bloc qui définissait
  le second. La sortie est contrainte par un schéma JSON où chaque cellule doit
  citer ses articles ; un numéro cité sans article correspondant est renvoyé
  dans `unmatched_citations` plutôt qu'avalé, et le texte intégral des sources
  accompagne chaque cellule pour que le lecteur vérifie lui-même.

Aucun des deux n'est mis en cache : chaque appel dépense un appel Gemini.

### Mesure

`RRF_K`, `TEXT_WEIGHT` et `SEMANTIC_WEIGHT` ne sont **pas** calibrés sur ce
corpus. Le harnais qui les calibre :

```bash
python -m scripts.eval.generate_eval_set --sample 120   # puis RELECTURE
python -m scripts.eval.validate_slicing --dim 768       # 40 appels
python -m scripts.eval.run_eval --dims 3072 1536 768    # la dimension vaut-elle son coût ?
python -m scripts.eval.run_eval --sweep rrf             # les poids
```

Le jeu d'évaluation est committé sous `tests/fixtures/eval/`, les résultats de
run vont dans `data/eval_runs/` (ignoré). Un item non relu ou un corpus qui a
bougé depuis la génération font **échouer** la mesure plutôt que produire des
chiffres faux. Avec 60 questions, l'erreur type est d'environ 6 points : un
écart de 2 points entre deux configurations n'est pas un résultat.

## Tests

La suite tourne sur un **vrai PostgreSQL** — le cœur du produit est du SQL
PostgreSQL brut, et les objets dont il dépend (`search_vector`, index GIN,
triggers, extensions) n'existent que dans les migrations, jamais dans
`Base.metadata`.

```bash
<!-- La base de test partage le serveur de developpement : un seul conteneur,
     deux bases. Deux conteneurs sur deux ports etaient annonces ici, et la
     realite n'en a jamais compte qu'un. -->

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
