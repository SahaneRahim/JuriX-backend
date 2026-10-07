# JuriX Backend

API de la plateforme juridique camerounaise JuriX.

Recherche et question-réponse sur un corpus de textes officiels : recherche
plein texte et sémantique, extraction OCR des PDF scannés, découpage par
article, réponses citées via Mistral, classements par Groq.

## Pile technique

| Composant | Choix |
|---|---|
| API | FastAPI (Python 3.11) |
| Base | PostgreSQL 16 + `pgvector` + `pg_trgm` |
| Recherche plein texte | `tsvector` / `websearch_to_tsquery`, index GIN, triggers |
| Recherche sémantique | `pgvector` + index HNSW, embeddings EmbeddingGemma 768 dim. calculés en local (Gemini en option) |
| Cache | tables `query_cache` et `embedding_cache` |
| Réponses du chat, explications, comparaisons | Mistral `ministral-14b` (API REST, `LLM_PROVIDER=mistral`) ; Gemini en option |
| Classement d'intention et des lois | Groq : `qwen3.8-27b` (chat), `gpt-oss-120b` (lois) |
| Extraction PDF | Docling en local, OCR pleine page (Gemini en option) |
| Tâches de fond | `BackgroundTasks` FastAPI |

Pas de Redis, pas de Meilisearch, pas de Celery : recherche, cache et files
d'attente sont assurés par PostgreSQL et par le serveur applicatif.

## Prérequis

- Python 3.11
- PostgreSQL 16 avec les extensions `vector` et `pg_trgm`
- Une clé API Mistral (console.mistral.ai), pour le chat
- Une clé API Groq (console.groq.com), pour le classement des messages et des
  lois
- Le modèle EmbeddingGemma sur disque (voir plus bas), pour la recherche
  sémantique
- Pour ingérer des PDF : Docling et torch (`requirements-ingestion.txt`), un GPU
  CUDA de préférence

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

# Schéma, extensions, index, triggers. Alembic ne lit PAS le .env : la base
# visée est nommée à chaque commande, et elle est affichée avant de migrer.
DATABASE_URL=postgresql://jurix:jurix@localhost:5433/jurix_dev alembic upgrade head
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

### Modèle d'embeddings

EmbeddingGemma (330 Mo), à la révision épinglée dans `app/core/config.py`.
Toujours avec `--local-dir` : le cache Hugging Face sépare le `.onnx` de son
`.onnx_data`, et onnxruntime refuse alors de le charger.

```bash
hf download onnx-community/embeddinggemma-300m-ONNX \
  onnx/model_quantized.onnx onnx/model_quantized.onnx_data tokenizer.json \
  --revision 5090578d9565bb06545b4552f76e6bc2c93e4a66 \
  --local-dir models/embeddinggemma-300m-onnx
```

Il est chargé au démarrage de l'API (~1,5 Go de RAM). Sans lui, l'API démarre
quand même — plein texte, chat et comptes fonctionnent — mais la recherche
sémantique est coupée et `GET /api/v1/search/health` répond `degraded`.

## Configuration

Toutes les variables de `.env.example` sont réellement lues par
`app/core/config.py`. Les principales :

| Variable | Rôle |
|---|---|
| `DATABASE_URL` | `postgresql+asyncpg://…` — le driver asyncpg est obligatoire |
| `LLM_PROVIDER` | `mistral` (défaut) ou `gemini` : le modèle des réponses, explications et comparaisons |
| `MISTRAL_API_KEY` / `MISTRAL_MODEL` | Modèle des réponses (`ministral-14b-latest`) ; `MISTRAL_MODEL_SECOURS` (`ministral-8b-latest`) sert quand le premier reste saturé |
| `GROQ_API_KEY` | Classement d'intention (`GROQ_MODEL`) et classement des lois (`GROQ_MODEL_CLASSEMENT`), deux modèles pour deux quotas |
| `INTENT_CLASSIFIER` | `groq` (défaut) ou `llm` (le modèle du chat, un appel de plus par question) |
| `GEMINI_API_KEY` / `GEMINI_MODEL` | Seulement si `LLM_PROVIDER=gemini` ou `EMBEDDING_PROVIDER=gemini` |
| `EMBEDDING_PROVIDER` | `gemma` (défaut, local et gratuit) ou `gemini` (API payante). En changer impose de régénérer les vecteurs |
| `GEMMA_MODEL_DIR` | Dossier du modèle EmbeddingGemma (défaut `models/embeddinggemma-300m-onnx`) |
| `PDF_EXTRACTION_PAGES_PER_CALL` | Pages envoyées par appel d'extraction (défaut 20) |
| `SECRET_KEY` | Signature JWT. **Obligatoire hors développement** : l'application refuse de démarrer si la valeur du dépôt est conservée |
| `ADMIN_EMAIL` / `ADMIN_PASSWORD` | Valeurs par défaut de `scripts/create_admin.py` |

### Fournisseurs et quotas

Paliers gratuits vérifiés le 07/10/2026 :

| Fournisseur | Modèle | Plafonds |
|---|---|---|
| Mistral | `ministral-14b` | 30 requêtes par minute |
| Mistral | `ministral-8b` (secours) | 188 requêtes par minute |
| Groq, **par modèle** | `qwen3.8-27b`, `gpt-oss-120b` | 30 requêtes et 8 000 jetons par minute, 1 000 requêtes et 200 000 jetons par jour |

Chaque appel passe par un **limiteur** (`app/core/limiteur.py`) qui fait
attendre avant d'envoyer, plutôt que de collectionner les refus.

- **Mistral.** Un 429 est rejoué (`Retry-After`), puis le modèle de secours
  prend le relais ; au-delà de 20 s d'attente, l'utilisateur reçoit « service
  saturé, réessayez » (503), jamais « quota épuisé ». Une réponse coupée — budget
  de jetons, ou flux rompu en cours de route — est **complétée** par une suite ;
  la mention « réponse interrompue » ne reste que si la suite échoue. Chaque
  appel journalise modèle, fin, jetons et durée.
- **Groq, classement d'intention.** Consignes courtes, verdicts en cache. Quand
  Groq ne peut pas répondre dans les temps (quota, attente, panne), le message
  est traité comme une question de droit : c'est l'erreur sans conséquence. Mesure
  sur 122 messages étiquetés : 121 justes, aucune question de droit perdue
  (`python -m scripts.eval.evaluer_intention --limite 122`).
- **Groq, classement des lois.** Par lots (40 titres par requête). Jamais de
  catégorie inventée : un classement impossible laisse la catégorie vide et le
  dit dans `processing_error`.

### Reclassement des lois

Les catégories se recalculent par lots, en deux temps, avec relecture entre
les deux (`scripts/maintenance/reclassify_domains.py`) :

```bash
# 1. Verdicts dans data/reclassement/verdicts.jsonl, rien en base. Reprenable :
#    une relance saute les lois déjà classées. Code 4 = quota du jour épuisé,
#    l'heure de reprise est affichée.
python scripts/maintenance/reclassify_domains.py classer --extrait 0 --lot 40
python scripts/maintenance/reclassify_domains.py classer --incertains --extrait 500 --lot 15

# 2. Simulation : rapport, changements.csv et a_revoir.csv. À RELIRE.
python scripts/maintenance/reclassify_domains.py appliquer --dry-run

# 3. Écriture en base, catégories déjà posées comprises.
python scripts/maintenance/reclassify_domains.py appliquer --force
```

Une loi « à revoir » ou sous le seuil de confiance (`--seuil`, 0,6) n'est jamais
appliquée. Le nom de la catégorie est figé dans le texte vectorisé
(`embed_text`) : les lois de `changements.csv` sont à redécouper puis
revectoriser. Pour une ingestion de masse, `ingest_corpus.py --sans-classement`
laisse le classement à ce script : une requête pour 40 lois au lieu de 40.

## Authentification

JWT porteur, trois rôles : `user`, `admin`, `superadmin`.

| Route | Usage |
|---|---|
| `POST /api/v1/auth/signup` | Inscription publique : nom, adresse, mot de passe |
| `POST /api/v1/auth/google` | Connexion par Google Identity Services |
| `POST /api/v1/auth/login` | Formulaire OAuth2 (utilisé par `/docs`) |
| `POST /api/v1/auth/login/json` | JSON (utilisé par le front) |
| `GET /api/v1/auth/me` | Compte associé au jeton |

Les écritures, les endpoints d'administration, l'OCR et l'upload exigent un
rôle administrateur. L'inscription publique, elle, écrit `role: "user"` côté
serveur : son schéma n'expose pas ce champ et refuse toute clé inconnue.

**Deux durées de session.** 30 jours pour un compte ordinaire — l'intérêt d'un
compte étant de retrouver ses conversations, être déconnecté toutes les 30
minutes le rendait inutilisable. 12 heures dès que le compte porte un rôle
privilégié : les JWT sont sans état ici, `/logout` ne révoque rien, et il
n'existe aucune liste de révocation.

**Levier d'urgence :** `get_current_user` relit l'utilisateur en base à chaque
requête et refuse un compte inactif. Désactiver un compte révoque donc ses
jetons immédiatement — c'est le seul moyen d'annuler une session en cours sans
changer `SECRET_KEY`, ce qui déconnecterait tout le monde.

### Connexion Google

Pas de Firebase : le navigateur obtient un jeton d'identité de Google, le
serveur le vérifie avec `google-auth`, puis émet le JWT maison. Une dépendance,
zéro paquet npm, et **un seul fichier d'utilisateurs**.

Console Google Cloud, une fois : écran de consentement *External*, périmètres
`openid email profile` uniquement, puis un identifiant client OAuth de type
*Web application* dont les origines JavaScript autorisées couvrent
`http://localhost:5173`, `http://localhost:4173` et le domaine de production.
Ni URI de redirection, ni client secret. Reporter l'identifiant dans
`GOOGLE_CLIENT_ID` (backend) et `VITE_GOOGLE_CLIENT_ID` (frontend).

`GOOGLE_CLIENT_ID` vide ⇒ `/auth/google` répond 503 et le bouton n'est pas
rendu ; le reste fonctionne. Ce n'est pas une commodité : vérifier un jeton
sans audience acceptrait n'importe quel jeton Google émis pour n'importe quelle
application.

Un compte créé par mot de passe qui se connecte ensuite par Google est **lié**
au même compte, et garde son mot de passe. L'inverse est refusé en 409 : sans
vérification d'adresse de notre côté, autoriser un mot de passe sur une adresse
déjà rattachée à Google serait une prise de contrôle de compte.

### Mot de passe oublié

**Il n'y a aucune infrastructure d'envoi d'e-mail**, donc ni vérification
d'adresse, ni lien de réinitialisation. Deux conséquences assumées :

- `is_verified` ne vaut `true` que lorsque Google a affirmé `email_verified`.
  Il ne conditionne rien.
- La seule remise à zéro est administrative :
  `PUT /api/v1/admin/users/{id}` accepte un champ `password`. Pour un compte
  lié à Google, se reconnecter par Google fait office de récupération.

### Conversations et appartenance

`Conversation.user_id` est le propriétaire ; `NULL` signifie « anonyme,
appartient à qui détient le `session_id` ». Une conversation demandée par un
autre compte répond **404**, jamais 403 — un 403 confirmerait l'existence du
`session_id` et permettrait de les énumérer.

Le chat reste utilisable **sans compte**. Une conversation anonyme reprise par
un compte connecté lui est rattachée : on peut discuter, puis s'inscrire en
gardant le fil.

## Chaîne d'ingestion

```
PDF → extraction Docling (OCR pleine page) → normalisation → découpage par article
    → embeddings EmbeddingGemma → tsvector (trigger) → published
```

Un document est créé en `processing`. Il passe à `published` en cas de succès,
à `refused` avec `processing_error` en cas d'échec. Une page que l'extraction
n'a pas pu lire ne bloque pas le document : il est publié sans elle, et sa liste
part dans `processing_error`.

**OCR sur toutes les pages, par Docling, en local.** 97 % des pages du corpus
prc.cm sont des scans ; la couche texte que portent 55 % d'entre elles est l'OCR
du scanner, bruité (« portantnomination deresponsables »), mesuré à environ 20 %
de rappel. Docling l'ignore et relit l'image de chaque page (mode OCR pleine
page), avec la mise en page et les tableaux sur le GPU s'il y en a un.
`PDF_EXTRACTION_ENGINE=gemini` repasse sur l'API Gemini, payante.

**Pagination.** L'extraction rend un marqueur `<<PAGE:n>>` par page physique ;
chaque article porte la page où il commence.

### Ingérer le corpus

Deux passes, séparées pour la mémoire : Docling et EmbeddingGemma ne tiennent
pas ensemble dans quelques Go de RAM. Toutes deux sont **reprenables** — une
interruption, quelle qu'elle soit, ne coûte que le travail en cours, et relancer
la même commande reprend où elle s'était arrêtée.

```bash
pip install -r requirements-ingestion.txt     # Docling, torch (voir le fichier)

# Passe 1 — extraction, sans base. Une vingtaine d'heures pour les 13 970 pages.
# Détachée : elle survit à la fermeture du terminal.
setsid nohup .venv/bin/python scripts/extraire_corpus.py \
    > data/ingestion/extraction.out 2>&1 &
python scripts/extraire_corpus.py --etat      # avancement, durée restante estimée
touch data/ingestion/STOP                     # arrêt propre après le lot en cours

# Passe 2 — indexation en base. Ne relit que l'extraction faite : un document
# que la passe 1 n'a pas fini est laissé pour la prochaine exécution.
python scripts/ingest_corpus.py

# Enfin : vecteurs manquants éventuels, puis index vectoriel reconstruit en masse
python scripts/regenerate_embeddings.py --all
python scripts/regenerate_embeddings.py --reindex
```

- **Passe 1** (`scripts/extraire_corpus.py`) : chaque document est converti par
  lots de pages, chaque lot écrit en cache dès qu'il est fait
  (`data/ocr_cache/docling/`). Le travail se fait dans un processus enfant,
  recyclé régulièrement — la mémoire de Docling grossit au fil des pages. Si ce
  processus meurt, le lot en cours compte une tentative perdue ; au bout de
  trois, le lot est repris page à page, et seule la page qui échoue encore est
  abandonnée et signalée, sans bloquer le reste. Les petits documents passent
  d'abord ; aucun n'est exclu pour sa taille. Si `.env` ou les paquets changent
  pendant l'extraction, elle s'arrête plutôt que d'écrire dans un autre cache :
  relancer la commande. Suivi : `data/ingestion/extraction.log` et
  `progression.json`.
- **Passe 2** (`scripts/ingest_corpus.py`) : une loi en échec passe `refused` ;
  la relance reprend les lois `refused`, `processing` et `pending`, et ignore
  les `published`. Journal : `data/ingestion/indexation.log`.
- **Cache d'extraction.** Son empreinte porte les versions de Docling et du
  moteur OCR : changer de version, ou de moteur, extrait de nouveau dans un
  autre dossier, sans rien écraser. Les versions sont donc épinglées dans
  `requirements-ingestion.txt`.

## Découpage

`app/utils/text_chunker.py` découpe le document en articles, puis
`app/utils/chunk_refiner.py` classe chaque chunk et décide de son sort :

| `kind` | vectorisé | pourquoi |
|---|---|---|
| `article` | oui | le texte normatif |
| `legal_basis`, `preamble` | non | les visas ne répondent à aucune question |
| `boilerplate` | non | « sera enregistré, publié au Journal Officiel », identique dans des milliers de décrets |
| `roster` | un vecteur par liste | listes nominatives : le contenu garde tous les noms (affichage, plein texte), le vecteur n'en porte que le résumé ; une longue liste est coupée entre ses lignes |
| `signature`, `annexe` | non, oui | la signature et ce qui suit le dernier article : lieu, date, signataire hors index ; une annexe reste un chunk à part entière |
| `fragment` | non | moins de 120 caractères hors vrais articles (30 pour un article) : 62 % du corpus sous ce seuil sont des lignes de tableau ou de liste |
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

Les embeddings font **768 dimensions** et sont calculés **en local** par
EmbeddingGemma (`onnx-community/embeddinggemma-300m-ONNX`, int8, exécuté par
onnxruntime) : gratuit, sans réseau, ~1,5 Go de RAM dans le processus de l'API.
`EMBEDDING_PROVIDER=gemini` repasse sur l'API Gemini, à 768 dimensions elle
aussi : le schéma ne change pas. Le modèle se télécharge une fois — commande
dans `.env.example` — et il est chargé au démarrage : s'il manque,
`GET /api/v1/search/health` répond `degraded` au lieu de laisser la recherche
retomber en silence sur le plein texte.

Chaque vecteur porte l'empreinte du modèle qui l'a produit
(`articles.embedding_model`). Après une migration de dimension, une
restauration ou un changement de fournisseur, les vecteurs manquants ou
d'un autre modèle doivent être régénérés :

```bash
DATABASE_URL=postgresql://jurix:jurix@localhost:5433/jurix_dev alembic upgrade head
python scripts/regenerate_embeddings.py --all --batch-size 16   # reprenable
python scripts/regenerate_embeddings.py --reindex               # index en masse
```

Tant que le backfill n'est pas terminé, la recherche sémantique ne renvoie rien
et le mode hybride dégrade en recherche plein texte ; la santé de la recherche
le signale (`vectors.missing`, `vectors.foreign`).

### Re-ranking

Les chunks remontés passent par `app/services/reranker.py` avant d'être tronqués
et mis en cache. L'étage 1 est lexical (numéro d'article demandé, densité des
termes, expression exacte, titre de loi, pénalité des formules d'exécution) :
sans dépendance, sans réseau, actif par défaut. L'étage 2 fait noter les 20
meilleurs chunks par le modèle du chat ; il ajoute un appel sur le chemin critique,
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
  le second. La sortie est contrainte par un schéma JSON **strict** où chaque
  cellule doit citer ses articles (une grille coupée est redemandée une fois, au
  double du budget) ; un numéro cité sans article correspondant est renvoyé
  dans `unmatched_citations` plutôt qu'avalé, et le texte intégral des sources
  accompagne chaque cellule pour que le lecteur vérifie lui-même.

Aucun des deux n'est mis en cache : chaque appel dépense un appel au modèle.

### Mesure

`RRF_K`, `TEXT_WEIGHT` et `SEMANTIC_WEIGHT` ne sont **pas** calibrés sur ce
corpus. Le harnais qui les calibre :

```bash
python -m scripts.eval.generate_eval_set --sample 120   # puis RELECTURE
python -m scripts.eval.run_eval --dims 768 512 256      # la dimension vaut-elle son coût ?
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
pytest                                   # tout, sans aucun appel reseau
pytest -m "not integration"              # sans base
pytest -m groq_live                      # appels reels a Groq (une poignee)
pytest -m mistral_live                   # appels reels a Mistral (cinq)
pytest --cov=app --cov-report=term-missing
```

Sans base joignable, les tests qui en dépendent sont **ignorés avec la commande
à lancer** — jamais silencieusement verts.

**Aucun appel réel par défaut.** Les clés de Gemini, Groq et Mistral sont
remplacées par des clés factices, et une garde réseau fait échouer tout test qui
tenterait de joindre Groq ou Mistral : une suite complète vidait le quota
gratuit de Groq. Les tests marqués `groq_live` ou `mistral_live` lèvent la garde
du seul service visé, avec les vraies clés, et ne tournent que sur demande
(`-m`). `JURIX_E2E=1` garde les vraies clés et ouvre tout, pour un essai de bout
en bout délibéré.

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

`Dockerfile` fourni. Il lance `alembic upgrade head` **puis** uvicorn : les
migrations partent donc toutes seules à chaque démarrage. C'est voulu —
`search_vector`, les index GIN et les déclencheurs n'existent que dans les
migrations, jamais dans `Base.metadata`, si bien qu'un schéma créé par
`create_all` serait incomplet.

### Bascule des embeddings sur une base peuplée

**À ne jamais déclencher sans avoir d'abord vérifié s'il existe une base de
production, et ce qu'elle contient.**

La migration `c4d5e6f7a8b9` passe `articles.embedding` de `vector(3072)` à
`vector(768)` et met **tous** les vecteurs à NULL : un vecteur Gemini 3072 n'a
pas d'équivalent dans l'espace d'EmbeddingGemma. Le conteneur lançant
`alembic upgrade head` à chaque démarrage, **déployer ce code suffit à
déclencher la purge**. Jusqu'à la régénération, la recherche sémantique ne rend
rien et l'hybride répond en plein texte seul.

1. **Lire l'état de la base visée**, sans rien modifier :

   ```sql
   SELECT version_num FROM alembic_version;
   SELECT count(*) AS articles, count(embedding) AS vecteurs FROM articles;
   ```

   Aucun article : déployer suffit. Déjà en `c4d5e6f7a8b9` : la purge a eu
   lieu, reprendre à l'étape 5 si des vecteurs manquent.

2. **Sauvegarder.** `alembic downgrade` rétablit le schéma 3072, pas les
   vecteurs. Seul un dump permet de revenir en arrière sans repayer l'API :

   ```bash
   pg_dump --format=custom --file=jurix-avant-768.dump "postgresql://…"
   ```

3. **Mettre le modèle à disposition de l'API.** L'image Docker ne l'embarque
   pas encore. Sans lui, l'API démarre, mais en `degraded` et sans recherche
   sémantique. À défaut, `EMBEDDING_PROVIDER=gemini` (payant) fonctionne avec
   le même schéma 768.

4. **Déployer.** Le démarrage applique la migration.

5. **Régénérer, hors du processus de l'API.** Le script charge sa propre copie
   du modèle (~1,5 Go) et occupe tous les cœurs physiques moins un. Sur un
   petit hébergement, le lancer depuis une machine qui a le modèle, contre la
   base de production. Il est reprenable : une interruption ne coûte qu'un
   lot.

   ```bash
   export DATABASE_URL="postgresql+asyncpg://…"   # la base de production
   export PGSSLMODE=require
   python scripts/regenerate_embeddings.py --all --dry-run   # combien
   python scripts/regenerate_embeddings.py --all
   python scripts/regenerate_embeddings.py --reindex
   ```

   Ordre de grandeur mesuré sur le poste de développement : ~700 jetons par
   seconde, soit 1 h 30 à 3 h pour le corpus.

6. **Vérifier** `GET /api/v1/search/health` : `status` à `healthy`,
   `vectors.missing` et `vectors.foreign` à 0.

Le même déroulé vaut pour tout changement de fournisseur à dimension égale
(gemma vers gemini, ou un meilleur modèle plus tard), à une différence près :
sans migration, les anciens vecteurs restent en place pendant la régénération.
Leurs similarités avec une question encodée par l'autre modèle n'ont pas de
sens ; la santé les compte dans `vectors.foreign` et reste `degraded` tant
qu'il en reste.

### L'URL de la base : deux pièges opposés

**Ne mettez jamais `?sslmode=` ni `?channel_binding=` dans `DATABASE_URL`.** Le
dialecte asyncpg de SQLAlchemy transmet les paramètres qu'il ne connaît pas
directement à `asyncpg.connect()`, dont la signature ne les accepte pas :

```
TypeError: connect() got an unexpected keyword argument 'sslmode'
```

L'erreur ne survient qu'au premier accès à la base, donc bien après un démarrage
en apparence réussi. Or c'est exactement l'URL que fournissent les boutons
« copier » des bases infogérées. La configuration refuse maintenant de se
charger dans ce cas, avec le remède dans le message : **posez `PGSSLMODE=require`
en variable d'environnement**, lue aussi bien par asyncpg que par libpq.

Symétriquement, `alembic/env.py` **retire la chaîne de requête** avant de passer
l'URL à psycopg2 : `prepared_statement_cache_size`, indispensable côté
application derrière un pooler en mode transaction, ferait échouer libpq
(`invalid URI query parameter`) et donc le démarrage du conteneur.

### Les autres points

- **`data/` est éphémère.** Sans réglage, les PDF disparaissent au redéploiement
  alors que les lignes en base subsistent. Poser `DOCUMENTS_BASE_URL` sur un
  magasin HTTPS public : les documents sont alors récupérés en flux et mis en
  cache sur disque, et rien d'autre ne change. Vide, le comportement historique
  (lecture dans `./data/uploads`) est conservé.
- **`--workers 1`** : plusieurs états sont en mémoire du processus (connexions
  WebSocket du suivi de lot, étranglement des envois de courriel par IP).
- **Mémoire** : le modèle d'embeddings occupe ~1,5 Go dans le processus de
  l'API. Une offre à 512 Mo ne suffit pas ; à défaut, `EMBEDDING_PROVIDER=gemini`
  (payant). L'image Docker n'embarque pas encore le modèle.
- **`SECRET_KEY`** : `openssl rand -hex 32`. Avec `ENVIRONMENT` différent de
  `development`, le conteneur **refuse de démarrer** sur la valeur du dépôt.
- **`ALLOWED_ORIGINS`** : origine exacte, sans slash final. `allow_credentials`
  étant actif, le joker `"*"` est refusé par les navigateurs.
- **Envoi de courriel** : `BREVO_API_KEY`, `BREVO_SENDER_EMAIL` et
  `FRONTEND_BASE_URL`. Sans elles, l'inscription et la connexion fonctionnent
  normalement ; seules la vérification d'adresse et la réinitialisation de mot
  de passe restent inertes. Le transport est l'API HTTPS de Brevo et **jamais
  SMTP** : les hébergeurs gratuits bloquent les ports sortants 25, 465 et 587.
- **Extensions PostgreSQL** : `vector`, `pg_trgm` et `unaccent` doivent exister
  **dans le schéma `public`**. Certains hébergeurs les installent ailleurs par
  défaut, ce qui casse `immutable_unaccent` (migration `a6b7c8d9e0f1`). Créer
  explicitement `CREATE EXTENSION ... WITH SCHEMA public;` avant la première
  migration.
- **Maintien en éveil** : `GET /health` ne touche pas la base, délibérément —
  c'est la cible d'une sonde fréquente. Une seconde sonde, bien plus espacée,
  doit viser une vraie route de lecture si l'hébergeur met la base en pause pour
  inactivité.

### Mot de passe oublié

La réinitialisation par courriel existe (`POST /auth/password/forgot` puis
`/auth/password/reset`). Sans clé Brevo configurée, elle est inerte : le seul
recours reste alors `PUT /admin/users/{id}`, ou une reconnexion par Google si le
compte y est lié.
