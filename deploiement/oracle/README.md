# Déploiement de JuriX sur Oracle Cloud (palier gratuit)

Tout JuriX tourne sur **une seule machine virtuelle** Oracle Cloud du palier
« Always Free » : une VM ARM de 2 cœurs et 12 Go de mémoire, avec 200 Go de
disque, gratuite tant qu'on reste dans ces limites.

```
navigateur ──https──> Caddy (ports 80/443, certificat automatique)
                        ├── /api/*, /health ──> api (FastAPI) ──https──> Voyage AI (embeddings)
                        │                          └──> db (PostgreSQL 16 + pgvector)
                        └── tout le reste ──────> front (Node : SvelteKit, rendu serveur)
                                                     └──> api, par le réseau interne (http://api:8000)
```

- **Un seul domaine** — le sous-domaine, par exemple `jurix.exemple.cm` — sert
  le front et l'API (sous `/api/v1`). Pas de CORS, un seul certificat.
- **Le front est un serveur Node** (SvelteKit, adapter-node) : les pages
  publiques arrivent remplies, lisibles par un moteur de recherche ou un aperçu
  de lien. Pour les remplir, il appelle l'API par le réseau interne de Docker,
  sans repasser par Caddy. Son build vit dans `front-node/`, envoyé par
  `deployer.sh`.
- **Front ou API injoignable** (conteneur arrêté, ou qui redémarre pendant une
  mise à jour) : Caddy sert `caddy/maintenance.html`, en 503, au lieu d'une
  erreur 502 vide. Une page en 503 rendue par le front lui-même (l'API ne lui
  répond pas) passe telle quelle.
- **Aucun port de la base n'est exposé** : seule l'API la voit.
- Mistral, Groq et Voyage AI (les embeddings de la recherche sémantique)
  restent des API externes ; rien ne tourne pour eux ici. L'API a donc besoin
  du HTTPS sortant, ouvert par défaut.
- L'ingestion de nouveaux PDF (Docling, OCR) se fait sur la machine de
  développement, pas sur la VM.

Besoins : ~2 à 3 Go pour PostgreSQL, et bien moins pour l'API depuis qu'elle ne
garde plus de modèle d'embeddings en mémoire (EmbeddingGemma prenait ~1,5 Go) :
à mesurer par `docker stats` après le premier déploiement. Le front Node est
plafonné à 384 Mo.

Les limites de mémoire de `docker-compose.yml` (base 4 Go, API 5 Go, front
384 Mo, Caddy 256 Mo : 9,6 Go en tout) supposent une VM de 12 Go. `mem_limit`
est un plafond, pas une réservation : rien n'est retenu d'avance, mais un
conteneur qui atteint le sien est tué, puis redémarre.

**Vérifier d'abord la mémoire de la VM avec `free -g`.** Si elle n'a que 6 Go
(créée en 1 OCPU et 6 Go, §2), réduire ces limites dans le `docker-compose.yml`
**du dépôt**, et commiter avant de lancer `deployer.sh` : une modification
faite sur la VM est écrasée à chaque déploiement. Pour 6 Go :

| Service | 12 Go (actuel) | 6 Go |
|---|---|---|
| `db` | `mem_limit: 4g`, `shared_buffers=1GB`, `effective_cache_size=4GB` | `mem_limit: 2g`, `shared_buffers=512MB`, `effective_cache_size=1536MB` |
| `api` | `mem_limit: 5g` | `mem_limit: 1536m` |
| `front` | `mem_limit: 384m` | inchangé |
| `caddy` | `mem_limit: 256m` | inchangé |

Pour l'API, 1,5 Go : l'image n'installe ni Docling ni torch (absents de
`requirements.txt`), et son seul modèle en mémoire est celui de fastText
(126 Mo) ; le même code tient ~250 Mo mesurés en développement. La marge sert à
l'OCR de l'administration (`/api/v1/ocr`), qui rend en mémoire toutes les pages
d'un PDF à 300 dpi, ~25 Mo chacune : au-delà d'une quarantaine de pages, le
conteneur serait tué. Les gros scans s'OCRisent sur la machine de
développement, comme le reste de l'ingestion. Après le premier déploiement,
comparer avec `docker stats --no-stream`.

## 1. Le compte Oracle (une fois, par toi)

1. Inscription sur cloud.oracle.com, offre gratuite. La carte bancaire sert à
   vérifier l'identité ; rien n'est débité tant que le compte reste gratuit (ne
   pas le passer en « Pay As You Go »).
2. **Région d'origine : Marseille (`eu-marseille-1`).** Le choix est
   **définitif** : les ressources gratuites n'existent que dans cette région.
   Marseille est une région mature, et un point d'arrivée majeur des câbles
   sous-marins africains, donc proche du Cameroun en réseau. Aucune donnée
   publique ne garantit la disponibilité des VM ARM gratuites dans une région :
   des pénuries sont signalées à Francfort comme à Johannesburg.

## 2. La machine (une fois, par toi)

Console Oracle → *Compute* → *Instances* → *Create instance* :

| Réglage | Valeur |
|---|---|
| Image | Canonical Ubuntu 24.04 (aarch64) |
| Shape | `VM.Standard.A1.Flex` (« Always Free-eligible »), **2 OCPU, 12 Go** |
| Réseau | VCN par défaut, sous-réseau public, **adresse IP publique** |
| Clé SSH | coller le contenu de `~/.ssh/jurix_oracle.pub` de la machine de développement |
| Disque de démarrage | 100 Go (le gratuit couvre 200 Go au total) |

Si Oracle répond **« Out of host capacity »** : la région manque
momentanément de machines ARM gratuites. Réessayer à d'autres heures, souvent
tôt le matin ; la capacité revient. Sinon, créer la VM avec 1 OCPU et 6 Go, puis
l'agrandir plus tard (*Edit* → *Shape*).

Puis **ouvrir les ports 80 et 443** : *Networking* → *Virtual cloud networks* →
le VCN → *Security Lists* → *Default Security List* → *Add Ingress Rules* :
source `0.0.0.0/0`, TCP, ports `80` et `443` (et UDP `443`, facultatif, pour
HTTP/3).

Noter l'**adresse IP publique** de la VM.

## 3. Le DNS (une fois, par ton frère)

Sur son domaine, un seul enregistrement :

| Type | Nom | Valeur |
|---|---|---|
| `A` | `jurix` (ou le sous-domaine choisi) | l'IP publique de la VM |

Son serveur n'héberge rien : l'enregistrement DNS envoie simplement les
visiteurs vers la VM.

## 4. Premier déploiement

Depuis la machine de développement, à la racine du dépôt backend :

```bash
IP=1.2.3.4                          # l'IP publique de la VM
export JURIX_DOMAINE=jurix.exemple.cm  # le vrai domaine : deployer.sh refuse cet exemple

# a. Le code, puis la préparation de la VM (Docker, pare-feu, swap, sauvegardes)
git archive HEAD | ssh -i ~/.ssh/jurix_oracle ubuntu@$IP "mkdir -p jurix && tar -x -C jurix"
ssh -i ~/.ssh/jurix_oracle ubuntu@$IP "bash jurix/deploiement/oracle/installer.sh"

# b. Les secrets, SUR la VM uniquement : domaine, mot de passe de la base et
#    SECRET_KEY (openssl rand -hex 32 / -hex 24), clés Mistral, Groq et Voyage
#    (compte Voyage AVEC moyen de paiement), ADMIN_EMAIL et ADMIN_PASSWORD
#    (le compte administrateur de la production)
ssh -i ~/.ssh/jurix_oracle ubuntu@$IP
cd jurix/deploiement/oracle && cp env.production.exemple .env && nano .env
exit

# c. Tout le reste : front, PDF (5,6 Go, reprenable), base, démarrage
deploiement/oracle/deployer.sh ubuntu@$IP --avec-donnees
```

Prérequis sur la machine de développement : `deploiement/oracle` **entièrement
commité** (la VM reçoit HEAD) ; le dépôt du front à côté de celui-ci (ou
`JURIX_FRONT`), sur la branche du rendu serveur (adapter-node), **sans
modification non commitée** ; l'accès à npm : `deployer.sh` lance `npm ci`
dans le front, ce qui **remplace ses `node_modules`** par ceux de son
`package-lock.json` ; le conteneur `jurix-pg` démarré ; `jurix_dev` à la
dernière migration **du commit envoyé**, ses vecteurs Voyage chargés
(`deployer.sh` refuse de l'envoyer sinon : la migration viderait les vecteurs
sur la VM, qui n'a pas de quoi les recharger sans tout recalculer).

`deployer.sh` s'arrête **avant tout envoi** :

- si `deploiement/oracle` a des modifications non commitées ou des fichiers non
  suivis (ailleurs dans le backend, il prévient seulement : seul HEAD part) ;
- si la VM ne répond pas, si son `.env` n'a pas le **même `JURIX_DOMAINE`**
  qu'ici (le domaine figé dans le front doit être celui d'`ORIGIN` et
  d'`ALLOWED_ORIGINS`), ou si c'est le domaine d'exemple `jurix.exemple.cm` ;
- si `JURIX_DOMAINE` (ici ou sur la VM) ou `GOOGLE_CLIENT_ID` contient des
  guillemets, des espaces, `$`, `#` ou `/` : ces valeurs sont lues telles
  quelles, là où docker compose les nettoierait ;
- avec `--avec-donnees`, si `jurix_dev` n'est pas à la tête des migrations
  **du commit envoyé** (extrait par `git archive`, et non pris dans la copie de
  travail), ou s'il lui manque des vecteurs. Le contrôle est refait juste avant
  le dump : une migration appliquée à `jurix_dev` pendant l'envoi des PDF
  arrête le déploiement ;
- si le dépôt du front a des modifications non commitées (le front envoyé doit
  être un commit, qu'on sait reconstruire), des `dependencies` dans son
  `package.json` (l'image n'a pas de `node_modules`), ou s'il a changé pendant
  la construction (nouveau commit, fichier modifié) ;
- si le build n'a pas `build/index.js` et `build/client` (build statique), ou si
  le domaine n'est pas figé dans `build/client/_app`.

Le front est construit avec `VITE_API_URL=https://<domaine>` et `VITE_WS_URL`
vide : une valeur du `.env` local du front ne s'y glisse pas. Seuls `build/` et
`package.json` partent, dans `front-node.nouveau/`, échangé avec `front-node/`
une fois l'envoi complet ; `front-node/VERSION`, écrit en dernier, décrit ce
**dernier front envoyé** (commit du backend, commit du front, date d'envoi). Le
build remplacé passe dans `front-node.ancien/` seulement s'il était complet
(avec son `VERSION`) et d'un autre commit du front : un envoi coupé est jeté,
et renvoyer le même front ne chasse pas ce secours. L'image du front **copie**
ces fichiers à sa construction : le conteneur en place n'est pas touché pendant
l'envoi.

Une clé changée dans `.env` ne prend effet qu'avec `docker compose up -d api`,
qui recrée le conteneur : `docker compose restart` ne relit pas le fichier.

Connexion Google (facultative) : `GOOGLE_CLIENT_ID` dans le `.env` de la VM
(`deployer.sh` le reprend pour construire le front), et, dans la console Google
Cloud, `https://jurix.exemple.cm` ajouté aux **origines JavaScript autorisées**
de cet identifiant. Sans cela, le bouton Google échoue.

La base de la VM n'est jetée (DROP), par `demarrer.sh`, que dans trois cas :

- elle n'a **aucune loi** et `sauvegardes/initial.dump` existe : le premier
  déploiement, avec le dump de `jurix_dev`, **sans les comptes ni les
  conversations** de développement (le corpus seulement) ;
- une restauration est **demandée à ce lancement**, par `JURIX_RESTAURER` ou
  `deployer.sh --restaurer` (§6) : une base qui a des lois est d'abord
  sauvegardée ;
- une restauration **interrompue** reprend, de zéro, avec le même dump, au
  lancement suivant de `demarrer.sh`.

Une fois restauré, `initial.dump` est renommé `initial.restaure_….dump` : aucune
mise à jour ne pourra plus le rejouer. Un `initial.dump` qui arrive sur une base
déjà peuplée n'est pas restauré, et devient `initial.ignore_….dump` : une
relance qui trouverait plus tard la base sans loi ne pourra pas le rejouer non
plus. Le compte
administrateur de `.env` est créé ensuite ; si un compte ordinaire existe déjà
à cette adresse, `create_admin.py` refuse de le promouvoir sans `--promouvoir`.
`demarrer.sh` refuse de démarrer si un secret du `.env` est vide ou recopié de
l'exemple, et l'API refuse une `SECRET_KEY` de moins de 32 caractères.

Le certificat HTTPS est demandé dès que Caddy démarre. Si le DNS ne pointait
pas encore sur la VM, Caddy réessaie de lui-même, à intervalles croissants
(jusqu'à plusieurs heures) : une fois le DNS en place, `docker compose restart
caddy` relance la demande tout de suite.

Vérifier : `https://jurix.exemple.cm`, `https://jurix.exemple.cm/health`, et
`https://jurix.exemple.cm/api/v1/search/health` : `status` à `healthy` et
`embedding_service` à `ready` (`demarrer.sh` le contrôle aussi et prévient,
sinon la recherche sémantique serait coupée sans bruit).

### Réglages du front

Le service `front` ne lit **pas** le `.env` (il porte les secrets de l'API) :
ses réglages sont dans `docker-compose.yml`, dérivés de `JURIX_DOMAINE`. Rien à
ajouter au `.env`.

| Variable | Valeur | Rôle |
|---|---|---|
| `ORIGIN` | `https://${JURIX_DOMAINE}` | l'origine publique, annoncée à l'API, dont `ALLOWED_ORIGINS` vient du même domaine ; `VITE_API_URL` doit la désigner aussi |
| `API_INTERNE` | `http://api:8000` | le rendu serveur appelle l'API par le réseau interne (`src/hooks.server.ts` du front) |
| `KEEP_ALIVE_TIMEOUT`, `HEADERS_TIMEOUT` | `125`, `126` (secondes) | Node garde ses connexions plus longtemps que Caddy (2 min) : pas de 502 sur une connexion fermée au mauvais moment |
| `SHUTDOWN_TIMEOUT` | `8` | arrêt propre en moins des 10 s que Docker accorde |
| `NODE_OPTIONS` | `--max-old-space-size=256` | le tas de Node sous la limite de 384 Mo du conteneur |
| `PORT`, `HOST` | `3000`, `0.0.0.0` | l'adresse que Caddy joint (`front:3000`) |

`VITE_API_URL` et `VITE_GOOGLE_CLIENT_ID`, eux, sont figés **au build** par
`deployer.sh` : en changer demande un nouveau déploiement. `VITE_WS_URL` y est
forcée vide : le front en dérive l'adresse de `VITE_API_URL`.

## 5. Mettre à jour

```bash
JURIX_DOMAINE=jurix.exemple.cm deploiement/oracle/deployer.sh ubuntu@$IP
```

Construit et contrôle d'abord le front (état **commité** lui aussi), l'envoie
dans `front-node/`, puis envoie l'état **commité** du backend (`git archive` :
jamais un `.env` local), et efface sur la VM les fichiers retirés du dépôt.
Fait une sauvegarde de la base, puis relance avec `docker compose up -d
--build` (les migrations s'appliquent au démarrage de l'API ; le front est
reconstruit si son build a changé, et Caddy fait patienter les visiteurs
jusqu'à 10 s pendant son redémarrage). Les données de la VM (PDF, sauvegardes,
`.env`, et l'ancien `front/` statique) ne sont pas touchées ; la base,
seulement par les migrations, sauf avec `--restaurer` (§6). Le code part en
`tar.gz`, extrait à part sur la VM : un envoi coupé échoue sans rien effacer.

`demarrer.sh` tient le journal `sauvegardes/deploiements.log` sur la VM, qu'il
soit lancé par `deployer.sh` ou à la main : une ligne `tentative` juste avant de
remplacer l'API et le front, une ligne `ok` une fois le rendu contrôlé.

```
<date> <tentative|ok> <commit du backend> <commit du front> <sauvegarde d'avant> <révision d'avant>
```

- **Sauvegarde d'avant** : la base telle qu'elle était avant ce passage, chemin
  prêt pour `JURIX_RESTAURER` ou `--restaurer`. `-` quand ce passage n'avait
  rien à sauvegarder : premier déploiement (le dump restauré est gardé en
  `initial.restaure_….dump`), ou reprise d'une restauration interrompue.
- **Révision d'avant** : la révision alembic de cette même base (`-` : pas de
  table `alembic_version`, ou reprise d'une restauration).

La **version en place** est la dernière ligne, `tentative` ou `ok` : après un
`ECHEC` aux étapes 5 à 7, c'est bien la nouvelle version qui tourne. Une ligne
`secours` note l'échange de front d'urgence (plus bas). `front-node/VERSION` ne
décrit que le dernier front envoyé.

Le premier déploiement de cette version recrée une fois le conteneur de la base
(rotation des journaux, délai d'arrêt porté à une minute), comme ceux de l'API
et de Caddy : quelques secondes de coupure, les données restant dans le volume
`db`.

`demarrer.sh` refuse de démarrer, avant de toucher à la base : si un autre
`demarrer.sh` ou une sauvegarde tourne (verrou `sauvegardes/.verrou`) ; un
`front-node/` sans `VERSION` (envoi incomplet), d'un autre envoi que le code
(le `BACKEND_VERSION` que `deployer.sh` ajoute à l'archive doit valoir le
premier champ de `front-node/VERSION`), ou construit pour un autre domaine que
le `JURIX_DOMAINE` de son `.env` ; un `JURIX_DOMAINE` ou un `GOOGLE_CLIENT_ID`
avec guillemets, espaces, `$`, `#` ou `/` ; un `JURIX_RESTAURER` qui n'est pas
un fichier `sauvegardes/….dump` existant ; ou une VM sans
`caddy/maintenance.html`. Il
contrôle enfin le rendu serveur, depuis le conteneur du front : `/laws` doit
lier des fiches, sans son bloc d'erreur, et une de ces fiches répondre 200 avec
un `<title>`. Sinon, il s'arrête en `ECHEC` : voir `docker compose logs front
api`.

### Revenir en arrière

Revenir en arrière, c'est redéployer un état précédent : un commit du backend
**et** le commit du front qui allait avec, lus dans
`sauvegardes/deploiements.log` sur la VM. La version en place, N, est sa
dernière ligne ; N-1 est la dernière ligne `ok` d'un **autre** commit du
backend. Les sauvegardes qu'il nomme sont gardées 14 jours, comme toutes (§6).

Si N n'a **pas** appliqué de migration :

1. Sur la machine de développement, ramener le backend au commit de N-1
   (`git switch --detach <commit>`), et le front au sien
   (`git switch --detach <commit>`, puis `npm ci`).
2. Relancer `deployer.sh` : il renvoie ce code, reconstruit `front-node/` depuis
   ce front, et `demarrer.sh` reconstruit l'image et contrôle le rendu.
3. Revenir ensuite sur les branches de travail : `git switch -` dans les deux
   dépôts, puis `npm ci` dans le front, dont `deployer.sh` a remplacé les
   `node_modules` par ceux du commit déployé.

Si N **a appliqué une migration**, la base doit revenir aussi. La tête de
N-1 : `.venv/bin/alembic heads` au commit N-1, sur la machine de développement.

0. Si la migration de N a un `downgrade` qui fonctionne, sur la VM :
   1. `docker compose stop api front` : plus rien n'écrit dans la base, et
      Caddy sert `maintenance.html` (arrêtés ainsi, les conteneurs le restent
      même si Docker redémarre) ;
   2. `docker compose run --rm --no-deps api alembic downgrade <tête de N-1>`,
      avec l'image de N, qui porte ce `downgrade` ;
   3. déployer N-1 comme ci-dessus. Pas de restauration.
1. Sinon, restaurer. La sauvegarde à prendre est la plus récente du journal
   dont la révision d'avant vaut la tête de N-1. Déployer N-1 comme ci-dessus,
   mais avec `--restaurer` (le `deployer.sh` de N-1 doit le connaître : à
   partir de cette version du déploiement) :

   ```bash
   deploiement/oracle/deployer.sh ubuntu@$IP --restaurer sauvegardes/jurix_….dump
   ```

   Le `demarrer.sh` de N-1 sauvegarde d'abord la base de N (les comptes et
   conversations créés depuis N restent dans ce fichier), la remplace par ce
   dump, puis démarre l'API de N-1.

Ne jamais restaurer une base d'avant N avec le code de N en place
(`JURIX_RESTAURER=… bash demarrer.sh` sur la VM) : l'API de N réappliquerait
sa migration. Le retour passe par le `deployer.sh` de N-1.

En urgence, le front seul, sans la machine de développement :
`front-node.ancien/` garde le dernier build complet d'un autre commit du front.
Valable seulement à partir du deuxième déploiement avec le rendu serveur (il
n'existe pas avant), et seulement pour un front construit pour le **backend en
place** (`BACKEND_VERSION`) ; sinon, redéployer en entier, ou revenir au site
statique. Sur la VM, la commande se relance sans risque : déjà en place, le
front voulu est seulement reconstruit, sans nouvelle ligne au journal.

```bash
cible=<commit du front voulu>   # tel que dans VERSION (12 caractères)
if [ "$(cut -d' ' -f2 front-node/VERSION)" = "$cible" ]; then
    docker compose up -d --build --no-deps front
elif [ -f front-node.ancien/build/index.js ] \
        && [ "$(cut -d' ' -f2 front-node.ancien/VERSION 2>/dev/null)" = "$cible" ] \
        && [ "$(cut -d' ' -f1 front-node.ancien/VERSION)" = "$(cat BACKEND_VERSION)" ]; then
    mv front-node front-node.tmp && mv front-node.ancien front-node && mv front-node.tmp front-node.ancien \
      && printf '%s secours %s %s - -\n' "$(date -Is)" "$(cat BACKEND_VERSION)" "$cible" >> sauvegardes/deploiements.log \
      && docker compose up -d --build --no-deps front \
      || echo "Echange inacheve. Si front-node.tmp existe, le remettre : mv front-node.tmp front-node (ou, si front-node existe deja, mv front-node.tmp front-node.ancien), puis relancer."
else
    echo "Refuse : front-node.ancien ne porte pas le front $cible pour le backend en place ($(cat BACKEND_VERSION))"
fi
```

Revenir au **site statique** d'avant le rendu serveur :

1. Sur la machine de développement, ramener le backend au commit d'avant la
   bascule, et le front au sien ; dans le front, `npm ci`, puis s'assurer que
   `VITE_API_URL=https://<domaine> npm run build` réussit **avant** de lancer
   l'ancien `deployer.sh`. Celui-ci remplace en effet sur la VM le
   `docker-compose.yml` et le Caddyfile, et efface `front-node/` (et ses
   copies) et `caddy/maintenance.html`, qu'il ne connaît pas, **avant** de
   construire le front : un build en échec à ce moment laisserait la VM à
   moitié revenue.
2. Lancer l'ancien `deployer.sh` : il reconstruit `front/` et l'envoie.
3. Sur la VM, retirer le conteneur du front, que l'ancienne composition ignore :
   `docker compose up -d --remove-orphans`.
4. Revenir sur les branches de travail : `git switch -` dans les deux dépôts,
   puis `npm ci` dans le front.

L'ancien `front/` est resté intact sur une VM déployée avant la bascule ; une
VM déployée d'emblée avec le rendu serveur n'en a pas, et c'est l'ancien
`deployer.sh` qui le crée.

### Une VM déployée avant la bascule vers Voyage

La migration `a8b9c0d1e2f3` vide les vecteurs EmbeddingGemma de la base
(incompatibles avec Voyage). `demarrer.sh` refuse donc de lancer une base qui
en porte encore, sauf demande explicite. Dans l'ordre :

1. Sur la VM, dans `.env` : ajouter `VOYAGE_API_KEY`, retirer
   `EMBEDDING_PROVIDER=gemma` et les `GEMMA_*` (`demarrer.sh` refuse sinon).
2. Depuis le poste : `JURIX_PURGER_VECTEURS=1 JURIX_DOMAINE=… deploiement/oracle/deployer.sh ubuntu@$IP`
   (la variable est transmise à `demarrer.sh`).
3. Sur la VM, recalculer les vecteurs par l'API Voyage (environ 10 M de jetons,
   quelques minutes), puis reconstruire l'index :
   `docker compose exec api python scripts/regenerate_embeddings.py --all`, puis
   `… --reindex`.
4. Vérifier `/api/v1/search/health`, puis supprimer l'ancien modèle :
   `sudo rm -rf modeles/`.

## 6. Sauvegardes

Chaque nuit à 03:15, et avant chaque mise à jour ou restauration,
`sauvegarde.sh` écrit `sauvegardes/jurix_AAAAMMJJ_HHMMSS.dump`, gardé 14 jours.
Il ne tourne jamais en même temps que `demarrer.sh` (verrou
`sauvegardes/.verrou`), ni pendant une restauration interrompue : la sauvegarde
de nuit est alors sautée, avec une ligne dans `sauvegarde.log`. Rien ne prévient d'un
échec : jeter un œil de temps en temps à `sauvegardes/sauvegarde.log`. À la main :

```bash
bash sauvegarde.sh
```

Restaurer une sauvegarde **à version égale** (la base actuelle est
**remplacée**) ; pour revenir à une version antérieure, voir « Revenir en
arrière » (§5) :

```bash
JURIX_RESTAURER=sauvegardes/jurix_….dump bash demarrer.sh
```

La demande ne vaut que pour ce lancement : rien n'est posé à l'avance, rien
n'est à annuler. `demarrer.sh` sauvegarde d'abord la base en place si elle a
des lois (son nom va au journal, comme sauvegarde d'avant), arrête l'API, le
front et Caddy, écrit lui-même le témoin `sauvegardes/.restauration_en_cours`
(le chemin du dump) juste avant de jeter la base, la restaure, puis redémarre
tout. Interrompue (coupure, mémoire), la restauration reprend de zéro au
prochain `bash demarrer.sh`, sans `JURIX_RESTAURER` ; d'ici là, `deployer.sh`
et `sauvegarde.sh` refusent de tourner. Le témoin ne se pose ni ne se retire
jamais à la main.

Copier de temps en temps une sauvegarde hors de la VM :
`scp -i ~/.ssh/jurix_oracle ubuntu@$IP:jurix/deploiement/oracle/sauvegardes/<fichier> .`

## 7. Surveiller et dépanner

```bash
docker compose ps                    # état des quatre conteneurs
docker compose logs -f api           # journal de l'API (une ligne par appel Mistral/Groq)
docker compose logs front            # rendu serveur du front
docker compose logs caddy            # certificat HTTPS, accès
cat front-node/VERSION               # dernier front envoyé (commits, date d'envoi)
tail -3 sauvegardes/deploiements.log # la dernière ligne : la version en place
free -h ; df -h                      # mémoire et disque
```

Les journaux de chaque conteneur tournent : 3 fichiers de 10 Mo au plus.

- **Page « JuriX revient dans un instant »** : Caddy ne joint plus le front
  (ou l'API, pour `/api`). `docker compose ps`, puis `docker compose logs
  front` ; `docker compose up -d` relance un conteneur arrêté.
- **Page d'erreur 503 aux couleurs du site** : le front tourne, mais l'API ne
  lui répond pas (`docker compose logs api`).
- **Pas de certificat** : le DNS ne pointe pas encore sur la VM, ou le port 80
  est fermé (liste de sécurité Oracle **et** pare-feu de la VM, que
  `installer.sh` ouvre).
- **La VM s'arrête seule** : Oracle récupère les VM gratuites « inactives » (sur
  7 jours, processeur, réseau ET mémoire sous 20 %). Sans le modèle
  d'embeddings, qui occupait ~1,5 Go en permanence, la mémoire utilisée peut
  passer sous ce seuil : surveiller la métrique mémoire dans la console Oracle
  la première semaine.
- **Déménager** : tout est dans Docker. Sur la nouvelle machine : y copier le
  dossier `jurix/` entier (le code et ce dossier, avec `.env`, `donnees/` et
  `front-node/`), placer la dernière sauvegarde sous `sauvegardes/initial.dump`,
  lancer `installer.sh`, **se reconnecter** (le groupe `docker` ne vaut que
  pour une nouvelle session), puis `bash demarrer.sh` : la base, vide, est
  restaurée avant le démarrage de l'API. Enfin, faire pointer le DNS sur la
  nouvelle IP.
