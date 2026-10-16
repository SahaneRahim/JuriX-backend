# Déploiement de JuriX sur Oracle Cloud (palier gratuit)

Tout JuriX tourne sur **une seule machine virtuelle** Oracle Cloud du palier
« Always Free » : une VM ARM de 2 cœurs et 12 Go de mémoire, avec 200 Go de
disque, gratuite tant qu'on reste dans ces limites.

```
navigateur ──https──> Caddy (ports 80/443, certificat automatique)
                        ├── /api/*, /health ──> api (FastAPI) ──https──> Voyage AI (embeddings)
                        │                          └──> db (PostgreSQL 16 + pgvector)
                        └── tout le reste ──────> front statique (SvelteKit)
```

- **Un seul domaine** — le sous-domaine, par exemple `jurix.exemple.cm` — sert
  le front et l'API (sous `/api/v1`). Pas de CORS, un seul certificat.
- **Aucun port de la base n'est exposé** : seule l'API la voit.
- Mistral, Groq et Voyage AI (les embeddings de la recherche sémantique)
  restent des API externes ; rien ne tourne pour eux ici. L'API a donc besoin
  du HTTPS sortant, ouvert par défaut.
- L'ingestion de nouveaux PDF (Docling, OCR) se fait sur la machine de
  développement, pas sur la VM.

Besoins : ~2 à 3 Go pour PostgreSQL, et bien moins pour l'API depuis qu'elle ne
garde plus de modèle d'embeddings en mémoire (EmbeddingGemma prenait ~1,5 Go) :
à mesurer par `docker stats` après le premier déploiement.

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
export JURIX_DOMAINE=jurix.exemple.cm

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

Prérequis sur la machine de développement : le dépôt du front à côté de celui-ci
(ou `JURIX_FRONT`), avec ses `node_modules` ; le conteneur `jurix-pg` démarré ;
`jurix_dev` à la dernière migration, ses vecteurs Voyage chargés
(`deployer.sh` refuse de l'envoyer sinon : la migration viderait les vecteurs
sur la VM, qui n'a pas de quoi les recharger sans tout recalculer).

Une clé changée dans `.env` ne prend effet qu'avec `docker compose up -d api`,
qui recrée le conteneur : `docker compose restart` ne relit pas le fichier.

Connexion Google (facultative) : `GOOGLE_CLIENT_ID` dans le `.env` de la VM
(`deployer.sh` le reprend pour construire le front), et, dans la console Google
Cloud, `https://jurix.exemple.cm` ajouté aux **origines JavaScript autorisées**
de cet identifiant. Sans cela, le bouton Google échoue.

La base de la VM est restaurée depuis un dump de `jurix_dev`, **seulement si
elle ne contient aucune loi**, et **sans les comptes ni les conversations** de
développement : le corpus seulement. Une fois restauré, le dump est renommé
`initial.restaure_….dump` : aucune mise à jour ne pourra plus le rejouer. Une
restauration interrompue est reprise de zéro à la relance. Le compte
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

## 5. Mettre à jour

```bash
JURIX_DOMAINE=jurix.exemple.cm deploiement/oracle/deployer.sh ubuntu@$IP
```

Envoie l'état **commité** du backend (`git archive` : jamais un `.env` local),
et efface sur la VM les fichiers retirés du dépôt. Reconstruit le front, fait
une sauvegarde de la base, puis relance avec `docker compose up -d --build` (les
migrations s'appliquent au démarrage de l'API). Les données de la VM (PDF,
sauvegardes, `.env`) ne sont pas touchées ; la base, seulement par les
migrations.

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

Chaque nuit à 03:15, et avant chaque mise à jour, `sauvegarde.sh` écrit
`sauvegardes/jurix_AAAAMMJJ_HHMM.dump`, gardé 14 jours. Rien ne prévient d'un
échec : jeter un œil de temps en temps à `sauvegardes/sauvegarde.log`. À la main :

```bash
bash sauvegarde.sh
```

Restaurer une sauvegarde (la base actuelle est **remplacée**) :

```bash
docker compose stop api caddy
docker compose exec -T db psql -U jurix -d postgres -c "DROP DATABASE jurix WITH (FORCE)" -c "CREATE DATABASE jurix OWNER jurix"
docker compose exec -T db pg_restore -U jurix -d jurix --no-owner --exit-on-error < sauvegardes/jurix_….dump
bash demarrer.sh
```

Copier de temps en temps une sauvegarde hors de la VM :
`scp -i ~/.ssh/jurix_oracle ubuntu@$IP:jurix/deploiement/oracle/sauvegardes/<fichier> .`

## 7. Surveiller et dépanner

```bash
docker compose ps                    # état des trois conteneurs
docker compose logs -f api           # journal de l'API (une ligne par appel Mistral/Groq)
docker compose logs caddy            # certificat HTTPS, accès
free -h ; df -h                      # mémoire et disque
```

- **Pas de certificat** : le DNS ne pointe pas encore sur la VM, ou le port 80
  est fermé (liste de sécurité Oracle **et** pare-feu de la VM, que
  `installer.sh` ouvre).
- **La VM s'arrête seule** : Oracle récupère les VM gratuites « inactives » (sur
  7 jours, processeur, réseau ET mémoire sous 20 %). Sans le modèle
  d'embeddings, qui occupait ~1,5 Go en permanence, la mémoire utilisée peut
  passer sous ce seuil : surveiller la métrique mémoire dans la console Oracle
  la première semaine.
- **Déménager** : tout est dans Docker. Sur la nouvelle machine : y copier le
  dossier `jurix/` entier (le code et ce dossier, avec `.env` et
  `donnees/`), placer la dernière sauvegarde sous `sauvegardes/initial.dump`,
  lancer `installer.sh`, **se reconnecter** (le groupe `docker` ne vaut que
  pour une nouvelle session), puis `bash demarrer.sh` : la base, vide, est
  restaurée avant le démarrage de l'API. Enfin, faire pointer le DNS sur la
  nouvelle IP.
