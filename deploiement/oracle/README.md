# Déploiement de JuriX sur Oracle Cloud (palier gratuit)

Tout JuriX tourne sur **une seule machine virtuelle** Oracle Cloud du palier
« Always Free » : une VM ARM de 2 cœurs et 12 Go de mémoire, avec 200 Go de
disque, gratuite tant qu'on reste dans ces limites.

```
navigateur ──https──> Caddy (ports 80/443, certificat automatique)
                        ├── /api/*, /health ──> api (FastAPI + EmbeddingGemma)
                        │                          └──> db (PostgreSQL 16 + pgvector)
                        └── tout le reste ──────> front statique (SvelteKit)
```

- **Un seul domaine** — le sous-domaine, par exemple `jurix.exemple.cm` — sert
  le front et l'API (sous `/api/v1`). Pas de CORS, un seul certificat.
- **Aucun port de la base n'est exposé** : seule l'API la voit.
- Mistral et Groq restent des API externes ; rien ne tourne pour eux ici.
- L'ingestion de nouveaux PDF (Docling, OCR) se fait sur la machine de
  développement, pas sur la VM.

Besoins mesurés : ~1,5 à 2 Go pour l'API (le modèle d'embeddings est en
mémoire), ~2 à 3 Go pour PostgreSQL, soit 4 à 5 Go sur 12.

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

# b. Les secrets, SUR la VM uniquement
ssh -i ~/.ssh/jurix_oracle ubuntu@$IP
cd jurix/deploiement/oracle && cp env.production.exemple .env && nano .env
exit

# c. Tout le reste : front, modèle, PDF (5,6 Go, reprenable), base, démarrage
deploiement/oracle/deployer.sh ubuntu@$IP --avec-donnees
```

La base de la VM est restaurée depuis un dump de `jurix_dev`, seulement si elle
est vide ; l'API applique ensuite les migrations au démarrage. Le certificat
HTTPS est obtenu dès que le DNS pointe sur la VM.

Vérifier : `https://jurix.exemple.cm` et `https://jurix.exemple.cm/health`.

## 5. Mettre à jour

```bash
JURIX_DOMAINE=jurix.exemple.cm deploiement/oracle/deployer.sh ubuntu@$IP
```

Envoie l'état **commité** du backend (`git archive` : jamais un `.env` local),
reconstruit le front, et relance avec `docker compose up -d --build`. Les
données de la VM (base, PDF, sauvegardes, `.env`) ne sont pas touchées.

## 6. Sauvegardes

Chaque nuit à 03:15, `sauvegarde.sh` écrit `sauvegardes/jurix_AAAAMMJJ_HHMM.dump`,
gardé 14 jours. À la main :

```bash
bash sauvegarde.sh
# Restaurer (base vide) :
docker compose exec -T db pg_restore -U jurix -d jurix --no-owner < sauvegardes/jurix_….dump
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
  7 jours, processeur, réseau ET mémoire sous 20 %). JuriX garde en permanence
  le modèle et PostgreSQL en mémoire (30 à 40 %) : elle ne devrait pas être
  jugée inactive.
- **Déménager** : tout est dans Docker. Sur une autre machine : installer
  Docker, copier ce dossier, son `.env` et une sauvegarde, `docker compose up`.
