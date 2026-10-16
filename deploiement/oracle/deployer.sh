#!/usr/bin/env bash
# Deploie (ou met a jour) JuriX sur la VM, depuis la machine de developpement.
#
#   JURIX_DOMAINE=jurix.exemple.cm deploiement/oracle/deployer.sh ubuntu@IP
#   JURIX_DOMAINE=jurix.exemple.cm deploiement/oracle/deployer.sh ubuntu@IP --avec-donnees
#   JURIX_DOMAINE=jurix.exemple.cm deploiement/oracle/deployer.sh ubuntu@IP --restaurer sauvegardes/<fichier>.dump
#
# Sans option : le front construit pour le domaine, d'abord, puis le code
# COMMITE du backend (git archive, jamais les fichiers locaux, donc jamais un
# .env), et `docker compose up -d --build`.
# --avec-donnees, au premier deploiement : en plus, les PDF (rsync,
# reprenable) et la base (dump de jurix_dev, restaure seulement si la base de
# la VM est vide).
# --restaurer : demarrer.sh restaure ce dump de la VM (apres avoir sauvegarde
# la base en place) avant de demarrer la nouvelle API ; le retour en arriere
# par-dessus une migration (README §5).
#
# Prerequis :
# - sur la VM : installer.sh deja lance, et deploiement/oracle/.env renseigne
#   (voir env.production.exemple), avec le MEME JURIX_DOMAINE qu'ici ;
# - ici : deploiement/oracle tout commite ; le depot du front a cote de
#   celui-ci (ou JURIX_FRONT), sur la branche du rendu serveur (adapter-node),
#   tout commite (`npm ci` y remplace ses node_modules) ; pour --avec-donnees,
#   le conteneur local jurix-pg demarre, et jurix_dev a la derniere migration
#   du commit envoye, ses vecteurs Voyage charges
#   (scripts/regenerate_embeddings.py --depuis-lots ou --all).
# JURIX_GOOGLE_CLIENT_ID : l'identifiant OAuth Google (fige dans le front au
# build). Facultatif : a defaut, GOOGLE_CLIENT_ID du .env de la VM.
set -euo pipefail

CIBLE=${1:?usage : JURIX_DOMAINE=... deployer.sh utilisateur@ip [--avec-donnees | --restaurer sauvegardes/<fichier>.dump]}
shift
AVEC_DONNEES=
RESTAURER=
while [ "$#" -gt 0 ]; do
    case "$1" in
        --avec-donnees) AVEC_DONNEES=--avec-donnees ;;
        --restaurer) RESTAURER=${2:?--restaurer : donner sauvegardes/<fichier>.dump}; shift ;;
        *) echo "Option inconnue : $1"; exit 1 ;;
    esac
    shift
done
# Le chemin part dans la commande ssh de l'etape 8 : un nom de fichier sous
# sauvegardes/, rien d'autre. Et l'un ou l'autre : --avec-donnees remplace
# initial.dump, --restaurer en restaure un autre.
if [ -n "$RESTAURER" ]; then
    if [[ ! $RESTAURER =~ ^sauvegardes/[A-Za-z0-9._-]+\.dump$ ]]; then
        printf -- '--restaurer refuse : %q (attendu : sauvegardes/<fichier>.dump)\n' "$RESTAURER"
        exit 1
    fi
    if [ -n "$AVEC_DONNEES" ]; then
        echo "--avec-donnees et --restaurer ensemble : l'un ou l'autre"
        exit 1
    fi
fi
: "${JURIX_DOMAINE:?renseigner JURIX_DOMAINE (le domaine du site)}"
# Valeurs lues telles quelles, ici et dans le .env de la VM (grep, cut), alors
# que docker compose en retire guillemets et espaces : « "jurix.cm" » ne
# serait pas le meme domaine des deux cotes, et partirait tel quel dans le
# front. Refusees plutot que nettoyees : un seul texte fait foi.
refuser_si_brute() {
    case "$2" in
        *[[:space:]\"\'\$#/]*)
            printf '%s refuse : %q (ni guillemets, ni espaces, ni $, ni #, ni /)\n' "$1" "$2"
            exit 1 ;;
    esac
}
refuser_si_brute JURIX_DOMAINE "$JURIX_DOMAINE"
# Le domaine d'exemple, fige dans le front, enverrait le navigateur vers un
# domaine que n'importe qui peut acheter.
if [ "$JURIX_DOMAINE" = "jurix.exemple.cm" ]; then
    echo "JURIX_DOMAINE vaut jurix.exemple.cm, le domaine d'exemple : donner celui du site"
    exit 1
fi

RACINE=$(cd "$(dirname "$0")/../.." && pwd)
FRONT=${JURIX_FRONT:-$RACINE/../JuriX-frontend-main}
CLE=${JURIX_CLE_SSH:-$HOME/.ssh/jurix_oracle}
DISTANT=jurix/deploiement/oracle
SSH=(ssh -i "$CLE" -o StrictHostKeyChecking=accept-new)
RSYNC=(rsync -az --partial --info=progress2 -e "ssh -i $CLE -o StrictHostKeyChecking=accept-new")
# Les donnees appartiennent a l'utilisateur du conteneur (uid 1000), qui n'est
# pas forcement celui de la VM : l'envoi se fait en root, cote VM.
RSYNC_ROOT=("${RSYNC[@]}" --rsync-path="sudo rsync")

# jurix_dev doit etre a la derniere migration du commit ENVOYE, ses vecteurs
# tous a jour pour le modele de ce commit. Un dump d'avant la bascule serait
# restaure, puis migre au demarrage du conteneur : la migration a8b9c0d1e2f3
# viderait ses vecteurs EN PRODUCTION, et la VM n'a pas data/mesure_voyage
# pour les recharger. Un vecteur manquant y resterait manquant : rien, sur la
# VM, ne le calcule.
controler_jurix_dev() {
    # Le code du commit envoye ($backend_sha), extrait a part, et non la copie
    # de travail : une migration ajoutee depuis, commitee ou non, passerait
    # sinon pour attendue, et la VM recevrait un dump plus migre que son code.
    # Le dossier n'a pas de .env : la tete et l'empreinte sont celles des
    # reglages par defaut de ce code, ceux de la VM ; le venv local ne fournit
    # que les bibliotheques.
    # `|| true` : un echec laisse les variables vides, et le controle plus bas
    # le dit, au lieu d'un arret muet de `set -e`.
    local vide tete empreinte version etrangers manquants vectorises
    vide=$(mktemp -d)
    git -C "$RACINE" archive "$backend_sha" alembic alembic.ini app | tar -x -C "$vide" \
        || { rm -rf "$vide"; echo "Extraction d'alembic/ et d'app/ du commit $backend_sha impossible"; exit 1; }
    read -r tete empreinte < <(cd "$vide" && env -u VOYAGE_EMBEDDING_MODEL -u EMBEDDING_PROVIDER \
        PYTHONPATH="$vide" "$RACINE/.venv/bin/python" -c "
from alembic.config import Config
from alembic.script import ScriptDirectory
from app.core.config import settings
from app.services.embedding_service import empreinte_voyage
config = Config('$vide/alembic.ini')
config.set_main_option('script_location', '$vide/alembic')
print('TETE', ScriptDirectory.from_config(config).get_current_head(), empreinte_voyage(settings.VOYAGE_EMBEDDING_MODEL))" 2>/dev/null \
        | grep '^TETE ' | cut -d' ' -f2-) || true
    rm -rf "$vide"
    read -r version etrangers manquants vectorises < <(docker exec jurix-pg psql -U jurix -d jurix_dev -tA -F ' ' -c \
        "SELECT (SELECT version_num FROM alembic_version),
                count(*) FILTER (WHERE embedding IS NOT NULL AND embedding_model IS DISTINCT FROM '$empreinte'),
                count(*) FILTER (WHERE embed AND embedding IS NULL),
                count(*) FILTER (WHERE embedding IS NOT NULL)
         FROM articles") || true
    if [ -z "$tete" ] || [ -z "$empreinte" ] || [ "$version" != "$tete" ] \
            || [ "$etrangers" != "0" ] || [ "$manquants" != "0" ] || [ "${vectorises:-0}" = "0" ]; then
        echo "jurix_dev n'est pas pret pour le commit $backend_sha : migration ${version:-?} (attendue ${tete:-?}), ${etrangers:-?} vecteur(s) hors ${empreinte:-?}, ${manquants:-?} manquant(s), ${vectorises:-?} vecteur(s)"
        if [ -n "$tete" ] && [ "${version:-}" != "$tete" ]; then
            echo "  -> jurix_dev et le commit envoye doivent avoir la meme tete : commiter puis deployer la migration, ou ramener jurix_dev (alembic downgrade)"
        else
            echo "  -> scripts/regenerate_embeddings.py --all (puis --reindex) sur jurix_dev"
        fi
        exit 1
    fi
}

echo "== 1. Fichiers de deploiement commites, VM joignable, meme domaine"
# La VM recoit le code de HEAD (git archive), mais ce script se lit dans la
# copie de travail : un fichier de deploiement modifie partirait dans sa
# version de HEAD, et un fichier non suivi (une page, un script) pas du tout.
etat_deploiement=$(git -C "$RACINE" status --porcelain -- deploiement/oracle)
if [ -n "$etat_deploiement" ]; then
    echo "Modifications non commitees sous deploiement/oracle : commiter d'abord, la VM recoit HEAD"
    printf '%s\n' "$etat_deploiement"
    exit 1
fi
# Ailleurs dans le backend, un simple avertissement : seul HEAD part.
autres=$(git -C "$RACINE" status --porcelain | wc -l)
if [ "$autres" -gt 0 ]; then
    echo "ATTENTION : $autres fichier(s) modifie(s) ou non suivi(s) dans le backend : ils ne partent pas, seul HEAD est envoye"
fi
# Le commit envoye, releve une fois pour toutes, avant le premier controle de
# jurix_dev : l'archive (etape 5), VERSION et ces controles nomment le meme,
# meme si HEAD bouge pendant le deploiement.
backend_sha=$(git -C "$RACINE" rev-parse --short=12 HEAD)
echo "Backend : etat commite de $backend_sha"
# L'archive de l'etape 5 porte BACKEND_VERSION (--add-virtual-file, git 2.39
# au moins). Essaye ici, sur un seul fichier : un git trop ancien echouerait
# sinon a l'etape 5, le front deja echange sur la VM.
git -C "$RACINE" archive --add-virtual-file=essai:x "$backend_sha" -- deploiement/oracle/demarrer.sh >/dev/null 2>&1 \
    || { echo "$(git --version) : git archive --add-virtual-file manque (git 2.39 au moins)"; exit 1; }
# Le domaine fige dans le front (VITE_API_URL) doit etre celui de la VM, d'ou
# viennent ORIGIN et ALLOWED_ORIGINS : sinon les pages s'affichent, mais le
# navigateur envoie ses appels a l'API (recherche, connexion, chat) a un autre
# domaine. Lu dans le .env de la VM, ou il n'est pas un secret ; premier
# contact ssh, il verifie aussi que la VM repond, avant tout travail.
domaine_vm=$("${SSH[@]}" "$CIBLE" "grep -E '^JURIX_DOMAINE=' $DISTANT/.env 2>/dev/null | tail -1 | cut -d= -f2-") \
    || { echo "VM injoignable ($CIBLE) : ssh a echoue"; exit 1; }
if [ -z "$domaine_vm" ]; then
    echo "Pas de JURIX_DOMAINE dans $DISTANT/.env sur la VM : le renseigner d'abord (README §4)"
    exit 1
fi
refuser_si_brute "JURIX_DOMAINE du .env de la VM" "$domaine_vm"
if [ "$domaine_vm" != "$JURIX_DOMAINE" ]; then
    printf 'JURIX_DOMAINE differe : %q ici, %q dans le .env de la VM\n' "$JURIX_DOMAINE" "$domaine_vm"
    exit 1
fi
# Une restauration coupee sur la VM se termine avec le code qui l'a
# commencee : envoyer un autre code d'abord, c'est la reprendre avec lui, ou
# l'abandonner sur une base jetee.
temoin_vm=$("${SSH[@]}" "$CIBLE" "if [ -e $DISTANT/sauvegardes/.restauration_en_cours ]; then echo present; fi") \
    || { echo "VM injoignable ($CIBLE) : ssh a echoue"; exit 1; }
if [ -n "$temoin_vm" ]; then
    echo "Restauration interrompue sur la VM : la terminer d'abord (bash demarrer.sh sur la VM)"
    exit 1
fi
if [ -n "$RESTAURER" ]; then
    "${SSH[@]}" "$CIBLE" "test -f $DISTANT/$RESTAURER" \
        || { echo "--restaurer : $RESTAURER introuvable sur la VM (ou ssh en echec)"; exit 1; }
fi

if [ "$AVEC_DONNEES" = "--avec-donnees" ]; then
    echo "== 2. jurix_dev : derniere migration, vecteurs a jour (avant tout envoi)"
    controler_jurix_dev
fi

echo "== 3. Front, construit pour https://$JURIX_DOMAINE (avant tout envoi)"
# Construit et controle avant que rien ne parte vers la VM : un build en echec
# apres l'envoi du code y laisserait une composition qui attend un front-node/
# absent ou perime.
# L'etat commite seulement, comme pour le backend : un build fait d'une copie
# de travail modifiee ne correspondrait a aucun commit, et on ne saurait plus
# le refaire pour revenir en arriere.
etat_front=$(git -C "$FRONT" status --porcelain) \
    || { echo "$FRONT n'est pas un depot git : JURIX_FRONT doit designer le depot du front"; exit 1; }
if [ -n "$etat_front" ]; then
    echo "Modifications non commitees dans le front ($FRONT) : commiter d'abord, le front envoye doit etre un commit"
    printf '%s\n' "$etat_front"
    exit 1
fi
front_sha=$(git -C "$FRONT" rev-parse --short=12 HEAD)
echo "Front : etat commite de $front_sha"
# L'image du front n'a pas de node_modules : adapter-node integre au build les
# devDependencies, mais laisse les "dependencies" en import externe, que
# `node build` ne trouverait pas (arret au demarrage, en boucle).
dependances=$(node -e '
const p = JSON.parse(require("fs").readFileSync(process.argv[1], "utf8"));
console.log(Object.keys(p.dependencies || {}).join(" "));' "$FRONT/package.json")
if [ -n "$dependances" ]; then
    echo "\"dependencies\" dans le package.json du front ($dependances) : l'image n'a pas de node_modules, les passer en devDependencies"
    exit 1
fi
# Les versions du package-lock.json commite, et non celles qu'a laissees le
# developpement : le build doit sortir de ce commit, et de lui seul.
( cd "$FRONT" && npm ci --no-audit --no-fund >/dev/null )
# L'identifiant Google est fige dans le front a la construction. Sans
# JURIX_GOOGLE_CLIENT_ID ici, il est repris du .env de la VM : sinon chaque
# mise a jour faite sans la variable retirait la connexion Google, en silence.
# Un ssh en echec arrete donc le deploiement ; une cle absente, elle, rend une
# chaine vide.
GOOGLE_ID=${JURIX_GOOGLE_CLIENT_ID:-$("${SSH[@]}" "$CIBLE" \
    "grep -E '^GOOGLE_CLIENT_ID=' $DISTANT/.env 2>/dev/null | tail -1 | cut -d= -f2-")} \
    || { echo "Lecture de GOOGLE_CLIENT_ID sur la VM en echec (ssh) : arret"; exit 1; }
refuser_si_brute "GOOGLE_CLIENT_ID (ou JURIX_GOOGLE_CLIENT_ID)" "$GOOGLE_ID"
echo "Connexion Google : $([ -n "$GOOGLE_ID" ] && echo activee || echo absente)"
# VITE_WS_URL impose vide : une valeur du .env local du front (hors de git)
# serait figee dans le build ; vide, le front la derive de VITE_API_URL.
( cd "$FRONT" && VITE_API_URL="https://$JURIX_DOMAINE" VITE_WS_URL='' \
    VITE_GOOGLE_CLIENT_ID="$GOOGLE_ID" npm run build >/dev/null )
# Le build dure : un commit ou une modification faite entre-temps dans le
# front ferait partir un build qui ne correspond plus au commit enregistre.
if [ -n "$(git -C "$FRONT" status --porcelain)" ] \
        || [ "$(git -C "$FRONT" rev-parse --short=12 HEAD)" != "$front_sha" ]; then
    echo "Le front a change pendant la construction : relancer"
    exit 1
fi
# Un build d'adapter-node : un build statique (adapter-static) n'a ni serveur
# ni build/client, et `node build` n'aurait rien a lancer.
if [ ! -f "$FRONT/build/index.js" ] || [ ! -d "$FRONT/build/client" ]; then
    echo "build/index.js ou build/client manquant : le front n'est pas sur la branche du rendu serveur (adapter-node)"
    exit 1
fi
# VITE_API_URL est figee dans le JavaScript du navigateur : sans le domaine
# dedans, ses appels a l'API partiraient ailleurs (localhost par defaut).
if ! grep -rqF -- "https://$JURIX_DOMAINE" "$FRONT/build/client/_app"; then
    echo "https://$JURIX_DOMAINE absent de build/client/_app : VITE_API_URL n'a pas ete figee dans le front"
    exit 1
fi

echo "== 4. Front envoye dans front-node/ (build/ et package.json seulement)"
# Seuls ces deux-la servent a l'execution. Envoyes dans front-node.nouveau/,
# copie de l'actuel (rsync n'envoie que les differences), puis echanges d'un
# coup : en place, un envoi coupe laisserait un front-node/ moitie neuf, moitie
# ancien, que demarrer.sh construirait. VERSION, ecrit en dernier, marque
# un envoi complet. Le conteneur en place n'en voit rien : l'image COPIE ces
# fichiers, et seul demarrer.sh la reconstruit. L'ancien front/ statique n'est
# plus touche.
horodatage=$(date -Is)
"${SSH[@]}" "$CIBLE" "set -e; cd $DISTANT
    rm -rf front-node.nouveau
    if [ -d front-node ]; then cp -a front-node front-node.nouveau; else mkdir front-node.nouveau; fi
    rm -f front-node.nouveau/VERSION"
"${RSYNC[@]}" --delete "$FRONT/build" "$FRONT/package.json" "$CIBLE:$DISTANT/front-node.nouveau/"
# Script passe sur l'entree standard, entre apostrophes : rien n'y est
# developpe ici, les valeurs arrivent en arguments. Aucune de ses commandes ne
# lit l'entree standard (le piege decrit a l'etape 8).
# front-node.ancien/ garde le dernier build COMPLET d'un AUTRE front : un
# front-node/ sans VERSION (envoi coupe, dossier vide d'installer.sh) est jete,
# et la relance d'un meme front n'evince pas le precedent, seul secours.
"${SSH[@]}" "$CIBLE" "bash -s -- $DISTANT $backend_sha $front_sha $horodatage" <<'FIN'
set -e
cd "$1"
printf '%s %s %s\n' "$2" "$3" "$4" > front-node.nouveau/VERSION
if [ -f front-node/VERSION ] && [ "$(cut -d' ' -f2 front-node/VERSION)" != "$3" ]; then
    rm -rf front-node.ancien
    mv front-node front-node.ancien
else
    rm -rf front-node
fi
mv front-node.nouveau front-node
FIN

echo "== 5. Code du backend (etat commite de $backend_sha)"
# Extrait a cote, puis aligne avec rsync --delete : un `tar -x` direct n'effacait
# rien, et un fichier supprime du depot restait sur la VM, jusque dans l'image.
# Les donnees de la VM (.env, PDF, sauvegardes, les front-node*, l'ancien front
# statique, et l'ancien modele EmbeddingGemma d'une VM d'avant la bascule, que
# son API en place monte encore) sont exclues : rsync ne les touche pas.
# Le commit releve a l'etape 1, et non HEAD relu ici : VERSION, le journal des
# deploiements et le controle de jurix_dev doivent nommer ce qui part vraiment.
# BACKEND_VERSION, ajoute a l'archive, le dit a demarrer.sh, qui le compare a
# front-node/VERSION.
# En tar.gz : un flux coupe en route fait echouer gzip (somme de controle,
# longueur), alors qu'un tar coupe entre deux fichiers pouvait passer pour
# complet, et rsync --delete aurait efface ce qui manquait. Extrait dans un
# dossier propre a ce passage, retire dans tous les cas. Entre apostrophes :
# $nouveau se developpe sur la VM, pas ici.
# shellcheck disable=SC2016
git -C "$RACINE" archive --format=tar.gz \
    --add-virtual-file="deploiement/oracle/BACKEND_VERSION:$backend_sha" "$backend_sha" \
    | "${SSH[@]}" "$CIBLE" 'set -e
    nouveau=$(mktemp -d jurix.nouveau.XXXXXX)
    tar -xz -C "$nouveau" || { rm -rf "$nouveau"; exit 1; }
    mkdir -p jurix
    rsync -a --delete \
        --exclude=/deploiement/oracle/.env \
        --exclude=/deploiement/oracle/modeles/ \
        --exclude=/deploiement/oracle/donnees/ \
        --exclude=/deploiement/oracle/sauvegardes/ \
        --exclude=/deploiement/oracle/front/ \
        --exclude=/deploiement/oracle/front-node/ \
        --exclude=/deploiement/oracle/front-node.ancien/ \
        --exclude=/deploiement/oracle/front-node.nouveau/ \
        "$nouveau"/ jurix/ || { rm -rf "$nouveau"; exit 1; }
    rm -rf "$nouveau"'

if [ "$AVEC_DONNEES" = "--avec-donnees" ]; then
    echo "== 6. PDF d'origine (reprenable : relancer en cas de coupure)"
    "${RSYNC_ROOT[@]}" "$RACINE/data/uploads/" "$CIBLE:$DISTANT/donnees/uploads/"

    echo "== 7. Base : dump de jurix_dev, SANS les comptes ni les conversations"
    # Le meme controle qu'a l'etape 2, refait, toujours contre le commit
    # envoye : l'envoi des PDF dure des heures, et une migration appliquee a
    # jurix_dev entre-temps doit arreter ici, avant le dump.
    controler_jurix_dev
    # Le corpus seulement. Les comptes de developpement (et leurs mots de
    # passe d'essai), leurs conversations, jetons et caches resteraient sinon
    # joignables depuis internet. Les tables gardent leur schema ; demarrer.sh
    # cree l'administrateur de production.
    dump=$(mktemp --suffix=.dump)
    docker exec jurix-pg pg_dump -U jurix -Fc --no-owner --no-privileges \
        --exclude-table-data=users --exclude-table-data=email_tokens \
        --exclude-table-data=conversations --exclude-table-data=messages \
        --exclude-table-data=message_feedback --exclude-table-data=persona_interactions \
        --exclude-table-data=persona_stats --exclude-table-data=search_events \
        --exclude-table-data=query_cache --exclude-table-data=embedding_cache \
        jurix_dev > "$dump"
    "${RSYNC[@]}" "$dump" "$CIBLE:$DISTANT/sauvegardes/initial.dump"
    rm -f "$dump"
fi

echo "== 8. Demarrage sur la VM (demarrer.sh)"
# Un FICHIER lance par ssh, et non un script passe sur l'entree standard : un
# `docker compose exec` y avalait la suite, et rien apres lui ne s'executait.
# JURIX_PURGER_VECTEURS et JURIX_RESTAURER : seule la VM les lit (demarrer.sh),
# ssh ne transmet pas l'environnement local. demarrer.sh ecrit lui-meme les
# lignes du journal des deploiements : une relance a la main y figure aussi.
"${SSH[@]}" "$CIBLE" "sudo chown -R 1000:1000 $DISTANT/donnees && JURIX_PURGER_VECTEURS=${JURIX_PURGER_VECTEURS:-} JURIX_RESTAURER=$RESTAURER bash $DISTANT/demarrer.sh"

echo
echo "Verifier : https://$JURIX_DOMAINE  et  https://$JURIX_DOMAINE/health"
