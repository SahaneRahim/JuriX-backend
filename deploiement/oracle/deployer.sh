#!/usr/bin/env bash
# Deploie (ou met a jour) JuriX sur la VM, depuis la machine de developpement.
#
#   JURIX_DOMAINE=jurix.exemple.cm deploiement/oracle/deployer.sh ubuntu@IP
#   JURIX_DOMAINE=jurix.exemple.cm deploiement/oracle/deployer.sh ubuntu@IP --avec-donnees
#
# Sans option : le code COMMITE du backend (git archive, jamais les fichiers
# locaux, donc jamais un .env), le front reconstruit pour le domaine, puis
# `docker compose up -d --build`.
# --avec-donnees, au premier deploiement : en plus, les PDF (rsync,
# reprenable) et la base (dump de jurix_dev, restaure seulement si la base de
# la VM est vide).
#
# Prerequis :
# - sur la VM : installer.sh deja lance, et deploiement/oracle/.env renseigne
#   (voir env.production.exemple) ;
# - ici : le depot du front a cote de celui-ci (ou JURIX_FRONT), sur la branche
#   du site statique, avec ses node_modules ; pour --avec-donnees, le conteneur
#   local jurix-pg demarre, et jurix_dev a la derniere migration, ses vecteurs
#   Voyage charges (scripts/regenerate_embeddings.py --depuis-lots ou --all).
# JURIX_GOOGLE_CLIENT_ID : l'identifiant OAuth Google (fige dans le front au
# build). Facultatif : a defaut, GOOGLE_CLIENT_ID du .env de la VM.
set -euo pipefail

CIBLE=${1:?usage : JURIX_DOMAINE=... deployer.sh utilisateur@ip [--avec-donnees]}
AVEC_DONNEES=${2:-}
: "${JURIX_DOMAINE:?renseigner JURIX_DOMAINE (le domaine du site)}"

RACINE=$(cd "$(dirname "$0")/../.." && pwd)
FRONT=${JURIX_FRONT:-$RACINE/../JuriX-frontend-main}
CLE=${JURIX_CLE_SSH:-$HOME/.ssh/jurix_oracle}
DISTANT=jurix/deploiement/oracle
SSH=(ssh -i "$CLE" -o StrictHostKeyChecking=accept-new)
RSYNC=(rsync -az --partial --info=progress2 -e "ssh -i $CLE -o StrictHostKeyChecking=accept-new")
# Les donnees appartiennent a l'utilisateur du conteneur (uid 1000), qui n'est
# pas forcement celui de la VM : l'envoi se fait en root, cote VM.
RSYNC_ROOT=("${RSYNC[@]}" --rsync-path="sudo rsync")

# jurix_dev doit etre a la derniere migration du code COMMITE, ses vecteurs
# tous a jour pour le modele du code commite. Un dump d'avant la bascule serait
# restaure, puis migre au demarrage du conteneur : la migration a8b9c0d1e2f3
# viderait ses vecteurs EN PRODUCTION, et la VM n'a pas data/mesure_voyage
# pour les recharger. Un vecteur manquant y resterait manquant : rien, sur la
# VM, ne le calcule.
controler_jurix_dev() {
    # Le code envoye est celui de HEAD : une modification non commitee de
    # alembic/ ou de app/ (modele, empreinte) fausserait le controle.
    if [ -n "$(git -C "$RACINE" status --porcelain -- alembic app)" ]; then
        echo "Modifications non commitees sous alembic/ ou app/ : commiter d'abord, le code envoye est celui de HEAD"
        exit 1
    fi
    # Lance depuis un dossier vide : sans le .env local, la tete et
    # l'empreinte sont celles des reglages par defaut du code, ceux de la VM.
    # `|| true` : un echec laisse les variables vides, et le controle plus bas
    # le dit, au lieu d'un arret muet de `set -e`.
    local vide tete empreinte version etrangers manquants vectorises
    vide=$(mktemp -d)
    read -r tete empreinte < <(cd "$vide" && env -u VOYAGE_EMBEDDING_MODEL -u EMBEDDING_PROVIDER \
        PYTHONPATH="$RACINE" "$RACINE/.venv/bin/python" -c "
from alembic.config import Config
from alembic.script import ScriptDirectory
from app.core.config import settings
from app.services.embedding_service import empreinte_voyage
config = Config('$RACINE/alembic.ini')
config.set_main_option('script_location', '$RACINE/alembic')
print('TETE', ScriptDirectory.from_config(config).get_current_head(), empreinte_voyage(settings.VOYAGE_EMBEDDING_MODEL))" 2>/dev/null \
        | grep '^TETE ' | cut -d' ' -f2-) || true
    rmdir "$vide"
    read -r version etrangers manquants vectorises < <(docker exec jurix-pg psql -U jurix -d jurix_dev -tA -F ' ' -c \
        "SELECT (SELECT version_num FROM alembic_version),
                count(*) FILTER (WHERE embedding IS NOT NULL AND embedding_model IS DISTINCT FROM '$empreinte'),
                count(*) FILTER (WHERE embed AND embedding IS NULL),
                count(*) FILTER (WHERE embedding IS NOT NULL)
         FROM articles") || true
    if [ -z "$tete" ] || [ -z "$empreinte" ] || [ "$version" != "$tete" ] \
            || [ "$etrangers" != "0" ] || [ "$manquants" != "0" ] || [ "${vectorises:-0}" = "0" ]; then
        echo "jurix_dev n'est pas pret : migration ${version:-?} (attendue ${tete:-?}), ${etrangers:-?} vecteur(s) hors ${empreinte:-?}, ${manquants:-?} manquant(s), ${vectorises:-?} vecteur(s)"
        echo "  -> scripts/regenerate_embeddings.py --all (puis --reindex) sur jurix_dev"
        exit 1
    fi
}

if [ "$AVEC_DONNEES" = "--avec-donnees" ]; then
    echo "== 0. jurix_dev : derniere migration, vecteurs a jour (avant tout envoi)"
    controler_jurix_dev
fi

echo "== 1. Code du backend (etat commite de $(git -C "$RACINE" rev-parse --short HEAD))"
# Extrait a cote, puis aligne avec rsync --delete : un `tar -x` direct n'effacait
# rien, et un fichier supprime du depot restait sur la VM, jusque dans l'image.
# Les donnees de la VM (.env, PDF, sauvegardes, front, et l'ancien modele
# EmbeddingGemma d'une VM d'avant la bascule, que son API en place monte
# encore) sont exclues :
# rsync ne les touche pas.
git -C "$RACINE" archive --format=tar HEAD | "${SSH[@]}" "$CIBLE" 'set -e
    rm -rf jurix.nouveau && mkdir jurix.nouveau && tar -x -C jurix.nouveau
    mkdir -p jurix
    rsync -a --delete \
        --exclude=/deploiement/oracle/.env \
        --exclude=/deploiement/oracle/modeles/ \
        --exclude=/deploiement/oracle/donnees/ \
        --exclude=/deploiement/oracle/sauvegardes/ \
        --exclude=/deploiement/oracle/front/ \
        jurix.nouveau/ jurix/
    rm -rf jurix.nouveau'

echo "== 2. Front, construit pour https://$JURIX_DOMAINE"
# L'identifiant Google est fige dans le front a la construction. Sans
# JURIX_GOOGLE_CLIENT_ID ici, il est repris du .env de la VM : sinon chaque
# mise a jour faite sans la variable retirait la connexion Google, en silence.
GOOGLE_ID=${JURIX_GOOGLE_CLIENT_ID:-$("${SSH[@]}" "$CIBLE" \
    "grep -E '^GOOGLE_CLIENT_ID=' $DISTANT/.env 2>/dev/null | tail -1 | cut -d= -f2-" || true)}
echo "Connexion Google : $([ -n "$GOOGLE_ID" ] && echo activee || echo absente)"
( cd "$FRONT" && VITE_API_URL="https://$JURIX_DOMAINE" \
    VITE_GOOGLE_CLIENT_ID="$GOOGLE_ID" npm run build >/dev/null )
"${RSYNC[@]}" --delete "$FRONT/build/" "$CIBLE:$DISTANT/front/"

if [ "$AVEC_DONNEES" = "--avec-donnees" ]; then
    echo "== 3. PDF d'origine (reprenable : relancer en cas de coupure)"
    "${RSYNC_ROOT[@]}" "$RACINE/data/uploads/" "$CIBLE:$DISTANT/donnees/uploads/"

    echo "== 4. Base : dump de jurix_dev, SANS les comptes ni les conversations"
    # Le meme controle qu'a l'etape 0, refait : l'envoi des PDF dure des
    # heures, et jurix_dev a pu changer entre-temps.
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

echo "== 5. Demarrage sur la VM (demarrer.sh)"
# Un FICHIER lance par ssh, et non un script passe sur l'entree standard : un
# `docker compose exec` y avalait la suite, et rien apres lui ne s'executait.
# JURIX_PURGER_VECTEURS : seule la VM le lit (demarrer.sh), ssh ne transmet
# pas l'environnement local.
"${SSH[@]}" "$CIBLE" "sudo chown -R 1000:1000 $DISTANT/donnees && JURIX_PURGER_VECTEURS=${JURIX_PURGER_VECTEURS:-} bash $DISTANT/demarrer.sh"

echo
echo "Verifier : https://$JURIX_DOMAINE  et  https://$JURIX_DOMAINE/health"
