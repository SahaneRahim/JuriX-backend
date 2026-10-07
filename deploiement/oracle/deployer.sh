#!/usr/bin/env bash
# Deploie (ou met a jour) JuriX sur la VM, depuis la machine de developpement.
#
#   JURIX_DOMAINE=jurix.exemple.cm deploiement/oracle/deployer.sh ubuntu@IP
#   JURIX_DOMAINE=jurix.exemple.cm deploiement/oracle/deployer.sh ubuntu@IP --avec-donnees
#
# Sans option : le code COMMITE du backend (git archive, jamais les fichiers
# locaux, donc jamais un .env), le front reconstruit pour le domaine, puis
# `docker compose up -d --build`.
# --avec-donnees, au premier deploiement : en plus, le modele d'embeddings, les
# PDF (rsync, reprenable) et la base (dump de jurix_dev, restaure seulement si
# la base de la VM est vide).
#
# Prerequis :
# - sur la VM : installer.sh deja lance, et deploiement/oracle/.env renseigne
#   (voir env.production.exemple) ;
# - ici : le depot du front a cote de celui-ci (ou JURIX_FRONT), sur la branche
#   du site statique, avec ses node_modules ; pour --avec-donnees, le conteneur
#   local jurix-pg demarre et le modele dans ~/modeles (ou JURIX_MODELE).
# JURIX_GOOGLE_CLIENT_ID : l'identifiant OAuth Google, si la connexion Google
# est activee (il est fige dans le front au build).
set -euo pipefail

CIBLE=${1:?usage : JURIX_DOMAINE=... deployer.sh utilisateur@ip [--avec-donnees]}
AVEC_DONNEES=${2:-}
: "${JURIX_DOMAINE:?renseigner JURIX_DOMAINE (le domaine du site)}"

RACINE=$(cd "$(dirname "$0")/../.." && pwd)
FRONT=${JURIX_FRONT:-$RACINE/../JuriX-frontend-main}
MODELE=${JURIX_MODELE:-$HOME/modeles/embeddinggemma-300m-onnx}
CLE=${JURIX_CLE_SSH:-$HOME/.ssh/jurix_oracle}
DISTANT=jurix/deploiement/oracle
SSH=(ssh -i "$CLE" -o StrictHostKeyChecking=accept-new)
RSYNC=(rsync -az --partial --info=progress2 -e "ssh -i $CLE -o StrictHostKeyChecking=accept-new")
# Les donnees appartiennent a l'utilisateur du conteneur (uid 1000), qui n'est
# pas forcement celui de la VM : l'envoi se fait en root, cote VM.
RSYNC_ROOT=("${RSYNC[@]}" --rsync-path="sudo rsync")

echo "== 1. Code du backend (etat commite de $(git -C "$RACINE" rev-parse --short HEAD))"
git -C "$RACINE" archive --format=tar HEAD | "${SSH[@]}" "$CIBLE" "mkdir -p jurix && tar -x -C jurix"

echo "== 2. Front, construit pour https://$JURIX_DOMAINE"
( cd "$FRONT" && VITE_API_URL="https://$JURIX_DOMAINE" \
    VITE_GOOGLE_CLIENT_ID="${JURIX_GOOGLE_CLIENT_ID:-}" npm run build >/dev/null )
"${RSYNC[@]}" --delete "$FRONT/build/" "$CIBLE:$DISTANT/front/"

if [ "$AVEC_DONNEES" = "--avec-donnees" ]; then
    echo "== 3. Modele d'embeddings"
    "${RSYNC_ROOT[@]}" "$MODELE/" "$CIBLE:$DISTANT/modeles/embeddinggemma-300m-onnx/"

    echo "== 4. PDF d'origine (reprenable : relancer en cas de coupure)"
    "${RSYNC_ROOT[@]}" "$RACINE/data/uploads/" "$CIBLE:$DISTANT/donnees/uploads/"

    echo "== 5. Base : dump de jurix_dev, SANS les comptes ni les conversations"
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

echo "== 6. Demarrage sur la VM (demarrer.sh)"
# Un FICHIER lance par ssh, et non un script passe sur l'entree standard : un
# `docker compose exec` y avalait la suite, et rien apres lui ne s'executait.
"${SSH[@]}" "$CIBLE" "sudo chown -R 1000:1000 $DISTANT/donnees && bash $DISTANT/demarrer.sh"

echo
echo "Verifier : https://$JURIX_DOMAINE  et  https://$JURIX_DOMAINE/health"
