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
# Prerequis sur la VM : installer.sh deja lance, et deploiement/oracle/.env
# renseigne (voir env.production.exemple).
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

echo "== 1. Code du backend (etat commite de $(git -C "$RACINE" rev-parse --short HEAD))"
git -C "$RACINE" archive --format=tar HEAD | "${SSH[@]}" "$CIBLE" "mkdir -p jurix && tar -x -C jurix"

echo "== 2. Front, construit pour https://$JURIX_DOMAINE"
( cd "$FRONT" && VITE_API_URL="https://$JURIX_DOMAINE" npm run build >/dev/null )
"${RSYNC[@]}" --delete "$FRONT/build/" "$CIBLE:$DISTANT/front/"

if [ "$AVEC_DONNEES" = "--avec-donnees" ]; then
    echo "== 3. Modele d'embeddings"
    "${RSYNC[@]}" "$MODELE/" "$CIBLE:$DISTANT/modeles/embeddinggemma-300m-onnx/"

    echo "== 4. PDF d'origine (reprenable : relancer en cas de coupure)"
    "${RSYNC[@]}" "$RACINE/data/uploads/" "$CIBLE:$DISTANT/donnees/uploads/"

    echo "== 5. Base : dump de jurix_dev"
    dump=$(mktemp --suffix=.dump)
    docker exec jurix-pg pg_dump -U jurix -Fc --no-owner --no-privileges jurix_dev > "$dump"
    "${RSYNC[@]}" "$dump" "$CIBLE:$DISTANT/sauvegardes/initial.dump"
    rm -f "$dump"
fi

echo "== 6. Demarrage sur la VM"
"${SSH[@]}" "$CIBLE" bash -s <<DISTANT_FIN
set -euo pipefail
cd $DISTANT
sudo chown -R 1000:1000 donnees
docker compose up -d db
until docker compose exec -T db pg_isready -U jurix -d jurix >/dev/null 2>&1; do sleep 2; done
lois=\$(docker compose exec -T db psql -U jurix -d jurix -tAc "SELECT to_regclass('public.laws') IS NOT NULL")
if [ "\$lois" = "f" ] && [ -f sauvegardes/initial.dump ]; then
    echo "Base vide : restauration du dump initial"
    docker compose exec -T db pg_restore -U jurix -d jurix --no-owner --no-privileges < sauvegardes/initial.dump || true
    n=\$(docker compose exec -T db psql -U jurix -d jurix -tAc "SELECT count(*) FROM laws")
    echo "\$n lois restaurees"
    [ "\$n" -gt 0 ] || { echo "Restauration en echec"; exit 1; }
fi
docker compose up -d --build
echo "Attente de l'API (chargement du modele, migrations)..."
for i in \$(seq 1 60); do
    if docker compose exec -T api curl -fsS http://localhost:8000/health >/dev/null 2>&1; then
        echo "API prete"; break
    fi
    sleep 5
done
docker compose ps
DISTANT_FIN

echo
echo "Verifier : https://$JURIX_DOMAINE  et  https://$JURIX_DOMAINE/health"
