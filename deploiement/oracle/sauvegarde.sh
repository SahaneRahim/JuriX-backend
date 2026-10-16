#!/usr/bin/env bash
# Sauvegarde de la base de production, gardee 14 jours.
# Lancee chaque nuit par cron (voir installer.sh), et par demarrer.sh avant
# chaque mise a jour ou restauration ; a la main :
#   bash sauvegarde.sh
set -euo pipefail
cd "$(dirname "$0")"

# Pendant une restauration, la base est partielle, ou absente : la sauvegarder
# donnerait un jurix_*.dump d'apparence saine, que la rotation garderait.
if [ -e sauvegardes/.restauration_en_cours ]; then
    echo "$(date -Is) restauration en cours (sauvegardes/.restauration_en_cours) : pas de sauvegarde, la terminer d'abord (bash demarrer.sh)"
    exit 1
fi
# Le verrou de demarrer.sh : pas de sauvegarde pendant qu'il restaure ou migre
# la base. Lancee par lui (JURIX_VERROU_TENU=1), elle le tient deja par son
# descripteur herite ; le reprendre ici l'attendrait en vain.
if [ "${JURIX_VERROU_TENU:-}" != 1 ]; then
    exec 9>sauvegardes/.verrou
    flock -n 9 || { echo "$(date -Is) demarrer.sh ou une autre sauvegarde tourne : sauvegarde non faite"; exit 1; }
fi

# A la seconde : deux sauvegardes de la meme minute portaient le meme nom, et
# la seconde ecrasait la sauvegarde d'avant mise a jour que nomme le journal
# des deploiements.
horodatage=$(date +%Y%m%d_%H%M%S)
cible="sauvegardes/jurix_${horodatage}.dump"

# Ecrite sous un nom provisoire, propre a ce passage, puis renommee : une
# sauvegarde interrompue ne se fait jamais passer pour une sauvegarde complete.
partiel="${cible}.$$.partiel"
docker compose exec -T db pg_dump -U jurix -Fc jurix </dev/null > "$partiel"
mv "$partiel" "$cible"

find sauvegardes -name 'jurix_*.dump' -mtime +14 -delete
find sauvegardes -name 'jurix_*.partiel' -mtime +1 -delete
# demarrer.sh lit le nom du fichier dans cette ligne, pour le journal des
# deploiements : en garder la forme « sauvegarde <chemin> ».
echo "$(date -Is) sauvegarde ${cible} ($(du -h "${cible}" | cut -f1))"
