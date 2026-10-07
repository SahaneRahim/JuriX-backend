#!/usr/bin/env bash
# Sauvegarde de la base de production, gardee 14 jours.
# Lancee chaque nuit par cron (voir installer.sh) ; a la main :
#   bash sauvegarde.sh
set -euo pipefail
cd "$(dirname "$0")"

horodatage=$(date +%Y%m%d_%H%M)
cible="sauvegardes/jurix_${horodatage}.dump"

# Ecrite sous un nom provisoire puis renommee : une sauvegarde interrompue ne
# se fait jamais passer pour une sauvegarde complete.
docker compose exec -T db pg_dump -U jurix -Fc jurix > "${cible}.partiel"
mv "${cible}.partiel" "${cible}"

find sauvegardes -name 'jurix_*.dump' -mtime +14 -delete
find sauvegardes -name 'jurix_*.partiel' -mtime +1 -delete
echo "$(date -Is) sauvegarde ${cible} ($(du -h "${cible}" | cut -f1))"
