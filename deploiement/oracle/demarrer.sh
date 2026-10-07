#!/usr/bin/env bash
# Demarre (ou met a jour) JuriX, SUR la VM. deployer.sh le lance apres avoir
# envoye le code ; a la main, depuis ce dossier :
#   bash demarrer.sh
#
# Chaque commande `docker compose exec` qui n'a rien a lire recoit </dev/null :
# lance par `ssh ... bash -s`, un exec qui garde l'entree standard avalait la
# suite du script, et rien apres lui ne s'executait.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1. Le .env : present, et sans secret oublie"
[ -f .env ] || { echo "Pas de .env : cp env.production.exemple .env, puis le renseigner (README §4)"; exit 1; }
valeur() { grep -E "^$1=" .env | tail -1 | cut -d= -f2- || true; }
manquants=()
for cle in JURIX_DOMAINE POSTGRES_PASSWORD SECRET_KEY MISTRAL_API_KEY GROQ_API_KEY; do
    case "$(valeur "$cle")" in
        ""|a-generer|a-renseigner|jurix.exemple.cm) manquants+=("$cle") ;;
    esac
done
if [ "${#manquants[@]}" -gt 0 ]; then
    echo ".env incomplet : ${manquants[*]}"
    exit 1
fi
if [ "$(valeur SECRET_KEY | tr -d '\n' | wc -c)" -lt 32 ]; then
    echo "SECRET_KEY trop courte : openssl rand -hex 32"
    exit 1
fi

echo "== 2. La base, seule d'abord"
docker compose up -d db
# Par TCP : a sa premiere initialisation, l'image demarre un serveur
# temporaire qui n'ecoute que sur le socket, et qu'il ne faut pas prendre
# pour le vrai.
until docker compose exec -T db pg_isready -h 127.0.0.1 -U jurix -d jurix </dev/null >/dev/null 2>&1; do
    sleep 2
done
requete() { docker compose exec -T db psql -U jurix -d "${2:-jurix}" -tAc "$1" </dev/null; }

echo "== 3. Restauration initiale, seulement si la base n'a AUCUNE loi"
# Le nombre de lois, et non l'existence de la table : l'API cree le schema
# (vide) a chaque demarrage, et un premier demarrage avant la restauration la
# faisait sauter pour toujours, sans un mot.
lois=$(requete "SELECT CASE WHEN to_regclass('public.laws') IS NULL THEN 0 ELSE (SELECT count(*) FROM laws) END")
if [ "$lois" = "0" ] && [ -f sauvegardes/initial.dump ]; then
    echo "Base sans loi : restauration de sauvegardes/initial.dump dans une base neuve"
    docker compose stop api caddy >/dev/null 2>&1 || true
    requete "DROP DATABASE IF EXISTS jurix WITH (FORCE)" postgres >/dev/null
    requete "CREATE DATABASE jurix OWNER jurix" postgres >/dev/null
    # --exit-on-error : une restauration partielle ne doit pas passer pour
    # reussie. Dans une base neuve, aucune erreur n'est attendue.
    docker compose exec -T db pg_restore -U jurix -d jurix --no-owner --no-privileges \
        --exit-on-error < sauvegardes/initial.dump
    lois=$(requete "SELECT count(*) FROM laws")
    echo "$lois lois restaurees"
    [ "$lois" -gt 0 ] || { echo "Restauration en echec : aucune loi"; exit 1; }
elif [ "$lois" = "0" ]; then
    echo "ATTENTION : base sans aucune loi, et pas de sauvegardes/initial.dump"
else
    echo "Base deja peuplee ($lois lois) : pas de restauration"
fi

echo "== 4. API (reconstruite si le code a change) et Caddy"
docker compose up -d --build
# Caddy ne relit pas sa configuration tout seul.
docker compose exec -T caddy caddy reload --config /etc/caddy/Caddyfile </dev/null >/dev/null 2>&1 \
    || docker compose restart caddy

echo "== 5. Attente de l'API (chargement du modele, migrations)"
prete=0
for _ in $(seq 1 60); do
    if docker compose exec -T api curl -fsS http://localhost:8000/health </dev/null >/dev/null 2>&1; then
        prete=1
        break
    fi
    sleep 5
done
[ "$prete" = 1 ] || { echo "L'API ne repond pas : docker compose logs api"; exit 1; }
echo "API prete"

echo "== 6. Compte administrateur"
# La base est restauree sans les comptes de developpement. Rien n'est fait s'il
# existe deja un administrateur : create_admin.py ne remet jamais un mot de
# passe, mais inutile de le lancer a chaque mise a jour.
admins=$(requete "SELECT count(*) FROM users WHERE role IN ('admin', 'superadmin')")
if [ "$admins" = "0" ]; then
    if [ -n "$(valeur ADMIN_EMAIL)" ] && [ -n "$(valeur ADMIN_PASSWORD)" ]; then
        docker compose exec -T api python scripts/create_admin.py </dev/null
    else
        echo "ATTENTION : aucun administrateur. Renseigner ADMIN_EMAIL et ADMIN_PASSWORD dans .env, puis relancer."
    fi
else
    echo "$admins administrateur(s) deja present(s)"
fi

docker compose ps
