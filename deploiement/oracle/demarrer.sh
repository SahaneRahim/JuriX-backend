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
# La politique de l'API (app/schemas/user.py), verifiee ici AVANT la
# construction de l'image : refuse a l'etape 6 seulement, le mot de passe
# faisait echouer un deploiement de vingt minutes. create_admin.py reste juge.
mdp=$(valeur ADMIN_PASSWORD)
# docker compose interprete le .env avant de le passer au conteneur : $NOM y
# devient une variable (vide si absente), ` #` ouvre un commentaire, les
# guillemets et les espaces du bord disparaissent. « Jurix2026$admin »
# arrivait a create_admin.py sous la forme « Jurix2026 » : un administrateur
# cree avec un autre mot de passe que celui du .env.
case "$mdp" in
    *'$'*|*'#'*|*'"'*|*"'"*|[[:space:]]*|*[[:space:]])
        echo "ADMIN_PASSWORD refuse : ni \$, ni #, ni guillemets, ni espace au debut ou a la fin (docker compose les interprete)"
        exit 1 ;;
esac
if [ -n "$mdp" ] && ! { [ "${#mdp}" -ge 8 ] && [ "${#mdp}" -le 100 ] \
        && [[ $mdp =~ [[:upper:]] ]] && [[ $mdp =~ [[:lower:]] ]] && [[ $mdp =~ [[:digit:]] ]]; }; then
    echo "ADMIN_PASSWORD refuse : 8 a 100 caracteres, dont une majuscule, une minuscule et un chiffre"
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
# Deux requetes, et non un CASE : PostgreSQL resout `laws` a l'analyse, meme
# dans la branche non prise, et sur une base neuve la requete echouait.
# Une affectation seule, puis un `case` qui n'accepte que t ou f : dans la
# condition d'un `if`, `set -e` ne s'applique pas, et un psql en echec
# (conteneur qui redemarre, demon Docker) passait pour « aucune table », donc
# pour une base a restaurer : DROP DATABASE de la production, puis retour au
# corpus du premier jour, sans un compte.
existe=$(requete "SELECT to_regclass('public.laws') IS NOT NULL")
case "$existe" in
    t) lois=$(requete "SELECT count(*) FROM laws") ;;
    f) lois=0 ;;
    *) echo "Reponse inattendue de la base (« $existe ») : arret, rien n'est touche"; exit 1 ;;
esac

# Un temoin pose avant le DROP, retire apres une restauration complete. Sans
# lui, une restauration interrompue (coupure, memoire pendant l'index HNSW)
# laissait des lois sans index ni contraintes, et la relance, qui ne compte
# que les lois, prenait cette base pour saine.
temoin=sauvegardes/.restauration_en_cours
a_restaurer=0
peuplee=0
if [ -f "$temoin" ]; then
    [ -f sauvegardes/initial.dump ] || {
        echo "Restauration interrompue, et plus de sauvegardes/initial.dump : restaurer a la main (README §6)"
        exit 1
    }
    echo "La restauration precedente n'a pas abouti : elle reprend de zero"
    a_restaurer=1
elif [ "$lois" = "0" ] && [ -f sauvegardes/initial.dump ]; then
    a_restaurer=1
fi

if [ "$a_restaurer" = 1 ]; then
    echo "Restauration de sauvegardes/initial.dump dans une base neuve"
    touch "$temoin"
    docker compose stop api caddy >/dev/null 2>&1 || true
    requete "DROP DATABASE IF EXISTS jurix WITH (FORCE)" postgres >/dev/null
    requete "CREATE DATABASE jurix OWNER jurix" postgres >/dev/null
    # --exit-on-error : une restauration partielle ne doit pas passer pour
    # reussie. Dans une base neuve, aucune erreur n'est attendue.
    docker compose exec -T db pg_restore -U jurix -d jurix --no-owner --no-privileges \
        --exit-on-error < sauvegardes/initial.dump
    lois=$(requete "SELECT count(*) FROM laws")
    [ "$lois" -gt 0 ] || { echo "Restauration en echec : aucune loi"; exit 1; }
    # Consomme : plus aucune mise a jour ne pourra la rejouer, quoi qu'il
    # arrive a la base ensuite. Gardee pour memoire, sous un autre nom (hors de
    # la rotation de sauvegarde.sh).
    mv sauvegardes/initial.dump "sauvegardes/initial.restaure_$(date +%Y%m%d_%H%M).dump"
    rm -f "$temoin"
    echo "$lois lois restaurees"
elif [ "$lois" = "0" ]; then
    echo "ATTENTION : base sans aucune loi, et pas de sauvegardes/initial.dump"
else
    echo "Base deja peuplee ($lois lois) : pas de restauration"
    peuplee=1
fi

echo "== 4. API (reconstruite si le code a change) et Caddy"
# Une sauvegarde AVANT une mise a jour : l'API applique les migrations a son
# demarrage, et la sauvegarde nocturne peut avoir jusqu'a 24 h.
if [ "$peuplee" = 1 ]; then
    bash sauvegarde.sh
fi
docker compose up -d --build
# Caddy ne relit pas sa configuration tout seul.
docker compose exec -T caddy caddy reload --config /etc/caddy/Caddyfile </dev/null >/dev/null 2>&1 \
    || docker compose restart caddy
# L'image remplacee reste sur le disque, sans nom : une de plus a chaque mise a
# jour. Seules celles de JuriX sont visees.
docker image prune -f --filter label=com.docker.compose.project=jurix >/dev/null

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
