#!/usr/bin/env bash
# Demarre (ou met a jour) JuriX, SUR la VM. deployer.sh le lance apres avoir
# envoye le code ; a la main, depuis ce dossier :
#   bash demarrer.sh
#   JURIX_RESTAURER=sauvegardes/<fichier>.dump bash demarrer.sh   (README §6)
#
# Chaque commande `docker compose exec` qui n'a rien a lire recoit </dev/null :
# lance par `ssh ... bash -s`, un exec qui garde l'entree standard avalait la
# suite du script, et rien apres lui ne s'executait.
set -euo pipefail
cd "$(dirname "$0")"

# Un seul passage a la fois, et aucune sauvegarde pendant qu'il tourne : deux
# demarrer.sh restaureraient ou migreraient la meme base ensemble, et la
# sauvegarde de nuit copierait une base en pleine restauration. sauvegarde.sh
# prend le meme verrou, sauf lancee d'ici (JURIX_VERROU_TENU) ; le
# descripteur 9 le tient jusqu'a la fin du script.
[ -d sauvegardes ] || { echo "Pas de dossier sauvegardes/ : lancer installer.sh d'abord"; exit 1; }
exec 9>sauvegardes/.verrou
flock -n 9 || { echo "Un autre demarrer.sh, ou une sauvegarde, tourne deja : relancer une fois fini"; exit 1; }

echo "== 1. Le .env (present, et sans secret oublie) et le front envoye"
[ -f .env ] || { echo "Pas de .env : cp env.production.exemple .env, puis le renseigner (README §4)"; exit 1; }
valeur() { grep -E "^$1=" .env | tail -1 | cut -d= -f2- || true; }
manquants=()
# VOYAGE_API_KEY : sans elle, l'API demarre quand meme et repond en plein
# texte seul, la recherche semantique coupee EN SILENCE. Seul ce controle
# l'empeche.
for cle in JURIX_DOMAINE POSTGRES_PASSWORD SECRET_KEY MISTRAL_API_KEY GROQ_API_KEY VOYAGE_API_KEY; do
    case "$(valeur "$cle")" in
        ""|a-generer|a-renseigner|jurix.exemple.cm) manquants+=("$cle") ;;
    esac
done
if [ "${#manquants[@]}" -gt 0 ]; then
    echo ".env incomplet : ${manquants[*]}"
    exit 1
fi
# Lus tels quels ici et par deployer.sh (grep, cut), alors que docker compose
# en retire guillemets et espaces : « "jurix.cm" » ne serait pas le domaine
# fige dans le front, et le controle du front plus bas echouerait sans dire
# pourquoi.
for cle in JURIX_DOMAINE GOOGLE_CLIENT_ID; do
    case "$(valeur "$cle")" in
        *[[:space:]\"\'\$#/]*)
            printf '%s refuse dans .env : %q (ni guillemets, ni espaces, ni $, ni #, ni /)\n' "$cle" "$(valeur "$cle")"
            exit 1 ;;
    esac
done
# Un .env d'avant la bascule (EMBEDDING_PROVIDER=gemma, GEMMA_*) : la
# composition fige voyage, mais la ligne trompe qui lit le fichier.
if grep -qE '^(EMBEDDING_PROVIDER=gemma|GEMMA_)' .env; then
    echo ".env d'avant la bascule vers Voyage : retirer EMBEDDING_PROVIDER=gemma et les GEMMA_*"
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
# Une restauration ne se demande qu'au lancement, jamais par un fichier pose a
# l'avance : une demande oubliee ferait sinon, au deploiement suivant, un DROP
# de la production. Verifiee ici, avant que la base ne soit touchee.
restaurer=${JURIX_RESTAURER:-}
if [ -n "$restaurer" ]; then
    if [[ ! $restaurer =~ ^sauvegardes/[A-Za-z0-9._-]+\.dump$ ]] || [ ! -f "$restaurer" ]; then
        printf 'JURIX_RESTAURER refuse : %q (un fichier existant, sauvegardes/<fichier>.dump)\n' "$restaurer"
        exit 1
    fi
fi
# Le front, envoye par deployer.sh. Controle ici, avant que la base ne soit
# touchee : la construction de son image, a l'etape 4, echouerait apres une
# eventuelle restauration. VERSION, que deployer.sh ecrit en dernier, prouve
# un envoi complet ; lu une seule fois, ici, pour le journal des deploiements.
if [ ! -f front-node/VERSION ] || [ ! -f front-node/build/index.js ] || [ ! -f front-node/package.json ]; then
    echo "front-node/ incomplet (VERSION, build/index.js ou package.json manquant) : le front s'envoie par deployer.sh (README §5)"
    exit 1
fi
read -r backend_envoye front_envoye date_envoi < front-node/VERSION
echo "Front : commit $front_envoye, envoye le $date_envoi avec le backend $backend_envoye"
# Le code et le front d'un meme envoi : deployer.sh ajoute BACKEND_VERSION a
# l'archive du code. Un deploiement coupe entre l'envoi du front et celui du
# code laisserait sinon l'un d'un envoi, l'autre du precedent.
[ -f BACKEND_VERSION ] || { echo "Pas de BACKEND_VERSION : ce code n'a pas ete envoye par deployer.sh"; exit 1; }
if [ "$(cat BACKEND_VERSION)" != "$backend_envoye" ]; then
    echo "Code du backend $(cat BACKEND_VERSION), front envoye avec le backend $backend_envoye : deux envois differents, relancer deployer.sh"
    exit 1
fi
# Construit pour le domaine de CE .env, d'ou viennent ORIGIN et
# ALLOWED_ORIGINS : sinon les pages s'affichent, mais le navigateur envoie ses
# appels a l'API (recherche, connexion, chat) a un autre domaine.
if ! grep -rqF -- "https://$(valeur JURIX_DOMAINE)" front-node/build/client/_app; then
    echo "front-node/ n'est pas construit pour https://$(valeur JURIX_DOMAINE), le JURIX_DOMAINE du .env : relancer deployer.sh avec ce domaine"
    exit 1
fi
# La page d'attente de Caddy : absente, une panne du front ou de l'API
# s'afficherait en erreur vide.
[ -f caddy/maintenance.html ] || { echo "Pas de caddy/maintenance.html : redeployer depuis un depot ou elle est commitee"; exit 1; }

echo "== 2. La base, seule d'abord"
docker compose up -d db
# Par TCP : a sa premiere initialisation, l'image demarre un serveur
# temporaire qui n'ecoute que sur le socket, et qu'il ne faut pas prendre
# pour le vrai.
until docker compose exec -T db pg_isready -h 127.0.0.1 -U jurix -d jurix </dev/null >/dev/null 2>&1; do
    sleep 2
done
requete() { docker compose exec -T db psql -U jurix -d "${2:-jurix}" -tAc "$1" </dev/null; }
# sauvegarde.sh, sous le verrou deja tenu, et le nom du fichier, lu dans la
# ligne qu'il ecrit : le journal le donne tel que JURIX_RESTAURER l'attend.
sauvegarder() {
    local rapport motif=' sauvegarde (sauvegardes/jurix_[0-9_]+\.dump) '
    rapport=$(JURIX_VERROU_TENU=1 bash sauvegarde.sh)
    echo "$rapport"
    [[ $rapport =~ $motif ]] || { echo "Nom de la sauvegarde introuvable dans : $rapport"; exit 1; }
    sauvegarde_avant=${BASH_REMATCH[1]}
}

echo "== 3. Restauration : premier deploiement, demande, ou reprise"
# La base n'est jetee (DROP) que dans trois cas : elle n'a aucune loi et
# sauvegardes/initial.dump existe (premier deploiement) ; JURIX_RESTAURER le
# demande a ce lancement (une base qui a des lois est d'abord sauvegardee) ;
# ou une restauration interrompue reprend.
# Le temoin n'est ecrit que par ce script, juste avant le DROP, et porte le
# chemin du dump : la relance apres une coupure (memoire pendant l'index HNSW,
# reseau) reprend la meme restauration, de zero, sans qu'on la redemande. Lu
# AVANT toute requete sur la base jurix : coupee entre le DROP et le CREATE,
# la restauration laisse une base absente, que la premiere requete ferait
# echouer.
temoin=sauvegardes/.restauration_en_cours
a_restaurer=
peuplee=0
sauvegarde_avant=-
revision_avant=-
if [ -f "$temoin" ]; then
    a_restaurer=$(cat "$temoin")
    # Vide : un temoin d'une version precedente, qui ne portait pas de chemin.
    [ -n "$a_restaurer" ] || a_restaurer=sauvegardes/initial.dump
    if [ -n "$restaurer" ] && [ "$restaurer" != "$a_restaurer" ]; then
        echo "Restauration interrompue de $a_restaurer : la terminer d'abord (bash demarrer.sh, sans JURIX_RESTAURER)"
        exit 1
    fi
    [ -f "$a_restaurer" ] || { echo "Restauration interrompue de $a_restaurer, introuvable : restaurer a la main (README §6)"; exit 1; }
    echo "Restauration interrompue de $a_restaurer : elle reprend de zero"
else
    # Le nombre de lois, et non l'existence de la table : l'API cree le schema
    # (vide) a chaque demarrage, et un premier demarrage avant la restauration
    # la faisait sauter pour toujours, sans un mot.
    # Deux requetes, et non un CASE : PostgreSQL resout `laws` a l'analyse,
    # meme dans la branche non prise, et sur une base neuve la requete
    # echouait.
    # Une affectation seule, puis un `case` qui n'accepte que t ou f : dans la
    # condition d'un `if`, `set -e` ne s'applique pas, et un psql en echec
    # (conteneur qui redemarre, demon Docker) passait pour « aucune table »,
    # donc pour une base a restaurer : DROP DATABASE de la production, puis
    # retour au corpus du premier jour, sans un compte.
    existe=$(requete "SELECT to_regclass('public.laws') IS NOT NULL")
    case "$existe" in
        t) lois=$(requete "SELECT count(*) FROM laws") ;;
        f) lois=0 ;;
        *) echo "Reponse inattendue de la base (« $existe ») : arret, rien n'est touche"; exit 1 ;;
    esac
    # La revision de la base avant ce passage, pour le journal : avec la
    # sauvegarde d'avant, elle dit quel etat restaurer pour revenir en arriere.
    alembic=$(requete "SELECT to_regclass('public.alembic_version') IS NOT NULL")
    case "$alembic" in
        t) revision_avant=$(requete "SELECT coalesce(string_agg(version_num, ','), '-') FROM alembic_version") ;;
        f) revision_avant=- ;;
        *) echo "Reponse inattendue de la base (« $alembic ») : arret, rien n'est touche"; exit 1 ;;
    esac
    if [ -n "$restaurer" ]; then
        if [ "$lois" != "0" ]; then
            echo "Base peuplee ($lois lois) : sauvegardee avant de restaurer $restaurer"
            sauvegarder
        fi
        a_restaurer=$restaurer
    elif [ "$lois" = "0" ] && [ -f sauvegardes/initial.dump ]; then
        a_restaurer=sauvegardes/initial.dump
    fi
fi

if [ -n "$a_restaurer" ]; then
    echo "Restauration de $a_restaurer dans une base neuve"
    docker compose stop api front caddy >/dev/null 2>&1 || true
    printf '%s\n' "$a_restaurer" > "$temoin"
    requete "DROP DATABASE IF EXISTS jurix WITH (FORCE)" postgres >/dev/null
    requete "CREATE DATABASE jurix OWNER jurix" postgres >/dev/null
    # --exit-on-error : une restauration partielle ne doit pas passer pour
    # reussie. Dans une base neuve, aucune erreur n'est attendue.
    docker compose exec -T db pg_restore -U jurix -d jurix --no-owner --no-privileges \
        --exit-on-error < "$a_restaurer"
    lois=$(requete "SELECT count(*) FROM laws")
    [ "$lois" -gt 0 ] || { echo "Restauration en echec : aucune loi"; exit 1; }
    # Le temoin d'abord : coupe juste apres, le script laisse au pire un
    # initial.dump sans temoin, que la relance, devant une base peuplee, met
    # de cote (initial.ignore_), et non un temoin sans son dump.
    rm -f "$temoin"
    # initial.dump consomme : plus aucune mise a jour ne pourra le rejouer,
    # quoi qu'il arrive a la base ensuite. Garde pour memoire, sous un autre
    # nom (hors de la rotation de sauvegarde.sh).
    if [ "$a_restaurer" = sauvegardes/initial.dump ]; then
        mv sauvegardes/initial.dump "sauvegardes/initial.restaure_$(date +%Y%m%d_%H%M%S).dump"
    fi
    echo "$lois lois restaurees"
elif [ "$lois" = "0" ]; then
    echo "ATTENTION : base sans aucune loi, et pas de sauvegardes/initial.dump"
else
    echo "Base deja peuplee ($lois lois) : pas de restauration"
    peuplee=1
    # Un initial.dump arrive sur une base deja peuplee (deployer.sh relance
    # avec --avec-donnees) n'a pas servi. Laisse la, il serait restaure par la
    # premiere relance qui trouverait la base sans loi : DROP DATABASE, et
    # retour a ce corpus, sans les comptes crees depuis. Le fichier reste, sous
    # un autre nom, hors de la rotation de sauvegarde.sh.
    if [ -f sauvegardes/initial.dump ]; then
        ignore="sauvegardes/initial.ignore_$(date +%Y%m%d_%H%M%S).dump"
        mv sauvegardes/initial.dump "$ignore"
        echo "sauvegardes/initial.dump inutilise, la base etant deja peuplee : renomme en $ignore"
    fi
fi

# La migration a8b9c0d1e2f3 (embeddings Voyage, halfvec(1024)) VIDE les
# vecteurs d'une base qui ne l'a pas encore, au demarrage de l'API, et la VM
# n'a pas de quoi les recharger sans tout recalculer par l'API (environ 10 M de
# jetons). Ce cas ne passe que sur demande explicite.
if [ "$lois" != "0" ]; then
    type_vecteurs=$(requete "SELECT format_type(atttypid, atttypmod) FROM pg_attribute WHERE attrelid = 'public.articles'::regclass AND attname = 'embedding'")
    vecteurs=$(requete "SELECT count(embedding) FROM articles")
    if [ "$type_vecteurs" != "halfvec(1024)" ] && [ "$vecteurs" != "0" ] \
            && [ "${JURIX_PURGER_VECTEURS:-}" != "1" ]; then
        echo "ARRET : $vecteurs vecteurs en $type_vecteurs, que la migration vers Voyage viderait."
        echo "Pour l'accepter : JURIX_PURGER_VECTEURS=1 bash demarrer.sh, puis recalculer :"
        echo "  docker compose exec api python scripts/regenerate_embeddings.py --all"
        echo "  docker compose exec api python scripts/regenerate_embeddings.py --reindex"
        exit 1
    fi
fi

echo "== 4. API et front (reconstruits si leur code a change), et Caddy"
# Une sauvegarde AVANT une mise a jour : l'API applique les migrations a son
# demarrage, et la sauvegarde nocturne peut avoir jusqu'a 24 h.
if [ "$peuplee" = 1 ]; then
    sauvegarder
fi
# Le journal des deploiements (README §5) : « tentative » avant de remplacer
# l'API et le front, « ok » une fois le rendu controle. La derniere ligne,
# l'une ou l'autre, est la version qui tourne ; la sauvegarde et la revision
# d'avant disent quel etat restaurer pour revenir avant ce passage.
journal() {
    printf '%s %s %s %s %s %s\n' "$(date -Is)" "$1" "$backend_envoye" "$front_envoye" \
        "$sauvegarde_avant" "$revision_avant" >> sauvegardes/deploiements.log
}
journal tentative
docker compose up -d --build
# Caddy ne relit pas sa configuration tout seul.
docker compose exec -T caddy caddy reload --config /etc/caddy/Caddyfile </dev/null >/dev/null 2>&1 \
    || docker compose restart caddy
# L'image remplacee reste sur le disque, sans nom : une de plus a chaque mise a
# jour. Seules celles de JuriX sont visees.
docker image prune -f --filter label=com.docker.compose.project=jurix >/dev/null

echo "== 5. Attente de l'API (migrations)"
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

# /health ne dit rien de la recherche semantique : une cle Voyage refusee, ou
# des vecteurs purges, laisseraient le site en plein texte seul, en silence.
sante=$(docker compose exec -T api curl -fsS http://localhost:8000/api/v1/search/health </dev/null 2>/dev/null || true)
if printf '%s' "$sante" | grep -q '"embedding_service":"ready"' \
        && printf '%s' "$sante" | grep -q '"status":"healthy"'; then
    echo "Recherche semantique prete"
else
    echo "ATTENTION : recherche semantique degradee ou coupee :"
    echo "  $sante"
    echo "  Cle : VOYAGE_API_KEY dans .env, puis docker compose up -d api"
    echo "  Vecteurs manquants : docker compose exec api python scripts/regenerate_embeddings.py --all"
    echo "                       docker compose exec api python scripts/regenerate_embeddings.py --reindex"
fi

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

echo "== 7. Rendu serveur du front"
# La sonde du front ne lit qu'un fichier du build : un front qui ne joint pas
# l'API la passe, et sert pourtant ses pages en 503. Controle depuis son
# conteneur, sans Caddy : /laws (load serveur) doit lier des fiches, sans son
# bloc d'erreur, et une fiche (load universel, l'autre chemin vers l'API)
# repondre 200 avec un titre. Un front construit pour un autre domaine
# passerait ici (seuls les appels du navigateur se trompent d'adresse) :
# l'etape 1 le refuse.
# -T 15 : sans delai, wget attendrait jusqu'a 900 s une reponse qui ne vient pas.
page() { docker compose exec -T front wget -q -T 15 -O- "http://127.0.0.1:3000$1" </dev/null; }
# Le front vient peut-etre d'etre recree : il a une minute pour repondre.
limite=$((SECONDS + 60))
liste=""
until liste=$(page /laws 2>&1); do
    erreur=$liste
    liste=""
    [ "$SECONDS" -lt "$limite" ] || break
    sleep 2
done
[ -n "$liste" ] || { echo "ECHEC : le front ne rend pas /laws (${erreur:-?}) : docker compose logs front api"; exit 1; }
if [[ $liste == *laws-retry* ]]; then
    echo "ECHEC : /laws porte son bloc d'erreur (laws-retry) : le front n'a pas eu la reponse de l'API"
    exit 1
fi
lien='href="/laws/([0-9]+)"'
[[ $liste =~ $lien ]] || { echo "ECHEC : /laws ne lie aucune fiche (href=\"/laws/<numero>\") : base vide, ou rendu casse"; exit 1; }
id=${BASH_REMATCH[1]}
fiche=$(page "/laws/$id" 2>&1) || { echo "ECHEC : /laws/$id ne repond pas 200 ($fiche) : docker compose logs front api"; exit 1; }
titre='<title>[^<]+</title>'
[[ $fiche =~ $titre ]] || { echo "ECHEC : /laws/$id rendue sans <title>"; exit 1; }
echo "Rendu serveur : /laws et /laws/$id complets"

journal ok
echo "Journal des deploiements : $(tail -n 1 sauvegardes/deploiements.log)"

docker compose ps
