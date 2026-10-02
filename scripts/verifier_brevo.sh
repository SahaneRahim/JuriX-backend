#!/usr/bin/env bash
# Verifie que la configuration Brevo fonctionne, SANS jamais afficher la cle.
#
# Lit .env (chmod 600, ignore par git). N'ecrit rien, n'envoie aucun message :
# il interroge le compte et la liste des expediteurs, et dit ce qui manque.
#
# Usage :  bash scripts/verifier_brevo.sh

set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

[ -f .env ] || { echo "✗ Aucun fichier .env ici."; exit 1; }
# shellcheck disable=SC1091
set -a; . ./.env; set +a

if [ -z "${BREVO_API_KEY:-}" ]; then
  echo "✗ BREVO_API_KEY est vide dans .env."
  echo "  Tableau de bord Brevo : Settings > SMTP & API > API Keys > Generate a new API key"
  exit 1
fi
echo "✓ BREVO_API_KEY presente (${#BREVO_API_KEY} caracteres, non affichee)"

# ---- 1. La cle est-elle acceptee ? ----
compte=$(curl -s -w '\n%{http_code}' https://api.brevo.com/v3/account \
  -H "api-key: ${BREVO_API_KEY}" -H 'accept: application/json')
code=$(printf '%s' "$compte" | tail -1)
corps=$(printf '%s' "$compte" | sed '$d')

case "$code" in
  200) echo "✓ Cle acceptee par Brevo" ;;
  401) echo "✗ Cle refusee (401). Elle n'est visible qu'a la creation : si elle a ete perdue, il faut en generer une autre."; exit 1 ;;
  *)   echo "✗ Reponse inattendue ($code) : $(printf '%s' "$corps" | head -c 200)"; exit 1 ;;
esac

# Credits restants du jour, s'ils sont exposes.
restants=$(printf '%s' "$corps" | grep -o '"credits":[0-9]*' | head -1 | cut -d: -f2)
[ -n "$restants" ] && echo "  credits annonces : $restants"

# ---- 2. L'expediteur est-il VERIFIE ? ----
# C'est le point qui bloque le plus souvent : la cle est bonne, mais l'envoi
# echoue parce que l'adresse d'expedition n'a jamais ete validee par son lien.
if [ -z "${BREVO_SENDER_EMAIL:-}" ]; then
  echo "✗ BREVO_SENDER_EMAIL est vide. Sans expediteur, rien ne peut partir."
  exit 1
fi

expediteurs=$(curl -s https://api.brevo.com/v3/senders \
  -H "api-key: ${BREVO_API_KEY}" -H 'accept: application/json')

if printf '%s' "$expediteurs" | grep -qiF "\"${BREVO_SENDER_EMAIL}\""; then
  echo "✓ Expediteur ${BREVO_SENDER_EMAIL} declare dans Brevo"
  # `active: true` signifie que le lien de validation a bien ete clique.
  if printf '%s' "$expediteurs" | tr '}' '\n' | grep -iF "${BREVO_SENDER_EMAIL}" | grep -q '"active":true'; then
    echo "✓ Expediteur VERIFIE — l'envoi est possible"
  else
    echo "✗ Expediteur declare mais NON verifie."
    echo "  Brevo a envoye un lien de validation a ${BREVO_SENDER_EMAIL} : il faut cliquer dessus."
  fi
else
  echo "✗ ${BREVO_SENDER_EMAIL} n'est pas dans la liste des expediteurs."
  echo "  Brevo : Senders, Domains & Dedicated IPs > Senders > Add a sender"
  printf '  Adresses declarees : %s\n' "$(printf '%s' "$expediteurs" | grep -o '"email":"[^"]*"' | cut -d'"' -f4 | paste -sd', ')"
fi

# ---- 3. Le lien des courriels pointe-t-il quelque part ? ----
if [ -z "${FRONTEND_BASE_URL:-}" ]; then
  echo "✗ FRONTEND_BASE_URL est vide : aucun message porteur d'un lien ne partira."
  echo "  C'est voulu — mieux vaut ne rien envoyer qu'expedier un lien mort."
else
  echo "✓ FRONTEND_BASE_URL = ${FRONTEND_BASE_URL}"
fi
