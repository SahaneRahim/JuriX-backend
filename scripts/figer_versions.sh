#!/usr/bin/env bash
# Regenere constraints.txt : les versions EXACTES que l'image Docker installe,
# prises dans le venv de developpement, la ou la suite de tests passe.
#
#   bash scripts/figer_versions.sh
#
# A relancer apres chaque changement de requirements.txt ou du venv, puis
# commiter constraints.txt.
#
# Pourquoi : requirements.txt donne des fourchettes, et l'image prenait les
# dernieres versions du jour de sa construction. SQLAlchemy 2.1, sorti entre
# deux, n'installait plus greenlet : l'image plantait au demarrage alors que
# tout passait en developpement, en 2.0.52.
#
# La resolution se fait dans l'image de base du Dockerfile (python:3.11-slim),
# avec les versions du venv pour contraintes : constraints.txt contient tout ce
# que requirements.txt fait installer, dependances comprises, et rien d'autre.
set -euo pipefail
cd "$(dirname "$0")/.."

venv=$(mktemp)
rapport=$(mktemp)
nouveau=$(mktemp)
trap 'rm -f "$venv" "$rapport" "$nouveau"' EXIT
# Un paquet installe depuis un fichier local (« nom @ file:///... », conda)
# n'est pas une contrainte valable : il est laisse libre.
.venv/bin/pip freeze | grep -v ' @ ' > "$venv"

# Le rapport passe par un fichier : en argument, il depasse la taille
# permise d'une ligne de commande.
docker run --rm \
    -v "$PWD/requirements.txt:/w/requirements.txt:ro" -v "$venv:/w/venv.txt:ro" -w /w \
    python:3.11-slim \
    pip install --dry-run --ignore-installed --quiet --report - \
        -r requirements.txt -c venv.txt > "$rapport"

{
    echo "# Versions exactes installees dans l'image (Dockerfile : pip install -c)."
    echo "# Generees par scripts/figer_versions.sh depuis le venv de developpement :"
    echo "# ne pas modifier a la main."
    python3 -c '
import json, sys
paquets = json.load(open(sys.argv[1]))["install"]
for p in sorted(paquets, key=lambda p: p["metadata"]["name"].lower()):
    print(p["metadata"]["name"] + "==" + p["metadata"]["version"])
' "$rapport"
} > "$nouveau"
# Remplace seulement en cas de succes : un echec en cours de route laissait un
# constraints.txt reduit a son en-tete.
chmod 644 "$nouveau"
mv "$nouveau" constraints.txt

echo "constraints.txt : $(grep -vc '^#' constraints.txt) paquets"
