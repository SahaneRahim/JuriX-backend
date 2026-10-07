#!/usr/bin/env bash
# Prepare une VM Ubuntu 24.04 (ARM, Oracle Cloud « Always Free ») pour JuriX.
# A lancer UNE fois, SUR la VM, depuis deploiement/oracle :
#   bash installer.sh
# Idempotent : le relancer ne casse rien.
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1. Docker et Docker Compose (depots Ubuntu)"
sudo apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q \
    docker.io docker-compose-v2 iptables-persistent rsync
sudo systemctl enable --now docker
sudo usermod -aG docker "$USER"

echo "== 2. Pare-feu de la VM : ports 80 et 443"
# Les images Ubuntu d'Oracle portent leur propre pare-feu (iptables), qui
# rejette tout sauf SSH. Ouvrir les ports dans la console Oracle (liste de
# securite) ne suffit pas : il faut AUSSI les ouvrir ici, avant la regle REJECT.
ouvrir() {
    local proto=$1 port=$2
    if ! sudo iptables -C INPUT -p "$proto" --dport "$port" -j ACCEPT 2>/dev/null; then
        local rang
        rang=$(sudo iptables -L INPUT --line-numbers | awk '/REJECT/ {print $1; exit}')
        sudo iptables -I INPUT "${rang:-1}" -p "$proto" --dport "$port" -j ACCEPT
    fi
}
ouvrir tcp 80
ouvrir tcp 443
ouvrir udp 443
sudo netfilter-persistent save

echo "== 3. Memoire d'appoint (4 Go de swap)"
# Filet pour la construction de l'image et les pointes : sans lui, un depassement
# fait tuer un conteneur au lieu de ralentir.
if ! swapon --show | grep -q /swapfile; then
    sudo fallocate -l 4G /swapfile
    sudo chmod 600 /swapfile
    sudo mkswap /swapfile
    sudo swapon /swapfile
    grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
fi

echo "== 4. Dossiers des donnees"
mkdir -p donnees/uploads modeles sauvegardes front
# L'API tourne sous l'utilisateur 1000 dans son conteneur : elle doit pouvoir
# ecrire les PDF envoyes.
sudo chown -R 1000:1000 donnees

echo "== 5. Sauvegarde quotidienne de la base (03:15)"
ligne="15 3 * * * cd $(pwd) && bash sauvegarde.sh >> sauvegardes/sauvegarde.log 2>&1"
( crontab -l 2>/dev/null | grep -v 'sauvegarde.sh' ; echo "$ligne" ) | crontab -

echo
echo "Pret. Se deconnecter puis se reconnecter (groupe docker), puis :"
echo "  cp env.production.exemple .env   # et le renseigner"
echo "  docker compose up -d --build"
