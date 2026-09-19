#!/usr/bin/env bash
# =============================================================================
# Aprovisionamiento inicial del VPS de netcup (Ubuntu 22.04/24.04).
# Ejecutar UNA VEZ como root:  bash setup-vps.sh
# Hace: hardening basico, Docker, WireGuard (servidor VPN) y estructura /opt.
# =============================================================================
set -euo pipefail

PROJECT="mlops-churn-platform"
DEPLOY_USER="deploy"
WG_PORT=51820
WG_NET="10.8.0"

echo "[1/6] Actualizacion del sistema y utilidades"
apt-get update -y && apt-get upgrade -y
apt-get install -y ufw fail2ban curl git unattended-upgrades wireguard qrencode

echo "[2/6] Usuario de despliegue sin privilegios interactivos"
if ! id "$DEPLOY_USER" &>/dev/null; then
  adduser --disabled-password --gecos "" "$DEPLOY_USER"
  mkdir -p /home/$DEPLOY_USER/.ssh && chmod 700 /home/$DEPLOY_USER/.ssh
  touch /home/$DEPLOY_USER/.ssh/authorized_keys && chmod 600 /home/$DEPLOY_USER/.ssh/authorized_keys
  chown -R $DEPLOY_USER:$DEPLOY_USER /home/$DEPLOY_USER/.ssh
  echo ">> Anade la clave publica de despliegue a /home/$DEPLOY_USER/.ssh/authorized_keys"
fi

echo "[3/6] Docker Engine + plugin compose"
if ! command -v docker &>/dev/null; then
  curl -fsSL https://get.docker.com | sh
fi
usermod -aG docker "$DEPLOY_USER"

echo "[4/6] Firewall: SSH + WireGuard publicos; el resto SOLO por VPN"
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp comment 'SSH (mover a puerto alto si se desea)'
ufw allow ${WG_PORT}/udp comment 'WireGuard'
ufw allow in on wg0 comment 'Trafico interno VPN'
ufw --force enable

echo "[5/6] WireGuard servidor (red privada ${WG_NET}.0/24)"
if [ ! -f /etc/wireguard/wg0.conf ]; then
  umask 077
  wg genkey | tee /etc/wireguard/server.key | wg pubkey > /etc/wireguard/server.pub
  cat > /etc/wireguard/wg0.conf <<WG
[Interface]
Address = ${WG_NET}.1/24
ListenPort = ${WG_PORT}
PrivateKey = $(cat /etc/wireguard/server.key)
# Anade un bloque [Peer] por cada cliente (portatil, runner CI...):
# [Peer]
# PublicKey = <clave publica del cliente>
# AllowedIPs = ${WG_NET}.X/32
WG
  systemctl enable --now wg-quick@wg0
  echo ">> Clave publica del servidor: $(cat /etc/wireguard/server.pub)"
fi

echo "[6/6] Estructura de despliegue"
mkdir -p /opt/${PROJECT}
chown -R $DEPLOY_USER:$DEPLOY_USER /opt/${PROJECT}

echo "Listo. Pasos siguientes: ver deploy/DESPLIEGUE_NETCUP.md"
