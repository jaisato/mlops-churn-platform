# Despliegue en produccion: VPS netcup + VPN privada (WireGuard) + GHCR

Guia operativa completa para desplegar **mlops-churn-platform** en un VPS de netcup cuya
superficie publica queda reducida a SSH y WireGuard: la API solo es accesible
desde dentro de la VPN privada.

## Arquitectura de despliegue

```
                 push tag vX.Y.Z
 Desarrollador ────────────────▶ GitHub
                                   │
                     ┌─────────────┴──────────────┐
                     │ GitHub Actions             │
                     │ 1. lint + tests            │
                     │ 2. build imagen Docker     │
                     │ 3. push a ghcr.io          │
                     │ 4. wg-quick up (VPN)       │
                     │ 5. ssh deploy@VPS          │
                     └─────────────┬──────────────┘
                                   │ tunel WireGuard (UDP 51820)
                 ┌─────────────────▼───────────────────┐
                 │ VPS netcup (Ubuntu 24.04)           │
                 │  ufw: solo 22/tcp y 51820/udp       │
                 │  wg0: 10.8.0.1/24 (red privada)     │
                 │  docker compose (prod):             │
                 │    caddy (TLS, bind IP VPN)         │
                 │    api (scoring) + mlflow + trainer │
                 └─────────────────────────────────────┘
```

## 1. Preparar el VPS (una sola vez)

1. Crea el VPS en netcup (recomendado: >= 4 vCPU / 8 GB RAM / 160 GB SSD) con Ubuntu 24.04.
2. Copia el proyecto y ejecuta el aprovisionamiento:

```bash
scp -r deploy root@IP_PUBLICA_VPS:/root/deploy
ssh root@IP_PUBLICA_VPS "bash /root/deploy/scripts/setup-vps.sh"
```

El script instala Docker, configura `ufw` (solo SSH + WireGuard expuestos),
`fail2ban`, crea el usuario `deploy` y levanta el servidor WireGuard `wg0`.

3. Anade tu clave publica SSH a `/home/deploy/.ssh/authorized_keys`.

> **Docker y ufw**: Docker publica los puertos manipulando iptables directamente, de modo
> que `ufw deny incoming` **no** protege los puertos publicados en `0.0.0.0`. Por eso
> `docker-compose.prod.yml` obliga a definir `VPN_BIND_IP` (Caddy) y liga la API y
> MLflow a `127.0.0.1`. Nunca publiques un puerto sin IP de bind en este VPS.

## 2. Configurar la VPN privada

Genera un par de claves por cliente (tu portatil y el runner de CI):

```bash
wg genkey | tee cliente.key | wg pubkey > cliente.pub
```

En el VPS, anade un bloque `[Peer]` en `/etc/wireguard/wg0.conf` por cliente y
recarga: `systemctl restart wg-quick@wg0`.

Config del cliente (la usaras tambien como secreto `WG_CONFIG` en GitHub):

```ini
[Interface]
PrivateKey = <contenido de cliente.key>
Address = 10.8.0.10/32
DNS = 1.1.1.1

[Peer]
PublicKey = <clave publica del servidor>
Endpoint = IP_PUBLICA_VPS:51820
AllowedIPs = 10.8.0.0/24
PersistentKeepalive = 25
```

> Con `AllowedIPs = 10.8.0.0/24` solo el trafico hacia la red privada pasa por
> el tunel (split-tunnel), suficiente para desplegar y consumir la API.

## 3. Publicacion de imagenes en GHCR

El workflow `.github/workflows/deploy.yml` publica la imagen en
`ghcr.io/<tu_usuario>/mlops-churn-platform` al crear un tag `vX.Y.Z`, con las
etiquetas `X.Y.Z` (sin la `v`), `sha-<commit>` y `latest`.

En el VPS, autentica el pull (token clasico con scope `read:packages`):

```bash
echo "<GHCR_PAT>" | docker login ghcr.io -u <tu_usuario> --password-stdin
```

## 4. Primera instalacion en el VPS

```bash
ssh deploy@10.8.0.1            # ya dentro de la VPN
sudo mkdir -p /opt/mlops-churn-platform && sudo chown deploy:deploy /opt/mlops-churn-platform
git clone https://github.com/<tu_usuario>/mlops-churn-platform.git /opt/mlops-churn-platform   # o scp del repo
cd /opt/mlops-churn-platform
cp .env.example .env && chmod 600 .env
```

Edita `.env`:

| Variable            | Valor                                                                   |
|---------------------|-------------------------------------------------------------------------|
| `GITHUB_OWNER`      | Tu usuario/organizacion de GitHub (propietario de la imagen en GHCR)    |
| `TAG`               | Tag por defecto para despliegues manuales (`latest`)                    |
| `CHURN_ADMIN_TOKEN` | Token aleatorio de **>= 16 caracteres**: `openssl rand -hex 32`. La API rechaza arrancar en produccion con el valor de ejemplo. |
| `VPN_BIND_IP`       | IP del VPS dentro de la VPN (`10.8.0.1`). Obligatoria.                  |

Y despliega:

```bash
TAG=latest ./deploy/scripts/deploy.sh
```

`deploy.sh` descarga la imagen, levanta el stack y espera a que `/health` responda.
En la **primera instalacion no hay modelo** en el volumen `/models`: el script lo
detecta (la API responde 503), ejecuta el entrenamiento inicial con el perfil
`train`, recarga la API y solo entonces da el despliegue por bueno. Si la API no
llega a responder 200, hace rollback al tag anterior y termina con error.

La API queda servida por Caddy con TLS en `https://10.8.0.1` (certificado interno
autofirmado de Caddy) y MLflow en `https://10.8.0.1/mlflow/`.

## 5. Secretos de GitHub Actions

Configura en *Settings -> Secrets and variables -> Actions*:

| Secreto           | Contenido                                                        |
|-------------------|------------------------------------------------------------------|
| `WG_CONFIG`       | Config WireGuard completa del peer del runner (fichero .conf)     |
| `DEPLOY_HOST`     | IP del VPS dentro de la VPN (p. ej. `10.8.0.1`)                   |
| `DEPLOY_USER`     | `deploy`                                                          |
| `DEPLOY_SSH_KEY`  | Clave privada ed25519 cuyo par publico esta en el VPS             |

> Alternativa recomendada si prefieres no abrir WireGuard a los runners
> efimeros: instala un **runner self-hosted** dentro de la VPN y elimina el
> paso del tunel; el workflow funciona igual.

## 6. Operacion del dia a dia

| Accion              | Comando                                                                 |
|---------------------|-------------------------------------------------------------------------|
| Desplegar version   | `git tag v1.2.0 && git push --tags` (todo lo demas es automatico)       |
| Despliegue manual   | `TAG=v1.2.0 ./deploy/scripts/deploy.sh` en `/opt/mlops-churn-platform` (la `v` es opcional) |
| Rollback            | `TAG=$(cat .previous_tag) ./deploy/scripts/deploy.sh`                   |
| Version en servicio | `cat .current_tag`                                                      |
| Reentrenar          | `docker compose -f docker-compose.prod.yml --profile train run --rm trainer` y despues `curl -fsS -X POST localhost:8010/model/reload -H "X-Admin-Token: $CHURN_ADMIN_TOKEN"`. Si el nuevo modelo no supera el gate de calidad (`CHURN_MIN_ROC_AUC`), el trainer termina con codigo 2 y el modelo en servicio no se toca. |
| Drift               | `curl -s localhost:8010/monitoring/drift \| python3 -m json.tool`       |
| Logs                | `docker compose -f docker-compose.prod.yml logs -f --tail 100`          |
| Estado              | `docker compose -f docker-compose.prod.yml ps`                          |
| Backup de volumenes | `docker run --rm -v <vol>:/v -v /backup:/b alpine tar czf /b/vol.tgz /v`|

Notas:

- `deploy.sh` da prioridad al `TAG` pasado por el invocador sobre el de `.env`.
- El estado del despliegue vive en `.current_tag` (tag en servicio) y `.previous_tag`
  (destino del rollback); ambos estan ignorados por git.

## 7. Checklist de seguridad

- [x] `ufw` deniega todo excepto 22/tcp y 51820/udp; Caddy solo escucha en la IP de la VPN y la API/MLflow en `127.0.0.1` (Docker no pasa por ufw: ver nota de la seccion 1).
- [x] La API rechaza arrancar en produccion con un token de administracion vacio, de ejemplo o corto; la comparacion del token es en tiempo constante.
- [x] Contenedores con usuario no root y limites de memoria.
- [x] Imagenes ancladas por tag inmutable (semver + sha) publicadas en GHCR.
- [x] Secretos solo en GitHub Secrets y `.env` del VPS (permisos 600, fuera de git).
- [x] `fail2ban` activo para SSH; desactiva el login por password en `/etc/ssh/sshd_config`.
- [ ] Recomendado: backups programados de volumenes via cron + restic/borg a almacenamiento externo.
