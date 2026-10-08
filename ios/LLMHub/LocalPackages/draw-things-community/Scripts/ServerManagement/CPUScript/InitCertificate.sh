#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# The domain is now hardcoded.
DOMAIN="compute.drawthings.ai"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_HOOK_SOURCE="$SCRIPT_DIR/restart-drawthings-tls.sh"
DEPLOY_HOOK_DIR="/etc/letsencrypt/renewal-hooks/deploy"
DEPLOY_HOOK_PATH="$DEPLOY_HOOK_DIR/restart-drawthings-tls.sh"

if [ ! -f "$DEPLOY_HOOK_SOURCE" ]; then
    echo "❌ Error: Missing Certbot deploy hook: $DEPLOY_HOOK_SOURCE"
    exit 1
fi

# Prompt user for the remaining required information
read -p "Enter your email address (for Let's Encrypt notifications): " EMAIL

# --- 2. Certbot (Let's Encrypt) Installation ---
echo "➡️ Step 2: Installing Certbot to obtain SSL certificate..."

# Install snapd if not present
if ! command -v snap &> /dev/null; then
    apt install -y snapd
fi

# Install Certbot via snap
snap install certbot --classic

echo "✅ Certbot installed. Requesting certificate for $DOMAIN..."
# Obtain the certificate using standalone mode (requires port 80 to be free)
certbot certonly --standalone --non-interactive --agree-tos --email "$EMAIL" -d "$DOMAIN"
echo "✅ SSL certificate obtained successfully."

# The proxy container runs as appuser and follows Certbot's live symlinks into
# the root-owned archive directory. Make the current key readable, then install
# a deploy hook that reapplies these permissions and restarts both TLS services
# after every successful renewal.
chmod 0711 /etc/letsencrypt/archive
chmod 0755 "/etc/letsencrypt/archive/$DOMAIN"
chmod 0644 "/etc/letsencrypt/live/$DOMAIN/privkey.pem"
install -d -m 0755 "$DEPLOY_HOOK_DIR"
install -m 0755 "$DEPLOY_HOOK_SOURCE" "$DEPLOY_HOOK_PATH"
echo "✅ Certbot deploy hook installed at $DEPLOY_HOOK_PATH"
