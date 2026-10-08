#!/bin/sh

# Installed by InitCertificate.sh as:
# /etc/letsencrypt/renewal-hooks/deploy/restart-drawthings-tls.sh
#
# Certbot runs this executable automatically after a successful renewal.
# Test the complete renewal and reload path with:
# certbot renew --dry-run --run-deploy-hooks

set -eu

CERT_ROOT="/etc/letsencrypt"
CERT_DOMAIN="compute.drawthings.ai"

# proxy_service runs as appuser and follows the live symlinks into archive.
chmod 0711 "$CERT_ROOT/archive"
chmod 0755 "$CERT_ROOT/archive/$CERT_DOMAIN"
chmod 0644 "$CERT_ROOT/live/$CERT_DOMAIN/privkey.pem"

# Both services load the certificate and private key into memory at startup.
/usr/bin/docker restart proxy_service
/usr/bin/docker restart envoy_grpc_web_proxy
