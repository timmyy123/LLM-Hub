#!/bin/bash

# This script automates the setup of a Draw Things proxy server environment.
# It requires a Datadog API key and a path to a model list file as arguments.
# It should be run with sudo privileges.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Argument Validation ---
# Check if at least two arguments are provided.
if [ "$#" -lt 2 ]; then
  echo "❌ Error: Missing required arguments."
  echo "This script requires a Datadog API key and a path to your model list file."
  echo ""
  echo "Usage:   $0 <DATADOG_API_KEY> <PATH_TO_MODEL_LIST_FILE> [PROXY_SERVICE_ARGS...]"
  echo "Example: $0 'dd_api_123abc...' /home/user/Documents/model-list -g 100.121.197.14:40001-40008:1"
  exit 1
fi

# --- Configuration ---
DATADOG_KEY="$1"
MODEL_LIST_PATH="$2"
shift 2
EXTRA_PROXY_SERVICE_ARGS=("$@")
DOMAIN="compute.drawthings.ai"

# Check if the provided model list file actually exists.
if [ ! -f "$MODEL_LIST_PATH" ]; then
  echo "❌ Error: The model list file was not found at the specified path:"
  echo "   $MODEL_LIST_PATH"
  exit 1
fi

echo "✅ Datadog API key and model list path provided."

MODEL_LIST_DIR="$(cd "$(dirname "$MODEL_LIST_PATH")" && pwd)"
MODEL_LIST_FILE="$(basename "$MODEL_LIST_PATH")"
CONTAINER_MODEL_LIST_DIR="/app/Documents"
CONTAINER_MODEL_LIST_PATH="$CONTAINER_MODEL_LIST_DIR/$MODEL_LIST_FILE"

# The control-panel update-model-list command writes the model list atomically.
# Keep the mounted directory writable by the container's non-root app user.
chmod a+rwx "$MODEL_LIST_DIR"
chmod a+rw "$MODEL_LIST_PATH"

TS_IP=$(tailscale ip -4)
echo "✅ Tailscale configured. This machine's Tailscale IP is: $TS_IP"
echo "DOMAIN : $DOMAIN"
echo "MODEL_LIST_PATH : $MODEL_LIST_PATH"
echo "CONTAINER_MODEL_LIST_PATH : $CONTAINER_MODEL_LIST_PATH"


# Start the Draw Things proxy service container
echo "Stopping Draw Things proxy service container..."
docker stop proxy_service || true
echo "Removing Draw Things proxy service container..."
docker rm proxy_service || true
echo "🚀 Starting Draw Things proxy service container..."
docker run -d \
  -p 443:8080 \
  -p "$TS_IP:50002:50000" \
  --name proxy_service \
  --restart unless-stopped \
  -v /etc/letsencrypt:/etc/letsencrypt:ro \
  -v "$MODEL_LIST_DIR:$CONTAINER_MODEL_LIST_DIR" \
  -u appuser \
  drawthingsai/draw-things-proxy-server-cli \
  /usr/local/bin/ProxyServiceCLI \
  --model-list-path "$CONTAINER_MODEL_LIST_PATH" \
  -p 8080 \
  --control-port 50000 \
  --cert "/etc/letsencrypt/live/$DOMAIN/fullchain.pem" \
  --key "/etc/letsencrypt/live/$DOMAIN/privkey.pem" \
  -d "$DATADOG_KEY" \
  "${EXTRA_PROXY_SERVICE_ARGS[@]}"
