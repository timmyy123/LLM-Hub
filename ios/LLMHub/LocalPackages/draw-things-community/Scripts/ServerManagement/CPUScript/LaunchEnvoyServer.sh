#!/bin/bash

# This script starts the Envoy proxy container.
# It requires the path to the envoy-config.yaml file as an argument.
# It should be run with sudo privileges.

# Exit immediately if a command exits with a non-zero status.
set -euo pipefail

# --- Argument Validation ---
# Check if exactly one argument is provided.
if [ "$#" -ne 1 ]; then
  echo "❌ Error: Missing Envoy configuration file path."
  echo ""
  echo "Usage:   $0 <PATH_TO_ENVOY_CONFIG>"
  echo "Example: $0 /home/user/envoy/envoy-config.yaml"
  exit 1
fi

# --- Configuration ---
ENVOY_CONFIG_PATH="$1"
ENVOY_IMAGE="envoyproxy/envoy:v1.39.1@sha256:be87c8b52663c1164a5bdf3c5419017a269cb3d8c74be1ec93638a71f1ffbd4b"
CONTAINER_NAME="envoy_grpc_web_proxy"

# Check if the provided Envoy config file actually exists.
if [ ! -f "$ENVOY_CONFIG_PATH" ]; then
  echo "❌ Error: The Envoy configuration file was not found at the specified path:"
  echo "   $ENVOY_CONFIG_PATH"
  exit 1
fi

echo "✅ Envoy configuration file found."

# Fail before stopping the service if a required command or image is unavailable.
command -v curl >/dev/null
docker image inspect "$ENVOY_IMAGE" >/dev/null 2>&1 || docker pull "$ENVOY_IMAGE"

# Each container keeps its own configuration, outside the proxy's writable mount.
install -d -m 0755 /etc/drawthings/envoy
INSTALLED_CONFIG=$(mktemp /etc/drawthings/envoy/envoy-config.XXXXXXXX.yaml)
install -m 0644 "$ENVOY_CONFIG_PATH" "$INSTALLED_CONFIG"
docker run --rm --network none --read-only --cap-drop ALL \
  --security-opt no-new-privileges --entrypoint /usr/local/bin/envoy \
  -v "$INSTALLED_CONFIG:/etc/envoy/envoy.yaml:ro" \
  -v /etc/letsencrypt:/etc/letsencrypt:ro \
  "$ENVOY_IMAGE" --mode validate --disable-hot-restart --concurrency 2 \
  -c /etc/envoy/envoy.yaml

ROLLBACK_CONTAINER=""
if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  ROLLBACK_CONTAINER="${CONTAINER_NAME}_rollback_$(date -u +%Y%m%dT%H%M%SZ)_$$"
  echo "Stopping Envoy; preserving it as $ROLLBACK_CONTAINER..."
  docker stop --timeout 60 "$CONTAINER_NAME"
  if ! docker rename "$CONTAINER_NAME" "$ROLLBACK_CONTAINER"; then
    docker start "$CONTAINER_NAME"
    exit 1
  fi
fi

restore_previous() {
  local status=$?
  trap - EXIT
  if [ "$status" -ne 0 ]; then
    echo "❌ Envoy replacement failed."
    if docker container inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
      docker stop --timeout 10 "$CONTAINER_NAME" || true
      docker rename "$CONTAINER_NAME" "${CONTAINER_NAME}_failed_$(date -u +%Y%m%dT%H%M%SZ)_$$" || true
    fi
    if [ -n "$ROLLBACK_CONTAINER" ]; then
      docker rename "$ROLLBACK_CONTAINER" "$CONTAINER_NAME" && docker start "$CONTAINER_NAME"
    fi
  fi
  exit "$status"
}
trap restore_previous EXIT

echo "🚀 Starting Envoy proxy container..."
docker run -d \
  --name "$CONTAINER_NAME" \
  --network host \
  --restart unless-stopped \
  -v "$INSTALLED_CONFIG:/etc/envoy/envoy.yaml:ro" \
  -v /etc/letsencrypt:/etc/letsencrypt:ro \
  "$ENVOY_IMAGE"

for attempt in {1..30}; do
  if curl --fail --silent --show-error --max-time 2 http://127.0.0.1:9901/ready >/dev/null 2>&1; then
    echo "✅ Envoy is ready. Configuration: $INSTALLED_CONFIG"
    if [ -n "$ROLLBACK_CONTAINER" ]; then
      echo "Rollback container: $ROLLBACK_CONTAINER"
    fi
    exit 0
  fi
  sleep 1
done

echo "❌ Envoy did not become ready within the startup checks."
exit 1
