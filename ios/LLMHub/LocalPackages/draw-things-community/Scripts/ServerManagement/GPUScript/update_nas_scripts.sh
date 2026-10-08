#!/bin/bash

# Script to upload UpdateModels scripts to the remote NAS server.
# Cleans the remote directory before copying for a fresh update.
# Usage: ./update_nas_scripts.sh root@hostname

# Exit immediately if a command exits with a non-zero status.
set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SOURCE_DIR="$SCRIPT_DIR/UpdateModels"
REMOTE_DIR="/root/utils/UpdateModels"

REMOTE_HOST="$1"

if [ -z "$REMOTE_HOST" ]; then
    echo "Usage: $0 root@hostname"
    echo ""
    echo "Copies UpdateModels/ to $REMOTE_DIR on the remote NAS server."
    echo "The remote directory is cleaned before copying."
    exit 1
fi

echo "=================================================="
echo "  Upload UpdateModels to NAS Server"
echo "=================================================="
echo "Target: $REMOTE_HOST"
echo ""

# Check if ssh is available
if ! command -v ssh &> /dev/null; then
    echo "Error: ssh command not found. Please install openssh-client."
    exit 1
fi

# Test SSH connection
echo "Testing SSH connection to $REMOTE_HOST..."
if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$REMOTE_HOST" "echo 'Connection successful'" 2>/dev/null < /dev/null; then
    echo "Error: Cannot connect to $REMOTE_HOST"
    echo "Please ensure:"
    echo "  1. The host is reachable"
    echo "  2. SSH keys are set up for passwordless authentication"
    echo "  3. You have root access"
    exit 1
fi
echo "✅ SSH connection successful"
echo ""

if [ ! -d "$SOURCE_DIR" ]; then
    echo "Error: Source directory not found: $SOURCE_DIR"
    exit 1
fi

# Clean remote directory and copy fresh
echo "Cleaning remote directory: $REMOTE_HOST:$REMOTE_DIR"
ssh "$REMOTE_HOST" "rm -rf '$REMOTE_DIR' && mkdir -p '$REMOTE_DIR'" < /dev/null

echo "Copying $SOURCE_DIR -> $REMOTE_HOST:$REMOTE_DIR"
scp -rq "$SOURCE_DIR"/. "$REMOTE_HOST:$REMOTE_DIR/"

# Make .sh and .py files executable
ssh "$REMOTE_HOST" "find '$REMOTE_DIR' -name '*.sh' -exec chmod +x {} \; && find '$REMOTE_DIR' -name '*.py' -exec chmod +x {} \;" < /dev/null

echo ""
echo "=================================================="
echo "✅ Upload complete!"
echo ""
echo "Files are located at: $REMOTE_HOST:$REMOTE_DIR"
