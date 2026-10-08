#!/bin/bash

# This script automates the setup of a Draw Things proxy server environment.
# It installs Docker, Certbot, Tailscale, and configures the necessary containers.
# The domain name is hardcoded to compute.drawthings.ai.
# It should be run with sudo privileges.

# Exit immediately if a command exits with a non-zero status.
set -e

# The domain is now hardcoded.
DOMAIN="compute.drawthings.ai"

echo "-----------------------------------------------------"
echo "Starting setup for the : $DOMAIN"
echo "-----------------------------------------------------"
sleep 2

# --- 1. System & Docker Installation ---
echo "➡️ Step 1: Updating system and installing Docker..."

# Update package index and install prerequisites
apt update
apt install -y apt-transport-https ca-certificates curl software-properties-common

# Add Docker's official GPG key
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update package index again and install Docker
apt update
apt install -y docker-ce docker-ce-cli containerd.io

# Add current user to the docker group (requires logout/login to take effect)
if [ -n "$SUDO_USER" ]; then
    usermod -aG docker "$SUDO_USER"
    echo "✅ User $SUDO_USER added to the docker group. You'll need to log out and back in for this to take effect."
fi
echo "✅ Docker installed successfully."

# --- 3. Tailscale Installation & Configuration ---
echo "➡️ Step 3: Installing and configuring Tailscale..."

# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# Start Tailscale and print login URL
echo "🔴 ACTION REQUIRED: Please log in to Tailscale in your browser."
tailscale up --ssh

# Get the Tailscale IP address for the proxy container
TS_IP=$(tailscale ip -4)
echo "✅ Tailscale configured. This machine's Tailscale IP is: $TS_IP"


# --- 4. Security & Directory Setup ---
echo "➡️ Step 4: Applying security settings and preparing directories..."

# Lock the root user's password for enhanced security
passwd -l root
echo "✅ Root password locked."


# --- 5. Start Services (Docker Containers) ---
echo "➡️ Step 5: Pulling images and starting services..."

# Pull the required Docker images
docker pull drawthingsai/draw-things-proxy-server-cli:latest
docker pull envoyproxy/envoy:v1.39.1@sha256:be87c8b52663c1164a5bdf3c5419017a269cb3d8c74be1ec93638a71f1ffbd4b

echo "✅ Docker images pulled."

echo "CPU Server init finished."
