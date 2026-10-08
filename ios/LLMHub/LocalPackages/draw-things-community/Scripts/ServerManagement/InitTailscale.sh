#!/bin/bash

echo "➡️ Installing and configuring Tailscale..."

# Install Tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# Start Tailscale and print login URL
echo "🔴 ACTION REQUIRED: Please log in to Tailscale in your browser."
tailscale up --ssh

# Get the Tailscale IP address for the proxy container
TS_IP=$(tailscale ip -4)
echo "✅ Tailscale configured. This machine's Tailscale IP is: $TS_IP"