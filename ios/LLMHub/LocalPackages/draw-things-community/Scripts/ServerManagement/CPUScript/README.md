# CPU Script Setup Guide

## Initial Setup

Initialize CPU server with dependencies (Docker, Docker image, Certbot, and Envoy):

```bash
/path/to/CPUScript/InitCPUServer.sh
```

This script installs all required dependencies and configures `envoy-config.yaml` for the gRPC web proxy.

## Certificate Setup

Initialize SSL certificates using Let's Encrypt:

```bash
/path/to/CPUScript/InitCertificate.sh
```

**Important:** Certificates are created in `/etc/letsencrypt/` and are required for both Envoy and the Proxy Server to function properly.

**Note:** Creating new certificates should not invalidate existing ones (per Gemini and Claude documentation), though this hasn't been fully tested.

## Tailscale Setup

Configure Tailscale for GPU server connectivity:

```bash
/path/to/InitTailscale.sh
```

**Required:** Tailscale must be configured for the CPU server to connect to GPU servers.

## Launch Services

### CPU Proxy Server

Start the ProxyServiceCLI CPU proxy server:

```bash
/path/to/CPUScript/LaunchCPUServer.sh <DATADOG_API_KEY> <PATH_TO_MODEL_LIST_FILE> [PROXY_SERVICE_ARGS...]
```

**Example:**
```bash
./LaunchCPUServer.sh 'dd_api_123abc...' /home/user/Documents/model-list \
  -g 100.70.225.70:40001-40008:1 \
  -g 100.111.187.6:40001-40008:1 \
  -g 100.115.89.107:40001-40008:1 \
  -g 100.112.144.13:40001-40008:1 \
  -g 100.121.197.14:40001-40008:1 \
  -g 100.110.198.32:40001-40008:1
```

The launcher uses Envoy 1.39.1 pinned to its Linux amd64 image digest. It copies the configuration into `/etc/drawthings/envoy` and validates it before stopping the existing service. The previous container is retained under a timestamped rollback name. If replacement startup or readiness fails, the launcher restores the previous container.

After an upgrade, verify public TLS, the binary/text gRPC-web `Pubkey` response, CORS preflight, and the 403/404 route restrictions. The canonical container name remains `envoy_grpc_web_proxy` so the certificate-renewal hook continues to find it.
The script mounts the directory containing `model-list` into the container as `/app/Documents`
so host-side edits and `update-model-list` control-panel writes target the same file.

### Envoy Proxy Server

Start the Envoy gRPC web proxy:

```bash
/path/to/CPUScript/LaunchEnvoyServer.sh <PATH_TO_ENVOY_CONFIG>
```

**Example:**
```bash
./LaunchEnvoyServer.sh /home/user/envoy/envoy-config.yaml
```

**Note:** This will stop and remove any existing container before starting a new one.

## Model List

A default model list based on September 11, 2025 GPU server configurations is included for both services.
