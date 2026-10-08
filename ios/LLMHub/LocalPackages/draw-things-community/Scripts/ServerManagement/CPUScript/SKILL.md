---
name: setup-cpu-proxy-server
description: Set up and verify a new Draw Things CPU proxy and Envoy server using the scripts in Scripts/ServerManagement/CPUScript. Use when Codex needs to bootstrap a root-access Linux CPU server over SSH/Tailscale, upload model-list and envoy-config.yaml, install Docker/Tailscale images, install or copy compute.drawthings.ai TLS certificates, launch proxy_service and envoy_grpc_web_proxy, and debug port 443, Envoy 8443, TLS, or gRPC Echo connectivity.
---

# Draw Things CPU Proxy Server Setup

Use this workflow from `Scripts/ServerManagement/CPUScript`.

## Files

- `InitCPUServer.sh`: install/update Docker, install Tailscale, run `tailscale up --ssh`, lock the root password, and pull `drawthingsai/draw-things-proxy-server-cli:latest` plus Envoy 1.39.1 pinned to its Linux amd64 image digest.
- `InitCertificate.sh`: request a Let's Encrypt cert for `compute.drawthings.ai` with standalone certbot on port 80, set the permissions required by the non-root proxy container, and install the renewal deploy hook.
- `restart-drawthings-tls.sh`: Certbot deploy hook that repairs renewed certificate permissions and restarts `proxy_service` and `envoy_grpc_web_proxy` so both processes load the renewed certificate.
- `LaunchCPUServer.sh`: start `proxy_service` on public `443` and Tailscale control port `TS_IP:50002`.
- `LaunchEnvoyServer.sh`: copy the provided Envoy config into root-owned `/etc/drawthings/envoy`, validate it, and start `envoy_grpc_web_proxy` with host networking. Preserve the previous container for rollback and restore it automatically if replacement startup or readiness fails.
- `model-list`: mounted through its containing directory as `/app/Documents/<filename>` so host-side edits and control-panel writes target the same file.
- `envoy-config.yaml`: copied to a separate file per deployment and mounted read-only as `/etc/envoy/envoy.yaml`; uses typed CORS and upstream HTTP/2 configuration.

## Inputs

Have these ready before launch:

- New CPU server SSH target, usually `root@<tailscale-ip>`.
- Datadog API key. Treat it as a secret; do not commit or echo it in summaries.
- Let's Encrypt notification email if issuing a new cert.
- Optional existing cert source server, if DNS is not routed to the new VM yet.
- Updated `model-list`.
- A known-good `envoy-config.yaml`; the admin socket must include a port, typically `127.0.0.1:9901`.

## Initial State Check

Check SSH, OS, Docker, Tailscale, certs, containers, and ports:

```bash
ssh root@<new-ts-ip> 'hostname; uname -a; id; command -v apt || true; command -v docker || true; command -v tailscale || true; tailscale ip -4 2>/dev/null || true'
ssh root@<new-ts-ip> 'docker ps -a --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || true; ss -ltnp | grep -E ":(80|443|8443|50002)" || true; ls -la /etc/letsencrypt/live/compute.drawthings.ai 2>/dev/null || true'
```

## Upload the Script Bundle

Install the CPUScript files on the new server:

```bash
ssh root@<new-ts-ip> 'mkdir -p /root/CPUScript'
scp InitCertificate.sh InitCPUServer.sh LaunchCPUServer.sh LaunchEnvoyServer.sh restart-drawthings-tls.sh envoy-config.yaml model-list root@<new-ts-ip>:/root/CPUScript/
ssh root@<new-ts-ip> 'chmod +x /root/CPUScript/*.sh; wc -l /root/CPUScript/model-list; ls -la /root/CPUScript'
```

If the model list changes, recopy only that file and verify checksum:

```bash
scp model-list root@<new-ts-ip>:/root/CPUScript/model-list
shasum -a 256 model-list
ssh root@<new-ts-ip> 'shasum -a 256 /root/CPUScript/model-list; wc -l /root/CPUScript/model-list'
```

If the local Envoy config is stale, copy the known-good config from the current CPU server, then upload it:

```bash
scp root@<current-cpu-ts-ip>:/root/envoy/envoy-config.yaml ./envoy-config.yaml
rg -n "port_value|address:|filename:|sni:|admin:" envoy-config.yaml
scp envoy-config.yaml root@<new-ts-ip>:/root/CPUScript/envoy-config.yaml
```

## Run CPU Server Init

Run the dependency/image setup:

```bash
ssh root@<new-ts-ip> 'cd /root/CPUScript && ./InitCPUServer.sh'
```

Expect apt output, Docker install/update, Tailscale install/update, root password lock, and Docker image pulls. A pending kernel upgrade warning is not a launch blocker.

## Certificate Strategy

The scripts expect:

```text
/etc/letsencrypt/live/compute.drawthings.ai/fullchain.pem
/etc/letsencrypt/live/compute.drawthings.ai/privkey.pem
```

If `compute.drawthings.ai` already routes to the new VM and public port 80 reaches it, run:

```bash
ssh root@<new-ts-ip> 'cd /root/CPUScript && printf "%s\n" "<email>" | ./InitCertificate.sh'
```

If DNS is not routed to the new VM yet, certbot standalone will fail. In that case, copy only the active cert and key from the current CPU server. Do not copy the whole `/etc/letsencrypt` unless deliberately migrating Certbot renewal ownership; that also copies ACME account keys, renewal configs, and archived cert material.

```bash
ssh root@<current-cpu-ts-ip> 'ls -l /etc/letsencrypt/live/compute.drawthings.ai/fullchain.pem /etc/letsencrypt/live/compute.drawthings.ai/privkey.pem; openssl x509 -in /etc/letsencrypt/live/compute.drawthings.ai/fullchain.pem -noout -subject -issuer -dates'

mkdir -p /private/tmp/cpu-cert-transfer
chmod 700 /private/tmp/cpu-cert-transfer
scp root@<current-cpu-ts-ip>:/etc/letsencrypt/live/compute.drawthings.ai/fullchain.pem root@<current-cpu-ts-ip>:/etc/letsencrypt/live/compute.drawthings.ai/privkey.pem /private/tmp/cpu-cert-transfer/
chmod 644 /private/tmp/cpu-cert-transfer/fullchain.pem
chmod 600 /private/tmp/cpu-cert-transfer/privkey.pem

ssh root@<new-ts-ip> 'mkdir -p /etc/letsencrypt/live/compute.drawthings.ai && chmod 755 /etc/letsencrypt /etc/letsencrypt/live /etc/letsencrypt/live/compute.drawthings.ai'
scp /private/tmp/cpu-cert-transfer/fullchain.pem /private/tmp/cpu-cert-transfer/privkey.pem root@<new-ts-ip>:/etc/letsencrypt/live/compute.drawthings.ai/
ssh root@<new-ts-ip> 'chmod 644 /etc/letsencrypt/live/compute.drawthings.ai/fullchain.pem; chmod 644 /etc/letsencrypt/live/compute.drawthings.ai/privkey.pem; openssl x509 -in /etc/letsencrypt/live/compute.drawthings.ai/fullchain.pem -noout -subject -issuer -dates'
rm -f /private/tmp/cpu-cert-transfer/fullchain.pem /private/tmp/cpu-cert-transfer/privkey.pem
```

Use `0644` for `privkey.pem` with the current Docker launch because the proxy process inside the container may not run as root. If `proxy_service` logs `Permission denied` opening `privkey.pem`, fix permissions and recreate the container.

### Automatic Renewal and TLS Reload

Copying only `fullchain.pem` and `privkey.pem` creates usable TLS files but does not migrate Certbot renewal ownership. A managed certificate must appear in `certbot certificates` and have all of:

```text
/etc/letsencrypt/renewal/compute.drawthings.ai.conf
/etc/letsencrypt/archive/compute.drawthings.ai/
/etc/letsencrypt/live/compute.drawthings.ai/ -> ../../archive/compute.drawthings.ai/
```

`InitCertificate.sh` installs this executable deploy hook:

```text
/etc/letsencrypt/renewal-hooks/deploy/restart-drawthings-tls.sh
```

Certbot normally creates `/etc/letsencrypt/archive` as root-only and creates each renewed private key as mode `0600`. The proxy runs as `appuser`, so following the `live` symlink fails unless the archive directories are searchable and the private key is readable. The hook reapplies:

```bash
chmod 0711 /etc/letsencrypt/archive
chmod 0755 /etc/letsencrypt/archive/compute.drawthings.ai
chmod 0644 /etc/letsencrypt/live/compute.drawthings.ai/privkey.pem
```

The proxy and the statically configured Envoy listener load the certificate into memory at process startup. After a successful renewal, the deploy hook restarts both containers so they load the new certificate:

```bash
/usr/bin/docker restart proxy_service
/usr/bin/docker restart envoy_grpc_web_proxy
```

Verify the snap timer and the complete renewal-to-reload path:

```bash
systemctl is-enabled snap.certbot.renew.timer
systemctl is-active snap.certbot.renew.timer
certbot renew --dry-run --run-deploy-hooks
docker ps --filter name=proxy_service
docker ps --filter name=envoy_grpc_web_proxy
```

Expected results are `enabled`, `active`, `Congratulations, all simulated renewals succeeded`, and both containers returning to `Up`. `--run-deploy-hooks` is needed only to make a dry run execute deploy hooks; a real successful renewal automatically executes every executable in `/etc/letsencrypt/renewal-hooks/deploy`.

Confirm that ports 443 and 8443 serve the same renewed certificate:

```bash
openssl s_client -connect 127.0.0.1:443 -servername compute.drawthings.ai </dev/null 2>/dev/null | openssl x509 -noout -dates -serial
openssl s_client -connect 127.0.0.1:8443 -servername compute.drawthings.ai </dev/null 2>/dev/null | openssl x509 -noout -dates -serial
```

## Launch Services

Launch the CPU proxy first. Any extra arguments after the model-list path are passed through to `ProxyServiceCLI`, including `-g` GPU worker ranges:

```bash
ssh root@<new-ts-ip> 'cd /root/CPUScript && ./LaunchCPUServer.sh "<DATADOG_API_KEY>" /root/CPUScript/model-list -g 100.70.225.70:40001-40008:1 -g 100.115.89.107:40001-40008:1 -g 100.102.100.71:40001-40008:1 -g 100.102.159.74:40001-40008:1 -g 100.121.197.14:40001-40008:1 -g 100.110.198.32:40001-40008:1'
```

Verify each host's current Tailscale address before executing this example. The rebuilt `dfw-242-009` and `dfw-243-010` replaced the retired `100.83.253.70` and `100.64.66.84` addresses. `LaunchCPUServer.sh` stops and recreates the central proxy container, which can interrupt in-flight generation requests; use a planned maintenance window. A plain restart of an existing container retains its previously saved `-g` arguments.

`LaunchCPUServer.sh` mounts the directory containing the model list as `/app/Documents` and passes `--model-list-path /app/Documents/<filename>`. It also makes that directory and file writable by the container's non-root `appuser`; this is required because `update-model-list` writes atomically by creating a temporary file in the same directory.

Do not use the old single-file read-only mount:

```bash
-v /root/CPUScript/model-list:/app/model-list:ro
```

That mount lets the server read the file, but it prevents the control-panel `update-model-list` RPC from writing back. The server currently uses `try?` around the file write, so the CLI can appear to succeed even when the file was not updated.

Then launch Envoy:

```bash
ssh root@<new-ts-ip> 'cd /root/CPUScript && ./LaunchEnvoyServer.sh /root/CPUScript/envoy-config.yaml'
```

Expected ports:

- `proxy_service`: `0.0.0.0:443->8080/tcp` and `<tailscale-ip>:50002->50000/tcp`.
- `envoy_grpc_web_proxy`: host network, listening on `0.0.0.0:8443`.
- Envoy admin: `127.0.0.1:9901`.

## Verification

Verify containers, listeners, logs, TLS, and Envoy readiness:

```bash
ssh root@<new-ts-ip> 'docker ps -a --format "table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Ports}}"'
ssh root@<new-ts-ip> 'docker inspect --format "{{range .Mounts}}{{println .Source \"->\" .Destination \"RW=\" .RW}}{{end}}" proxy_service'
ssh root@<new-ts-ip> 'docker inspect --format "{{range .Args}}{{println .}}{{end}}" proxy_service | grep -E "^100\\..*:40001-40008:1$" || true'
ssh root@<new-ts-ip> 'ss -ltnp | grep -E ":(443|8443|50002|9901)" || true'
ssh root@<new-ts-ip> 'docker logs --tail 50 proxy_service 2>&1; docker logs --tail 80 envoy_grpc_web_proxy 2>&1'
ssh root@<new-ts-ip> 'openssl s_client -connect 127.0.0.1:443 -servername compute.drawthings.ai </dev/null 2>/dev/null | openssl x509 -noout -subject -dates'
ssh root@<new-ts-ip> 'openssl s_client -connect 127.0.0.1:8443 -servername compute.drawthings.ai </dev/null 2>/dev/null | openssl x509 -noout -subject -dates'
ssh root@<new-ts-ip> 'curl -fsS --max-time 5 http://127.0.0.1:9901/ready'
```

Verify external GCP access:

```bash
curl -vk --connect-timeout 10 https://<external-ip>:443/ImageGenerationService/Echo
openssl s_client -connect <external-ip>:443 -servername compute.drawthings.ai </dev/null 2>/dev/null | openssl x509 -noout -subject -dates -ext subjectAltName
curl -vk --resolve compute.drawthings.ai:443:<external-ip> --connect-timeout 10 https://compute.drawthings.ai:443/ImageGenerationService/Echo
```

An HTTP `415` from a raw GET to a gRPC endpoint is acceptable; it means the request reached the service but was not a valid gRPC call.

## Model List Updates

With the updated launch script, both of these update the same model-list file:

- Editing `/root/CPUScript/model-list` on the CPU host.
- Running control-panel update from a local checkout:

```bash
bazel-bin/Apps/ProxyServerControlPanelCLI -h <new-ts-ip> -p 50002 update-model-list /path/to/model-list
```

The container should show:

```text
/root/CPUScript -> /app/Documents RW= true
```

If this mount is not writable, `update-model-list` will not reliably update the host file. After a successful update, `Echo` and `FilesExist` read `modelListPath` on demand, so they should reflect the new file without restarting. A restart is still useful after changing launch arguments such as `-g` GPU workers.

## DNS and App Client Notes

The copied Let's Encrypt cert is for `compute.drawthings.ai`, not an IP address. If the app connects with:

```swift
static let host = "<external-ip>"
static let port = 443
```

and uses full hostname verification, TLS fails because the certificate SAN has `DNS:compute.drawthings.ai` and no IP SAN. Symptoms may surface as `GRPC.ConnectionPoolError.shutdown`, and the server logs may show no Echo call because the client fails before the RPC reaches application code.

Before DNS is updated, either:

- connect to `<external-ip>` but set gRPC TLS `hostnameOverride: "compute.drawthings.ai"` / SNI to `compute.drawthings.ai`, or
- temporarily disable hostname verification only for a controlled test path.

After DNS is updated, prefer:

```swift
static let host = "compute.drawthings.ai"
static let port = 443
```

Then full TLS verification should pass without special handling.

## Troubleshooting

- `InitCertificate.sh` fails with Let's Encrypt unauthorized or `503`: `compute.drawthings.ai` is not routed to this VM on public port 80. Copy the existing cert/key to launch now, then fix DNS/load-balancer routing for renewal.
- `proxy_service` restarts with `fopen(... privkey.pem ... Permission denied)`: make `/etc/letsencrypt/live/compute.drawthings.ai/privkey.pem` readable by the container, then rerun `LaunchCPUServer.sh`.
- `update-model-list` prints success but the host file does not change: the model-list path is probably mounted read-only or the containing directory is not writable by the container user. Relaunch with the updated `LaunchCPUServer.sh` directory mount.
- Envoy restarts with validation error on `AdminValidationError.Address ... port_specifier is required`: update `envoy-config.yaml` so `admin.address.socket_address` includes `port_value: 9901`, or copy the known-good config from the current CPU server.
- `grpcurl <ip>:443 ...` fails certificate verification: use `-servername compute.drawthings.ai` or update DNS and use the hostname.
- No Echo log appears on the server: suspect client-side TLS/channel setup before debugging server Echo handling.
