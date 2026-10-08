# GPU Script Setup Guide

## Table of Contents

1. [Initial Setup](#initial-setup)
2. [Model Synchronization](#model-synchronization)
3. [Testing Setup](#testing-setup)
4. [Tailscale Setup](#tailscale-setup-script-optional-you-could-setup-tailscale-directly)
5. [Launch GPU Server](#launch-gpu-server)
6. [Model Synchronization & Verification](#model-synchronization--verification)
   - [compare_checksums.py](#compare_checksumspy)
   - [sync_models_multi_server.py](#sync_models_multi_serverpy)
   - [UpdateNasModel.sh](#updatenasmodelsh)
7. [Additional Utilities](#additional-utilities)
8. [Quick Start Workflow](#quick-start-workflow)
   - [Setting Up a New GPU Server](#setting-up-a-new-gpu-server)
   - [Maintaining NAS Server](#maintaining-nas-server)
   - [Verifying Model Integrity](#verifying-model-integrity)
   - [Troubleshooting](#troubleshooting)

## Initial Setup

After SSH into the server:

```bash
./InitGPUServer.sh  # Sets up Docker, CUDA, and Docker image
```

## Model Synchronization

### Option 1: Cloudflare R2 Sync (Recommended)
```bash
python3 /path/to/GPUScript/r2_sync_verification.py \
  -p /mnt/disk2/official-models/ \
  --account-id <your-account-id> \
  --access-key <your-access-key> \
  --secret-key <your-secret-key>
```

### Option 2: Rsync from Another Server
Use rsync with SSH pubkey authentication. Note: Must use physical IP address instead of Tailscale IP (10x slower).

## Testing Setup

1. Create test Docker container:
```bash
sudo docker run -it --name test3 \
  -v /path/to/official-models/:/models \
  -v /path/to/utils/:/utils \
  --gpus all \
  drawthingsai/draw-things-grpc-server-cli /bin/bash
```

2. Format tensor data (inside container):
```bash
./utils/TensorDataFormatter2 -p /models/ \
  --skip "qwen_image_edit_1.0_bf16.ckpt, qwen_image_1.0_bf16.ckpt"
```
**Note:** bf16 models may crash - issue pending investigation.

## Tailscale Setup script (Optional, you could setup tailscale directly)

```bash
/home/wlin1/Drawthings/draw-things/Scripts/ServerManagement/InitTailscale.sh
```

## Launch GPU Server

[LaunchGPU/LaunchGPUServer.sh](LaunchGPU/LaunchGPUServer.sh) starts 8 Docker containers, each running one gRPCServerCLI process with automatic restart on crash.

### Features:
- Supports both local and remote execution
- Optional tmpfs overlay for faster model loading (448GB by default)
- Automatically pulls latest Docker image
- Waits for GPUs to be idle before starting (5-minute timeout)
- Uses [mount_overlay.sh](LaunchGPU/mount_overlay.sh) for overlay filesystem setup

### Usage:

**Local execution:**
```bash
./LaunchGPU/LaunchGPUServer.sh <address> <models_path> <utils_path> <lora_models_path> [--skip-tmpfs]
```

**Remote execution:**
```bash
./LaunchGPU/LaunchGPUServer.sh <remote_host> <address> <models_path> <utils_path> <lora_models_path> [--skip-tmpfs]
```

### Examples:
```bash
# Local execution with tmpfs overlay
./LaunchGPU/LaunchGPUServer.sh \
  100.168.1.100 \
  /mnt/models/official-models \
  /root/utils \
  /disk2/loraModels/

# Remote execution
./LaunchGPU/LaunchGPUServer.sh \
  root@dfw-026-001 \
  100.168.1.100 \
  /mnt/models/official-models \
  /root/utils \
  /disk2/loraModels/

# Skip tmpfs overlay (use models path directly)
./LaunchGPU/LaunchGPUServer.sh \
  100.168.1.100 \
  /mnt/models/official-models \
  /root/utils \
  /disk2/loraModels/ \
  --skip-tmpfs
```

### Container Management:
```bash
# Check container status
sudo docker ps

# View logs for specific container
sudo docker logs grpc_service_0

# Stop all containers
sudo docker stop $(sudo docker ps -q --filter name=grpc_service_)
```

## Model Synchronization & Verification

### compare_checksums.py

[UpdateModels/compare_checksums.py](UpdateModels/compare_checksums.py) generates and compares SHA256 checksums for `.ckpt-tensordata` and `.ckpt` files. It automatically detects and removes corrupted or zero-size files.

#### Mode 1: Single Directory Mode

Generate/update `sha256-list.csv` for a directory and clean up corrupted files.

**Local directory:**
```bash
python3 UpdateModels/compare_checksums.py /path/to/models/
```

**Remote directory:**
```bash
python3 UpdateModels/compare_checksums.py root@hostname:/path/to/models/
```

This mode will:
- Download existing `sha256-list.csv` from remote (if exists)
- Calculate checksums for files without them
- Identify and remove corrupted files (checksum errors)
- Identify and remove zero-size files
- Upload updated CSV back to remote server

**Options:**
- `--dry-run`: Preview actions without modifying files
- `--verbose` or `-v`: Show detailed output

**Example:**
```bash
# Update checksums on a GPU server
python3 UpdateModels/compare_checksums.py root@dfw-026-001:/mnt/models/official-models

# Dry-run to preview
python3 UpdateModels/compare_checksums.py --dry-run /disk1/models/
```

#### Mode 2: CSV Comparison Mode

Compare two CSV files to identify missing or mismatched files.

```bash
python3 UpdateModels/compare_checksums.py <csv1> <csv2>
```

- `csv2` is treated as the source of truth
- Default output: Simple list of files to download (one per line)
- `--verbose` output: Detailed differences with categories

**Example:**
```bash
# Compare server CSV against NAS (source of truth)
python3 UpdateModels/compare_checksums.py \
  sha256-list-dfw-026-001.csv \
  nas-sha256-list.csv

# Get detailed comparison
python3 UpdateModels/compare_checksums.py --verbose \
  sha256-list-dfw-026-001.csv \
  nas-sha256-list.csv
```

**Exit codes:**
- `0`: No differences found
- `1`: Differences found (useful for scripting)

### sync_models_multi_server.py

[UpdateModels/sync_models_multi_server.py](UpdateModels/sync_models_multi_server.py) orchestrates model synchronization across multiple GPU servers using a NAS HTTP server as the source of truth.

#### Prerequisites:

1. **Start NAS HTTP server:**
```bash
# On NAS server
ssh root@dt-thpc-nas01 'cd /root/utils && ./start_nas_http_server.sh'
```

2. **Create GPU servers list:**

Create `UpdateModels/gpu_servers.txt` with one server per line:
```
root@dfw-026-001
root@dfw-026-002
root@dfw-026-003
```

The script uses the default path `/mnt/models/official-models` for all servers.

#### Workflow:

1. Check if NAS HTTP server is accessible (`http://100.104.93.82:8000`)
2. Load GPU servers from `gpu_servers.txt`
3. Download NAS `sha256-list.csv` as source of truth
4. For each GPU server:
   - Update checksums on GPU server
   - Compare with NAS CSV to identify missing/corrupted files
   - Download files from NAS HTTP server using wget
   - Update checksums again to verify

#### Usage:

```bash
python3 UpdateModels/sync_models_multi_server.py
```

#### Configuration:

Edit the script constants if needed:
```python
NAS_HOST = "root@dt-thpc-nas01"
NAS_IP = "100.104.93.82"
NAS_PATH = "/zfs/data/official-models-ckpt-tensordata"
HTTP_PORT = 8000
DEFAULT_GPU_PATH = "/mnt/models/official-models"
```

#### Example Output:
```
[Step 1/4] Check NAS HTTP server
   ✅ NAS HTTP server is accessible

[Step 2/4] Load GPU servers from gpu_servers.txt
✅ Loaded 3 GPU server(s)

[Step 3/4] Download NAS CSV (source of truth)
   ✅ Downloaded to: nas-sha256-list.csv

[Step 4/4] Sync GPU servers
   [1/3] Syncing: root@dfw-026-001:/mnt/models/official-models
   ✅ Downloaded 12/12 file(s)

Total: 3/3 servers synced successfully
```

### UpdateNasModel.sh

[UpdateModels/UpdateNasModel.sh](UpdateModels/UpdateNasModel.sh) is a comprehensive script for updating and maintaining NAS model storage. It orchestrates three key operations: R2 sync, TensorData formatting, and checksum verification.

#### Operation Modes:

1. **sync** - Syncs and verifies models from Cloudflare R2 storage
2. **format** - Formats models using TensorDataFormatter
3. **checksum** - Compares checksums to verify integrity
4. **all** - Runs all three operations in sequence (default)

#### Usage:

**Local execution:**
```bash
./UpdateModels/UpdateNasModel.sh \
  -p /path/to/models \
  -u /path/to/utils \
  -m MODE \
  --account-id ID \
  --access-key KEY \
  --secret-key SECRET
```

**Remote execution:**
```bash
./UpdateModels/UpdateNasModel.sh \
  root@hostname \
  -p /path/to/models \
  -u /path/to/utils \
  -m MODE \
  --account-id ID \
  --access-key KEY \
  --secret-key SECRET
```

#### Parameters:

**Required:**
- `-p, --path`: Path to models directory
- `-u, --utils`: Path to utils directory
- `-m, --mode`: Operation mode (sync, format, checksum, or all)

**Required for sync mode:**
- `--account-id`: Cloudflare R2 account ID
- `--access-key`: R2 access key
- `--secret-key`: R2 secret key

#### Examples:

**Run all operations on NAS:**
```bash
./UpdateModels/UpdateNasModel.sh \
  -p /zfs/data/official-models-ckpt-tensordata/ \
  -u /root/utils/ \
  --account-id YOUR_ID \
  --access-key YOUR_KEY \
  --secret-key YOUR_SECRET
```

**Run all operations on remote NAS:**
```bash
./UpdateModels/UpdateNasModel.sh \
  root@dt-thpc-nas01 \
  -p /zfs/data/official-models-ckpt-tensordata/ \
  -u /root/utils/ \
  --account-id YOUR_ID \
  --access-key YOUR_KEY \
  --secret-key YOUR_SECRET
```

**Only sync from R2:**
```bash
./UpdateModels/UpdateNasModel.sh \
  -p /zfs/data/official-models-ckpt-tensordata/ \
  -u /root/utils/ \
  -m sync \
  --account-id YOUR_ID \
  --access-key YOUR_KEY \
  --secret-key YOUR_SECRET
```

**Only format (no R2 credentials needed):**
```bash
./UpdateModels/UpdateNasModel.sh \
  -p /zfs/data/official-models-ckpt-tensordata/ \
  -u /root/utils/ \
  -m format
```

**Only verify checksums (no R2 credentials needed):**
```bash
./UpdateModels/UpdateNasModel.sh \
  -p /zfs/data/official-models-ckpt-tensordata/ \
  -u /root/utils/ \
  -m checksum
```

#### Workflow (all mode):

1. **R2 Sync Verification**: Uses [r2_sync_verification.py](UpdateModels/r2_sync_verification.py) to download/verify models from Cloudflare R2
2. **TensorData Formatting**: Runs TensorDataFormatter to convert models to Draw Things format
3. **Checksum Verification**: Uses [compare_checksums.py](UpdateModels/compare_checksums.py) to verify integrity and remove corrupted files

#### Dependencies:

- `r2_sync_verification.py` (for sync mode)
- `TensorDataFormatter` binary (for format mode)
- `compare_checksums.py` (for checksum mode)

All dependencies should be present in the utils directory.

## Additional Utilities

### NAS HTTP Server Management

**Start NAS HTTP server:**
```bash
ssh root@dt-thpc-nas01 'cd /root/utils && ./UpdateModels/start_nas_http_server.sh'
```

**Stop NAS HTTP server:**
```bash
ssh root@dt-thpc-nas01 'cd /root/utils && ./UpdateModels/stop_nas_http_server.sh'
```

The HTTP server serves models from `/zfs/data/official-models-ckpt-tensordata` on port 8000.

### LaunchGPU Utilities

- **[mount_overlay.sh](LaunchGPU/mount_overlay.sh)**: Creates overlay filesystem with tmpfs for faster model loading
- **[start_single_grpc.sh](LaunchGPU/start_single_grpc.sh)**: Starts a single gRPC service (used by LaunchGPUServer.sh)
- **[list_ckpts.sh](LaunchGPU/list_ckpts.sh)**: Lists checkpoint files in a directory
- **[sum_size.sh](LaunchGPU/sum_size.sh)**: Calculates total size of checkpoint files
- **[tmpfs.ls](LaunchGPU/tmpfs.ls)**: List of files to load into tmpfs overlay

### Legacy/WIP Scripts

- **[UpdateModels/UpdateNasModel_WIP.sh](UpdateModels/UpdateNasModel_WIP.sh)**: Work-in-progress NAS model update script
- **[LaunchGPU/SetupRAID0Storage_WIP.sh](LaunchGPU/SetupRAID0Storage_WIP.sh)**: Work-in-progress RAID0 storage setup

## Quick Start Workflow

### Setting Up a New GPU Server

1. **Initialize server:**
```bash
ssh root@new-server
./InitGPUServer.sh
```

2. **Sync models from NAS:**

   **Option A: Multi-server sync (recommended for multiple servers):**
   ```bash
   # Add server to gpu_servers.txt
   echo "root@new-server" >> UpdateModels/gpu_servers.txt

   # Run multi-server sync
   python3 UpdateModels/sync_models_multi_server.py
   ```

   **Option B: Direct sync from Cloudflare R2:**
   ```bash
   # Sync directly from R2 to GPU server
   python3 UpdateModels/r2_sync_verification.py \
     -p /mnt/models/official-models/ \
     --account-id YOUR_ID \
     --access-key YOUR_KEY \
     --secret-key YOUR_SECRET
   ```

3. **Launch GPU services:**
```bash
./LaunchGPU/LaunchGPUServer.sh \
  root@new-server \
  100.168.1.X \
  /mnt/models/official-models \
  /root/utils \
  /disk2/loraModels/
```

### Maintaining NAS Server

**Update NAS models from R2 (all operations):**
```bash
./UpdateModels/UpdateNasModel.sh \
  root@dt-thpc-nas01 \
  -p /zfs/data/official-models-ckpt-tensordata/ \
  -u /root/utils/ \
  --account-id YOUR_ID \
  --access-key YOUR_KEY \
  --secret-key YOUR_SECRET
```

**Only sync new models from R2:**
```bash
./UpdateModels/UpdateNasModel.sh \
  root@dt-thpc-nas01 \
  -p /zfs/data/official-models-ckpt-tensordata/ \
  -u /root/utils/ \
  -m sync \
  --account-id YOUR_ID \
  --access-key YOUR_KEY \
  --secret-key YOUR_SECRET
```

**Verify NAS model integrity:**
```bash
./UpdateModels/UpdateNasModel.sh \
  root@dt-thpc-nas01 \
  -p /zfs/data/official-models-ckpt-tensordata/ \
  -u /root/utils/ \
  -m checksum
```

### Verifying Model Integrity

**Check single server:**
```bash
python3 UpdateModels/compare_checksums.py root@server:/mnt/models/official-models
```

**Compare against source of truth:**
```bash
# Generate checksums for both
python3 UpdateModels/compare_checksums.py root@server:/mnt/models/official-models
python3 UpdateModels/compare_checksums.py root@dt-thpc-nas01:/zfs/data/official-models-ckpt-tensordata

# Compare
python3 UpdateModels/compare_checksums.py \
  sha256-list-server.csv \
  nas-sha256-list.csv
```

### Troubleshooting

**GPU services not starting:**
```bash
# Check GPU usage
nvidia-smi

# Check Docker containers
sudo docker ps -a

# View container logs
sudo docker logs grpc_service_0
```

**Model download failures:**
```bash
# Test NAS HTTP server
wget --spider http://100.104.93.82:8000/sha256-list.csv

# Check SSH connectivity
ssh root@server 'echo "Connection OK"'

# Verify checksums manually
ssh root@server 'sha256sum /mnt/models/official-models/model.ckpt'
```

**Corrupted models:**
```bash
# Run checksum verification (auto-removes corrupted files)
python3 UpdateModels/compare_checksums.py root@server:/mnt/models/official-models

# Re-download from NAS
python3 UpdateModels/sync_models_multi_server.py
```