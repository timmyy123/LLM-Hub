---
name: init-gpu-server
description: Initialize a Draw Things GPU server with GPUScript, including script sync, Docker/CUDA/NVIDIA runtime setup, 7T data disk mounting, mergerfs, and end-to-end GPU verification.
---

# Init GPU Server Skill

Use this workflow when bringing up a new Draw Things GPU server reachable as `root@HOST`.

## Goal

Prepare the server for Draw Things GPU workloads:

- sync GPUScript utilities to `/root/utils`
- install Docker, CUDA Toolkit, NVIDIA Container Toolkit, mergerfs, and Python utilities
- mount data disks at `/mnt/models` and `/mnt/loraModels`
- expose a merged model path at `/mnt/official-models`
- verify Docker can access all GPUs

## Source Scripts

Repo source:

```sh
Scripts/ServerManagement/GPUScript
```

Important files:

- `update_scripts.sh`: uploads files from `files_to_copy.txt` to `/root/utils/`
- `init_gpu_server.sh`: installs Docker/CUDA/NVIDIA runtime, Python deps, network sysctl, and the Draw Things Docker image
- `LaunchGPU/remount_disk.sh`: historical reference for disk remounting and mergerfs setup
- `verify_gpu_setup.sh`: remote verification helper

## Script Hygiene

Before running the init script:

```sh
bash -n Scripts/ServerManagement/GPUScript/init_gpu_server.sh
```

If working from an older branch, make sure `init_gpu_server.sh` has these properties:

- Uses noninteractive apt/dpkg with a lock timeout:

```sh
export DEBIAN_FRONTEND=noninteractive
apt-get -o DPkg::Lock::Timeout=600 \
  -o Dpkg::Options::=--force-confdef \
  -o Dpkg::Options::=--force-confold ...
```

- Uses the remote OS version for the CUDA repo, not a hardcoded Ubuntu release:

```sh
CUDA_REPO_ID=$( . /etc/os-release && echo "ubuntu$(echo "$VERSION_ID" | tr -d ".")" )
```

- Propagates the remote SSH exit code from `init_server`; do not force `return 0`.
- Does not require `ssh -t` for automation.
- Disk prompt `read` lines tolerate EOF, for example `read -p "..." DISK1 || DISK1=""`.
- Avoids single quotes inside the large single-quoted `INIT_COMMANDS` string.

## Upload Scripts

Accept the SSH host key if needed:

```sh
ssh -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new root@HOST 'echo ok'
```

Upload the GPUScript files:

```sh
bash Scripts/ServerManagement/GPUScript/update_scripts.sh root@HOST
```

Expected remote destination:

```text
/root/utils/
```

## Run Init

Run:

```sh
bash Scripts/ServerManagement/GPUScript/init_gpu_server.sh root@HOST
```

This should install/configure:

- Docker CE and containerd
- CUDA Toolkit
- NVIDIA Container Toolkit
- mergerfs
- `/opt/draw-things-venv` with `tqdm`, `schedule`, and `boto3`
- network sysctl tuning for large TCP buffers and BBR
- `drawthingsai/draw-things-grpc-server-cli:latest`

It is acceptable to skip disk mounting inside this script. Mount disks explicitly after inspecting `lsblk`.

## Apt/Dpkg Recovery

If apt or dpkg fails on a conffile prompt, such as `/etc/cloud/cloud.cfg`, repair the remote package state:

```sh
ssh root@HOST 'DEBIAN_FRONTEND=noninteractive dpkg --force-confdef --force-confold --configure -a'
ssh root@HOST 'DEBIAN_FRONTEND=noninteractive apt-get -y -f install -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold'
```

If a new run hits an apt lock, inspect first:

```sh
ssh root@HOST 'pgrep -af apt; pgrep -af dpkg'
```

If an earlier init process is still installing packages, wait. Do not kill package-manager processes unless the user explicitly asks.

## Disk Inspection

Inspect block devices and current mounts before changing disks:

```sh
ssh root@HOST 'lsblk -o NAME,SIZE,TYPE,MOUNTPOINTS,FSTYPE,UUID; echo ---; blkid || true; echo ---; df -h'
```

Target shape:

```text
DISK_Ap1 -> /mnt/models
DISK_Bp1 -> /mnt/loraModels
/mnt/models/official-models:/mnt/loraModels/models_extra -> /mnt/official-models
```

Do not assume device names. Pick the two 7T data disks from actual `lsblk` output. On one validated host the mapping was:

```text
/dev/nvme1n1p1 -> /mnt/models
/dev/nvme0n1p1 -> /mnt/loraModels
```

## Raw 7T Disk Setup

If the two 7T disks are raw disks with no partitions and no filesystem, get explicit user approval before partitioning and formatting. This destroys data on those disks.

After approval, replace `MODELS_DISK` and `LORA_DISK` with the inspected devices:

```sh
ssh root@HOST 'set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get -o DPkg::Lock::Timeout=600 -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold update
apt-get -o DPkg::Lock::Timeout=600 -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold install -y parted mergerfs

MODELS_DISK=/dev/nvme1n1
LORA_DISK=/dev/nvme0n1

for disk in "$MODELS_DISK" "$LORA_DISK"; do
  test -b "$disk"
  if lsblk -nrpo NAME "$disk" | tail -n +2 | grep -q .; then
    echo "Refusing to repartition $disk because it already has child block devices" >&2
    lsblk "$disk" >&2
    exit 1
  fi
done

parted -s "$MODELS_DISK" mklabel gpt mkpart primary ext4 0% 100%
parted -s "$LORA_DISK" mklabel gpt mkpart primary ext4 0% 100%
partprobe "$MODELS_DISK" "$LORA_DISK" || true
udevadm settle

mkfs.ext4 -F -L models "${MODELS_DISK}p1"
mkfs.ext4 -F -L loraModels "${LORA_DISK}p1"
'
```

For non-NVMe disks, adjust partition paths accordingly. NVMe partition paths normally use the `p1` suffix.

## Mount And Persist

After partitions/filesystems exist, mount them by UUID and persist in `/etc/fstab`:

```sh
ssh root@HOST 'set -euo pipefail
mkdir -p /mnt/models /mnt/loraModels /mnt/official-models

MODEL_PART=/dev/nvme1n1p1
LORA_PART=/dev/nvme0n1p1
MODEL_UUID=$(blkid -s UUID -o value "$MODEL_PART")
LORA_UUID=$(blkid -s UUID -o value "$LORA_PART")

cp /etc/fstab "/etc/fstab.drawthings.$(date +%Y%m%d%H%M%S).bak"
grep -vE "/mnt/models|/mnt/loraModels|/mnt/official-models|/mnt/official_models" /etc/fstab > /etc/fstab.drawthings.new
mv /etc/fstab.drawthings.new /etc/fstab

{
  echo "UUID=$MODEL_UUID /mnt/models ext4 defaults,nofail 0 2"
  echo "UUID=$LORA_UUID /mnt/loraModels ext4 defaults,nofail 0 2"
  echo "/mnt/models/official-models:/mnt/loraModels/models_extra /mnt/official-models fuse.mergerfs allow_other,use_ino,ro,nofail 0 0"
} >> /etc/fstab

if grep -q "^#user_allow_other" /etc/fuse.conf; then
  sed -i "s/^#user_allow_other/user_allow_other/" /etc/fuse.conf
elif ! grep -q "^user_allow_other" /etc/fuse.conf; then
  echo user_allow_other >> /etc/fuse.conf
fi

mountpoint -q /mnt/official-models && umount /mnt/official-models || true
mountpoint -q /mnt/models || mount /mnt/models
mountpoint -q /mnt/loraModels || mount /mnt/loraModels
mkdir -p /mnt/models/official-models /mnt/loraModels/models_extra /mnt/official-models
mountpoint -q /mnt/official-models || mount /mnt/official-models

lsblk -o NAME,SIZE,TYPE,MOUNTPOINTS,FSTYPE,UUID
df -h | grep -E "Filesystem|/mnt/models|/mnt/loraModels|/mnt/official-models"
'
```

Use `/mnt/official-models`, not `/mnt/official_models`, for Draw Things GPU server conventions.

## Verification

Always verify directly, even if the init script prints success:

```sh
ssh root@HOST 'set -e
echo "=== GPUs ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "=== CUDA ==="
/usr/local/cuda/bin/nvcc --version
echo "=== NVIDIA Container Toolkit ==="
nvidia-ctk --version
echo "=== Docker ==="
docker --version
docker info --format "Runtimes={{json .Runtimes}} Default={{.DefaultRuntime}}"
echo "=== Draw Things image ==="
docker image inspect drawthingsai/draw-things-grpc-server-cli:latest --format "{{.Id}} {{.RepoTags}}"
echo "=== Mounts ==="
lsblk -o NAME,SIZE,TYPE,MOUNTPOINTS,FSTYPE
df -h | grep -E "Filesystem|/mnt/models|/mnt/loraModels|/mnt/official-models"
'
```

Then run an end-to-end Docker GPU test:

```sh
ssh root@HOST 'docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi'
```

Success means the container sees GPUs through the NVIDIA runtime.

## Expected Healthy State

For an 8x RTX 5090 host, a healthy result looked like:

- `nvidia-smi` lists 8 GPUs
- host driver version is `580.173.02`
- `/usr/local/cuda/bin/nvcc --version` reports CUDA Toolkit `13.3`
- `nvidia-ctk --version` reports NVIDIA Container Toolkit `1.19.1`
- Docker has an `nvidia` runtime
- `drawthingsai/draw-things-grpc-server-cli:latest` exists locally
- Docker GPU test container runs `nvidia-smi` successfully
- `lsblk` shows both 7T partitions mounted:

```text
nvme1n1p1 /mnt/models     ext4
nvme0n1p1 /mnt/loraModels ext4
```

- `df -h` shows `/mnt/models`, `/mnt/loraModels`, and `/mnt/official-models`

## Reboot

If `/var/run/reboot-required` exists, tell the user. GPU/Docker may work before reboot, but reboot is the clean final state after kernel, firmware, CUDA, or driver setup:

```sh
ssh root@HOST 'test -f /var/run/reboot-required && cat /var/run/reboot-required || echo no'
```
