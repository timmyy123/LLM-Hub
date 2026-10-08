#!/bin/bash

# Script to initialize GPU servers with Docker, CUDA, and dependencies
# Usage:
#   ./InitGPUServer.sh root@hostname
#   ./InitGPUServer.sh --dry-run root@hostname

# Exit immediately if a command exits with a non-zero status.
set -e

# Parse arguments
REMOTE_HOST=""
DRY_RUN=false

for arg in "$@"; do
    if [ "$arg" == "--dry-run" ] || [ "$arg" == "-n" ]; then
        DRY_RUN=true
    elif [ -z "$REMOTE_HOST" ]; then
        REMOTE_HOST="$arg"
    fi
done

# The initialization commands to run on the remote server
INIT_COMMANDS='
set -e
export DEBIAN_FRONTEND=noninteractive

apt_get() {
    sudo DEBIAN_FRONTEND=noninteractive apt-get -o DPkg::Lock::Timeout=600 -o Dpkg::Options::=--force-confdef -o Dpkg::Options::=--force-confold "$@"
}

echo "=================================================="
echo "  GPU Server Initialization"
echo "=================================================="
echo ""

# --- 0. System & Docker Installation ---
echo "➡️ Step 1: Updating system and installing Docker..."

# Update package index and install prerequisites
apt_get update
apt_get install -y apt-transport-https ca-certificates curl software-properties-common wget

# Add Docker official GPG key
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

# Add Docker repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Update package index again and install Docker
apt_get update
apt_get install -y docker-ce docker-ce-cli containerd.io

echo "✅ Docker installed successfully."

# --- 1. Install CUDA and related dependencies ---

## Section 1: System Preparation
echo ""
echo "➡️ Step 2: Installing NVIDIA CUDA and Container Toolkit..."

## Section 2: Install CUDA Toolkit
echo "Adding NVIDIA CUDA repository..."

# Install the repository keyring
CUDA_REPO_ID=$( . /etc/os-release && echo "ubuntu$(echo "$VERSION_ID" | tr -d ".")" )
wget -q "https://developer.download.nvidia.com/compute/cuda/repos/${CUDA_REPO_ID}/x86_64/cuda-keyring_1.1-1_all.deb"
sudo DEBIAN_FRONTEND=noninteractive dpkg --force-confdef --force-confold -i cuda-keyring_1.1-1_all.deb
apt_get update
rm cuda-keyring_1.1-1_all.deb

echo "Installing CUDA Toolkit..."
apt_get -y install cuda-toolkit

# Add CUDA to the system PATH for all users
echo "export PATH=/usr/local/cuda/bin\${PATH:+:\${PATH}}" | sudo tee /etc/profile.d/cuda.sh
echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64\${LD_LIBRARY_PATH:+:\${LD_LIBRARY_PATH}}" | sudo tee -a /etc/profile.d/cuda.sh

echo "✅ CUDA Toolkit installed successfully."

## Section 3: Install NVIDIA Container Toolkit
echo "Adding NVIDIA Container Toolkit repository..."
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg --yes
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed "s#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g" | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

apt_get update

echo "Installing NVIDIA Container Toolkit..."
apt_get install -y nvidia-container-toolkit
apt_get install -y mergerfs

## Section 4: Configure Docker
echo "Configuring Docker to use the NVIDIA runtime..."
sudo nvidia-ctk runtime configure --runtime=docker

# Restart the Docker daemon to apply the new configuration
sudo systemctl restart docker

echo "✅ NVIDIA Container Toolkit installed and configured."

# --- 3. Security & Directory Setup ---
echo ""
echo "➡️ Step 4: Applying security settings..."

# Lock the root user password for enhanced security, should do this manually in case we need metal provider login
# sudo passwd -l root
# echo "✅ Root password locked."

# --- 4. Install Python dependencies ---
echo ""
echo "➡️ Step 5: Installing Python and dependencies..."

apt_get update
apt_get install -y python3-pip python3-venv nvtop htop

# Create a virtual environment for Python packages
python3 -m venv /opt/draw-things-venv
/opt/draw-things-venv/bin/pip install --upgrade pip
/opt/draw-things-venv/bin/pip install tqdm schedule boto3

# Add venv to PATH for all users
echo "export PATH=/opt/draw-things-venv/bin:\$PATH" | sudo tee /etc/profile.d/draw-things-venv.sh
sudo chmod +x /etc/profile.d/draw-things-venv.sh
source /etc/profile.d/draw-things-venv.sh

echo "✅ Python dependencies installed in /opt/draw-things-venv"

# --- 5. Network Performance Tuning ---
echo ""
echo "➡️ Step 6: Configuring network performance settings..."

# Apply TCP buffer and BBR settings immediately
sudo sysctl -w net.ipv4.tcp_rmem="4096 87380 134217728"
sudo sysctl -w net.ipv4.tcp_wmem="4096 65536 134217728"
sudo sysctl -w net.core.rmem_max=134217728
sudo sysctl -w net.core.wmem_max=134217728
sudo sysctl -w net.core.default_qdisc=fq
sudo sysctl -w net.ipv4.tcp_congestion_control=bbr

# Make settings persistent across reboots
cat <<EOF | sudo tee /etc/sysctl.d/99-network-performance.conf
# TCP buffer sizes for high-bandwidth connections
net.ipv4.tcp_rmem = 4096 87380 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728

# BBR congestion control
net.core.default_qdisc = fq
net.ipv4.tcp_congestion_control = bbr
EOF

echo "✅ Network performance settings configured and persisted."

# --- 6. Pull Docker images ---
echo ""
echo "➡️ Step 7: Pulling Docker images..."

sudo docker pull drawthingsai/draw-things-grpc-server-cli:latest

echo "✅ Docker images pulled."

# --- 7. Disk Mounting and MergerFS Setup ---
echo ""
echo "➡️ Step 8: Setting up disk mounts and MergerFS..."
echo ""

# Show current disk layout
echo "--- Current Disk Layout (df -h) ---"
df -h
echo ""
echo "--- Block Devices (lsblk) ---"
lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE
echo ""

# Prompt for disk devices
echo "Please enter the mount points to unmount and disk partitions to mount."
echo "(Press Enter on first prompt to skip disk mounting setup)"
echo ""
read -p "Enter current mount point to unmount for models disk (e.g., /mnt/disk1): " UNMOUNT1 || UNMOUNT1=""
read -p "Enter current mount point to unmount for loraModels disk (e.g., /mnt/disk2): " UNMOUNT2 || UNMOUNT2=""
read -p "Enter disk partition for /mnt/models (e.g., /dev/nvme2n1p1): " DISK1 || DISK1=""
read -p "Enter disk partition for /mnt/loraModels (e.g., /dev/nvme3n1p1): " DISK2 || DISK2=""

if [ -n "$UNMOUNT1" ] && [ -n "$UNMOUNT2" ] && [ -n "$DISK1" ] && [ -n "$DISK2" ]; then
    echo ""
    echo "Setting up disk mounts..."

    # Unmount existing locations
    echo "Unmounting existing mounts..."
    if [ -n "$UNMOUNT1" ]; then
        echo "  Unmounting $UNMOUNT1..."
        umount "$UNMOUNT1" 2>/dev/null || true
    fi
    if [ -n "$UNMOUNT2" ]; then
        echo "  Unmounting $UNMOUNT2..."
        umount "$UNMOUNT2" 2>/dev/null || true
    fi

    # Also unmount target mount points if something else is mounted there
    umount /mnt/models /mnt/loraModels /mnt/official-models 2>/dev/null || true

    # Ensure mount points exist
    echo "Creating mount points..."
    mkdir -p /mnt/models /mnt/loraModels /mnt/official-models

    # Mount the physical partitions
    echo "Mounting $DISK1 to /mnt/models..."
    mount "$DISK1" /mnt/models

    echo "Mounting $DISK2 to /mnt/loraModels..."
    mount "$DISK2" /mnt/loraModels

    # Create the extra directory on the second drive
    mkdir -p /mnt/loraModels/models_extra

    # Set up mergerfs to merge both paths
    echo "Setting up MergerFS overlay at /mnt/official-models..."
    mergerfs -o allow_other,use_ino,ro \
      /mnt/models:/mnt/loraModels/models_extra \
      /mnt/official-models

    # Add to fstab for persistence
    echo ""
    echo "Adding entries to /etc/fstab for persistence..."

    # Remove any existing entries for old and new mount points
    sed -i "\|$UNMOUNT1|d" /etc/fstab
    sed -i "\|$UNMOUNT2|d" /etc/fstab
    sed -i "\|/mnt/models|d" /etc/fstab
    sed -i "\|/mnt/loraModels|d" /etc/fstab
    sed -i "\|/mnt/official-models|d" /etc/fstab

    # Detect filesystem types
    FSTYPE1=$(blkid -s TYPE -o value "$DISK1" 2>/dev/null || echo "auto")
    FSTYPE2=$(blkid -s TYPE -o value "$DISK2" 2>/dev/null || echo "auto")
    UUID1=$(blkid -s UUID -o value "$DISK1" 2>/dev/null || true)
    UUID2=$(blkid -s UUID -o value "$DISK2" 2>/dev/null || true)
    FSTAB_DISK1="$DISK1"
    FSTAB_DISK2="$DISK2"
    if [ -n "$UUID1" ]; then
        FSTAB_DISK1="UUID=$UUID1"
    fi
    if [ -n "$UUID2" ]; then
        FSTAB_DISK2="UUID=$UUID2"
    fi

    # Add new entries
    echo "$FSTAB_DISK1 /mnt/models $FSTYPE1 defaults,nofail 0 2" >> /etc/fstab
    echo "$FSTAB_DISK2 /mnt/loraModels $FSTYPE2 defaults,nofail 0 2" >> /etc/fstab
    echo "/mnt/models:/mnt/loraModels/models_extra /mnt/official-models fuse.mergerfs allow_other,use_ino,ro,nofail 0 0" >> /etc/fstab

    echo "✅ Disk mounts and MergerFS configured successfully."
    echo ""
    echo "Mount summary:"
    echo "  Unmounted: $UNMOUNT1, $UNMOUNT2"
    echo "  $DISK1 ($FSTYPE1) → /mnt/models"
    echo "  $DISK2 ($FSTYPE2) → /mnt/loraModels"
    echo "  MergerFS overlay → /mnt/official-models (read-only)"
else
    echo "Skipping disk mounting setup."
fi

# --- 8. Final Summary ---
echo ""
echo "=================================================="
echo "🎉 GPU Server Initialization Complete!"
echo "=================================================="
echo ""
if [ -f /var/run/reboot-required ]; then
    echo "A system reboot is required by installed packages."
else
    echo "No package-required reboot marker found."
fi
echo ""
echo "Verification commands:"
echo "  1. nvidia-smi                    # Check GPU status"
echo "  2. nvcc --version                # Check CUDA version"
echo "  3. sudo docker run --rm --gpus all nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi"
echo "  4. df -h                         # Check disk mounts"
echo "  5. ls /mnt/official-models       # Verify MergerFS overlay"
echo ""
echo "=================================================="
'

# Function to perform dry-run check on a single server
dry_run_server() {
    local HOST="$1"

    echo "Target: $HOST"
    echo ""

    # Check if ssh is available
    if ! command -v ssh &> /dev/null; then
        echo "Error: ssh command not found. Please install openssh-client."
        return 1
    fi

    # Test SSH connection
    echo "Testing SSH connection to $HOST..."
    if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" "echo 'Connection successful'" 2>/dev/null < /dev/null; then
        echo "Error: Cannot connect to $HOST"
        echo "Please ensure:"
        echo "  1. The host is reachable"
        echo "  2. SSH keys are set up for passwordless authentication"
        echo "  3. You have root access"
        return 1
    fi
    echo "✅ SSH connection successful"
    echo ""

    # Get system information
    echo "--- System Information ---"
    ssh "$HOST" "lsb_release -a 2>/dev/null || cat /etc/os-release 2>/dev/null || echo 'Could not determine OS'" < /dev/null
    echo ""

    echo "--- GPU Information ---"
    ssh "$HOST" "nvidia-smi 2>/dev/null || echo 'nvidia-smi not available (GPU driver may not be installed yet)'" < /dev/null
    echo ""

    echo "--- Disk Information ---"
    ssh "$HOST" "df -h 2>/dev/null" < /dev/null
    echo ""
    ssh "$HOST" "lsblk -o NAME,SIZE,TYPE,MOUNTPOINT,FSTYPE 2>/dev/null || lsblk 2>/dev/null || echo 'lsblk not available'" < /dev/null
    echo ""

    echo "--- Commands that would be executed ---"
    echo "$INIT_COMMANDS" | head -80
    echo "... (truncated, see script for full commands)"
    echo ""

    return 0
}

# Function to initialize a single server
init_server() {
    local HOST="$1"

    echo "Target: $HOST"
    echo ""

    # Check if ssh is available
    if ! command -v ssh &> /dev/null; then
        echo "Error: ssh command not found. Please install openssh-client."
        return 1
    fi

    # Test SSH connection
    echo "Testing SSH connection to $HOST..."
    if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" "echo 'Connection successful'" 2>/dev/null < /dev/null; then
        echo "Error: Cannot connect to $HOST"
        echo "Please ensure:"
        echo "  1. The host is reachable"
        echo "  2. SSH keys are set up for passwordless authentication"
        echo "  3. You have root access"
        return 1
    fi
    echo "✅ SSH connection successful"
    echo ""

    # Execute initialization commands on remote server
    echo "Starting initialization on $HOST..."
    echo ""
    ssh "$HOST" "$INIT_COMMANDS" < /dev/null
}

# Validate arguments
if [ -z "$REMOTE_HOST" ]; then
    echo "Usage:"
    echo "  $0 root@hostname"
    echo "  $0 --dry-run root@hostname"
    echo ""
    echo "Options:"
    echo "  --dry-run, -n   Check server connectivity and show system info without making changes"
    echo ""
    echo "This script will initialize a GPU server with:"
    echo "  - Docker"
    echo "  - CUDA Toolkit"
    echo "  - NVIDIA Container Toolkit"
    echo "  - Python dependencies (tqdm, schedule, boto3)"
    echo "  - Network performance tuning (TCP buffers, BBR)"
    echo "  - Draw Things gRPC server Docker image"
    echo "  - Disk mounting with MergerFS overlay (interactive)"
    exit 1
fi

if [ "$DRY_RUN" = true ]; then
    echo "=================================================="
    echo "  [DRY RUN] Check GPU Server"
    echo "=================================================="

    if dry_run_server "$REMOTE_HOST"; then
        echo ""
        echo "=================================================="
        echo "✅ Dry run check passed!"
        echo ""
        echo "Run without --dry-run to perform the actual initialization."
        exit 0
    else
        echo ""
        echo "=================================================="
        echo "❌ Dry run check failed!"
        exit 1
    fi
else
    echo "=================================================="
    echo "  Initialize GPU Server"
    echo "=================================================="

    if init_server "$REMOTE_HOST"; then
        echo ""
        echo "=================================================="
        echo "✅ Initialization complete!"
        echo ""
        echo "Check /var/run/reboot-required before rebooting the server."
        exit 0
    else
        echo ""
        echo "=================================================="
        echo "❌ Initialization failed!"
        exit 1
    fi
fi
