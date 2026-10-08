#!/bin/bash

# Script to start 8 separate Docker containers, each running one gRPCServerCLI process
# Each container will automatically restart if the process crashes
# Usage:
#   Remote: ./LaunchGPUServer.sh <remote_host> <models_path> <utils_path> <lora_models_path> [--tmpfs-list <filename>]
#   Local:  ./LaunchGPUServer.sh <address_or_hostname> <models_path> <utils_path> <lora_models_path> [--tmpfs-list <filename>]
#
# If --tmpfs-list is omitted, the tmpfs overlay is skipped and the models path is mounted directly.
#
# For remote execution, the address is automatically derived from <remote_host> (e.g., root@dfw-026-001 -> dfw-026-001)
#
# Default paths (commonly used):
#   models_path:      /mnt/models/official-models
#   utils_path:       /root/utils
#   lora_models_path: /mnt/loraModels
#
# Environment variables:
#   PROXY_HOST - Proxy server host (default: 100.71.37.12)
#   PROXY_PORT - Proxy server port (default: 50002)
#   DRAW_THINGS_DIR - Path to draw-things repo for bazel commands (default: auto-detect)

# Exit immediately if a command exits with a non-zero status
set -e

# Proxy server configuration (can be overridden via environment variables)
PROXY_HOST="${PROXY_HOST:-100.71.37.12}"
PROXY_PORT="${PROXY_PORT:-50002}"

# Auto-detect draw-things directory (4 levels up from script location)
DRAW_THINGS_DIR="${DRAW_THINGS_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"

# Function to resolve hostname to IP address (via Tailscale)
resolve_hostname_to_ip() {
    local input="$1"

    # Check if input is already an IP address (IPv4)
    if [[ "$input" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "$input"
        return 0
    fi

    # Resolve via Tailscale
    local resolved_ip
    resolved_ip=$(tailscale ip -4 "$input" 2>/dev/null)

    if [ -z "$resolved_ip" ]; then
        echo "Error: Could not resolve hostname '$input' via Tailscale" >&2
        return 1
    fi

    echo "$resolved_ip"
    return 0
}

# Function to remove GPU from proxy server
remove_gpu_from_proxy() {
    local address="$1"
    local port_range="40001-40008"

    echo "Removing GPU from proxy server..."
    echo "  Proxy: ${PROXY_HOST}:${PROXY_PORT}"
    echo "  GPU address: ${address}:${port_range}"

    # Check if we're in a directory with bazel access
    if [ -d "$DRAW_THINGS_DIR" ] && [ -f "$DRAW_THINGS_DIR/WORKSPACE" ]; then
        echo "  Running: bazel run Apps:ProxyServerControlPanelCLI -- -h $PROXY_HOST -p $PROXY_PORT remove-gpu ${address}:${port_range}"
        (cd "$DRAW_THINGS_DIR" && bazel run Apps:ProxyServerControlPanelCLI -- -h "$PROXY_HOST" -p "$PROXY_PORT" remove-gpu "${address}:${port_range}") || {
            echo "  ⚠️ Warning: Failed to remove GPU from proxy server (may not exist or proxy unreachable)"
        }
        echo "  ✅ GPU removal command completed"
    else
        echo "  ⚠️ Warning: draw-things directory not found at $DRAW_THINGS_DIR"
        echo "  Skipping proxy server GPU removal"
    fi
}

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"

# Check if all required arguments are provided
if [ $# -lt 4 ]; then
    echo "Error: Insufficient arguments"
    echo ""
    echo "Usage:"
    echo "  Remote: $SCRIPT_NAME <remote_host> <models_path> <utils_path> <lora_models_path> [--tmpfs-list <filename>]"
    echo "  Local:  $SCRIPT_NAME <address> <models_path> <utils_path> <lora_models_path> [--tmpfs-list <filename>]"
    echo ""
    echo "Options:"
    echo "  --tmpfs-list <filename> File list (relative to script dir) for tmpfs overlay."
    echo "                          If omitted, tmpfs overlay is skipped and models path is used directly."
    echo ""
    echo "Environment variables:"
    echo "  PROXY_HOST       Proxy server host (default: 100.71.37.12)"
    echo "  PROXY_PORT       Proxy server port (default: 50002)"
    echo "  DRAW_THINGS_DIR  Path to draw-things repo (default: auto-detect)"
    echo ""
    echo "Default paths (commonly used):"
    echo "  models_path:      /mnt/models/official-models"
    echo "  utils_path:       /root/utils"
    echo "  lora_models_path: /mnt/loraModels"
    echo ""
    echo "Examples:"
    echo "  Remote: $SCRIPT_NAME root@dfw-026-001 /mnt/models/official-models /root/utils/LaunchGPU /mnt/loraModels"
    echo "  Local:  $SCRIPT_NAME 192.168.1.100 /mnt/models/official-models /root/utils/LaunchGPU /mnt/loraModels"
    echo ""
    echo "Note: For remote execution, address is derived from remote_host (e.g., root@dfw-026-001 -> dfw-026-001)"
    echo "      The target's Tailscale IPv4 is queried over the verified SSH connection"
    exit 1
fi

# Initialize flags
TMPFS_LIST_NAME=""

# Parse optional flags from all arguments
NEXT_IS_TMPFS_LIST=false
for arg in "$@"; do
    if [ "$NEXT_IS_TMPFS_LIST" = true ]; then
        TMPFS_LIST_NAME="$arg"
        NEXT_IS_TMPFS_LIST=false
        continue
    fi
    case "$arg" in
        --tmpfs-list) NEXT_IS_TMPFS_LIST=true ;;
    esac
done

# Validate that --tmpfs-list got a filename when used
if [ "$NEXT_IS_TMPFS_LIST" = true ]; then
    echo "Error: --tmpfs-list requires a filename argument"
    exit 1
fi

# Check if first argument is a remote host
REMOTE_HOST=""
if [[ "$1" =~ ^[a-zA-Z0-9_-]+@[a-zA-Z0-9._-]+$ ]]; then
    REMOTE_HOST="$1"
    shift
fi

# If we're executing remotely
if [ -n "$REMOTE_HOST" ]; then
    echo "=================================================="
    echo "  Remote GPU Server Launch"
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
    if ! ssh -o BatchMode=yes -o ConnectTimeout=5 "$REMOTE_HOST" "echo 'Connection successful'" 2>/dev/null; then
        echo "Error: Cannot connect to $REMOTE_HOST"
        echo "Please ensure:"
        echo "  1. The host is reachable"
        echo "  2. SSH keys are set up for passwordless authentication"
        echo "  3. You have appropriate access"
        exit 1
    fi
    echo "✅ SSH connection successful"
    echo ""

    # Extract hostname from remote_host (strip user@ prefix)
    ADDRESS_HOSTNAME="${REMOTE_HOST#*@}"

    # Parse remaining arguments for remote execution
    MODELS_PATH=$1
    UTILS_PATH=$2
    LORA_MODELS_PATH=$3

    # Validate argument count
    if [ -z "$LORA_MODELS_PATH" ]; then
        echo "Error: Missing required arguments"
        echo "Usage: $SCRIPT_NAME $REMOTE_HOST <models_path> <utils_path> <lora_models_path> [--tmpfs-list <filename>]"
        exit 1
    fi

    # Query the target server for its own Tailscale IPv4 address. SSH access
    # was already verified above, so this does not depend on a local Tailscale
    # CLI installation or local hostname resolution.
    if ! RESOLVED_ADDRESS=$(ssh -T -o BatchMode=yes -o ConnectTimeout=5 "$REMOTE_HOST" 'tailscale ip -4'); then
        echo "Error: Could not get the Tailscale IPv4 address from '$REMOTE_HOST'"
        exit 1
    fi
    if [[ ! "$RESOLVED_ADDRESS" =~ ^[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
        echo "Error: Invalid Tailscale IPv4 address from '$REMOTE_HOST': $RESOLVED_ADDRESS"
        exit 1
    fi

    echo "Configuration:"
    echo "  Remote host: $REMOTE_HOST"
    echo "  Address (hostname): $ADDRESS_HOSTNAME"
    echo "  Address (resolved): $RESOLVED_ADDRESS"
    echo "  Models path: $MODELS_PATH"
    echo "  Utils path: $UTILS_PATH"
    echo "  LoRA models path: $LORA_MODELS_PATH"
    echo "  Tmpfs list: ${TMPFS_LIST_NAME:-<none: tmpfs overlay skipped>}"
    echo ""

    # Remove GPU from proxy server before launching
    echo "=================================================="
    echo "  Proxy Server GPU Removal"
    echo "=================================================="
    remove_gpu_from_proxy "$RESOLVED_ADDRESS"
    echo ""

    # Build remote command (use resolved IP address)
    # Scripts are expected to be synced via update_scripts.sh
    REMOTE_DIR="${UTILS_PATH}"
    REMOTE_CMD="cd $REMOTE_DIR && sudo bash $SCRIPT_NAME $RESOLVED_ADDRESS $MODELS_PATH $UTILS_PATH $LORA_MODELS_PATH"
    if [ -n "$TMPFS_LIST_NAME" ]; then
        REMOTE_CMD="$REMOTE_CMD --tmpfs-list $TMPFS_LIST_NAME"
    fi

    # Execute the script remotely
    echo "Executing GPU server launch on $REMOTE_HOST..."
    echo "=================================================="
    ssh -t "$REMOTE_HOST" "$REMOTE_CMD"

    echo ""
    echo "=================================================="
    echo "✅ Remote GPU server launch complete!"
    echo "=================================================="
    exit 0
fi

# If we reach here, we're running locally
# Configuration from command line arguments
ADDRESS_INPUT=$1
MODELS_PATH=$2
UTILS_PATH=$3
LORA_MODELS_PATH=$4

# Resolve hostname to IP if needed (for local execution)
ADDRESS=$(resolve_hostname_to_ip "$ADDRESS_INPUT")
if [ $? -ne 0 ]; then
    echo "Error: Failed to resolve address '$ADDRESS_INPUT'"
    exit 1
fi

# Validate that paths exist
if [ ! -d "$MODELS_PATH" ]; then
    echo "Error: Models path '$MODELS_PATH' does not exist"
    exit 1
fi

if [ ! -d "$UTILS_PATH" ]; then
    echo "Error: Utils path '$UTILS_PATH' does not exist"
    exit 1
fi

if [ ! -d "$LORA_MODELS_PATH" ]; then
    echo "Error: LoRA models path '$LORA_MODELS_PATH' does not exist"
    exit 1
fi

echo "=================================================="
echo "  Local GPU Server Launch"
echo "=================================================="
echo "Configuration:"
echo "  Address (input): $ADDRESS_INPUT"
echo "  Address (resolved): $ADDRESS"
echo "  Models path: $MODELS_PATH"
echo "  Utils path: $UTILS_PATH"
echo "  LoRA models path: $LORA_MODELS_PATH"
echo "  Tmpfs list: ${TMPFS_LIST_NAME:-<none: tmpfs overlay skipped>}"
echo ""

OVERLAY_MOUNT="/mnt/unified-models"
if [ -n "$TMPFS_LIST_NAME" ]; then
    TMPFS_FILE_LIST="${SCRIPT_DIR}/${TMPFS_LIST_NAME}"
else
    TMPFS_FILE_LIST=""
fi

# Function to setup overlay using mount_overlay.sh
setup_overlay() {
    echo "Setting up overlay filesystem..."

    # Check if mount_overlay.sh exists in the same directory as this script
    local mount_script="${SCRIPT_DIR}/mount_overlay.sh"
    if [ ! -f "$mount_script" ]; then
        echo "  ⚠️ mount_overlay.sh not found at $mount_script"
        echo "  Skipping overlay setup, will use $MODELS_PATH directly"
        return 1
    fi

    # Check if tmpfs.ls file exists
    if [ ! -f "$TMPFS_FILE_LIST" ]; then
        echo "  ⚠️ File list not found at $TMPFS_FILE_LIST"
        echo "  Skipping overlay setup, will use $MODELS_PATH directly"
        return 1
    fi

    echo "  Running: sudo $mount_script -s $MODELS_PATH -l $MODELS_PATH -f $TMPFS_FILE_LIST -m $OVERLAY_MOUNT -z 700G"

    sudo "$mount_script" \
        -s "$MODELS_PATH" \
        -l "$MODELS_PATH" \
        -f "$TMPFS_FILE_LIST" \
        -m "$OVERLAY_MOUNT" \
        -z 700G

    if [ $? -eq 0 ]; then
        echo "  ✅ Overlay mounted successfully at $OVERLAY_MOUNT"
        return 0
    else
        echo "  ❌ Failed to mount overlay"
        return 1
    fi
}

# Function to check if all GPUs are idle using nvidia-smi
check_gpu_idle() {
    # Check if nvidia-smi is available
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Warning: nvidia-smi not found, skipping GPU usage check"
        return 0
    fi
    
    # Get GPU usage from nvidia-smi (simple and reliable)
    local gpu_usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null)
    
    if [ -z "$gpu_usage" ]; then
        echo "Warning: Could not read GPU usage from nvidia-smi, proceeding anyway"
        return 0
    fi
    
    # Check if all GPUs show 0% usage
    local all_idle=true
    local gpu_count=0
    local usage_list=""
    
    while IFS= read -r usage; do
        if [ -n "$usage" ]; then
            gpu_count=$((gpu_count + 1))
            # Handle [N/A] or non-numeric values from nvidia-smi (indicates GPU error state)
            if [[ "$usage" == *"N/A"* ]] || ! [[ "$usage" =~ ^[0-9]+$ ]]; then
                echo "Warning: GPU returned invalid usage value: $usage (may need reset)"
                usage_list="${usage_list}ERR "
                all_idle=false
            else
                usage_list="$usage_list$usage% "
                if [ "$usage" -gt 0 ]; then
                    all_idle=false
                fi
            fi
        fi
    done <<< "$gpu_usage"
    
    if [ "$gpu_count" -eq 0 ]; then
        echo "Warning: No GPU usage data found, proceeding anyway"
        return 0
    fi
    
    if [ "$all_idle" = true ]; then
        echo "All $gpu_count GPUs are idle (0% usage)"
        return 0
    else
        echo "Some GPUs are still busy (usage: $usage_list)"
        return 1
    fi
}

# Wait for all GPUs to be idle (with 5-minute timeout)
wait_for_gpu_idle() {
    echo "Checking GPU usage before starting containers..."
    local timeout=300  # 5 minutes in seconds
    local elapsed=0
    local check_interval=10  # Check every 10 seconds
    
    while [ $elapsed -lt $timeout ]; do
        if check_gpu_idle; then
            echo "All GPUs are idle, proceeding with container startup"
            return 0
        fi
        
        echo "Waiting for GPUs to become idle... (${elapsed}s/${timeout}s elapsed)"
        sleep $check_interval
        elapsed=$((elapsed + check_interval))
    done
    
    echo "Timeout reached (5 minutes), proceeding with container startup anyway"
    return 0
}

# Pull the latest image
echo "Pulling latest Docker image..."
sudo docker pull drawthingsai/draw-things-grpc-server-cli:latest

# Wait for GPUs to be idle before starting containers
wait_for_gpu_idle

# Stop and remove any existing containers
echo "Stopping and removing existing containers..."
for i in $(seq 0 7); do
    CONTAINER_NAME="grpc_service_$i"
    sudo docker stop $CONTAINER_NAME 2>/dev/null || true
    sudo docker rm $CONTAINER_NAME 2>/dev/null || true
done

# Setup overlay filesystem
echo ""
echo "=================================================="
echo "  Overlay Filesystem Setup"
echo "=================================================="

if [ -z "$TMPFS_LIST_NAME" ]; then
    echo "⏭️  Skipping tmpfs overlay setup (no --tmpfs-list specified)"
    MODELS_MOUNT_PATH="$MODELS_PATH"
    echo "Containers will use direct path: $MODELS_MOUNT_PATH"
else
    if setup_overlay; then
        MODELS_MOUNT_PATH="$OVERLAY_MOUNT"
        echo "✅ Containers will use overlay at: $MODELS_MOUNT_PATH"
    else
        MODELS_MOUNT_PATH="$MODELS_PATH"
        echo "⚠️  Containers will use direct path: $MODELS_MOUNT_PATH"
    fi
fi

echo "=================================================="
echo ""

# Start 4 separate containers
for i in $(seq 0 7); do
    PORT=$((40001 + i))
    GPU=$i
    CONTAINER_NAME="grpc_service_$i"
    
    echo "Starting container $CONTAINER_NAME on port $PORT using GPU $GPU at address $ADDRESS"
    
    sudo docker run -d \
        --name $CONTAINER_NAME \
        --network=host \
        --restart=unless-stopped \
        --gpus '"device='$GPU'"' \
        --tmpfs /tmp \
        -v "${MODELS_MOUNT_PATH}:/models" \
        -v "${UTILS_PATH}:/utils" \
        -v "${LORA_MODELS_PATH}:/loraModels" \
        drawthingsai/draw-things-grpc-server-cli:latest \
        /utils/start_single_grpc.sh $PORT $GPU $ADDRESS
        
    if [ $? -eq 0 ]; then
        echo "Successfully started container $CONTAINER_NAME"
    else
        echo "Failed to start container $CONTAINER_NAME"
    fi
    
    # Small delay between container starts
    sleep 2
done

echo "All containers started. Use 'sudo docker ps' to check status."
echo "To view logs for a specific container, use: sudo docker logs grpc_service_<number>"
echo "To stop all containers, run: sudo docker stop \$(sudo docker ps -q --filter name=grpc_service_)"
